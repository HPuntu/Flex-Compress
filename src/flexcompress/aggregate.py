"""Merge per-target results into tables, summaries and the universality analysis.

Outputs under ``<run>/report``:
    stage0.csv           one row per target: PCA counts, cosine content, RMSIP,
                         drift, native contacts, replica floors and correlations
    stage1.csv           one row per (target, fold, codec, rate) with absolute and
                         floor-normalised distortions
    stage1_fits.csv      fit diagnostics per (target, fold)
    summary.json         grouped medians, rank-needed analysis with N scaling,
                         model adequacy, reliability-vs-lag curves, universality
    figures/*.png        if matplotlib is available
"""

from __future__ import annotations

import csv
import hashlib
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io_utils import atomic_write_json, now_utc, read_json
from .metrics import METRICS

__all__ = ["aggregate", "universality"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _q(values) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return {"n": int(arr.size), "median": float(med), "q25": float(q25), "q75": float(q75)}


def _group_key(doc_or_row) -> str:
    target = doc_or_row["target"] if "target" in doc_or_row else doc_or_row
    return f"{target['source']}/T{target['temperature']}"


def _loglog_slope(x, y, n_boot: int, seed: int) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = np.log(x[ok]), np.log(y[ok])
    if x.size < 4 or np.ptp(x) == 0:
        return {"n": int(x.size), "slope": None, "ci95": None}
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, x.size, x.size)
        if np.ptp(x[pick]) > 0:
            boots.append(np.polyfit(x[pick], y[pick], 1)[0])
    ci = np.percentile(boots, [2.5, 97.5]).tolist() if boots else None
    return {"n": int(x.size), "slope": slope, "ci95": ci}


# --------------------------------------------------------------------------- stage 0


def _flatten_stage0(doc: dict) -> dict:
    t, s0 = doc["target"], doc.get("stage0", {})
    row = {
        "source": t["source"], "domain": t["domain_id"], "temperature": t["temperature"],
        "n_res": t["n_res"], "n_rep": len(doc.get("replicas_used", [])),
        "t_min_frames": min(doc.get("n_frames_used", [0]) or [0]), "dt_ps": doc.get("dt_ps"),
        "segment_frames": doc.get("segment_frames"),
    }
    for block in ("ca_pca_per_replica_mean", "ca_pca_pooled", "bb_pca_per_replica_mean", "bb_pca_pooled"):
        for key, value in (s0.get(block) or {}).items():
            row[f"{block.replace('_per_replica_mean', '_rep')}_{key}"] = value
    for i, value in enumerate(s0.get("cosine_content_mean", []), 1):
        row[f"cosine_pc{i}"] = value
    for k, value in (s0.get("rmsip") or {}).items():
        row[f"rmsip_{k}"] = value
    row["mean_shift_ratio"] = s0.get("mean_shift_ratio")
    row["half_drift_max"] = max(s0.get("half_drift", [np.nan]) or [np.nan])
    row["native_q_mean_min"] = min(s0.get("native_contacts_mean", [np.nan]) or [np.nan])
    row["native_q_last10_min"] = min(s0.get("native_contacts_last10", [np.nan]) or [np.nan])
    for m, value in (s0.get("floor_pairwise") or {}).items():
        row[f"floor_{m}"] = value
    for k, value in (s0.get("replica_corr") or {}).items():
        row[f"corr_{k}"] = value
    for k, value in (s0.get("replica_corr_spearman_brown") or {}).items():
        row[f"corr_sb_{k}"] = value
    row["acf_horizon_ps"] = (s0.get("acf_reliable_horizon_frames") or 0.0) * float(doc.get("dt_ps") or 0.0)
    asym_ratio = s0.get("asym_ratio_by_lag") or []
    asym_cos = s0.get("asym_replica_cos_by_lag") or []
    row["asym_ratio_mean"] = float(np.mean(asym_ratio)) if asym_ratio else None
    row["asym_replica_cos_mean"] = float(np.nanmean(asym_cos)) if asym_cos else None
    band = doc.get("band_reliability") or s0.get("band_reliability") or {}
    for ps, corr, frac in zip(band.get("bands_ps", []), band.get("log_energy_corr_by_band", []),
                              band.get("energy_fraction_by_band", [])):
        row[f"band_{ps:g}ps_corr"] = corr
        row[f"band_{ps:g}ps_frac"] = frac
    row["msf_total_A2"] = s0.get("msf_total_A2")
    return row


def _curve_on_grid(lags_ps, values, grid_ps) -> np.ndarray:
    lags_ps = np.asarray(lags_ps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = (lags_ps > 0) & np.isfinite(values)
    if ok.sum() < 2:
        return np.full(len(grid_ps), np.nan)
    x, y = np.log(lags_ps[ok]), values[ok]
    out = np.interp(np.log(grid_ps), x, y, left=np.nan, right=np.nan)
    return out


def _summarize_stage0(docs: list[dict], rows: list[dict], cfg: dict) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, doc in enumerate(docs):
        groups[_group_key(doc)].append(i)
    out = {}
    for key, index in sorted(groups.items()):
        sub_rows = [rows[i] for i in index]
        numeric = []
        for r in sub_rows:
            numeric += [k for k in r if k not in numeric and k not in {"source", "domain", "temperature"}]
        entry = {"n_targets": len(index), "medians": {k: _q([r.get(k) for r in sub_rows]) for k in numeric}}
        for col in ("ca_pca_rep_n_pc_90", "ca_pca_pooled_n_pc_90", "bb_pca_rep_n_pc_90", "bb_pca_pooled_n_pc_90"):
            if any(col in r for r in sub_rows):
                entry[f"scaling_{col}_vs_N"] = _loglog_slope(
                    [r["n_res"] for r in sub_rows], [r.get(col, np.nan) for r in sub_rows],
                    cfg["aggregate"]["bootstrap"], cfg["seed"])
        lag_ps_all = [np.asarray(docs[i]["lags_ps"][1:]) for i in index]
        lo = min(float(l.min()) for l in lag_ps_all if l.size)
        hi = max(float(l.max()) for l in lag_ps_all if l.size)
        grid = np.geomspace(lo, hi, 24)
        acf = np.array([_curve_on_grid(docs[i]["lags_ps"][1:], docs[i]["stage0"]["acf_corr_by_lag"], grid)
                        for i in index])
        pair_lags_ps = [np.asarray(docs[i]["pair_lags"][1:]) * docs[i]["dt_ps"] for i in index]
        ldcc = np.array([_curve_on_grid(pair_lags_ps[j], docs[i]["stage0"]["lagged_dcc_corr_by_lag"], grid)
                         for j, i in enumerate(index)])
        asym = np.array([_curve_on_grid(pair_lags_ps[j], docs[i]["stage0"]["asym_replica_cos_by_lag"], grid)
                         for j, i in enumerate(index)])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            entry["reliability_vs_lag"] = {
                "lag_ps": grid.tolist(),
                "acf_corr_median": np.nanmedian(acf, axis=0).tolist(),
                "lagged_dcc_corr_median": np.nanmedian(ldcc, axis=0).tolist(),
                "asym_replica_cos_median": np.nanmedian(asym, axis=0).tolist(),
                "n_contributing": np.sum(np.isfinite(acf), axis=0).tolist(),
            }
        out[key] = entry
    return out


# --------------------------------------------------------------------------- stage 1


def _stage1_tables(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, fits = [], []
    for doc in docs:
        s1 = doc.get("stage1")
        if not s1:
            continue
        t = doc["target"]
        base = {"source": t["source"], "domain": t["domain_id"], "temperature": t["temperature"],
                "n_res": t["n_res"], "t_min_frames": min(doc["n_frames_used"]), "dt_ps": doc["dt_ps"]}
        floors = {r["fold"]: r for r in s1["rows"] if r["method"] == "floor"}
        for r in s1["rows"]:
            row = dict(base)
            row.update({k: r.get(k) for k in ("fold", "n_fit", "method", "param", "params_per_res", "full_rank")})
            row["on_grid"] = bool(r.get("on_grid", True))
            row["rank_actual"] = r.get("param")
            if r.get("full_rank") and not row["on_grid"]:
                # The full rank can differ between folds (per-band codecs);
                # one key lets its folds average together.
                row["param"] = "full"
            floor = floors.get(r["fold"], {})
            for m in METRICS:
                row[m] = r.get(m)
                f = floor.get(m)
                row[f"rel_{m}"] = (r[m] / f) if (r.get(m) is not None and f not in (None, 0) and np.isfinite(f)) else None
            rel_vals = [row[f"rel_{m}"] for m in METRICS if row[f"rel_{m}"] is not None]
            row["rel_max"] = max(rel_vals) if rel_vals else None
            rows.append(row)
        for f in s1.get("fits", []):
            fit = dict(base)
            fit.update({k: v for k, v in f.items() if k != "band_variance_fraction"})
            for ps, frac in zip(s1.get("bands_ps", []), f.get("band_variance_fraction", [])):
                fit[f"band_var_{ps:.4g}ps"] = frac
            fits.append(fit)
    return rows, fits


def _per_target_means(rows: list[dict], on_grid_only: bool = False) -> dict:
    """(group, domain, method, param) -> fold-averaged rel metrics and params."""

    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if on_grid_only and not r.get("on_grid", True):
            continue
        key = (f"{r['source']}/T{r['temperature']}", r["domain"], r["method"],
               r["param"] if r["param"] not in ("", None) else None)
        for m in METRICS + ("max",):
            v = r.get(f"rel_{m}")
            if v is not None and np.isfinite(v):
                acc[key][f"rel_{m}"].append(v)
        for m in METRICS:
            v = r.get(m)
            if v is not None and np.isfinite(v):
                acc[key][m].append(v)
        if r.get("params_per_res") is not None:
            acc[key]["params_per_res"].append(r["params_per_res"])
        if isinstance(r.get("rank_actual"), (int, float)):
            acc[key]["rank_actual"].append(float(r["rank_actual"]))
        acc[key]["n_res"].append(r["n_res"])
        acc[key]["full_rank"].append(bool(r.get("full_rank")))
    out = {}
    for key, values in acc.items():
        out[key] = {k: (float(np.mean(v)) if k != "full_rank" else any(v)) for k, v in values.items() if v}
    return out


def _summarize_stage1(rows: list[dict], cfg: dict) -> dict:
    def grouped(per_target):
        groups: dict = defaultdict(lambda: defaultdict(list))
        for (group, domain, method, param), values in per_target.items():
            groups[group][(method, param)].append((domain, values))
        return groups

    # Curves use the configured rate grid only (a rank capped at a small
    # protein's maximum would otherwise mix proteins at one curve point);
    # rank-needed and adequacy use every row, including each full-rank fit.
    grid_groups = grouped(_per_target_means(rows, on_grid_only=True))
    all_groups = grouped(_per_target_means(rows))
    out = {}
    for group, methods in sorted(all_groups.items()):
        curves = []
        grid_methods = grid_groups.get(group, {})
        for (method, param), items in sorted(grid_methods.items(),
                                             key=lambda x: (x[0][0], -1 if x[0][1] is None else x[0][1])):
            entry = {"method": method, "param": param, "n_targets": len(items),
                     "params_per_res": _q([v.get("params_per_res") for _, v in items])}
            for m in METRICS + ("max",):
                entry[f"rel_{m}"] = _q([v.get(f"rel_{m}") for _, v in items])
            for m in METRICS:
                entry[m] = _q([v.get(m) for _, v in items])
            curves.append(entry)
        g = {"curves": curves, "rank_needed": {}, "full_model_adequacy": {}}
        # Smallest rate reaching the replica floor, per target and metric.
        for method in ("spectral_subspace_local", "spectral_subspace", "spectral_band_local", "pca_exp"):
            options = sorted([(p, items) for (mm, p), items in methods.items() if mm == method and p is not None],
                             key=lambda x: np.inf if x[0] == "full" else x[0])
            if not options:
                continue
            domains = sorted({d for _, items in options for d, _ in items})
            for threshold in cfg["aggregate"]["rank_threshold"]:
                for metric in METRICS + ("max",):
                    needed, n_res, params = [], [], []
                    for domain in domains:
                        hit = None
                        for p, items in options:
                            vals = dict(items).get(domain)
                            if vals is None:
                                continue
                            v = vals.get(f"rel_{metric}")
                            if v is not None and v <= threshold:
                                hit = (vals.get("rank_actual", p), vals)
                                break
                        size = next((dict(items)[domain]["n_res"] for _, items in options if domain in dict(items)), None)
                        needed.append(hit[0] if hit else np.nan)
                        params.append(hit[1].get("params_per_res") if hit else np.nan)
                        n_res.append(size)
                    reached = np.isfinite(np.asarray(needed, dtype=np.float64))
                    g["rank_needed"].setdefault(method, {}).setdefault(f"thr_{threshold:g}", {})[metric] = {
                        "fraction_reached": float(reached.mean()) if len(reached) else None,
                        "rank": _q(needed),
                        "params_per_res": _q(params),
                        # Rank 0 (local tokens suffice) enters the log-log fit as 0.5.
                        "scaling_vs_N": _loglog_slope(n_res, np.maximum(np.asarray(needed, dtype=float), 0.5),
                                                      cfg["aggregate"]["bootstrap"], cfg["seed"]),
                    }
        full = [v for (mm, p), items in methods.items() if mm == "spectral_subspace_local"
                for _, v in items if v.get("full_rank")]
        g["full_model_adequacy"] = {f"rel_{m}": _q([v.get(f"rel_{m}") for v in full]) for m in METRICS}
        out[group] = g
    return out


# --------------------------------------------------------------------------- universality


def _stable_half(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 2


def universality(items: list[tuple[str, np.ndarray]], dims_labels: list[str] | None = None) -> dict:
    """Dimensionality of the replica-reproducible part of per-residue descriptors.

    ``items``: (target key, array (n_rep, N, D)).  Signal covariance is the
    symmetrised cross-replica covariance (noise is independent across replicas),
    pooled over residues and proteins; the total covariance includes noise.
    Held-out curve: eigenvectors of the signal covariance from one half of the
    proteins, scored by the signal variance they capture in the other half.
    """

    items = [(k, np.asarray(a, dtype=np.float64)) for k, a in items if a.shape[0] >= 2 and a.size]
    if not items:
        return {"n_targets": 0}
    dim = items[0][1].shape[-1]
    flat = np.concatenate([a.reshape(-1, dim) for _, a in items])
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    mu, sd = flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-12)

    def accumulate(sel):
        # Each protein contributes per residue, averaged over its replicas (total)
        # or replica pairs (signal), so replica counts do not reweight proteins.
        signal = np.zeros((dim, dim))
        total = np.zeros((dim, dim))
        count = 0
        for _, a in sel:
            z = (a - mu) / sd
            z = np.where(np.isfinite(z), z, 0.0)
            n_rep = z.shape[0]
            summed = z.sum(axis=0)
            own = np.einsum("rni,rnj->ij", z, z)
            total += own / n_rep
            signal += (summed.T @ summed - own) / (n_rep * (n_rep - 1))
            count += z.shape[1]
        signal = 0.5 * (signal + signal.T) / max(count, 1)
        return signal, total / max(count, 1)

    signal, total = accumulate(items)
    w = np.clip(np.linalg.eigvalsh(signal)[::-1], 0.0, None)
    frac = np.cumsum(w) / max(w.sum(), 1e-300)
    dims = {f"dims_{int(t * 100)}": int(np.searchsorted(frac, t) + 1) for t in (0.8, 0.9, 0.95, 0.99)}
    halves = [[it for it in items if _stable_half(it[0]) == h] for h in (0, 1)]
    held_out = None
    if all(len(h) >= 2 for h in halves):
        curves = []
        for a, b in ((0, 1), (1, 0)):
            s_a, _ = accumulate(halves[a])
            s_b, _ = accumulate(halves[b])
            _, vec = np.linalg.eigh(s_a)
            vec = vec[:, ::-1]
            # Same normaliser as the in-sample curve: positive signal variance.
            denom = max(float(np.clip(np.linalg.eigvalsh(s_b), 0.0, None).sum()), 1e-300)
            curves.append([float(np.trace(vec[:, :k].T @ s_b @ vec[:, :k]) / denom) for k in range(1, dim + 1)])
        held_out = np.mean(curves, axis=0).tolist()
    per_dim = (np.diag(signal) / np.maximum(np.diag(total), 1e-300)).tolist()
    return {
        "n_targets": len(items),
        "n_residues": int(sum(a.shape[1] for _, a in items)),
        "descriptor_dim": dim,
        "reliable_fraction": float(np.trace(signal) / max(np.trace(total), 1e-300)),
        "signal_eigenvalues": w.tolist(),
        "signal_cumulative": frac.tolist(),
        **dims,
        "held_out_captured": held_out,
        "per_dim_reliability": per_dim,
        "dim_labels": dims_labels,
    }


def _descriptor_groups(run_dir: Path) -> dict:
    groups: dict = defaultdict(list)
    for path in sorted((run_dir / "descriptors").rglob("*.npz")):
        key = path.stem
        temperature = key.rsplit("__T", 1)[-1]
        source = path.parent.name
        with np.load(path, allow_pickle=False) as data:
            groups[f"{source}/T{temperature}"].append((key, data["descriptors"], data["bands_ps"]))
    return groups


# --------------------------------------------------------------------------- entry


def aggregate(run_dir: str | Path, figures: bool = True) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    cfg = read_json(run_dir / "config.json")
    listing = read_json(run_dir / "targets.json")
    report = run_dir / "report"
    report.mkdir(parents=True, exist_ok=True)

    docs = [read_json(p) for p in sorted((run_dir / "results").rglob("*.json"))]
    errors = [read_json(p) for p in sorted((run_dir / "errors").rglob("*.json"))] if (run_dir / "errors").is_dir() else []
    ok = [d for d in docs if d.get("status") == "ok"]
    stale = [d["target"]["domain_id"] for d in ok if d.get("config_hash") != cfg["run"]["config_hash"]]

    stage0_docs = [d for d in ok if "acf_corr_by_lag" in d.get("stage0", {})]
    stage0_rows = [_flatten_stage0(d) for d in stage0_docs]
    _write_csv(report / "stage0.csv", stage0_rows)
    stage1_rows, fit_rows = _stage1_tables(ok)
    _write_csv(report / "stage1.csv", stage1_rows)
    _write_csv(report / "stage1_fits.csv", fit_rows)

    uni = {}
    for group, entries in _descriptor_groups(run_dir).items():
        bands = entries[0][2]
        labels = [f"{b:g}ps:{c}" for b in bands for c in ("xx", "yy", "zz", "xy", "xz", "yz")]
        uni[group] = universality([(k, d) for k, d, _ in entries], labels)

    from .workqueue import progress

    coverage = progress(run_dir, listing)
    summary = {
        "created": now_utc(),
        "run_dir": str(run_dir),
        "config_hash": cfg["run"]["config_hash"],
        "coverage": coverage,
        "n_targets_configured": listing["n_targets"],
        "n_results": len(docs),
        "n_ok": len(ok),
        "n_errors": len(errors),
        "status_counts": {s: sum(1 for d in docs if d.get("status") == s) for s in {d.get("status") for d in docs}},
        "stale_results": stale,
        "errors": [{"target": e["target"]["domain_id"], "temperature": e["target"]["temperature"],
                    "error": e["error"]} for e in errors],
        "stage0": _summarize_stage0(stage0_docs, stage0_rows, cfg) if stage0_docs else {},
        "stage1": _summarize_stage1(stage1_rows, cfg) if stage1_rows else {},
        "universality": uni,
        "timing_s": _q([d.get("timing", {}).get("total") for d in ok]),
    }
    atomic_write_json(report / "summary.json", summary)
    if figures:
        try:
            from .report import make_figures

            make_figures(report, stage0_rows, stage1_rows, summary)
        except ImportError as exc:
            print(f"figures skipped: {exc}")
    print(f"aggregated {len(ok)}/{listing['n_targets']} targets ({len(errors)} errors"
          + ("" if coverage["complete"] else f"; {coverage['remaining'] + coverage['running']} not finished yet")
          + f") -> {report}")
    return summary


if __name__ == "__main__":
    import sys

    aggregate(Path(sys.argv[1]))
