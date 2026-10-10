"""Per-target Stage 0 (reliability/diagnostics) and Stage 1 (compressibility).

A *target* is one domain at one temperature with >= 2 replicas.  Everything
is computed from the same per-replica lagged statistics:

Stage 0
    replica-vs-replica reliability of every profile statistic (the noise
    floor / ceiling), PCA variance counts, Hess cosine content, replica RMSIP,
    between-replica mean shift, half-trajectory drift, native-contact
    fraction, and reproducibility of the antisymmetric (irreversible) part.

Descriptors
    per replica, per residue, band-resolved 3x3 tensors in residue frames on a
    fixed per-source band grid (log-Euclidean, 6 numbers per band), for the
    cross-protein universality analysis in ``aggregate``.

Stage 1
    leave-one-replica-out: fit codecs on the remaining replicas, score their
    expected statistics against the held-out replica, and normalise by the
    floor (pooled raw statistics of the fit replicas vs the held-out one).
    Codecs: spectral subspace / per-band low-rank (each optionally + per-residue
    local tensors), PCA + single exponential, truncated DCT.
"""

from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from . import diagnostics as dg
from .baselines import dct_truncate, fit_pca_exp, pca_exp_candidates, pca_exp_model
from .data import Target, load_replica, load_static
from .frames import residue_frames, to_local
from .kernels import kernel_matrix
from .metrics import METRICS, distortion, summarize
from .spectral import (
    band_timescales,
    fit_local_residual,
    fit_psd_bands,
    model_from_components,
    spectral_components,
    subspace_model,
)
from .stats import (
    Lagged,
    center,
    cross_sums,
    diag_block_sums,
    lag_grid,
    pair_rows,
    pair_subset,
    segmented,
    subset_lags,
    window_starts,
)

__all__ = ["config_hash", "process_target", "resolve_stride"]

RESULT_SCHEMA = 1


def config_hash(cfg: dict) -> str:
    """Hash of the scientific configuration (paths and run bookkeeping excluded)."""

    keep = {k: v for k, v in cfg.items() if k not in {"archive_root", "index_path", "run", "selection", "split"}}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def resolve_stride(source_interval_ps: float, target_interval_ps: float | None) -> int:
    if target_interval_ps is None:
        return 1
    ratio = float(target_interval_ps) / float(source_interval_ps)
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-6:
        raise ValueError(
            f"target interval {target_interval_ps} ps is not an integer multiple of {source_interval_ps} ps"
        )
    return stride


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def _spearman_brown(r: float, n: int) -> float:
    if not np.isfinite(r) or n < 2:
        return float("nan")
    return float(n * r / (1.0 + (n - 1) * r)) if (1.0 + (n - 1) * r) != 0 else float("nan")


def _nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def logm_sym6(tensors: np.ndarray, eps: float) -> np.ndarray:
    """Log-Euclidean embedding of PSD 3x3 tensors: 6 isometric coordinates."""

    w, v = np.linalg.eigh(0.5 * (tensors + np.swapaxes(tensors, -1, -2)))
    log_t = (v * np.log(np.clip(w, 0.0, None) + eps)[..., None, :]) @ np.swapaxes(v, -1, -2)
    root2 = np.sqrt(2.0)
    return np.stack(
        [
            log_t[..., 0, 0],
            log_t[..., 1, 1],
            log_t[..., 2, 2],
            root2 * log_t[..., 0, 1],
            root2 * log_t[..., 0, 2],
            root2 * log_t[..., 1, 2],
        ],
        axis=-1,
    )


class _Replica:
    __slots__ = ("name", "n_frames", "dx", "mean", "diag", "pair", "summary", "native", "ca_pcs")

    def __init__(self, name: str, ca: np.ndarray, native: np.ndarray):
        self.name = name
        self.n_frames = int(ca.shape[0])
        self.mean = ca.mean(axis=0)
        self.dx = center(ca)
        self.native = native
        self.diag = None
        self.pair = None
        self.summary = None
        self.ca_pcs = None


def process_target(target: Target, cfg: dict, log=print) -> tuple[dict, dict | None]:
    """Run all enabled stages for one target; returns (result JSON, descriptor arrays)."""

    timing: dict[str, float] = {}
    started = time.monotonic()
    diag_cfg, stats_cfg = cfg["diagnostics"], cfg["stats"]
    stride = resolve_stride(target.frame_interval_ps, cfg.get("frame_interval_ps"))
    dt_ps = float(target.frame_interval_ps) * stride

    static = load_static(target.archive_path)
    n_res = int(target.n_res)
    bb_pooled_ok = bool(diag_cfg["bb_pca"]) and 12 * n_res <= int(diag_cfg["bb_pooled_max_dim"])
    bb_sum = np.zeros(12 * n_res) if bb_pooled_ok else None
    bb_outer = np.zeros((12 * n_res, 12 * n_res)) if bb_pooled_ok else None
    bb_count = 0
    bb_rep: list[dict] = []
    thresholds = diag_cfg["variance_thresholds"]

    replicas: list[_Replica] = []
    for name, n_frames in zip(target.replicas, target.n_frames):
        loaded = load_replica(
            target.archive_path, target.temperature, name, n_frames, stride=stride,
            keep_backbone=bool(diag_cfg["bb_pca"]),
        )
        ca = loaded["ca"]
        if ca.shape[0] < int(cfg["min_frames"]):
            log(f"  replica {name}: {ca.shape[0]} frames < min_frames, skipped")
            continue
        if ca.shape[1] != n_res:
            raise ValueError(f"replica {name} has {ca.shape[1]} residues, expected {n_res}")
        rep = _Replica(name, ca, loaded["native_reference"])
        if loaded["backbone"] is not None:
            flat = loaded["backbone"].reshape(rep.n_frames, 12 * n_res)
            centred = flat - flat.mean(axis=0, keepdims=True)
            var, _, total = dg.top_pcs(centred, int(diag_cfg["pca_top"]), seed=int(cfg["seed"]))
            bb_rep.append(dg.variance_counts(var, total, thresholds))
            if bb_pooled_ok:
                f64 = flat.astype(np.float64)
                bb_sum += f64.sum(axis=0)
                bb_outer += f64.T @ f64
                bb_count += rep.n_frames
            del flat, centred
        del loaded
        replicas.append(rep)
    timing["load"] = time.monotonic() - started

    result = {
        "schema": RESULT_SCHEMA,
        "status": "ok",
        "target": target.to_dict(),
        "config_hash": config_hash(cfg),
        "dt_ps": dt_ps,
        "stride": stride,
    }
    if len(replicas) < int(cfg["min_replicas"]):
        result["status"] = "insufficient_replicas"
        result["replicas_used"] = [r.name for r in replicas]
        return result, None

    frames = [r.n_frames for r in replicas]
    # One common window length for every statistic of this target (default:
    # the shortest replica), so fit replicas, the held-out replica and every
    # model share a single estimator expectation regardless of replica lengths.
    segment = min(int(stats_cfg.get("segment_frames") or min(frames)), min(frames))
    lags = lag_grid(segment, int(stats_cfg["n_lags"]), float(stats_cfg["max_lag_fraction"]))
    pair_lags = subset_lags(lags, int(stats_cfg["n_pair_lags"]))
    residues = pair_subset(n_res, int(stats_cfg["pair_max_residues"]))
    rows = pair_rows(residues)
    result.update(
        replicas_used=[r.name for r in replicas],
        n_frames_used=frames,
        segment_frames=segment,
        n_windows=[int(len(window_starts(f, segment))) for f in frames],
        lags=lags.tolist(),
        lags_ps=(lags * dt_ps).tolist(),
        pair_lags=pair_lags.tolist(),
        n_pair_residues=int(residues.size),
    )

    t = time.monotonic()
    for rep in replicas:
        rep.diag = segmented(rep.dx, lags, segment, diag_block_sums)
        rep.pair = segmented(rep.dx.reshape(rep.n_frames, 3 * n_res)[:, rows], pair_lags, segment, cross_sums)
        rep.summary = summarize(rep.diag.mean(), rep.pair.mean())
    timing["statistics"] = time.monotonic() - t

    descriptors = None
    if cfg["stages"].get("stage0", True):
        t = time.monotonic()
        result["stage0"] = _stage0(replicas, static, cfg, n_res, residues, pair_lags, bb_rep,
                                   (bb_sum, bb_outer, bb_count) if bb_pooled_ok else None)
        timing["stage0"] = time.monotonic() - t
    if cfg["descriptors"].get("enabled", True):
        t = time.monotonic()
        descriptors, desc_rel = _descriptors(replicas, static, cfg, lags, dt_ps, segment)
        result["band_reliability"] = desc_rel
        timing["descriptors"] = time.monotonic() - t
    if cfg["stages"].get("stage1", True):
        t = time.monotonic()
        result["stage1"] = _stage1(replicas, cfg, n_res, lags, pair_lags, residues, dt_ps, segment, log)
        timing["stage1"] = time.monotonic() - t
    timing["total"] = time.monotonic() - started
    result["timing"] = timing
    return result, descriptors


# --------------------------------------------------------------------------- stage 0


def _stage0(replicas, static, cfg, n_res, residues, pair_lags, bb_rep, bb_pooled) -> dict:
    dcfg = cfg["diagnostics"]
    thresholds = dcfg["variance_thresholds"]
    n_rep = len(replicas)
    seed = int(cfg["seed"])
    out: dict = {"n_replicas": n_rep}

    # PCA on CA, per replica and pooled (grand mean, so between-replica shifts count).
    per_rep_counts, cosines, comps = [], [], []
    for rep in replicas:
        flat = rep.dx.reshape(rep.n_frames, 3 * n_res)
        var, vec, total = dg.top_pcs(flat, int(dcfg["pca_top"]), seed=seed)
        per_rep_counts.append(dg.variance_counts(var, total, thresholds))
        n_cos = int(dcfg["cosine_pcs"])
        cosines.append(dg.cosine_content(flat @ vec[:, :n_cos], n_cos).tolist())
        comps.append(vec)
    out["ca_pca_per_replica"] = per_rep_counts
    out["ca_pca_per_replica_mean"] = {k: _nanmean([c[k] for c in per_rep_counts]) for k in per_rep_counts[0]}
    weights_t = np.array([rep.n_frames for rep in replicas], dtype=np.float64)
    grand = (np.tensordot(weights_t, np.stack([rep.mean for rep in replicas]), axes=(0, 0)) / weights_t.sum()).reshape(-1)
    cov = np.zeros((3 * n_res, 3 * n_res))
    count = 0
    for rep in replicas:
        shifted = rep.dx.reshape(rep.n_frames, -1) + (rep.mean.reshape(-1) - grand)[None, :]
        cov += shifted.T @ shifted
        count += rep.n_frames
    w = np.clip(np.linalg.eigvalsh(cov / count)[::-1], 0.0, None)
    out["ca_pca_pooled"] = dg.variance_counts(w, float(w.sum()), thresholds)
    out["cosine_content"] = cosines
    out["cosine_content_mean"] = np.mean(np.asarray(cosines), axis=0).tolist()
    out["rmsip"] = {
        str(k): _nanmean([dg.rmsip(comps[a], comps[b], int(k)) for a, b in combinations(range(n_rep), 2)])
        for k in dcfg["rmsip_k"]
    }
    if bb_rep:
        out["bb_pca_per_replica_mean"] = {k: _nanmean([c[k] for c in bb_rep]) for k in bb_rep[0]}
    if bb_pooled is not None:
        total_sum, outer, count_bb = bb_pooled
        mean = total_sum / count_bb
        cov_bb = outer / count_bb - np.outer(mean, mean)
        k = min(int(dcfg["pca_top"]), cov_bb.shape[0])
        w_bb = scipy_eigh(cov_bb, eigvals_only=True, subset_by_index=[cov_bb.shape[0] - k, cov_bb.shape[0] - 1])
        out["bb_pca_pooled"] = dg.variance_counts(np.clip(w_bb[::-1], 0, None), float(np.trace(cov_bb)), thresholds)

    # Sampling / stationarity.
    msf_within = np.mean([float(np.sum(rep.summary["msf"])) for rep in replicas])
    means = np.stack([rep.mean for rep in replicas])
    between = float(np.sum((means - means.mean(axis=0)) ** 2) / max(n_rep - 1, 1))
    out["mean_shift_ratio"] = between / max(msf_within, 1e-300)
    drifts, q_mean, q_last = [], [], []
    for rep in replicas:
        half = rep.n_frames // 2
        drift = rep.dx[:half].mean(axis=0) - rep.dx[half:].mean(axis=0)
        drifts.append(float(np.sum(drift**2) / max(np.sum(rep.summary["msf"]), 1e-300)))
        q = dg.native_contact_fraction(
            rep.dx + rep.mean, rep.native[:, 1], int(dcfg["contact_frames"]),
            float(dcfg["contact_cutoff_A"]), float(dcfg["contact_tolerance"]),
        )
        q_mean.append(float(np.nanmean(q)))
        tail = q[max(len(q) - max(len(q) // 10, 1), 0):]
        q_last.append(float(np.nanmean(tail)))
    out["half_drift"] = drifts
    out["native_contacts_mean"] = q_mean
    out["native_contacts_last10"] = q_last

    # Reliability between replicas: floors and correlations.
    pairs = list(combinations(range(n_rep), 2))
    floors = []
    corr = {"log_msf": [], "shape": [], "dcc0": []}
    acf_corr = []
    lagged_dcc_corr = []
    asym_cos = []
    off = ~np.eye(residues.size, dtype=bool)
    upper = np.triu_indices(residues.size, k=1)
    tri3 = np.triu_indices(3)

    def deviatoric(shape):
        # Trace-1 shape tensors share a fixed diagonal/off-diagonal contrast;
        # correlate only the deviation from isotropy.
        return (shape - np.eye(3) / 3.0)[:, tri3[0], tri3[1]]
    antis = [dg.asymmetry(rep.pair.mean())[0] for rep in replicas]
    ratios = [dg.asymmetry(rep.pair.mean())[1] for rep in replicas]
    for a, b in pairs:
        sa, sb = replicas[a].summary, replicas[b].summary
        d_ab, d_ba = distortion(sa, sb), distortion(sb, sa)
        floors.append({m: 0.5 * (d_ab[m] + d_ba[m]) for m in METRICS})
        corr["log_msf"].append(_pearson(np.log(sa["msf"] + 1e-12), np.log(sb["msf"] + 1e-12)))
        corr["shape"].append(_pearson(deviatoric(sa["shape"]), deviatoric(sb["shape"])))
        corr["dcc0"].append(_pearson(sa["dcc"][0][upper], sb["dcc"][0][upper]))
        acf_corr.append([_pearson(sa["acf"][k], sb["acf"][k]) for k in range(sa["acf"].shape[0])])
        lagged_dcc_corr.append(
            [_pearson(sa["dcc"][k][off], sb["dcc"][k][off]) for k in range(1, sa["dcc"].shape[0])]
        )
        asym_cos.append(
            [
                float(np.sum(antis[a][k] * antis[b][k])
                      / max(np.linalg.norm(antis[a][k]) * np.linalg.norm(antis[b][k]), 1e-300))
                for k in range(1, len(pair_lags))
            ]
        )
    out["floor_pairwise"] = {m: _nanmean([f[m] for f in floors]) for m in METRICS}
    rel = {k: _nanmean(v) for k, v in corr.items()}
    out["replica_corr"] = rel
    out["replica_corr_spearman_brown"] = {k: _spearman_brown(v, n_rep) for k, v in rel.items()}
    acf_by_lag = np.nanmean(np.asarray(acf_corr, dtype=np.float64), axis=0) if acf_corr else np.array([])
    out["acf_corr_by_lag"] = acf_by_lag.tolist()
    lag_values = np.asarray(replicas[0].diag.lags[1:])
    below = np.nonzero(~(acf_by_lag >= 0.5))[0]  # NaN counts as unreliable
    first_bad = int(below[0]) if below.size else len(acf_by_lag)
    out["acf_reliable_horizon_frames"] = float(lag_values[first_bad - 1]) if first_bad > 0 else 0.0
    out["lagged_dcc_corr_by_lag"] = (
        np.nanmean(np.asarray(lagged_dcc_corr, dtype=np.float64), axis=0).tolist() if lagged_dcc_corr else []
    )
    out["asym_ratio_by_lag"] = np.mean(np.asarray([r[1:] for r in ratios]), axis=0).tolist()
    out["asym_replica_cos_by_lag"] = (
        np.nanmean(np.asarray(asym_cos, dtype=np.float64), axis=0).tolist() if asym_cos else []
    )
    out["msf_total_A2"] = float(msf_within)
    return out


# --------------------------------------------------------------------------- descriptors


def _descriptors(replicas, static, cfg, lags, dt_ps, segment):
    dcfg = cfg["descriptors"]
    bands_ps = np.asarray(dcfg["bands_ps"], dtype=np.float64)
    bands_frames = bands_ps / dt_ps
    frames = residue_frames(replicas[0].native)
    weights = np.ones(len(lags))
    desc, energy, msf = [], [], []
    kern = kernel_matrix(bands_frames, segment, lags)
    for rep in replicas:
        q, _ = fit_psd_bands(rep.diag.mean(), kern, weights, ridge=float(cfg["spectral"]["ridge"]),
                             max_iter=int(cfg["spectral"]["local_max_iter"]), tol=float(cfg["spectral"]["tol"]),
                             obj_tol=float(cfg["spectral"]["obj_tol"]))
        local = to_local(q, frames)  # (B, N, 3, 3)
        emb = logm_sym6(local, float(dcfg["eps_A2"]))  # (B, N, 6)
        desc.append(np.transpose(emb, (1, 0, 2)).reshape(emb.shape[1], -1))
        energy.append(np.trace(local, axis1=-2, axis2=-1))
        msf.append(rep.summary["msf"])
    energy = np.asarray(energy)  # (n_rep, B, N)
    reliability = []
    for b in range(energy.shape[1]):
        vals = [_pearson(np.log(energy[a, b] + float(dcfg["eps_A2"])), np.log(energy[c, b] + float(dcfg["eps_A2"])))
                for a, c in combinations(range(len(replicas)), 2)]
        reliability.append(_nanmean(vals))
    total = energy.sum(axis=(0, 2))
    arrays = {
        "descriptors": np.asarray(desc, dtype=np.float32),
        "band_energy": energy.astype(np.float32),
        "msf": np.asarray(msf, dtype=np.float32),
        "bands_ps": bands_ps,
        "replicas": np.asarray([r.name for r in replicas]),
        "residue_names": np.asarray(static["residue_names"]),
        "dssp": np.asarray(static["dssp"] if static["dssp"] is not None else ["C"] * len(static["residue_names"])),
    }
    summary = {
        "bands_ps": bands_ps.tolist(),
        "log_energy_corr_by_band": reliability,
        "energy_fraction_by_band": (total / max(total.sum(), 1e-300)).tolist(),
    }
    return arrays, summary


# --------------------------------------------------------------------------- stage 1


def _stage1(replicas, cfg, n_res, lags, pair_lags, residues, dt_ps, segment, log) -> dict:
    scfg, bcfg = cfg["spectral"], cfg["baselines"]
    n_rep = len(replicas)
    bands = band_timescales(segment, float(scfg["bands_per_decade"]), float(scfg["max_timescale_fraction"]),
                            subframe=bool(scfg["subframe_band"]))
    # Every statistic uses length-`segment` windows: one kernel for fits and targets.
    kern = kernel_matrix(bands, segment, lags)
    kern_pair = kernel_matrix(bands, segment, pair_lags)
    weights = np.ones(len(lags))
    max_folds = cfg["folds"].get("max_folds")
    holdouts = list(range(n_rep))[: int(max_folds) if max_folds else n_rep]
    candidates = pca_exp_candidates(segment, int(bcfg["pca_exp_candidates"]))
    kc = kernel_matrix(candidates, segment, lags)
    kc_pair = kernel_matrix(candidates, segment, pair_lags)
    common = dict(ridge=float(scfg["ridge"]), tol=float(scfg["tol"]), obj_tol=float(scfg["obj_tol"]))
    fit_kw = dict(common, max_iter=int(scfg["max_iter"]))
    local_kw = dict(common, max_iter=int(scfg["local_max_iter"]))
    rows_out: list[dict] = []
    fits: list[dict] = []
    dim = 3 * n_res
    k_pc = min(dim, int(scfg["pc_max"]))

    for h in holdouts:
        held = replicas[h]
        fit_set = [r for i, r in enumerate(replicas) if i != h]
        fit_frames = [r.n_frames for r in fit_set]
        target = held.summary
        diag_fit = Lagged.pool([r.diag for r in fit_set]).mean()
        pair_fit = Lagged.pool([r.pair for r in fit_set]).mean()
        base = {"fold": held.name, "n_fit": len(fit_set)}

        def record(method, param, params_per_res, metrics, base=base, **extra):
            row = dict(base, method=method, param=param, params_per_res=params_per_res)
            row.update(extra)
            row.update(metrics)
            rows_out.append(row)

        floor = distortion(summarize(diag_fit, pair_fit), target)
        record("floor", None, float("nan"), floor)

        # Common PC basis from the pooled lag-0 covariance of the fit replicas.
        cov0 = np.zeros((dim, dim))
        for r in fit_set:
            flat = r.dx.reshape(r.n_frames, dim)
            cov0 += flat.T @ flat
        cov0 /= float(sum(fit_frames))
        evals, basis = scipy_eigh(cov0, subset_by_index=[dim - k_pc, dim - 1])
        order = np.argsort(evals)[::-1]
        evals, basis = np.clip(evals[order], 0.0, None), basis[:, order]
        captured = float(evals.sum() / max(np.trace(cov0), 1e-300))
        pc_stats = Lagged.pool([segmented(r.dx.reshape(r.n_frames, dim) @ basis, lags, segment, cross_sums)
                                for r in fit_set]).mean()
        kern_fit = kern_h = kern
        kern_h_pair = kern_pair
        amplitudes, info = fit_psd_bands(pc_stats, kern_fit, weights, **fit_kw)
        comps = spectral_components(amplitudes, basis)
        band_var = np.array([np.trace(a) for a in amplitudes])
        fits.append(
            dict(
                fold=held.name, k_pc=k_pc, pc_captured=captured, n_components=len(comps),
                fit_iterations=info["iterations"], fit_converged=info["converged"], fit_stop=info["stop"],
                fit_rel_residual=info["rel_residual"],
                band_variance_fraction=(band_var / max(band_var.sum(), 1e-300)).tolist(),
            )
        )

        # Spectral codecs.  "subspace": shared top-R spatial subspace with a full
        # PSD core per band (primary).  "band": top-R (band, pattern) pairs.
        # "_local" adds per-residue band tensors fitted to the diagonal residual.
        for codec in scfg["codecs"]:
            truncation, use_local = codec.split("_")[0], codec.endswith("_local")
            local_init = None  # warm start: neighbouring ranks have similar residuals
            limit = k_pc if truncation == "subspace" else len(comps)
            grid = {int(r) for r in scfg["ranks"] if int(r) <= limit}
            for rank in sorted(grid | {limit}):
                if rank == 0 and not use_local:
                    continue
                if truncation == "subspace":
                    model = subspace_model(bands, amplitudes, basis, rank)
                    params = model.n_params(use_local)
                elif truncation == "band":
                    model = model_from_components(bands, comps, rank)
                    params = (dim * rank + rank) / n_res + (6 * len(bands) if use_local else 0)
                else:
                    raise ValueError(f"unknown spectral codec {codec!r}")
                if use_local:
                    low_diag = np.tensordot(kern_fit, model.band_diag(), axes=(1, 0))
                    local, _ = fit_local_residual(diag_fit, low_diag, kern_fit, weights, init=local_init,
                                                  **local_kw)
                    local_init = local
                    model = model.with_local(local)
                diag_p, pair_p = model.predict(kern_h, kern_h_pair, residues)
                record(f"spectral_{codec}", rank, params, distortion((diag_p, pair_p), target),
                       full_rank=bool(rank == limit), on_grid=bool(rank in grid))

        # PCA + one relaxation time per PC.
        amp, best = fit_pca_exp(np.diagonal(pc_stats, axis1=1, axis2=2), basis, [segment], lags, candidates, weights)
        kc_h, kc_h_pair = kc, kc_pair
        for rank in sorted({int(r) for r in bcfg["pca_exp_ranks"] if int(r) <= k_pc}):
            model = pca_exp_model(basis, amp, best, candidates, rank)
            diag_p, pair_p = model.predict(kc_h, kc_h_pair, residues)
            record("pca_exp", rank, (dim * rank + 2 * rank) / n_res, distortion((diag_p, pair_p), target))

        # Truncated DCT of each fit replica's realisation.
        keeps = [int(k) for k in bcfg["dct_keep"] if int(k) < min(fit_frames)]
        rows_cols = pair_rows(residues)
        for keep in keeps:
            diag_items, pair_items = [], []
            for r in fit_set:
                recon = dct_truncate(r.dx.reshape(r.n_frames, dim), keep)
                diag_items.append(segmented(recon.reshape(r.n_frames, n_res, 3), lags, segment, diag_block_sums))
                pair_items.append(segmented(recon[:, rows_cols], pair_lags, segment, cross_sums))
            pred = summarize(Lagged.pool(diag_items).mean(), Lagged.pool(pair_items).mean())
            record("dct", keep, 3.0 * keep * len(fit_set), distortion(pred, target))
        log(f"  fold {held.name}: K={k_pc} captured={captured:.4f} comps={len(comps)} "
            f"fit_iter={info['iterations']} rel_res={info['rel_residual']:.3g}")

    return {
        "bands_frames": bands.tolist(),
        "bands_ps": (bands * dt_ps).tolist(),
        "rows": rows_out,
        "fits": fits,
    }
