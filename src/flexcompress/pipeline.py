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

from . import accel
from . import diagnostics as dg
from .baselines import fit_pca_exp, pca_exp_candidates, pca_exp_model
from .data import Target, load_replica, load_static
from .frames import residue_frames, to_local
from .kernels import kernel_matrix
from .metrics import METRICS, distortion, summarize
from .spectral import (
    band_timescales,
    model_from_components,
    spectral_components,
    subspace_model,
)
from .stats import (
    Lagged,
    center,
    lag_grid,
    pair_rows,
    pair_subset,
    subset_lags,
    window_starts,
)

__all__ = ["config_hash", "process_target", "resolve_stride"]

RESULT_SCHEMA = 1


def config_hash(cfg: dict) -> str:
    """Hash of the scientific configuration (paths and run bookkeeping excluded)."""

    keep = {k: v for k, v in cfg.items()
            if k not in {"archive_root", "index_path", "run", "selection", "split", "compute"}}
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
    bb_pooled = accel.OuterSum(12 * n_res) if bb_pooled_ok else None
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
            var, _, total = accel.top_pcs(centred, int(diag_cfg["pca_top"]), seed=int(cfg["seed"]))
            bb_rep.append(dg.variance_counts(var, total, thresholds))
            if bb_pooled is not None:
                bb_pooled.add(flat)
            del flat, centred
        del loaded
        replicas.append(rep)
    timing["load"] = time.monotonic() - started

    result = {
        "schema": RESULT_SCHEMA,
        "status": "ok",
        "target": target.to_dict(),
        "config_hash": config_hash(cfg),
        "compute": accel.info(),
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
        rep.diag = accel.lagged_diag(rep.dx, lags, segment)
        rep.pair = accel.lagged_cross(rep.dx.reshape(rep.n_frames, 3 * n_res)[:, rows], pair_lags, segment)
        rep.summary = summarize(rep.diag.mean(), rep.pair.mean())
    timing["statistics"] = time.monotonic() - t

    descriptors = None
    if cfg["stages"].get("stage0", True):
        t = time.monotonic()
        result["stage0"] = _stage0(replicas, static, cfg, n_res, residues, pair_lags, bb_rep,
                                   bb_pooled)
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
        var, vec, total = accel.top_pcs(flat, int(dcfg["pca_top"]), seed=seed)
        per_rep_counts.append(dg.variance_counts(var, total, thresholds))
        n_cos = int(dcfg["cosine_pcs"])
        cosines.append(dg.cosine_content(flat @ vec[:, :n_cos], n_cos).tolist())
        comps.append(vec)
    out["ca_pca_per_replica"] = per_rep_counts
    out["ca_pca_per_replica_mean"] = {k: _nanmean([c[k] for c in per_rep_counts]) for k in per_rep_counts[0]}
    weights_t = np.array([rep.n_frames for rep in replicas], dtype=np.float64)
    grand = (np.tensordot(weights_t, np.stack([rep.mean for rep in replicas]), axes=(0, 0)) / weights_t.sum()).reshape(-1)
    w = accel.eigvalsh_gram([rep.dx.reshape(rep.n_frames, -1) + (rep.mean.reshape(-1) - grand)[None, :]
                             for rep in replicas])
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
        w_bb, trace_bb = bb_pooled.top_eigvals(int(dcfg["pca_top"]))
        out["bb_pca_pooled"] = dg.variance_counts(w_bb, trace_bb, thresholds)

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
    # One batched fit for all replicas: (L, n_rep, N, 3, 3) -> (B, n_rep, N, 3, 3).
    q_all, _ = accel.fit_psd_bands(np.stack([rep.diag.mean() for rep in replicas], axis=1), kern, weights,
                                   ridge=float(cfg["spectral"]["ridge"]),
                                   max_iter=int(cfg["spectral"]["local_max_iter"]),
                                   tol=float(cfg["spectral"]["tol"]), obj_tol=float(cfg["spectral"]["obj_tol"]))
    for i, rep in enumerate(replicas):
        local = to_local(q_all[:, i], frames)  # (B, N, 3, 3)
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
    """Leave-one-replica-out rate-distortion, organised for batched (GPU) solves.

    Every statistic shares one window length, so all folds share one kernel
    matrix.  That lets the collective fits of all folds run as a single batched
    ADMM solve, and the per-residue local fits of every (fold, codec, rank) as
    another, instead of dozens of small sequential solves.
    """

    scfg, bcfg = cfg["spectral"], cfg["baselines"]
    n_rep = len(replicas)
    bands = band_timescales(segment, float(scfg["bands_per_decade"]), float(scfg["max_timescale_fraction"]),
                            subframe=bool(scfg["subframe_band"]))
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
    dim = 3 * n_res
    k_pc = min(dim, int(scfg["pc_max"]))
    rows_cols = pair_rows(residues)

    # Phase 1: per-fold statistics, floor and PC basis.
    folds = []
    for h in holdouts:
        held = replicas[h]
        fit_set = [r for i, r in enumerate(replicas) if i != h]
        diag_fit = Lagged.pool([r.diag for r in fit_set]).mean()
        pair_fit = Lagged.pool([r.pair for r in fit_set]).mean()
        evals, basis, trace0 = accel.gram_top_eigh([r.dx.reshape(r.n_frames, dim) for r in fit_set], k_pc)
        pc_stats = Lagged.pool([accel.lagged_cross(r.dx.reshape(r.n_frames, dim), lags, segment, basis=basis)
                                for r in fit_set]).mean()
        fold = {"held": held, "fit_set": fit_set, "target": held.summary, "diag_fit": diag_fit, "basis": basis,
                "captured": float(evals.sum() / max(trace0, 1e-300)), "pc_stats": pc_stats, "rows": []}
        fold["rows"].append(_row(fold, "floor", None, float("nan"), distortion(summarize(diag_fit, pair_fit),
                                                                              held.summary)))
        folds.append(fold)

    # Phase 2: all folds' collective fits in one batched solve, (L, F, K, K) -> (B, F, K, K).
    amplitudes, info = accel.fit_psd_bands(np.stack([f["pc_stats"] for f in folds], axis=1), kern, weights,
                                           **fit_kw)
    fits = []
    for i, fold in enumerate(folds):
        fold["amplitudes"] = amplitudes[:, i]
        fold["comps"] = spectral_components(fold["amplitudes"], fold["basis"])
        band_var = np.array([np.trace(a) for a in fold["amplitudes"]])
        fits.append(dict(
            fold=fold["held"].name, k_pc=k_pc, pc_captured=fold["captured"], n_components=len(fold["comps"]),
            fit_iterations=info["iterations"], fit_converged=info["converged"], fit_stop=info["stop"],
            fit_rel_residual=info["rel_residual"], fit_batched_folds=len(folds),
            band_variance_fraction=(band_var / max(band_var.sum(), 1e-300)).tolist(),
        ))

    # Phase 3: build every spectral model ("subspace": shared top-R spatial
    # subspace with a PSD core per band, primary; "band": top-R (band, pattern)
    # pairs; "_local": + per-residue band tensors) and solve all local residual
    # fits together, (L, M, N, 3, 3) -> (B, M, N, 3, 3).
    pending, residuals = [], []
    for i, fold in enumerate(folds):
        for codec in scfg["codecs"]:
            truncation, use_local = codec.split("_")[0], codec.endswith("_local")
            limit = k_pc if truncation == "subspace" else len(fold["comps"])
            grid = {int(r) for r in scfg["ranks"] if int(r) <= limit}
            for rank in sorted(grid | {limit}):
                if rank == 0 and not use_local:
                    continue
                if truncation == "subspace":
                    model = subspace_model(bands, fold["amplitudes"], fold["basis"], rank)
                    params = model.n_params(use_local)
                elif truncation == "band":
                    model = model_from_components(bands, fold["comps"], rank)
                    params = (dim * rank + rank) / n_res + (6 * len(bands) if use_local else 0)
                else:
                    raise ValueError(f"unknown spectral codec {codec!r}")
                slot = None
                if use_local:
                    slot = len(residuals)
                    residuals.append(fold["diag_fit"] - np.tensordot(kern, accel.band_diag(model), axes=(1, 0)))
                pending.append((i, codec, rank, model, params, slot, rank == limit, rank in grid))
    local_all = None
    if residuals:
        local_all, _ = accel.fit_psd_bands(np.stack(residuals, axis=1), kern, weights, **local_kw)
    for i, codec, rank, model, params, slot, full, on_grid in pending:
        fold = folds[i]
        if slot is not None:
            model = model.with_local(local_all[:, slot])
        diag_p, pair_p = accel.predict(model, kern, kern_pair, residues)
        fold["rows"].append(_row(fold, f"spectral_{codec}", rank, params, distortion((diag_p, pair_p), fold["target"]),
                                 full_rank=bool(full), on_grid=bool(on_grid)))

    # Phase 4: baselines.
    for fold in folds:
        # PCA + one relaxation time per PC.
        amp, best = fit_pca_exp(np.diagonal(fold["pc_stats"], axis1=1, axis2=2), fold["basis"], [segment], lags,
                                candidates, weights)
        for rank in sorted({int(r) for r in bcfg["pca_exp_ranks"] if int(r) <= k_pc}):
            model = pca_exp_model(fold["basis"], amp, best, candidates, rank)
            diag_p, pair_p = accel.predict(model, kc, kc_pair, residues)
            fold["rows"].append(_row(fold, "pca_exp", rank, (dim * rank + 2 * rank) / n_res,
                                     distortion((diag_p, pair_p), fold["target"])))
        # Truncated DCT of each fit replica's realisation.
        fit_set = fold["fit_set"]
        for keep in [int(k) for k in bcfg["dct_keep"] if int(k) < min(r.n_frames for r in fit_set)]:
            items = [accel.dct_lagged(r.dx.reshape(r.n_frames, dim), keep, lags, pair_lags, rows_cols, n_res, segment)
                     for r in fit_set]
            pred = summarize(Lagged.pool([d for d, _ in items]).mean(), Lagged.pool([q for _, q in items]).mean())
            fold["rows"].append(_row(fold, "dct", keep, 3.0 * keep * len(fit_set), distortion(pred, fold["target"])))
        log(f"  fold {fold['held'].name}: K={k_pc} captured={fold['captured']:.4f} comps={len(fold['comps'])} "
            f"fit_iter={info['iterations']} rel_res={info['rel_residual']:.3g} local_fits={len(residuals)} "
            f"[{accel.info()['backend']}]")

    return {
        "bands_frames": bands.tolist(),
        "bands_ps": (bands * dt_ps).tolist(),
        "rows": [row for fold in folds for row in fold["rows"]],
        "fits": fits,
    }


def _row(fold: dict, method: str, param, params_per_res: float, metrics: dict, **extra) -> dict:
    row = {"fold": fold["held"].name, "n_fit": len(fold["fit_set"]), "method": method, "param": param,
           "params_per_res": params_per_res}
    row.update(extra)
    row.update(metrics)
    return row
