"""Distortion metrics between two sets of lagged second-moment statistics.

A statistics set is ``(diag, pair)``:
    diag  (L, N, 3, 3)   per-residue lagged covariance blocks, lag 0 first
    pair  (Lp, 3S, 3S)   full lagged covariance on a residue subset, lag 0 first

Metrics (all 0 for identical inputs, lower is better):
    msf_rel           relative L2 error of per-residue mean-square fluctuation
    log_msf_rmse      RMSE of log MSF (scale-free flexibility profile)
    aniso             mean Frobenius error of trace-normalised 3x3 shape tensors
    acf_rmse          RMSE of per-residue normalised autocorrelation over lags
    dir_acf_rmse      as above but direction resolved (all 9 tensor components)
    dccm_rmse         RMSE of the lag-0 dynamic cross-correlation matrix
    lagged_dccm_rmse  RMSE of lagged cross-correlations (lags > 0, i != j)
    frob_rel          relative Frobenius error of the full pair blocks, all lags
"""

from __future__ import annotations

import numpy as np

__all__ = ["METRICS", "distortion", "summarize"]

METRICS = (
    "msf_rel",
    "log_msf_rmse",
    "aniso",
    "acf_rmse",
    "dir_acf_rmse",
    "dccm_rmse",
    "lagged_dccm_rmse",
    "frob_rel",
)

_TINY = 1e-12


def summarize(diag: np.ndarray, pair: np.ndarray) -> dict[str, np.ndarray]:
    """Derived normalised quantities used by the metrics and reliability code."""

    diag = np.asarray(diag, dtype=np.float64)
    pair = np.asarray(pair, dtype=np.float64)
    msf = np.trace(diag[0], axis1=-2, axis2=-1)
    msf_safe = np.maximum(msf, _TINY)
    shape = diag[0] / msf_safe[:, None, None]
    acf = np.trace(diag[1:], axis1=-2, axis2=-1) / msf_safe[None, :]
    dir_acf = diag[1:] / msf_safe[None, :, None, None]
    n_lp, dim, _ = pair.shape
    n_sub = dim // 3
    tr_blocks = _trace_blocks(pair.reshape(n_lp, n_sub, 3, n_sub, 3))
    var =np.maximum(np.diagonal(tr_blocks[0]), _TINY)
    dcc = tr_blocks / np.sqrt(var[None, :, None] * var[None, None, :])
    return {
        "msf": msf,
        "shape": shape,
        "acf": acf,
        "dir_acf": dir_acf,
        "dcc": dcc,
        "pair": pair,
    }


def _trace_blocks(blocks: np.ndarray) -> np.ndarray:
    """(Lp, S, 3, S, 3) -> (Lp, S, S) traces of the 3x3 residue-pair blocks."""

    return np.einsum("liaja->lij", blocks)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def distortion(pred: dict | tuple, target: dict | tuple) -> dict[str, float]:
    """All metrics for a prediction against a target (each a summary or (diag, pair))."""

    p = pred if isinstance(pred, dict) else summarize(*pred)
    t = target if isinstance(target, dict) else summarize(*target)
    out: dict[str, float] = {}
    out["msf_rel"] = float(np.linalg.norm(p["msf"] - t["msf"]) / max(np.linalg.norm(t["msf"]), _TINY))
    out["log_msf_rmse"] = _rms(
        np.log(np.maximum(p["msf"], _TINY)) - np.log(np.maximum(t["msf"], _TINY))
    )
    out["aniso"] = float(np.mean(np.linalg.norm(p["shape"] - t["shape"], axis=(-2, -1))))
    out["acf_rmse"] = _rms(p["acf"] - t["acf"])
    out["dir_acf_rmse"] = _rms(p["dir_acf"] - t["dir_acf"])
    n_sub = t["dcc"].shape[-1]
    upper = np.triu_indices(n_sub, k=1)
    out["dccm_rmse"] = _rms(p["dcc"][0][upper] - t["dcc"][0][upper])
    if t["dcc"].shape[0] > 1:
        off = ~np.eye(n_sub, dtype=bool)
        out["lagged_dccm_rmse"] = _rms((p["dcc"][1:] - t["dcc"][1:])[:, off])
    else:
        out["lagged_dccm_rmse"] = float("nan")
    num = float(np.sum((p["pair"] - t["pair"]) ** 2))
    den = float(np.sum(t["pair"] ** 2))
    out["frob_rel"] = float(np.sqrt(num / max(den, _TINY)))
    return out
