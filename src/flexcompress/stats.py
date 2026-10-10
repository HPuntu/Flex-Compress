"""Lagged covariance statistics with exact replica pooling.

Statistics are stored as *sums* and *counts* so that pooling replicas of
different lengths is exact: pooled C(tau) = sum_r S_r(tau) / sum_r (T_r - tau),
the same weighting used by :func:`flexcompress.kernels.pooled_kernel_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Lagged",
    "center",
    "cross_sums",
    "diag_block_sums",
    "lag_grid",
    "pair_rows",
    "pair_subset",
    "segmented",
    "subset_lags",
    "window_starts",
]


def lag_grid(n_frames_min: int, n_lags: int, max_fraction: float) -> np.ndarray:
    """Lag 0 plus a log-spaced integer grid up to ``max_fraction * T_min`` frames."""

    max_lag = int(np.floor(float(max_fraction) * int(n_frames_min)))
    max_lag = min(max(max_lag, 1), int(n_frames_min) - 1)
    if n_lags <= 1 or max_lag < 1:
        return np.array([0], dtype=np.int64)
    values = np.unique(np.round(np.geomspace(1, max_lag, int(n_lags) - 1)).astype(np.int64))
    return np.concatenate([[0], values]).astype(np.int64)


def subset_lags(lags: np.ndarray, n: int) -> np.ndarray:
    """``n`` lags from ``lags`` (always including lag 0), spread across the grid."""

    lags = np.asarray(lags, dtype=np.int64)
    if len(lags) <= n:
        return lags.copy()
    index = np.unique(np.round(np.linspace(0, len(lags) - 1, int(n))).astype(int))
    return lags[index]


def pair_subset(n_res: int, max_residues: int) -> np.ndarray:
    """Evenly spaced residue subset used for pair (coupling) statistics."""

    if n_res <= max_residues:
        return np.arange(n_res, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, n_res - 1, int(max_residues))).astype(np.int64))


def pair_rows(residues: np.ndarray) -> np.ndarray:
    """Flattened (3N) coordinate rows of the given residues, xyz-minor."""

    residues = np.asarray(residues, dtype=np.int64)
    return (residues[:, None] * 3 + np.arange(3)[None, :]).ravel()


def center(x: np.ndarray) -> np.ndarray:
    """Subtract the replica time mean (float64)."""

    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)


@dataclass
class Lagged:
    """Additive lagged second-moment sums ``S(tau)`` with counts ``T - tau``."""

    lags: np.ndarray
    sums: np.ndarray
    counts: np.ndarray

    def mean(self) -> np.ndarray:
        shape = (-1,) + (1,) * (self.sums.ndim - 1)
        return self.sums / self.counts.reshape(shape)

    def select(self, lags: np.ndarray) -> Lagged:
        index = np.searchsorted(self.lags, lags)
        if not np.array_equal(self.lags[index], lags):
            raise KeyError("requested lags are not a subset of the stored grid")
        return Lagged(self.lags[index].copy(), self.sums[index].copy(), self.counts[index].copy())

    @staticmethod
    def pool(items: list[Lagged]) -> Lagged:
        if not items:
            raise ValueError("nothing to pool")
        lags = items[0].lags
        for item in items[1:]:
            if not np.array_equal(item.lags, lags):
                raise ValueError("cannot pool statistics on different lag grids")
        return Lagged(
            lags.copy(),
            np.sum([item.sums for item in items], axis=0),
            np.sum([item.counts for item in items], axis=0),
        )


def diag_block_sums(dx: np.ndarray, lags: np.ndarray) -> Lagged:
    """Per-residue 3x3 lagged sums: S_i(tau)[a, b] = sum_t dx_ia(t) dx_ib(t + tau).

    ``dx`` has shape (T, N, 3).  Memory stays O(T N) by looping over the nine
    components rather than materialising (T, N, 3, 3).
    """

    dx = np.asarray(dx, dtype=np.float64)
    n_frames, n_res, _ = dx.shape
    lags = np.asarray(lags, dtype=np.int64)
    out = np.empty((len(lags), n_res, 3, 3))
    for li, tau in enumerate(lags):
        head = dx[: n_frames - tau]
        tail = dx[tau:]
        for a in range(3):
            for b in range(3):
                out[li, :, a, b] = np.einsum("tn,tn->n", head[..., a], tail[..., b])
    return Lagged(lags.copy(), out, (n_frames - lags).astype(np.float64))


def cross_sums(z: np.ndarray, lags: np.ndarray) -> Lagged:
    """Full lagged sums S(tau) = Z[:T-tau]^T Z[tau:] for a (T, D) signal."""

    z = np.asarray(z, dtype=np.float64)
    n_frames = z.shape[0]
    lags = np.asarray(lags, dtype=np.int64)
    out = np.empty((len(lags), z.shape[1], z.shape[1]))
    for li, tau in enumerate(lags):
        out[li] = z[: n_frames - tau].T @ z[tau:]
    return Lagged(lags.copy(), out, (n_frames - lags).astype(np.float64))


def window_starts(n_frames: int, segment: int) -> np.ndarray:
    """Start frames of the fewest evenly spaced length-``segment`` windows covering a replica."""

    n_frames, segment = int(n_frames), int(segment)
    if segment > n_frames:
        raise ValueError("segment longer than the replica")
    count = int(np.ceil(n_frames / segment))
    return np.unique(np.round(np.linspace(0, n_frames - segment, count)).astype(np.int64))


def segmented(x: np.ndarray, lags: np.ndarray, segment: int, fn) -> Lagged:
    """Pool ``fn(window, lags)`` over windows of one common length, each re-centred.

    Every statistic in a target is computed on windows of the same length, so
    all replicas -- whatever their length -- share one estimator expectation
    (one set of kernels).  A replica exactly ``segment`` long is one window;
    longer replicas use evenly spaced (possibly overlapping) windows so no
    frame is discarded.
    """

    items = [fn(center(x[s : s + int(segment)]), lags) for s in window_starts(len(x), segment)]
    return Lagged.pool(items)
