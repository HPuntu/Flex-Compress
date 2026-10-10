"""Classical comparison codecs for the rate-distortion study.

``pca_exp``  Static PCA with one fitted relaxation time per PC ("PCA + temporal
             fit").  A statistical codec, like the spectral model, but spatial
             patterns are not allowed to depend on timescale.
``dct``      Truncated DCT of each replica's realisation (the DynaMode-style
             spectral representation).  A *realisation* codec: its rate grows
             with trajectory length because it stores one run's noise too.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

from .kernels import pooled_kernel_matrix
from .spectral import SpectralModel

__all__ = ["dct_truncate", "fit_pca_exp", "pca_exp_candidates"]


def pca_exp_candidates(n_frames_min: int, count: int = 64) -> np.ndarray:
    """Candidate single-exponential times (frames), including sub-frame."""

    return np.concatenate([[0.0], np.geomspace(0.25, 2.0 * int(n_frames_min), int(count) - 1)])


def fit_pca_exp(
    pc_autocov: np.ndarray,
    basis: np.ndarray,
    frame_counts,
    lags: np.ndarray,
    candidates: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Best single relaxation time and amplitude for every PC.

    ``pc_autocov`` is (L, K): the pooled lagged autocovariance of each PC.
    Returns (amplitude (K,), candidate index (K,)).
    """

    y = np.asarray(pc_autocov, dtype=np.float64)
    kern = pooled_kernel_matrix(candidates, frame_counts, lags)  # (L, n_cand)
    w = np.ones(len(lags)) if weights is None else np.asarray(weights, dtype=np.float64)
    num = kern.T @ (w[:, None] * y)  # (n_cand, K)
    den = np.einsum("lc,l,lc->c", kern, w, kern)[:, None]
    amp = np.clip(num / den, 0.0, None)
    sse = np.einsum("l,lk->k", w, y**2)[None, :] - 2.0 * amp * num + amp**2 * den
    best = np.argmin(sse, axis=0)
    columns = np.arange(y.shape[1])
    return amp[best, columns], best


def pca_exp_model(
    basis: np.ndarray, amplitude: np.ndarray, best: np.ndarray, candidates: np.ndarray, rank: int
) -> SpectralModel:
    rank = min(int(rank), basis.shape[1])
    factors = basis[:, :rank] * np.sqrt(amplitude[:rank])[None, :]
    return SpectralModel(np.asarray(candidates, dtype=np.float64), factors, best[:rank].astype(int))


def dct_truncate(signal: np.ndarray, keep: int) -> np.ndarray:
    """Keep the lowest ``keep`` DCT-II coefficients of each column of (T, D)."""

    coeffs = dct(np.asarray(signal, dtype=np.float64), type=2, norm="ortho", axis=0)
    coeffs[int(keep):] = 0.0
    return idct(coeffs, type=2, norm="ortho", axis=0)


__all__.append("pca_exp_model")
