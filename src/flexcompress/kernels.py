"""Finite-length expected-estimator kernels for relaxation bands.

Every lagged covariance in this project is estimated per replica as

    C_hat(tau) = 1/(T - tau) * sum_{s < T - tau} (x_s - xbar)(x_{s+tau} - xbar)^T

with the replica's own time mean xbar.  Subtracting xbar biases C_hat(tau)
downward for processes whose relaxation time is not << T (and removes a
non-decaying component entirely).  Rather than correct the data, the model
side uses the *expected value of this estimator* for each relaxation band.
For a stationary scalar process with autocovariance gamma(h):

    E[C_hat(tau)] = gamma(tau) - (1/(T-tau)) sum_{s<T-tau} (c_s + c_{s+tau}) + Var(xbar)
    c_s = Cov(x_s, xbar) = (1/T) sum_u gamma(|s-u|),   Var(xbar) = mean_s c_s

Fitted band amplitudes are therefore process-level amplitudes, and a model
can be compared like-for-like with a held-out replica of a different length.
All quantities are O(T) per band.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "autocov_exponential",
    "expected_lagged_cov",
    "kernel_matrix",
    "pooled_kernel_matrix",
]


def autocov_exponential(timescale_frames: float, n_frames: int) -> np.ndarray:
    """Unit-variance autocovariance gamma(h), h = 0..T-1, of one relaxation band.

    ``timescale_frames == 0`` is the sub-frame (white) band: all variance that
    decorrelates within one stored frame interval.
    """

    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    t = float(timescale_frames)
    if t < 0 or not np.isfinite(t):
        raise ValueError("timescale must be finite and non-negative")
    if t == 0.0:
        gamma = np.zeros(n_frames)
        gamma[0] = 1.0
        return gamma
    return np.exp(-np.arange(n_frames, dtype=np.float64) / t)


def expected_lagged_cov(gamma: np.ndarray, n_frames: int, lags) -> np.ndarray:
    """Expected mean-subtracted lagged covariance estimator, see module docstring."""

    n_frames = int(n_frames)
    gamma = np.asarray(gamma, dtype=np.float64)[:n_frames]
    if gamma.shape[0] < n_frames:
        raise ValueError("gamma must cover lags 0..T-1")
    lags = np.asarray(lags, dtype=np.int64)
    if lags.size and (lags.min() < 0 or lags.max() >= n_frames):
        raise ValueError("lags must lie in [0, T)")
    cumulative = np.cumsum(gamma)  # G[m] = sum_{h<=m} gamma(h)
    s = np.arange(n_frames)
    c = (cumulative[s] + cumulative[n_frames - 1 - s] - gamma[0]) / n_frames
    var_mean = float(c.mean())
    prefix = np.concatenate([[0.0], np.cumsum(c)])
    count = n_frames - lags
    head = prefix[count]  # sum_{s < T - tau} c_s
    tail = prefix[n_frames] - prefix[lags]  # sum_{s < T - tau} c_{s + tau}
    return gamma[lags] - (head + tail) / count + var_mean


def kernel_matrix(timescales_frames, n_frames: int, lags) -> np.ndarray:
    """(L, B) matrix of expected estimator values for unit-variance bands."""

    timescales = np.atleast_1d(np.asarray(timescales_frames, dtype=np.float64))
    out = np.empty((len(np.atleast_1d(lags)), timescales.size))
    for b, t in enumerate(timescales):
        out[:, b] = expected_lagged_cov(autocov_exponential(t, n_frames), n_frames, lags)
    return out


def pooled_kernel_matrix(timescales_frames, frame_counts, lags) -> np.ndarray:
    """Kernel for replica-pooled estimators, weighted by (T_r - tau) like the data."""

    lags = np.asarray(lags, dtype=np.int64)
    counts = [int(t) for t in frame_counts]
    if not counts:
        raise ValueError("at least one replica length is required")
    num = 0.0
    den = 0.0
    for n_frames in counts:
        weight = (n_frames - lags).astype(np.float64)[:, None]
        num = num + weight * kernel_matrix(timescales_frames, n_frames, lags)
        den = den + weight
    return num / den
