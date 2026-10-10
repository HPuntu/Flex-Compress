"""Stage-0 diagnostics: PCA compressibility, sampling artefacts, stationarity.

* PCA variance counts (how many PCs reach 50/80/90/95/99 % of variance).
* Cosine content of the top PCs (Hess 2000/2002).  Values near 1 mean the PC
  is a half-cosine in time, the signature of undersampled random diffusion.
* RMSIP between replicas' top-k PC subspaces (replicas share the native
  orientation, so their PCs are directly comparable).
* Fraction of native CA contacts retained (stationarity / unfolding filter).
* Antisymmetric lagged covariance: zero in expectation for reversible
  equilibrium dynamics (Onsager), so its reproducibility across replicas is
  a direct test for irreversibility / non-stationarity.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "asymmetry",
    "cosine_content",
    "native_contact_fraction",
    "top_pcs",
    "rmsip",
    "variance_counts",
]


def top_pcs(
    centered: np.ndarray,
    k: int,
    *,
    exact_max_dim: int = 3000,
    oversample: int = 16,
    power_iters: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Top-k PCA of a centred (T, D) matrix.

    Returns (variances (k,), components (D, k), total variance).  Uses the exact
    covariance eigendecomposition for small D, otherwise a randomized range
    finder with power iterations (Halko et al. 2011).
    """

    x = np.asarray(centered)
    n, dim = x.shape
    k = int(min(k, dim, n))
    total = float(np.sum(x.astype(np.float64) ** 2) / n)
    if dim <= exact_max_dim:
        cov = (x.T.astype(np.float64) @ x.astype(np.float64)) / n
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1][:k]
        return np.clip(w[order], 0.0, None), v[:, order], total
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal((dim, k + oversample)).astype(x.dtype)
    q, _ = np.linalg.qr(x @ probe)
    for _ in range(int(power_iters)):
        q, _ = np.linalg.qr(x.T @ q)
        q, _ = np.linalg.qr(x @ q)
    small = q.T @ x
    _, s, vt = np.linalg.svd(small.astype(np.float64), full_matrices=False)
    return (s[:k] ** 2) / n, vt[:k].T, total


def variance_counts(variances: np.ndarray, total: float, thresholds) -> dict[str, float]:
    """Number of leading PCs reaching each variance fraction (NaN if beyond k)."""

    frac = np.cumsum(np.asarray(variances, dtype=np.float64)) / max(float(total), 1e-300)
    out = {}
    for threshold in thresholds:
        hit = np.nonzero(frac >= float(threshold) - 1e-12)[0]
        out[f"n_pc_{int(round(float(threshold) * 100))}"] = float(hit[0] + 1) if hit.size else float("nan")
    return out


def cosine_content(projections: np.ndarray, n: int) -> np.ndarray:
    """Hess cosine content c_i of the first ``n`` PC projections (T, k)."""

    p = np.asarray(projections, dtype=np.float64)
    n_frames = p.shape[0]
    t = (np.arange(n_frames) + 0.5) / n_frames
    out = []
    for i in range(min(int(n), p.shape[1])):
        cosine = np.cos((i + 1) * np.pi * t)
        num = (2.0 / n_frames) * float(cosine @ p[:, i]) ** 2
        out.append(num / max(float(p[:, i] @ p[:, i]), 1e-300))
    return np.asarray(out)


def rmsip(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Root-mean-square inner product of two orthonormal k-dimensional subspaces."""

    k = int(min(k, a.shape[1], b.shape[1]))
    overlap = a[:, :k].T @ b[:, :k]
    return float(np.sqrt(np.sum(overlap**2) / k))


def native_contact_fraction(
    ca: np.ndarray,
    native_ca: np.ndarray,
    n_samples: int = 50,
    cutoff: float = 8.0,
    tolerance: float = 1.2,
    min_separation: int = 4,
) -> np.ndarray:
    """Q(t) on evenly spaced frames: fraction of native CA contacts retained."""

    native_ca = np.asarray(native_ca, dtype=np.float64)
    n_res = native_ca.shape[0]
    i, j = np.triu_indices(n_res, k=int(min_separation))
    d0 = np.linalg.norm(native_ca[i] - native_ca[j], axis=-1)
    contact = d0 < cutoff
    if not np.any(contact):
        return np.full(min(int(n_samples), len(ca)), np.nan)
    i, j, d0 = i[contact], j[contact], d0[contact]
    frames = np.unique(np.round(np.linspace(0, len(ca) - 1, int(n_samples))).astype(int))
    xyz = np.asarray(ca[frames], dtype=np.float64)
    d = np.linalg.norm(xyz[:, i] - xyz[:, j], axis=-1)
    return np.mean(d < tolerance * d0[None, :], axis=1)


def asymmetry(pair_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Antisymmetric parts (Lp, D, D) and ratios ||C_a|| / ||C_s|| per lag."""

    pair_mean = np.asarray(pair_mean, dtype=np.float64)
    anti = 0.5 * (pair_mean - np.swapaxes(pair_mean, -1, -2))
    sym = 0.5 * (pair_mean + np.swapaxes(pair_mean, -1, -2))
    ratio = np.linalg.norm(anti, axis=(-2, -1)) / np.maximum(np.linalg.norm(sym, axis=(-2, -1)), 1e-300)
    return anti, ratio
