"""Relaxation-spectrum ("dynamical fingerprint") models of lagged covariance.

For reversible equilibrium dynamics sampled on an overdamped timescale the
symmetric lagged covariance of the aligned coordinates has the form

    C(tau) = sum_b exp(-tau / t_b) A_b,      A_b PSD,

one positive semi-definite amplitude matrix per relaxation band.  With a fixed
log-spaced band grid this is a convex, PSD-constrained least-squares problem,
solved here by ADMM with eigenvalue projection.  Adjacent half-decade bands
are not separately identifiable (an inverse-Laplace problem), so read the
fitted spectrum as smoothed: variance in a timescale window is well
determined, the split between neighbouring bands is not.  The data-side estimator bias
is handled by fitting with expected-estimator kernels (``kernels.py``).

A fitted model is compressed by keeping the top-R eigencomponents across all
bands (each a 3N spatial pattern tied to one timescale; the "collective"
part), optionally plus a per-residue 3x3 PSD tensor per band fitted to the
diagonal-block residual (the "local" part).  The two parts mirror the
residue-token / collective-token split of the eventual learned latent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import pair_rows

__all__ = [
    "Components",
    "SpectralModel",
    "band_timescales",
    "fit_local_residual",
    "fit_psd_bands",
    "proj_psd",
    "spectral_components",
]


def band_timescales(
    n_frames_min: int,
    bands_per_decade: float,
    max_fraction: float,
    subframe: bool = True,
    min_frames: float = 1.0,
) -> np.ndarray:
    """Band relaxation times in frames: [0 (sub-frame)] + log grid [1, f*T_min]."""

    t_max = max(float(max_fraction) * int(n_frames_min), float(min_frames))
    decades = np.log10(t_max / float(min_frames))
    count = max(int(np.ceil(decades * float(bands_per_decade))) + 1, 1)
    grid = np.geomspace(float(min_frames), t_max, count) if count > 1 else np.array([min_frames])
    return np.concatenate([[0.0], grid]) if subframe else grid


def proj_psd(matrix: np.ndarray) -> np.ndarray:
    """Project (stacks of) symmetric matrices onto the PSD cone."""

    sym = 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
    w, v = np.linalg.eigh(sym)
    w = np.clip(w, 0.0, None)
    return (v * w[..., None, :]) @ np.swapaxes(v, -1, -2)


def fit_psd_bands(
    targets: np.ndarray,
    kernels: np.ndarray,
    weights: np.ndarray | None = None,
    ridge: float = 1e-6,
    max_iter: int = 500,
    tol: float = 1e-7,
    init: np.ndarray | None = None,
    obj_tol: float = 1e-6,
    check_every: int = 10,
) -> tuple[np.ndarray, dict]:
    """Solve min_A sum_l w_l ||sum_b K_lb A_b - C_l||_F^2,  A_b PSD.

    ``targets`` has shape (L, *batch, d, d) (symmetrised internally);
    ``kernels`` (L, B).  Returns ``A`` with shape (B, *batch, d, d).
    ``init`` warm-starts from a previous solution of a similar problem
    (e.g. the local fit at the neighbouring rank).

    Stops when the iterates settle (``tol``) or, more usefully, when the
    feasible objective has improved by less than ``obj_tol`` (relative) over
    ``check_every`` iterations while the primal residual is below 1e-3.  The
    fitted covariance converges long before the (non-identifiable) split of
    variance between adjacent bands, so the objective test saves most of the
    iterations without changing any prediction materially.
    """

    targets = np.asarray(targets, dtype=np.float64)
    kernels = np.asarray(kernels, dtype=np.float64)
    n_lags, n_bands = kernels.shape
    if targets.shape[0] != n_lags:
        raise ValueError("targets and kernels disagree on the number of lags")
    w = np.ones(n_lags) if weights is None else np.asarray(weights, dtype=np.float64)
    sym = 0.5 * (targets + np.swapaxes(targets, -1, -2))
    scale = float(np.abs(sym[0]).max()) if np.abs(sym[0]).max() > 0 else 1.0
    sym = sym / scale

    gram = kernels.T @ (w[:, None] * kernels)
    gram = gram + float(ridge) * np.trace(gram) / n_bands * np.eye(n_bands)
    rhs = np.tensordot((kernels * w[:, None]).T, sym, axes=(1, 0))

    # ADMM on  min f(A) + I_psd(Z)  s.t. A = Z.  The A-step only couples the
    # band axis, (G + rho I)^-1 is B x B; the Z-step is one PSD projection per
    # band.  Adjacent exponential bands make G ill-conditioned (cond ~ 1e3-1e4),
    # where ADMM with residual-balanced rho converges ~10x faster than FISTA.
    rho = float(np.trace(gram)) / n_bands
    if init is not None and np.shape(init) == rhs.shape:
        current = proj_psd(np.asarray(init, dtype=np.float64) / scale)
    else:
        current = proj_psd(np.tensordot(np.linalg.inv(gram), rhs, axes=(1, 0)))
    dual = np.zeros_like(current)
    eye = np.eye(n_bands)
    inverse = np.linalg.inv(gram + rho * eye)
    total = float(np.sum(w.reshape((-1,) + (1,) * (sym.ndim - 1)) * sym**2))
    flat_rhs = rhs.reshape(n_bands, -1)

    def rel_objective(z):
        flat = z.reshape(n_bands, -1)
        value = float(np.sum(gram * (flat @ flat.T)) - 2.0 * np.sum(flat * flat_rhs)) + total
        return value / max(total, 1e-300)

    iterations = 0
    rel_change = np.inf
    last_obj = np.inf
    stop = "max_iter"
    for iterations in range(1, int(max_iter) + 1):
        primal = np.tensordot(inverse, rhs + rho * (current - dual), axes=(1, 0))
        previous = current
        current = proj_psd(primal + dual)
        dual = dual + primal - current
        norm = max(float(np.linalg.norm(current)), 1e-300)
        r_primal = float(np.linalg.norm(primal - current)) / norm
        r_dual = float(np.linalg.norm(current - previous)) / norm
        rel_change = max(r_primal, r_dual)
        if rel_change < tol:
            stop = "iterates"
            break
        if obj_tol and iterations % int(check_every) == 0:
            obj = rel_objective(current)
            if r_primal < 1e-3 and abs(last_obj - obj) <= obj_tol * max(abs(obj), 1e-12):
                stop = "objective"
                break
            last_obj = obj
        if r_primal > 10.0 * r_dual:
            rho, dual = rho * 2.0, dual / 2.0
            inverse = np.linalg.inv(gram + rho * eye)
        elif r_dual > 10.0 * r_primal:
            rho, dual = rho / 2.0, dual * 2.0
            inverse = np.linalg.inv(gram + rho * eye)

    residual = np.tensordot(kernels, current, axes=(1, 0)) - sym
    objective = float(np.sum(w.reshape((-1,) + (1,) * (residual.ndim - 1)) * residual**2))
    info = {
        "iterations": iterations,
        "rel_change": rel_change,
        "converged": stop != "max_iter",
        "stop": stop,
        "rel_residual": float(np.sqrt(objective / max(total, 1e-300))),
    }
    return current * scale, info


@dataclass
class Components:
    """Eigencomponents of fitted band amplitudes, sorted by variance."""

    variance: np.ndarray  # (M,)
    band: np.ndarray  # (M,) band index
    vectors: np.ndarray  # (3N, M) unit spatial patterns

    def __len__(self) -> int:
        return int(self.variance.size)


def spectral_components(amplitudes: np.ndarray, basis: np.ndarray, rel_floor: float = 1e-10) -> Components:
    """Eigendecompose each A_b (in the PC basis) and lift to Cartesian space."""

    w_all, b_all, v_all = [], [], []
    peak = 0.0
    eig = []
    for b in range(amplitudes.shape[0]):
        w, v = np.linalg.eigh(0.5 * (amplitudes[b] + amplitudes[b].T))
        eig.append((w, v))
        peak = max(peak, float(w.max(initial=0.0)))
    for b, (w, v) in enumerate(eig):
        keep = w > rel_floor * max(peak, 1e-300)
        if np.any(keep):
            w_all.append(w[keep])
            b_all.append(np.full(int(keep.sum()), b))
            v_all.append(basis @ v[:, keep])
    if not w_all:
        return Components(np.zeros(0), np.zeros(0, dtype=int), np.zeros((basis.shape[0], 0)))
    variance = np.concatenate(w_all)
    order = np.argsort(variance)[::-1]
    return Components(
        variance[order],
        np.concatenate(b_all)[order].astype(int),
        np.concatenate(v_all, axis=1)[:, order],
    )


@dataclass
class SpectralModel:
    """C(tau) = sum_k K_k(tau) [ sum_{c: kid_c = k} u_c u_c^T + blockdiag(local_k) ]."""

    timescales: np.ndarray  # (Kt,) frames
    factors: np.ndarray  # (3N, R) variance-scaled spatial patterns
    kid: np.ndarray  # (R,) index into timescales
    local: np.ndarray | None = None  # (Kt, N, 3, 3)

    @property
    def n_res(self) -> int:
        return self.factors.shape[0] // 3

    @property
    def rank(self) -> int:
        return int(self.factors.shape[1])

    def band_diag(self) -> np.ndarray:
        n_res, n_t = self.n_res, len(self.timescales)
        out = np.zeros((n_t, n_res, 3, 3))
        per_res = self.factors.reshape(n_res, 3, -1)
        for k in np.unique(self.kid):
            sel = self.kid == k
            u = per_res[:, :, sel]
            out[k] = np.einsum("nir,njr->nij", u, u)
        if self.local is not None:
            out = out + self.local
        return out

    def band_pair(self, residues: np.ndarray) -> np.ndarray:
        rows = pair_rows(residues)
        n_t, dim = len(self.timescales), rows.size
        out = np.zeros((n_t, dim, dim))
        sub = self.factors[rows]
        for k in np.unique(self.kid):
            sel = self.kid == k
            out[k] = sub[:, sel] @ sub[:, sel].T
        if self.local is not None:
            for j, residue in enumerate(np.asarray(residues)):
                out[:, 3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += self.local[:, residue]
        return out

    def predict(self, kernel_diag: np.ndarray, kernel_pair: np.ndarray, residues: np.ndarray):
        """Expected-estimator diag blocks (L, N, 3, 3) and pair blocks (Lp, 3S, 3S)."""

        diag = np.tensordot(kernel_diag, self.band_diag(), axes=(1, 0))
        pair = np.tensordot(kernel_pair, self.band_pair(residues), axes=(1, 0))
        return diag, pair

    def truncated(self, rank: int) -> SpectralModel:
        return SpectralModel(self.timescales, self.factors[:, :rank], self.kid[:rank], None)

    def with_local(self, local: np.ndarray | None) -> SpectralModel:
        return SpectralModel(self.timescales, self.factors, self.kid, local)


def model_from_components(timescales: np.ndarray, comps: Components, rank: int | None = None) -> SpectralModel:
    rank = len(comps) if rank is None else min(int(rank), len(comps))
    factors = comps.vectors[:, :rank] * np.sqrt(comps.variance[:rank])[None, :]
    return SpectralModel(np.asarray(timescales, dtype=np.float64), factors, comps.band[:rank].copy())


@dataclass
class SubspaceModel:
    """C(tau) = sum_k K_k(tau) [ U M_k U^T + blockdiag(local_k) ],  U (3N, R) orthonormal.

    One shared R-dimensional spatial subspace (the top-R principal directions
    of the fitted total amplitude sum_k A_k), with a full PSD R x R core per
    relaxation band.  Patterns may still differ between timescales inside the
    subspace, but a collective motion whose variance leaks across adjacent
    bands costs one dimension, not one per band.
    """

    timescales: np.ndarray  # (Kt,) frames
    basis: np.ndarray  # (3N, R)
    cores: np.ndarray  # (Kt, R, R)
    local: np.ndarray | None = None  # (Kt, N, 3, 3)

    @property
    def n_res(self) -> int:
        return self.basis.shape[0] // 3

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def band_diag(self) -> np.ndarray:
        n_res, n_t = self.n_res, len(self.timescales)
        if self.rank == 0:
            out = np.zeros((n_t, n_res, 3, 3))
        else:
            per_res = self.basis.reshape(n_res, 3, self.rank)
            out = np.einsum("nir,krs,njs->knij", per_res, self.cores, per_res, optimize=True)
        if self.local is not None:
            out = out + self.local
        return out

    def band_pair(self, residues: np.ndarray) -> np.ndarray:
        rows = pair_rows(residues)
        sub = self.basis[rows]
        out = np.einsum("ir,krs,js->kij", sub, self.cores, sub, optimize=True) if self.rank else \
            np.zeros((len(self.timescales), rows.size, rows.size))
        if self.local is not None:
            for j, residue in enumerate(np.asarray(residues)):
                out[:, 3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += self.local[:, residue]
        return out

    predict = SpectralModel.predict

    def with_local(self, local: np.ndarray | None) -> SubspaceModel:
        return SubspaceModel(self.timescales, self.basis, self.cores, local)

    def n_params(self, include_local: bool) -> float:
        rank, n_t = self.rank, len(self.timescales)
        per_protein = 3 * self.n_res * rank + n_t * rank * (rank + 1) / 2
        return per_protein / self.n_res + (6 * n_t if include_local else 0)


def subspace_model(timescales: np.ndarray, amplitudes: np.ndarray, basis: np.ndarray, rank: int) -> SubspaceModel:
    """Truncate fitted band amplitudes (B, K, K) in a PC basis (3N, K) to rank R."""

    total = 0.5 * (amplitudes.sum(axis=0) + amplitudes.sum(axis=0).T)
    w, v = np.linalg.eigh(total)
    order = np.argsort(w)[::-1][: int(rank)]
    sub = v[:, order]  # (K, R)
    cores = np.einsum("kr,bkl,ls->brs", sub, amplitudes, sub, optimize=True)
    return SubspaceModel(np.asarray(timescales, dtype=np.float64), basis @ sub, cores)


__all__ += ["SubspaceModel", "subspace_model"]


def fit_local_residual(
    target_diag: np.ndarray,
    model_diag: np.ndarray,
    kernels: np.ndarray,
    weights: np.ndarray | None = None,
    **fit_kwargs,
) -> tuple[np.ndarray, dict]:
    """Per-residue band tensors (B, N, 3, 3) fitted to the diagonal-block residual."""

    residual = np.asarray(target_diag) - np.asarray(model_diag)
    return fit_psd_bands(residual, kernels, weights, **fit_kwargs)


__all__.append("model_from_components")
