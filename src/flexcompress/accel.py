"""Optional PyTorch acceleration of the hot kernels (the H100 of an Isambard-AI GH200).

Selected once per process with :func:`configure`:

    "numpy" / "cpu"   reference numpy/scipy path (also the fallback for "auto")
    "auto"            CUDA when torch sees a GPU, otherwise numpy
    "cuda" / "cuda:N" force the GPU (error if unavailable)
    "torch-cpu"       the torch code path on the CPU (tests and debugging)

Every accelerated function takes and returns numpy arrays exactly like its
numpy counterpart and computes the same quantity in float64, so results agree
to rounding.  One deliberate exception: inside ADMM, PSD projections of
matrices larger than 32 x 32 use a matmul-only Newton-Schulz iteration on the
GPU, because batched ``torch.linalg.eigh`` falls back to one-matrix-at-a-time
``syevd`` above n = 32 (pytorch/pytorch#175585) while matmuls are what an H100
does best.  Inexact projections are standard for ADMM; the returned solution
is projected exactly once at the end, so it is always PSD.  The 3 x 3
per-residue fits (most of the CPU time) use cuSOLVER's batched Jacobi eigh.

Accelerated: windowed lagged covariance sums, PC bases (gram + eigh),
randomized/exact PCA, pooled backbone covariance, ADMM band fits, model
predictions, and the DCT baseline.  Archive decoding and metrics stay on the
CPU.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from . import baselines as bl
from . import diagnostics as dg
from . import spectral as sp
from . import stats as st
from .stats import Lagged

__all__ = [
    "OuterSum",
    "band_diag",
    "configure",
    "dct_lagged",
    "device_memory_gb",
    "eigvalsh_gram",
    "enabled",
    "fit_psd_bands",
    "gram_top_eigh",
    "info",
    "lagged_cross",
    "lagged_diag",
    "predict",
    "release",
    "top_pcs",
]

_S: dict = {"torch": None, "device": None, "dtype": None, "name": "numpy", "psd_method": "auto", "ns_iters": 40}


# --------------------------------------------------------------------------- setup


def configure(device: str | None = "auto", dtype: str = "float64", psd_method: str = "auto",
              ns_iters: int = 40) -> str:
    """Select the compute backend for this process; returns its name."""

    choice = (device or "numpy").strip().lower()
    _S.update(torch=None, device=None, dtype=None, name="numpy",
              psd_method=str(psd_method), ns_iters=int(ns_iters))
    if choice in ("numpy", "cpu", "none"):
        return "numpy"
    try:
        import torch
    except ImportError:
        if choice == "auto":
            return "numpy"
        raise RuntimeError(f"compute device {choice!r} needs PyTorch, which is not installed") from None
    if choice == "auto":
        if not torch.cuda.is_available():
            return "numpy"
        choice = "cuda"
    if choice == "torch-cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device(choice)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"compute device {choice!r} requested but torch sees no CUDA GPU")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _S.update(torch=torch, device=dev, dtype=getattr(torch, dtype), name=f"torch:{dev}")
    return _S["name"]


def enabled() -> bool:
    return _S["torch"] is not None


def info() -> dict:
    out = {"backend": _S["name"], "psd_method": _S["psd_method"], "ns_iters": _S["ns_iters"]}
    torch = _S["torch"]
    if torch is not None:
        out["torch"] = torch.__version__
        if _S["device"].type == "cuda":
            out["gpu"] = torch.cuda.get_device_name(_S["device"])
            out["cuda"] = torch.version.cuda
    return out


def device_memory_gb() -> float:
    """Total memory of the configured GPU in GB (inf for CPU backends)."""

    torch = _S["torch"]
    if torch is None or _S["device"].type != "cuda":
        return float("inf")
    return torch.cuda.get_device_properties(_S["device"]).total_memory / 1e9


def release() -> None:
    """Return cached GPU memory to other processes sharing the device."""

    torch = _S["torch"]
    if torch is not None and _S["device"].type == "cuda":
        try:
            torch.cuda.synchronize(_S["device"])
        except Exception:  # a failed kernel may surface here; the cache is still released
            pass
        torch.cuda.empty_cache()


def on_gpu() -> bool:
    return _S["torch"] is not None and _S["device"].type == "cuda"


# Messages of errors caused by the GPU running short of memory: CUDA allocation
# failures, and cuSOLVER/cuBLAS failing to create a handle or workspace.
_GPU_MEMORY_MARKERS = ("out of memory", "CUSOLVER_STATUS_INTERNAL_ERROR", "CUSOLVER_STATUS_ALLOC_FAILED",
                       "CUBLAS_STATUS_ALLOC_FAILED", "CUBLAS_STATUS_NOT_INITIALIZED", "cusolverDnCreate")


def is_gpu_memory_error(exc: BaseException) -> bool:
    """True if ``exc`` means the shared GPU ran short of memory (worth retrying), not a bug."""

    if not on_gpu():
        return False
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _GPU_MEMORY_MARKERS)


def request_exceeds_device(exc: BaseException) -> bool:
    """True if a CUDA OOM message asked for more than the GPU's total memory."""

    import re

    match = re.search(r"Tried to allocate ([0-9.]+) (GiB|MiB)", str(exc))
    if not match or not on_gpu():
        return False
    size = float(match.group(1)) * (2**30 if match.group(2) == "GiB" else 2**20)
    return size > _S["torch"].cuda.get_device_properties(_S["device"]).total_memory


def reset_peak() -> None:
    if on_gpu():
        _S["torch"].cuda.reset_peak_memory_stats(_S["device"])


def peak_gb() -> float | None:
    """Peak GPU memory this process reserved since ``reset_peak`` (GB), plus its CUDA context."""

    if not on_gpu():
        return None
    return _S["torch"].cuda.max_memory_reserved(_S["device"]) / 1e9 + CUDA_CONTEXT_GB


CUDA_CONTEXT_GB = 0.6  # per-process CUDA context and library handles, outside the caching allocator


def _t(x):
    torch = _S["torch"]
    if isinstance(x, torch.Tensor):
        return x.to(device=_S["device"], dtype=_S["dtype"])
    return torch.from_numpy(np.ascontiguousarray(x)).to(device=_S["device"], dtype=_S["dtype"])


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy()


# --------------------------------------------------------------------------- lagged statistics


def _lagged_diag_t(x, lags: np.ndarray, segment: int):
    torch = _S["torch"]
    n_res = x.shape[1]
    sums = torch.zeros((len(lags), n_res, 3, 3), dtype=_S["dtype"], device=_S["device"])
    starts = st.window_starts(x.shape[0], segment)
    for s in starts:
        w = x[int(s) : int(s) + segment]
        w = (w - w.mean(dim=0, keepdim=True)).permute(1, 2, 0)  # (N, 3, T)
        for li, tau in enumerate(lags):
            tau = int(tau)
            sums[li] += torch.bmm(w[:, :, : segment - tau], w[:, :, tau:].transpose(1, 2))
    return sums, (segment - lags).astype(np.float64) * len(starts)


def _lagged_cross_t(x, lags: np.ndarray, segment: int):
    torch = _S["torch"]
    dim = x.shape[1]
    sums = torch.zeros((len(lags), dim, dim), dtype=_S["dtype"], device=_S["device"])
    starts = st.window_starts(x.shape[0], segment)
    for s in starts:
        w = x[int(s) : int(s) + segment]
        w = w - w.mean(dim=0, keepdim=True)
        for li, tau in enumerate(lags):
            tau = int(tau)
            sums[li] += w[: segment - tau].T @ w[tau:]
    return sums, (segment - lags).astype(np.float64) * len(starts)


def lagged_diag(dx: np.ndarray, lags, segment: int) -> Lagged:
    """Windowed per-residue lagged 3x3 sums (see ``stats.segmented``)."""

    lags = np.asarray(lags, dtype=np.int64)
    if not enabled():
        return st.segmented(dx, lags, segment, st.diag_block_sums)
    sums, counts = _lagged_diag_t(_t(dx), lags, int(segment))
    return Lagged(lags.copy(), _np(sums), counts)


def lagged_cross(z: np.ndarray, lags, segment: int, basis: np.ndarray | None = None) -> Lagged:
    """Windowed full lagged sums of ``z`` (optionally projected: ``z @ basis``)."""

    lags = np.asarray(lags, dtype=np.int64)
    if not enabled():
        projected = z @ basis if basis is not None else z
        return st.segmented(projected, lags, segment, st.cross_sums)
    x = _t(z)
    if basis is not None:
        x = x @ _t(basis)
    sums, counts = _lagged_cross_t(x, lags, int(segment))
    return Lagged(lags.copy(), _np(sums), counts)


# --------------------------------------------------------------------------- PCA / covariance


def gram_top_eigh(flats: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Top-k eigenpairs of the pooled lag-0 covariance sum_r X_r^T X_r / sum_r T_r."""

    dim = flats[0].shape[1]
    k = min(int(k), dim)
    frames = float(sum(f.shape[0] for f in flats))
    if not enabled():
        cov = np.zeros((dim, dim))
        for f in flats:
            cov += f.T @ f
        cov /= frames
        evals, vecs = scipy_eigh(cov, subset_by_index=[dim - k, dim - 1])
        order = np.argsort(evals)[::-1]
        return np.clip(evals[order], 0.0, None), vecs[:, order], float(np.trace(cov))
    torch = _S["torch"]
    cov = torch.zeros((dim, dim), dtype=_S["dtype"], device=_S["device"])
    for f in flats:
        x = _t(f)
        cov += x.T @ x
    cov /= frames
    evals, vecs = torch.linalg.eigh(cov)
    evals, vecs = evals.flip(0)[:k], vecs.flip(1)[:, :k]
    return np.clip(_np(evals), 0.0, None), _np(vecs), float(torch.trace(cov).item())


def eigvalsh_gram(arrays: list[np.ndarray]) -> np.ndarray:
    """Descending eigenvalues of sum_r X_r^T X_r / sum_r T_r (rows already centred)."""

    frames = float(sum(a.shape[0] for a in arrays))
    if not enabled():
        cov = sum(a.T @ a for a in arrays) / frames
        return np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)
    cov = None
    for a in arrays:
        x = _t(a)
        cov = x.T @ x if cov is None else cov + x.T @ x
    return np.clip(_np(_S["torch"].linalg.eigvalsh(cov / frames)).copy()[::-1], 0.0, None)


class OuterSum:
    """Running sums for a pooled (grand-mean) covariance of large frames."""

    def __init__(self, dim: int):
        self.dim, self.count = int(dim), 0
        if enabled():
            torch = _S["torch"]
            self.sum = torch.zeros(dim, dtype=_S["dtype"], device=_S["device"])
            self.outer = torch.zeros((dim, dim), dtype=_S["dtype"], device=_S["device"])
        else:
            self.sum, self.outer = np.zeros(dim), np.zeros((dim, dim))

    def add(self, frames: np.ndarray) -> None:
        if enabled():
            x = _t(frames)
        else:
            x = np.asarray(frames, dtype=np.float64)
        self.sum += x.sum(0)
        self.outer += x.T @ x
        self.count += int(frames.shape[0])

    def top_eigvals(self, k: int) -> tuple[np.ndarray, float]:
        k = min(int(k), self.dim)
        mean = self.sum / self.count
        if enabled():
            torch = _S["torch"]
            cov = self.outer / self.count - torch.outer(mean, mean)
            w = torch.linalg.eigvalsh(cov).flip(0)[:k]
            return np.clip(_np(w), 0.0, None), float(torch.trace(cov).item())
        cov = self.outer / self.count - np.outer(mean, mean)
        w = scipy_eigh(cov, eigvals_only=True, subset_by_index=[self.dim - k, self.dim - 1])
        return np.clip(w[::-1], 0.0, None), float(np.trace(cov))


def top_pcs(centered: np.ndarray, k: int, *, exact_max_dim: int = 3000, oversample: int = 16,
            power_iters: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Accelerated :func:`diagnostics.top_pcs` (same algorithm and outputs)."""

    if not enabled():
        return dg.top_pcs(centered, k, exact_max_dim=exact_max_dim, oversample=oversample,
                          power_iters=power_iters, seed=seed)
    torch = _S["torch"]
    x = _t(centered)
    n, dim = x.shape
    k = int(min(k, dim, n))
    total = float((x * x).sum().item() / n)
    if dim <= exact_max_dim:
        w, v = torch.linalg.eigh(x.T @ x / n)
        return np.clip(_np(w.flip(0)[:k]), 0.0, None), _np(v.flip(1)[:, :k]), total
    gen = torch.Generator(device=_S["device"]).manual_seed(int(seed))
    probe = torch.randn((dim, k + oversample), generator=gen, dtype=_S["dtype"], device=_S["device"])
    q, _ = torch.linalg.qr(x @ probe)
    for _ in range(int(power_iters)):
        q, _ = torch.linalg.qr(x.T @ q)
        q, _ = torch.linalg.qr(x @ q)
    _, s, vt = torch.linalg.svd(q.T @ x, full_matrices=False)
    return _np(s[:k] ** 2 / n), _np(vt[:k].T), total


# --------------------------------------------------------------------------- ADMM band fit


EIGH_CHUNK = 4096  # matrices per batched eigh call on CUDA (bounds cuSOLVER workspace)
JACOBI_MAX_DIM = 4
JACOBI_SWEEPS = 8


def _eigh(sym):
    """Batched eigh; on CUDA in bounded chunks.

    cuSOLVER's batched solver sizes its workspace with the whole batch, and for
    the hundreds of thousands of matrices of a stage-1 local fit it asked for
    more memory than the GPU has (100-360 GB for one protein).
    """

    torch = _S["torch"]
    batch = sym.shape[:-2]
    count = int(np.prod(batch)) if batch else 1
    if _S["device"].type != "cuda" or count <= EIGH_CHUNK:
        return torch.linalg.eigh(sym)
    flat = sym.reshape(count, *sym.shape[-2:])
    ws, vs = zip(*(torch.linalg.eigh(flat[i : i + EIGH_CHUNK]) for i in range(0, count, EIGH_CHUNK)))
    return torch.cat(ws).reshape(*batch, -1), torch.cat(vs).reshape(sym.shape)


def _jacobi_eigh(sym, sweeps: int = JACOBI_SWEEPS):
    """Batched cyclic Jacobi for small symmetric matrices: elementwise ops and tiny matmuls only.

    No solver library and no workspace; converges quadratically, so a few
    sweeps reach rounding error for 3x3.  Returns (eigenvalues, eigenvectors)
    like ``torch.linalg.eigh`` (unsorted, which the PSD projection does not need).
    """

    torch = _S["torch"]
    n = sym.shape[-1]
    a = sym.clone()
    eye = torch.eye(n, dtype=sym.dtype, device=sym.device)
    v = eye.expand_as(a).clone()
    pairs = [(p, q) for p in range(n - 1) for q in range(p + 1, n)]
    for _ in range(int(sweeps)):
        for p, q in pairs:
            apq = a[..., p, q]
            app, aqq = a[..., p, p], a[..., q, q]
            nonzero = apq != 0
            theta = (aqq - app) / torch.where(nonzero, 2.0 * apq, torch.ones_like(apq))
            t = torch.sign(theta) / (theta.abs() + torch.sqrt(theta * theta + 1.0))
            t = torch.where(nonzero & torch.isfinite(t), t, torch.zeros_like(t))
            t = torch.where(theta == 0, torch.where(nonzero, torch.ones_like(t), t), t)
            c = 1.0 / torch.sqrt(t * t + 1.0)
            sn = t * c
            rot = eye.expand_as(a).clone()
            rot[..., p, p] = c
            rot[..., q, q] = c
            rot[..., p, q] = sn
            rot[..., q, p] = -sn
            a = rot.transpose(-1, -2) @ a @ rot
            v = v @ rot
    return torch.diagonal(a, dim1=-2, dim2=-1), v


def _proj_exact(z):
    sym = 0.5 * (z + z.transpose(-1, -2))
    small = sym.shape[-1] <= JACOBI_MAX_DIM
    if small and (_S["psd_method"] == "jacobi" or _S["device"].type == "cuda"):
        w, v = _jacobi_eigh(sym)
    else:
        w, v = _eigh(sym)
    out = (v * w.clamp_min(0.0).unsqueeze(-2)) @ v.transpose(-1, -2)
    return 0.5 * (out + out.transpose(-1, -2))


def _proj_newton_schulz(z, iters: int):
    """P_+(S) = (S + |S|) / 2 with |S| = S sign(S); sign by Newton-Schulz (matmuls only).

    Eigenvalues above ~1.5^-iters of ||S||_F converge; smaller ones contribute an
    error no larger than themselves, which ADMM absorbs.
    """

    torch = _S["torch"]
    sym = 0.5 * (z + z.transpose(-1, -2))
    norm = torch.linalg.matrix_norm(sym, ord="fro", keepdim=True).clamp_min(1e-300)
    x = sym / norm
    eye = torch.eye(sym.shape[-1], dtype=sym.dtype, device=sym.device)
    for _ in range(int(iters)):
        x = x @ (1.5 * eye - 0.5 * (x @ x))
    out = 0.5 * (sym + sym @ x)
    return 0.5 * (out + out.transpose(-1, -2))


def _use_newton_schulz(dim: int) -> bool:
    method = _S["psd_method"]
    if method == "newton_schulz":
        return True
    if method in ("eigh", "jacobi"):
        return False
    return _S["device"].type == "cuda" and dim > 32


def fit_psd_bands(targets, kernels, weights=None, ridge: float = 1e-6, max_iter: int = 500, tol: float = 1e-7,
                  init=None, obj_tol: float = 1e-6, check_every: int = 10):
    """Accelerated :func:`spectral.fit_psd_bands` (same ADMM, same stopping rules)."""

    if not enabled():
        return sp.fit_psd_bands(targets, kernels, weights, ridge=ridge, max_iter=max_iter, tol=tol,
                                init=init, obj_tol=obj_tol, check_every=check_every)
    torch = _S["torch"]
    kernels = np.asarray(kernels, dtype=np.float64)
    n_lags, n_bands = kernels.shape
    w = np.ones(n_lags) if weights is None else np.asarray(weights, dtype=np.float64)
    data = _t(targets)
    if data.shape[0] != n_lags:
        raise ValueError("targets and kernels disagree on the number of lags")
    sym = 0.5 * (data + data.transpose(-1, -2))
    peak = sym[0].abs().amax(dim=(-2, -1), keepdim=True)  # per batch element, as spectral.py
    scale = torch.where(peak > 0, peak, torch.ones_like(peak))
    sym = sym / scale
    wshape = (-1,) + (1,) * (sym.ndim - 1)
    w_t = _t(w).reshape(wshape)

    gram = kernels.T @ (w[:, None] * kernels)
    gram = gram + float(ridge) * np.trace(gram) / n_bands * np.eye(n_bands)
    rhs = torch.tensordot(_t((kernels * w[:, None]).T), sym, dims=([1], [0]))
    gram_t = _t(gram)
    ns = _use_newton_schulz(int(sym.shape[-1]))

    def proj(z):
        return _proj_newton_schulz(z, _S["ns_iters"]) if ns else _proj_exact(z)

    rho = float(np.trace(gram)) / n_bands
    if init is not None and tuple(np.shape(init)) == tuple(rhs.shape):
        current = proj(_t(init) / scale)
    else:
        current = proj(torch.tensordot(_t(np.linalg.inv(gram)), rhs, dims=([1], [0])))
    dual = torch.zeros_like(current)
    eye = np.eye(n_bands)
    inverse = _t(np.linalg.inv(gram + rho * eye))
    total = float((w_t * sym**2).sum().item())
    flat_rhs = rhs.reshape(n_bands, -1)

    def rel_objective(z):
        flat = z.reshape(n_bands, -1)
        value = float(((gram_t * (flat @ flat.T)).sum() - 2.0 * (flat * flat_rhs).sum()).item()) + total
        return value / max(total, 1e-300)

    iterations, rel_change, last_obj, stop = 0, math.inf, math.inf, "max_iter"
    for iterations in range(1, int(max_iter) + 1):
        primal = torch.tensordot(inverse, rhs + rho * (current - dual), dims=([1], [0]))
        previous = current
        current = proj(primal + dual)
        dual = dual + primal - current
        norm = max(float(torch.linalg.norm(current).item()), 1e-300)
        r_primal = float(torch.linalg.norm(primal - current).item()) / norm
        r_dual = float(torch.linalg.norm(current - previous).item()) / norm
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
            inverse = _t(np.linalg.inv(gram + rho * eye))
        elif r_dual > 10.0 * r_primal:
            rho, dual = rho / 2.0, dual * 2.0
            inverse = _t(np.linalg.inv(gram + rho * eye))

    if ns:
        current = _proj_exact(current)  # guarantee a PSD result
    residual = torch.tensordot(_t(kernels), current, dims=([1], [0])) - sym
    objective = float((w_t * residual**2).sum().item())
    info = {
        "iterations": iterations,
        "rel_change": rel_change,
        "converged": stop != "max_iter",
        "stop": stop,
        "rel_residual": math.sqrt(objective / max(total, 1e-300)),
        "projection": "newton_schulz" if ns else "eigh",
    }
    return _np(current * scale), info


# --------------------------------------------------------------------------- model predictions


def _band_diag_t(model):
    torch = _S["torch"]
    n_t = len(model.timescales)
    n_res = model.n_res
    out = torch.zeros((n_t, n_res, 3, 3), dtype=_S["dtype"], device=_S["device"])
    if hasattr(model, "cores"):
        if model.rank:
            u = _t(model.basis).reshape(n_res, 3, model.rank)
            tmp = torch.einsum("nir,krs->knis", u, _t(model.cores))
            out = torch.einsum("knis,njs->knij", tmp, u)
    else:
        per_res = _t(model.factors).reshape(n_res, 3, -1)
        for k in np.unique(model.kid):
            idx = torch.as_tensor(np.nonzero(model.kid == k)[0], device=_S["device"])
            u = per_res.index_select(2, idx)
            out[int(k)] = torch.einsum("nir,njr->nij", u, u)
    if model.local is not None:
        out = out + _t(model.local)
    return out


def _band_pair_t(model, residues: np.ndarray):
    torch = _S["torch"]
    rows = st.pair_rows(residues)
    n_t = len(model.timescales)
    out = torch.zeros((n_t, rows.size, rows.size), dtype=_S["dtype"], device=_S["device"])
    if hasattr(model, "cores"):
        if model.rank:
            sub = _t(model.basis[rows])
            tmp = torch.einsum("ir,krs->kis", sub, _t(model.cores))
            out = torch.einsum("kis,js->kij", tmp, sub)
    else:
        sub = _t(model.factors[rows])
        for k in np.unique(model.kid):
            idx = torch.as_tensor(np.nonzero(model.kid == k)[0], device=_S["device"])
            u = sub.index_select(1, idx)
            out[int(k)] = u @ u.T
    if model.local is not None:
        local = _t(model.local[:, np.asarray(residues)])  # (Kt, S, 3, 3)
        for j in range(local.shape[1]):
            out[:, 3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += local[:, j]
    return out


def band_diag(model) -> np.ndarray:
    """Per-band diagonal blocks (Kt, N, 3, 3) of a Spectral/Subspace model."""

    if not enabled():
        return model.band_diag()
    return _np(_band_diag_t(model))


def predict(model, kernel_diag: np.ndarray, kernel_pair: np.ndarray, residues: np.ndarray):
    """Expected-estimator diag (L, N, 3, 3) and pair (Lp, 3S, 3S) blocks of a model."""

    if not enabled():
        return model.predict(kernel_diag, kernel_pair, residues)
    torch = _S["torch"]
    diag = torch.tensordot(_t(kernel_diag), _band_diag_t(model), dims=([1], [0]))
    pair = torch.tensordot(_t(kernel_pair), _band_pair_t(model, residues), dims=([1], [0]))
    return _np(diag), _np(pair)


# --------------------------------------------------------------------------- DCT baseline


def dct_lagged(flat: np.ndarray, keep: int, lags, pair_lags, rows: np.ndarray, n_res: int, segment: int):
    """Truncated-DCT reconstruction of a (T, 3N) replica, then its windowed statistics."""

    lags = np.asarray(lags, dtype=np.int64)
    pair_lags = np.asarray(pair_lags, dtype=np.int64)
    n_frames = flat.shape[0]
    if not enabled():
        recon = bl.dct_truncate(flat, keep)
        return (st.segmented(recon.reshape(n_frames, n_res, 3), lags, segment, st.diag_block_sums),
                st.segmented(recon[:, rows], pair_lags, segment, st.cross_sums))
    torch = _S["torch"]
    x = _t(flat)
    t = torch.arange(n_frames, dtype=_S["dtype"], device=_S["device"])
    k = torch.arange(int(keep), dtype=_S["dtype"], device=_S["device"])
    # Orthonormal DCT-II basis (scipy norm="ortho"): truncation = projection onto its first columns.
    phi = torch.cos(math.pi * (t[:, None] + 0.5) * k[None, :] / n_frames) * math.sqrt(2.0 / n_frames)
    phi[:, 0] = math.sqrt(1.0 / n_frames)
    recon = phi @ (phi.T @ x)
    d_sums, d_counts = _lagged_diag_t(recon.reshape(n_frames, n_res, 3), lags, int(segment))
    p_sums, p_counts = _lagged_cross_t(recon[:, torch.as_tensor(rows, device=_S["device"])], pair_lags, int(segment))
    return Lagged(lags.copy(), _np(d_sums), d_counts), Lagged(pair_lags.copy(), _np(p_sums), p_counts)
