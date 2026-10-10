"""Structure-fixed residue frames for rotation-invariant per-residue tensors.

Convention matches DynaMode's backbone codec: x along CA->C, y the component
of CA->N orthogonal to x, z = x cross y.  Frames come from the native
reference, so they are fixed through the trajectory and shared by replicas.
"""

from __future__ import annotations

import numpy as np

__all__ = ["residue_frames", "to_local"]


def residue_frames(backbone: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(N, 3, 3) rotations whose columns are the local x, y, z axes."""

    backbone = np.asarray(backbone, dtype=np.float64)
    if backbone.ndim != 3 or backbone.shape[-2:] != (4, 3):
        raise ValueError(f"expected (N, 4, 3) backbone, got {backbone.shape}")
    n_atom, ca, c_atom = backbone[:, 0], backbone[:, 1], backbone[:, 2]
    x = c_atom - ca
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(x_norm <= eps):
        raise ValueError("degenerate CA-C vector")
    x = x / x_norm
    n_vec = n_atom - ca
    y = n_vec - np.sum(n_vec * x, axis=-1, keepdims=True) * x
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(y_norm <= eps):
        raise ValueError("degenerate N-CA-C geometry")
    y = y / y_norm
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def to_local(blocks: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Rotate per-residue 3x3 tensors into residue frames: R_i^T C_i R_i.

    ``blocks`` has shape (..., N, 3, 3); ``frames`` (N, 3, 3).
    """

    return np.einsum("nai,...nab,nbj->...nij", frames, blocks, frames)
