"""Synthetic ground truth: OU-mixture dynamics written as DynaMode archives.

``ou_mixture`` draws x(t) = sum_b U_b eta_b(t) where each eta_b is a vector of
independent unit-variance AR(1)/OU processes with relaxation time t_b (frames;
t_b = 0 gives white noise).  Its exact lagged covariance is
C(tau) = sum_b exp(-tau / t_b) U_b U_b^T, so every estimator and fit in the
package can be checked against known truth.  ``rotational_ou`` gives a
non-reversible process (antisymmetric drift) for irreversibility tests.

``write_archive`` stores trajectories in the DynaMode-dev compressed backbone
format (cartesian12 codec, int32 quantisation, chunked delta coding) using the
submodule's own encoder, so tests exercise the real decode path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dynamode.dataloader.archive import FORMAT_NAME, FORMAT_VERSION
from dynamode.dataloader.compressed_backbone import (
    codec_scales,
    delta_encode_block,
    encode_backbone,
    open_zip_group,
    quantize_code,
)
from numcodecs import Blosc

__all__ = ["helix_backbone", "ou_mixture", "rotational_ou", "write_archive"]


def helix_backbone(n_res: int) -> np.ndarray:
    """Idealised helical N/CA/C/O backbone (Angstrom), non-degenerate frames."""

    i = np.arange(n_res, dtype=np.float64)
    angle = np.deg2rad(100.0) * i
    ca = np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * i], axis=-1)
    tangent = np.stack([-np.sin(angle), np.cos(angle), np.full_like(angle, 0.65)], axis=-1)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    radial = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)
    n_atom = ca - 1.2 * tangent + 0.6 * radial
    c_atom = ca + 1.3 * tangent + 0.5 * radial
    o_atom = c_atom + 1.0 * np.cross(tangent, radial) + 0.3 * radial
    return np.stack([n_atom, ca, c_atom, o_atom], axis=1)


def _ar1(n_frames: int, dims: int, timescale: float, rng: np.random.Generator) -> np.ndarray:
    if dims == 0:
        return np.zeros((n_frames, 0))
    if timescale == 0:
        return rng.standard_normal((n_frames, dims))
    phi = np.exp(-1.0 / timescale)
    noise = rng.standard_normal((n_frames, dims)) * np.sqrt(1.0 - phi**2)
    out = np.empty((n_frames, dims))
    out[0] = rng.standard_normal(dims)
    for t in range(1, n_frames):
        out[t] = phi * out[t - 1] + noise[t]
    return out


def ou_mixture(
    n_frames: int, timescales, factors, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """(T, D) samples of sum_b U_b eta_b(t); ``factors[b]`` is (D, r_b)."""

    rng = np.random.default_rng(rng)
    dim = factors[0].shape[0]
    x = np.zeros((int(n_frames), dim))
    for t_b, u_b in zip(timescales, factors):
        x += _ar1(int(n_frames), u_b.shape[1], float(t_b), rng) @ u_b.T
    return x


def rotational_ou(
    n_frames: int, timescale: float, omega: float, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """2D OU with drift -(1/t) I + omega J: stationary but not time-reversible."""

    rng = np.random.default_rng(rng)
    drift = np.array([[-1.0 / timescale, -omega], [omega, -1.0 / timescale]])
    w, v = np.linalg.eig(drift)
    step = (v @ np.diag(np.exp(w)) @ np.linalg.inv(v)).real
    q = np.eye(2) - step @ step.T  # unit stationary covariance
    chol = np.linalg.cholesky(q + 1e-12 * np.eye(2))
    x = np.zeros((int(n_frames), 2))
    x[0] = rng.standard_normal(2)
    for t in range(1, int(n_frames)):
        x[t] = step @ x[t - 1] + chol @ rng.standard_normal(2)
    return x


def write_archive(
    root: str | Path,
    source: str,
    domain_id: str,
    native: np.ndarray,
    trajectories: dict,
    frame_interval_ps: float,
    chunk_frames: int = 64,
    residue_name: str = "ALA",
) -> Path:
    """Write ``{(temperature, replica): (T, N, 4, 3)}`` aligned coordinates as one archive."""

    prefix = {"mdcath": "mdcath_backbone_", "atlas": "atlas_backbone_"}[source]
    domains = Path(root) / "domains"
    domains.mkdir(parents=True, exist_ok=True)
    path = domains / f"{prefix}{domain_id}.zarr.zip"
    native = np.asarray(native, dtype=np.float32)
    reference = native - native[:, 1, :].mean(axis=0)[None, None, :]
    n_res = native.shape[0]
    scales = codec_scales("cartesian12", 0.01, 0.001, 0.001)
    compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    zroot, store = open_zip_group(path, mode="w")
    try:
        zroot.attrs.put({
            "format": FORMAT_NAME, "format_version": FORMAT_VERSION, "complete": True,
            "source_dataset": source, "frame_interval_ps": float(frame_interval_ps),
            "coordinate_unit": "angstrom", "codec": "cartesian12",
        })
        static = zroot.require_group("static")
        static.array("native_backbone_coords", native, chunks=native.shape, compressor=compressor)
        static.array("residue_names", np.asarray([residue_name.encode()] * n_res, dtype="S4"),
                     chunks=(n_res,), compressor=compressor)
        dssp = np.zeros((n_res, 8), dtype=np.float32)
        dssp[:, 0] = 1.0
        static.array("dssp", dssp, chunks=dssp.shape, compressor=compressor)
        for (temperature, replica), coords in trajectories.items():
            coords = np.asarray(coords, dtype=np.float32)
            codes = quantize_code(encode_backbone(coords, reference, "cartesian12"), scales)
            encoded = np.concatenate([
                delta_encode_block(codes[s : s + chunk_frames]) for s in range(0, len(codes), chunk_frames)
            ])
            group = zroot.require_group(str(temperature)).require_group(str(replica))
            group.array("native_reference", reference, chunks=reference.shape, compressor=compressor)
            group.array("trajectory_code", encoded, chunks=(chunk_frames, n_res, 12), compressor=compressor)
            group.attrs.put({
                "codec": "cartesian12", "quantization_scales": scales.tolist(),
                "delta_chunk_frames": int(chunk_frames), "frame_interval_ps": float(frame_interval_ps),
            })
    finally:
        store.close()
    return path
