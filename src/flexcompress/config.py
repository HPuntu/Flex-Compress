"""Experiment configuration: defaults, YAML loading, env expansion, resolution.

A run's resolved configuration is written once to ``<run>/config.json`` by
``configure`` and is the only configuration workers and aggregation read.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import numpy as np

from .spectral import band_timescales

__all__ = ["DEFAULTS", "SOURCE_DEFAULTS", "load_config", "resolve_config"]

SOURCE_DEFAULTS = {
    "atlas": {
        "archive_root": "${DYNAMODE_ATLAS_ARCHIVE:-/projects/u6xq/hew/DATA/ATLAS/compressed_bb_zarr}",  # Isambard default
        "split_file": "DynaMode-dev/splits/ATLAS/atlas_train.csv",
        "nominal_interval_ps": 10.0,
        "nominal_frames": 10000,
        "temperatures": None,
    },
    "mdcath": {
        "archive_root": "${DYNAMODE_MDCATH_ARCHIVE:-/projects/u6xq/hew/DATA/mdCATH/compressed_bb_zarr}",  # Isambard default
        "split_file": "DynaMode-dev/splits/mdCATH/train_ids.txt",
        "nominal_interval_ps": 1000.0,
        "nominal_frames": 500,
        "temperatures": ["320", "348", "379", "413", "450"],
    },
}

DEFAULTS: dict = {
    "name": "stage01",
    "source": "atlas",
    "archive_root": None,  # default from SOURCE_DEFAULTS
    "index_path": None,  # DynaMode trajectory_index.v2.json if not under archive_root
    "split": {"file": None, "ids": None},
    "temperatures": None,
    "frame_interval_ps": None,  # null = native; e.g. 1000 views ATLAS at 1 ns
    "min_frames": 100,
    "min_replicas": 2,
    "selection": {
        "mode": "full",  # full | pilot
        "pilot_size": 32,  # domains, evenly spaced in length
        "min_residues": None,
        "max_residues": None,
        "domains": None,  # explicit domain list overrides the split
    },
    "stats": {
        "n_lags": 16,
        "max_lag_fraction": 0.1,
        "pair_max_residues": 128,
        "n_pair_lags": 6,
    },
    "spectral": {
        "bands_per_decade": 2.0,
        "max_timescale_fraction": 0.5,
        "subframe_band": True,
        "pc_max": 256,
        # Adjacent bands are nearly collinear; a small ridge picks the min-norm
        # split and speeds ADMM.  Fitted covariances converge long before the
        # band split does, so tol 1e-5 changes no distortion metric materially.
        "ridge": 1.0e-3,
        "max_iter": 400,
        "local_max_iter": 200,
        "tol": 1.0e-5,
        "obj_tol": 1.0e-6,  # stop when the fit objective stalls (see spectral.fit_psd_bands)
        "ranks": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        # subspace_local is the primary codec; see spectral.SubspaceModel.
        "codecs": ["subspace_local", "subspace", "band_local"],
    },
    "baselines": {
        "pca_exp_ranks": [1, 2, 4, 8, 16, 32, 64, 128],
        "pca_exp_candidates": 64,
        "dct_keep": [8, 32, 128, 512, 2048],
    },
    "folds": {"max_folds": None},
    "diagnostics": {
        "bb_pca": True,
        "pca_top": 128,
        "bb_pooled_max_dim": 6000,
        "cosine_pcs": 5,
        "rmsip_k": [10, 20],
        "contact_frames": 50,
        "contact_cutoff_A": 8.0,
        "contact_tolerance": 1.2,
        "variance_thresholds": [0.5, 0.8, 0.9, 0.95, 0.99],
    },
    "descriptors": {
        "enabled": True,
        "bands_ps": "auto",  # fixed per-source grid so residues pool across proteins
        "nominal_frames": None,
        "eps_A2": 1.0e-4,
    },
    "stages": {"stage0": True, "stage1": True},
    # Compute backend (not part of the result hash: devices agree to rounding).
    # device: auto (CUDA if torch sees a GPU, else numpy) | numpy | cuda | cuda:N | torch-cpu
    # psd_method: auto (on CUDA: Jacobi for <=4x4, Newton-Schulz for >32x32, chunked eigh between;
    #             eigh on CPU) | eigh | newton_schulz | jacobi (Jacobi for <=4x4 on any device)
    "compute": {"device": "auto", "dtype": "float64", "psd_method": "auto", "ns_iters": 40},
    "aggregate": {"rank_threshold": [1.0, 1.1], "bootstrap": 1000},
    "seed": 20261009,
}

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value):
    if isinstance(value, str):
        def sub(match):
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(f"environment variable {name} is not set (needed by the config)")
        return _ENV.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _merge(base: dict, update: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        where = f"{path}.{key}" if path else key
        if key not in out:
            raise KeyError(f"unknown config key: {where}")
        if isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value, where)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_override(text: str) -> dict:
    import yaml

    if "=" not in text:
        raise ValueError(f"override must be key=value, got {text!r}")
    key, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    node: dict = {}
    cursor = node
    parts = key.strip().split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return node


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict:
    """Merge DEFAULTS <- YAML file <- ``key.sub=value`` overrides (unresolved)."""

    import yaml

    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for text in overrides or []:
        cfg = _merge(cfg, _parse_override(text))
    return cfg


def resolve_config(cfg: dict, repo_root: str | Path) -> dict:
    """Fill source defaults, expand ${ENV} references and fix the descriptor band grid."""

    cfg = copy.deepcopy(cfg)
    source = str(cfg["source"]).lower()
    if source not in SOURCE_DEFAULTS:
        raise ValueError(f"source must be one of {sorted(SOURCE_DEFAULTS)}")
    cfg["source"] = source
    src = SOURCE_DEFAULTS[source]
    if cfg["archive_root"] is None:
        cfg["archive_root"] = src["archive_root"]
    if cfg["split"]["file"] is None and cfg["split"]["ids"] is None:
        cfg["split"]["file"] = src["split_file"]
    if cfg["temperatures"] is None and src["temperatures"] is not None:
        cfg["temperatures"] = list(src["temperatures"])
    if cfg["temperatures"] is not None:
        cfg["temperatures"] = [str(t) for t in cfg["temperatures"]]
    cfg = _expand(cfg)
    cfg["archive_root"] = str(Path(cfg["archive_root"]).expanduser().resolve())
    if cfg["split"]["file"]:
        split = Path(cfg["split"]["file"]).expanduser()
        cfg["split"]["file"] = str(split if split.is_absolute() else (Path(repo_root) / split).resolve())

    dt_ps = float(cfg["frame_interval_ps"] or src["nominal_interval_ps"])
    if cfg["descriptors"]["bands_ps"] == "auto":
        nominal = cfg["descriptors"]["nominal_frames"]
        if nominal is None:
            nominal = int(round(src["nominal_frames"] * src["nominal_interval_ps"] / dt_ps))
        frames = band_timescales(
            int(nominal),
            float(cfg["spectral"]["bands_per_decade"]),
            float(cfg["spectral"]["max_timescale_fraction"]),
            subframe=True,
        )
        cfg["descriptors"]["bands_ps"] = [float(x) for x in np.round(frames * dt_ps, 6)]
        cfg["descriptors"]["nominal_frames"] = int(nominal)
    cfg["descriptors"]["bands_ps"] = [float(x) for x in cfg["descriptors"]["bands_ps"]]
    return cfg
