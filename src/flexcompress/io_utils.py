"""Atomic file writes and small provenance helpers."""

from __future__ import annotations

import io
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__all__ = ["atomic_write_bytes", "atomic_write_json", "atomic_write_npz", "git_state", "now_utc", "read_json"]


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, document) -> None:
    text = json.dumps(document, cls=_Encoder, indent=1, allow_nan=True)
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def atomic_write_npz(path: str | Path, **arrays) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write_bytes(path, buffer.getvalue())


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_state(path: str | Path) -> dict:
    """Commit, branch and dirty flag of the git checkout containing ``path``."""

    def run(*args):
        try:
            done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=20)
            return done.stdout.strip() if done.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def environment() -> dict:
    versions = {}
    for name in ("numpy", "scipy", "zarr", "numcodecs", "yaml", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "packages": versions,
    }


__all__.append("environment")
