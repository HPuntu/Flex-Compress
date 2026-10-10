"""Read-only access to DynaMode-dev compressed backbone archives.

Layout (produced by DynaMode-dev ``scripts/download_*_compressed_bb.py``)::

    <archive_root>/domains/{mdcath,atlas}_backbone_<domain>.zarr.zip
    <archive_root>/trajectory_index.v2.json         (optional cache)

Each archive has ``static/{residue_names, dssp, native_backbone_coords}`` and
``<temperature>/<replica>/{trajectory_code, native_reference}``.  Coordinates
are Angstrom N/CA/C/O, aligned by two-pass native-CA Kabsch on the least-mobile
half of residues, centred on that core.  All replicas share the native
orientation.  Decoding is delegated to ``dynamode.dataloader.compressed_backbone``
(numpy/scipy/zarr only; no torch import).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from dynamode.dataloader.archive import FORMAT_NAME, FORMAT_VERSION
from dynamode.dataloader.compressed_backbone import (
    load_backbone_trajectory,
    load_backbone_trajectory_sampled,
    open_zip_group,
)

__all__ = [
    "ARCHIVE_PREFIXES",
    "Target",
    "TrajRow",
    "build_targets",
    "discover",
    "load_replica",
    "load_static",
    "read_split_ids",
]

ARCHIVE_PREFIXES = {"mdcath": "mdcath_backbone_", "atlas": "atlas_backbone_"}
INDEX_NAME = "trajectory_index.v2.json"
DEFAULT_INTERVAL_PS = {"mdcath": 1000.0, "atlas": 10.0}
DSSP_STATES = ("H", "B", "E", "G", "I", "T", "S", "C")


@dataclass(frozen=True)
class TrajRow:
    domain_id: str
    archive_name: str
    temperature: str
    replica: str
    n_frames: int
    n_res: int
    source: str
    frame_interval_ps: float


@dataclass
class Target:
    """All replicas of one domain at one temperature."""

    source: str
    domain_id: str
    archive_path: str
    temperature: str
    replicas: list[str]
    n_frames: list[int]
    n_res: int
    frame_interval_ps: float
    shard: int = -1
    cost: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.domain_id}__T{self.temperature}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> Target:
        return Target(**d)


def domains_dir(archive_root: str | Path) -> Path:
    root = Path(archive_root).expanduser()
    return root / "domains" if (root / "domains").is_dir() else root


def archive_path(archive_root: str | Path, source: str, domain_id: str) -> Path:
    return domains_dir(archive_root) / f"{ARCHIVE_PREFIXES[source]}{domain_id}.zarr.zip"


def read_index(archive_root: str | Path, index_path: str | Path | None = None) -> list[TrajRow] | None:
    """Rows of DynaMode's cached trajectory index, or None if absent/unreadable."""

    root = Path(archive_root).expanduser()
    candidates = [Path(index_path).expanduser()] if index_path else []
    candidates.append(root / INDEX_NAME)
    if root.name == "domains":
        candidates.append(root.parent / INDEX_NAME)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("schema_version") != 2 or not isinstance(document.get("trajectories"), list):
            continue
        rows = []
        for row in document["trajectories"]:
            if not isinstance(row, list) or len(row) != 8:
                return None
            rows.append(
                TrajRow(str(row[0]), str(row[1]), str(row[2]), str(row[3]), int(row[4]),
                        int(row[5]), str(row[6]).lower(), float(row[7]))
            )
        return rows
    return None


def scan_archive(path: Path, source: str) -> list[TrajRow]:
    """Index rows for one archive by reading only its hierarchy metadata."""

    domain_id = path.name[len(ARCHIVE_PREFIXES[source]) : -len(".zarr.zip")]
    rows = []
    root, store = open_zip_group(path)
    try:
        if root.attrs.get("format") != FORMAT_NAME or int(root.attrs.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("unexpected archive format/version")
        if not bool(root.attrs.get("complete", False)):
            raise ValueError("archive is not marked complete")
        root_interval = float(root.attrs.get("frame_interval_ps", DEFAULT_INTERVAL_PS[source]))
        for temperature, replica in list_archive_trajectories_open(root):
            group = root[temperature][replica]
            n_frames, n_res = map(int, group["trajectory_code"].shape[:2])
            interval = float(group.attrs.get("frame_interval_ps", root_interval))
            rows.append(TrajRow(domain_id, path.name, temperature, replica, n_frames, n_res, source, interval))
    finally:
        store.close()
    return rows


def list_archive_trajectories_open(root) -> list[tuple[str, str]]:
    pairs = []
    for temperature in root.group_keys():
        if temperature == "static":
            continue
        for replica in root[temperature].group_keys():
            group = root[temperature][replica]
            if "trajectory_code" in group and "native_reference" in group:
                pairs.append((str(temperature), str(replica)))
    return sorted(pairs)


def discover(
    archive_root: str | Path,
    source: str,
    domain_ids: list[str] | None = None,
    index_path: str | Path | None = None,
) -> list[TrajRow]:
    """Trajectory rows for ``source``, from the cached index when available."""

    if source not in ARCHIVE_PREFIXES:
        raise ValueError(f"unknown source {source!r}")
    wanted = set(domain_ids) if domain_ids is not None else None
    out: list[TrajRow] = []
    rows = read_index(archive_root, index_path)
    if rows is not None:
        out = [r for r in rows if r.source == source and (wanted is None or r.domain_id in wanted)]
        if wanted is None:
            return out
        # Scan only the requested domains the cached index does not cover.
        wanted = wanted - {r.domain_id for r in out}
        if not wanted:
            return out
    folder = domains_dir(archive_root)
    if wanted is None:
        paths = sorted(folder.glob(f"{ARCHIVE_PREFIXES[source]}*.zarr.zip"))
    else:
        paths = [archive_path(archive_root, source, d) for d in sorted(wanted)]
        paths = [p for p in paths if p.is_file()]
    for path in paths:
        try:
            out.extend(scan_archive(path, source))
        except Exception as exc:  # incomplete or corrupt archive: report and skip
            print(f"[discover] skipping {path.name}: {type(exc).__name__}: {exc}", flush=True)
    return out


def read_split_ids(path: str | Path) -> list[str]:
    """Domain IDs from a split CSV (``name`` column) or a one-ID-per-line text file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(text.splitlines())
        if "name" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no 'name' column")
        ids = [row["name"].strip() for row in reader if row.get("name")]
    else:
        ids = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {path}")
    return ids


def build_targets(
    rows: list[TrajRow],
    archive_root: str | Path,
    temperatures: list[str] | None = None,
    min_replicas: int = 2,
    min_frames: int = 1,
) -> list[Target]:
    """Group rows into (domain, temperature) targets with enough usable replicas."""

    wanted_t = None if temperatures is None else {str(t) for t in temperatures}
    groups: dict[tuple[str, str], list[TrajRow]] = {}
    for row in rows:
        if wanted_t is not None and row.temperature not in wanted_t:
            continue
        if row.n_frames < min_frames:
            continue
        groups.setdefault((row.domain_id, row.temperature), []).append(row)
    targets = []
    for (domain_id, temperature), members in sorted(groups.items()):
        members = sorted(members, key=lambda r: (len(r.replica), r.replica))
        if len(members) < min_replicas:
            continue
        intervals = {round(m.frame_interval_ps, 6) for m in members}
        n_res = {m.n_res for m in members}
        if len(intervals) != 1 or len(n_res) != 1:
            print(f"[targets] skipping {domain_id}/{temperature}: inconsistent replicas", flush=True)
            continue
        first = members[0]
        targets.append(
            Target(
                source=first.source,
                domain_id=domain_id,
                archive_path=str(domains_dir(archive_root) / first.archive_name),
                temperature=temperature,
                replicas=[m.replica for m in members],
                n_frames=[m.n_frames for m in members],
                n_res=first.n_res,
                frame_interval_ps=first.frame_interval_ps,
            )
        )
    return targets


def _decode_text_array(values) -> list[str]:
    out = []
    for v in np.asarray(values).ravel().tolist():
        out.append(v.decode("ascii", errors="replace").strip() if isinstance(v, bytes) else str(v).strip())
    return out


def load_static(path: str | Path) -> dict:
    """Residue names, native DSSP letters and native backbone of one archive."""

    root, store = open_zip_group(path)
    try:
        static = root["static"]
        names = _decode_text_array(static["residue_names"][...])
        dssp = None
        if "dssp" in static:
            raw = np.asarray(static["dssp"][...])
            if raw.ndim == 2 and np.issubdtype(raw.dtype, np.number):
                dssp = [DSSP_STATES[i] for i in np.argmax(raw, axis=-1)]
            else:
                dssp = [s[:1] if s else "C" for s in _decode_text_array(raw)]
        native = (
            np.asarray(static["native_backbone_coords"][...], dtype=np.float64)
            if "native_backbone_coords" in static
            else None
        )
    finally:
        store.close()
    return {"residue_names": names, "dssp": dssp, "native_backbone": native}


def load_native_reference(path: str | Path, temperature: str, replica: str) -> np.ndarray:
    root, store = open_zip_group(path)
    try:
        return np.asarray(root[str(temperature)][str(replica)]["native_reference"][...], dtype=np.float64)
    finally:
        store.close()


def load_replica(
    path: str | Path,
    temperature: str,
    replica: str,
    n_frames: int,
    stride: int = 1,
    chunk_frames: int = 2048,
    keep_backbone: bool = False,
) -> dict:
    """Decode one replica to CA (T, N, 3) float64 and optionally full backbone float32.

    Decoding is chunked to bound the float64 peak inside DynaMode's decoder.
    ``stride > 1`` uses DynaMode's sampled loader (stored sampling views when
    present, e.g. ATLAS stride 100 = 1 ns).
    """

    stride = int(stride)
    if stride == 1:
        pieces_ca, pieces_bb = [], []
        for start in range(0, int(n_frames), int(chunk_frames)):
            stop = min(start + int(chunk_frames), int(n_frames))
            bb = load_backbone_trajectory(path, temperature, replica, start=start, stop=stop)
            pieces_ca.append(bb[:, :, 1, :].astype(np.float64))
            if keep_backbone:
                pieces_bb.append(bb.astype(np.float32, copy=False))
        ca = np.concatenate(pieces_ca, axis=0)
        bb_all = np.concatenate(pieces_bb, axis=0) if keep_backbone else None
    else:
        count = (int(n_frames) - 1) // stride + 1
        bb = load_backbone_trajectory_sampled(path, temperature, replica, start=0, count=count, stride=stride)
        ca = bb[:, :, 1, :].astype(np.float64)
        bb_all = bb.astype(np.float32, copy=False) if keep_backbone else None
    return {
        "ca": ca,
        "backbone": bb_all,
        "native_reference": load_native_reference(path, temperature, replica),
    }


__all__ += ["archive_path", "domains_dir", "load_native_reference", "read_index", "scan_archive"]
