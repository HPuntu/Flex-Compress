"""Shared work queue: every array task of a run pulls targets from the whole list.

A run is the full training split of one dataset.  Instead of fixing each
array task's targets in advance (static shards, where one slow task can hold
up the run or hit the walltime while others sit idle), every task claims the
largest unfinished target that nobody else holds, so any number of GH200s can
work through one split and finish together.

Claims are files ``RUN/claims/<key>.claim`` created with ``O_CREAT|O_EXCL``,
which is atomic on Lustre, GPFS and NFSv3+.  A background thread refreshes
the claims a worker holds every ``heartbeat_s``.  A claim is dead, and its
target can be taken over (one taker at a time, under a per-target lock), when its heartbeat is older than ``stale_s`` or
when its owning Slurm array has been recorded as finished in
``RUN/claims/DEAD_JOBS`` (done by the finalize job that follows each round),
so a round killed at the walltime is picked up immediately by the next.

Workers also watch the walltime: a target is only started when the time left
exceeds its predicted runtime (learned from targets this worker has already
finished) with a safety margin, so tasks stop cleanly and leave the rest of
the split to the next round instead of being killed half-way.
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path

__all__ = ["ClaimQueue", "DeadlineGate", "parse_slurm_duration", "progress", "slurm_deadline"]

DEAD_JOBS = "DEAD_JOBS"


def _owner_job() -> str:
    """Identity of this worker's Slurm array (all its tasks share it)."""

    for name in ("SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"):
        if os.environ.get(name):
            return os.environ[name]
    return f"local-{socket.gethostname()}-{os.getpid()}"


class ClaimQueue:
    """Exclusive claims on targets, shared by every worker of a run."""

    def __init__(self, run_dir: str | Path, stale_s: float = 900.0, heartbeat_s: float = 60.0):
        self.dir = Path(run_dir) / "claims"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.job = _owner_job()
        self.token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
        self.stale_s = float(stale_s)
        self.heartbeat_s = float(heartbeat_s)
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dead_cache: tuple[tuple[int, int], set[str]] = ((-1, -1), set())

    # ------------------------------------------------------------ claim files
    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.claim"

    def dead_jobs(self) -> set[str]:
        path = self.dir / DEAD_JOBS
        try:
            st = path.stat()
        except FileNotFoundError:
            return set()
        stamp = (st.st_mtime_ns, st.st_size)  # size too: coarse mtimes on some file systems
        if stamp != self._dead_cache[0]:
            text = path.read_text(encoding="utf-8")
            self._dead_cache = (stamp, {line.strip() for line in text.splitlines() if line.strip()})
        return self._dead_cache[1]

    def mark_dead(self, job: str) -> None:
        """Record that Slurm job ``job`` has ended: its claims are released."""

        with open(self.dir / DEAD_JOBS, "a", encoding="utf-8") as fh:
            fh.write(f"{job}\n")

    def _read(self, path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def is_live(self, path: Path, now: float | None = None) -> bool:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return False
        if (time.time() if now is None else now) - mtime > self.stale_s:
            return False
        owner = self._read(path)
        # An empty read is a claim being written right now: treat it as live.
        return not owner or str(owner.get("job")) not in self.dead_jobs()

    def try_claim(self, key: str) -> bool:
        """Claim ``key`` exclusively; False if a live worker holds it."""

        with self._lock:
            if key in self._held:
                return True
        path = self._path(key)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if self.is_live(path):
                return False
            return self._take_over(key, path)
        self._write_claim(key, fd)
        return True

    def _write_claim(self, key: str, fd: int) -> None:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"job": self.job, "token": self.token, "host": socket.gethostname(),
                       "pid": os.getpid(), "claimed": time.time()}, fh)
        with self._lock:
            self._held.add(key)

    def _take_over(self, key: str, path: Path) -> bool:
        """Replace a dead claim with ours.  A per-target lock serialises takers,
        and the claim is re-checked under it, so a fresh claim made by another
        taker is never removed."""

        lock = path.with_name(f"{path.name}.takeover")
        try:
            lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:  # a taker that died holding the lock
                if time.time() - lock.stat().st_mtime > self.stale_s:
                    lock.unlink()
            except FileNotFoundError:
                pass
            return False
        os.close(lock_fd)
        try:
            if self.is_live(path):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                return False  # a worker on the normal path claimed it first: theirs
            self._write_claim(key, fd)
            return True
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def release(self, key: str) -> None:
        with self._lock:
            self._held.discard(key)
        path = self._path(key)
        if self._read(path).get("token") == self.token:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def held(self) -> set[str]:
        with self._lock:
            return set(self._held)

    def live_keys(self) -> set[str]:
        now = time.time()
        return {p.name[: -len(".claim")] for p in self.dir.glob("*.claim") if self.is_live(p, now)}

    def live_jobs(self) -> set[str]:
        """Owner arrays of the live claims."""

        now = time.time()
        return {str(self._read(p).get("job")) for p in self.dir.glob("*.claim") if self.is_live(p, now)}

    # -------------------------------------------------------------- heartbeat
    def _beat(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            for key in self.held():
                try:
                    os.utime(self._path(key))
                except OSError:
                    pass

    def start(self) -> ClaimQueue:
        if self._thread is None:
            self._thread = threading.Thread(target=self._beat, name="flexc-claims", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        """Stop the heartbeat and release anything still held."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        for key in self.held():
            self.release(key)


# ---------------------------------------------------------------- walltime


def parse_slurm_duration(text: str) -> float | None:
    """Seconds in a Slurm duration: ``D-HH:MM:SS``, ``HH:MM:SS``, ``MM:SS``."""

    text = (text or "").strip()
    if not text or not text[0].isdigit():
        return None  # UNLIMITED, INVALID, NOT_SET
    days = 0
    if "-" in text:
        head, text = text.split("-", 1)
        days = int(head)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return float(((days * 24 + h) * 60 + m) * 60 + s)


def slurm_deadline() -> float | None:
    """Unix time at which this Slurm job will be stopped; None outside Slurm."""

    end = os.environ.get("SLURM_JOB_END_TIME", "")
    if end.isdigit():
        return float(end)
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    left = parse_slurm_duration(out.splitlines()[0] if out.strip() else "")
    return None if left is None else time.time() + left


class DeadlineGate:
    """Start a target only if it should finish before the walltime.

    The runtime of a target is predicted as (wall seconds per unit cost) x its
    cost, using the median rate of targets this worker has finished, measured
    under the same concurrency.  Until one has finished, targets start while
    more than ``first_s`` remain.
    """

    def __init__(self, deadline: float | None, margin_s: float = 600.0, safety: float = 1.5,
                 first_s: float = 1800.0):
        self.deadline = deadline
        self.margin_s = float(margin_s)
        self.safety = float(safety)
        self.first_s = float(first_s)
        self.rates: list[float] = []
        self.refused = 0

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def record(self, cost: float, seconds: float) -> None:
        if cost > 0 and seconds > 0:
            self.rates.append(seconds / cost)

    def predicted(self, cost: float) -> float | None:
        return statistics.median(self.rates) * cost if self.rates else None

    def allows(self, cost: float) -> bool:
        if self.deadline is None:
            return True
        left = self.time_left()
        need = self.predicted(cost)
        ok = left > (self.first_s if need is None else self.safety * need + self.margin_s)
        if not ok:
            self.refused += 1
        return ok


# ---------------------------------------------------------------- progress


def _keys(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {p.name[:-5] for p in os.scandir(directory) if p.name.endswith(".json")}


def done_keys(run_dir: Path, sources) -> set[str]:
    """Targets with a result file (a run's config is frozen, so any result in it is current)."""

    out: set[str] = set()
    for source in sources:
        out |= _keys(Path(run_dir) / "results" / source)
    return out


def error_keys(run_dir: Path, sources, since: float | None = None) -> set[str]:
    """Targets whose last attempt failed (optionally: only failures written at or after ``since``)."""

    out: set[str] = set()
    for source in sources:
        directory = Path(run_dir) / "errors" / source
        if not directory.is_dir():
            continue
        for entry in os.scandir(directory):
            if entry.name.endswith(".json") and (since is None or entry.stat().st_mtime >= since):
                out.add(entry.name[:-5])
    return out


def progress(run_dir: str | Path, listing: dict | None = None) -> dict:
    """Coverage of a run: how much of the configured split is done."""

    run_dir = Path(run_dir)
    if listing is None:
        listing = json.loads((run_dir / "targets.json").read_text(encoding="utf-8"))
    keys = {f"{t['domain_id']}__T{t['temperature']}" for t in listing["targets"]}
    sources = sorted({t["source"] for t in listing["targets"]})
    done = done_keys(run_dir, sources) & keys
    failed = (error_keys(run_dir, sources) & keys) - done
    live = (ClaimQueue(run_dir).live_keys() & keys) - done
    remaining = keys - done - failed - live
    cov = listing.get("coverage", {})
    return {
        "mode": cov.get("mode"),
        "split_domains": cov.get("n_split_domains"),
        "domains_missing_archive": len(listing.get("missing_domains", [])),
        "targets_excluded": len(cov.get("excluded_targets", [])),
        "targets_deselected": cov.get("n_deselected", 0),
        "targets": len(keys),
        "done": len(done),
        "failed": len(failed),
        "running": len(live),
        "remaining": len(remaining),
        "complete": not remaining and not live,
        "failed_keys": sorted(failed),
    }
