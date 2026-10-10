"""Memory-aware admission of targets into a pool of worker processes.

Packing many proteins onto one GH200 is limited by memory, not cores: a
1000-residue ATLAS chain needs several GB of host memory and GPU memory, a
100-residue domain a few hundred MB.  Targets are started longest-first, and
only when their estimated host and GPU footprints fit the remaining budgets.
Smaller targets backfill while a large one waits, but once the head of the
queue has waited ``reserve_after_s`` its memory is reserved so it cannot
starve.  A target larger than the whole budget still runs, alone.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AdaptiveFootprints", "Footprint", "detect_host_memory_gb", "estimate_footprint", "run_admitted"]


@dataclass(frozen=True)
class Footprint:
    host_gb: float
    gpu_gb: float


def _cgroup_limit_bytes() -> int | None:
    """Memory limit of this process's own cgroup (Slurm job step), v2 or v1."""

    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            base, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):
            base, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        node = base / path.lstrip("/")
        while True:  # walk up: the limit may be set on the job, not the step
            try:
                text = (node / name).read_text().strip()
                if text and text != "max" and 0 < int(text) < 1 << 60:
                    return int(text)
            except (OSError, ValueError):
                pass
            if node == base or node.parent == node:
                break
            node = node.parent
    return None


def detect_host_memory_gb() -> float:
    """Memory this job may use: its cgroup limit (Slurm), else MemAvailable."""

    limit = _cgroup_limit_bytes()
    if limit is not None:
        return limit / 1e9
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9


def estimate_footprint(target, cfg: dict, gpu: bool, stride: int = 1) -> Footprint:
    """Conservative peak host/GPU memory (GB) of ``process_target`` for one target.

    Dominant terms: float64 CA trajectories of every replica (plus centred
    copies), one replica's backbone in float32 with a float64 working copy for
    its PCA, the pooled backbone covariance when enabled, and per-process
    runtime overhead (interpreter, numpy/torch, CUDA context).
    """

    n = int(target.n_res)
    frames = [max(1, int(f) // max(int(stride), 1)) for f in target.n_frames]
    t_max, t_sum = max(frames), sum(frames)
    gb = 1e-9
    ca = t_sum * n * 3 * 8 * 2.5
    backbone = t_max * n * 12 * (4 + 8 + 8)
    dim_bb = 12 * n
    pooled = 2 * dim_bb * dim_bb * 8 if dim_bb <= int(cfg["diagnostics"]["bb_pooled_max_dim"]) else 0.0
    k = min(3 * n, int(cfg["spectral"]["pc_max"]))
    # Batched stage-1 solves: residual targets (L x M x N x 9), local-fit ADMM
    # state (~6 copies of B x M x N x 9) and collective-fit state (6 x B x F x K^2),
    # with L~16 lags, M~66 local models, B~10 bands, F~5 folds.
    fits = 8.0 * (16 * 66 * n * 9 + 6 * 10 * 66 * n * 9 + 6 * 10 * 5 * k * k) + 1e8
    runtime_host = 1.2 if gpu else 0.4
    host = (ca + backbone + pooled + fits) * gb + runtime_host
    if not gpu:
        return Footprint(host, 0.0)
    device = (t_max * n * 3 * 8 * 4 + t_max * dim_bb * 8 * 2.5 + pooled + fits) * gb + 0.6
    return Footprint(host, device)


class AdaptiveFootprints:
    """Footprints whose GPU part is corrected by what targets actually used.

    ``estimate_footprint`` counts the arrays a target needs; on the GPU the
    caching allocator, fragmentation and library workspaces add a large,
    unknown factor.  GPU estimates are multiplied by ``factor``, which starts
    conservative, is recalibrated from observed peaks (1.25 x the largest
    observed/estimated ratio once three targets have finished), and is raised
    1.5x whenever a target still runs out of GPU memory (at most x16); it never
    drops below a value set by such an increase.
    """

    def __init__(self, base: dict | None = None, factor: float = 4.0):
        self.base: dict = dict(base or {})
        self.factor = float(factor)
        self.floor = 1.0
        self.max_factor = 16.0  # beyond this the estimate is not the problem
        self.ratios: list[float] = []

    def __getitem__(self, key) -> Footprint:
        fp = self.base[key]
        return Footprint(fp.host_gb, fp.gpu_gb * self.factor)

    def __contains__(self, key) -> bool:
        return key in self.base

    def values(self):
        return [self[k] for k in self.base]

    def observe(self, key, peak_gb: float | None) -> None:
        fp = self.base.get(key)
        if not peak_gb or fp is None or fp.gpu_gb <= 0:
            return
        self.ratios.append(peak_gb / fp.gpu_gb)
        if len(self.ratios) >= 3:
            self.factor = min(max(self.floor, 1.25 * max(self.ratios)), self.max_factor)

    def out_of_memory(self) -> None:
        self.floor = self.factor = min(self.factor * 1.5, self.max_factor)


def run_admitted(make_executor, func, tasks: list, footprints: dict, jobs: int, host_budget_gb: float,
                 gpu_budget_gb: float, on_done, reserve_after_s: float = 60.0, poll_s: float = 0.25,
                 claim=None) -> None:
    """Run ``func(task.to_dict())`` for every task under memory budgets.

    ``make_executor()`` returns a fresh ``concurrent.futures`` process pool.
    ``tasks`` are ordered by priority (longest first); ``footprints`` maps
    task.key -> Footprint; ``on_done(task, outcome)`` is called as each ends,
    and may return a task to run again, alone (e.g. after a GPU out-of-memory).
    ``claim(task) -> bool``, if given, is asked right before a task starts;
    False drops it (another worker holds it, or it would not finish before
    the walltime).  It must return True again for a task it already accepted.

    If a worker process dies (e.g. killed for exceeding memory) every in-flight
    future is lost.  The pool is rebuilt and those targets become *suspects*,
    re-run one at a time with nothing alongside: a suspect that kills the pool
    on its own is reported failed, the innocent ones simply complete.
    """

    from concurrent.futures.process import BrokenProcessPool

    pending = deque(tasks)
    suspects: deque = deque()
    running: dict = {}  # key -> (task, footprint, future, solo)
    used = {"host": 0.0, "gpu": 0.0}
    head_since = time.monotonic()
    head_key = pending[0].key if pending else None
    executor = make_executor()

    def fits(fp: Footprint, reserve: Footprint | None) -> bool:
        if not running:
            return True  # never deadlock: an oversized target runs alone
        extra_h = reserve.host_gb if reserve else 0.0
        extra_g = reserve.gpu_gb if reserve else 0.0
        return (used["host"] + fp.host_gb + extra_h <= host_budget_gb
                and used["gpu"] + fp.gpu_gb + extra_g <= gpu_budget_gb)

    def submit(task, solo: bool) -> bool:
        """Submit; False if the pool is already broken (task not consumed)."""
        try:
            future = executor.submit(func, task.to_dict())
        except BrokenProcessPool:
            return False
        fp = footprints[task.key]
        running[task.key] = (task, fp, future, solo)
        used["host"] += fp.host_gb
        used["gpu"] += fp.gpu_gb
        return True

    try:
        while pending or suspects or running:
            broken = False
            if suspects:
                if not running:
                    if submit(suspects[0], solo=True):
                        suspects.popleft()
                    else:
                        broken = True
            else:
                while pending and len(running) < jobs:
                    head = pending[0]
                    if head.key != head_key:
                        head_key, head_since = head.key, time.monotonic()
                    chosen = None
                    if fits(footprints[head.key], None):
                        chosen = 0
                    else:
                        reserve = footprints[head.key] if time.monotonic() - head_since > reserve_after_s else None
                        for i in range(1, len(pending)):
                            if fits(footprints[pending[i].key], reserve):
                                chosen = i
                                break
                    if chosen is None:
                        break
                    if claim is not None and not claim(pending[chosen]):
                        del pending[chosen]
                        continue
                    if not submit(pending[chosen], solo=False):
                        broken = True
                        break
                    del pending[chosen]
            finished = [key for key, (_, _, fut, _) in running.items() if fut.done()]
            if not finished and not broken:
                time.sleep(poll_s)
                continue
            for key in finished:
                task, fp, fut, solo = running[key]
                try:
                    outcome = fut.result()
                except BrokenProcessPool:
                    broken = True
                    continue
                running.pop(key)
                used["host"] -= fp.host_gb
                used["gpu"] -= fp.gpu_gb
                again = on_done(task, outcome)
                if again is not None:
                    suspects.append(again)
            if broken:
                executor.shutdown(wait=False, cancel_futures=True)
                executor = make_executor()
                lost = list(running.values())
                running.clear()
                used["host"] = used["gpu"] = 0.0
                if not lost:
                    pass  # pool broke between submissions: nothing in flight to retry
                elif len(lost) == 1 and lost[0][3]:
                    task = lost[0][0]
                    on_done(task, (task.key, "failed", float("nan"),
                                   "worker process died running this target alone (likely out of memory); "
                                   "lower FLEXC_JOBS or raise the memory request"))
                else:
                    suspects.extend(item[0] for item in lost)
    finally:
        executor.shutdown(wait=True)
