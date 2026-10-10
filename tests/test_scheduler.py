"""Memory-aware admission: budgets respected, no starvation, worker death survived."""

import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass

from flexcompress.config import DEFAULTS
from flexcompress.data import Target
from flexcompress.scheduler import AdaptiveFootprints, Footprint, estimate_footprint, run_admitted


@dataclass
class FakeTask:
    key: str
    seconds: float = 0.05

    def to_dict(self):
        return asdict(self)


_LOCK = threading.Lock()
_STATE = {"host": 0.0, "peak": 0.0, "running": 0, "max_running": 0}
_FP = {}


def _thread_job(task: dict):
    fp = _FP[task["key"]]
    with _LOCK:
        _STATE["host"] += fp.host_gb
        _STATE["running"] += 1
        _STATE["peak"] = max(_STATE["peak"], _STATE["host"])
        _STATE["max_running"] = max(_STATE["max_running"], _STATE["running"])
    time.sleep(task["seconds"])
    with _LOCK:
        _STATE["host"] -= fp.host_gb
        _STATE["running"] -= 1
    return task["key"], "ok", task["seconds"], None


def test_admission_respects_memory_budget_and_backfills():
    sizes = {"big1": 5.0, "big2": 5.0, "big3": 5.0, "s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 1.0}
    _FP.clear()
    _FP.update({k: Footprint(v, 0.0) for k, v in sizes.items()})
    _STATE.update(host=0.0, peak=0.0, running=0, max_running=0)
    tasks = [FakeTask(k) for k in sizes]  # longest first
    done = []
    run_admitted(lambda: ThreadPoolExecutor(max_workers=4), _thread_job, tasks, _FP, jobs=4,
                 host_budget_gb=7.0, gpu_budget_gb=float("inf"), on_done=lambda t, o: done.append(o[0]),
                 poll_s=0.005)
    assert sorted(done) == sorted(sizes)
    assert _STATE["peak"] <= 7.0 + 1e-9  # two bigs never run together
    assert _STATE["max_running"] >= 2  # small ones backfilled next to a big one


def test_oversized_target_runs_alone():
    _FP.clear()
    _FP.update({"huge": Footprint(50.0, 0.0), "s": Footprint(1.0, 0.0)})
    _STATE.update(host=0.0, peak=0.0, running=0, max_running=0)
    done = []
    run_admitted(lambda: ThreadPoolExecutor(max_workers=2), _thread_job, [FakeTask("huge"), FakeTask("s")], _FP,
                 jobs=2, host_budget_gb=10.0, gpu_budget_gb=float("inf"), on_done=lambda t, o: done.append(o),
                 poll_s=0.005)
    assert {o[0] for o in done} == {"huge", "s"} and _STATE["max_running"] == 1


def _die_or_work(task: dict):
    if task["key"] == "bad":
        os._exit(13)  # simulate an out-of-memory kill
    time.sleep(task["seconds"])
    return task["key"], "ok", task["seconds"], None


def test_dead_worker_is_retried_then_reported():
    fps = {k: Footprint(1.0, 0.0) for k in ("a", "bad", "b", "c")}
    done = {}
    ctx = mp.get_context("spawn")
    run_admitted(lambda: ProcessPoolExecutor(max_workers=2, mp_context=ctx), _die_or_work,
                 [FakeTask(k, 0.2) for k in fps], fps, jobs=2, host_budget_gb=100.0,
                 gpu_budget_gb=float("inf"), on_done=lambda t, o: done.__setitem__(o[0], o), poll_s=0.01)
    assert set(done) == set(fps)
    assert done["bad"][1] == "failed" and "out of memory" in done["bad"][3]  # culprit isolated
    assert all(done[k][1] == "ok" for k in ("a", "b", "c"))


def test_footprint_estimates_scale_with_size():
    def target(n, frames):
        return Target("atlas", "x", "", "300", ["0", "1", "2"], [frames] * 3, n, 10.0)

    small = estimate_footprint(target(100, 10000), DEFAULTS, gpu=True)
    large = estimate_footprint(target(1000, 10000), DEFAULTS, gpu=True)
    assert large.host_gb > 3 * small.host_gb and large.gpu_gb > 3 * small.gpu_gb
    assert 0.5 < small.host_gb < 4 and 2 < large.host_gb < 20
    assert estimate_footprint(target(100, 10000), DEFAULTS, gpu=False).gpu_gb == 0.0
    strided = estimate_footprint(target(1000, 10000), DEFAULTS, gpu=True, stride=100)
    assert strided.host_gb < 0.5 * large.host_gb


def test_on_done_can_requeue_a_task_to_run_alone():
    _FP.clear()
    _FP.update({k: Footprint(1.0, 0.0) for k in ("a", "b", "c")})
    _STATE.update(host=0.0, peak=0.0, running=0, max_running=0)
    seen, retried = [], set()

    def on_done(task, outcome):
        seen.append(task.key)
        if task.key == "b" and "b" not in retried:  # e.g. GPU out of memory: once more, alone
            retried.add("b")
            return FakeTask("b", 0.02)
        return None

    run_admitted(lambda: ThreadPoolExecutor(max_workers=3), _thread_job, [FakeTask(k) for k in "abc"], _FP,
                 jobs=3, host_budget_gb=10.0, gpu_budget_gb=float("inf"), on_done=on_done, poll_s=0.005)
    assert sorted(seen) == ["a", "b", "b", "c"]


def test_adaptive_gpu_footprints_learn_from_peaks_and_ooms():
    sizes = AdaptiveFootprints({k: Footprint(1.0, 2.0) for k in "abcd"}, factor=4.0)
    assert sizes["a"].gpu_gb == 8.0 and sizes["a"].host_gb == 1.0
    sizes.observe("a", 3.0)
    sizes.observe("b", 4.0)
    assert sizes.factor == 4.0  # needs three observations
    sizes.observe("c", 5.0)  # largest ratio 2.5 -> 1.25 x 2.5
    assert abs(sizes.factor - 3.125) < 1e-12
    sizes.out_of_memory()
    assert abs(sizes.factor - 4.6875) < 1e-12
    sizes.observe("d", 2.0)  # small peaks never lower it below the post-OOM floor
    assert abs(sizes.factor - 4.6875) < 1e-12


def test_adaptive_factor_is_capped():
    sizes = AdaptiveFootprints({"a": Footprint(1.0, 1.0)}, factor=4.0)
    for _ in range(20):
        sizes.out_of_memory()
    assert sizes.factor == 16.0
