"""Command line: configure a frozen run, work through it, aggregate.

    python -m flexcompress.run configure --config configs/stage01_atlas.yaml \
        --run-dir /path/to/runs/atlas-full --shards 16
    python -m flexcompress.run worker --run-dir RUN          # any number at once
    python -m flexcompress.run status --run-dir RUN
    python -m flexcompress.run aggregate --run-dir RUN
    python -m flexcompress.run local --config ... --run-dir RUN --set selection.domains=[1abcA00]

``configure`` freezes the resolved config, the target list (by default the
whole training split), a coverage record of split domains that yield no
target and why, provenance, and a copy of the code (Flex-Compress and the
DynaMode-dev data layer) under ``RUN/code/src``; the Slurm workers run that
snapshot, so later edits never change a running study.  Workers share one
queue (``workqueue.py``): each claims the largest unfinished target nobody
else holds, so any number of them finish the split together.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import accel
from .config import load_config, resolve_config
from .data import build_targets, discover, read_split_ids
from .io_utils import atomic_write_json, atomic_write_npz, environment, git_state, now_utc, read_json
from .pipeline import config_hash, process_target

REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


# --------------------------------------------------------------------------- configure


def _select(targets, cfg):
    sel = cfg["selection"]
    if sel["min_residues"] is not None:
        targets = [t for t in targets if t.n_res >= int(sel["min_residues"])]
    if sel["max_residues"] is not None:
        targets = [t for t in targets if t.n_res <= int(sel["max_residues"])]
    if sel["mode"] == "pilot":
        lengths = {}
        for t in targets:
            lengths[t.domain_id] = t.n_res
        domains = sorted(lengths, key=lambda d: (lengths[d], d))
        size = int(sel["pilot_size"])
        if len(domains) > size:
            pick = {domains[int(round(i * (len(domains) - 1) / max(size - 1, 1)))] for i in range(size)}
            targets = [t for t in targets if t.domain_id in pick]
    elif sel["mode"] != "full":
        raise ValueError("selection.mode must be 'full' or 'pilot'")
    return targets


def _exclusions(domains, rows, eligible, cfg) -> list[dict]:
    """Why each (split domain, temperature) with an archive yields no target."""

    by_domain: dict[str, list] = {}
    for row in rows:
        by_domain.setdefault(row.domain_id, []).append(row)
    have = {(t.domain_id, t.temperature) for t in eligible}
    wanted = None if cfg["temperatures"] is None else [str(t) for t in cfg["temperatures"]]
    min_frames, min_reps = int(cfg["min_frames"]), int(cfg["min_replicas"])
    out = []
    for domain in domains:
        members_all = by_domain.get(domain, [])
        if not members_all:
            continue  # no archive: listed under missing_domains
        for temp in wanted or sorted({r.temperature for r in members_all}):
            if (domain, temp) in have:
                continue
            members = [r for r in members_all if r.temperature == temp]
            usable = [r for r in members if r.n_frames >= min_frames]
            if not members:
                reason, detail = "no_replicas_at_temperature", f"no replicas at {temp} K"
            elif len(usable) < min_reps:
                reason = "too_few_replicas"
                detail = f"{len(usable)} of {len(members)} replicas have >= {min_frames} frames (need {min_reps})"
            else:
                reason, detail = "inconsistent_replicas", "replicas differ in frame interval or residue count"
            out.append({"domain": domain, "temperature": temp, "reason": reason, "detail": detail})
    return out


def _assign_shards(targets, n_shards: int):
    for t in targets:
        frames = float(sum(t.n_frames))
        dim = 3.0 * t.n_res
        t.cost = frames * (min(dim, 256.0) ** 2 + 9.0 * t.n_res) + dim**3 / 10.0
    loads = np.zeros(n_shards)
    for t in sorted(targets, key=lambda x: -x.cost):
        shard = int(np.argmin(loads))
        t.shard = shard
        loads[shard] += t.cost
    return loads


def _snapshot(run_dir: Path) -> None:
    code = run_dir / "code" / "src"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    shutil.copytree(REPO_ROOT / "src" / "flexcompress", code / "flexcompress", ignore=ignore)
    dynamode = REPO_ROOT / "DynaMode-dev" / "src" / "dynamode"
    if not dynamode.is_dir():
        raise FileNotFoundError(f"DynaMode-dev submodule not found at {dynamode}; run git submodule update --init")
    shutil.copytree(dynamode, code / "dynamode", ignore=ignore)
    for name in ("worker.sbatch", "aggregate.sbatch", "flexc_env.sh", "launch.sh"):
        source = REPO_ROOT / "sbatch" / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)
    if (REPO_ROOT / "sbatch" / "sites").is_dir():
        shutil.copytree(REPO_ROOT / "sbatch" / "sites", run_dir / "sites")


def configure(args) -> Path:
    cfg = resolve_config(load_config(args.config, args.set), REPO_ROOT)
    if args.mode:
        cfg["selection"]["mode"] = args.mode
    run_dir = Path(args.run_dir).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} exists and is not empty; use a fresh run directory")

    domains = cfg["selection"]["domains"]
    if domains is None:
        domains = cfg["split"]["ids"] or read_split_ids(cfg["split"]["file"])
    log(f"discovering {cfg['source']} archives under {cfg['archive_root']} for {len(domains)} domains")
    rows = discover(cfg["archive_root"], cfg["source"], list(domains), cfg["index_path"])
    eligible = build_targets(rows, cfg["archive_root"], cfg["temperatures"], int(cfg["min_replicas"]),
                             int(cfg["min_frames"]))
    with_archive = {r.domain_id for r in rows}
    missing = sorted(set(domains) - with_archive)
    excluded = _exclusions(domains, rows, eligible, cfg)
    targets = _select(eligible, cfg)
    if not targets:
        raise SystemExit("no targets selected; check archive_root, split and temperatures")
    reasons: dict[str, int] = {}
    for item in excluded:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    coverage = {
        "mode": cfg["selection"]["mode"],
        "split": cfg["split"]["file"] if cfg["selection"]["domains"] is None and cfg["split"]["ids"] is None
        else "explicit list",
        "n_split_domains": len(domains),
        "n_domains_with_archive": len(with_archive & set(domains)),
        "n_domains_with_targets": len({t.domain_id for t in eligible}),
        "n_targets_eligible": len(eligible),
        "n_deselected": len(eligible) - len(targets),
        "excluded_reasons": reasons,
        "excluded_targets": excluded,
    }
    n_shards = max(1, min(int(args.shards), len(targets)))
    loads = _assign_shards(targets, n_shards)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    cfg["run"] = {"n_shards": n_shards, "created": now_utc(), "config_hash": config_hash(cfg)}
    atomic_write_json(run_dir / "config.json", cfg)
    atomic_write_json(run_dir / "targets.json", {
        "n_targets": len(targets),
        "n_shards": n_shards,
        "missing_domains": missing,
        "coverage": coverage,
        "shard_cost_ratio_max_min": float(loads.max() / max(loads.min(), 1e-300)),
        "targets": [t.to_dict() for t in targets],
    })
    atomic_write_json(run_dir / "provenance.json", {
        "created": now_utc(),
        "argv": sys.argv,
        "flex_compress": git_state(REPO_ROOT),
        "dynamode_dev": git_state(REPO_ROOT / "DynaMode-dev"),
        "environment": environment(),
    })
    _snapshot(run_dir)
    log(f"split: {len(domains)} domains, {coverage['n_domains_with_archive']} with archives "
        f"({len(missing)} missing), {len(eligible)} eligible targets"
        + (f", {len(excluded)} (domain, temperature) excluded {reasons}" if excluded else ""))
    log(f"run {run_dir}: {len(targets)} targets ({coverage['mode']}"
        + (f", {coverage['n_deselected']} deselected" if coverage["n_deselected"] else "")
        + f") for {n_shards} workers per round")
    print(run_dir)
    return run_dir


# --------------------------------------------------------------------------- worker


def _result_paths(run_dir: Path, target) -> tuple[Path, Path, Path]:
    base = run_dir / "results" / target.source
    return (base / f"{target.key}.json",
            run_dir / "descriptors" / target.source / f"{target.key}.npz",
            run_dir / "errors" / target.source / f"{target.key}.json")


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_CHILD: dict = {}


def available_cpus() -> int:
    """CPUs this process may use (respects Slurm cgroups / affinity)."""

    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _device_of(backend: str) -> str:
    """The ``--device`` value that selects ``backend`` (an ``accel.info()['backend']``)."""

    if not backend.startswith("torch:"):
        return "numpy"
    dev = backend.split(":", 1)[1]
    return "torch-cpu" if dev == "cpu" else dev


GPU_RETRY = "gpu_retry"  # outcome status: the shared GPU ran out of memory; run the target again


def _process_one(run_dir: Path, cfg: dict, target, tag: str) -> tuple:
    """Run one target and write its outputs atomically; never raises.

    Returns ``(key, status, seconds, error, info)``.  A GPU out-of-memory error
    (or cuSOLVER failing for lack of memory) is not a property of the target:
    it returns status ``gpu_retry`` and records no failure, and the worker runs
    the target again, alone on the GPU and then on the CPU if need be.
    ``target.extra["device"]`` forces a backend for this one target.
    """

    out_json, out_npz, err_json = _result_paths(run_dir, target)
    started = time.monotonic()
    forced = target.extra.get("device")
    previous = accel.info()["backend"]
    if forced:
        _configure_compute(cfg, forced)
    log(f"{tag} {target.key}: start N={target.n_res} reps={len(target.replicas)} frames={target.n_frames}"
        + (f" device={forced}" if forced else ""))
    try:
        accel.reset_peak()
        result, descriptors = process_target(target, cfg, log=lambda m: log(f"{tag} {target.key}:{m}"))
        info = {"gpu_peak_gb": accel.peak_gb()}
        result["finished"] = now_utc()
        result["compute"]["gpu_peak_gb"] = info["gpu_peak_gb"]
        if descriptors is not None:
            atomic_write_npz(out_npz, **descriptors)
        atomic_write_json(out_json, result)
        if err_json.is_file():
            err_json.unlink()
        return target.key, result["status"], time.monotonic() - started, None, info
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".split("\n", 1)[0]
        if accel.is_gpu_memory_error(exc):
            return target.key, GPU_RETRY, time.monotonic() - started, message, {}
        atomic_write_json(err_json, {
            "target": target.to_dict(), "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "time": now_utc(), "compute": accel.info(),
        })
        return target.key, "failed", time.monotonic() - started, message, {}
    finally:
        accel.release()
        if forced:
            _configure_compute(cfg, _device_of(previous))


def _configure_compute(cfg: dict, device: str | None) -> str:
    comp = cfg.get("compute", {})
    return accel.configure(device or comp.get("device", "auto"), comp.get("dtype", "float64"),
                           comp.get("psd_method", "auto"), int(comp.get("ns_iters", 40)))


def _child_init(run_dir: str, device: str | None) -> None:
    _CHILD["run_dir"] = Path(run_dir)
    _CHILD["cfg"] = read_json(Path(run_dir) / "config.json")
    _configure_compute(_CHILD["cfg"], device)


def _child_run(target_dict: dict):
    from .data import Target

    return _process_one(_CHILD["run_dir"], _CHILD["cfg"], Target.from_dict(target_dict), f"[pid {os.getpid()}]")


def worker(args) -> int:
    """Work through a run together with every other worker of it.

    Workers share one queue: each claims the largest target that is neither
    finished, failed, nor held by a live worker, so any number of array tasks
    (and resubmitted rounds) cover the whole split between them.  ``--jobs N``
    runs N targets at once in spawned processes, admitted by memory, each with
    ``--threads`` BLAS threads, so a whole-superchip job (Isambard-AI: 72 cores
    and an H100 per GH200) stays busy.  Near the walltime no new target is
    started unless it is predicted to finish; the next round takes the rest.
    """

    from .data import Target
    from .workqueue import ClaimQueue, DeadlineGate, done_keys, error_keys, slurm_deadline

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = read_json(run_dir / "config.json")
    listing = read_json(run_dir / "targets.json")
    label = args.shard if args.shard is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    targets = [Target.from_dict(t) for t in listing["targets"]]
    sources = sorted({t.source for t in targets})
    started = time.time()
    force = bool(getattr(args, "force", False))
    # Failures recorded before the cutoff are retried.  resume.sh passes one
    # cutoff to every task of its array, so a target that fails again during
    # the resume is not retried by the tasks that start after it.
    retry_since = float(getattr(args, "retry_since", 0) or 0) or (
        started if getattr(args, "retry_failed", False) else None)
    req_jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    threads = int(getattr(args, "threads", 0) or 0) or max(1, available_cpus() // req_jobs)
    device = getattr(args, "device", None)
    backend = _configure_compute(cfg, device)  # also validates the device before spawning
    import dynamode.dataloader.compressed_backbone as cb

    wanted = float(getattr(args, "deadline", 0) or 0)
    deadline = None if wanted < 0 else (wanted or slurm_deadline())
    gate = DeadlineGate(deadline, margin_s=float(getattr(args, "end_margin_s", 600.0)))
    queue = ClaimQueue(run_dir, stale_s=float(getattr(args, "claim_stale_s", 900.0))).start()
    attempted: set[str] = set()
    claimed_at: dict[str, float] = {}
    outcomes: list = []
    from .scheduler import AdaptiveFootprints

    sizes = AdaptiveFootprints()
    breaker = {"tripped": False}
    log(f"worker {label}: {len(targets)} targets in run; jobs={req_jobs} threads/job={threads} "
        f"cpus={available_cpus()} compute={accel.info()}; "
        + (f"walltime left {gate.time_left() / 3600:.1f} h; " if deadline else "no walltime; ")
        + f"flexcompress={Path(__file__).parent} dynamode={Path(cb.__file__).parent}")

    def scan():
        done = set() if force else done_keys(run_dir, sources)
        failed = error_keys(run_dir, sources, since=retry_since) - done
        busy = queue.live_keys() - done
        skip = done | failed | busy | attempted
        todo = sorted((t for t in targets if t.key not in skip), key=lambda t: -t.cost)
        log(f"worker {label}: {len(done)} done, {len(failed)} failed, {len(busy)} held by other workers, "
            f"{len(todo)} available")
        return todo

    def claim(target) -> bool:
        if target.key in queue.held():
            return True
        if breaker["tripped"]:
            return False
        if not gate.allows(target.cost) or not queue.try_claim(target.key):
            return False
        out_json, _, err_json = _result_paths(run_dir, target)
        if not force:
            # Another worker may have finished it, or failed on it, since the scan.
            try:
                err_mtime = err_json.stat().st_mtime
            except FileNotFoundError:
                err_mtime = None
            failed_now = err_mtime is not None and (retry_since is None or err_mtime >= retry_since)
            if out_json.is_file() or failed_now:
                queue.release(target.key)
                return False
        attempted.add(target.key)
        claimed_at[target.key] = time.time()
        return True

    def finish(target, outcome):
        """Record an outcome; returns a target to run again (alone) after a GPU out-of-memory."""

        key, status, seconds, error = outcome[:4]
        info = outcome[4] if len(outcome) > 4 else {}
        if status == GPU_RETRY:
            stage = int(target.extra.get("gpu_retry", 0)) + 1
            sizes.out_of_memory()
            log(f"{key}: GPU out of memory ({error}); running it again "
                + ("alone on the GPU" if stage == 1 else "on the CPU")
                + f"; GPU estimates now x{sizes.factor:.1f}")
            from dataclasses import replace

            extra = {**target.extra, "gpu_retry": stage}
            if stage >= 2:
                extra["device"] = "numpy"
            return replace(target, extra=extra)
        outcomes.append(outcome[:4])
        if status == "failed":
            err_json = _result_paths(run_dir, target)[2]
            if not err_json.is_file() or err_json.stat().st_mtime < claimed_at.get(key, started) - 1.0:
                # the process died before it could record the failure: record it here
                atomic_write_json(err_json, {"target": target.to_dict(), "error": error, "traceback": None,
                                             "time": now_utc(), "compute": accel.info()})
        else:
            gate.record(target.cost, seconds)
            sizes.observe(key, info.get("gpu_peak_gb"))
        queue.release(key)
        log(f"[{len(outcomes)}] {key}: {status} in {seconds:.1f}s" + (f" ({error})" if error else ""))
        n_failed = sum(1 for o in outcomes if o[1] == "failed")
        if not breaker["tripped"] and len(outcomes) >= 20 and n_failed > 0.5 * len(outcomes):
            breaker["tripped"] = True
            log(f"worker {label}: {n_failed} of {len(outcomes)} targets failed; starting no more "
                f"(a systematic problem: see {run_dir / 'errors'})")
        return None

    def serial(todo) -> None:
        for target in todo:
            if claim(target):
                while target is not None:
                    target = finish(target, _process_one(run_dir, cfg, target, f"[{len(outcomes) + 1}]"))

    def parallel(todo) -> None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from .pipeline import resolve_stride
        from .scheduler import detect_host_memory_gb, estimate_footprint, run_admitted

        jobs = min(req_jobs, len(todo))
        # Children are spawned (fresh interpreters), so BLAS reads these at import.
        for name in THREAD_VARS:
            os.environ[name] = str(threads)
        ctx = mp.get_context("spawn")
        # Each child selects the same backend; GPU processes share the device.
        child_device = device or _device_of(backend)
        gpu = backend.startswith("torch:cuda")
        host_budget = float(getattr(args, "mem_budget_gb", 0) or 0) or 0.85 * detect_host_memory_gb()
        gpu_budget = float(getattr(args, "gpu_mem_budget_gb", 0) or 0) or (
            0.85 * accel.device_memory_gb() if gpu else float("inf"))
        for t in todo:
            stride = resolve_stride(t.frame_interval_ps, cfg.get("frame_interval_ps"))
            sizes.base[t.key] = estimate_footprint(t, cfg, gpu=gpu, stride=stride)
        biggest = max((sizes[t.key] for t in todo), key=lambda f: f.host_gb)
        log(f"memory budgets: host {host_budget:.0f} GB, gpu {gpu_budget:.0f} GB; "
            f"largest target ~{biggest.host_gb:.1f} GB host / {biggest.gpu_gb:.1f} GB gpu "
            f"(GPU estimates x{sizes.factor:.1f}, recalibrated from measured peaks)")
        # Less fragmentation when many processes share the GPU (PyTorch >= 2.1).
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        def make_executor():
            return ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_child_init,
                                       initargs=(str(run_dir), child_device))

        run_admitted(make_executor, _child_run, todo, sizes, jobs, host_budget, gpu_budget, finish,
                     claim=claim)

    try:
        while True:  # a second pass picks up targets whose workers died meanwhile
            todo = scan()
            if not todo:
                break
            before = len(outcomes)
            (serial if req_jobs == 1 else parallel)(todo)
            if len(outcomes) == before:
                break
    finally:
        queue.close()
    if gate.refused:
        log(f"worker {label}: stopped starting targets {gate.time_left() / 60:.0f} min before the walltime; "
            "the next round continues the split")
    errors = sum(1 for o in outcomes if o[1] == "failed")
    log(f"worker {label} finished: {len(outcomes) - errors} ok, {errors} failed")
    return 1 if (errors and args.strict) else 0


def status(args) -> int:
    """Coverage of a run; with ``--round``, also decide whether to continue."""

    from .workqueue import ClaimQueue, progress

    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.dead_job:
        ClaimQueue(run_dir).mark_dead(str(args.dead_job))  # that array has ended: its claims are void
    prog = progress(run_dir)
    log(f"{run_dir.name}: {prog['done']}/{prog['targets']} targets done, {prog['failed']} failed, "
        f"{prog['running']} running, {prog['remaining']} not started "
        f"(split: {prog['split_domains']} domains, {prog['domains_missing_archive']} without archive, "
        f"{prog['targets_excluded']} (domain, temperature) excluded, {prog['targets_deselected']} deselected)")
    if args.round is None:
        print(json.dumps({k: v for k, v in prog.items() if k != "failed_keys"}))
        return 0
    path = run_dir / "rounds.json"
    history = read_json(path) if path.is_file() else []
    series = str(args.series or os.environ.get("FLEXC_SERIES") or "default")
    settled = prog["done"] + prog["failed"]
    entry = {"series": series, "round": args.round, "time": now_utc(), "done": prog["done"],
             "failed": prog["failed"], "settled": settled, "running": prog["running"],
             "remaining": prog["remaining"]}
    if args.round == 0:  # start of a series (submit.sh / resume.sh): the baseline for round 1
        history.append({**entry, "action": "begin"})
        atomic_write_json(path, history)
        return 0
    mine = [h for h in history if h.get("series", "default") == series]
    prev_settled = mine[-1].get("settled", mine[-1]["done"]) if mine else 0
    new_done = prog["done"] - (mine[-1]["done"] if mine else 0)
    new_failed = prog["failed"] - (mine[-1]["failed"] if mine else 0)
    if prog["complete"]:
        action = "finished"
    elif prog["remaining"] == 0:
        action = "waiting"  # only targets held by another live array are left; its finalize continues
    elif new_failed >= 20 and new_failed > new_done:
        action = "failing"  # mostly failures this round: a systematic problem, do not burn the split
    elif args.round >= args.max_rounds:
        action = "max_rounds"
    elif settled <= prev_settled:
        action = "stalled"  # the round neither finished nor failed anything: something is wrong
    else:
        action = "resubmit"
    history.append({**entry, "action": action})
    atomic_write_json(path, history)
    log(f"round {args.round}/{args.max_rounds}: {action}")
    print(f"action={action}")
    return 0


# --------------------------------------------------------------------------- entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="flexcompress.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p):
        p.add_argument("--config", type=str, default=None)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override, e.g. --set stats.n_lags=20 --set selection.pilot_size=8")
        p.add_argument("--run-dir", required=True)
        p.add_argument("--mode", choices=["pilot", "full"], default=None)
        p.add_argument("--shards", type=int, default=1)

    add_config(sub.add_parser("configure", help="freeze config, targets, provenance and code"))
    p_work = sub.add_parser("worker", help="work through a configured run (shared queue; run any number)")
    p_work.add_argument("--run-dir", required=True)
    p_work.add_argument("--shard", type=int, default=None, help="label for logs (default: SLURM_ARRAY_TASK_ID)")
    p_work.add_argument("--all", action="store_true", help=argparse.SUPPRESS)  # kept for old commands: always
    p_work.add_argument("--force", action="store_true", help="redo targets that already have results (one worker)")
    p_work.add_argument("--retry-failed", action="store_true",
                        help="retry targets that failed before this worker started")
    p_work.add_argument("--retry-since", type=float, default=0,
                        help="retry targets whose failure was recorded before this unix time")
    p_work.add_argument("--deadline", type=float, default=0,
                        help="unix time to finish by (default: the Slurm job's end time; <0: none)")
    p_work.add_argument("--end-margin-s", type=float, default=600.0,
                        help="seconds kept free before the deadline when starting a target")
    p_work.add_argument("--claim-stale-s", type=float, default=900.0,
                        help="a claim without heartbeat for this long is taken over")
    p_work.add_argument("--strict", action="store_true", help="non-zero exit if any target failed")
    p_work.add_argument("--jobs", type=int, default=1, help="targets processed in parallel (spawned processes)")
    p_work.add_argument("--threads", type=int, default=0, help="BLAS threads per job (default: cpus // jobs)")
    p_work.add_argument("--device", default=None,
                        help="compute backend: auto | numpy | cuda | cuda:N | torch-cpu (default: config)")
    p_work.add_argument("--mem-budget-gb", type=float, default=0,
                        help="host memory for admitting targets with --jobs > 1 (default: 85%% of the job's)")
    p_work.add_argument("--gpu-mem-budget-gb", type=float, default=0,
                        help="GPU memory for admitting targets (default: 85%% of the device)")
    p_stat = sub.add_parser("status", help="how much of the run's split is done; round decisions")
    p_stat.add_argument("--run-dir", required=True)
    p_stat.add_argument("--round", type=int, default=None,
                        help="record this round and print action=... (0: record a series baseline)")
    p_stat.add_argument("--series", default=None, help="series of rounds (default: $FLEXC_SERIES)")
    p_stat.add_argument("--max-rounds", type=int, default=4)
    p_stat.add_argument("--dead-job", default=None, help="Slurm array that has ended (its claims are void)")
    p_agg = sub.add_parser("aggregate", help="merge results into tables, summaries and figures")
    p_agg.add_argument("--run-dir", required=True)
    p_agg.add_argument("--no-figures", action="store_true")
    p_local = sub.add_parser("local", help="configure + work + aggregate in this process")
    add_config(p_local)
    p_local.add_argument("--no-figures", action="store_true")
    p_local.add_argument("--jobs", type=int, default=1)
    p_local.add_argument("--threads", type=int, default=0)
    p_local.add_argument("--device", default=None)
    p_bench = sub.add_parser("bench", help="time one synthetic ATLAS-sized target on several backends")
    p_bench.add_argument("--devices", default="numpy,auto", help="comma-separated backends to compare")
    p_bench.add_argument("--n-res", type=int, default=300)
    p_bench.add_argument("--frames", type=int, default=10000)
    p_bench.add_argument("--replicas", type=int, default=3)
    p_bench.add_argument("--max-folds", type=int, default=1)
    p_bench.add_argument("--workdir", default=None)
    p_bench.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p_bench.add_argument("--throughput", action="store_true", help="proteins/hour of the worker vs --jobs-list")
    p_bench.add_argument("--proteins", type=int, default=24)
    p_bench.add_argument("--n-res-range", default="80,400")
    p_bench.add_argument("--jobs-list", default="8,16,32")
    p_bench.add_argument("--threads", type=int, default=0)
    p_bench.add_argument("--mem-budget-gb", type=float, default=0)
    p_bench.add_argument("--gpu-mem-budget-gb", type=float, default=0)

    args = parser.parse_args(argv)
    if args.command == "configure":
        configure(args)
        return 0
    if args.command == "worker":
        return worker(args)
    if args.command == "status":
        return status(args)
    if args.command == "aggregate":
        from .aggregate import aggregate

        aggregate(Path(args.run_dir), figures=not args.no_figures)
        return 0
    if args.command == "bench":
        from .bench import bench

        return bench(args)
    if args.command == "local":
        run_dir = configure(args)
        code = worker(argparse.Namespace(run_dir=str(run_dir), shard=None, all=True, force=False, strict=False,
                                         jobs=args.jobs, threads=args.threads, device=args.device,
                                         mem_budget_gb=0, gpu_mem_budget_gb=0, retry_failed=False))
        from .aggregate import aggregate

        aggregate(run_dir, figures=not args.no_figures)
        return code
    return 2


if __name__ == "__main__":
    sys.exit(main())
