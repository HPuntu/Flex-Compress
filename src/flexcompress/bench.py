"""Benchmark compute backends on synthetic ATLAS-sized targets.

    python -m flexcompress.run bench --devices numpy,cuda            # one target, per-stage timing
    python -m flexcompress.run bench --throughput --devices cuda --jobs-list 8,16,32
                                                                     # proteins/hour vs concurrency

Single mode:
Writes a synthetic archive (OU-mixture dynamics around an ideal helix), runs
``process_target`` with the default configuration on each backend, prints
per-stage timings and the largest relative disagreement of every Stage-1
distortion against the first backend.  Exit status 1 if backends disagree by
more than 1 % (the GPU Newton-Schulz projection typically agrees to ~1e-4).

Throughput mode writes ``--proteins`` synthetic targets with sizes spread over
``--n-res-range`` and runs the real worker (memory-aware scheduler, spawned
processes) for every device x jobs combination, reporting proteins per hour:
the number that decides how many proteins one GH200 should run at once.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from . import accel
from .config import load_config, resolve_config
from .data import build_targets, discover
from .metrics import METRICS
from .pipeline import process_target
from .synthetic import helix_backbone, ou_mixture, write_archive

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_archive(root: Path, n_res: int, n_frames: int, n_rep: int, seed: int = 0, domain: str = "bench_A") -> None:
    rng = np.random.default_rng(seed)
    dim = 3 * n_res
    bands = [0.0, 10.0, 100.0, 1000.0]
    factors = [0.15 * np.eye(dim), 0.05 * rng.standard_normal((dim, 20)),
               0.08 * rng.standard_normal((dim, 5)), 0.1 * rng.standard_normal((dim, 2))]
    native = helix_backbone(n_res)
    centre = native[:, 1].mean(0)[None, None]
    traj = {}
    for r in range(n_rep):
        disp = ou_mixture(n_frames, bands, factors, seed + r).reshape(n_frames, n_res, 3)
        traj[("300", str(r))] = native[None] - centre + disp[:, :, None, :]
    write_archive(root, "atlas", domain, native, traj, frame_interval_ps=10.0, chunk_frames=256)


def bench(args) -> int:
    if getattr(args, "throughput", False):
        return throughput(args)
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="flexc-bench-"))
    archive = work / "archive"
    if not (archive / "domains" / "atlas_backbone_bench_A.zarr.zip").is_file():
        print(f"writing synthetic archive: N={args.n_res}, {args.replicas} x {args.frames} frames -> {archive}",
              flush=True)
        _make_archive(archive, args.n_res, args.frames, args.replicas)
    overrides = [f"archive_root={archive}", f"folds.max_folds={int(args.max_folds)}", *args.set]
    cfg = resolve_config(load_config(None, overrides), REPO_ROOT)
    target = build_targets(discover(archive, "atlas"), archive, None, 2, 1)[0]

    runs = []
    for device in [d.strip() for d in args.devices.split(",") if d.strip()]:
        comp = cfg["compute"]
        backend = accel.configure(device, comp["dtype"], comp["psd_method"], comp["ns_iters"])
        started = time.monotonic()
        result, _ = process_target(target, cfg, log=lambda message: None)
        elapsed = time.monotonic() - started
        accel.release()
        runs.append((device, backend, elapsed, result))
        stage = {k: round(v, 1) for k, v in result["timing"].items()}
        print(f"{device:>10} -> {backend:<14} total {elapsed:7.1f} s   {stage}", flush=True)
    accel.configure("numpy")

    ref = runs[0][3]
    ref_rows = {(r["fold"], r["method"], r["param"]): r for r in ref["stage1"]["rows"]}
    worst_all = 0.0
    for device, _backend, elapsed, result in runs[1:]:
        worst = 0.0
        for r in result["stage1"]["rows"]:
            a = ref_rows[(r["fold"], r["method"], r["param"])]
            for m in METRICS:
                if a.get(m) is None or not np.isfinite(a[m]):
                    continue
                worst = max(worst, abs(r[m] - a[m]) / max(abs(a[m]), 1e-12))
        worst_all = max(worst_all, worst)
        speed = runs[0][2] / elapsed if elapsed > 0 else float("nan")
        print(f"{device}: {speed:.1f}x vs {runs[0][0]}, max relative Stage-1 difference {worst:.2e}", flush=True)
    summary = {"target": target.key, "runs": [
        {"device": d, "backend": b, "seconds": t, "timing": res["timing"], "compute": res.get("compute")}
        for d, b, t, res in runs], "max_relative_difference": worst_all}
    (work / "bench.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {work / 'bench.json'}")
    return 1 if worst_all > 1e-2 else 0


def throughput(args) -> int:
    """Proteins per hour of the real worker for each device x concurrency."""

    import argparse

    from .run import available_cpus, configure, worker

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="flexc-throughput-"))
    archive = work / "archive"
    lo, hi = (int(x) for x in args.n_res_range.split(","))
    sizes = np.linspace(lo, hi, int(args.proteins)).round().astype(int)
    ids = [f"tp{i:03d}_A" for i in range(len(sizes))]
    for i, (domain, n_res) in enumerate(zip(ids, sizes)):
        if not (archive / "domains" / f"atlas_backbone_{domain}.zarr.zip").is_file():
            _make_archive(archive, int(n_res), args.frames, args.replicas, seed=100 + i, domain=domain)
    (work / "ids.txt").write_text("\n".join(ids) + "\n")
    print(f"{len(ids)} synthetic proteins, N {lo}-{hi}, {args.replicas} x {args.frames} frames -> {archive}",
          flush=True)
    rows = []
    for device in [d.strip() for d in args.devices.split(",") if d.strip()]:
        for jobs in [int(j) for j in args.jobs_list.split(",") if j.strip()]:
            run_dir = work / f"run-{device.replace(':', '')}-j{jobs}"
            if run_dir.exists():
                import shutil

                shutil.rmtree(run_dir)
            configure(argparse.Namespace(
                config=None, run_dir=str(run_dir), mode=None, shards=1,
                set=[f"archive_root={archive}", f"split.file={work / 'ids.txt'}",
                     f"folds.max_folds={int(args.max_folds)}", *args.set]))
            threads = int(args.threads or 0) or max(1, available_cpus() // jobs)
            started = time.monotonic()
            code = worker(argparse.Namespace(
                run_dir=str(run_dir), shard=None, all=True, force=True, strict=False, jobs=jobs, threads=threads,
                device=device, mem_budget_gb=args.mem_budget_gb, gpu_mem_budget_gb=args.gpu_mem_budget_gb,
                deadline=-1))
            elapsed = time.monotonic() - started
            done = len(list((run_dir / "results" / "atlas").glob("*.json")))
            failed = len(list((run_dir / "errors").rglob("*.json"))) if (run_dir / "errors").is_dir() else 0
            rate = done / elapsed * 3600 if elapsed > 0 else float("nan")
            rows.append({"device": device, "jobs": jobs, "threads": threads, "seconds": elapsed, "done": done,
                         "failed": failed, "proteins_per_hour": rate, "exit": code})
            print(f"THROUGHPUT device={device:<9} jobs={jobs:<3} threads={threads:<2} "
                  f"{done}/{len(ids)} ok, {failed} failed in {elapsed:7.1f} s  ->  {rate:8.1f} proteins/hour",
                  flush=True)
    (work / "throughput.json").write_text(json.dumps({"proteins": len(ids), "n_res": sizes.tolist(),
                                                      "frames": args.frames, "rows": rows}, indent=1))
    print(f"wrote {work / 'throughput.json'}")
    return 1 if any(r["failed"] for r in rows) else 0
