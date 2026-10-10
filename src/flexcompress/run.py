"""Command line: configure a frozen run, process shards, aggregate.

    python -m flexcompress.run configure --config configs/stage01_atlas.yaml \
        --run-dir /path/to/runs/atlas-pilot --mode pilot --shards 16
    python -m flexcompress.run worker --run-dir RUN --shard $SLURM_ARRAY_TASK_ID
    python -m flexcompress.run aggregate --run-dir RUN
    python -m flexcompress.run local --config ... --run-dir RUN --set selection.domains=[1abcA00]

``configure`` freezes the resolved config, the target list with shard
assignments, provenance, and a copy of the code (Flex-Compress and the
DynaMode-dev data layer) under ``RUN/code/src``; the Slurm worker runs that
snapshot, so later edits never change a running study.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

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
    for name in ("worker.sbatch", "aggregate.sbatch", "flexc_env.sh"):
        source = REPO_ROOT / "sbatch" / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)


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
    targets = build_targets(rows, cfg["archive_root"], cfg["temperatures"], int(cfg["min_replicas"]),
                            int(cfg["min_frames"]))
    found = {t.domain_id for t in targets}
    missing = sorted(set(domains) - found)
    targets = _select(targets, cfg)
    if not targets:
        raise SystemExit("no targets selected; check archive_root, split and temperatures")
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
    log(f"run {run_dir}: {len(targets)} targets ({len(found)} domains found, {len(missing)} missing) "
        f"in {n_shards} shards")
    print(run_dir)
    return run_dir


# --------------------------------------------------------------------------- worker


def _result_paths(run_dir: Path, target) -> tuple[Path, Path, Path]:
    base = run_dir / "results" / target.source
    return (base / f"{target.key}.json",
            run_dir / "descriptors" / target.source / f"{target.key}.npz",
            run_dir / "errors" / target.source / f"{target.key}.json")


def worker(args) -> int:
    from .data import Target

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = read_json(run_dir / "config.json")
    listing = read_json(run_dir / "targets.json")
    shard = args.shard if args.shard is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    targets = [Target.from_dict(t) for t in listing["targets"]]
    mine = [t for t in targets if args.all or t.shard == shard]
    want_hash = config_hash(cfg)
    import dynamode.dataloader.compressed_backbone as cb

    log(f"shard {shard}: {len(mine)} targets; flexcompress={Path(__file__).parent} dynamode={Path(cb.__file__).parent}")
    errors = 0
    for i, target in enumerate(mine, 1):
        out_json, out_npz, err_json = _result_paths(run_dir, target)
        if out_json.is_file() and not args.force:
            try:
                previous = read_json(out_json)
                if previous.get("config_hash") == want_hash:
                    log(f"[{i}/{len(mine)}] {target.key}: done, skipping")
                    continue
            except Exception:
                pass
        started = time.monotonic()
        log(f"[{i}/{len(mine)}] {target.key}: N={target.n_res} reps={len(target.replicas)} frames={target.n_frames}")
        try:
            result, descriptors = process_target(target, cfg, log=log)
            result["finished"] = now_utc()
            if descriptors is not None:
                atomic_write_npz(out_npz, **descriptors)
            atomic_write_json(out_json, result)
            if err_json.is_file():
                err_json.unlink()
            log(f"[{i}/{len(mine)}] {target.key}: {result['status']} in {time.monotonic() - started:.1f}s")
        except Exception as exc:
            errors += 1
            atomic_write_json(err_json, {
                "target": target.to_dict(), "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "time": now_utc(),
            })
            log(f"[{i}/{len(mine)}] {target.key}: FAILED {type(exc).__name__}: {exc}")
    log(f"shard {shard} finished: {len(mine) - errors} ok, {errors} failed")
    return 1 if (errors and args.strict) else 0


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
    p_work = sub.add_parser("worker", help="process one shard of a configured run")
    p_work.add_argument("--run-dir", required=True)
    p_work.add_argument("--shard", type=int, default=None)
    p_work.add_argument("--all", action="store_true", help="process every shard serially")
    p_work.add_argument("--force", action="store_true")
    p_work.add_argument("--strict", action="store_true", help="non-zero exit if any target failed")
    p_agg = sub.add_parser("aggregate", help="merge results into tables, summaries and figures")
    p_agg.add_argument("--run-dir", required=True)
    p_agg.add_argument("--no-figures", action="store_true")
    p_local = sub.add_parser("local", help="configure + all shards + aggregate in this process")
    add_config(p_local)
    p_local.add_argument("--no-figures", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "configure":
        configure(args)
        return 0
    if args.command == "worker":
        return worker(args)
    if args.command == "aggregate":
        from .aggregate import aggregate

        aggregate(Path(args.run_dir), figures=not args.no_figures)
        return 0
    if args.command == "local":
        run_dir = configure(args)
        code = worker(argparse.Namespace(run_dir=str(run_dir), shard=None, all=True, force=False, strict=False))
        from .aggregate import aggregate

        aggregate(run_dir, figures=not args.no_figures)
        return code
    return 2


if __name__ == "__main__":
    sys.exit(main())
