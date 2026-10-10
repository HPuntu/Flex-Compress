"""Archive round trip and the full configure -> worker -> aggregate pipeline."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from flexcompress import run
from flexcompress.data import build_targets, discover, load_replica, load_static, read_split_ids
from flexcompress.synthetic import helix_backbone, ou_mixture, write_archive

BANDS = [0.0, 3.0, 30.0, 300.0]


def make_protein(n_res, n_reps, n_frames, seed, temps=("300",)):
    rng = np.random.default_rng(seed)
    dim = 3 * n_res
    factors = [
        0.15 * np.eye(dim),  # sub-frame local vibration
        0.25 * rng.standard_normal((dim, 2)),  # fast collective
        0.35 * rng.standard_normal((dim, 1)),  # slow collective
        np.zeros((dim, 0)),
    ]
    native = helix_backbone(n_res)
    trajectories = {}
    for t_i, temp in enumerate(temps):
        for r in range(n_reps):
            disp = ou_mixture(n_frames, BANDS, factors, seed * 100 + t_i * 10 + r).reshape(n_frames, n_res, 3)
            coords = native[None] - native[:, 1].mean(0)[None, None] + disp[:, :, None, :]
            trajectories[(temp, str(r))] = coords
    return native, trajectories


@pytest.fixture(scope="module")
def archive_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("archives")
    for domain, n_res, seed in (("1abc_A", 10, 1), ("2xyz_B", 16, 2), ("3pqr_C", 13, 3)):
        native, traj = make_protein(n_res, 3, 900, seed)
        write_archive(root, "atlas", domain, native, traj, frame_interval_ps=10.0, chunk_frames=128)
    native, traj = make_protein(12, 3, 300, 9, temps=("320", "450"))
    write_archive(root, "mdcath", "9mdcA01", native, traj, frame_interval_ps=1000.0, chunk_frames=64)
    return root


def test_archive_roundtrip_and_discovery(archive_root, tmp_path):
    native, traj = make_protein(10, 3, 900, 1)
    rows = discover(archive_root, "atlas")
    assert {r.domain_id for r in rows} == {"1abc_A", "2xyz_B", "3pqr_C"}
    targets = build_targets(rows, archive_root, None, 2, 100)
    target = next(t for t in targets if t.domain_id == "1abc_A")
    assert target.replicas == ["0", "1", "2"] and target.frame_interval_ps == 10.0
    loaded = load_replica(target.archive_path, "300", "1", 900, chunk_frames=200)
    expected = traj[("300", "1")][:, :, 1, :]
    assert np.max(np.abs(loaded["ca"] - expected)) <= 0.0051  # 0.01 A quantisation
    sampled = load_replica(target.archive_path, "300", "1", 900, stride=10)
    np.testing.assert_allclose(sampled["ca"], loaded["ca"][::10], atol=1e-5)
    static = load_static(target.archive_path)
    assert static["residue_names"] == ["ALA"] * 10 and static["dssp"] == ["H"] * 10
    md_rows = discover(archive_root, "mdcath")
    md_targets = build_targets(md_rows, archive_root, ["320", "450"], 2, 100)
    assert [t.key for t in md_targets] == ["9mdcA01__T320", "9mdcA01__T450"]
    split = tmp_path / "ids.csv"
    split.write_text("name,seqres\n1abc_A,AAAA\n2xyz_B,AAAA\n")
    assert read_split_ids(split) == ["1abc_A", "2xyz_B"]


def small_config(tmp_path, archive_root, source, ids):
    split = tmp_path / f"{source}_ids.txt"
    split.write_text("\n".join(ids) + "\n")
    cfg = {
        "source": source,
        "archive_root": str(archive_root),
        "split": {"file": str(split)},
        "min_frames": 100,
        "stats": {"n_lags": 10, "pair_max_residues": 12, "n_pair_lags": 4},
        "spectral": {"pc_max": 40, "ranks": [0, 1, 2, 3, 4, 8], "max_iter": 800, "local_max_iter": 400},
        "baselines": {"pca_exp_ranks": [1, 2, 4], "pca_exp_candidates": 32, "dct_keep": [8, 64]},
        "diagnostics": {"pca_top": 24},
        "aggregate": {"bootstrap": 50},
    }
    path = tmp_path / f"{source}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_local_pipeline_atlas(archive_root, tmp_path):
    cfg_path = small_config(tmp_path, archive_root, "atlas", ["1abc_A", "2xyz_B", "3pqr_C", "missing_Z"])
    run_dir = tmp_path / "run_atlas"
    assert run.main(["local", "--config", str(cfg_path), "--run-dir", str(run_dir), "--shards", "2"]) == 0

    listing = json.loads((run_dir / "targets.json").read_text())
    assert listing["n_targets"] == 3 and listing["missing_domains"] == ["missing_Z"]
    assert (run_dir / "code" / "src" / "flexcompress" / "pipeline.py").is_file()
    assert (run_dir / "code" / "src" / "dynamode" / "dataloader" / "compressed_backbone.py").is_file()

    result = json.loads((run_dir / "results" / "atlas" / "2xyz_B__T300.json").read_text())
    assert result["status"] == "ok" and len(result["replicas_used"]) == 3
    s0 = result["stage0"]
    for key in ("ca_pca_pooled", "bb_pca_pooled", "cosine_content_mean", "rmsip", "floor_pairwise",
                "acf_corr_by_lag", "asym_replica_cos_by_lag", "native_contacts_mean"):
        assert key in s0, key
    assert len(result["band_reliability"]["log_energy_corr_by_band"]) == len(result["band_reliability"]["bands_ps"])
    assert min(s0["native_contacts_mean"]) > 0.9  # small fluctuations keep native contacts
    rows = result["stage1"]["rows"]
    methods = {r["method"] for r in rows}
    assert methods == {"floor", "spectral_subspace_local", "spectral_subspace", "spectral_band_local", "pca_exp", "dct"}
    assert all(f["fit_converged"] for f in result["stage1"]["fits"])

    with np.load(run_dir / "descriptors" / "atlas" / "2xyz_B__T300.npz") as data:
        n_bands = len(data["bands_ps"])
        assert data["descriptors"].shape == (3, 16, 6 * n_bands)
        assert np.all(np.isfinite(data["descriptors"]))

    summary = json.loads((run_dir / "report" / "summary.json").read_text())
    assert summary["n_ok"] == 3 and summary["n_errors"] == 0
    group = summary["stage1"]["atlas/T300"]
    full = group["full_model_adequacy"]
    # The full statistical model is no further from a new replica than the raw statistics.
    for metric in ("msf_rel", "acf_rmse", "dccm_rmse", "frob_rel"):
        assert full[f"rel_{metric}"]["median"] < 1.25, (metric, full[f"rel_{metric}"])
    # True structure: 3 collective dims + isotropic local noise.  Rank 3 + local reaches the floor.
    curve = {c["param"]: c for c in group["curves"] if c["method"] == "spectral_subspace_local"}
    assert curve[3]["rel_dccm_rmse"]["median"] < 1.3
    assert curve[0]["rel_dccm_rmse"]["median"] > 2.0
    uni = summary["universality"]["atlas/T300"]
    assert uni["n_targets"] == 3 and 0 < uni["reliable_fraction"] <= 1.0
    figures = {p.name for p in (run_dir / "report" / "figures").glob("*.png")}
    assert {"rate_distortion_atlas_T300.png", "reliability_vs_lag_atlas.png",
            "universality_atlas_T300.png", "pca90_vs_N_atlas_T300.png"} <= figures

    # Re-running a shard skips finished targets.
    before = (run_dir / "results" / "atlas" / "1abc_A__T300.json").stat().st_mtime_ns
    assert run.main(["worker", "--run-dir", str(run_dir), "--all"]) == 0
    assert (run_dir / "results" / "atlas" / "1abc_A__T300.json").stat().st_mtime_ns == before


def test_local_pipeline_mdcath_temperatures(archive_root, tmp_path):
    cfg_path = small_config(tmp_path, archive_root, "mdcath", ["9mdcA01"])
    run_dir = tmp_path / "run_md"
    assert run.main(["local", "--config", str(cfg_path), "--run-dir", str(run_dir), "--no-figures",
                     "--set", "temperatures=['320','450']", "--set", "baselines.dct_keep=[8]"]) == 0
    summary = json.loads((run_dir / "report" / "summary.json").read_text())
    assert set(summary["stage1"]) == {"mdcath/T320", "mdcath/T450"}
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["descriptors"]["bands_ps"][0] == 0.0 and cfg["descriptors"]["bands_ps"][1] == 1000.0


def test_unequal_replica_lengths_share_one_window(tmp_path):
    native, traj = make_protein(9, 4, 600, 21)
    traj[("300", "3")] = traj[("300", "3")][:250]  # one short replica
    root = tmp_path / "arch"
    write_archive(root, "atlas", "4len_A", native, traj, frame_interval_ps=10.0, chunk_frames=128)
    cfg_path = small_config(tmp_path, root, "atlas", ["4len_A"])
    run_dir = tmp_path / "run_len"
    assert run.main(["local", "--config", str(cfg_path), "--run-dir", str(run_dir), "--no-figures"]) == 0
    result = json.loads((run_dir / "results" / "atlas" / "4len_A__T300.json").read_text())
    assert result["n_frames_used"] == [600, 600, 600, 250]
    assert result["segment_frames"] == 250 and result["n_windows"] == [3, 3, 3, 1]
    full = [r for r in result["stage1"]["rows"] if r["method"] == "spectral_subspace_local" and r["full_rank"]]
    floors = {r["fold"]: r for r in result["stage1"]["rows"] if r["method"] == "floor"}
    ratios = [r["msf_rel"] / floors[r["fold"]]["msf_rel"] for r in full]
    assert len(ratios) == 4 and 0.4 < float(np.median(ratios)) < 1.6


def test_parallel_worker_matches_sequential(archive_root, tmp_path):
    """`worker --jobs` (spawned processes, as on an Isambard GH200) gives the sequential results."""

    cfg_path = small_config(tmp_path, archive_root, "atlas", ["1abc_A", "2xyz_B", "3pqr_C"])
    seq, par = tmp_path / "seq", tmp_path / "par"
    for run_dir in (seq, par):
        assert run.main(["configure", "--config", str(cfg_path), "--run-dir", str(run_dir)]) == 0
    assert run.main(["worker", "--run-dir", str(seq), "--all", "--threads", "1"]) == 0
    assert run.main(["worker", "--run-dir", str(par), "--all", "--jobs", "3", "--threads", "1"]) == 0
    for key in ("1abc_A__T300", "2xyz_B__T300", "3pqr_C__T300"):
        a = json.loads((seq / "results" / "atlas" / f"{key}.json").read_text())
        b = json.loads((par / "results" / "atlas" / f"{key}.json").read_text())
        assert a["status"] == b["status"] == "ok"
        rows_a = {(r["fold"], r["method"], r["param"]): r for r in a["stage1"]["rows"]}
        rows_b = {(r["fold"], r["method"], r["param"]): r for r in b["stage1"]["rows"]}
        assert rows_a.keys() == rows_b.keys()
        for k, r in rows_a.items():
            np.testing.assert_allclose(r["frob_rel"], rows_b[k]["frob_rel"], rtol=1e-6, atol=1e-9)


def test_torch_backend_worker_matches_numpy(archive_root, tmp_path):
    """Whole pipeline on the torch backend (as on the GPU), incl. forced Newton-Schulz projections."""

    pytest.importorskip("torch")
    cfg_path = small_config(tmp_path, archive_root, "atlas", ["1abc_A", "2xyz_B"])
    runs = {}
    for name, extra in (("numpy", []), ("torch", []), ("ns", ["--set", "compute.psd_method=newton_schulz"])):
        run_dir = tmp_path / f"run_{name}"
        assert run.main(["configure", "--config", str(cfg_path), "--run-dir", str(run_dir), *extra]) == 0
        device = "numpy" if name == "numpy" else "torch-cpu"
        jobs = "2" if name == "torch" else "1"
        assert run.main(["worker", "--run-dir", str(run_dir), "--all", "--device", device, "--jobs", jobs,
                         "--threads", "1"]) == 0
        runs[name] = run_dir
    for key in ("1abc_A__T300", "2xyz_B__T300"):
        docs = {n: json.loads((d / "results" / "atlas" / f"{key}.json").read_text()) for n, d in runs.items()}
        assert docs["torch"]["compute"]["backend"] == "torch:cpu"
        assert docs["numpy"]["config_hash"] == docs["torch"]["config_hash"] == docs["ns"]["config_hash"]
        ref = {(r["fold"], r["method"], r["param"]): r for r in docs["numpy"]["stage1"]["rows"]}
        for name, rtol in (("torch", 1e-6), ("ns", 2e-2)):
            for r in docs[name]["stage1"]["rows"]:
                a = ref[(r["fold"], r["method"], r["param"])]
                np.testing.assert_allclose(r["frob_rel"], a["frob_rel"], rtol=rtol, atol=1e-9, err_msg=name)
                np.testing.assert_allclose(r["msf_rel"], a["msf_rel"], rtol=rtol, atol=1e-6, err_msg=name)
        s0n, s0t = docs["numpy"]["stage0"], docs["torch"]["stage0"]
        assert s0n["ca_pca_pooled"] == s0t["ca_pca_pooled"]
        np.testing.assert_allclose(s0t["floor_pairwise"]["frob_rel"], s0n["floor_pairwise"]["frob_rel"], rtol=1e-8)


def _status(run_dir, *extra):
    """Run `status` and return its last stdout line."""

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert run.main(["status", "--run-dir", str(run_dir), *extra]) == 0
    return buf.getvalue().strip().splitlines()[-1]


def test_full_split_shared_by_concurrent_workers(tmp_path):
    """Whole split: coverage record, two workers at once never duplicate, rounds continue to the end."""

    root = tmp_path / "arch"
    ids = ["5aaa_A", "5bbb_A", "5ccc_A", "5ddd_A"]
    for i, domain in enumerate(ids):
        native, traj = make_protein(8 + i, 3, 300, 40 + i)
        write_archive(root, "atlas", domain, native, traj, frame_interval_ps=10.0, chunk_frames=128)
    native, traj = make_protein(8, 1, 300, 50)  # a single replica: excluded, with the reason recorded
    write_archive(root, "atlas", "6one_A", native, traj, frame_interval_ps=10.0, chunk_frames=128)
    cfg_path = small_config(tmp_path, root, "atlas", [*ids, "6one_A", "missing_Z"])
    run_dir = tmp_path / "run_full"
    assert run.main(["configure", "--config", str(cfg_path), "--run-dir", str(run_dir), "--shards", "2",
                     "--set", "folds.max_folds=1", "--set", "baselines.dct_keep=[8]"]) == 0
    listing = json.loads((run_dir / "targets.json").read_text())
    cov = listing["coverage"]
    assert listing["n_targets"] == 4 and listing["missing_domains"] == ["missing_Z"]
    assert (run_dir / "launch.sh").is_file() and (run_dir / "aggregate.sbatch").is_file()
    assert cov["mode"] == "full" and cov["n_split_domains"] == 6 and cov["n_domains_with_archive"] == 5
    assert [(e["domain"], e["reason"]) for e in cov["excluded_targets"]] == [("6one_A", "too_few_replicas")]

    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{repo / 'src'}{os.pathsep}{repo / 'DynaMode-dev' / 'src'}"}
    env.pop("SLURM_JOB_ID", None)
    procs = [subprocess.Popen([sys.executable, "-m", "flexcompress.run", "worker", "--run-dir", str(run_dir),
                               "--shard", str(i), "--device", "numpy", "--threads", "1"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for i in range(2)]
    logs = [p.communicate(timeout=600)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), logs
    starts = [line.split()[3].rstrip(":") for log in logs for line in log.splitlines() if ": start N=" in line]
    assert sorted(starts) == sorted(f"{d}__T300" for d in ids), logs  # each target exactly once
    assert not list((run_dir / "claims").glob("*.claim"))
    assert json.loads(_status(run_dir))["done"] == 4
    assert _status(run_dir, "--round", "1") == "action=finished"

    # Rounds: lose two results (as if a round hit the walltime), then continue until finished.
    for d in ids[:2]:
        (run_dir / "results" / "atlas" / f"{d}__T300.json").unlink()
    (run_dir / "rounds.json").unlink()
    assert _status(run_dir, "--round", "1", "--max-rounds", "3") == "action=resubmit"  # 2 done > 0
    assert _status(run_dir, "--round", "2", "--max-rounds", "3") == "action=stalled"  # no progress
    assert _status(run_dir, "--round", "3", "--max-rounds", "3") == "action=max_rounds"
    # Progress is done + failed per series: a round that only records failures is not stalled.
    _status(run_dir, "--round", "0", "--series", "f")
    err_fake = run_dir / "errors" / "atlas" / f"{ids[1]}__T300.json"
    err_fake.parent.mkdir(parents=True, exist_ok=True)
    err_fake.write_text(json.dumps({"target": {"domain_id": ids[1], "temperature": "300"}, "error": "x"}))
    assert _status(run_dir, "--round", "1", "--series", "f") == "action=resubmit"
    assert _status(run_dir, "--round", "1", "--series", "g") == "action=resubmit"  # no baseline: vs 0
    assert _status(run_dir, "--round", "2", "--series", "f") == "action=stalled"
    history = json.loads((run_dir / "rounds.json").read_text())
    assert [h["series"] for h in history][-4:] == ["f", "f", "g", "f"]
    err_fake.unlink()

    # A claim left by an array that has ended blocks nothing once the finalize job records it.
    claim = run_dir / "claims" / f"{ids[0]}__T300.claim"
    claim.write_text(json.dumps({"job": "777", "token": "gone"}))
    assert json.loads(_status(run_dir))["running"] == 1
    assert json.loads(_status(run_dir, "--dead-job", "777"))["running"] == 0
    assert run.main(["worker", "--run-dir", str(run_dir), "--device", "numpy"]) == 0
    assert _status(run_dir, "--round", "4", "--max-rounds", "9") == "action=finished"

    # Failures are recorded, not retried by later rounds, and retried on request (resume).
    broken = root / "domains" / f"atlas_backbone_{ids[3]}.zarr.zip"
    broken.write_bytes(b"not a zip")
    (run_dir / "results" / "atlas" / f"{ids[3]}__T300.json").unlink()
    assert run.main(["worker", "--run-dir", str(run_dir), "--device", "numpy"]) == 0
    err = run_dir / "errors" / "atlas" / f"{ids[3]}__T300.json"
    first = err.stat().st_mtime_ns
    prog = json.loads(_status(run_dir))
    assert prog["failed"] == 1 and prog["remaining"] == 0 and prog["complete"]
    assert run.main(["worker", "--run-dir", str(run_dir), "--device", "numpy"]) == 0
    assert err.stat().st_mtime_ns == first
    time.sleep(0.01)
    assert run.main(["worker", "--run-dir", str(run_dir), "--device", "numpy", "--retry-failed"]) == 0
    second = err.stat().st_mtime_ns
    assert second > first
    # resume.sh gives every task one cutoff: failures recorded after it are not retried again.
    assert run.main(["worker", "--run-dir", str(run_dir), "--device", "numpy",
                     "--retry-since", str(time.time() - 3600)]) == 0
    assert err.stat().st_mtime_ns == second


def test_target_failed_elsewhere_is_not_rerun(archive_root, tmp_path, monkeypatch):
    """A target another worker failed on after this worker's scan is skipped, not run again."""

    cfg_path = small_config(tmp_path, archive_root, "atlas", ["1abc_A", "2xyz_B"])
    run_dir = tmp_path / "run_elsewhere"
    assert run.main(["configure", "--config", str(cfg_path), "--run-dir", str(run_dir)]) == 0
    calls = []

    def fake(run_dir_, cfg, target, tag):
        calls.append(target.key)
        other = "1abc_A__T300" if target.key == "2xyz_B__T300" else "2xyz_B__T300"
        err = run_dir_ / "errors" / "atlas" / f"{other}.json"  # "another worker" fails on the other one
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(json.dumps({"target": {"domain_id": other.split("__")[0], "temperature": "300"},
                                   "error": "elsewhere"}))
        out = run_dir_ / "results" / "atlas" / f"{target.key}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"status": "ok"}))
        return target.key, "ok", 0.1, None

    monkeypatch.setattr(run, "_process_one", fake)
    assert run.main(["worker", "--run-dir", str(run_dir), "--device", "numpy"]) == 0
    assert len(calls) == 1


def test_gpu_out_of_memory_is_retried_alone_then_on_cpu(archive_root, tmp_path, monkeypatch):
    """A GPU out-of-memory is not a failure: the target runs again (alone, then on the CPU)."""

    from flexcompress import accel

    cfg_path = small_config(tmp_path, archive_root, "atlas", ["1abc_A"])
    run_dir = tmp_path / "run_oom"
    assert run.main(["configure", "--config", str(cfg_path), "--run-dir", str(run_dir),
                     "--set", "folds.max_folds=1"]) == 0
    real, calls = run.process_target, []

    def flaky(target, cfg, log=print):
        calls.append(dict(target.extra))
        if len(calls) < 3:
            raise RuntimeError("CUDA error: out of memory")
        return real(target, cfg, log=log)

    monkeypatch.setattr(run, "process_target", flaky)
    monkeypatch.setattr(accel, "is_gpu_memory_error", lambda exc: "out of memory" in str(exc))
    assert run.main(["worker", "--run-dir", str(run_dir), "--device", "numpy"]) == 0
    assert calls == [{}, {"gpu_retry": 1}, {"gpu_retry": 2, "device": "numpy"}]
    assert (run_dir / "results" / "atlas" / "1abc_A__T300.json").is_file()
    assert not (run_dir / "errors").exists() or not list((run_dir / "errors").rglob("*.json"))


def test_systematic_failures_stop_the_worker_and_the_rounds(archive_root, tmp_path, monkeypatch):
    cfg_path = small_config(tmp_path, archive_root, "atlas", ["1abc_A"])
    run_dir = tmp_path / "run_breaker"
    assert run.main(["configure", "--config", str(cfg_path), "--run-dir", str(run_dir)]) == 0
    listing = json.loads((run_dir / "targets.json").read_text())
    template = listing["targets"][0]
    listing["targets"] = [{**template, "domain_id": f"x{i:03d}_A"} for i in range(40)]
    listing["n_targets"] = 40
    (run_dir / "targets.json").write_text(json.dumps(listing))
    _status(run_dir, "--round", "0")

    def broken(target, cfg, log=print):
        raise ValueError("bad archive layout")

    monkeypatch.setattr(run, "process_target", broken)
    assert run.main(["worker", "--run-dir", str(run_dir), "--device", "numpy"]) == 0
    prog = json.loads(_status(run_dir))
    assert prog["failed"] == 20 and prog["remaining"] == 20  # stopped after 20, half the split untouched
    assert _status(run_dir, "--round", "1") == "action=failing"


def test_worker_stops_before_the_walltime(archive_root, tmp_path):
    cfg_path = small_config(tmp_path, archive_root, "atlas", ["1abc_A", "2xyz_B"])
    run_dir = tmp_path / "run_deadline"
    assert run.main(["configure", "--config", str(cfg_path), "--run-dir", str(run_dir)]) == 0
    # 5 minutes left: too little to start a target before any runtime is known.
    assert run.main(["worker", "--run-dir", str(run_dir), "--deadline", str(time.time() + 300)]) == 0
    prog = json.loads(_status(run_dir))
    assert prog["done"] == 0 and prog["remaining"] == 2 and not list((run_dir / "claims").glob("*.claim"))


def test_bench_command(tmp_path):
    pytest.importorskip("torch")
    code = run.main(["bench", "--devices", "numpy,torch-cpu", "--n-res", "12", "--frames", "600",
                     "--workdir", str(tmp_path), "--set", "spectral.pc_max=24",
                     "--set", "spectral.ranks=[0,2,8]", "--set", "baselines.dct_keep=[8]"])
    assert code == 0
    summary = json.loads((tmp_path / "bench.json").read_text())
    assert [r["backend"] for r in summary["runs"]] == ["numpy", "torch:cpu"]
    assert summary["max_relative_difference"] < 1e-6


def test_bench_throughput_mode(tmp_path):
    code = run.main(["bench", "--throughput", "--devices", "numpy", "--jobs-list", "1,3", "--proteins", "4",
                     "--n-res-range", "8,14", "--frames", "400", "--threads", "1", "--workdir", str(tmp_path),
                     "--set", "spectral.pc_max=24", "--set", "spectral.ranks=[0,2]", "--set", "baselines.dct_keep=[8]",
                     "--set", "diagnostics.bb_pca=false", "--set", "min_frames=50"])
    assert code == 0
    rows = json.loads((tmp_path / "throughput.json").read_text())["rows"]
    assert [(r["jobs"], r["done"], r["failed"]) for r in rows] == [(1, 4, 0), (3, 4, 0)]
    assert all(r["proteins_per_hour"] > 0 for r in rows)


def test_isambard_archive_defaults(monkeypatch, tmp_path):
    from flexcompress.config import load_config, resolve_config

    monkeypatch.delenv("DYNAMODE_ATLAS_ARCHIVE", raising=False)
    monkeypatch.delenv("DYNAMODE_MDCATH_ARCHIVE", raising=False)
    root = run.REPO_ROOT
    atlas = resolve_config(load_config(root / "configs" / "stage01_atlas.yaml"), root)
    mdcath = resolve_config(load_config(root / "configs" / "stage01_mdcath.yaml"), root)
    assert atlas["archive_root"] == "/projects/u6xq/hew/DATA/ATLAS/compressed_bb_zarr"
    assert mdcath["archive_root"] == "/projects/u6xq/hew/DATA/mdCATH/compressed_bb_zarr"
    monkeypatch.setenv("DYNAMODE_ATLAS_ARCHIVE", str(tmp_path))
    assert resolve_config(load_config(root / "configs" / "stage01_atlas.yaml"), root)["archive_root"] == str(tmp_path)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("stats:\n  n_lagz: 3\n")
    with pytest.raises(KeyError):
        run.main(["configure", "--config", str(path), "--run-dir", str(tmp_path / "r")])
