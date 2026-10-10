"""Archive round trip and the full configure -> worker -> aggregate pipeline."""

import json

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


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("stats:\n  n_lagz: 3\n")
    with pytest.raises(KeyError):
        run.main(["configure", "--config", str(path), "--run-dir", str(tmp_path / "r")])
