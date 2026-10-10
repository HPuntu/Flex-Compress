"""Relaxation-spectrum fitting against synthetic ground truth."""

import numpy as np
import pytest

from flexcompress.baselines import dct_truncate, fit_pca_exp, pca_exp_candidates, pca_exp_model
from flexcompress.kernels import kernel_matrix, pooled_kernel_matrix
from flexcompress.metrics import distortion, summarize
from flexcompress.spectral import (
    band_timescales,
    fit_local_residual,
    fit_psd_bands,
    model_from_components,
    proj_psd,
    spectral_components,
    subspace_model,
)
from flexcompress.stats import Lagged, center, cross_sums, diag_block_sums, lag_grid
from flexcompress.synthetic import ou_mixture

BANDS = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0])


def truth(n_res=4, seed=0):
    rng = np.random.default_rng(seed)
    dim = 3 * n_res
    factors = [np.zeros((dim, 0)) for _ in BANDS]
    factors[0] = 0.3 * np.eye(dim)  # white, full rank (local vibration)
    factors[2] = rng.standard_normal((dim, 2)) * 0.6  # 3-frame collective, rank 2
    factors[4] = rng.standard_normal((dim, 1)) * 1.0  # 30-frame collective, rank 1
    return factors


def simulate(factors, n_rep=8, n_frames=4000, seed=10):
    return [center(ou_mixture(n_frames, BANDS, factors, seed + r)) for r in range(n_rep)]


def test_proj_psd_properties():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((3, 5, 5))
    p = proj_psd(m)
    assert np.all(np.linalg.eigvalsh(p) > -1e-10)
    np.testing.assert_allclose(proj_psd(p), p, atol=1e-10)


def test_band_timescales_grid():
    t = band_timescales(10000, 2.0, 0.5)
    assert t[0] == 0 and t[1] == 1.0 and t[-1] == pytest.approx(5000.0)
    assert np.allclose(np.diff(np.log10(t[1:])), np.log10(5000) / (len(t) - 2))


def test_fit_recovers_band_amplitudes_and_lagged_covariance():
    factors = truth()
    reps = simulate(factors)
    lags = lag_grid(4000, 18, 0.1)
    stats = Lagged.pool([cross_sums(x, lags) for x in reps]).mean()
    kern = pooled_kernel_matrix(BANDS, [len(x) for x in reps], lags)
    amp, info = fit_psd_bands(stats, kern, ridge=1e-8, max_iter=3000, tol=1e-9)
    true_amp = np.array([f @ f.T for f in factors])
    var_true = np.array([np.trace(a) for a in true_amp])
    var_fit = np.array([np.trace(a) for a in amp])
    # Variance in non-overlapping timescale windows is recovered; the split
    # between adjacent half-decade bands is not identifiable and is not tested.
    assert var_fit[:3].sum() == pytest.approx(var_true[:3].sum(), rel=0.08)
    assert var_fit[3:].sum() == pytest.approx(var_true[3:].sum(), rel=0.15)
    assert var_fit.sum() == pytest.approx(var_true.sum(), rel=0.05)

    def gamma(tau):
        return np.array([float(tau == 0) if t == 0 else np.exp(-tau / t) for t in BANDS])

    exact = np.array([np.tensordot(gamma(tau), true_amp, axes=(0, 0)) for tau in lags])
    model = np.array([np.tensordot(gamma(tau), amp, axes=(0, 0)) for tau in lags])
    rel = np.linalg.norm(model - exact) / np.linalg.norm(exact)
    assert rel < 0.08, rel
    assert info["converged"]
    assert info["rel_residual"] < 0.05


def test_rank_truncation_and_local_residual_reach_the_floor():
    factors = truth(n_res=6, seed=3)
    reps = simulate(factors, n_rep=4, n_frames=3000, seed=40)
    n_res = 6
    lags = lag_grid(3000, 14, 0.1)
    residues = np.arange(n_res)
    fit, held = reps[:3], reps[3]
    diag = Lagged.pool([diag_block_sums(x.reshape(-1, n_res, 3), lags) for x in fit]).mean()
    pair = Lagged.pool([cross_sums(x, lags) for x in fit]).mean()
    target = summarize(diag_block_sums(held.reshape(-1, n_res, 3), lags).mean(), cross_sums(held, lags).mean())
    floor = distortion(summarize(diag, pair), target)
    kern_fit = pooled_kernel_matrix(BANDS, [3000] * 3, lags)
    kern_h = kernel_matrix(BANDS, 3000, lags)
    amp, _ = fit_psd_bands(pair, kern_fit, max_iter=2000, tol=1e-9)
    comps = spectral_components(amp, np.eye(3 * n_res))
    results = {}
    for kind, rank in [("sub", 0), ("sub", 3), ("sub", 3 * n_res), ("band", 3), ("band", len(comps))]:
        if kind == "sub":
            model = subspace_model(BANDS, amp, np.eye(3 * n_res), rank)
        else:
            model = model_from_components(BANDS, comps, rank)
        low = np.tensordot(kern_fit, model.band_diag(), axes=(1, 0))
        local, _ = fit_local_residual(diag, low, kern_fit, max_iter=1000, tol=1e-9)
        d, p = model.with_local(local).predict(kern_h, kern_h, residues)
        results[(kind, rank)] = {k: v / floor[k] for k, v in distortion((d, p), target).items()}
    full = results[("sub", 3 * n_res)]
    # A statistical codec at full rank is as close to a new replica as raw stats (denoising: <= 1).
    assert max(full.values()) < 1.1
    np.testing.assert_allclose(list(full.values()), list(results[("band", len(comps))].values()), rtol=0.05)
    # Subspace rank 3 = the true collective rank reaches the floor on coupling ...
    assert results[("sub", 3)]["dccm_rmse"] < 1.2 and results[("sub", 3)]["frob_rel"] < 1.1
    # ... per-band truncation at the same rank does not (leakage across adjacent bands) ...
    assert results[("band", 3)]["dccm_rmse"] > 2.0
    # ... and local-only (rank 0) captures per-residue statistics but not coupling.
    assert results[("sub", 0)]["msf_rel"] < 1.1 and results[("sub", 0)]["acf_rmse"] < 1.1
    assert results[("sub", 0)]["dccm_rmse"] > 5.0


def test_fit_batched_equals_individual():
    rng = np.random.default_rng(5)
    lags = np.array([0, 1, 3, 9, 27])
    kern = kernel_matrix(BANDS[:4], 500, lags)
    targets = np.stack([proj_psd(rng.standard_normal((len(lags), 3, 3))) for _ in range(4)], axis=1)
    batched, _ = fit_psd_bands(targets, kern, max_iter=400, tol=1e-12, obj_tol=0)
    for i in range(4):
        single, _ = fit_psd_bands(targets[:, i] * (np.abs(targets[0]).max() / np.abs(targets[0, i]).max()),
                                  kern, max_iter=400, tol=1e-12, obj_tol=0)
        scale = np.abs(targets[0, i]).max() / np.abs(targets[0]).max()
        np.testing.assert_allclose(batched[:, i], single * scale, atol=2e-3 * np.abs(batched).max())


def test_pca_exp_recovers_single_timescales():
    rng = np.random.default_rng(7)
    basis = np.linalg.qr(rng.standard_normal((9, 9)))[0]
    factors = [basis[:, [0]] * 2.0, basis[:, [1]] * 1.0]
    reps = [center(ou_mixture(5000, [40.0, 4.0], factors, s)) for s in range(4)]
    lags = lag_grid(5000, 16, 0.1)
    pc = Lagged.pool([cross_sums(x @ basis, lags) for x in reps]).mean()
    cand = pca_exp_candidates(5000, 96)
    amp, best = fit_pca_exp(np.diagonal(pc, axis1=1, axis2=2), basis, [5000] * 4, lags, cand)
    assert cand[best[0]] == pytest.approx(40.0, rel=0.25)
    assert cand[best[1]] == pytest.approx(4.0, rel=0.25)
    assert amp[0] == pytest.approx(4.0, rel=0.2)
    model = pca_exp_model(basis, amp, best, cand, 2)
    assert model.factors.shape == (9, 2)


def test_dct_truncation_keeps_slow_and_drops_fast():
    t = np.arange(1000)
    slow = np.cos(np.pi * 2 * (t + 0.5) / 1000)
    fast = 0.5 * np.cos(np.pi * 600 * (t + 0.5) / 1000)
    recon = dct_truncate(np.stack([slow + fast], axis=1), 10)[:, 0]
    np.testing.assert_allclose(recon, slow, atol=1e-8)
