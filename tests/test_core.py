"""Unit tests: estimator kernels, lagged statistics, metrics, diagnostics."""

import numpy as np
import pytest
from scipy.linalg import toeplitz

from flexcompress import diagnostics as dg
from flexcompress.kernels import autocov_exponential, expected_lagged_cov, kernel_matrix, pooled_kernel_matrix
from flexcompress.metrics import METRICS, distortion, summarize
from flexcompress.stats import (
    Lagged,
    center,
    cross_sums,
    diag_block_sums,
    lag_grid,
    segmented,
    subset_lags,
    window_starts,
)
from flexcompress.synthetic import ou_mixture, rotational_ou


def exact_expected(gamma, n, tau):
    """E[C_hat(tau)] = tr(M Sigma) for the quadratic form defining the estimator."""

    sigma = toeplitz(gamma[:n])
    centre = np.eye(n) - np.ones((n, n)) / n
    shift = np.zeros((n, n))
    shift[np.arange(n - tau), np.arange(tau, n)] = 1.0
    m = centre @ shift @ centre / (n - tau)
    return float(np.trace(m @ sigma))


@pytest.mark.parametrize("t", [0.0, 0.7, 3.0, 25.0, 400.0])
def test_expected_estimator_matches_exact_quadratic_form(t):
    n = 40
    gamma = autocov_exponential(t, n)
    lags = np.array([0, 1, 2, 5, 13, 39])
    fast = expected_lagged_cov(gamma, n, lags)
    exact = [exact_expected(gamma, n, int(tau)) for tau in lags]
    np.testing.assert_allclose(fast, exact, rtol=1e-10, atol=1e-12)


def test_kernel_bias_is_physical():
    k = kernel_matrix([0.0, 5.0, 1e6], 200, [0, 1, 10])
    assert k[0, 0] == pytest.approx(1 - 1 / 200)  # white noise loses one dof
    assert abs(k[0, 2]) < 1e-3  # near-constant process is removed by mean subtraction
    assert k[1, 1] < np.exp(-1 / 5.0)  # finite-length estimator is biased low


def test_pooled_kernel_is_count_weighted():
    lags = np.array([0, 3])
    pooled = pooled_kernel_matrix([4.0], [50, 150], lags)
    w = np.array([[50 - lags], [150 - lags]], dtype=float)[:, 0, :, None]
    manual = (w[0] * kernel_matrix([4.0], 50, lags) + w[1] * kernel_matrix([4.0], 150, lags)) / (w[0] + w[1])
    np.testing.assert_allclose(pooled, manual)


def test_lag_grid_and_subset():
    lags = lag_grid(10000, 16, 0.1)
    assert lags[0] == 0 and lags[-1] == 1000 and np.all(np.diff(lags) > 0)
    sub = subset_lags(lags, 6)
    assert sub[0] == 0 and sub[-1] == 1000 and len(sub) == 6


def test_lagged_sums_match_naive_and_pool_exactly():
    rng = np.random.default_rng(0)
    a = center(rng.standard_normal((60, 4, 3)))
    b = center(rng.standard_normal((45, 4, 3)))
    lags = np.array([0, 1, 7])
    d = diag_block_sums(a, lags)
    for li, tau in enumerate(lags):
        naive = np.einsum("tni,tnj->nij", a[: 60 - tau], a[tau:])
        np.testing.assert_allclose(d.sums[li], naive, atol=1e-12)
    flat_a, flat_b = a.reshape(60, -1), b.reshape(45, -1)
    pooled = Lagged.pool([cross_sums(flat_a, lags), cross_sums(flat_b, lags)]).mean()
    tau = 7
    manual = (flat_a[:-tau].T @ flat_a[tau:] + flat_b[:-tau].T @ flat_b[tau:]) / (60 - tau + 45 - tau)
    np.testing.assert_allclose(pooled[2], manual, atol=1e-12)


def _stats(x, lags, pair_lags):
    dx = center(x)
    diag = diag_block_sums(dx, lags).mean()
    pair = cross_sums(dx.reshape(len(dx), -1), pair_lags).mean()
    return diag, pair


def test_metrics_zero_on_identity_and_positive_otherwise():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 6, 3)).cumsum(axis=0) * 0.01 + rng.standard_normal((300, 6, 3))
    lags = np.array([0, 1, 4, 16])
    s = _stats(x, lags, lags)
    d0 = distortion(s, s)
    assert set(d0) == set(METRICS)
    assert all(abs(v) < 1e-12 for v in d0.values())
    other = _stats(x * np.array([1.0, 2.0, 1.0]), lags, lags)
    d1 = distortion(other, s)
    assert d1["aniso"] > 0.05 and d1["msf_rel"] > 0.1
    summ = summarize(*s)
    assert summ["dcc"].shape == (4, 6, 6)
    np.testing.assert_allclose(np.diagonal(summ["dcc"][0]), 1.0)


def test_cosine_content_and_rmsip():
    n = 500
    t = (np.arange(n) + 0.5) / n
    p = np.stack([np.cos(np.pi * t), np.sin(7 * np.pi * t)], axis=1)
    c = dg.cosine_content(p, 2)
    assert c[0] == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(2)
    walk = center(rng.standard_normal((2000, 30)).cumsum(axis=0))
    var, vec, total = dg.top_pcs(walk, 3)
    assert dg.cosine_content(walk @ vec, 1)[0] > 0.6  # Hess: diffusion PC1 ~ half cosine
    q, _ = np.linalg.qr(rng.standard_normal((30, 5)))
    assert dg.rmsip(q, q, 5) == pytest.approx(1.0)
    counts = dg.variance_counts(np.array([5.0, 3.0, 1.0, 1.0]), 10.0, [0.5, 0.8, 0.95])
    assert counts == {"n_pc_50": 1.0, "n_pc_80": 2.0, "n_pc_95": 4.0}


def test_randomized_pca_matches_exact():
    rng = np.random.default_rng(3)
    basis = np.linalg.qr(rng.standard_normal((400, 6)))[0]
    x = center(rng.standard_normal((800, 6)) * np.array([9, 7, 5, 3, 2, 1.0]) @ basis.T
               + 0.05 * rng.standard_normal((800, 400)))
    exact = dg.top_pcs(x, 4, exact_max_dim=10000)[0]
    approx = dg.top_pcs(x, 4, exact_max_dim=10)[0]
    np.testing.assert_allclose(approx, exact, rtol=1e-3)


def test_irreversibility_detected_only_for_nonreversible_process():
    lags = np.array([0, 2, 5])
    rot = [rotational_ou(4000, 10.0, 0.15, seed) for seed in (1, 2)]
    rev = [rotational_ou(4000, 10.0, 0.0, seed) for seed in (3, 4)]

    def anti(x):
        return dg.asymmetry(cross_sums(center(x), lags).mean())

    a_rot = [anti(x) for x in rot]
    a_rev = [anti(x) for x in rev]
    assert a_rot[0][1][1] > 0.3 and a_rev[0][1][1] < 0.15
    cos_rot = np.sum(a_rot[0][0][1] * a_rot[1][0][1]) / (np.linalg.norm(a_rot[0][0][1]) * np.linalg.norm(a_rot[1][0][1]))
    assert cos_rot > 0.95  # reproducible circulation direction


def test_native_contacts_full_and_lost():
    native = np.stack([np.arange(20) * 1.0, np.zeros(20), np.zeros(20)], axis=1)  # dense line: contacts
    traj = np.repeat(native[None], 10, axis=0)
    q = dg.native_contact_fraction(traj, native, n_samples=5)
    np.testing.assert_allclose(q, 1.0)
    stretched = traj * np.linspace(1, 3, 10)[:, None, None]
    assert dg.native_contact_fraction(stretched, native, n_samples=10)[-1] < 0.2


def test_window_starts_cover_replica():
    assert list(window_starts(200, 200)) == [0]
    starts = window_starts(500, 200)
    assert starts[0] == 0 and starts[-1] == 300 and len(starts) == 3
    with pytest.raises(ValueError):
        window_starts(100, 200)


def test_segmented_estimator_matches_segment_kernel():
    """Long and short replicas analysed in common windows share one expectation."""

    lags = np.array([0, 2, 8, 20])
    t = 15.0
    est_long, est_short = [], []
    for seed in range(300):
        x = ou_mixture(600, [t], [np.ones((1, 1))], seed)
        est_long.append(segmented(x, lags, 200, cross_sums).mean()[:, 0, 0])
        est_short.append(cross_sums(center(x[:200]), lags).mean()[:, 0, 0])
    expected = kernel_matrix([t], 200, lags)[:, 0]
    np.testing.assert_allclose(np.mean(est_long, axis=0), expected, atol=0.025)
    np.testing.assert_allclose(np.mean(est_short, axis=0), expected, atol=0.04)
    # Without windowing the 600-frame estimator has a different (less biased) expectation.
    assert kernel_matrix([t], 600, lags)[1, 0] - expected[1] > 0.04
