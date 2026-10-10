"""The torch backend reproduces the numpy reference kernel by kernel.

Runs the torch code path on the CPU ("torch-cpu"); on a GPU node the same
code runs with device="cuda" (see ``python -m flexcompress.run bench``).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flexcompress import accel  # noqa: E402
from flexcompress import diagnostics as dg  # noqa: E402
from flexcompress import spectral as sp  # noqa: E402
from flexcompress.kernels import kernel_matrix  # noqa: E402
from flexcompress.stats import Lagged, center, lag_grid, pair_rows  # noqa: E402
from flexcompress.synthetic import ou_mixture  # noqa: E402


@pytest.fixture
def torch_cpu():
    assert accel.configure("torch-cpu") == "torch:cpu"
    yield
    accel.configure("numpy")


def both(fn):
    """Run ``fn`` under numpy and under torch-cpu; return (numpy, torch) results."""

    accel.configure("numpy")
    ref = fn()
    accel.configure("torch-cpu")
    try:
        out = fn()
    finally:
        accel.configure("numpy")
    return ref, out


def test_configure_fallbacks():
    assert accel.configure("numpy") == "numpy" and not accel.enabled()
    if not torch.cuda.is_available():
        assert accel.configure("auto") == "numpy"
        with pytest.raises(RuntimeError):
            accel.configure("cuda")
    accel.configure("numpy")


def test_lagged_statistics_match():
    rng = np.random.default_rng(0)
    dx = center(rng.standard_normal((700, 9, 3)).cumsum(axis=0) * 0.05 + rng.standard_normal((700, 9, 3)))
    lags = lag_grid(300, 8, 0.1)
    basis = np.linalg.qr(rng.standard_normal((27, 6)))[0]
    for fn in (lambda: accel.lagged_diag(dx, lags, 300),
               lambda: accel.lagged_cross(dx.reshape(700, -1), lags, 300),
               lambda: accel.lagged_cross(dx.reshape(700, -1), lags, 300, basis=basis)):
        ref, out = both(fn)
        assert isinstance(out, Lagged)
        np.testing.assert_allclose(out.sums, ref.sums, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(out.counts, ref.counts)


def test_pca_and_covariance_kernels_match():
    rng = np.random.default_rng(1)
    flats = [center(rng.standard_normal((400, 30)) @ rng.standard_normal((30, 30))) for _ in range(2)]
    (e0, v0, t0), (e1, v1, t1) = both(lambda: accel.gram_top_eigh(flats, 8))
    np.testing.assert_allclose(e1, e0, rtol=1e-10)
    np.testing.assert_allclose(v1 @ v1.T, v0 @ v0.T, atol=1e-8)  # same subspace (signs free)
    assert t1 == pytest.approx(t0)
    w0, w1 = both(lambda: accel.eigvalsh_gram(flats))
    np.testing.assert_allclose(w1, w0, rtol=1e-9, atol=1e-12)
    (p0, c0, s0), (p1, c1, s1) = both(lambda: accel.top_pcs(flats[0], 5))
    np.testing.assert_allclose(p1, p0, rtol=1e-9)
    approx = both(lambda: accel.top_pcs(flats[0], 5, exact_max_dim=10))  # randomized path
    np.testing.assert_allclose(approx[1][0], p0, rtol=1e-6)

    def outer():
        acc = accel.OuterSum(30)
        for f in flats:
            acc.add(f.astype(np.float32) + 3.0)
        return acc.top_eigvals(6)

    (o0, tr0), (o1, tr1) = both(outer)
    np.testing.assert_allclose(o1, o0, rtol=1e-8)
    assert tr1 == pytest.approx(tr0)


def _band_problem(n_res=5, seed=0):
    bands = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
    rng = np.random.default_rng(seed)
    dim = 3 * n_res
    factors = [0.3 * np.eye(dim), np.zeros((dim, 0)), 0.5 * rng.standard_normal((dim, 2)),
               np.zeros((dim, 0)), rng.standard_normal((dim, 1))]
    x = center(ou_mixture(3000, bands, factors, seed + 1))
    lags = lag_grid(3000, 12, 0.1)
    from flexcompress.stats import cross_sums, diag_block_sums

    pair = cross_sums(x, lags).mean()
    diag = diag_block_sums(x.reshape(3000, n_res, 3), lags).mean()
    return bands, lags, kernel_matrix(bands, 3000, lags), pair, diag


def test_admm_fit_matches_exactly_with_eigh_projection():
    bands, lags, kern, pair, diag = _band_problem()
    kw = dict(ridge=1e-3, max_iter=400, tol=1e-6, obj_tol=1e-6)
    (a0, i0), (a1, i1) = both(lambda: accel.fit_psd_bands(pair, kern, **kw))
    assert i1["iterations"] == i0["iterations"] and i1["projection"] == "eigh"
    np.testing.assert_allclose(a1, a0, rtol=1e-7, atol=1e-9 * np.abs(a0).max())
    (l0, _), (l1, _) = both(lambda: accel.fit_psd_bands(diag, kern, **kw))  # batched 3x3
    np.testing.assert_allclose(l1, l0, rtol=1e-7, atol=1e-9 * np.abs(l0).max())


def test_newton_schulz_projection_and_fit():
    rng = np.random.default_rng(3)
    m = rng.standard_normal((4, 40, 40))
    m = m + np.swapaxes(m, -1, -2)
    exact = sp.proj_psd(m)
    accel.configure("torch-cpu", psd_method="newton_schulz", ns_iters=60)
    try:
        ns = accel._np(accel._proj_newton_schulz(accel._t(m), 60))
        assert np.abs(ns - exact).max() < 1e-6 * np.abs(exact).max()
        bands, lags, kern, pair, diag = _band_problem(n_res=13)  # 39 x 39 > 32
        a_ns, info = accel.fit_psd_bands(pair, kern, ridge=1e-3, max_iter=400, tol=1e-6)
        assert info["projection"] == "newton_schulz"
    finally:
        accel.configure("numpy")
    a_ref, _ = sp.fit_psd_bands(pair, kern, ridge=1e-3, max_iter=400, tol=1e-6)
    assert np.all(np.linalg.eigvalsh(a_ns) > -1e-10)  # exact final projection
    pred_ns = np.tensordot(kern, a_ns, axes=(1, 0))
    pred_ref = np.tensordot(kern, a_ref, axes=(1, 0))
    assert np.linalg.norm(pred_ns - pred_ref) / np.linalg.norm(pred_ref) < 1e-3


def test_model_predictions_match():
    bands, lags, kern, pair, diag = _band_problem(n_res=6)
    amp, _ = sp.fit_psd_bands(pair, kern, ridge=1e-3)
    basis = np.eye(18)
    rng = np.random.default_rng(4)
    local = sp.proj_psd(rng.standard_normal((len(bands), 6, 3, 3)))
    residues = np.array([0, 2, 5])
    models = [sp.subspace_model(bands, amp, basis, 4).with_local(local),
              sp.subspace_model(bands, amp, basis, 0).with_local(local),
              sp.model_from_components(bands, sp.spectral_components(amp, basis), 5).with_local(local),
              sp.model_from_components(bands, sp.spectral_components(amp, basis), 3)]
    for model in models:
        (d0, p0), (d1, p1) = both(lambda model=model: accel.predict(model, kern, kern[:4], residues))
        np.testing.assert_allclose(d1, d0, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(p1, p0, rtol=1e-10, atol=1e-12)
        b0, b1 = both(lambda model=model: accel.band_diag(model))
        np.testing.assert_allclose(b1, b0, rtol=1e-10, atol=1e-12)


def test_dct_baseline_matches():
    rng = np.random.default_rng(5)
    flat = center(rng.standard_normal((600, 12)).cumsum(axis=0))
    lags = lag_grid(300, 6, 0.1)
    rows = pair_rows(np.array([0, 3]))
    (d0, q0), (d1, q1) = both(lambda: accel.dct_lagged(flat, 40, lags, lags[:3], rows, 4, 300))
    np.testing.assert_allclose(d1.sums, d0.sums, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(q1.sums, q0.sums, rtol=1e-8, atol=1e-8)


def test_diagnostics_untouched_by_backend(torch_cpu):
    assert accel.enabled()
    x = center(np.random.default_rng(6).standard_normal((200, 10)))
    var, vec, total = accel.top_pcs(x, 3)
    ref = dg.top_pcs(x, 3)
    np.testing.assert_allclose(var, ref[0], rtol=1e-9)


def test_jacobi_psd_projection_matches_eigh():
    """The CUDA path for 3x3 projections (no cuSOLVER workspace) is exact to rounding."""

    torch = pytest.importorskip("torch")  # noqa: F841
    from flexcompress import accel

    accel.configure("torch-cpu", psd_method="jacobi")
    try:
        rng = np.random.default_rng(0)
        m = rng.standard_normal((5000, 3, 3))
        m = m + m.transpose(0, 2, 1)
        special = np.stack([np.eye(3), np.zeros((3, 3)), np.diag([1.0, 1.0, -1.0]),
                            np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), -np.eye(3),
                            np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                            np.diag([1e-300, 1.0, 1e8])])
        m = np.concatenate([m, special, 1e-8 * m, 1e8 * m])
        w, v = np.linalg.eigh(m)
        ref = (v * np.clip(w, 0, None)[:, None, :]) @ v.transpose(0, 2, 1)
        got = accel._np(accel._proj_exact(accel._t(m)))
        scale = np.abs(m).max(axis=(1, 2), keepdims=True) + 1e-300
        assert np.max(np.abs(got - ref) / scale) < 1e-12
    finally:
        accel.configure("numpy")
