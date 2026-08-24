"""Numerical-stability tests for the Minkowski norm inside the Lorentz aggregators.

``lorentz_residual`` and ``lorentz_midpoint`` normalize a weighted sum ``h`` of on-sheet points
by ``sqrt(c * |<h,h>_L|)``. Evaluating ``<h,h>_L`` as the literal ``-h_0^2 + ||h_s||^2``
subtracts two ``O(||s||^2)`` squares to reach an ``O(1/c)`` result: in float32 the relative
error grows like ``eps * c * ||s||^2``, the computed value flips sign above ``||s|| ~ 1e4``,
``abs()`` hides the flip, and the ``eps = 1e-7`` floor then inflates the output by
``2/sqrt(1e-7) ~ 6325x`` with no warning.

Both functions instead evaluate ``<h,h>_L`` through an exact algebraic identity whose every
term is ``O(||x_i - x_j||^2)``. These tests
(a) prove the identity exactly in rational arithmetic (``fractions.Fraction``, no floats),
(b) pin the float32 accuracy gain against transcriptions of the old naive bodies, which double
    as anti-reversion oracles — a revert to the naive form makes them fail, and
(c) check the two forms still agree at ordinary radii, plus edge cases and jit/vmap, and
(d) pin that the outputs stay on the sheet even for inputs float storage cannot keep on it (a
    boundary-lifted Poincare point) — the identity assumes exactly on-sheet inputs, so the outputs'
    time coordinate is reconstructed from their spatial part to restore the invariant the old
    self-normalizing form had.

Dimension key:
  M: points per midpoint    N: midpoint outputs    A: ambient dim (= D + 1)
  D: spatial dim            B: batch
"""

import random
from fractions import Fraction

import jax
import jax.numpy as jnp
import pytest

# tests/conftest.py enables x64 at import time, but ``seed_jax`` is not autouse, so set it
# here too: every float64 reference in this file depends on it.
jax.config.update("jax_enable_x64", True)

from hyperbolix.decomposition.frechet import frechet_mean  # noqa: E402
from hyperbolix.manifolds import Hyperboloid, Poincare  # noqa: E402
from hyperbolix.manifolds.isometry_mappings import poincare_to_hyperboloid  # noqa: E402
from hyperbolix.nn_layers.hyperboloid_core import lorentz_midpoint, lorentz_residual  # noqa: E402

# ---------------------------------------------------------------------------------------
# Anti-reversion oracles: verbatim transcriptions of the pre-fix (naive) bodies.
# ---------------------------------------------------------------------------------------


def _naive_lorentz_residual(x, y, w_y, c, eps=1e-7):
    """Old ``lorentz_residual`` body: ``<ave,ave>_L`` as the literal ``-ave_0^2 + ||ave_s||^2``."""
    ave_A = x + w_y * y
    mink_1 = -(ave_A[..., 0:1] ** 2) + jnp.sum(ave_A[..., 1:] ** 2, axis=-1, keepdims=True)
    denom_1 = jnp.sqrt(jnp.maximum(c * jnp.abs(mink_1), eps))
    return ave_A / denom_1


def _naive_lorentz_midpoint(points, weights, c, eps=1e-7):
    """Old ``lorentz_midpoint`` body: ``<h,h>_L`` as the literal ``-h_0^2 + ||h_s||^2``."""
    h_NA = jnp.einsum("...nm,...ma->...na", weights, points)
    mink_1 = -(h_NA[..., 0:1] ** 2) + jnp.sum(h_NA[..., 1:] ** 2, axis=-1, keepdims=True)
    denom_1 = jnp.sqrt(jnp.maximum(c * jnp.abs(mink_1), eps))
    return h_NA / denom_1


# ---------------------------------------------------------------------------------------
# Float helpers
# ---------------------------------------------------------------------------------------


def _onsheet(key, d, c, snorm, dtype=jnp.float64):
    """Point on the upper sheet of curvature ``c`` whose spatial norm is exactly ``snorm``.

    Built in float64 (``x_0 = sqrt(||v||^2 + 1/c)``) and cast down, so the float32 input is the
    correctly-rounded image of an exactly-on-sheet point rather than an independently rounded one.
    """
    v_D = jax.random.normal(key, (d,), dtype=jnp.float64)
    v_D = v_D / jnp.linalg.norm(v_D) * snorm
    x0 = jnp.sqrt(jnp.sum(v_D**2) + 1.0 / c)
    return jnp.concatenate([x0[None], v_D]).astype(dtype)


def _reproject(p_A, c):
    """Recompute the time coordinate of ``p_A`` so it lies exactly on the sheet (float64)."""
    s_D = p_A[1:]
    x0 = jnp.sqrt(jnp.sum(s_D**2) + 1.0 / c)
    return jnp.concatenate([x0[None], s_D])


def _rel_err(approx, truth):
    """Max absolute deviation from ``truth``, relative to ``truth``'s largest entry."""
    approx = jnp.asarray(approx, dtype=jnp.float64)
    truth = jnp.asarray(truth, dtype=jnp.float64)
    return float(jnp.max(jnp.abs(approx - truth)) / jnp.maximum(jnp.max(jnp.abs(truth)), 1e-300))


def _near_pair(seed, d, c, snorm):
    """Two on-sheet float64 points a relative step of 1e-4 apart (the hardest realistic regime)."""
    kx, ky = jax.random.split(jax.random.PRNGKey(seed))
    x_A = _onsheet(kx, d, c, snorm)
    y_far_A = _onsheet(ky, d, c, snorm)
    return x_A, _reproject(x_A + 1e-4 * (y_far_A - x_A), c)


def _near_points(seed, d, c, snorm, m):
    """``M`` on-sheet float64 points, all within a relative step of 1e-4 of the first."""
    keys = jax.random.split(jax.random.PRNGKey(seed), m)
    x0_A = _onsheet(keys[0], d, c, snorm)
    pts = [x0_A] + [_reproject(x0_A + 1e-4 * (_onsheet(k, d, c, snorm) - x0_A), c) for k in keys[1:]]
    return jnp.stack(pts)  # (M, A)


def _softmax_weights(seed, n, m):
    """(N, M) row-stochastic float64 weights, as an attention layer would supply."""
    return jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(seed), (n, m), dtype=jnp.float64), axis=-1)


# ---------------------------------------------------------------------------------------
# 1. Exact identities in rational arithmetic (no floating point anywhere)
# ---------------------------------------------------------------------------------------


def _mink(a, b):
    """Minkowski inner product of two Fraction vectors, signature (-, +, ..., +)."""
    return -a[0] * b[0] + sum(ai * bi for ai, bi in zip(a[1:], b[1:], strict=True))


def _rational_sheet_point(rnd, d, q):
    """Exactly-on-sheet rational point of curvature ``c = 1/q^2`` via the rational parametrization.

    With ``|t| < 1`` rational, ``x_0 = q(1 + |t|^2)/(1 - |t|^2)`` and ``x_s = 2 q t/(1 - |t|^2)``
    satisfy ``-x_0^2 + ||x_s||^2 = -q^2 = -1/c`` identically.
    """
    t = [Fraction(rnd.randint(-10, 10), rnd.randint(1, 20)) for _ in range(d)]
    t_sq = sum(ti * ti for ti in t)
    # Shrink t by an exact rational factor so |t| < 1 (t_sq / k^2 < 1 with k > sqrt(t_sq)).
    k = int(t_sq) + 2
    scale = Fraction(rnd.randint(1, 9), 10) / k
    t = [ti * scale for ti in t]
    t_sq = sum(ti * ti for ti in t)
    assert t_sq < 1
    den = 1 - t_sq
    x0 = q * (1 + t_sq) / den
    xs = [2 * q * ti / den for ti in t]
    point = [x0, *xs]
    assert _mink(point, point) == -q * q  # == -1/c, exactly
    return point


def _rational_weight(rnd):
    """Random rational weight, deliberately including negatives and exact zero."""
    return Fraction(rnd.randint(-6, 6), rnd.randint(1, 7))


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6])
def test_lorentz_residual_exact_identity(d):
    """``<x + w y, x + w y>_L == -(1+w)^2/c - w <x-y, x-y>_L`` exactly, for any real ``w``."""
    rnd = random.Random(20260824 + d)
    for _ in range(25):
        q = Fraction(rnd.randint(1, 5), rnd.randint(1, 5))
        c_inv = q * q  # 1/c
        x = _rational_sheet_point(rnd, d, q)
        y = _rational_sheet_point(rnd, d, q)
        w = _rational_weight(rnd)

        ave = [xi + w * yi for xi, yi in zip(x, y, strict=True)]
        naive = _mink(ave, ave)

        diff = [xi - yi for xi, yi in zip(x, y, strict=True)]
        identity = -((1 + w) ** 2) * c_inv - w * _mink(diff, diff)

        assert naive == identity, f"d={d}, w={w}, q={q}: {naive} != {identity}"


@pytest.mark.parametrize("m", [2, 3, 9])
@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6])
def test_lorentz_midpoint_exact_identity(m, d):
    """``<h,h>_L == -W^2/c - W sum_m w_m <delta_m,delta_m>_L + <Delta,Delta>_L`` exactly."""
    rnd = random.Random(31415 + 100 * m + d)
    for _ in range(15):
        q = Fraction(rnd.randint(1, 5), rnd.randint(1, 5))
        c_inv = q * q  # 1/c
        pts = [_rational_sheet_point(rnd, d, q) for _ in range(m)]
        weights = [_rational_weight(rnd) for _ in range(m)]

        h = [sum(w * p[i] for w, p in zip(weights, pts, strict=True)) for i in range(d + 1)]
        naive = _mink(h, h)

        p0 = pts[0]  # reference point
        deltas = [[pi - p0i for pi, p0i in zip(p, p0, strict=True)] for p in pts]
        big_w = sum(weights)
        big_delta = [sum(w * dm[i] for w, dm in zip(weights, deltas, strict=True)) for i in range(d + 1)]
        identity = (
            -(big_w**2) * c_inv
            - big_w * sum(w * _mink(dm, dm) for w, dm in zip(weights, deltas, strict=True))
            + _mink(big_delta, big_delta)
        )

        assert naive == identity, f"d={d}, m={m}, q={q}: {naive} != {identity}"


# ---------------------------------------------------------------------------------------
# 2-3. float32 accuracy of lorentz_residual (with the naive form pinned as failing)
# ---------------------------------------------------------------------------------------

_C = 0.1
_SNORM = 1e3
_D = 16
_SEEDS = (0, 1, 2, 3)


def test_lorentz_residual_float32_forward_accuracy():
    """At ``||s|| = 1e3`` the fixed form keeps float32 forward accuracy; the naive one does not.

    The naive error is luck-dependent per seed (how the 16-term sum of squares happens to round),
    so the anti-reversion pin is two-part: on *every* seed the naive form must be at least 100x
    worse, and on the *worst* seed it must exceed 1e-3 outright.
    """
    naive_errs = []
    for seed in _SEEDS:
        x64_A, y64_A = _near_pair(seed, _D, _C, _SNORM)
        x32_A, y32_A = x64_A.astype(jnp.float32), y64_A.astype(jnp.float32)
        truth_A = lorentz_residual(x64_A, y64_A, 1.0, _C)  # identity is exact -> f64 is the truth

        lib = _rel_err(lorentz_residual(x32_A, y32_A, 1.0, _C), truth_A)
        naive = _rel_err(_naive_lorentz_residual(x32_A, y32_A, 1.0, _C), truth_A)
        naive_errs.append(naive)

        assert lib < 1e-5, f"seed {seed}: library float32 forward rel. error {lib:.2e}"
        assert naive > 100.0 * lib, f"seed {seed}: naive {naive:.2e} not >100x library {lib:.2e}"

    assert max(naive_errs) > 1e-3, f"naive float32 forward rel. error never exceeded 1e-3: {max(naive_errs):.2e}"


def test_lorentz_residual_float32_gradient_accuracy():
    """Same regime, gradient w.r.t. ``x`` of ``sum(out * g)`` for a fixed random cotangent ``g``."""
    naive_errs = []
    for seed in _SEEDS:
        x64_A, y64_A = _near_pair(seed, _D, _C, _SNORM)
        g64_A = jax.random.normal(jax.random.PRNGKey(500 + seed), x64_A.shape, dtype=jnp.float64)

        def loss(fn, g_A=g64_A):
            def inner(x_A, y_A):
                return jnp.sum(fn(x_A, y_A, 1.0, _C) * g_A.astype(x_A.dtype))

            return inner

        truth_A = jax.grad(loss(lorentz_residual))(x64_A, y64_A)
        x32_A, y32_A = x64_A.astype(jnp.float32), y64_A.astype(jnp.float32)

        lib = _rel_err(jax.grad(loss(lorentz_residual))(x32_A, y32_A), truth_A)
        naive = _rel_err(jax.grad(loss(_naive_lorentz_residual))(x32_A, y32_A), truth_A)
        naive_errs.append(naive)

        assert lib < 2e-3, f"seed {seed}: library float32 gradient rel. error {lib:.2e}"
        assert naive > 100.0 * lib, f"seed {seed}: naive {naive:.2e} not >100x library {lib:.2e}"

    assert max(naive_errs) > 5e-3, f"naive float32 gradient rel. error never exceeded 5e-3: {max(naive_errs):.2e}"


def test_lorentz_residual_no_eps_inflation():
    """``x == y`` at ``||s|| = 2e4``: the fixed form returns ``x``; the naive one hits the eps floor.

    For ``x == y`` and ``w_y = 1`` the exact result is ``2x / sqrt(c * 4/c) = x``. In float32 the
    naive Minkowski square rounds to (near) zero, ``maximum(., 1e-7)`` takes over, and the output
    is inflated by up to ``2/sqrt(1e-7) = 6324.6``. Whether the floor is hit depends on how the
    16-term sum happens to round, so the naive assertion is on the worst of 8 seeds.
    """
    snorm = 2e4
    naive_ratios = []
    for seed in range(8):
        x32_A = _onsheet(jax.random.PRNGKey(seed), _D, _C, snorm, dtype=jnp.float32)
        out_A = lorentz_residual(x32_A, x32_A, 1.0, _C)
        assert _rel_err(out_A, x32_A) < 1e-5, f"seed {seed}: lorentz_residual(x, x, 1) != x"
        naive_ratios.append(float(_naive_lorentz_residual(x32_A, x32_A, 1.0, _C)[0] / x32_A[0]))

    assert max(naive_ratios) > 1e3, f"naive time coordinate never inflated: max ratio {max(naive_ratios):.4g}"


# ---------------------------------------------------------------------------------------
# 5. float32 accuracy of lorentz_midpoint
# ---------------------------------------------------------------------------------------


def test_lorentz_midpoint_float32_accuracy():
    """M=8 points, N=3 softmax weight rows, ``||s|| = 1e3``, near regime: forward and gradient."""
    m, n = 8, 3
    naive_errs = []
    for seed in _SEEDS:
        pts64_MA = _near_points(seed, _D, _C, _SNORM, m)
        w64_NM = _softmax_weights(600 + seed, n, m)
        g64_NA = jax.random.normal(jax.random.PRNGKey(700 + seed), (n, _D + 1), dtype=jnp.float64)
        pts32_MA, w32_NM = pts64_MA.astype(jnp.float32), w64_NM.astype(jnp.float32)

        truth_NA = lorentz_midpoint(pts64_MA, w64_NM, _C)
        lib = _rel_err(lorentz_midpoint(pts32_MA, w32_NM, _C), truth_NA)
        naive = _rel_err(_naive_lorentz_midpoint(pts32_MA, w32_NM, _C), truth_NA)
        naive_errs.append(naive)
        assert lib < 1e-5, f"seed {seed}: library float32 forward rel. error {lib:.2e}"
        assert naive > 100.0 * lib, f"seed {seed}: naive {naive:.2e} not >100x library {lib:.2e}"

        def loss(fn, g_NA=g64_NA, w_NM=w64_NM):
            def inner(pts_MA):
                return jnp.sum(fn(pts_MA, w_NM.astype(pts_MA.dtype), _C) * g_NA.astype(pts_MA.dtype))

            return inner

        grad_truth_MA = jax.grad(loss(lorentz_midpoint))(pts64_MA)
        lib_grad = _rel_err(jax.grad(loss(lorentz_midpoint))(pts32_MA), grad_truth_MA)
        assert lib_grad < 3e-3, f"seed {seed}: library float32 gradient rel. error {lib_grad:.2e}"

    assert max(naive_errs) > 1e-3, f"naive float32 forward rel. error never exceeded 1e-3: {max(naive_errs):.2e}"


# ---------------------------------------------------------------------------------------
# 6. Ordinary radii: the fixed form must not move the answer
# ---------------------------------------------------------------------------------------


def _agreement_tol(c, snorm, w_sum=2.0):
    """How closely the two forms can possibly agree in float32 at this radius.

    They cannot agree more tightly than the *naive* form's own cancellation error, which is
    ``~ eps32 * (amplification)`` with ``amplification = c * ||s||^2 / w_sum^2`` — the ratio of
    the two cancelling squares to the ``W^2/c`` result. The factor 8 covers the accumulation of
    the 16-term sum of squares. Below ``||s|| ~ 5`` the amplification is ~1 and plain float32
    round-off (1e-6) dominates.

    Worked example (``c = 1``, ``||s|| = 20``, ``w_sum = 2``): amplification = 1*400/4 = 100, so
    ``8 * 1.19e-7 * 100 = 9.5e-5`` — and the measured disagreement there is 3.1e-5.
    """
    eps32 = float(jnp.finfo(jnp.float32).eps)
    return max(1e-6, 8.0 * eps32 * c * snorm**2 / w_sum**2)


@pytest.mark.parametrize("snorm", [0.5, 5.0, 20.0])
@pytest.mark.parametrize("c", [0.1, 1.0])
def test_lorentz_residual_small_norm_matches_naive(c, snorm):
    """Below the cancellation regime the fix does not move the answer.

    Two claims: the library form is exact to float32 round-off against the float64 truth, and it
    agrees with the old naive form to within the naive form's own round-off (see ``_agreement_tol``).
    """
    tol = _agreement_tol(c, snorm)
    for seed in _SEEDS:
        x64_A, y64_A = _near_pair(seed, _D, c, snorm)
        x32_A, y32_A = x64_A.astype(jnp.float32), y64_A.astype(jnp.float32)
        lib_A = lorentz_residual(x32_A, y32_A, 1.0, c)
        naive_A = _naive_lorentz_residual(x32_A, y32_A, 1.0, c)
        truth_A = lorentz_residual(x64_A, y64_A, 1.0, c)

        assert _rel_err(lib_A, truth_A) < 1e-6, f"c={c}, |s|={snorm}, seed {seed}: library vs f64 truth"
        assert _rel_err(lib_A, naive_A) < tol, f"c={c}, |s|={snorm}, seed {seed}: library vs naive (tol {tol:.1e})"


@pytest.mark.parametrize("snorm", [0.5, 5.0, 20.0])
@pytest.mark.parametrize("c", [0.1, 1.0])
def test_lorentz_midpoint_small_norm_matches_naive(c, snorm):
    """Below the cancellation regime the midpoint fix does not move the answer either.

    Softmax rows sum to 1, so the relevant ``w_sum`` in ``_agreement_tol`` is 1, not 2.
    """
    m, n = 8, 3
    tol = _agreement_tol(c, snorm, w_sum=1.0)
    for seed in _SEEDS:
        pts64_MA = _near_points(seed, _D, c, snorm, m)
        w64_NM = _softmax_weights(600 + seed, n, m)
        pts32_MA, w32_NM = pts64_MA.astype(jnp.float32), w64_NM.astype(jnp.float32)
        lib_NA = lorentz_midpoint(pts32_MA, w32_NM, c)
        naive_NA = _naive_lorentz_midpoint(pts32_MA, w32_NM, c)
        truth_NA = lorentz_midpoint(pts64_MA, w64_NM, c)

        assert _rel_err(lib_NA, truth_NA) < 1e-6, f"c={c}, |s|={snorm}, seed {seed}: library vs f64 truth"
        assert _rel_err(lib_NA, naive_NA) < tol, f"c={c}, |s|={snorm}, seed {seed}: library vs naive (tol {tol:.1e})"


# ---------------------------------------------------------------------------------------
# 7-9. Edge cases, broadcasting, jit/vmap
# ---------------------------------------------------------------------------------------


def test_lorentz_residual_edge_cases():
    """``x == y`` returns ``x``; a near-vertex point stays on the manifold; ``w_y = 0`` returns ``x``."""
    c = 1.0
    manifold = Hyperboloid(dtype=jnp.float64)

    # x == y with w_y = 1: 2x / sqrt(c * |<2x,2x>_L|) = 2x / 2 = x.
    x_A = _onsheet(jax.random.PRNGKey(11), _D, c, 3.0)
    assert _rel_err(lorentz_residual(x_A, x_A, 1.0, c), x_A) < 1e-6

    # Near the vertex (tiny spatial norm) the output must still satisfy <z,z>_L = -1/c.
    x_A = _onsheet(jax.random.PRNGKey(12), _D, c, 1e-3)
    y_A = _onsheet(jax.random.PRNGKey(13), _D, c, 1e-3)
    z_A = lorentz_residual(x_A, y_A, 1.0, c)
    lorentz_sq = -(z_A[0] ** 2) + jnp.sum(z_A[1:] ** 2)
    assert abs(float(lorentz_sq) + 1.0 / c) < 1e-9
    assert bool(manifold.is_in_manifold(z_A, c))

    # w_y = 0: ave = x, and c * |<x,x>_L| = 1, so the result is x unchanged.
    x_A = _onsheet(jax.random.PRNGKey(14), _D, c, 3.0)
    y_A = _onsheet(jax.random.PRNGKey(15), _D, c, 3.0)
    assert _rel_err(lorentz_residual(x_A, y_A, 0.0, c), x_A) < 1e-6


def test_lorentz_midpoint_broadcasts_points_over_weight_batch():
    """One shared ``(M, A)`` point set with ``(B, N, M)`` weights equals the per-batch loop."""
    c, m, n, b = 1.0, 5, 3, 4
    pts_MA = _near_points(21, _D, c, 2.0, m)  # (M, A) — no batch axis
    w_BNM = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(22), (b, n, m), dtype=jnp.float64), axis=-1)

    out_BNA = lorentz_midpoint(pts_MA, w_BNM, c)
    assert out_BNA.shape == (b, n, _D + 1)

    expected_BNA = jnp.stack([lorentz_midpoint(pts_MA, w_BNM[i], c) for i in range(b)])
    assert _rel_err(out_BNA, expected_BNA) < 1e-12


def test_jit_vmap_smoke():
    """Both aggregators compile and vectorize; outputs are finite and on the hyperboloid."""
    c, m, n, b = 1.0, 5, 3, 4
    manifold = Hyperboloid(dtype=jnp.float64)

    x_BA = jnp.stack([_onsheet(jax.random.PRNGKey(30 + i), _D, c, 2.0) for i in range(b)])
    y_BA = jnp.stack([_onsheet(jax.random.PRNGKey(40 + i), _D, c, 2.0) for i in range(b)])
    res_BA = jax.jit(jax.vmap(lambda a, d: lorentz_residual(a, d, 1.0, c)))(x_BA, y_BA)
    assert res_BA.shape == (b, _D + 1)
    assert bool(jnp.all(jnp.isfinite(res_BA)))
    assert bool(jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(res_BA, c)))

    pts_BMA = jnp.stack([_near_points(50 + i, _D, c, 2.0, m) for i in range(b)])
    w_BNM = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(60), (b, n, m), dtype=jnp.float64), axis=-1)
    mid_BNA = jax.jit(jax.vmap(lambda p, w: lorentz_midpoint(p, w, c)))(pts_BMA, w_BNM)
    assert mid_BNA.shape == (b, n, _D + 1)
    assert bool(jnp.all(jnp.isfinite(mid_BNA)))
    flat_KA = mid_BNA.reshape(-1, _D + 1)
    assert bool(jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(flat_KA, c)))


# ---------------------------------------------------------------------------------------
# 10. Inputs that float storage cannot keep on the sheet (boundary-lifted Poincare points)
# ---------------------------------------------------------------------------------------


def _boundary_lifted_point(d, c, dtype=jnp.float64):
    """Hyperboloid lift of a Poincare point sitting exactly on the ball boundary.

    ``Poincare.proj`` nudges the point just inside the boundary; the lift then puts it at
    ``x_0 ~ 5.5e11`` (c = 1, float64), a radius at which the sheet constraint cannot be *stored*:
    it needs ``eps * x_0^2 << 1/c``, and here ``eps64 * x_0^2 ~ 67``. The point's own computed
    Lorentz residual is O(1), so the identity-based Minkowski norm — which assumes exactly on-sheet
    inputs — has no valid input. This is precisely the ``test_horopca`` boundary-input case.
    """
    boundary_D = jnp.zeros(d, dtype=dtype).at[0].set(1.0 / jnp.sqrt(jnp.asarray(c, dtype=dtype)))
    inside_D = Poincare(dtype=dtype).proj(boundary_D, c)
    return poincare_to_hyperboloid(inside_D, c)


def _lorentz_sq(z_A):
    """``<z, z>_L`` of a single point (float64)."""
    return float(-(z_A[0] ** 2) + jnp.sum(z_A[1:] ** 2))


def _sheet_tol(z_A):
    """How closely ``<z,z>_L = -1/c`` can be *checked* for a point of this radius.

    The check itself subtracts two ``z_0^2``-sized squares, so it cannot resolve the constraint
    below ``eps64 * z_0^2`` no matter how exactly on-sheet ``z`` is; the factor 4 covers the
    accumulation over the spatial sum. Aggregating a boundary-lifted point (``x_0 ~ 5.5e11``) with
    ordinary ones puts the output far out too, so this floor is the binding one there.

    Worked example (``z_0 = 5.87e5``, the ``w_y = 0.5`` residual case):
    ``4 * 2.22e-16 * (5.87e5)^2 = 3.1e-4`` — and the measured deviation is 6.1e-5, i.e. 1 ulp of
    ``z_0^2`` (relative 1.8e-16). Below ``z_0 ~ 1.1e5`` the fixed ``1e-5`` is the binding bound.
    """
    eps64 = float(jnp.finfo(jnp.float64).eps)
    return max(1e-5, 4.0 * eps64 * float(z_A[0]) ** 2)


def _points_with_extreme(seed, d, c, m, extreme_at):
    """``M`` on-sheet points with the boundary-lifted point inserted at index ``extreme_at``."""
    keys = jax.random.split(jax.random.PRNGKey(seed), m - 1)
    pts = [_onsheet(k, d, c, 1.0 + 0.1 * i) for i, k in enumerate(keys)]
    pts.insert(extreme_at, _boundary_lifted_point(d, c))
    return jnp.stack(pts)  # (M, A)


@pytest.mark.parametrize("extreme_at", [0, 16])
def test_lorentz_midpoint_offsheet_extreme_input_stays_on_sheet(extreme_at):
    """A point too far out to be *stored* on the sheet must still yield an on-sheet midpoint.

    Regression for ``test_horopca.py::test_poincare_boundary_input_is_finite[float64]``: the
    identity-based Minkowski norm assumes exactly on-sheet inputs, so with this input its output
    landed off the sheet (measured residual 4.8e-3) and ``frechet_mean``'s Karcher loop then
    diverged to NaN. The output's time coordinate is now reconstructed from its spatial part, which
    restores the self-normalizing property the old naive form had. ``extreme_at = 0`` additionally
    covers the reference-point choice: with the old ``p = points_0`` every ``delta_m`` inherited the
    extreme radius.
    """
    c, m = 1.0, 32
    pts_MA = _points_with_extreme(80 + extreme_at, _D, c, m, extreme_at)
    w_NM = jnp.full((1, m), 1.0 / m, dtype=jnp.float64)

    out_NA = lorentz_midpoint(pts_MA, w_NM, c)

    assert bool(jnp.all(jnp.isfinite(out_NA))), f"extreme_at={extreme_at}: non-finite midpoint {out_NA}"
    assert float(out_NA[0, 0]) > 0.0, f"extreme_at={extreme_at}: midpoint not on the upper sheet"
    resid, tol = abs(_lorentz_sq(out_NA[0]) + 1.0 / c), _sheet_tol(out_NA[0])
    assert resid < tol, f"extreme_at={extreme_at}: midpoint off the sheet by {resid:.3e} (tol {tol:.1e})"


@pytest.mark.parametrize("w_y", [0.5, 1.0])
def test_lorentz_residual_offsheet_extreme_input_stays_on_sheet(w_y):
    """Same for the two-point aggregator: ``x`` unstorably far out, ``y`` ordinary."""
    c = 1.0
    x_A = _boundary_lifted_point(_D, c)
    y_A = _onsheet(jax.random.PRNGKey(90), _D, c, 1.5)

    out_A = lorentz_residual(x_A, y_A, w_y, c)

    assert bool(jnp.all(jnp.isfinite(out_A))), f"w_y={w_y}: non-finite output {out_A}"
    assert float(out_A[0]) > 0.0, f"w_y={w_y}: output not on the upper sheet"
    resid, tol = abs(_lorentz_sq(out_A) + 1.0 / c), _sheet_tol(out_A)
    assert resid < tol, f"w_y={w_y}: output off the sheet by {resid:.3e} (tol {tol:.1e})"


def test_frechet_mean_with_boundary_lifted_point_is_finite():
    """``frechet_mean`` initializes from ``lorentz_midpoint``; an off-sheet init NaNs the loop.

    Direct regression of the CI failure, independent of HoroPCA.
    """
    c, m = 1.0, 32
    manifold = Hyperboloid(dtype=jnp.float64)
    pts_MA = _points_with_extreme(81, _D, c, m, extreme_at=0)

    mean_A = frechet_mean(pts_MA, manifold, c)

    assert bool(jnp.all(jnp.isfinite(mean_A))), f"frechet_mean is not finite: {mean_A}"
    assert float(mean_A[0]) > 0.0, "frechet_mean not on the upper sheet"
    resid, tol = abs(_lorentz_sq(mean_A) + 1.0 / c), _sheet_tol(mean_A)
    assert resid < tol, f"frechet_mean off the sheet by {resid:.3e} (tol {tol:.1e})"
