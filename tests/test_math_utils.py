"""Tests for JAX math utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.utils.math_utils import (
    MIN_NORM,
    acosh,
    atanh,
    capped_exp,
    cosh,
    safe_hypot,
    safe_norm,
    safe_normalize,
    sinh,
    smooth_clamp,
    smooth_clamp_max,
    smooth_clamp_min,
    tanh,
)


def test_smooth_clamp_min():
    """Test smooth minimum clamping."""
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    min_val = 0.0
    result = smooth_clamp_min(x, min_val)

    # Values above min_val should be unchanged
    assert jnp.allclose(result[3:], x[3:], rtol=1e-6)  # [1.0, 2.0] unchanged

    # Values below min_val should be clamped and >= min_val
    assert jnp.all(result >= min_val)

    # Should be smooth (no discontinuities)
    assert result[0] < result[1] < result[2]  # monotonic


def test_smooth_clamp_max():
    """Test smooth maximum clamping."""
    x = jnp.array([-2.0, -1.0, 0.0, 1.5, 2.0])
    max_val = 1.0
    result = smooth_clamp_max(x, max_val)

    # Values well below max_val should be unchanged
    assert jnp.allclose(result[:3], x[:3], rtol=1e-6)  # [-2, -1, 0] unchanged

    # Values above max_val should be clamped and <= max_val
    assert jnp.all(result <= max_val + 1e-10)  # Small tolerance for numerical precision

    # Should be smooth (no discontinuities)
    assert result[3] > result[4] or jnp.allclose(result[3], result[4], rtol=1e-5)  # monotonic


def test_smooth_clamp():
    """Test smooth range clamping."""
    x = jnp.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    min_val, max_val = -1.5, 1.5
    result = smooth_clamp(x, min_val, max_val)

    # All values should be in range
    assert jnp.all(result >= min_val)
    assert jnp.all(result <= max_val)

    # Values in range should be approximately unchanged
    in_range_mask = (x >= min_val) & (x <= max_val)
    assert jnp.allclose(result[in_range_mask], x[in_range_mask], rtol=1e-5)


# ---------------------------------------------------------------------------------------------
# smooth_clamp gradient oracles (audit M1-07)
#
# The three tests above check forward VALUES only: that the output is in range, monotone, and
# unchanged well inside the interval. The whole point of the softplus clamp over a hard `jnp.clip`
# is the gradient in the saturated region — a hard clip passes every forward assertion in this file
# and gives gradient 0 exactly where the smooth version is supposed to keep pushing. The closed
# forms below come straight from the gate-free implementation (sp(u) = softplus(beta*u)/beta):
#
#     smooth_clamp_min(x) = min_value + sp(x - min_value)
#         d/dx = sigmoid(beta·(x - min_value))          everywhere — no gate, no shift
#     smooth_clamp_max(x) = max_value - sp(max_value - x)
#         d/dx = sigmoid(beta·(max_value - x))          everywhere
#     smooth_clamp(x)     = min_value + sp(x - min_value) - sp(x - max_value)
#         d/dx = sigmoid(beta·(x - min_value)) - sigmoid(beta·(x - max_value))
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [10.0, 50.0])
def test_smooth_clamp_min_gradient_is_the_softplus_sigmoid(beta: float):
    """d/dx smooth_clamp_min = sigmoid(beta·(x - min_value)) at every x, both sides of the bound.

    The ``beta`` axis is what pins the smoothing factor: dropping the multiplication (``arg = x -
    min_value``) leaves a valid, monotone, in-range clamp whose gradient no longer depends on ``beta``
    at all, so both parametrizations would collapse onto the same wrong curve.
    """
    min_val = 0.0
    xs = jnp.array([-1.0, -0.2, -0.01, 0.5, 2.0], dtype=jnp.float32)

    grads = jax.vmap(jax.grad(lambda v: smooth_clamp_min(v, min_val, smoothing_factor=beta)))(xs)

    expected = np.asarray(jax.nn.sigmoid(beta * (xs - min_val)))
    assert np.allclose(np.asarray(grads), expected, rtol=1e-5, atol=1e-6)
    # Just past the bound the gradient is attenuated but strictly positive — that is the whole
    # difference from a hard clip, which would return exactly 0 across the entire clamped region.
    assert 0.0 < float(grads[1]) < 1.0
    assert float(grads[0]) < float(grads[1]), "attenuation must deepen with distance past the bound"


@pytest.mark.parametrize("beta", [10.0, 50.0])
def test_smooth_clamp_max_gradient_is_the_softplus_sigmoid(beta: float):
    """d/dx smooth_clamp_max = sigmoid(beta·(max_value - x)) at every x, both sides of the bound."""
    max_val = 1.0
    xs = jnp.array([-1.0, 0.5, 0.99, 1.2, 3.0], dtype=jnp.float32)

    grads = jax.vmap(jax.grad(lambda v: smooth_clamp_max(v, max_val, smoothing_factor=beta)))(xs)

    expected = np.asarray(jax.nn.sigmoid(beta * (max_val - xs)))
    assert np.allclose(np.asarray(grads), expected, rtol=1e-5, atol=1e-6)
    assert 0.0 < float(grads[3]) < 1.0
    # Deep saturation: in float32 ``sigmoid(-100)`` underflows to exactly 0, so far past the bound
    # the smooth clamp degenerates to a hard clip. Pinned as monotone attenuation rather than
    # strict positivity, because the latter is false at beta=50 and x=3.
    assert 0.0 <= float(grads[-1]) <= float(grads[3])


@pytest.mark.parametrize("beta", [10.0, 50.0])
def test_smooth_clamp_gradient_is_the_sigmoid_difference(beta: float):
    """Two-sided ``smooth_clamp`` gradient = sigmoid(beta·(x - min)) - sigmoid(beta·(x - max)).

    ``smooth_clamp`` is the two-sided difference form, NOT a composition of the one-sided clamps
    (composing them overshoots narrow windows — pinned in test_smooth_clamp_vs_clip.py). Its
    derivative is therefore the plain difference of the two boundary sigmoids. Pinning it catches
    a dropped term or a re-simplification into the composition, which the range-only forward test
    cannot: the composition also keeps the output inside wide windows.
    """
    min_val, max_val = -1.5, 1.5
    xs = jnp.array([-4.0, -1.6, 0.0, 1.6, 4.0], dtype=jnp.float32)

    grads = jax.vmap(jax.grad(lambda v: smooth_clamp(v, min_val, max_val, smoothing_factor=beta)))(xs)

    x_np = np.asarray(xs, dtype=np.float64)
    expected = np.asarray(jax.nn.sigmoid(beta * (x_np - min_val))) - np.asarray(jax.nn.sigmoid(beta * (x_np - max_val)))

    assert np.allclose(np.asarray(grads), expected, rtol=1e-4, atol=1e-6)
    assert float(grads[2]) == pytest.approx(1.0, rel=1e-6), "no attenuation strictly inside the interval"
    # Symmetric attenuation: just outside each bound the gradient is alive but < 1; far outside it
    # decays further (to exactly 0 in float32 at beta=50 — see the max-side test).
    assert 0.0 < float(grads[1]) < 1.0 and 0.0 < float(grads[3]) < 1.0
    assert float(grads[0]) <= float(grads[1]) and float(grads[-1]) <= float(grads[3])


def test_cosh():
    """Test numerically stable cosh."""
    # Test normal values
    x_normal = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result_normal = cosh(x_normal)
    expected_normal = jnp.cosh(x_normal)
    assert jnp.allclose(result_normal, expected_normal)

    # Test extreme values that would overflow regular cosh
    x_extreme = jnp.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=jnp.float32)
    result_extreme = cosh(x_extreme)

    # Should not contain inf or nan
    assert jnp.all(jnp.isfinite(result_extreme))

    # Should be symmetric: cosh(-x) = cosh(x)
    assert jnp.allclose(result_extreme[0], result_extreme[4], rtol=1e-5)
    assert jnp.allclose(result_extreme[1], result_extreme[3], rtol=1e-5)


def test_sinh():
    """Test numerically stable sinh."""
    # Test normal values
    x_normal = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result_normal = sinh(x_normal)
    expected_normal = jnp.sinh(x_normal)
    assert jnp.allclose(result_normal, expected_normal)

    # Test extreme values
    x_extreme = jnp.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=jnp.float32)
    result_extreme = sinh(x_extreme)

    # Should not contain inf or nan
    assert jnp.all(jnp.isfinite(result_extreme))

    # Should be antisymmetric: sinh(-x) = -sinh(x)
    assert jnp.allclose(result_extreme[0], -result_extreme[4], rtol=1e-5)
    assert jnp.allclose(result_extreme[1], -result_extreme[3], rtol=1e-5)
    assert jnp.abs(result_extreme[2]) < 1e-10  # sinh(0) = 0


def test_sinh_cosh_f64_ulp_accuracy():
    """Regression test for XLA's CPU sinh/cosh float64 accuracy hole.

    XLA's CPU ``jnp.sinh``/``jnp.cosh`` are up to ~17 ulps off for ``|x|`` in ``[16, 512]`` and
    ~496 ulps for ``[512, 710]`` (the jump sits exactly at the power-of-two boundary 512); NumPy's
    ``sinh``/``cosh`` are <=1.8 ulps throughout. A naive exp-form sinh also fails via cancellation
    near 0. The library's expm1-form sinh / exp-form-with-custom_jvp cosh pass both regimes at
    <=8 ulps.
    """
    grids = [
        np.linspace(13.0, 700.0, 20001),  # XLA CPU sinh/cosh's bad regime
        np.linspace(-2.0, 2.0, 20001),  # cancellation regime for a naive exp-form sinh
    ]
    for fn, ref_fn in [(sinh, np.sinh), (cosh, np.cosh)]:
        for grid in grids:
            out = np.asarray(fn(jnp.asarray(grid, dtype=jnp.float64)))
            expected = ref_fn(grid)
            # Skip the exact zero (sinh(0) == 0): the relative-ulp division is undefined there.
            mask = expected != 0.0
            ulp = np.abs(out[mask] - expected[mask]) / np.spacing(np.abs(expected[mask]))
            assert ulp.max() <= 8.0


def test_sinh_cosh_odd_even_symmetry():
    """sinh(-x) == -sinh(x) and cosh(-x) == cosh(x) bitwise, for both dtypes."""
    for dt in (jnp.float32, jnp.float64):
        g = jnp.linspace(0.01, 80.0, 5001, dtype=dt)
        assert np.array_equal(np.asarray(sinh(-g)), np.asarray(-sinh(g)))
        assert np.array_equal(np.asarray(cosh(-g)), np.asarray(cosh(g)))


def test_sinh_cosh_gradients():
    """sinh'(0) == 1.0 and cosh'(0) == 0.0 exactly, for both dtypes.

    Also catches a regression of ``cosh``'s ``custom_jvp`` to the naive cancelling exp-form
    gradient ``0.5*(exp(x) - exp(-x))``: measured, that form errs by ~1.7e-4 relative at
    ``x = 1e-4`` in float32 while the custom_jvp errs by ~4.6e-8, so the 1e-5 tolerance sits
    between them with >1 order of margin on each side.
    """
    for dt in (jnp.float32, jnp.float64):
        assert float(jax.grad(sinh)(jnp.asarray(0.0, dtype=dt))) == 1.0
        assert float(jax.grad(cosh)(jnp.asarray(0.0, dtype=dt))) == 0.0

    x = jnp.asarray(1e-4, dtype=jnp.float32)
    g = float(jax.grad(cosh)(x))
    expected = float(np.sinh(np.float64(1e-4)))
    assert g == pytest.approx(expected, rel=1e-5)


def test_capped_exp():
    """capped_exp: bitwise exp below the cap, finite (never inf) above it, both dtypes.

    The guard exists for exp of unconstrained trainable log-scale params: a runaway param
    must saturate at a huge-but-finite value instead of overflowing to inf -> NaN.
    """
    for dt in (jnp.float32, jnp.float64):
        # Value- and gradient-identity to jnp.exp below the cap.
        x = jnp.asarray([-40.0, -1.0, 0.0, 1.0, 20.0], dtype=dt)
        assert np.array_equal(np.asarray(capped_exp(x)), np.asarray(jnp.exp(x)))
        g = float(jax.grad(capped_exp)(jnp.asarray(1.0, dtype=dt)))
        assert g == float(jnp.exp(jnp.asarray(1.0, dtype=dt)))

        # Above the cap: jnp.exp overflows to inf, capped_exp stays finite with zero gradient.
        big = jnp.asarray(1e30, dtype=dt)
        assert not np.isfinite(float(jnp.exp(big)))  # the guard is load-bearing
        assert np.isfinite(float(capped_exp(big)))
        assert float(jax.grad(capped_exp)(big)) == 0.0


def test_acosh():
    """Test numerically stable acosh."""
    # Values away from the domain boundary are exact
    x_valid = jnp.array([1.5, 2.0, 5.0, 10.0])
    result_valid = acosh(x_valid)
    expected_valid = jnp.acosh(x_valid)
    assert jnp.allclose(result_valid, expected_valid)

    # Test invalid domain values (should be clamped)
    x_invalid = jnp.array([0.5, 0.9, 1.0, 1.1, 2.0])
    result_invalid = acosh(x_invalid)

    # Should not contain nan
    assert jnp.all(jnp.isfinite(result_invalid))

    # Values <= 1 are clamped to 1 + 10*machine_eps, giving a forward value of
    # sqrt(2 * 10 * eps) instead of exactly 0 — the deliberate margin that
    # bounds acosh' (acosh'(1) = inf would otherwise NaN every gradient that
    # lands exactly on the boundary, e.g. dist(x, x)).
    for dtype in [jnp.float32, jnp.float64]:
        margin = 10.0 * float(jnp.finfo(dtype).eps)
        forward_error = float(jnp.sqrt(2.0 * margin))
        res = acosh(jnp.array([0.5, 1.0], dtype=dtype))
        assert jnp.allclose(res, 0.0, atol=1.01 * forward_error)


def test_acosh_gradient_at_boundary():
    """Gradient at and below the domain boundary is finite (regression: a hard
    clip at exactly 1.0 let x = 1.0 reach acosh'(1) = inf)."""
    for dtype in [jnp.float32, jnp.float64]:
        for x in [0.5, 1.0, 1.0 + 1e-9]:
            g = jax.grad(lambda a: acosh(a))(jnp.asarray(x, dtype=dtype))
            assert jnp.isfinite(g)


def test_atanh():
    """Test numerically stable atanh."""
    # Test valid domain values
    x_valid = jnp.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    result_valid = atanh(x_valid)
    expected_valid = jnp.atanh(x_valid)
    assert jnp.allclose(result_valid, expected_valid)

    # Boundary values are clamped to ±(1 - 10·eps), so the forward value is the EXACT arctanh of
    # that clamp point. The margin is the whole point of the guard, so it is what gets pinned:
    # widening it to 1e5·eps (a real accuracy regression near the Poincaré ball boundary — the f32
    # input error becomes 1.2e-2) drops the f32 value from 7.166 to ~2.6, whereas the previous
    # `abs(atanh(-0.9999)) < 1e10` bound was ~2e9x too loose to notice any of it.
    for dt in (jnp.float32, jnp.float64):
        margin = 10.0 * float(jnp.finfo(dt).eps)
        expected = float(np.arctanh(1.0 - margin))  # f32 -> 7.166473, f64 -> 17.217108
        out = np.asarray(atanh(jnp.array([-1.1, -1.0, 1.0, 1.1], dtype=dt)), dtype=np.float64)
        assert np.allclose(out, np.array([-expected, -expected, expected, expected]), rtol=1e-5)


def test_atanh_gradient_at_boundary():
    """The ±(1 - 10·eps) clamp bounds ``atanh'`` — pin the exact gradient on both sides of it.

    Outside the band the clip's VJP is zero, so the gradient is exactly 0.0; the previous
    ``isfinite`` assertion could not tell that apart from any other finite value. Just inside the
    band the clip is a gradient IDENTITY, so the gradient is exactly the unclamped 1/(1 - x²).
    Together the two pin where the clamp point sits: at a 1e5·eps margin the in-band point below
    would be clipped instead and its gradient would collapse to 0.
    """
    for dt in (jnp.float32, jnp.float64):
        margin = 10.0 * float(jnp.finfo(dt).eps)
        # At/beyond ±1, and anywhere inside the margin: saturated, gradient exactly 0.
        for x in (-1.1, -1.0, 1.0, 1.1, 1.0 - 0.5 * margin):
            g = jax.grad(atanh)(jnp.asarray(x, dtype=dt))
            assert float(g) == 0.0, f"clip VJP not saturated at x={x} ({dt.__name__})"
        # Just inside the clamp point: the guard is a gradient identity.
        #   f32 -> 2.097155e5, f64 -> 1.125900e14
        x_in = 1.0 - 2.0 * margin
        g_in = float(jax.grad(atanh)(jnp.asarray(x_in, dtype=dt)))
        assert g_in == pytest.approx(1.0 / (1.0 - x_in**2), rel=1e-5)


def test_atanh_f64_ulp_accuracy():
    """Regression test for the XLA float64 ``log1p`` accuracy hole.

    XLA evaluates ``jnp.atanh`` as ``0.5*(log1p(x) - log1p(-x))``. XLA's CPU float64 ``log1p`` is
    up to ~129 ulps off for arguments in ``[-0.53, -0.28]``, so the builtin ``jnp.atanh`` fails
    this test at ~129 ulps on the positive window ``[0.28, 0.53]``. A plain (non-odd) single-log1p
    rewrite ``0.5*log1p(2x/(1-x))`` also fails this test, at ~129 ulps, on the negative window
    ``[-0.40, -0.12]`` (its inner argument re-enters the bad ``log1p`` window there). The library's
    odd/where single-log1p form keeps ``log1p``'s argument non-negative for either sign and passes
    both windows at <=8 ulps.
    """
    reference = np.arctanh  # accurate to <1 ulp in these windows
    grids = [
        np.linspace(0.28, 0.53, 20001),  # builtin jnp.atanh's bad window
        np.linspace(-0.40, -0.12, 20001),  # window where the naive non-odd rewrite fails
    ]
    for grid in grids:
        out = np.asarray(atanh(jnp.asarray(grid, dtype=jnp.float64)))
        expected = reference(grid)
        ulp = np.abs(out - expected) / np.spacing(np.abs(expected))
        assert ulp.max() <= 8.0


def test_atanh_odd_symmetry():
    """atanh(-x) == -atanh(x) bitwise, for both dtypes.

    Pins the odd/where spelling: a plain (non-odd) single-log1p rewrite would not generally
    satisfy this to bit precision.
    """
    for dt in (jnp.float32, jnp.float64):
        g = jnp.linspace(0.001, 0.999, 5001, dtype=dt)
        assert np.array_equal(np.asarray(atanh(-g)), np.asarray(-atanh(g)))


def test_atanh_gradient_at_zero():
    """atanh'(0) == 1.0 exactly, for both dtypes.

    Catches a ``jnp.sign(x)*...`` mis-spelling of the odd rewrite, whose product-rule gradient at
    x == 0 collapses to 0 (analytic atanh'(0) = 1).
    """
    for dt in (jnp.float32, jnp.float64):
        g = jax.grad(atanh)(jnp.asarray(0.0, dtype=dt))
        assert float(g) == 1.0


def test_tanh():
    """Output stays strictly inside (-1, 1) even where float32 jnp.tanh saturates to exactly 1.0."""
    for dtype in [jnp.float32, jnp.float64]:
        # Saturated tail: |tanh| must stay < 1 so a downstream atanh cannot reach its pole.
        big = jnp.array([8.0, 12.0, 50.0, -8.0, -50.0], dtype=dtype)
        out = tanh(big)
        assert jnp.all(jnp.abs(out) < 1.0)
        assert jnp.all(jnp.isfinite(atanh(out)))  # atanh(tanh(x)) stays finite
        # Non-saturated regime: value-identity to jnp.tanh.
        mid = jnp.array([-1.0, -0.5, 0.0, 0.3, 1.0], dtype=dtype)
        assert jnp.allclose(tanh(mid), jnp.tanh(mid), atol=1e-6)


# ---------------------------------------------------------------------------------------------
# safe_norm / safe_hypot / safe_normalize
#
# These replace the library's ``sqrt(sum(v**2) + MIN_NORM**2)`` idiom wherever the *magnitude*
# range matters rather than just the gradient at zero. Two failure modes are pinned below, one at
# each end of the exponent range, plus the exact VJP:
#
#   overflow  — ``sum(v**2)`` leaves float32 at |v| ~ 1.8e19, while the norm itself is
#               representable up to 3.4e38 (the hyperboloid spatial radius passes 1.8e19 at
#               geodesic radius 44).
#   underflow — the ``+ MIN_NORM**2`` floor is 1e-30, so any true norm below 1e-15 is replaced by
#               1e-15. A float32 angular chord of 1e-34 is a legitimate input, not noise.
#   VJP       — with the scale under ``stop_gradient`` the derivative is exactly ``v/‖v‖``; at
#               v = 0 it is exactly 0, which requires sanitizing the sqrt ARGUMENT (a where on the
#               output alone still builds ``0 * sqrt'(0) = 0 * inf = NaN`` inside the VJP).
# ---------------------------------------------------------------------------------------------


def test_safe_norm_matches_numpy_across_300_orders_of_magnitude():
    """float64: correct norms for magnitudes 1e-300 .. 1e300.

    The reference is ``‖direction‖ · 10**exponent`` rather than ``np.linalg.norm(v)``, because
    NumPy's own norm squares first and therefore overflows past ~1e154 and flushes below ~1e-162 —
    the exact failure this function exists to avoid, on the exact same inputs.
    """
    directions = np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0], [-2.0, 0.5, 7.0]])
    unit_norms = np.linalg.norm(directions, axis=-1)
    for exponent in range(-300, 301, 10):
        v = directions * (10.0**exponent)
        out = np.asarray(safe_norm(jnp.asarray(v, dtype=jnp.float64)))
        expected = unit_norms * (10.0**exponent)
        assert np.allclose(out, expected, rtol=1e-15, atol=0.0), f"exponent {exponent}"


def test_safe_norm_does_not_overflow_or_underflow_in_float32():
    """The two regimes the ``sqrt(sum(v**2) + MIN_NORM**2)`` idiom gets wrong, in float32.

    ``1e30`` and ``1e-34`` are both perfectly representable float32 values (max 3.4e38, smallest
    normal 1.2e-38); only their *squares* are not (1e60 overflows, 1e-68 flushes to zero).
    """
    big = jnp.asarray([1e30, 0.0, 0.0], dtype=jnp.float32)
    assert float(safe_norm(big)) == pytest.approx(1e30, rel=1e-6)
    assert np.isfinite(float(safe_norm(big)))
    # The naive idiom overflows here — pin that the guard is load-bearing.
    assert not np.isfinite(float(jnp.sqrt(jnp.sum(big**2) + MIN_NORM**2)))

    tiny = jnp.asarray([3e-34, 4e-34, 0.0], dtype=jnp.float32)
    assert float(safe_norm(tiny)) == pytest.approx(5e-34, rel=1e-5)
    # The naive idiom returns the MIN_NORM floor, 19 orders of magnitude too large.
    assert float(jnp.sqrt(jnp.sum(tiny**2) + MIN_NORM**2)) == pytest.approx(1e-15, rel=1e-3)


def test_safe_norm_is_exactly_zero_with_an_exactly_zero_gradient_at_the_origin():
    """Forward 0 and VJP 0 at v = 0, both dtypes — the double-``where``'s whole job."""
    for dt in (jnp.float32, jnp.float64):
        zero = jnp.zeros(4, dtype=dt)
        assert float(safe_norm(zero)) == 0.0
        g = np.asarray(jax.grad(safe_norm)(zero))
        assert np.all(np.isfinite(g)) and np.all(g == 0.0)


def test_safe_norm_gradient_is_the_unit_vector():
    """grad(safe_norm)(v) == v/‖v‖ exactly, at ordinary and extreme magnitudes.

    ``stop_gradient`` on the max-scale is what makes this hold: differentiating through the scale
    would add a term proportional to the sign of the largest component.
    """
    for v_np in ([3.0, -4.0, 12.0], [1e-30, 2e-30, -2e-30], [1e150, -1e150, 0.0]):
        v = jnp.asarray(v_np, dtype=jnp.float64)
        g = np.asarray(jax.grad(safe_norm)(v))
        expected = np.asarray(v_np) / np.linalg.norm(v_np)
        assert np.allclose(g, expected, rtol=1e-14, atol=0.0)
        assert float(np.linalg.norm(g)) == pytest.approx(1.0, rel=1e-14)


def test_safe_norm_batches_over_leading_axes():
    """The norm is taken over the last axis only, for any number of leading axes."""
    v = jnp.asarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4))
    out = np.asarray(safe_norm(v))
    assert out.shape == (2, 3)
    assert np.allclose(out, np.linalg.norm(np.asarray(v), axis=-1), rtol=1e-15)


def test_safe_norm_passes_infinity_through():
    """A vector holding an inf returns inf, not NaN (out-of-range points stay visibly degenerate)."""
    v = jnp.asarray([np.inf, 1.0, 0.0], dtype=jnp.float64)
    assert np.isinf(float(safe_norm(v)))


def test_safe_hypot_matches_numpy_and_survives_extreme_legs():
    """Value oracle vs ``np.hypot``, plus the float32 leg that squares out of range."""
    legs = [(3.0, 4.0), (1e-20, 1e-25), (0.0, 2.5), (-7.0, 0.0), (1e150, 1e150)]
    for p, q in legs:
        out = float(safe_hypot(jnp.asarray(p, dtype=jnp.float64), jnp.asarray(q, dtype=jnp.float64)))
        assert out == pytest.approx(float(np.hypot(p, q)), rel=1e-14)

    p32 = jnp.asarray(1e30, dtype=jnp.float32)
    q32 = jnp.asarray(1.0, dtype=jnp.float32)
    assert np.isfinite(float(safe_hypot(p32, q32)))
    assert float(safe_hypot(p32, q32)) == pytest.approx(1e30, rel=1e-6)
    assert not np.isfinite(float(jnp.sqrt(p32**2 + q32**2)))  # the guard is load-bearing


def test_safe_hypot_is_exactly_zero_with_zero_gradients_at_the_origin():
    """0 forward and 0 in both partial derivatives at p == q == 0, both dtypes."""
    for dt in (jnp.float32, jnp.float64):
        z = jnp.asarray(0.0, dtype=dt)
        assert float(safe_hypot(z, z)) == 0.0
        gp, gq = jax.grad(safe_hypot, argnums=(0, 1))(z, z)
        assert float(gp) == 0.0 and float(gq) == 0.0


def test_safe_hypot_gradient_matches_the_closed_form():
    """d/dp hypot = p/hypot, d/dq hypot = q/hypot — including at a leg that would overflow."""
    p, q = 1e150, -3e149
    gp, gq = jax.grad(safe_hypot, argnums=(0, 1))(jnp.asarray(p), jnp.asarray(q))
    h = float(np.hypot(p, q))
    assert float(gp) == pytest.approx(p / h, rel=1e-14)
    assert float(gq) == pytest.approx(q / h, rel=1e-14)


def test_safe_normalize_returns_unit_vectors_and_an_exact_zero_at_the_origin():
    """Unit output at every magnitude; the exact zero vector (finite gradient) at v = 0."""
    # The reference direction is built at unit scale and then compared: ``np.linalg.norm`` itself
    # overflows on the 1e200 row.
    for direction, scale in (([3.0, -4.0, 0.0], 1.0), ([1.0, 0.0, 0.0], 1e-33), ([1.0, 1.0, 0.0], 1e200)):
        v_np = np.asarray(direction) * scale
        out = np.asarray(safe_normalize(jnp.asarray(v_np, dtype=jnp.float64)))
        assert float(np.linalg.norm(out)) == pytest.approx(1.0, rel=1e-14)
        assert np.allclose(out, np.asarray(direction) / np.linalg.norm(direction), rtol=1e-14)

    for dt in (jnp.float32, jnp.float64):
        zero = jnp.zeros(3, dtype=dt)
        out = np.asarray(safe_normalize(zero))
        assert np.all(out == 0.0)
        g = np.asarray(jax.jacobian(safe_normalize)(zero))
        assert np.all(np.isfinite(g))


def test_safe_normalize_has_no_min_norm_floor_on_the_denominator():
    """A direction of length 1e-19 normalizes to a UNIT vector, not to a 1e-4-scaled one.

    A ``maximum(‖v‖, MIN_NORM)`` floor (the library's older idiom) would divide by 1e-15 here and
    return a vector of length 1e-4. The angular leg of the hyperboloid geodesic frame is built
    from exactly such a vector, so the floor would silently shrink the direction.
    """
    v = jnp.asarray([1e-19, 0.0, 0.0], dtype=jnp.float64)
    out = np.asarray(safe_normalize(v))
    assert out[0] == pytest.approx(1.0, rel=1e-14)


def test_safe_norm_family_preserves_dtype():
    """float32 in, float32 out — no silent promotion under global x64."""
    for dt in (jnp.float32, jnp.float64):
        v = jnp.asarray([0.3, -0.4, 0.5], dtype=dt)
        assert safe_norm(v).dtype == dt
        assert safe_normalize(v).dtype == dt
        assert safe_hypot(v[0], v[1]).dtype == dt


def test_dtype_consistency():
    """Test that functions preserve dtype."""
    for dtype in [jnp.float32, jnp.float64]:
        x = jnp.array([0.5, 1.0, 1.5], dtype=dtype)

        # Test all functions preserve dtype
        assert smooth_clamp(x, 0.0, 2.0).dtype == dtype
        assert cosh(x).dtype == dtype
        assert sinh(x).dtype == dtype
        assert acosh(x).dtype == dtype
        assert atanh(x * 0.5).dtype == dtype  # Scale to valid domain
