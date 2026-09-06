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
    safe_hypot_norm,
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
# tanh / atanh float32 ulp accuracy
#
# ``tanh`` and ``atanh`` are the two primitives every Poincare map is built on, and in float32
# XLA's own kernels are 4 ulp / 3 ulp rational approximations (torch's are ~0.5 ulp). The wrappers
# close part of that gap with two rewrites, each gated on a measured seam:
#
#   * ``tanh(x) = expm1(2|x|)/(expm1(2|x|) + 2)`` for ``|x| >= 1/8`` (float32) / ``>= 1/2``
#     (float64); below the seam, ``x - x^3/3 + 2x^5/15 - 17x^7/315`` in float32 and XLA's
#     ``tanh`` in float64;
#   * ``atanh(x) = x + x^3/3 + ... + x^9/9`` for ``|x| < 1/8`` in float32 only, the single-log1p
#     form above it and on the whole float64 domain.
#
# The bounds below are the CPU-safe ones — they must hold on the CI CPU runner AND on GPU — and
# each docstring records the separately measured GPU and CPU numbers behind its bound. Ulp
# distance is exact, taken from the bit patterns, so a bound of "2 ulp" means "at most two
# float32 values away from the correctly rounded result".
# ---------------------------------------------------------------------------------------------

_ULP_SEED = 0
_ULP_N = 20_000


def _ordinal(a_N: np.ndarray) -> np.ndarray:
    """Monotone integer total order over float32: adjacent floats differ by exactly 1."""
    bits_N = np.ascontiguousarray(a_N).view(np.int32).astype(np.int64)
    out_N = bits_N.copy()
    neg_N = bits_N < 0
    # Masked assignment, not np.where: ``offset - bits`` would overflow int64 on the positive
    # bit patterns np.where evaluates and then discards.
    out_N[neg_N] = np.int64(-(2**31)) - bits_N[neg_N]
    return out_N


def _max_ulp_f32(got_N: np.ndarray, ref_N: np.ndarray) -> float:
    """Exact maximum bit-pattern ulp distance between two float32 arrays."""
    return float(np.abs(_ordinal(got_N) - _ordinal(ref_N)).max())


def _log_uniform_f32(lo: float, hi: float, seed: int = _ULP_SEED, n: int = _ULP_N) -> np.ndarray:
    """``n`` seeded log-uniform float32 samples on ``[lo, hi]`` — the probe script's sampling."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n)).astype(np.float32)


def _correctly_rounded_f32(np_fn, x_N: np.ndarray) -> np.ndarray:
    """Reference: evaluate in float64 on the stored float32 inputs, round back to float32.

    Double rounding can differ from a single correct rounding only for inputs whose exact result
    sits within 2^-29 relative of a float32 tie — far below the 1-4 ulp effects measured here.
    """
    return np_fn(x_N.astype(np.float64)).astype(np.float32)


def test_tanh_f32_ulp_accuracy_near_zero():
    """Below the 1/8 seam the truncated odd series must be correctly rounded to <=1 ulp.

    Measured on 20k log-uniform float32 inputs in [1e-4, 0.125] (max / mean ulp), at both signs:
    XLA's own kernel — which is what the wrapper used here before the series landed — is 4 / 0.87
    on XLA GPU and 4 / 0.96 on XLA CPU; the series
    ``x - x^3/3 + 2x^5/15 - 17x^7/315`` is 1 / 0.16 on both. Its own truncation is negligible on
    this range: the first dropped term is ``62 x^9/2835``, 1.6e-10 absolute (1.3e-9 relative) at
    x = 1/8 against a float32 eps of 1.2e-7, so what this bound pins is that the rewrite is in
    place at all.

    The series is also what keeps the ``expm1`` form from simply being pushed below 1/8: measured
    on this range ``expm1(2x)/(expm1(2x) + 2)`` is 3 ulp on GPU but **6** on CPU (mean 0.96 ->
    1.08), a regression on the backend CI runs on.
    """
    x_N = _log_uniform_f32(1e-4, 0.125)
    for signed_N in (x_N, -x_N):
        got_N = np.asarray(tanh(jnp.asarray(signed_N)), dtype=np.float32)
        assert _max_ulp_f32(got_N, _correctly_rounded_f32(np.tanh, signed_N)) <= 1.0


def test_tanh_f32_ulp_accuracy_in_the_tail():
    """Above the seam the ``expm1`` rewrite must hold <=2 ulp where XLA's ``tanh`` gives 4.

    Measured on 20k log-uniform float32 inputs in [0.9, 7] (max / mean ulp): ``jnp.tanh`` is
    4 / 0.82 on XLA GPU and 4 / 0.72 on XLA CPU; the wrapper is 2 / 0.40 on GPU and 1 / 0.04
    on CPU.

    The upper end is 7, not 8, deliberately: the wrapper clips its OUTPUT to +-(1 - 10*eps) so a
    downstream ``atanh`` can never reach the pole, and in float32 that clip starts binding at
    x = atanh(1 - 10*eps) = 7.168. Past that point the deviation from ``tanh`` is the guard (~16
    ulp at x = 8), not the kernel, so measuring the kernel means staying below it.
    """
    x_N = _log_uniform_f32(0.9, 7.0)
    got_N = np.asarray(tanh(jnp.asarray(x_N)), dtype=np.float32)
    assert _max_ulp_f32(got_N, _correctly_rounded_f32(np.tanh, x_N)) <= 2.0


def test_atanh_f32_ulp_accuracy_near_zero():
    """Below the 1/8 seam the truncated odd series must be correctly rounded to <=1 ulp.

    Measured on 20k log-uniform float32 inputs in [1e-4, 0.125] (max / mean ulp): the single-log1p
    form is 3 / 0.48 on XLA GPU and 2 / 0.28 on XLA CPU (``jnp.arctanh`` itself: 3 / 0.48 GPU,
    1 / 0.24 CPU); the series is 1 / 0.33 on both. The series' own truncation is negligible here —
    the first dropped term is x^11/11, 1.1e-11 relative at x = 1/8 against a float32 eps of 1.2e-7
    — so what this bound really pins is that the rewrite is in place at all.
    """
    x_N = _log_uniform_f32(1e-4, 0.125)
    got_N = np.asarray(atanh(jnp.asarray(x_N)), dtype=np.float32)
    assert _max_ulp_f32(got_N, _correctly_rounded_f32(np.arctanh, x_N)) <= 1.0


def test_atanh_f32_ulp_accuracy_above_the_seam():
    """Above 1/8 the single-log1p form carries the whole range and must stay within 3 ulp.

    Measured on 20k log-uniform float32 inputs in [0.125, 0.9] (max / mean ulp): 2 / 0.33 on XLA
    GPU, 3 / 0.42 on XLA CPU. The bound is the CPU one so the test holds on both backends.
    """
    x_N = _log_uniform_f32(0.125, 0.9)
    got_N = np.asarray(atanh(jnp.asarray(x_N)), dtype=np.float32)
    assert _max_ulp_f32(got_N, _correctly_rounded_f32(np.arctanh, x_N)) <= 3.0


def test_tanh_and_atanh_seams_are_continuous():
    """Both branches of each seam agree to <=2 ulp when evaluated at the seam +-1 ulp.

    A ``jnp.where`` seam is only harmless if the two kernels it switches between are
    indistinguishable there. Measured branch-to-branch distance at the three points:
    ``tanh``'s float32 seam (1/8, series vs ``expm1`` form) is 1/1/1 ulp on XLA CPU and 1/0/1 on
    XLA GPU, and its float64 seam (1/2, ``jnp.tanh`` vs ``expm1`` form) is 1/0/1 on CPU and 0/1/1
    on GPU; ``atanh``'s float32 seam (1/8) is 0/0/0 on both. A 1 ulp step is smaller than either
    branch's own error — what has to hold on top of it is that the wrapper stays monotone there,
    which is asserted separately below. (Before the float32 ``tanh`` series landed, its below-seam
    branch was XLA's own ``tanh`` and this seam measured 0/1/0 on CPU but 2/1/2 on GPU.)
    """

    def tanh_expm1_branch(x):
        t = jnp.expm1(2.0 * jnp.abs(x))
        return t / (t + 2.0)

    def tanh_series_branch(x):
        a = jnp.abs(x)
        x2 = a * a
        return a * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 - x2 * (17.0 / 315.0))))

    def atanh_log1p_branch(x):
        a = jnp.abs(x)
        return 0.5 * jnp.log1p(2.0 * a / (1.0 - a))

    def atanh_series_branch(x):
        x2 = x * x
        return x * (1.0 + x2 * (1.0 / 3.0 + x2 * (1.0 / 5.0 + x2 * (1.0 / 7.0 + x2 / 9.0))))

    for dt, np_dt in ((jnp.float32, np.float32), (jnp.float64, np.float64)):
        # Below the seam float32 uses the odd series and float64 uses XLA's own tanh.
        below_tanh = tanh_series_branch if dt is jnp.float32 else jnp.tanh
        seams = [(0.125 if dt is jnp.float32 else 0.5, below_tanh, tanh_expm1_branch, tanh)]
        if dt is jnp.float32:  # float64 atanh has no seam: the log1p form covers the whole domain
            seams.append((0.125, atanh_series_branch, atanh_log1p_branch, atanh))
        for seam, below_fn, above_fn, wrapper in seams:
            s = np_dt(seam)
            pts = np.array([np.nextafter(s, np_dt(0)), s, np.nextafter(s, np_dt(2))], dtype=np_dt)
            x = jnp.asarray(pts, dtype=dt)
            lo = np.asarray(below_fn(x), dtype=np_dt)
            hi = np.asarray(above_fn(x), dtype=np_dt)
            # Gap in ulps *of the working dtype* — np.spacing must therefore be taken before the
            # float64 widening, or a float32 seam would be measured against float64's grid.
            spacing = np.spacing(np.abs(lo)).astype(np.float64)
            gap = np.abs(lo.astype(np.float64) - hi.astype(np.float64)) / spacing
            assert gap.max() <= 2.0, f"{np_dt.__name__} seam at {seam}: {gap} ulp between branches"
            # The wrapper's own output must still be non-decreasing across the seam: a 2 ulp
            # branch gap that reordered two neighbouring inputs would be a real discontinuity.
            out = np.asarray(wrapper(x), dtype=np_dt)
            assert np.all(np.diff(out.astype(np.float64)) >= 0.0), f"{np_dt.__name__} seam at {seam}: {out}"


def test_tanh_odd_symmetry():
    """tanh(-x) == -tanh(x) bitwise, for both dtypes.

    ``expm1(2x)`` and ``expm1(-2x)`` are not negatives of each other, so the rewrite is only odd
    to the last bit because it is evaluated on ``|x|`` with the sign restored by ``jnp.where``.
    Spelling it on ``x`` directly breaks this (measured: ~5000 of 20001 grid points differ).
    """
    for dt in (jnp.float32, jnp.float64):
        g = jnp.asarray(np.concatenate([np.linspace(1e-6, 60.0, 20001), [0.125, 0.5, 8.0]]), dtype=dt)
        assert np.array_equal(np.asarray(tanh(-g)), np.asarray(-tanh(g)))


def test_tanh_gradient_is_finite_and_matches_sech2():
    """``tanh``'s gradient is finite on the whole real line and equals ``1 - tanh(x)**2``.

    Both ``jnp.where`` branches are evaluated *and differentiated*, so the ``expm1`` branch has to
    stay finite even where it is not selected: ``expm1(2x)`` overflows to inf past x = 44 (f32),
    and ``inf/inf = NaN`` would poison the selected branch's cotangent. The input clip to
    +-0.5*log(2/eps) is what prevents that, and x = +-100 below is past the overflow point.

    Beyond the guards the gradient is 0 by design (the clips' VJP), so the comparison against
    ``1 - tanh**2`` runs only where neither guard binds. That region is read off the *computed*
    output rather than from ``atanh(1 - 10*eps)``: in float32 the grid near 1 is 5.96e-8 coarse,
    so the output clip already binds at x ~ 7.13, well inside the analytic 7.168.
    """
    for dt, atol in ((jnp.float32, 1e-6), (jnp.float64, 1e-14)):
        xs_N = np.concatenate([np.linspace(-50.0, 50.0, 2001), [0.0, 0.125, -0.125, 0.5, -0.5, 100.0, -100.0]]).astype(
            np.float64
        )
        x = jnp.asarray(xs_N, dtype=dt)
        grad_N = np.asarray(jax.vmap(jax.grad(tanh))(x), dtype=np.float64)
        jac_N = np.asarray(jax.vmap(jax.jacfwd(tanh))(x), dtype=np.float64)
        assert np.all(np.isfinite(grad_N)), f"non-finite jax.grad ({dt.__name__})"
        assert np.all(np.isfinite(jac_N)), f"non-finite jax.jacfwd ({dt.__name__})"

        max_out = 1.0 - 10.0 * float(jnp.finfo(dt).eps)
        unsaturated = np.asarray(jnp.abs(tanh(x)), dtype=np.float64) < max_out
        expected_N = 1.0 - np.tanh(xs_N) ** 2
        assert np.abs(grad_N[unsaturated] - expected_N[unsaturated]).max() <= atol
        assert np.abs(jac_N[unsaturated] - expected_N[unsaturated]).max() <= atol
        # The seam itself and the origin, pinned exactly.
        assert float(jax.grad(tanh)(jnp.asarray(0.0, dtype=dt))) == 1.0


def test_atanh_gradient_is_finite_and_matches_the_closed_form():
    """``atanh``'s gradient is finite on [-0.999, 0.999] and equals ``1/(1 - x**2)``.

    Covers the seam at +-1/8 and the origin, where the series branch takes over: both branches are
    differentiated by ``jnp.where``, and the log1p branch's ``1 - |x|`` denominator is what the
    domain clip keeps bounded away from zero (>= 10*eps).

    The tolerance is the conditioning of the problem, not a fixed number: forming ``1 - x**2``
    near ``|x| = 1`` cancels, so no float implementation can beat a relative error of
    ``eps/(1 - x**2)`` (5.96e-5 at |x| = 0.999 in float32). Measured, both dtypes come in at
    1.4x that bound at worst and ~3 eps on |x| <= 0.9; the factor 8 leaves ~6x margin at the
    worst point without letting a genuinely wrong derivative through.
    """
    for dt in (jnp.float32, jnp.float64):
        eps = float(jnp.finfo(dt).eps)
        xs_N = np.concatenate([np.linspace(-0.999, 0.999, 2001), [0.0, 0.125, -0.125]]).astype(np.float64)
        x = jnp.asarray(xs_N, dtype=dt)
        grad_N = np.asarray(jax.vmap(jax.grad(atanh))(x), dtype=np.float64)
        jac_N = np.asarray(jax.vmap(jax.jacfwd(atanh))(x), dtype=np.float64)
        assert np.all(np.isfinite(grad_N)), f"non-finite jax.grad ({dt.__name__})"
        assert np.all(np.isfinite(jac_N)), f"non-finite jax.jacfwd ({dt.__name__})"
        expected_N = 1.0 / (1.0 - xs_N**2)
        tol_N = 8.0 * eps * expected_N  # = 8*eps/(1 - x**2), the cancellation floor
        assert np.all(np.abs(grad_N - expected_N) / expected_N <= tol_N)
        assert np.all(np.abs(jac_N - expected_N) / expected_N <= tol_N)


def test_tanh_float64_keeps_xla_tanh_below_its_own_seam_bitwise():
    """float64 ``tanh`` must be bit-for-bit XLA's ``tanh`` below 1/2 — the series is float32-only.

    The truncated degree-7 series is 1.3e-9 relative at x = 1/8, which is ~6e6 float64 ulps:
    usable in float32, catastrophic in float64. This pins the ``_is_low_precision`` dtype gate by
    rebuilding the expected float64 kernel inline (no dependency on a checkout of the previous
    version) on 20k seeded inputs spanning both sides of the float64 seam, at both signs.
    """
    rng = np.random.default_rng(_ULP_SEED)
    x_N = np.concatenate(
        [
            np.exp(rng.uniform(np.log(1e-6), np.log(0.5), size=_ULP_N // 2)),
            np.exp(rng.uniform(np.log(0.5), np.log(16.0), size=_ULP_N // 2)),
        ]
    )
    x_N = np.concatenate([x_N, -x_N])
    x = jnp.asarray(x_N, dtype=jnp.float64)

    # The inline expectation deliberately omits the wrapper's output clip, so the samples stop at
    # 16: in float64 that clip only starts binding at atanh(1 - 10*eps) = 17.2.
    abs_x = jnp.abs(x)
    t = jnp.expm1(2.0 * abs_x)
    magnitude = jnp.where(abs_x >= 0.5, t / (t + 2.0), jnp.tanh(abs_x))
    expected = jnp.where(x >= 0, magnitude, -magnitude)
    assert np.array_equal(np.asarray(tanh(x)), np.asarray(expected))


def test_atanh_float64_is_the_log1p_form_bitwise_on_the_whole_domain():
    """float64 ``atanh`` must be bit-for-bit the odd single-log1p form — the series is float32-only.

    The truncated series is 1.1e-11 relative at x = 1/8, which is ~5e4 float64 ulps: usable in
    float32, catastrophic in float64. This pins the ``_is_low_precision`` dtype gate by rebuilding
    the expected float64 kernel inline (no dependency on a checkout of the previous version), on
    20k seeded inputs spanning both sides of the float32 seam.
    """
    rng = np.random.default_rng(_ULP_SEED)
    x_N = np.concatenate(
        [
            np.exp(rng.uniform(np.log(1e-6), np.log(0.125), size=_ULP_N // 2)),
            np.exp(rng.uniform(np.log(0.125), np.log(0.999), size=_ULP_N // 2)),
        ]
    )
    x_N = np.concatenate([x_N, -x_N])
    x = jnp.asarray(x_N, dtype=jnp.float64)

    abs_x = jnp.abs(x)
    half_log1p = 0.5 * jnp.log1p(2.0 * abs_x / (1.0 - abs_x))
    expected = jnp.where(x >= 0, half_log1p, -half_log1p)
    assert np.array_equal(np.asarray(atanh(x)), np.asarray(expected))


# ---------------------------------------------------------------------------------------------
# safe_norm / safe_hypot / safe_normalize
#
# These replace the library's ``sqrt(sum(v**2) + MIN_NORM**2)`` idiom wherever the *magnitude*
# range matters rather than just the gradient at zero. Two failure modes are pinned below, one at
# each end of the exponent range, plus the exact VJP:
#
#   overflow  — ``sum(v**2)`` leaves float32 at |v| ~ 1.8e19, while the norm itself is
#               representable up to 3.4e38 (the hyperboloid spatial radius passes 1.8e19 at
#               geodesic radius ~45).
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


# ---------------------------------------------------------------------------------------------
# safe_hypot_norm and the power-of-two rescale
#
# The rescale the safe-norm family shares divides by the power of two just below ``max|v|``, not
# by ``max|v|``. Both are overflow-free; only the power of two is *exact*, so it adds no rounding
# of its own and the rescaled computation keeps the rounding of the plain ``sqrt(sum(v**2))``.
#
# That matters because callers recompute the same sum of squares. The hyperboloid time slot
# ``x0 = sqrt(||x_s||**2 + 1/c)`` is checked against ``-x0**2 + ||x_s||**2 = -1/c``, and at
# ``x0 ~ 34`` one float32 ulp of ``x0**2`` is 1.2e-4 -- larger than the 1e-4 tolerance the HRC
# norm layers are asserted at (``tests/nn_layers/test_hypformer.py``). The same argument forbids
# the two-leg spelling ``safe_hypot(safe_norm(v), q)``: it rounds ``||v||`` and squares it again,
# which is up to one ulp *of* ``sum(v**2)`` and cannot cancel. ``safe_hypot_norm`` is the
# one-reduction form.
# ---------------------------------------------------------------------------------------------


def _divide_by_max_norm_reference(v):
    """The scaler ``safe_norm`` used before: divide by ``max|v|`` itself, which is inexact."""
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(v), axis=-1))
    is_zero = scale == 0.0
    divisor = jnp.where((scale > 0.0) & jnp.isfinite(scale), scale, jnp.ones_like(scale))
    sq = jnp.sum((v / divisor[..., None]) ** 2, axis=-1)
    sq_safe = jnp.where(is_zero, jnp.ones_like(sq), sq)
    return jnp.where(is_zero, jnp.zeros_like(sq), divisor * jnp.sqrt(sq_safe))


def _ulp_distance(a, b, int_bits):
    """Distance in representable floats between two same-dtype arrays (both non-negative here)."""
    ia = np.asarray(a.view(int_bits)).astype(np.int64)
    ib = np.asarray(b.view(int_bits)).astype(np.int64)
    return np.abs(ia - ib)


def test_the_norm_rescale_adds_no_rounding_of_its_own():
    """``safe_norm`` tracks ``sqrt(sum(v**2))`` far more closely than the divide-by-max scaler did.

    The reference is jitted, like ``safe_norm`` itself: eager and jitted ``sqrt(sum(v**2))`` are
    not bit-identical to each other, so an eager reference would measure XLA's fusion rather than
    the scaler.

    Bit-identity is *not* asserted, because it is not XLA's to give: which association a reduction
    over the last axis gets depends on the surrounding fusion, and at D = 8 the two programs happen
    to differ (1-2 ulp) while at D in {2, 4, 16, 64} they agree exactly. The bound below separates
    that residue from the scaler's own error. Measured on this machine, mean ulp distance from the
    reference: power-of-two 0.000 at D in {2, 4, 16, 64} and 0.13-0.17 at D = 8, max 2; the
    divide-by-max form 0.34-0.62 at every D, max 3.
    """
    naive = jax.jit(lambda v: jnp.sqrt(jnp.sum(v**2, axis=-1)))
    for dt, int_bits in ((jnp.float32, jnp.uint32), (jnp.float64, jnp.uint64)):
        for dim in (2, 4, 8, 16, 64):
            for exponent in (-10, 0, 10):
                key = jax.random.PRNGKey(exponent + dim)
                v = (jax.random.normal(key, (500, dim), dtype=jnp.float64) * 10.0**exponent).astype(dt)
                reference = naive(v)
                d_pow2 = _ulp_distance(safe_norm(v), reference, int_bits)
                d_max = _ulp_distance(jax.jit(_divide_by_max_norm_reference)(v), reference, int_bits)
                where = f"{dt.__name__} D={dim} 1e{exponent}"
                assert d_pow2.max() <= 2, f"{where}: max ulp {d_pow2.max()}"
                assert d_pow2.mean() <= 0.25, f"{where}: mean ulp {d_pow2.mean()}"
                assert d_pow2.mean() <= d_max.mean(), f"{where}: {d_pow2.mean()} vs divide-by-max {d_max.mean()}"


def test_safe_hypot_norm_keeps_the_hyperboloid_constraint_residual_near_one_rounding():
    """The whole point of the one-reduction form: ``-x0**2 + ||s||**2 + 1/c`` still nearly cancels.

    ``x0`` is built three ways from the same spatial part, at the radius the HRC extreme-curvature
    tests reach (``c = 0.1``, ``x0`` up to ~89), and each is checked against the residual the
    manifold's own ``is_in_manifold`` forms. The two-leg ``safe_hypot(safe_norm(s), 1/sqrt(c))``
    re-squares an already-rounded norm, which is up to one ulp *of* ``sum(s**2)``; the
    one-reduction form stays within a small factor of the plain ``sqrt(sum(s**2) + 1/c)`` it
    replaces (it is not identical to it -- the float32 ``(1/sqrt(c))**2`` is not the float32
    ``1/c``, and XLA associates the two reductions independently). Measured here, mean residual:
    plain 4.8e-5, safe_hypot_norm 6.1e-5, two-leg 8.1e-5.
    """
    c = 0.1
    s = (17.0 * jax.random.normal(jax.random.PRNGKey(11), (5000, 4), dtype=jnp.float64)).astype(jnp.float32)
    sum_sq = jnp.sum(s**2, axis=-1)  # exactly what the constraint check recomputes
    inv_sqrt_c = jnp.asarray(1.0 / np.sqrt(c), dtype=jnp.float32)

    def residual(x0):
        return jnp.abs(-(x0.astype(jnp.float32) ** 2) + sum_sq + jnp.asarray(1.0 / c, dtype=jnp.float32))

    mean_plain = float(jnp.mean(residual(jnp.sqrt(sum_sq + jnp.asarray(1.0 / c, dtype=jnp.float32)))))
    mean_single = float(jnp.mean(residual(safe_hypot_norm(s, inv_sqrt_c))))
    mean_two_leg = float(jnp.mean(residual(safe_hypot(safe_norm(s), inv_sqrt_c))))

    assert mean_single <= 1.5 * mean_plain, f"{mean_single} vs plain {mean_plain}"
    assert mean_two_leg >= 1.2 * mean_single, f"two-leg {mean_two_leg} vs one-reduction {mean_single}"


def test_safe_hypot_norm_is_overflow_free_and_zero_gradient_safe():
    """Finite past the float32 ``sum(v**2)`` ceiling; exact 0 with an exactly-zero VJP at the origin."""
    big = jnp.full((8,), 1e20, dtype=jnp.float32)
    assert np.isfinite(float(safe_hypot_norm(big, jnp.float32(1e20))))
    assert float(safe_hypot_norm(big, jnp.float32(1e20))) == pytest.approx(3e20, rel=1e-6)

    for dt in (jnp.float32, jnp.float64):
        zero = jnp.zeros(4, dtype=dt)
        zero_q = jnp.asarray(0.0, dtype=dt)
        assert float(safe_hypot_norm(zero, zero_q)) == 0.0
        g = np.asarray(jax.grad(safe_hypot_norm)(zero, zero_q))
        assert np.all(g == 0.0)
        assert safe_hypot_norm(jnp.ones(4, dtype=dt), jnp.asarray(1.0, dtype=dt)).dtype == dt

    # Non-finite entries propagate as inf rather than becoming NaN (the family's convention).
    assert np.isinf(float(safe_hypot_norm(jnp.asarray([1.0, jnp.inf, 0.0], jnp.float32), jnp.float32(1.0))))
