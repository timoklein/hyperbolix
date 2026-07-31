"""Tests for JAX math utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.utils.math_utils import (
    acosh,
    atanh,
    cosh,
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
