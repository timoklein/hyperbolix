"""Tests for JAX math utilities."""

import jax.numpy as jnp
import numpy as np
from math_utils import (
    smooth_clamp_min,
    smooth_clamp_max,
    smooth_clamp,
    safe_cosh,
    safe_sinh,
    safe_acosh,
    safe_atanh,
    _get_array_eps,
)


def test_get_array_eps():
    """Test epsilon extraction for different dtypes."""
    print("Testing _get_array_eps...")

    x_f32 = jnp.array([1.0], dtype=jnp.float32)
    x_f64 = jnp.array([1.0], dtype=jnp.float64)

    eps32 = _get_array_eps(x_f32)
    eps64 = _get_array_eps(x_f64)

    assert eps32 == jnp.finfo(jnp.float32).eps
    assert eps64 == jnp.finfo(jnp.float64).eps
    assert eps32 > eps64  # float32 has larger epsilon

    print(f"  float32 eps: {eps32:.2e}")
    print(f"  float64 eps: {eps64:.2e}")
    print("  ✓ Epsilon extraction works correctly")


def test_smooth_clamp_min():
    """Test smooth minimum clamping."""
    print("\\nTesting smooth_clamp_min...")

    # Test basic functionality
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    min_val = 0.0
    result = smooth_clamp_min(x, min_val)

    # Values above min_val should be unchanged
    assert jnp.allclose(result[3:], x[3:], rtol=1e-6)  # [1.0, 2.0] unchanged

    # Values below min_val should be clamped and >= min_val
    assert jnp.all(result >= min_val)

    # Should be smooth (no discontinuities)
    assert result[0] < result[1] < result[2]  # monotonic

    print(f"  Input:  {x}")
    print(f"  Output: {result}")
    print(f"  All values >= {min_val}: {jnp.all(result >= min_val)}")
    print("  ✓ Smooth minimum clamping works")


def test_smooth_clamp_max():
    """Test smooth maximum clamping."""
    print("\\nTesting smooth_clamp_max...")

    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    max_val = 1.0
    result = smooth_clamp_max(x, max_val)

    # Values below max_val should be unchanged
    assert jnp.allclose(result[:4], x[:4], rtol=1e-6)  # [-2, -1, 0, 1] unchanged

    # Values above max_val should be clamped and <= max_val
    assert jnp.all(result <= max_val)

    print(f"  Input:  {x}")
    print(f"  Output: {result}")
    print(f"  All values <= {max_val}: {jnp.all(result <= max_val)}")
    print("  ✓ Smooth maximum clamping works")


def test_smooth_clamp():
    """Test smooth range clamping."""
    print("\\nTesting smooth_clamp...")

    x = jnp.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    min_val, max_val = -1.5, 1.5
    result = smooth_clamp(x, min_val, max_val)

    # All values should be in range
    assert jnp.all(result >= min_val)
    assert jnp.all(result <= max_val)

    # Values in range should be approximately unchanged
    in_range_mask = (x >= min_val) & (x <= max_val)
    assert jnp.allclose(result[in_range_mask], x[in_range_mask], rtol=1e-5)

    print(f"  Input:  {x}")
    print(f"  Output: {result}")
    print(f"  Range: [{min_val}, {max_val}]")
    print(f"  In range: {jnp.all((result >= min_val) & (result <= max_val))}")
    print("  ✓ Smooth range clamping works")


def test_safe_cosh():
    """Test numerically stable cosh."""
    print("\\nTesting safe_cosh...")

    # Test normal values
    x_normal = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result_normal = safe_cosh(x_normal)
    expected_normal = jnp.cosh(x_normal)
    assert jnp.allclose(result_normal, expected_normal)

    # Test extreme values that would overflow regular cosh
    x_extreme = jnp.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=jnp.float32)
    result_extreme = safe_cosh(x_extreme)

    # Should not contain inf or nan
    assert jnp.all(jnp.isfinite(result_extreme))

    # Should be symmetric: cosh(-x) = cosh(x)
    assert jnp.allclose(result_extreme[0], result_extreme[4], rtol=1e-5)
    assert jnp.allclose(result_extreme[1], result_extreme[3], rtol=1e-5)

    print(f"  Normal values: {x_normal}")
    print(f"  Normal cosh: {result_normal}")
    print(f"  Extreme values: {x_extreme}")
    print(f"  Extreme cosh (finite): {jnp.all(jnp.isfinite(result_extreme))}")
    print("  ✓ Safe cosh works")


def test_safe_sinh():
    """Test numerically stable sinh."""
    print("\\nTesting safe_sinh...")

    # Test normal values
    x_normal = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result_normal = safe_sinh(x_normal)
    expected_normal = jnp.sinh(x_normal)
    assert jnp.allclose(result_normal, expected_normal)

    # Test extreme values
    x_extreme = jnp.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=jnp.float32)
    result_extreme = safe_sinh(x_extreme)

    # Should not contain inf or nan
    assert jnp.all(jnp.isfinite(result_extreme))

    # Should be antisymmetric: sinh(-x) = -sinh(x)
    assert jnp.allclose(result_extreme[0], -result_extreme[4], rtol=1e-5)
    assert jnp.allclose(result_extreme[1], -result_extreme[3], rtol=1e-5)
    assert jnp.abs(result_extreme[2]) < 1e-10  # sinh(0) = 0

    print(f"  Normal values: {x_normal}")
    print(f"  Normal sinh: {result_normal}")
    print(f"  Extreme values: {x_extreme}")
    print(f"  Extreme sinh (finite): {jnp.all(jnp.isfinite(result_extreme))}")
    print("  ✓ Safe sinh works")


def test_safe_acosh():
    """Test numerically stable acosh."""
    print("\\nTesting safe_acosh...")

    # Test valid domain values
    x_valid = jnp.array([1.0, 1.5, 2.0, 5.0, 10.0])
    result_valid = safe_acosh(x_valid)
    expected_valid = jnp.acosh(x_valid)
    assert jnp.allclose(result_valid, expected_valid)

    # Test invalid domain values (should be clamped)
    x_invalid = jnp.array([0.5, 0.9, 1.0, 1.1, 2.0])
    result_invalid = safe_acosh(x_invalid)

    # Should not contain nan
    assert jnp.all(jnp.isfinite(result_invalid))

    # Values < 1 should be clamped to acosh(1) = 0
    assert result_invalid[0] == 0.0  # acosh(1) = 0
    assert result_invalid[1] == 0.0  # clamped
    assert result_invalid[2] == 0.0  # acosh(1) = 0

    print(f"  Valid domain: {x_valid}")
    print(f"  Valid acosh: {result_valid}")
    print(f"  Invalid domain: {x_invalid}")
    print(f"  Clamped acosh: {result_invalid}")
    print("  ✓ Safe acosh works")


def test_safe_atanh():
    """Test numerically stable atanh."""
    print("\\nTesting safe_atanh...")

    # Test valid domain values
    x_valid = jnp.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    result_valid = safe_atanh(x_valid)
    expected_valid = jnp.atanh(x_valid)
    assert jnp.allclose(result_valid, expected_valid)

    # Test boundary values (should be clamped away from ±1)
    x_boundary = jnp.array([-1.1, -1.0, -0.9999, 0.9999, 1.0, 1.1])
    result_boundary = safe_atanh(x_boundary)

    # Should not contain inf or nan
    assert jnp.all(jnp.isfinite(result_boundary))

    # Should be antisymmetric
    assert jnp.allclose(result_boundary[0], -result_boundary[-1], rtol=1e-5)
    assert jnp.abs(result_boundary[2]) < 1e10  # Should be finite but large

    print(f"  Valid domain: {x_valid}")
    print(f"  Valid atanh: {result_valid}")
    print(f"  Boundary values: {x_boundary}")
    print(f"  Clamped atanh (finite): {jnp.all(jnp.isfinite(result_boundary))}")
    print("  ✓ Safe atanh works")


def test_dtype_consistency():
    """Test that functions preserve dtype."""
    print("\\nTesting dtype consistency...")

    for dtype in [jnp.float32, jnp.float64]:
        x = jnp.array([0.5, 1.0, 1.5], dtype=dtype)

        # Test all functions preserve dtype
        assert smooth_clamp(x, 0.0, 2.0).dtype == dtype
        assert safe_cosh(x).dtype == dtype
        assert safe_sinh(x).dtype == dtype
        assert safe_acosh(x).dtype == dtype
        assert safe_atanh(x * 0.5).dtype == dtype  # Scale to valid domain

        print(f"  ✓ {dtype} dtype preserved across all functions")

    print("  ✓ Dtype consistency verified")


def run_all_tests():
    """Run all math utils tests."""
    print("=== Testing JAX Math Utils ===\\n")

    test_get_array_eps()
    test_smooth_clamp_min()
    test_smooth_clamp_max()
    test_smooth_clamp()
    test_safe_cosh()
    test_safe_sinh()
    test_safe_acosh()
    test_safe_atanh()
    test_dtype_consistency()

    print("\\n=== All Tests Passed! ===")


if __name__ == "__main__":
    run_all_tests()