"""Tests for Hyperboloid convolutional layers and HCat operation.

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypConv2DHyperboloid and hcat-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.scipy.special import digamma

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers.hyperboloid_conv import HypConv2DHyperboloid
from hyperbolix.nn_layers.hyperboloid_core import hcat_ambient_dim

# ============================================================================
# HCat Operation Tests
# ============================================================================


@pytest.mark.parametrize("N,n,c", [(2, 3, 1.0), (3, 4, 1.0), (4, 5, 0.5), (1, 3, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_output_on_manifold(N, n, c, dtype):
    """Test that HCat output lies on the Hyperboloid manifold."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)

    # Create N points on n-dimensional hyperboloid
    points = []
    for _i in range(N):
        key, subkey = jax.random.split(key)
        tangent = jax.random.normal(subkey, (n,), dtype=dtype) * 0.1
        tangent = tangent.at[0].set(0)  # Set time coordinate to 0 (tangent at origin)
        point = manifold.expmap_0(tangent, c)
        points.append(point)

    points = jnp.stack(points)  # (N, n)

    # Apply HCat
    result = manifold.hcat(points, c)

    # Check output dimension
    # Input: N points of ambient dimension n (manifold dim d = n-1)
    # Output: ambient dimension dN + 1 = (n-1)*N + 1
    d = n - 1  # Manifold dimension
    expected_dim = d * N + 1
    assert result.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {result.shape}"

    # Check output is on manifold
    # For a (dN)-dimensional manifold with curvature c:
    # Lorentz constraint: -x[0]^2 + sum(x[1:]^2) = -1/c
    lorentz_product = -(result[0] ** 2) + jnp.sum(result[1:] ** 2)
    expected = -1.0 / c
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.abs(lorentz_product - expected) < tolerance, f"HCat output not on manifold: {lorentz_product} != {expected}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_single_point(dtype):
    """Test HCat with a single point (edge case N=1)."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    n, c = 5, 1.0

    # Create single point
    tangent = jax.random.normal(key, (n,), dtype=dtype) * 0.1
    tangent = tangent.at[0].set(0)
    point = manifold.expmap_0(tangent, c)
    points = point.reshape(1, n)  # (1, n)

    # Apply HCat
    result = manifold.hcat(points, c)

    # For N=1, time coordinate should be sqrt(x[0]^2 + 0) = |x[0]|
    # and space coordinates should just be x[1:]
    assert result.shape == (n,)
    assert jnp.abs(result[0] - jnp.abs(point[0])) < 1e-6
    assert jnp.allclose(result[1:], point[1:], atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_dimensionality(dtype):
    """Test HCat correctly increases dimensionality."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    test_cases = [(2, 3), (3, 4), (5, 6)]

    for N, n in test_cases:
        # Create N points on n-dimensional hyperboloid (n is ambient dim)
        points = []
        for _i in range(N):
            key, subkey = jax.random.split(key)
            tangent = jax.random.normal(subkey, (n,), dtype=dtype) * 0.1
            tangent = tangent.at[0].set(0)
            point = manifold.expmap_0(tangent, 1.0)
            points.append(point)

        points = jnp.stack(points)  # (N, n)

        # Apply HCat
        result = manifold.hcat(points, 1.0)

        # Check dimensionality: (n-1)*N + 1
        d = n - 1  # Manifold dimension
        expected_dim = d * N + 1
        assert result.shape == (expected_dim,), f"Expected ({expected_dim},), got {result.shape}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_time_coordinate_formula(dtype):
    """Test HCat time coordinate computation formula."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    N, n, c = 3, 4, 1.0

    # Create N points
    points = []
    for _i in range(N):
        key, subkey = jax.random.split(key)
        tangent = jax.random.normal(subkey, (n,), dtype=dtype) * 0.1
        tangent = tangent.at[0].set(0)
        point = manifold.expmap_0(tangent, c)
        points.append(point)

    points = jnp.stack(points)  # (N, n)

    # Apply HCat
    result = manifold.hcat(points, c)

    # Manually compute expected time coordinate using the CORRECT formula
    # Formula: sqrt(sum(x_i[0]^2) - (N-1)/c)  [note the MINUS]
    time_coords = points[:, 0]  # (N,)
    expected_time = jnp.sqrt(jnp.sum(time_coords**2) - (N - 1) / c)

    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.abs(result[0] - expected_time) < tolerance, f"Time coordinate mismatch: {result[0]} != {expected_time}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_space_concatenation(dtype):
    """Test HCat correctly concatenates space coordinates."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    N, n, c = 3, 4, 1.0

    # Create N points
    points = []
    for _i in range(N):
        key, subkey = jax.random.split(key)
        tangent = jax.random.normal(subkey, (n,), dtype=dtype) * 0.1
        tangent = tangent.at[0].set(0)
        point = manifold.expmap_0(tangent, c)
        points.append(point)

    points = jnp.stack(points)  # (N, n)

    # Apply HCat
    result = manifold.hcat(points, c)

    # Check space coordinates are correctly concatenated
    expected_space = points[:, 1:].reshape(-1)  # (N*(n-1),)
    actual_space = result[1:]  # (N*n - 1,)

    assert jnp.allclose(actual_space, expected_space, atol=1e-6)


# ============================================================================
# Log-Radius Concatenation Tests (Shi et al. 2026, Sec. 4.3)
# ============================================================================


def _random_hyperboloid_points(key, N, n, c, dtype, scale=0.1):
    """Create N points on the (n-1)-dim hyperboloid (ambient dim n) via expmap_0."""
    manifold = Hyperboloid(dtype=dtype)
    points = []
    for _i in range(N):
        key, subkey = jax.random.split(key)
        tangent = jax.random.normal(subkey, (n,), dtype=dtype) * scale
        tangent = tangent.at[0].set(0)  # tangent at origin (time component 0)
        points.append(manifold.expmap_0(tangent, c))
    return jnp.stack(points)  # (N, n)


def _log_radius_scale(N, n, dtype):
    """Reference digamma scale s = exp(0.5*(psi(N*d/2) - psi(d/2))) with d = n - 1."""
    d = n - 1
    n_total = N * d
    return jnp.exp(0.5 * (digamma(n_total / 2.0) - digamma(d / 2.0))).astype(dtype)


@pytest.mark.parametrize("N,n,c", [(2, 3, 1.0), (3, 4, 1.0), (4, 5, 0.5), (1, 3, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_radius_concat_output_on_manifold(N, n, c, dtype):
    """Log-radius concat output lies on the (dN)-dimensional Hyperboloid."""
    manifold = Hyperboloid(dtype=dtype)
    points = _random_hyperboloid_points(jax.random.PRNGKey(42), N, n, c, dtype)

    result = manifold.log_radius_concat(points, c)

    d = n - 1
    expected_dim = d * N + 1
    assert result.shape == (expected_dim,)

    lorentz_product = -(result[0] ** 2) + jnp.sum(result[1:] ** 2)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.abs(lorentz_product - (-1.0 / c)) < tolerance


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_radius_concat_single_point_identity(dtype):
    """For N=1 the digamma scale is 1, so log_radius_concat reduces to hcat (identity)."""
    manifold = Hyperboloid(dtype=dtype)
    n, c = 5, 1.0
    points = _random_hyperboloid_points(jax.random.PRNGKey(42), 1, n, c, dtype)

    result = manifold.log_radius_concat(points, c)

    assert result.shape == (n,)
    # Scale == 1 ⇒ returns the input point and matches the unscaled hcat exactly.
    assert jnp.allclose(result, points[0], atol=1e-6)
    assert jnp.allclose(result, manifold.hcat(points, c), atol=1e-6)
    assert jnp.allclose(_log_radius_scale(1, n, dtype), jnp.asarray(1.0, dtype=dtype), atol=1e-6)


@pytest.mark.parametrize("N,n,c", [(2, 3, 1.0), (3, 4, 0.5), (4, 5, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_radius_concat_scale_and_space(N, n, c, dtype):
    """Spatial blocks are rescaled by s = exp(0.5*(psi(N*d/2) - psi(d/2))) before stacking."""
    manifold = Hyperboloid(dtype=dtype)
    points = _random_hyperboloid_points(jax.random.PRNGKey(7), N, n, c, dtype)

    result = manifold.log_radius_concat(points, c)

    scale = _log_radius_scale(N, n, dtype)
    expected_space = (scale * points[:, 1:]).reshape(-1)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.allclose(result[1:], expected_space, atol=tolerance)
    # Concatenation widens the spatial dimension (N>1), so blocks are upscaled.
    assert float(scale) > 1.0


@pytest.mark.parametrize("N,n,c", [(2, 3, 1.0), (3, 4, 0.5), (4, 5, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_radius_concat_time_formula(N, n, c, dtype):
    """Time coordinate equals sqrt(1/c + s^2 * sum(t_i^2 - 1/c))."""
    manifold = Hyperboloid(dtype=dtype)
    points = _random_hyperboloid_points(jax.random.PRNGKey(11), N, n, c, dtype)

    result = manifold.log_radius_concat(points, c)

    scale = _log_radius_scale(N, n, dtype)
    time_coords = points[:, 0]
    expected_time = jnp.sqrt(1.0 / c + scale**2 * jnp.sum(time_coords**2 - 1.0 / c))
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.abs(result[0] - expected_time) < tolerance


# ============================================================================
# HypConv2DHyperboloid Layer Tests
# ============================================================================


def test_hypconv_hyperboloid_kernel_init_reset_params():
    """Default kernel init is U(-0.02, 0.02); reset_params='fan_in' gives
    std=sqrt(1/hcat_out_ambient_dim)."""
    manifold = Hyperboloid(dtype=jnp.float32)
    in_channels, out_channels, kernel_size = 9, 17, 3

    layer_default = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(42),
    )
    layer_fan_in = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(42),
        reset_params="fan_in",
    )

    bound = 0.02
    expected_default_std = bound / jnp.sqrt(3.0)  # std of U(-bound, bound)
    assert abs(float(jnp.std(layer_default.kernel[...])) - float(expected_default_std)) < 0.2 * float(expected_default_std)

    hcat_out_ambient_dim = hcat_ambient_dim(in_channels, (kernel_size, kernel_size))
    expected_fan_in_std = float(jnp.sqrt(1.0 / hcat_out_ambient_dim))
    assert abs(float(jnp.std(layer_fan_in.kernel[...])) - expected_fan_in_std) < 0.2 * expected_fan_in_std


def test_hypconv_hyperboloid_invalid_reset_params_errors():
    with pytest.raises(ValueError, match="reset_params"):
        HypConv2DHyperboloid(
            manifold_module=Hyperboloid(dtype=jnp.float32),
            in_channels=9,
            out_channels=17,
            kernel_size=3,
            rngs=nnx.Rngs(0),
            reset_params="bogus",
        )
