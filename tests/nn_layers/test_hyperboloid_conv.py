"""Tests for Hyperboloid convolutional layers and HCat operation."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.scipy.special import digamma

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers.hyperboloid_conv import HypConv2DHyperboloid

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


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_hyperboloid_output_shape(kernel_size, padding, dtype):
    """Test HypConv2DHyperboloid output shape with different kernel sizes and padding."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    c = 1.0

    # Create input feature map (batch, height, width, in_channels)
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1

    # Project each point to manifold
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
        padding=padding,
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check output shape
    if padding == "SAME":
        expected_height, expected_width = height, width
    else:  # VALID
        expected_height = height - kernel_size + 1
        expected_width = width - kernel_size + 1

    assert y.shape == (batch_size, expected_height, expected_width, out_channels), (
        f"Expected shape ({batch_size}, {expected_height}, {expected_width}, {out_channels}), got {y.shape}"
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_hyperboloid_output_on_manifold(dtype):
    """Test that all outputs lie on the Hyperboloid manifold."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        padding="SAME",
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check all outputs are on manifold
    def check_manifold(point):
        return manifold.is_in_manifold(point, c)

    # vmap over all dimensions
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(check_manifold)))(y)
    assert is_on_manifold.all(), "Not all outputs lie on the manifold"


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_hyperboloid_stride(stride, dtype):
    """Test HypConv2DHyperboloid with different stride values."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    kernel_size = 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        rngs=rngs,
        padding="SAME",
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check output shape matches expected stride behavior
    expected_height = (height + stride - 1) // stride
    expected_width = (width + stride - 1) // stride
    assert y.shape == (batch_size, expected_height, expected_width, out_channels)


@pytest.mark.parametrize("input_space", ["manifold", "tangent"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_hyperboloid_input_space(input_space, dtype):
    """Test HypConv2DHyperboloid with different input_space settings."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1

    if input_space == "manifold":
        # Project to manifold
        x_input = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c), in_axes=0), in_axes=0), in_axes=0)(x)
    else:
        # Keep in tangent space (set time coordinate to 0)
        x_input = x.at[:, :, :, 0].set(0)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        input_space=input_space,
    )

    # Forward pass
    y = layer(x_input, c=c)

    # Check outputs are on manifold
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.is_in_manifold(p, c))))(y)
    assert is_on_manifold.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_hyperboloid_gradient(dtype):
    """Test HypConv2DHyperboloid has valid gradients."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    # Define loss function
    def loss_fn(model):
        y = model(x_manifold, c=c)
        return jnp.sum(y**2)

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    # Check gradients exist and are finite
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_hyperboloid_jitted(dtype):
    """Test HypConv2DHyperboloid under nnx.jit."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    # Forward pass
    y = forward(layer, x_manifold, c)

    # Check outputs are on manifold
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.is_in_manifold(p, c))))(y)
    assert is_on_manifold.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_hyperboloid_different_curvatures(dtype):
    """Test HypConv2DHyperboloid with different curvature values."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3

    curvatures = [0.5, 1.0, 2.0]

    for c in curvatures:
        # Create input
        x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
        proj_fn = partial(manifold.proj, c=c)
        x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypConv2DHyperboloid(
            manifold_module=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            rngs=rngs,
        )

        # Forward pass
        y = layer(x_manifold, c=c)

        # Check outputs are on manifold with correct curvature
        is_in_manifold_fn = partial(manifold.is_in_manifold, c=c)
        is_on_manifold = jax.vmap(jax.vmap(jax.vmap(is_in_manifold_fn)))(y)
        assert is_on_manifold.all(), f"Failed for curvature {c}"
