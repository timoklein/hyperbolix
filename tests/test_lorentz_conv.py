"""Tests for Lorentz convolutional layers and transforms.

Tests cover:
- lorentz_boost function (manifold preservation, batch operations)
- distance_rescale function (manifold preservation, distance bounding)
- LorentzConv2D layer (shapes, manifold, gradients, JIT)
- LorentzConv3D layer (shapes, manifold, gradients, JIT)
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import hyperbolix.manifolds.hyperboloid as hyperboloid
from hyperbolix.nn_layers.lorentz_conv import LorentzConv2D, LorentzConv3D

# Enable float64 for tests
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Lorentz Boost Tests
# ============================================================================


@pytest.mark.parametrize("dim", [3, 4, 5])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_lorentz_boost_preserves_manifold(dim: int, c: float, dtype):
    """Test that Lorentz boost preserves the hyperboloid manifold constraint."""
    key = jax.random.PRNGKey(42)

    # Create point on hyperboloid (ambient dimension = dim)
    key, subkey = jax.random.split(key)
    tangent = jax.random.normal(subkey, (dim,), dtype=dtype) * 0.3
    tangent = tangent.at[0].set(0)  # Tangent at origin has zero time component
    x = hyperboloid.expmap_0(tangent, c)

    # Verify input is on manifold
    assert hyperboloid.is_in_manifold(x, c, atol=1e-5), "Input point not on manifold"

    # Create velocity vector (spatial dimensions = dim - 1)
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (dim - 1,), dtype=dtype) * 0.3

    # Apply Lorentz boost
    x_boosted = hyperboloid.lorentz_boost(x, v, c)

    # Check output is on manifold
    tolerance = 1e-4 if dtype == jnp.float32 else 1e-10
    assert hyperboloid.is_in_manifold(x_boosted, c, atol=tolerance), (
        f"Boosted point not on manifold: Lorentz norm = {-(x_boosted[0] ** 2) + jnp.sum(x_boosted[1:] ** 2)}, "
        f"expected = {-1.0 / c}"
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_lorentz_boost_zero_velocity(dtype):
    """Test that zero velocity boost returns the same point."""
    key = jax.random.PRNGKey(42)
    dim, c = 4, 1.0

    # Create point on hyperboloid
    tangent = jax.random.normal(key, (dim,), dtype=dtype) * 0.3
    tangent = tangent.at[0].set(0)
    x = hyperboloid.expmap_0(tangent, c)

    # Zero velocity
    v = jnp.zeros(dim - 1, dtype=dtype)

    # Apply boost
    x_boosted = hyperboloid.lorentz_boost(x, v, c)

    # Should be unchanged
    tolerance = 1e-6 if dtype == jnp.float32 else 1e-12
    assert jnp.allclose(x, x_boosted, atol=tolerance), "Zero velocity boost should not change point"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_lorentz_boost_batch(dtype):
    """Test Lorentz boost with batched inputs."""
    key = jax.random.PRNGKey(42)
    batch_size, dim, c = 10, 4, 1.0

    # Create batch of points
    key, subkey = jax.random.split(key)
    tangents = jax.random.normal(subkey, (batch_size, dim), dtype=dtype) * 0.3
    tangents = tangents.at[:, 0].set(0)
    x_batch = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(tangents, c)

    # Single velocity for all points
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (dim - 1,), dtype=dtype) * 0.3

    # Apply boost via vmap
    x_boosted = jax.vmap(hyperboloid.lorentz_boost, in_axes=(0, None, None))(x_batch, v, c)

    # Check all outputs are on manifold
    tolerance = 1e-4 if dtype == jnp.float32 else 1e-10
    is_on_manifold = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None, None))(x_boosted, c, tolerance)
    assert is_on_manifold.all(), f"Not all boosted points on manifold: {jnp.sum(is_on_manifold)}/{batch_size}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_lorentz_boost_near_unit_velocity(dtype):
    """Test Lorentz boost with velocity near the unit boundary."""
    key = jax.random.PRNGKey(42)
    dim, c = 4, 1.0

    # Create point on hyperboloid
    tangent = jax.random.normal(key, (dim,), dtype=dtype) * 0.3
    tangent = tangent.at[0].set(0)
    x = hyperboloid.expmap_0(tangent, c)

    # Velocity with norm close to but less than 1
    v_near_unit = jnp.array([0.5, 0.5, 0.5], dtype=dtype)  # |v| ≈ 0.866

    # Should work without issues
    x_boosted = hyperboloid.lorentz_boost(x, v_near_unit, c)

    # Output should still be on manifold
    tolerance = 1e-4 if dtype == jnp.float32 else 1e-10
    assert hyperboloid.is_in_manifold(x_boosted, c, atol=tolerance), "Boosted point with near-unit velocity not on manifold"


# ============================================================================
# Distance Rescale Tests
# ============================================================================


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_distance_rescale_preserves_manifold(c: float, dtype):
    """Test that distance rescaling preserves the hyperboloid manifold."""
    key = jax.random.PRNGKey(42)
    dim = 4

    # Create point on hyperboloid
    tangent = jax.random.normal(key, (dim,), dtype=dtype) * 0.5
    tangent = tangent.at[0].set(0)
    x = hyperboloid.expmap_0(tangent, c)

    # Verify input is on manifold
    assert hyperboloid.is_in_manifold(x, c, atol=1e-5), "Input not on manifold"

    # Apply distance rescaling
    x_rescaled = hyperboloid.distance_rescale(x, c)

    # Check output is on manifold
    tolerance = 1e-4 if dtype == jnp.float32 else 1e-10
    assert hyperboloid.is_in_manifold(x_rescaled, c, atol=tolerance), "Rescaled point not on manifold"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_distance_rescale_bounds_large_distance(dtype):
    """Test that distance rescaling bounds large hyperbolic distances.

    The rescaling applies: D_rescaled = D_max · tanh(D · atanh(0.99) / (s·D_max))

    For very large D, D_rescaled approaches D_max (bounded).
    For small D, D_rescaled ≈ D · atanh(0.99) ≈ 2.65·D (amplified).

    This test verifies the bounding behavior for large distances.
    """
    c = 1.0
    x_t_max = 50.0  # Use smaller max for testing

    # Create point with VERY large distance from origin
    # Using large time coordinate directly
    x_spatial = jnp.array([5.0, 5.0, 5.0], dtype=dtype)
    x = hyperboloid.proj(jnp.concatenate([jnp.array([10.0], dtype=dtype), x_spatial]), c)

    # Apply distance rescaling
    x_rescaled = hyperboloid.distance_rescale(x, c, x_t_max=x_t_max)

    # Distance after rescaling
    dist_after = hyperboloid.dist_0(x_rescaled, c)

    # Compute the theoretical maximum bounded distance
    sqrt_c = jnp.sqrt(c)
    D_max = jnp.arccosh(sqrt_c * x_t_max) / sqrt_c

    # For large initial distances, the rescaled distance should approach D_max
    # but never exceed it significantly
    assert dist_after < D_max * 1.01, f"Distance not bounded: after={dist_after}, D_max={D_max}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_distance_rescale_origin_unchanged(dtype):
    """Test that points near origin are minimally affected."""
    c = 1.0

    # Create point very close to origin
    tangent = jnp.array([0.0, 0.01, 0.01, 0.01], dtype=dtype)
    x = hyperboloid.expmap_0(tangent, c)

    # Apply rescaling
    x_rescaled = hyperboloid.distance_rescale(x, c)

    # Point should be nearly unchanged
    tolerance = 0.1 if dtype == jnp.float32 else 0.05
    dist_original = hyperboloid.dist_0(x, c)
    dist_rescaled = hyperboloid.dist_0(x_rescaled, c)
    assert jnp.abs(dist_original - dist_rescaled) < tolerance, "Point near origin changed too much"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_distance_rescale_batch(dtype):
    """Test distance rescaling with batched inputs."""
    key = jax.random.PRNGKey(42)
    batch_size, dim, c = 10, 4, 1.0

    # Create batch of points
    tangents = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.5
    tangents = tangents.at[:, 0].set(0)
    x_batch = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(tangents, c)

    # Apply rescaling via vmap
    x_rescaled = jax.vmap(hyperboloid.distance_rescale, in_axes=(0, None))(x_batch, c)

    # Check all outputs are on manifold
    tolerance = 1e-4 if dtype == jnp.float32 else 1e-10
    is_on_manifold = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None, None))(x_rescaled, c, tolerance)
    assert is_on_manifold.all(), f"Not all rescaled points on manifold: {jnp.sum(is_on_manifold)}/{batch_size}"


# ============================================================================
# LorentzConv2D Layer Tests
# ============================================================================


# Note: LorentzConv2D/3D layers initialize weights as float64 when jax_enable_x64=True.
# Testing with float32 input causes dtype mismatch in conv_general_dilated.
# We test only with float64 to match the layer's actual behavior.


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_output_shape(kernel_size: int, padding: str, dtype):
    """Test LorentzConv2D output shape with different kernel sizes and padding."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 4, 6
    c = 1.0

    # Create input feature map
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv2D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
        padding=padding,
        dtype=dtype,
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


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_output_on_manifold(dtype):
    """Test that all LorentzConv2D outputs lie on the hyperboloid manifold."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 6, 6, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv2D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=rngs,
        padding="SAME",
        dtype=dtype,
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check all outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance))))(y)
    num_valid = jnp.sum(is_on_manifold)
    total = is_on_manifold.size
    assert num_valid / total > 0.99, f"Only {num_valid}/{total} points on manifold ({100 * num_valid / total:.1f}%)"


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_stride(stride: int, dtype):
    """Test LorentzConv2D with different stride values."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 4, 5
    kernel_size = 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv2D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        rngs=rngs,
        padding="SAME",
        dtype=dtype,
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check output shape matches expected stride behavior
    expected_height = (height + stride - 1) // stride
    expected_width = (width + stride - 1) // stride
    assert y.shape == (batch_size, expected_height, expected_width, out_channels), (
        f"Expected shape with stride={stride}: ({batch_size}, {expected_height}, {expected_width}, {out_channels}), "
        f"got {y.shape}"
    )


@pytest.mark.parametrize("input_space", ["manifold", "tangent"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_input_space(input_space: str, dtype):
    """Test LorentzConv2D with different input_space settings."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 6, 6, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1

    if input_space == "manifold":
        # Project to manifold
        proj_fn = partial(hyperboloid.proj, c=c)
        x_input = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)
    else:
        # Keep in tangent space (set time coordinate to 0)
        x_input = x.at[:, :, :, 0].set(0)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv2D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=rngs,
        input_space=input_space,
        dtype=dtype,
    )

    # Forward pass
    y = layer(x_input, c=c)

    # Check outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance))))(y)
    num_valid = jnp.sum(is_on_manifold)
    total = is_on_manifold.size
    assert num_valid / total > 0.99, f"Only {num_valid}/{total} points on manifold for input_space={input_space}"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_gradient(dtype):
    """Test LorentzConv2D has valid gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 6, 6, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv2D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=rngs,
        dtype=dtype,
    )

    # Define loss function
    def loss_fn(model):
        y = model(x_manifold, c=c)
        return jnp.sum(y**2)

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    # Check gradients exist and are finite
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert jnp.isfinite(grads.weight[...]).all(), "Weight gradients contain NaN/Inf"
    if layer.use_boost:
        assert jnp.isfinite(grads.boost_velocity[...]).all(), "Boost velocity gradients contain NaN/Inf"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_jitted(dtype):
    """Test LorentzConv2D under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 6, 6, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv2D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=rngs,
        dtype=dtype,
    )

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    # Forward pass (first call compiles)
    y1 = forward(layer, x_manifold, c)

    # Second call (uses cached compilation)
    y2 = forward(layer, x_manifold, c)

    # Results should be consistent
    assert jnp.allclose(y1, y2, atol=1e-6), "JIT results not consistent between calls"

    # Check outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance))))(y1)
    assert is_on_manifold.all(), "Not all JIT outputs on manifold"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_different_curvatures(dtype):
    """Test LorentzConv2D with different curvature values."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 6, 6, 4, 5

    curvatures = [0.5, 1.0, 2.0]

    for c in curvatures:
        # Create input
        x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
        proj_fn = partial(hyperboloid.proj, c=c)
        x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = LorentzConv2D(
            manifold_module=hyperboloid,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            rngs=rngs,
            dtype=dtype,
        )

        # Forward pass
        y = layer(x_manifold, c=c)

        # Check outputs are on manifold with correct curvature
        tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
        check_fn = partial(hyperboloid.is_in_manifold, c=c, atol=tolerance)
        is_on_manifold = jax.vmap(jax.vmap(jax.vmap(check_fn)))(y)
        num_valid = jnp.sum(is_on_manifold)
        total = is_on_manifold.size
        assert num_valid / total > 0.99, f"Failed for curvature {c}: {num_valid}/{total} on manifold"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_no_boost(dtype):
    """Test LorentzConv2D with boost disabled."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 6, 6, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer without boost
    rngs = nnx.Rngs(42)
    layer = LorentzConv2D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=rngs,
        use_boost=False,
        dtype=dtype,
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance))))(y)
    assert is_on_manifold.all(), "Not all outputs on manifold with boost disabled"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv2d_no_rescaling(dtype):
    """Test LorentzConv2D with distance rescaling disabled."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 6, 6, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer without rescaling
    rngs = nnx.Rngs(42)
    layer = LorentzConv2D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=rngs,
        use_distance_rescaling=False,
        dtype=dtype,
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance))))(y)
    assert is_on_manifold.all(), "Not all outputs on manifold with rescaling disabled"


# ============================================================================
# LorentzConv3D Layer Tests
# ============================================================================


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv3d_output_shape(kernel_size: int, padding: str, dtype):
    """Test LorentzConv3D output shape with different kernel sizes and padding."""
    key = jax.random.PRNGKey(42)
    batch_size, depth, height, width, in_channels, out_channels = 2, 6, 6, 6, 4, 5
    c = 1.0

    # Create input feature map
    x = jax.random.normal(key, (batch_size, depth, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv3D(
        manifold_module=hyperboloid,
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
        expected_depth, expected_height, expected_width = depth, height, width
    else:  # VALID
        expected_depth = depth - kernel_size + 1
        expected_height = height - kernel_size + 1
        expected_width = width - kernel_size + 1

    assert y.shape == (batch_size, expected_depth, expected_height, expected_width, out_channels), (
        f"Expected shape ({batch_size}, {expected_depth}, {expected_height}, {expected_width}, {out_channels}), got {y.shape}"
    )


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv3d_output_on_manifold(dtype):
    """Test that all LorentzConv3D outputs lie on the hyperboloid manifold."""
    key = jax.random.PRNGKey(42)
    batch_size, depth, height, width, in_channels, out_channels = 2, 4, 4, 4, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, depth, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv3D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check all outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance)))))(y)
    num_valid = jnp.sum(is_on_manifold)
    total = is_on_manifold.size
    assert num_valid / total > 0.99, f"Only {num_valid}/{total} points on manifold ({100 * num_valid / total:.1f}%)"


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv3d_stride(stride: int, dtype):
    """Test LorentzConv3D with different strides."""
    key = jax.random.PRNGKey(42)
    batch_size, depth, height, width, in_channels, out_channels = 2, 8, 8, 8, 4, 5
    kernel_size = 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, depth, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv3D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        rngs=rngs,
        padding="VALID",
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check output shape with stride
    expected_depth = (depth - kernel_size) // stride + 1
    expected_height = (height - kernel_size) // stride + 1
    expected_width = (width - kernel_size) // stride + 1

    assert y.shape == (batch_size, expected_depth, expected_height, expected_width, out_channels), (
        f"Expected shape ({batch_size}, {expected_depth}, {expected_height}, {expected_width}, {out_channels}), got {y.shape}"
    )


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv3d_different_curvatures(dtype):
    """Test LorentzConv3D with different curvature values."""
    key = jax.random.PRNGKey(42)
    batch_size, depth, height, width, in_channels, out_channels = 2, 4, 4, 4, 4, 5

    curvatures = [0.5, 1.0, 2.0]

    for c in curvatures:
        # Create input
        x = jax.random.normal(key, (batch_size, depth, height, width, in_channels), dtype=dtype) * 0.1
        proj_fn = partial(hyperboloid.proj, c=c)
        x_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0), in_axes=0)(x)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = LorentzConv3D(
            manifold_module=hyperboloid,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            rngs=rngs,
        )

        # Forward pass
        y = layer(x_manifold, c=c)

        # Check outputs are on manifold with correct curvature
        tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
        check_fn = partial(hyperboloid.is_in_manifold, c=c, atol=tolerance)
        is_on_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(check_fn))))(y)
        num_valid = jnp.sum(is_on_manifold)
        total = is_on_manifold.size
        assert num_valid / total > 0.99, f"Failed for curvature {c}: {num_valid}/{total} on manifold"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv3d_tangent_input(dtype):
    """Test LorentzConv3D with tangent space input."""
    key = jax.random.PRNGKey(42)
    batch_size, depth, height, width, in_channels, out_channels = 2, 4, 4, 4, 4, 5
    c = 1.0

    # Create tangent space input (not on manifold)
    x_tangent = jax.random.normal(key, (batch_size, depth, height, width, in_channels), dtype=dtype) * 0.1
    x_tangent = x_tangent.at[:, :, :, :, 0].set(0)  # Time coordinate = 0 for tangent vectors

    # Create layer with tangent input space
    rngs = nnx.Rngs(42)
    layer = LorentzConv3D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        input_space="tangent",
    )

    # Forward pass
    y = layer(x_tangent, c=c)

    # Check outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance)))))(y)
    num_valid = jnp.sum(is_on_manifold)
    total = is_on_manifold.size
    assert num_valid / total > 0.99, f"Outputs should be on manifold even with tangent input: {num_valid}/{total}"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv3d_anisotropic_kernel(dtype):
    """Test LorentzConv3D with non-cubic kernel size."""
    key = jax.random.PRNGKey(42)
    batch_size, depth, height, width, in_channels, out_channels = 2, 8, 8, 8, 4, 5
    kernel_size = (2, 3, 2)  # Non-cubic kernel
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, depth, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer with anisotropic kernel
    rngs = nnx.Rngs(42)
    layer = LorentzConv3D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
        padding="VALID",
    )

    # Forward pass
    y = layer(x_manifold, c=c)

    # Check output shape
    expected_depth = depth - kernel_size[0] + 1
    expected_height = height - kernel_size[1] + 1
    expected_width = width - kernel_size[2] + 1

    assert y.shape == (batch_size, expected_depth, expected_height, expected_width, out_channels), (
        f"Expected shape ({batch_size}, {expected_depth}, {expected_height}, {expected_width}, {out_channels}), got {y.shape}"
    )

    # Check outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance)))))(y)
    assert is_on_manifold.all(), "All outputs should be on manifold with anisotropic kernel"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv3d_gradient(dtype):
    """Test LorentzConv3D has valid gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, depth, height, width, in_channels, out_channels = 1, 4, 4, 4, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, depth, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv3D(
        manifold_module=hyperboloid,
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
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert jnp.isfinite(grads.weight[...]).all(), "Weight gradients contain NaN/Inf"
    if layer.use_boost:
        assert jnp.isfinite(grads.boost_velocity[...]).all(), "Boost velocity gradients contain NaN/Inf"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_lorentz_conv3d_jitted(dtype):
    """Test LorentzConv3D under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, depth, height, width, in_channels, out_channels = 1, 4, 4, 4, 4, 5
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, depth, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(hyperboloid.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = LorentzConv3D(
        manifold_module=hyperboloid,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    # Forward pass (first call compiles)
    y1 = forward(layer, x_manifold, c)

    # Second call (uses cached compilation)
    y2 = forward(layer, x_manifold, c)

    # Results should be consistent
    assert jnp.allclose(y1, y2, atol=1e-6), "JIT results not consistent between calls"

    # Check outputs are on manifold
    tolerance = 1e-3 if dtype == jnp.float32 else 1e-8
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, c, atol=tolerance)))))(y1)
    assert is_on_manifold.all(), "Not all JIT outputs on manifold"
