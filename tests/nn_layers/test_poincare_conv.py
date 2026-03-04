"""Tests for Poincaré convolutional layers and beta-concatenation operation."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers.poincare_conv import HypConv2DPoincare

# ============================================================================
# Helper: create random Poincaré ball points
# ============================================================================


def _make_poincare_points(key, shape, c, dtype):
    """Create random points on the Poincaré ball via expmap_0 of small tangent vectors."""
    manifold = Poincare(dtype=dtype)
    tangent = jax.random.normal(key, shape, dtype=dtype) * 0.1
    if tangent.ndim == 1:
        return manifold.expmap_0(tangent, c)
    return jax.vmap(manifold.expmap_0, in_axes=(0, None))(tangent, c)


def _make_tangent_input(key, shape, dtype):
    """Create random tangent-space input (small vectors near origin)."""
    return jax.random.normal(key, shape, dtype=dtype) * 0.1


# ============================================================================
# Beta-Concatenation Tests
# ============================================================================


@pytest.mark.parametrize("M,n_i,c", [(2, 3, 1.0), (3, 4, 1.0), (4, 5, 0.5), (1, 3, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_beta_concat_output_on_manifold(M, n_i, c, dtype):
    """Test that beta-concat output lies on the Poincaré ball."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)

    # Create M points on the Poincaré ball of dimension n_i
    points = _make_poincare_points(key, (M, n_i), c, dtype)

    # Apply beta-concat
    result = manifold.beta_concat(points, c)

    # Check output shape
    expected_dim = M * n_i
    assert result.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {result.shape}"

    # Check output is on manifold
    assert manifold.is_in_manifold(result, c), "Beta-concat output not on Poincaré ball"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_beta_concat_single_point(dtype):
    """Test beta-concat with a single point (M=1) is identity."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    n_i, c = 5, 1.0

    # Create single point
    point = _make_poincare_points(key, (n_i,), c, dtype)
    points = point.reshape(1, n_i)

    # Apply beta-concat
    result = manifold.beta_concat(points, c)

    # For M=1, scale = B(n_i/2, 0.5) / B(n_i/2, 0.5) = 1, so result should equal the input
    assert result.shape == (n_i,)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.allclose(result, point, atol=tolerance), f"Single-point beta-concat should be identity: {result} != {point}"


@pytest.mark.parametrize(
    "M,n_i",
    [(2, 3), (3, 4), (5, 6), (4, 2)],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_beta_concat_output_dimension(M, n_i, dtype):
    """Test beta-concat correctly produces output dimension M * n_i."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    c = 1.0

    points = _make_poincare_points(key, (M, n_i), c, dtype)
    result = manifold.beta_concat(points, c)

    expected_dim = M * n_i
    assert result.shape == (expected_dim,), f"Expected ({expected_dim},), got {result.shape}"


# ============================================================================
# HypConv2DPoincare Layer Tests
# ============================================================================
#
# NOTE: The conv layer now returns tangent-space output (matching the reference
# Poincaré ResNet implementation). Tests check for finite output and correct
# shapes, not manifold membership (tangent vectors are unconstrained).


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_output_shape(kernel_size, padding, dtype):
    """Test HypConv2DPoincare output shape with different kernel sizes and padding."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    c = 1.0

    # Create tangent-space input (default input_space="tangent")
    x = _make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=Poincare(dtype=dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
        padding=padding,
    )

    # Forward pass
    y = layer(x, c=c)

    # Check output shape
    if padding == "SAME":
        expected_height, expected_width = height, width
    else:  # VALID
        expected_height = height - kernel_size + 1
        expected_width = width - kernel_size + 1

    assert y.shape == (batch_size, expected_height, expected_width, out_channels), (
        f"Expected shape ({batch_size}, {expected_height}, {expected_width}, {out_channels}), got {y.shape}"
    )
    # Output is tangent space, should be finite
    assert jnp.isfinite(y).all(), "Output contains NaN or Inf"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_output_mappable_to_manifold(dtype):
    """Test that output can be mapped to manifold via expmap_0."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create tangent-space input
    x = _make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    # Forward pass (returns tangent space)
    y = layer(x, c=c)

    # Map to manifold and verify all points are valid
    y_flat = y.reshape(-1, out_channels)
    y_manifold = jax.vmap(manifold.expmap_0, in_axes=(0, None))(y_flat, c)
    is_on_manifold = jax.vmap(lambda p: manifold.is_in_manifold(p, c))(y_manifold)
    assert is_on_manifold.all(), "Output mapped to manifold is not valid"


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_stride(stride, dtype):
    """Test HypConv2DPoincare with different stride values."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    kernel_size = 3
    c = 1.0

    # Create tangent-space input
    x = _make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=Poincare(dtype=dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        rngs=rngs,
        padding="SAME",
    )

    # Forward pass
    y = layer(x, c=c)

    # Check output shape matches expected stride behavior
    expected_height = (height + stride - 1) // stride
    expected_width = (width + stride - 1) // stride
    assert y.shape == (batch_size, expected_height, expected_width, out_channels)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("input_space", ["manifold", "tangent"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_input_space(input_space, dtype):
    """Test HypConv2DPoincare with different input_space settings."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1

    if input_space == "manifold":
        # Project to manifold
        proj_fn = partial(manifold.proj, c=c)
        x_input = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)
    else:
        # Keep in tangent space (small tangent vectors at origin)
        x_input = x

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        input_space=input_space,
    )

    # Forward pass
    y = layer(x_input, c=c)

    # Output is in tangent space, should be finite
    assert jnp.isfinite(y).all(), f"Output contains NaN/Inf for input_space={input_space}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_gradient(dtype):
    """Test HypConv2DPoincare has valid gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create tangent-space input
    x = _make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=Poincare(dtype=dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    # Define loss function
    def loss_fn(model):
        y = model(x, c=c)
        return jnp.sum(y**2)

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    # Check gradients exist and are finite
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert jnp.isfinite(grads.linear.weight[...]).all(), "Weight gradients contain NaN/Inf"
    assert jnp.isfinite(grads.linear.bias[...]).all(), "Bias gradients contain NaN/Inf"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_jitted(dtype):
    """Test HypConv2DPoincare under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create tangent-space input
    x = _make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=Poincare(dtype=dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    # Forward pass
    y = forward(layer, x, c)

    # Check output is finite
    assert jnp.isfinite(y).all(), "JIT output contains NaN/Inf"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_different_curvatures(dtype):
    """Test HypConv2DPoincare with different curvature values."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3

    curvatures = [0.5, 1.0, 2.0]

    for c in curvatures:
        # Create tangent-space input
        x = _make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypConv2DPoincare(
            manifold_module=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            rngs=rngs,
        )

        # Forward pass
        y = layer(x, c=c)

        # Output is tangent space, verify finite
        assert jnp.isfinite(y).all(), f"Output contains NaN/Inf for curvature {c}"

        # Verify output can be mapped to manifold at this curvature
        y_flat = y.reshape(-1, out_channels)
        y_manifold = jax.vmap(manifold.expmap_0, in_axes=(0, None))(y_flat, c)
        is_on = jax.vmap(partial(manifold.is_in_manifold, c=c))(y_manifold)
        assert is_on.all(), f"Mapped output not on manifold for curvature {c}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_chained_tangent_flow(dtype):
    """Test chaining two conv layers with standard relu in tangent space.

    This is the primary use case: conv → relu → conv (all in tangent space).
    """
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width = 2, 8, 8
    c = 1.0

    # Create tangent-space input
    x = _make_tangent_input(key, (batch_size, height, width, 3), dtype)

    # Two conv layers
    rngs = nnx.Rngs(42)
    conv1 = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        rngs=rngs,
        stride=2,
    )
    conv2 = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        rngs=nnx.Rngs(43),
        stride=2,
    )

    # Forward: conv → relu → conv → relu (all tangent space)
    y = conv1(x, c)
    y = jax.nn.relu(y)
    y = conv2(y, c)
    y = jax.nn.relu(y)

    # Check output
    expected_h = (height + 1) // 2  # stride=2 SAME
    expected_h2 = (expected_h + 1) // 2
    assert y.shape == (batch_size, expected_h2, expected_h2, 16)
    assert jnp.isfinite(y).all(), "Chained conv output contains NaN/Inf"

    # Gradients through the chain
    def loss_fn(m1, m2):
        h = m1(x, c)
        h = jax.nn.relu(h)
        h = m2(h, c)
        return jnp.sum(h**2)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=(0, 1))(conv1, conv2)
    grads1, grads2 = grads
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads1.linear.weight[...]).all()
    assert jnp.isfinite(grads2.linear.weight[...]).all()
