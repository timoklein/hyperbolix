"""Tests for Poincaré convolutional layers and beta-concatenation operation."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers.poincare_conv import HypConv2DPoincare

# Enable float64 for tests
jax.config.update("jax_enable_x64", True)


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


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_output_shape(kernel_size, padding, dtype):
    """Test HypConv2DPoincare output shape with different kernel sizes and padding."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    c = 1.0

    # Create input on Poincaré ball
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(manifold.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
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
def test_hypconv_poincare_output_on_manifold(dtype):
    """Test that all outputs lie on the Poincaré ball."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(manifold.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
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
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.is_in_manifold(p, c))))(y)
    assert is_on_manifold.all(), "Not all outputs lie on the Poincaré ball"


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_stride(stride, dtype):
    """Test HypConv2DPoincare with different stride values."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    kernel_size = 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(manifold.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
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

    # Check outputs are on manifold
    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.is_in_manifold(p, c))))(y)
    assert is_on_manifold.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_gradient(dtype):
    """Test HypConv2DPoincare has valid gradients."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(manifold.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
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
    assert jnp.isfinite(grads.linear.weight[...]).all()
    assert jnp.isfinite(grads.linear.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_jitted(dtype):
    """Test HypConv2DPoincare under nnx.jit."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    proj_fn = partial(manifold.proj, c=c)
    x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
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
def test_hypconv_poincare_different_curvatures(dtype):
    """Test HypConv2DPoincare with different curvature values."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3

    curvatures = [0.5, 1.0, 2.0]

    for c in curvatures:
        # Create input
        x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
        proj_fn = partial(manifold.proj, c=c)
        x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)

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
        y = layer(x_manifold, c=c)

        # Check outputs are on manifold with correct curvature
        is_in_manifold_fn = partial(manifold.is_in_manifold, c=c)
        is_on_manifold = jax.vmap(jax.vmap(jax.vmap(is_in_manifold_fn)))(y)
        assert is_on_manifold.all(), f"Failed for curvature {c}"
