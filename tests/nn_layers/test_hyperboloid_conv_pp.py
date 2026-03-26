"""Tests for HypConv2DHyperboloidPP (HNN++ hyperboloid convolutional layer)."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HypConv2DHyperboloidPP


def _proj_image(x, manifold, c):
    """Project each pixel in a (B, H, W, C) feature map to the hyperboloid."""
    return jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)


def _check_on_hyperboloid(x, c, atol=1e-5):
    """Check Minkowski constraint: -x0^2 + ||x_s||^2 = -1/c."""
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_output_shape(kernel_size, padding, dtype):
    """Test HypConv2DHyperboloidPP output shape with different kernel sizes and padding."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidPP(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
        padding=padding,
    )

    y = layer(x_manifold, c=c)

    if padding == "SAME":
        expected_height, expected_width = height, width
    else:
        expected_height = height - kernel_size + 1
        expected_width = width - kernel_size + 1

    assert y.shape == (batch_size, expected_height, expected_width, out_channels)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_output_on_manifold(dtype):
    """Test that all outputs lie on the Hyperboloid manifold."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidPP(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    y = layer(x_manifold, c=c)

    # Flatten to (N, C) and check constraint
    y_flat = y.reshape(-1, out_channels)
    assert _check_on_hyperboloid(y_flat, c=c, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted_forward(dtype):
    """Test HypConv2DHyperboloidPP under nnx.jit."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidPP(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y = forward(layer, x_manifold, c)

    assert y.shape == (batch_size, height, width, out_channels)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradient(dtype):
    """Test HypConv2DHyperboloidPP has valid gradients."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidPP(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    def loss_fn(model):
        y = model(x_manifold, c=c)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted_gradient(dtype):
    """Test HypConv2DHyperboloidPP gradients under nnx.jit."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidPP(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    @nnx.jit
    def loss_fn(module, inputs, curvature):
        y = module(inputs, c=curvature)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, x_manifold, c))(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tangent_input(dtype):
    """Test HypConv2DHyperboloidPP with tangent space input."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    # Tangent vectors at origin (time coordinate is 0)
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x = x.at[:, :, :, 0].set(0.0)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidPP(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        input_space="tangent",
    )

    y = layer(x, c=c)

    assert y.shape == (batch_size, height, width, out_channels)
    assert jnp.isfinite(y).all()
    y_flat = y.reshape(-1, out_channels)
    assert _check_on_hyperboloid(y_flat, c=c, atol=atol)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_stride(stride, dtype):
    """Test HypConv2DHyperboloidPP with different stride values."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    kernel_size = 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidPP(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        rngs=rngs,
    )

    y = layer(x_manifold, c=c)

    expected_height = (height + stride - 1) // stride
    expected_width = (width + stride - 1) // stride
    assert y.shape == (batch_size, expected_height, expected_width, out_channels)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_different_curvatures(dtype):
    """Test HypConv2DHyperboloidPP with different curvature values."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    for c in [0.5, 1.0, 2.0]:
        x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
        proj_fn = partial(manifold.proj, c=c)
        x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn)))(x)

        rngs = nnx.Rngs(42)
        layer = HypConv2DHyperboloidPP(
            manifold_module=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            rngs=rngs,
        )

        y = layer(x_manifold, c=c)

        y_flat = y.reshape(-1, out_channels)
        assert _check_on_hyperboloid(y_flat, c=c, atol=atol), f"Failed for curvature {c}"
