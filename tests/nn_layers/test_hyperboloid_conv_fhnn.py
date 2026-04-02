"""Tests for HypConv2DHyperboloidFHNN (Chen et al. 2021 hyperboloid conv layer)."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HypConv2DHyperboloidFHNN


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_output_shape(kernel_size, padding, dtype):
    """Test HypConv2DHyperboloidFHNN output shape with different kernel sizes and padding."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidFHNN(
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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_output_on_manifold(dtype):
    """Test that all outputs lie on the Hyperboloid manifold."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidFHNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        padding="SAME",
    )

    y = layer(x_manifold, c=c)

    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.is_in_manifold(p, c))))(y)
    assert is_on_manifold.all(), "Not all outputs lie on the manifold"


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_stride(stride, dtype):
    """Test HypConv2DHyperboloidFHNN with different stride values."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    kernel_size = 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidFHNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        rngs=rngs,
        padding="SAME",
    )

    y = layer(x_manifold, c=c)

    expected_height = (height + stride - 1) // stride
    expected_width = (width + stride - 1) // stride
    assert y.shape == (batch_size, expected_height, expected_width, out_channels)


@pytest.mark.parametrize("input_space", ["manifold", "tangent"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_input_space(input_space, dtype):
    """Test HypConv2DHyperboloidFHNN with different input_space settings."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1

    if input_space == "manifold":
        x_input = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)
    else:
        x_input = x.at[:, :, :, 0].set(0)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidFHNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        input_space=input_space,
    )

    y = layer(x_input, c=c)

    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.is_in_manifold(p, c))))(y)
    assert is_on_manifold.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradient(dtype):
    """Test HypConv2DHyperboloidFHNN has valid gradients (kernel, bias, scale)."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidFHNN(
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
    assert jnp.isfinite(grads.scale[...])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted(dtype):
    """Test HypConv2DHyperboloidFHNN under nnx.jit."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidFHNN(
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

    is_on_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.is_in_manifold(p, c))))(y)
    assert is_on_manifold.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_different_curvatures(dtype):
    """Test HypConv2DHyperboloidFHNN with different curvature values."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3

    for c in [0.5, 1.0, 2.0]:
        x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
        proj_fn = partial(manifold.proj, c=c)
        x_manifold = jax.vmap(jax.vmap(jax.vmap(proj_fn)))(x)

        rngs = nnx.Rngs(42)
        layer = HypConv2DHyperboloidFHNN(
            manifold_module=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            rngs=rngs,
        )

        y = layer(x_manifold, c=c)

        is_in_manifold_fn = partial(manifold.is_in_manifold, c=c)
        is_on_manifold = jax.vmap(jax.vmap(jax.vmap(is_in_manifold_fn)))(y)
        assert is_on_manifold.all(), f"Failed for curvature {c}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_time_floor(dtype):
    """Test that FHNN conv time coordinate y0 >= 1/sqrt(c) always holds."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidFHNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    y = layer(x_manifold, c=c)

    assert (y[..., 0] >= 1.0 / jnp.sqrt(c)).all()
