"""Tests for HypLinearHyperboloidPP (HNN++ hyperboloid linear layer)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import HypLinearHyperboloidPP


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


def _check_on_hyperboloid(x, c, atol=1e-5):
    """Check Minkowski constraint: -x0^2 + ||x_s||^2 = -1/c."""
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_forward_shape(dtype):
    """Test HypLinearHyperboloidPP output shape and finiteness."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPP(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_on_manifold(dtype):
    """Test HypLinearHyperboloidPP output lies on the hyperboloid."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPP(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=1.0)

    assert _check_on_hyperboloid(y, c=1.0, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted_forward(dtype):
    """Test HypLinearHyperboloidPP under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPP(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y = forward(layer, x, 1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradient(dtype):
    """Test HypLinearHyperboloidPP has valid gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPP(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted_gradient(dtype):
    """Test HypLinearHyperboloidPP gradients under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPP(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def loss_fn(module, inputs, curvature):
        y = module(inputs, c=curvature)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, x, 1.0))(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tangent_input(dtype):
    """Test HypLinearHyperboloidPP with tangent space input."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    # Create tangent vector at origin (time coordinate is 0)
    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    v = v.at[:, 0].set(0.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPP(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs, input_space="tangent")

    y = layer(v, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()
    assert _check_on_hyperboloid(y, c=1.0, atol=atol)
