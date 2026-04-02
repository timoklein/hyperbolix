"""Tests for HypLinearHyperboloidFHNN (Chen et al. 2021 hyperboloid linear layer)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import HypLinearHyperboloidFHNN


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


def _check_on_hyperboloid(x, c, atol=1e-5):
    """Check Minkowski constraint: -x0^2 + ||x_s||^2 = -1/c."""
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_forward_shape(dtype):
    """Test HypLinearHyperboloidFHNN output shape and finiteness."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_on_manifold(dtype):
    """Test HypLinearHyperboloidFHNN output lies on the hyperboloid."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=1.0)

    assert _check_on_hyperboloid(y, c=1.0, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted_forward(dtype):
    """Test HypLinearHyperboloidFHNN under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y = forward(layer, x, 1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradient(dtype):
    """Test HypLinearHyperboloidFHNN has valid gradients (kernel, bias, scale)."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()
    assert jnp.isfinite(grads.scale[...])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted_gradient(dtype):
    """Test HypLinearHyperboloidFHNN gradients under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def loss_fn(module, inputs, curvature):
        y = module(inputs, c=curvature)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, x, 1.0))(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()
    assert jnp.isfinite(grads.scale[...])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tangent_input(dtype):
    """Test HypLinearHyperboloidFHNN with tangent space input."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    # Create tangent vector at origin (time coordinate is 0)
    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    v = v.at[:, 0].set(0.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs, input_space="tangent")

    y = layer(v, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()
    assert _check_on_hyperboloid(y, c=1.0, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_time_floor(dtype, c):
    """Test that FHNN time coordinate y0 > 1/sqrt(c) always holds."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 16, 6, 10

    # Use larger magnitude inputs to stress-test the floor
    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 1.0
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, c)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=c)

    assert (y[:, 0] >= 1.0 / jnp.sqrt(c)).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_activation_and_dropout(dtype):
    """Test HypLinearHyperboloidFHNN with activation and dropout."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(params=42, dropout=43)
    layer = HypLinearHyperboloidFHNN(
        get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs, activation=jax.nn.relu, dropout_rate=0.1
    )

    # deterministic=False (training mode)
    y_train = layer(x, c=1.0, deterministic=False)
    assert y_train.shape == (batch_size, out_dim)
    assert jnp.isfinite(y_train).all()
    assert _check_on_hyperboloid(y_train, c=1.0, atol=atol)

    # deterministic=True (eval mode)
    y_eval = layer(x, c=1.0, deterministic=True)
    assert y_eval.shape == (batch_size, out_dim)
    assert jnp.isfinite(y_eval).all()
    assert _check_on_hyperboloid(y_eval, c=1.0, atol=atol)


def test_init_time_column_zeroed():
    """Test that FHNN initializes kernel time column (column 0) to zero."""
    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(jnp.float32), 6, 10, rngs=rngs)

    assert jnp.allclose(layer.kernel[...][:, 0], 0.0)
