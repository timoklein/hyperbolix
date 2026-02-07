"""Tests for hyperbolic neural network layers."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import hyperbolix.manifolds.poincare as poincare
from hyperbolix.nn_layers import (
    HypLinearPoincare,
    HypLinearPoincarePP,
)

# Enable float64 for tests
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_poincare_forward(dtype):
    """Test HypLinearPoincare forward pass."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 5, 3

    # Create input on manifold
    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypLinearPoincare(poincare, in_dim, out_dim, rngs=rngs)

    # Forward pass
    y = layer(x, c=1.0)

    # Check output shape
    assert y.shape == (batch_size, out_dim)
    # Check output is on manifold (vmap over batch)
    assert jax.vmap(poincare.is_in_manifold, in_axes=(0, None))(y, 1.0).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_poincare_jitted_forward(dtype):
    """Test HypLinearPoincare forward pass under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 5, 3

    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearPoincare(poincare, in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y = forward(layer, x, 1.0)

    assert y.shape == (batch_size, out_dim)
    assert jax.vmap(poincare.is_in_manifold, in_axes=(0, None))(y, 1.0).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_poincare_gradient(dtype):
    """Test HypLinearPoincare has valid gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 5, 3

    # Create input
    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypLinearPoincare(poincare, in_dim, out_dim, rngs=rngs)

    # Define loss function
    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    # Check gradients exist and are finite
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.weight[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_poincare_jitted_gradient(dtype):
    """Test HypLinearPoincare gradients under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 5, 3

    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearPoincare(poincare, in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def loss_fn(module, inputs, curvature):
        y = module(inputs, c=curvature)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, x, 1.0))(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.weight[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_poincare_pp_forward(dtype):
    """Test HypLinearPoincarePP forward pass."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 5, 3

    # Create input on manifold
    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypLinearPoincarePP(poincare, in_dim, out_dim, rngs=rngs)

    # Forward pass
    y = layer(x, c=1.0)

    # Check output shape
    assert y.shape == (batch_size, out_dim)
    # Check output is on manifold (vmap over batch)
    assert jax.vmap(poincare.is_in_manifold, in_axes=(0, None))(y, 1.0).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_poincare_pp_jitted_forward(dtype):
    """Test HypLinearPoincarePP forward pass under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 5, 3

    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearPoincarePP(poincare, in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y = forward(layer, x, 1.0)

    assert y.shape == (batch_size, out_dim)
    assert jax.vmap(poincare.is_in_manifold, in_axes=(0, None))(y, 1.0).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_poincare_pp_jitted_gradient(dtype):
    """Test HypLinearPoincarePP gradients under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 5, 3

    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearPoincarePP(poincare, in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def loss_fn(module, inputs, curvature):
        y = module(inputs, c=curvature)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, x, 1.0))(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.weight[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()
