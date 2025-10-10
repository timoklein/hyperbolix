"""Tests for hyperbolic neural network layers."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import hyperbolix_jax.manifolds.hyperboloid as hyperboloid
import hyperbolix_jax.manifolds.poincare as poincare
from hyperbolix_jax.nn_layers import (
    HypLinearHyperboloid,
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
    assert jnp.isfinite(grads.weight.value).all()
    assert jnp.isfinite(grads.bias.value).all()


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
def test_hyp_linear_hyperboloid_forward(dtype):
    """Test HypLinearHyperboloid forward pass."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 5, 4

    # Create input on manifold
    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None, None))(v, 1.0, True)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloid(hyperboloid, in_dim, out_dim, rngs=rngs)

    # Forward pass
    y = layer(x, c=1.0)

    # Check output shape
    assert y.shape == (batch_size, out_dim)
    # Check output is on manifold (vmap over batch)
    assert jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_hyperboloid_gradient(dtype):
    """Test HypLinearHyperboloid has valid gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 5, 4

    # Create input
    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None, None))(v, 1.0, True)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloid(hyperboloid, in_dim, out_dim, rngs=rngs)

    # Define loss function
    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    # Check gradients exist and are finite
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.weight.value).all()
    assert jnp.isfinite(grads.bias.value).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_hyperboloid_tangent_input(dtype):
    """Test HypLinearHyperboloid with tangent space input."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 5, 4

    # Create tangent vector at origin
    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    # Ensure it's a valid tangent vector (time coordinate is 0)
    v = v.at[:, 0].set(0.0)

    # Create layer with tangent input
    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloid(hyperboloid, in_dim, out_dim, rngs=rngs, input_space="tangent")

    # Forward pass
    y = layer(v, c=1.0)

    # Check output shape
    assert y.shape == (batch_size, out_dim)
    # Check output is on manifold (vmap over batch)
    assert jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0).all()
