"""Tests for hyperbolic neural network layers."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import hyperbolix_jax.manifolds.hyperboloid as hyperboloid
import hyperbolix_jax.manifolds.poincare as poincare
from hyperbolix_jax.nn_layers import (
    Expmap,
    Expmap0,
    HyperbolicActivation,
    HypLinearHyperboloid,
    HypLinearPoincare,
    HypLinearPoincarePP,
    Logmap,
    Logmap0,
    Proj,
    Retraction,
    TanProj,
)

# Enable float64 for tests
jax.config.update("jax_enable_x64", True)


class TestStandardLayers:
    """Test standard hyperbolic layers (wrappers)."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("manifold_module", [poincare, hyperboloid])
    def test_expmap0_logmap0_inverse(self, dtype, manifold_module):
        """Test that Expmap0 and Logmap0 are inverses."""
        # Create points on manifold
        key = jax.random.PRNGKey(42)
        v = jax.random.normal(key, (5, 3), dtype=dtype) * 0.1

        # Create layers
        expmap0 = Expmap0(manifold_module)
        logmap0 = Logmap0(manifold_module)

        # Test: expmap0(logmap0(expmap0(v))) ≈ expmap0(v)
        x = expmap0(v, c=1.0)
        v_recovered = logmap0(x, c=1.0)
        x_recovered = expmap0(v_recovered, c=1.0)

        assert jnp.allclose(x, x_recovered, atol=1e-5)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_proj_keeps_on_manifold(self, dtype):
        """Test that Proj projects points onto the manifold."""
        key = jax.random.PRNGKey(42)
        # Create points slightly off the manifold
        x = jax.random.normal(key, (5, 3), dtype=dtype) * 0.5

        proj = Proj(poincare)
        x_proj = proj(x, c=1.0)

        # Check that projected points are on the manifold
        assert poincare.is_in_manifold(x_proj, c=1.0, axis=-1)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyperbolic_activation_poincare(self, dtype):
        """Test HyperbolicActivation with Poincaré ball."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (5, 3), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer with ReLU activation
        hyp_act = HyperbolicActivation(poincare, jax.nn.relu)

        # Apply activation
        y = hyp_act(x, c=1.0)

        # Check output is on manifold
        assert poincare.is_in_manifold(y, c=1.0, axis=-1)
        # Check output shape is preserved
        assert y.shape == x.shape


class TestPoincareLinearLayers:
    """Test Poincaré ball linear layers."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_linear_poincare_forward(self, dtype):
        """Test HypLinearPoincare forward pass."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 8, 5, 3

        # Create input on manifold
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypLinearPoincare(poincare, in_dim, out_dim, rngs=rngs)

        # Forward pass
        y = layer(x, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is on manifold
        assert poincare.is_in_manifold(y, c=1.0, axis=-1)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_linear_poincare_gradient(self, dtype):
        """Test HypLinearPoincare has valid gradients."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 4, 5, 3

        # Create input
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

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
    def test_hyp_linear_poincare_pp_forward(self, dtype):
        """Test HypLinearPoincarePP forward pass."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 8, 5, 3

        # Create input on manifold
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypLinearPoincarePP(poincare, in_dim, out_dim, rngs=rngs)

        # Forward pass
        y = layer(x, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is on manifold
        assert poincare.is_in_manifold(y, c=1.0, axis=-1)


class TestHyperboloidLinearLayers:
    """Test Hyperboloid linear layers."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_linear_hyperboloid_forward(self, dtype):
        """Test HypLinearHyperboloid forward pass."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 8, 5, 4

        # Create input on manifold
        v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = hyperboloid.expmap_0(v, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypLinearHyperboloid(hyperboloid, in_dim, out_dim, rngs=rngs)

        # Forward pass
        y = layer(x, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is on manifold
        assert hyperboloid.is_in_manifold(y, c=1.0, axis=-1)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_linear_hyperboloid_gradient(self, dtype):
        """Test HypLinearHyperboloid has valid gradients."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 4, 5, 4

        # Create input
        v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = hyperboloid.expmap_0(v, c=1.0, axis=-1)

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
    def test_hyp_linear_hyperboloid_tangent_input(self, dtype):
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
        # Check output is on manifold
        assert hyperboloid.is_in_manifold(y, c=1.0, axis=-1)


class TestLayerComposition:
    """Test composing multiple layers."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_two_layer_network_poincare(self, dtype):
        """Test a simple 2-layer hyperbolic network."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, hidden_dim, out_dim = 4, 5, 8, 3

        # Create input
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layers
        rngs = nnx.Rngs(42)
        layer1 = HypLinearPoincare(poincare, in_dim, hidden_dim, rngs=rngs)
        activation = HyperbolicActivation(poincare, jax.nn.relu)
        layer2 = HypLinearPoincare(poincare, hidden_dim, out_dim, rngs=rngs)

        # Forward pass
        h = layer1(x, c=1.0)
        h = activation(h, c=1.0)
        y = layer2(h, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is on manifold
        assert poincare.is_in_manifold(y, c=1.0, axis=-1)
