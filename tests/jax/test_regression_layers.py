"""Tests for hyperbolic regression neural network layers."""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import hyperbolix_jax.manifolds.hyperboloid as hyperboloid
import hyperbolix_jax.manifolds.poincare as poincare
from hyperbolix_jax.nn_layers import (
    HypRegressionHyperboloid,
    HypRegressionPoincare,
    HypRegressionPoincarePP,
    HypRegressionPoincareHDRL,
)

# Enable float64 for tests
jax.config.update("jax_enable_x64", True)


class TestPoincareRegressionLayers:
    """Test Poincaré ball regression layers."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_regression_poincare_forward(self, dtype):
        """Test HypRegressionPoincare forward pass."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 8, 5, 3

        # Create input on manifold
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypRegressionPoincare(poincare, in_dim, out_dim, rngs=rngs)

        # Forward pass
        y = layer(x, c=1.0)

        # Check output shape (regression scores, not on manifold)
        assert y.shape == (batch_size, out_dim)
        # Check output is finite
        assert jnp.isfinite(y).all()

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_regression_poincare_gradient(self, dtype):
        """Test HypRegressionPoincare has valid gradients."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 4, 5, 3

        # Create input
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypRegressionPoincare(poincare, in_dim, out_dim, rngs=rngs)

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
    def test_hyp_regression_poincare_pp_forward(self, dtype):
        """Test HypRegressionPoincarePP forward pass."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 8, 5, 3

        # Create input on manifold
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypRegressionPoincarePP(poincare, in_dim, out_dim, rngs=rngs)

        # Forward pass
        y = layer(x, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is finite
        assert jnp.isfinite(y).all()

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_regression_poincare_pp_gradient(self, dtype):
        """Test HypRegressionPoincarePP has valid gradients."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 4, 5, 3

        # Create input
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypRegressionPoincarePP(poincare, in_dim, out_dim, rngs=rngs)

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
    @pytest.mark.parametrize("version", ["standard", "rs"])
    def test_hyp_regression_poincare_hdrl_forward(self, dtype, version):
        """Test HypRegressionPoincareHDRL forward pass."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 8, 5, 3

        # Create input on manifold
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypRegressionPoincareHDRL(poincare, in_dim, out_dim, rngs=rngs, version=version)

        # Forward pass
        y = layer(x, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is finite
        assert jnp.isfinite(y).all()

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_regression_poincare_hdrl_gradient(self, dtype):
        """Test HypRegressionPoincareHDRL has valid gradients."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 4, 5, 3

        # Create input
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypRegressionPoincareHDRL(poincare, in_dim, out_dim, rngs=rngs)

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


class TestHyperboloidRegressionLayers:
    """Test Hyperboloid regression layers."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_regression_hyperboloid_forward(self, dtype):
        """Test HypRegressionHyperboloid forward pass."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 8, 5, 3

        # Create input on manifold
        v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = hyperboloid.expmap_0(v, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypRegressionHyperboloid(hyperboloid, in_dim, out_dim, rngs=rngs)

        # Forward pass
        y = layer(x, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is finite
        assert jnp.isfinite(y).all()

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_hyp_regression_hyperboloid_gradient(self, dtype):
        """Test HypRegressionHyperboloid has valid gradients."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 4, 5, 3

        # Create input
        v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = hyperboloid.expmap_0(v, c=1.0, axis=-1)

        # Create layer
        rngs = nnx.Rngs(42)
        layer = HypRegressionHyperboloid(hyperboloid, in_dim, out_dim, rngs=rngs)

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
    def test_hyp_regression_hyperboloid_tangent_input(self, dtype):
        """Test HypRegressionHyperboloid with tangent space input."""
        key = jax.random.PRNGKey(42)
        batch_size, in_dim, out_dim = 8, 5, 3

        # Create tangent vector at origin
        v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        # Ensure it's a valid tangent vector (time coordinate is 0)
        v = v.at[:, 0].set(0.0)

        # Create layer with tangent input
        rngs = nnx.Rngs(42)
        layer = HypRegressionHyperboloid(hyperboloid, in_dim, out_dim, rngs=rngs, input_space="tangent")

        # Forward pass
        y = layer(v, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is finite
        assert jnp.isfinite(y).all()


class TestRegressionLayerComposition:
    """Test composing regression layers with other layers."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_linear_then_regression_poincare(self, dtype):
        """Test linear layer followed by regression layer."""
        from hyperbolix_jax.nn_layers import HypLinearPoincare

        key = jax.random.PRNGKey(42)
        batch_size, in_dim, hidden_dim, out_dim = 4, 5, 8, 3

        # Create input
        x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
        x = poincare.proj(x, c=1.0, axis=-1)

        # Create layers
        rngs = nnx.Rngs(42)
        linear = HypLinearPoincare(poincare, in_dim, hidden_dim, rngs=rngs)
        regression = HypRegressionPoincarePP(poincare, hidden_dim, out_dim, rngs=rngs)

        # Forward pass
        h = linear(x, c=1.0)
        y = regression(h, c=1.0)

        # Check output shape
        assert y.shape == (batch_size, out_dim)
        # Check output is finite
        assert jnp.isfinite(y).all()
