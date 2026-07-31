"""Tests for hyperbolic regression neural network layers.

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypRegression{Poincare,PoincarePP,Hyperboloid}-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import (
    HypRegressionHyperboloid,
    HypRegressionPoincarePP,
)


def get_poincare(dtype: jnp.dtype) -> Poincare:
    """Get dtype-specific Poincaré manifold instance."""
    return Poincare(dtype=dtype)


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_regression_hyperboloid_kernel_init_std(dtype):
    """Kernel init is fan-scaled by the spatial fan-in (in_dim - 1); an unscaled N(0,1)
    would give row norms ~= sqrt(in_dim - 1), overwhelming the MLR output scaling."""
    in_dim, out_dim = 65, 10
    rngs = nnx.Rngs(42)
    layer = HypRegressionHyperboloid(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    kernel_std = jnp.std(layer.kernel[...])
    expected_std = 1.0 / jnp.sqrt(2.0 * (in_dim - 1) * out_dim)
    assert jnp.abs(kernel_std - expected_std) < 0.2 * expected_std


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_linear_then_regression_poincare(dtype):
    """Test linear layer followed by regression layer."""
    from hyperbolix.nn_layers import HypLinearPoincare

    key = jax.random.PRNGKey(42)
    batch_size, in_dim, hidden_dim, out_dim = 4, 5, 8, 3

    # Create input
    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = get_poincare(dtype).proj(x, c=1.0)

    # Create layers
    rngs = nnx.Rngs(42)
    linear = HypLinearPoincare(get_poincare(dtype), in_dim, hidden_dim, rngs=rngs)
    regression = HypRegressionPoincarePP(get_poincare(dtype), hidden_dim, out_dim, rngs=rngs)

    # Forward pass
    h = linear(x, c=1.0)
    y = regression(h, c=1.0)

    # Check output shape
    assert y.shape == (batch_size, out_dim)
    # Check output is finite
    assert jnp.isfinite(y).all()
