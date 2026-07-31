"""Tests for hyperbolic neural network layers.

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypLinearPoincare / HypLinearPoincarePP-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import (
    HypLinearPoincarePP,
)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_linear_poincare_pp_jitted_gradient(dtype):
    """Test HypLinearPoincarePP gradients under nnx.jit."""
    poincare = Poincare(dtype=dtype)
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
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()
