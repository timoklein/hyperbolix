"""Tests for hyperbolic neural network layers.

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypLinearPoincare / HypLinearPoincarePP-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import (
    HypLinearPoincare,
    HypLinearPoincarePP,
)


def _mobius_linear_reference(x_BI, kernel_OI, bias_O, c):
    """NumPy transcription of the Ganea et al. 2018 Mobius matvec + bias.

    Independent of the library --- the gyrovector formulas written out directly::

        log_0(x) = atanh(sqrt(c)||x||) * x / (sqrt(c)||x||)
        z        = W log_0(x)
        exp_0(z) = tanh(sqrt(c)||z||) * z / (sqrt(c)||z||)
        y        = exp_0(z) (+)_c b        (Mobius addition)
    """
    x_BI = np.asarray(x_BI, dtype=np.float64)
    w_OI = np.asarray(kernel_OI, dtype=np.float64)
    b_O = np.asarray(bias_O, dtype=np.float64)
    sqrt_c = np.sqrt(c)

    xn_B1 = np.linalg.norm(x_BI, axis=-1, keepdims=True)
    u_BI = np.arctanh(sqrt_c * xn_B1) * x_BI / (sqrt_c * xn_B1)

    z_BO = u_BI @ w_OI.T
    zn_B1 = np.linalg.norm(z_BO, axis=-1, keepdims=True)
    p_BO = np.tanh(sqrt_c * zn_B1) * z_BO / (sqrt_c * zn_B1)

    # Mobius addition p (+)_c b
    pb_B1 = np.sum(p_BO * b_O[None, :], axis=-1, keepdims=True)
    p2_B1 = np.sum(p_BO**2, axis=-1, keepdims=True)
    b2 = float(np.sum(b_O**2))
    num_BO = (1.0 + 2.0 * c * pb_B1 + c * b2) * p_BO + (1.0 - c * p2_B1) * b_O[None, :]
    den_B1 = 1.0 + 2.0 * c * pb_B1 + c**2 * p2_B1 * b2
    return num_BO / den_B1


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_hyp_linear_poincare_matches_mobius_transcription(c):
    """HypLinearPoincare equals ``exp_0(W log_0(x)) (+) b`` transcribed in NumPy.

    Value oracle: without it the layer was covered only by shape / on-manifold /
    finite-gradient checks, all of which a zeroed Mobius matvec (a constant layer
    returning the bias for every input) satisfies.
    """
    dtype = jnp.float64
    manifold = Poincare(dtype=dtype)
    batch_size, in_dim, out_dim = 6, 5, 4

    v = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_dim), dtype=dtype) * 0.3
    x = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, c)

    layer = HypLinearPoincare(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), curvature=c, param_dtype=dtype)
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(1), (out_dim, in_dim), dtype=dtype) * 0.5
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(2), (out_dim,), dtype=dtype) * 0.1

    y = layer(x, c=c)
    expected = _mobius_linear_reference(x, layer.kernel[...], layer.bias[...], c)

    assert np.allclose(np.asarray(y), expected, atol=1e-12)


def test_hyp_linear_poincare_output_depends_on_input():
    """Two distinct inputs give distinct outputs (constant-collapse guard)."""
    dtype = jnp.float64
    c = 1.0
    manifold = Poincare(dtype=dtype)
    in_dim, out_dim = 5, 4

    v = jax.random.normal(jax.random.PRNGKey(3), (2, in_dim), dtype=dtype) * 0.4
    x = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, c)

    layer = HypLinearPoincare(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), curvature=c, param_dtype=dtype)
    y = layer(x, c=c)

    assert float(jnp.max(jnp.abs(y[0] - y[1]))) > 1e-6


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
