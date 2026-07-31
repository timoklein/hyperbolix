"""Tests for HypLinearHyperboloidFHNN (Chen et al. 2021 hyperboloid linear layer).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypLinearHyperboloidFHNN-specific tests stay here.
"""

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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_fhnn_fhcnn_gradients_at_zero_spatial_norm(dtype):
    """Gradients are finite when the linear output's spatial part is exactly 0.

    Regression guard: both forwards divided by an unguarded ``linalg.norm`` of
    the spatial part and masked the result *afterwards* with ``jnp.where`` —
    the norm's NaN VJP at zero survives that masking. A zero kernel + zero
    bias forces every row through the singular point.
    """
    from hyperbolix.nn_layers.hyperboloid_linear import _fhcnn_forward, _fhnn_forward

    manifold = get_hyperboloid(dtype)
    batch_size, in_dim, out_dim = 2, 4, 4
    kernel_OI = jnp.zeros((out_dim, in_dim), dtype=dtype)
    bias_1O = jnp.zeros((1, out_dim), dtype=dtype)
    x_BI = jnp.ones((batch_size, in_dim), dtype=dtype) * 0.3
    scale = jnp.asarray(1.0, dtype=dtype)

    def loss_fhcnn(kernel):
        out = _fhcnn_forward(x_BI, kernel, bias_1O, manifold, 1.0, "tangent", None, True, scale, 1e-5)
        return jnp.sum(out)

    def loss_fhnn(kernel):
        out = _fhnn_forward(x_BI, kernel, bias_1O, manifold, 1.0, "tangent", None, None, scale, 1e-5)
        return jnp.sum(out)

    assert jnp.all(jnp.isfinite(jax.grad(loss_fhcnn)(kernel_OI)))
    assert jnp.all(jnp.isfinite(jax.grad(loss_fhnn)(kernel_OI)))

    # Forward: zero spatial rows still map to the hyperboloid origin
    out = _fhcnn_forward(x_BI, kernel_OI, bias_1O, manifold, 1.0, "tangent", None, True, scale, 1e-5)
    origin = jnp.concatenate([jnp.ones((1,), dtype=dtype), jnp.zeros((out_dim - 1,), dtype=dtype)])
    assert jnp.allclose(out, origin, atol=1e-6)
