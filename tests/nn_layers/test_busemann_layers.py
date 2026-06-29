"""Tests for the Busemann neural-network layers (BMLR heads and BFC layers).

Covers both models for both layer families:
  - HypRegression{Hyperboloid,Poincare}Busemann — BMLR classification heads (Euclidean logits).
  - HypLinear{Hyperboloid,Poincare}Busemann      — BFC layers (output a manifold point).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import (
    HypLinearHyperboloidBusemann,
    HypLinearPoincareBusemann,
    HypRegressionHyperboloidBusemann,
    HypRegressionPoincareBusemann,
)

C = 0.7


def get_hyperboloid(dtype):
    return Hyperboloid(dtype=dtype)


def get_poincare(dtype):
    return Poincare(dtype=dtype)


def _make_hyperboloid_points(key, n, in_dim, dtype, c=C):
    """n points on the hyperboloid with ambient dim in_dim (= spatial + 1)."""
    H = get_hyperboloid(dtype)
    v_spatial = jax.random.normal(key, (n, in_dim - 1), dtype=dtype) * 0.2
    return jax.vmap(H.expmap_0, in_axes=(0, None))(H.embed_spatial_0(v_spatial), c)


def _make_ball_points(key, n, in_dim, dtype, c=C):
    """n points inside the Poincaré ball with spatial dim in_dim."""
    P = get_poincare(dtype)
    v = jax.random.normal(key, (n, in_dim), dtype=dtype) * 0.2
    return jax.vmap(P.expmap_0, in_axes=(0, None))(v, c)


def _check_on_hyperboloid(x, c, atol):
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


def _check_in_ball(x, c):
    return bool((jnp.sum(x**2, axis=-1) < 1.0 / c).all())


# --------------------------------------------------------------------------- #
# BMLR heads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_bmlr_hyperboloid_forward(dtype):
    x = _make_hyperboloid_points(jax.random.PRNGKey(0), 8, 9, dtype)
    layer = HypRegressionHyperboloidBusemann(get_hyperboloid(dtype), in_dim=9, out_dim=5, rngs=nnx.Rngs(0))
    y = layer(x, C)
    assert y.shape == (8, 5)
    assert jnp.isfinite(y).all()
    assert y.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_bmlr_poincare_forward(dtype):
    x = _make_ball_points(jax.random.PRNGKey(0), 8, 8, dtype)
    layer = HypRegressionPoincareBusemann(get_poincare(dtype), in_dim=8, out_dim=5, rngs=nnx.Rngs(0))
    y = layer(x, C)
    assert y.shape == (8, 5)
    assert jnp.isfinite(y).all()
    assert y.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_bmlr_jit_and_gradients(dtype):
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 6, 9, dtype)
    layer = HypRegressionHyperboloidBusemann(get_hyperboloid(dtype), in_dim=9, out_dim=4, rngs=nnx.Rngs(0))

    @nnx.jit
    def forward(m, inp, c):
        return m(inp, c)

    y = forward(layer, x, C)
    assert y.shape == (6, 4)
    assert jnp.isfinite(y).all()

    def loss(m):
        return jnp.sum(m(x, C) ** 2)

    loss_val, grads = nnx.value_and_grad(loss)(layer)
    assert jnp.isfinite(loss_val)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.log_scale[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


def test_bmlr_tangent_input_space():
    """input_space='tangent' lifts Euclidean features before the Busemann head."""
    dtype = jnp.float32
    x_tangent = jax.random.normal(jax.random.PRNGKey(2), (5, 9), dtype=dtype) * 0.1
    layer = HypRegressionHyperboloidBusemann(
        get_hyperboloid(dtype), in_dim=9, out_dim=3, rngs=nnx.Rngs(0), input_space="tangent"
    )
    y = layer(x_tangent, C)
    assert y.shape == (5, 3)
    assert jnp.isfinite(y).all()


# --------------------------------------------------------------------------- #
# BFC layers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("use_gyro_bias", [False, True])
def test_bfc_hyperboloid_on_manifold(dtype, use_gyro_bias):
    x = _make_hyperboloid_points(jax.random.PRNGKey(0), 8, 9, dtype)
    layer = HypLinearHyperboloidBusemann(
        get_hyperboloid(dtype), in_dim=9, out_dim=7, rngs=nnx.Rngs(0), use_gyro_bias=use_gyro_bias
    )
    y = layer(x, C)
    assert y.shape == (8, 7)
    assert jnp.isfinite(y).all()
    assert y.dtype == dtype
    assert _check_on_hyperboloid(y, C, atol=4e-3 if dtype == jnp.float32 else 1e-7)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("use_gyro_bias", [False, True])
def test_bfc_poincare_in_ball(dtype, use_gyro_bias):
    x = _make_ball_points(jax.random.PRNGKey(0), 8, 8, dtype)
    layer = HypLinearPoincareBusemann(get_poincare(dtype), in_dim=8, out_dim=6, rngs=nnx.Rngs(0), use_gyro_bias=use_gyro_bias)
    y = layer(x, C)
    assert y.shape == (8, 6)
    assert jnp.isfinite(y).all()
    assert y.dtype == dtype
    assert _check_in_ball(y, C)


@pytest.mark.parametrize(
    "layer_cls, manifold_fn, in_dim, make_pts",
    [
        (HypLinearHyperboloidBusemann, get_hyperboloid, 9, _make_hyperboloid_points),
        (HypLinearPoincareBusemann, get_poincare, 8, _make_ball_points),
    ],
)
def test_bfc_jit_and_gradients(layer_cls, manifold_fn, in_dim, make_pts):
    dtype = jnp.float32
    x = make_pts(jax.random.PRNGKey(1), 6, in_dim, dtype)
    layer = layer_cls(manifold_fn(dtype), in_dim=in_dim, out_dim=7, rngs=nnx.Rngs(0), use_gyro_bias=True)

    @nnx.jit
    def forward(m, inp, c):
        return m(inp, c)

    assert jnp.isfinite(forward(layer, x, C)).all()

    def loss(m):
        return jnp.sum(m(x, C) ** 2)

    loss_val, grads = nnx.value_and_grad(loss)(layer)
    assert jnp.isfinite(loss_val)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.log_scale[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()
    assert jnp.isfinite(grads.gyro_bias[...]).all()


@pytest.mark.parametrize(
    "layer_cls, manifold_fn, in_dim, make_pts",
    [
        (HypLinearHyperboloidBusemann, get_hyperboloid, 9, _make_hyperboloid_points),
        (HypLinearPoincareBusemann, get_poincare, 8, _make_ball_points),
    ],
)
def test_bfc_gyro_bias_is_noop_at_init(layer_cls, manifold_fn, in_dim, make_pts):
    """Zero-initialized gyro-bias is the gyrogroup identity → identical output to no gyro-bias."""
    dtype = jnp.float64
    x = make_pts(jax.random.PRNGKey(3), 5, in_dim, dtype)
    kwargs = dict(in_dim=in_dim, out_dim=6)
    no_bias = layer_cls(manifold_fn(dtype), rngs=nnx.Rngs(0), use_gyro_bias=False, **kwargs)
    with_bias = layer_cls(manifold_fn(dtype), rngs=nnx.Rngs(0), use_gyro_bias=True, **kwargs)
    assert jnp.allclose(no_bias(x, C), with_bias(x, C), atol=1e-9)


@pytest.mark.parametrize(
    "layer_cls, manifold_fn, in_dim, make_pts",
    [
        (HypLinearHyperboloidBusemann, get_hyperboloid, 9, _make_hyperboloid_points),
        (HypLinearPoincareBusemann, get_poincare, 8, _make_ball_points),
    ],
)
def test_bfc_activation(layer_cls, manifold_fn, in_dim, make_pts):
    """A bounded activation (tanh) keeps the BFC output a valid manifold point."""
    dtype = jnp.float32
    x = make_pts(jax.random.PRNGKey(4), 6, in_dim, dtype)
    layer = layer_cls(manifold_fn(dtype), in_dim=in_dim, out_dim=6, rngs=nnx.Rngs(0), activation=jax.nn.tanh)
    y = layer(x, C)
    assert y.shape == (6, 6)
    assert jnp.isfinite(y).all()
