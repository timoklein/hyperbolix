"""Busemann-specific behaviour of the BFC layers (HypLinear{Hyperboloid,Poincare}Busemann).

The shared forward/gradient/JIT/tangent contract for all four Busemann layers
(the BMLR heads HypRegression{Hyperboloid,Poincare}Busemann included) lives in
``test_layer_contract.py``; only the tests that are specific to this family stay
here.
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


# --------------------------------------------------------------------------- #
# BFC layers
# --------------------------------------------------------------------------- #
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
    manifold = manifold_fn(dtype)
    layer = layer_cls(manifold, in_dim=in_dim, out_dim=6, rngs=nnx.Rngs(0), activation=jax.nn.tanh)
    y = layer(x, C)
    assert y.shape == (6, 6)
    assert jnp.isfinite(y).all()
    assert jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(y, C).all()
