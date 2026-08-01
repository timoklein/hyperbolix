"""Busemann-specific behaviour of the BFC layers (HypLinear{Hyperboloid,Poincare}Busemann).

The shared forward/gradient/JIT/tangent contract for all four Busemann layers
(the BMLR heads HypRegression{Hyperboloid,Poincare}Busemann included) lives in
``test_layer_contract.py``; only the tests that are specific to this family stay
here.
"""

import jax
import jax.numpy as jnp
import numpy as np
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


# --------------------------------------------------------------------------- #
# BMLR logit sign / value oracle (audit A9-06 + M3-G16)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "layer_cls, manifold_fn, in_dim, embed_tangent, lambda_0",
    [
        # Hyperboloid: tangent vectors carry the Minkowski norm, so exp_0(t*v) sits at
        # geodesic distance t. Poincare: lambda_0 = 2/(1 - c*0) = 2, so the Riemannian
        # length of the tangent vector t*v is 2t and exp_0(t*v) sits at distance 2t.
        (HypRegressionHyperboloidBusemann, get_hyperboloid, 9, True, 1.0),
        (HypRegressionPoincareBusemann, get_poincare, 8, False, 2.0),
    ],
    ids=["hyperboloid", "poincare"],
)
def test_bmlr_logit_is_affine_along_the_ideal_ray(layer_cls, manifold_fn, in_dim, embed_tangent, lambda_0):
    """``u_k`` grows by ``alpha_k`` per unit of geodesic distance toward the ideal point ``v_k``.

    Closed-form oracle, independent of the library: along the unit-speed geodesic
    ray toward ``v``, the Busemann function is ``B^v = -distance`` (both models),
    so Chen et al. 2026 Eq. 8, ``u_k(x) = -alpha_k*B^{v_k}(x) + b_k``, becomes

        u_k(exp_0(t*v_k)) = alpha_k * lambda_0 * t + b_k,   alpha_k = exp(log_scale_k).

    This pins the logit *sign* (a flipped sign makes the logit fall toward the
    ideal point), the magnitude ``alpha_k``, the bias offset at the origin
    (``B^v(origin) = 0``), and the direction normalization of the kernel rows.
    """
    dtype = jnp.float64
    manifold = manifold_fn(dtype)
    out_dim = 4
    layer = layer_cls(manifold, in_dim=in_dim, out_dim=out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)

    kernel_KI = np.asarray(layer.kernel[...], dtype=np.float64)
    alpha_K = np.exp(np.asarray(layer.log_scale[...], dtype=np.float64))
    bias_K = np.asarray(layer.bias[...], dtype=np.float64)

    t_T = jnp.array([0.0, 0.3, 0.7, 1.2], dtype=dtype)
    for k in range(out_dim):
        v_I = jnp.asarray(kernel_KI[k] / np.linalg.norm(kernel_KI[k]), dtype=dtype)
        tangent_TI = t_T[:, None] * v_I[None, :]
        if embed_tangent:
            tangent_TI = jax.vmap(manifold.embed_spatial_0)(tangent_TI)
        pts_TI = jax.vmap(manifold.expmap_0, in_axes=(0, None))(tangent_TI, C)

        logits_T = np.asarray(layer(pts_TI, c=C)[:, k], dtype=np.float64)
        expected_T = alpha_K[k] * lambda_0 * np.asarray(t_T, dtype=np.float64) + bias_K[k]

        assert np.allclose(logits_T, expected_T, atol=1e-9), f"class {k}: {logits_T} != {expected_T}"
        assert np.all(np.diff(logits_T) > 0.0)  # rises toward the ideal point


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
