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
# Construction: each family takes its own model only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("in_dim", [5, 2])
@pytest.mark.parametrize(
    "layer_cls, wrong_manifold_fn, family",
    [
        (HypRegressionPoincareBusemann, get_hyperboloid, "Poincare"),
        (HypLinearPoincareBusemann, get_hyperboloid, "Poincare"),
        (HypRegressionHyperboloidBusemann, get_poincare, "Hyperboloid"),
        (HypLinearHyperboloidBusemann, get_poincare, "Hyperboloid"),
    ],
    ids=["bmlr-poincare", "bfc-poincare", "bmlr-hyperboloid", "bfc-hyperboloid"],
)
def test_busemann_layers_reject_the_other_model_at_construction(layer_cls, wrong_manifold_fn, family, in_dim):
    """The Poincaré layers refuse a ``Hyperboloid``, the Hyperboloid layers a ``Poincare``.

    Both manifolds have ``busemann`` and ``_busemann_tangent``, so the method-name validation cannot
    tell them apart. The mismatched layer then failed at its first call on a shape error, except at
    ``in_dim = 2``: there one operand has a single spatial entry, which broadcasts against the
    other's two, and the layer ran without an error.
    """
    with pytest.raises(TypeError, match=f"{family} Busemann layers need a class-based {family} manifold instance"):
        layer_cls(wrong_manifold_fn(jnp.float32), in_dim, 3, rngs=nnx.Rngs(0))


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


# --------------------------------------------------------------------------- #
# Far-field accuracy of the Lorentz Busemann layers
# --------------------------------------------------------------------------- #
FAR_A = 9.5  # scaled geodesic radius a = sqrt(c) * d_0(x)
FAR_A_SUB = 6.5  # subdominant output channels, ~3 nats below the radius-carrying one
FAR_THETA = 1e-3  # angle between the near-aligned direction v_0 and the input point direction


def _far_points_BAi(a_B, in_dim, dtype, seed, one_ray, c=C):
    """Hyperboloid points at the scaled radii ``a_B = sqrt(c)*d``, built in closed form.

    ``one_ray``: every point shares a direction (one geodesic ray); otherwise the directions are
    drawn independently. Built directly rather than through ``expmap_0`` so the radius is exact.
    """
    key = jax.random.PRNGKey(seed)
    if one_ray:
        dir_BI = jax.random.normal(key, (in_dim - 1,), dtype=jnp.float64)[None, :]
    else:
        dir_BI = jax.random.normal(key, (len(a_B), in_dim - 1), dtype=jnp.float64)
    dir_BI = dir_BI / jnp.linalg.norm(dir_BI, axis=-1, keepdims=True)
    a_B = jnp.asarray(a_B, dtype=jnp.float64)
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=jnp.float64))
    time_B1 = (jnp.cosh(a_B) / sqrt_c)[:, None]
    return jnp.concatenate([time_B1, (jnp.sinh(a_B)[:, None] / sqrt_c) * dir_BI], axis=-1).astype(dtype)


def _tilted_rows_KI(seed, n_out, in_spatial, x_hat_I, theta):
    """Unit direction rows; row 0 is rotated to sit ``theta`` radians from ``x_hat_I``."""
    rows_KI = jax.random.normal(jax.random.PRNGKey(seed), (n_out, in_spatial), dtype=jnp.float64)
    rows_KI = rows_KI / jnp.linalg.norm(rows_KI, axis=-1, keepdims=True)
    perp_I = rows_KI[0] - jnp.dot(rows_KI[0], x_hat_I) * x_hat_I
    perp_I = perp_I / jnp.linalg.norm(perp_I)
    v0_I = jnp.cos(theta) * x_hat_I + jnp.sin(theta) * perp_I
    return rows_KI.at[0].set(v0_I / jnp.linalg.norm(v0_I))


def _scaled_radius(x_BA, c=C):
    """``a = sqrt(c)*d_0(x) = arcsinh(sqrt(c)*||x_s||)``, read off the spatial part."""
    return jnp.arcsinh(jnp.sqrt(c) * jnp.linalg.norm(jnp.asarray(x_BA, dtype=jnp.float64)[..., 1:], axis=-1))


def _max_rel(a32, a64):
    """Max-abs difference normalized by the max-abs of the float64 array."""
    a32, a64 = np.asarray(a32, dtype=np.float64), np.asarray(a64, dtype=np.float64)
    return float(np.max(np.abs(a32 - a64)) / np.max(np.abs(a64)))


def test_bfc_hyperboloid_far_field_matches_float64():
    """float32 Lorentz BFC with a gyro-bias tracks float64 at scaled geodesic radius a ~ 9.5.

    Two rewritten primitives meet here: ``busemann`` reads the input at a = 9.5 (channel 0 points
    within 1e-3 rad of the input direction, where ``x_t - <x_s, v>`` is a cancelling difference of
    two numbers of size cosh(a)), and the gyro-bias boosts the output point, also at a ~ 9.5. The
    pre-2026-09-08 spelling of ``busemann`` — the literal difference, floored at ``MIN_NORM`` —
    stays finite and is wrong by 9.9e-1 in geodesic distance and 1.3e-1 relative in the kernel
    gradient; the old ``addition`` returns NaN at this radius.

    Measured float32-vs-float64 error of the current spelling (float64 reference on the same
    float32-rounded inputs and parameters): 1.3e-3 geodesic distance, 1.5e-5 relative on the
    kernel gradient (the other three parameters are at 8e-7)
    (``logs/2026-09-08_hyperboloid_tangent_primitives/probe_layers_far_field5.out``).
    """
    in_dim, out_dim = 17, 9
    manifold64 = get_hyperboloid(jnp.float64)

    # The float64 reference must see exactly the points the float32 layer sees, so it runs on the
    # float32-rounded values cast back up: only the compute precision differs.
    x32_BAi = _far_points_BAi([9.5, 9.4, 9.3], in_dim, jnp.float32, seed=20, one_ray=True)
    x64_BAi = x32_BAi.astype(jnp.float64)

    x_hat_I = x64_BAi[0, 1:] / jnp.linalg.norm(x64_BAi[0, 1:])
    kernel_OI = _tilted_rows_KI(21, out_dim - 1, in_dim - 1, x_hat_I, FAR_THETA)
    busemann_O = jax.vmap(manifold64.busemann, in_axes=(None, 0, None))(x64_BAi[0], kernel_OI, C)
    # ``u_k = -alpha_k * B^{v_k}(x)`` is linear in ``alpha_k = exp(log_scale_k) > 0``, so one
    # division per channel puts sqrt(c)*|u_k| at a chosen target. Channel 1 (a generic direction)
    # carries the output radius; the near-aligned channel 0 is kept subdominant, since its output
    # coordinate is the one whose float32 error is transverse to the output direction.
    target_O = jnp.asarray([3.0, FAR_A] + [FAR_A_SUB] * (out_dim - 3), dtype=jnp.float64)
    log_scale_O = jnp.log(target_O / (jnp.sqrt(C) * jnp.abs(busemann_O)))

    gyro_O = jax.random.normal(jax.random.PRNGKey(22), (out_dim - 1,), dtype=jnp.float64)
    gyro_O = 0.2 * gyro_O / jnp.linalg.norm(gyro_O)

    def run(dtype, x_BAi):
        layer = HypLinearHyperboloidBusemann(
            get_hyperboloid(dtype), in_dim=in_dim, out_dim=out_dim, rngs=nnx.Rngs(0), use_gyro_bias=True, param_dtype=dtype
        )
        # Round to float32 first in both runs: the float64 reference must differ from the float32
        # layer only in compute precision, not in the parameter values it was handed.
        layer.kernel[...] = jnp.asarray(kernel_OI, dtype=jnp.float32).astype(dtype)
        layer.log_scale[...] = jnp.asarray(log_scale_O, dtype=jnp.float32).astype(dtype)
        layer.gyro_bias[...] = jnp.asarray(gyro_O, dtype=jnp.float32).astype(dtype)
        x_BAi = jnp.asarray(x_BAi, dtype=dtype)

        def loss_fn(model):
            return jnp.sum(model(x_BAi, c=C)[:, 1:])

        _, grads = nnx.value_and_grad(loss_fn)(layer)
        return layer(x_BAi, c=C), (grads.kernel[...], grads.log_scale[...], grads.bias[...], grads.gyro_bias[...])

    y64_BAo, grads64 = run(jnp.float64, x64_BAi)
    y32_BAo, grads32 = run(jnp.float32, x32_BAi)

    # The regime is the test: a construction that drifted back to small radius would pass vacuously.
    a_out_B = _scaled_radius(y64_BAo)
    assert 9.0 < float(jnp.min(a_out_B)) and float(jnp.max(a_out_B)) < 10.0, a_out_B

    dist_B = jax.vmap(manifold64.dist, in_axes=(0, 0, None))(
        manifold64.proj_batch(y32_BAo.astype(jnp.float64), C), manifold64.proj_batch(y64_BAo, C), C
    )
    assert jnp.isfinite(y32_BAo).all()
    assert float(jnp.max(dist_B)) < 6e-3  # measured 1.3e-3; old busemann 9.9e-1, old addition NaN
    for g32, g64 in zip(grads32, grads64, strict=True):
        assert _max_rel(g32, g64) < 1e-4  # measured 1.5e-5; old busemann 1.3e-1


def test_bmlr_hyperboloid_far_field_matches_float64():
    """float32 Lorentz BMLR logits and cross-entropy gradients track float64 at a ~ 9.5.

    Class 0's ideal direction sits ``1e-3`` rad from the input point direction, the cancelling case
    for the Busemann argument ``x_t - <x_s, v>``: at a = 9.5 both terms are of size cosh(a) ~ 6.7e3
    and their difference is ~3e-3, so the literal float32 subtraction keeps no significant digits.
    The pre-2026-09-08 spelling stays finite and is wrong by 3.7e-3 relative in the logits and
    7.1e-2 relative in the kernel gradient. The labels avoid class 0 on purpose: with it as the
    target the softmax saturates and the cancelling row receives no gradient at all.

    Measured float32-vs-float64 error of the current spelling (float64 reference on the same
    float32-rounded inputs and parameters): 1.1e-7 relative on the logits, 2.3e-5 relative on the
    kernel gradient (log_scale and bias are at 1.4e-7)
    (``logs/2026-09-08_hyperboloid_tangent_primitives/probe_layers_far_field5.out``).
    """
    in_dim, out_dim, batch = 17, 8, 4

    x32_BAi = _far_points_BAi([FAR_A] * batch, in_dim, jnp.float32, seed=30, one_ray=False)
    x64_BAi = x32_BAi.astype(jnp.float64)

    x_hat_I = x64_BAi[0, 1:] / jnp.linalg.norm(x64_BAi[0, 1:])
    dir_KI = _tilted_rows_KI(31, out_dim, in_dim - 1, x_hat_I, FAR_THETA)
    # Reference parameterization: alpha_k = ||kernel_k||, i.e. log_scale_k = log ||kernel_k||.
    norm_K1 = jnp.abs(jax.random.normal(jax.random.PRNGKey(32), (out_dim, 1), dtype=jnp.float64)) + 0.5
    kernel_KI = norm_K1 * dir_KI
    log_scale_K = jnp.log(jnp.linalg.norm(kernel_KI, axis=-1))
    label_B = (jnp.arange(batch) + 1) % out_dim

    def run(dtype, x_BAi):
        layer = HypRegressionHyperboloidBusemann(
            get_hyperboloid(dtype), in_dim=in_dim, out_dim=out_dim, rngs=nnx.Rngs(0), param_dtype=dtype
        )
        # Round to float32 first in both runs: the float64 reference must differ from the float32
        # layer only in compute precision, not in the parameter values it was handed.
        layer.kernel[...] = jnp.asarray(kernel_KI, dtype=jnp.float32).astype(dtype)
        layer.log_scale[...] = jnp.asarray(log_scale_K, dtype=jnp.float32).astype(dtype)
        x_BAi = jnp.asarray(x_BAi, dtype=dtype)

        def loss_fn(model):
            logits_BK = model(x_BAi, c=C)
            return -jnp.mean(jax.nn.log_softmax(logits_BK, axis=-1)[jnp.arange(batch), label_B])

        _, grads = nnx.value_and_grad(loss_fn)(layer)
        return layer(x_BAi, c=C), (grads.kernel[...], grads.log_scale[...], grads.bias[...])

    logits64_BK, grads64 = run(jnp.float64, x64_BAi)
    logits32_BK, grads32 = run(jnp.float32, x32_BAi)

    # The regime is the test: a construction that drifted back to small radius would pass vacuously.
    a_in_B = _scaled_radius(x64_BAi)
    assert np.allclose(np.asarray(a_in_B), FAR_A, atol=1e-6), a_in_B

    assert jnp.isfinite(logits32_BK).all()
    assert _max_rel(logits32_BK, logits64_BK) < 1e-6  # measured 1.1e-7; old busemann 3.7e-3
    for g32, g64 in zip(grads32, grads64, strict=True):
        assert _max_rel(g32, g64) < 2e-4  # measured 2.3e-5; old busemann 7.1e-2


def test_busemann_row_with_infinite_time_coordinate_stays_non_finite():
    """A diverged input row comes out non-finite, and leaves its batch neighbours bit-identical.

    ``HypLinearHyperboloidBusemann`` shares ``sinh_lift_to_hyperboloid`` with the PLFC layer, so it
    inherits the same contract: the ``±v_max`` clip guards finite scores only, and a Busemann logit
    that has already gone ``inf``/NaN is passed through rather than clipped to a plausible point.
    """
    in_dim, out_dim, dtype = 5, 4, jnp.float32
    layer = HypLinearHyperboloidBusemann(
        get_hyperboloid(dtype), in_dim=in_dim, out_dim=out_dim, rngs=nnx.Rngs(0), use_gyro_bias=True, param_dtype=dtype
    )
    layer.gyro_bias[...] = jnp.full_like(layer.gyro_bias[...], 0.1)  # zero-init would be a no-op

    good_BAi = _make_hyperboloid_points(jax.random.PRNGKey(0), 4, in_dim, dtype)
    bad_Ai = jnp.concatenate([jnp.array([jnp.inf], dtype=dtype), jnp.full((in_dim - 1,), 0.3, dtype=dtype)])

    y_BAo = layer(jnp.concatenate([good_BAi, bad_Ai[None, :]], axis=0), c=C)
    good_only_BAo = layer(good_BAi, c=C)

    assert not bool(jnp.isfinite(y_BAo[4]).any()), f"diverged row must stay loud, got {y_BAo[4]}"
    assert np.asarray(y_BAo[:4]).tobytes() == np.asarray(good_only_BAo).tobytes()


@pytest.mark.parametrize("c", [0.01, 1.0])
def test_bfc_poincare_saturated_scores_stay_at_boundary_f32(c):
    """Saturated Busemann scores must lift to the ball's edge in float32, not collapse to the origin.

    Regression: the Poincaré output map squared ω = sinh(√c·u)/√c, so ``sum(ω**2)`` overflowed once
    O·sinh²(v_max)/c passed FLT_MAX (here v_max = 44, O = 64: sinh(44)² ≈ 4e37), and the output became
    exactly the origin with an exactly-zero gradient and no NaN. ``log_scale + log(1e6)`` scales every
    ``alpha_k`` by 1e6, which pushes most of the √c·u past the ``v_max`` clip.
    """
    in_dim, out_dim, v_max = 16, 64, 44.0
    layer = HypLinearPoincareBusemann(get_poincare(jnp.float32), in_dim, out_dim, rngs=nnx.Rngs(0), v_max=v_max)
    u_BI = jax.random.normal(jax.random.PRNGKey(0), (4, in_dim), dtype=jnp.float32)
    x_BI = 0.9 / jnp.sqrt(c) * u_BI / jnp.linalg.norm(u_BI, axis=-1, keepdims=True)  # 0.9 of the ball radius
    layer.log_scale[...] = layer.log_scale[...] + jnp.log(1e6)

    def loss_fn(m):
        return jnp.sum(m(x_BI, c))

    y_BO = nnx.jit(lambda m: m(x_BI, c))(layer)
    _loss, grads = nnx.jit(nnx.value_and_grad(loss_fn))(layer)
    radius_B = jnp.sqrt(c) * jnp.linalg.norm(y_BO, axis=-1)

    assert jnp.all(jnp.isfinite(y_BO)), "non-finite output"
    assert jnp.all(radius_B > 0.99), f"rows left the ball's edge (√c·‖y‖ = {radius_B})"
    for _, value in jax.tree_util.tree_flatten_with_path(nnx.state(grads, nnx.Param))[0]:
        assert jnp.all(jnp.isfinite(value)), "gradients contain NaN or Inf"


# --------------------------------------------------------------------------- #
# Poincaré tangent input: scored from v, without the ball point expmap_0(v)
# --------------------------------------------------------------------------- #
POINCARE_LAYERS = [HypRegressionPoincareBusemann, HypLinearPoincareBusemann]
POINCARE_IDS = ["bmlr", "bfc"]


def _tangent_rows(key, batch, in_dim, t_B1, c):
    """Float64 rows of scaled norm ``t = √c‖v‖`` in random directions."""
    u_BI = jax.random.normal(key, (batch, in_dim), dtype=jnp.float64)
    return jnp.asarray(t_B1, dtype=jnp.float64) / jnp.sqrt(c) * u_BI / jnp.linalg.norm(u_BI, axis=-1, keepdims=True)


def _copy_params(layer, source, dtype):
    """Copy ``source``'s kernel, log_scale and bias into ``layer``, cast to ``dtype``."""
    for name in ("kernel", "log_scale", "bias"):
        getattr(layer, name)[...] = getattr(source, name)[...].astype(dtype)


def _out_and_grads(route, layer, v_BI, w_BO):
    """``route(layer, v)`` and the gradients of ``sum(w * route(layer, v))`` wrt ``v`` and the parameters."""
    grads, g_BI = nnx.grad(lambda m, v: jnp.sum(w_BO * route(m, v)), argnums=(0, 1))(layer, v_BI)
    return route(layer, v_BI), g_BI, (grads.kernel[...], grads.log_scale[...], grads.bias[...])


@pytest.mark.parametrize("c", [0.3, 1.0])
@pytest.mark.parametrize("layer_cls", POINCARE_LAYERS, ids=POINCARE_IDS)
def test_poincare_busemann_tangent_input_matches_the_ball_route(layer_cls, c):
    """float64, t = √c‖v‖ ≤ 3: the tangent-input layer equals the route through the ball it replaced.

    That route lifted v first: expmap_0, then the manifold-input layer. The rows straddle the t = 1
    seam between the two closed forms of ``Poincare._busemann_tangent``, three of them within 1e-7 of
    it. Outputs, input gradients and parameter gradients agree to 1e-12 relative to their largest
    entry (measured ≤ 3e-14).
    """
    dtype, batch, in_dim, out_dim = jnp.float64, 32, 12, 8
    manifold = get_poincare(dtype)
    tangent = layer_cls(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), input_space="tangent", param_dtype=dtype)
    ball = layer_cls(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)
    key_s, key_b, key_t, key_u, key_w = jax.random.split(jax.random.PRNGKey(1), 5)
    # Move alpha and the bias off their init so every parameter gradient is exercised.
    tangent.log_scale[...] = tangent.log_scale[...] + 0.3 * jax.random.normal(key_s, (out_dim,), dtype=dtype)
    tangent.bias[...] = 0.3 * jax.random.normal(key_b, (out_dim,), dtype=dtype)
    _copy_params(ball, tangent, dtype)
    seam_B1 = jnp.array([[1.0 - 1e-7], [1.0], [1.0 + 1e-7]], dtype=dtype)
    t_B1 = jnp.concatenate([jax.random.uniform(key_t, (batch - 3, 1), dtype=dtype, minval=0.05, maxval=3.0), seam_B1])
    v_BI = _tangent_rows(key_u, batch, in_dim, t_B1, c)
    w_BO = jax.random.normal(key_w, (batch, out_dim), dtype=dtype)

    def ball_route(model, v_BI):
        return model(jax.vmap(manifold.expmap_0, in_axes=(0, None))(v_BI, c), c)

    y_new, g_new, params_new = _out_and_grads(lambda m, v: m(v, c), tangent, v_BI, w_BO)
    y_old, g_old, params_old = _out_and_grads(ball_route, ball, v_BI, w_BO)
    assert _max_rel(y_new, y_old) < 1e-12, _max_rel(y_new, y_old)
    assert _max_rel(g_new, g_old) < 1e-12, _max_rel(g_new, g_old)
    for p_new, p_old in zip(params_new, params_old, strict=True):
        assert _max_rel(p_new, p_old) < 1e-12, _max_rel(p_new, p_old)


@pytest.mark.parametrize("c", [0.3, 1.0])
@pytest.mark.parametrize("layer_cls", POINCARE_LAYERS, ids=POINCARE_IDS)
def test_poincare_busemann_tangent_input_origin_jacobian(layer_cls, c):
    """float64: the input Jacobian at v = 0 matches central finite differences.

    The cancellation-free form of ``Poincare._busemann_tangent`` is built from t = √c‖v‖ and
    v̂ = v/‖v‖, whose cone singularities at the origin cancel only in the sum. Differentiated as
    written it returns a zero Jacobian there, which would silently stop the gradient to whatever
    produced an all-zero feature row. The BMLR Jacobian is also closed-form, ∂u_k/∂v = 2·alpha_k·ω_k,
    since B^ω(expmap_0(v)) = -2⟨ω, v⟩ + O(‖v‖²).
    """
    dtype, in_dim, out_dim, h = jnp.float64, 12, 8, 1e-6
    layer = layer_cls(get_poincare(dtype), in_dim, out_dim, rngs=nnx.Rngs(0), input_space="tangent", param_dtype=dtype)
    layer.bias[...] = 0.3 * jax.random.normal(jax.random.PRNGKey(2), (out_dim,), dtype=dtype)

    def layer_O(v_I):
        return layer(v_I[None, :], c)[0]

    jac_OI = jax.jacobian(layer_O)(jnp.zeros((in_dim,), dtype=dtype))
    fd_OI = jnp.stack([(layer_O(h * e_I) - layer_O(-h * e_I)) / (2 * h) for e_I in jnp.eye(in_dim, dtype=dtype)], axis=-1)
    assert _max_rel(jac_OI, fd_OI) < 1e-7, _max_rel(jac_OI, fd_OI)  # measured ≤ 2.2e-10 (FD rounding, ~eps/h)
    if layer_cls is HypRegressionPoincareBusemann:
        kernel_OI = layer.kernel[...]
        omega_OI = kernel_OI / jnp.linalg.norm(kernel_OI, axis=-1, keepdims=True)
        expected_OI = 2 * jnp.exp(layer.log_scale[...])[:, None] * omega_OI
        assert _max_rel(jac_OI, expected_OI) < 1e-12, _max_rel(jac_OI, expected_OI)


@pytest.mark.parametrize("c", [0.3, 1.0])
@pytest.mark.parametrize("layer_cls", POINCARE_LAYERS, ids=POINCARE_IDS)
def test_poincare_busemann_tangent_input_far_point_f32(layer_cls, c):
    """float32 at t = √c‖v‖ = 8: outputs and input gradients match float64 on the same inputs and parameters.

    The lift onto the ball capped this input at the float32 ceiling t ≈ 6.33 (c = 1), past which the
    score was constant along the ray: outputs 6.3e-2 … 2.1e-1 and input gradients 1.0 … 1.2 off
    relative to their largest entry, the radial derivative being zero. Here 4.8e-7 and 1.3e-6 at most.
    """
    batch, in_dim, out_dim, t = 32, 16, 12, 8.0
    layer32, layer64 = (
        layer_cls(get_poincare(dt), in_dim, out_dim, rngs=nnx.Rngs(0), input_space="tangent", param_dtype=dt)
        for dt in (jnp.float32, jnp.float64)
    )
    layer32.bias[...] = 0.3 * jax.random.normal(jax.random.PRNGKey(2), (out_dim,), dtype=jnp.float32)
    _copy_params(layer64, layer32, jnp.float64)
    v32_BI = _tangent_rows(jax.random.PRNGKey(0), batch, in_dim, jnp.full((batch, 1), t), c).astype(jnp.float32)
    w32_BO = jax.random.normal(jax.random.PRNGKey(1), (batch, out_dim), dtype=jnp.float32)

    def out_and_grad(layer, v_BI, w_BO):
        return layer(v_BI, c), jax.grad(lambda v: jnp.sum(w_BO * layer(v, c)))(v_BI)

    y32_BO, g32_BI = out_and_grad(layer32, v32_BI, w32_BO)
    y64_BO, g64_BI = out_and_grad(layer64, v32_BI.astype(jnp.float64), w32_BO.astype(jnp.float64))
    assert jnp.isfinite(y32_BO).all() and jnp.isfinite(g32_BI).all()
    assert _max_rel(y32_BO, y64_BO) < 5e-6, _max_rel(y32_BO, y64_BO)
    assert _max_rel(g32_BI, g64_BI) < 2e-5, _max_rel(g32_BI, g64_BI)


# --------------------------------------------------------------------------- #
# Hyperboloid tangent input: scored from v, without the lifted point expmap_0(v)
# --------------------------------------------------------------------------- #
HYPERBOLOID_LAYERS = [HypRegressionHyperboloidBusemann, HypLinearHyperboloidBusemann]
HYPERBOLOID_IDS = ["bmlr", "bfc"]


def _ambient_tangent_rows(key, batch, in_dim, t_B1, c):
    """Float64 tangent vectors ``(0, v_s)`` at the origin, ``t = √c‖v_s‖``, random directions."""
    return jnp.pad(_tangent_rows(key, batch, in_dim - 1, t_B1, c), ((0, 0), (1, 0)))


@pytest.mark.parametrize("c", [0.3, 1.0])
@pytest.mark.parametrize("layer_cls", HYPERBOLOID_LAYERS, ids=HYPERBOLOID_IDS)
def test_hyperboloid_busemann_tangent_input_matches_the_lift_route(layer_cls, c):
    """float64, t = √c‖v_s‖ ≤ 12: the tangent-input layer equals the route through the lifted point it replaced.

    That route lifted v first: expmap_0, then the manifold-input layer. The rows straddle the t = 2
    seam between the two closed forms of ``Hyperboloid._busemann_tangent``, three of them within 1e-7
    of it. Outputs, input gradients and parameter gradients agree to 1e-12 relative to their largest
    entry (measured ≤ 1.4e-15).
    """
    dtype, batch, in_dim, out_dim = jnp.float64, 32, 13, 9
    manifold = get_hyperboloid(dtype)
    tangent = layer_cls(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), input_space="tangent", param_dtype=dtype)
    lifted = layer_cls(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)
    n_rows = tangent.bias[...].shape[0]
    key_s, key_b, key_t, key_u, key_w = jax.random.split(jax.random.PRNGKey(1), 5)
    # Move alpha and the bias off their init so every parameter gradient is exercised.
    tangent.log_scale[...] = tangent.log_scale[...] + 0.3 * jax.random.normal(key_s, (n_rows,), dtype=dtype)
    tangent.bias[...] = 0.3 * jax.random.normal(key_b, (n_rows,), dtype=dtype)
    _copy_params(lifted, tangent, dtype)
    seam_B1 = jnp.array([[2.0 - 1e-7], [2.0], [2.0 + 1e-7]], dtype=dtype)
    t_B1 = jnp.concatenate([jax.random.uniform(key_t, (batch - 3, 1), dtype=dtype, minval=0.05, maxval=12.0), seam_B1])
    v_BA = _ambient_tangent_rows(key_u, batch, in_dim, t_B1, c)
    w_BO = jax.random.normal(key_w, (batch, out_dim), dtype=dtype)

    def lift_route(model, v_BA):
        return model(jax.vmap(manifold.expmap_0, in_axes=(0, None))(v_BA, c), c)

    y_new, g_new, params_new = _out_and_grads(lambda m, v: m(v, c), tangent, v_BA, w_BO)
    y_old, g_old, params_old = _out_and_grads(lift_route, lifted, v_BA, w_BO)
    assert _max_rel(y_new, y_old) < 1e-12, _max_rel(y_new, y_old)
    assert _max_rel(g_new, g_old) < 1e-12, _max_rel(g_new, g_old)
    for p_new, p_old in zip(params_new, params_old, strict=True):
        assert _max_rel(p_new, p_old) < 1e-12, _max_rel(p_new, p_old)


@pytest.mark.parametrize("c", [0.3, 1.0])
@pytest.mark.parametrize("layer_cls", HYPERBOLOID_LAYERS, ids=HYPERBOLOID_IDS)
def test_hyperboloid_busemann_tangent_input_derivatives_at_origin_and_seam(layer_cls, c):
    """float64: the input Jacobian at v = 0 and at the t = 2 seam matches central finite differences.

    The points lie on one ray ``s·u``; kernel row 0 is ``u`` (aligned) and row 1 is ``-u``
    (anti-aligned). At t = 2 the stencil straddles the seam between the two closed forms of
    ``Hyperboloid._busemann_tangent``; t = 2 ± 1e-3 test each side. The cancellation-free form is built
    from t = √c‖v_s‖ and v̂, whose cone singularities at the origin cancel only in the sum:
    differentiated as written it would return a zero Jacobian at v = 0. The BMLR Jacobian is also
    closed-form there, and on the two ray-aligned rows at every point: B^{±u}(expmap_0(s·u)) = ∓s, so
    ∂u_k/∂v_s = alpha_k·ω_k for those rows, and for every row at v = 0. The time slot is not read.
    """
    dtype, in_dim, out_dim, h = jnp.float64, 13, 9, 1e-6
    layer = layer_cls(get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(0), input_space="tangent", param_dtype=dtype)
    n_rows = layer.bias[...].shape[0]
    key_u, key_b = jax.random.split(jax.random.PRNGKey(2))
    u_I = jax.random.normal(key_u, (in_dim - 1,), dtype=dtype)
    u_I = u_I / jnp.linalg.norm(u_I)
    layer.kernel[...] = layer.kernel[...].at[0].set(0.7 * u_I).at[1].set(-1.3 * u_I)
    layer.bias[...] = 0.3 * jax.random.normal(key_b, (n_rows,), dtype=dtype)
    omega_KI = layer.kernel[...] / jnp.linalg.norm(layer.kernel[...], axis=-1, keepdims=True)
    alpha_K = jnp.exp(layer.log_scale[...])

    def layer_O(v_A):
        return layer(v_A[None, :], c)[0]

    for t in (0.0, 2.0 - 1e-3, 2.0, 2.0 + 1e-3):
        v_A = jnp.concatenate([jnp.zeros((1,), dtype=dtype), (t / jnp.sqrt(c)) * u_I])
        jac_OA = jax.jacobian(layer_O)(v_A)
        fd_OA = jnp.stack(
            [(layer_O(v_A + h * e_A) - layer_O(v_A - h * e_A)) / (2 * h) for e_A in jnp.eye(in_dim, dtype=dtype)], -1
        )
        assert _max_rel(jac_OA, fd_OA) < 1e-7, (t, _max_rel(jac_OA, fd_OA))  # measured ≤ 2.2e-9 (FD rounding)
        assert bool(jnp.all(jac_OA[:, 0] == 0.0)), (t, jac_OA[:, 0])
        if layer_cls is HypRegressionHyperboloidBusemann:
            rows = slice(None) if t == 0.0 else slice(0, 2)
            expected_OI = alpha_K[:, None] * omega_KI
            assert _max_rel(jac_OA[rows, 1:], expected_OI[rows]) < 1e-12, (t, _max_rel(jac_OA[rows, 1:], expected_OI[rows]))


@pytest.mark.parametrize("c", [0.3, 1.0])
@pytest.mark.parametrize("layer_cls", HYPERBOLOID_LAYERS, ids=HYPERBOLOID_IDS)
def test_hyperboloid_busemann_tangent_input_far_point_f32(layer_cls, c):
    """float32 at t = √c‖v_s‖ ∈ {12, 20, 40, 50}: outputs and all gradients match float64 on the same inputs and parameters.

    Through the lifted point the float32 gradients broke from t = 30: the backward pass of the
    rationalized division in ``hyperboloid._busemann_arg`` flushed a ~e^{-3t} intermediate to zero,
    so the input and kernel gradients were 5e-2 … 1.07 off relative to their largest entry while the
    scores stayed right, and at t = 50 the lifted point overflowed and the outputs were NaN. Here every
    quantity is within 1.4e-6 (measured). The directions are random: an ω within float32 rounding of
    v̂ is ill-conditioned by either route (see ``Hyperboloid._busemann_tangent``). alpha = 0.15 keeps
    the BFC's √c·u below its v_max clip at t = 50.
    """
    batch, in_dim, out_dim = 32, 17, 12
    layer32, layer64 = (
        layer_cls(get_hyperboloid(dt), in_dim, out_dim, rngs=nnx.Rngs(0), input_space="tangent", param_dtype=dt)
        for dt in (jnp.float32, jnp.float64)
    )
    n_rows = layer32.bias[...].shape[0]
    layer32.log_scale[...] = jnp.full((n_rows,), jnp.log(0.15), dtype=jnp.float32)
    layer32.bias[...] = 0.3 * jax.random.normal(jax.random.PRNGKey(2), (n_rows,), dtype=jnp.float32)
    _copy_params(layer64, layer32, jnp.float64)
    w32_BO = jax.random.normal(jax.random.PRNGKey(1), (batch, out_dim), dtype=jnp.float32)

    for t in (12.0, 20.0, 40.0, 50.0):
        v32_BA = _ambient_tangent_rows(jax.random.PRNGKey(0), batch, in_dim, jnp.full((batch, 1), t), c).astype(jnp.float32)
        y32, g32, params32 = _out_and_grads(lambda m, v: m(v, c), layer32, v32_BA, w32_BO)
        y64, g64, params64 = _out_and_grads(
            lambda m, v: m(v, c), layer64, v32_BA.astype(jnp.float64), w32_BO.astype(jnp.float64)
        )
        assert jnp.isfinite(y32).all() and jnp.isfinite(g32).all(), t
        assert _max_rel(y32, y64) < 1e-5, (t, _max_rel(y32, y64))
        assert _max_rel(g32, g64) < 1e-5, (t, _max_rel(g32, g64))
        for p32, p64 in zip(params32, params64, strict=True):
            assert _max_rel(p32, p64) < 1e-5, (t, _max_rel(p32, p64))
