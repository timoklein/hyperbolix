"""Tests for HypLinearHyperboloidPLFC (point-to-hyperplane Lorentz FC layer).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypLinearHyperboloidPLFC-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import HypLinearHyperboloidPLFC


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


def _check_on_hyperboloid(x, c, atol=1e-5):
    """Check Minkowski constraint: -x0^2 + ||x_s||^2 = -1/c."""
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


def plfc_reference(x_BAi, kernel_OI, bias_O1, c, v_max=10.0):
    """NumPy transcription of the PLFC forward (Shi et al. 2026, Thm. 1 + Sec. 4.1).

    Independent of the library — the published equations written out directly:

        alpha_k(x) = -x_t*sinh(sqrt(c)*r_k)*||z_k|| + cosh(sqrt(c)*r_k)*<x_s, z_k>
        v_k(x)     = (||z_k||/sqrt(c)) * asinh(sqrt(c)*alpha_k(x)/||z_k||)   (MLR score)
        y_s        = sinh(clip(sqrt(c)*v, +-v_max))/sqrt(c)                  (sinh diffeo)
        y_t        = sqrt(||y_s||^2 + 1/c)                                   (constraint)

    The library applies no clamp to the asinh argument, so this is the exact same expression.
    """
    x_BAi = np.asarray(x_BAi, dtype=np.float64)
    z_OI = np.asarray(kernel_OI, dtype=np.float64)
    r_O1 = np.asarray(bias_O1, dtype=np.float64)
    sqrt_c = np.sqrt(c)

    z_norm_1O = np.linalg.norm(z_OI, axis=-1)[None, :]
    sqrt_cr_1O = sqrt_c * r_O1.T
    xt_B1, xs_BI = x_BAi[:, 0:1], x_BAi[:, 1:]

    alpha_BO = -xt_B1 * np.sinh(sqrt_cr_1O) * z_norm_1O + np.cosh(sqrt_cr_1O) * (xs_BI @ z_OI.T)
    v_BO = (z_norm_1O / sqrt_c) * np.arcsinh(sqrt_c * alpha_BO / z_norm_1O)

    ys_BO = np.sinh(np.clip(sqrt_c * v_BO, -v_max, v_max)) / sqrt_c
    yt_B1 = np.sqrt(np.sum(ys_BO**2, axis=-1, keepdims=True) + 1.0 / c)
    return np.concatenate([yt_B1, ys_BO], axis=-1)


def hyperboloid_points(key, batch, ambient, c, dtype, scale=0.3):
    """Batch of hyperboloid points from spatial-only tangent vectors at the origin."""
    manifold = get_hyperboloid(dtype)
    v = jax.random.normal(key, (batch, ambient), dtype=dtype) * scale
    v = v.at[:, 0].set(0.0)
    return jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, c)


# --------------------------------------------------------------------------- #
# Forward value oracle (audit A6-03)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", [0.5, 1.0])
def test_plfc_forward_matches_shi_transcription(c):
    """PLFC forward equals the Shi et al. 2026 equations transcribed in NumPy.

    Catches both an origin-collapsed forward (spatial output scaled to zero) and
    any sign flip inside the MLR score or the sinh diffeomorphism.
    """
    dtype = jnp.float64
    manifold = get_hyperboloid(dtype)
    batch_size, in_dim, out_dim = 5, 6, 7

    x = hyperboloid_points(jax.random.PRNGKey(0), batch_size, in_dim, c, dtype)
    layer = HypLinearHyperboloidPLFC(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(1), (out_dim - 1, in_dim - 1), dtype=dtype) * 0.6
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(2), (out_dim - 1, 1), dtype=dtype) * 0.3

    y = layer(x, c=c)
    expected = plfc_reference(x, layer.kernel[...], layer.bias[...], c, v_max=layer.v_max)

    assert np.allclose(np.asarray(y), expected, atol=1e-11)
    # The oracle itself must be non-degenerate (a collapsed reference would match anything).
    assert np.max(np.abs(expected[:, 1:])) > 0.1


def test_plfc_forward_is_input_dependent():
    """Two distinct inputs give distinct outputs (constant-collapse guard)."""
    dtype = jnp.float64
    c = 1.0
    manifold = get_hyperboloid(dtype)
    in_dim, out_dim = 6, 7

    x = hyperboloid_points(jax.random.PRNGKey(7), 2, in_dim, c, dtype, scale=0.6)
    layer = HypLinearHyperboloidPLFC(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)

    y = layer(x, c=c)

    assert float(jnp.max(jnp.abs(y[0] - y[1]))) > 1e-6
    # And the output is not pinned at the manifold origin [1/sqrt(c), 0, ..., 0].
    assert float(jnp.max(jnp.abs(y[:, 1:]))) > 1e-6


def test_default_init_scale():
    """Default kernel init follows the Shi et al. 2026 PLFC reference (std=0.02)."""
    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(jnp.float32), 65, 65, rngs=rngs)

    kernel = layer.kernel[...]
    assert jnp.std(kernel) == pytest.approx(0.02, rel=0.2)
    assert jnp.all(layer.bias[...] == 0.0)


@pytest.mark.parametrize("c", [0.1, 1.0])
def test_v_max_guard_bounds_output(c):
    """The output-side guard bounds the spatial norm by sinh(v_max)/sqrt(c).

    Regression test for the float32 blow-up: with large kernels (std=1.0, the
    pre-PLFC init) and inputs far from the origin, the unguarded sinh produces
    spatial coordinates ~1e17 whose squared norm overflows in stacked layers.
    """
    key = jax.random.PRNGKey(1)
    batch_size, in_dim, out_dim = 4, 129, 129
    v_max = 10.0

    # Input with spatial norm 20 (geodesic distance ~3.7 from origin at c=1)
    xs = jax.random.normal(key, (batch_size, in_dim - 1), dtype=jnp.float32)
    xs = 20.0 * xs / jnp.linalg.norm(xs, axis=-1, keepdims=True)
    x0 = jnp.sqrt(jnp.sum(xs**2, axis=-1, keepdims=True) + 1.0 / c)
    x = jnp.concatenate([x0, xs], axis=-1)

    rngs = nnx.Rngs(42)
    # kernel_init_std=1.0 restores the pre-guard worst case
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(jnp.float32), in_dim, out_dim, rngs=rngs, kernel_init_std=1.0)

    y = layer(x, c=c)

    spatial_bound = jnp.sinh(v_max) / jnp.sqrt(c)
    assert jnp.isfinite(y).all()
    assert jnp.max(jnp.abs(y[:, 1:])) <= spatial_bound * 1.01

    # Gradients must stay finite in the saturated regime
    def loss_fn(model):
        return jnp.sum(model(x, c=c) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gyro_bias_zero_is_identity(dtype):
    """At init the gyro-bias is zero -> gyroaddition with the origin is a no-op."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 1e-5 if dtype == jnp.float32 else 1e-12

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    layer_plain = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(42))
    layer_gyro = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(42), use_gyro_bias=True)

    y_plain = layer_plain(x, c=1.0)
    y_gyro = layer_gyro(x, c=1.0)

    assert jnp.allclose(y_plain, y_gyro, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gyro_bias_on_manifold_and_trainable(dtype):
    """Nonzero gyro-bias keeps the output on the manifold and receives gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    layer = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(42), use_gyro_bias=True)
    bias_key = jax.random.PRNGKey(7)
    layer.gyro_bias[...] = jax.random.normal(bias_key, (out_dim - 1,), dtype=layer.gyro_bias[...].dtype) * 0.3

    y = layer(x, c=1.0)

    assert jnp.isfinite(y).all()
    assert _check_on_hyperboloid(y, c=1.0, atol=atol)

    def loss_fn(model):
        return jnp.sum(model(x, c=1.0) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.gyro_bias[...]).all()
    assert jnp.any(grads.gyro_bias[...] != 0.0)


# --------------------------------------------------------------------------- #
# Far-field accuracy of the gyro-bias boost
# --------------------------------------------------------------------------- #
FAR_C = 1.0
FAR_A = 9.5  # scaled geodesic radius a = sqrt(c) * d_0(x)
FAR_A_SUB = 6.5  # subdominant output channels, ~3 nats below the radius-carrying one


def ray_points(a_B, in_dim, c, seed):
    """Float64 hyperboloid points on one geodesic ray, at the scaled radii ``a_B``."""
    dir_I = jax.random.normal(jax.random.PRNGKey(seed), (in_dim - 1,), dtype=jnp.float64)
    dir_I = dir_I / jnp.linalg.norm(dir_I)
    a_B = jnp.asarray(a_B, dtype=jnp.float64)
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=jnp.float64))
    time_B1 = (jnp.cosh(a_B) / sqrt_c)[:, None]
    spatial_BI = (jnp.sinh(a_B)[:, None] / sqrt_c) * dir_I[None, :]
    return jnp.concatenate([time_B1, spatial_BI], axis=-1)


def scaled_radius(x_BA, c):
    """``a = sqrt(c)*d_0(x) = arcsinh(sqrt(c)*||x_s||)``, read off the spatial part."""
    return jnp.arcsinh(jnp.sqrt(c) * jnp.linalg.norm(jnp.asarray(x_BA, dtype=jnp.float64)[..., 1:], axis=-1))


def max_rel(a32, a64):
    """Max-abs difference normalized by the max-abs of the float64 array."""
    a32, a64 = np.asarray(a32, dtype=np.float64), np.asarray(a64, dtype=np.float64)
    return float(np.max(np.abs(a32 - a64)) / np.max(np.abs(a64)))


def test_gyro_bias_far_field_matches_float64():
    """float32 PLFC with a gyro-bias tracks float64 at scaled geodesic radius a ~ 9.5.

    The gyro-bias is ``y <- y (+) exp_0([0, b])``, i.e. the Lorentz boost of the output point,
    which is the hot path of every gyro-bias in the library. Its accuracy at large radius is what
    a finiteness check cannot see: the pre-2026-09-08 spelling (``Exp_x . PT_{0->x} . Log_0``,
    three maps each forming an O(1) quantity as a difference of O(cosh^2 a) Minkowski terms) stays
    finite here and is wrong by 5.1e-2 in geodesic distance and 3.0e-2 relative in every parameter
    gradient.

    Measured float32-vs-float64 error of the current spelling (float64 reference on the same
    float32-rounded inputs and parameters): 6.3e-4 geodesic distance, 6.0e-7 relative on the
    kernel / bias / gyro-bias gradients
    (``logs/2026-09-08_hyperboloid_tangent_primitives/probe_layers_far_field5.out``).
    """
    c, in_dim, out_dim = FAR_C, 17, 9
    manifold64 = get_hyperboloid(jnp.float64)

    # The float64 reference must see exactly the points the float32 layer sees, so it runs on the
    # float32-rounded values cast back up: only the compute precision differs.
    x32_BAi = ray_points([9.2, 9.35, 9.5], in_dim, c, seed=0).astype(jnp.float32)
    x64_BAi = x32_BAi.astype(jnp.float64)

    # At a zero bias the score is exactly linear in the kernel row norm,
    # ``sqrt(c)*v_k = ||z_k|| * arcsinh(sqrt(c) <x_s, z_k/||z_k||>)``, so one division per row puts
    # the sinh-lift argument at a chosen target — 9.5 on channel 0, 6.5 on the rest. The output
    # point then sits at a ~ 9.5 with its direction carried by the one dominant channel.
    rows_OI = jax.random.normal(jax.random.PRNGKey(1), (out_dim - 1, in_dim - 1), dtype=jnp.float64)
    rows_OI = rows_OI / jnp.linalg.norm(rows_OI, axis=-1, keepdims=True)
    score_O = jnp.arcsinh(jnp.sqrt(c) * (x64_BAi[0, 1:] @ rows_OI.T))
    target_O = jnp.asarray([FAR_A] + [FAR_A_SUB] * (out_dim - 2), dtype=jnp.float64)
    kernel_OI = (target_O / score_O)[:, None] * rows_OI

    gyro_O = jax.random.normal(jax.random.PRNGKey(2), (out_dim - 1,), dtype=jnp.float64)
    gyro_O = 0.2 * gyro_O / jnp.linalg.norm(gyro_O)

    def run(dtype, x_BAi):
        layer = HypLinearHyperboloidPLFC(
            get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(0), use_gyro_bias=True, param_dtype=dtype
        )
        # Round to float32 first in both runs: the float64 reference must differ from the float32
        # layer only in compute precision, not in the parameter values it was handed.
        layer.kernel[...] = jnp.asarray(kernel_OI, dtype=jnp.float32).astype(dtype)
        layer.gyro_bias[...] = jnp.asarray(gyro_O, dtype=jnp.float32).astype(dtype)
        x_BAi = jnp.asarray(x_BAi, dtype=dtype)

        def loss_fn(model):
            return jnp.sum(model(x_BAi, c=c)[:, 1:])

        _, grads = nnx.value_and_grad(loss_fn)(layer)
        return layer(x_BAi, c=c), (grads.kernel[...], grads.bias[...], grads.gyro_bias[...])

    y64_BAo, grads64 = run(jnp.float64, x64_BAi)
    y32_BAo, grads32 = run(jnp.float32, x32_BAi)

    # The regime is the test: a construction that drifted back to small radius would pass vacuously.
    a_out_B = scaled_radius(y64_BAo, c)
    assert 9.2 < float(jnp.min(a_out_B)) and float(jnp.max(a_out_B)) < 10.2, a_out_B

    dist_B = jax.vmap(manifold64.dist, in_axes=(0, 0, None))(
        manifold64.proj_batch(y32_BAo.astype(jnp.float64), c), manifold64.proj_batch(y64_BAo, c), c
    )
    assert jnp.isfinite(y32_BAo).all()
    assert float(jnp.max(dist_B)) < 3e-3  # measured 6.3e-4; old spelling 5.1e-2
    for g32, g64 in zip(grads32, grads64, strict=True):
        assert max_rel(g32, g64) < 5e-6  # measured 6.0e-7; old spelling 3.0e-2


# --------------------------------------------------------------------------- #
# Divergence stays loud (shared sinh-lift output map)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("r", [0.0, 1.0])
def test_row_with_infinite_time_coordinate_stays_non_finite(r: float):
    """An input row whose time coordinate has already overflowed comes out non-finite.

    With the MLR bias ``r = 0`` the score is NaN and was always loud. With ``r != 0`` the
    ``-x_t·sinh(√c·r)·‖z‖`` term makes the score ``-inf``, which the old output guard clipped to
    ``-v_max``: the layer returned a plausible on-sheet point, the loss stayed finite, and only the
    NaN kernel gradient (one minibatch later, through the dead parameters) gave the divergence
    away. Both bias regimes are pinned here; ``r = 1`` is the regression.

    The four healthy rows must be bit-identical to a call that contains only them — a diverged row
    may not perturb its batch neighbours.
    """
    c, in_dim, out_dim, dtype = 0.7, 5, 4, jnp.float32
    layer = HypLinearHyperboloidPLFC(
        get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(0), use_gyro_bias=True, param_dtype=dtype
    )
    layer.bias[...] = jnp.full_like(layer.bias[...], r)
    layer.gyro_bias[...] = jnp.full_like(layer.gyro_bias[...], 0.1)  # zero-init would be a no-op

    good_BAi = hyperboloid_points(jax.random.PRNGKey(0), 4, in_dim, c, dtype)
    bad_Ai = jnp.concatenate([jnp.array([jnp.inf], dtype=dtype), jnp.full((in_dim - 1,), 0.3, dtype=dtype)])

    y_BAo = layer(jnp.concatenate([good_BAi, bad_Ai[None, :]], axis=0), c=c)
    good_only_BAo = layer(good_BAi, c=c)

    assert not bool(jnp.isfinite(y_BAo[4]).any()), f"diverged row must stay loud, got {y_BAo[4]}"
    assert np.asarray(y_BAo[:4]).tobytes() == np.asarray(good_only_BAo).tobytes()
