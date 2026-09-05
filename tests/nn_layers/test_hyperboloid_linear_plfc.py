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
