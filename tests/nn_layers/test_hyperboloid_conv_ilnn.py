"""Tests for HypConv2DHyperboloidILNN (Intrinsic Lorentz convolution: LogCat + PLFC, Shi et al. 2026).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypConv2DHyperboloidILNN-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HypConv2DHyperboloidILNN, HypLinearHyperboloidPLFC


def _proj_image(x, manifold, c):
    """Project each pixel in a (B, H, W, C) feature map to the hyperboloid."""
    return jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)


def _check_on_hyperboloid(x, c, atol=1e-5):
    """Check Minkowski constraint: -x0^2 + ||x_s||^2 = -1/c."""
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


def _implied_logcat_scale(points_KC, logcat_A):
    """Recover the radius-matching scale that LogCat applied, from one entry.

    Deliberately *not* a transcription of the digamma formula: the value (and the
    direction) of that scale is under review upstream, while the block layout and
    the time formula asserted below hold for whatever scale the library picks.
    Consuming exactly one degree of freedom keeps everything else pinned.
    """
    ref_flat = np.asarray(points_KC, dtype=np.float64)[:, 1:].reshape(-1)
    idx = int(np.argmax(np.abs(ref_flat)))
    return float(np.asarray(logcat_A, dtype=np.float64)[1:][idx] / ref_flat[idx])


def _logcat_reference(points_KC, c, scale):
    """LogCat structure (Shi et al. 2026, Sec. 4.3) for a given radius-matching scale.

    Independent of ``Hyperboloid.log_radius_concat`` except for the scalar ``scale``::

        spatial  = concat_i(scale * x_i[1:])       (input order, blocks kept intact)
        time     = sqrt(1/c + scale^2 * sum_i(x_i[0]^2 - 1/c))    (Lorentz constraint)
    """
    pts_KC = np.asarray(points_KC, dtype=np.float64)
    spatial = (scale * pts_KC[:, 1:]).reshape(-1)
    time = np.sqrt(1.0 / c + scale**2 * np.sum(pts_KC[:, 0] ** 2 - 1.0 / c))
    return np.concatenate([[time], spatial])


def _plfc_reference(x_BAi, kernel_OI, bias_O1, c, v_max=10.0):
    """NumPy transcription of the PLFC forward (Shi et al. 2026, Thm. 1 + Sec. 4.1).

    See ``test_hyperboloid_linear_plfc.plfc_reference`` for the same equations at
    the linear layer; duplicated here so this file's mutation coverage stands alone.
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


def _single_patch_setup(c, dtype, kernel_size=2, in_channels=3, out_channels=4, seed=0):
    """One receptive field: image size == kernel size with VALID padding."""
    manifold = Hyperboloid(dtype=dtype)
    x = jax.random.normal(jax.random.PRNGKey(seed), (1, kernel_size, kernel_size, in_channels), dtype=dtype) * 0.3
    x_manifold = _proj_image(x, manifold, c)
    layer = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(42),
        padding="VALID",
        param_dtype=dtype,
    )
    return manifold, x_manifold, layer


def test_single_patch_uses_row_major_logcat_ordering():
    """The receptive field enters LogCat in row-major (h, w) order, then PLFC.

    The LogCat point is rebuilt from the patch (``_logcat_reference``) instead of
    being taken from ``manifold.log_radius_concat`` wholesale, so this pins the
    block layout and the time formula rather than comparing the library to itself.
    Only the scalar radius-matching factor is read off the library output — its
    value is not asserted (see ``_implied_logcat_scale``).
    """
    dtype = jnp.float64
    c = 1.0
    kernel_size, in_channels, out_channels = 2, 3, 4
    manifold, x_manifold, layer = _single_patch_setup(c, dtype, kernel_size, in_channels, out_channels)

    y = layer(x_manifold, c=c)  # (1, 1, 1, out_channels)

    points_KC = x_manifold.reshape(kernel_size * kernel_size, in_channels)
    scale = _implied_logcat_scale(points_KC, manifold.log_radius_concat(points_KC, c))
    logcat_A = jnp.asarray(_logcat_reference(points_KC, c, scale), dtype=dtype)

    logcat_dim = (in_channels - 1) * kernel_size**2 + 1
    plfc = HypLinearHyperboloidPLFC(manifold, logcat_dim, out_channels, rngs=nnx.Rngs(7), param_dtype=dtype)
    plfc.kernel[...] = layer.kernel[...]
    plfc.bias[...] = layer.bias[...]
    expected = plfc(logcat_A[None, :], c=c)

    assert jnp.allclose(y.reshape(1, out_channels), expected, atol=1e-10)


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_ilnn_single_patch_matches_numpy_transcription(c):
    """Conv forward equals the full LogCat + PLFC pipeline transcribed in NumPy.

    Value oracle (audit A6-03): independent of every library code path the layer
    uses, so an origin-collapsed PLFC output (or a sign flip in the MLR score)
    fails here.
    """
    dtype = jnp.float64
    kernel_size, in_channels, out_channels = 2, 3, 5
    manifold, x_manifold, layer = _single_patch_setup(c, dtype, kernel_size, in_channels, out_channels)
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(1), layer.kernel[...].shape, dtype=dtype) * 0.6
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(2), layer.bias[...].shape, dtype=dtype) * 0.3

    y = layer(x_manifold, c=c)

    points_KC = x_manifold.reshape(kernel_size * kernel_size, in_channels)
    scale = _implied_logcat_scale(points_KC, manifold.log_radius_concat(points_KC, c))
    logcat_A = _logcat_reference(points_KC, c, scale)
    expected = _plfc_reference(logcat_A[None, :], layer.kernel[...], layer.bias[...], c, v_max=layer.v_max)

    assert np.allclose(np.asarray(y).reshape(1, out_channels), expected, atol=1e-10)
    assert np.max(np.abs(expected[:, 1:])) > 0.05  # oracle is non-degenerate


def test_ilnn_forward_is_input_dependent():
    """Two distinct feature maps give distinct outputs (constant-collapse guard)."""
    dtype = jnp.float64
    c = 1.0
    manifold = Hyperboloid(dtype=dtype)
    in_channels, out_channels = 3, 4

    x = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 4, in_channels), dtype=dtype) * 0.5
    x_manifold = _proj_image(x, manifold, c)
    layer = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=nnx.Rngs(42),
        param_dtype=dtype,
    )

    y = layer(x_manifold, c=c)

    assert float(jnp.max(jnp.abs(y[0] - y[1]))) > 1e-6
    # Not pinned at the manifold origin [1/sqrt(c), 0, ..., 0].
    assert float(jnp.max(jnp.abs(y[..., 1:]))) > 1e-6


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_origin_vs_edge_padding(dtype):
    """SAME padding fills the border with the manifold origin by default; edge mode differs at borders only."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 4
    c = 1.0
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    layer_origin = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=nnx.Rngs(42),
    )
    layer_edge = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=nnx.Rngs(42),
        pad_mode="edge",
    )

    y_origin = layer_origin(x_manifold, c=c)
    y_edge = layer_edge(x_manifold, c=c)

    # Same weights (same seed): interior windows (no padding) agree, border windows differ
    assert jnp.allclose(y_origin[:, 1:3, 1:3, :], y_edge[:, 1:3, 1:3, :], atol=1e-6)
    assert jnp.max(jnp.abs(y_origin - y_edge)) > 1e-6
    # Origin padding keeps the output on the manifold
    assert _check_on_hyperboloid(y_origin.reshape(-1, out_channels), c=c, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gyro_bias_zero_is_identity(dtype):
    """At init the gyro-bias is zero -> gyroaddition with the origin is a no-op."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 4
    c = 1.0
    atol = 1e-5 if dtype == jnp.float32 else 1e-12

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    layer_plain = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=nnx.Rngs(42),
    )
    layer_gyro = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=nnx.Rngs(42),
        use_gyro_bias=True,
    )

    y_plain = layer_plain(x_manifold, c=c)
    y_gyro = layer_gyro(x_manifold, c=c)

    assert jnp.allclose(y_plain, y_gyro, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gyro_bias_on_manifold_and_trainable(dtype):
    """Nonzero gyro-bias keeps the output on the manifold and receives gradients."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 4
    c = 1.0
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    layer = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=nnx.Rngs(42),
        use_gyro_bias=True,
    )
    bias_key = jax.random.PRNGKey(7)
    layer.gyro_bias[...] = jax.random.normal(bias_key, (out_channels - 1,), dtype=layer.gyro_bias[...].dtype) * 0.3

    y = layer(x_manifold, c=c)

    assert jnp.isfinite(y).all()
    assert _check_on_hyperboloid(y.reshape(-1, out_channels), c=c, atol=atol)

    def loss_fn(model):
        return jnp.sum(model(x_manifold, c=c) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.gyro_bias[...]).all()
    assert jnp.any(grads.gyro_bias[...] != 0.0)


def test_kernel_init_std():
    """Default kernel init follows the PLFC reference (std=0.02); kernel_init_std=1.0 recovers HNN++."""
    manifold = Hyperboloid(dtype=jnp.float32)
    in_channels, out_channels, kernel_size = 9, 17, 3  # (16, 72) kernel -> 1152 samples

    layer_default = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(42),
    )
    layer_hnnpp = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(42),
        kernel_init_std=1.0,
    )

    assert abs(float(jnp.std(layer_default.kernel[...])) - 0.02) < 0.005
    assert 0.8 < float(jnp.std(layer_hnnpp.kernel[...])) < 1.2
