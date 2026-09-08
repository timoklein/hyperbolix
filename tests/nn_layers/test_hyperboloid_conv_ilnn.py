"""Tests for HypConv2DHyperboloidILNN (Intrinsic Lorentz convolution: LogCat + PLFC, Shi et al. 2026).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypConv2DHyperboloidILNN-specific tests stay here.
"""

import itertools

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
    """Default kernel init is fan-out ``sqrt(1/out_spatial)``; explicit values are passed through.

    ``kernel_init_std=0.02`` must still reproduce the Shi et al. 2026 reference draw
    bit-for-bit (same seed), since that is the documented escape hatch.
    """
    manifold = Hyperboloid(dtype=jnp.float32)
    in_channels, out_channels, kernel_size = 9, 17, 3  # (16, 72) kernel -> 1152 samples
    out_spatial = out_channels - 1

    def build(**kwargs):
        return HypConv2DHyperboloidILNN(
            manifold_module=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            rngs=nnx.Rngs(42),
            **kwargs,
        )

    layer_default, layer_ref, layer_hnnpp = build(), build(kernel_init_std=0.02), build(kernel_init_std=1.0)

    expected_std = (1.0 / out_spatial) ** 0.5  # = 0.25 at out_spatial=16
    assert abs(float(jnp.std(layer_default.kernel[...])) - expected_std) < 0.1 * expected_std
    assert 0.8 < float(jnp.std(layer_hnnpp.kernel[...])) < 1.2
    # Same seed ⇒ the three inits differ only by their scalar multiplier.
    assert jnp.allclose(layer_ref.kernel[...], 0.02 * layer_hnnpp.kernel[...], atol=1e-9)
    assert jnp.allclose(layer_default.kernel[...], expected_std * layer_hnnpp.kernel[...], atol=1e-8)


@pytest.mark.parametrize("kernel_size", [2, 3])
def test_depth3_stack_preserves_spatial_norm_at_init(kernel_size):
    """A depth-3 stack at the default init neither collapses to the origin nor blows up.

    The default ``kernel_init_std`` is coupled to the LogCat digamma sign fix
    (2026-07-31): the old fixed ``0.02`` was calibrated against the pre-fix ~sqrt(N)
    amplification, so under the corrected shrink it contracts the mean spatial norm by
    ~15x per layer (probe-measured ratio 0.068 for this configuration) and the stack
    is pinned at the manifold origin by layer 3. Deterministic: float64, fixed seeds.
    """
    dtype = jnp.float64
    c = 0.1
    channels = 17  # ambient; out_spatial = 16
    manifold = Hyperboloid(dtype=dtype)

    tangent_BHWC = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, channels), dtype=dtype) * 0.3
    tangent_BHWC = tangent_BHWC.at[..., 0].set(0.0)  # tangent at the origin
    x_BHWC = jax.vmap(jax.vmap(jax.vmap(lambda t: manifold.expmap_0(t, c))))(tangent_BHWC)

    rngs = nnx.Rngs(1234)
    norms = [float(jnp.mean(jnp.linalg.norm(x_BHWC[..., 1:], axis=-1)))]
    for _ in range(3):
        layer = HypConv2DHyperboloidILNN(
            manifold_module=manifold,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            rngs=rngs,
            param_dtype=dtype,
        )
        x_BHWC = layer(x_BHWC, c=c)
        norms.append(float(jnp.mean(jnp.linalg.norm(x_BHWC[..., 1:], axis=-1))))

    assert norms[0] > 0.5, f"probe input is degenerate: {norms}"
    for i, (prev, cur) in enumerate(itertools.pairwise(norms), start=1):
        assert 0.1 * prev < cur < 10.0 * prev, f"layer {i} spatial-norm gain {cur / prev:.3g} out of band: {norms}"


# --------------------------------------------------------------------------- #
# Far-field accuracy of the gyro-bias boost
# --------------------------------------------------------------------------- #
FAR_C = 1.0
FAR_A = 9.5  # scaled geodesic radius a = sqrt(c) * d_0(x)
FAR_A_SUB = 6.5  # subdominant output channels, ~3 nats below the radius-carrying one


def _far_pixels_KC(n, in_channels, c, a, seed):
    """Float64 hyperboloid points at scaled radius ``a``, random directions."""
    dir_KI = jax.random.normal(jax.random.PRNGKey(seed), (n, in_channels - 1), dtype=jnp.float64)
    dir_KI = dir_KI / jnp.linalg.norm(dir_KI, axis=-1, keepdims=True)
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=jnp.float64))
    time_K1 = jnp.full((n, 1), jnp.cosh(jnp.asarray(a, dtype=jnp.float64)) / sqrt_c)
    return jnp.concatenate([time_K1, (jnp.sinh(jnp.asarray(a, dtype=jnp.float64)) / sqrt_c) * dir_KI], axis=-1)


def _scaled_radius(x_A, c):
    """``a = sqrt(c)*d_0(x) = arcsinh(sqrt(c)*||x_s||)``, read off the spatial part."""
    return jnp.arcsinh(jnp.sqrt(c) * jnp.linalg.norm(jnp.asarray(x_A, dtype=jnp.float64)[..., 1:], axis=-1))


def _max_rel(a32, a64):
    """Max-abs difference normalized by the max-abs of the float64 array."""
    a32, a64 = np.asarray(a32, dtype=np.float64), np.asarray(a64, dtype=np.float64)
    return float(np.max(np.abs(a32 - a64)) / np.max(np.abs(a64)))


def test_gyro_bias_far_field_matches_float64():
    """float32 ILNN with a gyro-bias tracks float64 at scaled geodesic radius a ~ 9.5.

    One receptive field (image size == kernel size, VALID), every pixel at a = 9.5, so LogCat hands
    the PLFC a point at a = 9.4 and the kernel puts the output back at a = 9.5 — the radius at which
    the gyro-bias boost ``y <- y (+) exp_0([0, b])`` used to lose its significant bits silently.
    The pre-2026-09-08 spelling of that boost stays finite here and is wrong by 5.3e-2 in geodesic
    distance and 3.0e-2 relative in the parameter gradients.

    Measured float32-vs-float64 error of the current spelling (float64 reference on the same
    float32-rounded inputs and parameters): 5.4e-4 geodesic distance, 1.6e-7 relative on the
    kernel / bias / gyro-bias gradients
    (``logs/2026-09-08_hyperboloid_tangent_primitives/probe_layers_far_field5.out``).
    """
    c, in_channels, out_channels, kernel_size = FAR_C, 5, 9, 2
    manifold64 = Hyperboloid(dtype=jnp.float64)

    # The float64 reference must see exactly the pixels the float32 layer sees.
    pixels_KC = _far_pixels_KC(kernel_size**2, in_channels, c, FAR_A, seed=10)
    x32_BHWC = pixels_KC.reshape(1, kernel_size, kernel_size, in_channels).astype(jnp.float32)
    x64_BHWC = x32_BHWC.astype(jnp.float64)

    # Kernel rows scaled so the sinh-lift argument sqrt(c)*v is 9.5 on channel 0 and 6.5 on the
    # rest: at a zero bias the PLFC score is exactly linear in the row norm, so one division per
    # row sets it. The output point then sits at a ~ 9.5 with one dominant channel.
    logcat_A = manifold64.log_radius_concat(x64_BHWC.reshape(kernel_size**2, in_channels), c)
    rows_OI = jax.random.normal(jax.random.PRNGKey(11), (out_channels - 1, logcat_A.shape[0] - 1), dtype=jnp.float64)
    rows_OI = rows_OI / jnp.linalg.norm(rows_OI, axis=-1, keepdims=True)
    score_O = jnp.arcsinh(jnp.sqrt(c) * (logcat_A[1:] @ rows_OI.T))
    target_O = jnp.asarray([FAR_A] + [FAR_A_SUB] * (out_channels - 2), dtype=jnp.float64)
    kernel_OI = (target_O / score_O)[:, None] * rows_OI

    gyro_O = jax.random.normal(jax.random.PRNGKey(12), (out_channels - 1,), dtype=jnp.float64)
    gyro_O = 0.2 * gyro_O / jnp.linalg.norm(gyro_O)

    def run(dtype, x_BHWC):
        layer = HypConv2DHyperboloidILNN(
            manifold_module=Hyperboloid(dtype=dtype),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            rngs=nnx.Rngs(0),
            padding="VALID",
            use_gyro_bias=True,
            param_dtype=dtype,
        )
        # Round to float32 first in both runs: the float64 reference must differ from the float32
        # layer only in compute precision, not in the parameter values it was handed.
        layer.kernel[...] = jnp.asarray(kernel_OI, dtype=jnp.float32).astype(dtype)
        layer.gyro_bias[...] = jnp.asarray(gyro_O, dtype=jnp.float32).astype(dtype)
        x_BHWC = jnp.asarray(x_BHWC, dtype=dtype)

        def loss_fn(model):
            return jnp.sum(model(x_BHWC, c=c)[..., 1:])

        _, grads = nnx.value_and_grad(loss_fn)(layer)
        return layer(x_BHWC, c=c), (grads.kernel[...], grads.bias[...], grads.gyro_bias[...])

    y64_BHWC, grads64 = run(jnp.float64, x64_BHWC)
    y32_BHWC, grads32 = run(jnp.float32, x32_BHWC)

    # The regime is the test: a construction that drifted back to small radius would pass vacuously.
    a_out = _scaled_radius(y64_BHWC, c)
    assert 9.2 < float(jnp.min(a_out)) and float(jnp.max(a_out)) < 10.2, a_out

    y32_NC = manifold64.proj_batch(y32_BHWC.astype(jnp.float64), c).reshape(-1, out_channels)
    y64_NC = manifold64.proj_batch(y64_BHWC, c).reshape(-1, out_channels)
    dist_N = jax.vmap(manifold64.dist, in_axes=(0, 0, None))(y32_NC, y64_NC, c)
    assert jnp.isfinite(y32_BHWC).all()
    assert float(jnp.max(dist_N)) < 3e-3  # measured 5.4e-4; old spelling 5.3e-2
    for g32, g64 in zip(grads32, grads64, strict=True):
        assert _max_rel(g32, g64) < 2e-6  # measured 1.6e-7; old spelling 3.0e-2
