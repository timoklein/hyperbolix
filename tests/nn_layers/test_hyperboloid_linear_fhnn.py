"""Tests for HypLinearHyperboloidFHNN (Chen et al. 2021 hyperboloid linear layer).

Also home to the value oracle for the FHCNN pure function ``_fhcnn_forward``
(Bdeir et al. 2023), whose ``normalize=True`` branch no conv layer reaches:
``HypConv2DHyperboloid`` always calls it with ``normalize=False``, and
``HypLinearHyperboloidFHCNN`` is the only public entry point into that branch.

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypLinearHyperboloid{FHNN,FHCNN}-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import HypLinearHyperboloidFHCNN, HypLinearHyperboloidFHNN


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _fhnn_reference(x_BI, kernel_OI, bias_1O, scale, c, eps=1e-5):
    """NumPy transcription of the FHNN forward (Chen et al. 2021, time-primary form).

    Independent of the library: the paper's four steps written out in NumPy.

        z    = W x + b
        y0   = exp(s) * sigmoid(z0) + 1/sqrt(c) + eps
        ||ys|| = sqrt(y0^2 - 1/c)                    (hyperboloid constraint)
        ys   = ||ys|| * z_s / ||z_s||                (spatial *direction* preserved)
    """
    z_BO = np.asarray(x_BI, dtype=np.float64) @ np.asarray(kernel_OI, dtype=np.float64).T
    z_BO = z_BO + np.asarray(bias_1O, dtype=np.float64)
    z0_B1, zs_BD = z_BO[:, 0:1], z_BO[:, 1:]

    y0_B1 = np.exp(float(scale)) * _sigmoid(z0_B1) + 1.0 / np.sqrt(c) + eps
    target_norm_B1 = np.sqrt(y0_B1**2 - 1.0 / c)
    ys_BD = target_norm_B1 * zs_BD / np.linalg.norm(zs_BD, axis=-1, keepdims=True)
    return np.concatenate([y0_B1, ys_BD], axis=-1)


def _fhcnn_normalize_reference(x_BI, kernel_OI, bias_1O, scale, c, eps=1e-5):
    """NumPy transcription of the FHCNN ``normalize=True`` forward (Bdeir et al. 2023).

    z   = W x + b
    s   = exp(scale) * sigmoid(z0)               (spatial-primary: norm is learned)
    ys  = s * z_s / ||z_s||
    y0  = sqrt(s^2 + 1/c + eps)                  (time from the constraint)
    """
    z_BO = np.asarray(x_BI, dtype=np.float64) @ np.asarray(kernel_OI, dtype=np.float64).T
    z_BO = z_BO + np.asarray(bias_1O, dtype=np.float64)
    z0_B1, zs_BD = z_BO[:, 0:1], z_BO[:, 1:]

    s_B1 = np.exp(float(scale)) * _sigmoid(z0_B1)
    ys_BD = s_B1 * zs_BD / np.linalg.norm(zs_BD, axis=-1, keepdims=True)
    y0_B1 = np.sqrt(s_B1**2 + 1.0 / c + eps)
    return np.concatenate([y0_B1, ys_BD], axis=-1)


def _hyperboloid_points(key, batch, ambient, c, dtype):
    """Batch of hyperboloid points from spatial-only tangent vectors at the origin."""
    manifold = get_hyperboloid(dtype)
    v = jax.random.normal(key, (batch, ambient), dtype=dtype) * 0.3
    v = v.at[:, 0].set(0.0)
    return jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, c)


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
    """Dropout is actually wired in: training and eval outputs differ, both on-manifold."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(params=42, dropout=43)
    layer = HypLinearHyperboloidFHNN(
        get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs, activation=jax.nn.relu, dropout_rate=0.5
    )

    y_train = layer(x, c=1.0, deterministic=False)
    y_eval = layer(x, c=1.0, deterministic=True)

    assert jnp.isfinite(y_train).all()
    assert jnp.isfinite(y_eval).all()
    assert _check_on_hyperboloid(y_train, c=1.0, atol=atol)
    assert _check_on_hyperboloid(y_eval, c=1.0, atol=atol)
    # A dropped or ignored ``deterministic`` flag would make the two identical.
    assert not jnp.allclose(y_train, y_eval, atol=1e-4)


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


# --------------------------------------------------------------------------- #
# Target spatial norm: sqrt(y0 - 1/sqrt(c)) * sqrt(y0 + 1/sqrt(c)), not the
# algebraically equal sqrt(y0**2 - 1/c). `y0**2` overflows float32 to inf past
# y0 = sqrt(FLT_MAX) ~ 1.844e19, which `capped_exp` explicitly allows y0 to
# reach: it caps a runaway `scale` at exp(0.99*log(FLT_MAX)) ~ 6.1e37 to keep
# the value finite, and the squaring then threw that finiteness away.
# --------------------------------------------------------------------------- #
def _fhnn_layer_with_scale(c, scale, out_dim=4, in_dim=4, spatial_bias=(1.0, -2.0, 3.0)):
    """FHNN layer wired so that z0 == 0 exactly and the spatial logits are a fixed constant.

    Zero kernel + zero time bias => sigmoid(z0) == 0.5, so y0 == 0.5*exp(scale) + 1/sqrt(c) + eps
    is set purely by `scale`. That is the only handle that reaches the y0 >= 1e19 regime without
    hand-calling the private forward.
    """
    dtype = jnp.float32
    layer = HypLinearHyperboloidFHNN(get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)
    layer.kernel[...] = jnp.zeros((out_dim, in_dim), dtype=dtype)
    bias = jnp.asarray([0.0, *spatial_bias], dtype=dtype)[None, :]
    layer.bias[...] = bias
    layer.scale[...] = jnp.asarray(scale, dtype=dtype)
    return layer


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_fhnn_huge_time_coordinate_stays_finite(c):
    """y0 ~ 1e20 in float32: output finite, where `sqrt(y0**2 - 1/c)` gave inf.

    Gate for the factored target-norm form. `capped_exp` guarantees y0 itself is finite here, so
    an inf in the output can only come from the squaring.
    """
    dtype = jnp.float32
    # y0 = 0.5*exp(scale) + 1/sqrt(c) + eps ~ 1e20 -- past sqrt(FLT_MAX) = 1.844e19.
    layer = _fhnn_layer_with_scale(c, float(np.log(2e20)))
    x = jnp.zeros((3, 4), dtype=dtype)

    y = layer(x, c=c)

    y0 = np.asarray(y[:, 0], dtype=np.float32)
    assert np.all(np.isfinite(y0)) and np.all(y0 > 1e19), y0
    # The pre-fix expression, evaluated on exactly this y0, is where the inf came from.
    with np.errstate(over="ignore"):
        assert not np.all(np.isfinite(np.sqrt(y0**2 - np.float32(1.0 / c))))
    # ... and the layer no longer inherits it.
    assert np.all(np.isfinite(np.asarray(y))), np.asarray(y)
    # At y0 >> 1/sqrt(c) the constraint forces ||y_s|| -> y0.
    ys_norm = np.linalg.norm(np.asarray(y[:, 1:], dtype=np.float64), axis=-1)
    np.testing.assert_allclose(ys_norm, np.asarray(y0, dtype=np.float64), rtol=1e-6)
    # Gradients through the huge-y0 branch stay finite too.
    grad = nnx.grad(lambda m: jnp.sum(m(x, c=c)))(layer)
    assert np.all(np.isfinite(np.asarray(grad["kernel"][...])))
    assert np.all(np.isfinite(np.asarray(grad["bias"][...])))


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("scale", [-3.0, 0.0, 2.0, 8.0, 20.0])
def test_fhnn_target_norm_matches_old_expression_in_range(c, scale):
    """Ordinary range: the factored form equals `sqrt(y0**2 - 1/c)` to float32 rounding.

    The float32 spread between the two forms peaks at ~3e-6 relative where the margin
    ``y0 - 1/sqrt(c)`` is smallest (~1e-2); everywhere above that it is at the 1-ulp level.
    """
    layer = _fhnn_layer_with_scale(c, scale)
    x = jnp.zeros((2, 4), dtype=jnp.float32)

    y = layer(x, c=c)

    y0 = np.asarray(y[:, 0], dtype=np.float32)
    old_norm = np.sqrt(y0**2 - np.float32(1.0 / c))  # the pre-fix expression, in float32
    new_norm = np.linalg.norm(np.asarray(y[:, 1:], dtype=np.float64), axis=-1)
    assert np.all(np.isfinite(old_norm))
    np.testing.assert_allclose(new_norm, np.asarray(old_norm, dtype=np.float64), rtol=1e-5)


# --------------------------------------------------------------------------- #
# Forward value oracles (audit A6-02): independent NumPy transcriptions of the
# published formulas, so a sign flip or a collapsed spatial output fails here.
# float64 only — these are exact-agreement assertions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", [0.5, 1.0])
def test_fhnn_forward_matches_numpy_transcription(c):
    """FHNN forward equals the Chen et al. 2021 formulas written out in NumPy.

    Pins the *signed* spatial output: the layer rescales ``z_s`` to the norm the
    hyperboloid constraint demands but must leave its direction untouched, so
    negating the spatial branch (or zeroing it) breaks the comparison.
    """
    dtype = jnp.float64
    manifold = get_hyperboloid(dtype)
    batch_size, in_dim, out_dim = 5, 6, 7

    x = _hyperboloid_points(jax.random.PRNGKey(0), batch_size, in_dim, c, dtype)
    layer = HypLinearHyperboloidFHNN(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)
    # Overwrite the (tiny, time-column-zeroed) default init with a generic dense kernel.
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(1), (out_dim, in_dim), dtype=dtype) * 0.4
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(2), (1, out_dim), dtype=dtype) * 0.2

    y = layer(x, c=c)
    expected = _fhnn_reference(x, layer.kernel[...], layer.bias[...], layer.scale[...], c, eps=layer.eps)

    assert np.allclose(np.asarray(y), expected, atol=1e-12)
    # Spatial direction is preserved, not reflected.
    z_BO = np.asarray(x) @ np.asarray(layer.kernel[...]).T + np.asarray(layer.bias[...])
    cos = np.sum(np.asarray(y)[:, 1:] * z_BO[:, 1:], axis=-1) / (
        np.linalg.norm(np.asarray(y)[:, 1:], axis=-1) * np.linalg.norm(z_BO[:, 1:], axis=-1)
    )
    assert np.allclose(cos, 1.0, atol=1e-12)


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_fhcnn_normalize_matches_numpy_transcription(c):
    """FHCNN (``normalize=True``) equals the Bdeir et al. 2023 formulas in NumPy.

    This is the only public entry point into the ``normalize=True`` branch of
    ``_fhcnn_forward`` (``HypConv2DHyperboloid`` hard-codes ``normalize=False``),
    so without this test a sign flip on ``scale * z_s / ||z_s||`` is invisible.
    """
    dtype = jnp.float64
    manifold = get_hyperboloid(dtype)
    batch_size, in_dim, out_dim = 5, 6, 7
    init_scale, eps = 2.3, 1e-5

    x = _hyperboloid_points(jax.random.PRNGKey(3), batch_size, in_dim, c, dtype)
    layer = HypLinearHyperboloidFHCNN(
        manifold,
        in_dim,
        out_dim,
        rngs=nnx.Rngs(0),
        normalize=True,
        init_scale=init_scale,
        eps=eps,
        param_dtype=dtype,
    )
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(4), (out_dim, in_dim), dtype=dtype) * 0.4
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(5), (1, out_dim), dtype=dtype) * 0.2

    y = layer(x, c=c)
    expected = _fhcnn_normalize_reference(x, layer.kernel[...], layer.bias[...], init_scale, c, eps=eps)

    assert np.allclose(np.asarray(y), expected, atol=1e-12)
