"""Tests for HypConv2DHyperboloidILNN (Intrinsic Lorentz convolution: LogCat + PLFC, Shi et al. 2026).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypConv2DHyperboloidILNN-specific tests stay here.
"""

import jax
import jax.numpy as jnp
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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_forward_is_plfc_of_log_radius_concat(dtype):
    """Conv forward equals PLFC(LogCat(patch)) — the Shi et al. 2026 Eq. (11) pipeline."""
    key = jax.random.PRNGKey(0)
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    kernel_size, in_channels, out_channels = 2, 3, 4

    # Image size == kernel size with VALID padding -> exactly one receptive field
    x = jax.random.normal(key, (1, kernel_size, kernel_size, in_channels), dtype=dtype) * 0.1
    x_manifold = _proj_image(x, manifold, c)

    layer = HypConv2DHyperboloidILNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(42),
        padding="VALID",
    )
    y = layer(x_manifold, c=c)  # (1, 1, 1, out_channels)

    # Manual pipeline: LogCat over the receptive field, then PLFC with shared weights
    points_KC = x_manifold.reshape(kernel_size * kernel_size, in_channels)
    logcat_A = manifold.log_radius_concat(points_KC, c)

    logcat_dim = (in_channels - 1) * kernel_size**2 + 1
    plfc = HypLinearHyperboloidPLFC(manifold, logcat_dim, out_channels, rngs=nnx.Rngs(7))
    plfc.kernel[...] = layer.kernel[...]
    plfc.bias[...] = layer.bias[...]
    expected = plfc(logcat_A[None, :], c=c)

    atol = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.allclose(y.reshape(1, out_channels), expected, atol=atol)


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
