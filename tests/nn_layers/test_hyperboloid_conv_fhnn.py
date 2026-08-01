"""Tests for HypConv2DHyperboloidFHNN (Chen et al. 2021 hyperboloid conv layer).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypConv2DHyperboloidFHNN-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HypConv2DHyperboloidFHNN


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_time_floor(dtype):
    """Test that FHNN conv time coordinate y0 >= 1/sqrt(c) always holds."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1
    x_manifold = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.proj(p, c))))(x)

    rngs = nnx.Rngs(42)
    layer = HypConv2DHyperboloidFHNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    y = layer(x_manifold, c=c)

    assert (y[..., 0] >= 1.0 / jnp.sqrt(c)).all()


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_conv_fhnn_1x1_matches_numpy_transcription(c):
    """A 1x1 conv is HCat(single point) = identity followed by the FHNN linear map.

    Value oracle (audit A6-02): the Chen et al. 2021 formulas transcribed in NumPy,

        z  = W x + b,   y0 = exp(s)*sigmoid(z0) + 1/sqrt(c) + eps,
        ys = sqrt(y0^2 - 1/c) * z_s/||z_s||,

    so negating (or zeroing) the spatial branch fails here. Without it the FHNN
    conv only checked the time floor, which a sign flip leaves untouched.
    """
    dtype = jnp.float64
    manifold = Hyperboloid(dtype=dtype)
    batch, height, width, in_channels, out_channels = 2, 3, 3, 4, 5
    eps = 1e-5

    v = jax.random.normal(jax.random.PRNGKey(0), (batch, height, width, in_channels), dtype=dtype) * 0.3
    v = v.at[..., 0].set(0.0)
    x = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.expmap_0(p, c))))(v)

    layer = HypConv2DHyperboloidFHNN(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding="VALID",
        eps=eps,
        rngs=nnx.Rngs(0),
        param_dtype=dtype,
    )
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(1), (out_channels, in_channels), dtype=dtype) * 0.4
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(2), (1, out_channels), dtype=dtype) * 0.2

    y = layer(x, c=c)

    x_NC = np.asarray(x, dtype=np.float64).reshape(-1, in_channels)
    z_NO = x_NC @ np.asarray(layer.kernel[...], dtype=np.float64).T + np.asarray(layer.bias[...], dtype=np.float64)
    z0_N1, zs_ND = z_NO[:, 0:1], z_NO[:, 1:]
    scale = float(layer.scale[...])
    y0_N1 = np.exp(scale) / (1.0 + np.exp(-z0_N1)) + 1.0 / np.sqrt(c) + eps
    ys_ND = np.sqrt(y0_N1**2 - 1.0 / c) * zs_ND / np.linalg.norm(zs_ND, axis=-1, keepdims=True)
    expected = np.concatenate([y0_N1, ys_ND], axis=-1).reshape(batch, height, width, out_channels)

    assert np.allclose(np.asarray(y), expected, atol=1e-11)
