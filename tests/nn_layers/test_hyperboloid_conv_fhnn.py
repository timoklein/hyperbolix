"""Tests for HypConv2DHyperboloidFHNN (Chen et al. 2021 hyperboloid conv layer).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypConv2DHyperboloidFHNN-specific tests stay here.
"""

import jax
import jax.numpy as jnp
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
