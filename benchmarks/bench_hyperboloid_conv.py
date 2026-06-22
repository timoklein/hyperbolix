"""Benchmarks for hyperboloid convolutional layers with JIT compilation.

These benchmarks measure:
1. Forward pass performance (JIT) for the four hyperboloid conv layers
2. Forward + backward pass (JIT) for the same layers
3. The shared ``extract_patches`` op in isolation

Patch extraction (the ``conv_general_dilated_patches`` im2col) is the dominant cost of
these convolutions under JIT, so benchmark (3) is the regression guard for the unified
extraction path.

Run with:
    uv run pytest benchmarks/bench_hyperboloid_conv.py --benchmark-only -v

For GPU timings, install a CUDA jaxlib, e.g.:
    CUDA_VISIBLE_DEVICES=0 uv run --with "jax[cuda12]" \
        pytest benchmarks/bench_hyperboloid_conv.py --benchmark-only -q

Dimension key:
  B: batch size    H/W: spatial dims    C: ambient channels (time + spatial)
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import (
    FGGConv2D,
    HypConv2DHyperboloid,
    HypConv2DHyperboloidFHNN,
    HypConv2DHyperboloidILNN,
    extract_patches,
)

# ============================================================================
# Config + fixtures
# ============================================================================

# Float32 manifold (typical training precision) even though conftest enables x64.
manifold = Hyperboloid(dtype=jnp.float32)

# Representative conv config: B=32, 16x16 spatial, C=33 ambient (32 spatial), k=3, s=1.
BATCH = 32
HEIGHT = WIDTH = 16
IN_CHANNELS = 33
OUT_CHANNELS = 33
KERNEL = 3
CURVATURE = 1.0


@pytest.fixture
def conv_input():
    """Random hyperboloid feature map (B, H, W, C) in float32."""
    key = jax.random.PRNGKey(42)
    spatial = jax.random.normal(key, (BATCH, HEIGHT, WIDTH, IN_CHANNELS - 1), dtype=jnp.float32) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0 / CURVATURE)
    return jnp.concatenate([time, spatial], axis=-1)  # (B, H, W, C) float32


# The four hyperboloid conv layers sharing the unified extract_patches.
LAYER_BUILDERS = {
    "HypConv2DHyperboloid": lambda: HypConv2DHyperboloid(manifold, IN_CHANNELS, OUT_CHANNELS, KERNEL, rngs=nnx.Rngs(0)),
    "FHNN": lambda: HypConv2DHyperboloidFHNN(manifold, IN_CHANNELS, OUT_CHANNELS, KERNEL, rngs=nnx.Rngs(0)),
    "FGGConv2D": lambda: FGGConv2D(manifold, IN_CHANNELS, OUT_CHANNELS, KERNEL, rngs=nnx.Rngs(0)),
    "ILNN": lambda: HypConv2DHyperboloidILNN(manifold, IN_CHANNELS, OUT_CHANNELS, KERNEL, rngs=nnx.Rngs(0)),
}


@pytest.fixture(params=list(LAYER_BUILDERS), ids=list(LAYER_BUILDERS))
def conv_layer(request):
    """Instantiate one of the four hyperboloid conv layers."""
    return LAYER_BUILDERS[request.param]()


# ============================================================================
# Layer benchmarks
# ============================================================================


def test_conv_forward_with_jit(benchmark, conv_layer, conv_input):
    """Benchmark conv forward pass with JIT."""

    @nnx.jit
    def forward(model, x, c):
        return model(x, c)

    # Warmup
    _ = forward(conv_layer, conv_input, CURVATURE).block_until_ready()

    def run():
        return forward(conv_layer, conv_input, CURVATURE).block_until_ready()

    benchmark(run)


def test_conv_forward_backward_with_jit(benchmark, conv_layer, conv_input):
    """Benchmark conv forward + backward pass with JIT."""

    def loss_fn(model, x, c):
        return jnp.sum(model(x, c) ** 2)

    grad_fn = nnx.jit(nnx.value_and_grad(loss_fn))

    # Warmup
    _ = grad_fn(conv_layer, conv_input, CURVATURE)

    def run():
        loss, grads = grad_fn(conv_layer, conv_input, CURVATURE)
        jax.tree.map(lambda g: g.block_until_ready(), grads)
        return loss.block_until_ready()

    benchmark(run)


# ============================================================================
# Shared extract_patches benchmark (the unified op this PR touches)
# ============================================================================


def test_extract_patches_with_jit(benchmark, conv_input):
    """Benchmark the shared patch-extraction op in isolation (origin padding)."""

    @jax.jit
    def extract(x):
        return extract_patches(x, (KERNEL, KERNEL), (1, 1), "SAME", "origin", CURVATURE)

    # Warmup
    _ = extract(conv_input).block_until_ready()

    def run():
        return extract(conv_input).block_until_ready()

    benchmark(run)
