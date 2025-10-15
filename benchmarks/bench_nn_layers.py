"""Benchmarks for neural network layers with JIT compilation.

These benchmarks measure:
1. Forward pass performance
2. Forward + backward pass (with gradient computation)
3. JIT vs non-JIT comparison

Run with:
    uv run pytest benchmarks/bench_nn_layers.py --benchmark-only -v
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import hyperbolix_jax.manifolds as manifolds
from hyperbolix_jax.nn_layers import (
    HypLinearHyperboloid,
    HypLinearPoincare,
    HypRegressionPoincare,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def layer_input(batch_size, dim):
    """Generate random layer input."""
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (batch_size, dim)) * 0.1


@pytest.fixture
def poincare_linear_layer(dim):
    """Create a Poincaré linear layer."""
    rngs = nnx.Rngs(42)
    return HypLinearPoincare(
        manifold_module=manifolds.poincare,
        in_dim=dim,
        out_dim=dim,
        rngs=rngs,
        input_space="manifold",
    )


@pytest.fixture
def hyperboloid_linear_layer(dim):
    """Create a Hyperboloid linear layer.

    Note: Hyperboloid manifold lives in R^(dim+1), so in_dim and out_dim are dim+1.
    """
    rngs = nnx.Rngs(42)
    return HypLinearHyperboloid(
        manifold_module=manifolds.hyperboloid,
        in_dim=dim + 1,  # Ambient dimension
        out_dim=dim + 1,  # Ambient dimension
        rngs=rngs,
        input_space="manifold",
    )


@pytest.fixture
def poincare_regression_layer(dim):
    """Create a Poincaré regression layer."""
    rngs = nnx.Rngs(42)
    return HypRegressionPoincare(
        manifold_module=manifolds.poincare,
        in_dim=dim,
        out_dim=10,  # 10 classes
        rngs=rngs,
    )


@pytest.fixture
def two_layer_network(dim):
    """Create a 2-layer Poincaré network."""
    rngs = nnx.Rngs(42)
    layer1 = HypLinearPoincare(
        manifold_module=manifolds.poincare,
        in_dim=dim,
        out_dim=dim * 2,
        rngs=rngs,
        input_space="manifold",
    )
    layer2 = HypLinearPoincare(
        manifold_module=manifolds.poincare,
        in_dim=dim * 2,
        out_dim=dim,
        rngs=rngs,
        input_space="manifold",
    )
    return layer1, layer2


# ============================================================================
# Poincaré Layer Benchmarks
# ============================================================================


def test_poincare_forward_pass_no_jit(benchmark, poincare_linear_layer, layer_input):
    """Benchmark Poincaré forward pass without JIT."""

    def run():
        result = poincare_linear_layer(layer_input, c=1.0)
        return result.block_until_ready()

    benchmark(run)


def test_poincare_forward_pass_with_jit(benchmark, poincare_linear_layer, layer_input):
    """Benchmark Poincaré forward pass with JIT."""

    @nnx.jit
    def forward(model, x, c):
        return model(x, c)

    # Warmup
    _ = forward(poincare_linear_layer, layer_input, 1.0).block_until_ready()

    def run():
        result = forward(poincare_linear_layer, layer_input, 1.0)
        return result.block_until_ready()

    benchmark(run)


def test_poincare_forward_backward_with_jit(benchmark, poincare_linear_layer, layer_input):
    """Benchmark Poincaré forward + backward pass with JIT."""

    def loss_fn(model, x, c):
        output = model(x, c)
        return jnp.sum(output**2)

    # Create jitted grad function
    grad_fn = nnx.jit(nnx.value_and_grad(loss_fn))

    # Warmup
    _ = grad_fn(poincare_linear_layer, layer_input, 1.0)

    def run():
        loss, grads = grad_fn(poincare_linear_layer, layer_input, 1.0)
        # Ensure computation completes
        jax.tree.map(lambda x: x.block_until_ready(), grads)
        return loss.block_until_ready()

    benchmark(run)


# ============================================================================
# Hyperboloid Layer Benchmarks
# ============================================================================


def test_hyperboloid_forward_pass_with_jit(benchmark, hyperboloid_linear_layer, layer_input):
    """Benchmark Hyperboloid forward pass with JIT."""
    # Hyperboloid needs dim+1: (batch, dim) -> (batch, dim+1)
    layer_input_3d = jnp.concatenate([layer_input, jnp.ones((layer_input.shape[0], 1))], axis=1)

    # Project to hyperboloid: proj(x, c) for each x[i]
    proj_fn = jax.vmap(
        manifolds.hyperboloid.proj,
        in_axes=(0, None),  # (x: batch, c: scalar)
    )
    layer_input_3d = proj_fn(layer_input_3d, 1.0)

    @nnx.jit
    def forward(model, x, c):
        return model(x, c)

    # Warmup
    _ = forward(hyperboloid_linear_layer, layer_input_3d, 1.0).block_until_ready()

    def run():
        result = forward(hyperboloid_linear_layer, layer_input_3d, 1.0)
        return result.block_until_ready()

    benchmark(run)


# ============================================================================
# Regression Layer Benchmarks
# ============================================================================


def test_regression_forward_with_jit(benchmark, poincare_regression_layer, layer_input):
    """Benchmark regression layer forward pass with JIT."""

    @nnx.jit
    def forward(model, x, c):
        return model(x, c)

    # Warmup
    _ = forward(poincare_regression_layer, layer_input, 1.0).block_until_ready()

    def run():
        result = forward(poincare_regression_layer, layer_input, 1.0)
        return result.block_until_ready()

    benchmark(run)


# ============================================================================
# Multi-Layer Network Benchmarks
# ============================================================================


def test_two_layer_forward_with_jit(benchmark, two_layer_network, layer_input):
    """Benchmark 2-layer Poincaré network with JIT."""
    layer1, layer2 = two_layer_network

    @nnx.jit
    def forward(l1, l2, x, c):
        x = l1(x, c)
        x = l2(x, c)
        return x

    # Warmup
    _ = forward(layer1, layer2, layer_input, 1.0).block_until_ready()

    def run():
        result = forward(layer1, layer2, layer_input, 1.0)
        return result.block_until_ready()

    benchmark(run)


def test_two_layer_forward_backward_with_jit(benchmark, two_layer_network, layer_input):
    """Benchmark 2-layer network forward + backward pass with JIT."""
    layer1, layer2 = two_layer_network

    def loss_fn(l1, l2, x, c):
        x = l1(x, c)
        x = l2(x, c)
        return jnp.sum(x**2)

    # Create jitted grad function
    grad_fn = nnx.jit(nnx.value_and_grad(loss_fn, argnums=(0, 1)))

    # Warmup
    _ = grad_fn(layer1, layer2, layer_input, 1.0)

    def run():
        loss, (grads1, grads2) = grad_fn(layer1, layer2, layer_input, 1.0)
        # Ensure computation completes
        jax.tree.map(lambda x: x.block_until_ready(), grads1)
        jax.tree.map(lambda x: x.block_until_ready(), grads2)
        return loss.block_until_ready()

    benchmark(run)
