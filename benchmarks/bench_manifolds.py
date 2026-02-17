"""Benchmarks for manifold operations with JIT compilation.

These benchmarks measure:
1. Non-JIT baseline performance
2. JIT compilation overhead (first call)
3. JIT runtime performance (subsequent calls)

Run with:
    uv run pytest benchmarks/bench_manifolds.py --benchmark-only -v
"""

import jax
import jax.numpy as jnp
import pytest

import hyperbolix as hj
from hyperbolix.manifolds import Hyperboloid, Poincare

poincare = Poincare()
hyperboloid_m = Hyperboloid()

# ============================================================================
# Poincaré Ball Benchmarks
# ============================================================================


def test_poincare_dist_no_jit(benchmark, benchmark_points, curvature):
    """Benchmark Poincaré distance without JIT (baseline)."""
    points_a, points_b = jnp.array_split(benchmark_points, 2)

    # vmap over batch dimension: each point is shape (dim,)
    dist_fn = jax.vmap(
        poincare.dist,
        in_axes=(0, 0, None, None),  # (x: batch, y: batch, c: scalar, version_idx: scalar)
    )

    def run():
        result = dist_fn(points_a, points_b, curvature, hj.manifolds.poincare.VERSION_MOBIUS_DIRECT)
        return result.block_until_ready()

    benchmark(run)


def test_poincare_dist_with_jit(benchmark, benchmark_points, curvature):
    """Benchmark Poincaré distance with JIT (after warmup)."""
    points_a, points_b = jnp.array_split(benchmark_points, 2)

    dist_fn = jax.jit(
        jax.vmap(
            poincare.dist,
            in_axes=(0, 0, None, None),
        ),
        static_argnames=["version_idx"],
    )

    # Warmup JIT compilation
    _ = dist_fn(points_a, points_b, curvature, hj.manifolds.poincare.VERSION_MOBIUS_DIRECT).block_until_ready()

    def run():
        result = dist_fn(points_a, points_b, curvature, hj.manifolds.poincare.VERSION_MOBIUS_DIRECT)
        return result.block_until_ready()

    benchmark(run)


def test_poincare_expmap_no_jit(benchmark, benchmark_points, curvature):
    """Benchmark Poincaré exponential map without JIT."""
    tangent_vecs, base_points = jnp.array_split(benchmark_points, 2)

    # vmap over batch: expmap(v, x, c) for each v[i], x[i]
    expmap_fn = jax.vmap(
        poincare.expmap,
        in_axes=(0, 0, None),  # (v: batch, x: batch, c: scalar)
    )

    def run():
        result = expmap_fn(tangent_vecs, base_points, curvature)
        return result.block_until_ready()

    benchmark(run)


def test_poincare_expmap_with_jit(benchmark, benchmark_points, curvature):
    """Benchmark Poincaré exponential map with JIT."""
    tangent_vecs, base_points = jnp.array_split(benchmark_points, 2)

    expmap_fn = jax.jit(jax.vmap(poincare.expmap, in_axes=(0, 0, None)))

    # Warmup
    _ = expmap_fn(tangent_vecs, base_points, curvature).block_until_ready()

    def run():
        result = expmap_fn(tangent_vecs, base_points, curvature)
        return result.block_until_ready()

    benchmark(run)


def test_poincare_logmap_with_jit(benchmark, benchmark_points, curvature):
    """Benchmark Poincaré logarithmic map with JIT."""
    points_y, points_x = jnp.array_split(benchmark_points, 2)

    # vmap over batch: logmap(y, x, c) for each y[i], x[i]
    logmap_fn = jax.jit(
        jax.vmap(
            poincare.logmap,
            in_axes=(0, 0, None),  # (y: batch, x: batch, c: scalar)
        )
    )

    # Warmup
    _ = logmap_fn(points_y, points_x, curvature).block_until_ready()

    def run():
        result = logmap_fn(points_y, points_x, curvature)
        return result.block_until_ready()

    benchmark(run)


@pytest.mark.parametrize("version_idx", [0, 1, 2, 3])
def test_poincare_dist_versions(benchmark, benchmark_points, curvature, version_idx):
    """Compare performance of different Poincaré distance implementations.

    Tests all 4 versions:
    - 0: VERSION_MOBIUS_DIRECT (fastest)
    - 1: VERSION_MOBIUS
    - 2: VERSION_METRIC_TENSOR
    - 3: VERSION_LORENTZIAN_PROXY
    """
    points_a, points_b = jnp.array_split(benchmark_points, 2)

    dist_fn = jax.jit(
        jax.vmap(
            poincare.dist,
            in_axes=(0, 0, None, None),
        ),
        static_argnames=["version_idx"],
    )

    # Warmup
    _ = dist_fn(points_a, points_b, curvature, version_idx).block_until_ready()

    def run():
        result = dist_fn(points_a, points_b, curvature, version_idx)
        return result.block_until_ready()

    benchmark(run)


# ============================================================================
# Hyperboloid Benchmarks
# ============================================================================


def test_hyperboloid_dist_with_jit(benchmark, benchmark_points, curvature):
    """Benchmark Hyperboloid distance with JIT."""
    # Hyperboloid points need extra dimension: (batch, dim) -> (batch, dim+1)
    points_3d = jnp.concatenate([benchmark_points, jnp.ones((benchmark_points.shape[0], 1))], axis=1)
    points_a, points_b = jnp.array_split(points_3d, 2)

    # Project to hyperboloid: proj(x, c) for each x[i]
    proj_fn = jax.vmap(
        hyperboloid_m.proj,
        in_axes=(0, None),  # (x: batch, c: scalar)
    )
    points_a = proj_fn(points_a, curvature)
    points_b = proj_fn(points_b, curvature)

    dist_fn = jax.jit(
        jax.vmap(
            hyperboloid_m.dist,
            in_axes=(0, 0, None, None),  # (x: batch, y: batch, c: scalar, version_idx: scalar)
        ),
        static_argnames=["version_idx"],
    )

    # Warmup
    _ = dist_fn(points_a, points_b, curvature, hj.manifolds.hyperboloid.VERSION_DEFAULT).block_until_ready()

    def run():
        result = dist_fn(points_a, points_b, curvature, hj.manifolds.hyperboloid.VERSION_DEFAULT)
        return result.block_until_ready()

    benchmark(run)


# ============================================================================
# Math Utils Benchmarks
# ============================================================================


def test_acosh_with_jit(benchmark, benchmark_points):
    """Benchmark acosh function (already jitted, element-wise operation).

    Note: acosh operates element-wise on arrays of any shape, no vmap needed.
    """
    # acosh requires x > 1
    x = jnp.abs(benchmark_points) + 1.5

    # acosh is already jitted and works element-wise on entire array
    def run():
        result = hj.utils.math_utils.acosh(x)
        return result.block_until_ready()

    benchmark(run)


def test_atanh_with_jit(benchmark, benchmark_points):
    """Benchmark atanh function (already jitted, element-wise operation).

    Note: atanh operates element-wise on arrays of any shape, no vmap needed.
    """
    # atanh requires |x| < 1
    x = benchmark_points * 0.5

    # atanh is already jitted and works element-wise on entire array
    def run():
        result = hj.utils.math_utils.atanh(x)
        return result.block_until_ready()

    benchmark(run)


def test_cosh_with_jit(benchmark, benchmark_points):
    """Benchmark cosh function (already jitted, element-wise operation)."""
    x = benchmark_points * 2.0

    def run():
        result = hj.utils.math_utils.cosh(x)
        return result.block_until_ready()

    benchmark(run)


def test_sinh_with_jit(benchmark, benchmark_points):
    """Benchmark sinh function (already jitted, element-wise operation)."""
    x = benchmark_points * 2.0

    def run():
        result = hj.utils.math_utils.sinh(x)
        return result.block_until_ready()

    benchmark(run)
