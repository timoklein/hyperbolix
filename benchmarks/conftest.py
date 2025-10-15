"""Shared fixtures for benchmarks."""

import jax
import pytest

# Enable float64 for numerical accuracy
jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=[2, 10, 50, 128])
def dim(request):
    """Parametrize over dimensions."""
    return request.param


@pytest.fixture(params=[100, 1000])
def batch_size(request):
    """Parametrize over batch sizes."""
    return request.param


@pytest.fixture
def random_key():
    """Random key for reproducibility."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def benchmark_points(dim, batch_size, random_key):
    """Generate random points for benchmarking.

    Returns points scaled to be well within the manifold (not near boundary).
    """
    points = jax.random.normal(random_key, (batch_size, dim))
    # Scale to ~0.5 magnitude to avoid boundary issues
    return points * 0.1


@pytest.fixture
def curvature():
    """Fixed curvature for benchmarks."""
    return 1.0
