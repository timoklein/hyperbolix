"""JAX test fixtures mirroring the PyTorch conftest.py.

This file contains global fixtures for JAX-based tests, providing compatible
interfaces with the PyTorch test fixtures but using JAX/NumPy random operations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hyperbolix_jax as hj

# Enable float64 support in JAX
jax.config.update("jax_enable_x64", True)



@pytest.fixture(scope="package", params=[*range(10, 13)])
def seed_jax(request: pytest.FixtureRequest) -> int:
    """Global seed for JAX reproducibility (mirrors PyTorch seeds 10, 11, 12)."""
    return request.param


@pytest.fixture(scope="package")
def rng(seed_jax: int) -> np.random.Generator:
    """Shared NumPy RNG for JAX tests, seeded to match PyTorch test runs."""
    return np.random.default_rng(seed=seed_jax)


@pytest.fixture(scope="package", params=[jnp.float32, jnp.float64])
def dtype(request: pytest.FixtureRequest) -> jnp.dtype:
    """Test both float32 and float64 precision (mirroring PyTorch)."""
    return jnp.dtype(request.param)


@pytest.fixture(scope="package")
def tolerance(dtype: jnp.dtype) -> tuple[float, float]:
    """Numerical tolerances for floating point comparisons (matches PyTorch)."""
    if dtype == jnp.float32:
        return 4e-3, 4e-3  # atol, rtol
    return 1e-7, 1e-7  # float64


@pytest.fixture(
    scope="package",
    params=[
        ("euclidean", 0.0),
        ("poincare", 1.0),
        ("hyperboloid", 1.0),
    ],
    ids=["Euclidean", "PoincareBall", "Hyperboloid"],
)
def manifold_and_c(request: pytest.FixtureRequest, rng: np.random.Generator):
    """Fixture providing (manifold_module, curvature) tuples.

    Mirrors the PyTorch manifold fixture but returns functional modules
    instead of class instances. Curvatures are sampled the same way as PyTorch.
    """
    manifold_name, _ = request.param

    if manifold_name == "euclidean":
        # Euclidean always has c=0
        return hj.manifolds.euclidean, 0.0
    elif manifold_name == "poincare":
        # Poincaré with random positive curvature (exponential distribution, rate=0.5)
        # Matches PyTorch: torch.empty(1).exponential_(0.5)
        c = float(rng.exponential(scale=2.0))  # scale = 1/rate
        return hj.manifolds.poincare, c
    elif manifold_name == "hyperboloid":
        # Hyperboloid with random positive curvature
        c = float(rng.exponential(scale=2.0))
        return hj.manifolds.hyperboloid, c
    else:
        raise ValueError(f"Unknown manifold: {manifold_name}")


@pytest.fixture(scope="package", params=[2, 5, 10, 15])
def uniform_points(manifold_and_c, dtype: jnp.dtype, request: pytest.FixtureRequest, rng: np.random.Generator) -> jnp.ndarray:
    """Generate uniformly distributed points on the manifold.

    Mirrors the PyTorch uniform_points fixture, generating the same number
    and distribution of points but using NumPy arrays converted to JAX.
    """
    manifold, c = manifold_and_c
    dim = request.param
    num_pts = 2_500 * 6  # Same as PyTorch
    np_dtype = np.dtype(dtype.name)

    if manifold == hj.manifolds.euclidean:
        # Euclidean: uniform in box [-100, 100]^d
        lower, upper = -100.0, 100.0
        data = rng.uniform(lower, upper, size=(num_pts, dim)).astype(np_dtype)
        return jnp.asarray(data)

    elif manifold == hj.manifolds.poincare:
        # Poincaré ball: uniform sampling using spherical coordinates
        # Matches PyTorch approach
        random_dirs = rng.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
        random_dirs /= np.linalg.norm(random_dirs, axis=-1, keepdims=True)
        random_radii = rng.random((num_pts, 1)).astype(np_dtype) ** (1.0 / dim)
        # Scale to ball of radius 1/√c
        points = (random_dirs * random_radii) / np.sqrt(c)
        points = jnp.asarray(points, dtype=dtype)
        proj_batch = jax.vmap(manifold.proj, in_axes=(0, None))
        return proj_batch(points, c)

    elif manifold == hj.manifolds.hyperboloid:
        # Hyperboloid: generate points on upper sheet
        # Mirrors PyTorch approach: generate in Poincaré, scale, convert
        random_dirs = rng.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
        random_dirs /= np.linalg.norm(random_dirs, axis=-1, keepdims=True)
        random_radii = rng.random((num_pts, 1)).astype(np_dtype) ** (1.0 / dim)
        poincare_points = (random_dirs * random_radii) / np.sqrt(c)

        # Scale by 0.5 to account for representational limitations (matches PyTorch)
        poincare_points = poincare_points * 0.5

        # Convert Poincaré to Hyperboloid
        # Using the standard conversion: given Poincaré point p with ||p||² < 1/c
        # Hyperboloid point: [x₀, x_rest] where
        # x₀ = (1 + c||p||²) / (√c * (1 - c||p||²))
        # x_rest = 2p / (√c * (1 - c||p||²))
        p = poincare_points
        p_sqnorm = np.sum(p**2, axis=-1, keepdims=True)
        denom = 1.0 - c * p_sqnorm
        denom = np.maximum(denom, 1e-15)  # Avoid division by zero

        sqrt_c = np.sqrt(c)
        x0 = (1.0 + c * p_sqnorm) / (sqrt_c * denom)
        x_rest = 2.0 * p / (sqrt_c * denom)

        points = np.concatenate([x0, x_rest], axis=-1).astype(np_dtype)
        points = jnp.asarray(points, dtype=dtype)

        # Project to ensure they're on the manifold
        proj_batch = jax.vmap(manifold.proj, in_axes=(0, None))
        return proj_batch(points, c)

    else:
        raise ValueError("Unknown manifold module")
