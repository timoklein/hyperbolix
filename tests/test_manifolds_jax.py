"""Initial JAX ports of manifold tests.

The cases mirror the Torch suite in ``tests/test_manifolds.py`` but run
against the emerging ``hyperbolix_jax`` backend. Implementations are still
stubs, so the assertions currently exercise the ``NotImplementedError`` paths
to anchor forthcoming work.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from hyperbolix_jax.manifolds import Euclidean, Hyperboloid, Manifold, PoincareBall


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Shared RNG to keep the JAX fixtures reproducible."""

    return np.random.default_rng(seed=7727)


@pytest.fixture(scope="session", params=[jnp.float32, jnp.float64])
def dtype(request: pytest.FixtureRequest) -> jnp.dtype:
    return jnp.dtype(request.param)


@pytest.fixture(scope="session")
def tolerance(dtype: jnp.dtype) -> tuple[float, float]:
    if dtype == jnp.float32:
        return 4e-3, 4e-3
    return 1e-7, 1e-7


@pytest.fixture(scope="session", params=[Euclidean, Hyperboloid, PoincareBall], ids=["Euclidean", "Hyperboloid", "PoincareBall"])
def manifold(dtype: jnp.dtype, request: pytest.FixtureRequest, rng: np.random.Generator) -> Manifold:
    c = jnp.asarray(rng.exponential(scale=2.0, size=(1,)), dtype=dtype)
    return request.param(c=c, dtype=dtype)


@pytest.fixture(scope="session", params=[2, 5, 10, 15])
def uniform_points(manifold: Manifold, dtype: jnp.dtype, request: pytest.FixtureRequest, rng: np.random.Generator) -> jnp.ndarray:
    dim = request.param
    num_pts = 2_500 * 6
    np_dtype = np.dtype(dtype.name)
    if isinstance(manifold, Euclidean):
        lower, upper = -100.0, 100.0
        data = rng.uniform(lower, upper, size=(num_pts, dim)).astype(np_dtype)
        return jnp.asarray(data)
    random_dirs = rng.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
    random_dirs /= np.linalg.norm(random_dirs, axis=-1, keepdims=True)
    random_radii = rng.random((num_pts, 1)).astype(np_dtype) ** (1.0 / dim)
    points = np.asarray(manifold.c ** -0.5, dtype=np_dtype) * (random_dirs * random_radii)
    if isinstance(manifold, Hyperboloid):
        pytest.skip("Hyperboloid sampling path pending JAX conversion utilities")
    return jnp.asarray(points, dtype=dtype)


# ---------------------------------------------------------------------------
# Tests (ported from Torch suite)


def _split(points: jnp.ndarray, parts: int) -> tuple[jnp.ndarray, ...]:
    return tuple(jnp.array_split(points, parts, axis=0))


def test_addition(manifold: Manifold, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    if isinstance(manifold, Hyperboloid):
        pytest.skip("Hyperboloid addition undefined in Torch baseline; skip for parity")
    atol, rtol = tolerance
    x, y = _split(uniform_points, 2)
    identity = jnp.zeros_like(uniform_points)
    # Expectation: operations raise until Phase 2 implementations arrive.
    with pytest.raises(NotImplementedError):
        manifold.addition(identity, uniform_points)
    with pytest.raises(NotImplementedError):
        manifold.addition(uniform_points, identity)
    with pytest.raises(NotImplementedError):
        manifold.addition(-uniform_points, uniform_points)
    with pytest.raises(NotImplementedError):
        manifold.addition(x, y)


def test_scalar_mul(manifold: Manifold, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    atol, rtol = tolerance
    identity = jnp.ones((uniform_points.shape[0], 1), dtype=uniform_points.dtype)
    r1 = jnp.ones_like(identity) * 0.5
    r2 = jnp.ones_like(identity) * 0.25
    with pytest.raises(NotImplementedError):
        manifold.scalar_mul(identity, uniform_points)
    with pytest.raises(NotImplementedError):
        manifold.scalar_mul(r1 * r2, uniform_points)


def test_expmap_retraction_logmap(manifold: Manifold, uniform_points: jnp.ndarray) -> None:
    if isinstance(manifold, Hyperboloid):
        pytest.skip("Hyperboloid requires tangent projections not yet ported")
    v = jnp.ones_like(uniform_points)
    with pytest.raises(NotImplementedError):
        manifold.expmap(v, uniform_points)
    with pytest.raises(NotImplementedError):
        manifold.logmap(uniform_points, uniform_points)
