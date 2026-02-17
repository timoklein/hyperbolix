"""Tests for JAX helper utilities.

This module tests the helper functions for computing pairwise distances,
delta-hyperbolicity, and related geometric measures.
"""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.utils.helpers import (
    compute_hyperbolic_delta,
    compute_pairwise_distances,
    get_delta,
)


@pytest.fixture
def hyperboloid():
    """Hyperboloid manifold instance for tests."""
    return Hyperboloid()


@pytest.fixture
def poincare():
    """Poincare manifold instance for tests."""
    return Poincare()


@pytest.mark.parametrize("n_points", [5, 10, 20])
@pytest.mark.parametrize("dim", [5, 10])
def test_hyperboloid_shape(hyperboloid, n_points: int, dim: int):
    """Test that output shape is correct for hyperboloid."""
    key = jax.random.PRNGKey(42)
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    assert distmat.shape == (n_points, n_points)


@pytest.mark.parametrize("n_points", [5, 10, 20])
@pytest.mark.parametrize("dim", [5, 10])
def test_poincare_shape(poincare, n_points: int, dim: int):
    """Test that output shape is correct for Poincaré ball."""
    key = jax.random.PRNGKey(42)
    points = jax.random.normal(key, (n_points, dim)) * 0.3
    points = jax.vmap(poincare.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, poincare, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)

    assert distmat.shape == (n_points, n_points)


def test_hyperboloid_symmetry(hyperboloid):
    """Test that distance matrix is symmetric."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 10, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    assert jnp.allclose(distmat, distmat.T, rtol=1e-5)


def test_hyperboloid_diagonal_zero(hyperboloid):
    """Test that diagonal elements are zero (distance to self is 0)."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 10, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    diagonal = jnp.diag(distmat)
    assert jnp.allclose(diagonal, 0.0, atol=1e-6)


def test_hyperboloid_positive_distances(hyperboloid):
    """Test that all distances are non-negative."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 10, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    assert jnp.all(distmat >= 0.0)


def test_poincare_symmetry(poincare):
    """Test that Poincaré distance matrix is symmetric."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 10, 5
    points = jax.random.normal(key, (n_points, dim)) * 0.3
    points = jax.vmap(poincare.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, poincare, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)

    assert jnp.allclose(distmat, distmat.T, rtol=1e-5)


def test_poincare_diagonal_zero(poincare):
    """Test that Poincaré diagonal elements are zero."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 10, 5
    points = jax.random.normal(key, (n_points, dim)) * 0.3
    points = jax.vmap(poincare.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, poincare, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)

    diagonal = jnp.diag(distmat)
    assert jnp.allclose(diagonal, 0.0, atol=1e-6)


@pytest.mark.parametrize("version_idx", [0, 1])
def test_hyperboloid_versions(hyperboloid, version_idx: int):
    """Test that different distance versions work."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 5, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, hyperboloid, c=1.0, version_idx=version_idx)

    assert distmat.shape == (n_points, n_points)
    assert jnp.all(distmat >= 0.0)


@pytest.mark.parametrize("version_idx", [0, 1, 2, 3])
def test_poincare_versions(poincare, version_idx: int):
    """Test that different Poincaré distance versions work.

    Note: VERSION_LORENTZIAN_PROXY (3) is a pseudo-distance that can be
    negative, so we don't check non-negativity for it.
    """
    key = jax.random.PRNGKey(42)
    n_points, dim = 5, 5
    points = jax.random.normal(key, (n_points, dim)) * 0.3
    points = jax.vmap(poincare.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, poincare, c=1.0, version_idx=version_idx)

    assert distmat.shape == (n_points, n_points)
    if version_idx != poincare.VERSION_LORENTZIAN_PROXY:
        assert jnp.all(distmat >= 0.0)


def test_output_shape():
    """Test that output is a scalar."""
    distmat = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])

    delta = compute_hyperbolic_delta(distmat, version="average")

    assert delta.shape == ()
    assert delta.ndim == 0


def test_symmetric_matrix():
    """Test with symmetric distance matrix."""
    distmat = jnp.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0],
        ]
    )

    delta_avg = compute_hyperbolic_delta(distmat, version="average")
    delta_max = compute_hyperbolic_delta(distmat, version="smallest")

    assert delta_avg >= 0.0
    assert delta_max >= 0.0
    assert delta_max >= delta_avg


def test_zero_delta_for_tree_metric():
    """Test that tree metrics have zero delta (or very small)."""
    distmat = jnp.array([[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 2.0, 2.0], [1.0, 2.0, 0.0, 2.0], [1.0, 2.0, 2.0, 0.0]])

    delta = compute_hyperbolic_delta(distmat, version="average")

    assert jnp.allclose(delta, 0.0, atol=1e-6)


@pytest.mark.parametrize("version", ["average", "smallest"])
def test_both_versions(version: str):
    """Test that both versions work."""
    key = jax.random.PRNGKey(42)
    n_points = 10
    A = jax.random.uniform(key, (n_points, n_points))
    distmat = (A + A.T) / 2
    distmat = distmat.at[jnp.diag_indices(n_points)].set(0.0)

    delta = compute_hyperbolic_delta(distmat, version=version)

    assert delta.shape == ()
    assert delta >= 0.0


def test_real_hyperbolic_points(hyperboloid):
    """Test with actual hyperbolic points."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 20, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)
    delta = compute_hyperbolic_delta(distmat, version="average")

    assert delta >= 0.0
    assert jnp.isfinite(delta)


def test_output_tuple(hyperboloid):
    """Test that output is a tuple of three scalars."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 50, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    delta, diam, rel_delta = get_delta(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    assert isinstance(delta, jax.Array)
    assert isinstance(diam, jax.Array)
    assert isinstance(rel_delta, jax.Array)
    assert delta.shape == ()
    assert diam.shape == ()
    assert rel_delta.shape == ()


def test_positive_values(hyperboloid):
    """Test that delta, diameter, and relative delta are positive."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 50, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    delta, diam, rel_delta = get_delta(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    assert delta >= 0.0
    assert diam > 0.0
    assert rel_delta >= 0.0


def test_relative_delta_computation(hyperboloid):
    """Test that relative delta = delta / diameter."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 50, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    delta, diam, rel_delta = get_delta(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    expected_rel_delta = delta / diam
    assert jnp.allclose(rel_delta, expected_rel_delta, rtol=1e-6)


def test_subsampling(hyperboloid):
    """Test that subsampling works correctly."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 2000, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    key, subkey = jax.random.split(key)
    delta, diam, rel_delta = get_delta(
        points,
        hyperboloid,
        c=1.0,
        version_idx=hyperboloid.VERSION_DEFAULT,
        sample_size=100,
        key=subkey,
    )

    assert jnp.isfinite(delta)
    assert jnp.isfinite(diam)
    assert jnp.isfinite(rel_delta)


def test_no_subsampling(hyperboloid):
    """Test that no subsampling occurs when n_points < sample_size."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 50, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    delta, diam, rel_delta = get_delta(
        points,
        hyperboloid,
        c=1.0,
        version_idx=hyperboloid.VERSION_DEFAULT,
        sample_size=1500,
        key=None,
    )

    assert jnp.isfinite(delta)
    assert jnp.isfinite(diam)
    assert jnp.isfinite(rel_delta)


def test_requires_key_for_subsampling(hyperboloid):
    """Test that key is required when subsampling."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 2000, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    with pytest.raises(ValueError, match="Random key required for subsampling"):
        get_delta(
            points,
            hyperboloid,
            c=1.0,
            version_idx=hyperboloid.VERSION_DEFAULT,
            sample_size=100,
            key=None,
        )


@pytest.mark.parametrize("version", ["average", "smallest"])
def test_delta_versions(hyperboloid, version: str):
    """Test both delta computation versions."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 50, 5
    points = jax.random.normal(key, (n_points, dim + 1))
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(points, 1.0)

    delta, diam, rel_delta = get_delta(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT, version=version)

    assert jnp.isfinite(delta)
    assert jnp.isfinite(diam)
    assert jnp.isfinite(rel_delta)


def test_poincare_ball(poincare):
    """Test get_delta with Poincaré ball manifold."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 50, 5
    points = jax.random.normal(key, (n_points, dim)) * 0.3
    points = jax.vmap(poincare.proj, in_axes=(0, None))(points, 1.0)

    delta, diam, rel_delta = get_delta(points, poincare, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)

    assert delta >= 0.0
    assert diam > 0.0
    assert rel_delta >= 0.0
    assert jnp.allclose(rel_delta, delta / diam, rtol=1e-6)
