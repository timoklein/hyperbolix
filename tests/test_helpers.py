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


def _sample_points(manifold, n_points: int, dim: int, seed: int = 42):
    """``n_points`` on-manifold points; ``dim`` is the intrinsic dimension for both models."""
    key = jax.random.PRNGKey(seed)
    if isinstance(manifold, Hyperboloid):
        raw = jax.random.normal(key, (n_points, dim + 1))
    else:
        raw = jax.random.normal(key, (n_points, dim)) * 0.3
    return jax.vmap(manifold.proj, in_axes=(0, None))(raw, 1.0)


@pytest.mark.parametrize("manifold_name", ["hyperboloid", "poincare"])
def test_distmat_contract(hyperboloid, poincare, manifold_name: str):
    """``compute_pairwise_distances`` returns the (n, n) matrix of ``manifold.dist`` values.

    Merged from six former tests (``*_shape`` x n_points{5,10,20} x dim{5,10}, ``*_symmetry``,
    ``*_diagonal_zero``, ``*_positive_distances``). The dropped axes were pure vectorization width —
    the nested ``vmap`` has no ``n_points``- or ``dim``-dependent branch — and symmetry/diagonal/
    non-negativity are properties of ``dist`` itself, owned by ``tests/test_manifolds.py``; here
    they only confirm the double ``vmap`` did not transpose.

    The last assertion is the one real oracle: an off-diagonal entry against a direct
    ``manifold.dist`` call. Without it a 0.7x scale on every distance passed the whole file
    (audit A3-06, mutation M5).
    """
    manifold = hyperboloid if manifold_name == "hyperboloid" else poincare
    version_idx = manifold.VERSION_DEFAULT if manifold_name == "hyperboloid" else manifold.VERSION_MOBIUS_DIRECT
    n_points = 10
    points = _sample_points(manifold, n_points, dim=5)

    distmat = compute_pairwise_distances(points, manifold, c=1.0, version_idx=version_idx)

    assert distmat.shape == (n_points, n_points)
    assert jnp.allclose(distmat, distmat.T, rtol=1e-5)
    assert jnp.allclose(jnp.diag(distmat), 0.0, atol=1e-6)
    assert jnp.all(distmat >= 0.0)
    # Value oracle: entry (i, j) is the geodesic distance between points i and j.
    for i, j in ((0, 1), (3, 7)):
        assert jnp.allclose(distmat[i, j], manifold.dist(points[i], points[j], 1.0, version_idx), rtol=1e-6)


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


@pytest.mark.parametrize("version_idx", [0, 1, 2])
def test_poincare_versions(poincare, version_idx: int):
    """Test that different Poincaré distance versions work."""
    key = jax.random.PRNGKey(42)
    n_points, dim = 5, 5
    points = jax.random.normal(key, (n_points, dim)) * 0.3
    points = jax.vmap(poincare.proj, in_axes=(0, None))(points, 1.0)

    distmat = compute_pairwise_distances(points, poincare, c=1.0, version_idx=version_idx)

    assert distmat.shape == (n_points, n_points)
    assert jnp.all(distmat >= 0.0)


# ---------------------------------------------------------------------------------------------
# compute_hyperbolic_delta / get_delta
#
# KNOWN GAP: nothing below pins a delta VALUE. Audit A3-01 found why — the library's
# `compute_hyperbolic_delta` indexes both operands of its max-min product on row `i`, so
# `delta_matrix` is identically 0 for every input and `get_delta` always reports `rel_delta = 0`.
# A closed-form oracle (star / path / C4 metrics, plus a brute-force 4-point cross-check) cannot be
# written against the current implementation and lands together with the library fix. Until then
# these tests are structural on purpose; do not read them as evidence the delta is correct.
# ---------------------------------------------------------------------------------------------


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


def test_get_delta_composes_delta_and_diameter_of_the_distance_matrix(hyperboloid):
    """``get_delta`` returns ``(delta(D), max(D), delta(D)/max(D))`` for ``D`` the pairwise matrix.

    The previous body asserted ``rel_delta == delta / diam`` from the *returned* ``delta`` and
    ``diam`` — literally ``x == x``. Rebuilding ``D`` from the public helper pins the composition
    instead: which matrix each output is taken over, and that ``diam`` is the maximum rather than
    (say) the mean. It deliberately does not pin the delta VALUE — no test in this file does; see
    the note above ``test_output_shape``.
    """
    points = _sample_points(hyperboloid, n_points=50, dim=5)
    distmat = compute_pairwise_distances(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    delta, diam, rel_delta = get_delta(points, hyperboloid, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    assert jnp.allclose(diam, jnp.max(distmat), rtol=1e-6)
    assert jnp.allclose(delta, compute_hyperbolic_delta(distmat, version="average"), rtol=1e-6, atol=1e-9)
    assert jnp.allclose(rel_delta, delta / diam, rtol=1e-6)


def test_subsampling(hyperboloid):
    """Subsampling actually subsamples: the result depends on the key and is a sub-metric.

    The former body asserted only ``isfinite`` on the three outputs, which holds equally if
    ``indices = permutation(key, n)[:sample_size]`` were replaced by ``[:1]`` or dropped entirely.
    ``diam`` is a maximum over the sampled pairs, so it is the cheapest observable of *which*
    points were drawn: it must move with the key, and it can never exceed the full diameter.
    """
    points = _sample_points(hyperboloid, n_points=2000, dim=5)
    kwargs = dict(c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    _, diam_a, rel_a = get_delta(points, hyperboloid, sample_size=100, key=jax.random.PRNGKey(0), **kwargs)
    _, diam_b, _ = get_delta(points, hyperboloid, sample_size=100, key=jax.random.PRNGKey(1), **kwargs)
    _, diam_full, _ = get_delta(points, hyperboloid, sample_size=2000, key=None, **kwargs)

    assert jnp.isfinite(rel_a)
    assert not jnp.allclose(diam_a, diam_b), "diameter identical across keys — the draw is not random"
    assert float(diam_a) <= float(diam_full) + 1e-6, "a subsample cannot be wider than the full set"
    assert float(diam_b) <= float(diam_full) + 1e-6


def test_no_subsampling(hyperboloid):
    """With ``sample_size >= n_points`` the subsampling branch is skipped entirely.

    Pinned by equality with the un-subsampled call rather than by ``isfinite``: taking the branch
    anyway (with a stale or partial index set) would change the numbers, not their finiteness.
    """
    points = _sample_points(hyperboloid, n_points=50, dim=5)
    kwargs = dict(c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    gated = get_delta(points, hyperboloid, sample_size=1500, key=None, **kwargs)
    plain = get_delta(points, hyperboloid, **kwargs)

    assert all(jnp.isfinite(v) for v in gated)
    assert all(jnp.allclose(g, p, rtol=1e-9, atol=1e-12) for g, p in zip(gated, plain, strict=True))


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
    # The `rel_delta == delta / diam` line that used to close this test compared the returned
    # tuple against itself; the composition is pinned once, over an independently rebuilt distance
    # matrix, in test_get_delta_composes_delta_and_diameter_of_the_distance_matrix.
