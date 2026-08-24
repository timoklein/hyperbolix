"""Tests for JAX helper utilities.

This module tests the helper functions for computing pairwise distances,
delta-hyperbolicity, and related geometric measures.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.manifolds import (
    Euclidean,
    Hyperboloid,
    Poincare,
    ProductManifold,
    ProperVelocity,
    Stereographic,
)
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


# ---------------------------------------------------------------------------------------------
# Manifold coverage of the two public helpers (audit C2)
#
# Both helpers forward ``version_idx`` positionally to ``manifold.dist``. Only Poincare,
# Hyperboloid and ProperVelocity used to accept it, so
# ``compute_pairwise_distances(pts, Euclidean(), 0.0)`` raised
# ``TypeError: dist() takes from 3 to 4 positional arguments but 5 were given`` — the same for
# Stereographic and ProductManifold. Three of the library's six manifolds could not be used with
# either of its two public geometry utilities.
# ---------------------------------------------------------------------------------------------


def _manifold_case(name: str, n_points: int = 12, seed: int = 3):
    """``(manifold, c, points, version_idx)`` for each of the six manifolds.

    Curvature is a per-factor tuple for ``ProductManifold`` and signed for ``Stereographic``,
    so the case table also covers the two non-scalar/non-positive curvature shapes the helpers
    have to pass through untouched.
    """
    gen = np.random.default_rng(seed)
    dim = 4

    if name == "euclidean":
        pts = jnp.asarray(gen.normal(0.0, 1.0, size=(n_points, dim)), dtype=jnp.float64)
        return Euclidean(dtype=jnp.float64), 0.0, pts, 0

    if name == "poincare":
        manifold = Poincare(dtype=jnp.float64)
        raw = jnp.asarray(0.3 * gen.normal(0.0, 1.0, size=(n_points, dim)), dtype=jnp.float64)
        return manifold, 1.0, jax.vmap(manifold.proj, in_axes=(0, None))(raw, 1.0), manifold.VERSION_MOBIUS_DIRECT

    if name == "hyperboloid":
        manifold = Hyperboloid(dtype=jnp.float64)
        raw = jnp.asarray(gen.normal(0.0, 1.0, size=(n_points, dim + 1)), dtype=jnp.float64)
        return manifold, 1.0, jax.vmap(manifold.proj, in_axes=(0, None))(raw, 1.0), manifold.VERSION_DEFAULT

    if name == "proper_velocity":
        pts = jnp.asarray(gen.normal(0.0, 1.0, size=(n_points, dim)), dtype=jnp.float64)
        return ProperVelocity(dtype=jnp.float64), 1.0, pts, 0

    if name == "stereographic_hyperbolic":
        manifold = Stereographic(dtype=jnp.float64)
        raw = jnp.asarray(0.3 * gen.normal(0.0, 1.0, size=(n_points, dim)), dtype=jnp.float64)
        return manifold, 1.0, jax.vmap(manifold.proj, in_axes=(0, None))(raw, 1.0), 0

    if name == "stereographic_spherical":
        manifold = Stereographic(dtype=jnp.float64)
        raw = jnp.asarray(0.3 * gen.normal(0.0, 1.0, size=(n_points, dim)), dtype=jnp.float64)
        return manifold, -1.0, jax.vmap(manifold.proj, in_axes=(0, None))(raw, -1.0), 0

    if name == "product":
        ball, sheet = Poincare(dtype=jnp.float64), Hyperboloid(dtype=jnp.float64)
        manifold = ProductManifold((ball, 2), (sheet, 3), dtype=jnp.float64)
        cs = (1.0, 0.5)
        ball_pts = jax.vmap(ball.proj, in_axes=(0, None))(
            jnp.asarray(0.3 * gen.normal(0.0, 1.0, size=(n_points, 2)), dtype=jnp.float64), cs[0]
        )
        sheet_pts = jax.vmap(sheet.proj, in_axes=(0, None))(
            jnp.asarray(gen.normal(0.0, 1.0, size=(n_points, 3)), dtype=jnp.float64), cs[1]
        )
        return manifold, cs, jnp.concatenate([ball_pts, sheet_pts], axis=-1), 0

    raise ValueError(f"unknown case {name!r}")


MANIFOLD_CASES = [
    "euclidean",
    "poincare",
    "hyperboloid",
    "proper_velocity",
    "stereographic_hyperbolic",
    "stereographic_spherical",
    "product",
]


@pytest.mark.parametrize("case", MANIFOLD_CASES)
def test_compute_pairwise_distances_works_on_every_manifold(case: str):
    """The distance matrix is the manifold's own ``dist`` on every pair, for all six manifolds.

    The value oracle (an off-diagonal entry against a direct ``manifold.dist`` call) is what
    makes this more than a smoke test: three of these cases used to raise ``TypeError`` outright,
    and an accept-and-ignore ``version_idx`` added carelessly could just as easily have silenced
    the error by returning something that is not the geodesic distance.
    """
    manifold, c, points, version_idx = _manifold_case(case)

    distmat = compute_pairwise_distances(points, manifold, c, version_idx)

    assert distmat.shape == (points.shape[0], points.shape[0])
    assert jnp.all(jnp.isfinite(distmat))
    assert jnp.allclose(distmat, distmat.T, atol=1e-10)
    assert jnp.allclose(jnp.diag(distmat), 0.0, atol=1e-7)
    for i, j in ((0, 1), (2, 7)):
        assert float(distmat[i, j]) == pytest.approx(float(manifold.dist(points[i], points[j], c)), rel=1e-9)


@pytest.mark.parametrize("case", MANIFOLD_CASES)
def test_get_delta_works_on_every_manifold(case: str):
    """``get_delta`` composes over every manifold and returns ``(delta, max(D), delta/max(D))``.

    Rebuilds the distance matrix through the sibling helper rather than asserting only
    finiteness, so the two helpers are pinned to the same manifold dispatch.
    """
    manifold, c, points, version_idx = _manifold_case(case)

    delta, diam, rel_delta = get_delta(points, manifold, c, version_idx)
    distmat = compute_pairwise_distances(points, manifold, c, version_idx)

    assert float(diam) > 0.0
    assert jnp.allclose(diam, jnp.max(distmat), rtol=1e-12)
    assert jnp.allclose(delta, compute_hyperbolic_delta(distmat, version="average"), rtol=1e-9, atol=1e-12)
    assert jnp.allclose(rel_delta, delta / diam, rtol=1e-9)


@pytest.mark.parametrize("case", ["euclidean", "stereographic_hyperbolic", "product"])
def test_version_idx_is_accepted_and_ignored_by_the_single_implementation_manifolds(case: str):
    """Manifolds with one distance implementation return the same matrix for any ``version_idx``.

    Pins the documented contract rather than just the absence of a ``TypeError``: these three
    accept the argument for protocol uniformity and must not let it select anything.
    """
    manifold, c, points, _ = _manifold_case(case)

    base = compute_pairwise_distances(points, manifold, c, 0)

    for version_idx in (1, 2, 7):
        assert jnp.allclose(compute_pairwise_distances(points, manifold, c, version_idx), base, rtol=0, atol=0)


@pytest.mark.parametrize("version_idx", [0, 1, 2, 3])
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
# The delta VALUE oracles live in the block at the bottom of this file (audit M1-03). They landed
# together with the library fix for audit A3-01: `compute_hyperbolic_delta` used to index both
# operands of its max-min product on row `i`, so `max_k min(G[i,k], G[i,j]) == G[i,j]`,
# `delta_matrix` was identically 0 for every input and `get_delta` always reported `rel_delta = 0`.
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


def test_get_delta_reports_nonzero_delta_on_real_hyperbolic_points(hyperboloid):
    """``get_delta`` must report a strictly positive delta for a generic point cloud.

    A hyperbolic sample of 30 points is not a tree, so its 4-point condition is violated
    somewhere and both delta statistics are > 0. Before the A3-01 fix every call returned
    exactly 0.0 and ``rel_delta`` with it; the ``> 0`` floor here is the file's end-to-end
    guard against that regression coming back through ``get_delta``.
    """
    points = _sample_points(hyperboloid, n_points=30, dim=5)
    kwargs = dict(c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    delta_avg, diam, rel_delta = get_delta(points, hyperboloid, version="average", **kwargs)
    delta_max, _, _ = get_delta(points, hyperboloid, version="smallest", **kwargs)

    assert float(delta_avg) > 0.0
    assert float(delta_max) >= float(delta_avg)
    assert 0.0 < float(rel_delta) < 1.0
    assert jnp.allclose(rel_delta, delta_avg / diam, rtol=1e-6)


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

    EVERY output of every call is forced (memory-bug regression guard): the un-subsampled
    2000-point call used to launch a 64 GB ``(n, n, n)`` allocation inside
    ``compute_hyperbolic_delta``, and because JAX dispatches asynchronously and this test
    discarded ``delta_full``, the failure never surfaced — the machine either OOM-killed the
    whole pytest process or the poisoned buffer was silently garbage-collected. With the
    O(n^2) scan implementation this call must complete and return finite values.
    """
    points = _sample_points(hyperboloid, n_points=2000, dim=5)
    kwargs = dict(c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

    delta_a, diam_a, rel_a = get_delta(points, hyperboloid, sample_size=100, key=jax.random.PRNGKey(0), **kwargs)
    delta_b, diam_b, rel_b = get_delta(points, hyperboloid, sample_size=100, key=jax.random.PRNGKey(1), **kwargs)
    delta_full, diam_full, rel_full = get_delta(points, hyperboloid, sample_size=2000, key=None, **kwargs)

    for v in (delta_a, diam_a, rel_a, delta_b, diam_b, rel_b, delta_full, diam_full, rel_full):
        assert jnp.isfinite(v)
    assert float(delta_full) > 0.0  # a generic hyperbolic sample is not a tree
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


# =============================================================================================
# Delta-hyperbolicity VALUE oracles (audit M1-03)
#
# `compute_hyperbolic_delta` fixes the base point p = 0 and returns
#
#     2 * stat_{i,j} [ max_k min( (i|k)_0 , (k|j)_0 )  -  (i|j)_0 ],
#     (i|j)_0 = ( d(0,i) + d(0,j) - d(i,j) ) / 2,
#
# with `stat` = mean for version="average" and max for version="smallest". Every oracle below is
# written against that definition, computed by hand (small graph metrics) or by an explicit Python
# loop (random metric) — never by calling the library back.
# =============================================================================================

# 4-cycle C4 with unit edges: the smallest non-tree metric. Antipodal pairs are at distance 2.
_C4_METRIC = jnp.array(
    [
        [0.0, 1.0, 2.0, 1.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [1.0, 2.0, 1.0, 0.0],
    ]
)

# 3-leaf star (centre 0, leaves at distance 1) and the 4-vertex path 0-1-2-3.
_STAR_METRIC = jnp.array(
    [
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 2.0, 2.0],
        [1.0, 2.0, 0.0, 2.0],
        [1.0, 2.0, 2.0, 0.0],
    ]
)
_PATH_METRIC = jnp.array([[float(abs(i - j)) for j in range(4)] for i in range(4)])


def _delta_reference(distmat: np.ndarray, version: str) -> float:
    """Independent transcription of the fixed-base-point delta, as an explicit Python loop.

    Each term ranges over the quadruple ``(0, i, j, k)`` — the base point plus three free indices —
    so this is the naive O(n^4)-work form of the vectorized max-min matrix product in the library.
    Written with plain loops on purpose: the A3-01 defect was in the broadcasting layout of that
    product, which no reimplementation using the same fancy indexing could have caught.
    """
    n = distmat.shape[0]
    gromov = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            gromov[i, j] = 0.5 * (distmat[i, 0] + distmat[0, j] - distmat[i, j])

    delta_terms = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            best = -np.inf
            for k in range(n):
                best = max(best, min(gromov[i, k], gromov[k, j]))
            delta_terms[i, j] = best - gromov[i, j]

    stat = float(delta_terms.mean()) if version == "average" else float(delta_terms.max())
    return 2.0 * stat


@pytest.mark.parametrize(
    ("name", "distmat"),
    [("star", _STAR_METRIC), ("path", _PATH_METRIC)],
)
@pytest.mark.parametrize("version", ["average", "smallest"])
def test_delta_is_exactly_zero_on_tree_metrics(name: str, distmat: jnp.ndarray, version: str):
    """Tree metrics are 0-hyperbolic: both statistics must be exactly 0.

    Non-discriminating on its own — the pre-fix implementation returned 0 for *every* input and
    passed this too. It is the over-correction guard for the C4 oracle below: a "fix" that
    inflates delta by, say, dropping the ``- (i|j)_0`` subtraction would break here, not there.
    """
    delta = compute_hyperbolic_delta(distmat, version=version)

    assert float(delta) == pytest.approx(0.0, abs=1e-12)
    assert float(delta) == pytest.approx(_delta_reference(np.asarray(distmat, dtype=np.float64), version), abs=1e-12)


@pytest.mark.parametrize(("version", "expected"), [("average", 0.25), ("smallest", 2.0)])
def test_delta_matches_closed_form_on_the_4_cycle(version: str, expected: float):
    """C4 pins the VALUE: 0.25 (average) / 2.0 (max).

    Hand-derived. With base point 0 the Gromov product matrix is
    ``[[0,0,0,0],[0,1,1,0],[0,1,2,1],[0,0,1,1]]``; the max-min product exceeds it only at the
    antipodal pairs (1,3) and (3,1), each by 1, so mean = 2/16 and max = 1 — doubled by the
    fixed-base-point rescaling to 0.25 and 2.0.

    This is the test the A3-01 defect fails: the buggy max-min product collapsed to ``(i|j)_0``
    and returned 0.0 for both statistics.
    """
    delta = compute_hyperbolic_delta(_C4_METRIC, version=version)

    assert float(delta) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("version", ["average", "smallest"])
def test_delta_is_positively_homogeneous_in_the_metric(version: str):
    """delta(s*D) == s*delta(D): delta carries the units of the metric (hence ``rel_delta``).

    Run on C4 rather than a tree, and paired with a strict ``> 0`` floor: on a 0-hyperbolic input
    the scaling identity degenerates to ``0 == 0`` and holds for any implementation, including the
    pre-fix one.
    """
    base = compute_hyperbolic_delta(_C4_METRIC, version=version)
    scaled = compute_hyperbolic_delta(2.0 * _C4_METRIC, version=version)

    assert float(base) > 0.0
    assert float(scaled) == pytest.approx(2.0 * float(base), rel=1e-12)


@pytest.mark.parametrize("version", ["average", "smallest"])
def test_delta_matches_bruteforce_loop_on_a_random_4_point_metric(version: str):
    """Random 4-point metric vs the explicit quadruple loop in ``_delta_reference``.

    The random matrix is built as a genuine metric (shortest-path closure of random symmetric
    weights) so the Gromov products are the ones a real distance matrix would produce, and it is
    checked to be non-tree-like (delta > 0) so the comparison is not the vacuous 0 == 0.
    """
    gen = np.random.default_rng(7)
    n = 4
    weights = gen.uniform(0.5, 2.0, size=(n, n))
    distmat = (weights + weights.T) / 2.0
    np.fill_diagonal(distmat, 0.0)
    # Floyd-Warshall closure -> triangle inequality holds, so this is a metric.
    for k in range(n):
        distmat = np.minimum(distmat, distmat[:, k : k + 1] + distmat[k : k + 1, :])

    delta = compute_hyperbolic_delta(jnp.asarray(distmat), version=version)

    assert float(delta) > 0.0, "degenerate draw: pick another seed, the comparison would be 0 == 0"
    assert float(delta) == pytest.approx(_delta_reference(distmat, version), abs=1e-12)
