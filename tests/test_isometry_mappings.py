"""Tests for isometry mappings between hyperbolic manifold models.

Tests the distance-preserving transformations between the hyperboloid model
and the Poincaré ball model. Verifies that conversions preserve geodesic
distances and manifold constraints.
"""

import jax
import jax.numpy as jnp
import pytest

import hyperbolix as hj

# Enable float64 support in JAX for numerical precision
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def curvature():
    """Curvature parameter for tests."""
    return 1.0


@pytest.fixture
def tolerance():
    """Tolerance for float64 comparisons."""
    return (1e-6, 1e-6)  # (atol, rtol)


@pytest.fixture
def hyperboloid_points(curvature: float):
    """Generate test points on the hyperboloid manifold."""
    c = curvature
    dim = 3  # 3D hyperboloid (4D ambient space)

    # Create random points in Poincaré ball, then convert to hyperboloid
    # to ensure we have valid hyperboloid points
    rng = jax.random.PRNGKey(42)
    n_points = 20

    # Sample from uniform distribution in ball
    samples = jax.random.normal(rng, (n_points, dim))
    # Scale to be within ball (with margin for safety)
    norms = jnp.linalg.norm(samples, axis=1, keepdims=True)
    max_norm = 0.95 / jnp.sqrt(c)  # Stay away from boundary
    samples = samples * (max_norm / jnp.maximum(norms, 1.0))

    # Convert to hyperboloid using the mapping we're testing
    # (this is circular but we'll verify correctness separately)
    to_hyperboloid_batch = jax.vmap(
        hj.manifolds.isometry_mappings.poincare_to_hyperboloid,
        in_axes=(0, None),
    )
    hyperboloid_pts = to_hyperboloid_batch(samples, c)

    # Verify points are on hyperboloid
    is_valid = jax.vmap(hj.manifolds.hyperboloid._is_in_manifold, in_axes=(0, None))
    assert jnp.all(is_valid(hyperboloid_pts, c)), "Generated hyperboloid points are invalid"

    return hyperboloid_pts


@pytest.fixture
def poincare_points(curvature: float):
    """Generate test points in the Poincaré ball."""
    c = curvature
    dim = 3  # 3D Poincaré ball

    rng = jax.random.PRNGKey(123)
    n_points = 20

    # Sample from uniform distribution in ball
    samples = jax.random.normal(rng, (n_points, dim))
    # Scale to be within ball (with margin for safety)
    norms = jnp.linalg.norm(samples, axis=1, keepdims=True)
    max_norm = 0.95 / jnp.sqrt(c)  # Stay away from boundary
    poincare_pts = samples * (max_norm / jnp.maximum(norms, 1.0))

    # Verify points are in Poincaré ball
    is_valid = jax.vmap(hj.manifolds.poincare._is_in_manifold, in_axes=(0, None))
    assert jnp.all(is_valid(poincare_pts, c)), "Generated Poincaré points are invalid"

    return poincare_pts


def test_hyperboloid_to_poincare_manifold_validity(
    hyperboloid_points: jnp.ndarray,
    curvature: float,
):
    """Test that hyperboloid_to_poincare produces valid Poincaré ball points."""
    c = curvature

    # Convert points
    to_poincare_batch = jax.vmap(
        hj.manifolds.isometry_mappings.hyperboloid_to_poincare,
        in_axes=(0, None),
    )
    poincare_pts = to_poincare_batch(hyperboloid_points, c)

    # Verify all converted points are in Poincaré ball
    is_valid = jax.vmap(hj.manifolds.poincare._is_in_manifold, in_axes=(0, None))
    assert jnp.all(is_valid(poincare_pts, c)), "Converted points are not in Poincaré ball"


def test_poincare_to_hyperboloid_manifold_validity(
    poincare_points: jnp.ndarray,
    curvature: float,
):
    """Test that poincare_to_hyperboloid produces valid hyperboloid points."""
    c = curvature

    # Convert points
    to_hyperboloid_batch = jax.vmap(
        hj.manifolds.isometry_mappings.poincare_to_hyperboloid,
        in_axes=(0, None),
    )
    hyperboloid_pts = to_hyperboloid_batch(poincare_points, c)

    # Verify all converted points are on hyperboloid
    is_valid = jax.vmap(hj.manifolds.hyperboloid._is_in_manifold, in_axes=(0, None))
    assert jnp.all(is_valid(hyperboloid_pts, c)), "Converted points are not on hyperboloid"


def test_origin_mapping(curvature: float, tolerance: tuple[float, float]):
    """Test that origins map correctly between models."""
    c = curvature
    atol, rtol = tolerance

    # Hyperboloid origin: [sqrt(1/c), 0, ..., 0]
    dim = 3
    hyperboloid_origin = jnp.zeros(dim + 1)
    hyperboloid_origin = hyperboloid_origin.at[0].set(jnp.sqrt(1.0 / c))

    # Poincaré origin: [0, ..., 0]
    poincare_origin = jnp.zeros(dim)

    # Test hyperboloid origin -> Poincaré origin
    poincare_result = hj.manifolds.isometry_mappings.hyperboloid_to_poincare(hyperboloid_origin, c)
    assert jnp.allclose(poincare_result, poincare_origin, atol=atol, rtol=rtol), (
        "Hyperboloid origin does not map to Poincaré origin"
    )

    # Test Poincaré origin -> hyperboloid origin
    hyperboloid_result = hj.manifolds.isometry_mappings.poincare_to_hyperboloid(poincare_origin, c)
    assert jnp.allclose(hyperboloid_result, hyperboloid_origin, atol=atol, rtol=rtol), (
        "Poincaré origin does not map to hyperboloid origin"
    )


def test_round_trip_hyperboloid_to_poincare_to_hyperboloid(
    hyperboloid_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Test that hyperboloid -> Poincaré -> hyperboloid is identity."""
    c = curvature
    atol, rtol = tolerance

    # Create batched conversion functions
    to_poincare_batch = jax.vmap(
        hj.manifolds.isometry_mappings.hyperboloid_to_poincare,
        in_axes=(0, None),
    )
    to_hyperboloid_batch = jax.vmap(
        hj.manifolds.isometry_mappings.poincare_to_hyperboloid,
        in_axes=(0, None),
    )

    # Round-trip conversion
    poincare_pts = to_poincare_batch(hyperboloid_points, c)
    reconstructed_pts = to_hyperboloid_batch(poincare_pts, c)

    # Verify round-trip is identity
    assert jnp.allclose(reconstructed_pts, hyperboloid_points, atol=atol, rtol=rtol), (
        "Round-trip hyperboloid -> Poincaré -> hyperboloid failed"
    )


def test_round_trip_poincare_to_hyperboloid_to_poincare(
    poincare_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Test that Poincaré -> hyperboloid -> Poincaré is identity."""
    c = curvature
    atol, rtol = tolerance

    # Create batched conversion functions
    to_hyperboloid_batch = jax.vmap(
        hj.manifolds.isometry_mappings.poincare_to_hyperboloid,
        in_axes=(0, None),
    )
    to_poincare_batch = jax.vmap(
        hj.manifolds.isometry_mappings.hyperboloid_to_poincare,
        in_axes=(0, None),
    )

    # Round-trip conversion
    hyperboloid_pts = to_hyperboloid_batch(poincare_points, c)
    reconstructed_pts = to_poincare_batch(hyperboloid_pts, c)

    # Verify round-trip is identity
    assert jnp.allclose(reconstructed_pts, poincare_points, atol=atol, rtol=rtol), (
        "Round-trip Poincaré -> hyperboloid -> Poincaré failed"
    )


def test_isometry_preserves_distances(
    hyperboloid_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Test that the mapping preserves geodesic distances (is an isometry).

    The most important property: d_hyperboloid(x, y) = d_poincare(φ(x), φ(y))
    where φ is the hyperboloid_to_poincare mapping.
    """
    c = curvature
    atol, rtol = tolerance

    # Split points for pairwise distance computation
    n_pairs = min(10, len(hyperboloid_points) // 2)
    x_hyp = hyperboloid_points[:n_pairs]
    y_hyp = hyperboloid_points[n_pairs : 2 * n_pairs]

    # Convert to Poincaré ball
    to_poincare_batch = jax.vmap(
        hj.manifolds.isometry_mappings.hyperboloid_to_poincare,
        in_axes=(0, None),
    )
    x_poinc = to_poincare_batch(x_hyp, c)
    y_poinc = to_poincare_batch(y_hyp, c)

    # Compute distances in hyperboloid model
    dist_hyp_fn = jax.vmap(
        lambda x, y: hj.manifolds.hyperboloid._dist(x, y, c, version_idx=hj.manifolds.hyperboloid.VERSION_DEFAULT)
    )
    distances_hyperboloid = dist_hyp_fn(x_hyp, y_hyp)

    # Compute distances in Poincaré model
    dist_poinc_fn = jax.vmap(
        lambda x, y: hj.manifolds.poincare._dist(x, y, c, version_idx=hj.manifolds.poincare.VERSION_MOBIUS_DIRECT)
    )
    distances_poincare = dist_poinc_fn(x_poinc, y_poinc)

    # Verify distances are preserved
    assert jnp.allclose(distances_hyperboloid, distances_poincare, atol=atol, rtol=rtol), (
        "Isometry does not preserve distances"
    )


def test_isometry_preserves_distance_from_origin(
    hyperboloid_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Test that distance from origin is preserved under the mapping."""
    c = curvature
    atol, rtol = tolerance

    # Convert to Poincaré ball
    to_poincare_batch = jax.vmap(
        hj.manifolds.isometry_mappings.hyperboloid_to_poincare,
        in_axes=(0, None),
    )
    poincare_pts = to_poincare_batch(hyperboloid_points, c)

    # Compute distance from origin in hyperboloid model
    dist_0_hyp = jax.vmap(
        lambda x: hj.manifolds.hyperboloid._dist_0(x, c, version_idx=hj.manifolds.hyperboloid.VERSION_DEFAULT)
    )
    distances_hyperboloid = dist_0_hyp(hyperboloid_points)

    # Compute distance from origin in Poincaré model
    dist_0_poinc = jax.vmap(
        lambda x: hj.manifolds.poincare._dist_0(x, c, version_idx=hj.manifolds.poincare.VERSION_MOBIUS_DIRECT)
    )
    distances_poincare = dist_0_poinc(poincare_pts)

    # Verify distances are preserved
    assert jnp.allclose(distances_hyperboloid, distances_poincare, atol=atol, rtol=rtol), (
        "Isometry does not preserve distance from origin"
    )


def test_jit_compatibility(hyperboloid_points: jnp.ndarray, curvature: float):
    """Test that conversion functions are JIT-compatible."""
    c = curvature

    # JIT compile the conversion functions
    to_poincare_jit = jax.jit(hj.manifolds.isometry_mappings.hyperboloid_to_poincare)
    to_hyperboloid_jit = jax.jit(hj.manifolds.isometry_mappings.poincare_to_hyperboloid)

    # Test single point
    x_hyp = hyperboloid_points[0]
    y_poinc = to_poincare_jit(x_hyp, c)
    x_reconstructed = to_hyperboloid_jit(y_poinc, c)

    assert jnp.allclose(x_reconstructed, x_hyp, atol=1e-6, rtol=1e-6)


def test_vmap_compatibility(hyperboloid_points: jnp.ndarray, curvature: float):
    """Test that conversion functions work correctly with vmap."""
    c = curvature

    # Create vmapped functions
    to_poincare_batch = jax.vmap(
        hj.manifolds.isometry_mappings.hyperboloid_to_poincare,
        in_axes=(0, None),
    )
    to_hyperboloid_batch = jax.vmap(
        hj.manifolds.isometry_mappings.poincare_to_hyperboloid,
        in_axes=(0, None),
    )

    # Apply batched conversions
    poincare_pts = to_poincare_batch(hyperboloid_points, c)
    reconstructed_pts = to_hyperboloid_batch(poincare_pts, c)

    # Verify shapes
    assert poincare_pts.shape == (len(hyperboloid_points), hyperboloid_points.shape[1] - 1)
    assert reconstructed_pts.shape == hyperboloid_points.shape

    # Verify round-trip
    assert jnp.allclose(reconstructed_pts, hyperboloid_points, atol=1e-6, rtol=1e-6)


def test_multiple_curvatures():
    """Test that conversions work correctly for different curvature values."""
    curvatures = [0.5, 1.0, 2.0, 5.0]
    dim = 2

    for c in curvatures:
        # Create hyperboloid origin
        hyperboloid_origin = jnp.zeros(dim + 1)
        hyperboloid_origin = hyperboloid_origin.at[0].set(jnp.sqrt(1.0 / c))

        # Create Poincaré origin
        poincare_origin = jnp.zeros(dim)

        # Test conversions
        poincare_result = hj.manifolds.isometry_mappings.hyperboloid_to_poincare(hyperboloid_origin, c)
        assert jnp.allclose(poincare_result, poincare_origin, atol=1e-6, rtol=1e-6)

        hyperboloid_result = hj.manifolds.isometry_mappings.poincare_to_hyperboloid(poincare_origin, c)
        assert jnp.allclose(hyperboloid_result, hyperboloid_origin, atol=1e-6, rtol=1e-6)


def test_dimension_consistency():
    """Test that conversions handle different dimensions correctly."""
    c = 1.0
    dimensions = [1, 2, 3, 5, 10]

    for dim in dimensions:
        # Create hyperboloid point (dim+1 dimensional)
        x_hyp = jnp.zeros(dim + 1)
        x_hyp = x_hyp.at[0].set(jnp.sqrt(1.0 / c))
        x_hyp = x_hyp.at[1].set(0.1)  # Small perturbation

        # Project to hyperboloid
        x_hyp = hj.manifolds.hyperboloid._proj(x_hyp, c)

        # Convert to Poincaré (should be dim-dimensional)
        y_poinc = hj.manifolds.isometry_mappings.hyperboloid_to_poincare(x_hyp, c)
        assert y_poinc.shape == (dim,), f"Poincaré point has wrong dimension for dim={dim}"

        # Convert back to hyperboloid (should be (dim+1)-dimensional)
        x_reconstructed = hj.manifolds.isometry_mappings.poincare_to_hyperboloid(y_poinc, c)
        assert x_reconstructed.shape == (dim + 1,), f"Hyperboloid point has wrong dimension for dim={dim}"

        # Verify round-trip
        assert jnp.allclose(x_reconstructed, x_hyp, atol=1e-6, rtol=1e-6)
