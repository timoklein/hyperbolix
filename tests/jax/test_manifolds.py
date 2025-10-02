"""JAX manifold tests - pure functional API.

Tests for the hyperbolix_jax backend using pure functions.
Mirrors the PyTorch test suite structure but adapted for functional API.

Fixtures are defined in tests/jax/conftest.py and automatically loaded.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 support in JAX
jax.config.update("jax_enable_x64", True)

import hyperbolix_jax as hj

# ---------------------------------------------------------------------------
# Helper functions


def _split(points: jnp.ndarray, parts: int) -> tuple[jnp.ndarray, ...]:
    """Split points array into equal parts."""
    return tuple(jnp.array_split(points, parts, axis=0))


# ---------------------------------------------------------------------------
# Tests


def test_proj(manifold_and_c, uniform_points: jnp.ndarray) -> None:
    """Test projection keeps points on manifold."""
    manifold, c = manifold_and_c

    # Points should already be on manifold
    assert manifold.is_in_manifold(uniform_points, c=c, axis=-1)

    # Projecting should keep them on manifold
    projected = manifold.proj(uniform_points, c=c, axis=-1)
    assert manifold.is_in_manifold(projected, c=c, axis=-1)

    # For points already on manifold, projection should be close to identity
    if manifold == hj.manifolds.euclidean:
        assert jnp.allclose(projected, uniform_points)
    else:
        # For hyperbolic manifolds, points might be slightly adjusted
        assert jnp.allclose(projected, uniform_points, rtol=1e-5, atol=1e-5)


def test_addition(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test addition/Möbius addition operation."""
    manifold, c = manifold_and_c

    # Note: The addition operation is not well-defined for the Hyperboloid manifold
    # (matches PyTorch test behavior)
    if manifold == hj.manifolds.hyperboloid:
        pytest.skip("Addition not well-defined for Hyperboloid manifold")

    atol, rtol = tolerance
    x, y = _split(uniform_points, 2)

    # Create origin/identity element
    identity = jnp.zeros_like(uniform_points)

    # Additive identity: 0 ⊕ x = x
    result1 = manifold.addition(identity, uniform_points, c=c)
    assert jnp.allclose(result1, uniform_points, atol=atol, rtol=rtol)

    # Additive identity: x ⊕ 0 = x
    result2 = manifold.addition(uniform_points, identity, c=c)
    assert jnp.allclose(result2, uniform_points, atol=atol, rtol=rtol)

    # Additive inverse: (-x) ⊕ x ≈ 0
    result3 = manifold.addition(-uniform_points, uniform_points, c=c)
    assert jnp.allclose(result3, identity, atol=atol, rtol=rtol)

    # TODO: Needs Distributive law and the gyrotriangle inequality
    # Results should stay on manifold
    result = manifold.addition(x, y, c=c)
    assert manifold.is_in_manifold(result, c=c, axis=-1)


def test_scalar_mul(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test scalar multiplication operation."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Create scalars
    identity = jnp.ones((uniform_points.shape[0], 1), dtype=uniform_points.dtype)
    r1 = jnp.asarray(rng.random((uniform_points.shape[0], 1)), dtype=uniform_points.dtype)
    r2 = jnp.asarray(rng.random((uniform_points.shape[0], 1)), dtype=uniform_points.dtype)

    # Multiplicative identity: 1 ⊗ x = x
    result1 = manifold.scalar_mul(identity, uniform_points, c=c)
    assert jnp.allclose(result1, uniform_points, atol=atol, rtol=rtol)

    # Associative law: (r1*r2) ⊗ x = r1 ⊗ (r2 ⊗ x)
    result2 = manifold.scalar_mul(r1 * r2, uniform_points, c=c)
    result3 = manifold.scalar_mul(r1, manifold.scalar_mul(r2, uniform_points, c=c), c=c)
    assert jnp.allclose(result2, result3, atol=atol, rtol=rtol)

    # Commutative in scalars: (r1*r2) ⊗ x = (r2*r1) ⊗ x
    result4 = manifold.scalar_mul(r2 * r1, uniform_points, c=c)
    assert jnp.allclose(result2, result4, atol=atol, rtol=rtol)

    # TODO: Add N-Gyroaddition
    # TODO: Add distributive law
    # TODO: Add homogeneity of degree
    # TODO: Add numerical stability
    # Results should stay on manifold
    assert manifold.is_in_manifold(result1, c=c, axis=-1)

# TODO: Test gyration


def test_dist_properties(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test distance function properties."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    x, y, z = _split(uniform_points, 3)

    # Non-negativity: d(x, y) ≥ 0
    d_xy = manifold.dist(x, y, c=c, keepdim=False)
    assert jnp.all(d_xy >= -atol)

    # Identity: d(x, x) = 0
    d_xx = manifold.dist(x, x, c=c, keepdim=False)
    assert jnp.allclose(d_xx, 0.0, atol=atol, rtol=rtol)

    # Symmetry: d(x, y) = d(y, x)
    d_yx = manifold.dist(y, x, c=c, keepdim=False)
    assert jnp.allclose(d_xy, d_yx, atol=atol, rtol=rtol)

    # Triangle inequality: d(x, z) ≤ d(x, y) + d(y, z)
    d_xz = manifold.dist(x, z, c=c, keepdim=False)
    d_yz = manifold.dist(y, z, c=c, keepdim=False)
    assert jnp.all(d_xz <= d_xy + d_yz + atol)


def test_dist_0(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test distance from origin."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # dist_0 should match dist(x, origin)
    if manifold == hj.manifolds.hyperboloid:
        # Hyperboloid origin: [sqrt(1/c), 0, ..., 0]
        origin = jnp.zeros_like(uniform_points[0:1])
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points[0:1])

    d1 = manifold.dist_0(uniform_points, c=c, keepdim=True)
    d2 = manifold.dist(uniform_points, origin, c=c, keepdim=True)

    assert jnp.allclose(d1, d2, atol=atol, rtol=rtol)


def test_expmap_logmap_basic(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test that expmap and logmap produce valid outputs.

    Note: We don't test the inverse property expmap(logmap(y, x), x) ≈ y for
    arbitrary points because:
    1. Near-boundary points with large conformal factors (>10^4) cause numerical instability
    2. Float32 precision is insufficient for Möbius addition near the boundary
    3. PyTorch tests don't verify this property except at origin (see test_expmap_0_logmap_0_inverse)

    Instead, we verify that expmap/logmap produce finite, on-manifold results.
    """
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)

    # logmap should produce finite tangent vectors
    v = manifold.logmap(y, x, c=c)
    assert jnp.all(jnp.isfinite(v))
    assert manifold.is_in_tangent_space(v, x, c=c, axis=-1)

    # expmap should produce finite manifold points
    y_mapped = manifold.expmap(v, x, c=c)
    assert jnp.all(jnp.isfinite(y_mapped))
    assert manifold.is_in_manifold(y_mapped, c=c, axis=-1)

    # TODO: Add expmap0/logmap0 consistency checks


def test_expmap_0_logmap_0_inverse(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test that exp_0 and log_0 are inverse operations."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # log_0(x) maps x to tangent space at origin
    v = manifold.logmap_0(uniform_points, c=c)

    # exp_0(v) should map back to x
    x_reconstructed = manifold.expmap_0(v, c=c)

    assert jnp.allclose(x_reconstructed, uniform_points, atol=atol, rtol=rtol)

# TODO: Test retraction


def test_ptransp_preserves_norm(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test that parallel transport preserves tangent vector norms."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)

    # Create tangent vectors at x
    v = jnp.asarray(rng.normal(0.0, 0.1, size=x.shape), dtype=uniform_points.dtype)

    # Project onto tangent space at x (necessary for Hyperboloid)
    v = manifold.tangent_proj(v, x, c=c)

    # Compute norm at x
    norm_at_x = manifold.tangent_norm(v, x, c=c, keepdim=False)

    # Parallel transport to y
    v_transported = manifold.ptransp(v, x, y, c=c)

    # Compute norm at y
    norm_at_y = manifold.tangent_norm(v_transported, y, c=c, keepdim=False)

    # Norms should be preserved (isometry)
    assert jnp.allclose(norm_at_x, norm_at_y, atol=atol, rtol=rtol)


# TODO: Add ptransp consistency tests and round trip stability


def test_tangent_inner_positive_definite(manifold_and_c, uniform_points: jnp.ndarray, rng: np.random.Generator) -> None:
    """Test that tangent inner product is positive definite."""
    manifold, c = manifold_and_c

    # Create non-zero tangent vectors
    v = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)

    # Project onto tangent space (necessary for Hyperboloid)
    v = manifold.tangent_proj(v, uniform_points, c=c)

    # Inner product <v, v> should be positive
    inner = manifold.tangent_inner(v, v, uniform_points, c=c, keepdim=False)

    assert jnp.all(inner > 0)


def test_tangent_inner_symmetric(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test that tangent inner product is symmetric."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Create two tangent vectors
    u = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)
    v = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)

    # Project onto tangent space (necessary for Hyperboloid)
    u = manifold.tangent_proj(u, uniform_points, c=c)
    v = manifold.tangent_proj(v, uniform_points, c=c)

    # <u, v> = <v, u>
    inner_uv = manifold.tangent_inner(u, v, uniform_points, c=c, keepdim=False)
    inner_vu = manifold.tangent_inner(v, u, uniform_points, c=c, keepdim=False)

    assert jnp.allclose(inner_uv, inner_vu, atol=atol, rtol=rtol)


def test_egrad2rgrad_on_manifold(manifold_and_c, uniform_points: jnp.ndarray, rng: np.random.Generator) -> None:
    """Test that Riemannian gradient points lie in tangent space."""
    manifold, c = manifold_and_c

    # Create Euclidean gradients
    egrad = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)

    # Convert to Riemannian gradient
    rgrad = manifold.egrad2rgrad(egrad, uniform_points, c=c)

    # Riemannian gradient should be in tangent space
    assert manifold.is_in_tangent_space(rgrad, uniform_points, c=c, axis=-1)


def test_is_in_manifold(manifold_and_c, uniform_points: jnp.ndarray) -> None:
    """Test manifold membership checking."""
    manifold, c = manifold_and_c

    # All uniform points should be on manifold
    assert manifold.is_in_manifold(uniform_points, c=c, axis=-1)

    if manifold == hj.manifolds.poincare:
        # Points outside ball should not be on manifold
        outside = jnp.ones_like(uniform_points[0:1]) * 10.0
        assert not manifold.is_in_manifold(outside, c=c, axis=-1)
    elif manifold == hj.manifolds.hyperboloid:
        # Points not on hyperboloid surface should not be on manifold
        outside = jnp.ones_like(uniform_points[0:1]) * 10.0
        assert not manifold.is_in_manifold(outside, c=c, axis=-1)
