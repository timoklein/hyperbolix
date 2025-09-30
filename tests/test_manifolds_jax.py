"""JAX manifold tests - pure functional API.

Tests for the hyperbolix_jax backend using pure functions.
Mirrors the PyTorch test suite structure but adapted for functional API.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import hyperbolix_jax as hj


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Shared RNG to keep the JAX fixtures reproducible."""
    return np.random.default_rng(seed=7727)


@pytest.fixture(scope="session", params=[jnp.float32, jnp.float64])
def dtype(request: pytest.FixtureRequest) -> jnp.dtype:
    """Test both float32 and float64 precision."""
    return jnp.dtype(request.param)


@pytest.fixture(scope="session")
def tolerance(dtype: jnp.dtype) -> tuple[float, float]:
    """Tolerance values for numerical comparisons."""
    if dtype == jnp.float32:
        return 4e-3, 4e-3
    return 1e-7, 1e-7


@pytest.fixture(
    scope="session",
    params=[
        ("euclidean", 0.0),
        ("poincare", 1.0),
        ("hyperboloid", 1.0),
    ],
    ids=["Euclidean", "PoincareBall", "Hyperboloid"]
)
def manifold_and_c(request: pytest.FixtureRequest, rng: np.random.Generator):
    """Fixture providing (manifold_module, curvature) tuples."""
    manifold_name, c_base = request.param

    if manifold_name == "euclidean":
        # Euclidean always has c=0
        return hj.manifolds.euclidean, 0.0
    elif manifold_name == "poincare":
        # Poincaré with random positive curvature
        c = float(rng.exponential(scale=2.0))
        return hj.manifolds.poincare, c
    elif manifold_name == "hyperboloid":
        # Hyperboloid with random positive curvature
        c = float(rng.exponential(scale=2.0))
        return hj.manifolds.hyperboloid, c
    else:
        raise ValueError(f"Unknown manifold: {manifold_name}")


@pytest.fixture(scope="session", params=[2, 5, 10, 15])
def uniform_points(
    manifold_and_c,
    dtype: jnp.dtype,
    request: pytest.FixtureRequest,
    rng: np.random.Generator
) -> jnp.ndarray:
    """Generate uniformly distributed points on the manifold."""
    manifold, c = manifold_and_c
    dim = request.param
    num_pts = 2_500 * 6
    np_dtype = np.dtype(dtype.name)

    if manifold == hj.manifolds.euclidean:
        # Euclidean: uniform in box [-100, 100]^d
        lower, upper = -100.0, 100.0
        data = rng.uniform(lower, upper, size=(num_pts, dim)).astype(np_dtype)
        return jnp.asarray(data)

    elif manifold == hj.manifolds.poincare:
        # Poincaré ball: uniform sampling using spherical coordinates
        random_dirs = rng.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
        random_dirs /= np.linalg.norm(random_dirs, axis=-1, keepdims=True)
        random_radii = rng.random((num_pts, 1)).astype(np_dtype) ** (1.0 / dim)
        # Scale to ball of radius 1/√c
        points = (random_dirs * random_radii) / np.sqrt(c)
        # Project to ensure they're strictly inside
        return jnp.asarray(manifold.proj(jnp.asarray(points, dtype=dtype), c=c))

    elif manifold == hj.manifolds.hyperboloid:
        # Hyperboloid: generate points on upper sheet
        # Points satisfy: -x₀² + ||x_rest||² = -1/c with x₀ > 0
        # Generate random spatial components
        x_rest = rng.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
        # Scale to have varying radii
        scale = rng.exponential(scale=0.5, size=(num_pts, 1)).astype(np_dtype)
        x_rest = x_rest * scale
        # Compute temporal component: x₀ = sqrt(1/c + ||x_rest||²)
        x_rest_sqnorm = np.sum(x_rest ** 2, axis=-1, keepdims=True)
        x0 = np.sqrt(1.0 / c + x_rest_sqnorm)
        # Concatenate [x₀, x_rest]
        points = np.concatenate([x0, x_rest], axis=-1).astype(np_dtype)
        return jnp.asarray(points, dtype=dtype)

    else:
        raise ValueError(f"Unknown manifold module")


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


def test_addition(
    manifold_and_c,
    tolerance: tuple[float, float],
    uniform_points: jnp.ndarray
) -> None:
    """Test addition/Möbius addition operation."""
    manifold, c = manifold_and_c

    # Skip if manifold doesn't support addition
    if not hasattr(manifold, 'addition'):
        pytest.skip(f"{manifold.__name__} doesn't implement addition")

    atol, rtol = tolerance
    x, y = _split(uniform_points, 2)

    # Create origin/identity element
    if manifold == hj.manifolds.euclidean:
        identity = jnp.zeros_like(uniform_points)
    elif manifold == hj.manifolds.hyperboloid:
        # Hyperboloid identity: [sqrt(1/c), 0, ..., 0]
        identity = jnp.zeros_like(uniform_points)
        identity = identity.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        identity = jnp.zeros_like(uniform_points)

    # Additive identity: 0 ⊕ x = x
    result1 = manifold.addition(identity, uniform_points, c=c)
    assert jnp.allclose(result1, uniform_points, atol=atol, rtol=rtol)

    # Additive identity: x ⊕ 0 = x
    result2 = manifold.addition(uniform_points, identity, c=c)
    assert jnp.allclose(result2, uniform_points, atol=atol, rtol=rtol)

    # Additive inverse: (-x) ⊕ x ≈ 0
    if manifold != hj.manifolds.hyperboloid:
        # For Euclidean and Poincaré, simple negation works
        result3 = manifold.addition(-uniform_points, uniform_points, c=c)
        assert jnp.allclose(result3, identity, atol=atol, rtol=rtol)
    else:
        # For hyperboloid, skip this test as negation is more complex
        pass

    # Results should stay on manifold
    result = manifold.addition(x, y, c=c)
    assert manifold.is_in_manifold(result, c=c, axis=-1)


def test_scalar_mul(
    manifold_and_c,
    tolerance: tuple[float, float],
    uniform_points: jnp.ndarray,
    rng: np.random.Generator
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

    # Results should stay on manifold
    assert manifold.is_in_manifold(result1, c=c, axis=-1)


def test_dist_properties(
    manifold_and_c,
    tolerance: tuple[float, float],
    uniform_points: jnp.ndarray
) -> None:
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


def test_dist_0(
    manifold_and_c,
    tolerance: tuple[float, float],
    uniform_points: jnp.ndarray
) -> None:
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


def test_expmap_logmap_inverse(
    manifold_and_c,
    tolerance: tuple[float, float],
    uniform_points: jnp.ndarray
) -> None:
    """Test that exp and log are inverse operations."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)

    # log_x(y) maps y to tangent space at x
    v = manifold.logmap(y, x, c=c)

    # exp_x(v) should map back to y
    y_reconstructed = manifold.expmap(v, x, c=c)

    assert jnp.allclose(y_reconstructed, y, atol=atol, rtol=rtol)


def test_expmap_0_logmap_0_inverse(
    manifold_and_c,
    tolerance: tuple[float, float],
    uniform_points: jnp.ndarray
) -> None:
    """Test that exp_0 and log_0 are inverse operations."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # log_0(x) maps x to tangent space at origin
    v = manifold.logmap_0(uniform_points, c=c)

    # exp_0(v) should map back to x
    x_reconstructed = manifold.expmap_0(v, c=c)

    assert jnp.allclose(x_reconstructed, uniform_points, atol=atol, rtol=rtol)


def test_ptransp_preserves_norm(
    manifold_and_c,
    tolerance: tuple[float, float],
    uniform_points: jnp.ndarray,
    rng: np.random.Generator
) -> None:
    """Test that parallel transport preserves tangent vector norms."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)

    # Create tangent vectors at x
    v = jnp.asarray(
        rng.normal(0.0, 0.1, size=x.shape),
        dtype=uniform_points.dtype
    )

    # Compute norm at x
    norm_at_x = manifold.tangent_norm(v, x, c=c, keepdim=False)

    # Parallel transport to y
    v_transported = manifold.ptransp(v, x, y, c=c)

    # Compute norm at y
    norm_at_y = manifold.tangent_norm(v_transported, y, c=c, keepdim=False)

    # Norms should be preserved (isometry)
    assert jnp.allclose(norm_at_x, norm_at_y, atol=atol, rtol=rtol)


def test_tangent_inner_positive_definite(
    manifold_and_c,
    uniform_points: jnp.ndarray,
    rng: np.random.Generator
) -> None:
    """Test that tangent inner product is positive definite."""
    manifold, c = manifold_and_c

    # Create non-zero tangent vectors
    v = jnp.asarray(
        rng.normal(0.0, 1.0, size=uniform_points.shape),
        dtype=uniform_points.dtype
    )

    # Inner product <v, v> should be positive
    inner = manifold.tangent_inner(v, v, uniform_points, c=c, keepdim=False)

    assert jnp.all(inner > 0)


def test_tangent_inner_symmetric(
    manifold_and_c,
    tolerance: tuple[float, float],
    uniform_points: jnp.ndarray,
    rng: np.random.Generator
) -> None:
    """Test that tangent inner product is symmetric."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Create two tangent vectors
    u = jnp.asarray(
        rng.normal(0.0, 1.0, size=uniform_points.shape),
        dtype=uniform_points.dtype
    )
    v = jnp.asarray(
        rng.normal(0.0, 1.0, size=uniform_points.shape),
        dtype=uniform_points.dtype
    )

    # <u, v> = <v, u>
    inner_uv = manifold.tangent_inner(u, v, uniform_points, c=c, keepdim=False)
    inner_vu = manifold.tangent_inner(v, u, uniform_points, c=c, keepdim=False)

    assert jnp.allclose(inner_uv, inner_vu, atol=atol, rtol=rtol)


def test_egrad2rgrad_on_manifold(
    manifold_and_c,
    uniform_points: jnp.ndarray,
    rng: np.random.Generator
) -> None:
    """Test that Riemannian gradient points lie in tangent space."""
    manifold, c = manifold_and_c

    # Create Euclidean gradients
    egrad = jnp.asarray(
        rng.normal(0.0, 1.0, size=uniform_points.shape),
        dtype=uniform_points.dtype
    )

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


def test_consistency_across_dtypes(
    manifold_and_c,
    rng: np.random.Generator
) -> None:
    """Test that operations give consistent results across dtypes."""
    manifold, c = manifold_and_c

    # Create test points in float32
    if manifold == hj.manifolds.euclidean:
        points_f32 = jnp.asarray(
            rng.uniform(-10, 10, size=(100, 5)),
            dtype=jnp.float32
        )
    elif manifold == hj.manifolds.poincare:
        raw = rng.normal(0, 1, size=(100, 5)).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
        raw *= rng.random((100, 1)).astype(np.float32) ** 0.2
        points_f32 = manifold.proj(jnp.asarray(raw / np.sqrt(c), dtype=jnp.float32), c=c)
    else:  # hyperboloid
        x_rest = rng.normal(0, 0.5, size=(100, 5)).astype(np.float32)
        x_rest_sqnorm = np.sum(x_rest ** 2, axis=-1, keepdims=True)
        x0 = np.sqrt(1.0 / c + x_rest_sqnorm)
        points_f32 = jnp.asarray(np.concatenate([x0, x_rest], axis=-1), dtype=jnp.float32)

    # Convert to float64
    points_f64 = points_f32.astype(jnp.float64)

    # Compute distances
    x_f32, y_f32 = _split(points_f32, 2)
    x_f64, y_f64 = _split(points_f64, 2)

    d_f32 = manifold.dist(x_f32, y_f32, c=c, keepdim=False)
    d_f64 = manifold.dist(x_f64, y_f64, c=c, keepdim=False)

    # Results should be close (accounting for precision differences)
    assert jnp.allclose(d_f32, d_f64.astype(jnp.float32), rtol=1e-5, atol=1e-5)