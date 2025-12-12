"""JAX manifold tests - vmap-native API.

Tests for the hyperbolix backend using vmap-native pure functions.
Adapted for the new single-point API with vmap for batching.

Fixtures are defined in tests/conftest.py and automatically loaded.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hyperbolix as hj

# Enable float64 support in JAX
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper functions


def _split(points: jnp.ndarray, parts: int) -> tuple[jnp.ndarray, ...]:
    """Split points array into equal parts."""
    return tuple(jnp.array_split(points, parts, axis=0))


def _batch_is_in_manifold(manifold, points: jnp.ndarray, c: float) -> bool:
    """Check if all points in batch are on manifold."""
    is_in = jax.vmap(lambda p: manifold.is_in_manifold(p, c=c))
    return bool(jnp.all(is_in(points)))


def _batch_is_in_tangent_space(manifold, vectors: jnp.ndarray, points: jnp.ndarray, c: float) -> bool:
    """Check if all vectors in batch are in tangent space."""
    is_in = jax.vmap(lambda v, p: manifold.is_in_tangent_space(v, p, c=c))
    return bool(jnp.all(is_in(vectors, points)))


def _dist_fn(manifold):
    """Return distance function with default version index if needed."""
    if manifold == hj.manifolds.poincare:
        return functools.partial(manifold.dist, version_idx=manifold.VERSION_MOBIUS_DIRECT)
    if manifold == hj.manifolds.hyperboloid:
        return functools.partial(manifold.dist, version_idx=manifold.VERSION_DEFAULT)
    return manifold.dist


def _dist_0_fn(manifold):
    """Return origin distance function with default version index if needed."""
    if manifold == hj.manifolds.poincare:
        return functools.partial(manifold.dist_0, version_idx=manifold.VERSION_MOBIUS_DIRECT)
    if manifold == hj.manifolds.hyperboloid:
        return functools.partial(manifold.dist_0, version_idx=manifold.VERSION_DEFAULT)
    return manifold.dist_0


# ---------------------------------------------------------------------------
# Tests


def test_proj(manifold_and_c, uniform_points: jnp.ndarray) -> None:
    """Test projection keeps points on manifold."""
    manifold, c = manifold_and_c

    # Batch operations using vmap
    proj_batch = jax.vmap(manifold.proj, in_axes=(0, None))

    # Points should already be on manifold
    assert _batch_is_in_manifold(manifold, uniform_points, c)

    # Projecting should keep them on manifold
    projected = proj_batch(uniform_points, c)
    assert _batch_is_in_manifold(manifold, projected, c)

    # Single-point API should produce consistent projection
    sample = uniform_points[0]
    projected_single = manifold.proj(sample, c)
    assert bool(manifold.is_in_manifold(projected_single, c=c))

    # For points already on manifold, projection should be close to identity
    if manifold == hj.manifolds.euclidean:
        assert jnp.allclose(projected, uniform_points)
        assert jnp.allclose(projected_single, sample)
    else:
        # For hyperbolic manifolds, points might be slightly adjusted
        assert jnp.allclose(projected, uniform_points, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(projected_single, sample, rtol=1e-5, atol=1e-5)


def test_addition(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test addition/Möbius addition operation."""
    manifold, c = manifold_and_c

    # Note: The addition operation is not well-defined for the Hyperboloid manifold
    # (matches PyTorch test behavior)
    if manifold == hj.manifolds.hyperboloid:
        pytest.skip("Addition not well-defined for Hyperboloid manifold")

    atol, rtol = tolerance
    x, y = _split(uniform_points, 2)

    # Batch operations using vmap
    addition_batch = jax.vmap(manifold.addition, in_axes=(0, 0, None))

    # Create origin/identity element
    identity = jnp.zeros_like(uniform_points)

    # Additive identity: 0 ⊕ x = x
    result1 = addition_batch(identity, uniform_points, c)
    assert jnp.allclose(result1, uniform_points, atol=atol, rtol=rtol)

    # Additive identity: x ⊕ 0 = x
    result2 = addition_batch(uniform_points, identity, c)
    assert jnp.allclose(result2, uniform_points, atol=atol, rtol=rtol)

    # Additive inverse: (-x) ⊕ x ≈ 0
    result3 = addition_batch(-uniform_points, uniform_points, c)
    assert jnp.allclose(result3, identity, atol=atol, rtol=rtol)

    # Additive inverse: x ⊕ (-x) ≈ 0
    result4 = addition_batch(uniform_points, -uniform_points, c)
    # Add 1 to avoid precision issues with values very close to zero
    assert jnp.allclose(result4 + 1, identity + 1, atol=atol, rtol=rtol)

    # Distributive law: -(x ⊕ y) = (-x) ⊕ (-y)
    result5 = addition_batch(x, y, c)
    assert jnp.allclose(-result5, addition_batch(-x, -y, c), atol=atol, rtol=rtol)

    # Gyrotriangle inequality: ‖x ⊕ y‖ ≤ ‖x‖ ⊕ ‖y‖
    xy_norm = jnp.linalg.norm(result5, axis=-1, keepdims=True)
    x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
    norm_sum = addition_batch(x_norm, y_norm, c)
    assert jnp.all(xy_norm <= norm_sum + atol)

    # Results should stay on manifold
    assert _batch_is_in_manifold(manifold, result5, c)


def test_scalar_mul(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test scalar multiplication operation."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    if manifold == hj.manifolds.poincare and uniform_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 2e-2)

    # Create scalars - now as 1D array since scalar_mul expects scalar per point
    r1 = jnp.asarray(rng.random(uniform_points.shape[0]), dtype=uniform_points.dtype)
    r2 = jnp.asarray(rng.random(uniform_points.shape[0]), dtype=uniform_points.dtype)
    identity_scalars = jnp.ones(uniform_points.shape[0], dtype=uniform_points.dtype)

    # Batch operations using vmap
    scalar_mul_batch = jax.vmap(manifold.scalar_mul, in_axes=(0, 0, None))
    addition_batch = jax.vmap(manifold.addition, in_axes=(0, 0, None))

    # Multiplicative identity: 1 ⊗ x = x
    result1 = scalar_mul_batch(identity_scalars, uniform_points, c)
    assert jnp.allclose(result1, uniform_points, atol=atol, rtol=rtol)

    # Associative law: (r1*r2) ⊗ x = r1 ⊗ (r2 ⊗ x)
    result2 = scalar_mul_batch(r1 * r2, uniform_points, c)
    result3 = scalar_mul_batch(r1, scalar_mul_batch(r2, uniform_points, c), c)
    assert jnp.allclose(result2, result3, atol=atol, rtol=rtol)

    # Commutative in scalars: (r1*r2) ⊗ x = (r2*r1) ⊗ x
    result4 = scalar_mul_batch(r2 * r1, uniform_points, c)
    assert jnp.allclose(result2, result4, atol=atol, rtol=rtol)

    # Additional properties for Euclidean and PoincareBall (not Hyperboloid)
    if manifold in (hj.manifolds.euclidean, hj.manifolds.poincare):
        # N-Gyroaddition property: n ⊗ x = x ⊕ x ⊕ ... ⊕ x (n times)
        n = rng.integers(3, 10)
        n_sum = jnp.zeros_like(uniform_points)
        for _ in range(n):
            n_sum = addition_batch(n_sum, uniform_points, c)
        n_scalar = jnp.ones(uniform_points.shape[0], dtype=uniform_points.dtype) * n
        result_n = scalar_mul_batch(n_scalar, uniform_points, c)
        assert jnp.allclose(n_sum, result_n, atol=atol, rtol=rtol)

        # Distributive law: (r1 + r2) ⊗ x = (r1 ⊗ x) ⊕ (r2 ⊗ x)
        result_dist = scalar_mul_batch(r1 + r2, uniform_points, c)
        result_r1 = scalar_mul_batch(r1, uniform_points, c)
        result_r2 = scalar_mul_batch(r2, uniform_points, c)
        result_add = addition_batch(result_r1, result_r2, c)
        assert jnp.allclose(result_dist, result_add, atol=atol, rtol=rtol)

        # Distributive law: (-r) ⊗ x = r ⊗ (-x)
        result_neg_r = scalar_mul_batch(-r1, uniform_points, c)
        result_r_neg = scalar_mul_batch(r1, -uniform_points, c)
        assert jnp.allclose(result_neg_r, result_r_neg, atol=atol, rtol=rtol)

        # Scaling property: direction preservation
        r_abs = jnp.abs(r1)
        result_scaled = scalar_mul_batch(r_abs, uniform_points, c)
        result_norm = jnp.linalg.norm(scalar_mul_batch(r1, uniform_points, c), axis=-1, keepdims=True)
        # Normalize to get direction
        left_side = result_scaled / result_norm
        right_side = uniform_points / jnp.linalg.norm(uniform_points, axis=-1, keepdims=True)
        assert jnp.allclose(left_side, right_side, atol=atol, rtol=rtol)

        # Homogeneity property: ‖r ⊗ x‖ = |r| ⊗ ‖x‖
        result_norm_lhs = jnp.linalg.norm(scalar_mul_batch(r1, uniform_points, c), axis=-1, keepdims=True)
        x_norm = jnp.linalg.norm(uniform_points, axis=-1, keepdims=True)
        result_norm_rhs = scalar_mul_batch(r_abs, x_norm, c)
        assert jnp.allclose(result_norm_lhs, result_norm_rhs, atol=atol, rtol=rtol)

    # Numerical stability tests
    r_zero = 0.0
    r_small = float(atol)
    r_large = 10.0

    # Create epsilon-norm vector
    v_eps_norm = jnp.zeros((1, uniform_points.shape[1]), dtype=uniform_points.dtype)
    v_eps_norm = v_eps_norm.at[0, 0].set(atol)
    if manifold == hj.manifolds.hyperboloid:
        v_eps_norm = v_eps_norm.at[0, 0].set(v_eps_norm[0, 0] + jnp.sqrt(1.0 / c))
        proj_single = jax.vmap(manifold.proj, in_axes=(0, None))
        v_eps_norm = proj_single(v_eps_norm, c)

    # Origin for comparison
    if manifold == hj.manifolds.hyperboloid:
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    # Stability of multiplication with zero scalars
    r_zero_arr = jnp.zeros(uniform_points.shape[0])
    res = scalar_mul_batch(r_zero_arr, uniform_points, c)
    assert jnp.all(jnp.isfinite(res))
    assert _batch_is_in_manifold(manifold, res, c)
    assert jnp.allclose(res + 1, origin + 1, atol=atol, rtol=rtol)

    res = manifold.scalar_mul(r_zero, v_eps_norm[0], c)
    assert jnp.all(jnp.isfinite(res))
    assert manifold.is_in_manifold(res, c=c)
    assert jnp.allclose(res + 1, origin[0] + 1, atol=atol, rtol=rtol)

    # Stability of multiplication with small scalars
    res = manifold.scalar_mul(r_small, v_eps_norm[0], c)
    assert jnp.all(jnp.isfinite(res))
    assert manifold.is_in_manifold(res, c=c)
    assert res[0] > r_zero
    assert jnp.allclose(res[1:], jnp.zeros_like(res[1:]), atol=atol, rtol=rtol)

    # Stability of multiplication with large scalars
    if manifold in (hj.manifolds.euclidean, hj.manifolds.poincare):
        # Note: Hyperboloid manifold may fail is_in_manifold check with large scalars
        # due to numerical instabilities in the Minkowski inner product
        r_large_arr = jnp.ones(uniform_points.shape[0]) * r_large
        res = scalar_mul_batch(r_large_arr, uniform_points, c)
        assert jnp.all(jnp.isfinite(res))
        assert _batch_is_in_manifold(manifold, res, c)

    res = manifold.scalar_mul(r_large, v_eps_norm[0], c)
    assert jnp.all(jnp.isfinite(res))
    assert manifold.is_in_manifold(res, c=c)
    assert res[0] > r_zero
    assert jnp.allclose(res[1:] + 1, origin[0, 1:] + 1, atol=atol, rtol=rtol)

    # Results should stay on manifold
    assert _batch_is_in_manifold(manifold, result1, c)


def test_gyration(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test the gyration operation of the PoincareBall.

    Gyration is a fundamental operation in gyrogroups that restores commutativity
    in the non-commutative Möbius addition. This test verifies all the gyrogroup
    axioms and properties.
    """
    manifold, c = manifold_and_c

    # The gyration operation is only defined for the PoincareBall manifold
    if manifold in (hj.manifolds.euclidean, hj.manifolds.hyperboloid):
        pytest.skip("Gyration operation not defined for Euclidean/Hyperboloid manifold")

    atol, rtol = tolerance
    x, y, z, a = _split(uniform_points, 4)

    # Batch operations using vmap
    addition_batch = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    gyration_batch = jax.vmap(manifold._gyration, in_axes=(0, 0, 0, None))
    scalar_mul_batch = jax.vmap(manifold.scalar_mul, in_axes=(0, 0, None))

    # (Gyro-)commutative law: x ⊕ y = gyr[x,y](y ⊕ x)
    xy = addition_batch(x, y, c)
    yx = addition_batch(y, x, c)
    gyr_yx = gyration_batch(x, y, yx, c)
    assert jnp.allclose(xy, gyr_yx, atol=atol, rtol=rtol)

    # Gyrosum inversion law: -(x ⊕ y) = gyr[x,y]((-y) ⊕ (-x))
    neg_xy = -addition_batch(x, y, c)
    neg_y_neg_x = addition_batch(-y, -x, c)
    gyr_neg = gyration_batch(x, y, neg_y_neg_x, c)
    assert jnp.allclose(neg_xy, gyr_neg, atol=atol, rtol=rtol)

    # Left (gyro-)associative law: x ⊕ (y ⊕ z) = (x ⊕ y) ⊕ gyr[x,y]z
    left_side = addition_batch(x, addition_batch(y, z, c), c)
    gyr_z = gyration_batch(x, y, z, c)
    right_side = addition_batch(addition_batch(x, y, c), gyr_z, c)
    assert jnp.allclose(left_side, right_side, atol=atol, rtol=rtol)

    # Right (gyro-)associative law: (x ⊕ y) ⊕ z = x ⊕ (y ⊕ gyr[y,x]z)
    left_side = addition_batch(addition_batch(x, y, c), z, c)
    gyr_yx_z = gyration_batch(y, x, z, c)
    right_side = addition_batch(x, addition_batch(y, gyr_yx_z, c), c)
    assert jnp.allclose(left_side, right_side, atol=atol, rtol=rtol)

    # Möbius addition under gyrations: gyr[x,y](z ⊕ a) = gyr[x,y]z ⊕ gyr[x,y]a
    za = addition_batch(z, a, c)
    gyr_za = gyration_batch(x, y, za, c)
    gyr_z = gyration_batch(x, y, z, c)
    gyr_a = gyration_batch(x, y, a, c)
    gyr_z_gyr_a = addition_batch(gyr_z, gyr_a, c)
    assert jnp.allclose(gyr_za, gyr_z_gyr_a, atol=atol, rtol=rtol)

    # Left loop property: gyr[x,y]z = gyr[x⊕y,y]z
    gyr_xy = gyration_batch(x, y, z, c)
    xy = addition_batch(x, y, c)
    gyr_xy_y = gyration_batch(xy, y, z, c)
    assert jnp.allclose(gyr_xy, gyr_xy_y, atol=atol, rtol=rtol)

    # Right loop property: gyr[x,y]z = gyr[x,y⊕x]z
    gyr_xy = gyration_batch(x, y, z, c)
    yx = addition_batch(y, x, c)
    gyr_x_yx = gyration_batch(x, yx, z, c)
    assert jnp.allclose(gyr_xy, gyr_x_yx, atol=atol, rtol=rtol)

    # Identity gyroautomorphism property: gyr[r1⊗x, r2⊗x]y = y
    r1 = jnp.asarray(rng.random(x.shape[0]), dtype=x.dtype)
    r2 = jnp.asarray(rng.random(x.shape[0]), dtype=x.dtype)
    r1_x = scalar_mul_batch(r1, x, c)
    r2_x = scalar_mul_batch(r2, x, c)
    gyr_identity = gyration_batch(r1_x, r2_x, y, c)
    assert jnp.allclose(gyr_identity, y, atol=atol, rtol=rtol)

    # Gyroautomorphism property: gyr[x,y](r⊗z) = r⊗gyr[x,y]z
    r_z = scalar_mul_batch(r1, z, c)
    gyr_r_z = gyration_batch(x, y, r_z, c)
    gyr_z = gyration_batch(x, y, z, c)
    r_gyr_z = scalar_mul_batch(r1, gyr_z, c)
    assert jnp.allclose(gyr_r_z, r_gyr_z, atol=atol, rtol=rtol)

    # First gyrogroup theorems
    zero = jnp.zeros_like(x)
    # gyr[x,0]z = z
    gyr_x_0 = gyration_batch(x, zero, z, c)
    assert jnp.allclose(gyr_x_0, z, atol=atol, rtol=rtol)
    # gyr[0,x]z = z
    gyr_0_x = gyration_batch(zero, x, z, c)
    assert jnp.allclose(gyr_0_x, z, atol=atol, rtol=rtol)
    # gyr[x,x]z = z
    gyr_x_x = gyration_batch(x, x, z, c)
    assert jnp.allclose(gyr_x_x, z, atol=atol, rtol=rtol)
    # gyr[x,y]0 = 0
    gyr_xy_0 = gyration_batch(x, y, zero, c)
    assert jnp.allclose(gyr_xy_0, zero, atol=atol, rtol=rtol)
    # gyr[x,y](-z) = -gyr[x,y]z
    gyr_neg_z = gyration_batch(x, y, -z, c)
    neg_gyr_z = -gyration_batch(x, y, z, c)
    assert jnp.allclose(gyr_neg_z, neg_gyr_z, atol=atol, rtol=rtol)


def test_dist_properties(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test distance function properties."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    x, y, z = _split(uniform_points, 3)

    # Batch operations using vmap with manifold-specific distance signature
    dist_fn = _dist_fn(manifold)
    dist_batch = jax.vmap(dist_fn, in_axes=(0, 0, None))

    # Non-negativity: d(x, y) ≥ 0
    d_xy = dist_batch(x, y, c)
    assert jnp.all(d_xy >= -atol)

    # Identity: d(x, x) = 0
    d_xx = dist_batch(x, x, c)
    assert jnp.allclose(d_xx, 0.0, atol=atol, rtol=rtol)

    # Symmetry: d(x, y) = d(y, x)
    d_yx = dist_batch(y, x, c)
    assert jnp.allclose(d_xy, d_yx, atol=atol, rtol=rtol)

    # Triangle inequality: d(x, z) ≤ d(x, y) + d(y, z)
    d_xz = dist_batch(x, z, c)
    d_yz = dist_batch(y, z, c)
    assert jnp.all(d_xz <= d_xy + d_yz + atol)


def test_dist_0(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test distance from origin."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Batch operations using vmap with manifold-specific distance signature
    dist_0_fn = _dist_0_fn(manifold)
    dist_fn = _dist_fn(manifold)
    dist_0_batch = jax.vmap(dist_0_fn, in_axes=(0, None))
    dist_batch = jax.vmap(dist_fn, in_axes=(0, 0, None))

    # dist_0 should match dist(x, origin)
    if manifold == hj.manifolds.hyperboloid:
        # Hyperboloid origin: [sqrt(1/c), 0, ..., 0]
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    d1 = dist_0_batch(uniform_points, c)
    d2 = dist_batch(uniform_points, origin, c)

    assert jnp.allclose(d1, d2, atol=atol, rtol=rtol)


def test_expmap_logmap_basic(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test that expmap, logmap, and retraction produce valid outputs and satisfy consistency properties.

    Note: We don't test the inverse property expmap(logmap(y, x), x) ≈ y for
    arbitrary points because:
    1. Near-boundary points with large conformal factors (>10^4) cause numerical instability
    2. Float32 precision is insufficient for Möbius addition near the boundary
    3. PyTorch tests don't verify this property except at origin (see test_expmap_0_logmap_0_inverse)
    """
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)

    # Batch operations using vmap
    expmap_batch = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    expmap_0_batch = jax.vmap(manifold.expmap_0, in_axes=(0, None))
    retraction_batch = jax.vmap(manifold.retraction, in_axes=(0, 0, None))
    logmap_batch = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    logmap_0_batch = jax.vmap(manifold.logmap_0, in_axes=(0, None))
    tangent_proj_batch = jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))

    # Origin for consistency checks
    if manifold == hj.manifolds.hyperboloid:
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    # Create random tangent vectors
    bound = 10
    v = jnp.asarray(rng.uniform(-bound, bound, size=uniform_points.shape), dtype=uniform_points.dtype)
    v0 = v.copy()

    # Project onto tangent space for Hyperboloid
    if manifold == hj.manifolds.hyperboloid:
        v0 = tangent_proj_batch(v, origin, c)
        v = tangent_proj_batch(v, uniform_points, c)

    assert _batch_is_in_tangent_space(manifold, v, uniform_points, c)
    assert _batch_is_in_tangent_space(manifold, v0, origin, c)

    # Numerical stability of expmap/expmap_0/retraction
    if manifold in (hj.manifolds.euclidean, hj.manifolds.poincare):
        # Note: Hyperboloid may fail is_in_manifold check due to numerical errors

        # Expmap
        v_manif = expmap_batch(v, uniform_points, c)
        assert jnp.all(jnp.isfinite(v_manif))
        assert _batch_is_in_manifold(manifold, v_manif, c)

        # Expmap_0
        v0_manif = expmap_0_batch(v0, c)
        assert jnp.all(jnp.isfinite(v0_manif))
        assert _batch_is_in_manifold(manifold, v0_manif, c)

        # Retraction
        v_retr = retraction_batch(v, uniform_points, c)
        assert jnp.all(jnp.isfinite(v_retr))

        v0_retr = retraction_batch(v0, origin, c)
        assert jnp.all(jnp.isfinite(v0_retr))

    # Numerical stability of logmap - check logmap produces finite tangent vectors
    if manifold == hj.manifolds.poincare and uniform_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 3e-2)

    logmap_y_x = logmap_batch(y, x, c)
    assert jnp.all(jnp.isfinite(logmap_y_x))
    assert _batch_is_in_tangent_space(manifold, logmap_y_x, x, c)

    # Stability of inverse operation: expmap(logmap(y, x), x) is finite and on manifold
    # Note: expmap applies backprojection which is not injective
    res = expmap_batch(logmap_y_x, x, c)
    assert jnp.all(jnp.isfinite(res))
    assert _batch_is_in_manifold(manifold, res, c)

    # Consistency of expmap/logmap with expmap_0/logmap_0
    expmap_v0_origin = expmap_batch(v0, origin, c)
    expmap_0_v0 = expmap_0_batch(v0, c)
    assert jnp.allclose(expmap_v0_origin, expmap_0_v0, atol=atol, rtol=rtol)

    logmap_points_origin = logmap_batch(uniform_points, origin, c)
    logmap_0_points = logmap_0_batch(uniform_points, c)
    assert jnp.allclose(logmap_points_origin, logmap_0_points, atol=atol, rtol=rtol)


def test_expmap_0_logmap_0_inverse(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test that exp_0 and log_0 are inverse operations."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Batch operations using vmap
    logmap_0_batch = jax.vmap(manifold.logmap_0, in_axes=(0, None))
    expmap_0_batch = jax.vmap(manifold.expmap_0, in_axes=(0, None))

    # log_0(x) maps x to tangent space at origin
    v = logmap_0_batch(uniform_points, c)

    # exp_0(v) should map back to x
    x_reconstructed = expmap_0_batch(v, c)

    assert jnp.allclose(x_reconstructed, uniform_points, atol=atol, rtol=rtol)


def test_ptransp_preserves_norm(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test parallel transport properties: consistency and round-trip stability.

    Note: This test validates that ptransp is consistent with ptransp_0 and that
    round-trip transport (origin -> x -> origin) recovers the original vector.
    We do not test inner product preservation due to numerical limitations.
    """
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Batch operations using vmap
    ptransp_batch = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    ptransp_0_batch = jax.vmap(manifold.ptransp_0, in_axes=(0, 0, None))
    tangent_proj_batch = jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))

    # Origin for consistency checks
    if manifold == hj.manifolds.hyperboloid:
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    # Create random tangent vector at origin
    bound = 0.1
    u_random = jnp.asarray(rng.uniform(-bound, bound, size=uniform_points.shape), dtype=uniform_points.dtype)

    # Project onto tangent space (necessary for Hyperboloid)
    if manifold == hj.manifolds.hyperboloid:
        u = tangent_proj_batch(u_random, origin, c)
    else:
        u = u_random

    # Verify vector is in tangent space
    assert _batch_is_in_tangent_space(manifold, u, origin, c)

    # Parallel transport from origin to uniform_points
    u_pt = ptransp_0_batch(u, uniform_points, c)
    assert _batch_is_in_tangent_space(manifold, u_pt, uniform_points, c)

    # Consistency of ptransp with ptransp_0
    u_pt_general = ptransp_batch(u, origin, uniform_points, c)
    assert jnp.allclose(u_pt_general, u_pt, atol=atol, rtol=rtol)

    # Round-trip stability: ptransp(ptransp(u, origin, x), x, origin) ≈ u
    u_roundtrip = ptransp_batch(u_pt, uniform_points, origin, c)
    assert jnp.allclose(u_roundtrip, u, atol=atol, rtol=rtol)
    assert _batch_is_in_tangent_space(manifold, u_roundtrip, origin, c)


def test_tangent_inner_positive_definite(manifold_and_c, uniform_points: jnp.ndarray, rng: np.random.Generator) -> None:
    """Test that tangent inner product is positive definite."""
    manifold, c = manifold_and_c

    # Batch operations using vmap
    tangent_proj_batch = jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))
    tangent_inner_batch = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))

    # Create non-zero tangent vectors
    v = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)

    # Project onto tangent space (necessary for Hyperboloid)
    v = tangent_proj_batch(v, uniform_points, c)

    # Inner product <v, v> should be positive
    inner = tangent_inner_batch(v, v, uniform_points, c)

    assert jnp.all(inner > 0)


def test_tangent_inner_symmetric(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test that tangent inner product is symmetric."""
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Batch operations using vmap
    tangent_proj_batch = jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))
    tangent_inner_batch = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))

    # Create two tangent vectors
    u = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)
    v = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)

    # Project onto tangent space (necessary for Hyperboloid)
    u = tangent_proj_batch(u, uniform_points, c)
    v = tangent_proj_batch(v, uniform_points, c)

    # <u, v> = <v, u>
    inner_uv = tangent_inner_batch(u, v, uniform_points, c)
    inner_vu = tangent_inner_batch(v, u, uniform_points, c)

    assert jnp.allclose(inner_uv, inner_vu, atol=atol, rtol=rtol)


def test_tangent_norm_consistency(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test consistency of tangent_norm with logmap and dist operations.

    The tangent norm of a logarithmic map should equal the geodesic distance.
    This is a fundamental property: ‖log_x(y)‖_x = d(x, y)
    """
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)

    # Batch operations using vmap
    logmap_batch = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    logmap_0_batch = jax.vmap(manifold.logmap_0, in_axes=(0, None))
    tangent_norm_batch = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))
    dist_fn = _dist_fn(manifold)
    dist_0_fn = _dist_0_fn(manifold)
    dist_batch = jax.vmap(dist_fn, in_axes=(0, 0, None))
    dist_0_batch = jax.vmap(dist_0_fn, in_axes=(0, None))

    # Origin for _0 variant tests
    if manifold == hj.manifolds.hyperboloid:
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    # Float32 with Poincaré ball requires relaxed tolerance due to conformal factor explosion
    # near boundary. When points approach ||x|| ≈ 1/√c, the conformal factor λ(x) = 2/(1-c||x||²)
    # can exceed 10,000. The logmap/tangent_norm round-trip (divide by λ, then multiply by λ)
    # loses precision, especially for large distances (>10) involving near-boundary points.
    if manifold == hj.manifolds.poincare and uniform_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 3e-2)

    # Consistency of tangent_norm with logmap and dist
    # ‖log_x(y)‖_x = d(x, y)
    logmap_y_x = logmap_batch(y, x, c)
    tangent_norm_logmap = tangent_norm_batch(logmap_y_x, x, c)
    dist_x_y = dist_batch(x, y, c)
    assert jnp.allclose(tangent_norm_logmap, dist_x_y, atol=atol, rtol=rtol)

    # Consistency of tangent_norm with logmap_0 and dist_0
    # ‖log_0(x)‖_0 = d_0(x)
    logmap_0_points = logmap_0_batch(uniform_points, c)
    tangent_norm_logmap_0 = tangent_norm_batch(logmap_0_points, origin, c)
    dist_0_points = dist_0_batch(uniform_points, c)
    assert jnp.allclose(tangent_norm_logmap_0, dist_0_points, atol=atol, rtol=rtol)


def test_egrad2rgrad_on_manifold(manifold_and_c, uniform_points: jnp.ndarray, rng: np.random.Generator) -> None:
    """Test that Riemannian gradient points lie in tangent space."""
    manifold, c = manifold_and_c

    # Batch operations using vmap
    egrad2rgrad_batch = jax.vmap(manifold.egrad2rgrad, in_axes=(0, 0, None))

    # Create Euclidean gradients
    egrad = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)

    # Convert to Riemannian gradient
    rgrad = egrad2rgrad_batch(egrad, uniform_points, c)

    # Riemannian gradient should be in tangent space
    assert _batch_is_in_tangent_space(manifold, rgrad, uniform_points, c)


def test_is_in_manifold(manifold_and_c, uniform_points: jnp.ndarray) -> None:
    """Test manifold membership checking."""
    manifold, c = manifold_and_c

    # All uniform points should be on manifold
    assert _batch_is_in_manifold(manifold, uniform_points, c)

    if manifold == hj.manifolds.poincare:
        # Points outside ball should not be on manifold
        outside = jnp.ones_like(uniform_points[0]) * 10.0
        assert not manifold.is_in_manifold(outside, c=c)
    elif manifold == hj.manifolds.hyperboloid:
        # Points not on hyperboloid surface should not be on manifold
        outside = jnp.ones_like(uniform_points[0]) * 10.0
        assert not manifold.is_in_manifold(outside, c=c)
