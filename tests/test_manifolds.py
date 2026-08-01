"""JAX manifold tests - vmap-native API.

Tests for the hyperbolix backend using vmap-native pure functions.
Adapted for the new single-point API with vmap for batching.

Fixtures are defined in tests/conftest.py and automatically loaded. The generic tests run on
``manifold_and_c`` (Euclidean / Poincaré / Hyperboloid); ProperVelocity is covered by
``tests/test_pv_manifold.py``, whose namesakes are strictly stronger. Manifold-specific tests
request the dedicated ``poincare_and_c`` / ``hyperboloid_and_c`` fixtures instead of skipping
three quarters of a four-way parametrization.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hyperbolix as hj
import hyperbolix.manifolds.poincare as poincare_impl
from hyperbolix.manifolds import isometry_mappings

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
    """Return distance function.

    Class-based manifolds already provide the desired default distance version.
    """
    return manifold.dist


def _dist_0_fn(manifold):
    """Return origin distance function.

    Class-based manifolds already provide the desired default distance version.
    """
    return manifold.dist_0


def _is_euclidean(manifold) -> bool:
    return isinstance(manifold, hj.manifolds.Euclidean)


def _is_poincare(manifold) -> bool:
    return isinstance(manifold, hj.manifolds.Poincare)


def _is_hyperboloid(manifold) -> bool:
    return isinstance(manifold, hj.manifolds.Hyperboloid)


def _is_gyrovector(manifold) -> bool:
    """Manifolds that carry a gyrovector-space structure (non-trivial scalar mul axioms)."""
    return _is_euclidean(manifold) or _is_poincare(manifold)


def _random_ball_point(rng: np.random.Generator, dim: int, max_radius: float) -> np.ndarray:
    """Random point strictly inside the ball of the given radius (uniform-on-ball direction/radius)."""
    direction = rng.normal(0.0, 1.0, size=dim)
    direction /= np.linalg.norm(direction)
    r = max_radius * rng.random() ** (1.0 / dim)
    return direction * r


def _shrink_for_float32(manifold, points: jnp.ndarray, c: float) -> jnp.ndarray:
    """Pull points toward the origin in float32 so geodesic distances stay in the library's
    reliable range (< 7); a no-op in float64.

    The Apollonian identities tested below (symmetrization, closed-form values, δ(x,x)=0) hold
    at every interior point, so shrinking is harmless — it just avoids the near-boundary regime
    where float32 underflows the conformal factor. The float64 parametrization runs the same
    tests unshrunk and so still covers the full near-boundary range.
    """
    if points.dtype != jnp.dtype("float32"):
        return points
    scalar_mul_batch = jax.vmap(manifold.scalar_mul, in_axes=(0, 0, None))
    factor = jnp.full(points.shape[0], 0.2, dtype=points.dtype)
    return scalar_mul_batch(factor, points, c)


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
    if _is_euclidean(manifold):
        assert jnp.allclose(projected, uniform_points)
        assert jnp.allclose(projected_single, sample)
    else:
        # For hyperbolic manifolds, points might be slightly adjusted
        assert jnp.allclose(projected, uniform_points, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(projected_single, sample, rtol=1e-5, atol=1e-5)


def test_addition(manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray) -> None:
    """Test addition/Möbius addition operation."""
    manifold, c = manifold_and_c

    # The Hyperboloid implements the Lorentz gyrovector addition (Shi et al. 2026, Eq. 1):
    #   x ⊕ y = Exp_x(PT_{0→x}(Log_0(y))).
    # Its identity is the origin [√(1/c), 0…] (NOT zeros) and its inverse is
    # ⊖x = (-1) ⊙ x = [x₀, -x_s] (NOT -x), so the generic Euclidean-style body below does
    # not apply — verify the gyrogroup axioms here with the correct identity/inverse.
    if _is_hyperboloid(manifold):
        atol, rtol = tolerance
        if uniform_points.dtype == jnp.dtype("float32"):
            # Gyroaddition is a log_0 → ptransp → exp round-trip; relax f32 like other manifolds.
            atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)

        addition_batch = jax.vmap(manifold.addition, in_axes=(0, 0, None))
        scalar_mul_batch = jax.vmap(manifold.scalar_mul, in_axes=(0, 0, None))

        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))

        # Closed-form inverse: ⊖x = (-1) ⊙ x = [x₀, -x_s]
        neg_ones = -jnp.ones(uniform_points.shape[0], dtype=uniform_points.dtype)
        ominus = scalar_mul_batch(neg_ones, uniform_points, c)
        ominus_expected = uniform_points.at[:, 1:].multiply(-1.0)
        assert jnp.allclose(ominus, ominus_expected, atol=atol, rtol=rtol)

        # Left/right identity: 0 ⊕ x = x and x ⊕ 0 = x
        assert jnp.allclose(addition_batch(origin, uniform_points, c), uniform_points, atol=atol, rtol=rtol)
        assert jnp.allclose(addition_batch(uniform_points, origin, c), uniform_points, atol=atol, rtol=rtol)

        # Left/right inverse: (⊖x) ⊕ x = 0 and x ⊕ (⊖x) = 0
        assert jnp.allclose(addition_batch(ominus, uniform_points, c), origin, atol=atol, rtol=rtol)
        assert jnp.allclose(addition_batch(uniform_points, ominus, c), origin, atol=atol, rtol=rtol)

        # Result stays on the manifold
        x_h, y_h = _split(uniform_points, 2)
        assert _batch_is_in_manifold(manifold, addition_batch(x_h, y_h, c), c)
        return

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


@pytest.mark.parametrize("n", [2, 5])
def test_hyperboloid_gyro_addition(
    n: int, hyperboloid_and_c, tolerance: tuple[float, float], hyperboloid_points: jnp.ndarray
) -> None:
    """Deep correctness checks for the Hyperboloid Lorentz gyroaddition (Shi et al. 2026, Eq. 1).

    Validated three independent ways:
      1. Isometry cross-check against the trusted Poincaré Möbius addition (primary oracle):
         the stereographic projection is a gyrogroup isomorphism, so hyperboloid ⊕ must agree
         with adding in the Poincaré ball and mapping back.
      2. n-gyroaddition: n ⊙ x = x ⊕ x ⊕ … ⊕ x links the gyroaddition (Eq. 1) to the gyro
         scalar multiplication (Eq. 2).
      3. Left cancellation: (⊖x) ⊕ (x ⊕ y) = y, the defining law of a (left) gyrogroup.

    ``n`` (the multiplicity in check 2) is an explicit axis rather than an ``rng`` draw so a
    single-seed run still exercises more than one multiplicity.
    """
    manifold, c = hyperboloid_and_c
    uniform_points = hyperboloid_points

    atol, rtol = tolerance
    if uniform_points.dtype == jnp.dtype("float32"):
        atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)

    addition_batch = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    scalar_mul_batch = jax.vmap(manifold.scalar_mul, in_axes=(0, 0, None))
    x, y = _split(uniform_points, 2)

    # (1) Isometry cross-check against Poincaré Möbius addition (single addition — robust).
    poincare = hj.manifolds.Poincare(dtype=uniform_points.dtype)
    h2p = jax.vmap(isometry_mappings.hyperboloid_to_poincare, in_axes=(0, None))
    p2h = jax.vmap(isometry_mappings.poincare_to_hyperboloid, in_axes=(0, None))
    mobius_add = jax.vmap(poincare.addition, in_axes=(0, 0, None))

    got = addition_batch(x, y, c)
    expected = p2h(mobius_add(h2p(x, c), h2p(y, c), c), c)
    assert jnp.allclose(got, expected, atol=atol, rtol=rtol)
    assert _batch_is_in_manifold(manifold, got, c)

    # Shrink points so the multi-addition checks below keep geodesic distances in the
    # float32-reliable range (< 7); the gyrogroup laws are scale-independent.
    quarter = jnp.full(x.shape[0], 0.2, dtype=x.dtype)
    xs = scalar_mul_batch(quarter, x, c)
    ys = scalar_mul_batch(quarter, y, c)

    # (2) n-gyroaddition: n ⊙ xs = xs ⊕ xs ⊕ … ⊕ xs (n times).
    n_sum = xs
    for _ in range(n - 1):
        n_sum = addition_batch(n_sum, xs, c)
    n_scalar = jnp.full(xs.shape[0], float(n), dtype=xs.dtype)
    n_scaled = scalar_mul_batch(n_scalar, xs, c)
    assert jnp.allclose(n_sum, n_scaled, atol=atol, rtol=rtol)

    # (3) Left cancellation: (⊖xs) ⊕ (xs ⊕ ys) = ys.
    neg_ones = -jnp.ones(xs.shape[0], dtype=xs.dtype)
    ominus_xs = scalar_mul_batch(neg_ones, xs, c)
    left_cancel = addition_batch(ominus_xs, addition_batch(xs, ys, c), c)
    assert jnp.allclose(left_cancel, ys, atol=atol, rtol=rtol)


def test_hyperboloid_scalar_mul_eq2(
    hyperboloid_and_c, tolerance: tuple[float, float], hyperboloid_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Verify Hyperboloid.scalar_mul equals the paper's Eq. 2: t ⊙ x = Exp_0(t · Log_0(x)).

    ``scalar_mul`` is written in the normalized form Exp_0(t · d(0,x) · Log_0(x)/‖Log_0(x)‖);
    since ‖Log_0(x)‖_L = d(0,x) this collapses to Eq. 2. This test confirms the equality.
    """
    manifold, c = hyperboloid_and_c
    uniform_points = hyperboloid_points

    atol, rtol = tolerance
    if uniform_points.dtype == jnp.dtype("float32"):
        atol, rtol = max(atol, 4e-3), max(rtol, 2e-2)

    scalar_mul_batch = jax.vmap(manifold.scalar_mul, in_axes=(0, 0, None))
    logmap_0_batch = jax.vmap(manifold.logmap_0, in_axes=(0, None))
    expmap_0_batch = jax.vmap(manifold.expmap_0, in_axes=(0, None))

    # Keep |t| · d(0,x) within the float32-reliable range.
    t = jnp.asarray(rng.uniform(-1.5, 1.5, size=uniform_points.shape[0]), dtype=uniform_points.dtype)

    got = scalar_mul_batch(t, uniform_points, c)
    # Eq. 2 directly: scale the origin-tangent by t, then exp back.
    v0 = logmap_0_batch(uniform_points, c)
    expected = expmap_0_batch(t[:, None] * v0, c)
    assert jnp.allclose(got, expected, atol=atol, rtol=rtol)
    assert _batch_is_in_manifold(manifold, got, c)


@pytest.mark.parametrize("n", [3, 6])
def test_scalar_mul(
    n: int, manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test scalar multiplication operation.

    ``n`` (the multiplicity in the n-gyroaddition check) is an explicit axis rather than an
    ``rng`` draw so a single-seed run still exercises more than one multiplicity.
    """
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    if _is_poincare(manifold) and uniform_points.dtype == jnp.dtype("float32"):
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

    # Additional gyrovector properties (Euclidean, Poincaré, PV — not Hyperboloid)
    if _is_gyrovector(manifold):
        # N-Gyroaddition property: n ⊗ x = x ⊕ x ⊕ ... ⊕ x (n times)
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
    if _is_hyperboloid(manifold):
        v_eps_norm = v_eps_norm.at[0, 0].set(v_eps_norm[0, 0] + jnp.sqrt(1.0 / c))
        proj_single = jax.vmap(manifold.proj, in_axes=(0, None))
        v_eps_norm = proj_single(v_eps_norm, c)

    # Origin for comparison
    if _is_hyperboloid(manifold):
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
    if _is_gyrovector(manifold):
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
    poincare_and_c, tolerance: tuple[float, float], poincare_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Test the gyration operation of the PoincareBall.

    Gyration is a fundamental operation in gyrogroups that restores commutativity
    in the non-commutative Möbius addition. This test verifies all the gyrogroup
    axioms and properties.

    Poincaré-only: this exercises the Möbius ``_gyration`` helper (PV has its own, distinct
    gyration algebra, which is not routed through this helper).
    """
    manifold, c = poincare_and_c
    uniform_points = poincare_points

    atol, rtol = tolerance
    x, y, z, a = _split(uniform_points, 4)

    # Batch operations using vmap
    addition_batch = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    gyration_batch = jax.vmap(poincare_impl._gyration, in_axes=(0, 0, 0, None))
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
    if _is_hyperboloid(manifold):
        # Hyperboloid origin: [sqrt(1/c), 0, ..., 0]
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    d1 = dist_0_batch(uniform_points, c)
    d2 = dist_batch(uniform_points, origin, c)

    assert jnp.allclose(d1, d2, atol=atol, rtol=rtol)


def test_hyperboloid_sqdist(hyperboloid_and_c, tolerance: tuple[float, float], hyperboloid_points: jnp.ndarray) -> None:
    """Squared Lorentzian distance (Law et al. 2019), the acosh-free distance proxy.

    Checks it is zero at coincidence, non-negative, symmetric, and monotonically tied to the
    geodesic distance via the closed form d_L²(x, y) = (2/c)(cosh(√c · d(x, y)) - 1) — verified
    against an independent acosh-based ``dist``.
    """
    manifold, c = hyperboloid_and_c
    uniform_points = hyperboloid_points
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)
    # Keep geodesic distances in the float32-reliable range so cosh(√c·d) does not overflow the
    # closed-form check; a no-op in float64.
    x, y = _shrink_for_float32(manifold, x, c), _shrink_for_float32(manifold, y, c)

    sqdist_batch = jax.vmap(manifold.sqdist, in_axes=(0, 0, None))
    dist_batch = jax.vmap(manifold.dist, in_axes=(0, 0, None))

    # Non-negativity: d_L²(x, y) ≥ 0
    d2_xy = sqdist_batch(x, y, c)
    assert jnp.all(d2_xy >= -atol)

    # Identity: d_L²(x, x) = 0
    d2_xx = sqdist_batch(x, x, c)
    assert jnp.allclose(d2_xx, 0.0, atol=atol, rtol=rtol)

    # Symmetry: d_L²(x, y) = d_L²(y, x)
    d2_yx = sqdist_batch(y, x, c)
    assert jnp.allclose(d2_xy, d2_yx, atol=atol, rtol=rtol)

    # Closed-form tie to the (independent) geodesic distance: d_L² = (2/c)(cosh(√c·d) - 1).
    d_xy = dist_batch(x, y, c)
    expected = (2.0 / c) * (jnp.cosh(jnp.sqrt(c) * d_xy) - 1.0)
    is_f32 = uniform_points.dtype == jnp.dtype("float32")
    assert jnp.allclose(d2_xy, expected, atol=max(atol, 1e-3), rtol=max(rtol, 1e-2 if is_f32 else 1e-6))


# ---------------------------------------------------------------------------
# NaN-gradient regression guards.
#
# These are singular-point guards: the failure mode is a 0/0 or 0·inf in a VJP at one specific
# point, which is independent of the sampled curvature, the ambient dimension and the batch.
# They therefore use a hardcoded point instead of the fixture matrix (one collected item each).


def test_hyperboloid_tangent_norm_zero_vector_finite_grad() -> None:
    """``tangent_norm`` must have a finite gradient at the zero tangent vector.

    sqrt'(0) = inf, so a bare ``sqrt(clip(⟨v,v⟩_L, 0))`` yields a 0·inf = NaN gradient at v = 0;
    the ``+ MIN_NORM²`` floor (matching ``_expmap``) keeps it finite.
    """
    manifold, c = hj.manifolds.Hyperboloid(dtype=jnp.float64), 1.0

    x0 = manifold.proj(jnp.array([1.0, 0.3, -0.2], dtype=jnp.float64), c)
    v0 = jnp.zeros_like(x0)  # zero tangent vector at x0

    n = manifold.tangent_norm(v0, x0, c)
    assert jnp.isfinite(n)

    grad = jax.grad(lambda v: manifold.tangent_norm(v, x0, c))(v0)
    assert jnp.all(jnp.isfinite(grad))


def test_poincare_tangent_norm_zero_vector_finite_grad() -> None:
    """``tangent_norm`` must have a finite gradient at the zero tangent vector (Poincaré).

    ``λ(x)·||v||`` with a bare ``jnp.linalg.norm`` has VJP 0/0 = NaN at v = 0; the safe norm
    ``sqrt(||v||² + MIN_NORM²)`` (matching ``_expmap``/``_proj``) keeps it finite.
    """
    manifold, c = hj.manifolds.Poincare(dtype=jnp.float64), 1.0

    x0 = jnp.array([0.3, -0.2], dtype=jnp.float64)
    v0 = jnp.zeros_like(x0)  # zero tangent vector at x0

    n = manifold.tangent_norm(v0, x0, c)
    assert jnp.isfinite(n)

    grad = jax.grad(lambda v: manifold.tangent_norm(v, x0, c))(v0)
    assert jnp.all(jnp.isfinite(grad))


def test_poincare_logmap_coincident_finite_grad() -> None:
    """``logmap`` must have a finite gradient when y coincides with x (Poincaré).

    ``num = ||y - x||`` via a bare ``jnp.linalg.norm`` has VJP 0/0 = NaN at y = x; the safe norm
    keeps the gradient finite w.r.t. both arguments. The forward value is 0 either way.
    """
    manifold, c = hj.manifolds.Poincare(dtype=jnp.float64), 1.0

    x0 = jnp.array([0.3, -0.2], dtype=jnp.float64)
    y0 = x0  # coincident target

    v = manifold.logmap(y0, x0, c)
    assert jnp.all(jnp.isfinite(v))

    grad_y = jax.grad(lambda y: jnp.sum(manifold.logmap(y, x0, c)))(y0)
    grad_x = jax.grad(lambda x: jnp.sum(manifold.logmap(y0, x, c)))(x0)
    assert jnp.all(jnp.isfinite(grad_y))
    assert jnp.all(jnp.isfinite(grad_x))


# ---------------------------------------------------------------------------
# Apollonian weak metric (Poincaré-only) — Papadopoulos & Troyanov, Theorem 2


def test_apollonian_symmetrization_is_dist(
    poincare_and_c, tolerance: tuple[float, float], poincare_points: jnp.ndarray
) -> None:
    """Symmetrizing the Apollonian weak metric recovers the geodesic distance.

    δ(x, y) + δ(y, x) = √c · dist(x, y)  (Papadopoulos & Troyanov Cor 5.1).
    The √c factor reflects that the paper's Poincaré metric h_{D²} uses the curvature -4
    normalization — exactly half of hyperbolix's curvature -1 ``dist`` (so the factor is 1 at c=1).
    """
    manifold, c = poincare_and_c
    uniform_points = poincare_points
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)
    x, y = _shrink_for_float32(manifold, x, c), _shrink_for_float32(manifold, y, c)
    if uniform_points.dtype == jnp.dtype("float32"):
        atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)

    apoll_batch = jax.vmap(manifold.apollonian_dist, in_axes=(0, 0, None))
    dist_batch = jax.vmap(manifold.dist, in_axes=(0, 0, None))

    symmetrized = apoll_batch(x, y, c) + apoll_batch(y, x, c)
    geodesic = jnp.sqrt(c) * dist_batch(x, y, c)

    assert jnp.allclose(symmetrized, geodesic, atol=atol, rtol=rtol)


def test_apollonian_special_values(poincare_and_c, tolerance: tuple[float, float], poincare_points: jnp.ndarray) -> None:
    """Closed-form values against the origin (Papadopoulos & Troyanov Cor 5.2, generalized to c):

    δ(x, 0) = log(1 + √c‖x‖)   and   δ(0, x) = -log(1 - √c‖x‖).

    These two closed forms also pin the weak-metric asymmetry δ(x, y) ≠ δ(y, x): they are
    unequal by construction for x ≠ 0, which is why no separate "non-symmetric" test is needed.
    """
    manifold, c = poincare_and_c
    uniform_points = poincare_points
    atol, rtol = tolerance

    x = _shrink_for_float32(manifold, uniform_points, c)
    if uniform_points.dtype == jnp.dtype("float32"):
        atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)
    origin = jnp.zeros_like(x)
    apoll_batch = jax.vmap(manifold.apollonian_dist, in_axes=(0, 0, None))

    sqrt_c_norm = jnp.sqrt(c) * jnp.linalg.norm(x, axis=-1)  # √c‖x‖, shape (N,)

    d_x0 = apoll_batch(x, origin, c)  # δ(x, 0)
    d_0x = apoll_batch(origin, x, c)  # δ(0, x)

    assert jnp.allclose(d_x0, jnp.log1p(sqrt_c_norm), atol=atol, rtol=rtol)
    assert jnp.allclose(d_0x, -jnp.log1p(-sqrt_c_norm), atol=atol, rtol=rtol)

    # δ is a *weak* metric: it is not symmetric. Tight tolerance so "not close" is a meaningful
    # assertion, not float noise.
    assert not jnp.allclose(d_x0, d_0x, atol=1e-4, rtol=1e-4)


def test_apollonian_basic_properties(poincare_and_c, tolerance: tuple[float, float], poincare_points: jnp.ndarray) -> None:
    """Weak-metric basics: δ(x, x) = 0 and δ(x, y) ≥ 0."""
    manifold, c = poincare_and_c
    uniform_points = poincare_points
    atol, rtol = tolerance

    x, y = _split(uniform_points, 2)
    x, y = _shrink_for_float32(manifold, x, c), _shrink_for_float32(manifold, y, c)

    apoll_batch = jax.vmap(manifold.apollonian_dist, in_axes=(0, 0, None))

    d_xx = apoll_batch(x, x, c)
    assert jnp.allclose(d_xx, 0.0, atol=atol, rtol=rtol)

    d_xy = apoll_batch(x, y, c)
    assert jnp.all(d_xy >= -atol)


@pytest.mark.parametrize("dim", [2, 3])
def test_apollonian_matches_boundary_supremum(dim: int, rng: np.random.Generator) -> None:
    """Validate the closed form directly against the definition (Papadopoulos & Troyanov Eq. 6):

        δ(x, y) = sup_{‖a‖=1/√c} log(‖x - a‖ / ‖y - a‖),

    the supremum over the ball boundary sphere. Running dim=3 (not just the disk) confirms the
    maximizer lies in span(x, y) — i.e. the closed form genuinely generalizes beyond n=2.
    """
    manifold = hj.manifolds.Poincare(dtype=jnp.float64)
    c = 1.0
    radius = 1.0 / np.sqrt(c)
    n_theta = 8192
    theta = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False), dtype=jnp.float64)

    for _ in range(5):
        # Keep points off the boundary so the finite-sample sup is clean (δ blows up at ∂ball).
        x = jnp.asarray(_random_ball_point(rng, dim, radius * 0.85), dtype=jnp.float64)
        y = jnp.asarray(_random_ball_point(rng, dim, radius * 0.85), dtype=jnp.float64)

        # Orthonormal basis {e1, e2} of span(x, y) via Gram-Schmidt (non-collinear almost surely).
        e1 = x / jnp.linalg.norm(x)
        w = y - jnp.dot(y, e1) * e1
        e2 = w / jnp.linalg.norm(w)

        # Sweep the great circle a(θ) = radius·(cosθ·e1 + sinθ·e2) on the boundary sphere.
        a_TD = radius * (jnp.outer(jnp.cos(theta), e1) + jnp.outer(jnp.sin(theta), e2))  # (n_theta, dim)
        f_T = jnp.log(jnp.linalg.norm(x - a_TD, axis=-1) / jnp.linalg.norm(y - a_TD, axis=-1))
        sup_numeric = jnp.max(f_T)

        closed_form = manifold.apollonian_dist(x, y, c)

        # Closed form is the true supremum (dominates any finite sample) and a dense sweep matches.
        assert closed_form >= sup_numeric - 1e-9
        assert jnp.allclose(closed_form, sup_numeric, atol=1e-4, rtol=1e-4)


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
    if _is_hyperboloid(manifold):
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    # Create random tangent vectors.
    bound = 10
    v = jnp.asarray(rng.uniform(-bound, bound, size=uniform_points.shape), dtype=uniform_points.dtype)
    v0 = v.copy()

    # Project onto tangent space for Hyperboloid
    if _is_hyperboloid(manifold):
        v0 = tangent_proj_batch(v, origin, c)
        v = tangent_proj_batch(v, uniform_points, c)

    assert _batch_is_in_tangent_space(manifold, v, uniform_points, c)
    assert _batch_is_in_tangent_space(manifold, v0, origin, c)

    # Numerical stability of expmap/expmap_0/retraction
    if _is_gyrovector(manifold):
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
    if _is_poincare(manifold) and uniform_points.dtype == jnp.dtype("float32"):
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


def test_ptransp_is_an_isometry_and_round_trips(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """Parallel transport is a linear isometry between tangent spaces.

    Checks (a) norm preservation ‖PT_{0→x}(u)‖_x = ‖u‖_0 — the defining property, which the
    previous version of this test claimed in its name but never asserted (audit A1-F5) —
    plus (b) consistency of ``ptransp`` with ``ptransp_0`` and (c) an origin → x → origin
    round trip.
    """
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Batch operations using vmap
    ptransp_batch = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    ptransp_0_batch = jax.vmap(manifold.ptransp_0, in_axes=(0, 0, None))
    tangent_proj_batch = jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))
    tangent_norm_batch = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))

    # Origin for consistency checks
    if _is_hyperboloid(manifold):
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    # Create random tangent vector at origin
    bound = 0.1
    u_random = jnp.asarray(rng.uniform(-bound, bound, size=uniform_points.shape), dtype=uniform_points.dtype)

    # Project onto tangent space (necessary for Hyperboloid)
    if _is_hyperboloid(manifold):
        u = tangent_proj_batch(u_random, origin, c)
    else:
        u = u_random

    # Verify vector is in tangent space
    assert _batch_is_in_tangent_space(manifold, u, origin, c)

    # Parallel transport from origin to uniform_points
    u_pt = ptransp_0_batch(u, uniform_points, c)
    assert _batch_is_in_tangent_space(manifold, u_pt, uniform_points, c)

    # Isometry: transport preserves the Riemannian norm of the transported vector
    assert jnp.allclose(
        tangent_norm_batch(u_pt, uniform_points, c),
        tangent_norm_batch(u, origin, c),
        atol=atol,
        rtol=rtol,
    )

    # Consistency of ptransp with ptransp_0
    u_pt_general = ptransp_batch(u, origin, uniform_points, c)
    assert jnp.allclose(u_pt_general, u_pt, atol=atol, rtol=rtol)

    # Round-trip stability: ptransp(ptransp(u, origin, x), x, origin) ≈ u
    u_roundtrip = ptransp_batch(u_pt, uniform_points, origin, c)
    assert jnp.allclose(u_roundtrip, u, atol=atol, rtol=rtol)
    assert _batch_is_in_tangent_space(manifold, u_roundtrip, origin, c)


def test_tangent_inner_is_an_inner_product(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """``tangent_inner`` must be symmetric, positive definite and bilinear.

    (Merged from the former ``test_tangent_inner_symmetric`` / ``..._positive_definite``, which
    shared the same setup and one assertion each; linearity in the first slot is added here.)
    """
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Batch operations using vmap
    tangent_proj_batch = jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))
    tangent_inner_batch = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))

    # Create two tangent vectors (projection onto the tangent space is necessary for Hyperboloid)
    u = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)
    v = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)
    u = tangent_proj_batch(u, uniform_points, c)
    v = tangent_proj_batch(v, uniform_points, c)

    # Symmetry: <u, v> = <v, u>
    inner_uv = tangent_inner_batch(u, v, uniform_points, c)
    inner_vu = tangent_inner_batch(v, u, uniform_points, c)
    assert jnp.allclose(inner_uv, inner_vu, atol=atol, rtol=rtol)

    # Positive definiteness: <v, v> > 0 for the (a.s. nonzero) random tangent vectors
    assert jnp.all(tangent_inner_batch(v, v, uniform_points, c) > 0)

    # Bilinearity (first slot): <a·u + b·v, v> = a·<u, v> + b·<v, v>
    a = jnp.asarray(rng.uniform(-2.0, 2.0, size=(uniform_points.shape[0], 1)), dtype=uniform_points.dtype)
    b = jnp.asarray(rng.uniform(-2.0, 2.0, size=(uniform_points.shape[0], 1)), dtype=uniform_points.dtype)
    combo = a * u + b * v  # a linear combination of tangent vectors is tangent
    inner_combo = tangent_inner_batch(combo, v, uniform_points, c)
    expected = a[:, 0] * inner_uv + b[:, 0] * tangent_inner_batch(v, v, uniform_points, c)
    assert jnp.allclose(inner_combo, expected, atol=atol, rtol=rtol)


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
    if _is_hyperboloid(manifold):
        origin = jnp.zeros_like(uniform_points)
        origin = origin.at[:, 0].set(jnp.sqrt(1.0 / c))
    else:
        origin = jnp.zeros_like(uniform_points)

    # Float32 with Poincaré ball requires relaxed tolerance due to conformal factor explosion
    # near boundary. When points approach ||x|| ≈ 1/√c, the conformal factor λ(x) = 2/(1-c||x||²)
    # can exceed 10,000. The logmap/tangent_norm round-trip (divide by λ, then multiply by λ)
    # loses precision, especially for large distances (>10) involving near-boundary points.
    if _is_poincare(manifold) and uniform_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 5e-2)

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


def test_egrad2rgrad_is_metric_dual(
    manifold_and_c, tolerance: tuple[float, float], uniform_points: jnp.ndarray, rng: np.random.Generator
) -> None:
    """``egrad2rgrad`` must be the metric dual of the Euclidean gradient, not just tangent.

    Defining property: for every tangent vector v at x,

        ⟨egrad, v⟩_euclidean = g_x(egrad2rgrad(egrad, x), v).

    The tangent-space check alone (the former body of this test) is vacuous on three of the four
    manifolds — ``Euclidean``/``Poincare._is_in_tangent_space`` return a constant ``True`` and
    ``ProperVelocity``'s only checks finiteness — so ``egrad2rgrad`` could be replaced by the
    identity without failing it (audit A1-F3/UM1). It is kept below as a secondary assertion,
    since it is a real check on the Hyperboloid.
    """
    manifold, c = manifold_and_c
    atol, rtol = tolerance

    # Batch operations using vmap
    egrad2rgrad_batch = jax.vmap(manifold.egrad2rgrad, in_axes=(0, 0, None))
    tangent_proj_batch = jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))
    tangent_inner_batch = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))

    # Create Euclidean gradients and test tangent directions
    egrad = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)
    v = jnp.asarray(rng.normal(0.0, 1.0, size=uniform_points.shape), dtype=uniform_points.dtype)
    v = tangent_proj_batch(v, uniform_points, c)  # necessary for Hyperboloid

    # Convert to Riemannian gradient
    rgrad = egrad2rgrad_batch(egrad, uniform_points, c)

    # Metric duality
    euclid_inner = jnp.sum(egrad * v, axis=-1)
    riem_inner = tangent_inner_batch(rgrad, v, uniform_points, c)
    assert jnp.allclose(euclid_inner, riem_inner, atol=atol, rtol=rtol)

    # Riemannian gradient should be in tangent space
    assert _batch_is_in_tangent_space(manifold, rgrad, uniform_points, c)


def test_is_in_manifold(manifold_and_c, uniform_points: jnp.ndarray) -> None:
    """Test manifold membership checking."""
    manifold, c = manifold_and_c

    # All uniform points should be on manifold
    assert _batch_is_in_manifold(manifold, uniform_points, c)

    if _is_poincare(manifold):
        # Points outside ball should not be on manifold
        outside = jnp.ones_like(uniform_points[0]) * 10.0
        assert not manifold.is_in_manifold(outside, c=c)
    elif _is_hyperboloid(manifold):
        # Points not on hyperboloid surface should not be on manifold
        outside = jnp.ones_like(uniform_points[0]) * 10.0
        assert not manifold.is_in_manifold(outside, c=c)
    else:
        # Euclidean: unconstrained, so `is_in_manifold` only checks finiteness — this keeps the
        # Euclidean parametrization of this test from being assertion-free (audit A1-F9).
        # Mirrors ProperVelocity's finite-check expectations (test_pv_manifold.py::
        # test_pv_is_in_manifold_finite_inputs).
        assert _is_euclidean(manifold)
        far_away = jnp.ones_like(uniform_points[0]) * 1e12
        assert bool(manifold.is_in_manifold(far_away, c=c))
        nan_point = uniform_points[0].at[0].set(jnp.nan)
        assert not bool(manifold.is_in_manifold(nan_point, c=c))
        inf_point = uniform_points[0].at[0].set(jnp.inf)
        assert not bool(manifold.is_in_manifold(inf_point, c=c))
