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
from hyperbolix.manifolds._base import default_atol

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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
@pytest.mark.parametrize("t", [0.5, 2.0])
def test_hyperboloid_scalar_mul_matches_the_closed_form_near_the_origin(dtype, c: float, t: float) -> None:
    """``t ⊙ x`` against its closed form, down to spatial radius 1e-8.

    With ``d₀ = arcsinh(√c‖x_s‖)/√c`` and ``x̂_s = x_s/‖x_s‖``::

        (t ⊙ x)₀   = cosh(√c·t·d₀)/√c
        (t ⊙ x)_s  = sinh(√c·t·d₀)/√c · x̂_s

    ``scalar_mul`` used to renormalize ``log_0(x)`` through ``sqrt(maximum(‖v‖², MIN_NORM))``,
    an effective 3.16e-8 floor on the tangent norm that this grid walks straight into (it was
    hidden behind ``dist_0``'s own, larger 1.5e-3 acosh floor and only became reachable once that
    was removed). It is now ``exp_0(t · log_0(x))`` with no floor at all.

    Float32 stops at radius 1e-3: below that ``cosh(√c·t·d₀)`` is 1.0 to the last bit, so the
    time coordinate carries no information and only the spatial part is meaningful.
    """
    manifold = hj.manifolds.Hyperboloid(dtype=dtype)
    sqrt_c = np.sqrt(c)
    radii = (1e-8, 1e-6, 1e-3, 1.0) if dtype == jnp.float64 else (1e-3, 1.0)
    rtol = 1e-12 if dtype == jnp.float64 else 1e-5

    direction = np.zeros(8)
    direction[0] = 1.0

    for r in radii:
        x_s = r * direction
        x_A = jnp.asarray(np.concatenate([[np.sqrt(1.0 / c + r**2)], x_s]).astype(dtype))

        d0 = np.arcsinh(sqrt_c * r) / sqrt_c
        expected_A = np.concatenate([[np.cosh(sqrt_c * t * d0) / sqrt_c], (np.sinh(sqrt_c * t * d0) / sqrt_c) * direction])

        got_A = np.asarray(manifold.scalar_mul(t, x_A, c), dtype=np.float64)
        assert np.allclose(got_A, expected_A, rtol=rtol, atol=0.0), f"r={r}: {got_A} != {expected_A}"


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
# Cancellation-free dist/logmap/tangent_norm at large radius (WS-A hyperbolic-haversine fix)
# ---------------------------------------------------------------------------


def _stored_radius(spatial: np.ndarray, c: float) -> float:
    """``√c·d₀`` of the point whose **stored** spatial part is ``spatial``, in float64.

    ``arcsinh(√c·‖x_s‖)`` from the exact stored values, so it is the radius of the point the
    library is handed rather than of the real-arithmetic point it was meant to be.
    """
    s_64 = np.asarray(spatial, dtype=np.float64)
    return float(np.arcsinh(np.sqrt(c) * np.sqrt(float(np.dot(s_64, s_64)))))


def _radial_pair_exact_in_dtype(manifold, t: float, log2_ratio: int, c: float, dim: int, dtype):
    """Two points on one radial geodesic, collinear **in the dtype** and not merely in the reals.

    ``x = expmap_0(t·v_hat)`` for a one-hot ``v_hat``, and ``y`` carries the spatial part
    ``ldexp(x_s, log2_ratio)`` — an exact power-of-two rescale, every mantissa unchanged — with
    ``y₀ = sqrt(1/c + ‖y_s‖²)`` derived in float64.

    Why that is exact rather than lucky: a float divide is *exponent-invariant*
    (``fl((2^k·p)/(2^k·q)) == fl(p/q)``, the exact quotient being unchanged), and ``safe_norm``
    divides by ``max|v|``, which also scales by exactly ``2^k``. So ``_polar_frame``'s two unit
    directions ``x_s/r_x`` and ``y_s/r_y`` come back **bit-identical**, its angular term ``chord``
    is exactly 0, and the distance reduces to the cancellation-free radial gap term. Building the
    two points from two independently rounded scalings instead leaves them proportional only to
    within an ulp, which the angular term amplifies by ``sqrt(r_x·r_y) ~ e^((a+b)/2)/(2·sqrt(c))``.

    Exponent invariance is measured, not assumed — XLA:GPU's float32 divide is only faithfully
    rounded and ``x/x != 1`` for 15.3 % of float32 values on an A100, yet 0 of 200 000 x 7 exponent
    shifts disagree there (``logs/2026-09-03_collinearity_tests/divide_invariance_gpu.json``).

    Returns ``(x, y, expected_distance)``, the distance being the difference of the two stored
    radii: ``(arcsinh(2^k·sinh(√c·t)) - √c·t)/√c``, which tends to ``k·ln 2/√c`` at large radius.
    """
    v_hat = jnp.zeros(dim + 1, dtype=dtype).at[1].set(1.0)  # unit spatial tangent at the origin
    x = manifold.expmap_0(jnp.asarray(t, dtype=dtype) * v_hat, c)
    x_s = np.asarray(x)[1:]
    y_s = np.ldexp(x_s, log2_ratio).astype(dtype)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y_s)), f"radial pair not representable at t={t}"
    y_s_64 = np.asarray(y_s, dtype=np.float64)
    y_time = np.sqrt(1.0 / c + float(np.dot(y_s_64, y_s_64)))
    y = jnp.asarray(np.concatenate([[y_time], y_s_64]).astype(dtype))
    expected = (_stored_radius(y_s, c) - _stored_radius(x_s, c)) / np.sqrt(c)
    return x, y, expected


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyperboloid_dist_stable_exact_along_radial_geodesic(dtype: jnp.dtype) -> None:
    """dist along a shared radial geodesic is exactly the radius difference — a closed form the
    Minkowski-inner cancellation bug corrupted past radius ~10 (f32) / ~20 (f64) under the old
    acosh arm, now VERSION_LEGACY.

    ``x = expmap_0(t·v_hat)`` and ``y``, its spatial part rescaled by an exact factor 2, lie on the
    same geodesic ray through the origin, so ``d(x, y)`` is the difference of their radii — see
    :func:`_radial_pair_exact_in_dtype` for why that rescale makes the pair collinear *in the
    dtype*. VERSION_LEGACY is checked to be off by more than 10x the dtype tolerance at the largest
    radius: a regression guard that fails loudly if the switch wiring is ever reverted so
    VERSION_DEFAULT points at the legacy arm.

    Both dtypes now run their full radius list. The old version built ``y = expmap_0(t2·v_hat)`` as
    a second independent rounding and so could only assert collinearity where the two normalizations
    happened to agree: measured over 1300 (radius x curvature x dim) cells, that construction
    resolves the pair in 87.3 % of cells on an A100 in float32 — the one-hot ``v_hat`` hid the
    problem on CPU, where ``x_s/‖x_s‖`` is a correctly rounded ``s/s = 1``, but XLA:GPU's float32
    divide is only faithfully rounded and returns ``s/s != 1`` for 15.3 % of values. With the exact
    rescale it is 100 % of cells on CPU and on the A100 in both dtypes
    (``logs/2026-09-03_collinearity_tests/``). The float32 list stops at ``t = 43`` because
    ``sum(x_s²)`` inside ``expmap_0`` overflows float32 past spatial radius 1.8e19, which is a
    construction limit of the exponential map, not of ``dist``.
    """
    c = 1.0
    manifold = hj.manifolds.Hyperboloid(dtype=dtype)
    dim = 5

    if dtype == jnp.float32:
        atol, rtol = 4e-3, 4e-3
        radii = [9.0, 20.0, 30.0, 43.0]
    else:
        atol, rtol = 1e-7, 1e-7
        radii = [19.0, 40.0, 55.0]

    for t in radii:
        x, y, expected = _radial_pair_exact_in_dtype(manifold, t, 1, c, dim, dtype)
        d_default = manifold.dist(x, y, c)
        assert jnp.allclose(d_default, expected, atol=atol, rtol=rtol), f"t={t}: {d_default} vs {expected}"

    # Regression guard at the largest, most cancellation-prone radius: the legacy arm must be
    # visibly wrong, or this test would silently stop catching a reverted switch.
    x, y, expected = _radial_pair_exact_in_dtype(manifold, radii[-1], 1, c, dim, dtype)
    d_legacy = manifold.dist(x, y, c, version_idx=hj.manifolds.Hyperboloid.VERSION_LEGACY)
    assert abs(float(d_legacy) - expected) > 10.0 * atol


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyperboloid_logmap_finite_and_norm_consistent_at_large_radius(dtype: jnp.dtype) -> None:
    """logmap stays finite and its tangent_norm matches dist at large radius.

    Same radial-geodesic construction as the dist test above, kept within the tangent_norm
    validity WS-A measured (exact to radius ~25 f64 / ~15 f32): the old ``_minkowski_inner``
    route NaN'd logmap from radius ~10 (f32) / ~20 (f64) and floored tangent_norm to ~0 past
    radius 8 (f32) / 20 (f64) on exactly-unit tangent vectors.
    """
    c = 1.0
    manifold = hj.manifolds.Hyperboloid(dtype=dtype)
    dim = 5
    v_hat = jnp.zeros(dim + 1, dtype=dtype).at[1].set(1.0)

    if dtype == jnp.float32:
        atol, rtol = 4e-3, 4e-3
        t1, t2 = 9.0, 10.0
    else:
        atol, rtol = 1e-7, 1e-7
        t1, t2 = 19.0, 20.0

    x = manifold.expmap_0(jnp.asarray(t1, dtype=dtype) * v_hat, c)
    y = manifold.expmap_0(jnp.asarray(t2, dtype=dtype) * v_hat, c)

    v = manifold.logmap(y, x, c)
    assert jnp.all(jnp.isfinite(v))

    d = manifold.dist(x, y, c)
    n = manifold.tangent_norm(v, x, c)
    assert jnp.allclose(n, d, atol=atol, rtol=rtol)


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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_hyperboloid_logmap_0_of_an_infinite_point_is_infinite_not_nan(dtype) -> None:
    """An ``inf`` spatial entry must not turn the whole tangent vector into NaN.

    ``safe_norm`` deliberately passes an ``inf`` through as ``inf`` so an out-of-range point stays
    visibly degenerate; the scale ``arcsinh(u)/u`` then evaluated ``inf/inf`` and NaN-poisoned every
    entry, the time slot included. The convention for an out-of-range input elsewhere in the module
    (``dist``/``_polar_frame``'s ``isfinite`` guard) is ``±inf``, so this asserts ``±inf`` on the
    infinite entries, their sign preserved, and a time slot that is still exactly 0.
    """
    manifold, c = hj.manifolds.Hyperboloid(dtype=dtype), 1.0
    inf = jnp.asarray(jnp.inf, dtype=dtype)

    # x₀ = inf is what the sheet constraint gives for an infinite spatial part; logmap_0 ignores it.
    y_A = jnp.asarray([inf, inf, -inf, 0.3, -0.5], dtype=dtype)
    v_A = manifold.logmap_0(y_A, c)

    assert not jnp.any(jnp.isnan(v_A)), f"logmap_0 must not return NaN for an infinite point: {v_A}"
    assert v_A[0] == 0.0, "log_0 is tangent at the origin, so its time slot stays exactly 0"
    assert v_A[1] == jnp.inf
    assert v_A[2] == -jnp.inf
    assert jnp.all(jnp.isfinite(v_A[3:])), "the finite spatial entries stay finite"

    # A finite point is unaffected: the guard fires only on a non-finite norm.
    y_finite_A = manifold.expmap_0(jnp.asarray([0.0, 0.4, -0.2, 0.1, 0.3], dtype=dtype), c)
    assert jnp.all(jnp.isfinite(manifold.logmap_0(y_finite_A, c)))


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


# ---------------------------------------------------------------------------------------------
# Poincaré expmap_0: the boundary clamp moved from the (dim,) result onto the scalar coefficient
# ---------------------------------------------------------------------------------------------
#
# ``_expmap_0`` used to compute ``tanh(√c‖v‖)/(√c‖v‖) · v`` and then call ``_proj`` on the result.
# ``_proj`` re-reduces ‖·‖ over the op's own output, which under ``jit(vmap)`` at B = 1e7 costs a
# whole extra XLA reduction kernel over (B, dim)-sized data. The current implementation caps the
# scalar coefficient instead, which is mathematically the same clamp. The three tests below pin
# (a) agreement with the old formula, including on the rows the old ``_proj`` actually clamped,
# (b) the postcondition ``‖exp_0(v)‖ ≤ max_norm`` the removed ``_proj`` used to provide, and
# (c) the gradient behaviour (finite at v = 0, Jacobian unchanged away from the clamp).

_EXPMAP_0_DIM = 32
_EXPMAP_0_NORMS = (1e-3, 1e-2, 0.1, 1.0, 5.0, 8.0, 10.0, 20.0)


def _expmap_0_reference(v_D: jnp.ndarray, c: float) -> jnp.ndarray:
    """The pre-change ``poincare._expmap_0``, written out verbatim as the reference.

    Raw ``jnp.tanh`` (not the clamped ``math_utils`` wrapper) followed by the ``_proj`` boundary
    clamp on the ``(dim,)`` result — deliberately independent of the implementation under test.
    """
    v_norm = jnp.sqrt(jnp.sum(v_D**2) + poincare_impl.MIN_NORM**2)
    c_norm_prod = jnp.sqrt(c) * v_norm
    return poincare_impl._proj(jnp.tanh(c_norm_prod) / c_norm_prod * v_D, c)


def _expmap_0_directions(dtype: jnp.dtype, seed: int, n: int = 64) -> jnp.ndarray:
    """``n`` random unit directions in ``_EXPMAP_0_DIM`` dimensions."""
    dirs_ND = np.random.default_rng(seed).normal(size=(n, _EXPMAP_0_DIM))
    dirs_ND /= np.linalg.norm(dirs_ND, axis=1, keepdims=True)
    return jnp.asarray(dirs_ND, dtype=dtype)


def _expmap_0_max_norm(dtype: jnp.dtype, c: float) -> float:
    """``_proj``'s boundary, recomputed from the documented formula rather than from the helper."""
    return 1.0 / np.sqrt(c) - float(jnp.finfo(dtype).eps ** 0.75)


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_poincare_expmap_0_matches_the_projected_tanh_formula(dtype: jnp.dtype, c: float) -> None:
    """Scalar-clamp ``expmap_0`` reproduces ``_proj(tanh(√c‖v‖)/(√c‖v‖) · v)`` to a few ulps.

    The ``‖v‖ ∈ {8, 10, 20}`` rows are the point of the test: float32 ``jnp.tanh`` saturates to
    exactly 1.0 at ``√c‖v‖ ≈ 8``, so those are precisely the rows the old ``_proj`` clamped. The
    test asserts that at least one row on the grid was clamped, so a rewrite that stopped
    clamping altogether could not pass by making both sides trivially equal.

    Tolerance is stated in ulps because the two paths differ only in the rounding of a handful of
    scalar operations (and, off the clamp, in ``math_utils.tanh``'s ``expm1`` rewrite of the raw
    ``jnp.tanh`` the reference uses). Worst row-relative move measured on this grid, jax 0.9.1:
    3.05e-7 = 2.6 ulps at ``c = 2.5`` in float32 and 6.0e-16 = 2.7 ulps in float64 on CPU, lower
    on the A100; the bounds below are 4e-7 (3.4 ulps) and 1e-15 (4.5 ulps). Dropping the clamp
    would move the affected rows by ``eps**0.75`` ≈ 6.4e-6 relative in float32 — 16x the bound —
    so this still bites.
    """
    atol_rel = 4e-7 if dtype == jnp.float32 else 1e-15
    dirs_ND = _expmap_0_directions(dtype, seed=424242)
    max_norm = _expmap_0_max_norm(dtype, c)

    new_fn = jax.vmap(poincare_impl._expmap_0, in_axes=(0, None))
    old_fn = jax.vmap(_expmap_0_reference, in_axes=(0, None))

    clamped_rows = 0
    for norm in _EXPMAP_0_NORMS:
        v_ND = dirs_ND * jnp.asarray(norm, dtype=dtype)
        new_ND = np.asarray(new_fn(v_ND, c), dtype=np.float64)
        old_ND = np.asarray(old_fn(v_ND, c), dtype=np.float64)

        relative = np.linalg.norm(new_ND - old_ND, axis=1) / np.linalg.norm(old_ND, axis=1)
        assert relative.max() <= atol_rel, f"‖v‖={norm}, c={c}: max relative move {relative.max():.3e}"

        # A row sitting on the boundary is a row the old path had to clamp.
        clamped_rows += int(np.sum(np.linalg.norm(old_ND, axis=1) >= max_norm * (1.0 - 1e-6)))

    assert clamped_rows > 0, "grid never reached the boundary — the clamp was never exercised"


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_poincare_expmap_0_stays_inside_the_projection_boundary(dtype: jnp.dtype, c: float) -> None:
    """Every output row satisfies ``‖exp_0(v)‖ ≤ max_norm``, the postcondition ``_proj`` provided.

    ``max_norm = 1/√c - eps**0.75`` is recomputed here from the documented formula, not read back
    from ``_gyrovector_core``. The ``4·eps`` slack covers the rounding of the scalar coefficient;
    measured worst overshoot is 1.4 ulps (float32) / 2.3 ulps (float64) of ``max_norm``.
    """
    dirs_ND = _expmap_0_directions(dtype, seed=424242)
    max_norm = _expmap_0_max_norm(dtype, c)
    bound = max_norm * (1.0 + 4.0 * float(jnp.finfo(dtype).eps))
    new_fn = jax.vmap(poincare_impl._expmap_0, in_axes=(0, None))

    for norm in _EXPMAP_0_NORMS:
        v_ND = dirs_ND * jnp.asarray(norm, dtype=dtype)
        out_norms = np.linalg.norm(np.asarray(new_fn(v_ND, c), dtype=np.float64), axis=1)
        assert out_norms.max() <= bound, f"‖v‖={norm}, c={c}: {out_norms.max():.17g} > {bound:.17g}"


def test_poincare_expmap_0_gradients_survive_the_scalar_clamp(dtype: jnp.dtype) -> None:
    """``grad`` is finite at ``v = 0`` and the Jacobian is unchanged away from the boundary.

    The ``sqrt(‖v‖² + MIN_NORM²)`` safe norm is what keeps the gradient at the origin finite
    (``jnp.linalg.norm``'s VJP there is 0/0); the clamp must not disturb it. Away from the clamp
    the two coefficients differ only in ``tanh``'s own rounding, so the Jacobians agree far inside
    the stated tolerances: measured max absolute difference 0.0 (CPU, both dtypes) and 8.9e-8
    (float32) / 1.1e-16 (float64) on the A100.
    """
    c = 1.0
    atol = 1e-5 if dtype == jnp.float32 else 1e-12

    grad_D = jax.grad(lambda v: jnp.sum(poincare_impl._expmap_0(v, c)))(jnp.zeros((_EXPMAP_0_DIM,), dtype=dtype))
    assert bool(jnp.all(jnp.isfinite(grad_D)))

    # ‖v‖ ≤ 2 keeps tanh(√c‖v‖) well below saturation, i.e. strictly inside the clamp.
    dirs_ND = _expmap_0_directions(dtype, seed=7, n=4)
    for norm in (1e-3, 0.1, 1.0, 2.0):
        for direction_D in dirs_ND:
            v_D = direction_D * jnp.asarray(norm, dtype=dtype)
            jac_new_DD = jax.jacfwd(lambda u: poincare_impl._expmap_0(u, c))(v_D)
            jac_old_DD = jax.jacfwd(lambda u: _expmap_0_reference(u, c))(v_D)
            assert jnp.allclose(jac_new_DD, jac_old_DD, atol=atol, rtol=0.0)


# ---------------------------------------------------------------------------------------------
# Möbius addition: the boundary clamp moved from the (dim,) result onto an analytic scalar norm
# ---------------------------------------------------------------------------------------------
#
# ``_addition`` used to build ``num = A·x + B·y`` and call ``_proj(num/denom, c)``, which reduces
# ‖·‖ over the op's own (dim,) output — under ``jit(vmap)`` that costs XLA a whole extra kernel
# plus a round trip through (B, dim)-sized memory. The current implementation regroups the
# numerator as ``B·(x+y) + c‖x+y‖²·x`` (identical in exact arithmetic, since ``A - B = c‖x+y‖²``)
# and expands ‖num‖² in those same two coefficients, so the clamp becomes a scalar comparison
# computed entirely from reductions over the *inputs*.
#
# The regrouping is not cosmetic. Near the ball boundary with ``y ≈ -x`` the two pieces of
# ``A·x + B·y`` cancel to ``O(ε²)`` from ``O(ε)`` each (``ε := 1 - c‖x‖²``), so any norm derived
# analytically from ``A`` and ``B`` disagrees with the floating-point vector by ``eps/ε`` and the
# clamp can scale a row to sit *outside* the ball. ``B·(x+y)`` and ``c‖x+y‖²·x`` are individually
# the same order as their sum, which is what makes the analytic norm usable at all. Measured on
# 1.35M adversarial rows per dtype (logs/2026-09-03_b2_mobius_norm/probe_mobius_norm_cpu.json):
# the ``alpha/beta`` and naive three-dot forms overshoot the ceiling by up to 6.5e9 and 3.5e5 ulps
# respectively, this one by at most 10.
#
# The tests below pin (a) agreement with the old formula on well-conditioned inputs, including
# rows the old ``_proj`` actually clamped, (b) the postcondition ``‖x ⊕ y‖ ≤ max_norm`` that the
# removed ``_proj`` provided, held across the adversarial near-boundary antipodal grid, (c) that
# accuracy against a float64 reference *improved* rather than regressed, (d) the exact-antipode
# and coincident-point gradient behaviour, and (e) that an unclamped row is left untouched.

_MOBIUS_DIM = 32
_MOBIUS_CS = (0.3, 1.0, 2.5)
# ‖n‖/(ε/√c) for y = -x + n. The exact output norm passes through the ball ceiling around 1, so
# the rows near 1 are the ones where the clamp decision is both hardest and worst conditioned;
# they are kept in every finiteness/ball assertion and excluded only from the value comparison.
_MOBIUS_ETA_RATIOS = (0.01, 0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 10.0)
_MOBIUS_ILL_CONDITIONED = (0.9, 0.99, 1.0, 1.01, 1.1)


def _addition_reference(x_D: jnp.ndarray, y_D: jnp.ndarray, c: float) -> jnp.ndarray:
    """The pre-change ``_gyrovector_core._addition``, written out verbatim as the reference.

    ``num = A·x + B·y`` followed by the ``_proj`` boundary clamp on the ``(dim,)`` result —
    deliberately independent of the implementation under test.
    """
    x2 = jnp.dot(x_D, x_D)
    y2 = jnp.dot(y_D, y_D)
    xy = jnp.dot(x_D, y_D)
    num_D = (1 + 2 * c * xy + c * y2) * x_D + (1 - c * x2) * y_D
    denom = jnp.maximum(1 + 2 * c * xy + c**2 * x2 * y2, poincare_impl.MIN_NORM)
    return poincare_impl._proj(num_D / denom, c)


def _mobius_max_norm(dtype: jnp.dtype, c: float) -> float:
    """``_proj``'s ceiling, recomputed from the documented formula rather than from the helper."""
    return 1.0 / np.sqrt(c) - float(jnp.finfo(dtype).eps) ** 0.75


def _mobius_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v_ND = rng.normal(size=(n, _MOBIUS_DIM))
    return v_ND / np.linalg.norm(v_ND, axis=1, keepdims=True)


def _mobius_random_pairs(dtype: jnp.dtype, c: float, seed: int, n: int = 512):
    """Independent x, y uniform in radius on [0, ceiling] — the well-conditioned bulk."""
    rng = np.random.default_rng(seed)
    mx = _mobius_max_norm(dtype, c)
    x_ND = _mobius_unit(rng, n) * rng.uniform(0.0, mx, size=(n, 1))
    y_ND = _mobius_unit(rng, n) * rng.uniform(0.0, mx, size=(n, 1))
    return jnp.asarray(x_ND, dtype=dtype), jnp.asarray(y_ND, dtype=dtype)


def _mobius_boundary_pairs(dtype: jnp.dtype, c: float, seed: int, n: int = 512):
    """Both points on the ceiling and nearly parallel — every row of this grid gets clamped.

    For ``x = y = m·u`` the exact result has norm ``2m/(1 + c·m²)``, which for ``m`` at the
    ceiling ``1/√c - eps**0.75`` evaluates to ``1/√c`` — above ``max_norm``. Tilting ``y`` away
    from ``x`` by up to 0.05 rad keeps the rows on the clamped side while varying the direction.
    """
    rng = np.random.default_rng(seed)
    mx = _mobius_max_norm(dtype, c)
    u_ND = _mobius_unit(rng, n)
    w_ND = _mobius_unit(rng, n)
    tilt_N1 = np.linspace(0.0, 0.05, n)[:, None]
    v_ND = u_ND + tilt_N1 * (w_ND - np.sum(w_ND * u_ND, axis=1, keepdims=True) * u_ND)
    v_ND /= np.linalg.norm(v_ND, axis=1, keepdims=True)
    return jnp.asarray(u_ND * mx, dtype=dtype), jnp.asarray(v_ND * mx, dtype=dtype)


def _mobius_antipodal_pairs(dtype: jnp.dtype, c: float, eps_val: float, ratio: float, kind: str, seed: int, n: int = 256):
    """``x`` at ``1 - c‖x‖² = eps_val``, ``y = -x + n`` with ``‖n‖ = ratio·eps_val/√c``.

    ``kind`` picks the direction of ``n``: random, orthogonal to ``x``, or collinear with ``±x``
    (collinear is the worst case for this numerator grouping — it is the only configuration in
    which its two terms can cancel). A ``y`` that lands outside the ceiling is projected back onto
    it; a point outside the ball is not a legal input to ``⊕``.
    """
    rng = np.random.default_rng(seed)
    mx = _mobius_max_norm(dtype, c)
    xhat_ND = _mobius_unit(rng, n)
    x_ND = xhat_ND * np.sqrt((1.0 - eps_val) / c)
    if kind == "rand":
        nhat_ND = _mobius_unit(rng, n)
    elif kind == "orth":
        g_ND = _mobius_unit(rng, n)
        g_ND = g_ND - np.sum(g_ND * xhat_ND, axis=1, keepdims=True) * xhat_ND
        nhat_ND = g_ND / np.linalg.norm(g_ND, axis=1, keepdims=True)
    else:
        nhat_ND = rng.choice([-1.0, 1.0], size=(n, 1)) * xhat_ND
    y_ND = -x_ND + nhat_ND * (ratio * eps_val / np.sqrt(c))
    ynorm_N1 = np.linalg.norm(y_ND, axis=1, keepdims=True)
    y_ND = np.where(ynorm_N1 > mx, y_ND * (mx / np.maximum(ynorm_N1, 1e-300)), y_ND)
    return jnp.asarray(x_ND, dtype=dtype), jnp.asarray(y_ND, dtype=dtype)


def _mobius_eps_grid(dtype: jnp.dtype, c: float) -> tuple[float, ...]:
    """``ε`` values down to the smallest the ceiling admits: ``1 - c·max_norm²``."""
    e_mach = float(jnp.finfo(dtype).eps)
    margin = 2.0 * np.sqrt(c) * e_mach**0.75 - c * e_mach**1.5
    return (1e-2, 1e-3, 1e-4, float(margin))


def _mobius_ref_f64(x_ND: jnp.ndarray, y_ND: jnp.ndarray, c: float, max_norm: float) -> np.ndarray:
    """float64 reference, clamped at the ceiling of the dtype under test."""
    x = np.asarray(x_ND, dtype=np.float64)
    y = np.asarray(y_ND, dtype=np.float64)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True)
    xy = np.sum(x * y, axis=1, keepdims=True)
    s = x + y
    s2 = np.sum(s * s, axis=1, keepdims=True)
    denom = np.maximum(1 + 2 * c * xy + c**2 * x2 * y2, poincare_impl.MIN_NORM)
    out = ((1.0 - c * x2) * s + (c * s2) * x) / denom
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(nrm > max_norm, out * (max_norm / np.maximum(nrm, 1e-300)), out)


_mobius_add_batch = jax.vmap(poincare_impl._addition, in_axes=(0, 0, None))
_mobius_ref_batch = jax.vmap(_addition_reference, in_axes=(0, 0, None))


@pytest.mark.parametrize("c", _MOBIUS_CS)
def test_poincare_mobius_add_matches_the_projected_reference(dtype: jnp.dtype, c: float) -> None:
    """The regrouped numerator + scalar clamp reproduces ``_proj(A·x + B·y)`` to a few ulps.

    Two grids: independent random pairs (no row clamped) and pairs on the ceiling (every row
    clamped — the second grid is the point of the test, since a rewrite that stopped clamping
    would otherwise pass trivially). Differences are measured against ``max_norm`` rather than
    against ‖out‖ because whole input families here have an exact answer of 0.

    Worst measured move on these grids, jax 0.9.1 CPU: 3.4e-7 (float32) / 6.2e-16 (float64) of
    the ball radius, i.e. ~3 ulps; the bounds below are 12x and 16x that.
    """
    bound = 4e-6 if dtype == jnp.float32 else 1e-14
    max_norm = _mobius_max_norm(dtype, c)
    clamped_rows = 0
    for grid in (_mobius_random_pairs, _mobius_boundary_pairs):
        x_ND, y_ND = grid(dtype, c, seed=4242)
        new_ND = np.asarray(_mobius_add_batch(x_ND, y_ND, c), dtype=np.float64)
        old_ND = np.asarray(_mobius_ref_batch(x_ND, y_ND, c), dtype=np.float64)
        moved = np.linalg.norm(new_ND - old_ND, axis=1) / max_norm
        assert moved.max() <= bound, f"c={c}: max move {moved.max():.3e} of the ball radius"
        clamped_rows += int(np.sum(np.linalg.norm(old_ND, axis=1) >= max_norm * (1.0 - 1e-6)))

    assert clamped_rows > 0, "no row reached the boundary — the clamp was never exercised"


@pytest.mark.parametrize("c", _MOBIUS_CS)
def test_poincare_mobius_add_stays_inside_the_projection_boundary(dtype: jnp.dtype, c: float) -> None:
    """Every row satisfies ``‖x ⊕ y‖ ≤ max_norm``, the postcondition ``_proj`` used to provide.

    This is the assertion the change could plausibly break: the clamp is now decided on an
    analytic norm and applied to ``num/denom`` as it was really computed, so the two have to agree
    for the bound to hold. The adversarial near-boundary antipodal grid is included precisely
    because that is where an analytic norm can drift away from its own vector.

    The ``16·eps`` slack covers the rounding of ``sqrt``, the division by ``denom`` and the final
    scaling; measured worst overshoot is 10 ulps of ``max_norm`` over 1.35M rows per dtype. Even
    at 16 ulps the result stays strictly inside the true radius ``1/√c``, since the ceiling sits
    ``eps**0.75`` below it — a margin of ``eps**-0.25`` = 54 ulps (float32) / 26000 (float64).
    """
    max_norm = _mobius_max_norm(dtype, c)
    bound = max_norm * (1.0 + 16.0 * float(jnp.finfo(dtype).eps))

    grids = [_mobius_random_pairs(dtype, c, seed=4242), _mobius_boundary_pairs(dtype, c, seed=4242)]
    for eps_val in _mobius_eps_grid(dtype, c):
        for ratio in _MOBIUS_ETA_RATIOS:
            for kind in ("rand", "orth", "collin"):
                grids.append(_mobius_antipodal_pairs(dtype, c, eps_val, ratio, kind, seed=99))

    for x_ND, y_ND in grids:
        out_ND = np.asarray(_mobius_add_batch(x_ND, y_ND, c), dtype=np.float64)
        assert np.all(np.isfinite(out_ND))
        norms = np.linalg.norm(out_ND, axis=1)
        assert norms.max() <= bound, f"c={c}: {norms.max():.17g} > {bound:.17g}"


@pytest.mark.parametrize("c", _MOBIUS_CS)
def test_poincare_mobius_add_is_more_accurate_near_the_boundary(dtype: jnp.dtype, c: float) -> None:
    """Against a float64 reference the new grouping is strictly better than the old one.

    An absolute tolerance would say little here: at ``ε = 1e-4`` in float32 *both* forms are off
    by a whole ball radius, because ``A`` and ``B`` are ``O(ε)`` differences of ``O(1)`` terms and
    float32 has no bits left. What the change buys is a factor ~2 at every ``ε`` — and, once
    ``ε`` reaches the ceiling margin in float64, a factor of ~1e7, since there the old form's
    numerator is pure rounding noise while ``B·(x+y) + c‖x+y‖²·x`` still resolves it.

    The ``‖n‖/(εR) ∈ [0.9, 1.1]`` rows are excluded from this comparison only: that is where the
    exact result crosses the ball ceiling, so one form may clamp and the other not, and the
    difference then says nothing about accuracy. They stay in the finiteness and ball assertions.
    Measured ratio new/old on this grid: 0.48-0.60 everywhere.
    """
    max_norm = _mobius_max_norm(dtype, c)
    for eps_val in _mobius_eps_grid(dtype, c):
        worst_new, worst_old = 0.0, 0.0
        for ratio in _MOBIUS_ETA_RATIOS:
            if ratio in _MOBIUS_ILL_CONDITIONED:
                continue
            for kind in ("rand", "orth", "collin"):
                x_ND, y_ND = _mobius_antipodal_pairs(dtype, c, eps_val, ratio, kind, seed=99)
                ref_ND = _mobius_ref_f64(x_ND, y_ND, c, max_norm)
                new_ND = np.asarray(_mobius_add_batch(x_ND, y_ND, c), dtype=np.float64)
                old_ND = np.asarray(_mobius_ref_batch(x_ND, y_ND, c), dtype=np.float64)
                worst_new = max(worst_new, float(np.max(np.linalg.norm(new_ND - ref_ND, axis=1))) / max_norm)
                worst_old = max(worst_old, float(np.max(np.linalg.norm(old_ND - ref_ND, axis=1))) / max_norm)
        assert worst_new <= 0.75 * worst_old + 1e-15, f"c={c}, eps={eps_val:.2e}: new {worst_new:.3e} vs old {worst_old:.3e}"

    # Absolute bounds where the input is still well enough conditioned for one to mean something.
    # Measured: 2.6e-2 (float32, eps=1e-2) and 4.9e-7 (float64, eps=1e-4).
    abs_bound, abs_eps = (5e-2, 1e-2) if dtype == jnp.float32 else (1e-6, 1e-4)
    worst = 0.0
    for ratio in _MOBIUS_ETA_RATIOS:
        if ratio in _MOBIUS_ILL_CONDITIONED:
            continue
        for kind in ("rand", "orth", "collin"):
            x_ND, y_ND = _mobius_antipodal_pairs(dtype, c, abs_eps, ratio, kind, seed=99)
            ref_ND = _mobius_ref_f64(x_ND, y_ND, c, max_norm)
            new_ND = np.asarray(_mobius_add_batch(x_ND, y_ND, c), dtype=np.float64)
            worst = max(worst, float(np.max(np.linalg.norm(new_ND - ref_ND, axis=1))) / max_norm)
    assert worst <= abs_bound, f"c={c}: {worst:.3e} > {abs_bound:.3e}"


@pytest.mark.parametrize("c", _MOBIUS_CS)
def test_poincare_mobius_add_gradients_survive_the_analytic_norm(dtype: jnp.dtype, c: float) -> None:
    """``x ⊕ (-x)`` is exactly 0 and no configuration produces a NaN gradient.

    ``‖num‖²`` is now built from an explicit sum of three terms, so ``sqrt`` sees an exact 0
    whenever ``y = -x`` — and ``sqrt`` has an infinite derivative there. The ``degenerate`` guard
    is a ``where`` placed *before* the ``sqrt`` for exactly that reason; this test is what would
    catch it being moved after. The gradients of ``dist`` and ``logmap`` are checked too, since
    those are the two callers that feed ``(-x) ⊕ y`` with nearby near-boundary points.

    The old form did *not* give an exact zero here — ``A`` and ``B`` are rounded independently, so
    ``A·x + B·y`` left a residue of up to 1.3e-3 of the ball radius in float32.
    """
    x_ND, _ = _mobius_random_pairs(dtype, c, seed=7)
    zero_D = jnp.zeros((_MOBIUS_DIM,), dtype=dtype)

    assert float(jnp.max(jnp.abs(_mobius_add_batch(x_ND, -x_ND, c)))) == 0.0

    for name, fn in (
        ("antipode", lambda a: jnp.sum(_mobius_add_batch(a, -a, c))),
        ("coincident", lambda a: jnp.sum(_mobius_add_batch(a, a, c))),
        ("dist", lambda a: jnp.sum(jax.vmap(poincare_impl._dist_mobius, in_axes=(0, 0, None))(a, a, c))),
        ("logmap", lambda a: jnp.sum(jax.vmap(poincare_impl._logmap, in_axes=(0, 0, None))(a, a, c))),
    ):
        assert jnp.all(jnp.isfinite(jax.grad(fn)(x_ND))), f"NaN grad: {name}, c={c}, {dtype}"

    # The origin: x = y = 0 makes every reduction, and therefore `terms`, exactly 0.
    assert jnp.all(jnp.isfinite(jax.grad(lambda a: jnp.sum(poincare_impl._addition(a, -a, c)))(zero_D)))
    assert jnp.all(jnp.isfinite(jax.grad(lambda a: jnp.sum(poincare_impl._addition(a, zero_D, c)))(zero_D)))


@pytest.mark.parametrize("c", _MOBIUS_CS)
def test_poincare_mobius_add_leaves_unclamped_rows_untouched(dtype: jnp.dtype, c: float) -> None:
    """A row that is not clamped comes back as exactly ``num / denom`` — the scale factor is 1.0.

    ``scale`` multiplies the whole result, so if the unclamped branch were ``max_norm/norm`` with
    ``norm`` a hair below ``max_norm`` instead of a literal 1.0, every row in the library would
    pick up a silent relative shift. Multiplying by an exact 1.0 is exact, so the assertion is
    bit-equality against the documented numerator.

    CPU only: on GPU the five reductions are free to associate differently between this test's
    expression and the compiled library kernel, and a 1-ulp difference there is not a defect.
    """
    if jax.default_backend() != "cpu":
        pytest.skip("bit-equality across two separately compiled reduction trees is CPU-only")

    def unclamped(x_D, y_D):
        """``B·(x+y) + c‖x+y‖²·x`` over ``denom`` — the documented numerator, no clamp."""
        x2 = jnp.dot(x_D, x_D)
        y2 = jnp.dot(y_D, y_D)
        xy = jnp.dot(x_D, y_D)
        s_D = x_D + y_D
        s2 = jnp.dot(s_D, s_D)
        num_D = (1 - c * x2) * s_D + (c * s2) * x_D
        return num_D / jnp.maximum(1 + 2 * c * xy + c**2 * x2 * y2, poincare_impl.MIN_NORM)

    max_norm = _mobius_max_norm(dtype, c)
    x_ND, y_ND = _mobius_random_pairs(dtype, c, seed=4242)
    out_ND = _mobius_add_batch(x_ND, y_ND, c)
    raw_ND = jax.vmap(unclamped)(x_ND, y_ND)

    keep = np.linalg.norm(np.asarray(raw_ND, dtype=np.float64), axis=1) < max_norm * (1.0 - 1e-4)
    assert int(np.sum(keep)) > 0, "no unclamped rows on this grid"
    assert jnp.array_equal(out_ND[keep], raw_ND[keep])


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


# ---------------------------------------------------------------------------
# Constraint-tolerance convention (audit B9 / D3)
#
# One convention across all five manifolds: ``atol=None`` resolves through
# ``manifolds._base.default_atol`` (= sqrt of the dtype's machine epsilon) and an explicit
# ``atol`` is honoured as given. Historically the three constrained manifolds disagreed —
# Hyperboloid floored it at 1e-4 (so ``atol=1e-9`` and ``atol=1e-4`` were indistinguishable),
# Poincaré documented it as "not used", ProperVelocity ``del``-ed it.
# ---------------------------------------------------------------------------


def test_default_atol_is_sqrt_eps_and_dtype_aware() -> None:
    """``default_atol`` is ``sqrt(finfo(dtype).eps)`` — looser in float32 than in float64."""
    atol_f32 = default_atol(jnp.float32)
    atol_f64 = default_atol(jnp.float64)

    assert atol_f32 == pytest.approx(float(np.finfo(np.float32).eps) ** 0.5, rel=1e-12)
    assert atol_f64 == pytest.approx(float(np.finfo(np.float64).eps) ** 0.5, rel=1e-12)
    assert atol_f64 < atol_f32, "a float64 check must be strictly tighter than a float32 one"


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_hyperboloid_is_in_manifold_honours_an_explicit_atol(c: float) -> None:
    """A point off the sheet by exactly ``R`` is accepted iff ``atol > R``.

    Constructed so the Lorentz-norm residual is *exact*: replacing ``x₀`` by ``sqrt(x₀² - R)``
    makes ``⟨x, x⟩_L = -1/c + R`` identically. float64 throughout so ``R = 1e-6`` is far above
    the arithmetic noise.

    This is the test the old implementation fails: ``_is_in_manifold`` opened with
    ``tol = max(atol, 1e-4)``, so the ``atol=1e-7`` call below accepted the point (1e-6 < 1e-4)
    and no caller could ever tighten the check. It also pins that the floor's *removal* did not
    turn into "ignore atol entirely" — the loose call must still accept.
    """
    manifold = hj.manifolds.Hyperboloid(dtype=jnp.float64)
    residual = 1e-6

    on_sheet = manifold.proj(jnp.array([0.0, 0.4, -0.7, 0.2], dtype=jnp.float64), c)
    off_sheet = on_sheet.at[0].set(jnp.sqrt(on_sheet[0] ** 2 - residual))

    # The construction is exact: verify the residual before relying on it.
    lorentz = float(manifold.minkowski_inner(off_sheet, off_sheet))
    assert lorentz == pytest.approx(-1.0 / c + residual, abs=1e-12)

    assert bool(manifold.is_in_manifold(off_sheet, c, atol=1e-5)), "atol > residual must accept"
    assert not bool(manifold.is_in_manifold(off_sheet, c, atol=1e-7)), "atol < residual must reject"
    # The genuinely on-sheet point survives a tolerance far below the old 1e-4 floor.
    assert bool(manifold.is_in_manifold(on_sheet, c, atol=1e-12))


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_poincare_is_in_manifold_honours_an_explicit_atol(c: float) -> None:
    """A point outside the ball by exactly ``R`` in ``c‖x‖² - 1`` is accepted iff ``atol > R``.

    ``atol`` used to be documented as "kept for API consistency but not used", so both calls
    below returned False; the check was a hard ``‖x‖² < 1/c`` with no slack at all. The
    dimensionless residual ``c‖x‖² - 1`` is what is toleranced, so the same ``atol`` means the
    same thing at every curvature — hence the three-curvature parametrization.
    """
    manifold = hj.manifolds.Poincare(dtype=jnp.float64)
    residual = 1e-6

    direction = jnp.array([0.4, -0.7, 0.2], dtype=jnp.float64)
    outside = direction / jnp.linalg.norm(direction) * jnp.sqrt((1.0 + residual) / c)

    assert float(c * jnp.dot(outside, outside)) == pytest.approx(1.0 + residual, abs=1e-12)

    assert bool(manifold.is_in_manifold(outside, c, atol=1e-5)), "atol > residual must accept"
    assert not bool(manifold.is_in_manifold(outside, c, atol=1e-7)), "atol < residual must reject"
    # A projected point is strictly inside, so it passes even with zero slack.
    inside = manifold.proj(direction, c)
    assert bool(manifold.is_in_manifold(inside, c, atol=0.0))


def test_is_in_tangent_space_rejects_non_finite_vectors(manifold_and_c, uniform_points: jnp.ndarray, rng) -> None:
    """NaN/Inf tangent vectors are rejected on every manifold.

    ``Euclidean._is_in_tangent_space`` and ``Poincare._is_in_tangent_space`` returned the
    literal constant ``True`` — an assertion-free check that accepted NaN and Inf, the same
    defect commit 68c05e3 fixed for ``Euclidean.is_in_manifold`` alone. The tangent space of an
    open subset of R^n *is* R^n, so finiteness is the whole constraint there; the Hyperboloid
    already rejected them through its ``|⟨v, x⟩_L| < atol`` test.

    A genuine tangent vector is asserted accepted in the same test so "always False" is not a
    passing implementation either.
    """
    manifold, c = manifold_and_c
    point = uniform_points[0]

    v = jnp.asarray(rng.normal(0.0, 1.0, size=point.shape), dtype=point.dtype)
    if _is_hyperboloid(manifold):
        v = manifold.tangent_proj(v, point, c)
    assert bool(manifold.is_in_tangent_space(v, point, c))

    assert not bool(manifold.is_in_tangent_space(v.at[0].set(jnp.nan), point, c))
    assert not bool(manifold.is_in_tangent_space(v.at[0].set(jnp.inf), point, c))


def test_poincare_proj_batch_matches_the_vmapped_single_point_proj(poincare_and_c, poincare_points: jnp.ndarray) -> None:
    """``Poincare.proj_batch`` equals ``vmap(proj)``, and handles extra leading axes.

    Added to close the sibling gap against ``Hyperboloid.proj_batch``; ``decomposition/`` used
    to hand-roll the vmap at two sites for want of it. Equality is asserted to 2 ulp of the
    dtype rather than bit-for-bit: the two calls reduce over differently shaped arrays
    (``(8, 10)`` vs ``(2, 8, 10)``), and XLA:GPU picks its reduce kernel per shape at compile
    time, so the two sums can differ by 1-2 ulp — non-deterministically, since the choice comes
    from autotuning and varies across processes. 2 ulp is far tighter than any real defect: a
    batched rewrite that dropped the ``keepdims`` or clamped by the wrong norm moves the result
    by orders of magnitude, not by an ulp.

    Points outside the ball are included: on already-inside points ``proj`` is the identity, so
    a ``proj_batch`` that returned its input unchanged would pass a test built only from the
    on-manifold fixture.
    """
    manifold, c = poincare_and_c
    dtype = poincare_points.dtype

    outside = poincare_points[:4] * jnp.asarray(50.0, dtype=dtype)  # well past the boundary
    points = jnp.concatenate([poincare_points[:4], outside], axis=0)

    two_ulp = 2.0 * float(jnp.finfo(dtype).eps)

    batched = manifold.proj_batch(points, c)
    looped = jax.vmap(manifold.proj, in_axes=(0, None))(points, c)

    assert batched.shape == points.shape
    assert jnp.allclose(batched, looped, rtol=two_ulp, atol=0)
    assert _batch_is_in_manifold(manifold, batched, c)
    # The clamp actually fired: the far points moved.
    assert not jnp.allclose(batched[4:], points[4:])

    # Arbitrary leading dimensions, matching Hyperboloid.proj_batch's contract.
    stacked = jnp.stack([points, points[::-1]])  # (2, N, dim)
    assert jnp.allclose(manifold.proj_batch(stacked, c)[0], batched, rtol=two_ulp, atol=0)
