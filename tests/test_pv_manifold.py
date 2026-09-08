"""Proper Velocity manifold tests.

Covers the invariants required for a gyrovector-space / Riemannian-manifold
implementation: gyroaddition and scalar multiplication identities, distance
metric properties, exp/log inverse, parallel-transport round-trip, tangent
inner product structure, and numerical stability at large radii.

Uses ``dtype``, ``tolerance``, ``rng``, ``seed_jax`` from ``tests/conftest.py``
and local PV-only fixtures here so the shared ``manifold_and_c`` fixture is
not affected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hyperbolix.manifolds.proper_velocity as pv_impl
from hyperbolix.manifolds import ProperVelocity

# ---------------------------------------------------------------------------
# Fixtures (local to PV tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[0.1, 2.0], ids=lambda c: f"c{c}")
def curvature(request: pytest.FixtureRequest) -> float:
    """Positive curvature parameter c (sectional curvature = -c).

    Two values spanning a factor of 20 on both sides of 1 are enough to catch any missing or
    extra √c factor; the intermediate values only re-ran the same identities.
    """
    return float(request.param)


@pytest.fixture(scope="module")
def pv_manifold(dtype: jnp.dtype) -> ProperVelocity:
    """Fresh PV manifold with the dtype fixture's precision."""
    return ProperVelocity(dtype=dtype)


@pytest.fixture(scope="module", params=[2, 10], ids=lambda d: f"dim{d}")
def dim(request: pytest.FixtureRequest) -> int:
    """Ambient dimension. PV has no dimension-dependent code path, so this is pure
    vectorization width: the minimal case plus one generic value."""
    return int(request.param)


@pytest.fixture(scope="module")
def pv_points(dim: int, dtype: jnp.dtype, seed_jax: int) -> jnp.ndarray:
    """Batch of PV points sampled as Gaussians in R^n. PV is unconstrained, so no projection.

    Uses a generator derived from (seed, stream, dim) rather than the shared ``rng`` fixture so
    the sample does not depend on which other tests ran first.
    """
    num_pts = 256
    np_dtype = np.dtype(dtype.name)
    gen = np.random.default_rng([seed_jax, 20, dim])
    data = gen.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
    return jnp.asarray(data)


@pytest.fixture(scope="module")
def pv_tangent_vectors(dim: int, dtype: jnp.dtype, seed_jax: int) -> jnp.ndarray:
    """Random tangent vectors (any R^n vector is tangent in PV)."""
    num_pts = 256
    np_dtype = np.dtype(dtype.name)
    gen = np.random.default_rng([seed_jax, 21, dim])
    data = gen.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype) * 0.3
    return jnp.asarray(data)


def _split(points: jnp.ndarray, parts: int) -> tuple[jnp.ndarray, ...]:
    """Split into ``parts`` equal-size chunks (truncates remainder for vmap compatibility)."""
    chunk = points.shape[0] // parts
    return tuple(points[i * chunk : (i + 1) * chunk] for i in range(parts))


# ---------------------------------------------------------------------------
# Gyroaddition
# ---------------------------------------------------------------------------


def test_pv_gyroaddition_identity(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """0 ⊕ x = x  and  x ⊕ 0 = x."""
    atol, rtol = tolerance
    add = jax.vmap(pv_manifold.addition, in_axes=(0, 0, None))
    zero = jnp.zeros_like(pv_points)

    left = add(zero, pv_points, curvature)
    assert jnp.allclose(left, pv_points, atol=atol, rtol=rtol)

    right = add(pv_points, zero, curvature)
    assert jnp.allclose(right, pv_points, atol=atol, rtol=rtol)


def test_pv_gyroaddition_inverse(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """(-x) ⊕ x = 0  and  x ⊕ (-x) = 0."""
    atol, rtol = tolerance
    add = jax.vmap(pv_manifold.addition, in_axes=(0, 0, None))
    zero = jnp.zeros_like(pv_points)

    lhs = add(-pv_points, pv_points, curvature)
    assert jnp.allclose(lhs + 1.0, zero + 1.0, atol=atol, rtol=rtol)

    rhs = add(pv_points, -pv_points, curvature)
    assert jnp.allclose(rhs + 1.0, zero + 1.0, atol=atol, rtol=rtol)


def test_pv_gyro_difference_matches_the_ambient_gyroaddition(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """``gyro_difference(x, y) == addition(scalar_mul(-1, x), y)`` on the 256-point fixture.

    ``gyro_difference`` reads ``(⊖x) ⊕ y`` off the hyperboloid polar frame instead of forming the
    ambient Lorentz boost. The two are the same map, which is what this pins: at the fixture's
    radius (Gaussian coordinates, scaled radius ≲ 2) neither form cancels, so any disagreement
    beyond rounding would be a wrong lift, not a conditioning difference.

    Measured worst ``|new - ambient|`` on this grid: 1.8e-14 in float64 and 9.5e-6 in float32
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_new_test_measurements.out``,
    section 2), against the fixture tolerances of 1e-7 and 4e-3. The large-radius separation the
    change is actually for is measured in
    :func:`test_pv_gyro_difference_is_accurate_at_large_radius_in_float32`.
    """
    atol, rtol = tolerance
    gyro_difference = jax.vmap(pv_manifold.gyro_difference, in_axes=(0, 0, None))
    add = jax.vmap(pv_manifold.addition, in_axes=(0, 0, None))
    neg = jax.vmap(pv_manifold.scalar_mul, in_axes=(None, 0, None))

    x, y = _split(pv_points, 2)

    lhs = gyro_difference(x, y, curvature)
    rhs = add(neg(-1.0, x, curvature), y, curvature)
    assert jnp.allclose(lhs, rhs, atol=atol, rtol=rtol)

    # ``(⊖x) ⊕ x`` is the origin: the case the boost reaches by cancelling three O(e^{2a}) terms.
    coincident = gyro_difference(x, x, curvature)
    assert jnp.allclose(coincident + 1.0, jnp.ones_like(coincident), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Scalar multiplication
# ---------------------------------------------------------------------------


def test_pv_scalar_mul_identity_and_zero(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """0 ⊗ x = 0 and 1 ⊗ x = x."""
    atol, rtol = tolerance
    scalar_mul = jax.vmap(pv_manifold.scalar_mul, in_axes=(0, 0, None))

    ones = jnp.ones(pv_points.shape[0], dtype=pv_points.dtype)
    zeros = jnp.zeros_like(ones)

    identity_result = scalar_mul(ones, pv_points, curvature)
    assert jnp.allclose(identity_result, pv_points, atol=atol, rtol=rtol)

    zero_result = scalar_mul(zeros, pv_points, curvature)
    assert jnp.allclose(zero_result + 1.0, jnp.ones_like(pv_points), atol=atol, rtol=rtol)


def test_pv_scalar_mul_associative(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
    rng: np.random.Generator,
) -> None:
    """(r₁·r₂) ⊗ x = r₁ ⊗ (r₂ ⊗ x)."""
    atol, rtol = tolerance
    # Float32 PV at moderate curvature needs a loose rtol because sinh(·asinh(·))
    # compounds rounding error for large |t|.
    if pv_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 2e-2)

    scalar_mul = jax.vmap(pv_manifold.scalar_mul, in_axes=(0, 0, None))

    r1 = jnp.asarray(rng.uniform(-1.5, 1.5, pv_points.shape[0]), dtype=pv_points.dtype)
    r2 = jnp.asarray(rng.uniform(-1.5, 1.5, pv_points.shape[0]), dtype=pv_points.dtype)

    lhs = scalar_mul(r1 * r2, pv_points, curvature)
    rhs = scalar_mul(r1, scalar_mul(r2, pv_points, curvature), curvature)
    assert jnp.allclose(lhs, rhs, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def test_pv_dist_symmetry_and_identity(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """d(x, y) = d(y, x) and d(x, x) = 0."""
    atol, rtol = tolerance
    dist = jax.vmap(pv_manifold.dist, in_axes=(0, 0, None))

    x, y = _split(pv_points, 2)

    d_xy = dist(x, y, curvature)
    d_yx = dist(y, x, curvature)
    assert jnp.allclose(d_xy, d_yx, atol=atol, rtol=rtol)
    assert jnp.all(d_xy >= -atol)

    d_xx = dist(x, x, curvature)
    assert jnp.allclose(d_xx, jnp.zeros_like(d_xx), atol=atol, rtol=rtol)


def test_pv_dist_triangle_inequality(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """d(x, z) ≤ d(x, y) + d(y, z)."""
    atol, _ = tolerance
    dist = jax.vmap(pv_manifold.dist, in_axes=(0, 0, None))

    x, y, z = _split(pv_points, 3)
    d_xz = dist(x, z, curvature)
    d_xy = dist(x, y, curvature)
    d_yz = dist(y, z, curvature)
    assert jnp.all(d_xz <= d_xy + d_yz + 10.0 * atol)


def test_pv_dist_0_matches_dist_to_origin(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """dist_0(x) = dist(x, 0)."""
    atol, rtol = tolerance
    dist = jax.vmap(pv_manifold.dist, in_axes=(0, 0, None))
    dist_0 = jax.vmap(pv_manifold.dist_0, in_axes=(0, None))

    origin = jnp.zeros_like(pv_points)

    d1 = dist_0(pv_points, curvature)
    d2 = dist(pv_points, origin, curvature)
    assert jnp.allclose(d1, d2, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Exponential / logarithmic maps
# ---------------------------------------------------------------------------


def test_pv_expmap_0_logmap_0_inverse(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """exp_0(log_0(x)) = x."""
    atol, rtol = tolerance
    if pv_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 1e-2)
    logmap_0 = jax.vmap(pv_manifold.logmap_0, in_axes=(0, None))
    expmap_0 = jax.vmap(pv_manifold.expmap_0, in_axes=(0, None))

    v = logmap_0(pv_points, curvature)
    x_reconstructed = expmap_0(v, curvature)
    assert jnp.allclose(x_reconstructed, pv_points, atol=atol, rtol=rtol)


def test_pv_logmap_0_expmap_0_inverse(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """log_0(exp_0(v)) = v, for reasonable tangent norms."""
    atol, rtol = tolerance
    if pv_tangent_vectors.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 1e-2)
    logmap_0 = jax.vmap(pv_manifold.logmap_0, in_axes=(0, None))
    expmap_0 = jax.vmap(pv_manifold.expmap_0, in_axes=(0, None))

    y = expmap_0(pv_tangent_vectors, curvature)
    v_reconstructed = logmap_0(y, curvature)
    assert jnp.allclose(v_reconstructed, pv_tangent_vectors, atol=atol, rtol=rtol)


def test_pv_expmap_logmap_inverse(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """log_x(exp_x(v)) = v (away from the origin)."""
    atol, rtol = tolerance
    if pv_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 3e-2)
    expmap = jax.vmap(pv_manifold.expmap, in_axes=(0, 0, None))
    logmap = jax.vmap(pv_manifold.logmap, in_axes=(0, 0, None))

    y = expmap(pv_tangent_vectors, pv_points, curvature)
    v_reconstructed = logmap(y, pv_points, curvature)
    assert jnp.allclose(v_reconstructed, pv_tangent_vectors, atol=atol, rtol=rtol)


def test_pv_expmap_matches_expmap_0_at_origin(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """expmap(v, 0) = expmap_0(v)."""
    atol, rtol = tolerance
    expmap = jax.vmap(pv_manifold.expmap, in_axes=(0, 0, None))
    expmap_0 = jax.vmap(pv_manifold.expmap_0, in_axes=(0, None))

    origin = jnp.zeros_like(pv_tangent_vectors)
    out_general = expmap(pv_tangent_vectors, origin, curvature)
    out_origin = expmap_0(pv_tangent_vectors, curvature)
    assert jnp.allclose(out_general, out_origin, atol=atol, rtol=rtol)


def test_pv_logmap_matches_logmap_0_at_origin(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """logmap(y, 0) = logmap_0(y)."""
    atol, rtol = tolerance
    logmap = jax.vmap(pv_manifold.logmap, in_axes=(0, 0, None))
    logmap_0 = jax.vmap(pv_manifold.logmap_0, in_axes=(0, None))

    origin = jnp.zeros_like(pv_points)
    out_general = logmap(pv_points, origin, curvature)
    out_origin = logmap_0(pv_points, curvature)
    assert jnp.allclose(out_general, out_origin, atol=atol, rtol=rtol)


def test_pv_retraction_is_exact_euclidean_addition(
    pv_manifold: ProperVelocity,
    curvature: float,
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """retraction(v, x) = x + v exactly, R_x(0) = x, and R_x(tv) = exp_x(tv) + O(t^2).

    PV is unconstrained R^n, so the documented retraction is plain vector addition.
    Quadratic-order agreement with expmap is the defining retraction property:
    halving the step must shrink the batch-total deviation from expmap by ~4x.
    """
    retraction = jax.vmap(pv_manifold.retraction, in_axes=(0, 0, None))
    expmap = jax.vmap(pv_manifold.expmap, in_axes=(0, 0, None))

    out = retraction(pv_tangent_vectors, pv_points, curvature)
    assert jnp.array_equal(out, pv_points + pv_tangent_vectors)
    assert jnp.array_equal(retraction(jnp.zeros_like(pv_points), pv_points, curvature), pv_points)

    # err(t) = ||R_x(tv) - exp_x(tv)|| per point; summed over the batch to average out
    # per-point noise before taking the quadratic-shrink ratio.
    def batch_err(t: float) -> float:
        v_scaled = t * pv_tangent_vectors
        diff = retraction(v_scaled, pv_points, curvature) - expmap(v_scaled, pv_points, curvature)
        return float(jnp.sum(jnp.linalg.norm(diff, axis=-1)))

    t = 0.1
    assert batch_err(t / 2) <= 0.35 * batch_err(t)


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------


def test_pv_ptransp_0_matches_ptransp_from_origin(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """PT_{0→y}(v) = PT_{x=0, y}(v)."""
    atol, rtol = tolerance
    ptransp = jax.vmap(pv_manifold.ptransp, in_axes=(0, 0, 0, None))
    ptransp_0 = jax.vmap(pv_manifold.ptransp_0, in_axes=(0, 0, None))

    origin = jnp.zeros_like(pv_points)
    via_general = ptransp(pv_tangent_vectors, origin, pv_points, curvature)
    via_zero = ptransp_0(pv_tangent_vectors, pv_points, curvature)
    assert jnp.allclose(via_general, via_zero, atol=atol, rtol=rtol)


def test_pv_ptransp_0_round_trip(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """PT_{y→0} ∘ PT_{0→y}(v) = v."""
    atol, rtol = tolerance
    if pv_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 2e-2)
    ptransp = jax.vmap(pv_manifold.ptransp, in_axes=(0, 0, 0, None))
    ptransp_0 = jax.vmap(pv_manifold.ptransp_0, in_axes=(0, 0, None))

    origin = jnp.zeros_like(pv_points)

    forward = ptransp_0(pv_tangent_vectors, pv_points, curvature)
    back = ptransp(forward, pv_points, origin, curvature)
    assert jnp.allclose(back, pv_tangent_vectors, atol=atol, rtol=rtol)


def test_pv_ptransp_preserves_tangent_norm(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """‖PT_{0→y}(v)‖_y = ‖v‖_0."""
    atol, rtol = tolerance
    if pv_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 2e-2)
    ptransp_0 = jax.vmap(pv_manifold.ptransp_0, in_axes=(0, 0, None))
    tangent_norm = jax.vmap(pv_manifold.tangent_norm, in_axes=(0, 0, None))

    origin = jnp.zeros_like(pv_points)

    norm_at_origin = tangent_norm(pv_tangent_vectors, origin, curvature)
    transported = ptransp_0(pv_tangent_vectors, pv_points, curvature)
    norm_at_y = tangent_norm(transported, pv_points, curvature)
    assert jnp.allclose(norm_at_origin, norm_at_y, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Tangent inner product
# ---------------------------------------------------------------------------


def test_pv_tangent_inner_symmetric(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
    rng: np.random.Generator,
) -> None:
    """g_x(u, v) = g_x(v, u)."""
    atol, rtol = tolerance
    tangent_inner = jax.vmap(pv_manifold.tangent_inner, in_axes=(0, 0, 0, None))

    u = pv_tangent_vectors
    v = jnp.asarray(
        rng.normal(0.0, 1.0, size=u.shape).astype(np.dtype(u.dtype.name)) * 0.3,
        dtype=u.dtype,
    )

    inner_uv = tangent_inner(u, v, pv_points, curvature)
    inner_vu = tangent_inner(v, u, pv_points, curvature)
    assert jnp.allclose(inner_uv, inner_vu, atol=atol, rtol=rtol)


def test_pv_tangent_inner_positive_definite(
    pv_manifold: ProperVelocity,
    curvature: float,
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """g_x(v, v) > 0 for nonzero v."""
    tangent_inner = jax.vmap(pv_manifold.tangent_inner, in_axes=(0, 0, 0, None))

    inner = tangent_inner(pv_tangent_vectors, pv_tangent_vectors, pv_points, curvature)
    # Discard any accidentally-zero rows.
    norms = jnp.linalg.norm(pv_tangent_vectors, axis=-1)
    nonzero = norms > 0.0
    assert jnp.all(inner[nonzero] > 0)


def test_pv_tangent_norm_equals_dist(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """‖log_x(y)‖_x = d(x, y)."""
    atol, rtol = tolerance
    if pv_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 3e-2)
    logmap = jax.vmap(pv_manifold.logmap, in_axes=(0, 0, None))
    tangent_norm = jax.vmap(pv_manifold.tangent_norm, in_axes=(0, 0, None))
    dist = jax.vmap(pv_manifold.dist, in_axes=(0, 0, None))

    x, y = _split(pv_points, 2)
    v = logmap(y, x, curvature)
    lhs = tangent_norm(v, x, curvature)
    rhs = dist(x, y, curvature)
    assert jnp.allclose(lhs, rhs, atol=atol, rtol=rtol)


def test_pv_egrad2rgrad_is_metric_dual(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
    rng: np.random.Generator,
) -> None:
    """⟨grad, v⟩ = g_x(egrad2rgrad(grad), v) for all v."""
    atol, rtol = tolerance
    if pv_points.dtype == jnp.dtype("float32"):
        rtol = max(rtol, 5e-3)
    egrad2rgrad = jax.vmap(pv_manifold.egrad2rgrad, in_axes=(0, 0, None))
    tangent_inner = jax.vmap(pv_manifold.tangent_inner, in_axes=(0, 0, 0, None))

    grad = jnp.asarray(
        rng.normal(0.0, 1.0, size=pv_points.shape).astype(np.dtype(pv_points.dtype.name)),
        dtype=pv_points.dtype,
    )
    rgrad = egrad2rgrad(grad, pv_points, curvature)

    euclid_inner = jnp.sum(grad * pv_tangent_vectors, axis=-1)
    riem_inner = tangent_inner(rgrad, pv_tangent_vectors, pv_points, curvature)
    assert jnp.allclose(euclid_inner, riem_inner, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Validation / projection
# ---------------------------------------------------------------------------


def test_pv_is_in_manifold_finite_inputs(pv_manifold: ProperVelocity, curvature: float, pv_points: jnp.ndarray) -> None:
    """Finite points are in the manifold; NaN/Inf points are not."""
    is_in = jax.vmap(pv_manifold.is_in_manifold, in_axes=(0, None))
    assert bool(jnp.all(is_in(pv_points, curvature)))

    # A point with NaN must fail validation.
    bad = pv_points[0].at[0].set(jnp.nan)
    assert not bool(pv_manifold.is_in_manifold(bad, curvature))

    bad_inf = pv_points[0].at[0].set(jnp.inf)
    assert not bool(pv_manifold.is_in_manifold(bad_inf, curvature))


def test_pv_is_in_tangent_space_finite_inputs(
    pv_manifold: ProperVelocity,
    curvature: float,
    pv_points: jnp.ndarray,
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """Any finite vector is tangent at any PV point (T_x PV = R^n); NaN/Inf vectors are not."""
    is_tan = jax.vmap(pv_manifold.is_in_tangent_space, in_axes=(0, 0, None))
    assert bool(jnp.all(is_tan(pv_tangent_vectors, pv_points, curvature)))

    bad_nan = pv_tangent_vectors[0].at[0].set(jnp.nan)
    assert not bool(pv_manifold.is_in_tangent_space(bad_nan, pv_points[0], curvature))

    bad_inf = pv_tangent_vectors[0].at[0].set(jnp.inf)
    assert not bool(pv_manifold.is_in_tangent_space(bad_inf, pv_points[0], curvature))


def test_pv_proj_is_identity_on_finite_inputs(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_points: jnp.ndarray,
) -> None:
    """proj(x) = x whenever x is finite; NaN rows are scrubbed to 0."""
    atol, rtol = tolerance
    proj = jax.vmap(pv_manifold.proj, in_axes=(0, None))
    projected = proj(pv_points, curvature)
    assert jnp.allclose(projected, pv_points, atol=atol, rtol=rtol)

    dim = pv_points.shape[1]
    nan_row = jnp.full((dim,), jnp.nan, dtype=pv_points.dtype)
    cleaned = pv_manifold.proj(nan_row, curvature)
    assert jnp.all(jnp.isfinite(cleaned))


# ---------------------------------------------------------------------------
# Closed-form checks against paper Thm 4.3 (simplified at origin)
# ---------------------------------------------------------------------------


def test_pv_matches_paper_thm_4_3_at_origin(
    pv_manifold: ProperVelocity,
    curvature: float,
    tolerance: tuple[float, float],
    pv_tangent_vectors: jnp.ndarray,
) -> None:
    """At x = 0, Eq. 10-13 collapse to the simple asinh/sinh forms.

    Eq. 10 with x = 0 gives exp_0(v) = sinh(√c·||v||)·v/(√c·||v||).
    Eq. 13 with x = 0 gives d(0, y) = (1/√c)·asinh(√c·||y||).
    """
    atol, rtol = tolerance
    sqrt_c = float(np.sqrt(curvature))

    # Expected exp_0(v) from Thm 4.3 simplified form.
    v = pv_tangent_vectors
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    v_norm_safe = jnp.maximum(v_norm, 1e-12)
    expected_exp = jnp.sinh(sqrt_c * v_norm_safe) * v / (sqrt_c * v_norm_safe)

    expmap_0 = jax.vmap(pv_manifold.expmap_0, in_axes=(0, None))
    got_exp = expmap_0(v, curvature)
    assert jnp.allclose(got_exp, expected_exp, atol=atol, rtol=rtol)

    # Expected d(0, y) from Thm 4.3 simplified form.
    y = pv_tangent_vectors  # any finite R^n batch works
    y_norm = jnp.linalg.norm(y, axis=-1)
    expected_dist = jnp.asinh(sqrt_c * y_norm) / sqrt_c

    dist_0 = jax.vmap(pv_manifold.dist_0, in_axes=(0, None))
    got_dist = dist_0(y, curvature)
    assert jnp.allclose(got_dist, expected_dist, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Numerical stability at large radii
# ---------------------------------------------------------------------------


def test_pv_stability_at_large_norms(pv_manifold: ProperVelocity, curvature: float) -> None:
    """PV claims float32 stability up to large ‖x‖ — check finiteness up to 100.

    Paper Tables 1-2 claim PV operators remain finite where Poincaré's gradient
    vanishes and Hyperboloid's explodes. This guards that claim in our port.
    """
    # Build a ray of points with norms up to 100.
    dim = 8
    radii = jnp.asarray([0.0, 1.0, 5.0, 20.0, 50.0, 100.0], dtype=pv_manifold.dtype)
    unit = jnp.zeros((radii.shape[0], dim), dtype=pv_manifold.dtype).at[:, 0].set(1.0)
    pts = unit * radii[:, None]

    dist_0 = jax.vmap(pv_manifold.dist_0, in_axes=(0, None))
    logmap_0 = jax.vmap(pv_manifold.logmap_0, in_axes=(0, None))
    expmap_0 = jax.vmap(pv_manifold.expmap_0, in_axes=(0, None))

    d = dist_0(pts, curvature)
    assert jnp.all(jnp.isfinite(d))

    v = logmap_0(pts, curvature)
    assert jnp.all(jnp.isfinite(v))

    reconstructed = expmap_0(v, curvature)
    assert jnp.all(jnp.isfinite(reconstructed))


# ---------------------------------------------------------------------------
# Cross-check: private helpers
# ---------------------------------------------------------------------------


def test_pv_beta_matches_formula(curvature: float, pv_points: jnp.ndarray) -> None:
    """β_x = 1/√(1 + c·‖x‖²)."""
    sqnorm = jnp.sum(pv_points**2, axis=-1)
    expected = 1.0 / jnp.sqrt(1.0 + curvature * sqnorm)
    got = jax.vmap(pv_impl._beta, in_axes=(0, None))(pv_points, curvature)
    assert jnp.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_pv_dpi_norm_identity(curvature: float, pv_points: jnp.ndarray, pv_tangent_vectors: jnp.ndarray) -> None:
    """((1+β_x)/β_x)² · ‖dπ_x(v)‖² = g_x(v, v).

    This identity underlies the simplified expmap form; if it fails, the
    closed-form expmap is wrong by a scalar factor.
    """
    beta = jax.vmap(pv_impl._beta, in_axes=(0, None))(pv_points, curvature)
    dpi = jax.vmap(pv_impl._dpi_x, in_axes=(0, 0, None))(pv_points, pv_tangent_vectors, curvature)
    dpi_sqnorm = jnp.sum(dpi**2, axis=-1)

    scale = ((1.0 + beta) / beta) ** 2
    lhs = scale * dpi_sqnorm

    xv = jnp.sum(pv_points * pv_tangent_vectors, axis=-1)
    rhs = jnp.sum(pv_tangent_vectors**2, axis=-1) - curvature * beta**2 * xv**2

    assert jnp.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Radial tangents at large scaled geodesic radius
# ---------------------------------------------------------------------------


def _radial_unit_tangent(a: float, c: float, dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A point at scaled geodesic radius ``a`` and an *exactly* unit radial tangent there (f64).

    Derivation. ``dist_0(x) = asinh(√c‖x‖)/√c``, so ``√c·d(0, x) = a`` means
    ``‖x‖ = sinh(a)/√c`` and ``1/β_x = √(1 + c‖x‖²) = cosh(a)``. For a purely radial
    ``v = x̂·(1/β_x)`` the radial/perp split gives ``perp(v) = 0`` and ``radial(v) = 1/β_x``, so
    ``g_x(v, v) = (β_x·radial(v))² = 1``: the reference norm is exactly 1, no oracle needed.

    ``1/β_x`` comes from the module's own ``_beta_inv`` rather than from ``cosh(a)`` so the
    construction cannot drift from the implementation's own conventions.

    The direction is a coordinate axis (as in ``test_pv_stability_at_large_norms``) so that the
    float32 cast keeps ``v`` exactly radial. With a generic direction the cast leaves a
    perpendicular residue of ``‖v‖·eps``, whose ``(‖v‖·eps)²/2`` contribution to the norm is a
    float32 *storage* floor — measured 4.4e-5 at ``a = 12`` — that no arithmetic can remove.
    """
    u_D = jnp.zeros(dim, dtype=jnp.float64).at[0].set(1.0)
    x_D = (jnp.sinh(jnp.asarray(a, dtype=jnp.float64)) / jnp.sqrt(jnp.asarray(c, dtype=jnp.float64))) * u_D
    return x_D, u_D * pv_impl._beta_inv(x_D, c)


@pytest.mark.parametrize("a", [8.0, 10.0, 12.0])
def test_pv_radial_unit_tangent_keeps_its_norm_in_float32(a: float) -> None:
    """‖v‖_x = 1 for an exactly-unit radial tangent at ``√c·d`` ∈ {8, 10, 12}, in float32.

    The metric form ``⟨v, v⟩ - c·β_x²·⟨x, v⟩²`` is, for a radial ``v``, the difference of two
    terms of size ``cosh²(a)·‖v‖²`` reaching an answer of size ``‖v‖²``: at ``a = 8`` that
    amplification is 2e6, well past float32's ~1e7 of headroom. The output was finite and
    plausible throughout — a *norm*, positive, of the right order — which is why this asserts
    accuracy against the known value 1 rather than finiteness.

    Measured (c = 0.5, D = 16): ``|‖v‖_x - 1|`` ≤ 1.2e-7 and ``|g_x(v, v) - 1|`` ≤ 2.4e-7 at all
    three radii. The pre-fix spelling returns 1.118 (a = 8) and exactly 0 (a = 10 and 12, the
    form having rounded negative and been clipped), with ``g_x(v, v)`` off by 0.25, 1.0 and 1.5e3.
    """
    c, dim = 0.5, 16
    pv32, pv64 = ProperVelocity(dtype=jnp.float32), ProperVelocity(dtype=jnp.float64)
    x64_D, v64_D = _radial_unit_tangent(a, c, dim)

    # The construction's own claim, checked before it is used as the reference.
    assert float(pv64.tangent_norm(v64_D, x64_D, c)) == pytest.approx(1.0, abs=1e-12)

    x32_D, v32_D = x64_D.astype(jnp.float32), v64_D.astype(jnp.float32)
    assert float(pv32.tangent_norm(v32_D, x32_D, c)) == pytest.approx(1.0, abs=1e-5)
    assert float(pv32.tangent_inner(v32_D, v32_D, x32_D, c)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("a", [8.0, 10.0])
def test_pv_expmap_step_length_matches_the_tangent_norm_at_large_radius(a: float) -> None:
    """``d(x, exp_x(v)) = ‖v‖_x`` for a radial ``v`` of length 0.1 at ``√c·d`` ∈ {8, 10}.

    ``_expmap`` lands through the exact hyperboloid lift ``exp^H_X(V)[1:]``, so neither the step's
    length nor its direction cancels. Two earlier spellings did. The one before ``_tangent_norm``
    was rewritten took its geodesic *length* from the literal metric form and landed at 0.0849
    instead of 0.1 at ``a = 8`` and 4.28 at ``a = 10``. The ambient one after it had the length
    right but built the *direction* from ``dπ_x(v)``, whose two terms cancel by a factor
    ``β_x ≈ e^{-a}`` on a radial tangent, and then landed through the gyro-addition on top of that:
    measured 0.100011573 at ``a = 8`` (1.2e-4 relative, both backends) and 0.099952025 on the CPU /
    0.099870228 on an A100 at ``a = 10`` (4.8e-4 and 1.3e-3 — the second is 1.3x this assert's
    bound, which is how it surfaced). The current form is 2.0e-7 (CPU) / 1.3e-7 (GPU) at ``a = 8``
    and 5.7e-7 at ``a = 10`` on both
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_expmap_test_margins_{cpu,gpu}.out``).

    The yardstick is deliberately float64: ``ProperVelocity.dist`` between two points 0.1 apart
    at this radius is *itself* a float32 cancellation (it returns 0.315 for a true 0.1 at
    ``a = 8``), which this change does not address, so a float32 distance would measure that
    instead. The float32 *storage* of the landing point is not the issue — a relative coordinate
    perturbation of 6e-8 is a geodesic perturbation of the same order, five orders below the step.
    """
    c, dim = 0.5, 16
    pv32, pv64 = ProperVelocity(dtype=jnp.float32), ProperVelocity(dtype=jnp.float64)
    x64_D, unit_v64_D = _radial_unit_tangent(a, c, dim)
    v64_D = 0.1 * unit_v64_D
    x32_D, v32_D = x64_D.astype(jnp.float32), v64_D.astype(jnp.float32)

    step = float(pv32.tangent_norm(v32_D, x32_D, c))
    landed_D = pv32.expmap(v32_D, x32_D, c)
    landing = float(pv64.dist(x64_D, landed_D.astype(jnp.float64), c))

    assert landing == pytest.approx(step, rel=1e-3)


# ---------------------------------------------------------------------------
# dist / logmap through the exact hyperboloid lift
# ---------------------------------------------------------------------------


def _pair_at_scaled_radius(kind: str, a_x: float, a_y: float, c: float, dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Two float64 PV points at scaled geodesic radii ``a_x``, ``a_y`` in the named relative position.

    ``d(0, x) = asinh(√c‖x‖)/√c``, so ``√c·d(0, x) = a`` means ``‖x‖ = sinh(a)/√c``. The four
    geometries are the ones the polar frame treats differently: a generic pair, the two degenerate
    angles (same ray and opposite rays, where the angular leg of ``sinh²(θ/2)`` vanishes), and a
    right angle.
    """
    gen = np.random.default_rng([41, dim])
    u_D = np.zeros(dim)
    u_D[0] = 1.0
    w_D = np.zeros(dim)
    w_D[1] = 1.0
    scale_x, scale_y = np.sinh(a_x) / np.sqrt(c), np.sinh(a_y) / np.sqrt(c)
    if kind == "random":
        d_D = gen.normal(size=dim)
        v_D = d_D / np.linalg.norm(d_D)
    elif kind == "parallel":
        v_D = u_D
    elif kind == "antiparallel":
        v_D = -u_D
    elif kind == "perpendicular":
        v_D = w_D
    else:
        raise ValueError(kind)
    return (
        jnp.asarray(scale_x * u_D, dtype=jnp.float64),
        jnp.asarray(scale_y * v_D, dtype=jnp.float64),
    )


@pytest.mark.parametrize(("a_max", "round_trip_bound"), [(3.0, 1e-12), (6.0, 3e-9)])
def test_pv_logmap_carries_the_distance_and_inverts_expmap_in_float64(a_max: float, round_trip_bound: float) -> None:
    """``‖log_x(y)‖_x == d(x, y)`` and ``exp_x(log_x(y)) == y``, float64, scaled radius ≤ ``a_max``.

    These are the two identities that make the log map *the* inverse of the exponential map rather
    than merely a vector of about the right size, and they are what the change to the exact
    hyperboloid lift has to preserve. ``dist`` and ``logmap`` now read the same
    ``hyperboloid._polar_frame``, so the first identity holds by construction; asserting it is what
    would catch a lift that dropped a ``√c`` or transported into the wrong tangent space.

    Measured over ``c ∈ {0.1, 0.5, 1, 3}``, dims 2/5/64 and four pair geometries: ``|‖log‖_x - d|``
    ≤ 5.3e-15 at ``a ≤ 3`` and ≤1.8e-14 at ``a ≤ 6``.

    The round trip is bounded separately and more loosely at ``a = 6``, but not because of anything
    either map spells. ``_expmap`` is now the same exact lift and cancels nowhere, and feeding this
    very float64 log map into an 80-bit exponential map still leaves 2.2e-13 there, so neither arm
    of the composition is the floor. What is left is the conditioning of ``exp_x(v)`` for a *long*
    step: ``‖log_x(y)‖_x`` reaches 30 at ``c = 0.1`` on this grid, and
    ``cosh(√c‖v‖_x)·X + sinhc(√c‖v‖_x)·V`` then combines two terms of size ``e^(a_x + √c‖v‖_x)``
    into a point of size ``e^(a_y)``. Measured 1.6e-13 at ``a ≤ 3`` and 7.5e-10 at ``a ≤ 6``,
    identical on the CPU and on an A100, against 2.5e-13 / 6.5e-10 for the ambient ``_expmap`` this
    replaced -- the bound is 4x the measured value
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_expmap_landing_{cpu,gpu}.out``,
    section 2).
    """
    pv64 = ProperVelocity(dtype=jnp.float64)
    worst_identity, worst_round_trip = 0.0, 0.0
    for c in (0.1, 0.5, 1.0, 3.0):
        for dim in (2, 5, 64):
            for kind in ("random", "parallel", "antiparallel", "perpendicular"):
                x_D, y_D = _pair_at_scaled_radius(kind, a_max, 0.6 * a_max, c, dim)
                d = float(pv64.dist(x_D, y_D, c))
                log_D = pv64.logmap(y_D, x_D, c)
                worst_identity = max(worst_identity, abs(float(pv64.tangent_norm(log_D, x_D, c)) - d))
                worst_round_trip = max(worst_round_trip, float(pv64.dist(pv64.expmap(log_D, x_D, c), y_D, c)))

    assert worst_identity <= 1e-12
    assert worst_round_trip <= round_trip_bound


@pytest.mark.parametrize("a", [8.0, 10.0])
def test_pv_dist_and_logmap_resolve_a_short_step_at_large_radius_in_float32(a: float) -> None:
    """``d(x, y)`` and ``‖log_x(y)‖_x`` for two points 0.1 apart at ``√c·d`` ∈ {8, 10}, in float32.

    Both used to be built on the gyro-difference ``z = (⊖x) ⊕ y``, a sum of three terms of size
    ``e^(a+b)/√c`` cancelling down to ``sinh(θ)/√c``. The surviving significand is ``e^(a+b-θ)``
    times smaller than the operands, so float32 (``ln(1/eps) = 15.9``) has nothing left at these
    radii. The outputs were finite and plausible -- a distance, positive, of the right units -- which
    is why this asserts accuracy against a float64 oracle rather than finiteness.

    Measured with the pre-fix spelling on this construction: 0.0599 at ``a = 8`` (40 % low) and
    4.383 at ``a = 10`` (4280 % high), against ≤1.1e-6 relative for the current one -- which is the
    float32 *storage* floor of the pair itself (9.2e-7), so there is nothing further to win. The
    bound below is ~10x the measured error.

    The direction is a coordinate axis, as in :func:`_radial_unit_tangent` and for the same reason:
    with a generic direction the float32 cast leaves a perpendicular residue of ~eps in each unit
    vector, which the polar frame reads as a real angle and which moves the *true* distance of the
    stored pair by 1.4e-5 relative at ``a = 10``. That is a property of the stored points, not of
    the arithmetic, and it would be all this test measured.
    """
    c, dim, step = 0.5, 16, 0.1
    pv32, pv64 = ProperVelocity(dtype=jnp.float32), ProperVelocity(dtype=jnp.float64)
    x64_D, unit_v64_D = _radial_unit_tangent(a, c, dim)
    y64_D = pv64.expmap(step * unit_v64_D, x64_D, c)

    # The oracle's own claim, checked before it is used as the reference.
    d_true = float(pv64.dist(x64_D, y64_D, c))
    assert d_true == pytest.approx(step, rel=1e-9)

    x32_D, y32_D = x64_D.astype(jnp.float32), y64_D.astype(jnp.float32)
    assert float(pv32.dist(x32_D, y32_D, c)) == pytest.approx(d_true, rel=1e-5)
    log32_D = pv32.logmap(y32_D, x32_D, c)
    assert float(pv32.tangent_norm(log32_D, x32_D, c)) == pytest.approx(d_true, rel=1e-5)


# ---------------------------------------------------------------------------
# gyro-difference / parallel transport through the exact hyperboloid lift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [9.0, 12.0])
def test_pv_gyro_difference_is_accurate_at_large_radius_in_float32(a: float) -> None:
    """``(⊖x) ⊕ y`` for two points 0.1 apart at ``√c·d`` ∈ {9, 12}, in float32.

    This is the ``ProperVelocityGyroBatchNorm`` centering case: both operands far out, the result
    ``O(1)``. The ambient ``addition(scalar_mul(-1, x), y)`` is the Lorentz boost ``Λ_{⊖x} y``,
    three terms of size ``e^{2a}/√c`` that cancel *identically* at ``y = x``, so float32 has
    nothing left there — and it returns a perfectly plausible point, which is why this asserts
    accuracy against float64 rather than finiteness. Measured geodesic error of the ambient form on
    this construction: **2.39 nats at a = 9** and **10.2 at a = 12**, for a true separation of 0.1.

    The bound is 4x the float32 *storage* floor of the operands, ``eps32·sinh(a)/√c`` — the
    accuracy at which the pair can be held at all, 6.8e-4 at ``a = 9`` and 1.4e-2 at ``a = 12``.
    The new form measures 2.2e-9 / 7.6e-9 on the CPU and 7.3e-4 / 2.2e-8 on an A100, i.e. at or
    below that floor on both backends; the CPU's near-exactness at ``a = 9`` is the coordinate-axis
    construction, not something to assert
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_new_test_measurements{,_gpu}.out``,
    section 1; the random-direction sweep over ``a ∈ {8, 9, 10, 12}`` and separations 0.1 to 1 is
    ``step2c_pv_gyro_difference_accuracy.out``, section A, where the new form sits at 0.14x to
    0.54x the same floor propagated to the result and the ambient one at 0.75 to 11 nats).

    The direction is a coordinate axis, as in :func:`_radial_unit_tangent` and for the same reason:
    a generic direction's float32 cast moves the *true* separation of the stored pair, which is a
    property of the points rather than of the arithmetic.
    """
    c, dim, step = 0.5, 16, 0.1
    pv32, pv64 = ProperVelocity(dtype=jnp.float32), ProperVelocity(dtype=jnp.float64)
    x64_D, unit_v64_D = _radial_unit_tangent(a, c, dim)
    y64_D = pv64.expmap(step * unit_v64_D, x64_D, c)

    # The oracle's own claim, checked before it is used as the reference.
    assert float(pv64.dist(x64_D, y64_D, c)) == pytest.approx(step, rel=1e-9)

    x32_D, y32_D = x64_D.astype(jnp.float32), y64_D.astype(jnp.float32)
    ref_D = pv64.gyro_difference(x32_D.astype(jnp.float64), y32_D.astype(jnp.float64), c)
    got_D = pv32.gyro_difference(x32_D, y32_D, c)

    floor = float(np.finfo(np.float32).eps) * float(np.sinh(a)) / float(np.sqrt(c))
    err = float(pv64.dist(got_D.astype(jnp.float64), ref_D, c))
    assert err < 4.0 * floor, f"float32 gyro_difference is {err:.3e} nats off, floor is {floor:.3e}"


@pytest.mark.parametrize("a", [10.0, 14.0])
def test_pv_ptransp_is_an_isometry_at_large_radius_in_float32(a: float) -> None:
    """``‖PT_{x→y} v‖_y = ‖v‖_x`` and the transported direction, at ``√c·d`` ∈ {10, 14}, float32.

    ``_ptransp`` used paper Eq. 12, which routes ``v`` through ``_dpi_x`` — whose two terms cancel
    to a factor ``β_x ≈ e^{-a}`` of the operands on a radial ``v`` — and then multiplies the
    survivor back up by ``(1+β_x)/β_x ≈ e^{a}``. It now transports the exact hyperboloid lift
    instead. Measured isometry defect ``|‖PT v‖_y/‖v‖_x - 1|`` of the old spelling on random-direction
    operands at ``c = 0.5``, ``D = 16``: **0.34 at a = 8, 22.8 at a = 10, 782 at a = 12 and 7047 at
    a = 14**, with the transported direction flipped outright (angle π in the PV metric) at
    ``a ∈ {10, 12}`` — a momentum with the wrong length *and* the wrong sign, still finite and
    still plausible, which is why this asserts accuracy and not finiteness. The new form is
    9.4e-6 / 4.2e-5 / 4.8e-4 / 1.8e-3 on the same grid
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_ptransp_accuracy.out``).

    On the coordinate-axis construction used here both quantities are 4.0e-8 or below, identical
    on the CPU and on an A100, so the direction bound is 1.6e-7 = 4x the worst observed
    (``step2c_pv_new_test_measurements{,_gpu}.out``, section 3). The isometry bound is the looser
    1e-4: what remains at ``a = 14`` is ``Hyperboloid.ptransp``'s own float32 behaviour near the
    representation floor, measured at 1.803e-3 on the lifted operands against 1.804e-3 here
    (``step2c_pv_ptransp_accuracy.out``, section D), so this construction should not be tightened
    to the axis-aligned value.

    In float64 the new and old spellings agree to 1.2e-14 relative over ``c ∈ {0.1, 0.5, 1, 3}``,
    dims 2/5/64, four pair and three tangent geometries at ``a ≤ 3``, which pins the rewrite to a
    re-spelling (``step2c_pv_ptransp_equivalence.out``).
    """
    c, dim, step = 0.5, 16, 0.05
    pv32, pv64 = ProperVelocity(dtype=jnp.float32), ProperVelocity(dtype=jnp.float64)
    x64_D, radial_v64_D = _radial_unit_tangent(a, c, dim)

    gen = np.random.default_rng([151, dim])
    generic_v64_D = jnp.asarray(gen.normal(size=dim), dtype=jnp.float64)
    generic_v64_D = generic_v64_D / pv64.tangent_norm(generic_v64_D, x64_D, c)
    t64_D = jnp.asarray(gen.normal(size=dim), dtype=jnp.float64)
    t64_D = t64_D / pv64.tangent_norm(t64_D, x64_D, c)
    y64_D = pv64.expmap(step * t64_D, x64_D, c)

    x32_D, y32_D = x64_D.astype(jnp.float32), y64_D.astype(jnp.float32)
    xw_D, yw_D = x32_D.astype(jnp.float64), y32_D.astype(jnp.float64)

    for v64_D in (radial_v64_D, generic_v64_D):
        v32_D = v64_D.astype(jnp.float32)
        vw_D = v32_D.astype(jnp.float64)
        # The float64 leg is fed the same float32 numbers, so only the precision differs.
        ref_D = pv64.ptransp(vw_D, xw_D, yw_D, c)
        got_D = pv32.ptransp(v32_D, x32_D, y32_D, c).astype(jnp.float64)

        src = float(pv64.tangent_norm(vw_D, xw_D, c))
        iso = abs(float(pv64.tangent_norm(got_D, yw_D, c)) / src - 1.0)
        assert iso < 1e-4, f"transport is {iso:.3e} off being an isometry at a = {a}"

        direction = float(pv64.tangent_norm(got_D - ref_D, yw_D, c)) / float(pv64.tangent_norm(ref_D, yw_D, c))
        assert direction < 1.6e-7, f"transported vector is {direction:.3e} relative off float64"


# ---------------------------------------------------------------------------
# Gradients at the non-smooth points of the distance (audit D1)
# ---------------------------------------------------------------------------


def test_pv_distance_gradients_are_finite_at_the_origin(pv_manifold: ProperVelocity, curvature: float) -> None:
    """``grad dist(0, 0)`` and ``grad dist_0(0)`` must be finite, not NaN.

    ``_dist`` and ``_dist_0`` used a bare ``jnp.linalg.norm``, whose VJP at the zero vector is
    ``0/0``. At ``x = y = 0`` the gyro-difference ``-x ⊕ y`` is *exactly* zero, so both
    gradients came back all-NaN — the same failure class as the wrapped-normal NaN-at-the-mean
    bug, and exactly what the module's own ``_safe_norm`` (used by every other norm here) exists
    to prevent. Measured before the fix: ``[nan nan nan]`` for both, in float32 and float64.

    The origin is not incidental: ``dist_0`` at ``x = 0`` is what a "pull the embedding to the
    origin" regularizer differentiates, and one NaN there poisons a whole parameter tree.

    The finite off-origin gradients are asserted alongside so a "fix" that zeroes the gradient
    everywhere (e.g. a ``stop_gradient``) fails: at a generic point ``dist_0`` must still have
    the radial gradient ``x / (‖x‖·√(1 + c‖x‖²))``, checked here against that closed form.
    """
    dtype = pv_manifold.dtype
    zero = jnp.zeros(3, dtype=dtype)
    x = jnp.array([0.3, -0.4, 0.5], dtype=dtype)

    grad_dist_at_origin = jax.grad(lambda a, b: pv_manifold.dist(a, b, curvature))(zero, zero)
    grad_dist_0_at_origin = jax.grad(lambda a: pv_manifold.dist_0(a, curvature))(zero)

    assert bool(jnp.all(jnp.isfinite(grad_dist_at_origin)))
    assert bool(jnp.all(jnp.isfinite(grad_dist_0_at_origin)))

    # Values stay ~0 there: the safe-norm floor shifts them by at most MIN_NORM = 1e-15.
    assert float(pv_manifold.dist(zero, zero, curvature)) == pytest.approx(0.0, abs=1e-12)
    assert float(pv_manifold.dist_0(zero, curvature)) == pytest.approx(0.0, abs=1e-12)

    # Off the origin the gradient is unchanged and non-degenerate.
    grad_dist_0_at_x = jax.grad(lambda a: pv_manifold.dist_0(a, curvature))(x)
    x_norm = float(np.linalg.norm(np.asarray(x, dtype=np.float64)))
    expected = np.asarray(x, dtype=np.float64) / (x_norm * np.sqrt(1.0 + curvature * x_norm**2))
    assert bool(jnp.all(jnp.isfinite(grad_dist_0_at_x)))
    assert np.allclose(np.asarray(grad_dist_0_at_x, dtype=np.float64), expected, atol=1e-5, rtol=1e-5)


def test_pv_distance_gradients_are_finite_at_coincident_points(
    pv_manifold: ProperVelocity, curvature: float, pv_points: jnp.ndarray
) -> None:
    """``grad_x dist(x, x)`` must be finite for a whole batch of coincident pairs.

    ``dist`` is genuinely non-differentiable at ``x == y`` (the metric has a cone point there),
    so no particular value is asserted — only that autodiff returns numbers. Run over the full
    256-point fixture and under ``vmap`` because the pre-fix NaN was produced by whichever pair
    happened to cancel exactly; a single hand-picked pair can miss it.
    """
    grad_x = jax.vmap(jax.grad(lambda a, b: pv_manifold.dist(a, b, curvature)), in_axes=(0, 0))

    grads = grad_x(pv_points, pv_points)

    assert bool(jnp.all(jnp.isfinite(grads)))
