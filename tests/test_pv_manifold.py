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
