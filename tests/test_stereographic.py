"""Tests for the κ-stereographic manifold (Bachmann et al. 2020, signed curvature).

Unlike the shared ``manifold_and_c`` fixture in ``conftest.py`` (which only samples ``c > 0`` and has a
hyperbolic-only point sampler), this file owns a **sign-aware** sampler and exercises all three regimes:

- ``c > 0`` hyperbolic — the shared gyrovector core is bit-for-bit :class:`~hyperbolix.manifolds.Poincare`
  (same function objects; guarded by the ``is``-identity test), and the κ-trig-based maps are
  cross-validated against Poincaré's independent tanh closed forms.
- ``c = 0`` Euclidean — with the gyrovector factor-2 metric (``dist → 2‖x-y‖``, ``expmap → x+v``).
- ``c < 0`` spherical — cross-validated against the great-circle distance on the sphere embedding.

The sphere embedding (``inv_sproj``) is used as an *independent* oracle for the spherical regime so no
test re-derives the manifold's own formulas, and so no test touches the stereographic chart's
coordinate singularity (the antipode), where ``dist`` is an unavoidable ``0/0``.

Fixtures ``dtype``, ``seed_jax``, ``rng``, ``tolerance`` come from ``tests/conftest.py``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import hyperbolix.manifolds.stereographic as stereo_impl
from hyperbolix.manifolds import Euclidean, Poincare, ProductManifold, Stereographic
from hyperbolix.optim import mark_manifold_param, riemannian_adam

# ---------------------------------------------------------------------------
# NumPy oracles for the spherical regime (independent of the manifold code).
# For κ = -c > 0 the model is the stereographic projection of the sphere of radius R = 1/√κ.
# inv_sproj lifts a chart point x ∈ R^d to the sphere point X ∈ R^{d+1}.
# ---------------------------------------------------------------------------


def _sphere_embed(x: np.ndarray, c: float) -> np.ndarray:
    """Lift a stereographic chart point to the sphere of radius R = 1/√|κ| (κ = -c > 0)."""
    k = -c
    sqrt_abs_k = abs(k) ** 0.5
    lam = 2.0 / (1.0 - c * float(x @ x))
    return np.concatenate([lam * np.asarray(x), [(lam - 1.0) / sqrt_abs_k]])


def _great_circle_dist(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """Great-circle geodesic distance R·arccos(⟨X, Y⟩/R²) between the sphere lifts of x and y."""
    k = -c
    radius = 1.0 / abs(k) ** 0.5
    big_x, big_y = _sphere_embed(x, c), _sphere_embed(y, c)
    cos_angle = float(big_x @ big_y) / (radius * radius)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    return radius * math.acos(cos_angle)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Curvatures spanning the three regimes. Kept moderate so f32 stays in the reliable range.
_CURVATURES = [2.0, 0.5, 0.0, -0.5, -2.0]
_CURV_IDS = ["hyp_c2", "hyp_c0.5", "euclid_c0", "sph_c-0.5", "sph_c-2"]


@pytest.fixture(params=_CURVATURES, ids=_CURV_IDS)
def c(request: pytest.FixtureRequest) -> float:
    """Signed curvature spanning hyperbolic (c>0), Euclidean (c=0), spherical (c<0)."""
    return float(request.param)


@pytest.fixture(params=[2, 10], ids=["dim2", "dim10"])
def dim(request: pytest.FixtureRequest) -> int:
    """Ambient dimension. {2, 10}: no κ-stereographic code path branches on ``dim`` — it is pure
    vectorization width — so 2 (the disk/circle edge case) plus one generic value suffices."""
    return int(request.param)


@pytest.fixture
def manifold(dtype: jnp.dtype) -> Stereographic:
    return Stereographic(dtype=dtype)


def _sample_points(c: float, dim: int, n: int, rng: np.random.Generator, dtype: jnp.dtype) -> np.ndarray:
    """Sign-aware sampler kept away from each chart's singular locus.

    - c > 0 (ball radius 1/√c): uniform-on-ball directions/radii, shrunk (more in f32) to keep geodesic
      distances in the reliable range and off the boundary.
    - c < 0 (spherical, unbounded): moderate Gaussian so √|κ|‖x‖ is small — well away from the tan pole
      / antipodal locus where the chart degenerates.
    - c = 0: a moderate box.
    """
    np_dtype = np.dtype(jnp.dtype(dtype).name)
    is_f32 = np.dtype(dtype) == np.dtype(np.float32)
    if c > 0:
        dirs = rng.normal(size=(n, dim))
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        radii = rng.random((n, 1)) ** (1.0 / dim)
        shrink = 0.5 if is_f32 else 0.85  # stay off the boundary; tighter in f32
        pts = dirs * radii * shrink / np.sqrt(c)
    elif c < 0:
        # Dim-invariant ball so √|κ|‖x‖ ≤ r_max for every dim. A bare Gaussian's norm grows as √dim,
        # which pushes higher-dim points into the steep tan region and inflates f32 error; sampling a
        # direction x radius keeps the chart uniformly well-conditioned, off the tan pole / antipode.
        dirs = rng.normal(size=(n, dim))
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        radii = rng.random((n, 1)) ** (1.0 / dim)
        r_max = 0.5 if is_f32 else 0.7
        pts = dirs * radii * r_max / np.sqrt(abs(c))
    else:
        pts = rng.normal(size=(n, dim)) * 0.5
    return pts.astype(np_dtype)


@pytest.fixture
def points(manifold: Stereographic, c: float, dim: int, rng: np.random.Generator, dtype: jnp.dtype) -> jnp.ndarray:
    """A batch of on-manifold points for the given (curvature, dim, dtype)."""
    pts = _sample_points(c, dim, 384, rng, dtype)
    proj = jax.vmap(manifold.proj, in_axes=(0, None))
    return proj(jnp.asarray(pts), c)


def _split3(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, y, z = jnp.array_split(points, 3, axis=0)
    n = min(x.shape[0], y.shape[0], z.shape[0])
    return x[:n], y[:n], z[:n]


def _batch_in_manifold(manifold: Stereographic, pts: jnp.ndarray, c: float) -> bool:
    return bool(jnp.all(jax.vmap(lambda p: manifold.is_in_manifold(p, c))(pts)))


# ---------------------------------------------------------------------------
# Projection & membership
# ---------------------------------------------------------------------------


def test_proj_keeps_points_on_manifold(manifold, points, c):
    assert _batch_in_manifold(manifold, points, c)
    reproj = jax.vmap(manifold.proj, in_axes=(0, None))(points, c)
    assert _batch_in_manifold(manifold, reproj, c)
    # Already-projected points are (near-)fixed points of proj.
    assert jnp.allclose(reproj, points, atol=1e-5, rtol=1e-5)


def test_is_in_manifold_regimes(manifold, points, c, dim, dtype):
    assert _batch_in_manifold(manifold, points, c)
    if c > 0:
        # A point outside the ball ‖x‖ ≥ 1/√c is rejected.
        outside = jnp.ones((dim,), dtype=dtype) * (2.0 / jnp.sqrt(jnp.asarray(c)))
        assert not bool(manifold.is_in_manifold(outside, c))
    else:
        # Euclidean / spherical: all of R^d is valid, even large points.
        huge = jnp.ones((dim,), dtype=dtype) * 1e3
        assert bool(manifold.is_in_manifold(huge, c))


# ---------------------------------------------------------------------------
# Gyrovector algebra
# ---------------------------------------------------------------------------


def test_addition_identity_and_inverse(manifold, points, c, tolerance):
    atol, rtol = tolerance
    add = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    zero = jnp.zeros_like(points)
    # 0 ⊕ x = x and x ⊕ 0 = x
    assert jnp.allclose(add(zero, points, c), points, atol=atol, rtol=rtol)
    assert jnp.allclose(add(points, zero, c), points, atol=atol, rtol=rtol)
    # (-x) ⊕ x = 0 and x ⊕ (-x) = 0  (+1 shift dodges near-zero rtol blowup)
    assert jnp.allclose(add(-points, points, c) + 1.0, zero + 1.0, atol=atol, rtol=rtol)
    assert jnp.allclose(add(points, -points, c) + 1.0, zero + 1.0, atol=atol, rtol=rtol)
    # Results stay on the manifold.
    x, y, _ = _split3(points)
    assert _batch_in_manifold(manifold, add(x, y, c), c)


def test_scalar_mul_axioms(manifold, points, c, tolerance):
    atol, rtol = tolerance
    smul = jax.vmap(manifold.scalar_mul, in_axes=(0, 0, None))
    add = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    ones = jnp.ones(points.shape[0], dtype=points.dtype)
    # 1 ⊗ x = x
    assert jnp.allclose(smul(ones, points, c), points, atol=atol, rtol=rtol)
    # 0 ⊗ x = 0
    assert jnp.allclose(smul(0.0 * ones, points, c) + 1.0, jnp.ones_like(points), atol=atol, rtol=rtol)
    # Scalar distributivity: (r1 + r2) ⊗ x = (r1 ⊗ x) ⊕ (r2 ⊗ x)
    r1, r2 = 0.7 * ones, 1.3 * ones
    lhs = smul(r1 + r2, points, c)
    rhs = add(smul(r1, points, c), smul(r2, points, c), c)
    assert jnp.allclose(lhs, rhs, atol=atol, rtol=rtol)
    # 2 ⊗ x = x ⊕ x
    assert jnp.allclose(smul(2.0 * ones, points, c), add(points, points, c), atol=atol, rtol=rtol)


def test_scalar_mul_negative_is_gyro_inverse(manifold, points, c, tolerance):
    # (-1) ⊗ x = -x, the gyrogroup inverse — negative scalars were previously untested.
    atol, rtol = tolerance
    smul = jax.vmap(manifold.scalar_mul, in_axes=(0, 0, None))
    ones = jnp.ones(points.shape[0], dtype=points.dtype)
    assert jnp.allclose(smul(-ones, points, c), -points, atol=atol, rtol=rtol)


def test_gyration_gyrocommutative_law(manifold, points, c, tolerance):
    """``x ⊕ y = gyr[x, y](y ⊕ x)`` — the gyrocommutative law, in every curvature regime.

    This ties ``gyration`` to the oracle-anchored ``addition``: the ptransp-isometry test alone cannot
    pin gyration, because ANY tangent-space isometry preserves norms and inner products — a gyration
    that rotated by the wrong angle would pass it. Gyrocommutativity fails for any wrong rotation."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    add = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    gyr = jax.vmap(manifold.gyration, in_axes=(0, 0, 0, None))
    lhs = add(x, y, c)
    rhs = gyr(x, y, add(y, x, c), c)
    assert jnp.allclose(lhs, rhs, atol=atol, rtol=rtol)


def test_addition_left_cancellation(manifold, points, c, tolerance):
    """``(-x) ⊕ (x ⊕ y) = y`` — the gyrogroup left-cancellation law, in every curvature regime."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    add = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    assert jnp.allclose(add(-x, add(x, y, c), c), y, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def test_dist_metric_axioms(manifold, points, c, tolerance):
    atol, rtol = tolerance
    x, y, z = _split3(points)
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    assert jnp.all(dist(x, y, c) >= -atol)  # non-negativity
    assert jnp.allclose(dist(x, x, c), 0.0, atol=atol, rtol=rtol)  # identity of indiscernibles
    assert jnp.allclose(dist(x, y, c), dist(y, x, c), atol=atol, rtol=rtol)  # symmetry
    # Triangle inequality (points are sampled locally, so within the injectivity radius on the sphere).
    d_xz, d_xy, d_yz = dist(x, z, c), dist(x, y, c), dist(y, z, c)
    assert jnp.all(d_xz <= d_xy + d_yz + 10.0 * atol)


def test_dist_0_matches_dist_to_origin(manifold, points, c, tolerance):
    atol, rtol = tolerance
    origin = jnp.zeros_like(points)
    d0 = jax.vmap(manifold.dist_0, in_axes=(0, None))(points, c)
    d = jax.vmap(manifold.dist, in_axes=(0, 0, None))(points, origin, c)
    assert jnp.allclose(d0, d, atol=atol, rtol=rtol)


def test_dist_hyperbolic_matches_poincare(manifold, points, c, dtype, tolerance):
    if c <= 0:
        pytest.skip("Poincaré cross-check only applies to the hyperbolic regime (c > 0).")
    atol, rtol = tolerance
    poincare = Poincare(dtype=dtype)
    x, y, _ = _split3(points)
    d_stereo = jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, c)
    d_poincare = jax.vmap(poincare.dist, in_axes=(0, 0, None))(x, y, c)
    assert jnp.allclose(d_stereo, d_poincare, atol=atol, rtol=rtol)


def test_dist_spherical_matches_great_circle(manifold, points, c):
    if c >= 0:
        pytest.skip("Great-circle oracle only applies to the spherical regime (c < 0).")
    x, y, _ = _split3(points)
    d = np.asarray(jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, c))
    x_np, y_np = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ref = np.array([_great_circle_dist(x_np[i], y_np[i], c) for i in range(x_np.shape[0])])
    tol = 5e-4 if np.dtype(x.dtype) == np.dtype(np.float32) else 1e-9
    assert np.allclose(d, ref, atol=tol, rtol=tol)


def test_dist_euclidean_limit_has_factor_two(manifold, dtype):
    # The κ-stereographic metric is 4·I at the origin (λ_0 = 2), so d_0(x, y) = 2‖x - y‖ — NOT ‖x - y‖.
    x = jnp.array([0.1, 0.2, -0.05, 0.3], dtype=dtype)
    y = jnp.array([0.2, -0.1, 0.15, -0.2], dtype=dtype)
    d = manifold.dist(x, y, 0.0)
    assert jnp.allclose(d, 2.0 * jnp.linalg.norm(x - y), atol=1e-6)
    # tangent_norm carries the same factor 2; tangent_inner a factor 4.
    v = jnp.array([0.3, 0.1, -0.2, 0.05], dtype=dtype)
    assert jnp.allclose(manifold.tangent_norm(v, x, 0.0), 2.0 * jnp.linalg.norm(v), atol=1e-6)
    assert jnp.allclose(manifold.tangent_inner(v, v, x, 0.0), 4.0 * jnp.dot(v, v), atol=1e-6)


def _chart_point(theta: float, direction: np.ndarray, c: float) -> np.ndarray:
    """Chart coordinates of the sphere point at polar angle ``theta`` in the given direction:
    ``tan(θ/2)·R·d̂`` with ``R = 1/√|κ|`` (κ = -c > 0). Independent of the manifold code."""
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    radius = 1.0 / abs(-c) ** 0.5
    return math.tan(theta / 2.0) * radius * d


@pytest.mark.parametrize(
    ("c_val", "test_dtype", "tol"),
    [(-5e-6, jnp.float32, 1e-4), (-1e-6, jnp.float32, 1e-4), (-1e-10, jnp.float64, 1e-9)],
    ids=["f32_c-5e-6", "f32_c-1e-6", "f64_c-1e-10"],
)
def test_dist_spherical_below_taylor_cutover_matches_great_circle(c_val, test_dtype, tol):
    """Regression: spherical ``dist`` for ``|c|`` BELOW the κ-trig Taylor cutover (1e-5 float32,
    1e-9 float64) against the great-circle oracle, at separations spanning both sides of the
    ``|u| = |κ|·x² < _TAYLOR_MAX_U`` gate.

    Before the u-gate fix, any pair beyond a quarter great-circle fed the order-5 Taylor series a
    divergent argument (``tan(θ/2) > 1``): at ``c = -5e-6`` a 0.75π-separated pair returned
    **-1.09e6** for a true distance of 1053.7 — a silently wrong NEGATIVE distance, not a NaN. The
    original suite never caught this because its sampler caps chart norms at ``0.7R`` and only uses
    ``|c| ∈ {0.5, 2}`` (far above the cutover): an oracle only covers what its inputs reach."""
    m = Stereographic(dtype=test_dtype)
    direction = np.array([1.0, 0.2, -0.1])
    theta_x = 0.05 * math.pi
    x64 = _chart_point(theta_x, direction, c_val)
    x = jnp.asarray(x64, dtype=test_dtype)
    # Separations straddle the u-gate: 0.02π lands in the Taylor branch (tan²(sep/2) < 0.01), the
    # rest exercise the closed-form branch up to nearly antipodal.
    for theta_y in (0.07 * math.pi, 0.45 * math.pi, 0.80 * math.pi, 0.95 * math.pi):
        y64 = _chart_point(theta_y, direction, c_val)
        y = jnp.asarray(y64, dtype=test_dtype)
        got = float(m.dist(x, y, test_dtype(c_val)))
        ref = _great_circle_dist(x64, y64, c_val)
        assert got == pytest.approx(ref, rel=tol), f"sep={(theta_y - theta_x) / math.pi:.2f}π: {got} vs {ref}"


def test_flat_space_large_norm_no_overflow_float32():
    """Regression: the x**-power Taylor form NaN'd ``dist_0(x, c=0)`` in float32 for ``‖x‖ ≳ 3200``
    (``x**11 → inf`` while ``k⁵ = 0``, so the selected branch computed ``0·inf``). The Horner form in
    ``u = k·x²`` is exact at ``c = 0`` for any norm: ``dist_0 = 2‖x‖``, ``dist = 2‖x - y‖``."""
    m = Stereographic(dtype=jnp.float32)
    x = jnp.full((3,), 5000.0 / math.sqrt(3.0), dtype=jnp.float32)  # ‖x‖ = 5000
    y = -0.5 * x
    assert float(m.dist_0(x, 0.0)) == pytest.approx(2.0 * 5000.0, rel=1e-6)
    assert float(m.dist(x, y, 0.0)) == pytest.approx(2.0 * float(jnp.linalg.norm(x - y)), rel=1e-6)


# ---------------------------------------------------------------------------
# Exp / log maps
# ---------------------------------------------------------------------------


def test_expmap_0_logmap_0_inverse(manifold, points, c, tolerance):
    atol, rtol = tolerance
    logmap0 = jax.vmap(manifold.logmap_0, in_axes=(0, None))
    expmap0 = jax.vmap(manifold.expmap_0, in_axes=(0, None))
    reconstructed = expmap0(logmap0(points, c), c)
    assert jnp.allclose(reconstructed, points, atol=atol, rtol=rtol)


def test_expmap_logmap_inverse(manifold, points, c, tolerance):
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    logmap = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    expmap = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    reconstructed = expmap(logmap(y, x, c), x, c)
    assert jnp.allclose(reconstructed, y, atol=atol, rtol=rtol)
    assert _batch_in_manifold(manifold, reconstructed, c)


def test_logmap_expmap_roundtrip(manifold, points, c, rng, dtype, tolerance):
    """``log_x(exp_x(v)) = v`` — the reverse composition to ``test_expmap_logmap_inverse`` (which tests
    ``exp∘log``). Testing only one order can hide a shared systematic error; this pins the other.

    A *small* random tangent vector keeps ``exp_x(v)`` inside the injectivity radius in every regime
    (off the ball boundary for ``c > 0``, below ``π/√|c|`` for ``c < 0``), so the round-trip is exact
    up to floating point.
    """
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    v = jnp.asarray((rng.normal(size=x.shape) * 0.1).astype(np.dtype(jnp.dtype(dtype).name)))
    z = jax.vmap(manifold.expmap, in_axes=(0, 0, None))(v, x, c)
    v_rec = jax.vmap(manifold.logmap, in_axes=(0, 0, None))(z, x, c)
    assert jnp.allclose(v_rec, v, atol=atol, rtol=rtol)
    assert _batch_in_manifold(manifold, z, c)


def test_expmap_euclidean_limit(manifold, dtype):
    x = jnp.array([0.1, 0.2, -0.05], dtype=dtype)
    v = jnp.array([0.3, -0.1, 0.2], dtype=dtype)
    y = jnp.array([0.15, 0.25, -0.1], dtype=dtype)
    # At c = 0: expmap = x + v, logmap = y - x (factor 1, unlike dist).
    assert jnp.allclose(manifold.expmap(v, x, 0.0), x + v, atol=1e-6)
    assert jnp.allclose(manifold.logmap(y, x, 0.0), y - x, atol=1e-6)


def test_tangent_norm_consistency(manifold, points, c, tolerance):
    # ‖log_x(y)‖_x = d(x, y) at every point (holds for all c, including the factor-2 limit).
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    v = jax.vmap(manifold.logmap, in_axes=(0, 0, None))(y, x, c)
    tn = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))(v, x, c)
    d = jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, c)
    assert jnp.allclose(tn, d, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Transport, tangent metric, gradient conversion
# ---------------------------------------------------------------------------


def test_ptransp_preserves_riemannian_norm(manifold, points, c, tolerance):
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    origin = jnp.zeros_like(x)
    # Transport a tangent-at-origin vector 0 → y; its Riemannian norm is preserved.
    v0 = jax.vmap(manifold.logmap_0, in_axes=(0, None))(y, c)  # tangent at origin
    v_y = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))(v0, origin, y, c)
    n_src = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))(v0, origin, c)
    n_dst = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))(v_y, y, c)
    assert jnp.allclose(n_src, n_dst, atol=atol, rtol=rtol)
    # ptransp from the origin matches the general ptransp with x = 0.
    v_y0 = jax.vmap(manifold.ptransp_0, in_axes=(0, 0, None))(v0, y, c)
    assert jnp.allclose(v_y, v_y0, atol=atol, rtol=rtol)


def test_ptransp_isometry_general(manifold, points, c, rng, dtype, tolerance):
    """Parallel transport ``x → y`` is a linear ISOMETRY of tangent spaces: ``⟨Pu, Pw⟩_y = ⟨u, w⟩_x``
    for ARBITRARY vectors ``u, w`` and interior ``x ≠ 0``, ``y ≠ 0``.

    ``test_ptransp_preserves_riemannian_norm`` only checks single-vector norm preservation FROM THE
    ORIGIN; this covers the general two-point, two-vector bilinear form — so it also catches a transport
    that preserves individual norms but wrongly rotates *within* the tangent plane (which would leave the
    cross term ``⟨Pu, Pw⟩`` off).
    """
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    np_dtype = np.dtype(jnp.dtype(dtype).name)
    u = jnp.asarray(rng.normal(size=x.shape).astype(np_dtype))
    w = jnp.asarray(rng.normal(size=x.shape).astype(np_dtype))
    transport = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    inner = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))
    src = inner(u, w, x, c)
    dst = inner(transport(u, x, y, c), transport(w, x, y, c), y, c)
    assert jnp.allclose(src, dst, atol=atol, rtol=rtol)


def test_tangent_inner_positive_definite_and_symmetric(manifold, points, c, dim, rng, dtype):
    x, _, _ = _split3(points)
    u = jnp.asarray(rng.normal(size=x.shape).astype(np.dtype(jnp.dtype(dtype).name)))
    v = jnp.asarray(rng.normal(size=x.shape).astype(np.dtype(jnp.dtype(dtype).name)))
    inner = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))
    # Symmetry.
    assert jnp.allclose(inner(u, v, x, c), inner(v, u, x, c), atol=1e-5, rtol=1e-4)
    # Positive definiteness: ⟨u, u⟩_x > 0 for u ≠ 0.
    assert jnp.all(inner(u, u, x, c) > 0.0)


def test_egrad2rgrad_metric_duality(manifold, points, c, dim, rng, dtype, tolerance):
    """``⟨egrad2rgrad(g), v⟩_x = ⟨g, v⟩`` for arbitrary ``v`` — the DEFINING property of the Riemannian
    gradient (metric duality). This constrains ``egrad2rgrad`` through ``tangent_inner``, which is
    itself externally anchored (factor-4 Euclidean limit, ``tangent_norm``↔``dist`` consistency, the
    spherical oracles). A recompute-the-formula check (``rgrad == g/λ²`` using the implementation's own
    ``conformal_factor``) could only ever catch a wrong exponent, not a wrong ``λ`` — this can."""
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    np_dtype = np.dtype(jnp.dtype(dtype).name)
    g = jnp.asarray(rng.normal(size=x.shape).astype(np_dtype))
    v = jnp.asarray(rng.normal(size=x.shape).astype(np_dtype))
    rgrad = jax.vmap(manifold.egrad2rgrad, in_axes=(0, 0, None))(g, x, c)
    lhs = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))(rgrad, v, x, c)
    rhs = jnp.einsum("bi,bi->b", g, v)
    assert jnp.allclose(lhs, rhs, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Poincaré cross-validation (correctness anchor for the κ-trig layer)
# ---------------------------------------------------------------------------


def test_hyperbolic_ktrig_ops_match_poincare(manifold, points, c, dtype, tolerance):
    """Cross-validate the κ-trig-based ops against the independent Poincaré implementation for
    ``c > 0``.

    SCOPE (kept honest): only ops whose Stereographic implementation genuinely differs from
    Poincaré's are compared — the κ-trig ``where``-branches + clamped ``tanh`` here vs. Poincaré's
    direct ``tanh``/``atanh`` closed forms. NOT compared, deliberately: ``addition``/``gyration``/
    ``proj``/``conformal_factor`` are the SAME function object on both sides (guarded by
    ``test_shared_gyrovector_core_is_single_source``), and ``ptransp``/``ptransp_0``/
    ``tangent_inner``/``tangent_norm``/``egrad2rgrad``/``retraction`` are textually identical bodies
    over those shared helpers — comparing any of them would be ``f(x) == f(x)``, vacuous by
    construction. Their correctness is carried by the pre-existing Poincaré suite (c > 0), the
    spherical oracles (c < 0), and the property tests in this file."""
    if c <= 0:
        pytest.skip("Cross-validation against Poincaré only applies for c > 0.")
    atol, rtol = tolerance
    poincare = Poincare(dtype=dtype)
    x, y, _ = _split3(points)
    # Genuine tangent vectors of moderate norm for the two-point maps.
    v = jax.vmap(manifold.logmap, in_axes=(0, 0, None))(y, x, c)

    checks = {
        "expmap_0": (jax.vmap(manifold.expmap_0, in_axes=(0, None)), jax.vmap(poincare.expmap_0, in_axes=(0, None)), (x,)),
        "logmap_0": (jax.vmap(manifold.logmap_0, in_axes=(0, None)), jax.vmap(poincare.logmap_0, in_axes=(0, None)), (y,)),
        "dist_0": (jax.vmap(manifold.dist_0, in_axes=(0, None)), jax.vmap(poincare.dist_0, in_axes=(0, None)), (x,)),
        "expmap": (
            jax.vmap(manifold.expmap, in_axes=(0, 0, None)),
            jax.vmap(poincare.expmap, in_axes=(0, 0, None)),
            (v, x),
        ),
        "logmap": (
            jax.vmap(manifold.logmap, in_axes=(0, 0, None)),
            jax.vmap(poincare.logmap, in_axes=(0, 0, None)),
            (y, x),
        ),
        "scalar_mul": (
            jax.vmap(manifold.scalar_mul, in_axes=(None, 0, None)),
            jax.vmap(poincare.scalar_mul, in_axes=(None, 0, None)),
            (1.7, x),
        ),
    }
    for name, (fn_s, fn_p, args) in checks.items():
        got, ref = fn_s(*args, c), fn_p(*args, c)
        assert jnp.allclose(got, ref, atol=atol, rtol=rtol), f"stereographic.{name} != poincare.{name}"


# ---------------------------------------------------------------------------
# Geodesics & spherical antipode
# ---------------------------------------------------------------------------


def test_geodesic_endpoints_and_speed(manifold, points, c, tolerance):
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    geo = jax.vmap(manifold.geodesic, in_axes=(None, 0, 0, None))
    assert jnp.allclose(geo(0.0, x, y, c), x, atol=atol, rtol=rtol)
    assert jnp.allclose(geo(1.0, x, y, c), y, atol=atol, rtol=rtol)
    # Constant-speed property: d(x, gamma(t)) = t·d(x, y).
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    total = dist(x, y, c)
    half = dist(x, geo(0.5, x, y, c), c)
    assert jnp.allclose(half, 0.5 * total, atol=atol, rtol=rtol)


def test_geodesic_extrapolation_beyond_endpoint(manifold, points, c, tolerance):
    """``d(x, gamma(2)) = 2·d(x, y)`` — the geodesic parametrization holds OUTSIDE [0, 1] too
    (previously only interpolation was tested). On the sphere the identity only holds while the
    doubled distance stays inside the injectivity radius ``π·R``, so those pairs are masked out."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    geo = jax.vmap(manifold.geodesic, in_axes=(None, 0, 0, None))
    d = dist(x, y, c)
    doubled = dist(x, geo(2.0, x, y, c), c)
    if c < 0:
        mask = np.asarray(2.0 * d < 0.95 * math.pi / math.sqrt(abs(c)))
        assert mask.any(), "no pair inside the injectivity radius — sampler drifted"
    else:
        mask = np.ones(d.shape[0], dtype=bool)
    assert jnp.allclose(doubled[mask], (2.0 * d)[mask], atol=atol, rtol=rtol)


def test_geodesic_unit_contract(manifold, points, c, rng, dtype, tolerance):
    """``geodesic_unit(t, x, u)`` contract — this public method has no Poincaré counterpart, so it is
    validated against its own definition:

    - ``gamma(0) = x``;
    - it is UNIT SPEED: ``d(x, gamma(t)) = t`` (from ``2·tan_κ⁻¹(tan_κ(t/2)) = t``, holding for every
      curvature — the whole point of the κ-generalized construction);
    - its initial direction is ∝ ``u`` (``log_x(gamma(t))`` is a positive multiple of ``u``).

    ``t`` stays below the spherical injectivity radius ``π/√|c|`` so ``tan_κ⁻¹∘tan_κ`` stays principal.
    """
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    u = jnp.asarray(rng.normal(size=x.shape).astype(np.dtype(jnp.dtype(dtype).name)))

    def gu(t: float) -> jnp.ndarray:
        return jax.vmap(manifold.geodesic_unit, in_axes=(None, 0, 0, None))(t, x, u, c)

    # gamma(0) = x.
    assert jnp.allclose(gu(0.0), x, atol=atol, rtol=rtol)
    # Unit speed: d(x, gamma(t)) = t for t inside the injectivity radius.
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    for t in (0.3, 0.7, 1.2):
        assert jnp.allclose(dist(x, gu(float(t)), c), t, atol=atol, rtol=rtol), f"not unit speed at t={t}"
    # Initial direction ∝ u: log_x(gamma(t)) is a positive multiple of u (cosine similarity 1).
    v = jax.vmap(manifold.logmap, in_axes=(0, 0, None))(gu(0.3), x, c)
    cos = jnp.sum(v * u, axis=-1) / (jnp.linalg.norm(v, axis=-1) * jnp.linalg.norm(u, axis=-1))
    assert jnp.allclose(cos, 1.0, atol=atol, rtol=rtol)


def test_geodesic_midpoint_matches_sphere_midpoint(manifold, points, c):
    """Independent spherical oracle for the geodesic: the sphere lift of the constant-speed midpoint
    ``gamma(0.5, x, y)`` is the great-circle midpoint ``R·(X+Y)/‖X+Y‖`` of the endpoint lifts ``X, Y``
    (valid since all lifts have norm exactly ``R = 1/√|κ|``).

    This anchors ``geodesic`` — hence the ``scalar_mul`` / ``addition`` it is built from — in the
    spherical regime to something OTHER than its own inverse maps: the ``expmap∘logmap`` round-trips
    could in principle mask a shared error, an external oracle cannot.
    """
    if c >= 0:
        pytest.skip("Sphere-midpoint oracle only applies to the spherical regime (c < 0).")
    x, y, _ = _split3(points)
    mid = jax.vmap(manifold.geodesic, in_axes=(None, 0, 0, None))(0.5, x, y, c)
    x_np, y_np, mid_np = (np.asarray(a, dtype=np.float64) for a in (x, y, mid))
    radius = 1.0 / abs(-c) ** 0.5
    tol = 2e-3 if np.dtype(x.dtype) == np.dtype(np.float32) else 1e-8
    for i in range(x_np.shape[0]):
        bisector = _sphere_embed(x_np[i], c) + _sphere_embed(y_np[i], c)
        expected = radius * bisector / np.linalg.norm(bisector)
        assert np.max(np.abs(_sphere_embed(mid_np[i], c) - expected)) < tol


def test_antipode_is_sphere_antipode(manifold, points, c):
    if c >= 0:
        pytest.skip("Antipode is only the diametric point in the spherical regime (c < 0).")
    x, _, _ = _split3(points)
    ap = jax.vmap(manifold.antipode, in_axes=(0, None))(x, c)
    x_np, ap_np = np.asarray(x, dtype=np.float64), np.asarray(ap, dtype=np.float64)
    # The sphere lift of the antipode is the negated sphere lift of x (the true sphere antipode).
    for i in range(x_np.shape[0]):
        err = np.max(np.abs(_sphere_embed(ap_np[i], c) + _sphere_embed(x_np[i], c)))
        assert err < 1e-6, f"antipode is not the sphere antipode (err={err:.2e})"


@pytest.mark.parametrize("c_val", [-1e-6, -1e-7, -1e-8, -1e-10])
def test_antipode_tiny_negative_curvature_float32(c_val):
    """Regression: for every ``|c|`` below the float32 Taylor cutover (1e-5), the old
    ``geodesic_unit(πR, …)`` antipode fed the κ-trig Taylor series ``t/2 = π/(2√|c|)`` —
    ``|u| = π²/4``, outside the convergence region — returning wrong-signed garbage at ``c = -1e-6``
    and NaN at ``c = -1e-7`` and below. The prior antipode test never saw this: its curvature fixture
    only samples ``|c| ∈ {0.5, 2}``, far above the cutover.

    Oracle: the (independent, numpy) sphere lift — the lift of the antipode must be the NEGATED lift
    of ``x``. Compared as unit vectors, since the lifts' magnitude grows as ``R = 1/√|c|``."""
    m = Stereographic(dtype=jnp.float32)
    x = jnp.array([0.3, -0.5, 0.2], dtype=jnp.float32)
    ap = m.antipode(x, jnp.float32(c_val))
    assert bool(jnp.all(jnp.isfinite(ap))), f"antipode not finite at c={c_val}: {ap}"
    lift_x = _sphere_embed(np.asarray(x, dtype=np.float64), c_val)
    lift_ap = _sphere_embed(np.asarray(ap, dtype=np.float64), c_val)
    u_x = lift_x / np.linalg.norm(lift_x)
    u_ap = lift_ap / np.linalg.norm(lift_ap)
    assert np.max(np.abs(u_ap + u_x)) < 1e-4, f"antipode lift not antipodal at c={c_val}"


def test_antipode_involution(manifold, points, c):
    """``antipode(antipode(x)) = x`` in the spherical regime — the docstring's involution claim,
    exact for the closed-form chart inversion."""
    if c >= 0:
        pytest.skip("The involution is only nontrivial in the spherical regime (c < 0).")
    x, _, _ = _split3(points)
    ap = jax.vmap(manifold.antipode, in_axes=(0, None))(x, c)
    ap2 = jax.vmap(manifold.antipode, in_axes=(0, None))(ap, c)
    assert jnp.allclose(ap2, x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# κ-trig primitives: closed forms, κ→0 limit, inverse relations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2.0, 1.0, 0.5, -0.5, -1.0, -2.0])
def test_ktrig_closed_forms(k):
    # Domain-safe range: √|k|·|x| ≤ ~0.8 keeps every branch (atanh, tan) well inside its domain.
    xs = jnp.linspace(-0.5, 0.5, 41, dtype=jnp.float64)
    sqrt_abs_k = abs(k) ** 0.5
    if k < 0:  # hyperbolic branch
        assert jnp.allclose(stereo_impl._tan_k(xs, k), jnp.tanh(sqrt_abs_k * xs) / sqrt_abs_k, atol=1e-10)
        assert jnp.allclose(stereo_impl._artan_k(xs, k), jnp.arctanh(sqrt_abs_k * xs) / sqrt_abs_k, atol=1e-10)
    else:  # spherical branch
        assert jnp.allclose(stereo_impl._tan_k(xs, k), jnp.tan(sqrt_abs_k * xs) / sqrt_abs_k, atol=1e-10)
        assert jnp.allclose(stereo_impl._artan_k(xs, k), jnp.arctan(sqrt_abs_k * xs) / sqrt_abs_k, atol=1e-10)


def test_ktrig_inverse_relations():
    xs = jnp.linspace(-0.4, 0.4, 41, dtype=jnp.float64)
    for k in (2.0, 0.5, -0.5, -2.0):
        assert jnp.allclose(stereo_impl._artan_k(stereo_impl._tan_k(xs, k), k), xs, atol=1e-9)


def test_ktrig_taylor_coeffs_match_closed_form():
    # The κ→0 Taylor series only execute at |k| < K_ZERO_EPS (1e-9), where every term past the leading
    # `x` is ~1e-9 and vanishes under the other κ-trig tests' tolerances — so a mis-transcribed
    # higher-order coefficient would pass every test above. Pin the coefficients directly: evaluate each
    # `*_zero_taylor` at a MODERATE k (where the higher orders actually contribute above the truncation
    # error) against the independent jnp closed form. A wrong coefficient yields error ≥ 1e-3 ≫ atol.
    xs = jnp.linspace(-0.6, 0.6, 25, dtype=jnp.float64)
    for k in (0.3, -0.3, 0.15, -0.15):
        sqrt_abs_k = abs(k) ** 0.5
        tan_cf = (jnp.tan(sqrt_abs_k * xs) if k > 0 else jnp.tanh(sqrt_abs_k * xs)) / sqrt_abs_k
        artan_cf = (jnp.arctan(sqrt_abs_k * xs) if k > 0 else jnp.arctanh(sqrt_abs_k * xs)) / sqrt_abs_k
        assert jnp.allclose(stereo_impl._tan_k_zero_taylor(xs, k), tan_cf, atol=1e-6)
        assert jnp.allclose(stereo_impl._artan_k_zero_taylor(xs, k), artan_cf, atol=1e-6)


def test_ktrig_zero_curvature_limit():
    xs = jnp.linspace(-1.0, 1.0, 41, dtype=jnp.float64)
    # At k = 0 every κ-trig reduces to the identity, continuously.
    for fn in (stereo_impl._tan_k, stereo_impl._artan_k):
        assert jnp.allclose(fn(xs, 0.0), xs, atol=1e-12)
        assert jnp.allclose(fn(xs, 1e-12), xs, atol=1e-9)  # continuity across the Taylor seam
        assert jnp.allclose(fn(xs, -1e-12), xs, atol=1e-9)


def test_ktrig_seam_continuity_float32():
    # f32 analogue of test_ktrig_seam_continuity: the float32 cutover (1e-5) is the |k|-seam a signed
    # LearnableCurvature actually crosses in the library's DEFAULT dtype — the f64-only seam test
    # (at 1e-9) never exercises it.
    xs = jnp.linspace(-1.0, 1.0, 41, dtype=jnp.float32)
    eps = stereo_impl._K_ZERO_EPS_F32
    for fn in (stereo_impl._tan_k, stereo_impl._artan_k):
        for k in (eps, -eps):
            below = fn(xs, jnp.float32(k * 0.999))  # Taylor branch
            above = fn(xs, jnp.float32(k * 1.001))  # closed-form branch
            assert jnp.allclose(below, above, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("k_val", [9e-6, -9e-6])
def test_ktrig_u_gate_seam_float32(k_val):
    """Value accuracy on both sides of the ``|u| = |k|·x² < _TAYLOR_MAX_U`` gate, at fixed ``k`` just
    below the float32 cutover. Crossing this seam changes ``x`` (unlike the |k|-seam), so each side is
    compared against the float64 closed form as ground truth rather than against the other side. This
    pins the u-gate handover the bug fix introduced: Taylor just below, closed form just above."""
    x_seam = math.sqrt(stereo_impl._TAYLOR_MAX_U / abs(k_val))
    for fn in (stereo_impl._tan_k, stereo_impl._artan_k):
        for scale in (0.999, 1.001):  # Taylor side / closed-form side
            xv = x_seam * scale
            got = float(fn(jnp.float32(xv), jnp.float32(k_val)))
            ref = float(fn(jnp.float64(xv), jnp.float64(k_val)))
            assert got == pytest.approx(ref, rel=1e-5), f"u={k_val * xv**2:+.5f}"


# ---------------------------------------------------------------------------
# Gradient finiteness: the κ=0 Taylor seam and the singular points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_val", [0.0, 1e-4, -1e-4, 1e-9, -1e-9])
def test_finite_grad_wrt_curvature_near_zero(manifold, c_val):
    # Learnable curvature differentiates through c; the κ→0 Taylor seam must keep grad finite.
    x = jnp.array([0.1, 0.2, -0.05], dtype=jnp.float64)
    y = jnp.array([0.15, -0.1, 0.2], dtype=jnp.float64)
    v = jnp.array([0.3, 0.1, -0.2], dtype=jnp.float64)
    assert jnp.isfinite(jax.grad(lambda cc: manifold.dist(x, y, cc))(c_val))
    assert jnp.isfinite(jax.grad(lambda cc: jnp.sum(manifold.expmap(v, x, cc)))(c_val))
    assert jnp.isfinite(jax.grad(lambda cc: jnp.sum(manifold.scalar_mul(1.5, x, cc)))(c_val))
    assert jnp.isfinite(jax.grad(lambda cc: jnp.sum(manifold.logmap(y, x, cc)))(c_val))
    assert jnp.isfinite(jax.grad(lambda cc: manifold.dist_0(x, cc))(c_val))


@pytest.mark.parametrize("c_val", [1.0, 0.0, -1.0])
def test_finite_grad_at_singular_points(manifold, c_val):
    x = jnp.array([0.1, 0.2, -0.05], dtype=jnp.float64)
    # v = 0: tangent_norm / expmap have a 0/‖0‖ that the safe-norm must keep finite.
    g_tn = jax.grad(lambda v: manifold.tangent_norm(v, x, c_val))(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(g_tn))
    g_exp = jax.jacobian(lambda v: manifold.expmap(v, x, c_val))(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(g_exp))
    # x = y: dist / logmap have a coincident-point 0/0 that the safe-norm must keep finite.
    g_dist = jax.grad(lambda y: manifold.dist(x, y, c_val))(x)
    assert jnp.all(jnp.isfinite(g_dist))
    g_log = jax.jacobian(lambda y: manifold.logmap(y, x, c_val))(x)
    assert jnp.all(jnp.isfinite(g_log))


# ---------------------------------------------------------------------------
# JIT / vmap / grad-wrt-curvature smoke across regimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_val", [1.5, 0.0, -1.5])
def test_jit_vmap_grad_smoke(c_val):
    manifold = Stereographic(dtype=jnp.float64)
    x = jnp.array([[0.1, 0.2], [0.05, -0.1], [-0.2, 0.15]])
    y = jnp.array([[0.2, -0.1], [0.15, 0.05], [0.1, -0.2]])

    dist_jit = jax.jit(jax.vmap(manifold.dist, in_axes=(0, 0, None)))
    d = dist_jit(x, y, c_val)
    assert d.shape == (3,) and jnp.all(jnp.isfinite(d))

    # Differentiate a scalar loss through the curvature (learnable-curvature readiness).
    def loss(cc):
        return jnp.sum(jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, cc))

    g = jax.jit(jax.grad(loss))(c_val)
    assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# float32 gradient safety near c = 0 (the library default dtype). These pin the two float32-only
# hazards that the float64-only seam/grad tests above cannot see: a signed LearnableCurvature crossing
# zero lives here, in the default dtype.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_val", [1e-9, 5e-9, 1e-7, 1e-6, -1e-9, -1e-7, -1e-6])
def test_curvature_gradient_correct_near_zero_float32(c_val):
    # Regression (float32): the closed-form κ-trig κ-gradient dies to catastrophic cancellation for
    # |c| ≲ 1e-5 in float32 — even wrong-SIGNED below ~1e-8 — so a signed ``LearnableCurvature`` crossing
    # zero received garbage curvature gradients in the DEFAULT dtype. The dtype-aware Taylor cutover
    # (``_k_zero_eps``: 1e-5 for float32 vs 1e-9 for float64) must keep grad_c correct. Ground truth is
    # the float64 autodiff value (float64 has no cancellation here); note k = -c.
    g32 = float(jax.grad(lambda c: stereo_impl._artan_k(jnp.float32(0.5), -c))(jnp.float32(c_val)))
    g64 = float(jax.grad(lambda c: stereo_impl._artan_k(jnp.float64(0.5), -c))(jnp.float64(c_val)))
    assert math.copysign(1.0, g32) == math.copysign(1.0, g64), f"grad_c wrong sign at c={c_val}: f32={g32} f64={g64}"
    assert abs(g32 - g64) < 1e-2, f"grad_c inaccurate at c={c_val}: f32={g32} f64={g64}"

    # Same corruption surfaces through the public ``dist`` — the path ``LearnableCurvature`` differentiates.
    x = jnp.array([0.1, 0.2, -0.05])
    y = jnp.array([0.15, -0.1, 0.2])
    d32 = float(
        jax.grad(lambda c: Stereographic(dtype=jnp.float32).dist(x.astype(jnp.float32), y.astype(jnp.float32), c))(
            jnp.float32(c_val)
        )
    )
    d64 = float(jax.grad(lambda c: Stereographic(dtype=jnp.float64).dist(x, y, c))(jnp.float64(c_val)))
    assert abs(d32 - d64) < 1e-2, f"dist grad_c inaccurate at c={c_val}: f32={d32} f64={d64}"


@pytest.mark.parametrize("c_val", [0.0, 1e-9, -1e-6, -1e-7, 0.5, 2.0, -0.5, -2.0])
def test_antipode_gradient_finite_float32(c_val):
    # Regression (float32): the antipode is the closed-form chart inversion x/(c‖x‖²); the safe-norm
    # floor and the benign substituted curvature in the discarded c ≥ 0 branch must keep grad_x and
    # grad_c finite across all regimes — including tiny NEGATIVE c, where the old geodesic_unit(πR, …)
    # route produced NaN values/gradients (this parametrization originally tested +1e-9 but no tiny
    # negative c, which is exactly the hole the bug lived in).
    m = Stereographic(dtype=jnp.float32)
    x = jnp.array([0.3, -0.5, 0.2], dtype=jnp.float32)
    gx = jax.grad(lambda xx: jnp.sum(m.antipode(xx, jnp.float32(c_val))))(x)
    assert jnp.all(jnp.isfinite(gx)), f"antipode grad_x not finite at c={c_val}: {gx}"
    gc = jax.grad(lambda cc: jnp.sum(m.antipode(x, cc)))(jnp.float32(c_val))
    assert jnp.isfinite(gc), f"antipode grad_c not finite at c={c_val}: {gc}"


# ---------------------------------------------------------------------------
# Remaining protocol methods & continuity coverage
# ---------------------------------------------------------------------------


def test_retraction_on_manifold(manifold, points, c):
    # retraction = proj(x + v). A tiny step from a well-interior point stays on-manifold, and (since
    # proj is then a no-op / identity for c <= 0) equals the plain Euclidean update x + v.
    x, _, _ = _split3(points)
    v = 0.01 * jnp.ones_like(x)
    r = jax.vmap(manifold.retraction, in_axes=(0, 0, None))(v, x, c)
    assert _batch_in_manifold(manifold, r, c)
    assert jnp.allclose(r, x + v, atol=1e-5, rtol=1e-5)


def test_trivial_tangent_ops(manifold, points, c, dtype, rng):
    # Tangent space == ambient space here: tangent_proj is the identity and every vector is tangent.
    x, _, _ = _split3(points)
    v = jnp.asarray(rng.normal(size=x.shape).astype(np.dtype(jnp.dtype(dtype).name)))
    assert jnp.array_equal(jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))(v, x, c), v)
    in_tangent = jax.vmap(lambda vv, xx: manifold.is_in_tangent_space(vv, xx, c))(v, x)
    assert bool(jnp.all(in_tangent))


def test_stereographic_satisfies_manifold_protocol(dtype):
    # Structural conformance: Stereographic must expose the full runtime-checkable Manifold interface,
    # so it drops into any Manifold-typed API (and ProductManifold) unchanged. Guards against a public
    # method being renamed/dropped without a corresponding failure.
    from hyperbolix.manifolds.protocol import Manifold

    assert isinstance(Stereographic(dtype=dtype), Manifold)


def test_ktrig_seam_continuity():
    # No value jump where each κ-trig switches Taylor → closed form (evaluate both sides at ~the same k
    # so any residual is a true discontinuity, not the function's slope).
    xs = jnp.linspace(-1.0, 1.0, 41, dtype=jnp.float64)
    eps = stereo_impl.K_ZERO_EPS
    for fn in (stereo_impl._tan_k, stereo_impl._artan_k):
        for k in (eps, -eps):
            below = fn(xs, k * 0.999999)  # Taylor branch
            above = fn(xs, k * 1.000001)  # closed-form branch
            assert jnp.allclose(below, above, atol=1e-10, rtol=1e-8)


def test_stereographic_as_product_factor(dtype):
    # The design claims Stereographic drops into ProductManifold unchanged — verify end to end with
    # BOTH a hyperbolic (c>0) and a spherical (c<0) Stereographic factor alongside a Euclidean one.
    product = ProductManifold(
        (Stereographic(dtype=dtype), 3),
        (Stereographic(dtype=dtype), 3),
        (Euclidean(dtype=dtype), 2),
    )
    cs = (1.0, -1.0, 0.0)  # per-factor signed curvatures, supplied at call time
    raw_x = jnp.array([0.1, 0.2, -0.05, 0.15, -0.1, 0.2, 1.0, -2.0], dtype=dtype)
    raw_y = jnp.array([-0.1, 0.05, 0.2, -0.2, 0.1, -0.05, 0.5, 1.5], dtype=dtype)
    x, y = product.proj(raw_x, cs), product.proj(raw_y, cs)
    assert bool(product.is_in_manifold(x, cs)) and bool(product.is_in_manifold(y, cs))

    # Per-factor distances must match the standalone manifolds (i.e. the product really delegates).
    comp = product.component_dist(x, y, cs)
    stereo = Stereographic(dtype=dtype)
    expected = jnp.stack(
        [
            stereo.dist(x[0:3], y[0:3], 1.0),  # hyperbolic factor
            stereo.dist(x[3:6], y[3:6], -1.0),  # spherical factor
            Euclidean(dtype=dtype).dist(x[6:8], y[6:8], 0.0),  # euclidean factor
        ]
    )
    assert jnp.allclose(comp, expected, atol=1e-5, rtol=1e-5)
    # L2 product distance is Pythagorean over the per-factor distances.
    assert jnp.allclose(product.dist(x, y, cs), jnp.sqrt(jnp.sum(comp**2)), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Shared gyrovector-core dedup guard (Part B): Poincaré and Stereographic must
# import the c>0-bit-identical algebra from one module, not re-fork it.
# ---------------------------------------------------------------------------


def test_shared_gyrovector_core_is_single_source():
    """The core algebra is one function object across all three modules.

    If a future edit re-adds a local ``_addition`` / ``_gyration`` / ``_conformal_factor`` / ``_proj``
    to either manifold (silently re-forking the shared math), these ``is`` checks fail loudly.
    """
    import hyperbolix.manifolds._gyrovector_core as gc
    import hyperbolix.manifolds.poincare as poincare_impl

    for name in ("_addition", "_gyration", "_conformal_factor", "_conformal_factor_batch", "_proj"):
        shared = getattr(gc, name)
        assert getattr(poincare_impl, name) is shared, f"poincare.{name} diverged from the shared core"
        assert getattr(stereo_impl, name) is shared, f"stereographic.{name} diverged from the shared core"


def test_class_wrappers_add_no_divergent_pre_or_post_processing(dtype):
    """For ``c > 0`` the shared core is EXACTLY (not merely ``allclose``) Poincaré's result.

    This is a CLASS-PLUMBING guard, not independent validation: both methods dispatch to the same
    function object (see ``test_shared_gyrovector_core_is_single_source``), so a math error cannot
    fail it. What it CAN catch — and the ``is``-check cannot — is either class's *method* growing
    divergent pre/post-processing (extra casting, projection, reordering) around the shared call.

    Not parametrized over ``c``: neither side depends on the curvature except through the one shared
    call, so extra curvatures re-run the same plumbing."""
    c = 3.0
    x = jnp.array([0.12, -0.2, 0.05], dtype=dtype)
    y = jnp.array([-0.15, 0.1, 0.2], dtype=dtype)
    pm = Poincare(dtype=dtype)
    sm = Stereographic(dtype=dtype)
    assert bool(jnp.all(pm.addition(x, y, c) == sm.addition(x, y, c)))
    assert bool(jnp.all(pm.gyration(x, y, x, c) == sm.gyration(x, y, x, c)))
    assert bool(jnp.all(pm.proj(x * 6.0, c) == sm.proj(x * 6.0, c)))  # x*6 exits the c=3 ball → projects
    assert bool(jnp.all(pm.conformal_factor(x, c) == sm.conformal_factor(x, c)))


# ---------------------------------------------------------------------------
# Riemannian optimizer integration: egrad2rgrad/expmap/retraction/ptransp exist
# precisely so a ManifoldParam can live on this manifold — exercise that path
# end to end in both curved regimes (nothing else tests the dispatch).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_val", [1.0, -1.0], ids=["hyperbolic", "spherical"])
def test_riemannian_adam_on_stereographic_embedding(c_val):
    """A ManifoldParam on the Stereographic manifold trains with riemannian_adam: the loss (squared
    geodesic distance to a target) decreases and the point stays on-manifold. This is the only test
    of the optimizer dispatch path (egrad2rgrad → Adam moments → expmap → ptransp) for this manifold,
    and the only one that runs it at spherical curvature at all."""
    manifold = Stereographic(dtype=jnp.float32)
    target = jnp.array([0.3, 0.4], dtype=jnp.float32)
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, -0.1], dtype=jnp.float32)),
        manifold=manifold,
        curvature=c_val,
    )
    tx = riemannian_adam(learning_rate=0.05)
    opt_state = tx.init(param)

    def loss_fn(x):
        return manifold.dist(x, target, c_val) ** 2

    initial_loss = float(loss_fn(param[...]))
    for _ in range(50):
        grad = jax.grad(loss_fn)(param[...])
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates  # type: ignore[operator]

    final_loss = float(loss_fn(param[...]))
    assert jnp.all(jnp.isfinite(param[...]))
    assert final_loss < 0.1 * initial_loss, f"loss did not decrease: {initial_loss} -> {final_loss}"
    assert bool(manifold.is_in_manifold(param[...], c_val))
