"""Tests for the κ-stereographic manifold (Bachmann et al. 2020, signed curvature).

Unlike the shared ``manifold_and_c`` fixture in ``conftest.py`` (which only samples ``c > 0`` and has a
hyperbolic-only point sampler), this file owns a **sign-aware** sampler and exercises all three regimes:

- ``c > 0`` hyperbolic — cross-validated bit-for-bit against :class:`~hyperbolix.manifolds.Poincare`.
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

import hyperbolix.manifolds.stereographic as stereo_impl
from hyperbolix.manifolds import Euclidean, Poincare, ProductManifold, Stereographic

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


@pytest.fixture(params=[2, 5, 10], ids=["dim2", "dim5", "dim10"])
def dim(request: pytest.FixtureRequest) -> int:
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
    if np.dtype(points.dtype) == np.dtype(np.float32):
        atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)
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
    if np.dtype(points.dtype) == np.dtype(np.float32):
        atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)
    x, y, _ = _split3(points)
    logmap = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    expmap = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    reconstructed = expmap(logmap(y, x, c), x, c)
    assert jnp.allclose(reconstructed, y, atol=atol, rtol=rtol)
    assert _batch_in_manifold(manifold, reconstructed, c)


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
    if np.dtype(points.dtype) == np.dtype(np.float32):
        atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)
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
    if np.dtype(points.dtype) == np.dtype(np.float32):
        atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)
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


def test_tangent_inner_positive_definite_and_symmetric(manifold, points, c, dim, rng, dtype):
    x, _, _ = _split3(points)
    u = jnp.asarray(rng.normal(size=x.shape).astype(np.dtype(jnp.dtype(dtype).name)))
    v = jnp.asarray(rng.normal(size=x.shape).astype(np.dtype(jnp.dtype(dtype).name)))
    inner = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))
    # Symmetry.
    assert jnp.allclose(inner(u, v, x, c), inner(v, u, x, c), atol=1e-5, rtol=1e-4)
    # Positive definiteness: ⟨u, u⟩_x > 0 for u ≠ 0.
    assert jnp.all(inner(u, u, x, c) > 0.0)


def test_egrad2rgrad_scales_by_inverse_metric(manifold, points, c, dim, rng, dtype):
    x, _, _ = _split3(points)
    grad = jnp.asarray(rng.normal(size=x.shape).astype(np.dtype(jnp.dtype(dtype).name)))
    rgrad = jax.vmap(manifold.egrad2rgrad, in_axes=(0, 0, None))(grad, x, c)
    lam = jax.vmap(lambda p: manifold.conformal_factor(p[None, :], c)[0, 0])(x)
    assert jnp.allclose(rgrad, grad / lam[:, None] ** 2, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Poincaré cross-validation (correctness anchor for the whole hyperbolic half)
# ---------------------------------------------------------------------------


def test_hyperbolic_ops_match_poincare(manifold, points, c, dtype, tolerance):
    if c <= 0:
        pytest.skip("Cross-validation against Poincaré only applies for c > 0.")
    atol, rtol = tolerance
    poincare = Poincare(dtype=dtype)
    x, y, z = _split3(points)
    checks = {
        "addition": (
            jax.vmap(manifold.addition, in_axes=(0, 0, None)),
            jax.vmap(poincare.addition, in_axes=(0, 0, None)),
            (x, y),
        ),
        "gyration": (
            jax.vmap(manifold.gyration, in_axes=(0, 0, 0, None)),
            jax.vmap(poincare.gyration, in_axes=(0, 0, 0, None)),
            (x, y, z),
        ),
        "expmap_0": (jax.vmap(manifold.expmap_0, in_axes=(0, None)), jax.vmap(poincare.expmap_0, in_axes=(0, None)), (x,)),
        "logmap_0": (jax.vmap(manifold.logmap_0, in_axes=(0, None)), jax.vmap(poincare.logmap_0, in_axes=(0, None)), (y,)),
        "ptransp_0": (
            jax.vmap(manifold.ptransp_0, in_axes=(0, 0, None)),
            jax.vmap(poincare.ptransp_0, in_axes=(0, 0, None)),
            (x, y),
        ),
    }
    for name, (fn_s, fn_p, args) in checks.items():
        got, ref = fn_s(*args, c), fn_p(*args, c)
        assert jnp.allclose(got, ref, atol=atol, rtol=rtol), f"stereographic.{name} != poincare.{name}"
    # Conformal factor (batched).
    cf_s, cf_p = manifold.conformal_factor(x, c), poincare.conformal_factor(x, c)
    assert jnp.allclose(cf_s, cf_p, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Geodesics & spherical antipode
# ---------------------------------------------------------------------------


def test_geodesic_endpoints_and_speed(manifold, points, c, tolerance):
    atol, rtol = tolerance
    if np.dtype(points.dtype) == np.dtype(np.float32):
        atol, rtol = max(atol, 1e-2), max(rtol, 2e-2)
    x, y, _ = _split3(points)
    geo = jax.vmap(manifold.geodesic, in_axes=(None, 0, 0, None))
    assert jnp.allclose(geo(0.0, x, y, c), x, atol=atol, rtol=rtol)
    assert jnp.allclose(geo(1.0, x, y, c), y, atol=atol, rtol=rtol)
    # Constant-speed property: d(x, gamma(t)) = t·d(x, y).
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    total = dist(x, y, c)
    half = dist(x, geo(0.5, x, y, c), c)
    assert jnp.allclose(half, 0.5 * total, atol=atol, rtol=rtol)


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


@pytest.mark.parametrize("c_val", [0.0, 1e-9, 0.5, 2.0, -0.5, -2.0])
def test_antipode_gradient_finite_float32(c_val):
    # Regression (float32): the antipode builds a radius 1/√|c| that → ∞ as c → 0; the DISCARDED spherical
    # branch (c ≥ 0 returns -x) then fed ~1e8 into the κ-trig Taylor x**11 term, overflowing float32 to inf
    # whose 0·inf gradient leaked through jnp.where into the SELECTED -x branch → NaN grad. The radius
    # guard must keep both grad_x and grad_c finite across all regimes.
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


@pytest.mark.parametrize("c", [0.1, 1.0, 3.0])
def test_shared_core_bit_identical_to_poincare(dtype, c):
    """For ``c > 0`` the shared core is EXACTLY (not merely ``allclose``) Poincaré's result — they call
    the same underlying function object, so equality must be bit-for-bit."""
    x = jnp.array([0.12, -0.2, 0.05], dtype=dtype)
    y = jnp.array([-0.15, 0.1, 0.2], dtype=dtype)
    pm = Poincare(dtype=dtype)
    sm = Stereographic(dtype=dtype)
    assert bool(jnp.all(pm.addition(x, y, c) == sm.addition(x, y, c)))
    assert bool(jnp.all(pm.gyration(x, y, x, c) == sm.gyration(x, y, x, c)))
    assert bool(jnp.all(pm.proj(x * 6.0, c) == sm.proj(x * 6.0, c)))  # x*6 exits the c=3 ball → projects
    assert bool(jnp.all(pm.conformal_factor(x, c) == sm.conformal_factor(x, c)))
