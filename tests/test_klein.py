"""Tests for the Beltrami-Klein ball manifold (:class:`~hyperbolix.manifolds.Klein`).

Oracle / round-trip / consistency batteries. The oracles at the top of the file are written from the
textbook formulas in NumPy (or ``decimal`` for the near-antipodal Einstein addition reference) and
never call the library: the hyperboloid lift ``X = (1/(√c·√g), x/√g)``, the literal Klein metric
``g_x(u, v) = (u·v)/g + c(x·u)(x·v)/g²`` and the literal Einstein addition
``(x + y/gamma + c·gamma/(1+gamma)(x·y)x)/(1 + c x·y)``, with ``g = 1 - c‖x‖²`` and ``gamma = 1/√g``.

The high-precision Decimal battery for every primitive lives in ``test_manifold_oracles.py`` and the
isometry-equivariance tests in ``test_isometry_mappings.py``; they are not duplicated here.

Fixtures ``dtype``, ``seed_jax``, ``rng``, ``tolerance`` come from ``tests/conftest.py`` (which also
enables float64).
"""

from __future__ import annotations

import decimal
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.manifolds import Euclidean, Klein, Poincare, ProductManifold
from hyperbolix.manifolds.protocol import Manifold

# Dimension key:
#   B: batch of points (a slice of the ``points`` fixture)
#   D: manifold dimension (``dim``)
#   N: points averaged by ``einstein_midpoint``

# Largest scaled geodesic radius a = √c·d(0, x) of the sampled points: √c‖x‖ = tanh(a) ≤ tanh(3) = 0.995.
# Every Klein op divides by the boundary gap g = 1 - c‖x‖² = 1/cosh²(a), which float storage knows only to
# relative eps·cosh²(a): ≈ 100·eps at a = 3, i.e. 1.2e-5 in float32 — 300x below the shared 4e-3 tolerance,
# so two-point quantities (d ≤ 6) and round trips through twice the radius stay inside it. At a = 4.5 the
# same factor is ≈ 2000·eps = 2.4e-4, and quantities scaling with the metric 1/g² would eat the margin.
A_MAX = 3.0


# ---------------------------------------------------------------------------
# Independent oracles (NumPy / decimal; never call the library)
# ---------------------------------------------------------------------------


def _np_lift(x_D: np.ndarray, c) -> np.ndarray:
    """Klein point -> hyperboloid point ``(1/(√c·√g), x/√g)`` on ``-X₀² + ‖X_s‖² = -1/c`` (input dtype)."""
    g = 1 - c * (x_D @ x_D)
    return np.concatenate([[1 / (np.sqrt(c) * np.sqrt(g))], x_D / np.sqrt(g)])


def _np_minkowski(u: np.ndarray, v: np.ndarray):
    return -u[0] * v[0] + u[1:] @ v[1:]


def _np_dist(x_D: np.ndarray, y_D: np.ndarray, c: float) -> float:
    """``arcosh(-c⟨X, Y⟩_L)/√c`` of the hyperboloid lifts, in long double."""
    ld = np.longdouble
    x_ld, y_ld, c_ld = x_D.astype(ld), y_D.astype(ld), ld(c)
    arg = -c_ld * _np_minkowski(_np_lift(x_ld, c_ld), _np_lift(y_ld, c_ld))
    return float(np.arccosh(max(arg, ld(1))) / np.sqrt(c_ld))


def _np_klein_inner(u_D: np.ndarray, v_D: np.ndarray, x_D: np.ndarray, c: float) -> float:
    """Literal Klein metric ``(u·v)/g + c(x·u)(x·v)/g²``."""
    g = 1.0 - c * (x_D @ x_D)
    return float((u_D @ v_D) / g + c * (x_D @ u_D) * (x_D @ v_D) / g**2)


def _np_einstein_add(x_D: np.ndarray, y_D: np.ndarray, c) -> np.ndarray:
    """Literal Einstein addition (Ungar 2009), evaluated in the dtype of its inputs (pass ``c`` as that dtype)."""
    gamma = 1 / np.sqrt(1 - c * (x_D @ x_D))
    xy = x_D @ y_D
    return (x_D + y_D / gamma + c * gamma / (1 + gamma) * xy * x_D) / (1 + c * xy)


def _dec_einstein_add(x_D: np.ndarray, y_D: np.ndarray, c: float) -> np.ndarray:
    """Literal Einstein addition in 50-digit decimal arithmetic on the exact binary values of the inputs."""
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        xs = [decimal.Decimal(float(v)) for v in x_D]
        ys = [decimal.Decimal(float(v)) for v in y_D]
        c_d = decimal.Decimal(float(c))
        xx = sum(a * a for a in xs)
        xy = sum(a * b for a, b in zip(xs, ys, strict=True))
        gamma = 1 / (1 - c_d * xx).sqrt()
        coef = c_d * gamma / (1 + gamma) * xy
        den = 1 + c_d * xy
        return np.array([float((a + b / gamma + coef * a) / den) for a, b in zip(xs, ys, strict=True)])


def _np_lorentz_centroid_klein(x_ND: np.ndarray, w_N: np.ndarray, c: float) -> np.ndarray:
    """Normalized weighted Lorentz centroid of the lifted points, mapped back with ``k = X_s/(√c·X₀)``."""
    lifts_ND1 = np.stack([_np_lift(x, c) for x in x_ND])  # (N, D+1)
    z = w_N @ lifts_ND1  # (D+1,)
    z = z / (np.sqrt(c) * np.sqrt(-_np_minkowski(z, z)))  # back onto -X₀² + ‖X_s‖² = -1/c
    return z[1:] / (np.sqrt(c) * z[0])


def _central_diff_grad(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of a scalar function of a float64 vector."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        grad[i] = (float(f(jnp.asarray(x + e))) - float(f(jnp.asarray(x - e)))) / (2 * h)
    return grad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[0.1, 0.5, 1.0, 4.0], ids=["c0.1", "c0.5", "c1", "c4"])
def c(request: pytest.FixtureRequest) -> float:
    """Curvature magnitude (sectional curvature -c); spans a ball radius 1/√c from 3.16 to 0.5."""
    return float(request.param)


@pytest.fixture(params=[2, 10], ids=["dim2", "dim10"])
def dim(request: pytest.FixtureRequest) -> int:
    """No Klein code path branches on ``dim``: 2 (the disk) plus one generic width."""
    return int(request.param)


@pytest.fixture
def manifold(dtype: jnp.dtype) -> Klein:
    return Klein(dtype=dtype)


def _np_dtype(dtype) -> np.dtype:
    return np.dtype(jnp.dtype(dtype).name)


def _sample_klein(c: float, dim: int, n: int, rng: np.random.Generator, a_max: float = A_MAX) -> np.ndarray:
    """Float64 Klein points with uniform direction and scaled radius a ~ U(0, a_max), √c‖x‖ = tanh(a)."""
    dirs_BD = rng.normal(size=(n, dim))
    dirs_BD /= np.linalg.norm(dirs_BD, axis=-1, keepdims=True)
    a_B1 = rng.uniform(0.0, a_max, size=(n, 1))
    return dirs_BD * np.tanh(a_B1) / np.sqrt(c)


@pytest.fixture
def points(manifold: Klein, c: float, dim: int, rng: np.random.Generator, dtype: jnp.dtype) -> jnp.ndarray:
    """288 on-manifold points for the given (curvature, dim, dtype)."""
    pts_BD = _sample_klein(c, dim, 288, rng).astype(_np_dtype(dtype))
    return jax.vmap(manifold.proj, in_axes=(0, None))(jnp.asarray(pts_BD), c)


def _split3(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = points.shape[0] // 3
    return points[:n], points[n : 2 * n], points[2 * n : 3 * n]


def _tangent_vectors(x_BD, c: float, rng: np.random.Generator, r_lo: float, r_hi: float) -> jnp.ndarray:
    """Random tangent vectors at each ``x`` with Klein norm ‖v‖_x ~ U(r_lo, r_hi), scaled by the NumPy metric."""
    x64_BD = np.asarray(x_BD, dtype=np.float64)
    u_BD = rng.normal(size=x64_BD.shape)
    norms_B = np.array([math.sqrt(_np_klein_inner(u, u, x, c)) for u, x in zip(u_BD, x64_BD, strict=True)])
    r_B = rng.uniform(r_lo, r_hi, size=norms_B.shape)
    return jnp.asarray((u_BD * (r_B / norms_B)[:, None]).astype(np.asarray(x_BD).dtype))


def _batch_in_manifold(manifold: Klein, pts_BD: jnp.ndarray, c: float) -> bool:
    return bool(jnp.all(jax.vmap(lambda p: manifold.is_in_manifold(p, c))(pts_BD)))


def _margin(dtype) -> float:
    """The ``proj`` margin ``eps**0.75`` of the dtype: max norm ``1/√c - eps**0.75``."""
    return float(jnp.finfo(dtype).eps) ** 0.75


# ---------------------------------------------------------------------------
# 1. Projection & membership
# ---------------------------------------------------------------------------


def test_proj_leaves_interior_points_unchanged(manifold, points, c):
    # Sampled points sit at √c‖x‖ ≤ tanh(3), inside the margin, so proj is the exact identity.
    assert jnp.array_equal(jax.vmap(manifold.proj, in_axes=(0, None))(points, c), points)
    assert _batch_in_manifold(manifold, points, c)


def test_proj_outside_points_land_on_margin(manifold, c, dim, rng, dtype):
    dirs_BD = rng.normal(size=(16, dim))
    dirs_BD /= np.linalg.norm(dirs_BD, axis=-1, keepdims=True)
    scales_B1 = np.geomspace(1.0 + 1e-3, 1e3, 16)[:, None] / math.sqrt(c)
    raw_BD = jnp.asarray((dirs_BD * scales_B1).astype(_np_dtype(dtype)))
    out_BD = jax.vmap(manifold.proj, in_axes=(0, None))(raw_BD, c)
    eps = float(jnp.finfo(dtype).eps)
    expected_norm = 1.0 / math.sqrt(c) - _margin(dtype)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(out_BD, np.float64), axis=-1), expected_norm, rtol=8 * eps)
    # Direction preserved: out is a positive multiple of raw.
    cos_B = jnp.sum(out_BD * raw_BD, -1) / (jnp.linalg.norm(out_BD, axis=-1) * jnp.linalg.norm(raw_BD, axis=-1))
    np.testing.assert_allclose(np.asarray(cos_B), 1.0, rtol=8 * eps)
    assert _batch_in_manifold(manifold, out_BD, c)


def test_is_in_manifold_accepts_and_rejects(manifold, points, c, dim, dtype):
    assert _batch_in_manifold(manifold, points, c)
    unit_D = jnp.ones((dim,), dtype=dtype) / math.sqrt(dim)
    assert not bool(manifold.is_in_manifold(unit_D * (1.01 / math.sqrt(c)), c))
    assert not bool(manifold.is_in_manifold(unit_D * (10.0 / math.sqrt(c)), c))
    assert bool(manifold.is_in_manifold(unit_D * (1.0 / math.sqrt(c) - _margin(dtype)), c))


def test_proj_batch_matches_vmap_proj(manifold, points, c, dtype):
    # Mix interior points with points pushed outside the ball, and use two leading dims.
    scale_B1 = jnp.where(jnp.arange(points.shape[0])[:, None] % 2 == 0, 1.0, 3.0).astype(dtype)
    raw_BD = points * scale_B1
    expected_BD = jax.vmap(manifold.proj, in_axes=(0, None))(raw_BD, c)
    got = manifold.proj_batch(raw_BD.reshape(2, -1, raw_BD.shape[-1]), c).reshape(raw_BD.shape)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected_BD), rtol=4 * float(jnp.finfo(dtype).eps), atol=0)


# ---------------------------------------------------------------------------
# 2. Einstein gyrovector algebra
# ---------------------------------------------------------------------------


def test_addition_identity_and_inverse(manifold, points, c, tolerance):
    atol, rtol = tolerance
    add = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    zero_BD = jnp.zeros_like(points)
    np.testing.assert_allclose(add(zero_BD, points, c), points, atol=atol, rtol=rtol)
    np.testing.assert_allclose(add(points, zero_BD, c), points, atol=atol, rtol=rtol)
    np.testing.assert_allclose(add(points, -points, c), zero_BD, atol=atol)
    np.testing.assert_allclose(add(-points, points, c), zero_BD, atol=atol)


def test_addition_left_cancellation(manifold, points, c, tolerance):
    """``(-x) ⊕ (x ⊕ y) = y`` — the gyrogroup left-cancellation law."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    add = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    np.testing.assert_allclose(add(-x, add(x, y, c), c), y, atol=atol, rtol=rtol)
    assert _batch_in_manifold(manifold, add(x, y, c), c)


def test_gyro_difference_matches_addition_and_distance(manifold, points, c, tolerance):
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    gd_BD = jax.vmap(manifold.gyro_difference, in_axes=(0, 0, None))(x, y, c)
    np.testing.assert_allclose(gd_BD, jax.vmap(manifold.addition, in_axes=(0, 0, None))(-x, y, c), atol=atol, rtol=rtol)
    # ‖(⊖x) ⊕_E y‖ = tanh(√c·d(x, y))/√c.
    d_B = jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, c)
    sqrt_c = math.sqrt(c)
    np.testing.assert_allclose(jnp.linalg.norm(gd_BD, axis=-1), jnp.tanh(sqrt_c * d_B) / sqrt_c, atol=atol, rtol=rtol)


def test_addition_matches_literal_einstein_formula(manifold, points, c, dtype, tolerance):
    """Library (``s = x + y`` spelling) vs the literal Ungar formula in float64 on generic point pairs."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    got_BD = jax.vmap(manifold.addition, in_axes=(0, 0, None))(x, y, c)
    x64_BD, y64_BD = np.asarray(x, np.float64), np.asarray(y, np.float64)
    c_in = float(_np_dtype(dtype).type(c))  # the curvature value the library sees after weak-type casting
    ref_BD = np.stack([_np_einstein_add(a, b, c_in) for a, b in zip(x64_BD, y64_BD, strict=True)])
    np.testing.assert_allclose(np.asarray(got_BD, np.float64), ref_BD, atol=atol, rtol=rtol)


def test_addition_near_antipodal_beats_literal_form(manifold, c, dim, rng, dtype):
    """``x ⊕_E y`` at ``y = -x + δ`` near the boundary: small result, cancelling literal form.

    The literal numerator ``x + y/gamma + (…)x`` cancels O(‖x‖) terms down to O(‖δ‖/gamma), and its denominator
    ``1 + c x·y`` cancels to O(1/gamma²). The library's ``s = x + y`` spelling forms ``s = δ`` exactly
    (Sterbenz) and keeps only the storage floor of the gap: relative error ≲ eps·cosh²(a), with
    a = max scaled radius of x and y (measured ≤ 1.2x that in a probe). Reference: the literal formula
    in 50-digit decimal on the exact binary inputs. Negative control: the same literal formula in the
    test dtype misses the bound by 10x or more on the smallest δ at moderate radius.
    """
    eps = float(jnp.finfo(dtype).eps)
    np_dt = _np_dtype(dtype)
    c_in = float(np_dt.type(c))
    lib_ratio, literal_ratio_small_delta = [], []
    for a in (1.0, 2.0, 3.0, 4.0, 5.0):
        for delta in (1e-2, 1e-3, 1e-4, 1e-5):
            for _ in range(4):
                x_dir_D = rng.normal(size=dim)
                x_dir_D /= np.linalg.norm(x_dir_D)
                d_D = rng.normal(size=dim)
                d_D = d_D / np.linalg.norm(d_D) + x_dir_D  # lean δ along +x so y = -x + δ stays inside
                d_D /= np.linalg.norm(d_D)
                x_D = (np.tanh(a) / math.sqrt(c) * x_dir_D).astype(np_dt)
                y_D = (-x_D.astype(np.float64) + delta / math.sqrt(c) * d_D).astype(np_dt)
                sqrt_c_norm_y = math.sqrt(c) * float(np.linalg.norm(y_D.astype(np.float64)))
                if sqrt_c_norm_y >= 1.0 - 1e-3 / math.cosh(a) ** 2:
                    continue
                bound = eps * math.cosh(max(a, math.atanh(sqrt_c_norm_y))) ** 2
                ref_D = _dec_einstein_add(x_D, y_D, c_in)
                ref_norm = float(np.linalg.norm(ref_D))
                lib_D = np.asarray(manifold.addition(jnp.asarray(x_D), jnp.asarray(y_D), c), np.float64)
                lib_ratio.append(float(np.linalg.norm(lib_D - ref_D)) / ref_norm / bound)
                if delta == 1e-5 and a <= 3.0:
                    lit_D = _np_einstein_add(x_D, y_D, np_dt.type(c)).astype(np.float64)
                    literal_ratio_small_delta.append(float(np.linalg.norm(lit_D - ref_D)) / ref_norm / bound)
    assert len(lib_ratio) >= 60, "too many grid points rejected as outside the ball"
    assert max(lib_ratio) <= 4.0, f"library Einstein addition error {max(lib_ratio):.2f}x eps·cosh²(a)"
    assert max(literal_ratio_small_delta) >= 10.0, "negative control: literal form unexpectedly accurate"


# ---------------------------------------------------------------------------
# 3. Einstein scalar multiplication
# ---------------------------------------------------------------------------


def test_scalar_mul_identity_zero_negation(manifold, points, c, tolerance):
    atol, rtol = tolerance
    smul = jax.vmap(manifold.scalar_mul, in_axes=(None, 0, None))
    np.testing.assert_allclose(smul(1.0, points, c), points, atol=atol, rtol=rtol)
    np.testing.assert_allclose(smul(0.0, points, c), jnp.zeros_like(points), atol=atol)
    np.testing.assert_allclose(smul(-1.0, points, c), -points, atol=atol, rtol=rtol)


def test_scalar_mul_associativity(manifold, points, c, tolerance):
    """``(r1·r2) ⊗ x = r1 ⊗ (r2 ⊗ x)``; |r1·r2| ≤ 1.5 keeps the result at a ≤ 4.5 < float32 ceiling."""
    atol, rtol = tolerance
    smul = jax.vmap(manifold.scalar_mul, in_axes=(None, 0, None))
    for r1, r2 in ((0.5, 1.5), (-0.7, 0.4), (1.2, -1.1)):
        np.testing.assert_allclose(smul(r1 * r2, points, c), smul(r1, smul(r2, points, c), c), atol=atol, rtol=rtol)


def test_scalar_mul_scales_origin_distance(manifold, points, c, tolerance):
    atol, rtol = tolerance
    smul = jax.vmap(manifold.scalar_mul, in_axes=(None, 0, None))
    dist_0 = jax.vmap(manifold.dist_0, in_axes=(0, None))
    d0_B = dist_0(points, c)
    for r in (0.3, -0.8, 1.5):
        np.testing.assert_allclose(dist_0(smul(r, points, c), c), abs(r) * d0_B, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 4. Distance, exp/log round trips, geodesics
# ---------------------------------------------------------------------------


def test_dist_matches_hyperboloid_lift_oracle(manifold, points, c, dtype, tolerance):
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    got_B = jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, c)
    c_in = float(_np_dtype(dtype).type(c))
    ref_B = np.array([_np_dist(a, b, c_in) for a, b in zip(np.asarray(x), np.asarray(y), strict=True)])
    np.testing.assert_allclose(np.asarray(got_B, np.float64), ref_B, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        jax.vmap(manifold.dist_0, in_axes=(0, None))(x, c),
        jnp.arctanh(math.sqrt(c) * jnp.linalg.norm(x, axis=-1)) / math.sqrt(c),
        atol=atol,
        rtol=rtol,
    )


def test_expmap0_logmap0_round_trips(manifold, points, c, rng, tolerance):
    atol, rtol = tolerance
    exp0 = jax.vmap(manifold.expmap_0, in_axes=(0, None))
    log0 = jax.vmap(manifold.logmap_0, in_axes=(0, None))
    np.testing.assert_allclose(exp0(log0(points, c), c), points, atol=atol, rtol=rtol)
    # The metric at the origin is the identity, so ‖v‖_0 = ‖v‖; ‖v‖ ≤ 2/√c keeps exp_0(v) at a ≤ 2.
    v_BD = _tangent_vectors(jnp.zeros_like(points), c, rng, 0.01, 2.0 / math.sqrt(c))
    np.testing.assert_allclose(log0(exp0(v_BD, c), c), v_BD, atol=atol, rtol=rtol)


def test_expmap_logmap_round_trips(manifold, points, c, rng, tolerance):
    """``exp_x(log_x y) = y`` and ``log_x(exp_x v) = v``; tangent errors measured in the metric at x."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    log = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    tnorm = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))
    np.testing.assert_allclose(exp(log(y, x, c), x, c), y, atol=atol, rtol=rtol)
    # Klein norms up to 1.5/√c (scaled step ≤ 1.5): exp_x(v) stays at a ≤ 4.5, below the float32 ceiling.
    v_BD = _tangent_vectors(x, c, rng, 0.01, 1.5 / math.sqrt(c))
    err_B = tnorm(log(exp(v_BD, x, c), x, c) - v_BD, x, c)
    assert jnp.all(err_B <= atol + rtol * tnorm(v_BD, x, c))


def test_maps_at_origin_match_origin_variants(manifold, points, c, rng, tolerance):
    atol, rtol = tolerance
    zero_BD = jnp.zeros_like(points)
    v_BD = _tangent_vectors(zero_BD, c, rng, 0.01, 2.0 / math.sqrt(c))
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    log = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    np.testing.assert_allclose(
        exp(v_BD, zero_BD, c), jax.vmap(manifold.expmap_0, in_axes=(0, None))(v_BD, c), atol=atol, rtol=rtol
    )
    np.testing.assert_allclose(
        log(points, zero_BD, c), jax.vmap(manifold.logmap_0, in_axes=(0, None))(points, c), atol=atol, rtol=rtol
    )


def test_logmap_norm_is_distance(manifold, points, c, rng, tolerance):
    """``‖log_x y‖_x = d(x, y)`` and ``d(x, exp_x v) = ‖v‖_x``."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    tnorm = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    np.testing.assert_allclose(
        tnorm(jax.vmap(manifold.logmap, in_axes=(0, 0, None))(y, x, c), x, c), dist(x, y, c), atol=atol, rtol=rtol
    )
    v_BD = _tangent_vectors(x, c, rng, 0.01, 1.5 / math.sqrt(c))
    np.testing.assert_allclose(
        dist(x, jax.vmap(manifold.expmap, in_axes=(0, 0, None))(v_BD, x, c), c), tnorm(v_BD, x, c), atol=atol, rtol=rtol
    )


def test_logmap_at_base_point_is_exactly_zero(manifold, points, c):
    log_BD = jax.vmap(manifold.logmap, in_axes=(0, 0, None))(points, points, c)
    assert jnp.array_equal(log_BD, jnp.zeros_like(points))
    assert jnp.array_equal(
        jax.vmap(manifold.dist, in_axes=(0, 0, None))(points, points, c), jnp.zeros(points.shape[0], points.dtype)
    )


def test_geodesics_are_chords(manifold, points, c, rng, tolerance):
    """``exp_x(t·v)`` lies on the ray ``x + s·v̂`` (s > 0, increasing in t): Klein geodesics are straight."""
    atol, _ = tolerance
    x, _, _ = _split3(points)
    v_BD = _tangent_vectors(x, c, rng, 0.1, 1.5 / math.sqrt(c))
    v_hat_BD = v_BD / jnp.linalg.norm(v_BD, axis=-1, keepdims=True)
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    prev_B = jnp.zeros(x.shape[0], x.dtype)
    for t in (0.25, 0.5, 1.0):
        r_BD = exp(t * v_BD, x, c) - x
        along_B = jnp.sum(r_BD * v_hat_BD, axis=-1)
        perp_BD = r_BD - along_B[:, None] * v_hat_BD
        assert jnp.all(jnp.linalg.norm(perp_BD, axis=-1) <= atol / math.sqrt(c))
        assert jnp.all(along_B > prev_B)
        prev_B = along_B


# ---------------------------------------------------------------------------
# 5. Parallel transport
# ---------------------------------------------------------------------------


def test_ptransp_bilinear_isometry(manifold, points, c, rng, dtype, tolerance):
    """``⟨P u, P w⟩_y = ⟨u, w⟩_x`` for arbitrary ``u, w`` — catches a wrong in-plane rotation too."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    u_BD = _tangent_vectors(x, c, rng, 0.1, 2.0)
    w_BD = _tangent_vectors(x, c, rng, 0.1, 2.0)
    transport = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    inner = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))
    np.testing.assert_allclose(
        inner(transport(u_BD, x, y, c), transport(w_BD, x, y, c), y, c), inner(u_BD, w_BD, x, c), atol=atol, rtol=rtol
    )


def test_ptransp_round_trip(manifold, points, c, rng, tolerance):
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    u_BD = _tangent_vectors(x, c, rng, 0.1, 2.0)
    transport = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    tnorm = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))
    err_B = tnorm(transport(transport(u_BD, x, y, c), y, x, c) - u_BD, x, c)
    assert jnp.all(err_B <= atol + rtol * tnorm(u_BD, x, c))


def test_ptransp_from_origin_matches_ptransp_0(manifold, points, c, rng, tolerance):
    atol, rtol = tolerance
    zero_BD = jnp.zeros_like(points)
    v_BD = _tangent_vectors(zero_BD, c, rng, 0.1, 2.0)
    general = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))(v_BD, zero_BD, points, c)
    np.testing.assert_allclose(
        general, jax.vmap(manifold.ptransp_0, in_axes=(0, 0, None))(v_BD, points, c), atol=atol, rtol=rtol
    )


def test_ptransp_carries_log_to_negative_reverse_log(manifold, points, c, tolerance):
    """``P_{x→y}(log_x y) = -log_y x``: the geodesic's velocity is parallel along it."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    log = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    moved_BD = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))(log(y, x, c), x, y, c)
    reverse_BD = -log(x, y, c)
    err_B = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))(moved_BD - reverse_BD, y, c)
    assert jnp.all(err_B <= atol + rtol * jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, c))


# ---------------------------------------------------------------------------
# 6. Metric
# ---------------------------------------------------------------------------


def test_tangent_inner_matches_literal_metric(manifold, points, c, rng, dtype, tolerance):
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    np_dt = _np_dtype(dtype)
    u_BD = jnp.asarray(rng.normal(size=x.shape).astype(np_dt))
    v_BD = jnp.asarray(rng.normal(size=x.shape).astype(np_dt))
    inner = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))
    got_B = inner(u_BD, v_BD, x, c)
    ref_B = np.array(
        [_np_klein_inner(*(np.asarray(t, np.float64) for t in args), c) for args in zip(u_BD, v_BD, x, strict=True)]
    )
    np.testing.assert_allclose(np.asarray(got_B, np.float64), ref_B, atol=atol, rtol=rtol)
    # Symmetric up to rounding ((c·x·u)·x·v vs (c·x·v)·x·u associate differently): bound relative to the
    # magnitude of the two metric terms, since their sum can cancel.
    x64_BD, u64_BD, v64_BD = (np.asarray(t, np.float64) for t in (x, u_BD, v_BD))
    g_B = 1.0 - c * np.sum(x64_BD**2, -1)
    scale_B = (
        np.abs(np.sum(u64_BD * v64_BD, -1)) / g_B
        + c * np.abs(np.sum(x64_BD * u64_BD, -1) * np.sum(x64_BD * v64_BD, -1)) / g_B**2
    )
    sym_err_B = np.abs(np.asarray(got_B, np.float64) - np.asarray(inner(v_BD, u_BD, x, c), np.float64))
    assert np.all(sym_err_B <= 16 * float(jnp.finfo(dtype).eps) * scale_B)
    assert jnp.all(inner(u_BD, u_BD, x, c) > 0)  # positive definite
    # Bilinear: ⟨a·u + b·v, v⟩ = a⟨u, v⟩ + b⟨v, v⟩; tolerance scaled by the terms' magnitude.
    a, b = 1.7, -0.6
    lhs_B = inner(a * u_BD + b * v_BD, v_BD, x, c)
    uv_B, vv_B = inner(u_BD, v_BD, x, c), inner(v_BD, v_BD, x, c)
    assert jnp.all(jnp.abs(lhs_B - (a * uv_B + b * vv_B)) <= atol + rtol * (abs(a) * jnp.abs(uv_B) + abs(b) * vv_B))


def test_tangent_norm_squared_is_inner(manifold, points, c, rng, dtype, tolerance):
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    v_BD = jnp.asarray(rng.normal(size=x.shape).astype(_np_dtype(dtype)))
    norm_B = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))(v_BD, x, c)
    np.testing.assert_allclose(
        norm_B**2, jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))(v_BD, v_BD, x, c), atol=atol, rtol=rtol
    )


def test_egrad2rgrad_metric_duality(manifold, points, c, rng, dtype, tolerance):
    """``g_x(egrad2rgrad(e), v) = e·v`` for arbitrary ``v`` — the defining property of the Riemannian gradient."""
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    np_dt = _np_dtype(dtype)
    e_BD = jnp.asarray(rng.normal(size=x.shape).astype(np_dt))
    v_BD = jnp.asarray(rng.normal(size=x.shape).astype(np_dt))
    rgrad_BD = jax.vmap(manifold.egrad2rgrad, in_axes=(0, 0, None))(e_BD, x, c)
    lhs_B = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))(rgrad_BD, v_BD, x, c)
    np.testing.assert_allclose(lhs_B, jnp.einsum("bd,bd->b", e_BD, v_BD), atol=atol, rtol=rtol)


def test_lorentz_factor_matches_literal(manifold, points, c, tolerance):
    atol, rtol = tolerance
    got_B = jax.vmap(manifold.lorentz_factor, in_axes=(0, None))(points, c)
    x64_BD = np.asarray(points, np.float64)
    np.testing.assert_allclose(
        np.asarray(got_B, np.float64), 1.0 / np.sqrt(1.0 - c * np.sum(x64_BD**2, -1)), atol=atol, rtol=rtol
    )


def test_trivial_tangent_ops_and_retraction(manifold, points, c, rng, dtype):
    x, _, _ = _split3(points)
    v_BD = jnp.asarray(rng.normal(size=x.shape).astype(_np_dtype(dtype)))
    assert jnp.array_equal(jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))(v_BD, x, c), v_BD)
    assert bool(jnp.all(jax.vmap(lambda vv, xx: manifold.is_in_tangent_space(vv, xx, c))(v_BD, x)))
    assert not bool(manifold.is_in_tangent_space(v_BD[0].at[0].set(jnp.nan), x[0], c))
    retr_BD = jax.vmap(manifold.retraction, in_axes=(0, 0, None))(v_BD, x, c)
    assert jnp.array_equal(retr_BD, jax.vmap(manifold.proj, in_axes=(0, None))(x + v_BD, c))
    assert _batch_in_manifold(manifold, retr_BD, c)


# ---------------------------------------------------------------------------
# 7. Einstein midpoint
# ---------------------------------------------------------------------------


def test_einstein_midpoint_single_point(manifold, points, c, tolerance):
    atol, rtol = tolerance
    mid_BD = jax.vmap(lambda p: manifold.einstein_midpoint(p[None], None, c))(points)
    np.testing.assert_allclose(mid_BD, points, atol=atol, rtol=rtol)


def test_einstein_midpoint_two_points_is_geodesic_midpoint(manifold, points, c, tolerance):
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    mid_BD = jax.vmap(lambda a, b: manifold.einstein_midpoint(jnp.stack([a, b]), None, c))(x, y)
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    d_xm_B, d_my_B = dist(x, mid_BD, c), dist(mid_BD, y, c)
    np.testing.assert_allclose(d_xm_B, d_my_B, atol=atol, rtol=rtol)
    np.testing.assert_allclose(d_xm_B + d_my_B, dist(x, y, c), atol=atol, rtol=rtol)
    # On the chord between x and y: mid = x + t(y - x) with t ∈ [0, 1].
    w_BD = y - x
    t_B = jnp.sum((mid_BD - x) * w_BD, -1) / jnp.sum(w_BD * w_BD, -1)
    perp_BD = mid_BD - x - t_B[:, None] * w_BD
    assert jnp.all(jnp.linalg.norm(perp_BD, axis=-1) <= atol / math.sqrt(c))
    assert jnp.all((t_B >= -atol) & (t_B <= 1 + atol))


def test_einstein_midpoint_weights_permutation_and_uniform(manifold, points, c, rng, dtype, tolerance):
    atol, rtol = tolerance
    x_ND = points[:7]
    w_N = jnp.asarray(rng.uniform(0.1, 2.0, size=7).astype(_np_dtype(dtype)))
    perm = rng.permutation(7)
    mid_D = manifold.einstein_midpoint(x_ND, w_N, c)
    np.testing.assert_allclose(manifold.einstein_midpoint(x_ND[perm], w_N[perm], c), mid_D, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        manifold.einstein_midpoint(x_ND, None, c),
        manifold.einstein_midpoint(x_ND, jnp.ones(7, dtype), c),
        atol=atol,
        rtol=rtol,
    )
    # Only relative weights matter.
    np.testing.assert_allclose(manifold.einstein_midpoint(x_ND, 3.0 * w_N, c), mid_D, atol=atol, rtol=rtol)


def test_einstein_midpoint_matches_lorentz_centroid_oracle(manifold, points, c, rng, dtype, tolerance):
    atol, rtol = tolerance
    for start in (0, 5, 10):
        x_ND = points[start : start + 5]
        w_N = rng.uniform(0.1, 2.0, size=5)
        got_D = manifold.einstein_midpoint(x_ND, jnp.asarray(w_N.astype(_np_dtype(dtype))), c)
        ref_D = _np_lorentz_centroid_klein(np.asarray(x_ND, np.float64), w_N, c)
        np.testing.assert_allclose(np.asarray(got_D, np.float64), ref_D, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 8. Derivatives (float64, independent oracles)
# ---------------------------------------------------------------------------

_K64 = Klein(dtype=jnp.float64)


def _pair64(c: float, dim: int, seed: int, a_max: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Two float64 Klein points at scaled radius ≤ a_max (moderate: finite differences stay well conditioned)."""
    pts_BD = _sample_klein(c, dim, 2, np.random.default_rng(seed), a_max=a_max)
    return pts_BD[0], pts_BD[1]


def test_dist_grad_at_coincidence_is_zero(c):
    x_D, _ = _pair64(c, 5, 0)
    for base_D in (x_D, np.zeros(5)):
        base = jnp.asarray(base_D)
        grad_y = jax.grad(_K64.dist, argnums=1)(base, base, c)
        grad_x = jax.grad(_K64.dist, argnums=0)(base, base, c)
        assert jnp.all(jnp.isfinite(grad_y)) and jnp.array_equal(grad_y, jnp.zeros(5))
        assert jnp.all(jnp.isfinite(grad_x)) and jnp.array_equal(grad_x, jnp.zeros(5))


def test_dist_grad_at_origin_is_minus_unit_direction(c):
    """At x = 0 the Klein metric is the identity, so ∇_x d(x, y) = -ŷ exactly (unit Riemannian gradient)."""
    _, y_D = _pair64(c, 5, 1)
    grad_x = jax.grad(lambda x: _K64.dist(x, jnp.asarray(y_D), c))(jnp.zeros(5))
    np.testing.assert_allclose(np.asarray(grad_x), -y_D / np.linalg.norm(y_D), rtol=1e-12, atol=1e-14)


def test_expmap_jacobian_at_zero_is_identity(c):
    """``d exp_x(v)/dv |_{v=0} = I`` at every base point (the differential of exp at 0 is the identity)."""
    x_D, _ = _pair64(c, 5, 2)
    for base_D in (np.zeros(5), x_D):
        jac = jax.jacfwd(_K64.expmap, argnums=0)(jnp.zeros(5), jnp.asarray(base_D), c)
        np.testing.assert_allclose(np.asarray(jac), np.eye(5), atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jacfwd(lambda v: _K64.expmap_0(v, c))(jnp.zeros(5))), np.eye(5), atol=1e-12)


def test_logmap_jacobian_at_base_point_is_identity(c):
    """``d log_x(y)/dy |_{y=x} = I`` — the inverse of the exp differential."""
    x_D, _ = _pair64(c, 5, 3)
    for base_D in (np.zeros(5), x_D):
        base = jnp.asarray(base_D)
        jac = jax.jacfwd(_K64.logmap, argnums=0)(base, base, c)
        np.testing.assert_allclose(np.asarray(jac), np.eye(5), atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jacfwd(lambda y: _K64.logmap_0(y, c))(jnp.zeros(5))), np.eye(5), atol=1e-12)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_dist_grads_match_central_differences(c, seed):
    x_D, y_D = _pair64(c, 5, seed)
    x, y = jnp.asarray(x_D), jnp.asarray(y_D)
    checks = [
        (lambda p: _K64.dist(p, y, c), x_D),
        (lambda p: _K64.dist(x, p, c), y_D),
        (lambda p: _K64.dist_0(p, c), x_D),
    ]
    for f, at_D in checks:
        f_jit = jax.jit(f)
        got = np.asarray(jax.grad(f)(jnp.asarray(at_D)))
        np.testing.assert_allclose(got, _central_diff_grad(f_jit, at_D), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", [7, 8])
def test_dist_curvature_grad_matches_central_difference(c, seed):
    x, y = (jnp.asarray(p) for p in _pair64(c, 5, seed))
    f = jax.jit(lambda cc: _K64.dist(x, y, cc))
    got = float(jax.grad(f)(jnp.asarray(c, jnp.float64)))
    h = 1e-6 * c
    fd = (float(f(jnp.asarray(c + h))) - float(f(jnp.asarray(c - h)))) / (2 * h)
    np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-8)


def test_float32_grads_match_float64(c):
    """At moderate radius (a ≤ 1.5, pairs ≥ 0.2/√c apart) float32 gradients agree with float64 to 1e-4 relative."""
    k32 = Klein(dtype=jnp.float32)
    rng = np.random.default_rng(9)
    pts_BD = _sample_klein(c, 5, 64, rng, a_max=1.5).astype(np.float32)
    x32, y32 = jnp.asarray(pts_BD[:32]), jnp.asarray(pts_BD[32:])
    x64, y64 = x32.astype(jnp.float64), y32.astype(jnp.float64)
    keep = np.asarray(jax.vmap(_K64.dist, in_axes=(0, 0, None))(x64, y64, c)) >= 0.2 / math.sqrt(c)
    assert keep.sum() >= 16
    g_x32, g_y32 = jax.vmap(jax.grad(k32.dist, argnums=(0, 1)), in_axes=(0, 0, None))(x32, y32, c)
    g_x64, g_y64 = jax.vmap(jax.grad(_K64.dist, argnums=(0, 1)), in_axes=(0, 0, None))(x64, y64, c)
    g0_32 = jax.vmap(jax.grad(k32.dist_0), in_axes=(0, None))(x32, c)
    g0_64 = jax.vmap(jax.grad(_K64.dist_0), in_axes=(0, None))(x64, c)
    for g32, g64 in ((g_x32, g_x64), (g_y32, g_y64), (g0_32, g0_64)):
        g32_BD, g64_BD = np.asarray(g32, np.float64)[keep], np.asarray(g64)[keep]
        rel_B = np.linalg.norm(g32_BD - g64_BD, axis=-1) / np.linalg.norm(g64_BD, axis=-1)
        assert rel_B.max() <= 1e-4, rel_B.max()


# ---------------------------------------------------------------------------
# 8b. expmap: inward steps (x·v < 0)
# ---------------------------------------------------------------------------


def _np_expmap_oracle(x_D: np.ndarray, v_D: np.ndarray, c: float) -> np.ndarray:
    """Long-double hyperboloid geodesic of the lifted ``(x, v)``, mapped back with ``k = Y_s/(√c·Y₀)``.

    ``V = (√c(x·v)/g^{3/2}, v/√g + c(x·v)·x/g^{3/2})`` is the differential of :func:`_np_lift`, and
    ``exp_X(V) = cosh(θ)·X + sinh(θ)/θ·V`` with ``θ = √c·‖v‖_x`` from the literal Klein metric (the lift
    is an isometry).
    """
    ld = np.longdouble
    x, v, c_ld = x_D.astype(ld), v_D.astype(ld), ld(c)
    g = 1 - c_ld * (x @ x)
    xv = x @ v
    lift_x = np.concatenate([[1 / (np.sqrt(c_ld) * np.sqrt(g))], x / np.sqrt(g)])
    lift_v = np.concatenate([[np.sqrt(c_ld) * xv / g**1.5], v / np.sqrt(g) + c_ld * xv * x / g**1.5])
    theta = np.sqrt(c_ld) * np.sqrt((v @ v) / g + c_ld * xv**2 / g**2)
    y = np.cosh(theta) * lift_x + np.sinh(theta) / theta * lift_v
    return y[1:] / (np.sqrt(c_ld) * y[0])


def _inward_step(a: float, theta: float, phi: float, c: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Float64 base point at scaled radius ``a`` and a step of scaled length ``θ`` at ``φ`` radians off ``-x̂`` (dim 5)."""
    x_hat = rng.normal(size=5)
    x_hat /= np.linalg.norm(x_hat)
    n_D = rng.normal(size=5)
    n_D -= (n_D @ x_hat) * x_hat
    n_D /= np.linalg.norm(n_D)
    x_D = x_hat * math.tanh(a) / math.sqrt(c)
    u_D = -math.cos(phi) * x_hat + math.sin(phi) * n_D
    return x_D, u_D * (theta / (math.sqrt(c) * math.sqrt(_np_klein_inner(u_D, u_D, x_D, c))))


@pytest.mark.parametrize(("a", "theta"), [(4.0, 8.0), (3.0, 6.0)])
def test_expmap_inward_step_float32_matches_oracle(c, a, theta):
    """Float32 steps through the origin against the long-double geodesic of the same float32 inputs.

    Radial and near-radial (φ ≤ 3e-2) steps from scaled radius ``a`` land at ≈ ``θ - a`` on the far side,
    inside the float32 chart. For ``x·v < 0`` the literal denominator ``θ·coth θ + u`` (``u = c(x·v)/g_x``)
    is a difference of two ≈θ terms — and past θ ≈ 7.2 the saturated ``tanh`` inside ``_xcothx`` adds
    ``10·eps·θ`` to it: the worst step of this battery was 0.84 nats off at (a, θ) = (4, 8) and 2.7e-3
    at (3, 6). What remains is the chart's floor, the rounding of ``g = 1 - c‖x‖²`` (relative
    ``eps·cosh²(a)``, 8.9e-5 at a = 4): measured worst 5.1e-4 (5.7 floors) and 2.9e-5 (2.5 floors).
    Tolerance 16 floors.
    """
    k32 = Klein(dtype=jnp.float32)
    rng = np.random.default_rng(31)
    atol = 16 * float(jnp.finfo(jnp.float32).eps) * math.cosh(a) ** 2
    for phi in (0.0, 1e-3, 1e-2, 3e-2):
        x_D, v_D = _inward_step(a, theta, phi, c, rng)
        x32 = k32.proj(jnp.asarray(x_D, jnp.float32), c)
        v32 = jnp.asarray(v_D, jnp.float32)
        y_D = np.asarray(k32.expmap(v32, x32, c), np.float64)
        y_ref_D = _np_expmap_oracle(np.asarray(x32, np.float64), np.asarray(v32, np.float64), c)
        err = math.sqrt(c) * _np_dist(y_D, y_ref_D, c)
        assert err < atol, f"φ = {phi}: {err:.2e} nats (tolerance {atol:.1e})"


def test_expmap_outward_and_short_steps_float32_match_oracle(c):
    """Outward (``x·v > 0``) and short (θ ≤ 0.45) float32 steps against the long-double geodesic of the same inputs.

    They share the inward steps' denominator ``A + B`` (``B = θ + u`` a plain sum when outward), so the bound is
    the inward battery's: 16 chart floors ``eps·cosh²(a)``, with ``a`` the larger of the base and destination
    radius — for an outward step the destination, which stays inside the float32 chart (scaled radius ≤ 5).
    θ = 0.1 and 1e-3 fall below the float32 small-θ threshold 0.196, where ``A`` is its series. Measured worst
    0.69 floors (0.53 for the literal ``θ·coth θ + u`` these steps evaluated before, on the same inputs); over a
    random battery of 248 outward steps 0.88, against 1.5 for the literal (``logs/2026-09-29_cancellation-free/fixup/``).
    """
    k32 = Klein(dtype=jnp.float32)
    rng = np.random.default_rng(32)
    eps = float(jnp.finfo(jnp.float32).eps)
    outward = [(0.5, 3.0), (2.0, 1.5), (3.0, 2.0), (2.0, 0.1), (3.0, 1e-3)]  # (a, θ)
    inward = [(1.0, 0.3), (4.0, 0.45), (2.0, 0.1), (3.0, 1e-3)]
    for sign, cases in ((-1.0, outward), (1.0, inward)):  # sign -1 flips _inward_step's step outward
        for a, theta in cases:
            for phi in (0.0, 0.4, 1.2):
                x_D, v_D = _inward_step(a, theta, phi, c, rng)
                x32 = k32.proj(jnp.asarray(x_D, jnp.float32), c)
                v32 = jnp.asarray(sign * v_D, jnp.float32)
                y_D = np.asarray(k32.expmap(v32, x32, c), np.float64)
                y_ref_D = _np_expmap_oracle(np.asarray(x32, np.float64), np.asarray(v32, np.float64), c)
                a_dest = math.sqrt(c) * _np_dist(np.zeros(5), y_ref_D, c)
                atol = 16 * eps * math.cosh(max(a, a_dest)) ** 2
                err = math.sqrt(c) * _np_dist(y_D, y_ref_D, c)
                assert err < atol, f"(a, θ, sign, φ) = ({a}, {theta}, {sign}, {phi}): {err:.2e} nats (tol {atol:.1e})"


@pytest.mark.parametrize("theta", [3.0, 0.5])
def test_expmap_inward_grads_match_central_differences(c, theta):
    """Float64 derivatives of an inward step w.r.t. ``v``, ``x`` and ``c``, and of ``v = 0``, against central differences.

    θ = 3 runs the rewritten inward branch, θ = 0.5 sits on its switch. At ``v = 0`` the gradient is taken in
    reverse mode, where a NaN derivative of the untaken branch would reach the cotangent.
    """
    rng = np.random.default_rng(33)
    x_D, v_D = _inward_step(2.0, theta, 0.4, c, rng)
    w = jnp.asarray(rng.normal(size=5))
    x, v = jnp.asarray(x_D), jnp.asarray(v_D)
    checks = [
        (lambda p: w @ _K64.expmap(p, x, c), v_D),
        (lambda p: w @ _K64.expmap(v, p, c), x_D),
        (lambda p: w @ _K64.expmap(p, x, c), np.zeros(5)),
    ]
    for f, at_D in checks:
        got = np.asarray(jax.grad(f)(jnp.asarray(at_D)))
        np.testing.assert_allclose(got, _central_diff_grad(jax.jit(f), at_D), rtol=1e-6, atol=1e-8)
    f_c = jax.jit(lambda cc: w @ _K64.expmap(v, x, cc))
    h = 1e-6 * c
    fd = (float(f_c(jnp.asarray(c + h))) - float(f_c(jnp.asarray(c - h)))) / (2 * h)
    np.testing.assert_allclose(float(jax.grad(f_c)(jnp.asarray(c, jnp.float64))), fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    ("theta", "phi"),
    [(3.0, math.pi - 0.4), (1.4e-2, math.pi - 0.4), (3.5e-3, math.pi - 0.4), (1.0, math.pi / 2), (3.5e-3, math.pi / 2)],
)
def test_expmap_outward_and_switch_grads_match_central_differences(c, theta, phi):
    """Float64 derivatives of outward steps and of steps on the switch ``x·v = 0``, against central differences.

    The denominator ``A + B`` changes form twice: ``B = θ + u`` with the sign of ``x·v`` (φ = π/2 sits on it, so
    the central difference straddles both forms) and ``A`` at the float64 small-θ threshold 6.9e-3 (θ = 1.4e-2 and
    3.5e-3 lie on either side). φ = π - 0.4 is an outward step.
    """
    rng = np.random.default_rng(34)
    x_D, v_D = _inward_step(2.0, theta, phi, c, rng)
    w = jnp.asarray(rng.normal(size=5))
    x, v = jnp.asarray(x_D), jnp.asarray(v_D)
    for f, at_D in ((lambda p: w @ _K64.expmap(p, x, c), v_D), (lambda p: w @ _K64.expmap(v, p, c), x_D)):
        got = np.asarray(jax.grad(f)(jnp.asarray(at_D)))
        np.testing.assert_allclose(got, _central_diff_grad(jax.jit(f), at_D), rtol=1e-6, atol=1e-8)
    f_c = jax.jit(lambda cc: w @ _K64.expmap(v, x, cc))
    h = 1e-6 * c
    fd = (float(f_c(jnp.asarray(c + h))) - float(f_c(jnp.asarray(c - h)))) / (2 * h)
    np.testing.assert_allclose(float(jax.grad(f_c)(jnp.asarray(c, jnp.float64))), fd, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# 9. Chart ceiling
# ---------------------------------------------------------------------------


def test_expmap0_huge_vector_saturates_at_margin(manifold, c, dim, rng, dtype):
    """A huge tangent vector lands on the proj margin √c‖x‖ = 1 - √c·m (m = eps**0.75), finite.

    There ``dist_0 = artanh(z)/√c`` with ``z = √c‖x‖ = 1 - √c·m``. The stored ``z`` and the computed gap
    ``1 - z² ≈ 2√c·m`` each carry absolute rounding ~eps, i.e. relative eps/(√c·m) = eps**0.25/√c on the
    gap (``m`` is an absolute margin, so the gap shrinks with √c), which moves
    ``dist_0 ≈ ½·ln(2/gap)/√c`` by ≈ eps**0.25/(2c) per eps of rounding. Measured: 6.1e-4 in float64 at
    c = 0.1 (= 1.0 x eps**0.25/(2c)). Tolerance: ``2·eps**0.25/c`` (4 eps of rounding) — 3.7e-2/c in
    float32, 2.4e-4/c in float64, against dist_0 ≈ 20/√c resp. 45/√c there.
    """
    eps = float(jnp.finfo(dtype).eps)
    sqrt_c = math.sqrt(c)
    d_D = rng.normal(size=dim)
    v_D = jnp.asarray((1e3 / sqrt_c * d_D / np.linalg.norm(d_D)).astype(_np_dtype(dtype)))
    x_D = manifold.expmap_0(v_D, c)
    assert bool(jnp.all(jnp.isfinite(x_D))) and bool(manifold.is_in_manifold(x_D, c))
    np.testing.assert_allclose(float(jnp.linalg.norm(x_D)), 1.0 / sqrt_c - _margin(dtype), rtol=8 * eps)
    expected_d0 = math.atanh(1.0 - sqrt_c * _margin(dtype)) / sqrt_c
    atol = 2.0 * eps**0.25 / c
    np.testing.assert_allclose(float(manifold.dist_0(x_D, c)), expected_d0, atol=atol, rtol=0)
    log_D = manifold.logmap_0(x_D, c)
    assert bool(jnp.all(jnp.isfinite(log_D)))
    np.testing.assert_allclose(float(jnp.linalg.norm(log_D)), expected_d0, atol=atol, rtol=0)
    # Direction survives the saturation.
    np.testing.assert_allclose(
        np.asarray(log_D / jnp.linalg.norm(log_D)), np.asarray(v_D / jnp.linalg.norm(v_D)), atol=8 * eps**0.5
    )
    # A huge step from a non-origin base point also stays finite and in the ball.
    base_D = jnp.asarray((0.5 / sqrt_c * np.ones(dim) / math.sqrt(dim)).astype(_np_dtype(dtype)))
    far_D = manifold.expmap(v_D, base_D, c)
    assert bool(jnp.all(jnp.isfinite(far_D))) and bool(manifold.is_in_manifold(far_D, c))


# ---------------------------------------------------------------------------
# 10. Protocol & integration
# ---------------------------------------------------------------------------


def test_klein_satisfies_manifold_protocol(dtype):
    assert isinstance(Klein(dtype=dtype), Manifold)


@pytest.mark.parametrize(("manifold_dtype", "input_dtype"), [(jnp.float32, jnp.float64), (jnp.float64, jnp.float32)])
def test_dtype_casting(manifold_dtype, input_dtype):
    m = Klein(dtype=manifold_dtype)
    x = jnp.asarray([0.1, -0.2, 0.3], dtype=input_dtype)
    y = jnp.asarray([-0.3, 0.05, 0.2], dtype=input_dtype)
    outputs = [
        m.proj(x, 1.0),
        m.addition(x, y, 1.0),
        m.gyro_difference(x, y, 1.0),
        m.scalar_mul(0.5, x, 1.0),
        m.dist(x, y, 1.0),
        m.dist_0(x, 1.0),
        m.expmap(y, x, 1.0),
        m.expmap_0(y, 1.0),
        m.logmap(y, x, 1.0),
        m.logmap_0(y, 1.0),
        m.ptransp(y, x, y, 1.0),
        m.ptransp_0(y, x, 1.0),
        m.tangent_inner(x, y, x, 1.0),
        m.tangent_norm(y, x, 1.0),
        m.egrad2rgrad(y, x, 1.0),
        m.lorentz_factor(x, 1.0),
        m.einstein_midpoint(jnp.stack([x, y]), jnp.asarray([1.0, 2.0], dtype=input_dtype), 1.0),
    ]
    for out in outputs:
        assert out.dtype == jnp.dtype(manifold_dtype)


def test_jit_vmap_matches_eager(manifold, points, c, tolerance):
    atol, rtol = tolerance
    x, y, z = (p[:8] for p in _split3(points))
    ops = {
        "dist": (manifold.dist, (x, y), (0, 0, None)),
        "expmap": (manifold.expmap, (0.3 * z, x), (0, 0, None)),
        "logmap": (manifold.logmap, (y, x), (0, 0, None)),
        "ptransp": (manifold.ptransp, (z, x, y), (0, 0, 0, None)),
    }
    for name, (fn, args, in_axes) in ops.items():
        compiled = jax.jit(jax.vmap(fn, in_axes=in_axes))(*args, c)
        eager = jnp.stack([fn(*(a[i] for a in args), c) for i in range(8)])
        np.testing.assert_allclose(compiled, eager, atol=atol, rtol=rtol, err_msg=name)


def test_version_idx_accepted_and_ignored(manifold, points, c):
    x, y, _ = _split3(points)
    x0, y0 = x[0], y[0]
    assert manifold.VERSION_DEFAULT == 0
    assert jnp.array_equal(manifold.dist(x0, y0, c, version_idx=0), manifold.dist(x0, y0, c))
    assert jnp.array_equal(manifold.dist(x0, y0, c, version_idx=3), manifold.dist(x0, y0, c))
    assert jnp.array_equal(manifold.dist_0(x0, c, version_idx=1), manifold.dist_0(x0, c))


def test_klein_as_product_factor(c, dtype, tolerance):
    atol, rtol = tolerance
    product = ProductManifold((Klein(dtype=dtype), 3), (Poincare(dtype=dtype), 2), (Euclidean(dtype=dtype), 2), dtype=dtype)
    cs = (c, 1.0, 0.0)
    origin = product.origin(cs)
    assert jnp.array_equal(origin, jnp.zeros(7, dtype))
    v = jnp.asarray([0.4, -0.2, 0.1, 0.3, 0.2, 1.0, -2.0], dtype=dtype)
    x = product.expmap_0(v, cs)
    assert bool(product.is_in_manifold(x, cs))
    np.testing.assert_allclose(x[:3], Klein(dtype=dtype).expmap_0(v[:3], c), atol=atol, rtol=rtol)
    y = product.expmap_0(jnp.asarray([-0.3, 0.5, 0.2, -0.1, 0.4, 0.5, 1.5], dtype=dtype), cs)
    comp = product.component_dist(x, y, cs)
    np.testing.assert_allclose(comp[0], Klein(dtype=dtype).dist(x[:3], y[:3], c), atol=atol, rtol=rtol)
    np.testing.assert_allclose(product.dist(x, y, cs), jnp.sqrt(jnp.sum(comp**2)), atol=atol, rtol=rtol)
    np.testing.assert_allclose(product.logmap_0(x, cs), v, atol=atol, rtol=rtol)
