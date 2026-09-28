"""Tests for the Poincaré upper half-space manifold (:class:`~hyperbolix.manifolds.HalfSpace`).

Oracle / round-trip / consistency batteries. The oracles at the top of the file are written from the
textbook formulas in NumPy and never call the library: the literal distance
``arcosh(1 + ‖x - y‖²/(2x_n y_n))/√c`` in long double, the literal metric ``(u·v)/(c·x_n²)``, the
gyro-inverse ``⊖x = (-x_s, x_n)/(c‖x‖²)``, and Möbius addition carried into the half-space by the
isometry through the hyperboloid (``o = e_n/√c`` corresponds to the ball's origin). Points are built
in NumPy from a hyperboloid-style polar parametrization (scaled radius ``a`` from ``o``, uniform
direction), so ``dist_0`` has the exact reference ``a/√c``.

The half-space has two exact isometry families that act linearly on chart coordinates — dilations
``x ↦ λx`` and horizontal translations ``x ↦ x + (b, 0)`` — and its exp/log/transport do not depend
on ``c`` in chart coordinates. Sections 4 and 9 use them as oracles: a dilation by a power of 4 is
exact in floating point (``√λ`` too), so those checks are tight.

Fixtures ``dtype``, ``seed_jax``, ``rng``, ``tolerance`` come from ``tests/conftest.py`` (which also
enables float64).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.manifolds import Euclidean, HalfSpace, Poincare, ProductManifold
from hyperbolix.manifolds.protocol import Manifold

# Dimension key:
#   B: batch of points (a slice of the ``points`` fixture)
#   D: manifold dimension (``dim``); the height ``x_n`` is the last coordinate

# Largest scaled geodesic radius a = √c·d(o, x) of the sampled points. The storage floor of a stored point is
# ≈ 0.4·(eps/2)·cosh(√c·δ)/√c as a distance (δ ≤ d(o, x) its distance to the vertical axis through o):
# ≤ 2e-7/√c nats in float32 at a = 3, far below the shared 4e-3 tolerance even for results at radius 6.
A_MAX = 3.0


# ---------------------------------------------------------------------------
# Independent oracles (NumPy; never call the library)
# ---------------------------------------------------------------------------


def _np_dist(x_D: np.ndarray, y_D: np.ndarray, c: float) -> float:
    """Literal ``arcosh(1 + ‖x - y‖²/(2x_n y_n))/√c`` in long double."""
    ld = np.longdouble
    x_ld, y_ld, c_ld = x_D.astype(ld), y_D.astype(ld), ld(c)
    w = y_ld - x_ld
    return float(np.arccosh(1 + (w @ w) / (2 * x_ld[-1] * y_ld[-1])) / np.sqrt(c_ld))


def _np_gyro_inverse(x_BD, c: float) -> np.ndarray:
    """``⊖x = (-x_s, x_n)/(c‖x‖²)`` in float64 (the image of ``-p`` in the ball)."""
    x64_BD = np.asarray(x_BD, np.float64)
    flip_BD = np.concatenate([-x64_BD[..., :-1], x64_BD[..., -1:]], axis=-1)
    return flip_BD / (c * np.sum(x64_BD**2, axis=-1, keepdims=True))


def _np_halfspace_to_ball(x_D: np.ndarray, c: float) -> np.ndarray:
    """Half-space -> Poincaré ball through the hyperboloid, ``o = e_n/√c ↦ 0`` (long double).

    With ``o`` as above and ``q = ‖x + o‖²``: ``p_s = 2x_s/(c·q)``, ``p_n = (‖x‖² - 1/c)/(√c·q)``.
    """
    ld = np.longdouble
    x, c_ld = x_D.astype(ld), ld(c)
    sc = np.sqrt(c_ld)
    x_plus_o = x.copy()
    x_plus_o[-1] += 1 / sc
    q = x_plus_o @ x_plus_o
    return np.concatenate([2 * x[:-1] / (c_ld * q), [((x @ x) - 1 / c_ld) / (sc * q)]])


def _np_ball_to_halfspace(p_D: np.ndarray, c: float) -> np.ndarray:
    """Inverse of :func:`_np_halfspace_to_ball`: ``x_n = g/(√c·‖√c·p - e_n‖²)``, ``x_s = 2p_s/‖√c·p - e_n‖²``."""
    ld = np.longdouble
    p, c_ld = p_D.astype(ld), ld(c)
    sc = np.sqrt(c_ld)
    g = 1 - c_ld * (p @ p)
    diff = sc * p
    diff[-1] -= 1
    q = diff @ diff
    return np.concatenate([2 * p[:-1] / q, [g / (sc * q)]])


def _np_mobius_add(x_D: np.ndarray, y_D: np.ndarray, c) -> np.ndarray:
    """Literal Möbius addition on the ball (Ungar 2009), in the dtype of its inputs."""
    xy, xx, yy = x_D @ y_D, x_D @ x_D, y_D @ y_D
    num = (1 + 2 * c * xy + c * yy) * x_D + (1 - c * xx) * y_D
    return num / (1 + 2 * c * xy + c**2 * xx * yy)


def _np_mobius_add_halfspace(x_D: np.ndarray, y_D: np.ndarray, c: float) -> np.ndarray:
    ld = np.longdouble
    p = _np_mobius_add(_np_halfspace_to_ball(x_D, c), _np_halfspace_to_ball(y_D, c), ld(c))
    return _np_ball_to_halfspace(p, c).astype(np.float64)


def _np_htorch_ptransp_vertical(v_D: np.ndarray, x_D: np.ndarray, y_D: np.ndarray, c: float) -> np.ndarray:
    """HTorch's ``v - (⟨log_x y, v⟩_x/d²)(log_x y + log_y x)`` for a vertical pair (``x_s = y_s``).

    On the vertical geodesic ``log_x y = (0, x_n·ln(y_n/x_n))`` and ``log_y x = (0, y_n·ln(x_n/y_n))`` in
    closed form, and ``d = |ln(y_n/x_n)|/√c``.
    """
    ln_r = math.log(y_D[-1] / x_D[-1])
    log_xy = np.zeros_like(v_D)
    log_xy[-1] = x_D[-1] * ln_r
    log_yx = np.zeros_like(v_D)
    log_yx[-1] = -y_D[-1] * ln_r
    inner = (log_xy @ v_D) / (c * x_D[-1] ** 2)
    return v_D - inner / (ln_r**2 / c) * (log_xy + log_yx)


def _central_diff_grad(f, x: np.ndarray, h: float) -> np.ndarray:
    """Central finite-difference gradient of a scalar function of a float64 vector."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        grad[i] = (float(f(jnp.asarray(x + e))) - float(f(jnp.asarray(x - e)))) / (2 * h)
    return grad


def _central_jacobian(f, x: np.ndarray, h: float) -> np.ndarray:
    """Central finite-difference Jacobian ``J[i, j] = ∂f_i/∂x_j`` of a vector function of a float64 vector."""
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        cols.append((np.asarray(f(jnp.asarray(x + e))) - np.asarray(f(jnp.asarray(x - e)))) / (2 * h))
    return np.stack(cols, axis=-1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[0.1, 0.5, 1.0, 4.0], ids=["c0.1", "c0.5", "c1", "c4"])
def c(request: pytest.FixtureRequest) -> float:
    """Curvature magnitude (sectional curvature -c); the origin height 1/√c spans 3.16 to 0.5."""
    return float(request.param)


@pytest.fixture(params=[1, 2, 10], ids=["dim1", "dim2", "dim10"])
def dim(request: pytest.FixtureRequest) -> int:
    """1 (the hyperbolic line: ``x_s`` empty, every op on the vertical axis), 2 (the half-plane), one generic width."""
    return int(request.param)


@pytest.fixture
def manifold(dtype: jnp.dtype) -> HalfSpace:
    return HalfSpace(dtype=dtype)


def _np_dtype(dtype) -> np.dtype:
    return np.dtype(jnp.dtype(dtype).name)


def _sample_halfspace(
    c: float, dim: int, n: int, rng: np.random.Generator, a_max: float = A_MAX
) -> tuple[np.ndarray, np.ndarray]:
    """Float64 half-space points at scaled radius a ~ U(0, a_max) from ``o``, uniform direction ``u``.

    The hyperboloid point ``(cosh a, u·sinh a)/√c`` mapped by ``x_n = 1/(c(X₀ - X_n))``, ``x_s = √c·x_n·X_s``:
    ``x_n = 1/(√c·(cosh a - u_n sinh a))``, ``x_s = x_n·u_s·sinh a``. Returns the points and ``a``.
    """
    dirs_BD = rng.normal(size=(n, dim))
    dirs_BD /= np.linalg.norm(dirs_BD, axis=-1, keepdims=True)
    a_B = rng.uniform(0.0, a_max, size=n)
    a_B1 = a_B[:, None]
    x_n_B1 = 1.0 / (math.sqrt(c) * (np.cosh(a_B1) - dirs_BD[:, -1:] * np.sinh(a_B1)))
    x_s_BD = x_n_B1 * dirs_BD[:, :-1] * np.sinh(a_B1)
    return np.concatenate([x_s_BD, x_n_B1], axis=-1), a_B


@pytest.fixture
def points(c: float, dim: int, rng: np.random.Generator, dtype: jnp.dtype) -> jnp.ndarray:
    """288 on-manifold points for the given (curvature, dim, dtype)."""
    pts_BD, _ = _sample_halfspace(c, dim, 288, rng)
    return jnp.asarray(pts_BD.astype(_np_dtype(dtype)))


def _split3(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = points.shape[0] // 3
    return points[:n], points[n : 2 * n], points[2 * n : 3 * n]


def _origin(c: float, dim: int, dtype) -> jnp.ndarray:
    """``o = e_n/√c`` spelled as the library documents it (``o_n = 1/sqrt(c)`` in the dtype)."""
    return jnp.zeros((dim,), dtype).at[-1].set(1.0 / jnp.sqrt(jnp.asarray(c, dtype)))


def _tangent_vectors(x_BD, c: float, rng: np.random.Generator, r_lo: float, r_hi: float) -> jnp.ndarray:
    """Random tangent vectors at each ``x`` with metric norm ``‖v‖/(√c·x_n) ~ U(r_lo, r_hi)`` (NumPy float64)."""
    x64_BD = np.asarray(x_BD, dtype=np.float64)
    u_BD = rng.normal(size=x64_BD.shape)
    u_BD /= np.linalg.norm(u_BD, axis=-1, keepdims=True)
    r_B1 = rng.uniform(r_lo, r_hi, size=(x64_BD.shape[0], 1))
    return jnp.asarray((u_BD * r_B1 * math.sqrt(c) * x64_BD[:, -1:]).astype(np.asarray(x_BD).dtype))


def _batch_in_manifold(manifold: HalfSpace, pts_BD: jnp.ndarray, c: float) -> bool:
    return bool(jnp.all(jax.vmap(lambda p: manifold.is_in_manifold(p, c))(pts_BD)))


def _assert_tangent_close(manifold, got_BD, want_BD, x_BD, c, atol, rtol):
    """``‖got - want‖_x ≤ atol + rtol·‖want‖_x``: tangent errors measured in the metric at the base point."""
    tnorm = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))
    err_B = tnorm(got_BD - want_BD, x_BD, c)
    bound_B = atol + rtol * tnorm(want_BD, x_BD, c)
    assert jnp.all(err_B <= bound_B), float(jnp.max(err_B - bound_B))


# ---------------------------------------------------------------------------
# 1. Projection & membership
# ---------------------------------------------------------------------------


def test_proj_leaves_valid_points_unchanged(manifold, points, c, dtype):
    """Every finite height ≥ ``finfo.tiny`` is left bit-identical, so proj commutes with dilations of valid points."""
    proj = jax.vmap(manifold.proj, in_axes=(0, None))
    assert jnp.array_equal(proj(points, c), points)
    fi = jnp.finfo(dtype)
    heights = jnp.asarray([float(fi.tiny), 1e-30, 1.0, 1e30, float(fi.max)], dtype)
    extreme_BD = points[:5].at[:, -1].set(heights)
    assert jnp.array_equal(proj(extreme_BD, c), extreme_BD)
    for lam in (1e-20, 1e20):
        scaled_BD = points * jnp.asarray(lam, dtype)
        assert jnp.array_equal(proj(scaled_BD, c), scaled_BD)


def test_proj_floors_nonpositive_heights(manifold, points, c, dtype):
    fi = jnp.finfo(dtype)
    tiny = float(fi.tiny)
    heights = jnp.asarray([0.0, -0.0, -1.0, -1e30, -jnp.inf, tiny / 4], dtype)  # tiny/4 is subnormal
    raw_BD = points[:6].at[:, -1].set(heights)
    out_BD = jax.vmap(manifold.proj, in_axes=(0, None))(raw_BD, c)
    assert jnp.array_equal(out_BD[:, -1], jnp.full((6,), tiny, dtype))
    assert jnp.array_equal(out_BD[:, :-1], raw_BD[:, :-1])
    assert _batch_in_manifold(manifold, out_BD, c)
    # NaN passes through (loud divergence): in the height and in the horizontal part.
    nan_height = manifold.proj(points[0].at[-1].set(jnp.nan), c)
    assert bool(jnp.isnan(nan_height[-1])) and jnp.array_equal(nan_height[:-1], points[0, :-1])
    if points.shape[-1] > 1:
        nan_s = manifold.proj(points[0].at[0].set(jnp.nan), c)
        assert bool(jnp.isnan(nan_s[0])) and jnp.array_equal(nan_s[1:], points[0, 1:])


def test_is_in_manifold_accepts_and_rejects(manifold, points, c, dtype):
    assert _batch_in_manifold(manifold, points, c)
    x = points[0]
    assert bool(manifold.is_in_manifold(x.at[-1].set(float(jnp.finfo(dtype).tiny)), c))
    for bad in (0.0, -0.0, -1.0, jnp.inf, -jnp.inf, jnp.nan):
        assert not bool(manifold.is_in_manifold(x.at[-1].set(bad), c)), bad
    if points.shape[-1] > 1:
        for bad in (jnp.inf, jnp.nan):
            assert not bool(manifold.is_in_manifold(x.at[0].set(bad), c)), bad
    # atol is accepted and unused: the constraint x_n > 0 is exact.
    assert bool(manifold.is_in_manifold(x, c, atol=0.0)) and bool(manifold.is_in_manifold(x, c, atol=1e3))
    assert not bool(manifold.is_in_manifold(x.at[-1].set(-1e-3), c, atol=1e3))


def test_proj_batch_matches_vmap_proj(manifold, points, c, dtype):
    # Mix valid points with zero / negative heights, and use two leading dims.
    sign_B = jnp.where(jnp.arange(points.shape[0]) % 3 == 0, -1.0, 1.0).astype(dtype)
    raw_BD = points.at[:, -1].multiply(sign_B).at[1::7, -1].set(0.0)
    expected_BD = jax.vmap(manifold.proj, in_axes=(0, None))(raw_BD, c)
    got = manifold.proj_batch(raw_BD.reshape(2, -1, raw_BD.shape[-1]), c).reshape(raw_BD.shape)
    assert jnp.array_equal(got, expected_BD)


# ---------------------------------------------------------------------------
# 2. Gyrovector algebra
# ---------------------------------------------------------------------------


def test_addition_identity_and_inverse(manifold, points, c, dtype, tolerance):
    """``o`` is a two-sided identity and the NumPy ``⊖x`` a two-sided inverse."""
    atol, rtol = tolerance
    patol = atol / math.sqrt(c)  # point coordinates carry the length unit 1/√c
    add = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    o_BD = jnp.broadcast_to(_origin(c, points.shape[-1], dtype), points.shape)
    np.testing.assert_allclose(add(o_BD, points, c), points, atol=patol, rtol=rtol)
    np.testing.assert_allclose(add(points, o_BD, c), points, atol=patol, rtol=rtol)
    inv_BD = jnp.asarray(_np_gyro_inverse(points, c).astype(_np_dtype(dtype)))
    np.testing.assert_allclose(add(inv_BD, points, c), o_BD, atol=patol, rtol=rtol)
    np.testing.assert_allclose(add(points, inv_BD, c), o_BD, atol=patol, rtol=rtol)


def test_gyro_difference_matches_addition_of_inverse(manifold, points, c, dtype, tolerance):
    atol, rtol = tolerance
    patol = atol / math.sqrt(c)
    x, y, _ = _split3(points)
    gd = jax.vmap(manifold.gyro_difference, in_axes=(0, 0, None))
    inv_x = jnp.asarray(_np_gyro_inverse(x, c).astype(_np_dtype(dtype)))
    np.testing.assert_allclose(
        gd(x, y, c), jax.vmap(manifold.addition, in_axes=(0, 0, None))(inv_x, y, c), atol=patol, rtol=rtol
    )
    # (⊖x) ⊕ x = o exactly: log_x x = 0, transport of 0 is 0, exp_o(0) = o.
    assert jnp.array_equal(gd(x, x, c), jnp.broadcast_to(_origin(c, x.shape[-1], dtype), x.shape))


def test_gyro_translation_is_isometry(manifold, points, c, tolerance):
    """``d(o, (⊖x) ⊕ y) = d(x, y)``: left gyro-translation is an isometry."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    gd_BD = jax.vmap(manifold.gyro_difference, in_axes=(0, 0, None))(x, y, c)
    np.testing.assert_allclose(
        jax.vmap(manifold.dist_0, in_axes=(0, None))(gd_BD, c),
        jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, c),
        atol=atol,
        rtol=rtol,
    )


def test_addition_left_cancellation(manifold, points, c, dtype, tolerance):
    """``⊖x ⊕ (x ⊕ y) = y`` — the gyrogroup left-cancellation law, via ``addition`` and ``gyro_difference``."""
    atol, rtol = tolerance
    patol = atol / math.sqrt(c)
    x, y, _ = _split3(points)
    add = jax.vmap(manifold.addition, in_axes=(0, 0, None))
    xy_BD = add(x, y, c)
    assert _batch_in_manifold(manifold, xy_BD, c)
    inv_x = jnp.asarray(_np_gyro_inverse(x, c).astype(_np_dtype(dtype)))
    np.testing.assert_allclose(add(inv_x, xy_BD, c), y, atol=patol, rtol=rtol)
    np.testing.assert_allclose(jax.vmap(manifold.gyro_difference, in_axes=(0, 0, None))(x, xy_BD, c), y, atol=patol, rtol=rtol)


def test_addition_matches_mobius_through_ball_isometry(manifold, points, c, dtype, tolerance):
    """``x ⊕ y`` = image of the literal Möbius sum of the ball preimages (long double oracle).

    Möbius addition commutes with rotations about the ball's origin, so any isometry taking ``0`` to ``o``
    carries the same operation; the one through the hyperboloid is used.
    """
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    got_BD = np.asarray(jax.vmap(manifold.addition, in_axes=(0, 0, None))(x, y, c), np.float64)
    c_in = float(_np_dtype(dtype).type(c))
    x64_BD, y64_BD = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ref_BD = np.stack([_np_mobius_add_halfspace(a, b, c_in) for a, b in zip(x64_BD, y64_BD, strict=True)])
    np.testing.assert_allclose(got_BD, ref_BD, atol=atol / math.sqrt(c), rtol=rtol)


def test_scalar_mul_identity_zero_negation(manifold, points, c, dtype, tolerance):
    atol, rtol = tolerance
    patol = atol / math.sqrt(c)
    smul = jax.vmap(manifold.scalar_mul, in_axes=(None, 0, None))
    np.testing.assert_allclose(smul(1.0, points, c), points, atol=patol, rtol=rtol)
    o_BD = jnp.broadcast_to(_origin(c, points.shape[-1], dtype), points.shape)
    assert jnp.array_equal(smul(0.0, points, c), o_BD)  # exp_o(0·v) = exp_o(0) = o exactly
    inv_BD = _np_gyro_inverse(points, c)
    np.testing.assert_allclose(np.asarray(smul(-1.0, points, c), np.float64), inv_BD, atol=patol, rtol=rtol)


def test_scalar_mul_associativity(manifold, points, c, tolerance):
    """``(r1·r2) ⊗ x = r1 ⊗ (r2 ⊗ x)``; |r2|, |r1·r2| ≤ 1.5 keep every result at a ≤ 4.5."""
    atol, rtol = tolerance
    smul = jax.vmap(manifold.scalar_mul, in_axes=(None, 0, None))
    for r1, r2 in ((0.5, 1.5), (-0.7, 0.4), (1.2, -1.1)):
        np.testing.assert_allclose(
            smul(r1 * r2, points, c), smul(r1, smul(r2, points, c), c), atol=atol / math.sqrt(c), rtol=rtol
        )


def test_scalar_mul_scales_origin_distance(manifold, points, c, tolerance):
    atol, rtol = tolerance
    smul = jax.vmap(manifold.scalar_mul, in_axes=(None, 0, None))
    dist_0 = jax.vmap(manifold.dist_0, in_axes=(0, None))
    d0_B = dist_0(points, c)
    for r in (0.3, -0.8, 1.5):
        np.testing.assert_allclose(dist_0(smul(r, points, c), c), abs(r) * d0_B, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 3. Distance, exp/log round trips
# ---------------------------------------------------------------------------


def test_dist_matches_acosh_oracle(manifold, points, c, dtype, tolerance):
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    got_B = jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, c)
    c_in = float(_np_dtype(dtype).type(c))
    ref_B = np.array([_np_dist(a, b, c_in) for a, b in zip(np.asarray(x), np.asarray(y), strict=True)])
    np.testing.assert_allclose(np.asarray(got_B, np.float64), ref_B, atol=atol, rtol=rtol)


def test_dist_0_matches_sampled_radius(c, dim, rng, dtype, tolerance):
    """``dist_0`` of a point built at scaled radius ``a`` is ``a/√c`` (the sampler's own parameter)."""
    atol, rtol = tolerance
    manifold = HalfSpace(dtype=dtype)
    pts_BD, a_B = _sample_halfspace(c, dim, 96, rng)
    got_B = jax.vmap(manifold.dist_0, in_axes=(0, None))(jnp.asarray(pts_BD.astype(_np_dtype(dtype))), c)
    np.testing.assert_allclose(np.asarray(got_B, np.float64), a_B / math.sqrt(c), atol=atol, rtol=rtol)


def test_dist_symmetric_and_dist_0_is_dist_from_origin(manifold, points, c, dtype):
    eps = float(jnp.finfo(dtype).eps)
    x, y, _ = _split3(points)
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    np.testing.assert_allclose(dist(x, y, c), dist(y, x, c), rtol=8 * eps, atol=0)
    o_BD = jnp.broadcast_to(_origin(c, x.shape[-1], dtype), x.shape)
    np.testing.assert_allclose(jax.vmap(manifold.dist_0, in_axes=(0, None))(x, c), dist(o_BD, x, c), rtol=4 * eps, atol=0)
    # Documented exact values at the origin.
    o_D = _origin(c, x.shape[-1], dtype)
    assert float(manifold.dist_0(o_D, c)) == 0.0
    assert jnp.array_equal(manifold.logmap_0(o_D, c), jnp.zeros_like(o_D))
    assert jnp.array_equal(manifold.expmap_0(jnp.zeros_like(o_D), c), o_D)


def test_expmap0_logmap0_round_trips(manifold, points, c, dtype, rng, tolerance):
    atol, rtol = tolerance
    exp0 = jax.vmap(manifold.expmap_0, in_axes=(0, None))
    log0 = jax.vmap(manifold.logmap_0, in_axes=(0, None))
    np.testing.assert_allclose(exp0(log0(points, c), c), points, atol=atol / math.sqrt(c), rtol=rtol)
    o_BD = jnp.broadcast_to(_origin(c, points.shape[-1], dtype), points.shape)
    v_BD = _tangent_vectors(o_BD, c, rng, 0.01, 2.0)
    _assert_tangent_close(manifold, log0(exp0(v_BD, c), c), v_BD, o_BD, c, atol, rtol)


def test_expmap_logmap_round_trips(manifold, points, c, rng, tolerance):
    """``exp_x(log_x y) = y`` and ``log_x(exp_x v) = v``; tangent errors measured in the metric at x."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    log = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    np.testing.assert_allclose(exp(log(y, x, c), x, c), y, atol=atol / math.sqrt(c), rtol=rtol)
    v_BD = _tangent_vectors(x, c, rng, 0.01, 1.5)
    _assert_tangent_close(manifold, log(exp(v_BD, x, c), x, c), v_BD, x, c, atol, rtol)


def test_maps_at_origin_match_origin_variants(manifold, points, c, dtype, rng, tolerance):
    atol, rtol = tolerance
    o_BD = jnp.broadcast_to(_origin(c, points.shape[-1], dtype), points.shape)
    v_BD = _tangent_vectors(o_BD, c, rng, 0.01, 2.0)
    np.testing.assert_allclose(
        jax.vmap(manifold.expmap, in_axes=(0, 0, None))(v_BD, o_BD, c),
        jax.vmap(manifold.expmap_0, in_axes=(0, None))(v_BD, c),
        atol=atol / math.sqrt(c),
        rtol=rtol,
    )
    _assert_tangent_close(
        manifold,
        jax.vmap(manifold.logmap, in_axes=(0, 0, None))(points, o_BD, c),
        jax.vmap(manifold.logmap_0, in_axes=(0, None))(points, c),
        o_BD,
        c,
        atol,
        rtol,
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
    v_BD = _tangent_vectors(x, c, rng, 0.01, 1.5)
    np.testing.assert_allclose(
        dist(x, jax.vmap(manifold.expmap, in_axes=(0, 0, None))(v_BD, x, c), c), tnorm(v_BD, x, c), atol=atol, rtol=rtol
    )


def test_logmap_at_base_point_is_exactly_zero(manifold, points, c):
    log_BD = jax.vmap(manifold.logmap, in_axes=(0, 0, None))(points, points, c)
    assert jnp.array_equal(log_BD, jnp.zeros_like(points))
    assert jnp.array_equal(
        jax.vmap(manifold.dist, in_axes=(0, 0, None))(points, points, c), jnp.zeros(points.shape[0], points.dtype)
    )


# ---------------------------------------------------------------------------
# 4. Exact isometries of the chart: dilations, horizontal translations, c-independence
# ---------------------------------------------------------------------------


def test_dilation_invariance_and_equivariance(manifold, points, c, rng, dtype):
    """``x ↦ λx`` is an isometry: ``d`` invariant, exp/log/transport equivariant (tangents scale by λ).

    ``λ = 4^k`` scales every intermediate exactly (``√λ = 2^k`` too), so the results agree to a few ulps.
    """
    rtol = 4 * float(jnp.finfo(dtype).eps)
    x, y, _ = _split3(points)
    v_BD = _tangent_vectors(x, c, rng, 0.1, 1.5)
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    log = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    transport = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    for lam in (4.0**3, 4.0**-3):
        lx, ly, lv = lam * x, lam * y, lam * v_BD
        np.testing.assert_allclose(dist(lx, ly, c), dist(x, y, c), rtol=rtol, atol=0)
        np.testing.assert_allclose(exp(lv, lx, c), lam * exp(v_BD, x, c), rtol=rtol, atol=0)
        np.testing.assert_allclose(log(ly, lx, c), lam * log(y, x, c), rtol=rtol, atol=0)
        np.testing.assert_allclose(transport(lv, lx, ly, c), lam * transport(v_BD, x, y, c), rtol=rtol, atol=0)


def test_horizontal_translation_invariance_and_equivariance(manifold, points, c, rng, dtype, tolerance):
    """``x ↦ x + (b, 0)`` is an isometry: ``d`` invariant, ``exp`` shifts by ``(b, 0)``, log/transport unchanged."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    dim = x.shape[-1]
    b_D = jnp.asarray(np.append(rng.normal(size=dim - 1) * 2.0 / math.sqrt(c), 0.0).astype(_np_dtype(dtype)))
    v_BD = _tangent_vectors(x, c, rng, 0.1, 1.5)
    w_BD = _tangent_vectors(x, c, rng, 0.1, 1.5)
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    log = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    transport = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    bx, by = x + b_D, y + b_D
    np.testing.assert_allclose(dist(bx, by, c), dist(x, y, c), atol=atol, rtol=rtol)
    np.testing.assert_allclose(exp(v_BD, bx, c), exp(v_BD, x, c) + b_D, atol=atol / math.sqrt(c), rtol=rtol)
    _assert_tangent_close(manifold, log(by, bx, c), log(y, x, c), x, c, atol, rtol)
    _assert_tangent_close(manifold, transport(w_BD, bx, by, c), transport(w_BD, x, y, c), y, c, atol, rtol)


def test_chart_maps_are_curvature_independent(manifold, points, c, rng):
    """exp/log/transport/retraction are ``c``-free in chart coordinates: identical outputs for two curvatures."""
    x, y, _ = _split3(points)
    v_BD = _tangent_vectors(x, c, rng, 0.1, 1.5)
    c_other = 2.5 * c + 0.3
    for name, fn, args, in_axes in (
        ("expmap", manifold.expmap, (v_BD, x), (0, 0, None)),
        ("logmap", manifold.logmap, (y, x), (0, 0, None)),
        ("ptransp", manifold.ptransp, (v_BD, x, y), (0, 0, 0, None)),
        ("retraction", manifold.retraction, (v_BD, x), (0, 0, None)),
    ):
        vf = jax.vmap(fn, in_axes=in_axes)
        assert jnp.array_equal(vf(*args, c), vf(*args, c_other)), name


# ---------------------------------------------------------------------------
# 5. Closed-form geodesics
# ---------------------------------------------------------------------------


def test_vertical_geodesic_closed_form(manifold, points, c, dtype):
    """``exp_x((0, θ·x_n)) = (x_s, x_n·e^θ)`` for up- and downward steps; the distance travelled is ``|θ|/√c``."""
    eps = float(jnp.finfo(dtype).eps)
    x = points[:32]
    x_n64 = np.asarray(x[:, -1], np.float64)
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    for theta in (0.1, 1.0, 3.0, -0.1, -1.0, -3.0):
        v_BD = jnp.zeros_like(x).at[:, -1].set(theta * x[:, -1])
        y_BD = exp(v_BD, x, c)
        assert jnp.array_equal(y_BD[:, :-1], x[:, :-1])
        v_n64 = np.asarray(v_BD[:, -1], np.float64)  # the stored step, θ·x_n rounded
        np.testing.assert_allclose(np.asarray(y_BD[:, -1], np.float64), x_n64 * np.exp(v_n64 / x_n64), rtol=8 * eps)
        np.testing.assert_allclose(
            jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y_BD, c), abs(theta) / math.sqrt(c), rtol=16 * eps
        )


def test_vertical_distance_closed_form(manifold, c, dim, rng, dtype):
    """``d((x_s, a), (x_s, b)) = |ln(b/a)|/√c``, for height ratios up to e^±20."""
    eps = float(jnp.finfo(dtype).eps)
    np_dt = _np_dtype(dtype)
    a_B = np.geomspace(1e-3, 1e3, 12).astype(np_dt)
    b_B = (a_B.astype(np.float64) * np.exp(rng.uniform(-20.0, 20.0, size=12))).astype(np_dt)
    xs_BD = np.tile(rng.normal(size=(1, dim - 1)), (12, 1)).astype(np_dt)
    x_BD = jnp.asarray(np.concatenate([xs_BD, a_B[:, None]], -1))
    y_BD = jnp.asarray(np.concatenate([xs_BD, b_B[:, None]], -1))
    ref_B = np.abs(np.log(b_B.astype(np.float64) / a_B.astype(np.float64))) / math.sqrt(c)
    np.testing.assert_allclose(
        np.asarray(jax.vmap(manifold.dist, in_axes=(0, 0, None))(x_BD, y_BD, c), np.float64), ref_B, rtol=16 * eps
    )


def test_unit_semicircle_closed_form(dtype):
    """From ``(0, 1)`` with ``v = (t, 0)`` the geodesic is the unit semicircle: ``exp = (tanh t, sech t)``.

    The chart maps are ``c``-free, so the point is the same at every curvature; the distance is ``|t|/√c``.
    """
    eps = float(jnp.finfo(dtype).eps)
    m = HalfSpace(dtype=dtype)
    x = jnp.asarray([0.0, 1.0], dtype)
    for t in (0.3, 1.0, 2.5, -1.5):
        v = jnp.asarray([t, 0.0], dtype)
        for c in (1.0, 0.5):
            y = m.expmap(v, x, c)
            np.testing.assert_allclose(np.asarray(y, np.float64), [math.tanh(t), 1.0 / math.cosh(t)], rtol=8 * eps)
            np.testing.assert_allclose(float(m.dist(x, y, c)), abs(t) / math.sqrt(c), rtol=16 * eps)


# ---------------------------------------------------------------------------
# 6. Parallel transport
# ---------------------------------------------------------------------------


def test_ptransp_bilinear_isometry(manifold, points, c, rng, tolerance):
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
    _assert_tangent_close(manifold, transport(transport(u_BD, x, y, c), y, x, c), u_BD, x, c, atol, rtol)


def test_ptransp_carries_log_to_negative_reverse_log(manifold, points, c, tolerance):
    """``P_{x→y}(log_x y) = -log_y x``: the geodesic's velocity is parallel along it."""
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    log = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    moved_BD = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))(log(y, x, c), x, y, c)
    _assert_tangent_close(manifold, moved_BD, -log(x, y, c), y, c, atol, rtol)


def test_ptransp_from_origin_and_to_same_point(manifold, points, c, dtype, rng, tolerance):
    atol, rtol = tolerance
    o_BD = jnp.broadcast_to(_origin(c, points.shape[-1], dtype), points.shape)
    v_BD = _tangent_vectors(o_BD, c, rng, 0.1, 2.0)
    general = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))(v_BD, o_BD, points, c)
    _assert_tangent_close(
        manifold, general, jax.vmap(manifold.ptransp_0, in_axes=(0, 0, None))(v_BD, points, c), points, c, atol, rtol
    )
    # P_{x→x} is exactly the identity. Not in float32 on GPU: XLA:GPU's float32 divide is approximate (y_n/x_n ≠ 1).
    if jax.default_backend() != "cpu" and dtype == jnp.float32:
        pytest.skip("float32 bit-exact equality is CPU-only")
    u_BD = _tangent_vectors(points, c, rng, 0.1, 2.0)
    assert jnp.array_equal(jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))(u_BD, points, points, c), u_BD)


def test_ptransp_vertical_is_height_ratio_scaling(manifold, points, c, rng, dtype):
    """Along a vertical geodesic every vector is transported to ``(y_n/x_n)·v`` (horizontal ones included).

    HTorch's transport returns a horizontal ``v`` unchanged; this pins the correct value.
    """
    eps = float(jnp.finfo(dtype).eps)
    x = points[:32]
    y = x.at[:, -1].multiply(jnp.asarray(np.exp(rng.uniform(-3.0, 3.0, size=32)).astype(_np_dtype(dtype))))
    v_BD = _tangent_vectors(x, c, rng, 0.1, 2.0)
    got_BD = np.asarray(jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))(v_BD, x, y, c), np.float64)
    ratio_B1 = np.asarray(y[:, -1:], np.float64) / np.asarray(x[:, -1:], np.float64)
    np.testing.assert_allclose(got_BD, ratio_B1 * np.asarray(v_BD, np.float64), rtol=4 * eps, atol=0)


def test_ptransp_htorch_formula_negative_control(dtype):
    """HTorch's ``v - (⟨log_x y, v⟩_x/d²)(log_x y + log_y x)`` is not an isometry in the chart.

    ``(1, 0)`` at ``(0, 1)`` to ``(0, 2)`` at ``c = 1``: the formula returns ``(1, 0)`` (norm 0.5 at ``(0, 2)``),
    the transport is ``(2, 0)`` (norm 1).
    """
    m = HalfSpace(dtype=dtype)
    x_D, y_D, v_D = np.array([0.0, 1.0]), np.array([0.0, 2.0]), np.array([1.0, 0.0])
    htorch_D = _np_htorch_ptransp_vertical(v_D, x_D, y_D, 1.0)
    np.testing.assert_allclose(htorch_D, [1.0, 0.0])
    assert math.isclose(np.linalg.norm(htorch_D) / y_D[-1], 0.5)
    got = m.ptransp(jnp.asarray(v_D, dtype), jnp.asarray(x_D, dtype), jnp.asarray(y_D, dtype), 1.0)
    assert jnp.array_equal(got, jnp.asarray([2.0, 0.0], dtype))
    assert float(m.tangent_norm(got, jnp.asarray(y_D, dtype), 1.0)) == 1.0


# ---------------------------------------------------------------------------
# 7. Metric, tangent space, retraction
# ---------------------------------------------------------------------------


def test_tangent_inner_matches_literal_metric(manifold, points, c, rng, dtype, tolerance):
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    np_dt = _np_dtype(dtype)
    u_BD = jnp.asarray(rng.normal(size=x.shape).astype(np_dt))
    v_BD = jnp.asarray(rng.normal(size=x.shape).astype(np_dt))
    inner = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))
    got_B = np.asarray(inner(u_BD, v_BD, x, c), np.float64)
    x64_BD, u64_BD, v64_BD = (np.asarray(t, np.float64) for t in (x, u_BD, v_BD))
    ref_B = np.sum(u64_BD * v64_BD, -1) / (c * x64_BD[:, -1] ** 2)
    # Scale the tolerance by |u|·|v|/(c x_n²): the dot product can cancel.
    scale_B = np.linalg.norm(u64_BD, axis=-1) * np.linalg.norm(v64_BD, axis=-1) / (c * x64_BD[:, -1] ** 2)
    assert np.all(np.abs(got_B - ref_B) <= atol + rtol * scale_B)
    assert jnp.all(inner(u_BD, u_BD, x, c) > 0)  # positive definite


def test_tangent_norm_squared_is_inner(manifold, points, c, rng, dtype, tolerance):
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    v_BD = jnp.asarray(rng.normal(size=x.shape).astype(_np_dtype(dtype)))
    norm_B = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))(v_BD, x, c)
    np.testing.assert_allclose(
        norm_B**2, jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))(v_BD, v_BD, x, c), atol=atol, rtol=rtol
    )


def test_egrad2rgrad_metric_duality(manifold, points, c, rng, dtype, tolerance):
    """``g_x(egrad2rgrad(e), v) = e·v`` for arbitrary ``v``, and ``egrad2rgrad(e) = c·x_n²·e``."""
    atol, rtol = tolerance
    x, _, _ = _split3(points)
    np_dt = _np_dtype(dtype)
    e_BD = jnp.asarray(rng.normal(size=x.shape).astype(np_dt))
    v_BD = jnp.asarray(rng.normal(size=x.shape).astype(np_dt))
    rgrad_BD = jax.vmap(manifold.egrad2rgrad, in_axes=(0, 0, None))(e_BD, x, c)
    lhs_B = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))(rgrad_BD, v_BD, x, c)
    scale_B = jnp.linalg.norm(e_BD, axis=-1) * jnp.linalg.norm(v_BD, axis=-1)
    assert jnp.all(jnp.abs(lhs_B - jnp.einsum("bd,bd->b", e_BD, v_BD)) <= atol + rtol * scale_B)
    x64_BD = np.asarray(x, np.float64)
    np.testing.assert_allclose(
        np.asarray(rgrad_BD, np.float64), c * x64_BD[:, -1:] ** 2 * np.asarray(e_BD, np.float64), atol=0, rtol=rtol
    )


def test_tangent_proj_and_tangent_space_membership(manifold, points, c, rng, dtype):
    """``T_x H = R^n``: tangent_proj is the identity, membership is finiteness."""
    x, _, _ = _split3(points)
    v_BD = jnp.asarray(rng.normal(size=x.shape).astype(_np_dtype(dtype))) * 1e3
    assert jnp.array_equal(jax.vmap(manifold.tangent_proj, in_axes=(0, 0, None))(v_BD, x, c), v_BD)
    assert bool(jnp.all(jax.vmap(lambda vv, xx: manifold.is_in_tangent_space(vv, xx, c))(v_BD, x)))
    for bad in (jnp.nan, jnp.inf, -jnp.inf):
        assert not bool(manifold.is_in_tangent_space(v_BD[0].at[-1].set(bad), x[0], c)), bad


def test_retraction_stays_on_manifold_and_is_exact_vertically(manifold, points, c, rng, dtype):
    """Large downward steps leave ``x + v`` off the half-space but not the retraction; vertical steps equal expmap."""
    eps = float(jnp.finfo(dtype).eps)
    x, _, _ = _split3(points)
    retr = jax.vmap(manifold.retraction, in_axes=(0, 0, None))
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    # Downward step of 5 heights plus a random horizontal part: x + v has x_n - 5x_n < 0.
    v_BD = _tangent_vectors(x, c, rng, 0.1, 2.0).at[:, -1].set(-5.0 * x[:, -1])
    assert not bool(jnp.any((x + v_BD)[:, -1] > 0))
    out_BD = retr(v_BD, x, c)
    assert _batch_in_manifold(manifold, out_BD, c)
    np.testing.assert_allclose(out_BD[:, -1], x[:, -1] * jnp.exp(-5.0), rtol=8 * eps)
    for theta in (-10.0, -1.0, 0.5, 3.0):
        vert_BD = jnp.zeros_like(x).at[:, -1].set(theta * x[:, -1])
        r_BD, e_BD = retr(vert_BD, x, c), exp(vert_BD, x, c)
        assert jnp.array_equal(r_BD[:, :-1], e_BD[:, :-1])
        np.testing.assert_allclose(r_BD[:, -1], e_BD[:, -1], rtol=8 * eps, atol=0)


def test_retraction_agrees_with_expmap_to_first_order(c):
    """``d(R_x(tv), exp_x(tv)) = O(t²)``: halving ``t`` divides the gap by ≈ 4 (float64)."""
    h64 = HalfSpace(dtype=jnp.float64)
    rng = np.random.default_rng(11)
    pts_BD, _ = _sample_halfspace(c, 4, 16, rng, a_max=2.0)
    x = jnp.asarray(pts_BD)
    v_BD = _tangent_vectors(x, c, rng, 0.5, 1.0)
    gaps = []
    for t in (0.1, 0.05, 0.025):
        r_BD = jax.vmap(h64.retraction, in_axes=(0, 0, None))(t * v_BD, x, c)
        e_BD = jax.vmap(h64.expmap, in_axes=(0, 0, None))(t * v_BD, x, c)
        gaps.append(np.asarray(jax.vmap(h64.dist, in_axes=(0, 0, None))(r_BD, e_BD, c)))
    ratio1_B, ratio2_B = gaps[0] / gaps[1], gaps[1] / gaps[2]
    assert np.all((ratio1_B > 3.0) & (ratio1_B < 5.0)), ratio1_B
    assert np.all((ratio2_B > 3.5) & (ratio2_B < 4.5)), ratio2_B


# ---------------------------------------------------------------------------
# 8. Derivatives (float64, independent oracles)
# ---------------------------------------------------------------------------

_H64 = HalfSpace(dtype=jnp.float64)


def _pair64(c: float, dim: int, seed: int, a_max: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Two float64 half-space points at scaled radius ≤ a_max (finite differences stay well conditioned)."""
    pts_BD, _ = _sample_halfspace(c, dim, 2, np.random.default_rng(seed), a_max=a_max)
    return pts_BD[0], pts_BD[1]


def _o64(c: float, dim: int) -> np.ndarray:
    return np.asarray(_origin(c, dim, jnp.float64))


def test_dist_grad_at_coincidence_is_zero(c):
    x_D, _ = _pair64(c, 5, 0)
    for base_D in (x_D, _o64(c, 5)):
        base = jnp.asarray(base_D)
        grad_y = jax.grad(_H64.dist, argnums=1)(base, base, c)
        grad_x = jax.grad(_H64.dist, argnums=0)(base, base, c)
        assert jnp.all(jnp.isfinite(grad_y)) and jnp.array_equal(grad_y, jnp.zeros(5))
        assert jnp.all(jnp.isfinite(grad_x)) and jnp.array_equal(grad_x, jnp.zeros(5))
    grad_0 = jax.grad(_H64.dist_0)(jnp.asarray(_o64(c, 5)), c)
    assert jnp.all(jnp.isfinite(grad_0)) and jnp.array_equal(grad_0, jnp.zeros(5))


def _bases(c: float, dtype, seed: int) -> jnp.ndarray:
    """``o`` plus 24 sampled points at a ≤ 2, in ``dtype``.

    Enough bases that a Jacobian which is exact only where ``x_n·fl(1/x_n) = 1`` (true for roughly 7 heights
    in 8) cannot pass by luck.
    """
    pts_BD, _ = _sample_halfspace(c, 5, 24, np.random.default_rng(seed), a_max=2.0)
    bases_BD = np.concatenate([_o64(c, 5)[None], pts_BD]).astype(_np_dtype(dtype))
    return jnp.asarray(bases_BD)


def _assert_all_equal(jac_BDD, want_DD: np.ndarray, what: str, atol: float = 0.0) -> None:
    got_BDD = np.asarray(jac_BDD)
    bad_B = ~np.all(np.abs(got_BDD - want_DD.astype(got_BDD.dtype)) <= atol, axis=(-2, -1))
    if bad_B.any():
        example = np.diag(got_BDD[bad_B][0]).tolist()
        raise AssertionError(f"{what}: off by more than {atol} at {int(bad_B.sum())}/{bad_B.size} bases, e.g. diag {example}")


def _expmap_dv_modes(dtype) -> tuple:
    """(name, jacobian transform, atol) for the ``d exp_x(v)/dv`` checks (on CPU; see the float32 GPU skip below).

    Reverse mode is exactly ``I``: the unit cotangent comes back as ``(1·x_n)/x_n = 1``. Forward mode is ``I`` to
    one ulp: the tangent of ``p = v/x_n`` is ``fl(1/x_n)``, multiplied back by ``x_n``, and ``x_n·fl(1/x_n) ≠ 1``
    at about one height in eight (``x_n = 49``: ``1 - 2⁻⁵³``). No spelling built on ``p = v/x_n`` avoids that
    without forming ``x_n²``, which leaves the dtype at heights ``1e±19`` in float32.
    """
    return (("jacfwd", jax.jacfwd, float(np.finfo(_np_dtype(dtype)).eps)), ("jacrev", jax.jacrev, 0.0))


def test_expmap_jacobian_at_zero_step_is_identity(c):
    """``d exp_x(v)/dv |_{v=0} = I``: exactly in reverse mode, to one ulp in forward mode, at ``o`` and generic points."""
    bases_BD = _bases(c, jnp.float64, 2)
    zero_D = jnp.zeros(5)
    for name, jac_fn, atol in _expmap_dv_modes(jnp.float64):
        jac_BDD = jax.vmap(jac_fn(_H64.expmap, argnums=0), in_axes=(None, 0, None))(zero_D, bases_BD, c)
        _assert_all_equal(jac_BDD, np.eye(5), f"expmap dv, {name}", atol)
        _assert_all_equal(jac_fn(lambda v: _H64.expmap_0(v, c))(zero_D)[None], np.eye(5), f"expmap_0 dv, {name}", atol)


@pytest.mark.parametrize(("step_dtype", "step"), [(jnp.float64, 1e-300), (jnp.float32, 1e-40)])
def test_expmap_jacobian_at_tiny_vertical_step_is_identity(c, step_dtype, step):
    """A step far below ``MIN_NORM`` along ``e_n`` stays on the direct branch: Jacobian ``I`` (as at ``v = 0``), not NaN.

    (float32 ``1e-40`` is subnormal; either it survives or flushes to the ``v = 0`` case.) The finiteness check runs
    everywhere; the float32 comparison with ``I`` is CPU-only, since XLA:GPU's float32 divide is approximate.
    """
    m = HalfSpace(dtype=step_dtype)
    bases_BD = _bases(c, step_dtype, 3)
    v_D = jnp.zeros(5, step_dtype).at[-1].set(step)
    jacs = []
    for name, jac_fn, atol in _expmap_dv_modes(step_dtype):
        jac_BDD = jax.vmap(jac_fn(m.expmap, argnums=0), in_axes=(None, 0, None))(v_D, bases_BD, c)
        assert bool(jnp.all(jnp.isfinite(jac_BDD))), name
        jacs.append((name, jac_BDD, atol))
    if jax.default_backend() != "cpu" and step_dtype == jnp.float32:
        pytest.skip("float32 bit-exact equality is CPU-only")
    for name, jac_BDD, atol in jacs:
        _assert_all_equal(jac_BDD, np.eye(5), f"expmap dv at v = {step}·e_n, {name}", atol)


def test_expmap_jacobian_wrt_base_at_zero_step_is_identity(c):
    bases_BD = _bases(c, jnp.float64, 4)
    for name, jac_fn in (("jacfwd", jax.jacfwd), ("jacrev", jax.jacrev)):
        jac_BDD = jax.vmap(jac_fn(_H64.expmap, argnums=1), in_axes=(None, 0, None))(jnp.zeros(5), bases_BD, c)
        _assert_all_equal(jac_BDD, np.eye(5), f"expmap dx, {name}")


def test_logmap_jacobian_at_base_point(c):
    """``d log_x(y)/dy |_{y=x} = I`` and ``d log_x(y)/dx |_{y=x} = -I`` exactly, forward and reverse mode."""
    bases_BD = _bases(c, jnp.float64, 5)
    for name, jac_fn in (("jacfwd", jax.jacfwd), ("jacrev", jax.jacrev)):
        jac_y_BDD = jax.vmap(jac_fn(_H64.logmap, argnums=0), in_axes=(0, 0, None))(bases_BD, bases_BD, c)
        jac_x_BDD = jax.vmap(jac_fn(_H64.logmap, argnums=1), in_axes=(0, 0, None))(bases_BD, bases_BD, c)
        _assert_all_equal(jac_y_BDD, np.eye(5), f"logmap dy, {name}")
        _assert_all_equal(jac_x_BDD, -np.eye(5), f"logmap dx, {name}")


@pytest.mark.parametrize("seed", [6, 7, 8])
def test_dist_grads_match_central_differences(c, seed):
    x_D, y_D = _pair64(c, 5, seed)
    x, y = jnp.asarray(x_D), jnp.asarray(y_D)
    h = 1e-6 / math.sqrt(c)
    checks = [
        (lambda p: _H64.dist(p, y, c), x_D),
        (lambda p: _H64.dist(x, p, c), y_D),
        (lambda p: _H64.dist_0(p, c), x_D),
    ]
    for f, at_D in checks:
        got = np.asarray(jax.grad(f)(jnp.asarray(at_D)))
        np.testing.assert_allclose(got, _central_diff_grad(jax.jit(f), at_D, h), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", [9, 10])
def test_logmap_jacobians_match_central_differences(c, seed):
    x_D, y_D = _pair64(c, 5, seed)
    x, y = jnp.asarray(x_D), jnp.asarray(y_D)
    h = 1e-6 / math.sqrt(c)
    for f, at_D in ((lambda p: _H64.logmap(p, x, c), y_D), (lambda p: _H64.logmap(y, p, c), x_D)):
        got = np.asarray(jax.jacfwd(f)(jnp.asarray(at_D)))
        np.testing.assert_allclose(got, _central_jacobian(jax.jit(f), at_D, h), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", [12, 13])
def test_dist_curvature_grad_matches_central_difference(c, seed):
    """``∂d/∂c`` vs a central difference, and the closed form ``-d/(2c)`` (``d ∝ 1/√c`` in the fixed chart)."""
    x, y = (jnp.asarray(p) for p in _pair64(c, 5, seed))
    f = jax.jit(lambda cc: _H64.dist(x, y, cc))
    got = float(jax.grad(f)(jnp.asarray(c, jnp.float64)))
    h = 1e-6 * c
    fd = (float(f(jnp.asarray(c + h))) - float(f(jnp.asarray(c - h)))) / (2 * h)
    np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got, -float(f(jnp.asarray(c))) / (2 * c), rtol=1e-12)


def test_float32_grads_match_float64(c):
    """At moderate radius (a ≤ 1.5, pairs ≥ 0.2/√c apart) float32 gradients agree with float64 to 1e-4 relative."""
    h32 = HalfSpace(dtype=jnp.float32)
    rng = np.random.default_rng(14)
    pts_BD, _ = _sample_halfspace(c, 5, 64, rng, a_max=1.5)
    pts_BD = pts_BD.astype(np.float32)
    x32, y32 = jnp.asarray(pts_BD[:32]), jnp.asarray(pts_BD[32:])
    x64, y64 = x32.astype(jnp.float64), y32.astype(jnp.float64)
    keep = np.asarray(jax.vmap(_H64.dist, in_axes=(0, 0, None))(x64, y64, c)) >= 0.2 / math.sqrt(c)
    assert keep.sum() >= 16
    g_x32, g_y32 = jax.vmap(jax.grad(h32.dist, argnums=(0, 1)), in_axes=(0, 0, None))(x32, y32, c)
    g_x64, g_y64 = jax.vmap(jax.grad(_H64.dist, argnums=(0, 1)), in_axes=(0, 0, None))(x64, y64, c)
    g0_32 = jax.vmap(jax.grad(h32.dist_0), in_axes=(0, None))(x32, c)
    g0_64 = jax.vmap(jax.grad(_H64.dist_0), in_axes=(0, None))(x64, c)
    for g32, g64 in ((g_x32, g_x64), (g_y32, g_y64), (g0_32, g0_64)):
        g32_BD, g64_BD = np.asarray(g32, np.float64)[keep], np.asarray(g64)[keep]
        rel_B = np.linalg.norm(g32_BD - g64_BD, axis=-1) / np.linalg.norm(g64_BD, axis=-1)
        assert rel_B.max() <= 1e-4, rel_B.max()


# ---------------------------------------------------------------------------
# 9. Chart range: extreme heights
# ---------------------------------------------------------------------------


def _extreme_scales(dtype) -> tuple[float, float]:
    """Dilations whose squares over/underflow the dtype: 1e±30 in float32 (1e60 > 3.4e38), 1e±200 in float64."""
    return (1e30, 1e-30) if jnp.dtype(dtype) == jnp.float32 else (1e200, 1e-200)


def test_vertical_axis_at_extreme_heights(manifold, c, dim, dtype):
    """``d((0, s), (0, r·s)) = |ln r|/√c`` at heights ``s`` whose products ``s²`` over- or underflow.

    Storage has no radius ceiling along the vertical axis, and the distance is built from the scaled chord
    ``((y - x)/√x_n)/√y_n`` with sequential divisions, so ``x_n·y_n`` (1e±60 in float32) is never formed.
    """
    eps = float(jnp.finfo(dtype).eps)
    for s in _extreme_scales(dtype):
        for ratio in (3.0, math.exp(5.0), math.exp(-12.0)):
            x = jnp.zeros((dim,), dtype).at[-1].set(s)
            y = jnp.zeros((dim,), dtype).at[-1].set(s * ratio)
            y_n64 = float(np.asarray(y[-1], np.float64))
            ref = abs(math.log(y_n64 / float(np.asarray(x[-1], np.float64)))) / math.sqrt(c)
            got = float(manifold.dist(x, y, c))
            assert math.isfinite(got)
            np.testing.assert_allclose(got, ref, rtol=16 * eps)


def test_dilation_invariance_at_extreme_scales(manifold, points, c, rng, dtype, tolerance):
    """``d(λx, λy) = d(x, y)`` and exp/log/transport/norm equivariance at ``λ = 1e±30`` (float32) / ``1e±200`` (float64).

    Pins the overflow-free spellings: sequential divisions by ``√x_n``, ``√y_n`` in ``dist``/``logmap``, the
    ratio ``y_n/x_n`` in ``ptransp``, ``(u/x_n)·(v/x_n)`` in ``tangent_inner`` — ``x_n·y_n`` or ``x_n²`` at these
    scales would be inf/0. Every value must be finite; ``λ`` is not a power of 2, so the fixture tolerance applies.
    """
    atol, rtol = tolerance
    x, y, _ = _split3(points)
    v_BD = _tangent_vectors(x, c, rng, 0.1, 1.5)
    dist = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    exp = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
    log = jax.vmap(manifold.logmap, in_axes=(0, 0, None))
    transport = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    tnorm = jax.vmap(manifold.tangent_norm, in_axes=(0, 0, None))
    inner = jax.vmap(manifold.tangent_inner, in_axes=(0, 0, 0, None))
    for lam in _extreme_scales(dtype):
        lx, ly, lv = lam * x, lam * y, lam * v_BD
        outs = {
            "dist": (dist(lx, ly, c), dist(x, y, c)),
            "expmap": (exp(lv, lx, c) / lam, exp(v_BD, x, c)),
            "logmap": (log(ly, lx, c) / lam, log(y, x, c)),
            "ptransp": (transport(lv, lx, ly, c) / lam, transport(v_BD, x, y, c)),
            "tangent_norm": (tnorm(lv, lx, c), tnorm(v_BD, x, c)),
            "tangent_inner": (inner(lv, lv, lx, c), inner(v_BD, v_BD, x, c)),
        }
        for name, (got, want) in outs.items():
            assert bool(jnp.all(jnp.isfinite(got))), (name, lam)
            np.testing.assert_allclose(got, want, atol=atol / math.sqrt(c), rtol=rtol, err_msg=f"{name} λ={lam}")


# ---------------------------------------------------------------------------
# 10. Protocol & integration
# ---------------------------------------------------------------------------


def test_halfspace_satisfies_manifold_protocol(dtype):
    assert isinstance(HalfSpace(dtype=dtype), Manifold)


@pytest.mark.parametrize(
    ("manifold_dtype", "input_dtype"),
    [(jnp.float32, jnp.float64), (jnp.float64, jnp.float32), (jnp.float32, jnp.float32)],
)
def test_dtype_casting(manifold_dtype, input_dtype):
    """Outputs follow the manifold dtype; float32 stays float32 under x64 (``c`` a Python float)."""
    m = HalfSpace(dtype=manifold_dtype)
    x = jnp.asarray([0.1, -0.2, 0.8], dtype=input_dtype)
    y = jnp.asarray([-0.3, 0.05, 1.7], dtype=input_dtype)
    outputs = [
        m.proj(x, 1.0),
        m.proj_batch(jnp.stack([x, y]), 1.0),
        m.addition(x, y, 1.0),
        m.gyro_difference(x, y, 1.0),
        m.scalar_mul(0.5, x, 1.0),
        m.dist(x, y, 1.0),
        m.dist_0(x, 1.0),
        m.expmap(y, x, 1.0),
        m.expmap_0(y, 1.0),
        m.retraction(y, x, 1.0),
        m.logmap(y, x, 1.0),
        m.logmap_0(y, 1.0),
        m.ptransp(y, x, y, 1.0),
        m.ptransp_0(y, x, 1.0),
        m.tangent_inner(x, y, x, 1.0),
        m.tangent_norm(y, x, 1.0),
        m.egrad2rgrad(y, x, 1.0),
        m.tangent_proj(y, x, 1.0),
    ]
    for i, out in enumerate(outputs):
        assert out.dtype == jnp.dtype(manifold_dtype), i


def test_jit_vmap_matches_eager(manifold, points, c, tolerance):
    atol, rtol = tolerance
    x, y, z = (p[:8] for p in _split3(points))
    v = 0.3 * z / z[:, -1:] * x[:, -1:]  # tangent vectors of moderate metric size at x
    ops = {
        "dist": (manifold.dist, (x, y), (0, 0, None)),
        "dist_0": (manifold.dist_0, (x,), (0, None)),
        "addition": (manifold.addition, (x, y), (0, 0, None)),
        "gyro_difference": (manifold.gyro_difference, (x, y), (0, 0, None)),
        "expmap": (manifold.expmap, (v, x), (0, 0, None)),
        "logmap": (manifold.logmap, (y, x), (0, 0, None)),
        "ptransp": (manifold.ptransp, (v, x, y), (0, 0, 0, None)),
    }
    for name, (fn, args, in_axes) in ops.items():
        compiled = jax.jit(jax.vmap(fn, in_axes=in_axes))(*args, c)
        eager = jnp.stack([fn(*(a[i] for a in args), c) for i in range(8)])
        np.testing.assert_allclose(compiled, eager, atol=atol / math.sqrt(c), rtol=rtol, err_msg=name)


def test_version_idx_accepted_and_ignored(manifold, points, c):
    x, y, _ = _split3(points)
    x0, y0 = x[0], y[0]
    assert manifold.VERSION_DEFAULT == 0
    assert jnp.array_equal(manifold.dist(x0, y0, c, version_idx=0), manifold.dist(x0, y0, c))
    assert jnp.array_equal(manifold.dist(x0, y0, c, version_idx=7), manifold.dist(x0, y0, c))
    assert jnp.array_equal(manifold.dist_0(x0, c, version_idx=1), manifold.dist_0(x0, c))


def test_implicit_batch_matches_per_row(manifold, points, c):
    """``proj``/``expmap_0``/``logmap_0``/``scalar_mul`` accept a ``(B, dim)`` array and equal the per-row calls.

    CPU only: on GPU the ``(B, dim)`` program and the per-row one are compiled separately and are not bit-identical
    (at dim 10, in float32 or float64 depending on the launch).
    """
    if jax.default_backend() != "cpu":
        pytest.skip("bit-equality across two separately compiled reduction trees is CPU-only")
    x = points[:16]
    v = 0.5 * (x - x[::-1])
    cases = {
        "proj": (manifold.proj(x.at[::3, -1].set(-1.0), c), [manifold.proj(r, c) for r in x.at[::3, -1].set(-1.0)]),
        "expmap_0": (manifold.expmap_0(v, c), [manifold.expmap_0(r, c) for r in v]),
        "logmap_0": (manifold.logmap_0(x, c), [manifold.logmap_0(r, c) for r in x]),
        "scalar_mul": (manifold.scalar_mul(-0.7, x, c), [manifold.scalar_mul(-0.7, r, c) for r in x]),
    }
    for name, (batched, rows) in cases.items():
        assert batched.shape == x.shape, name
        assert jnp.array_equal(batched, jnp.stack(rows)), name


def test_halfspace_as_product_factor(c, dtype, tolerance):
    atol, rtol = tolerance
    h = HalfSpace(dtype=dtype)
    product = ProductManifold((h, 3), (Poincare(dtype=dtype), 2), (Euclidean(dtype=dtype), 2), dtype=dtype)
    cs = (c, 1.0, 0.0)
    origin = product.origin(cs)
    assert jnp.array_equal(origin[:3], _origin(c, 3, dtype))
    assert jnp.array_equal(origin[3:], jnp.zeros(4, dtype))
    v = jnp.asarray([0.4, -0.2, 0.1, 0.3, 0.2, 1.0, -2.0], dtype=dtype)  # the metric at o is the identity
    x = product.expmap_0(v, cs)
    assert bool(product.is_in_manifold(x, cs))
    np.testing.assert_allclose(x[:3], h.expmap_0(v[:3], c), atol=atol, rtol=rtol)
    y = product.expmap_0(jnp.asarray([-0.3, 0.5, 0.2, -0.1, 0.4, 0.5, 1.5], dtype=dtype), cs)
    comp = product.component_dist(x, y, cs)
    np.testing.assert_allclose(comp[0], h.dist(x[:3], y[:3], c), atol=atol, rtol=rtol)
    np.testing.assert_allclose(product.dist(x, y, cs), jnp.sqrt(jnp.sum(comp**2)), atol=atol, rtol=rtol)
    np.testing.assert_allclose(product.logmap_0(x, cs), v, atol=atol, rtol=rtol)
