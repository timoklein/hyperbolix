"""Tests for isometry mappings between hyperbolic manifold models.

Covers the distance-preserving transformations among the Poincaré ball,
hyperboloid (Lorentz), Proper Velocity (PV), and Beltrami-Klein models:

    Poincaré ↔ Hyperboloid    (curvature-aware stereographic projection)
    Poincaré ↔ PV             (PVNN Eq. 4 gyro-isomorphism)
    Hyperboloid ↔ PV          (direct: PV coords = space-like 4-velocity part)
    Klein ↔ Poincaré / Hyperboloid / PV   (Einstein half / central projection / Lorentz factor)

Every map is verified for: target-manifold validity, round-trip identity,
origin↦origin, geodesic-distance preservation (the defining isometry property),
the cross-model commutative diagram, and JIT/vmap compatibility. Tests are
parametrized over both dtypes (conftest ``dtype``/``tolerance``) and over a
range of curvatures — the latter specifically guards against curvature-dependent
bugs that a single ``c=1.0`` test cannot catch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.manifolds import Hyperboloid, Poincare, ProperVelocity
from hyperbolix.manifolds import isometry_mappings as iso
from hyperbolix.manifolds.hyperboloid import VERSION_DEFAULT
from hyperbolix.manifolds.poincare import VERSION_MOBIUS_DIRECT

DIM = 3  # spatial dimension (hyperboloid ambient dimension is DIM + 1)
N_POINTS = 20


def _batch(fn):
    """vmap a single-point ``(point, c) -> point`` map over the batch axis."""
    return jax.vmap(fn, in_axes=(0, None))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#
# ``dtype`` and ``tolerance`` are provided by conftest (parametrized f32/f64).
# ``curvature`` spans low/unit/high values, all != 1.0-only, so the maps are
# exercised away from the special c = 1 case. Values are kept >= 0.5 to keep
# float32 geodesic distances comfortably below the ~7 precision ceiling; an
# additional float64-only test below stresses extreme curvatures.


@pytest.fixture(params=[0.5, 1.0, 4.0])
def curvature(request: pytest.FixtureRequest) -> float:
    """Curvature values spanning low / unit / high."""
    return request.param


@pytest.fixture
def manifolds(dtype: jnp.dtype) -> tuple[Poincare, Hyperboloid, ProperVelocity]:
    """Manifold instances built at the test dtype."""
    return Poincare(dtype=dtype), Hyperboloid(dtype=dtype), ProperVelocity(dtype=dtype)


@pytest.fixture
def pv_points(curvature: float, dtype: jnp.dtype) -> jnp.ndarray:
    """PV points: unconstrained Gaussians scaled by 1/√c (no projection needed)."""
    c = curvature
    key = jax.random.PRNGKey(7)
    return jax.random.normal(key, (N_POINTS, DIM), dtype=dtype) / jnp.sqrt(c)


@pytest.fixture
def poincare_points(curvature: float, dtype: jnp.dtype) -> jnp.ndarray:
    """Poincaré points sampled inside the radius-1/√c ball (away from boundary)."""
    c = curvature
    key = jax.random.PRNGKey(123)
    k_dir, k_rad = jax.random.split(key)
    dirs = jax.random.normal(k_dir, (N_POINTS, DIM), dtype=dtype)
    dirs = dirs / jnp.maximum(jnp.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    # Uniform-in-ball radius, capped at 0.7/√c to stay clear of the boundary.
    radii = jax.random.uniform(k_rad, (N_POINTS, 1), dtype=dtype) ** (1.0 / DIM)
    return dirs * radii * (0.7 / jnp.sqrt(c))


@pytest.fixture
def hyperboloid_points(curvature: float, dtype: jnp.dtype) -> jnp.ndarray:
    """Valid hyperboloid points, generated independently of the maps under test.

    Random spatial parts are projected onto the manifold via ``Hyperboloid.proj``,
    which reconstructs the time component as x₀ = √(1/c + ||x_rest||²).
    """
    c = curvature
    key = jax.random.PRNGKey(42)
    spatial = jax.random.normal(key, (N_POINTS, DIM), dtype=dtype) / jnp.sqrt(c)
    ambient = jnp.concatenate([jnp.zeros((N_POINTS, 1), dtype=dtype), spatial], axis=1)
    return _batch(Hyperboloid(dtype=dtype).proj)(ambient, c)


# ---------------------------------------------------------------------------
# Target-manifold validity
# ---------------------------------------------------------------------------


def test_target_manifold_validity(
    manifolds: tuple[Poincare, Hyperboloid, ProperVelocity],
    pv_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    hyperboloid_points: jnp.ndarray,
    curvature: float,
):
    """Each map lands on its target manifold."""
    poincare, hyperboloid, pv = manifolds
    c = curvature

    # Poincaré <-> Hyperboloid
    assert jnp.all(_batch(poincare.is_in_manifold)(_batch(iso.hyperboloid_to_poincare)(hyperboloid_points, c), c))
    assert jnp.all(_batch(hyperboloid.is_in_manifold)(_batch(iso.poincare_to_hyperboloid)(poincare_points, c), c))

    # PV <-> Poincaré
    assert jnp.all(_batch(poincare.is_in_manifold)(_batch(iso.pv_to_poincare)(pv_points, c), c))
    assert jnp.all(_batch(pv.is_in_manifold)(_batch(iso.poincare_to_pv)(poincare_points, c), c))

    # PV <-> Hyperboloid
    assert jnp.all(_batch(hyperboloid.is_in_manifold)(_batch(iso.pv_to_hyperboloid)(pv_points, c), c))
    assert jnp.all(_batch(pv.is_in_manifold)(_batch(iso.hyperboloid_to_pv)(hyperboloid_points, c), c))


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------


def test_round_trip_poincare_hyperboloid(
    poincare_points: jnp.ndarray,
    hyperboloid_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Poincaré <-> Hyperboloid round-trips to identity (both directions)."""
    c = curvature
    atol, rtol = tolerance

    p_rt = _batch(iso.hyperboloid_to_poincare)(_batch(iso.poincare_to_hyperboloid)(poincare_points, c), c)
    assert jnp.allclose(p_rt, poincare_points, atol=atol, rtol=rtol)

    h_rt = _batch(iso.poincare_to_hyperboloid)(_batch(iso.hyperboloid_to_poincare)(hyperboloid_points, c), c)
    assert jnp.allclose(h_rt, hyperboloid_points, atol=atol, rtol=rtol)


def test_round_trip_poincare_pv(
    poincare_points: jnp.ndarray,
    pv_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """PV <-> Poincaré round-trips to identity (both directions)."""
    c = curvature
    atol, rtol = tolerance

    pv_rt = _batch(iso.poincare_to_pv)(_batch(iso.pv_to_poincare)(pv_points, c), c)
    assert jnp.allclose(pv_rt, pv_points, atol=atol, rtol=rtol)

    p_rt = _batch(iso.pv_to_poincare)(_batch(iso.poincare_to_pv)(poincare_points, c), c)
    assert jnp.allclose(p_rt, poincare_points, atol=atol, rtol=rtol)


def test_round_trip_hyperboloid_pv(
    hyperboloid_points: jnp.ndarray,
    pv_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """PV <-> Hyperboloid round-trips to identity (both directions)."""
    c = curvature
    atol, rtol = tolerance

    pv_rt = _batch(iso.hyperboloid_to_pv)(_batch(iso.pv_to_hyperboloid)(pv_points, c), c)
    assert jnp.allclose(pv_rt, pv_points, atol=atol, rtol=rtol)

    h_rt = _batch(iso.pv_to_hyperboloid)(_batch(iso.hyperboloid_to_pv)(hyperboloid_points, c), c)
    assert jnp.allclose(h_rt, hyperboloid_points, atol=atol, rtol=rtol)


@pytest.mark.parametrize("c", [0.01, 0.1, 10.0, 100.0])
def test_poincare_hyperboloid_extreme_curvatures(c: float):
    """float64-only: P <-> H round-trips at extreme curvatures.

    Directly pins the curvature-correctness fix: the previous unit-ball formulas
    only round-tripped at c = 1.0 and drifted badly for c far from 1.
    """
    poincare = Poincare(dtype=jnp.float64)
    key = jax.random.PRNGKey(2024)
    dirs = jax.random.normal(key, (N_POINTS, DIM), dtype=jnp.float64)
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    radii = jax.random.uniform(jax.random.fold_in(key, 1), (N_POINTS, 1), dtype=jnp.float64)
    pts = dirs * radii * (0.7 / jnp.sqrt(c))
    assert jnp.all(_batch(poincare.is_in_manifold)(pts, c)), "test points not in ball"

    rt = _batch(iso.hyperboloid_to_poincare)(_batch(iso.poincare_to_hyperboloid)(pts, c), c)
    assert jnp.allclose(rt, pts, atol=1e-9, rtol=1e-9), f"P<->H round-trip failed at c={c}"


# ---------------------------------------------------------------------------
# Origin mapping
# ---------------------------------------------------------------------------


def test_origin_mapping(curvature: float, dtype: jnp.dtype, tolerance: tuple[float, float]):
    """Every origin maps to every other model's origin."""
    c = curvature
    atol, rtol = tolerance

    poincare_origin = jnp.zeros(DIM, dtype=dtype)
    pv_origin = jnp.zeros(DIM, dtype=dtype)
    hyperboloid_origin = jnp.zeros(DIM + 1, dtype=dtype).at[0].set(jnp.sqrt(1.0 / c))

    def close(a, b):
        return jnp.allclose(a, b, atol=atol, rtol=rtol)

    # Poincaré <-> Hyperboloid
    assert close(iso.hyperboloid_to_poincare(hyperboloid_origin, c), poincare_origin)
    assert close(iso.poincare_to_hyperboloid(poincare_origin, c), hyperboloid_origin)
    # PV <-> Poincaré
    assert close(iso.pv_to_poincare(pv_origin, c), poincare_origin)
    assert close(iso.poincare_to_pv(poincare_origin, c), pv_origin)
    # PV <-> Hyperboloid
    assert close(iso.pv_to_hyperboloid(pv_origin, c), hyperboloid_origin)
    assert close(iso.hyperboloid_to_pv(hyperboloid_origin, c), pv_origin)


# ---------------------------------------------------------------------------
# Isometry: geodesic-distance preservation (the defining property)
# ---------------------------------------------------------------------------


def test_isometry_preserves_pairwise_distance(
    manifolds: tuple[Poincare, Hyperboloid, ProperVelocity],
    pv_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """d_PV(x, y) == d_Poincaré(φ(x), φ(y)) == d_Hyperboloid(ψ(x), ψ(y))."""
    poincare, hyperboloid, pv = manifolds
    c = curvature
    atol, rtol = tolerance

    n = N_POINTS // 2
    xs, ys = pv_points[:n], pv_points[n : 2 * n]

    d_pv = jax.vmap(lambda a, b: pv.dist(a, b, c))(xs, ys)

    xp, yp = _batch(iso.pv_to_poincare)(xs, c), _batch(iso.pv_to_poincare)(ys, c)
    d_p = jax.vmap(lambda a, b: poincare.dist(a, b, c, version_idx=VERSION_MOBIUS_DIRECT))(xp, yp)

    xh, yh = _batch(iso.pv_to_hyperboloid)(xs, c), _batch(iso.pv_to_hyperboloid)(ys, c)
    d_h = jax.vmap(lambda a, b: hyperboloid.dist(a, b, c, version_idx=VERSION_DEFAULT))(xh, yh)

    assert jnp.allclose(d_pv, d_p, atol=atol, rtol=rtol), "PV -> Poincaré not an isometry"
    assert jnp.allclose(d_pv, d_h, atol=atol, rtol=rtol), "PV -> Hyperboloid not an isometry"


def test_isometry_preserves_distance_from_origin(
    manifolds: tuple[Poincare, Hyperboloid, ProperVelocity],
    pv_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Distance-from-origin is preserved by the PV maps."""
    poincare, hyperboloid, pv = manifolds
    c = curvature
    atol, rtol = tolerance

    d_pv = jax.vmap(lambda a: pv.dist_0(a, c))(pv_points)
    d_p = jax.vmap(lambda a: poincare.dist_0(a, c, version_idx=VERSION_MOBIUS_DIRECT))(
        _batch(iso.pv_to_poincare)(pv_points, c)
    )
    d_h = jax.vmap(lambda a: hyperboloid.dist_0(a, c, version_idx=VERSION_DEFAULT))(
        _batch(iso.pv_to_hyperboloid)(pv_points, c)
    )

    assert jnp.allclose(d_pv, d_p, atol=atol, rtol=rtol)
    assert jnp.allclose(d_pv, d_h, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Cross-model consistency (commutative diagram)
# ---------------------------------------------------------------------------


def test_commutative_diagram(
    pv_points: jnp.ndarray,
    hyperboloid_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """The direct PV<->H map agrees with the route composed through Poincaré.

    pv_to_hyperboloid == poincare_to_hyperboloid ∘ pv_to_poincare, and the
    inverse. This pins the direct maps and the P<->H fix together: if either
    drifted in curvature, the two routes would disagree for c != 1.
    """
    c = curvature
    atol, rtol = tolerance

    direct_h = _batch(iso.pv_to_hyperboloid)(pv_points, c)
    composed_h = _batch(iso.poincare_to_hyperboloid)(_batch(iso.pv_to_poincare)(pv_points, c), c)
    assert jnp.allclose(direct_h, composed_h, atol=atol, rtol=rtol), "PV->H != PV->P->H"

    direct_pv = _batch(iso.hyperboloid_to_pv)(hyperboloid_points, c)
    composed_pv = _batch(iso.poincare_to_pv)(_batch(iso.hyperboloid_to_poincare)(hyperboloid_points, c), c)
    assert jnp.allclose(direct_pv, composed_pv, atol=atol, rtol=rtol), "H->PV != H->P->PV"


# ---------------------------------------------------------------------------
# JIT / vmap / shape compatibility
# ---------------------------------------------------------------------------


def test_jit_and_vmap_compatibility(pv_points: jnp.ndarray, curvature: float, tolerance: tuple[float, float]):
    """All four new PV maps are JIT- and vmap-compatible, shape-correct, and produce the SAME
    values under ``jax.jit`` as eagerly.

    The jitted-vs-eager comparison is the part that makes this a jit test at all: without it the
    body only re-ran the round-trips already covered by ``test_round_trip_hyperboloid_pv`` and
    ``test_round_trip_poincare_pv`` (``jax.jit`` is semantics-preserving, so a round-trip through
    jitted maps says nothing the eager one does not). The shape assertions stay — they are the only
    place ``DIM+1`` vs ``DIM`` is pinned.
    """
    c = curvature
    atol, rtol = tolerance

    to_h = jax.jit(_batch(iso.pv_to_hyperboloid))
    from_h = jax.jit(_batch(iso.hyperboloid_to_pv))
    to_p = jax.jit(_batch(iso.pv_to_poincare))
    from_p = jax.jit(_batch(iso.poincare_to_pv))

    z = to_h(pv_points, c)
    assert z.shape == (N_POINTS, DIM + 1)  # gains the time component
    assert jnp.allclose(z, _batch(iso.pv_to_hyperboloid)(pv_points, c), atol=atol, rtol=rtol)
    assert jnp.allclose(from_h(z, c), _batch(iso.hyperboloid_to_pv)(z, c), atol=atol, rtol=rtol)

    y = to_p(pv_points, c)
    assert y.shape == (N_POINTS, DIM)
    assert jnp.allclose(y, _batch(iso.pv_to_poincare)(pv_points, c), atol=atol, rtol=rtol)
    assert jnp.allclose(from_p(y, c), _batch(iso.poincare_to_pv)(y, c), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dim", [1, 2, 5, 10])
def test_dimension_consistency(dim: int):
    """PV <-> Hyperboloid handles arbitrary dimensions (time component add/drop)."""
    c = 1.0
    x = jnp.linspace(-0.3, 0.3, dim, dtype=jnp.float64)

    z = iso.pv_to_hyperboloid(x, c)
    assert z.shape == (dim + 1,)
    x_rt = iso.hyperboloid_to_pv(z, c)
    assert x_rt.shape == (dim,)
    assert jnp.allclose(x_rt, x, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Beltrami-Klein maps
# ---------------------------------------------------------------------------
#
# The Klein class is not imported: membership of a Klein image is checked directly as
# ``c·||k||² < 1``, and Klein distances against an independent NumPy float64 closed form.


def _in_klein_ball(k_BD: jnp.ndarray, c: float) -> jnp.ndarray:
    """``c·||k||² < 1`` row-wise — the Klein model's only constraint."""
    return c * jnp.sum(k_BD**2, axis=-1) < 1.0


def _klein_dist_oracle(x_BD: np.ndarray, y_BD: np.ndarray, c: float) -> np.ndarray:
    """Klein geodesic distance in NumPy float64: ``acosh((1 - c x·y) / √((1-c||x||²)(1-c||y||²))) / √c``."""
    x_BD, y_BD = np.asarray(x_BD, dtype=np.float64), np.asarray(y_BD, dtype=np.float64)
    gap_x_B = 1.0 - c * np.sum(x_BD**2, axis=-1)
    gap_y_B = 1.0 - c * np.sum(y_BD**2, axis=-1)
    arg_B = (1.0 - c * np.sum(x_BD * y_BD, axis=-1)) / np.sqrt(gap_x_B * gap_y_B)
    return np.arccosh(arg_B) / np.sqrt(c)


def _klein_ball_points(key: jax.Array, c: float, dtype: jnp.dtype, dim: int = DIM, cap: float = 0.9) -> jnp.ndarray:
    """Uniform-in-ball Klein points with radius capped at ``cap/√c``, built without any map under test.

    Why the cap is 0.9: the Klein gap ``1 - c·||k||² = sech²(a)`` at scaled radius ``a`` cancels,
    costing a relative error ``eps·cosh²(a)`` — the chart loses precision twice as fast in ``a`` as
    the Poincaré ball. At ``√c·||k|| = 0.9`` (``a = atanh(0.9) ≈ 1.47``, ``cosh²(a) ≈ 5.3``) float32
    still carries ~6 significant digits in the gap, so the fixture tolerance holds for every map.
    """
    k_dir, k_rad = jax.random.split(key)
    dirs = jax.random.normal(k_dir, (N_POINTS, dim), dtype=dtype)
    dirs = dirs / jnp.maximum(jnp.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    radii = jax.random.uniform(k_rad, (N_POINTS, 1), dtype=dtype) ** (1.0 / dim)
    return dirs * radii * (cap / jnp.sqrt(c))


@pytest.fixture
def klein_points(curvature: float, dtype: jnp.dtype) -> jnp.ndarray:
    """Klein points inside the radius-1/√c ball, capped at 0.9/√c (see ``_klein_ball_points``)."""
    return _klein_ball_points(jax.random.PRNGKey(99), curvature, dtype)


def test_klein_target_manifold_validity(
    manifolds: tuple[Poincare, Hyperboloid, ProperVelocity],
    klein_points: jnp.ndarray,
    pv_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    hyperboloid_points: jnp.ndarray,
    curvature: float,
):
    """Each Klein map lands on its target; maps into Klein land inside the Klein ball."""
    poincare, hyperboloid, pv = manifolds
    c = curvature

    assert jnp.all(_in_klein_ball(klein_points, c)), "test points not in the Klein ball"

    assert jnp.all(_batch(poincare.is_in_manifold)(_batch(iso.klein_to_poincare)(klein_points, c), c))
    assert jnp.all(_batch(hyperboloid.is_in_manifold)(_batch(iso.klein_to_hyperboloid)(klein_points, c), c))
    assert jnp.all(_batch(pv.is_in_manifold)(_batch(iso.klein_to_pv)(klein_points, c), c))

    assert jnp.all(_in_klein_ball(_batch(iso.poincare_to_klein)(poincare_points, c), c))
    assert jnp.all(_in_klein_ball(_batch(iso.hyperboloid_to_klein)(hyperboloid_points, c), c))
    assert jnp.all(_in_klein_ball(_batch(iso.pv_to_klein)(pv_points, c), c))


def test_round_trip_klein_poincare(
    klein_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Klein <-> Poincaré round-trips to identity (both directions)."""
    c = curvature
    atol, rtol = tolerance

    k_rt = _batch(iso.poincare_to_klein)(_batch(iso.klein_to_poincare)(klein_points, c), c)
    assert jnp.allclose(k_rt, klein_points, atol=atol, rtol=rtol)

    p_rt = _batch(iso.klein_to_poincare)(_batch(iso.poincare_to_klein)(poincare_points, c), c)
    assert jnp.allclose(p_rt, poincare_points, atol=atol, rtol=rtol)


def test_round_trip_klein_hyperboloid(
    klein_points: jnp.ndarray,
    hyperboloid_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Klein <-> Hyperboloid round-trips to identity (both directions)."""
    c = curvature
    atol, rtol = tolerance

    k_rt = _batch(iso.hyperboloid_to_klein)(_batch(iso.klein_to_hyperboloid)(klein_points, c), c)
    assert jnp.allclose(k_rt, klein_points, atol=atol, rtol=rtol)

    h_rt = _batch(iso.klein_to_hyperboloid)(_batch(iso.hyperboloid_to_klein)(hyperboloid_points, c), c)
    assert jnp.allclose(h_rt, hyperboloid_points, atol=atol, rtol=rtol)


def test_round_trip_klein_pv(
    klein_points: jnp.ndarray,
    pv_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Klein <-> PV round-trips to identity (both directions)."""
    c = curvature
    atol, rtol = tolerance

    k_rt = _batch(iso.pv_to_klein)(_batch(iso.klein_to_pv)(klein_points, c), c)
    assert jnp.allclose(k_rt, klein_points, atol=atol, rtol=rtol)

    pv_rt = _batch(iso.klein_to_pv)(_batch(iso.pv_to_klein)(pv_points, c), c)
    assert jnp.allclose(pv_rt, pv_points, atol=atol, rtol=rtol)


@pytest.mark.parametrize("c", [0.01, 0.1, 10.0, 100.0])
def test_klein_extreme_curvatures(c: float):
    """float64-only: K <-> P and K <-> H round-trip at extreme curvatures (points scaled by 1/√c)."""
    klein_pts = _klein_ball_points(jax.random.PRNGKey(2025), c, jnp.float64)
    assert jnp.all(_in_klein_ball(klein_pts, c)), "test points not in the Klein ball"

    k_rt_p = _batch(iso.poincare_to_klein)(_batch(iso.klein_to_poincare)(klein_pts, c), c)
    assert jnp.allclose(k_rt_p, klein_pts, atol=1e-9, rtol=1e-9), f"K->P->K round-trip failed at c={c}"

    k_rt_h = _batch(iso.hyperboloid_to_klein)(_batch(iso.klein_to_hyperboloid)(klein_pts, c), c)
    assert jnp.allclose(k_rt_h, klein_pts, atol=1e-9, rtol=1e-9), f"K->H->K round-trip failed at c={c}"

    poincare_pts = _batch(iso.klein_to_poincare)(klein_pts, c)
    p_rt = _batch(iso.klein_to_poincare)(_batch(iso.poincare_to_klein)(poincare_pts, c), c)
    assert jnp.allclose(p_rt, poincare_pts, atol=1e-9, rtol=1e-9), f"P->K->P round-trip failed at c={c}"

    spatial = jax.random.normal(jax.random.PRNGKey(2026), (N_POINTS, DIM), dtype=jnp.float64) / jnp.sqrt(c)
    ambient = jnp.concatenate([jnp.zeros((N_POINTS, 1), dtype=jnp.float64), spatial], axis=1)
    hyperboloid_pts = _batch(Hyperboloid(dtype=jnp.float64).proj)(ambient, c)
    h_rt = _batch(iso.klein_to_hyperboloid)(_batch(iso.hyperboloid_to_klein)(hyperboloid_pts, c), c)
    assert jnp.allclose(h_rt, hyperboloid_pts, atol=1e-9, rtol=1e-9), f"H->K->H round-trip failed at c={c}"


def test_klein_origin_mapping(curvature: float, dtype: jnp.dtype, tolerance: tuple[float, float]):
    """The Klein origin maps to every other model's origin, and back."""
    c = curvature
    atol, rtol = tolerance

    zero_D = jnp.zeros(DIM, dtype=dtype)  # Klein, Poincaré and PV origins coincide at 0
    hyperboloid_origin = jnp.zeros(DIM + 1, dtype=dtype).at[0].set(jnp.sqrt(1.0 / c))

    def close(a, b):
        return jnp.allclose(a, b, atol=atol, rtol=rtol)

    assert close(iso.klein_to_poincare(zero_D, c), zero_D)
    assert close(iso.poincare_to_klein(zero_D, c), zero_D)
    assert close(iso.klein_to_hyperboloid(zero_D, c), hyperboloid_origin)
    assert close(iso.hyperboloid_to_klein(hyperboloid_origin, c), zero_D)
    assert close(iso.klein_to_pv(zero_D, c), zero_D)
    assert close(iso.pv_to_klein(zero_D, c), zero_D)


def test_klein_isometry_preserves_pairwise_distance(
    manifolds: tuple[Poincare, Hyperboloid, ProperVelocity],
    klein_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """d_K(x, y) == d_Poincaré(K→P) == d_Hyperboloid(K→H) == d_PV(K→PV), d_K from a NumPy closed form.

    The oracle's ``acosh`` loses digits only near argument 1 (coincident points); the fixture's
    random pairs sit at distances ≳ 0.1, where the float64 oracle is exact to ~1e-14.
    """
    poincare, hyperboloid, pv = manifolds
    c = curvature
    atol, rtol = tolerance

    n = N_POINTS // 2
    xs, ys = klein_points[:n], klein_points[n : 2 * n]

    d_oracle = _klein_dist_oracle(np.asarray(xs), np.asarray(ys), c)

    xp, yp = _batch(iso.klein_to_poincare)(xs, c), _batch(iso.klein_to_poincare)(ys, c)
    d_p = jax.vmap(lambda a, b: poincare.dist(a, b, c, version_idx=VERSION_MOBIUS_DIRECT))(xp, yp)

    xh, yh = _batch(iso.klein_to_hyperboloid)(xs, c), _batch(iso.klein_to_hyperboloid)(ys, c)
    d_h = jax.vmap(lambda a, b: hyperboloid.dist(a, b, c, version_idx=VERSION_DEFAULT))(xh, yh)

    xv, yv = _batch(iso.klein_to_pv)(xs, c), _batch(iso.klein_to_pv)(ys, c)
    d_pv = jax.vmap(lambda a, b: pv.dist(a, b, c))(xv, yv)

    assert np.allclose(np.asarray(d_p), d_oracle, atol=atol, rtol=rtol), "K -> Poincaré not an isometry"
    assert np.allclose(np.asarray(d_h), d_oracle, atol=atol, rtol=rtol), "K -> Hyperboloid not an isometry"
    assert np.allclose(np.asarray(d_pv), d_oracle, atol=atol, rtol=rtol), "K -> PV not an isometry"
    assert jnp.allclose(d_p, d_h, atol=atol, rtol=rtol)
    assert jnp.allclose(d_p, d_pv, atol=atol, rtol=rtol)


def test_klein_isometry_preserves_distance_from_origin(
    manifolds: tuple[Poincare, Hyperboloid, ProperVelocity],
    klein_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Distance-from-origin is preserved by the Klein maps; oracle ``atanh(√c·||k||)/√c`` in NumPy."""
    poincare, hyperboloid, pv = manifolds
    c = curvature
    atol, rtol = tolerance

    k_np = np.asarray(klein_points, dtype=np.float64)
    d_oracle = np.arctanh(np.sqrt(c) * np.linalg.norm(k_np, axis=-1)) / np.sqrt(c)

    d_p = jax.vmap(lambda a: poincare.dist_0(a, c, version_idx=VERSION_MOBIUS_DIRECT))(
        _batch(iso.klein_to_poincare)(klein_points, c)
    )
    d_h = jax.vmap(lambda a: hyperboloid.dist_0(a, c, version_idx=VERSION_DEFAULT))(
        _batch(iso.klein_to_hyperboloid)(klein_points, c)
    )
    d_pv = jax.vmap(lambda a: pv.dist_0(a, c))(_batch(iso.klein_to_pv)(klein_points, c))

    assert np.allclose(np.asarray(d_p), d_oracle, atol=atol, rtol=rtol)
    assert np.allclose(np.asarray(d_h), d_oracle, atol=atol, rtol=rtol)
    assert np.allclose(np.asarray(d_pv), d_oracle, atol=atol, rtol=rtol)


def test_klein_commutative_diagrams(
    klein_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    pv_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Every direct Klein map agrees with the route composed through a third model."""
    c = curvature
    atol, rtol = tolerance

    def close(a, b):
        return jnp.allclose(a, b, atol=atol, rtol=rtol)

    k_to_h = _batch(iso.klein_to_hyperboloid)(klein_points, c)
    assert close(k_to_h, _batch(iso.poincare_to_hyperboloid)(_batch(iso.klein_to_poincare)(klein_points, c), c)), (
        "K->H != K->P->H"
    )
    assert close(_batch(iso.klein_to_pv)(klein_points, c), _batch(iso.hyperboloid_to_pv)(k_to_h, c)), "K->PV != K->H->PV"
    assert close(
        _batch(iso.poincare_to_klein)(poincare_points, c),
        _batch(iso.hyperboloid_to_klein)(_batch(iso.poincare_to_hyperboloid)(poincare_points, c), c),
    ), "P->K != P->H->K"
    assert close(
        _batch(iso.pv_to_klein)(pv_points, c),
        _batch(iso.hyperboloid_to_klein)(_batch(iso.pv_to_hyperboloid)(pv_points, c), c),
    ), "PV->K != PV->H->K"


def test_klein_poincare_maps_are_the_einstein_half_and_mobius_double(
    klein_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    curvature: float,
    dtype: jnp.dtype,
    tolerance: tuple[float, float],
):
    """K→P is ``k·gamma_k/(1+gamma_k)`` (gamma_k the Einstein Lorentz factor) and P→K is the Möbius ``2 ⊗ p``."""
    c = curvature
    atol, rtol = tolerance

    k_np = np.asarray(klein_points, dtype=np.float64)
    gamma_B1 = 1.0 / np.sqrt(1.0 - c * np.sum(k_np**2, axis=-1, keepdims=True))
    expected_p = k_np * gamma_B1 / (1.0 + gamma_B1)
    assert np.allclose(np.asarray(_batch(iso.klein_to_poincare)(klein_points, c)), expected_p, atol=atol, rtol=rtol)

    mobius_double = jax.vmap(lambda p: Poincare(dtype=dtype).scalar_mul(2.0, p, c))(poincare_points)
    assert jnp.allclose(_batch(iso.poincare_to_klein)(poincare_points, c), mobius_double, atol=atol, rtol=rtol)


def test_klein_jit_and_vmap_compatibility(
    klein_points: jnp.ndarray,
    curvature: float,
    dtype: jnp.dtype,
    tolerance: tuple[float, float],
):
    """All six Klein maps are JIT- and vmap-compatible, shape- and dtype-correct, and match eager."""
    c = curvature
    atol, rtol = tolerance

    forward = [(iso.klein_to_poincare, DIM), (iso.klein_to_hyperboloid, DIM + 1), (iso.klein_to_pv, DIM)]
    inverse = {
        iso.klein_to_poincare: iso.poincare_to_klein,
        iso.klein_to_hyperboloid: iso.hyperboloid_to_klein,
        iso.klein_to_pv: iso.pv_to_klein,
    }
    for fn, out_dim in forward:
        out = jax.jit(_batch(fn))(klein_points, c)
        assert out.shape == (N_POINTS, out_dim), fn.__name__
        assert out.dtype == dtype, fn.__name__
        assert jnp.allclose(out, _batch(fn)(klein_points, c), atol=atol, rtol=rtol), fn.__name__

        inv = inverse[fn]
        back = jax.jit(_batch(inv))(out, c)
        assert back.shape == (N_POINTS, DIM), inv.__name__
        assert back.dtype == dtype, inv.__name__
        assert jnp.allclose(back, _batch(inv)(out, c), atol=atol, rtol=rtol), inv.__name__


@pytest.mark.parametrize("dim", [1, 2, 5, 10])
def test_klein_dimension_consistency(dim: int):
    """Klein <-> {Poincaré, Hyperboloid, PV} handle arbitrary dimensions."""
    c = 1.0
    k = jnp.linspace(-0.3, 0.3, dim, dtype=jnp.float64)

    for to_fn, from_fn, out_dim in [
        (iso.klein_to_poincare, iso.poincare_to_klein, dim),
        (iso.klein_to_hyperboloid, iso.hyperboloid_to_klein, dim + 1),
        (iso.klein_to_pv, iso.pv_to_klein, dim),
    ]:
        out = to_fn(k, c)
        assert out.shape == (out_dim,), to_fn.__name__
        k_rt = from_fn(out, c)
        assert k_rt.shape == (dim,), from_fn.__name__
        assert jnp.allclose(k_rt, k, atol=1e-9, rtol=1e-9), from_fn.__name__
