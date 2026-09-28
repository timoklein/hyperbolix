"""Tests for isometry mappings between hyperbolic manifold models.

Covers the distance-preserving transformations among the Poincaré ball,
hyperboloid (Lorentz), Proper Velocity (PV), and Beltrami-Klein models:

    Poincaré ↔ Hyperboloid    (curvature-aware stereographic projection)
    Poincaré ↔ PV             (PVNN Eq. 4 gyro-isomorphism)
    Hyperboloid ↔ PV          (direct: PV coords = space-like 4-velocity part)
    Klein ↔ Poincaré / Hyperboloid / PV   (Einstein half / central projection / Lorentz factor)

Every map is verified for: target-manifold validity, round-trip identity,
origin↦origin, geodesic-distance preservation (the defining isometry property),
the cross-model commutative diagram, and JIT/vmap compatibility; every ``Klein``
operation is also checked for equivariance (the Klein op equals another model's op
transported through the maps, tangents via ``jax.jvp``). Tests are
parametrized over both dtypes (conftest ``dtype``/``tolerance``) and over a
range of curvatures — the latter specifically guards against curvature-dependent
bugs that a single ``c=1.0`` test cannot catch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.manifolds import Hyperboloid, Klein, Poincare, ProperVelocity
from hyperbolix.manifolds import isometry_mappings as iso
from hyperbolix.manifolds.hyperboloid import VERSION_DEFAULT
from hyperbolix.manifolds.poincare import VERSION_MOBIUS_DIRECT
from hyperbolix.nn_layers.hyperboloid_core import lorentz_midpoint

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
# The map tests in this block do not call the Klein class: membership of a Klein image is checked
# directly as ``c·||k||² < 1``, and Klein distances against an independent NumPy float64 closed
# form. The ``Klein`` operations themselves are tested through the maps in the next block.


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


# ---------------------------------------------------------------------------
# Klein operations are equivariant under the maps
# ---------------------------------------------------------------------------
#
# An isometry carries every geometric operation with it, so each ``Klein`` op must equal the same
# op of another model transported through the library maps: ``f_K(x, ...) = φ⁻¹(f_M(φ(x), ...))``.
# Tangent vectors move with the map's differential, computed with ``jax.jvp`` of the map itself —
# not a hand-derived pushforward that could share an algebra slip with the Klein code. The
# ``klein_points`` fixture caps the scaled radius at ``atanh(0.9) ≈ 1.47``, and every derived point
# (sums, exponentials) stays below scaled radius ≈ 3, where float32 still carries ~5 digits in the
# Klein gap, so the conftest tolerance holds.


def _klein_tangents(key: jax.Array, k_BD: jnp.ndarray, c: float, dtype: jnp.dtype) -> jnp.ndarray:
    """Random tangent vectors at ``k_BD`` with scaled Klein-metric norm ``√c·‖v‖_x`` uniform in [0.1, 1.5].

    The norm is the closed form ``‖v‖_x = √(g_x‖v‖² + c(x·v)²)/g_x``, ``g_x = 1 - c‖x‖²``, so the step
    length is set independently of ``Klein.tangent_norm`` (which the tests below also check).
    """
    k_dir, k_len = jax.random.split(key)
    v_BD = jax.random.normal(k_dir, k_BD.shape, dtype=dtype)
    tau_B = jax.random.uniform(k_len, (k_BD.shape[0],), dtype=dtype, minval=0.1, maxval=1.5)
    g_B = 1.0 - c * jnp.sum(k_BD**2, axis=-1)
    xv_B = jnp.sum(k_BD * v_BD, axis=-1)
    norm_B = jnp.sqrt(g_B * jnp.sum(v_BD**2, axis=-1) + c * xv_B**2) / g_B
    return v_BD * (tau_B / (jnp.sqrt(c) * norm_B))[:, None]


def _push(fn, c: float):
    """Batched ``(φ(k), dφ_k(u))`` for the map ``fn``, via ``jax.jvp``."""
    return jax.vmap(lambda k, u: jax.jvp(lambda p: fn(p, c), (k,), (u,)))


def test_klein_dist_and_dist_0_match_poincare_and_hyperboloid_through_the_maps(
    klein_points: jnp.ndarray,
    curvature: float,
    dtype: jnp.dtype,
    tolerance: tuple[float, float],
):
    """``Klein.dist(x, y) == Poincare.dist(φx, φy) == Hyperboloid.dist(ψx, ψy)``, and ``dist_0`` likewise."""
    c = curvature
    atol, rtol = tolerance
    klein, poincare, hyperboloid = Klein(dtype=dtype), Poincare(dtype=dtype), Hyperboloid(dtype=dtype)

    n = N_POINTS // 2
    xs, ys = klein_points[:n], klein_points[n : 2 * n]
    to_p, to_h = _batch(iso.klein_to_poincare), _batch(iso.klein_to_hyperboloid)

    d_k = jax.vmap(lambda a, b: klein.dist(a, b, c))(xs, ys)
    d_p = jax.vmap(lambda a, b: poincare.dist(a, b, c, version_idx=VERSION_MOBIUS_DIRECT))(to_p(xs, c), to_p(ys, c))
    d_h = jax.vmap(lambda a, b: hyperboloid.dist(a, b, c, version_idx=VERSION_DEFAULT))(to_h(xs, c), to_h(ys, c))
    assert jnp.allclose(d_k, d_p, atol=atol, rtol=rtol), "Klein.dist != Poincare.dist through K->P"
    assert jnp.allclose(d_k, d_h, atol=atol, rtol=rtol), "Klein.dist != Hyperboloid.dist through K->H"

    d0_k = jax.vmap(lambda a: klein.dist_0(a, c))(klein_points)
    d0_p = jax.vmap(lambda a: poincare.dist_0(a, c))(to_p(klein_points, c))
    d0_h = jax.vmap(lambda a: hyperboloid.dist_0(a, c))(to_h(klein_points, c))
    assert jnp.allclose(d0_k, d0_p, atol=atol, rtol=rtol), "Klein.dist_0 != Poincare.dist_0 through K->P"
    assert jnp.allclose(d0_k, d0_h, atol=atol, rtol=rtol), "Klein.dist_0 != Hyperboloid.dist_0 through K->H"


def test_klein_gyro_operations_match_mobius_through_the_poincare_map(
    klein_points: jnp.ndarray,
    curvature: float,
    dtype: jnp.dtype,
    tolerance: tuple[float, float],
):
    """Einstein ``⊕``, ``(⊖x) ⊕ y`` and ``⊗`` equal their Möbius counterparts conjugated by K->P.

    ``klein_to_poincare`` is the Einstein half ``p = ½ ⊗ k``, a gyrovector-space isomorphism from
    ``(K, ⊕_E, ⊗)`` onto ``(P, ⊕_M, ⊗)`` (Ungar 2009): it carries ``⊕_E`` to ``⊕_M``, the gyro-inverse
    ``-k`` to ``-p``, and commutes with scalar multiplication. So
    ``x ⊕_E y = P->K(φx ⊕_M φy)``, ``(⊖x) ⊕_E y = P->K((-φx) ⊕_M φy)`` and
    ``r ⊗ x = P->K(r ⊗ φx)``. The last one is not vacuous even though ``Klein.scalar_mul`` reuses
    the Möbius formula: the two sides apply it at different radii (``tanh(a)`` vs ``tanh(a/2)``).
    """
    c = curvature
    atol, rtol = tolerance
    klein, poincare = Klein(dtype=dtype), Poincare(dtype=dtype)
    to_p, to_k = _batch(iso.klein_to_poincare), _batch(iso.poincare_to_klein)

    n = N_POINTS // 2
    xs, ys = klein_points[:n], klein_points[n : 2 * n]
    xp, yp = to_p(xs, c), to_p(ys, c)

    add_k = jax.vmap(lambda a, b: klein.addition(a, b, c))(xs, ys)
    add_p = to_k(jax.vmap(lambda a, b: poincare.addition(a, b, c))(xp, yp), c)
    assert jnp.allclose(add_k, add_p, atol=atol, rtol=rtol), "Einstein addition != Mobius addition through K<->P"

    diff_k = jax.vmap(lambda a, b: klein.gyro_difference(a, b, c))(xs, ys)
    diff_p = to_k(jax.vmap(lambda a, b: poincare.addition(-a, b, c))(xp, yp), c)
    assert jnp.allclose(diff_k, diff_p, atol=atol, rtol=rtol), "Einstein gyro_difference != (-p) (+)_M q through K<->P"

    for r in (-1.5, 0.3, 2.0):
        mul_k = jax.vmap(lambda a, r=r: klein.scalar_mul(r, a, c))(klein_points)
        mul_p = to_k(jax.vmap(lambda a, r=r: poincare.scalar_mul(r, a, c))(to_p(klein_points, c)), c)
        assert jnp.allclose(mul_k, mul_p, atol=atol, rtol=rtol), f"Einstein scalar_mul({r}) != Mobius through K<->P"


def test_klein_origin_exp_and_log_match_poincare_through_the_half_differential(
    klein_points: jnp.ndarray,
    curvature: float,
    dtype: jnp.dtype,
    tolerance: tuple[float, float],
):
    """``K->P ∘ exp^K_0 = exp^P_0 ∘ d(K->P)_0`` and ``d(K->P)_0 ∘ log^K_0 = log^P_0 ∘ K->P``.

    Tangent correspondence at the origin: ``K->P(k) = k/(1 + √(1 - c‖k‖²)) = k/2 + O(‖k‖³)``, so
    its differential at 0 is ``½I``. The Klein metric at 0 is ``I`` and Poincaré's is
    ``λ_0²I = 4I``, so ``½I`` is an isometry of the two tangent spaces
    (``‖v/2‖_P = 2·‖v‖/2 = ‖v‖``) — the correspondence is ``v ↦ v/2``, not the identity, even though
    ``Klein.expmap_0`` and ``Poincare.expmap_0`` are the same function. Hence
    ``K->P(exp^K_0(v)) = exp^P_0(v/2)`` and ``log^K_0(k)/2 = log^P_0(K->P(k))``. The ``½`` is
    read off ``jax.jvp`` of ``klein_to_poincare`` at 0 and asserted to be ``½I``.
    """
    c = curvature
    atol, rtol = tolerance
    klein, poincare = Klein(dtype=dtype), Poincare(dtype=dtype)
    to_p = _batch(iso.klein_to_poincare)

    zeros_BD = jnp.zeros_like(klein_points)
    v_BD = _klein_tangents(jax.random.PRNGKey(5), zeros_BD, c, dtype)
    _, v_p_BD = _push(iso.klein_to_poincare, c)(zeros_BD, v_BD)
    assert jnp.allclose(v_p_BD, 0.5 * v_BD, atol=atol, rtol=rtol), "d(K->P) at the origin is not I/2"

    exp_k = to_p(jax.vmap(lambda v: klein.expmap_0(v, c))(v_BD), c)
    exp_p = jax.vmap(lambda v: poincare.expmap_0(v, c))(v_p_BD)
    assert jnp.allclose(exp_k, exp_p, atol=atol, rtol=rtol), "K->P(exp^K_0(v)) != exp^P_0(dφ_0 v)"

    log_k = jax.vmap(lambda k: klein.logmap_0(k, c))(klein_points)
    _, log_k_pushed = _push(iso.klein_to_poincare, c)(zeros_BD, log_k)
    log_p = jax.vmap(lambda p: poincare.logmap_0(p, c))(to_p(klein_points, c))
    assert jnp.allclose(log_k_pushed, log_p, atol=atol, rtol=rtol), "dφ_0(log^K_0(k)) != log^P_0(K->P(k))"


def test_klein_expmap_logmap_ptransp_match_the_hyperboloid_through_the_jvp_pushforward(
    klein_points: jnp.ndarray,
    curvature: float,
    dtype: jnp.dtype,
    tolerance: tuple[float, float],
):
    """``exp``, ``log``, ``PT`` and ``‖·‖_x`` of Klein equal the hyperboloid's under ``ψ = klein_to_hyperboloid``.

    With ``(X, V) = (ψ(x), dψ_x(v))`` from ``jax.jvp``:

    * ``Klein.expmap(v, x) == H->K(Hyperboloid.expmap(V, X))``
    * ``dψ_x(Klein.logmap(y, x)) == Hyperboloid.logmap(ψy, X)``
    * ``dψ_y(Klein.ptransp(v, x, y)) == Hyperboloid.ptransp(V, X, ψy)``
    * ``Klein.tangent_norm(v, x) == Hyperboloid.tangent_norm(V, X)`` (the metric is the pullback)

    Written out, ``dψ_x(u) = (√c·G³(x·u), G·u + c·G³(x·u)·x)`` with ``G = 1/√(1 - c‖x‖²)``; the jvp
    of the library map is used instead of that formula so no hand algebra sits in the oracle.
    """
    c = curvature
    atol, rtol = tolerance
    klein, hyperboloid = Klein(dtype=dtype), Hyperboloid(dtype=dtype)
    push = _push(iso.klein_to_hyperboloid, c)

    n = N_POINTS // 2
    xs, ys = klein_points[:n], klein_points[n : 2 * n]
    v_BD = _klein_tangents(jax.random.PRNGKey(11), xs, c, dtype)
    x_BA, v_BA = push(xs, v_BD)
    y_BA = _batch(iso.klein_to_hyperboloid)(ys, c)

    norm_k = jax.vmap(lambda v, x: klein.tangent_norm(v, x, c))(v_BD, xs)
    norm_h = jax.vmap(lambda v, x: hyperboloid.tangent_norm(v, x, c))(v_BA, x_BA)
    assert jnp.allclose(norm_k, norm_h, atol=atol, rtol=rtol), "Klein metric != pullback of the Minkowski metric"

    exp_k = jax.vmap(lambda v, x: klein.expmap(v, x, c))(v_BD, xs)
    exp_h = _batch(iso.hyperboloid_to_klein)(jax.vmap(lambda v, x: hyperboloid.expmap(v, x, c))(v_BA, x_BA), c)
    assert jnp.allclose(exp_k, exp_h, atol=atol, rtol=rtol), "Klein.expmap != H->K(Hyperboloid.expmap(dψ v))"

    log_k = jax.vmap(lambda y, x: klein.logmap(y, x, c))(ys, xs)
    _, log_k_pushed = push(xs, log_k)
    log_h = jax.vmap(lambda y, x: hyperboloid.logmap(y, x, c))(y_BA, x_BA)
    assert jnp.allclose(log_k_pushed, log_h, atol=atol, rtol=rtol), "dψ(Klein.logmap) != Hyperboloid.logmap"

    pt_k = jax.vmap(lambda v, x, y: klein.ptransp(v, x, y, c))(v_BD, xs, ys)
    _, pt_k_pushed = push(ys, pt_k)
    pt_h = jax.vmap(lambda v, x, y: hyperboloid.ptransp(v, x, y, c))(v_BA, x_BA, y_BA)
    assert jnp.allclose(pt_k_pushed, pt_h, atol=atol, rtol=rtol), "dψ(Klein.ptransp) != Hyperboloid.ptransp(dψ v)"


def test_klein_einstein_midpoint_is_the_lorentz_midpoint_through_the_maps(
    klein_points: jnp.ndarray,
    curvature: float,
    dtype: jnp.dtype,
    tolerance: tuple[float, float],
):
    """``Klein.einstein_midpoint == H->K(lorentz_midpoint(K->H(points)))``, weighted and uniform.

    The Einstein midpoint ``Σ wᵢGᵢxᵢ/Σ wᵢGᵢ`` (``Gᵢ`` the Lorentz factors) is the Klein coordinate of the weighted Lorentz
    centroid ``Σ wᵢXᵢ`` (normalized onto the sheet), and ``hyperboloid_core.lorentz_midpoint``
    computes that centroid with its own cancellation-free normalization. Checked on the full
    20-point cloud and on a 3-point subset, whose midpoint sits far from the origin.
    """
    c = curvature
    atol, rtol = tolerance
    klein = Klein(dtype=dtype)
    weights_N = jax.random.uniform(jax.random.PRNGKey(13), (N_POINTS,), dtype=dtype, minval=0.1, maxval=1.0)

    for pts_ND, w_N in ((klein_points, weights_N), (klein_points[:3], weights_N[:3])):
        lifted_NA = _batch(iso.klein_to_hyperboloid)(pts_ND, c)
        mid_h = iso.hyperboloid_to_klein(lorentz_midpoint(lifted_NA, w_N[None, :], c)[0], c)
        assert jnp.allclose(klein.einstein_midpoint(pts_ND, w_N, c), mid_h, atol=atol, rtol=rtol), "weighted"

        uniform_N = jnp.full((pts_ND.shape[0],), 1.0 / pts_ND.shape[0], dtype=dtype)
        mid_h_uniform = iso.hyperboloid_to_klein(lorentz_midpoint(lifted_NA, uniform_N[None, :], c)[0], c)
        assert jnp.allclose(klein.einstein_midpoint(pts_ND, None, c), mid_h_uniform, atol=atol, rtol=rtol), "uniform"


# ---------------------------------------------------------------------------
# Poincaré half-space maps — shared helpers
# ---------------------------------------------------------------------------
#
# The map tests below do not call the ``HalfSpace`` class: membership of a half-space image is
# checked directly as ``x_n > 0``, and half-space distances against an independent NumPy float64
# closed form. The ``HalfSpace`` operations themselves are tested through the maps in a later block.


def _in_halfspace(x_BD: jnp.ndarray) -> jnp.ndarray:
    """``x_n > 0`` with finite entries, row-wise — the half-space model's only constraint."""
    return jnp.all(jnp.isfinite(x_BD), axis=-1) & (x_BD[..., -1] > 0.0)


def _halfspace_dist_oracle(x_BD: np.ndarray, y_BD: np.ndarray, c: float) -> np.ndarray:
    """Half-space geodesic distance in NumPy float64: ``acosh(1 + ||x - y||²/(2 x_n y_n)) / √c``."""
    x_BD, y_BD = np.asarray(x_BD, dtype=np.float64), np.asarray(y_BD, dtype=np.float64)
    arg_B = 1.0 + np.sum((x_BD - y_BD) ** 2, axis=-1) / (2.0 * x_BD[..., -1] * y_BD[..., -1])
    return np.arccosh(arg_B) / np.sqrt(c)


def _halfspace_origin(c: float, dtype: jnp.dtype, dim: int = DIM) -> jnp.ndarray:
    """The half-space origin ``e_n/√c``."""
    return jnp.zeros(dim, dtype=dtype).at[-1].set(1.0 / jnp.sqrt(jnp.asarray(c, dtype=dtype)))


@pytest.fixture
def halfspace_points(curvature: float, dtype: jnp.dtype) -> jnp.ndarray:
    """Half-space points built without any map under test.

    ``x_s ~ N(0, 0.5²)/√c`` and ``log(√c·x_n) ~ N(0, 0.5²)``: the scaled distance from the origin
    stays below ≈ 3, well inside the range where float32 carries the fixture tolerance.
    """
    c = curvature
    k_s, k_n = jax.random.split(jax.random.PRNGKey(2718))
    x_s = 0.5 * jax.random.normal(k_s, (N_POINTS, DIM - 1), dtype=dtype) / jnp.sqrt(c)
    x_n = jnp.exp(0.5 * jax.random.normal(k_n, (N_POINTS, 1), dtype=dtype)) / jnp.sqrt(c)
    return jnp.concatenate([x_s, x_n], axis=1)


# ---------------------------------------------------------------------------
# Half-space ↔ Poincaré maps
# ---------------------------------------------------------------------------

#
# The Cayley transform is checked against an independent NumPy route through the hyperboloid: the
# half-space lift ``X = ((||x||² + 1/c)/(2 x_n), x_s/(√c x_n), (||x||² - 1/c)/(2 x_n))`` (time first)
# and its inverse ``x_n = 1/(c (X_0 - X_n))``, ``x_s = √c x_n X_s``, written below without any map
# under test.


def _hs_poincare_lift(x_BD: np.ndarray, c: float) -> np.ndarray:
    """NumPy float64 half-space → hyperboloid lift (time coordinate first)."""
    x_BD = np.asarray(x_BD, dtype=np.float64)
    sqnorm_B1 = np.sum(x_BD**2, axis=-1, keepdims=True)
    x_n_B1 = x_BD[..., -1:]
    time_B1 = (sqnorm_B1 + 1.0 / c) / (2.0 * x_n_B1)
    last_B1 = (sqnorm_B1 - 1.0 / c) / (2.0 * x_n_B1)
    return np.concatenate([time_B1, x_BD[..., :-1] / (np.sqrt(c) * x_n_B1), last_B1], axis=-1)


def _hs_poincare_unlift(h_BD1: np.ndarray, c: float) -> np.ndarray:
    """NumPy float64 hyperboloid → half-space, the inverse of :func:`_hs_poincare_lift`."""
    h_BD1 = np.asarray(h_BD1, dtype=np.float64)
    x_n_B1 = 1.0 / (c * (h_BD1[..., :1] - h_BD1[..., -1:]))
    return np.concatenate([np.sqrt(c) * x_n_B1 * h_BD1[..., 1:-1], x_n_B1], axis=-1)


def _hs_poincare_ball_points(key: jax.Array, c: float, dtype: jnp.dtype, dim: int = DIM) -> jnp.ndarray:
    """Uniform-in-ball Poincaré points of radius < 0.7/√c, as the ``poincare_points`` fixture, at any ``c``/``dim``."""
    k_dir, k_rad = jax.random.split(key)
    dirs = jax.random.normal(k_dir, (N_POINTS, dim), dtype=dtype)
    dirs = dirs / jnp.maximum(jnp.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    radii = jax.random.uniform(k_rad, (N_POINTS, 1), dtype=dtype) ** (1.0 / dim)
    return dirs * radii * (0.7 / jnp.sqrt(c))


def _hs_poincare_halfspace_points(key: jax.Array, c: float, dtype: jnp.dtype, dim: int = DIM) -> jnp.ndarray:
    """Half-space points distributed as the ``halfspace_points`` fixture, at any ``c``/``dim``."""
    k_s, k_n = jax.random.split(key)
    x_s = 0.5 * jax.random.normal(k_s, (N_POINTS, dim - 1), dtype=dtype) / jnp.sqrt(c)
    x_n = jnp.exp(0.5 * jax.random.normal(k_n, (N_POINTS, 1), dtype=dtype)) / jnp.sqrt(c)
    return jnp.concatenate([x_s, x_n], axis=1)


def test_halfspace_poincare_target_manifold_validity(
    manifolds: tuple[Poincare, Hyperboloid, ProperVelocity],
    halfspace_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    curvature: float,
):
    """HS→P lands inside the ball; P→HS lands in the half-space (finite, ``x_n > 0``)."""
    poincare = manifolds[0]
    c = curvature

    assert jnp.all(_in_halfspace(halfspace_points)), "test points not in the half-space"
    p_BD = _batch(iso.halfspace_to_poincare)(halfspace_points, c)
    assert jnp.all(_batch(poincare.is_in_manifold)(p_BD, c))
    assert jnp.all(_in_halfspace(_batch(iso.poincare_to_halfspace)(poincare_points, c)))


def test_halfspace_poincare_round_trips(
    halfspace_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """Half-space <-> Poincaré round-trips to identity (both directions)."""
    c = curvature
    atol, rtol = tolerance

    x_rt = _batch(iso.poincare_to_halfspace)(_batch(iso.halfspace_to_poincare)(halfspace_points, c), c)
    assert jnp.allclose(x_rt, halfspace_points, atol=atol, rtol=rtol)

    p_rt = _batch(iso.halfspace_to_poincare)(_batch(iso.poincare_to_halfspace)(poincare_points, c), c)
    assert jnp.allclose(p_rt, poincare_points, atol=atol, rtol=rtol)


@pytest.mark.parametrize("c", [1e-3, 1e-1, 10.0, 100.0])
def test_halfspace_poincare_extreme_curvatures(c: float):
    """float64-only: HS -> P -> HS and P -> HS -> P round-trip at extreme curvatures (points scaled by 1/√c)."""
    x_BD = _hs_poincare_halfspace_points(jax.random.PRNGKey(2025), c, jnp.float64)
    p_BD = _hs_poincare_ball_points(jax.random.PRNGKey(2026), c, jnp.float64)

    x_rt = _batch(iso.poincare_to_halfspace)(_batch(iso.halfspace_to_poincare)(x_BD, c), c)
    assert jnp.allclose(x_rt, x_BD, atol=1e-9, rtol=1e-9), f"HS->P->HS round-trip failed at c={c}"
    p_rt = _batch(iso.halfspace_to_poincare)(_batch(iso.poincare_to_halfspace)(p_BD, c), c)
    assert jnp.allclose(p_rt, p_BD, atol=1e-9, rtol=1e-9), f"P->HS->P round-trip failed at c={c}"


def test_halfspace_poincare_origin_and_half_differential(curvature: float, dtype: jnp.dtype):
    """Origins map to origins (to 1 ulp), and the float64 differential at the origin is ``½·I`` (inverse: ``2·I``)."""
    c = curvature
    eps = float(jnp.finfo(dtype).eps)

    o_D = _halfspace_origin(c, dtype)
    zero_D = jnp.zeros(DIM, dtype=dtype)
    assert jnp.allclose(iso.halfspace_to_poincare(o_D, c), zero_D, atol=eps / np.sqrt(c), rtol=0.0)
    assert jnp.allclose(iso.poincare_to_halfspace(zero_D, c), o_D, atol=0.0, rtol=eps)

    o64_D = _halfspace_origin(c, jnp.float64)
    eye_DD = jnp.eye(DIM, dtype=jnp.float64)
    jac_fwd_DD = jax.jacfwd(iso.halfspace_to_poincare)(o64_D, c)
    jac_inv_DD = jax.jacfwd(iso.poincare_to_halfspace)(jnp.zeros(DIM, dtype=jnp.float64), c)
    assert jnp.allclose(jac_fwd_DD, 0.5 * eye_DD, atol=1e-14, rtol=0.0), "HS->P differential at o is not I/2"
    assert jnp.allclose(jac_inv_DD, 2.0 * eye_DD, atol=1e-14, rtol=0.0), "P->HS differential at 0 is not 2I"


def test_halfspace_poincare_isometry_preserves_distance(
    manifolds: tuple[Poincare, Hyperboloid, ProperVelocity],
    halfspace_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """``d_P(φx, φy)`` equals the NumPy half-space distance of ``(x, y)``, and ``d_HS(ψp, ψq)`` equals ``d_P(p, q)``."""
    poincare = manifolds[0]
    c = curvature
    atol, rtol = tolerance
    n = N_POINTS // 2

    def poincare_dist(a_BD, b_BD):
        return np.asarray(jax.vmap(lambda a, b: poincare.dist(a, b, c, version_idx=VERSION_MOBIUS_DIRECT))(a_BD, b_BD))

    xs, ys = halfspace_points[:n], halfspace_points[n : 2 * n]
    d_hs = _halfspace_dist_oracle(np.asarray(xs), np.asarray(ys), c)
    to_p = _batch(iso.halfspace_to_poincare)
    assert np.allclose(poincare_dist(to_p(xs, c), to_p(ys, c)), d_hs, atol=atol, rtol=rtol), "HS -> P not an isometry"

    ps, qs = poincare_points[:n], poincare_points[n : 2 * n]
    to_hs = _batch(iso.poincare_to_halfspace)
    d_back = _halfspace_dist_oracle(np.asarray(to_hs(ps, c)), np.asarray(to_hs(qs, c)), c)
    assert np.allclose(d_back, poincare_dist(ps, qs), atol=atol, rtol=rtol), "P -> HS not an isometry"


def test_halfspace_poincare_commutative_diagram(
    halfspace_points: jnp.ndarray,
    poincare_points: jnp.ndarray,
    curvature: float,
    tolerance: tuple[float, float],
):
    """HS→P equals the NumPy lift then ``hyperboloid_to_poincare``; P→HS equals ``poincare_to_hyperboloid`` then the unlift."""
    c = curvature
    atol, rtol = tolerance
    dtype = halfspace_points.dtype

    h_BD1 = jnp.asarray(_hs_poincare_lift(np.asarray(halfspace_points), c), dtype=dtype)
    via_h = _batch(iso.hyperboloid_to_poincare)(h_BD1, c)
    assert jnp.allclose(_batch(iso.halfspace_to_poincare)(halfspace_points, c), via_h, atol=atol, rtol=rtol), (
        "HS->P != HS->H->P"
    )

    via_h_back = _hs_poincare_unlift(np.asarray(_batch(iso.poincare_to_hyperboloid)(poincare_points, c)), c)
    direct = np.asarray(_batch(iso.poincare_to_halfspace)(poincare_points, c))
    assert np.allclose(direct, via_h_back, atol=atol, rtol=rtol), "P->HS != P->H->HS"


def test_halfspace_poincare_jit_and_vmap_compatibility(
    halfspace_points: jnp.ndarray,
    curvature: float,
    dtype: jnp.dtype,
    tolerance: tuple[float, float],
):
    """Both maps are JIT- and vmap-compatible, shape- and dtype-correct, and match eager."""
    c = curvature
    atol, rtol = tolerance

    p_BD = jax.jit(_batch(iso.halfspace_to_poincare))(halfspace_points, c)
    assert p_BD.shape == (N_POINTS, DIM) and p_BD.dtype == dtype
    assert jnp.allclose(p_BD, _batch(iso.halfspace_to_poincare)(halfspace_points, c), atol=atol, rtol=rtol)

    x_BD = jax.jit(_batch(iso.poincare_to_halfspace))(p_BD, c)
    assert x_BD.shape == (N_POINTS, DIM) and x_BD.dtype == dtype
    assert jnp.allclose(x_BD, _batch(iso.poincare_to_halfspace)(p_BD, c), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dim", [1, 2, 5, 10])
def test_halfspace_poincare_dimension_consistency(dim: int):
    """Both maps handle any dimension, including dim = 1 (empty ``x_s``), and preserve the distance to the origin."""
    c = 0.5
    x_D = jnp.concatenate([jnp.linspace(-0.3, 0.3, dim - 1, dtype=jnp.float64), jnp.array([0.7], dtype=jnp.float64)])
    p_D = jnp.linspace(-0.3, 0.3, dim, dtype=jnp.float64)

    p_img = iso.halfspace_to_poincare(x_D, c)
    assert p_img.shape == (dim,)
    assert jnp.allclose(iso.poincare_to_halfspace(p_img, c), x_D, atol=1e-12, rtol=1e-12)

    x_img = iso.poincare_to_halfspace(p_D, c)
    assert x_img.shape == (dim,)
    assert jnp.allclose(iso.halfspace_to_poincare(x_img, c), p_D, atol=1e-12, rtol=1e-12)

    d_p = Poincare(dtype=jnp.float64).dist_0(p_img, c, version_idx=VERSION_MOBIUS_DIRECT)
    d_hs = _halfspace_dist_oracle(np.asarray(x_D)[None], np.asarray(_halfspace_origin(c, jnp.float64, dim))[None], c)
    assert np.allclose(float(d_p), d_hs[0], atol=1e-12, rtol=1e-12)


def test_halfspace_poincare_gradients_are_finite_and_correct():
    """float64 reverse-mode gradients: ``w/2`` (``2w``) at the origin, central differences at scaled radius ≈ 8.

    The test function is ``w · φ(x)`` for a fixed ``w``, so its gradient is ``Jᵀw``: at the origin the
    differential is ``½·I`` (inverse ``2·I``); elsewhere a central finite difference is the oracle.
    """
    c = 0.5
    sqrt_c = np.sqrt(c)
    w_D = jnp.array([0.3, -1.1, 0.7], dtype=jnp.float64)

    def fwd_loss(x):
        return jnp.dot(w_D, iso.halfspace_to_poincare(x, c))

    def inv_loss(p):
        return jnp.dot(w_D, iso.poincare_to_halfspace(p, c))

    o_D = _halfspace_origin(c, jnp.float64)
    assert jnp.allclose(jax.grad(fwd_loss)(o_D), 0.5 * w_D, atol=1e-14, rtol=0.0)
    assert jnp.allclose(jax.grad(inv_loss)(jnp.zeros(DIM, dtype=jnp.float64)), 2.0 * w_D, atol=1e-14, rtol=0.0)

    # Scaled radius 8 from the origin: up and down the x_n axis, and sideways on the horosphere
    # x_n = 1/√c; in the ball, radius tanh(4)/√c along a generic direction. Each point carries its
    # local length scale for the finite-difference step: x_n in the half-space, the distance to the
    # boundary in the ball.
    far_hs = [
        jnp.array([0.0, 0.0, np.exp(8.0)], dtype=jnp.float64) / sqrt_c,
        jnp.array([0.0, 0.0, np.exp(-8.0)], dtype=jnp.float64) / sqrt_c,
        jnp.array([np.sqrt(2.0 * np.cosh(8.0) - 2.0), 0.0, 1.0], dtype=jnp.float64) / sqrt_c,
    ]
    direction_D = jnp.array([0.48, -0.6, 0.64], dtype=jnp.float64)
    far_p = [np.tanh(4.0) / sqrt_c * direction_D, -np.tanh(4.0) / sqrt_c * direction_D]
    cases = [(fwd_loss, z_D, float(z_D[-1])) for z_D in far_hs]
    cases += [(inv_loss, z_D, (1.0 - np.tanh(4.0)) / sqrt_c) for z_D in far_p]

    def central_difference(loss, z_D, scale):
        h = 1e-5 * scale
        steps_DD = h * jnp.eye(DIM, dtype=jnp.float64)
        return jax.vmap(lambda e: (loss(z_D + e) - loss(z_D - e)) / (2.0 * h))(steps_DD)

    for loss, z_D, scale in cases:
        g_D = jax.grad(loss)(z_D)
        assert jnp.all(jnp.isfinite(g_D)), z_D
        fd_D = central_difference(loss, z_D, scale)
        assert jnp.allclose(g_D, fd_D, rtol=1e-5, atol=1e-6 * jnp.max(jnp.abs(fd_D))), (z_D, g_D, fd_D)


def test_halfspace_poincare_keeps_float32_under_x64(halfspace_points: jnp.ndarray, dtype: jnp.dtype):
    """float32 in → float32 out with x64 enabled, also for a float64 curvature array."""
    for c in (0.5, jnp.asarray(0.5, dtype=jnp.float64)):
        p_BD = _batch(iso.halfspace_to_poincare)(halfspace_points, c)
        assert p_BD.dtype == dtype
        assert _batch(iso.poincare_to_halfspace)(p_BD, c).dtype == dtype


# ---------------------------------------------------------------------------
# Half-space ↔ Hyperboloid maps
# ---------------------------------------------------------------------------

# (halfspace-hyperboloid map tests go here)


# ---------------------------------------------------------------------------
# Half-space ↔ Klein maps
# ---------------------------------------------------------------------------

# (halfspace-klein map tests go here)


# ---------------------------------------------------------------------------
# Half-space ↔ PV maps
# ---------------------------------------------------------------------------

# (halfspace-pv map tests go here)
