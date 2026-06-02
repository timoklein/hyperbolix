"""Tests for isometry mappings between hyperbolic manifold models.

Covers the distance-preserving transformations among the Poincaré ball,
hyperboloid (Lorentz), and Proper Velocity (PV) models:

    Poincaré ↔ Hyperboloid    (curvature-aware stereographic projection)
    Poincaré ↔ PV             (PVNN Eq. 4 gyro-isomorphism)
    Hyperboloid ↔ PV          (direct: PV coords = space-like 4-velocity part)

Every map is verified for: target-manifold validity, round-trip identity,
origin↦origin, geodesic-distance preservation (the defining isometry property),
the cross-model commutative diagram, and JIT/vmap compatibility. Tests are
parametrized over both dtypes (conftest ``dtype``/``tolerance``) and over a
range of curvatures — the latter specifically guards against curvature-dependent
bugs that a single ``c=1.0`` test cannot catch.
"""

import jax
import jax.numpy as jnp
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
    """All four new PV maps are JIT- and vmap-compatible and shape-correct."""
    c = curvature
    atol, rtol = tolerance

    to_h = jax.jit(_batch(iso.pv_to_hyperboloid))
    from_h = jax.jit(_batch(iso.hyperboloid_to_pv))
    to_p = jax.jit(_batch(iso.pv_to_poincare))
    from_p = jax.jit(_batch(iso.poincare_to_pv))

    z = to_h(pv_points, c)
    assert z.shape == (N_POINTS, DIM + 1)  # gains the time component
    assert jnp.allclose(from_h(z, c), pv_points, atol=atol, rtol=rtol)

    y = to_p(pv_points, c)
    assert y.shape == (N_POINTS, DIM)
    assert jnp.allclose(from_p(y, c), pv_points, atol=atol, rtol=rtol)


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
