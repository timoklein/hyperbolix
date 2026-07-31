"""Tests for the closed-form Busemann function on the Hyperboloid and Poincaré models.

The Busemann function ``B^v(x)`` (Chen et al. 2026, Eqs. 3/4) is a point-to-horosphere
coordinate. Correctness anchors used here:

- ``B^v(origin) = 0`` for unit ``v`` (both models).
- Cross-model isometry consistency: ``B`` is intrinsic, so the Poincaré and Lorentz formulas
  agree under ``poincare_to_hyperboloid`` (to machine precision).
- Busemann definition along the ideal ray: ``B^v(gamma(t)) = -t`` for the unit-speed geodesic ray
  gamma from the origin toward the ideal point ``v`` — pins the sign and the curvature scaling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.manifolds import isometry_mappings as iso

# The Busemann identities below are algebraic and hold pointwise, and each item already
# checks them on many (point, direction) pairs — so the seed axis buys nothing and is folded
# into the bodies (extra draws) instead of parametrized. ``dim`` enters only as a summation
# length (no dim-dependent branch); {2, 10} keeps the degenerate-ball edge and a wide case.
# The ``c`` axis IS load-bearing: dropping the 1/√c factor from the Poincaré Busemann fails
# the ray and cross-model tests at c ∈ {0.3, 2.5} and is a no-op at c = 1.0.
SEED = 10
DIMS = [2, 10]
ALL_DIMS = [2, 5, 10]
CURVATURES = [0.3, 1.0, 2.5]


def _tol(dtype: jnp.dtype) -> tuple[float, float]:
    return (4e-3, 4e-3) if dtype == jnp.float32 else (1e-7, 1e-7)


def _unit_dirs(rng: np.random.Generator, n: int, dim: int, dtype) -> jnp.ndarray:
    np_dtype = np.dtype(dtype)
    v = rng.normal(size=(n, dim)).astype(np_dtype)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.asarray(v, dtype=dtype)


def _ball_points(rng: np.random.Generator, n: int, dim: int, c: float, dtype) -> jnp.ndarray:
    """Random points strictly inside the Poincaré ball of radius 1/√c."""
    np_dtype = np.dtype(dtype)
    dirs = rng.normal(size=(n, dim)).astype(np_dtype)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    radii = rng.random((n, 1)).astype(np_dtype) ** (1.0 / dim) * (0.9 / np.sqrt(c))
    return jnp.asarray(dirs * radii, dtype=dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [1.0, 2.5])
def test_busemann_origin_is_zero(dtype, c):
    """B^v(origin) = 0 for unit v, both models.

    ``log(‖v‖²/1)/√c = 0`` and ``log(√c·(1/√c))/√c = 0`` are identically zero for every ``c``
    and every ``dim``, so the dims are looped in-body and only two curvatures are kept — this
    anchor cannot discriminate on either axis (verified: the ``1/√c`` mutation passes it 18/18).
    """
    atol, _ = _tol(dtype)
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    rng = np.random.default_rng(0)

    for dim in ALL_DIMS:
        v = _unit_dirs(rng, 4, dim, dtype)
        oH = H.create_origin(c, dim)
        oP = jnp.zeros(dim, dtype=dtype)
        bH = jax.vmap(H.busemann, in_axes=(None, 0, None))(oH, v, c)
        bP = jax.vmap(P.busemann, in_axes=(None, 0, None))(oP, v, c)
        assert jnp.allclose(bH, 0.0, atol=atol)
        assert jnp.allclose(bP, 0.0, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", DIMS)
def test_busemann_cross_model_consistency(dtype, c, dim):
    """Poincaré and Lorentz Busemann agree under the stereographic isometry (intrinsic quantity)."""
    atol, rtol = _tol(dtype)
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    rng = np.random.default_rng(SEED)

    xP = jax.vmap(P.proj, in_axes=(0, None))(_ball_points(rng, 16, dim, c, dtype), c)
    xL = jax.vmap(iso.poincare_to_hyperboloid, in_axes=(0, None))(xP, c)
    v = _unit_dirs(rng, 6, dim, dtype)

    bP = jax.vmap(jax.vmap(P.busemann, in_axes=(None, 0, None)), in_axes=(0, None, None))(xP, v, c)
    bL = jax.vmap(jax.vmap(H.busemann, in_axes=(None, 0, None)), in_axes=(0, None, None))(xL, v, c)
    assert jnp.allclose(bP, bL, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", DIMS)
def test_busemann_along_ideal_ray(dtype, c, dim):
    """B^v(gamma) = -d_0(gamma) along the geodesic ray from the origin toward the ideal point v.

    This is the coordinate-free Busemann identity (the ray toward v is the steepest descent of
    B^v, and B^v(origin) = 0), so it holds without assuming either model's exp-map is unit-speed.

    The old ``seed`` axis drew exactly one direction per item; the three directions are drawn
    in-body and vmapped over instead, which keeps the same coverage in one item.
    """
    # f32 Busemann drifts at large geodesic distance; keep the ray length modest.
    atol = 5e-3 if dtype == jnp.float32 else 1e-6
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    rng = np.random.default_rng(SEED)
    v_MD = _unit_dirs(rng, 3, dim, dtype)  # M ideal directions
    ss = jnp.asarray(np.linspace(0.0, 1.5, 7), dtype=dtype)  # S ray parameters

    # Lorentz ray toward v: expmap_0 of the spatial tangent [0, v] scaled by s.
    tangentL_MA = H.embed_spatial_0(v_MD)
    ptsL_MSA = jax.vmap(lambda t_A: jax.vmap(lambda s: H.expmap_0(s * t_A, c))(ss))(tangentL_MA)
    bL = jax.vmap(jax.vmap(H.busemann, in_axes=(0, None, None)), in_axes=(0, 0, None))(ptsL_MSA, v_MD, c)
    dL = jax.vmap(jax.vmap(H.dist_0, in_axes=(0, None)), in_axes=(0, None))(ptsL_MSA, c)
    assert jnp.allclose(bL, -dL, atol=atol)

    # Poincaré ray toward v: expmap_0 of the tangent v scaled by s.
    ptsP_MSD = jax.vmap(lambda v_D: jax.vmap(lambda s: P.expmap_0(s * v_D, c))(ss))(v_MD)
    bP = jax.vmap(jax.vmap(P.busemann, in_axes=(0, None, None)), in_axes=(0, 0, None))(ptsP_MSD, v_MD, c)
    dP = jax.vmap(jax.vmap(P.dist_0, in_axes=(0, None)), in_axes=(0, None))(ptsP_MSD, c)
    assert jnp.allclose(bP, -dP, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("manifold_cls, is_lorentz", [(Hyperboloid, True), (Poincare, False)])
def test_busemann_jit_vmap_finite_dtype(dtype, manifold_cls, is_lorentz):
    """busemann is jittable, vmappable, finite, and preserves the manifold dtype."""
    c = 1.0
    M = manifold_cls(dtype=dtype)
    rng = np.random.default_rng(0)
    dim = 5
    v = _unit_dirs(rng, 3, dim, dtype)

    xP = jax.vmap(Poincare(dtype=dtype).proj, in_axes=(0, None))(_ball_points(rng, 8, dim, c, dtype), c)
    x = jax.vmap(iso.poincare_to_hyperboloid, in_axes=(0, None))(xP, c) if is_lorentz else xP

    double_vmap = jax.vmap(jax.vmap(M.busemann, in_axes=(None, 0, None)), in_axes=(0, None, None))
    b = jax.jit(double_vmap)(x, v, c)
    assert b.shape == (x.shape[0], v.shape[0])
    assert jnp.isfinite(b).all()
    assert b.dtype == dtype

    # jit must reproduce eager to machine precision — without this the assertions above are
    # the eager sibling's assertions re-run, and a jit-only divergence would be invisible.
    # (Not bit-exact: XLA fuses the log/dot chain differently from the eager path, which
    # moves the Poincaré result by a few ULP.)
    ulps = 10.0 * float(jnp.finfo(dtype).eps)
    assert jnp.allclose(b, double_vmap(x, v, c), atol=ulps, rtol=ulps)

    # Gradient w.r.t. the input point is finite.
    g = jax.grad(lambda xi: M.busemann(xi, v[0], c))(x[0])
    assert jnp.isfinite(g).all()
