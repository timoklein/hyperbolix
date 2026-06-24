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

SEEDS = [10, 11, 12]
DIMS = [2, 5, 10]
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
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", DIMS)
def test_busemann_origin_is_zero(dtype, c, dim):
    """B^v(origin) = 0 for unit v, both models."""
    atol, _ = _tol(dtype)
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    rng = np.random.default_rng(0)
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
@pytest.mark.parametrize("seed", SEEDS)
def test_busemann_cross_model_consistency(dtype, c, dim, seed):
    """Poincaré and Lorentz Busemann agree under the stereographic isometry (intrinsic quantity)."""
    atol, rtol = _tol(dtype)
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    rng = np.random.default_rng(seed)

    xP = jax.vmap(P.proj, in_axes=(0, None))(_ball_points(rng, 16, dim, c, dtype), c)
    xL = jax.vmap(iso.poincare_to_hyperboloid, in_axes=(0, None))(xP, c)
    v = _unit_dirs(rng, 6, dim, dtype)

    bP = jax.vmap(jax.vmap(P.busemann, in_axes=(None, 0, None)), in_axes=(0, None, None))(xP, v, c)
    bL = jax.vmap(jax.vmap(H.busemann, in_axes=(None, 0, None)), in_axes=(0, None, None))(xL, v, c)
    assert jnp.allclose(bP, bL, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("seed", SEEDS)
def test_busemann_along_ideal_ray(dtype, c, dim, seed):
    """B^v(gamma) = -d_0(gamma) along the geodesic ray from the origin toward the ideal point v.

    This is the coordinate-free Busemann identity (the ray toward v is the steepest descent of
    B^v, and B^v(origin) = 0), so it holds without assuming either model's exp-map is unit-speed.
    """
    # f32 Busemann drifts at large geodesic distance; keep the ray length modest.
    atol = 5e-3 if dtype == jnp.float32 else 1e-6
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    rng = np.random.default_rng(seed)
    v = _unit_dirs(rng, 1, dim, dtype)[0]
    ss = jnp.asarray(np.linspace(0.0, 1.5, 7), dtype=dtype)

    # Lorentz ray toward v: expmap_0 of the spatial tangent [0, v] scaled by s.
    tangentL = H.embed_spatial_0(v)
    ptsL = jax.vmap(lambda s: H.expmap_0(s * tangentL, c))(ss)
    bL = jax.vmap(H.busemann, in_axes=(0, None, None))(ptsL, v, c)
    dL = jax.vmap(H.dist_0, in_axes=(0, None))(ptsL, c)
    assert jnp.allclose(bL, -dL, atol=atol)

    # Poincaré ray toward v: expmap_0 of the tangent v scaled by s.
    ptsP = jax.vmap(lambda s: P.expmap_0(s * v, c))(ss)
    bP = jax.vmap(P.busemann, in_axes=(0, None, None))(ptsP, v, c)
    dP = jax.vmap(P.dist_0, in_axes=(0, None))(ptsP, c)
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

    busemann_fn = jax.jit(jax.vmap(jax.vmap(M.busemann, in_axes=(None, 0, None)), in_axes=(0, None, None)))
    b = busemann_fn(x, v, c)
    assert b.shape == (x.shape[0], v.shape[0])
    assert jnp.isfinite(b).all()
    assert b.dtype == dtype

    # Gradient w.r.t. the input point is finite.
    g = jax.grad(lambda xi: M.busemann(xi, v[0], c))(x[0])
    assert jnp.isfinite(g).all()
