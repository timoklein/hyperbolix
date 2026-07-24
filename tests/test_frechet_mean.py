"""Tests for the Karcher/Fréchet mean (``hyperbolix.decomposition.frechet_mean``).

Correctness anchors (all independent of the iteration itself):
- Karcher stationarity: the averaged log-map tangent vanishes at the mean.
- Two points → geodesic midpoint (equidistant, at half the pairwise distance; matches the
  closed-form Lorentz midpoint on the hyperboloid).
- A singleton is its own mean.
- Cross-model consistency: the Poincaré and hyperboloid means agree under the isometry.
- The output dtype is preserved and an explicit ``init_D`` reaches the same fixed point.
"""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.decomposition import frechet_mean
from hyperbolix.distributions import wrapped_normal_hyperboloid as wn
from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.manifolds import isometry_mappings as iso
from hyperbolix.nn_layers.hyperboloid_core import lorentz_midpoint

SEEDS = [10, 11, 12]
DIMS = [2, 5, 10]
CURVATURES = [0.3, 1.0, 2.5]


def _hyp_points(seed: int, n: int, dim: int, c: float, sigma: float, dtype) -> jnp.ndarray:
    """Wrapped-normal samples on the hyperboloid, shape (n, dim+1)."""
    H = Hyperboloid(dtype=dtype)
    mu0 = H.create_origin(c, dim)
    return wn.sample(jax.random.PRNGKey(seed), mu0, jnp.asarray(sigma, dtype=dtype), c, sample_shape=(n,), manifold_module=H)


def _stationarity_residual(x_NR, manifold, mean_R, c) -> float:
    """Riemannian norm of the averaged log-map tangent at ``mean_R`` (0 at a Karcher mean)."""
    logs = jax.vmap(manifold.logmap, in_axes=(0, None, None))(x_NR, mean_R, c)
    v = jnp.mean(logs, axis=0)
    return float(manifold.tangent_norm(v, mean_R, c))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", [2, 5, 10])
@pytest.mark.parametrize("seed", SEEDS)
def test_frechet_mean_karcher_stationarity(dtype, c, dim, seed):
    """The averaged log-map tangent vanishes at the mean (both models)."""
    # High dim + high curvature needs more Karcher steps to reach the fixed point; a modest
    # spread keeps the far-out points inside the float32-reliable regime (dist ≲ few).
    atol = 5e-5 if dtype == jnp.float32 else 1e-6
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    x_hyp = _hyp_points(seed, 24, dim, c, sigma=0.25, dtype=dtype)
    x_ball = jax.vmap(iso.hyperboloid_to_poincare, in_axes=(0, None))(x_hyp, c)

    mean_hyp = frechet_mean(x_hyp, H, c, max_iters=300)
    mean_ball = frechet_mean(x_ball, P, c, max_iters=300)
    assert _stationarity_residual(x_hyp, H, mean_hyp, c) < atol
    assert _stationarity_residual(x_ball, P, mean_ball, c) < atol
    assert bool(H.is_in_manifold(mean_hyp, c))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("seed", SEEDS)
def test_frechet_mean_two_points_is_geodesic_midpoint(dtype, c, seed):
    """Mean of two points is equidistant at half the pairwise distance and equals the Lorentz midpoint."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    H = Hyperboloid(dtype=dtype)
    x = _hyp_points(seed, 2, 5, c, sigma=0.5, dtype=dtype)
    x1, x2 = x[0], x[1]

    mean = frechet_mean(x, H, c)
    d1 = H.dist(x1, mean, c)
    d2 = H.dist(x2, mean, c)
    half = H.dist(x1, x2, c) / 2.0
    assert d1 == pytest.approx(float(half), abs=atol)
    assert d2 == pytest.approx(float(half), abs=atol)

    # Independent oracle: the closed-form uniform Lorentz midpoint.
    weights = jnp.full((1, 2), 0.5, dtype=dtype)
    midpoint = lorentz_midpoint(x, weights, c)[0]
    assert jnp.allclose(mean, midpoint, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
def test_frechet_mean_singleton_is_itself(dtype, c):
    """The Fréchet mean of a single point is that point (geodesic distance ≈ 0).

    Checks geodesic distance, not raw coordinates: at high curvature the point's ambient
    coordinates are large (~5), so float32's ~1e-6 relative resolution shows up as a ~1e-2
    coordinate gap despite the two points being geodesically ~2e-3 apart.
    """
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    H = Hyperboloid(dtype=dtype)
    x = _hyp_points(3, 1, 5, c, sigma=0.5, dtype=dtype)
    mean = frechet_mean(x, H, c)
    assert float(H.dist(mean, x[0], c)) == pytest.approx(0.0, abs=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("seed", SEEDS)
def test_frechet_mean_cross_model_consistency(dtype, c, seed):
    """Poincaré and hyperboloid Fréchet means agree under the stereographic isometry."""
    atol = 5e-3 if dtype == jnp.float32 else 1e-6
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    x_hyp = _hyp_points(seed, 24, 5, c, sigma=0.3, dtype=dtype)
    x_ball = jax.vmap(iso.hyperboloid_to_poincare, in_axes=(0, None))(x_hyp, c)

    mean_hyp = frechet_mean(x_hyp, H, c, max_iters=300)
    mean_ball = frechet_mean(x_ball, P, c, max_iters=300)
    mean_hyp_as_ball = iso.hyperboloid_to_poincare(mean_hyp, c)
    assert jnp.allclose(mean_ball, mean_hyp_as_ball, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
def test_frechet_mean_dtype_and_explicit_init(dtype, c):
    """Output dtype is preserved and an explicit init_D reaches the same fixed point."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    H = Hyperboloid(dtype=dtype)
    x = _hyp_points(11, 20, 5, c, sigma=0.3, dtype=dtype)

    mean_default = frechet_mean(x, H, c, max_iters=300)
    assert mean_default.dtype == dtype

    mean_explicit = frechet_mean(x, H, c, init_D=x[0], max_iters=300)
    assert jnp.allclose(mean_default, mean_explicit, atol=atol)
