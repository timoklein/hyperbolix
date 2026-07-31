"""Tests for the wrapped normal distribution (hyperboloid and Poincaré ball).

Dimension key:
  N: number of points      S: sample dimensions (from sample_shape)
  B: batch dimensions (from a batched mean)
  D: spatial dim (n)       A: ambient dim (n+1, hyperboloid time coordinate first)
  Q: radial quadrature nodes                T: angular quadrature nodes
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from hyperbolix.distributions import wrapped_normal_hyperboloid, wrapped_normal_poincare
from hyperbolix.distributions._common import gaussian_log_prob, sigma_to_cov
from hyperbolix.distributions._wrapped_normal_base import _log_det_jacobian_from_r
from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.manifolds import isometry_mappings as iso
from hyperbolix.manifolds.hyperboloid import VERSION_DEFAULT

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _create_hyperboloid_origin(manifold: Hyperboloid, c: float, dim: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Create hyperboloid origin via class-based expmap at the origin."""
    tangent_0 = jnp.zeros(dim + 1, dtype=dtype)
    return manifold.expmap_0(tangent_0, c)


def _batch_is_on_hyperboloid(manifold: Hyperboloid, points: jnp.ndarray, c: float, atol: float = 1e-5) -> bool:
    """Check if all points in batch are on hyperboloid."""
    is_in = jax.vmap(lambda p: manifold.is_in_manifold(p, c=c, atol=atol))
    return bool(jnp.all(is_in(points)))


def _batch_is_in_poincare(manifold: Poincare, points: jnp.ndarray, c: float, atol: float = 1e-5) -> bool:
    """Check if all points in batch are in Poincaré ball."""
    is_in = jax.vmap(lambda p: manifold.is_in_manifold(p, c=c, atol=atol))
    return bool(jnp.all(is_in(points)))


# ---------------------------------------------------------------------------
# Tests - Sampling
# ---------------------------------------------------------------------------


def test_sample_single_point_isotropic(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test basic sampling with a single mean point and isotropic covariance."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    # Create mean at origin in H^2 (3D ambient space)
    c = 1.0
    hyperboloid = Hyperboloid(dtype=dtype)
    mu = _create_hyperboloid_origin(hyperboloid, c, dim=2, dtype=dtype)

    # Sample with isotropic covariance
    sigma = 0.1
    z = wrapped_normal_hyperboloid.sample(key, mu, sigma, c)

    # Check output shape
    assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"

    # Check point is on manifold
    assert hyperboloid.is_in_manifold(z, c, atol=atol), "Sampled point not on hyperboloid"

    # JIT must reproduce the eager draw exactly (folded in from the former standalone
    # test_sample_jit_compatibility, which only re-asserted the eager properties above).
    sample_jit = jax.jit(wrapped_normal_hyperboloid.sample, static_argnames=["sample_shape"])
    assert jnp.allclose(sample_jit(key, mu, sigma, c, sample_shape=()), z, atol=atol)


def test_sample_shape_parameter(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test that sample_shape parameter works correctly."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    # Create mean
    c = 1.0
    hyperboloid = Hyperboloid(dtype=dtype)
    mu = _create_hyperboloid_origin(hyperboloid, c, dim=2, dtype=dtype)
    sigma = 0.1

    # Sample with sample_shape
    z = wrapped_normal_hyperboloid.sample(key, mu, sigma, c, sample_shape=(5, 3))

    # Check output shape: sample_shape + mu.shape
    assert z.shape == (5, 3, 3), f"Expected shape (5, 3, 3), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"

    # Check all points are on manifold
    z_flat = z.reshape(-1, 3)
    assert _batch_is_on_hyperboloid(hyperboloid, z_flat, c, atol=atol), "Not all sampled points on hyperboloid"


def test_sample_diagonal_covariance(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test sampling with diagonal covariance."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    # Create mean in H^2
    c = 1.0
    hyperboloid = Hyperboloid(dtype=dtype)
    mu = _create_hyperboloid_origin(hyperboloid, c, dim=2, dtype=dtype)

    # Diagonal covariance for spatial dimension (n=2)
    sigma_diag = jnp.array([0.1, 0.2], dtype=dtype)

    z = wrapped_normal_hyperboloid.sample(key, mu, sigma_diag, c)

    # Check output shape
    assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"

    # Check point is on manifold
    assert hyperboloid.is_in_manifold(z, c, atol=atol), "Sampled point not on hyperboloid"


def test_sample_full_covariance(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test sampling with full covariance matrix."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    # Create mean in H^2
    c = 1.0
    hyperboloid = Hyperboloid(dtype=dtype)
    mu = _create_hyperboloid_origin(hyperboloid, c, dim=2, dtype=dtype)

    # Full covariance for spatial dimension (n=2)
    sigma_full = jnp.array([[0.1, 0.01], [0.01, 0.1]], dtype=dtype)

    z = wrapped_normal_hyperboloid.sample(key, mu, sigma_full, c)

    # Check output shape
    assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"

    # Check point is on manifold
    assert hyperboloid.is_in_manifold(z, c, atol=atol), "Sampled point not on hyperboloid"


def test_sample_dtype_propagation(tolerance: tuple[float, float]) -> None:
    """Test that dtype parameter is respected."""
    key = jax.random.PRNGKey(42)
    _atol, _rtol = tolerance

    # Create mean in H^2 with float32
    c = 1.0
    hyperboloid_f32 = Hyperboloid(dtype=jnp.float32)
    hyperboloid_f64 = Hyperboloid(dtype=jnp.float64)
    mu = _create_hyperboloid_origin(hyperboloid_f32, c, dim=2, dtype=jnp.float32)
    sigma = 0.1

    # Sample with explicit float64 dtype
    z_f64 = wrapped_normal_hyperboloid.sample(key, mu, sigma, c, dtype=jnp.float64)

    assert z_f64.dtype == jnp.float64, f"Expected dtype float64, got {z_f64.dtype}"
    assert hyperboloid_f64.is_in_manifold(z_f64, c, atol=1e-10), "Float64 sample not on hyperboloid"

    # Sample with explicit float32 dtype
    z_f32 = wrapped_normal_hyperboloid.sample(key, mu, sigma, c, dtype=jnp.float32)

    assert z_f32.dtype == jnp.float32, f"Expected dtype float32, got {z_f32.dtype}"
    assert hyperboloid_f32.is_in_manifold(z_f32, c, atol=1e-5), "Float32 sample not on hyperboloid"


def test_sample_batched_means(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test sampling with batched mean points."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    # Create batch of means in H^2
    c = 1.0
    hyperboloid = Hyperboloid(dtype=dtype)
    mu_batch = jnp.array(
        [
            [1.0, 0.0, 0.0],  # origin
            [1.0, 0.1, 0.1],  # another point (will be projected)
        ],
        dtype=dtype,
    )
    # Project to ensure they're on manifold
    mu_batch = jax.vmap(lambda m: hyperboloid.proj(m, c))(mu_batch)

    sigma = 0.1

    z = wrapped_normal_hyperboloid.sample(key, mu_batch, sigma, c)

    # Check output shape: should match mu_batch.shape
    assert z.shape == (2, 3), f"Expected shape (2, 3), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"

    # Check all points are on manifold
    assert _batch_is_on_hyperboloid(hyperboloid, z, c, atol=atol), "Not all batched samples on hyperboloid"


def test_sample_different_curvatures(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test sampling with different curvature values."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance
    hyperboloid = Hyperboloid(dtype=dtype)

    for c in [0.5, 1.0, 2.0]:
        mu = _create_hyperboloid_origin(hyperboloid, c, dim=2, dtype=dtype)
        sigma = 0.1

        z = wrapped_normal_hyperboloid.sample(key, mu, sigma, c)

        assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"
        assert hyperboloid.is_in_manifold(z, c, atol=atol), f"Sample not on hyperboloid with c={c}"


# ---------------------------------------------------------------------------
# Tests - JAX Compatibility
# ---------------------------------------------------------------------------


def test_sample_vmap_compatibility(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test that sampling works with JAX vmap."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    # Create multiple keys
    keys = jax.random.split(key, 10)

    c = 1.0
    hyperboloid = Hyperboloid(dtype=dtype)
    mu = _create_hyperboloid_origin(hyperboloid, c, dim=2, dtype=dtype)
    sigma = 0.1

    # vmap over keys
    sample_vmap = jax.vmap(lambda k: wrapped_normal_hyperboloid.sample(k, mu, sigma, c))
    z_batch = sample_vmap(keys)

    assert z_batch.shape == (10, 3), f"Expected shape (10, 3), got {z_batch.shape}"
    assert _batch_is_on_hyperboloid(hyperboloid, z_batch, c, atol=atol), "Not all vmapped samples on hyperboloid"


def test_sample_gradient_flow(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test that sampling is differentiable w.r.t. mean."""
    key = jax.random.PRNGKey(42)
    _atol, _rtol = tolerance

    c = 1.0
    hyperboloid = Hyperboloid(dtype=dtype)
    mu = _create_hyperboloid_origin(hyperboloid, c, dim=3, dtype=dtype)
    sigma = 0.1

    # Define a simple loss that uses sampling
    def loss_fn(mu_param):
        # Sample and compute some loss (e.g., distance from origin)
        z = wrapped_normal_hyperboloid.sample(key, mu_param, sigma, c)
        origin = _create_hyperboloid_origin(hyperboloid, c, dim=3, dtype=dtype)
        dist = hyperboloid.dist(z, origin, c, version_idx=VERSION_DEFAULT)
        return dist

    # Compute gradient
    grad_mu = jax.grad(loss_fn)(mu)

    # Check that gradient is finite and has correct shape
    assert grad_mu.shape == mu.shape, f"Gradient shape mismatch: {grad_mu.shape} vs {mu.shape}"
    assert jnp.all(jnp.isfinite(grad_mu)), "Gradient contains NaN or Inf"


# ---------------------------------------------------------------------------
# Tests - Helper Functions
# ---------------------------------------------------------------------------


def test_embed_spatial_0_single(dtype: jnp.dtype) -> None:
    """Test embedding a single spatial vector at origin."""
    hyperboloid = Hyperboloid(dtype=dtype)
    v_spatial = jnp.array([0.1, 0.2], dtype=dtype)
    v_tangent = hyperboloid.embed_spatial_0(v_spatial)

    assert v_tangent.shape == (3,), f"Expected shape (3,), got {v_tangent.shape}"
    assert v_tangent[0] == 0.0, f"First component should be 0, got {v_tangent[0]}"
    assert jnp.allclose(v_tangent[1:], v_spatial), "Spatial components should match"
    assert v_tangent.dtype == dtype, f"Expected dtype {dtype}, got {v_tangent.dtype}"


def test_embed_spatial_0_batch(dtype: jnp.dtype) -> None:
    """Test embedding batched spatial vectors at origin."""
    hyperboloid = Hyperboloid(dtype=dtype)
    v_spatial = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=dtype)
    v_tangent = hyperboloid.embed_spatial_0(v_spatial)

    assert v_tangent.shape == (2, 3), f"Expected shape (2, 3), got {v_tangent.shape}"
    assert jnp.all(v_tangent[:, 0] == 0.0), "First component should be 0"
    assert jnp.allclose(v_tangent[:, 1:], v_spatial), "Spatial components should match"
    assert v_tangent.dtype == dtype, f"Expected dtype {dtype}, got {v_tangent.dtype}"


# ---------------------------------------------------------------------------
# Tests - Poincaré Ball Sampling
# ---------------------------------------------------------------------------


def test_sample_poincare_single_point_isotropic(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test basic Poincaré sampling with a single mean point and isotropic covariance."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    # Create mean at origin
    c = 1.0
    poincare = Poincare(dtype=dtype)
    mu = jnp.zeros(2, dtype=dtype)

    # Sample with isotropic covariance
    sigma = 0.1
    z = wrapped_normal_poincare.sample(key, mu, sigma, c)

    # Check output shape
    assert z.shape == (2,), f"Expected shape (2,), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"

    # Check point is in Poincaré ball
    assert poincare.is_in_manifold(z, c, atol=atol), "Sampled point not in Poincaré ball"

    # JIT must reproduce the eager draw exactly (folded in from the former standalone
    # test_sample_poincare_jit_compatibility, which only re-asserted the eager properties).
    sample_jit = jax.jit(wrapped_normal_poincare.sample, static_argnames=["sample_shape"])
    assert jnp.allclose(sample_jit(key, mu, sigma, c, sample_shape=()), z, atol=atol)


def test_sample_poincare_shape_parameter(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test that sample_shape parameter works for Poincaré."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    c = 1.0
    poincare = Poincare(dtype=dtype)
    mu = jnp.zeros(2, dtype=dtype)
    sigma = 0.1

    # Sample with sample_shape
    z = wrapped_normal_poincare.sample(key, mu, sigma, c, sample_shape=(5, 3))

    # Check output shape: sample_shape + mu.shape
    assert z.shape == (5, 3, 2), f"Expected shape (5, 3, 2), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"

    # Check all points are in ball
    z_flat = z.reshape(-1, 2)
    assert _batch_is_in_poincare(poincare, z_flat, c, atol=atol), "Not all sampled points in Poincaré ball"


def test_sample_poincare_diagonal_covariance(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test Poincaré sampling with diagonal covariance."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    c = 1.0
    poincare = Poincare(dtype=dtype)
    mu = jnp.zeros(2, dtype=dtype)

    # Diagonal covariance
    sigma_diag = jnp.array([0.1, 0.2], dtype=dtype)

    z = wrapped_normal_poincare.sample(key, mu, sigma_diag, c)

    assert z.shape == (2,), f"Expected shape (2,), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"
    assert poincare.is_in_manifold(z, c, atol=atol), "Sampled point not in Poincaré ball"


def test_sample_poincare_full_covariance(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test Poincaré sampling with full covariance matrix."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    c = 1.0
    poincare = Poincare(dtype=dtype)
    mu = jnp.zeros(2, dtype=dtype)

    # Full covariance
    sigma_full = jnp.array([[0.1, 0.01], [0.01, 0.1]], dtype=dtype)

    z = wrapped_normal_poincare.sample(key, mu, sigma_full, c)

    assert z.shape == (2,), f"Expected shape (2,), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"
    assert poincare.is_in_manifold(z, c, atol=atol), "Sampled point not in Poincaré ball"


def test_sample_poincare_batched_means(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test Poincaré sampling with batched mean points."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    c = 1.0
    poincare = Poincare(dtype=dtype)
    mu_batch = jnp.array(
        [
            [0.0, 0.0],  # origin
            [0.1, 0.1],  # another point
        ],
        dtype=dtype,
    )
    # Project to ensure they're in ball
    mu_batch = jax.vmap(lambda m: poincare.proj(m, c))(mu_batch)

    sigma = 0.1

    z = wrapped_normal_poincare.sample(key, mu_batch, sigma, c)

    assert z.shape == (2, 2), f"Expected shape (2, 2), got {z.shape}"
    assert z.dtype == dtype, f"Expected dtype {dtype}, got {z.dtype}"
    assert _batch_is_in_poincare(poincare, z, c, atol=atol), "Not all batched samples in Poincaré ball"


def test_sample_poincare_different_curvatures(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test Poincaré sampling with different curvature values."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance
    poincare = Poincare(dtype=dtype)

    for c in [0.5, 1.0, 2.0]:
        mu = jnp.zeros(2, dtype=dtype)
        sigma = 0.1

        z = wrapped_normal_poincare.sample(key, mu, sigma, c)

        assert z.shape == (2,), f"Expected shape (2,), got {z.shape}"
        assert poincare.is_in_manifold(z, c, atol=atol), f"Sample not in Poincaré ball with c={c}"


def test_sample_poincare_vmap_compatibility(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test that Poincaré sampling works with JAX vmap."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    # Create multiple keys
    keys = jax.random.split(key, 10)

    c = 1.0
    poincare = Poincare(dtype=dtype)
    mu = jnp.zeros(2, dtype=dtype)
    sigma = 0.1

    # vmap over keys
    sample_vmap = jax.vmap(lambda k: wrapped_normal_poincare.sample(k, mu, sigma, c))
    z_batch = sample_vmap(keys)

    assert z_batch.shape == (10, 2), f"Expected shape (10, 2), got {z_batch.shape}"
    assert _batch_is_in_poincare(poincare, z_batch, c, atol=atol), "Not all vmapped samples in Poincaré ball"


def test_sample_poincare_gradient_flow(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test that Poincaré sampling is differentiable w.r.t. mean."""
    key = jax.random.PRNGKey(42)
    _atol, _rtol = tolerance

    c = 1.0
    mu = jnp.zeros(3, dtype=dtype)
    sigma = 0.1

    # Define a simple loss that uses sampling
    def loss_fn(mu_param):
        # Sample and compute some loss (e.g., norm)
        z = wrapped_normal_poincare.sample(key, mu_param, sigma, c)
        return jnp.linalg.norm(z)

    # Compute gradient
    grad_mu = jax.grad(loss_fn)(mu)

    # Check that gradient is finite and has correct shape
    assert grad_mu.shape == mu.shape, f"Gradient shape mismatch: {grad_mu.shape} vs {mu.shape}"
    assert jnp.all(jnp.isfinite(grad_mu)), "Gradient contains NaN or Inf"


# ---------------------------------------------------------------------------
# Tests - Distributional semantics
#
# Everything above is satisfied by the constant map z ↦ mu: a ``sample_gaussian``
# that returns zeros (sigma ignored, variance identically 0) passes 40 of the 42
# items this file used to collect. The tests below pin what ``sigma`` actually does,
# for all three of its parameterizations and both models.
# ---------------------------------------------------------------------------

_DIST_N = 20_000
"""Sample count for the distributional checks. At sigma ≈ 0.3 the standard error of the
tangent mean is sigma/sqrt(N) ≈ 2e-3 and of a covariance entry ≈ sigma²/sqrt(N) ≈ 6e-4."""


def _sigma_for(kind: str, n: int, dtype: jnp.dtype):
    """Sigma in each of the three parameterizations ``sigma_to_cov`` accepts."""
    if kind == "scalar":
        return jnp.asarray(0.3, dtype=dtype)
    if kind == "diag":
        return jnp.asarray([0.2, 0.3, 0.4][:n], dtype=dtype)
    # Full covariance: SPD with genuine off-diagonal correlation (L @ L.T, L lower-triangular).
    chol = jnp.asarray([[0.30, 0.0, 0.0], [0.12, 0.25, 0.0], [0.05, 0.08, 0.20]], dtype=dtype)[:n, :n]
    return chol @ chol.T


def _assert_tangent_moments(v_ND: jnp.ndarray, cov_DD: jnp.ndarray) -> None:
    """The recovered origin-tangent noise must have mean 0 and covariance ``cov_DD``."""
    n_samples = v_ND.shape[0]

    # (i) Mean within 3 standard errors of zero, per coordinate.
    se_D = jnp.sqrt(jnp.diag(cov_DD) / n_samples)
    mean_D = jnp.mean(v_ND, axis=0)
    assert jnp.all(jnp.abs(mean_D) < 3.0 * se_D), f"tangent mean {mean_D} outside 3 s.e. {3.0 * se_D}"

    # (ii) Empirical covariance matches sigma_to_cov. The atol floor (5% of the largest
    # variance, ≈ 7 standard errors at N = 20k) is what makes the zero off-diagonals of the
    # scalar/diagonal parameterizations checkable at all.
    cov_emp_DD = jnp.cov(v_ND, rowvar=False, bias=True)
    atol = 0.05 * float(jnp.max(jnp.diag(cov_DD)))
    assert jnp.allclose(cov_emp_DD, cov_DD, rtol=0.1, atol=atol), f"empirical cov\n{cov_emp_DD}\nexpected\n{cov_DD}"


@pytest.mark.parametrize("sigma_kind", ["scalar", "diag", "full"])
def test_sample_hyperboloid_tangent_moments_match_sigma(sigma_kind: str) -> None:
    """Samples transported back to the origin have moments (0, sigma_to_cov(sigma)).

    ``sample`` is ``z = exp_mu(PT_{o→mu}([0, v]))`` with ``v ~ N(0, Sigma)``; both maps are
    invertible, so ``PT_{mu→o}(log_mu(z))`` recovers ``[0, v]`` exactly and its sample moments
    are a direct read-out of ``Sigma``. Fixed seed, float64, non-origin mean.
    """
    dtype, c, n = jnp.float64, 1.0, 3
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(0)

    origin = _create_hyperboloid_origin(hyperboloid, c, dim=n, dtype=dtype)
    mu = hyperboloid.expmap_0(hyperboloid.embed_spatial_0(jnp.asarray([0.4, -0.3, 0.2], dtype=dtype)), c)
    sigma = _sigma_for(sigma_kind, n, dtype)

    z_NA = wrapped_normal_hyperboloid.sample(key, mu, sigma, c, sample_shape=(_DIST_N,), manifold_module=hyperboloid)
    u_NA = jax.vmap(hyperboloid.logmap, in_axes=(0, None, None))(z_NA, mu, c)
    v_NA = jax.vmap(hyperboloid.ptransp, in_axes=(0, None, None, None))(u_NA, mu, origin, c)

    assert jnp.allclose(v_NA[:, 0], 0.0, atol=1e-9), "transported tangent is not purely spatial at the origin"
    _assert_tangent_moments(v_NA[:, 1:], sigma_to_cov(sigma, n, dtype))


@pytest.mark.parametrize("sigma_kind", ["scalar", "diag", "full"])
def test_sample_poincare_tangent_moments_match_sigma(sigma_kind: str) -> None:
    """Samples mapped back to the origin tangent space have moments (0, sigma_to_cov(sigma)).

    ``sample`` is ``z = mu ⊕ exp_0(v/2)`` with ``v ~ N(0, Sigma)`` (the ``/2`` is the conformal
    factor at the origin), so ``v = 2·log_0(⊖mu ⊕ z)`` inverts it.
    """
    dtype, c, n = jnp.float64, 1.0, 3
    poincare = Poincare(dtype=dtype)
    key = jax.random.PRNGKey(0)

    mu = jnp.asarray([0.2, -0.1, 0.15], dtype=dtype)
    sigma = _sigma_for(sigma_kind, n, dtype)

    z_ND = wrapped_normal_poincare.sample(key, mu, sigma, c, sample_shape=(_DIST_N,), manifold_module=poincare)
    z0_ND = jax.vmap(poincare.addition, in_axes=(None, 0, None))(-mu, z_ND, c)
    v_ND = 2.0 * jax.vmap(poincare.logmap_0, in_axes=(0, None))(z0_ND, c)

    _assert_tangent_moments(v_ND, sigma_to_cov(sigma, n, dtype))


# =============================================================================================
# log_prob value oracles (audit M2-01)
#
# Nothing in this file used to assert a log_prob *value*: the entire log-det-Jacobian term could
# be deleted and every item still passed. The four oracles below pin the value from four
# independent directions, and with it the reading of the density — it is taken w.r.t. the
# RIEMANNIAN VOLUME measure of H^n, not the Lebesgue measure of a coordinate chart.
#
# All float64 with fixed seeds: these are exact analytic identities, so a float32 axis would
# only measure rounding.
# =============================================================================================

_F64 = jnp.float64
_QUAD_R_MAX = 6.0
"""Radial cut-off of the polar quadrature. At sigma = 0.5 the tangent-space mass beyond
r = 6 (12 sigma) is ~e^-72, far below the 1e-3 tolerance."""


def _geodesic_frame_h2(c: float, d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy H² frame: a mean at geodesic distance ``d`` from the origin plus two unit tangents.

    Built without the library. With ``⟨x,x⟩_L = -1/c`` the unit-speed geodesic ``g`` leaving the
    origin ``o = (1/√c, 0, 0)`` along the spacelike unit vector ``e₁`` is
    ``g(t) = cosh(√c·t)·o + sinh(√c·t)/√c · e₁``; the mean is ``g(d)``, its velocity ``g'(d)`` is
    a unit tangent there, and ``e₂`` (untouched by the geodesic) completes the orthonormal pair.
    """
    sqrt_c = np.sqrt(c)
    o_A = np.array([1.0 / sqrt_c, 0.0, 0.0])
    e1_A = np.array([0.0, 1.0, 0.0])
    e2_A = np.array([0.0, 0.0, 1.0])
    mu_A = np.cosh(sqrt_c * d) * o_A + np.sinh(sqrt_c * d) / sqrt_c * e1_A
    tangent_A = sqrt_c * np.sinh(sqrt_c * d) * o_A + np.cosh(sqrt_c * d) * e1_A  # g'(d)
    return mu_A, tangent_A, e2_A


@pytest.mark.parametrize("c", [0.4, 1.0, 2.0])
def test_log_prob_integrates_to_one_over_h2(c: float) -> None:
    """(a) ``∫ exp(log_prob) dV = 1`` in geodesic polar coordinates, both models, non-origin mean.

    The area element ``(sinh(√c·r)/√c)^{n-1} dr dθ`` is the Riemannian volume of H², the measure
    the density is defined against — integrating against the Lebesgue element of any chart would
    not give 1. Discriminating case: every factor of the log-det Jacobian (the ``n-1`` exponent,
    the ``√c`` inside the sinh, the term itself) moves this integral off 1.
    """
    sqrt_c = np.sqrt(c)
    sigma = 0.5
    mu_A, b1_A, b2_A = _geodesic_frame_h2(c, d=0.7)

    # Gauss-Legendre in r, periodic trapezoid in theta (spectrally accurate for a smooth
    # periodic integrand). The observed quadrature error is ~1e-13, 10 orders below the tol.
    nodes_Q, weights_Q = np.polynomial.legendre.leggauss(160)
    r_Q = 0.5 * _QUAD_R_MAX * (nodes_Q + 1.0)
    w_r_Q = 0.5 * _QUAD_R_MAX * weights_Q
    n_theta = 64
    theta_T = 2.0 * np.pi * np.arange(n_theta) / n_theta

    r_QT, theta_QT = np.meshgrid(r_Q, theta_T, indexing="ij")
    dir_QTA = np.cos(theta_QT)[..., None] * b1_A + np.sin(theta_QT)[..., None] * b2_A
    z_QTA = np.cosh(sqrt_c * r_QT)[..., None] * mu_A + (np.sinh(sqrt_c * r_QT) / sqrt_c)[..., None] * dir_QTA
    area_QT = np.sinh(sqrt_c * r_QT) / sqrt_c  # (sinh(√c·r)/√c)^{n-1} with n = 2
    quad_QT = w_r_Q[:, None] * (2.0 * np.pi / n_theta) * area_QT

    hyperboloid, poincare = Hyperboloid(dtype=_F64), Poincare(dtype=_F64)
    z_NA = jnp.asarray(z_QTA.reshape(-1, 3))
    mu_jax_A = jnp.asarray(mu_A)

    lp_N = wrapped_normal_hyperboloid.log_prob(z_NA, mu_jax_A, sigma, c, manifold_module=hyperboloid)
    mass_hyp = float(np.sum(quad_QT * np.asarray(jnp.exp(lp_N)).reshape(quad_QT.shape)))
    assert mass_hyp == pytest.approx(1.0, abs=1e-3), f"hyperboloid density integrates to {mass_hyp}"

    # Same grid, same volume element: the stereographic isometry preserves dV, so the ball
    # density must integrate to 1 against the very same weights.
    z_ball_ND = jax.vmap(iso.hyperboloid_to_poincare, in_axes=(0, None))(z_NA, c)
    mu_ball_D = iso.hyperboloid_to_poincare(mu_jax_A, c)
    lp_ball_N = wrapped_normal_poincare.log_prob(z_ball_ND, mu_ball_D, sigma, c, manifold_module=poincare)
    mass_ball = float(np.sum(quad_QT * np.asarray(jnp.exp(lp_ball_N)).reshape(quad_QT.shape)))
    assert mass_ball == pytest.approx(1.0, abs=1e-3), f"ball density integrates to {mass_ball}"


@pytest.mark.parametrize("c", [0.4, 1.3])
def test_log_prob_matches_pushforward_change_of_variables(c: float) -> None:
    """(b) log_prob equals the change-of-variables density of the sampling map itself.

    ``sample`` pushes ``v ~ N(0, Σ)`` through a map ``T``; the image's Lebesgue density in a
    chart is ``N(v; 0, Σ)/|det ∂T/∂v|`` (Jacobian from ``jacfwd``, never from the library's
    closed form), and converting a volume-measure density into that chart multiplies by the
    metric volume factor: ``λ(z)^n`` on the ball, ``√det G = 1/(√c·x₀)`` in the hyperboloid's
    spatial chart (``G = I - s sᵀ/x₀²``, so ``det G = 1/(c·x₀²)``).
    """
    n = 3
    hyperboloid, poincare = Hyperboloid(dtype=_F64), Poincare(dtype=_F64)
    sigma_DD = jnp.asarray([[0.30, 0.05, 0.0], [0.05, 0.22, 0.03], [0.0, 0.03, 0.18]], dtype=_F64)
    mu_A = hyperboloid.expmap_0(hyperboloid.embed_spatial_0(jnp.asarray([0.4, -0.25, 0.1], dtype=_F64)), c)
    mu_ball_D = iso.hyperboloid_to_poincare(mu_A, c)
    gauss = scipy.stats.multivariate_normal(mean=np.zeros(n), cov=np.asarray(sigma_DD))

    def spatial_of(v_D):
        """Sampling map into the hyperboloid's spatial chart: s(v) = expmap_μ(PT_{0→μ}[0, v])_s."""
        u_A = hyperboloid.ptransp_0(hyperboloid.embed_spatial_0(v_D), mu_A, c)
        return hyperboloid.expmap(u_A, mu_A, c)[1:]

    def ball_of(v_D):
        """Sampling map into the ball: z(v) = μ ⊕ exp_0(v/2) (the /2 is λ(0) = 2)."""
        return poincare.addition(mu_ball_D, poincare.expmap_0(v_D / 2.0, c), c)

    draws_ND = np.random.default_rng(0).multivariate_normal(np.zeros(n), np.asarray(sigma_DD), size=4)
    for v_D in jnp.asarray(draws_ND, dtype=_F64):
        log_n_v = float(gauss.logpdf(np.asarray(v_D)))

        s_D = spatial_of(v_D)
        x0 = jnp.sqrt(1.0 / c + jnp.sum(s_D**2))
        z_A = jnp.concatenate([x0[None], s_D])
        lp_A = wrapped_normal_hyperboloid.log_prob(z_A, mu_A, sigma_DD, c, manifold_module=hyperboloid)
        lhs_hyp = float(lp_A - jnp.log(jnp.sqrt(c) * x0))  # log(p_vol · √det G)
        rhs_hyp = log_n_v - float(jnp.log(jnp.abs(jnp.linalg.det(jax.jacfwd(spatial_of)(v_D)))))
        assert lhs_hyp == pytest.approx(rhs_hyp, abs=1e-10)

        z_D = ball_of(v_D)
        lam = 2.0 / (1.0 - c * jnp.sum(z_D**2))
        lp_D = wrapped_normal_poincare.log_prob(z_D, mu_ball_D, sigma_DD, c, manifold_module=poincare)
        lhs_ball = float(lp_D + n * jnp.log(lam))  # log(p_vol · λ^n)
        rhs_ball = log_n_v - float(jnp.log(jnp.abs(jnp.linalg.det(jax.jacfwd(ball_of)(v_D)))))
        assert lhs_ball == pytest.approx(rhs_ball, abs=1e-10)


def test_log_prob_flat_limit_matches_scipy_gaussian() -> None:
    """(c) At c = 1e-4 both models reduce to a Euclidean Gaussian in Riemannian coordinates.

    Hyperbolic space flattens as c → 0, so ``log p(exp_0(v)) → log N(v; 0, Σ)`` with the volume
    measure tending to the Lebesgue one. The residual here is the O(c·r²) curvature correction
    (~1e-4), while a dropped 2π, a dropped ``|Σ|`` or a mishandled full covariance is an O(1)
    offset — hence the 1e-3 tolerance rather than a loose one.
    """
    c, n = 1e-4, 3
    hyperboloid, poincare = Hyperboloid(dtype=_F64), Poincare(dtype=_F64)
    sigma_DD = jnp.asarray([[0.30, 0.05, 0.0], [0.05, 0.22, 0.03], [0.0, 0.03, 0.18]], dtype=_F64)
    v_ND = jnp.asarray(np.random.default_rng(1).multivariate_normal(np.zeros(n), np.asarray(sigma_DD), size=8), dtype=_F64)
    reference_N = scipy.stats.multivariate_normal(mean=np.zeros(n), cov=np.asarray(sigma_DD)).logpdf(np.asarray(v_ND))

    mu_0_A = hyperboloid.create_origin(c, n)
    z_NA = jax.vmap(lambda v_D: hyperboloid.expmap(hyperboloid.embed_spatial_0(v_D), mu_0_A, c))(v_ND)
    lp_N = wrapped_normal_hyperboloid.log_prob(z_NA, mu_0_A, sigma_DD, c, manifold_module=hyperboloid)
    assert np.allclose(np.asarray(lp_N), reference_N, atol=1e-3)

    # Ball: exp_0 consumes v/2 because λ(0) = 2, so v is again the Riemannian displacement.
    z_ND = jax.vmap(lambda v_D: poincare.expmap_0(v_D / 2.0, c))(v_ND)
    lp_ball_N = wrapped_normal_poincare.log_prob(z_ND, jnp.zeros(n, dtype=_F64), sigma_DD, c, manifold_module=poincare)
    assert np.allclose(np.asarray(lp_ball_N), reference_N, atol=1e-3)


def test_log_prob_agrees_across_the_stereographic_isometry() -> None:
    """(d) The two models assign the same log-density to corresponding points — no correction.

    ``hyperboloid_to_poincare`` is an isometry, so it carries the Riemannian volume measure over
    unchanged and the two densities must agree pointwise with *no* Jacobian factor. A model that
    silently returned a Lebesgue-chart density on one side would need a ``λ^n`` correction here.
    """
    c = 1.3
    hyperboloid, poincare = Hyperboloid(dtype=_F64), Poincare(dtype=_F64)
    sigma_DD = jnp.asarray([[0.30, 0.05, 0.0], [0.05, 0.22, 0.03], [0.0, 0.03, 0.18]], dtype=_F64)
    mu_A = hyperboloid.expmap_0(hyperboloid.embed_spatial_0(jnp.asarray([0.4, -0.25, 0.1], dtype=_F64)), c)

    z_NA = wrapped_normal_hyperboloid.sample(
        jax.random.PRNGKey(3), mu_A, sigma_DD, c, sample_shape=(16,), manifold_module=hyperboloid
    )
    lp_hyp_N = wrapped_normal_hyperboloid.log_prob(z_NA, mu_A, sigma_DD, c, manifold_module=hyperboloid)

    z_ball_ND = jax.vmap(iso.hyperboloid_to_poincare, in_axes=(0, None))(z_NA, c)
    mu_ball_D = iso.hyperboloid_to_poincare(mu_A, c)
    lp_ball_N = wrapped_normal_poincare.log_prob(z_ball_ND, mu_ball_D, sigma_DD, c, manifold_module=poincare)

    assert jnp.allclose(lp_hyp_N, lp_ball_N, atol=1e-10)


# =============================================================================================
# Batched-mean log_prob (audit M2-03, library fix)
# =============================================================================================


def test_log_prob_accepts_batched_mu_with_sample_shape() -> None:
    """log_prob handles ``sample``'s own output when the mean is batched (used to raise).

    ``z`` is ``(*S, *B, dim)`` and ``mu`` is ``(*B, dim)``: the sample axes broadcast the mean
    while the batch axes pair with it. Step 2 (parallel transport to the origin) used to vmap
    ``u`` and ``mu`` together on axis 0, pairing the leading *sample* axis with the *batch*
    axis — a TypeError whenever S != B, and a silently mispaired answer at S == B. The
    per-entry scalar-mean calls below are the oracle for the pairing.
    """
    c = 1.0
    hyperboloid, poincare = Hyperboloid(dtype=_F64), Poincare(dtype=_F64)
    sigma = 0.3
    key = jax.random.PRNGKey(0)

    mu_hyp_BA = jax.vmap(lambda m_A: hyperboloid.proj(m_A, c))(
        jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.2, -0.1], [1.0, -0.3, 0.15]], dtype=_F64)
    )
    mu_ball_BD = jnp.asarray([[0.0, 0.0], [0.2, -0.1], [-0.3, 0.15]], dtype=_F64)

    # S != B (the TypeError case) and S == B (the silent-mispairing case).
    for n_samples in (5, 3):
        z_SBA = wrapped_normal_hyperboloid.sample(
            key, mu_hyp_BA, sigma, c, sample_shape=(n_samples,), manifold_module=hyperboloid
        )
        lp_SB = wrapped_normal_hyperboloid.log_prob(z_SBA, mu_hyp_BA, sigma, c, manifold_module=hyperboloid)
        assert lp_SB.shape == (n_samples, 3)
        for s in range(n_samples):
            for b in range(3):
                one = wrapped_normal_hyperboloid.log_prob(z_SBA[s, b], mu_hyp_BA[b], sigma, c, manifold_module=hyperboloid)
                assert float(lp_SB[s, b]) == pytest.approx(float(one), abs=1e-12)

        z_ball_SBD = wrapped_normal_poincare.sample(
            key, mu_ball_BD, sigma, c, sample_shape=(n_samples,), manifold_module=poincare
        )
        lp_ball_SB = wrapped_normal_poincare.log_prob(z_ball_SBD, mu_ball_BD, sigma, c, manifold_module=poincare)
        assert lp_ball_SB.shape == (n_samples, 3)
        for s in range(n_samples):
            for b in range(3):
                one = wrapped_normal_poincare.log_prob(z_ball_SBD[s, b], mu_ball_BD[b], sigma, c, manifold_module=poincare)
                assert float(lp_ball_SB[s, b]) == pytest.approx(float(one), abs=1e-12)

    # Multi-axis sample_shape keeps every sample axis outside the batch axis.
    z_224A = wrapped_normal_hyperboloid.sample(key, mu_hyp_BA, sigma, c, sample_shape=(2, 4), manifold_module=hyperboloid)
    lp_224 = wrapped_normal_hyperboloid.log_prob(z_224A, mu_hyp_BA, sigma, c, manifold_module=hyperboloid)
    assert lp_224.shape == (2, 4, 3)
    assert float(lp_224[1, 2, 0]) == pytest.approx(
        float(wrapped_normal_hyperboloid.log_prob(z_224A[1, 2, 0], mu_hyp_BA[0], sigma, c, manifold_module=hyperboloid)),
        abs=1e-12,
    )


# =============================================================================================
# Gradients at the mean (audit M2-08, library fix)
# =============================================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_prob_gradient_at_the_mean_is_finite_and_zero(dtype) -> None:
    """∇_z log p(z) is finite and vanishes at z = μ, in both models and both dtypes.

    z = μ is the density's maximum, so the gradient there is exactly 0 — but the Poincaré path
    reached it through ``sqrt(Σv²)`` with v = 0, whose derivative is infinite, and returned NaN
    (in float64 as much as float32). Reaching for a KL term or a MAP objective at the mean is
    the routine way to hit this.
    """
    c = 1.0
    sigma = jnp.asarray(0.5, dtype=dtype)
    hyperboloid, poincare = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)

    spatial_D = jnp.asarray([0.3, -0.2, 0.1], dtype=dtype)
    cases = [
        ("hyperboloid origin", wrapped_normal_hyperboloid, hyperboloid, hyperboloid.create_origin(c, 3)),
        (
            "hyperboloid offset",
            wrapped_normal_hyperboloid,
            hyperboloid,
            hyperboloid.expmap_0(hyperboloid.embed_spatial_0(spatial_D), c),
        ),
        ("ball origin", wrapped_normal_poincare, poincare, jnp.zeros(3, dtype=dtype)),
        ("ball offset", wrapped_normal_poincare, poincare, jnp.asarray([0.2, -0.1, 0.05], dtype=dtype)),
    ]
    for name, module, manifold, mu in cases:
        grad = jax.grad(lambda z: module.log_prob(z, mu, sigma, c, manifold_module=manifold))(mu)  # noqa: B023
        assert bool(jnp.all(jnp.isfinite(grad))), f"{name}: non-finite gradient {grad}"
        assert float(jnp.max(jnp.abs(grad))) == pytest.approx(0.0, abs=1e-6), f"{name}: gradient {grad}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("n", [2, 3, 5])
def test_log_det_jacobian_at_zero_radius(dtype, n: int) -> None:
    """``_log_det_jacobian_from_r`` is 0 with a finite 0 gradient at r = 0, and exact elsewhere.

    A single ``jnp.where`` cannot hide the singular branch: ``log(sinh 0) - log 0`` is NaN and
    reverse-mode AD multiplies the *unselected* branch by zero, giving 0·NaN = NaN. Neither
    log_prob can reach r = 0 (both floor the radius first), so only a direct call exercises it.
    """
    c = 1.7
    value = _log_det_jacobian_from_r(jnp.asarray(0.0, dtype=dtype), c, n)
    grad = jax.grad(lambda r: _log_det_jacobian_from_r(r, c, n))(jnp.asarray(0.0, dtype=dtype))
    assert float(value) == pytest.approx(0.0, abs=1e-12)
    assert bool(jnp.isfinite(grad)) and float(grad) == pytest.approx(0.0, abs=1e-12)

    # Away from the singularity the value is the plain closed form (independent NumPy).
    for r in (0.25, 2.0):
        x = np.sqrt(c) * r
        expected = (n - 1) * np.log(np.sinh(x) / x)
        got = float(_log_det_jacobian_from_r(jnp.asarray(r, dtype=_F64), c, n))
        assert got == pytest.approx(float(expected), rel=1e-12)


# =============================================================================================
# _common building blocks (audit M2-09)
# =============================================================================================


@pytest.mark.parametrize("sigma_kind", ["scalar", "diag", "full"])
def test_gaussian_log_prob_matches_scipy(sigma_kind: str) -> None:
    """``gaussian_log_prob`` equals scipy's multivariate normal for all three sigma forms.

    Also pins the sigma → covariance convention: a scalar/diagonal sigma is a standard
    *deviation* (squared into the covariance), a matrix is the covariance itself. Swapping those
    readings leaves every shape and finiteness check in this file green.
    """
    n = 3
    sigma = _sigma_for(sigma_kind, n, _F64)
    cov_DD = np.asarray(sigma_to_cov(sigma, n, _F64))

    if sigma_kind == "scalar":
        assert np.allclose(cov_DD, 0.3**2 * np.eye(n))
    elif sigma_kind == "diag":
        assert np.allclose(cov_DD, np.diag(np.asarray([0.2, 0.3, 0.4]) ** 2))
    else:
        assert np.allclose(cov_DD, np.asarray(sigma))

    v_ND = np.random.default_rng(2).multivariate_normal(np.zeros(n), cov_DD, size=6)
    expected_N = scipy.stats.multivariate_normal(mean=np.zeros(n), cov=cov_DD).logpdf(v_ND)
    got_N = gaussian_log_prob(jnp.asarray(v_ND, dtype=_F64), sigma, n, _F64)
    assert np.allclose(np.asarray(got_N), expected_N, atol=1e-12)


def test_sigma_to_cov_rejects_mismatched_dimensions() -> None:
    """Both validation branches raise: a diagonal of the wrong length and a non-(n, n) matrix."""
    with pytest.raises(ValueError, match="Diagonal sigma must have length 3"):
        sigma_to_cov(jnp.asarray([0.1, 0.2], dtype=_F64), 3, _F64)
    with pytest.raises(ValueError, match=r"Covariance matrix must be \(3, 3\)"):
        sigma_to_cov(jnp.eye(2, dtype=_F64), 3, _F64)
