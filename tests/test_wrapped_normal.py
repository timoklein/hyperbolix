"""Tests for wrapped normal distribution on hyperboloid."""

import jax
import jax.numpy as jnp

from hyperbolix.distributions import wrapped_normal_hyperboloid, wrapped_normal_poincare
from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.manifolds.hyperboloid import VERSION_DEFAULT

# Enable float64 support for numerical precision in tests
jax.config.update("jax_enable_x64", True)


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


def test_sample_jit_compatibility(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test that sampling works with JAX jit."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    c = 1.0
    hyperboloid = Hyperboloid(dtype=dtype)
    mu = _create_hyperboloid_origin(hyperboloid, c, dim=2, dtype=dtype)
    sigma = 0.1

    # JIT compile the sampling function
    sample_jit = jax.jit(wrapped_normal_hyperboloid.sample, static_argnames=["sample_shape"])

    z = sample_jit(key, mu, sigma, c, sample_shape=())

    assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"
    assert hyperboloid.is_in_manifold(z, c, atol=atol), "JIT-compiled sample not on hyperboloid"


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


def test_sample_poincare_jit_compatibility(dtype: jnp.dtype, tolerance: tuple[float, float]) -> None:
    """Test that Poincaré sampling works with JAX jit."""
    key = jax.random.PRNGKey(42)
    atol, _rtol = tolerance

    c = 1.0
    poincare = Poincare(dtype=dtype)
    mu = jnp.zeros(2, dtype=dtype)
    sigma = 0.1

    # JIT compile the sampling function
    sample_jit = jax.jit(wrapped_normal_poincare.sample, static_argnames=["sample_shape"])

    z = sample_jit(key, mu, sigma, c, sample_shape=())

    assert z.shape == (2,), f"Expected shape (2,), got {z.shape}"
    assert poincare.is_in_manifold(z, c, atol=atol), "JIT-compiled sample not in Poincaré ball"


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
