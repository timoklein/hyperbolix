"""Tests for wrapped normal distribution on hyperboloid."""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix_jax.distributions import wrapped_normal
from hyperbolix_jax.manifolds import hyperboloid

# Enable float64 support for numerical precision in tests
jax.config.update("jax_enable_x64", True)


class TestWrappedNormalSampling:
    """Test the wrapped normal sampling functionality."""

    def test_basic_sampling_single_point(self):
        """Test basic sampling with a single mean point and isotropic covariance."""
        key = jax.random.PRNGKey(42)

        # Create mean at origin in H^2 (3D ambient space)
        c = 1.0
        mu = hyperboloid._create_origin(c, dim=2, dtype=jnp.float32)

        # Sample with isotropic covariance
        sigma = 0.1
        z = wrapped_normal.sample(key, mu, sigma, c)

        # Check output shape
        assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"

        # Check point is on manifold
        assert hyperboloid.is_in_manifold(z, c, atol=1e-5), "Sampled point not on hyperboloid"

    def test_sample_shape_parameter(self):
        """Test that sample_shape parameter works correctly."""
        key = jax.random.PRNGKey(42)

        # Create mean
        c = 1.0
        mu = hyperboloid._create_origin(c, dim=2, dtype=jnp.float32)
        sigma = 0.1

        # Sample with sample_shape
        z = wrapped_normal.sample(key, mu, sigma, c, sample_shape=(5, 3))

        # Check output shape: sample_shape + mu.shape
        assert z.shape == (5, 3, 3), f"Expected shape (5, 3, 3), got {z.shape}"

        # Check all points are on manifold
        for i in range(5):
            for j in range(3):
                assert hyperboloid.is_in_manifold(z[i, j], c, atol=1e-5), f"Point [{i}, {j}] not on hyperboloid"

    def test_diagonal_covariance(self):
        """Test sampling with diagonal covariance."""
        key = jax.random.PRNGKey(42)

        # Create mean in H^2
        c = 1.0
        mu = hyperboloid._create_origin(c, dim=2, dtype=jnp.float32)

        # Diagonal covariance for spatial dimension (n=2)
        sigma_diag = jnp.array([0.1, 0.2])

        z = wrapped_normal.sample(key, mu, sigma_diag, c)

        # Check output shape
        assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"

        # Check point is on manifold
        assert hyperboloid.is_in_manifold(z, c, atol=1e-5), "Sampled point not on hyperboloid"

    def test_full_covariance(self):
        """Test sampling with full covariance matrix."""
        key = jax.random.PRNGKey(42)

        # Create mean in H^2
        c = 1.0
        mu = hyperboloid._create_origin(c, dim=2, dtype=jnp.float32)

        # Full covariance for spatial dimension (n=2)
        sigma_full = jnp.array([[0.1, 0.01], [0.01, 0.1]])

        z = wrapped_normal.sample(key, mu, sigma_full, c)

        # Check output shape
        assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"

        # Check point is on manifold
        assert hyperboloid.is_in_manifold(z, c, atol=1e-5), "Sampled point not on hyperboloid"

    def test_dtype_propagation(self):
        """Test that dtype parameter is respected."""
        key = jax.random.PRNGKey(42)

        # Create mean in H^2 with float32
        c = 1.0
        mu = hyperboloid._create_origin(c, dim=2, dtype=jnp.float32)
        sigma = 0.1

        # Sample with explicit float64 dtype
        z_f64 = wrapped_normal.sample(key, mu, sigma, c, dtype=jnp.float64)

        assert z_f64.dtype == jnp.float64, f"Expected dtype float64, got {z_f64.dtype}"
        assert hyperboloid.is_in_manifold(z_f64, c, atol=1e-10), "Float64 sample not on hyperboloid"

    def test_batched_means(self):
        """Test sampling with batched mean points."""
        key = jax.random.PRNGKey(42)

        # Create batch of means in H^2
        c = 1.0
        mu_batch = jnp.array([
            [1.0, 0.0, 0.0],  # origin
            [1.0, 0.1, 0.1],  # another point (will be projected)
        ])
        # Project to ensure they're on manifold
        mu_batch = jax.vmap(lambda m: hyperboloid.proj(m, c))(mu_batch)

        sigma = 0.1

        z = wrapped_normal.sample(key, mu_batch, sigma, c)

        # Check output shape: should match mu_batch.shape
        assert z.shape == (2, 3), f"Expected shape (2, 3), got {z.shape}"

        # Check all points are on manifold
        for i in range(2):
            assert hyperboloid.is_in_manifold(z[i], c, atol=1e-5), f"Point {i} not on hyperboloid"

    def test_jit_compatibility(self):
        """Test that sampling works with JAX jit."""
        key = jax.random.PRNGKey(42)

        c = 1.0
        mu = hyperboloid._create_origin(c, dim=2, dtype=jnp.float32)
        sigma = 0.1

        # JIT compile the sampling function
        sample_jit = jax.jit(wrapped_normal.sample, static_argnames=["sample_shape"])

        z = sample_jit(key, mu, sigma, c, sample_shape=())

        assert z.shape == (3,), f"Expected shape (3,), got {z.shape}"
        assert hyperboloid.is_in_manifold(z, c, atol=1e-5), "JIT-compiled sample not on hyperboloid"

    def test_vmap_compatibility(self):
        """Test that sampling works with JAX vmap."""
        # Create multiple keys
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 10)

        c = 1.0
        mu = hyperboloid._create_origin(c, dim=2, dtype=jnp.float32)
        sigma = 0.1

        # vmap over keys
        sample_vmap = jax.vmap(lambda k: wrapped_normal.sample(k, mu, sigma, c))
        z_batch = sample_vmap(keys)

        assert z_batch.shape == (10, 3), f"Expected shape (10, 3), got {z_batch.shape}"

        # Check all points are on manifold
        for i in range(10):
            assert hyperboloid.is_in_manifold(z_batch[i], c, atol=1e-5), f"Point {i} not on hyperboloid"


class TestGaussianHelper:
    """Test the _sample_gaussian helper function."""

    def test_isotropic_covariance(self):
        """Test Gaussian sampling with isotropic covariance."""
        key = jax.random.PRNGKey(42)
        sigma = 0.1
        n = 2

        samples = wrapped_normal._sample_gaussian(key, sigma, n, sample_shape=(100,))

        assert samples.shape == (100, 2), f"Expected shape (100, 2), got {samples.shape}"

        # Check approximate statistics (with tolerance for randomness)
        mean = jnp.mean(samples, axis=0)
        assert jnp.allclose(mean, 0.0, atol=0.1), f"Mean should be near 0, got {mean}"

    def test_diagonal_covariance(self):
        """Test Gaussian sampling with diagonal covariance."""
        key = jax.random.PRNGKey(42)
        sigma = jnp.array([0.1, 0.2])
        n = 2

        samples = wrapped_normal._sample_gaussian(key, sigma, n, sample_shape=(100,))

        assert samples.shape == (100, 2), f"Expected shape (100, 2), got {samples.shape}"

    def test_full_covariance(self):
        """Test Gaussian sampling with full covariance."""
        key = jax.random.PRNGKey(42)
        sigma = jnp.array([[0.1, 0.01], [0.01, 0.1]])
        n = 2

        samples = wrapped_normal._sample_gaussian(key, sigma, n, sample_shape=(100,))

        assert samples.shape == (100, 2), f"Expected shape (100, 2), got {samples.shape}"


class TestEmbedHelper:
    """Test the _embed_in_tangent_space helper function."""

    def test_single_vector(self):
        """Test embedding a single spatial vector."""
        v_spatial = jnp.array([0.1, 0.2])
        v_tangent = wrapped_normal._embed_in_tangent_space(v_spatial)

        assert v_tangent.shape == (3,), f"Expected shape (3,), got {v_tangent.shape}"
        assert v_tangent[0] == 0.0, f"First component should be 0, got {v_tangent[0]}"
        assert jnp.allclose(v_tangent[1:], v_spatial), "Spatial components should match"

    def test_batched_vectors(self):
        """Test embedding batched spatial vectors."""
        v_spatial = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        v_tangent = wrapped_normal._embed_in_tangent_space(v_spatial)

        assert v_tangent.shape == (2, 3), f"Expected shape (2, 3), got {v_tangent.shape}"
        assert jnp.all(v_tangent[:, 0] == 0.0), "First component should be 0"
        assert jnp.allclose(v_tangent[:, 1:], v_spatial), "Spatial components should match"
