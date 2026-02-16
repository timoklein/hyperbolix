"""Tests for class-based manifold API dtype precision control."""

import jax
import jax.numpy as jnp

from hyperbolix.manifolds import hyperboloid as hyperboloid_module
from hyperbolix.manifolds import poincare as poincare_module
from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare

# Enable float64 for these tests
jax.config.update("jax_enable_x64", True)


class TestPoincareClassConstants:
    """Test that module constants are still accessible."""

    def test_module_constants(self):
        """Verify constants are available from module."""
        assert hasattr(poincare_module, "VERSION_MOBIUS_DIRECT")
        assert hasattr(poincare_module, "VERSION_MOBIUS")
        assert hasattr(poincare_module, "MIN_NORM")

    def test_hyperboloid_module_constants(self):
        """Verify hyperboloid constants are available from module."""
        assert hasattr(hyperboloid_module, "VERSION_DEFAULT")
        assert hasattr(hyperboloid_module, "VERSION_SMOOTHENED")
        assert hasattr(hyperboloid_module, "MIN_NORM")


class TestPoincarePrecision:
    """Test auto-casting for Poincaré ball operations."""

    def test_dist_float64_output(self):
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.1, 0.2])  # float32
        y = jnp.array([0.3, 0.4])  # float32
        d = manifold.dist(x, y, c=1.0)
        assert d.dtype == jnp.float64

    def test_proj_float64_output(self):
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.9, 0.1])  # float32
        result = manifold.proj(x, c=1.0)
        assert result.dtype == jnp.float64

    def test_expmap_float64_output(self):
        manifold = Poincare(dtype=jnp.float64)
        v = jnp.array([0.1, 0.2])
        x = jnp.array([0.05, 0.05])
        result = manifold.expmap(v, x, c=1.0)
        assert result.dtype == jnp.float64

    def test_logmap_float64_output(self):
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])
        result = manifold.logmap(y, x, c=1.0)
        assert result.dtype == jnp.float64

    def test_addition_float64_output(self):
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.05, 0.05])
        result = manifold.addition(x, y, c=1.0)
        assert result.dtype == jnp.float64

    def test_dist_values_match(self):
        """Verify class version produces same values as manual casting."""
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        d_class = manifold.dist(x, y, c=1.0)
        d_manual = poincare_module._dist(x.astype(jnp.float64), y.astype(jnp.float64), c=1.0)
        assert jnp.allclose(d_class, d_manual)


class TestHyperboloidPrecision:
    """Test auto-casting for hyperboloid operations."""

    def test_dist_float64_output(self):
        manifold = Hyperboloid(dtype=jnp.float64)
        x = manifold.proj(jnp.array([1.0, 0.1, 0.2]), c=1.0)
        y = manifold.proj(jnp.array([1.0, 0.3, 0.4]), c=1.0)
        d = manifold.dist(x, y, c=1.0)
        assert d.dtype == jnp.float64

    def test_proj_float64_output(self):
        manifold = Hyperboloid(dtype=jnp.float64)
        x = jnp.array([1.0, 0.1, 0.2])
        result = manifold.proj(x, c=1.0)
        assert result.dtype == jnp.float64

    def test_expmap_float64_output(self):
        manifold = Hyperboloid(dtype=jnp.float64)
        x = manifold.proj(jnp.array([1.0, 0.1, 0.2]), c=1.0)
        v = jnp.array([0.0, 0.1, 0.2])
        v = manifold.tangent_proj(v, x, c=1.0)
        result = manifold.expmap(v, x, c=1.0)
        assert result.dtype == jnp.float64


class TestVmapJitCompat:
    """Test that class-based manifolds work with jax.vmap and jax.jit."""

    def test_vmap_poincare_dist(self):
        manifold = Poincare(dtype=jnp.float64)
        x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])
        y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45]])

        dist_fn = jax.vmap(manifold.dist, in_axes=(0, 0, None))
        distances = dist_fn(x_batch, y_batch, 1.0)

        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64

    def test_jit_poincare_dist(self):
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        dist_jit = jax.jit(manifold.dist, static_argnames=["version_idx"])
        d = dist_jit(x, y, c=1.0, version_idx=poincare_module.VERSION_MOBIUS_DIRECT)

        assert d.dtype == jnp.float64

    def test_vmap_hyperboloid_dist(self):
        manifold = Hyperboloid(dtype=jnp.float64)
        x_batch = jnp.array([[1.0, 0.1, 0.2], [1.0, 0.15, 0.25]])
        y_batch = jnp.array([[1.0, 0.3, 0.4], [1.0, 0.35, 0.45]])
        x_batch = jax.vmap(manifold.proj, in_axes=(0, None))(x_batch, 1.0)
        y_batch = jax.vmap(manifold.proj, in_axes=(0, None))(y_batch, 1.0)

        dist_fn = jax.vmap(manifold.dist, in_axes=(0, 0, None))
        distances = dist_fn(x_batch, y_batch, 1.0)

        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64

    def test_jit_vmap_combined(self):
        manifold = Poincare(dtype=jnp.float64)
        x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])
        y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45]])

        @jax.jit
        def compute_dists(x, y):
            return jax.vmap(manifold.dist, in_axes=(0, 0, None))(x, y, 1.0)

        distances = compute_dists(x_batch, y_batch)
        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64


class TestMLRFunctions:
    """Test that MLR functions work correctly via class methods."""

    def test_poincare_conformal_factor(self):
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        cf = manifold.conformal_factor(x, c=1.0)
        assert cf.shape == (2, 1)
        assert jnp.all(cf > 0)
        assert cf.dtype == jnp.float64

    def test_poincare_compute_mlr_pp(self):
        manifold = Poincare(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (4, 3)) * 0.3  # batch=4, in_dim=3
        x = jax.vmap(manifold.proj, in_axes=(0, None))(x, 1.0)
        z = jax.random.normal(k2, (5, 3)) * 0.1  # out_dim=5
        r = jax.random.normal(k3, (5, 1)) * 0.01

        result = manifold.compute_mlr_pp(x, z, r, c=1.0, clamping_factor=1.0, smoothing_factor=50.0)
        assert result.shape == (4, 5)
        assert jnp.all(jnp.isfinite(result))
        assert result.dtype == jnp.float64

    def test_hyperboloid_compute_mlr(self):
        manifold = Hyperboloid(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (4, 4))  # batch=4, in_dim=4 (ambient)
        x = jax.vmap(manifold.proj, in_axes=(0, None))(x, 1.0)
        z = jax.random.normal(k2, (5, 3)) * 0.1  # out_dim=5, in_dim-1=3
        r = jax.random.normal(k3, (5, 1)) * 0.01

        result = manifold.compute_mlr(x, z, r, c=1.0, clamping_factor=1.0, smoothing_factor=50.0)
        assert result.shape == (4, 5)
        assert jnp.all(jnp.isfinite(result))
        assert result.dtype == jnp.float64


class TestDistributionManifoldParam:
    """Test that distributions accept manifold_module parameter."""

    def test_hyperboloid_sample_with_module(self):
        from hyperbolix.distributions import wrapped_normal_hyperboloid

        manifold = Hyperboloid(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        mu = jnp.array([1.0, 0.0, 0.0])
        mu = manifold.proj(mu, c=1.0)

        # Test with class instance
        z = wrapped_normal_hyperboloid.sample(key, mu, sigma=0.1, c=1.0, manifold_module=manifold)
        assert z.dtype == jnp.float64

    def test_poincare_sample_with_module(self):
        from hyperbolix.distributions import wrapped_normal_poincare

        manifold = Poincare(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        mu = jnp.array([0.0, 0.0])

        z = wrapped_normal_poincare.sample(key, mu, sigma=0.1, c=1.0, manifold_module=manifold)
        assert z.dtype == jnp.float64

    def test_hyperboloid_log_prob_with_module(self):
        from hyperbolix.distributions import wrapped_normal_hyperboloid

        manifold = Hyperboloid(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        mu = manifold.proj(jnp.array([1.0, 0.0, 0.0]), c=1.0)
        z = wrapped_normal_hyperboloid.sample(key, mu, sigma=0.1, c=1.0, manifold_module=manifold)

        lp = wrapped_normal_hyperboloid.log_prob(z, mu, sigma=0.1, c=1.0, manifold_module=manifold)
        assert lp.dtype == jnp.float64

    def test_poincare_log_prob_with_module(self):
        from hyperbolix.distributions import wrapped_normal_poincare

        manifold = Poincare(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        mu = jnp.array([0.0, 0.0])
        z = wrapped_normal_poincare.sample(key, mu, sigma=0.1, c=1.0, manifold_module=manifold)

        lp = wrapped_normal_poincare.log_prob(z, mu, sigma=0.1, c=1.0, manifold_module=manifold)
        assert lp.dtype == jnp.float64
