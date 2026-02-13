"""Tests for the with_precision() manifold wrapper."""

import jax
import jax.numpy as jnp

from hyperbolix.manifolds import hyperboloid, poincare, with_precision
from hyperbolix.manifolds.precision import PrecisionWrapped

# Enable float64 for these tests
jax.config.update("jax_enable_x64", True)


class TestWithPrecision:
    """Test that with_precision creates a valid proxy."""

    def test_returns_precision_wrapped(self):
        wrapped = with_precision(poincare, jnp.float64)
        assert isinstance(wrapped, PrecisionWrapped)

    def test_constants_pass_through(self):
        wrapped = with_precision(poincare, jnp.float64)
        assert wrapped.VERSION_MOBIUS_DIRECT == poincare.VERSION_MOBIUS_DIRECT
        assert wrapped.VERSION_MOBIUS == poincare.VERSION_MOBIUS
        assert wrapped.MIN_NORM == poincare.MIN_NORM

    def test_hyperboloid_constants_pass_through(self):
        wrapped = with_precision(hyperboloid, jnp.float64)
        assert wrapped.VERSION_DEFAULT == hyperboloid.VERSION_DEFAULT
        assert wrapped.VERSION_SMOOTHENED == hyperboloid.VERSION_SMOOTHENED
        assert wrapped.MIN_NORM == hyperboloid.MIN_NORM


class TestPoincarePrecision:
    """Test auto-casting for Poincaré ball operations."""

    def test_dist_float64_output(self):
        poincare_f64 = with_precision(poincare, jnp.float64)
        x = jnp.array([0.1, 0.2])  # float32
        y = jnp.array([0.3, 0.4])  # float32
        d = poincare_f64.dist(x, y, c=1.0)
        assert d.dtype == jnp.float64

    def test_proj_float64_output(self):
        poincare_f64 = with_precision(poincare, jnp.float64)
        x = jnp.array([0.9, 0.1])  # float32
        result = poincare_f64.proj(x, c=1.0)
        assert result.dtype == jnp.float64

    def test_expmap_float64_output(self):
        poincare_f64 = with_precision(poincare, jnp.float64)
        v = jnp.array([0.1, 0.2])
        x = jnp.array([0.05, 0.05])
        result = poincare_f64.expmap(v, x, c=1.0)
        assert result.dtype == jnp.float64

    def test_logmap_float64_output(self):
        poincare_f64 = with_precision(poincare, jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])
        result = poincare_f64.logmap(y, x, c=1.0)
        assert result.dtype == jnp.float64

    def test_addition_float64_output(self):
        poincare_f64 = with_precision(poincare, jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.05, 0.05])
        result = poincare_f64.addition(x, y, c=1.0)
        assert result.dtype == jnp.float64

    def test_dist_values_match(self):
        """Verify wrapped version produces same values as manual casting."""
        poincare_f64 = with_precision(poincare, jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        d_wrapped = poincare_f64.dist(x, y, c=1.0)
        d_manual = poincare.dist(x.astype(jnp.float64), y.astype(jnp.float64), c=1.0)
        assert jnp.allclose(d_wrapped, d_manual)


class TestHyperboloidPrecision:
    """Test auto-casting for hyperboloid operations."""

    def test_dist_float64_output(self):
        hyp_f64 = with_precision(hyperboloid, jnp.float64)
        x = hyperboloid.proj(jnp.array([1.0, 0.1, 0.2]), c=1.0)
        y = hyperboloid.proj(jnp.array([1.0, 0.3, 0.4]), c=1.0)
        d = hyp_f64.dist(x, y, c=1.0)
        assert d.dtype == jnp.float64

    def test_proj_float64_output(self):
        hyp_f64 = with_precision(hyperboloid, jnp.float64)
        x = jnp.array([1.0, 0.1, 0.2])
        result = hyp_f64.proj(x, c=1.0)
        assert result.dtype == jnp.float64

    def test_expmap_float64_output(self):
        hyp_f64 = with_precision(hyperboloid, jnp.float64)
        x = hyperboloid.proj(jnp.array([1.0, 0.1, 0.2]), c=1.0)
        v = jnp.array([0.0, 0.1, 0.2])
        v = hyperboloid.tangent_proj(v, x, c=1.0)
        result = hyp_f64.expmap(v, x, c=1.0)
        assert result.dtype == jnp.float64


class TestVmapJitCompat:
    """Test that wrapped modules work with jax.vmap and jax.jit."""

    def test_vmap_poincare_dist(self):
        poincare_f64 = with_precision(poincare, jnp.float64)
        x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])
        y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45]])

        dist_fn = jax.vmap(poincare_f64.dist, in_axes=(0, 0, None))
        distances = dist_fn(x_batch, y_batch, 1.0)

        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64

    def test_jit_poincare_dist(self):
        poincare_f64 = with_precision(poincare, jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        dist_jit = jax.jit(poincare_f64.dist, static_argnames=["version_idx"])
        d = dist_jit(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)

        assert d.dtype == jnp.float64

    def test_vmap_hyperboloid_dist(self):
        hyp_f64 = with_precision(hyperboloid, jnp.float64)
        x_batch = jnp.array([[1.0, 0.1, 0.2], [1.0, 0.15, 0.25]])
        y_batch = jnp.array([[1.0, 0.3, 0.4], [1.0, 0.35, 0.45]])
        x_batch = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x_batch, 1.0)
        y_batch = jax.vmap(hyperboloid.proj, in_axes=(0, None))(y_batch, 1.0)

        dist_fn = jax.vmap(hyp_f64.dist, in_axes=(0, 0, None))
        distances = dist_fn(x_batch, y_batch, 1.0)

        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64

    def test_jit_vmap_combined(self):
        poincare_f64 = with_precision(poincare, jnp.float64)
        x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])
        y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45]])

        @jax.jit
        def compute_dists(x, y):
            return jax.vmap(poincare_f64.dist, in_axes=(0, 0, None))(x, y, 1.0)

        distances = compute_dists(x_batch, y_batch)
        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64


class TestMLRFunctions:
    """Test that moved MLR functions work correctly via the manifold modules."""

    def test_poincare_conformal_factor(self):
        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        cf = poincare.conformal_factor(x, c=1.0)
        assert cf.shape == (2, 1)
        assert jnp.all(cf > 0)

    def test_poincare_compute_mlr_pp(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (4, 3)) * 0.3  # batch=4, in_dim=3
        x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)
        z = jax.random.normal(k2, (5, 3)) * 0.1  # out_dim=5
        r = jax.random.normal(k3, (5, 1)) * 0.01

        result = poincare.compute_mlr_pp(x, z, r, c=1.0, clamping_factor=1.0, smoothing_factor=50.0)
        assert result.shape == (4, 5)
        assert jnp.all(jnp.isfinite(result))

    def test_hyperboloid_compute_mlr(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (4, 4))  # batch=4, in_dim=4 (ambient)
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)
        z = jax.random.normal(k2, (5, 3)) * 0.1  # out_dim=5, in_dim-1=3
        r = jax.random.normal(k3, (5, 1)) * 0.01

        result = hyperboloid.compute_mlr(x, z, r, c=1.0, clamping_factor=1.0, smoothing_factor=50.0)
        assert result.shape == (4, 5)
        assert jnp.all(jnp.isfinite(result))

    def test_precision_wrapped_mlr(self):
        """Test that MLR functions work through precision wrapper."""
        poincare_f64 = with_precision(poincare, jnp.float64)
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (4, 3)) * 0.3
        x = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)
        z = jax.random.normal(k2, (5, 3)) * 0.1
        r = jax.random.normal(k3, (5, 1)) * 0.01

        result = poincare_f64.compute_mlr_pp(x, z, r, c=1.0, clamping_factor=1.0, smoothing_factor=50.0)
        assert result.dtype == jnp.float64


class TestDistributionManifoldParam:
    """Test that distributions accept manifold_module parameter."""

    def test_hyperboloid_sample_with_module(self):
        from hyperbolix.distributions import wrapped_normal_hyperboloid

        key = jax.random.PRNGKey(0)
        mu = jnp.array([1.0, 0.0, 0.0])
        mu = hyperboloid.proj(mu, c=1.0)

        # Default (no module)
        z1 = wrapped_normal_hyperboloid.sample(key, mu, sigma=0.1, c=1.0)
        # Explicit module
        z2 = wrapped_normal_hyperboloid.sample(key, mu, sigma=0.1, c=1.0, manifold_module=hyperboloid)

        assert jnp.allclose(z1, z2)

    def test_hyperboloid_sample_with_wrapped_module(self):
        from hyperbolix.distributions import wrapped_normal_hyperboloid

        hyp_f64 = with_precision(hyperboloid, jnp.float64)
        key = jax.random.PRNGKey(0)
        mu = jnp.array([1.0, 0.0, 0.0])
        mu = hyperboloid.proj(mu, c=1.0)

        z = wrapped_normal_hyperboloid.sample(key, mu, sigma=0.1, c=1.0, manifold_module=hyp_f64)
        assert z.dtype == jnp.float64

    def test_poincare_sample_with_module(self):
        from hyperbolix.distributions import wrapped_normal_poincare

        key = jax.random.PRNGKey(0)
        mu = jnp.array([0.0, 0.0])

        z1 = wrapped_normal_poincare.sample(key, mu, sigma=0.1, c=1.0)
        z2 = wrapped_normal_poincare.sample(key, mu, sigma=0.1, c=1.0, manifold_module=poincare)

        assert jnp.allclose(z1, z2)

    def test_poincare_sample_with_wrapped_module(self):
        from hyperbolix.distributions import wrapped_normal_poincare

        poincare_f64 = with_precision(poincare, jnp.float64)
        key = jax.random.PRNGKey(0)
        mu = jnp.array([0.0, 0.0])

        z = wrapped_normal_poincare.sample(key, mu, sigma=0.1, c=1.0, manifold_module=poincare_f64)
        assert z.dtype == jnp.float64

    def test_hyperboloid_log_prob_with_module(self):
        from hyperbolix.distributions import wrapped_normal_hyperboloid

        key = jax.random.PRNGKey(0)
        mu = hyperboloid.proj(jnp.array([1.0, 0.0, 0.0]), c=1.0)
        z = wrapped_normal_hyperboloid.sample(key, mu, sigma=0.1, c=1.0)

        lp1 = wrapped_normal_hyperboloid.log_prob(z, mu, sigma=0.1, c=1.0)
        lp2 = wrapped_normal_hyperboloid.log_prob(z, mu, sigma=0.1, c=1.0, manifold_module=hyperboloid)

        assert jnp.allclose(lp1, lp2)

    def test_poincare_log_prob_with_module(self):
        from hyperbolix.distributions import wrapped_normal_poincare

        key = jax.random.PRNGKey(0)
        mu = jnp.array([0.0, 0.0])
        z = wrapped_normal_poincare.sample(key, mu, sigma=0.1, c=1.0)

        lp1 = wrapped_normal_poincare.log_prob(z, mu, sigma=0.1, c=1.0)
        lp2 = wrapped_normal_poincare.log_prob(z, mu, sigma=0.1, c=1.0, manifold_module=poincare)

        assert jnp.allclose(lp1, lp2)
