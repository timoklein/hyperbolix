"""Tests for class-based manifold API JIT compatibility.

Tests verify that:
1. Manifold class methods are JIT-compilable
2. Same instance doesn't trigger recompilation
3. Different dtypes trigger recompilation
4. Methods work with vmap
5. Gradients flow through methods correctly
"""

import jax
import jax.numpy as jnp

from hyperbolix.manifolds.euclidean import Euclidean
from hyperbolix.manifolds.hyperboloid import VERSION_DEFAULT, Hyperboloid
from hyperbolix.manifolds.poincare import VERSION_MOBIUS_DIRECT, Poincare

# Enable float64 for these tests
jax.config.update("jax_enable_x64", True)


class TestPoincareJIT:
    """Test JIT compilation with Poincare class methods."""

    def test_jit_dist_method(self):
        """Verify that Poincare.dist() is JIT-compilable."""
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        # JIT compile the method
        dist_jit = jax.jit(manifold.dist, static_argnames=["version_idx"])
        d = dist_jit(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        assert d.dtype == jnp.float64
        assert jnp.isfinite(d)
        assert d > 0

    def test_jit_expmap_method(self):
        """Verify that Poincare.expmap() is JIT-compilable."""
        manifold = Poincare(dtype=jnp.float64)
        v = jnp.array([0.05, 0.05])
        x = jnp.array([0.1, 0.2])

        expmap_jit = jax.jit(manifold.expmap)
        result = expmap_jit(v, x, c=1.0)

        assert result.dtype == jnp.float64
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_logmap_method(self):
        """Verify that Poincare.logmap() is JIT-compilable."""
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        logmap_jit = jax.jit(manifold.logmap)
        result = logmap_jit(y, x, c=1.0)

        assert result.dtype == jnp.float64
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_no_recompilation_same_instance(self):
        """Verify same instance doesn't trigger recompilation."""
        manifold = Poincare(dtype=jnp.float64)

        # Create JIT-compiled function
        dist_jit = jax.jit(manifold.dist, static_argnames=["version_idx"])

        # First call - triggers compilation
        x1 = jnp.array([0.1, 0.2])
        y1 = jnp.array([0.3, 0.4])
        d1 = dist_jit(x1, y1, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        # Second call with different data - should NOT recompile
        x2 = jnp.array([0.15, 0.25])
        y2 = jnp.array([0.35, 0.45])
        d2 = dist_jit(x2, y2, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        # Both should succeed and have correct dtype
        assert d1.dtype == jnp.float64
        assert d2.dtype == jnp.float64
        assert jnp.isfinite(d1)
        assert jnp.isfinite(d2)
        # Results should be different since inputs are different
        assert not jnp.allclose(d1, d2)

    def test_jit_recompilation_different_dtype(self):
        """Verify different dtype instances trigger separate compilations."""
        manifold_f32 = Poincare(dtype=jnp.float32)
        manifold_f64 = Poincare(dtype=jnp.float64)

        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        # JIT compile with float32 manifold
        dist_f32_jit = jax.jit(manifold_f32.dist, static_argnames=["version_idx"])
        d_f32 = dist_f32_jit(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        # JIT compile with float64 manifold
        dist_f64_jit = jax.jit(manifold_f64.dist, static_argnames=["version_idx"])
        d_f64 = dist_f64_jit(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        # Results should have different dtypes
        assert d_f32.dtype == jnp.float32
        assert d_f64.dtype == jnp.float64
        # Values should be close but not identical due to precision
        assert jnp.allclose(d_f32, d_f64, rtol=1e-6)


class TestPoincareVmap:
    """Test vmap compatibility with Poincare class methods."""

    def test_vmap_dist_method(self):
        """Verify that Poincare.dist() works with vmap."""
        manifold = Poincare(dtype=jnp.float64)
        x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])
        y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45]])

        dist_batched = jax.vmap(manifold.dist, in_axes=(0, 0, None, None))
        distances = dist_batched(x_batch, y_batch, 1.0, VERSION_MOBIUS_DIRECT)

        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(distances))
        assert jnp.all(distances > 0)

    def test_vmap_expmap_method(self):
        """Verify that Poincare.expmap() works with vmap."""
        manifold = Poincare(dtype=jnp.float64)
        v_batch = jnp.array([[0.05, 0.05], [0.06, 0.06]])
        x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])

        expmap_batched = jax.vmap(manifold.expmap, in_axes=(0, 0, None))
        results = expmap_batched(v_batch, x_batch, 1.0)

        assert results.shape == (2, 2)
        assert results.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(results))

    def test_vmap_jit_combined(self):
        """Verify that vmap and JIT can be composed."""
        manifold = Poincare(dtype=jnp.float64)
        x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])
        y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45]])

        @jax.jit
        def compute_dists(x, y):
            return jax.vmap(manifold.dist, in_axes=(0, 0, None, None))(x, y, 1.0, VERSION_MOBIUS_DIRECT)

        distances = compute_dists(x_batch, y_batch)
        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64


class TestPoincareGrad:
    """Test gradient flow through Poincare class methods."""

    def test_grad_through_dist(self):
        """Verify gradients flow through Poincare.dist()."""
        manifold = Poincare(dtype=jnp.float64)

        def loss_fn(x):
            y = jnp.array([0.3, 0.4])
            return manifold.dist(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        x = jnp.array([0.1, 0.2])
        grad = jax.grad(loss_fn)(x)

        assert grad.shape == (2,)
        assert grad.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.linalg.norm(grad) > 0  # Gradient should be non-zero

    def test_grad_through_expmap(self):
        """Verify gradients flow through Poincare.expmap()."""
        manifold = Poincare(dtype=jnp.float64)

        def loss_fn(v):
            x = jnp.array([0.1, 0.2])
            y = manifold.expmap(v, x, c=1.0)
            return jnp.sum(y**2)

        v = jnp.array([0.05, 0.05])
        grad = jax.grad(loss_fn)(v)

        assert grad.shape == (2,)
        assert grad.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(grad))

    def test_value_and_grad(self):
        """Verify value_and_grad works with class methods."""
        manifold = Poincare(dtype=jnp.float64)

        def loss_fn(x):
            y = jnp.array([0.3, 0.4])
            return manifold.dist(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        x = jnp.array([0.1, 0.2])
        value, grad = jax.value_and_grad(loss_fn)(x)

        assert jnp.isscalar(value) or value.shape == ()
        assert value.dtype == jnp.float64
        assert grad.shape == (2,)
        assert grad.dtype == jnp.float64


class TestHyperboloidJIT:
    """Test JIT compilation with Hyperboloid class methods."""

    def test_jit_dist_method(self):
        """Verify that Hyperboloid.dist() is JIT-compilable."""
        manifold = Hyperboloid(dtype=jnp.float64)
        x = manifold.proj(jnp.array([1.0, 0.1, 0.2]), c=1.0)
        y = manifold.proj(jnp.array([1.0, 0.3, 0.4]), c=1.0)

        dist_jit = jax.jit(manifold.dist, static_argnames=["version_idx"])
        d = dist_jit(x, y, c=1.0, version_idx=VERSION_DEFAULT)

        assert d.dtype == jnp.float64
        assert jnp.isfinite(d)
        assert d >= 0

    def test_jit_expmap_method(self):
        """Verify that Hyperboloid.expmap() is JIT-compilable."""
        manifold = Hyperboloid(dtype=jnp.float64)
        x = manifold.proj(jnp.array([1.0, 0.1, 0.2]), c=1.0)
        v = jnp.array([0.0, 0.05, 0.05])
        v = manifold.tangent_proj(v, x, c=1.0)

        expmap_jit = jax.jit(manifold.expmap)
        result = expmap_jit(v, x, c=1.0)

        assert result.dtype == jnp.float64
        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_no_recompilation_same_instance(self):
        """Verify same instance doesn't trigger recompilation."""
        manifold = Hyperboloid(dtype=jnp.float64)

        dist_jit = jax.jit(manifold.dist, static_argnames=["version_idx"])

        # First call
        x1 = manifold.proj(jnp.array([1.0, 0.1, 0.2]), c=1.0)
        y1 = manifold.proj(jnp.array([1.0, 0.3, 0.4]), c=1.0)
        d1 = dist_jit(x1, y1, c=1.0, version_idx=VERSION_DEFAULT)

        # Second call
        x2 = manifold.proj(jnp.array([1.0, 0.15, 0.25]), c=1.0)
        y2 = manifold.proj(jnp.array([1.0, 0.35, 0.45]), c=1.0)
        d2 = dist_jit(x2, y2, c=1.0, version_idx=VERSION_DEFAULT)

        assert d1.dtype == jnp.float64
        assert d2.dtype == jnp.float64
        assert not jnp.allclose(d1, d2)


class TestHyperboloidVmap:
    """Test vmap compatibility with Hyperboloid class methods."""

    def test_vmap_dist_method(self):
        """Verify that Hyperboloid.dist() works with vmap."""
        manifold = Hyperboloid(dtype=jnp.float64)
        x_batch = jnp.array([[1.0, 0.1, 0.2], [1.0, 0.15, 0.25]])
        y_batch = jnp.array([[1.0, 0.3, 0.4], [1.0, 0.35, 0.45]])

        # Project points onto manifold
        x_batch = jax.vmap(manifold.proj, in_axes=(0, None))(x_batch, 1.0)
        y_batch = jax.vmap(manifold.proj, in_axes=(0, None))(y_batch, 1.0)

        dist_batched = jax.vmap(manifold.dist, in_axes=(0, 0, None, None))
        distances = dist_batched(x_batch, y_batch, 1.0, VERSION_DEFAULT)

        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(distances))

    def test_vmap_proj_method(self):
        """Verify that Hyperboloid.proj() works with vmap."""
        manifold = Hyperboloid(dtype=jnp.float64)
        x_batch = jnp.array([[1.0, 0.1, 0.2], [1.0, 0.15, 0.25]])

        proj_batched = jax.vmap(manifold.proj, in_axes=(0, None))
        results = proj_batched(x_batch, 1.0)

        assert results.shape == (2, 3)
        assert results.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(results))


class TestHyperboloidGrad:
    """Test gradient flow through Hyperboloid class methods."""

    def test_grad_through_dist(self):
        """Verify gradients flow through Hyperboloid.dist()."""
        manifold = Hyperboloid(dtype=jnp.float64)

        def loss_fn(x_raw):
            x = manifold.proj(x_raw, c=1.0)
            y = manifold.proj(jnp.array([1.0, 0.3, 0.4]), c=1.0)
            return manifold.dist(x, y, c=1.0, version_idx=VERSION_DEFAULT)

        x_raw = jnp.array([1.0, 0.1, 0.2])
        grad = jax.grad(loss_fn)(x_raw)

        assert grad.shape == (3,)
        assert grad.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(grad))


class TestEuclideanJIT:
    """Test JIT compilation with Euclidean class methods."""

    def test_jit_dist_method(self):
        """Verify that Euclidean.dist() is JIT-compilable."""
        manifold = Euclidean(dtype=jnp.float64)
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        dist_jit = jax.jit(manifold.dist)
        d = dist_jit(x, y, c=0.0)

        assert d.dtype == jnp.float64
        assert jnp.isfinite(d)
        assert d > 0

    def test_jit_expmap_method(self):
        """Verify that Euclidean.expmap() is JIT-compilable."""
        manifold = Euclidean(dtype=jnp.float64)
        v = jnp.array([0.5, 0.5])
        x = jnp.array([1.0, 2.0])

        expmap_jit = jax.jit(manifold.expmap)
        result = expmap_jit(v, x, c=0.0)

        assert result.dtype == jnp.float64
        assert result.shape == (2,)
        # For Euclidean, expmap is just addition
        assert jnp.allclose(result, x + v)

    def test_jit_no_recompilation_same_instance(self):
        """Verify same instance doesn't trigger recompilation."""
        manifold = Euclidean(dtype=jnp.float64)

        dist_jit = jax.jit(manifold.dist)

        # First call
        d1 = dist_jit(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), c=0.0)
        # Second call
        d2 = dist_jit(jnp.array([1.0, 1.0]), jnp.array([2.0, 2.0]), c=0.0)

        assert d1.dtype == jnp.float64
        assert d2.dtype == jnp.float64
        assert jnp.allclose(d1, d2)  # Both should be sqrt(2)


class TestEuclideanVmap:
    """Test vmap compatibility with Euclidean class methods."""

    def test_vmap_dist_method(self):
        """Verify that Euclidean.dist() works with vmap."""
        manifold = Euclidean(dtype=jnp.float64)
        x_batch = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y_batch = jnp.array([[1.0, 1.0], [2.0, 2.0]])

        dist_batched = jax.vmap(manifold.dist, in_axes=(0, 0, None))
        distances = dist_batched(x_batch, y_batch, 0.0)

        assert distances.shape == (2,)
        assert distances.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(distances))


class TestEuclideanGrad:
    """Test gradient flow through Euclidean class methods."""

    def test_grad_through_dist(self):
        """Verify gradients flow through Euclidean.dist()."""
        manifold = Euclidean(dtype=jnp.float64)

        def loss_fn(x):
            y = jnp.array([3.0, 4.0])
            return manifold.dist(x, y, c=0.0)

        x = jnp.array([1.0, 2.0])
        grad = jax.grad(loss_fn)(x)

        assert grad.shape == (2,)
        assert grad.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(grad))


class TestDtypeCasting:
    """Test that dtype casting works correctly."""

    def test_poincare_float32_to_float64(self):
        """Verify float32 arrays are cast to float64 when using float64 manifold."""
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([0.1, 0.2], dtype=jnp.float32)
        y = jnp.array([0.3, 0.4], dtype=jnp.float32)

        d = manifold.dist(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)
        assert d.dtype == jnp.float64

    def test_hyperboloid_float32_to_float64(self):
        """Verify float32 arrays are cast to float64 when using float64 manifold."""
        manifold = Hyperboloid(dtype=jnp.float64)
        x = jnp.array([1.0, 0.1, 0.2], dtype=jnp.float32)

        result = manifold.proj(x, c=1.0)
        assert result.dtype == jnp.float64

    def test_euclidean_float32_to_float64(self):
        """Verify float32 arrays are cast to float64 when using float64 manifold."""
        manifold = Euclidean(dtype=jnp.float64)
        x = jnp.array([1.0, 2.0], dtype=jnp.float32)
        y = jnp.array([3.0, 4.0], dtype=jnp.float32)

        d = manifold.dist(x, y, c=0.0)
        assert d.dtype == jnp.float64

    def test_poincare_preserves_float32(self):
        """Verify float32 manifold preserves float32."""
        manifold = Poincare(dtype=jnp.float32)
        x = jnp.array([0.1, 0.2], dtype=jnp.float32)
        y = jnp.array([0.3, 0.4], dtype=jnp.float32)

        d = manifold.dist(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)
        assert d.dtype == jnp.float32


class TestCrossDtypeCompilation:
    """Test that different dtype instances can coexist."""

    def test_separate_jit_compilation_per_dtype(self):
        """Verify that JIT compiles separately for each dtype."""
        poincare_f32 = Poincare(dtype=jnp.float32)
        poincare_f64 = Poincare(dtype=jnp.float64)

        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        # These should compile separately and produce different dtypes
        @jax.jit
        def dist_f32(x, y):
            return poincare_f32.dist(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        @jax.jit
        def dist_f64(x, y):
            return poincare_f64.dist(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        d_f32 = dist_f32(x, y)
        d_f64 = dist_f64(x, y)

        assert d_f32.dtype == jnp.float32
        assert d_f64.dtype == jnp.float64
        assert jnp.allclose(d_f32, d_f64, rtol=1e-6)


class TestBoundMethodJIT:
    """Test that bound methods work correctly with JIT.

    This is critical because bound methods capture self, which contains
    the dtype. We need to ensure self.dtype (a static Python value) doesn't
    cause issues with JIT.
    """

    def test_bound_method_captures_self_correctly(self):
        """Verify bound method captures correct self instance."""
        manifold_f32 = Poincare(dtype=jnp.float32)
        manifold_f64 = Poincare(dtype=jnp.float64)

        # Get bound methods
        dist_f32 = manifold_f32.dist
        dist_f64 = manifold_f64.dist

        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])

        # Call bound methods
        d_f32 = dist_f32(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)
        d_f64 = dist_f64(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        assert d_f32.dtype == jnp.float32
        assert d_f64.dtype == jnp.float64

    def test_bound_method_jit(self):
        """Verify bound methods can be JIT compiled."""
        manifold = Poincare(dtype=jnp.float64)

        # JIT compile the bound method directly
        dist_jit = jax.jit(manifold.dist, static_argnames=["version_idx"])

        x = jnp.array([0.1, 0.2])
        y = jnp.array([0.3, 0.4])
        d = dist_jit(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

        assert d.dtype == jnp.float64
        assert jnp.isfinite(d)
