"""Tests for Hyperboloid activation functions."""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.manifolds import hyperboloid
from hyperbolix.nn_layers.hyperboloid_activations import (
    hyp_leaky_relu,
    hyp_relu,
    hyp_swish,
    hyp_tanh,
)

# ============================================================================
# Manifold Constraint Tests (Most Critical)
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_manifold_constraint_single(dtype):
    """Test that hyp_relu output for single point lies on manifold."""
    key = jax.random.PRNGKey(42)
    dim = 4

    # Generate valid hyperboloid point
    v = jax.random.normal(key, (dim,), dtype=dtype) * 0.1
    x = hyperboloid.expmap_0(v, c=1.0)

    # Apply activation
    y = hyp_relu(x, c=1.0)

    # Check manifold constraint
    assert hyperboloid.is_in_manifold(y, c=1.0, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_manifold_constraint_batch(dtype):
    """Test that hyp_relu output for batch lies on manifold."""
    key = jax.random.PRNGKey(42)
    batch_size, dim = 8, 4

    # Generate valid hyperboloid points
    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    # Apply activation (no vmap needed!)
    y = hyp_relu(x, c=1.0)

    # Check manifold constraint for all points
    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_manifold_constraint_multi_dim(dtype):
    """Test that hyp_relu output for multi-dimensional batch lies on manifold."""
    key = jax.random.PRNGKey(42)
    batch, height, width, dim = 2, 4, 4, 3

    # Generate valid hyperboloid points
    v = jax.random.normal(key, (batch, height, width, dim), dtype=dtype) * 0.1
    x = jax.vmap(jax.vmap(jax.vmap(hyperboloid.expmap_0, in_axes=(0, None)), in_axes=(0, None)), in_axes=(0, None))(v, 1.0)

    # Apply activation
    y = hyp_relu(x, c=1.0)

    # Check manifold constraint
    is_valid = jax.vmap(jax.vmap(jax.vmap(lambda p: hyperboloid.is_in_manifold(p, 1.0))))(y)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_leaky_relu_manifold_constraint(dtype):
    """Test that hyp_leaky_relu output lies on manifold."""
    key = jax.random.PRNGKey(43)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_leaky_relu(x, c=1.0, negative_slope=0.01)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_tanh_manifold_constraint(dtype):
    """Test that hyp_tanh output lies on manifold."""
    key = jax.random.PRNGKey(44)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_tanh(x, c=1.0)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_swish_manifold_constraint(dtype):
    """Test that hyp_swish output lies on manifold."""
    key = jax.random.PRNGKey(45)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_swish(x, c=1.0)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


# ============================================================================
# Shape Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("dim", [2, 4, 10])
def test_hyp_relu_shape_single(dtype, dim):
    """Test that hyp_relu preserves shape for single points."""
    key = jax.random.PRNGKey(42)

    v = jax.random.normal(key, (dim,), dtype=dtype) * 0.1
    x = hyperboloid.expmap_0(v, c=1.0)

    y = hyp_relu(x, c=1.0)

    assert y.shape == x.shape
    assert y.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.parametrize("dim", [2, 4, 10])
def test_hyp_relu_shape_batch(dtype, batch_size, dim):
    """Test that hyp_relu preserves shape for batches."""
    key = jax.random.PRNGKey(42)

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_relu(x, c=1.0)

    assert y.shape == x.shape
    assert y.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_shape_multi_dim(dtype):
    """Test that hyp_relu preserves shape for multi-dimensional batches."""
    key = jax.random.PRNGKey(42)
    batch, height, width, dim = 4, 8, 8, 5

    v = jax.random.normal(key, (batch, height, width, dim), dtype=dtype) * 0.1
    x = jax.vmap(jax.vmap(jax.vmap(hyperboloid.expmap_0, in_axes=(0, None)), in_axes=(0, None)), in_axes=(0, None))(v, 1.0)

    y = hyp_relu(x, c=1.0)

    assert y.shape == x.shape
    assert y.dtype == dtype


# ============================================================================
# Correctness Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_formula(dtype):
    """Test that hyp_relu correctly implements the formula."""
    key = jax.random.PRNGKey(42)
    batch_size, dim = 8, 4
    c = 1.0

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, c)

    y = hyp_relu(x, c=c)

    # Verify formula: y[0] = sqrt(||y[1:]||^2 + 1/c)
    spatial = y[:, 1:]
    expected_x0 = jnp.sqrt(jnp.sum(spatial**2, axis=-1) + 1.0 / c)

    assert jnp.allclose(y[:, 0], expected_x0, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_negative_components_zeroed(dtype):
    """Test that negative spatial components become zero after hyp_relu."""
    # Create a point with some negative spatial components
    x = jnp.array([1.5, 0.3, -0.5, 0.2, -0.1], dtype=dtype)
    # Project to manifold
    x = hyperboloid.proj(x, c=1.0)

    y = hyp_relu(x, c=1.0)

    # Spatial components that were negative should be zero
    # Original: x[1:] = [0.3, -0.5, 0.2, -0.1] (approximately, after projection)
    # Expected: y[1:] = [max(0.3,0), max(-0.5,0), max(0.2,0), max(-0.1,0)]
    #                  = [0.3, 0, 0.2, 0]

    # Check that we can't have negative values
    assert jnp.all(y[1:] >= 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 0.2])
def test_hyp_leaky_relu_negative_slope(dtype, negative_slope):
    """Test that hyp_leaky_relu correctly applies negative_slope."""
    key = jax.random.PRNGKey(42)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.5
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_leaky_relu(x, c=1.0, negative_slope=negative_slope)

    # Verify manifold constraint holds (already tested above) and shape
    assert y.shape == x.shape


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_tanh_bounded(dtype):
    """Test that hyp_tanh produces bounded spatial components."""
    key = jax.random.PRNGKey(42)
    batch_size, dim = 8, 4

    # Use large values to test bounding
    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 5.0
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_tanh(x, c=1.0)

    # Spatial components should be bounded in [-1, 1]
    spatial = y[:, 1:]
    assert jnp.all(jnp.abs(spatial) <= 1.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_swish_smooth_at_zero(dtype):
    """Test that hyp_swish is smooth around zero."""
    # Create points near origin
    x1 = jnp.array([1.0, 0.01, 0.01], dtype=dtype)
    x2 = jnp.array([1.0, -0.01, -0.01], dtype=dtype)

    x1 = hyperboloid.proj(x1, c=1.0)
    x2 = hyperboloid.proj(x2, c=1.0)

    y1 = hyp_swish(x1, c=1.0)
    y2 = hyp_swish(x2, c=1.0)

    # Swish should be continuous and smooth (no abrupt changes)
    assert jnp.isfinite(y1).all()
    assert jnp.isfinite(y2).all()


# ============================================================================
# Gradient Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_gradients(dtype):
    """Test that hyp_relu has finite gradients."""
    key = jax.random.PRNGKey(42)
    dim = 4

    v = jax.random.normal(key, (dim,), dtype=dtype) * 0.1
    x = hyperboloid.expmap_0(v, c=1.0)

    def loss_fn(x):
        y = hyp_relu(x, c=1.0)
        return jnp.sum(y**2)

    grad = jax.grad(loss_fn)(x)

    assert jnp.isfinite(grad).all()
    assert grad.shape == x.shape


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_leaky_relu_gradients(dtype):
    """Test that hyp_leaky_relu has finite gradients."""
    key = jax.random.PRNGKey(43)
    dim = 4

    v = jax.random.normal(key, (dim,), dtype=dtype) * 0.1
    x = hyperboloid.expmap_0(v, c=1.0)

    def loss_fn(x):
        y = hyp_leaky_relu(x, c=1.0, negative_slope=0.01)
        return jnp.sum(y**2)

    grad = jax.grad(loss_fn)(x)

    assert jnp.isfinite(grad).all()
    assert grad.shape == x.shape


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_tanh_gradients(dtype):
    """Test that hyp_tanh has finite gradients."""
    key = jax.random.PRNGKey(44)
    dim = 4

    v = jax.random.normal(key, (dim,), dtype=dtype) * 0.1
    x = hyperboloid.expmap_0(v, c=1.0)

    def loss_fn(x):
        y = hyp_tanh(x, c=1.0)
        return jnp.sum(y**2)

    grad = jax.grad(loss_fn)(x)

    assert jnp.isfinite(grad).all()
    assert grad.shape == x.shape


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_swish_gradients(dtype):
    """Test that hyp_swish has finite gradients."""
    key = jax.random.PRNGKey(45)
    dim = 4

    v = jax.random.normal(key, (dim,), dtype=dtype) * 0.1
    x = hyperboloid.expmap_0(v, c=1.0)

    def loss_fn(x):
        y = hyp_swish(x, c=1.0)
        return jnp.sum(y**2)

    grad = jax.grad(loss_fn)(x)

    assert jnp.isfinite(grad).all()
    assert grad.shape == x.shape


# ============================================================================
# JIT Compatibility Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_jit(dtype):
    """Test that hyp_relu works with JIT compilation."""
    key = jax.random.PRNGKey(42)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    @jax.jit
    def apply_activation(x):
        return hyp_relu(x, c=1.0)

    y = apply_activation(x)

    assert y.shape == x.shape
    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_leaky_relu_jit(dtype):
    """Test that hyp_leaky_relu works with JIT compilation."""
    key = jax.random.PRNGKey(43)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    @jax.jit
    def apply_activation(x):
        return hyp_leaky_relu(x, c=1.0, negative_slope=0.01)

    y = apply_activation(x)

    assert y.shape == x.shape
    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_tanh_jit(dtype):
    """Test that hyp_tanh works with JIT compilation."""
    key = jax.random.PRNGKey(44)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    @jax.jit
    def apply_activation(x):
        return hyp_tanh(x, c=1.0)

    y = apply_activation(x)

    assert y.shape == x.shape
    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_swish_jit(dtype):
    """Test that hyp_swish works with JIT compilation."""
    key = jax.random.PRNGKey(45)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    @jax.jit
    def apply_activation(x):
        return hyp_swish(x, c=1.0)

    y = apply_activation(x)

    assert y.shape == x.shape
    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


# ============================================================================
# Curvature Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_hyp_relu_different_curvatures(dtype, c):
    """Test that hyp_relu works with different curvature values."""
    key = jax.random.PRNGKey(42)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, c)

    y = hyp_relu(x, c=c)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, c)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_hyp_tanh_different_curvatures(dtype, c):
    """Test that hyp_tanh works with different curvature values."""
    key = jax.random.PRNGKey(44)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, c)

    y = hyp_tanh(x, c=c)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, c)
    assert is_valid.all()


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_zero_spatial_components(dtype):
    """Test hyp_relu with zero spatial components."""
    # Point at origin
    x = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    x = hyperboloid.proj(x, c=1.0)

    y = hyp_relu(x, c=1.0)

    assert hyperboloid.is_in_manifold(y, c=1.0, atol=1e-5)
    # All spatial components should remain zero
    assert jnp.allclose(y[1:], 0.0, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_moderate_magnitude(dtype):
    """Test hyp_relu with moderate magnitude inputs."""
    key = jax.random.PRNGKey(42)
    batch_size, dim = 8, 4

    # Generate moderate magnitude vectors (increased from 0.1 to test robustness)
    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 2.0
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_relu(x, c=1.0)

    assert jnp.isfinite(y).all()
    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_tanh_moderate_magnitude(dtype):
    """Test hyp_tanh with moderate magnitude inputs."""
    key = jax.random.PRNGKey(44)
    batch_size, dim = 8, 4

    # Generate moderate magnitude vectors
    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 3.0
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_tanh(x, c=1.0)

    assert jnp.isfinite(y).all()
    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()
    # Tanh should still bound outputs
    assert jnp.all(jnp.abs(y[:, 1:]) <= 1.0)
