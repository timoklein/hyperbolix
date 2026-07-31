"""Tests for hyp_avg_pool2d — hyperbolic 2D global average pooling."""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers.hyperboloid_core import hyp_avg_pool2d

# ============================================================================
# Helpers
# ============================================================================


def _make_hyperboloid_feature_map(key, batch, height, width, spatial_dim, c, dtype):
    """Generate a valid hyperboloid feature map of shape (batch, H, W, spatial_dim+1).

    Parameters
    ----------
    spatial_dim : int
        Spatial dimension.  Ambient (output) dimension is spatial_dim + 1.
    """
    hyperboloid = Hyperboloid(dtype=dtype)
    # Tangent vectors at origin have zero time component: [0, v_1, ..., v_d]
    v_spatial = jax.random.normal(key, (batch, height, width, spatial_dim), dtype=dtype) * 0.1
    v = jnp.concatenate([jnp.zeros_like(v_spatial[..., :1]), v_spatial], axis=-1)  # (..., d+1)
    proj = jax.vmap(
        jax.vmap(jax.vmap(hyperboloid.expmap_0, in_axes=(0, None)), in_axes=(0, None)),
        in_axes=(0, None),
    )
    return proj(v, c)


# ============================================================================
# Shape Tests
# ============================================================================


def test_pool_shape_4d():
    """(B, H, W, A) -> (B, A) where A = spatial_dim + 1."""
    key = jax.random.PRNGKey(0)
    x = _make_hyperboloid_feature_map(key, batch=4, height=7, width=7, spatial_dim=64, c=1.0, dtype=jnp.float32)
    assert x.shape == (4, 7, 7, 65)
    y = hyp_avg_pool2d(x, c=1.0)
    assert y.shape == (4, 65)


def test_pool_shape_3d():
    """(H, W, A) -> (A,)."""
    hyperboloid = Hyperboloid(dtype=jnp.float32)
    key = jax.random.PRNGKey(1)
    # Tangent vectors at origin: [0, v_1, ..., v_8] → ambient dim 9
    v_spatial = jax.random.normal(key, (3, 3, 8), dtype=jnp.float32) * 0.1
    v = jnp.concatenate([jnp.zeros_like(v_spatial[..., :1]), v_spatial], axis=-1)
    x = jax.vmap(jax.vmap(hyperboloid.expmap_0, in_axes=(0, None)), in_axes=(0, None))(v, 1.0)
    assert x.shape == (3, 3, 9)
    y = hyp_avg_pool2d(x, c=1.0)
    assert y.shape == (9,)


def test_pool_shape_5d():
    """(B1, B2, H, W, A) -> (B1, B2, A)."""
    key = jax.random.PRNGKey(2)
    # Build (6, H, W, 17) then reshape to (2, 3, H, W, 17)
    x = _make_hyperboloid_feature_map(key, batch=6, height=4, width=4, spatial_dim=16, c=1.0, dtype=jnp.float32)
    assert x.shape == (6, 4, 4, 17)
    x = x.reshape(2, 3, 4, 4, 17)
    y = hyp_avg_pool2d(x, c=1.0)
    assert y.shape == (2, 3, 17)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pool_dtype_preserved(dtype):
    """Output dtype matches input dtype."""
    key = jax.random.PRNGKey(3)
    x = _make_hyperboloid_feature_map(key, batch=2, height=3, width=3, spatial_dim=8, c=1.0, dtype=dtype)
    y = hyp_avg_pool2d(x, c=1.0)
    assert y.dtype == dtype


# ============================================================================
# Manifold Constraint Tests
# ============================================================================


@pytest.mark.parametrize("c", [0.5, 1.0, 5.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pool_manifold_constraint(dtype, c):
    """All output points satisfy the hyperboloid constraint."""
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(10)
    x = _make_hyperboloid_feature_map(key, batch=8, height=4, width=4, spatial_dim=16, c=c, dtype=dtype)
    y = hyp_avg_pool2d(x, c=c)  # (8, 17)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, c)
    assert is_valid.all()


@pytest.mark.parametrize("hw", [1, 7])
def test_pool_manifold_various_spatial(hw):
    """Manifold constraint holds for various spatial sizes."""
    hyperboloid = Hyperboloid(dtype=jnp.float64)
    key = jax.random.PRNGKey(20)
    x = _make_hyperboloid_feature_map(key, batch=4, height=hw, width=hw, spatial_dim=8, c=1.0, dtype=jnp.float64)
    y = hyp_avg_pool2d(x, c=1.0)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


# ============================================================================
# Correctness Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pool_matches_manual(dtype):
    """Output matches the hand-rolled extract+pool+reconstruct pattern."""
    key = jax.random.PRNGKey(30)
    c = 1.0
    x = _make_hyperboloid_feature_map(key, batch=4, height=7, width=7, spatial_dim=32, c=c, dtype=dtype)

    # Hand-rolled (the pattern from benchmarks)
    x_space = x[..., 1:]  # (4, 7, 7, 32)
    x_pooled = jnp.mean(x_space, axis=(1, 2))  # (4, 32)
    time_coord = jnp.sqrt(jnp.sum(x_pooled**2, axis=-1, keepdims=True) + 1.0 / c)
    expected = jnp.concatenate([time_coord, x_pooled], axis=-1)  # (4, 33)

    y = hyp_avg_pool2d(x, c=c)

    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    assert jnp.allclose(y, expected, atol=atol)


def test_pool_1x1_identity():
    """When H=W=1, spatial components pass through unchanged."""
    key = jax.random.PRNGKey(32)
    c = 1.0
    x = _make_hyperboloid_feature_map(key, batch=4, height=1, width=1, spatial_dim=16, c=c, dtype=jnp.float64)

    y = hyp_avg_pool2d(x, c=c)  # (4, 17)
    # Squeeze H,W from input for comparison
    x_squeezed = x[:, 0, 0, :]  # (4, 17)

    assert jnp.allclose(y, x_squeezed, atol=1e-7)


def test_pool_uniform_spatial():
    """When all spatial positions are identical, output equals that point."""
    hyperboloid = Hyperboloid(dtype=jnp.float64)
    key = jax.random.PRNGKey(33)
    c = 1.0

    # Create a single hyperboloid point and tile across H, W
    v = jax.random.normal(key, (4, 8), dtype=jnp.float64) * 0.1
    points = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, c)  # (4, 9)
    x = jnp.tile(points[:, None, None, :], (1, 5, 5, 1))  # (4, 5, 5, 9)

    y = hyp_avg_pool2d(x, c=c)  # (4, 9)

    assert jnp.allclose(y, points, atol=1e-7)


# ============================================================================
# Gradient Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pool_gradients_finite(dtype):
    """Gradients through hyp_avg_pool2d are finite."""
    key = jax.random.PRNGKey(40)
    c = 1.0
    x = _make_hyperboloid_feature_map(key, batch=2, height=4, width=4, spatial_dim=8, c=c, dtype=dtype)

    def loss_fn(x_in):
        y = hyp_avg_pool2d(x_in, c=c)
        return jnp.sum(y**2)

    grads = jax.grad(loss_fn)(x)
    assert grads.shape == x.shape
    assert jnp.all(jnp.isfinite(grads))


# ============================================================================
# JIT Compatibility Tests
# ============================================================================


def test_pool_jit():
    """JIT-compiled output matches eager execution."""
    key = jax.random.PRNGKey(50)
    c = 1.0
    x = _make_hyperboloid_feature_map(key, batch=4, height=4, width=4, spatial_dim=16, c=c, dtype=jnp.float64)

    y_eager = hyp_avg_pool2d(x, c=c)
    y_jit = jax.jit(hyp_avg_pool2d, static_argnames="c")(x, c=c)

    assert jnp.allclose(y_eager, y_jit, atol=1e-10)
