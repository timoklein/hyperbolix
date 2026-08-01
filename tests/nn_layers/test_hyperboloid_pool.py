"""Tests for the hyperboloid conv → FC bridges.

``hyp_avg_pool2d`` (dimension-preserving global average pool) and ``hyp_flatten2d``
(LogCat flatten of the whole feature map into one wider point).

Dimension key:
  B: batch    H: height    W: width    N: H*W blocks
  D: per-pixel spatial dim             A: per-pixel ambient dim (D+1)
  Af: flattened ambient dim (H*W*D + 1)
"""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers.hyperboloid_core import hyp_avg_pool2d, hyp_flatten2d

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


# ============================================================================
# hyp_flatten2d — LogCat flatten at the conv → FC boundary
# ============================================================================


@pytest.mark.parametrize(
    "batch,height,width,spatial_dim",
    [(2, 3, 3, 8), (4, 2, 5, 4), (3, 7, 7, 2), (1, 1, 1, 16)],
)
def test_flatten_shape_bookkeeping(batch, height, width, spatial_dim):
    """(B, H, W, D+1) -> (B, H*W*D + 1): H*W blocks of D spatial dims plus one time coordinate."""
    hyperboloid = Hyperboloid(dtype=jnp.float64)
    key = jax.random.PRNGKey(60)
    x_BHWA = _make_hyperboloid_feature_map(key, batch, height, width, spatial_dim, c=1.0, dtype=jnp.float64)
    assert x_BHWA.shape == (batch, height, width, spatial_dim + 1)

    y_BAf = hyp_flatten2d(x_BHWA, hyperboloid, c=1.0)

    assert y_BAf.shape == (batch, height * width * spatial_dim + 1)


def test_flatten_shape_unbatched():
    """A bare (H, W, A) feature map (no leading batch axis) flattens to (H*W*D + 1,)."""
    hyperboloid = Hyperboloid(dtype=jnp.float64)
    key = jax.random.PRNGKey(61)
    x_BHWA = _make_hyperboloid_feature_map(key, batch=1, height=4, width=4, spatial_dim=8, c=1.0, dtype=jnp.float64)

    y_Af = hyp_flatten2d(x_BHWA[0], hyperboloid, c=1.0)

    assert y_Af.shape == (4 * 4 * 8 + 1,)
    assert jnp.allclose(y_Af, hyp_flatten2d(x_BHWA, hyperboloid, c=1.0)[0], atol=1e-12)


@pytest.mark.parametrize("c", [0.5, 1.0, 5.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_flatten_output_on_manifold(dtype, c):
    """Flattened points satisfy the hyperboloid constraint of the widened manifold."""
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(62)
    x_BHWA = _make_hyperboloid_feature_map(key, batch=6, height=4, width=4, spatial_dim=8, c=c, dtype=dtype)

    y_BAf = hyp_flatten2d(x_BHWA, hyperboloid, c=c)

    assert y_BAf.dtype == dtype
    assert jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y_BAf, c).all()

    # Independent time-component reconstruction: t = sqrt(||space||^2 + 1/c).
    expected_time_B = jnp.sqrt(jnp.sum(y_BAf[:, 1:] ** 2, axis=-1) + 1.0 / c)
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    assert jnp.allclose(y_BAf[:, 0], expected_time_B, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_flatten_matches_manual_log_radius_concat(dtype):
    """Oracle: identical to reshaping each sample to (H*W, A) and calling log_radius_concat."""
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(63)
    c = 1.0
    batch, height, width, spatial_dim = 4, 3, 5, 6
    x_BHWA = _make_hyperboloid_feature_map(key, batch, height, width, spatial_dim, c=c, dtype=dtype)

    expected_BAf = jnp.stack(
        [hyperboloid.log_radius_concat(x_BHWA[b].reshape(height * width, spatial_dim + 1), c) for b in range(batch)]
    )
    y_BAf = hyp_flatten2d(x_BHWA, hyperboloid, c=c)

    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    assert jnp.allclose(y_BAf, expected_BAf, atol=atol)


def test_flatten_single_pixel_reduces_to_identity():
    """H = W = 1 means N = 1 blocks, so the digamma scale is 1 and the point passes through."""
    hyperboloid = Hyperboloid(dtype=jnp.float64)
    key = jax.random.PRNGKey(64)
    c = 1.0
    x_BHWA = _make_hyperboloid_feature_map(key, batch=4, height=1, width=1, spatial_dim=12, c=c, dtype=jnp.float64)

    y_BAf = hyp_flatten2d(x_BHWA, hyperboloid, c=c)

    assert jnp.allclose(y_BAf, x_BHWA[:, 0, 0, :], atol=1e-9)


def test_flatten_radius_stays_order_of_per_pixel_radius():
    """The point of the helper: the flattened radius tracks one pixel's, not ~sqrt(H*W) times it.

    Naive ``hcat`` flattening widens the spatial dimension without rescaling, so the
    concatenated spatial norm grows like ``sqrt(N)`` (N = H*W = 64 here). LogCat's digamma
    shrink ``≈ 1/sqrt(N)`` cancels that. Compared against the mean per-pixel geodesic radius,
    which is the quantity LogCat claims to preserve.
    """
    hyperboloid = Hyperboloid(dtype=jnp.float64)
    key = jax.random.PRNGKey(65)
    c = 1.0
    batch, height, width, spatial_dim = 4, 8, 8, 8
    x_BHWA = _make_hyperboloid_feature_map(key, batch, height, width, spatial_dim, c=c, dtype=jnp.float64)

    dist_0_grid = jax.vmap(jax.vmap(jax.vmap(hyperboloid.dist_0, in_axes=(0, None)), in_axes=(0, None)), in_axes=(0, None))
    per_pixel_radius = float(jnp.mean(dist_0_grid(x_BHWA, c)))

    logcat_BAf = hyp_flatten2d(x_BHWA, hyperboloid, c=c)
    logcat_radius = float(jnp.mean(jax.vmap(hyperboloid.dist_0, in_axes=(0, None))(logcat_BAf, c)))

    points_BNA = x_BHWA.reshape(batch, height * width, spatial_dim + 1)
    naive_BAf = jax.vmap(hyperboloid.hcat, in_axes=(0, None))(points_BNA, c)
    naive_radius = float(jnp.mean(jax.vmap(hyperboloid.dist_0, in_axes=(0, None))(naive_BAf, c)))

    assert 0.5 < logcat_radius / per_pixel_radius < 1.5, f"LogCat radius drifted: {logcat_radius / per_pixel_radius:.3f}"
    # Not vacuous: the naive flatten inflates by several times the per-pixel radius.
    assert naive_radius / per_pixel_radius > 3.0
    assert naive_radius > 3.0 * logcat_radius


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_flatten_jit_matches_eager(dtype):
    """JIT-compiled output matches eager execution (manifold instance passed as a static arg)."""
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(66)
    c = 1.0
    x_BHWA = _make_hyperboloid_feature_map(key, batch=4, height=4, width=4, spatial_dim=8, c=c, dtype=dtype)

    y_eager = hyp_flatten2d(x_BHWA, hyperboloid, c=c)
    y_jit = jax.jit(hyp_flatten2d, static_argnums=(1,), static_argnames="c")(x_BHWA, hyperboloid, c=c)

    # float32 tolerance is loose on purpose: XLA fuses the reshape/vmap chain differently
    # than the eager path, so bit-exactness is not expected.
    atol = 1e-5 if dtype == jnp.float32 else 1e-12
    assert jnp.allclose(y_eager, y_jit, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_flatten_gradients_finite(dtype):
    """Gradients through hyp_flatten2d are finite."""
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(67)
    c = 1.0
    x_BHWA = _make_hyperboloid_feature_map(key, batch=2, height=3, width=3, spatial_dim=8, c=c, dtype=dtype)

    def loss_fn(x_in):
        return jnp.sum(hyp_flatten2d(x_in, hyperboloid, c=c) ** 2)

    grads = jax.grad(loss_fn)(x_BHWA)

    assert grads.shape == x_BHWA.shape
    assert jnp.all(jnp.isfinite(grads))


def test_flatten_rejects_non_hyperboloid_manifold():
    """A manifold without log_radius_concat (Poincaré) is rejected up front."""
    key = jax.random.PRNGKey(68)
    x_BHWA = _make_hyperboloid_feature_map(key, batch=2, height=2, width=2, spatial_dim=4, c=1.0, dtype=jnp.float64)

    with pytest.raises(TypeError, match="Hyperboloid"):
        hyp_flatten2d(x_BHWA, Poincare(dtype=jnp.float64), c=1.0)


def test_flatten_rejects_input_without_hw_axes():
    """Fewer than 3 axes means there is no (H, W, ambient) tail to flatten."""
    hyperboloid = Hyperboloid(dtype=jnp.float64)
    x_BA = jnp.zeros((4, 9), dtype=jnp.float64).at[:, 0].set(1.0)

    with pytest.raises(ValueError, match="at least 3 axes"):
        hyp_flatten2d(x_BA, hyperboloid, c=1.0)
