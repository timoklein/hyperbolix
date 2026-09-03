"""Tests for Hyperboloid activation functions."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers.hyperboloid_activations import (
    hrc_gelu,
    hrc_leaky_relu,
    hrc_relu,
    hrc_swish,
    hrc_tanh,
    hyp_gelu,
    hyp_leaky_relu,
    hyp_relu,
    hyp_swish,
    hyp_tanh,
)

# (id, curvature-preserving wrapper f(x, c), curvature-changing hrc f(x, c_in, c_out),
#  the plain Euclidean function that must be applied to the SPATIAL components).
SPATIAL_ACTIVATIONS = [
    ("relu", hyp_relu, hrc_relu, jax.nn.relu),
    (
        "leaky_relu",
        partial(hyp_leaky_relu, negative_slope=0.01),
        partial(hrc_leaky_relu, negative_slope=0.01),
        lambda z: jax.nn.leaky_relu(z, 0.01),
    ),
    ("tanh", hyp_tanh, hrc_tanh, jnp.tanh),
    ("swish", hyp_swish, hrc_swish, jax.nn.swish),
    ("gelu", hyp_gelu, hrc_gelu, jax.nn.gelu),
]
SPATIAL_ACTIVATION_IDS = [a[0] for a in SPATIAL_ACTIVATIONS]


def _points(dtype, c, spatial):
    """On-manifold points at curvature ``c`` from spatial tangent rows."""
    manifold = Hyperboloid(dtype=dtype)
    v_ND = jnp.asarray(spatial, dtype=dtype)
    v_amb = manifold.embed_spatial_0(v_ND)
    return manifold, jax.vmap(manifold.expmap_0, in_axes=(0, None))(v_amb, c)


# Spatial rows with a healthy mix of signs and magnitudes — the negative entries
# are what make an identity-substituted activation visible.
_SPATIAL_ROWS = [[0.4, -0.6, 0.2, -1.1], [-0.3, 0.9, -0.5, 0.05]]

# ============================================================================
# Manifold Constraint Tests (Most Critical)
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_manifold_constraint_single(dtype):
    """Test that hyp_relu output for single point lies on manifold."""
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(45)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_swish(x, c=1.0)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_gelu_manifold_constraint(dtype):
    """Test that hyp_gelu output lies on manifold."""
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(46)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_gelu(x, c=1.0)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(y, 1.0)
    assert is_valid.all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_hyp_gelu_delegates_to_hrc_gelu(dtype, c):
    """hyp_gelu is the curvature-preserving wrapper hrc_gelu(x, c_in=c, c_out=c).

    The wrapper body is one line of argument wiring with no caller inside the
    library, so nothing else pins the curvature reaching both c_in and c_out.
    Parametrized over c because c_in == c_out makes any single-curvature check
    blind to the two arguments being swapped or one being dropped.
    """
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(46)
    batch_size, dim = 8, 4

    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 0.1
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, c)

    assert jnp.array_equal(hyp_gelu(x, c=c), hrc_gelu(x, c_in=c, c_out=c))


# ============================================================================
# Shape Tests
# ============================================================================


def test_hyp_relu_shape_multi_dim():
    """Test that hyp_relu preserves shape/dtype for multi-dimensional batches.

    float32 because that is the dtype a stray ``Hyperboloid(dtype=float64)``
    instance would silently promote under ``JAX_ENABLE_X64=1``.
    """
    dtype = jnp.float32
    hyperboloid = Hyperboloid(dtype=dtype)
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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("name,hyp_fn,hrc_fn,euclidean_fn", SPATIAL_ACTIVATIONS, ids=SPATIAL_ACTIVATION_IDS)
def test_activation_applies_euclidean_fn_to_spatial(name, hyp_fn, hrc_fn, euclidean_fn, dtype):
    """``y = [sqrt(‖f(x_s)‖² + 1/c), f(x_s)]`` for the Euclidean function ``f``.

    The per-activation semantics check: an independent transcription of the HRC
    formula against the plain ``jax.nn`` function on the spatial block. Without
    it nothing distinguishes ``hyp_leaky_relu`` / ``hyp_swish`` from the identity
    (the manifold-constraint and shape tests pass for a pass-through), and the
    ``1/c`` in the time reconstruction is only pinned at a non-unit curvature.

    The jit leg is folded in here (A8-13): a compiled call must reproduce the
    eager values exactly, which is strictly more than the old jit-only tests
    (shape + on-manifold) checked.
    """
    c = 0.7
    _manifold, x_NF = _points(dtype, c, _SPATIAL_ROWS)

    y_NF = hyp_fn(x_NF, c)

    expected_space = euclidean_fn(x_NF[:, 1:])
    expected_time = jnp.sqrt(jnp.sum(expected_space**2, axis=-1) + 1.0 / c)
    atol = 1e-6 if dtype == jnp.float32 else 1e-12
    assert jnp.allclose(y_NF[:, 1:], expected_space, atol=atol)
    assert jnp.allclose(y_NF[:, 0], expected_time, atol=atol)
    # Identity-substitution guard: f must actually move the spatial components.
    assert not jnp.allclose(y_NF[:, 1:], x_NF[:, 1:], atol=1e-3)
    # jit fold: the compiled output equals the eager output to 1 ulp. Not bit-for-bit: on GPU a
    # transcendental fused into the compiled kernel and the same op run eagerly are different
    # kernels, and XLA does not promise them identical — `tanh` in float64 differs by 1 ulp on
    # one element of ten. The two *value* assertions above stay strict; this one only guards
    # against jit changing the computation, which would move it far more than an ulp.
    assert jnp.allclose(jax.jit(lambda z: hyp_fn(z, c))(x_NF), y_NF, rtol=1e-15, atol=0)


@pytest.mark.parametrize("name,hyp_fn,hrc_fn,euclidean_fn", SPATIAL_ACTIVATIONS, ids=SPATIAL_ACTIVATION_IDS)
def test_hrc_scales_spatial_by_sqrt_curvature_ratio(name, hyp_fn, hrc_fn, euclidean_fn):
    """``hrc_*(x, c_in, c_out)`` spatial block is ``sqrt(c_in/c_out)·f(x_s)``.

    Pins the curvature-change leg that the ``c_in == c_out`` wrappers can never
    exercise: with c_in = c_out the ratio is 1 and a dropped rescale is invisible.
    """
    c_in, c_out = 0.5, 2.0
    _manifold, x_NF = _points(jnp.float64, c_in, _SPATIAL_ROWS)

    y_NF = hrc_fn(x_NF, c_in=c_in, c_out=c_out)

    expected_space = jnp.sqrt(c_in / c_out) * euclidean_fn(x_NF[:, 1:])
    expected_time = jnp.sqrt(jnp.sum(expected_space**2, axis=-1) + 1.0 / c_out)
    assert jnp.allclose(y_NF[:, 1:], expected_space, atol=1e-12)
    assert jnp.allclose(y_NF[:, 0], expected_time, atol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_negative_components_zeroed(dtype):
    """Test that negative spatial components become zero after hyp_relu."""
    hyperboloid = Hyperboloid(dtype=dtype)
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


@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 0.2])
def test_hyp_leaky_relu_negative_slope(negative_slope):
    """``negative_slope`` actually scales the negative spatial components.

    The old version only re-checked the shape, so any value of ``negative_slope``
    (including one that never reached ``jax.nn.leaky_relu``) passed. The
    scaled-negatives assertion plus the differs-from-ReLU leg pin the argument.
    """
    c = 1.0
    _manifold, x_NF = _points(jnp.float64, c, _SPATIAL_ROWS)
    x_space = x_NF[:, 1:]
    assert jnp.any(x_space < 0)  # the test data must exercise the negative branch

    y_NF = hyp_leaky_relu(x_NF, c=c, negative_slope=negative_slope)

    expected_space = jnp.where(x_space > 0, x_space, negative_slope * x_space)
    assert jnp.allclose(y_NF[:, 1:], expected_space, atol=1e-12)
    # A slope of 0 would be plain ReLU; a slope of 1 would be the identity.
    assert not jnp.allclose(y_NF[:, 1:], jax.nn.relu(x_space), atol=1e-6)
    assert not jnp.allclose(y_NF[:, 1:], x_space, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_tanh_bounded(dtype):
    """Test that hyp_tanh produces bounded spatial components."""
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(42)
    batch_size, dim = 8, 4

    # Use large values to test bounding
    v = jax.random.normal(key, (batch_size, dim), dtype=dtype) * 5.0
    x = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))(v, 1.0)

    y = hyp_tanh(x, c=1.0)

    # Spatial components should be bounded in [-1, 1]
    spatial = y[:, 1:]
    assert jnp.all(jnp.abs(spatial) <= 1.0)


# ============================================================================
# Gradient Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_relu_gradients(dtype):
    """Test that hyp_relu has finite gradients."""
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_gelu_gradients(dtype):
    """Test that hyp_gelu has finite gradients."""
    hyperboloid = Hyperboloid(dtype=dtype)
    key = jax.random.PRNGKey(46)
    dim = 4

    v = jax.random.normal(key, (dim,), dtype=dtype) * 0.1
    x = hyperboloid.expmap_0(v, c=1.0)

    def loss_fn(x):
        y = hyp_gelu(x, c=1.0)
        return jnp.sum(y**2)

    grad = jax.grad(loss_fn)(x)

    assert jnp.isfinite(grad).all()
    assert grad.shape == x.shape


# ============================================================================
# Curvature Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_hyp_relu_different_curvatures(dtype, c):
    """Test that hyp_relu works with different curvature values."""
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
    hyperboloid = Hyperboloid(dtype=dtype)
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
