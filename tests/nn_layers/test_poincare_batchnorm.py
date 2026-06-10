"""Tests for Poincaré batch normalization layer and helpers.

Dimension key:
  B: batch size     H: height     W: width
  C: channels       N: flattened (B*H*W)
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers.poincare_batchnorm import (
    PoincareBatchNorm2D,
    frechet_variance,
    poincare_midpoint,
)

# ============================================================================
# Helper: create random tangent/manifold inputs
# ============================================================================


def _make_poincare_points(key, shape, c, dtype):
    """Create random points on the Poincaré ball via expmap_0 of small tangent vectors."""
    manifold = Poincare(dtype=dtype)
    tangent = jax.random.normal(key, shape, dtype=dtype) * 0.1
    if tangent.ndim == 1:
        return manifold.expmap_0(tangent, c)
    return jax.vmap(manifold.expmap_0, in_axes=(0, None))(tangent, c)


def _make_tangent_input(key, shape, dtype):
    """Create random tangent-space input (small vectors near origin)."""
    return jax.random.normal(key, shape, dtype=dtype) * 0.1


# ============================================================================
# Midpoint Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_midpoint_on_manifold(dtype):
    """Midpoint lies on the Poincaré ball."""
    key = jax.random.PRNGKey(42)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    x_NC = _make_poincare_points(key, (16, 8), c, dtype)

    mid_C = poincare_midpoint(x_NC, manifold, c)

    assert mid_C.shape == (8,)
    assert manifold.is_in_manifold(mid_C, c, atol=1e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_midpoint_identical_points(dtype):
    """Midpoint of N copies of the same point equals that point."""
    key = jax.random.PRNGKey(0)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    single = _make_poincare_points(key, (4,), c, dtype)  # (C,)
    repeated_NC = jnp.tile(single, (10, 1))  # (10, C)

    mid_C = poincare_midpoint(repeated_NC, manifold, c)

    atol = 1e-3 if dtype == jnp.float32 else 1e-7
    assert jnp.allclose(mid_C, single, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_midpoint_equidistant_two_points(dtype):
    """Midpoint of two points is geodesically equidistant from both.

    Regression guard: the old ``Σλ²x/Σλ²`` formula gave (0.8686, 0) for this
    pair — geodesic distances 0.289 / 2.655 — because the λ² weighting drags
    the mean toward boundary points. The gyromidpoint has the closed form
    ``tanh(atanh(0.9)/2)·e₁`` here.
    """
    c = 1.0
    manifold = Poincare(dtype=dtype)
    x_NC = jnp.array([[0.9, 0.0], [0.0, 0.0]], dtype=dtype)

    mid_C = poincare_midpoint(x_NC, manifold, c)

    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    expected_x = jnp.tanh(jnp.arctanh(jnp.asarray(0.9, dtype=dtype)) / 2.0)
    assert jnp.allclose(mid_C, jnp.array([expected_x, 0.0], dtype=dtype), atol=atol)

    d0 = manifold.dist(mid_C, x_NC[0], c)
    d1 = manifold.dist(mid_C, x_NC[1], c)
    assert jnp.allclose(d0, d1, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_midpoint_gradients(dtype):
    """Gradients through midpoint are finite."""
    key = jax.random.PRNGKey(1)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    x_NC = _make_poincare_points(key, (8, 4), c, dtype)

    def loss_fn(x):
        mid = poincare_midpoint(x, manifold, c)
        return jnp.sum(mid**2)

    grads = jax.grad(loss_fn)(x_NC)
    assert jnp.all(jnp.isfinite(grads))


# ============================================================================
# Variance Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_frechet_variance_nonnegative(dtype):
    """Fréchet variance is non-negative."""
    key = jax.random.PRNGKey(42)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    x_NC = _make_poincare_points(key, (16, 8), c, dtype)
    mean_C = poincare_midpoint(x_NC, manifold, c)

    var = frechet_variance(x_NC, mean_C, manifold, c)
    assert var >= 0.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_frechet_variance_identical_points(dtype):
    """Variance of identical points is zero."""
    key = jax.random.PRNGKey(0)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    single = _make_poincare_points(key, (4,), c, dtype)
    repeated_NC = jnp.tile(single, (10, 1))

    var = frechet_variance(repeated_NC, single, manifold, c)
    assert var < 1e-6


# ============================================================================
# BatchNorm Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_output_shape(dtype):
    """Output shape matches input shape (B, H, W, C)."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=8)
    key = jax.random.PRNGKey(42)
    x_BHWC = _make_tangent_input(key, (2, 4, 4, 8), dtype)

    out = bn(x_BHWC, c=1.0)
    assert out.shape == (2, 4, 4, 8)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_output_finite(dtype):
    """No NaN or Inf in output."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=8)
    key = jax.random.PRNGKey(42)
    x_BHWC = _make_tangent_input(key, (2, 4, 4, 8), dtype)

    out = bn(x_BHWC, c=1.0)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_output_mappable_to_manifold(dtype):
    """Tangent output maps to valid manifold points via expmap_0."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = _make_tangent_input(key, (2, 3, 3, 4), dtype)

    out_BHWC = bn(x_BHWC, c=1.0)
    # Flatten and map to manifold
    out_NC = out_BHWC.reshape(-1, 4)
    on_manifold_NC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(out_NC, 1.0)

    # All points should be on the manifold
    checks = jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(on_manifold_NC, 1.0)
    assert jnp.all(checks)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_gradient(dtype):
    """Finite gradients for mean and var params."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = _make_tangent_input(key, (2, 3, 3, 4), dtype)

    def loss_fn(bn):
        out = bn(x_BHWC, c=1.0)
        return jnp.sum(out**2)

    loss, grads = nnx.value_and_grad(loss_fn)(bn)
    assert jnp.isfinite(loss)

    # Check mean and var gradients are finite
    mean_grad = grads.mean[...]
    var_grad = grads.var[...]
    assert jnp.all(jnp.isfinite(mean_grad)), f"mean grad not finite: {mean_grad}"
    assert jnp.all(jnp.isfinite(var_grad)), f"var grad not finite: {var_grad}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_jitted(dtype):
    """Works under @nnx.jit."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = _make_tangent_input(key, (2, 3, 3, 4), dtype)

    @nnx.jit
    def forward(bn, x):
        return bn(x, c=1.0)

    out = forward(bn, x_BHWC)
    assert out.shape == x_BHWC.shape
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_different_curvatures(c, dtype):
    """Works with different curvature values."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    # Smaller tangent vectors for higher curvature (smaller ball)
    scale = 0.05 / jnp.sqrt(c)
    x_BHWC = jax.random.normal(key, (2, 3, 3, 4), dtype=dtype) * scale

    out = bn(x_BHWC, c=c)
    assert out.shape == x_BHWC.shape
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_running_stats_update(dtype):
    """Running stats change during training, stay fixed during eval."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = _make_tangent_input(key, (2, 3, 3, 4), dtype)

    # Save initial running stats
    init_running_mean = bn.running_mean[...].copy()
    init_running_var = bn.running_var[...].copy()

    # Training forward pass — should update running stats
    _ = bn(x_BHWC, c=1.0, use_running_average=False)

    assert not jnp.allclose(bn.running_mean[...], init_running_mean), "Running mean should have changed after training step"
    assert not jnp.allclose(bn.running_var[...], init_running_var), "Running var should have changed after training step"

    # Save updated running stats
    updated_running_mean = bn.running_mean[...].copy()
    updated_running_var = bn.running_var[...].copy()

    # Eval forward pass — should NOT update running stats
    _ = bn(x_BHWC, c=1.0, use_running_average=True)

    assert jnp.array_equal(bn.running_mean[...], updated_running_mean)
    assert jnp.array_equal(bn.running_var[...], updated_running_var)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_train_vs_eval(dtype):
    """Different outputs in train vs eval mode."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = _make_tangent_input(key, (2, 3, 3, 4), dtype)

    # Run a few training steps to update running stats
    for i in range(3):
        k = jax.random.PRNGKey(i + 10)
        x = _make_tangent_input(k, (2, 3, 3, 4), dtype)
        _ = bn(x, c=1.0, use_running_average=False)

    # Compare train vs eval output on same input
    out_train = bn(x_BHWC, c=1.0, use_running_average=False)
    out_eval = bn(x_BHWC, c=1.0, use_running_average=True)

    assert not jnp.allclose(out_train, out_eval), "Train and eval outputs should differ"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_chained_with_conv(dtype):
    """conv → bn → relu → conv → bn → relu chain produces finite output and gradients."""
    from hyperbolix.nn_layers.poincare_conv import HypConv2DPoincare

    manifold = Poincare(dtype=dtype)
    c = 0.5

    conv1 = HypConv2DPoincare(
        manifold,
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        rngs=nnx.Rngs(0),
        padding="SAME",
        input_space="tangent",
    )
    bn1 = PoincareBatchNorm2D(manifold, num_features=4)

    conv2 = HypConv2DPoincare(
        manifold,
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        rngs=nnx.Rngs(1),
        padding="SAME",
        input_space="tangent",
    )
    bn2 = PoincareBatchNorm2D(manifold, num_features=4)

    key = jax.random.PRNGKey(42)
    x_BHWC = _make_tangent_input(key, (2, 4, 4, 4), dtype)

    def forward(conv1, bn1, conv2, bn2, x):
        h = conv1(x, c=c)  # tangent → tangent
        h = bn1(h, c=c)  # tangent → tangent
        h = jax.nn.relu(h)  # tangent activation
        h = conv2(h, c=c)  # tangent → tangent
        h = bn2(h, c=c)  # tangent → tangent
        h = jax.nn.relu(h)  # tangent activation
        return h

    out = forward(conv1, bn1, conv2, bn2, x_BHWC)
    assert out.shape == x_BHWC.shape
    assert jnp.all(jnp.isfinite(out)), "Output has non-finite values"

    # Check gradients through the full chain
    def loss_fn(conv1, bn1, conv2, bn2):
        out = forward(conv1, bn1, conv2, bn2, x_BHWC)
        return jnp.sum(out**2)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(conv1, bn1, conv2, bn2)
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"

    # Check bn gradient params are finite
    bn1_mean_grad = grads[1].mean[...]
    bn1_var_grad = grads[1].var[...]
    assert jnp.all(jnp.isfinite(bn1_mean_grad)), "bn1 mean grad not finite"
    assert jnp.all(jnp.isfinite(bn1_var_grad)), "bn1 var grad not finite"
