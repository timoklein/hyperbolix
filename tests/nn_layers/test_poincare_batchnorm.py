"""Tests for Poincaré batch normalization layer and helpers.

Dimension key:
  B: batch size     H: height     W: width
  C: channels       N: flattened (B*H*W)

The ``make_poincare_points`` / ``make_tangent_input`` factories come from
``tests/nn_layers/conftest.py`` (shared with ``test_poincare_conv.py``).
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
# Midpoint Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_midpoint_on_manifold(make_poincare_points, dtype):
    """Midpoint lies on the Poincaré ball."""
    key = jax.random.PRNGKey(42)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    x_NC = make_poincare_points(key, (16, 8), c, dtype)

    mid_C = poincare_midpoint(x_NC, manifold, c)

    assert mid_C.shape == (8,)
    assert manifold.is_in_manifold(mid_C, c, atol=1e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_midpoint_identical_points(make_poincare_points, dtype):
    """Midpoint of N copies of the same point equals that point."""
    key = jax.random.PRNGKey(0)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    single = make_poincare_points(key, (4,), c, dtype)  # (C,)
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
def test_poincare_midpoint_gradients(make_poincare_points, dtype):
    """Gradients through midpoint are finite."""
    key = jax.random.PRNGKey(1)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    x_NC = make_poincare_points(key, (8, 4), c, dtype)

    def loss_fn(x):
        mid = poincare_midpoint(x, manifold, c)
        return jnp.sum(mid**2)

    grads = jax.grad(loss_fn)(x_NC)
    assert jnp.all(jnp.isfinite(grads))


# ============================================================================
# Variance Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0], ids=["c0.5", "c1.0", "c2.0"])
def test_frechet_variance_two_symmetric_points(dtype, c):
    """Two antipodal points at geodesic distance d from the origin → var == d².

    Closed form, no library call on the reference side: ``dist_0(expmap_0(v)) =
    2‖v‖`` on this ball (the λ(0) = 2 conformal factor is inside the map), so
    ``x = {p, -p}`` with ``p = expmap_0(v)`` sits at ``d = 2‖v‖`` from the
    origin and the Fréchet variance about the origin is exactly ``d²`` — for
    every curvature, since the radius is set by the tangent vector.

    The absolute-value anchor the variance never had: the previous tests were
    ``var >= 0`` (true of the constant 0) and ``var(identical points) < 1e-6``
    (also true of the constant 0), so ``frechet_variance ≡ 0`` passed both.
    """
    manifold = Poincare(dtype=dtype)
    v_C = jnp.array([0.45, 0.0, 0.0], dtype=dtype)
    d = 2.0 * float(jnp.linalg.norm(v_C))

    p_C = manifold.expmap_0(v_C, c)
    x_NC = jnp.stack([p_C, -p_C])
    origin_C = jnp.zeros((3,), dtype=dtype)

    var = frechet_variance(x_NC, origin_C, manifold, c)

    atol = 4e-3 if dtype == jnp.float32 else 1e-10
    assert jnp.allclose(var, d**2, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_frechet_variance_identical_points(make_poincare_points, dtype):
    """Variance of identical points is zero."""
    key = jax.random.PRNGKey(0)
    c = 1.0
    manifold = Poincare(dtype=dtype)
    single = make_poincare_points(key, (4,), c, dtype)
    repeated_NC = jnp.tile(single, (10, 1))

    var = frechet_variance(repeated_NC, single, manifold, c)
    assert var < 1e-6


# ============================================================================
# BatchNorm Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_output_shape(make_tangent_input, dtype):
    """Output shape matches input shape (B, H, W, C)."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=8)
    key = jax.random.PRNGKey(42)
    x_BHWC = make_tangent_input(key, (2, 4, 4, 8), dtype)

    out = bn(x_BHWC, c=1.0)
    assert out.shape == (2, 4, 4, 8)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_output_finite_and_jit_matches_eager(make_tangent_input, dtype):
    """No NaN/Inf, and ``nnx.jit`` reproduces the eager output exactly.

    The jit leg is folded in here (was a separate shape-only ``jitted`` test):
    two freshly built layers with identical parameters are used so the
    running-statistic mutation in the first call cannot make the comparison
    trivially unequal.
    """
    manifold = Poincare(dtype=dtype)
    key = jax.random.PRNGKey(42)
    x_BHWC = make_tangent_input(key, (2, 4, 4, 8), dtype)

    bn_eager = PoincareBatchNorm2D(manifold, num_features=8)
    bn_jit = PoincareBatchNorm2D(manifold, num_features=8)

    @nnx.jit
    def forward(bn, x):
        return bn(x, c=1.0)

    out_eager = bn_eager(x_BHWC, c=1.0)
    out_jit = forward(bn_jit, x_BHWC)

    assert jnp.all(jnp.isfinite(out_eager))
    assert jnp.allclose(out_jit, out_eager, atol=1e-6 if dtype == jnp.float32 else 1e-12)
    # The BatchStat mutation must also survive the jit boundary.
    assert jnp.allclose(bn_jit.running_var[...], bn_eager.running_var[...], atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_poincare_bn_rescales_geodesic_radii_about_learned_mean(dtype):
    """``d(out_i, learned_mean) == scale · d(x_i, batch_midpoint)`` exactly.

    The layer's own algorithm (steps 4-7) is a logmap at the batch midpoint, a
    parallel transport to the learned mean, a radial rescale by
    ``sqrt(var / (batch_var + eps))`` and an expmap at the learned mean. Parallel
    transport is a linear isometry and ``expmap`` is a radial isometry, so the
    geodesic radius about the learned mean is *exactly* the scaled input radius
    about the batch midpoint — an absolute oracle for the whole pipeline.

    Regression guard: every other test in this file (shape, finiteness,
    on-manifold after ``expmap_0``, finite gradients, running-stat mutation)
    passes when ``__call__`` returns its input unchanged. ``batch_var`` is
    recomputed here from ``manifold.dist`` rather than via ``frechet_variance``,
    so a ``frechet_variance ≡ 0`` mutation also fails this test.
    """
    c = 0.8
    C = 4
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=C, param_dtype=dtype)
    bn.mean[...] = jnp.array([0.2, -0.1, 0.05, 0.3], dtype=dtype)
    bn.var[...] = jnp.asarray(0.25, dtype=dtype)

    x_BHWC = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 3, C), dtype=dtype) * 0.4

    out_BHWC = bn(x_BHWC, c=c, use_running_average=False)

    x_NC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(x_BHWC.reshape(-1, C), c)
    midpoint_C = poincare_midpoint(x_NC, manifold, c)
    d_in_N = jax.vmap(manifold.dist, in_axes=(0, None, None))(x_NC, midpoint_C, c)
    batch_var = jnp.mean(d_in_N**2)
    scale = jnp.sqrt(bn.var[...] / (batch_var + bn.eps))

    learned_mean_C = manifold.expmap_0(bn.mean[...], c)
    out_NC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(out_BHWC.reshape(-1, C), c)
    d_out_N = jax.vmap(manifold.dist, in_axes=(0, None, None))(out_NC, learned_mean_C, c)

    atol = 4e-3 if dtype == jnp.float32 else 1e-9
    assert jnp.allclose(d_out_N, scale * d_in_N, atol=atol)
    # Non-vacuity: this batch is genuinely transformed (rules out a no-op layer).
    assert not jnp.allclose(out_BHWC, x_BHWC, atol=1e-3)
    assert abs(float(scale) - 1.0) > 0.2  # the rescale is a real change of radius, not ≈1


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_output_mappable_to_manifold(make_tangent_input, dtype):
    """Tangent output maps to valid manifold points via expmap_0."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = make_tangent_input(key, (2, 3, 3, 4), dtype)

    out_BHWC = bn(x_BHWC, c=1.0)
    # Flatten and map to manifold
    out_NC = out_BHWC.reshape(-1, 4)
    on_manifold_NC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(out_NC, 1.0)

    # All points should be on the manifold
    checks = jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(on_manifold_NC, 1.0)
    assert jnp.all(checks)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_gradient(make_tangent_input, dtype):
    """Finite gradients for mean and var params."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = make_tangent_input(key, (2, 3, 3, 4), dtype)

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
def test_poincare_bn_running_stats_update(make_tangent_input, dtype):
    """Running stats change during training, stay fixed during eval."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = make_tangent_input(key, (2, 3, 3, 4), dtype)

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
def test_poincare_bn_train_vs_eval(make_tangent_input, dtype):
    """Different outputs in train vs eval mode."""
    manifold = Poincare(dtype=dtype)
    bn = PoincareBatchNorm2D(manifold, num_features=4)
    key = jax.random.PRNGKey(42)
    x_BHWC = make_tangent_input(key, (2, 3, 3, 4), dtype)

    # Run a few training steps to update running stats
    for i in range(3):
        k = jax.random.PRNGKey(i + 10)
        x = make_tangent_input(k, (2, 3, 3, 4), dtype)
        _ = bn(x, c=1.0, use_running_average=False)

    # Compare train vs eval output on same input
    out_train = bn(x_BHWC, c=1.0, use_running_average=False)
    out_eval = bn(x_BHWC, c=1.0, use_running_average=True)

    assert not jnp.allclose(out_train, out_eval), "Train and eval outputs should differ"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_bn_chained_with_conv(make_tangent_input, dtype):
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
    x_BHWC = make_tangent_input(key, (2, 4, 4, 4), dtype)

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
