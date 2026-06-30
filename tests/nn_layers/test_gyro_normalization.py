"""Tests for gyrogroup normalization layers (GyroBatchNorm + radial GyroRMSNorm).

Covers both manifolds (Hyperboloid, Proper Velocity) and both families:

- GyroBatchNorm: shape/flex-shape, on-manifold output, running-stat updates,
  train-vs-eval, finite gradients, jit, and the degenerate identical-batch case.
- Gyro radial RMSNorm: shape, on-manifold output, the radius-normalization
  property (the core correctness check), per-sample batch independence, absence of
  batch state, gradients, jit, and the origin-input edge case.

Dimension key:
  B: batch size     N: flattened batch     D: spatial feature dim
  F: input feature dim (ambient: D+1 Hyperboloid, D PV)
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid, ProperVelocity
from hyperbolix.nn_layers import (
    HyperboloidGyroBatchNorm,
    HyperboloidGyroRMSNorm,
    ProperVelocityGyroBatchNorm,
    ProperVelocityGyroRMSNorm,
)

# ============================================================================
# Per-manifold configuration + input helpers
# ============================================================================


def _hyp_points(key, shape, c, dtype):
    """Random hyperboloid points: expmap_0 of small spatial tangent vectors.

    ``shape`` is the *spatial* shape ``(..., D)``; output is ``(..., D+1)``.
    """
    m = Hyperboloid(dtype=dtype)
    v_D = jax.random.normal(key, shape, dtype=dtype) * 0.2
    v_amb = m.embed_spatial_0(v_D)
    flat = v_amb.reshape(-1, v_amb.shape[-1])
    pts = jax.vmap(m.expmap_0, in_axes=(0, None))(flat, c)
    return pts.reshape(*shape[:-1], shape[-1] + 1)


def _pv_points(key, shape, c, dtype):
    """Random PV points: unconstrained R^D (already on the manifold)."""
    return jax.random.normal(key, shape, dtype=dtype) * 0.2


CONFIGS = {
    "hyperboloid": dict(
        make=lambda dt: Hyperboloid(dtype=dt),
        bn=HyperboloidGyroBatchNorm,
        rms=HyperboloidGyroRMSNorm,
        points=_hyp_points,
        time=1,  # ambient = D + time
    ),
    "pv": dict(
        make=lambda dt: ProperVelocity(dtype=dt),
        bn=ProperVelocityGyroBatchNorm,
        rms=ProperVelocityGyroRMSNorm,
        points=_pv_points,
        time=0,
    ),
}


@pytest.fixture(params=["hyperboloid", "pv"])
def cfg(request):
    return CONFIGS[request.param]


DTYPES = [jnp.float32, jnp.float64]
DIMS = [2, 5, 10, 15]
DIM_IDS = [f"dim{d}" for d in DIMS]


def _tol(dtype):
    return (4e-3, 4e-3) if dtype == jnp.float32 else (1e-7, 1e-7)


# ============================================================================
# GyroBatchNorm
# ============================================================================


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", DIMS, ids=DIM_IDS)
def test_bn_shape_and_on_manifold(cfg, dim, dtype):
    """Output preserves shape and stays on the manifold."""
    manifold = cfg["make"](dtype)
    bn = cfg["bn"](manifold, num_features=dim)
    x = cfg["points"](jax.random.PRNGKey(0), (16, dim), 1.0, dtype)

    out = bn(x, c=1.0)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))
    checks = jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(out, 1.0)
    assert jnp.all(checks)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(8, 5), (4, 3, 5), (2, 3, 3, 5)], ids=["2d", "3d", "4d"])
def test_bn_flexible_leading_dims(cfg, shape, dtype):
    """Works for (B,F), (B,L,F) and (B,H,W,F) inputs; shape preserved, on-manifold."""
    manifold = cfg["make"](dtype)
    D = shape[-1]
    bn = cfg["bn"](manifold, num_features=D)
    x = cfg["points"](jax.random.PRNGKey(1), shape, 1.0, dtype)

    out = bn(x, c=1.0)
    assert out.shape == x.shape
    flat = out.reshape(-1, out.shape[-1])
    assert jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(flat, 1.0))


@pytest.mark.parametrize("dtype", DTYPES)
def test_bn_running_stats_update(cfg, dtype):
    """Running stats change in train mode, are frozen in eval mode."""
    manifold = cfg["make"](dtype)
    bn = cfg["bn"](manifold, num_features=6)
    x = cfg["points"](jax.random.PRNGKey(2), (16, 6), 1.0, dtype)

    init_mean = bn.running_mean[...].copy()
    init_var = bn.running_var[...].copy()

    _ = bn(x, c=1.0, use_running_average=False)
    assert not jnp.allclose(bn.running_mean[...], init_mean)
    assert not jnp.allclose(bn.running_var[...], init_var)

    frozen_mean = bn.running_mean[...].copy()
    frozen_var = bn.running_var[...].copy()
    _ = bn(x, c=1.0, use_running_average=True)
    assert jnp.array_equal(bn.running_mean[...], frozen_mean)
    assert jnp.array_equal(bn.running_var[...], frozen_var)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bn_train_vs_eval(cfg, dtype):
    """Train and eval outputs differ after running stats diverge from the batch."""
    manifold = cfg["make"](dtype)
    bn = cfg["bn"](manifold, num_features=6)
    for i in range(3):
        x_i = cfg["points"](jax.random.PRNGKey(10 + i), (16, 6), 1.0, dtype)
        _ = bn(x_i, c=1.0, use_running_average=False)

    x = cfg["points"](jax.random.PRNGKey(99), (16, 6), 1.0, dtype)
    out_train = bn(x, c=1.0, use_running_average=False)
    out_eval = bn(x, c=1.0, use_running_average=True)
    assert not jnp.allclose(out_train, out_eval)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bn_gradients(cfg, dtype):
    """Finite gradients for the affine parameters (bias, gamma)."""
    manifold = cfg["make"](dtype)
    bn = cfg["bn"](manifold, num_features=6)
    x = cfg["points"](jax.random.PRNGKey(3), (16, 6), 1.0, dtype)

    def loss_fn(bn):
        return jnp.sum(bn(x, c=1.0) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(bn)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grads.bias[...]))
    assert jnp.isfinite(grads.gamma[...])


@pytest.mark.parametrize("dtype", DTYPES)
def test_bn_jitted(cfg, dtype):
    """Forward runs cleanly under nnx.jit (BatchStat mutation included)."""
    manifold = cfg["make"](dtype)
    bn = cfg["bn"](manifold, num_features=6)
    x = cfg["points"](jax.random.PRNGKey(4), (16, 6), 1.0, dtype)

    @nnx.jit
    def forward(bn, x):
        return bn(x, c=1.0)

    out = forward(bn, x)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", DTYPES)
def test_bn_degenerate_identical_batch(cfg, dtype):
    """Identical points (var == 0) must not produce NaN and must stay on-manifold.

    Stresses the var->0 path: the scale factor ``gamma / sqrt(var + eps)`` saturates
    at ``1 / sqrt(eps)``, but the origin is a fixed point of ``scalar_mul`` and the
    output is re-projected, so the result is a finite, valid manifold point (no NaN /
    no escape off the manifold). Note: unlike Euclidean BN, the gyro centering does
    not cancel to the origin in exact float, so the output is not literally the bias
    point — only finiteness and on-manifold validity are guaranteed here.
    """
    manifold = cfg["make"](dtype)
    D = 5
    bn = cfg["bn"](manifold, num_features=D)
    bn.bias[...] = jnp.full((D,), 0.1, dtype=bn.bias[...].dtype)

    single = cfg["points"](jax.random.PRNGKey(5), (D,), 1.0, dtype)
    x = jnp.broadcast_to(single, (12, single.shape[-1]))
    out = bn(x, c=1.0)

    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(out, 1.0))


# ============================================================================
# Gyro radial RMSNorm
# ============================================================================


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", DIMS, ids=DIM_IDS)
def test_rms_shape_and_on_manifold(cfg, dim, dtype):
    """Output preserves shape and stays on the manifold."""
    manifold = cfg["make"](dtype)
    rms = cfg["rms"](manifold, num_features=dim)
    x = cfg["points"](jax.random.PRNGKey(0), (16, dim), 1.0, dtype)

    out = rms(x, c=1.0)
    assert out.shape == x.shape
    assert jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(out, 1.0))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", DIMS, ids=DIM_IDS)
@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_rms_radius_normalization(cfg, dim, gamma, dtype):
    """Core property: ``scalar_mul`` sends each radius to ``gamma * r / (r + eps)``.

    The eps regularizer makes the target ``gamma * r / (r + eps)`` rather than exactly
    ``gamma`` (a ~gamma*eps/r shortfall, negligible for r >> eps). Asserting against
    the exact target validates the radial-scaling math to full f64 precision.
    """
    manifold = cfg["make"](dtype)
    rms = cfg["rms"](manifold, num_features=dim)
    rms.gamma[...] = jnp.asarray(gamma, dtype=rms.gamma[...].dtype)
    x = cfg["points"](jax.random.PRNGKey(7), (16, dim), 1.0, dtype)

    out = rms(x, c=1.0)
    r_in = jax.vmap(manifold.dist_0, in_axes=(0, None))(x, 1.0)
    radii = jax.vmap(manifold.dist_0, in_axes=(0, None))(out, 1.0)
    expected = gamma * r_in / (r_in + rms.eps)
    atol, _ = _tol(dtype)
    assert jnp.allclose(radii, expected, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_batch_independence(cfg, dtype):
    """A point's output is identical whether normalized alone or inside a batch."""
    manifold = cfg["make"](dtype)
    rms = cfg["rms"](manifold, num_features=6)
    batch = cfg["points"](jax.random.PRNGKey(8), (10, 6), 1.0, dtype)

    out_batch = rms(batch, c=1.0)
    out_single = rms(batch[3:4], c=1.0)
    atol, _ = _tol(dtype)
    assert jnp.allclose(out_single[0], out_batch[3], atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_no_batch_state(cfg, dtype):
    """RMSNorm holds no running statistics (no nnx.BatchStat leaves)."""
    manifold = cfg["make"](dtype)
    rms = cfg["rms"](manifold, num_features=6)
    batch_stats = nnx.state(rms, nnx.BatchStat)
    assert len(jax.tree.leaves(batch_stats)) == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_use_bias(cfg, dtype):
    """use_bias=True applies a learned gyro-bias and keeps gradients finite."""
    manifold = cfg["make"](dtype)
    rms = cfg["rms"](manifold, num_features=6, use_bias=True)
    rms.bias[...] = jnp.full((6,), 0.1, dtype=rms.bias[...].dtype)
    x = cfg["points"](jax.random.PRNGKey(9), (8, 6), 1.0, dtype)

    rms_nobias = cfg["rms"](manifold, num_features=6, use_bias=False)
    out_bias = rms(x, c=1.0)
    out_nobias = rms_nobias(x, c=1.0)
    assert out_bias.shape == x.shape
    assert not jnp.allclose(out_bias, out_nobias)  # bias shifts the output

    def loss_fn(rms):
        return jnp.sum(rms(x, c=1.0) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(rms)
    assert jnp.all(jnp.isfinite(grads.bias[...]))
    assert jnp.isfinite(grads.gamma[...])


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_gradients_and_jit(cfg, dtype):
    """Finite gradient for gamma and a clean jitted forward."""
    manifold = cfg["make"](dtype)
    rms = cfg["rms"](manifold, num_features=6)
    x = cfg["points"](jax.random.PRNGKey(11), (8, 6), 1.0, dtype)

    def loss_fn(rms):
        return jnp.sum(rms(x, c=1.0) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(rms)
    assert jnp.isfinite(grads.gamma[...])

    @nnx.jit
    def forward(rms, x):
        return rms(x, c=1.0)

    out = forward(rms, x)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_origin_input_stays_finite(cfg, dtype):
    """A point at the origin (radius 0) stays at the origin — no gamma/eps blow-up."""
    manifold = cfg["make"](dtype)
    rms = cfg["rms"](manifold, num_features=6)
    origin = manifold.create_origin(1.0, 6)
    x = jnp.broadcast_to(origin, (4, origin.shape[-1]))

    out = rms(x, c=1.0)
    assert jnp.all(jnp.isfinite(out))
    atol, _ = _tol(dtype)
    assert jnp.allclose(out, jnp.broadcast_to(origin, out.shape), atol=atol)


# ============================================================================
# End-to-end composition in a real network
# ============================================================================


@pytest.mark.parametrize("dtype", DTYPES)
def test_end_to_end_hyperboloid(dtype):
    """HTCLinear -> GyroBatchNorm -> GyroRMSNorm -> HypRegression: finite loss/grads."""
    from hyperbolix.nn_layers import HTCLinear, HypRegressionHyperboloid

    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    rngs = nnx.Rngs(0)
    fc = HTCLinear(7, 6, rngs=rngs)  # in ambient 7 (D=6), out spatial 6 -> ambient 7
    bn = HyperboloidGyroBatchNorm(manifold, num_features=6)
    rms = HyperboloidGyroRMSNorm(manifold, num_features=6)
    head = HypRegressionHyperboloid(manifold, 7, 3, rngs=rngs)
    x = _hyp_points(jax.random.PRNGKey(1), (8, 6), c, dtype)  # ambient (8, 7)

    def loss_fn(fc, bn, rms, head):
        h = fc(x, c_in=c, c_out=c)
        h = bn(h, c=c)
        h = rms(h, c=c)
        return jnp.sum(head(h, c) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(fc, bn, rms, head)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grads[1].gamma[...]))
    assert jnp.all(jnp.isfinite(grads[2].gamma[...]))


@pytest.mark.parametrize("dtype", DTYPES)
def test_end_to_end_pv(dtype):
    """HypLinearPV -> GyroBatchNorm -> GyroRMSNorm -> HypRegressionPV: finite loss/grads."""
    from hyperbolix.nn_layers import HypLinearPV, HypRegressionPV

    manifold = ProperVelocity(dtype=dtype)
    c = 1.0
    rngs = nnx.Rngs(0)
    fc = HypLinearPV(manifold, 6, 6, rngs=rngs)
    bn = ProperVelocityGyroBatchNorm(manifold, num_features=6)
    rms = ProperVelocityGyroRMSNorm(manifold, num_features=6)
    head = HypRegressionPV(manifold, 6, 3, rngs=rngs)
    x = _pv_points(jax.random.PRNGKey(1), (8, 6), c, dtype)

    def loss_fn(fc, bn, rms, head):
        h = fc(x, c)
        h = bn(h, c=c)
        h = rms(h, c=c)
        return jnp.sum(head(h, c) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(fc, bn, rms, head)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grads[1].gamma[...]))
    assert jnp.all(jnp.isfinite(grads[2].gamma[...]))
