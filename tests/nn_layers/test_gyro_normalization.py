"""Tests for gyrogroup normalization layers (GyroBatchNorm + radial GyroRMSNorm).

Covers both families across their manifolds:

- GyroBatchNorm (Hyperboloid, Proper Velocity): shape/flex-shape, on-manifold
  output, running-stat updates, train-vs-eval, finite gradients, jit, and the
  degenerate identical-batch case. (Poincaré BatchNorm is the separate tangent-space
  ``PoincareBatchNorm2D``, tested in ``test_poincare_batchnorm.py``.)
- Gyro radial RMSNorm (Hyperboloid, Proper Velocity, Poincaré): shape, on-manifold
  output, the radius-normalization property (the core correctness check), per-sample
  batch independence, absence of batch state, gradients, jit, and the origin-input
  edge case.

Dimension key:
  B: batch size     N: flattened batch     D: spatial feature dim
  F: input feature dim (ambient: D+1 Hyperboloid, D PV/Poincaré)
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid, Poincare, ProperVelocity
from hyperbolix.nn_layers import (
    HyperboloidGyroBatchNorm,
    HyperboloidGyroRMSNorm,
    PoincareGyroRMSNorm,
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


def _poincare_points(key, shape, c, dtype):
    """Random Poincaré-ball points: expmap_0 of small spatial tangent vectors.

    The ball has no time coordinate, so ``shape`` is already the ambient shape ``(..., D)``.
    """
    m = Poincare(dtype=dtype)
    v_D = jax.random.normal(key, shape, dtype=dtype) * 0.2
    flat = v_D.reshape(-1, v_D.shape[-1])
    pts = jax.vmap(m.expmap_0, in_axes=(0, None))(flat, c)
    return pts.reshape(shape)


def _lorentz_centroid_ref(manifold, x_NF, c):
    """Uniform Lorentz centroid, transcribed from the definition (HELM).

    ``mu = s / (sqrt(c)·sqrt(-⟨s,s⟩_L))`` with ``s = Σ x_i`` — the ambient sum
    renormalized back onto the ``⟨mu,mu⟩_L = -1/c`` sheet. Independent of
    ``lorentz_midpoint``.
    """
    s_F = jnp.sum(x_NF, axis=0)
    minkowski_sq = -(s_F[0] ** 2) + jnp.sum(s_F[1:] ** 2)
    return s_F / (jnp.sqrt(c) * jnp.sqrt(-minkowski_sq))


def _log_euclidean_mean_ref(manifold, x_NF, c):
    """``expmap_0(mean_i logmap_0(x_i))`` — the PV batch-mean estimator."""
    v_NF = jax.vmap(manifold.logmap_0, in_axes=(0, None))(x_NF, c)
    return manifold.expmap_0(jnp.mean(v_NF, axis=0), c)


CONFIGS = {
    "hyperboloid": dict(
        make=lambda dt: Hyperboloid(dtype=dt),
        bn=HyperboloidGyroBatchNorm,
        rms=HyperboloidGyroRMSNorm,
        points=_hyp_points,
        origin=lambda m, dim, dt: m.create_origin(1.0, dim),
        batch_mean=_lorentz_centroid_ref,
        time=1,  # ambient = D + time
    ),
    "pv": dict(
        make=lambda dt: ProperVelocity(dtype=dt),
        bn=ProperVelocityGyroBatchNorm,
        rms=ProperVelocityGyroRMSNorm,
        points=_pv_points,
        origin=lambda m, dim, dt: m.create_origin(1.0, dim),
        batch_mean=_log_euclidean_mean_ref,
        time=0,
    ),
    "poincare": dict(
        make=lambda dt: Poincare(dtype=dt),
        bn=None,  # Poincaré BatchNorm is the tangent-space PoincareBatchNorm2D (tested elsewhere)
        rms=PoincareGyroRMSNorm,
        points=_poincare_points,
        origin=lambda m, dim, dt: jnp.zeros((dim,), dtype=dt),  # ball origin is the zero vector
        time=0,
    ),
}


@pytest.fixture(params=["hyperboloid", "pv"])
def cfg(request):
    """BatchNorm-family manifolds (Hyperboloid, PV)."""
    return CONFIGS[request.param]


@pytest.fixture(params=["hyperboloid", "pv", "poincare"])
def cfg_rms(request):
    """RMSNorm-family manifolds (Hyperboloid, PV, Poincaré)."""
    return CONFIGS[request.param]


DTYPES = [jnp.float32, jnp.float64]
DIMS = [2, 5, 10, 15]
DIM_IDS = [f"dim{d}" for d in DIMS]
# RMSNorm's radial rescale is a scalar operation on the geodesic radius, so the
# feature width only needs a narrow and a wide representative.
RMS_DIMS = [2, 10]
RMS_DIM_IDS = [f"dim{d}" for d in RMS_DIMS]


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
def test_bn_gradients_and_jit_matches_eager(cfg, dtype):
    """Finite gradients for (bias, gamma), and ``nnx.jit`` reproduces eager values.

    The jit leg is folded in here (was a separate shape-only ``jitted`` test):
    two freshly built layers are compared so the BatchStat mutation in the first
    call cannot make the comparison trivially unequal.
    """
    manifold = cfg["make"](dtype)
    bn = cfg["bn"](manifold, num_features=6)
    x = cfg["points"](jax.random.PRNGKey(3), (16, 6), 1.0, dtype)

    def loss_fn(bn):
        return jnp.sum(bn(x, c=1.0) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(bn)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grads.bias[...]))
    assert jnp.isfinite(grads.gamma[...])

    bn_eager = cfg["bn"](manifold, num_features=6)
    bn_jit = cfg["bn"](manifold, num_features=6)

    @nnx.jit
    def forward(bn, x):
        return bn(x, c=1.0)

    out_eager = bn_eager(x, c=1.0)
    out_jit = forward(bn_jit, x)
    atol, _ = _tol(dtype)
    assert jnp.allclose(out_jit, out_eager, atol=atol)
    assert jnp.allclose(bn_jit.running_var[...], bn_eager.running_var[...], atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bn_centers_scales_and_biases(cfg, dtype):
    """``d(y_i, w) == (gamma/sqrt(var+eps)) · d(x_i, mu)`` — the whole GyroBN pipeline.

    ``y = w ⊕ ((gamma/sqrt(var+eps)) ⊗ ((⊖mu) ⊕ x))``. Gyro-translation by ``w`` is
    an isometry taking the origin to ``w``, and ``scalar_mul`` scales geodesic
    radius linearly, so the geodesic radius of the output about the bias point is
    *exactly* the scaled input radius about the batch mean. Same closed-form style
    as the RMSNorm sibling's radius oracle, and it pins all three GyroBN steps at
    once:

    * centering — the batch is deliberately gyro-translated off the origin, so
      dropping ``(⊖mu) ⊕ ·`` measures radii about the wrong point;
    * scaling — the factor is checked by value, not just for finiteness;
    * bias — the radii are measured about ``w``, not the origin.

    ``mu`` and ``var`` are recomputed here (Lorentz centroid / log-Euclidean mean
    transcribed in ``CONFIGS[...]["batch_mean"]``, variance straight from
    ``manifold.dist``), so a ``frechet_variance ≡ 0`` mutation also fails: the
    library would then use the ``min_var`` floor while the reference uses the
    true variance.
    """
    c = 1.0
    D = 6
    manifold = cfg["make"](dtype)
    bn = cfg["bn"](manifold, num_features=D, param_dtype=dtype)
    bn.bias[...] = jnp.array([0.2, -0.1, 0.05, 0.3, -0.25, 0.15], dtype=dtype)
    bn.gamma[...] = jnp.asarray(1.5, dtype=dtype)

    # Gyro-translate a random batch away from the origin so the batch mean is a
    # genuinely non-origin point (at the origin, centering would be a no-op).
    pts_NF = cfg["points"](jax.random.PRNGKey(12), (16, D), c, dtype)
    shift_F = manifold.expmap_0(manifold.embed_spatial_0(jnp.full((D,), 0.35, dtype=dtype)), c)
    x_NF = jax.vmap(manifold.addition, in_axes=(None, 0, None))(shift_F, pts_NF, c)

    out_NF = bn(x_NF, c=c, use_running_average=False)

    mu_F = cfg["batch_mean"](manifold, x_NF, c)
    assert float(manifold.dist_0(mu_F, c)) > 0.5  # centering is not a no-op here
    d_in_N = jax.vmap(manifold.dist, in_axes=(0, None, None))(x_NF, mu_F, c)
    var = jnp.maximum(jnp.mean(d_in_N**2), bn.min_var)
    factor = bn.gamma[...] / jnp.sqrt(var + bn.eps)

    bias_pt_F = manifold.expmap_0(manifold.embed_spatial_0(bn.bias[...]), c)
    d_out_N = jax.vmap(manifold.dist, in_axes=(0, None, None))(out_NF, bias_pt_F, c)

    atol, _ = _tol(dtype)
    assert jnp.allclose(d_out_N, factor * d_in_N, atol=atol)
    # Fréchet std about the bias point is gamma (up to the eps in the denominator).
    frechet_std = jnp.sqrt(jnp.mean(d_out_N**2))
    assert jnp.allclose(frechet_std, bn.gamma[...] * jnp.sqrt(var / (var + bn.eps)), atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bn_degenerate_identical_batch(cfg, dtype):
    """Identical points (var == 0) must not produce NaN and must stay on-manifold.

    Stresses the var->0 path: the scale factor ``gamma / sqrt(var + eps)`` is now
    floored at ``gamma / sqrt(min_var)`` (see ``test_bn_degenerate_batch_variance_floor``
    for the bound itself), but the origin is a fixed point of ``scalar_mul`` and the
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


@pytest.mark.parametrize("dtype", DTYPES)
def test_bn_degenerate_batch_variance_floor(cfg, dtype):
    """A collapsed (identical-point) batch floors Fréchet variance at min_var, bounding
    the scale factor (and gamma's gradient) by 1/sqrt(min_var) instead of saturating
    near 1/sqrt(eps) (eps=1e-6 -> ~1000x; min_var=1e-2 -> ~10x), and never lets a
    collapsed batch write a sub-floor value into running_var.
    """
    manifold = cfg["make"](dtype)
    D = 5
    min_var = 1e-2
    bn = cfg["bn"](manifold, num_features=D, min_var=min_var)

    single = cfg["points"](jax.random.PRNGKey(5), (D,), 1.0, dtype)
    x = jnp.broadcast_to(single, (12, single.shape[-1]))

    def loss_fn(bn):
        return jnp.sum(bn(x, c=1.0) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(bn)

    bound = float(jnp.abs(bn.gamma[...])) / jnp.sqrt(min_var) * 20  # generous margin
    assert jnp.abs(grads.gamma[...]) < bound
    assert bn.running_var[...] >= min_var


def test_bn_hyperboloid_float32_tracks_float64_at_scaled_radius_9():
    """Train-mode GyroBN in float32 stays accurate at scaled geodesic radius ``a = √c·d = 9``.

    Accuracy, not finiteness: on the pre-fix ``_addition`` this same configuration returns a
    perfectly finite, plausible batch of points that is 0.56 geodesic nats away from the right
    one (and NaN on other seeds), which is invisible to every other test in this file. Both legs
    are handed **bit-identical** inputs — the points are rounded to float32 and then widened for
    the float64 leg — so the only difference between them is the arithmetic precision.

    Where the radius bites: ``_addition(x, y)`` is the Lorentz boost ``Λ_x y``, and it is the
    *base point* ``x`` that sets the conditioning. GyroBN calls it twice, ``(⊖mu) ⊕ x`` and
    ``w ⊕ x_scaled``, so a far batch is not enough on its own — 32 points in random directions at
    ``a = 9`` have their Lorentz centroid back near the origin, and even the pre-fix spelling is
    accurate to 4e-7 on that. The learned gyro-bias is the base point that is genuinely far here:
    ``bias`` has spatial norm ``9/√c``, so ``w = expmap_0(bias)`` sits at ``a = 9`` and the output
    does too (measured ``a`` range 8.77 to 9.75). That is the case the ``_addition`` docstring calls
    the hot path of every gyro-bias in the library.

    Measured max geodesic error between the two legs: 7.8e-4 (bound 4e-3, ≈ 5.1x); the pre-fix
    spelling gives 0.56. Measured max relative error of the ``bias`` gradient: 1.5e-7 (bound 1e-5,
    ≈ 66x); the pre-fix spelling is 15% off.
    Evidence: ``logs/2026-09-08_hyperboloid_tangent_primitives/probe_final_configs.py``.
    """
    c, a, D, n_points = 0.5, 9.0, 16, 32
    hyp32, hyp64 = Hyperboloid(dtype=jnp.float32), Hyperboloid(dtype=jnp.float64)
    radius = a / jnp.sqrt(jnp.asarray(c, dtype=jnp.float64))

    dirs_ND = jax.random.normal(jax.random.PRNGKey(31), (n_points, D), dtype=jnp.float64)
    dirs_ND /= jnp.linalg.norm(dirs_ND, axis=-1, keepdims=True)
    x_NF = jax.vmap(hyp64.expmap_0, in_axes=(0, None))(hyp64.embed_spatial_0(radius * dirs_ND), c)
    x_NF_32 = jax.vmap(hyp32.proj, in_axes=(0, None))(x_NF.astype(jnp.float32), c)
    x_NF_64 = x_NF_32.astype(jnp.float64)  # same numbers, wider arithmetic

    radii_N = jax.vmap(hyp64.dist_0, in_axes=(0, None))(x_NF_64, c) * jnp.sqrt(c)
    assert jnp.allclose(radii_N, a, atol=1e-3), "inputs are not at the radius the test intends"

    bias_D = jax.random.normal(jax.random.PRNGKey(41), (D,), dtype=jnp.float64)
    bias_D = radius * bias_D / jnp.linalg.norm(bias_D)  # target mean point also at a = 9

    def build(dtype):
        bn = HyperboloidGyroBatchNorm(Hyperboloid(dtype=dtype), num_features=D, param_dtype=dtype)
        bn.bias[...] = bias_D.astype(dtype)
        bn.gamma[...] = jnp.asarray(1.3, dtype=dtype)
        return bn

    out_NF_32 = build(jnp.float32)(x_NF_32, c=c, use_running_average=False)
    out_NF_64 = build(jnp.float64)(x_NF_64, c=c, use_running_average=False)
    assert jnp.all(jnp.isfinite(out_NF_32))

    err_N = jax.vmap(hyp64.dist, in_axes=(0, 0, None))(out_NF_32.astype(jnp.float64), out_NF_64, c)
    assert float(jnp.max(err_N)) < 4e-3, f"float32 output is {float(jnp.max(err_N)):.3e} nats off the float64 one"

    # The gradient is the quantity that failed silently: a bias that stopped receiving a correct
    # one looks exactly like a converged bias. ``nnx.value_and_grad``, not the merged-module
    # ``jax.value_and_grad``, which raises on the running-statistic write.
    def loss_fn(bn, x):
        return jnp.sum(bn(x, c=c, use_running_average=False) ** 2)

    _, g32 = nnx.value_and_grad(lambda bn: loss_fn(bn, x_NF_32))(build(jnp.float32))
    _, g64 = nnx.value_and_grad(lambda bn: loss_fn(bn, x_NF_64))(build(jnp.float64))
    scale = jnp.max(jnp.abs(g64.bias[...]))
    rel = float(jnp.max(jnp.abs(g32.bias[...].astype(jnp.float64) - g64.bias[...])) / scale)
    assert rel < 1e-5, f"float32 bias gradient is {rel:.3e} relative off the float64 one"


def test_bn_hyperboloid_float32_tracks_float64_on_a_tight_cluster_at_scaled_radius_9():
    """The far batch of the test above, but angularly CLUSTERED — the centering's hard case.

    The test above spreads its 32 points over random directions at ``a = 9``, so their Lorentz
    centroid falls back near the origin and the centering ``(⊖mu) ⊕ x`` never cancels; what it
    stresses is the gyro-bias ``w ⊕ x``. Cluster the points instead and the mean lands *among*
    them: both operands of the centering sit at ``a = 9`` while the result is ``O(1)``, which is
    where the ambient Lorentz boost's three ``O(e^{2a})`` terms cancel identically and float32 has
    nothing left. ``HyperboloidGyroBatchNorm`` therefore centers with
    ``Hyperboloid.gyro_difference``, which reads the same result off the polar frame.

    Accuracy, not finiteness — the ambient spelling returns a perfectly plausible batch here. Both
    legs get **bit-identical** inputs (rounded to float32, then widened), so the only difference is
    the arithmetic precision.

    Angular spread is set from the target geodesic separation through
    ``cosh(sep) ≈ 1 + sinh²(a)·θ²/2``; this configuration lands at mean pairwise separation 1.84.
    Measured max geodesic error between the legs: 9.6e-4 with this centering against a bound of
    4e-3 (4.2x), where the ambient spelling gives 1.77 nats. The bias gradient goes the same way:
    7.1e-4 relative here against a bound of 1e-2 (14x), and 0.97 with the ambient centering. A
    sweep over the separation shows the gap widening as the cluster tightens — 2.14 vs 9.0e-4 nats
    at separation 1.80, 5.22 vs 4.0e-3 at 0.56 — and closing as it spreads, 1.5e-3 vs 1.8e-5 at
    11.7, which is why the shipped random-direction test above does not see it. Evidence:
    ``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_gyro_bn_centering.out`` (the sweep) and
    ``step2c_committed_configs.out`` (this configuration exactly).
    """
    c, a, D, n_points, target_sep = 0.5, 9.0, 16, 32, 1.0
    hyp32, hyp64 = Hyperboloid(dtype=jnp.float32), Hyperboloid(dtype=jnp.float64)
    radius = a / jnp.sqrt(jnp.asarray(c, dtype=jnp.float64))

    # theta = sqrt(2(cosh(sep) - 1))/sinh(a) is the angular spread giving that geodesic separation.
    theta = jnp.sqrt(2.0 * (jnp.cosh(jnp.asarray(target_sep, dtype=jnp.float64)) - 1.0)) / jnp.sinh(a)
    k_dir, k_jitter = jax.random.split(jax.random.PRNGKey(53))
    u_D = jax.random.normal(k_dir, (D,), dtype=jnp.float64)
    u_D /= jnp.linalg.norm(u_D)
    w_ND = u_D[None, :] + (theta / jnp.sqrt(D)) * jax.random.normal(k_jitter, (n_points, D), dtype=jnp.float64)
    w_ND /= jnp.linalg.norm(w_ND, axis=-1, keepdims=True)

    x_NF = jax.vmap(hyp64.expmap_0, in_axes=(0, None))(hyp64.embed_spatial_0(radius * w_ND), c)
    x_NF_32 = jax.vmap(hyp32.proj, in_axes=(0, None))(x_NF.astype(jnp.float32), c)
    x_NF_64 = x_NF_32.astype(jnp.float64)  # same numbers, wider arithmetic

    radii_N = jax.vmap(hyp64.dist_0, in_axes=(0, None))(x_NF_64, c) * jnp.sqrt(c)
    assert jnp.allclose(radii_N, a, atol=1e-3), "inputs are not at the radius the test intends"
    seps_NN = jax.vmap(lambda p: jax.vmap(hyp64.dist, in_axes=(0, None, None))(x_NF_64, p, c))(x_NF_64)
    assert 1.0 < float(jnp.mean(seps_NN)) < 3.0, "the batch is not the tight cluster the test intends"

    bias_D = 0.2 * jax.random.normal(jax.random.PRNGKey(61), (D,), dtype=jnp.float64)

    def build(dtype):
        bn = HyperboloidGyroBatchNorm(Hyperboloid(dtype=dtype), num_features=D, param_dtype=dtype)
        bn.bias[...] = bias_D.astype(dtype)
        bn.gamma[...] = jnp.asarray(1.3, dtype=dtype)
        return bn

    out_NF_32 = build(jnp.float32)(x_NF_32, c=c, use_running_average=False)
    out_NF_64 = build(jnp.float64)(x_NF_64, c=c, use_running_average=False)
    assert jnp.all(jnp.isfinite(out_NF_32))

    err_N = jax.vmap(hyp64.dist, in_axes=(0, 0, None))(out_NF_32.astype(jnp.float64), out_NF_64, c)
    assert float(jnp.max(err_N)) < 4e-3, f"float32 output is {float(jnp.max(err_N)):.3e} nats off the float64 one"

    def loss_fn(bn, x):
        return jnp.sum(bn(x, c=c, use_running_average=False) ** 2)

    _, g32 = nnx.value_and_grad(lambda bn: loss_fn(bn, x_NF_32))(build(jnp.float32))
    _, g64 = nnx.value_and_grad(lambda bn: loss_fn(bn, x_NF_64))(build(jnp.float64))
    rel = float(jnp.max(jnp.abs(g32.bias[...].astype(jnp.float64) - g64.bias[...])) / jnp.max(jnp.abs(g64.bias[...])))
    assert rel < 1e-2, f"float32 bias gradient is {rel:.3e} relative off the float64 one"


def test_bn_pv_float32_tracks_float64_on_a_tight_cluster_at_scaled_radius_9():
    """The PV twin of the hyperboloid tight-cluster test above -- the centering's hard case.

    PV gyroaddition *is* the spatial part of the Lorentz boost (see
    ``ProperVelocity._addition``), so a clustered far batch puts
    ``ProperVelocityGyroBatchNorm``'s centering in exactly the regime the hyperboloid one is in:
    the log-Euclidean batch mean lands *among* the points, both operands sit at ``a = 9`` and the
    result is ``O(1)``, where the boost's three ``O(e^{2a})`` terms cancel identically.
    ``ProperVelocityGyroBatchNorm`` therefore centres with
    :meth:`ProperVelocity.gyro_difference`.

    Accuracy, not finiteness -- the ambient spelling returns a perfectly plausible batch here.
    Both legs get **bit-identical** inputs (rounded to float32, then widened), so the only
    difference is the arithmetic precision.

    Measured on this configuration (mean pairwise separation 1.84): max geodesic error between the
    legs 6.3e-4 on the CPU / 5.7e-4 on an A100, against a bound of 2.6e-3 (4.1x); max relative
    error of ``∂loss/∂bias`` 4.9e-4 / 4.7e-4 against a bound of 2e-3 (4.0x)
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_new_test_measurements{,_gpu}.out``,
    section 4). With the base class's ambient centering the same configuration gives 2.75 nats and
    a bias gradient 6.7 relative off at ``a = 9``, and 10.9 nats / 2.9e+4 at ``a = 12``; the
    spread batch (random directions, mean back near the origin) is 3.4e-7 either way, which is why
    a far batch alone does not see this
    (``step2c_pv_gyro_difference_accuracy.out``, section B).
    """
    c, a, D, n_points, target_sep = 0.5, 9.0, 16, 32, 1.0
    pv64 = ProperVelocity(dtype=jnp.float64)

    # theta = sqrt(2(cosh(sep) - 1))/sinh(a) is the angular spread giving that geodesic separation.
    theta = jnp.sqrt(2.0 * (jnp.cosh(jnp.asarray(target_sep, dtype=jnp.float64)) - 1.0)) / jnp.sinh(a)
    k_dir, k_jitter = jax.random.split(jax.random.PRNGKey(53))
    u_D = jax.random.normal(k_dir, (D,), dtype=jnp.float64)
    u_D /= jnp.linalg.norm(u_D)
    w_ND = u_D[None, :] + (theta / jnp.sqrt(D)) * jax.random.normal(k_jitter, (n_points, D), dtype=jnp.float64)
    w_ND /= jnp.linalg.norm(w_ND, axis=-1, keepdims=True)

    # PV coordinates: scaled radius a means ||x|| = sinh(a)/sqrt(c).
    x_ND = (jnp.sinh(jnp.asarray(a, dtype=jnp.float64)) / jnp.sqrt(c)) * w_ND
    x_ND_32 = x_ND.astype(jnp.float32)
    x_ND_64 = x_ND_32.astype(jnp.float64)  # same numbers, wider arithmetic

    radii_N = jax.vmap(pv64.dist_0, in_axes=(0, None))(x_ND_64, c) * jnp.sqrt(c)
    assert jnp.allclose(radii_N, a, atol=1e-3), "inputs are not at the radius the test intends"
    seps_NN = jax.vmap(lambda p: jax.vmap(pv64.dist, in_axes=(0, None, None))(x_ND_64, p, c))(x_ND_64)
    assert 1.0 < float(jnp.mean(seps_NN)) < 3.0, "the batch is not the tight cluster the test intends"

    bias_D = 0.2 * jax.random.normal(jax.random.PRNGKey(61), (D,), dtype=jnp.float64)

    def build(dtype):
        bn = ProperVelocityGyroBatchNorm(ProperVelocity(dtype=dtype), num_features=D, param_dtype=dtype)
        bn.bias[...] = bias_D.astype(dtype)
        bn.gamma[...] = jnp.asarray(1.3, dtype=dtype)
        return bn

    out_ND_32 = build(jnp.float32)(x_ND_32, c=c, use_running_average=False)
    out_ND_64 = build(jnp.float64)(x_ND_64, c=c, use_running_average=False)
    assert jnp.all(jnp.isfinite(out_ND_32))

    err_N = jax.vmap(pv64.dist, in_axes=(0, 0, None))(out_ND_32.astype(jnp.float64), out_ND_64, c)
    assert float(jnp.max(err_N)) < 2.6e-3, f"float32 output is {float(jnp.max(err_N)):.3e} nats off the float64 one"

    def loss_fn(bn, x):
        return jnp.sum(bn(x, c=c, use_running_average=False) ** 2)

    _, g32 = nnx.value_and_grad(lambda bn: loss_fn(bn, x_ND_32))(build(jnp.float32))
    _, g64 = nnx.value_and_grad(lambda bn: loss_fn(bn, x_ND_64))(build(jnp.float64))
    rel = float(jnp.max(jnp.abs(g32.bias[...].astype(jnp.float64) - g64.bias[...])) / jnp.max(jnp.abs(g64.bias[...])))
    assert rel < 2e-3, f"float32 bias gradient is {rel:.3e} relative off the float64 one"


# ============================================================================
# Gyro radial RMSNorm
# ============================================================================


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", RMS_DIMS, ids=RMS_DIM_IDS)
@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_rms_radius_normalization(cfg_rms, dim, gamma, dtype):
    """Core property: ``scalar_mul`` sends each radius to ``gamma * r / (r + eps)``.

    The eps regularizer makes the target ``gamma * r / (r + eps)`` rather than exactly
    ``gamma`` (a ~gamma*eps/r shortfall, negligible for r >> eps). Asserting against
    the exact target validates the radial-scaling math to full f64 precision. This is
    the test that proves Möbius (Poincaré) and Lorentz/PV ``scalar_mul`` scale geodesic
    radius identically.

    The shape / on-manifold assertions of the old ``test_rms_shape_and_on_manifold``
    are folded in: a radius oracle that also checks the output is a valid point of
    the right shape subsumes it. The dim axis is {2, 10} — the radial rescale is a
    scalar operation, so extra feature widths only duplicate items.
    """
    manifold = cfg_rms["make"](dtype)
    rms = cfg_rms["rms"](manifold, num_features=dim)
    rms.gamma[...] = jnp.asarray(gamma, dtype=rms.gamma[...].dtype)
    x = cfg_rms["points"](jax.random.PRNGKey(7), (16, dim), 1.0, dtype)

    out = rms(x, c=1.0)
    assert out.shape == x.shape
    assert jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(out, 1.0))

    r_in = jax.vmap(manifold.dist_0, in_axes=(0, None))(x, 1.0)
    radii = jax.vmap(manifold.dist_0, in_axes=(0, None))(out, 1.0)
    expected = gamma * r_in / (r_in + rms.eps)
    atol, _ = _tol(dtype)
    assert jnp.allclose(radii, expected, atol=atol)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("input_radius", [1e-4, 1e-2])
def test_rms_radius_normalization_from_a_tiny_input_radius_float32(gamma, input_radius):
    """Samples inside the old float32 ``dist_0`` floor still normalize to ``gamma``.

    ``HyperboloidGyroRMSNorm`` divides by ``dist_0(x)``. The pre-fix ``acosh`` arm reported
    1.5441e-3 for *every* float32 radius below that, so a batch at radius 1e-4 was rescaled by a
    factor ~15x too small and came out at ~0.065·gamma instead of gamma; at 1e-2 the ``x₀``
    resolution alone cost ~5e-4 relative. Float32 only — this is a float32 failure mode.

    ``eps`` is set to 1e-9 rather than the 1e-6 default so the regularizer's deliberate
    ``gamma·r/(r + eps)`` shortfall (1% at r = 1e-4 with the default) does not mask the effect
    under test; the exact target is asserted as well, exactly as in ``test_rms_radius_normalization``.
    """
    dtype = jnp.float32
    manifold = Hyperboloid(dtype=dtype)
    rms = HyperboloidGyroRMSNorm(manifold, num_features=8, eps=1e-9)
    rms.gamma[...] = jnp.asarray(gamma, dtype=rms.gamma[...].dtype)

    # Points at an exact geodesic radius: expmap_0 of a tangent vector of that spatial norm.
    directions_ND = jax.random.normal(jax.random.PRNGKey(21), (16, 8), dtype=dtype)
    directions_ND /= jnp.linalg.norm(directions_ND, axis=-1, keepdims=True)
    v_NA = manifold.embed_spatial_0(input_radius * directions_ND)
    x_NA = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v_NA, 1.0)

    out_NA = rms(x_NA, c=1.0)
    assert jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(out_NA, 1.0))

    r_in_N = jax.vmap(manifold.dist_0, in_axes=(0, None))(x_NA, 1.0)
    radii_N = jax.vmap(manifold.dist_0, in_axes=(0, None))(out_NA, 1.0)

    assert jnp.allclose(r_in_N, input_radius, rtol=1e-4), "input radius is not what the test intends"
    assert jnp.allclose(radii_N, gamma, rtol=1e-4)
    assert jnp.allclose(radii_N, gamma * r_in_N / (r_in_N + rms.eps), rtol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_batch_independence(cfg_rms, dtype):
    """A point's output is identical whether normalized alone or inside a batch."""
    manifold = cfg_rms["make"](dtype)
    rms = cfg_rms["rms"](manifold, num_features=6)
    batch = cfg_rms["points"](jax.random.PRNGKey(8), (10, 6), 1.0, dtype)

    out_batch = rms(batch, c=1.0)
    out_single = rms(batch[3:4], c=1.0)
    atol, _ = _tol(dtype)
    assert jnp.allclose(out_single[0], out_batch[3], atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_no_batch_state(cfg_rms, dtype):
    """RMSNorm holds no running statistics (no nnx.BatchStat leaves)."""
    manifold = cfg_rms["make"](dtype)
    rms = cfg_rms["rms"](manifold, num_features=6)
    batch_stats = nnx.state(rms, nnx.BatchStat)
    assert len(jax.tree.leaves(batch_stats)) == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_use_bias(cfg_rms, dtype):
    """use_bias=True applies a learned gyro-bias and keeps gradients finite."""
    manifold = cfg_rms["make"](dtype)
    rms = cfg_rms["rms"](manifold, num_features=6, use_bias=True)
    rms.bias[...] = jnp.full((6,), 0.1, dtype=rms.bias[...].dtype)
    x = cfg_rms["points"](jax.random.PRNGKey(9), (8, 6), 1.0, dtype)

    rms_nobias = cfg_rms["rms"](manifold, num_features=6, use_bias=False)
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
def test_rms_gradients_and_jit(cfg_rms, dtype):
    """Finite gradient for gamma, and the jitted forward reproduces eager values.

    RMSNorm is stateless, so the jit leg can be the strict ``jitted == eager``
    comparison rather than the old shape-and-finiteness check.
    """
    manifold = cfg_rms["make"](dtype)
    rms = cfg_rms["rms"](manifold, num_features=6)
    x = cfg_rms["points"](jax.random.PRNGKey(11), (8, 6), 1.0, dtype)

    def loss_fn(rms):
        return jnp.sum(rms(x, c=1.0) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(rms)
    assert jnp.isfinite(grads.gamma[...])

    @nnx.jit
    def forward(rms, x):
        return rms(x, c=1.0)

    out_eager = rms(x, c=1.0)
    out = forward(rms, x)
    assert out.shape == x.shape
    assert jnp.allclose(out, out_eager, atol=_tol(dtype)[0])


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_origin_input_stays_finite(cfg_rms, dtype):
    """A point at the origin (radius 0) stays at the origin — no gamma/eps blow-up."""
    manifold = cfg_rms["make"](dtype)
    rms = cfg_rms["rms"](manifold, num_features=6)
    origin = cfg_rms["origin"](manifold, 6, dtype)
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
def test_end_to_end_poincare(dtype):
    """GyroRMSNorm -> HypRegressionPoincarePP: finite loss/grads on on-ball inputs.

    Poincaré has no gyro BatchNorm (that role is the tangent-space PoincareBatchNorm2D),
    so the radial RMSNorm feeds the on-ball regression head directly.
    """
    from hyperbolix.nn_layers import HypRegressionPoincarePP

    manifold = Poincare(dtype=dtype)
    c = 1.0
    rngs = nnx.Rngs(0)
    rms = PoincareGyroRMSNorm(manifold, num_features=6)
    head = HypRegressionPoincarePP(manifold, 6, 3, rngs=rngs)  # input_space="manifold" by default
    x = _poincare_points(jax.random.PRNGKey(1), (8, 6), c, dtype)  # on-ball (8, 6)

    def loss_fn(rms, head):
        h = rms(x, c=c)
        return jnp.sum(head(h, c) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=(0, 1))(rms, head)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grads[0].gamma[...]))


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
