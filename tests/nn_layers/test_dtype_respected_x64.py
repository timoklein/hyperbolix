"""Regression guards: float32 NN layers must stay float32 under global x64.

``tests/conftest.py`` enables ``jax_enable_x64`` globally. Under that flag any
array built without an explicit ``dtype=`` (weight inits via ``jax.random.*``,
``jnp.zeros/full``, and ``jnp.arange``) is float64. If a layer then combines such
a value with a float32 input in a raw einsum/matmul/concat — i.e. *before* a
``_cast``-ing manifold method — the whole computation silently promotes to
float64, ignoring the manifold dtype contract.

Each test below constructs a float32 manifold + layer, feeds an explicitly
float32 input, and asserts the output (and any persistent state) stays float32.
These would fail on the un-cast code paths the boundary casts now guard.
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import (
    FGGLinear,
    FGGLorentzMLR,
    FGGMeanOnlyBatchNorm,
    HRCBatchNorm,
    HTCLinear,
    HypConv2DHyperboloid,
    HyperbolicFullAttention,
    HypLinearHyperboloidFHCNN,
    HypLinearHyperboloidFHNN,
    HypLinearPoincare,
    HypLinearPoincarePP,
    HypVQEmbeddingPoincare,
    LorentzConv2D,
    PoincareBatchNorm2D,
    hope,
)
from hyperbolix.optim import mark_manifold_param, riemannian_adam, riemannian_sgd

F32 = jnp.float32
F64 = jnp.float64


def _hyperboloid_points(key, batch, ambient, c=1.0):
    """A batch of float32 points on the hyperboloid of ambient dim ``ambient``."""
    v = jax.random.normal(key, (batch, ambient), dtype=F32) * 0.1
    return jax.vmap(Hyperboloid(dtype=F32).expmap_0, in_axes=(0, None))(v, c)


def test_x64_is_globally_enabled():
    """Sanity check: the leaks only matter when x64 is on (conftest enables it)."""
    assert jax.config.jax_enable_x64 is True


def test_hope_respects_float32():
    # HOPE rotary encoding: jnp.arange(0, d, 2)/d previously forced float64 freqs.
    key = jax.random.PRNGKey(0)
    batch, seq, d = 2, 6, 8  # spatial d must be even; ambient = d + 1
    tangent = jax.random.normal(key, (batch, seq, d + 1), dtype=F32) * 0.1
    z = jax.vmap(jax.vmap(Hyperboloid(dtype=F32).expmap_0, in_axes=(0, None)), in_axes=(0, None))(tangent, 1.0)
    positions = jnp.arange(seq)  # int — must not drag the output to float64

    out = hope(z, positions, c=1.0)
    assert out.dtype == F32


def test_htclinear_respects_float32():
    # HTCLinear: out = z @ kernel; kernel is float64 under x64 without the cast.
    x = _hyperboloid_points(jax.random.PRNGKey(1), batch=8, ambient=5)
    layer = HTCLinear(in_features=5, out_features=8, rngs=nnx.Rngs(0))
    assert layer(x, c_in=1.0, c_out=1.0).dtype == F32


def test_fhcnn_linear_respects_float32():
    # _fhcnn_forward: einsum(x, kernel) + bias before spatial_to_hyperboloid.
    manifold = Hyperboloid(dtype=F32)
    x = _hyperboloid_points(jax.random.PRNGKey(2), batch=8, ambient=6)
    layer = HypLinearHyperboloidFHCNN(manifold, 6, 10, rngs=nnx.Rngs(0))
    assert layer(x, c=1.0).dtype == F32


def test_fhnn_linear_respects_float32():
    # _fhnn_forward: same einsum-before-reconstruction path as FHCNN.
    manifold = Hyperboloid(dtype=F32)
    x = _hyperboloid_points(jax.random.PRNGKey(3), batch=8, ambient=6)
    layer = HypLinearHyperboloidFHNN(manifold, 6, 10, rngs=nnx.Rngs(0))
    assert layer(x, c=1.0).dtype == F32


def test_poincare_linear_respects_float32():
    # HypLinearPoincare: jnp.einsum(x_BI, kernel) in tangent space.
    manifold = Poincare(dtype=F32)
    x = jax.random.normal(jax.random.PRNGKey(4), (8, 5), dtype=F32) * 0.1  # tangent input
    layer = HypLinearPoincare(manifold, in_dim=5, out_dim=3, input_space="tangent", rngs=nnx.Rngs(0))
    assert layer(x, c=1.0).dtype == F32


def test_poincare_batchnorm_respects_float32():
    # var / running stats were float64 params; the tangent-space scaling promoted.
    manifold = Poincare(dtype=F32)
    bn = PoincareBatchNorm2D(manifold, num_features=8)
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 4, 8), dtype=F32) * 0.1  # tangent input
    out = bn(x, c=1.0)  # train mode (default) — also updates running stats
    assert out.dtype == F32
    # Persistent state must not silently become float64 either.
    assert bn.running_mean[...].dtype == F32
    assert bn.running_var[...].dtype == F32
    assert bn.var[...].dtype == F32


def test_fgg_mean_only_batchnorm_respects_float32():
    # bias / running_mean were float64; z - mean + bias promoted under x64.
    bn = FGGMeanOnlyBatchNorm(num_features=8)
    x = _hyperboloid_points(jax.random.PRNGKey(6), batch=16, ambient=9)
    out = bn(x, c_in=1.0, c_out=1.0)  # train mode — updates running_mean
    assert out.dtype == F32
    assert bn.running_mean[...].dtype == F32


def test_beta_concat_respects_float32():
    # jax.scipy.special.beta returns a STRONG float64 scalar under x64 (most
    # scalar math stays weak-typed); the unscaled ratio promoted the tangent
    # concat to float64 even for a float32 manifold.
    manifold = Poincare(dtype=F32)
    points = jax.random.uniform(jax.random.PRNGKey(7), (4, 3), dtype=F32, minval=-0.1, maxval=0.1)
    assert manifold.beta_concat(points, 1.0).dtype == F32


def _one_adam_step(layer, x, c):
    """Run a single Adam step so weak-typed scalar params turn strong float64.

    Init-time forwards cannot catch the scale-param leak: jnp.array(2.3) is
    weak-typed and adopts float32 in the forward. Optimizer arithmetic drops
    the weak flag, so from step 2 onward an un-cast scale promotes the layer.
    """
    optimizer = nnx.Optimizer(layer, optax.adam(1e-3), wrt=nnx.Param)
    grads = nnx.grad(lambda model: jnp.sum(model(x, c=c) ** 2))(layer)
    optimizer.update(layer, grads)


def test_fhcnn_learnable_scale_respects_float32_after_optimizer_step():
    # learnable_scale=True + normalize=True exercises jnp.exp(scale) in
    # _fhcnn_forward; kernel/bias were cast but the scale param was not.
    manifold = Hyperboloid(dtype=F32)
    x = _hyperboloid_points(jax.random.PRNGKey(8), batch=8, ambient=6)
    layer = HypLinearHyperboloidFHCNN(manifold, 6, 10, rngs=nnx.Rngs(0), learnable_scale=True, normalize=True)
    _one_adam_step(layer, x, c=1.0)
    assert layer(x, c=1.0).dtype == F32


def test_fhnn_linear_respects_float32_after_optimizer_step():
    # FHNN's scale is always learnable and always used (time coordinate).
    manifold = Hyperboloid(dtype=F32)
    x = _hyperboloid_points(jax.random.PRNGKey(9), batch=8, ambient=6)
    layer = HypLinearHyperboloidFHNN(manifold, 6, 10, rngs=nnx.Rngs(0))
    _one_adam_step(layer, x, c=1.0)
    assert layer(x, c=1.0).dtype == F32


# ---------------------------------------------------------------------------
# param_dtype storage contract: param_dtype (default float32) controls the
# STORAGE dtype of all trainable params and persistent state; manifold.dtype
# controls the COMPUTE precision of manifold operations. The tests below all
# use float64 manifolds to prove the two are decoupled.
# ---------------------------------------------------------------------------


def _assert_param_leaves(layer, dtype):
    leaves = jax.tree.leaves(nnx.state(layer, nnx.Param))
    assert leaves, "expected at least one Param leaf"
    for leaf in leaves:
        assert leaf.dtype == dtype, f"param leaf is {leaf.dtype}, expected {dtype}"


def test_param_storage_float32_with_float64_manifold():
    # One representative layer per family/init-path: random.normal kernels,
    # random.uniform kernels, jnp.eye/jnp.full inits (FGG), scalar params
    # (attention), wrapped nnx built-ins (LorentzConv2D, HRCBatchNorm), and a
    # ManifoldParam bias (HypLinearPoincare).
    p64 = Poincare(dtype=F64)
    h64 = Hyperboloid(dtype=F64)
    layers = [
        HypLinearPoincare(p64, 5, 3, rngs=nnx.Rngs(0)),
        FGGLinear(5, 9, rngs=nnx.Rngs(0), use_weight_norm=True),
        HTCLinear(5, 8, rngs=nnx.Rngs(0)),
        FGGLorentzMLR(9, 4, rngs=nnx.Rngs(0)),
        HypConv2DHyperboloid(h64, 4, 8, 3, rngs=nnx.Rngs(0)),
        LorentzConv2D(4, 8, (3, 3), rngs=nnx.Rngs(0)),
        HyperbolicFullAttention(5, 4, num_heads=2, rngs=nnx.Rngs(0)),
        HRCBatchNorm(8, rngs=nnx.Rngs(0)),
    ]
    for layer in layers:
        _assert_param_leaves(layer, F32)


def test_poincare_batchnorm_state_float32_with_float64_manifold():
    # Stats are computed via manifold ops (float64 here); the EMA update must
    # cast back to the float32 storage dtype at store time.
    p64 = Poincare(dtype=F64)
    bn = PoincareBatchNorm2D(p64, num_features=8)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4, 8), dtype=F32) * 0.1
    bn(x, c=1.0)  # train mode — updates running stats
    assert bn.mean[...].dtype == F32
    assert bn.var[...].dtype == F32
    assert bn.running_mean[...].dtype == F32
    assert bn.running_var[...].dtype == F32


def test_vq_embedding_state_float32_with_float64_manifold():
    # The EMA codebook update runs entirely in manifold dtype (float64); the
    # codebook/cluster-size buffers must keep their float32 storage.
    p64 = Poincare(dtype=F64)
    vq = HypVQEmbeddingPoincare(p64, num_codes=6, code_dim=3, rngs=nnx.Rngs(0))
    assert vq.codebook[...].dtype == F32
    assert vq.cluster_size[...].dtype == F32
    x = jax.random.normal(jax.random.PRNGKey(1), (12, 3), dtype=F32) * 0.1
    out = vq(x, c=1.0)
    vq.ema_update(out.z, out.indices, c=1.0)
    assert vq.codebook[...].dtype == F32
    assert vq.cluster_size[...].dtype == F32


def test_riemannian_adam_update_and_moments_keep_param_dtype():
    # Direct tx.update: optax.apply_updates casts updates to the param dtype
    # at apply time, which would MASK a float64 update through nnx.Optimizer.
    # The transformation contract itself (updates and moments in the param's
    # storage dtype) is what this asserts. The (N, D) table covers the
    # vmapped point_update path.
    manifold = Poincare(dtype=F64)
    # mark_manifold_param RETURNS the tagged param (it does not mutate).
    p = mark_manifold_param(
        nnx.Param(jax.random.normal(jax.random.PRNGKey(0), (4, 3), dtype=F32) * 0.01),
        manifold=manifold,
        curvature=1.0,
    )
    params = {"emb": p}
    tx = riemannian_adam(1e-2)
    state = tx.init(params)
    grads = {"emb": jax.random.normal(jax.random.PRNGKey(1), (4, 3), dtype=F32)}
    updates, new_state = tx.update(grads, state, params)
    assert updates["emb"].dtype == F32
    assert new_state[0]["emb"].dtype == F32  # m1
    assert new_state[1]["emb"].dtype == F32  # m2


def test_riemannian_sgd_momentum_keeps_param_dtype():
    # SGD momentum buffers are parallel-transported through manifold.ptransp,
    # whose output is manifold.dtype (float64 here) — without the storage
    # cast-back the optimizer state would silently turn float64 after step 1.
    manifold = Poincare(dtype=F64)
    p = mark_manifold_param(
        nnx.Param(jax.random.normal(jax.random.PRNGKey(2), (4, 3), dtype=F32) * 0.01),
        manifold=manifold,
        curvature=1.0,
    )
    params = {"emb": p}
    tx = riemannian_sgd(1e-2, momentum=0.9)
    state = tx.init(params)
    grads = {"emb": jax.random.normal(jax.random.PRNGKey(3), (4, 3), dtype=F32)}
    updates, new_state = tx.update(grads, state, params)
    assert updates["emb"].dtype == F32
    assert new_state[0]["emb"].dtype == F32  # momentum


def test_hyplinear_poincare_params_float32_after_riemannian_step():
    # End-to-end nnx path: kernel (Euclidean update) and bias (ManifoldParam,
    # Riemannian update through float64 expmap) both keep float32 storage.
    manifold = Poincare(dtype=F64)
    layer = HypLinearPoincare(manifold, 5, 3, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(layer, riemannian_adam(1e-2), wrt=nnx.Param)
    x = jax.random.normal(jax.random.PRNGKey(4), (4, 5), dtype=F32) * 0.1
    grads = nnx.grad(lambda m: jnp.sum(m(x, c=1.0) ** 2))(layer)
    optimizer.update(layer, grads)
    assert layer.kernel[...].dtype == F32
    assert layer.bias[...].dtype == F32
    # Optimizer state (Adam moments) must stay float32 too — optax's
    # apply_updates casts param updates but nothing downstream rescues moments.
    for leaf in jax.tree.leaves(nnx.state(optimizer)):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == F32, f"optimizer state leaf is {leaf.dtype}"


def test_param_dtype_float64_opt_in():
    # Explicit param_dtype=float64 must reach every creation site, including
    # the renamed FGGMeanOnlyBatchNorm argument (previously ``dtype``).
    p64 = Poincare(dtype=F64)
    layer = HypLinearPoincarePP(p64, 5, 3, rngs=nnx.Rngs(0), param_dtype=F64)
    _assert_param_leaves(layer, F64)
    bn = PoincareBatchNorm2D(p64, num_features=8, param_dtype=F64)
    assert bn.running_mean[...].dtype == F64
    fgg_bn = FGGMeanOnlyBatchNorm(8, param_dtype=F64)
    assert fgg_bn.bias[...].dtype == F64
    assert fgg_bn.running_mean[...].dtype == F64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
