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

The same leak enters through a float64 *curvature*: a Python-float ``c`` is weak-typed and
adopts float32, but ``np.float64(c)`` or a float64 ``jnp`` scalar is strong and promoted every
layer that used ``c`` before casting it. ``test_float64_curvature_keeps_float32`` covers every
public layer and functional op with both spellings.

Dimension key: B batch  S sequence  H W feature-map height/width  A ambient dim (spatial + 1)
  D spatial dim  M points averaged  N midpoints  I O in/out dim
"""

import functools
import json
import os
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.scipy.special import digamma

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import (
    FGGConv2D,
    FGGLinear,
    FGGLorentzMLR,
    FGGMeanOnlyBatchNorm,
    HRCBatchNorm,
    HRCDropout,
    HRCLayerNorm,
    HRCRMSNorm,
    HTCLinear,
    HypConv2DHyperboloid,
    HypConv2DHyperboloidILNN,
    HypConv2DPoincare,
    HyperbolicFullAttention,
    HyperbolicLinearAttention,
    HyperbolicRoPE,
    HyperbolicSoftmaxAttention,
    HyperPPFeatureScaling,
    HypformerPositionalEncoding,
    HypLinearHyperboloidBusemann,
    HypLinearHyperboloidFHCNN,
    HypLinearHyperboloidFHNN,
    HypLinearHyperboloidPLFC,
    HypLinearPoincare,
    HypLinearPoincareBusemann,
    HypLinearPoincarePP,
    HypRegressionHyperboloidBusemann,
    HypRegressionPoincare,
    HypRegressionPoincareBusemann,
    HypVQEmbeddingPoincare,
    LorentzConv2D,
    LorentzEmbedding,
    LorentzMLA,
    LorentzMoE,
    LorentzMoEGate,
    LorentzResidual,
    LorentzSwiGLU,
    PoincareBatchNorm2D,
    build_spacelike_V,
    busemann_fc_poincare_output,
    extract_patches,
    hope,
    hope_rotate_space,
    hrc,
    hrc_gelu,
    hrc_leaky_relu,
    hrc_relu,
    hrc_swish,
    hrc_tanh,
    htc,
    hyp_avg_pool2d,
    hyp_gelu,
    hyp_leaky_relu,
    hyp_relu,
    hyp_swish,
    hyp_tanh,
    lorentz_midpoint,
    lorentz_residual,
    lorentz_scale,
    sinh_lift_to_hyperboloid,
    spatial_to_hyperboloid,
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


def test_hope_rotate_space_respects_float32():
    # The rotation angles positions * theta come from jnp.arange / a float base; they must not promote.
    space = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 8), dtype=F32)
    assert hope_rotate_space(space, jnp.arange(6)).dtype == F32  # (S,) positions
    assert hope_rotate_space(space, jnp.tile(jnp.arange(6), (2, 1))).dtype == F32  # (B, S) positions


def test_htclinear_respects_float32():
    # HTCLinear: out = z @ kernel; kernel is float64 under x64 without the cast.
    x = _hyperboloid_points(jax.random.PRNGKey(1), batch=8, ambient=5)
    layer = HTCLinear(in_features=5, out_features=8, rngs=nnx.Rngs(0))
    assert layer(x, c_in=1.0, c_out=1.0).dtype == F32


def test_lorentz_mla_respects_float32():
    # HOPE angles, the score temperature, the softmax and the centroid all mix float32 data with
    # scalars/tables that are float64 under x64 unless cast.
    x = _hyperboloid_points(jax.random.PRNGKey(10), batch=2 * 5, ambient=9).reshape(2, 5, 9)
    layer = LorentzMLA(9, 2, 4, 3, 5, 4, rngs=nnx.Rngs(0))
    assert layer(x, c=1.0).dtype == F32
    assert layer(x, c=1.0, positions=jnp.tile(jnp.arange(5), (2, 1))).dtype == F32


def test_lorentz_moe_respects_float32():
    # Learnable expert curvatures are float32 params; the gate scores, one-hot weights and the
    # stacked-expert vmap must keep the float32 activations.
    x = _hyperboloid_points(jax.random.PRNGKey(11), batch=2 * 5, ambient=9).reshape(2, 5, 9)
    moe = LorentzMoE(9, 6, 4, 1, 2, rngs=nnx.Rngs(0))
    out, stats = moe(x, c=1.0)
    assert out.dtype == F32
    assert stats.affinity.dtype == F32
    assert stats.mask.dtype == F32
    weights, _, _ = moe.gate(x.reshape(-1, 9))
    assert weights.dtype == F32


def test_lorentz_swiglu_respects_float32():
    x = _hyperboloid_points(jax.random.PRNGKey(12), batch=8, ambient=9)
    expert = LorentzSwiGLU(9, 6, rngs=nnx.Rngs(0))
    assert expert(x, 0.5, 1.0).dtype == F32


@pytest.mark.parametrize("parameterization", ["manifold", "spatial"])
def test_lorentz_embedding_respects_float32(parameterization):
    # The table init (jax.random.normal) and the sqrt(c / c_out) map are the promotion sites.
    emb = LorentzEmbedding(16, 9, rngs=nnx.Rngs(0), parameterization=parameterization)
    ids = jnp.arange(6).reshape(2, 3)
    assert emb.embedding[...].dtype == F32
    assert emb(ids).dtype == F32
    assert emb(ids, c_out=0.5).dtype == F32


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
# A float64 curvature (np.float64 or a float64 jnp scalar) must not promote a
# float32 layer. Each case below is fn(c) -> output on float32 inputs and, where
# the entry point has one, a float32 manifold; it is called once with the
# Python float C_VAL (the reference) and once with each float64 spelling.
# ---------------------------------------------------------------------------

C_VAL = 0.7
C64_SPELLINGS = {
    "np.float64": np.float64,
    "jnp_f64": lambda value: jnp.asarray(value, dtype=F64),
}


def _like_c(c, value):
    """``value`` spelled like ``c`` (Python float, np.float64 or float64 jnp scalar)."""
    if isinstance(c, jax.Array):
        return jnp.asarray(value, dtype=c.dtype)
    if isinstance(c, np.floating):
        return type(c)(value)
    return value


@functools.cache
def _h(shape, seed=0):
    """Float32 hyperboloid points on the C_VAL sheet; ``shape`` ends in the ambient dim A."""
    ambient = shape[-1]
    batch = int(np.prod(shape[:-1]))
    return _hyperboloid_points(jax.random.PRNGKey(100 + seed), batch, ambient, c=C_VAL).reshape(shape)


@functools.cache
def _t(shape, seed=0, scale=0.3):
    """Float32 Euclidean / tangent draw."""
    return jax.random.normal(jax.random.PRNGKey(200 + seed), shape, dtype=F32) * scale


@functools.cache
def _p(shape, seed=0):
    """Float32 Poincaré-ball points at curvature C_VAL."""
    flat_ND = _t(shape, seed).reshape(-1, shape[-1])
    return jax.vmap(Poincare(dtype=F32).expmap_0, in_axes=(0, None))(flat_ND, C_VAL).reshape(shape)


def _rngs():
    return nnx.Rngs(0)


H32 = Hyperboloid(dtype=F32)
P32 = Poincare(dtype=F32)
SEQ_A = 9  # ambient dim of the sequence layers (attention, MLA, MoE, ...)


def _core_cases():
    x_BA, y_BA = _h((4, 6)), _h((4, 6), seed=1)
    x_BHWA = _h((2, 4, 4, 4), seed=2)
    x_BMA = _h((2, 5, 6), seed=3)
    w_BNM = jnp.abs(_t((2, 3, 5), seed=4))
    U_IO, b_O = _t((5, 7), seed=5), _t((7,), seed=6)
    s_BD = _t((4, 5), seed=7, scale=1.0)
    W_AD = _t((6, 5), seed=8)
    return {
        "core.build_spacelike_V": lambda c: build_spacelike_V(U_IO, b_O, c),
        "core.spatial_to_hyperboloid": lambda c: spatial_to_hyperboloid(s_BD, c, c),
        "core.sinh_lift_to_hyperboloid": lambda c: sinh_lift_to_hyperboloid(s_BD, c, 10.0),
        "core.lorentz_midpoint[variance]": lambda c: lorentz_midpoint(x_BMA, w_BNM, c),
        "core.lorentz_midpoint[gemm]": lambda c: lorentz_midpoint(x_BMA, w_BNM, c, form="gemm"),
        "core.lorentz_residual": lambda c: lorentz_residual(x_BA, y_BA, 0.7, c),
        # gamma is spelled like c, so the float64 legs also pass a float64 gamma.
        "core.lorentz_scale": lambda c: lorentz_scale(x_BA, _like_c(c, 1.5), c),
        "core.hyp_avg_pool2d": lambda c: hyp_avg_pool2d(x_BHWA, c),
        "core.hrc": lambda c: hrc(x_BA, jax.nn.relu, c, c),
        "core.htc": lambda c: htc(x_BA, lambda z_BA: z_BA @ W_AD, c, c),
        # np.float64 c used to raise a JAX FutureWarning here; the test turns it into an error.
        "core.extract_patches[origin]": lambda c: extract_patches(x_BHWA, (3, 3), (1, 1), "SAME", "origin", c),
    }


def _activation_cases():
    x_BA = _h((4, 6))
    cases = {
        f"act.{f.__name__}": functools.partial(f, x_BA) for f in (hyp_relu, hyp_leaky_relu, hyp_tanh, hyp_swish, hyp_gelu)
    }
    for f in (hrc_relu, hrc_leaky_relu, hrc_tanh, hrc_swish, hrc_gelu):
        cases[f"act.{f.__name__}"] = functools.partial(lambda f_, c: f_(x_BA, c, c), f)
    return cases


def _hyperboloid_layer_cases():
    x_BA = _h((4, 6))
    x_BHWA = _h((2, 4, 4, 4), seed=2)
    return {
        "hyp.FHCNN[normalize]": lambda c: HypLinearHyperboloidFHCNN(H32, 6, 8, rngs=_rngs(), normalize=True)(x_BA, c),
        "hyp.PLFC": lambda c: HypLinearHyperboloidPLFC(H32, 6, 8, rngs=_rngs())(x_BA, c),
        "hyp.HTCLinear": lambda c: HTCLinear(6, 7, rngs=_rngs())(x_BA, c, c),
        "hyp.LorentzConv2D": lambda c: LorentzConv2D(H32, 4, 5, 3, rngs=_rngs())(x_BHWA, c),
        "hyp.HypConv2DHyperboloid": lambda c: HypConv2DHyperboloid(H32, 4, 5, 3, rngs=_rngs())(x_BHWA, c),
        "hyp.ILNN": lambda c: HypConv2DHyperboloidILNN(H32, 4, 5, 3, rngs=_rngs())(x_BHWA, c),
        "hyp.FGGLinear": lambda c: FGGLinear(6, 8, rngs=_rngs())(x_BA, c),
        "hyp.FGGConv2D": lambda c: FGGConv2D(H32, 4, 5, 3, rngs=_rngs())(x_BHWA, c),
        "hyp.FGGLorentzMLR": lambda c: FGGLorentzMLR(6, 3, rngs=_rngs())(x_BA, c),
        # Train mode: the same Rngs(0) seed draws the same mask in every leg.
        "hyp.HRCDropout": lambda c: HRCDropout(0.3, rngs=_rngs())(x_BA, c, c),
        "hyp.HRCLayerNorm": lambda c: HRCLayerNorm(5, rngs=_rngs())(x_BA, c, c),
        "hyp.HRCRMSNorm": lambda c: HRCRMSNorm(5, rngs=_rngs())(x_BA, c, c),
        "hyp.HRCBatchNorm": lambda c: HRCBatchNorm(5, rngs=_rngs())(x_BA, c, c),
        "hyp.FGGMeanOnlyBatchNorm": lambda c: FGGMeanOnlyBatchNorm(5)(x_BA, c, c),
    }


def _sequence_cases():
    a = SEQ_A
    x_BSA, y_BSA = _h((2, 5, a), seed=10), _h((2, 5, a), seed=11)
    ids_BS = jnp.arange(10).reshape(2, 5)
    positions_S = jnp.arange(5)
    return {
        "seq.HyperbolicFullAttention": lambda c: HyperbolicFullAttention(a, a - 1, num_heads=2, rngs=_rngs())(x_BSA, c, c, c),
        "seq.HyperbolicLinearAttention": lambda c: HyperbolicLinearAttention(a, a - 1, num_heads=2, rngs=_rngs())(
            x_BSA, c, c, c
        ),
        "seq.HyperbolicSoftmaxAttention": lambda c: HyperbolicSoftmaxAttention(a, a - 1, num_heads=2, rngs=_rngs())(
            x_BSA, c, c, c
        ),
        "seq.LorentzMLA": lambda c: LorentzMLA(a, 2, 4, 3, 5, 4, rngs=_rngs())(x_BSA, c),
        "seq.LorentzSwiGLU": lambda c: LorentzSwiGLU(a, 6, rngs=_rngs())(x_BSA, 2.0 * c, c),
        "seq.LorentzResidual": lambda c: LorentzResidual()(x_BSA, y_BSA, c=c),
        "seq.LorentzResidual[scaled]": lambda c: LorentzResidual(scale=True, init_gamma=3.0, learnable_scale=True)(
            x_BSA, y_BSA, c=c
        ),
        "seq.HypformerPositionalEncoding": lambda c: HypformerPositionalEncoding(a, a - 1, rngs=_rngs())(x_BSA, c),
        "seq.HyperbolicRoPE": lambda c: HyperbolicRoPE(a - 1)(x_BSA, positions_S, c),
        "seq.hope": lambda c: hope(x_BSA, positions_S, c),
        "seq.LorentzMoE": lambda c: LorentzMoE(a, 6, 4, 1, 2, rngs=_rngs())(x_BSA, c)[0],
        "seq.LorentzEmbedding": lambda c: LorentzEmbedding(16, a, c=c, rngs=_rngs())(ids_BS, c_out=c),
    }


def _poincare_busemann_cases():
    x5_BI, x6_BI = _p((8, 5)), _p((8, 6), seed=1)
    x7_BA = _h((8, 7), seed=12)
    v_BHWC = _t((2, 4, 4, 3), seed=13)
    x_BHWC = _p((2, 4, 4, 3), seed=13)
    return {
        "poincare.HypConv2DPoincare[tangent]": lambda c: HypConv2DPoincare(
            P32, 3, 4, 2, rngs=_rngs(), input_space="tangent", id_init=False
        )(v_BHWC, c),
        "poincare.HypConv2DPoincare[manifold]": lambda c: HypConv2DPoincare(
            P32, 3, 4, 2, rngs=_rngs(), input_space="manifold", id_init=False
        )(x_BHWC, c),
        "poincare.HypRegressionPoincare": lambda c: HypRegressionPoincare(P32, 5, 3, rngs=_rngs())(x5_BI, c),
        "poincare.HypLinearPoincarePP": lambda c: HypLinearPoincarePP(P32, 5, 3, rngs=_rngs())(x5_BI, c),
        "busemann.HypLinearPoincareBusemann": lambda c: HypLinearPoincareBusemann(P32, 6, 4, rngs=_rngs(), use_gyro_bias=True)(
            x6_BI, c
        ),
        "busemann.HypLinearHyperboloidBusemann[gyro_bias]": lambda c: HypLinearHyperboloidBusemann(
            H32, 7, 5, rngs=_rngs(), use_gyro_bias=True
        )(x7_BA, c),
        "busemann.HypLinearHyperboloidBusemann": lambda c: HypLinearHyperboloidBusemann(H32, 7, 5, rngs=_rngs())(x7_BA, c),
        "busemann.HypRegressionPoincareBusemann": lambda c: HypRegressionPoincareBusemann(P32, 6, 4, rngs=_rngs())(x6_BI, c),
        "busemann.HypRegressionHyperboloidBusemann": lambda c: HypRegressionHyperboloidBusemann(H32, 7, 4, rngs=_rngs())(
            x7_BA, c
        ),
        "busemann.busemann_fc_poincare_output": lambda c: busemann_fc_poincare_output(_t((8, 4), seed=14, scale=2.0), c, 10.0),
        "hybrid.HyperPPFeatureScaling[alpha]": lambda c: HyperPPFeatureScaling(5, alpha=0.9, rngs=_rngs())(
            _t((8, 5), seed=15, scale=1.0), c
        ),
    }


def _c64_cases():
    return {
        **_core_cases(),
        **_activation_cases(),
        **_hyperboloid_layer_cases(),
        **_sequence_cases(),
        **_poincare_busemann_cases(),
    }


C64_CASE_NAMES = (
    "core.build_spacelike_V",
    "core.spatial_to_hyperboloid",
    "core.sinh_lift_to_hyperboloid",
    "core.lorentz_midpoint[variance]",
    "core.lorentz_midpoint[gemm]",
    "core.lorentz_residual",
    "core.lorentz_scale",
    "core.hyp_avg_pool2d",
    "core.hrc",
    "core.htc",
    "core.extract_patches[origin]",
    *(f"act.{f.__name__}" for f in (hyp_relu, hyp_leaky_relu, hyp_tanh, hyp_swish, hyp_gelu)),
    *(f"act.{f.__name__}" for f in (hrc_relu, hrc_leaky_relu, hrc_tanh, hrc_swish, hrc_gelu)),
    "hyp.FHCNN[normalize]",
    "hyp.PLFC",
    "hyp.HTCLinear",
    "hyp.LorentzConv2D",
    "hyp.HypConv2DHyperboloid",
    "hyp.ILNN",
    "hyp.FGGLinear",
    "hyp.FGGConv2D",
    "hyp.FGGLorentzMLR",
    "hyp.HRCDropout",
    "hyp.HRCLayerNorm",
    "hyp.HRCRMSNorm",
    "hyp.HRCBatchNorm",
    "hyp.FGGMeanOnlyBatchNorm",
    "seq.HyperbolicFullAttention",
    "seq.HyperbolicLinearAttention",
    "seq.HyperbolicSoftmaxAttention",
    "seq.LorentzMLA",
    "seq.LorentzSwiGLU",
    "seq.LorentzResidual",
    "seq.LorentzResidual[scaled]",
    "seq.HypformerPositionalEncoding",
    "seq.HyperbolicRoPE",
    "seq.hope",
    "seq.LorentzMoE",
    "seq.LorentzEmbedding",
    "poincare.HypConv2DPoincare[tangent]",
    "poincare.HypConv2DPoincare[manifold]",
    "poincare.HypRegressionPoincare",
    "poincare.HypLinearPoincarePP",
    "busemann.HypLinearPoincareBusemann",
    "busemann.HypLinearHyperboloidBusemann[gyro_bias]",
    "busemann.HypLinearHyperboloidBusemann",
    "busemann.HypRegressionPoincareBusemann",
    "busemann.HypRegressionHyperboloidBusemann",
    "busemann.busemann_fc_poincare_output",
    "hybrid.HyperPPFeatureScaling[alpha]",
)


def test_c64_case_names_match_the_table():
    # The parametrize ids are spelled out so collection builds no arrays; keep them in sync.
    assert sorted(C64_CASE_NAMES) == sorted(_c64_cases())


def _assert_float32_like(out, ref, what):
    assert out.dtype == F32, f"{what}: output is {out.dtype}, expected float32"
    # Both legs compute in float32 from the same float32 inputs (the boundary casts turn every
    # float64 c into the same float32 value a Python float becomes); measured bit-identical on
    # CPU, the tolerance leaves a few float32 ulps.
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-6, atol=1e-6, err_msg=what)


@pytest.mark.parametrize("spelling", sorted(C64_SPELLINGS))
@pytest.mark.parametrize("name", C64_CASE_NAMES)
def test_float64_curvature_keeps_float32(name, spelling):
    fn = _c64_cases()[name]
    ref = fn(C_VAL)
    assert ref.dtype == F32, f"{name}: Python-float c already gives {ref.dtype}"
    with warnings.catch_warnings():
        # extract_patches(pad_mode="origin") built its padding value from c with a FutureWarning.
        warnings.simplefilter("error", FutureWarning)
        out = fn(C64_SPELLINGS[spelling](C_VAL))
    _assert_float32_like(out, ref, f"{name} with c as {spelling}")


# Layers with a manifold whose input is a manifold point: a float64 input array (what
# jnp.asarray(numpy_array) gives under x64) must still compute in the manifold's float32.
# busemann_fc_poincare_output and HyperPPFeatureScaling have no manifold, so their output
# legitimately follows a float64 input and they are not listed.
FLOAT64_INPUT_CASES = {
    "HypRegressionPoincareBusemann": lambda: (HypRegressionPoincareBusemann(P32, 6, 4, rngs=_rngs()), _p((8, 6), seed=1)),
    "HypRegressionHyperboloidBusemann": lambda: (
        HypRegressionHyperboloidBusemann(H32, 7, 4, rngs=_rngs()),
        _h((8, 7), seed=12),
    ),
    "HypLinearHyperboloidBusemann": lambda: (HypLinearHyperboloidBusemann(H32, 7, 5, rngs=_rngs()), _h((8, 7), seed=12)),
    "HypLinearHyperboloidBusemann[gyro_bias]": lambda: (
        HypLinearHyperboloidBusemann(H32, 7, 5, rngs=_rngs(), use_gyro_bias=True),
        _h((8, 7), seed=12),
    ),
    "HypLinearPoincareBusemann": lambda: (HypLinearPoincareBusemann(P32, 6, 4, rngs=_rngs()), _p((8, 6), seed=1)),
    "HypLinearPoincareBusemann[gyro_bias]": lambda: (
        HypLinearPoincareBusemann(P32, 6, 4, rngs=_rngs(), use_gyro_bias=True),
        _p((8, 6), seed=1),
    ),
}


@pytest.mark.parametrize("name", sorted(FLOAT64_INPUT_CASES))
def test_float64_input_on_float32_manifold_keeps_float32(name):
    layer, x_BI = FLOAT64_INPUT_CASES[name]()
    ref = layer(x_BI, C_VAL)
    out = layer(x_BI.astype(F64), C_VAL)
    _assert_float32_like(out, ref, f"{name} with a float64 input")


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
        LorentzConv2D(h64, 4, 8, (3, 3), rngs=nnx.Rngs(0)),
        HyperbolicFullAttention(5, 4, num_heads=2, rngs=nnx.Rngs(0)),
        HRCBatchNorm(8, rngs=nnx.Rngs(0)),
        LorentzMLA(9, 2, 4, 3, 5, 4, rngs=nnx.Rngs(0)),
        LorentzMoE(9, 6, 4, 1, 2, rngs=nnx.Rngs(0)),
        LorentzSwiGLU(9, 6, rngs=nnx.Rngs(0)),
        LorentzMoEGate(9, 4, 2, rngs=nnx.Rngs(0)),
        LorentzEmbedding(16, 9, rngs=nnx.Rngs(0)),
        LorentzEmbedding(16, 9, rngs=nnx.Rngs(0), parameterization="spatial"),
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
    mla = LorentzMLA(9, 2, 4, 3, 5, 4, rngs=nnx.Rngs(0), param_dtype=F64)
    _assert_param_leaves(mla, F64)
    # The routed experts' raw curvatures stay float32 by design, so fix them here.
    moe = LorentzMoE(9, 6, 4, 1, 2, rngs=nnx.Rngs(0), param_dtype=F64, learnable_curvature=False)
    _assert_param_leaves(moe, F64)
    _assert_param_leaves(LorentzEmbedding(16, 9, rngs=nnx.Rngs(0), param_dtype=F64), F64)
    fgg_bn = FGGMeanOnlyBatchNorm(8, param_dtype=F64)
    assert fgg_bn.bias[...].dtype == F64
    assert fgg_bn.running_mean[...].dtype == F64


# Builds every layer whose init scale was a JAX op on Python numbers (`jnp.sqrt(k / n)`), at a size
# where float32 and float64 round that scale differently (`np.float32(np.sqrt(k / n))` against
# `np.sqrt(np.float32(k / n))`), plus the LogCat digamma scale and the two sites that took `jnp.sqrt`
# of a Python-float `c`, at c = 0.9, which float32 and float64 round differently there.
# Run once in this process (x64 on) and once in a fresh interpreter (x64 off).
_SEEDED_X64_BUILD = """
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from hyperbolix.distributions import uniform_poincare
from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.nn_layers import (
    FGGConv2D, FGGLinear, FGGLorentzMLR, HypConv2DHyperboloid, HypConv2DPoincare, HypLinearHyperboloidFHNN,
    HypLinearPoincare, HypLinearPoincarePP, HypRegressionHyperboloid, HypRegressionPoincare,
    HypRegressionPoincarePP,
)


def build():
    hyp, poi = Hyperboloid(dtype=jnp.float32), Poincare(dtype=jnp.float32)
    r = lambda: nnx.Rngs(0)
    out = {}
    # FGG: (reset_params, in_features, out_features), each at a size its scale formula leaks at.
    for rp, i, o in [("fan_out", 6, 24), ("xavier", 14, 16), ("kaiming", 15, 8),
                     ("lorentz_kaiming", 23, 8), ("mlr", 7, 8)]:
        out[f"FGGLinear-{rp}"] = FGGLinear(i, o, rngs=r(), reset_params=rp).kernel[...]
    out["FGGLinear-weight-norm-scale"] = FGGLinear(14, 16, rngs=r(), use_weight_norm=True).kernel_scale[...]
    out["FGGConv2D-fan_out"] = FGGConv2D(hyp, 4, 24, 3, rngs=r()).kernel[...]
    out["FGGConv2D-mlr"] = FGGConv2D(hyp, 4, 5, 3, rngs=r(), reset_params="mlr").kernel[...]
    out["FGGConv2D-weight-norm-scale"] = FGGConv2D(hyp, 4, 11, 3, rngs=r(), use_weight_norm=True).kernel_scale[...]
    out["FGGLorentzMLR-mlr"] = FGGLorentzMLR(7, 3, rngs=r()).kernel[...]
    out["HypRegressionHyperboloid"] = HypRegressionHyperboloid(hyp, 4, 3, rngs=r()).kernel[...]
    out["HypRegressionPoincare"] = HypRegressionPoincare(poi, 3, 3, rngs=r()).kernel[...]
    out["HypRegressionPoincarePP"] = HypRegressionPoincarePP(poi, 3, 3, rngs=r()).kernel[...]
    out["HypConv2DHyperboloid-fan_in"] = HypConv2DHyperboloid(hyp, 10, 4, 3, rngs=r(), reset_params="fan_in").kernel[...]
    out["HypLinearPoincare"] = HypLinearPoincare(poi, 6, 3, rngs=r()).kernel[...]
    out["HypLinearPoincarePP"] = HypLinearPoincarePP(poi, 6, 3, rngs=r()).kernel[...]
    out["HypConv2DPoincare-random"] = HypConv2DPoincare(poi, 2, 3, 3, rngs=r(), id_init=False).kernel[...]
    # FHNN origin fallback: a zero kernel sends every row to the origin, whose time coordinate is 1/sqrt(c).
    fhnn = HypLinearHyperboloidFHNN(hyp, 4, 3, rngs=r())
    fhnn.kernel[...] = jnp.zeros_like(fhnn.kernel[...])
    x_1A = np.float32([[np.sqrt(np.float32(1) / np.float32(0.9)), 0.0, 0.0, 0.0]])
    out["HypLinearHyperboloidFHNN-origin"] = fhnn(jnp.asarray(x_1A), 0.9)
    # Rejection sampler (n >= 3): sqrt(c) sets u_max and so every draw.
    out["uniform_poincare.sample-n3"] = uniform_poincare.sample(
        jax.random.key(0), 3, 0.9, 1.0, sample_shape=(6,), dtype=jnp.float32
    )
    # LogCat of 3 points with 4 spatial dims, built in NumPy float32 so both processes get the same input.
    space_ND = np.linspace(-0.9, 1.3, 12, dtype=np.float32).reshape(3, 4)
    time_N1 = np.sqrt(np.float32(1) + np.sum(space_ND**2, axis=-1, keepdims=True, dtype=np.float32))
    out["Hyperboloid.log_radius_concat"] = hyp.log_radius_concat(jnp.asarray(np.concatenate([time_N1, space_ND], -1)), 1.0)
    return out
"""


def test_seeded_init_and_logcat_scale_same_with_and_without_x64():
    """Seeded inits and the LogCat scale give the same float32 values with and without x64.

    Their scales were JAX ops on Python numbers (``jnp.sqrt(5.0 / in_features)``, ``digamma(d / 2)``):
    float32 without x64 and float64 rounded afterwards with it, so at 12-23 % of layer sizes a
    seed gave weights 1 ulp apart. They are now computed in float32 (float64 for float64
    parameters), which leaves every value without x64 as it was. The FHNN origin fallback and the
    Poincaré-ball rejection sampler took ``jnp.sqrt`` of a Python-float ``c`` the same way.
    """
    code = _SEEDED_X64_BUILD + (
        "\nimport json\n"
        "print(json.dumps({k: [str(v.dtype), [float(x).hex() for x in np.asarray(v).ravel()]] for k, v in build().items()}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "JAX_ENABLE_X64": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    reference = json.loads(proc.stdout.strip().splitlines()[-1])

    namespace: dict = {}
    exec(_SEEDED_X64_BUILD, namespace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = namespace["build"]()
    assert jax.config.jax_enable_x64
    assert set(values) == set(reference)
    mismatched = []
    for name, value in values.items():
        ref_dtype, ref_hex = reference[name]
        assert ref_dtype == "float32" and value.dtype == F32, (name, ref_dtype, value.dtype)
        if not np.array_equal(np.asarray(value).ravel(), np.float32([float.fromhex(h) for h in ref_hex])):
            mismatched.append(name)
    assert not mismatched, f"differ between x64 on and off: {mismatched}"


@pytest.mark.parametrize(
    "half_dtype, d, n_points",
    [(jnp.bfloat16, 63, 25), (jnp.float16, 63, 2)],
)
def test_logcat_half_precision_scale_is_float32_scale_rounded(half_dtype, d, n_points):
    """A half-precision LogCat scales by the float32 digamma scale rounded once to its dtype.

    That is what it always did without x64 (the scale was a weakly typed float32). Computing the
    digamma scale in bfloat16 or float16 instead moves it by 1-2 ulps of that dtype at these sizes.
    """
    hyp = Hyperboloid(dtype=half_dtype)
    space_ND = np.linspace(-0.5, 0.7, n_points * d, dtype=np.float32).reshape(n_points, d)
    time_N1 = np.sqrt(np.float32(1) + np.sum(space_ND**2, axis=-1, keepdims=True, dtype=np.float32))
    points_NA = jnp.asarray(np.concatenate([time_N1, space_ND], -1), dtype=half_dtype)
    out_A = hyp.log_radius_concat(points_NA, 1.0)

    half_d, half_n = jnp.float32(d / 2), jnp.float32(n_points * d / 2)
    scale_f32 = jnp.exp(0.5 * (digamma(half_d) - digamma(half_n)))
    expected_space = (scale_f32.astype(half_dtype) * points_NA[:, 1:]).reshape(-1)
    assert out_A.dtype == half_dtype
    np.testing.assert_array_equal(np.asarray(out_A[1:], np.float32), np.asarray(expected_space, np.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
