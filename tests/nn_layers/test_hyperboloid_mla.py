"""Tests for LorentzMLA, the hyperbolic multi-head latent attention of HELM (He et al. 2025).

Dimension key:
  B batch  S query positions  T key positions  H heads  A model ambient dim
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers.hyperboloid_core import spatial_to_hyperboloid
from hyperbolix.nn_layers.hyperboloid_mla import LorentzMLA

# Small HELM-shaped config: dim ambient; qk_nope spatial; qk_rope and v ambient.
DIM, HEADS, KV_RANK, QK_NOPE, QK_ROPE, V_HEAD = 9, 2, 4, 3, 5, 4
BATCH, SEQ = 2, 6


def _make_layer(dtype=jnp.float32, seed=0, **kwargs):
    return LorentzMLA(DIM, HEADS, KV_RANK, QK_NOPE, QK_ROPE, V_HEAD, rngs=nnx.Rngs(seed), param_dtype=dtype, **kwargs)


def _make_points(key, c, dtype=jnp.float32, batch=BATCH, seq=SEQ, scale=0.5):
    """(B, S, A) hyperboloid points at moderate radius."""
    spatial = jax.random.normal(key, (batch, seq, DIM - 1), dtype=dtype) * scale
    return spatial_to_hyperboloid(spatial, c, c)


def _randomize_params(layer, seed=1):
    """Move biases, RMSNorm scale, tau and b off their init so the reference match exercises them."""
    rng = np.random.default_rng(seed)
    for proj in (layer.wq, layer.wkv_a, layer.wkv_b, layer.wo):
        bias = proj.bias
        bias[...] = jnp.asarray(0.1 * rng.standard_normal(bias.shape), dtype=bias[...].dtype)
    scale = layer.kv_norm.rms.scale
    scale[...] = jnp.asarray(1.0 + 0.1 * rng.standard_normal(scale.shape), dtype=scale[...].dtype)
    layer.softmax_scale[...] = jnp.asarray(3.0, dtype=layer.softmax_scale[...].dtype)
    layer.attn_bias[...] = jnp.asarray(0.3, dtype=layer.attn_bias[...].dtype)


def _lorentz_residual(y, c):
    """|<y, y>_L + 1/c| per point."""
    y = np.asarray(y, dtype=np.float64)
    return np.abs(-(y[..., 0] ** 2) + np.sum(y[..., 1:] ** 2, axis=-1) + 1.0 / c)


# ---------------------------------------------------------------------------
# Float64 NumPy transcription of HELM's forward (helm/modules/hmla.py, q_lora_rank = 0),
# written against the reference's own code, not the layer: torch-layout weights, the complex
# rotary multiply, HELM's `project`, `cinner`, clamped `lorentzian_centroid`, LorentzRMSNorm and
# LorentzLinear time rebuilds, and masked_fill(mask, -1e18). HELM's k is 1/c here.
# ---------------------------------------------------------------------------


def _ref_linear(x, kernel, bias):
    weight = np.asarray(kernel, np.float64).T  # torch nn.Linear layout (out, in)
    return x @ weight.T + np.asarray(bias, np.float64)


def _ref_apply_rotary_emb(x, freqs_cis):
    """x (B, S, H, d); freqs_cis (S, d/2) complex. torch.view_as_complex over adjacent pairs."""
    x_complex = x[..., 0::2] + 1j * x[..., 1::2]
    y = x_complex * freqs_cis[None, :, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = y.real
    out[..., 1::2] = y.imag
    return out


def _ref_project(x_space, k):
    return np.concatenate([np.sqrt(np.sum(x_space**2, axis=-1, keepdims=True) + k), x_space], axis=-1)


def _helm_reference_forward(layer, x, c, positions, masked_BST, rope_base=10000.0, rms_eps=1e-8):
    x = np.asarray(x, np.float64)
    k = 1.0 / c
    bsz, seqlen, _ = x.shape
    n_heads, rank = HEADS, KV_RANK
    qk_head_dim = QK_NOPE + QK_ROPE
    p = lambda v: np.asarray(v[...], np.float64)  # noqa: E731

    d = QK_ROPE - 1
    freqs = 1.0 / (rope_base ** (np.arange(0, d, 2, dtype=np.float64) / d))
    freqs_cis = np.exp(1j * np.outer(np.asarray(positions, np.float64), freqs))  # (S, d/2)

    q = _ref_linear(x, layer.wq.kernel, layer.wq.bias)  # return_space=True
    q = q.reshape(bsz, seqlen, n_heads, qk_head_dim - 1)
    q_nope, q_pe = q[..., :QK_NOPE], q[..., QK_NOPE:]
    q_pe = _ref_apply_rotary_emb(q_pe, freqs_cis)
    kv = _ref_linear(x, layer.wkv_a.kernel, layer.wkv_a.bias)
    kv, k_pe = kv[..., :rank], kv[..., rank:]
    k_pe = _ref_apply_rotary_emb(k_pe[:, :, None, :], freqs_cis)
    q = np.concatenate([q_nope, q_pe], axis=-1)

    # LorentzRMSNorm(space_only=True): F.rms_norm, then time = sqrt(clamp(|n|^2 + k, 1e-4)).
    normed = kv / np.sqrt(np.mean(kv**2, axis=-1, keepdims=True) + rms_eps) * p(layer.kv_norm.rms.scale)
    kv_point = np.concatenate([np.sqrt(np.maximum(np.sum(normed**2, -1, keepdims=True) + k, 1e-4)), normed], -1)
    kv = _ref_linear(kv_point, layer.wkv_b.kernel, layer.wkv_b.bias)
    kv = kv.reshape(bsz, seqlen, n_heads, QK_NOPE + V_HEAD - 1)
    k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
    kk = np.concatenate([k_nope, np.broadcast_to(k_pe, (bsz, seqlen, n_heads, QK_ROPE - 1))], axis=-1)

    qs = _ref_project(q, k).transpose(0, 2, 1, 3)  # (B, H, S, L)
    ks = _ref_project(kk, k).transpose(0, 2, 1, 3)
    qs_neg = qs.copy()
    qs_neg[..., 0] *= -1  # cinner
    scores = 2 * k + 2 * (qs_neg @ np.swapaxes(ks, -1, -2))  # (B, H, S, T)
    scores = scores / p(layer.softmax_scale) + p(layer.attn_bias)
    if masked_BST is not None:
        scores = np.where(np.asarray(masked_BST)[:, None], -1e18, scores)  # shape_mask: [B,N,N] -> [B,1,N,N]
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    scores = scores / scores.sum(axis=-1, keepdims=True)

    vs = _ref_project(v, k).transpose(0, 2, 1, 3)  # (B, H, T, W)
    ave = scores @ vs
    l_inner = -(ave[..., :1] ** 2) + np.sum(ave[..., 1:] ** 2, axis=-1, keepdims=True)
    denom = np.sqrt(np.maximum(np.abs(-l_inner), 1e-8))
    centroid = math.sqrt(k) * ave / denom  # (B, H, S, W)
    flat = centroid.transpose(0, 2, 1, 3).reshape(bsz, seqlen, n_heads * V_HEAD)
    out_space = _ref_linear(flat, layer.wo.kernel, layer.wo.bias)
    out_time = np.sqrt(np.maximum(np.sum(out_space**2, -1, keepdims=True) + k, 1e-8))
    return np.concatenate([out_time, out_space], axis=-1)


def _masks(kind):
    """(causal, segment_ids, attention_mask, HELM masked_BST with True = masked)."""
    masked = np.zeros((BATCH, SEQ, SEQ), dtype=bool)
    causal = kind != "none"
    segment_ids = attention_mask = None
    if causal:
        masked |= np.triu(np.ones((SEQ, SEQ), dtype=bool), k=1)[None]
    if kind == "causal_segment_padding":
        segment_ids = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=np.int32)
        attention_mask = np.array([[True] * 6, [True] * 4 + [False] * 2])
        masked |= segment_ids[:, :, None] != segment_ids[:, None, :]  # train.py: block where seq_ids differ
        masked |= ~attention_mask[:, None, :]
        segment_ids, attention_mask = jnp.asarray(segment_ids), jnp.asarray(attention_mask)
    return causal, segment_ids, attention_mask, (masked if kind != "none" else None)


@pytest.mark.parametrize("mask_kind", ["none", "causal", "causal_segment_padding"])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_matches_helm_reference_float64(c, mask_kind):
    """Same weights through the layer and a float64 transcription of HELM's forward agree."""
    layer = _make_layer(jnp.float64)
    _randomize_params(layer)
    x = _make_points(jax.random.PRNGKey(3), c, jnp.float64)
    causal, segment_ids, attention_mask, masked = _masks(mask_kind)
    out = layer(x, c, causal=causal, segment_ids=segment_ids, attention_mask=attention_mask)
    ref = _helm_reference_forward(layer, x, c, np.arange(SEQ), masked)
    assert out.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-10, rtol=1e-10)


def test_matches_helm_reference_float32():
    """Float32 layer against the float64 reference at moderate radius."""
    c = 1.0
    layer = _make_layer(jnp.float32)
    _randomize_params(layer)
    x = _make_points(jax.random.PRNGKey(4), c, jnp.float32)
    causal, segment_ids, attention_mask, masked = _masks("causal_segment_padding")
    out = layer(x, c, causal=causal, segment_ids=segment_ids, attention_mask=attention_mask)
    ref = _helm_reference_forward(layer, x, c, np.arange(SEQ), masked)
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(out, np.float64), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_output_shape_and_on_manifold(c, dtype):
    layer = _make_layer(dtype)
    x = _make_points(jax.random.PRNGKey(0), c, dtype)
    out = layer(x, c)
    assert out.shape == (BATCH, SEQ, DIM)
    assert out.dtype == dtype
    manifold = Hyperboloid(dtype=dtype)
    atol = 1e-4 if dtype == jnp.float32 else 1e-10
    flat = out.reshape(-1, DIM)
    assert bool(jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None, None))(flat, c, atol)))
    assert np.max(_lorentz_residual(out, c)) < atol


def test_init_matches_helm_reference():
    """Xavier-uniform gain sqrt(2) per projection (ambient in, spatial out), zero biases, tau, b."""
    layer = _make_layer()
    shapes = {
        "wq": (DIM, HEADS * (QK_NOPE + QK_ROPE - 1)),
        "wkv_a": (DIM, KV_RANK + QK_ROPE - 1),
        "wkv_b": (KV_RANK + 1, HEADS * (QK_NOPE + V_HEAD - 1)),
        "wo": (HEADS * V_HEAD, DIM - 1),
    }
    for name, (fan_in, fan_out) in shapes.items():
        proj = getattr(layer, name)
        kernel = np.asarray(proj.kernel[...])
        bound = math.sqrt(12.0 / (fan_in + fan_out))
        assert kernel.shape == (fan_in, fan_out)
        assert np.max(np.abs(kernel)) <= bound
        assert np.max(np.abs(kernel)) > 0.5 * bound
        assert np.all(np.asarray(proj.bias[...]) == 0.0)
    assert float(layer.softmax_scale[...]) == pytest.approx(math.sqrt(HEADS * (QK_NOPE + QK_ROPE)))
    assert float(layer.attn_bias[...]) == 0.0
    fixed = _make_layer(init_bound=0.05)
    assert np.max(np.abs(np.asarray(fixed.wq.kernel[...]))) <= 0.05


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qk_rope_head_dim": 4},  # spatial 3: odd
        {"qk_rope_head_dim": 1},  # spatial 0
        {"v_head_dim": 1},
        {"kv_lora_rank": 0},
        {"dim": 1},
    ],
)
def test_rejects_invalid_dims(kwargs):
    config = {
        "dim": DIM,
        "num_heads": HEADS,
        "kv_lora_rank": KV_RANK,
        "qk_nope_head_dim": QK_NOPE,
        "qk_rope_head_dim": QK_ROPE,
        "v_head_dim": V_HEAD,
    }
    config.update(kwargs)
    with pytest.raises(ValueError):
        LorentzMLA(**config, rngs=nnx.Rngs(0))


def test_rejects_bad_call_shapes():
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), 1.0)
    with pytest.raises(ValueError):
        layer(x[..., :-1], 1.0)
    with pytest.raises(ValueError):
        layer(x, 1.0, positions=jnp.arange(SEQ + 1))
    with pytest.raises(ValueError):
        layer(x, 1.0, segment_ids=jnp.zeros((BATCH, SEQ + 1), dtype=jnp.int32))


def _perturb(x, key, where_BS, c):
    """Replace the tokens at where_BS by different on-sheet points."""
    other = _make_points(key, c, x.dtype, scale=1.0)
    return jnp.where(where_BS[..., None], other, x)


def test_causal_no_leakage():
    c = 1.0
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c)
    t = 2
    future_BS = jnp.broadcast_to(jnp.arange(SEQ) > t, (BATCH, SEQ))
    x_pert = _perturb(x, jax.random.PRNGKey(9), future_BS, c)
    out, out_pert = layer(x, c), layer(x_pert, c)
    np.testing.assert_allclose(out[:, : t + 1], out_pert[:, : t + 1], atol=1e-6)
    assert float(jnp.max(jnp.abs(out[:, t + 1 :] - out_pert[:, t + 1 :]))) > 1e-3  # non-vacuous


def test_segment_no_leakage():
    c = 1.0
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c)
    segment_ids = jnp.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    # Non-causal, so the later segment could see the earlier one without the segment mask.
    first_segment_BS = segment_ids == 0
    x_pert = _perturb(x, jax.random.PRNGKey(9), first_segment_BS, c)
    out = layer(x, c, causal=False, segment_ids=segment_ids)
    out_pert = layer(x_pert, c, causal=False, segment_ids=segment_ids)
    second = ~np.asarray(first_segment_BS)
    np.testing.assert_allclose(np.asarray(out)[second], np.asarray(out_pert)[second], atol=1e-6)
    unmasked_diff = layer(x, c, causal=False) - layer(x_pert, c, causal=False)
    assert float(jnp.max(jnp.abs(np.asarray(unmasked_diff)[second]))) > 1e-3  # non-vacuous


def test_padding_mask_no_influence():
    c = 1.0
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c)
    attention_mask = jnp.array([[True] * 6, [True] * 4 + [False] * 2])
    x_pert = _perturb(x, jax.random.PRNGKey(9), ~attention_mask, c)
    out = layer(x, c, causal=False, attention_mask=attention_mask)
    out_pert = layer(x_pert, c, causal=False, attention_mask=attention_mask)
    valid = np.asarray(attention_mask)
    np.testing.assert_allclose(np.asarray(out)[valid], np.asarray(out_pert)[valid], atol=1e-6)
    unmasked_diff = layer(x, c, causal=False) - layer(x_pert, c, causal=False)
    assert float(jnp.max(jnp.abs(np.asarray(unmasked_diff)[valid]))) > 1e-3  # non-vacuous


def test_fully_masked_row_is_finite():
    """A query with no valid key softmaxes to a uniform row instead of NaN."""
    c = 1.0
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c)
    attention_mask = jnp.zeros((BATCH, SEQ), dtype=jnp.bool_)
    out = layer(x, c, attention_mask=attention_mask)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_hope_shift_invariance():
    """Shifting every position by a constant leaves the output unchanged (relative encoding)."""
    c = 1.0
    layer = _make_layer(jnp.float64)
    _randomize_params(layer)
    x = _make_points(jax.random.PRNGKey(5), c, jnp.float64)
    out = layer(x, c, positions=jnp.arange(SEQ, dtype=jnp.int32))
    out_shift = layer(x, c, positions=jnp.arange(SEQ, dtype=jnp.int32) + 37)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_shift), atol=1e-9)
    # Non-vacuous: permuting positions (a non-uniform change) does move the output.
    out_perm = layer(x, c, positions=jnp.array([0, 2, 1, 3, 5, 4], dtype=jnp.int32))
    assert float(jnp.max(jnp.abs(out - out_perm))) > 1e-6


def test_batched_positions_match_per_row_calls():
    c = 1.0
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c)
    positions_BS = jnp.stack([jnp.arange(SEQ), jnp.array([4, 1, 7, 2, 9, 3])]).astype(jnp.int32)
    out = layer(x, c, positions=positions_BS)
    for b in range(BATCH):
        out_row = layer(x[b : b + 1], c, positions=positions_BS[b])
        np.testing.assert_allclose(np.asarray(out[b]), np.asarray(out_row[0]), atol=1e-6)


def test_jit_parity():
    c = 0.5
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c)
    segment_ids = jnp.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=jnp.int32)

    @nnx.jit
    def forward(model, x, segment_ids):
        return model(x, c, segment_ids=segment_ids)

    np.testing.assert_allclose(
        np.asarray(forward(layer, x, segment_ids)), np.asarray(layer(x, c, segment_ids=segment_ids)), atol=1e-6
    )


def test_gradients_finite():
    c = 1.0
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c)

    def loss_fn(model, x):
        return jnp.sum(model(x, c)[..., 1:] ** 2)

    param_grads, x_grad = nnx.jit(nnx.grad(loss_fn, argnums=(0, 1)))(layer, x)
    leaves = jax.tree_util.tree_leaves_with_path(nnx.state(param_grads, nnx.Param))
    assert leaves
    for path, leaf in leaves:
        assert bool(jnp.all(jnp.isfinite(leaf))), path
        name = jax.tree_util.keystr(path)
        # HELM's scalar score bias is a softmax shift: its gradient is exactly zero by construction.
        if "attn_bias" not in name:
            assert float(jnp.max(jnp.abs(leaf))) > 0.0, name
    assert bool(jnp.all(jnp.isfinite(x_grad)))
    assert float(jnp.max(jnp.abs(x_grad))) > 0.0


def test_bfloat16_input_runs():
    c = 1.0
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c).astype(jnp.bfloat16)
    out = layer(x, c, segment_ids=jnp.zeros((BATCH, SEQ), dtype=jnp.int32))
    assert out.dtype == jnp.bfloat16
    assert out.shape == (BATCH, SEQ, DIM)
    assert bool(jnp.all(jnp.isfinite(out.astype(jnp.float32))))


def test_overfit_reduces_loss():
    c = 1.0
    layer = _make_layer()
    x = _make_points(jax.random.PRNGKey(0), c)
    target = _make_points(jax.random.PRNGKey(1), c)
    optimizer = nnx.Optimizer(layer, optax.adam(1e-2), wrt=nnx.Param)

    def loss_fn(model):
        return jnp.mean((model(x, c)[..., 1:] - target[..., 1:]) ** 2)

    @nnx.jit
    def step(model, optimizer):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    initial = float(loss_fn(layer))
    for _ in range(60):
        step(layer, optimizer)
    final = float(loss_fn(layer))
    assert np.isfinite(final)
    assert final < 0.5 * initial
