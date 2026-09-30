"""Tests for LorentzMoE, HELM's mixture of curvature experts (He et al. 2025), in the paper's form.

The reference transcriptions below follow the HELM paper's equations (hyperbolix ``c = -K``),
not the released code, which the layer deliberately does not copy (see the LorentzMoE docstring).

Dimension key:
  B batch  S sequence  N tokens  A model ambient dim  E routed experts  K selected experts
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from hyperbolix.nn_layers.hyperboloid_core import spatial_to_hyperboloid
from hyperbolix.nn_layers.hyperboloid_moe import (
    LorentzMoE,
    LorentzMoEGate,
    LorentzSwiGLU,
    MoERoutingStats,
    RoutingBias,
    moe_sequence_balance_loss,
)

DIM, INTER, NUM_ROUTED, NUM_SHARED, TOP_K = 9, 6, 4, 1, 2
BATCH, SEQ = 2, 8
DTYPES = [jnp.float32, jnp.float64]
DTYPE_IDS = ["float32", "float64"]
# Layer-vs-float64-reference tolerance per layer dtype.
RTOL = {jnp.float32: 2e-5, jnp.float64: 1e-10}
ATOL = {jnp.float32: 2e-5, jnp.float64: 1e-10}


def _make_points(key, c, dtype=jnp.float32, shape=(BATCH, SEQ), scale=0.5):
    """Hyperboloid points of shape (*shape, A) at moderate radius."""
    spatial = jax.random.normal(key, (*shape, DIM - 1), dtype=dtype) * scale
    return spatial_to_hyperboloid(spatial, c, c)


def _make_moe(dtype=jnp.float32, seed=0, num_shared=NUM_SHARED, top_k=TOP_K, **kwargs):
    return LorentzMoE(DIM, INTER, NUM_ROUTED, num_shared, top_k, rngs=nnx.Rngs(seed), param_dtype=dtype, **kwargs)


def _randomize_biases(module, seed=1):
    """Move every HTCLinear bias off its zero init so the reference match exercises it."""
    rng = np.random.default_rng(seed)
    for _, leaf in nnx.iter_graph(module):
        if isinstance(leaf, LorentzSwiGLU):
            for proj in (leaf.w1, leaf.w2, leaf.w3):
                bias = proj.bias
                bias[...] = jnp.asarray(0.1 * rng.standard_normal(bias.shape), dtype=bias[...].dtype)


def _sheet_residual(y, c):
    """|<y, y>_L + 1/c| per point, relative to y_0^2 (the size of the stored squares)."""
    y = np.asarray(y, dtype=np.float64)
    return np.abs(-(y[..., 0] ** 2) + np.sum(y[..., 1:] ** 2, axis=-1) + 1.0 / c) / y[..., 0] ** 2


def _f64(v):
    """A Variable's (or array's) value as a float64 NumPy array."""
    return np.asarray(v[...], dtype=np.float64)


# ---------------------------------------------------------------------------
# Float64 NumPy transcription of the HELM paper (Sec. 4), with c = -K:
#   gating  s_j = act(x_s . y_j);  top-k on s + b;  g_i = s_i / sum_{TopK} s * route_scale
#   expert  y_i = sqrt(c_i/c) * HFFN_i(sqrt(c/c_i) * x)   (whole ambient vectors scaled)
#   merge   (sum_i g_i y_i + sum_j z_j) / (sqrt(c) * ||sum_i g_i y_i + sum_j z_j||_L)
# ---------------------------------------------------------------------------


def _ref_gate(x, kernel, bias, top_k, score_func, route_scale):
    """Returns the dense (N, E) weights, the (N, E) mask and the (N, E) normalised affinities."""
    x = np.asarray(x, np.float64).reshape(-1, DIM)
    logits = x[:, 1:] @ np.asarray(kernel, np.float64)  # (N, E)
    if score_func == "softmax":
        scores = np.exp(logits - logits.max(-1, keepdims=True))
        scores = scores / scores.sum(-1, keepdims=True)
    else:
        scores = 1.0 / (1.0 + np.exp(-logits))
    idx = np.argsort(-(scores + np.asarray(bias, np.float64)), axis=-1)[:, :top_k]  # (N, K)
    selected = np.take_along_axis(scores, idx, axis=-1)
    weights = route_scale * selected / selected.sum(-1, keepdims=True)
    dense = np.zeros_like(scores)
    mask = np.zeros_like(scores)
    np.put_along_axis(dense, idx, weights, axis=-1)
    np.put_along_axis(mask, idx, 1.0, axis=-1)
    return dense, mask, scores / scores.sum(-1, keepdims=True)


def _ref_time(space, c):
    return np.concatenate([np.sqrt(np.sum(space**2, -1, keepdims=True) + 1.0 / c), space], axis=-1)


def _ref_expert(x, params, c_e, c):
    """params: dict of float64 (w1, b1, w2, b2, w3, b3) in the Flax (in, out) layout."""
    silu = lambda v: v / (1.0 + np.exp(-v))  # noqa: E731
    x_e = math.sqrt(c / c_e) * x  # the paper's sqrt(K/K_e) x
    hidden = silu(x_e @ params["w1"] + params["b1"]) * (x_e @ params["w3"] + params["b3"])
    out_e = _ref_time(_ref_time(hidden, c_e) @ params["w2"] + params["b2"], c_e)  # on the c_e sheet
    return math.sqrt(c_e / c) * out_e  # the paper's sqrt(K_e/K) map back


def _expert_params(stacked, i):
    return {
        "w1": _f64(stacked.w1.kernel)[i],
        "b1": _f64(stacked.w1.bias)[i],
        "w2": _f64(stacked.w2.kernel)[i],
        "b2": _f64(stacked.w2.bias)[i],
        "w3": _f64(stacked.w3.kernel)[i],
        "b3": _f64(stacked.w3.bias)[i],
    }


def _ref_moe(moe, x, c, score_func="softmax", route_scale=1.0):
    x = np.asarray(x, np.float64).reshape(-1, DIM)
    dense, _, _ = _ref_gate(x, _f64(moe.gate.kernel), _f64(moe.gate.bias), moe.top_k, score_func, route_scale)
    curvatures = np.asarray(moe.routed_curvatures(), np.float64)
    ave = np.zeros_like(x)
    for i in range(moe.num_routed):
        ave += dense[:, i : i + 1] * _ref_expert(x, _expert_params(moe.experts, i), curvatures[i], c)
    for j in range(moe.num_shared):
        ave += _ref_expert(x, _expert_params(moe.shared_experts, j), c, c)
    neg_inner = ave[:, :1] ** 2 - np.sum(ave[:, 1:] ** 2, -1, keepdims=True)  # -<ave, ave>_L
    return ave / (math.sqrt(c) * np.sqrt(neg_inner))


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("route_scale", [1.0, 2.5])
def test_gate_matches_paper_transcription(dtype, score_func, route_scale):
    gate = LorentzMoEGate(DIM, NUM_ROUTED, TOP_K, rngs=nnx.Rngs(0), score_func=score_func, route_scale=route_scale)
    x = _make_points(jax.random.key(1), 1.0, dtype, shape=(32,), scale=1.0)
    weights_NK, indices_NK, stats = gate(x)
    dense = np.zeros((32, NUM_ROUTED))
    np.put_along_axis(dense, np.asarray(indices_NK), np.asarray(weights_NK, np.float64), axis=-1)

    ref_dense, ref_mask, ref_affinity = _ref_gate(x, _f64(gate.kernel), _f64(gate.bias), TOP_K, score_func, route_scale)
    np.testing.assert_allclose(dense, ref_dense, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(stats.mask), ref_mask)
    np.testing.assert_allclose(np.asarray(stats.affinity, np.float64), ref_affinity, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(weights_NK).sum(-1), route_scale, rtol=1e-6)  # renormalised over the top-k
    assert weights_NK.dtype == jnp.promote_types(dtype, jnp.float32)


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
def test_gate_bias_changes_selection_not_weights(score_func):
    gate = LorentzMoEGate(DIM, NUM_ROUTED, TOP_K, rngs=nnx.Rngs(0), score_func=score_func)
    x = _make_points(jax.random.key(1), 1.0, jnp.float64, shape=(32,), scale=1.0)
    _, _, stats_before = gate(x)
    bias = np.array([0.0, 0.0, 0.0, 10.0])  # forces expert 3 into every selection
    gate.bias[...] = jnp.asarray(bias, dtype=gate.bias[...].dtype)
    weights_NK, indices_NK, stats = gate(x)

    assert np.all(np.asarray(stats.mask)[:, 3] == 1.0)
    assert not np.array_equal(np.asarray(stats.mask), np.asarray(stats_before.mask))
    dense = np.zeros((32, NUM_ROUTED))
    np.put_along_axis(dense, np.asarray(indices_NK), np.asarray(weights_NK, np.float64), axis=-1)
    ref_dense, _, _ = _ref_gate(x, _f64(gate.kernel), bias, TOP_K, score_func, 1.0)
    np.testing.assert_allclose(dense, ref_dense, rtol=1e-6, atol=1e-7)  # unbiased scores, renormalised
    # The affinities (the balance loss input) do not see the bias either.
    np.testing.assert_allclose(np.asarray(stats.affinity), np.asarray(stats_before.affinity), rtol=0, atol=0)


def test_gate_bias_is_state_without_gradient():
    moe = _make_moe(jnp.float32)
    moe.gate.bias[...] = jnp.asarray([0.3, -0.2, 0.1, 0.0], dtype=jnp.float32)
    assert isinstance(moe.gate.bias, RoutingBias)
    assert not isinstance(moe.gate.bias, nnx.Param)
    assert "bias" not in nnx.state(moe.gate, nnx.Param)
    assert "bias" in nnx.state(moe.gate, RoutingBias)

    x = _make_points(jax.random.key(2), 1.0, jnp.float32)

    def loss_fn(model):
        out, stats = model(x, 1.0)
        return jnp.sum(out[..., 1:] ** 2) + moe_sequence_balance_loss(stats, 1e-2)

    grads = nnx.grad(loss_fn)(moe)
    assert "bias" not in grads["gate"]
    assert float(jnp.abs(grads["gate"]["kernel"][...]).sum()) > 0

    bias_before = np.asarray(moe.gate.bias[...])
    optimizer = nnx.Optimizer(moe, optax.adam(1e-2), wrt=nnx.Param)
    optimizer.update(moe, grads)
    np.testing.assert_array_equal(np.asarray(moe.gate.bias[...]), bias_before)


def test_gate_update_bias_moves_toward_balance():
    num_experts = 4
    gate = LorentzMoEGate(DIM, num_experts, 1, rngs=nnx.Rngs(0))
    # Tokens aligned with expert 0's gate vector: expert 0 takes almost all of them.
    kernel_OE = np.asarray(gate.kernel[...], np.float64)
    noise = np.random.default_rng(0).standard_normal((64, DIM - 1))
    spatial = 3.0 * kernel_OE[:, 0] / np.linalg.norm(kernel_OE[:, 0]) + 0.3 * noise
    x = spatial_to_hyperboloid(jnp.asarray(spatial, dtype=jnp.float32), 1.0, 1.0)

    _, _, stats = gate(x)
    load0 = np.asarray(stats.mask).sum(0)
    assert load0[0] > load0.mean()
    speed = 0.01
    gate.update_bias(stats.mask, speed)
    expected = speed * np.sign(load0.mean() - load0)
    np.testing.assert_allclose(np.asarray(gate.bias[...]), expected, rtol=0, atol=1e-7)

    @nnx.jit
    def step(g, x):
        _, _, st = g(x)
        g.update_bias(st.mask, speed)
        return st.mask.sum(0)

    for _ in range(300):
        load = np.asarray(step(gate, x))
    assert load.max() - load.min() < load0.max() - load0.min()
    assert load.max() < load0.max()

    # Exactly balanced load leaves the bias unchanged.
    before = np.asarray(gate.bias[...])
    gate.update_bias(jnp.eye(num_experts, dtype=jnp.float32), speed)
    np.testing.assert_array_equal(np.asarray(gate.bias[...]), before)


def _hand_mask():
    """(2, 2, 4) routing mask, top_k = 2: loads 3, 1, 0, 4 over 8 selections."""
    rows = [[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]]
    return jnp.asarray(rows, dtype=jnp.float32).reshape(2, 2, 4)


def test_gate_update_bias_proportional_matches_hand_computation_and_keeps_sum():
    """HELM code's rule: ``b += speed * (mean(util) - util)``, ``util = load / sum(load)``."""
    gate = LorentzMoEGate(DIM, 4, 2, rngs=nnx.Rngs(0))
    bias0 = np.array([0.1, -0.2, 0.3, 0.05], dtype=np.float32)
    gate.bias[...] = jnp.asarray(bias0)
    speed = 0.005
    gate.update_bias(_hand_mask(), speed, rule="proportional")
    # util = (3, 1, 0, 4) / 8, mean(util) = 1/4.
    expected = bias0 + speed * np.array([0.25 - 3 / 8, 0.25 - 1 / 8, 0.25 - 0.0, 0.25 - 4 / 8])
    np.testing.assert_allclose(np.asarray(gate.bias[...]), expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(jnp.sum(gate.bias[...])), float(bias0.sum()), rtol=0, atol=1e-7)

    # A step with no selections leaves the bias unchanged (HELM returns early).
    before = np.asarray(gate.bias[...])
    gate.update_bias(jnp.zeros((3, 4), dtype=jnp.float32), speed, rule="proportional")
    np.testing.assert_array_equal(np.asarray(gate.bias[...]), before)


def test_gate_update_bias_sign_rule_unchanged_and_rule_validated():
    """``rule="sign"`` is the default and the DeepSeek-V3 rule; an unknown rule raises."""
    bias0 = jnp.asarray([0.1, -0.2, 0.3, 0.05], dtype=jnp.float32)
    gates = [LorentzMoEGate(DIM, 4, 2, rngs=nnx.Rngs(0)) for _ in range(2)]
    for gate in gates:
        gate.bias[...] = bias0
    gates[0].update_bias(_hand_mask(), 0.01)
    gates[1].update_bias(_hand_mask(), 0.01, rule="sign")
    expected = np.asarray(bias0) + 0.01 * np.sign(2.0 - np.array([3.0, 1.0, 0.0, 4.0]))
    np.testing.assert_array_equal(np.asarray(gates[0].bias[...]), np.asarray(gates[1].bias[...]))
    np.testing.assert_allclose(np.asarray(gates[0].bias[...]), expected, rtol=0, atol=1e-8)
    with pytest.raises(ValueError):
        gates[0].update_bias(_hand_mask(), 0.01, rule="linear")  # type: ignore[arg-type]


def test_moe_update_bias_forwards_rule():
    moe = LorentzMoE(DIM, INTER, 4, NUM_SHARED, 2, rngs=nnx.Rngs(0))
    gate = LorentzMoEGate(DIM, 4, 2, rngs=nnx.Rngs(0))
    moe.update_bias(_hand_mask(), 0.005, rule="proportional")
    gate.update_bias(_hand_mask(), 0.005, rule="proportional")
    np.testing.assert_array_equal(np.asarray(moe.gate.bias[...]), np.asarray(gate.bias[...]))


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("c,c_expert", [(1.0, 1.0), (1.0, 0.1), (0.5, 2.0), (2.0, 0.7)])
def test_expert_output_on_model_sheet(dtype, c, c_expert):
    expert = LorentzSwiGLU(DIM, INTER, rngs=nnx.Rngs(0), param_dtype=dtype)
    _randomize_biases(expert)
    x = _make_points(jax.random.key(3), c, dtype, shape=(16,))
    y = expert(x, c_expert, c)
    assert y.shape == x.shape and y.dtype == dtype
    tol = 1e-6 if dtype == jnp.float32 else 1e-14
    assert np.max(_sheet_residual(y, c)) < tol


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("c,c_expert", [(1.0, 0.1), (0.5, 2.0), (1.0, 1.0)])
def test_expert_matches_paper_transcription(dtype, c, c_expert):
    expert = LorentzSwiGLU(DIM, INTER, rngs=nnx.Rngs(0), param_dtype=dtype)
    _randomize_biases(expert)
    x = _make_points(jax.random.key(4), c, dtype, shape=(16,))
    y = expert(x, c_expert, c)
    params = {
        "w1": _f64(expert.w1.kernel),
        "b1": _f64(expert.w1.bias),
        "w2": _f64(expert.w2.kernel),
        "b2": _f64(expert.w2.bias),
        "w3": _f64(expert.w3.kernel),
        "b3": _f64(expert.w3.bias),
    }
    ref = _ref_expert(np.asarray(x, np.float64), params, c_expert, c)
    np.testing.assert_allclose(np.asarray(y, np.float64), ref, rtol=RTOL[dtype], atol=ATOL[dtype])


def test_expert_init_is_helm_xavier():
    expert = LorentzSwiGLU(DIM, INTER, rngs=nnx.Rngs(0))
    for proj, (fan_in, fan_out) in ((expert.w1, (DIM, INTER)), (expert.w3, (DIM, INTER)), (expert.w2, (INTER + 1, DIM - 1))):
        bound = math.sqrt(12.0 / (fan_in + fan_out))
        kernel = np.asarray(proj.kernel[...])
        assert kernel.shape == (fan_in, fan_out)
        assert np.abs(kernel).max() <= bound
        assert np.abs(kernel).max() > 0.5 * bound
        np.testing.assert_array_equal(np.asarray(proj.bias[...]), 0.0)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def test_moe_default_curvatures_are_paper_linspace():
    moe = _make_moe()
    np.testing.assert_allclose(np.asarray(moe.routed_curvatures()), np.linspace(0.1, 2.0, NUM_ROUTED), rtol=1e-6)
    assert len(moe.curvatures) == NUM_ROUTED

    fixed = _make_moe(learnable_curvature=False, expert_curvatures=(0.3, 0.6, 0.9, 1.2))
    assert fixed.curvatures is None
    np.testing.assert_allclose(np.asarray(fixed.routed_curvatures()), [0.3, 0.6, 0.9, 1.2], rtol=1e-7)
    assert not any("curvatures" in str(path) for path, _ in nnx.to_flat_state(nnx.state(fixed, nnx.Param)))


def test_moe_rejects_bad_config():
    with pytest.raises(ValueError, match="top_k"):
        _make_moe(top_k=NUM_ROUTED + 1)
    with pytest.raises(ValueError, match="expert_curvatures"):
        _make_moe(expert_curvatures=(0.5, 1.0))
    with pytest.raises(ValueError, match="positive"):
        _make_moe(expert_curvatures=(0.5, 1.0, -1.0, 2.0), learnable_curvature=False)
    with pytest.raises(ValueError, match="route_scale"):
        _make_moe(route_scale=0.0)
    with pytest.raises(ValueError, match="score_func"):
        _make_moe(score_func="relu")


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("c", [1.0, 0.3])
@pytest.mark.parametrize("shape", [(BATCH, SEQ), (12,)], ids=["BSA", "NA"])
def test_moe_output_on_manifold(dtype, c, shape):
    moe = _make_moe(dtype)
    _randomize_biases(moe)
    x = _make_points(jax.random.key(5), c, dtype, shape=shape)
    out, stats = moe(x, c)
    assert out.shape == x.shape and out.dtype == dtype
    assert stats.affinity.shape == (*shape, NUM_ROUTED) and stats.mask.shape == (*shape, NUM_ROUTED)
    np.testing.assert_array_equal(np.asarray(stats.mask).sum(-1), TOP_K)
    tol = 1e-6 if dtype == jnp.float32 else 1e-14
    assert np.max(_sheet_residual(out, c)) < tol


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("num_shared", [0, 1, 2])
def test_moe_matches_paper_transcription(dtype, score_func, num_shared):
    route_scale = 1.5
    c = 0.7
    moe = _make_moe(dtype, num_shared=num_shared, score_func=score_func, route_scale=route_scale)
    _randomize_biases(moe)
    moe.gate.bias[...] = jnp.asarray([0.05, -0.05, 0.02, 0.0], dtype=moe.gate.bias[...].dtype)
    x = _make_points(jax.random.key(6), c, dtype)
    out, _ = moe(x, c)
    ref = _ref_moe(moe, x, c, score_func, route_scale).reshape(out.shape)
    np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=RTOL[dtype], atol=ATOL[dtype])


def test_moe_independent_of_expert_order():
    moe = _make_moe(jnp.float64)
    _randomize_biases(moe)
    moe.gate.bias[...] = jnp.asarray([0.05, -0.05, 0.02, 0.0], dtype=jnp.float64)
    x = _make_points(jax.random.key(7), 1.0, jnp.float64)
    out, stats = moe(x, 1.0)

    perm = np.array([2, 0, 3, 1])
    permuted = nnx.clone(moe)
    nnx.update(permuted.experts, jax.tree.map(lambda a: a[perm], nnx.state(moe.experts)))
    permuted.gate.kernel[...] = moe.gate.kernel[...][:, perm]
    permuted.gate.bias[...] = moe.gate.bias[...][perm]
    for j, i in enumerate(perm):
        permuted.curvatures[j].raw[...] = moe.curvatures[int(i)].raw[...]
    np.testing.assert_allclose(np.asarray(permuted.routed_curvatures()), np.asarray(moe.routed_curvatures())[perm])

    out_perm, stats_perm = permuted(x, 1.0)
    np.testing.assert_allclose(np.asarray(out_perm), np.asarray(out), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(stats_perm.mask), np.asarray(stats.mask)[..., perm])


def test_moe_unselected_experts_get_zero_gradient():
    moe = _make_moe(jnp.float32, top_k=1, expert_curvatures=(0.3, 0.7, 1.2, 1.8))
    _randomize_biases(moe)
    x = _make_points(jax.random.key(8), 1.0, jnp.float32, shape=(1,))  # one token
    _, stats = moe(x, 1.0)
    selected = int(np.argmax(np.asarray(stats.mask)[0]))
    direction = jax.random.normal(jax.random.key(9), x.shape, dtype=jnp.float32)

    grads = nnx.grad(lambda m: jnp.sum(m(x, 1.0)[0] * direction))(moe)
    for i in range(NUM_ROUTED):
        grad_norms = [
            float(jnp.abs(grads["experts"][name][leaf][...][i]).sum())
            for name in ("w1", "w2", "w3")
            for leaf in ("kernel", "bias")
        ]
        grad_norms.append(abs(float(grads["curvatures"][i]["raw"][...])))
        if i == selected:
            assert all(g > 0 for g in grad_norms), f"selected expert {i}: {grad_norms}"
        else:
            assert all(g == 0.0 for g in grad_norms), f"unselected expert {i}: {grad_norms}"


def _grads_over_batch(moe, seed=10):
    x = _make_points(jax.random.key(seed), 1.0, jnp.float32, shape=(4, 16), scale=1.0)
    _, stats = moe(x, 1.0)
    assert np.all(np.asarray(stats.mask).sum((0, 1)) > 0), "every routed expert must be selected by some token"
    direction = jax.random.normal(jax.random.key(seed + 1), x.shape, dtype=jnp.float32)

    def loss_fn(model):
        out, st = model(x, 1.0)
        return jnp.sum(out * direction) + moe_sequence_balance_loss(st, 1e-2)

    return nnx.grad(loss_fn)(moe)


@pytest.mark.parametrize("straight_through_clamp", [True, False], ids=["straight_through", "hard_clamp"])
@pytest.mark.parametrize("expert", range(NUM_ROUTED))
def test_moe_gradient_reaches_every_expert_curvature(expert, straight_through_clamp):
    """Default (paper) curvatures linspace(0.1, 2.0, E), default log parameterization, at init.

    Expert 0 starts at c = 0.1, the LearnableCurvature clamp floor. Under the hard clamp with the
    pre-fix LearnableCurvature, float32 rounds exp(log 0.1) below the floor and this gradient is 0.
    """
    moe = _make_moe(jnp.float32, straight_through_clamp=straight_through_clamp)
    _randomize_biases(moe)
    grads = _grads_over_batch(moe)
    raw_grad = float(grads["curvatures"][expert]["raw"][...])
    assert np.isfinite(raw_grad) and raw_grad != 0.0, f"expert {expert} (c = {float(moe.routed_curvatures()[expert])})"


def test_moe_gradients_reach_gate_and_all_expert_kernels():
    moe = _make_moe(jnp.float32, num_shared=2)
    _randomize_biases(moe)
    grads = _grads_over_batch(moe)
    assert float(jnp.abs(grads["gate"]["kernel"][...]).sum()) > 0
    for group, count in (("experts", NUM_ROUTED), ("shared_experts", 2)):
        for name in ("w1", "w2", "w3"):
            kernel_grad = np.asarray(grads[group][name]["kernel"][...])
            assert kernel_grad.shape[0] == count
            assert np.all(np.isfinite(kernel_grad))
            assert np.all(np.abs(kernel_grad).reshape(count, -1).sum(-1) > 0), f"{group}.{name}"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_moe_jit_matches_eager(dtype):
    moe = _make_moe(dtype)
    _randomize_biases(moe)
    x = _make_points(jax.random.key(11), 0.5, dtype)
    out, stats = moe(x, 0.5)
    out_jit, stats_jit = nnx.jit(lambda m, x: m(x, 0.5))(moe, x)
    tol = 1e-5 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(np.asarray(out_jit), np.asarray(out), rtol=tol, atol=tol)
    np.testing.assert_array_equal(np.asarray(stats_jit.mask), np.asarray(stats.mask))


def test_moe_bfloat16_smoke():
    moe = _make_moe(jnp.float32)
    x = _make_points(jax.random.key(12), 1.0, jnp.float32).astype(jnp.bfloat16)
    out, stats = moe(x, 1.0)
    assert out.dtype == jnp.bfloat16
    assert np.all(np.isfinite(np.asarray(out, np.float32)))
    assert np.all(np.isfinite(np.asarray(stats.affinity)))
    assert float(jnp.abs(out[..., 1:].astype(jnp.float32)).sum()) > 0


# ---------------------------------------------------------------------------
# Sequence-wise balance loss
# ---------------------------------------------------------------------------


def _stats(affinity, mask):
    return MoERoutingStats(affinity=jnp.asarray(affinity, jnp.float64), mask=jnp.asarray(mask, jnp.float64))


def test_balance_loss_hand_example():
    alpha = 0.3
    # Sequence 0 (E = 2, k = 1, T = 3): two tokens on expert 0 at s' = (0.6, 0.4), one on expert 1 at (0, 1).
    # f = 2 / 3 * (2, 1) = (4/3, 2/3), P = (1.2/3, 1.8/3) = (0.4, 0.6): sum f P = 0.5333 + 0.4 = 14/15.
    # Sequence 1: tokens on experts 0, 1, 0 at s' = (0.9, 0.1), (0.2, 0.8), (0.5, 0.5).
    # f = (4/3, 2/3), P = (1.6/3, 1.4/3): sum f P = 4/3 * 1.6/3 + 2/3 * 1.4/3.
    affinity = [
        [[0.6, 0.4], [0.6, 0.4], [0.0, 1.0]],
        [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]],
    ]
    mask = [
        [[1, 0], [1, 0], [0, 1]],
        [[1, 0], [0, 1], [1, 0]],
    ]
    seq0 = 14.0 / 15.0
    seq1 = 4.0 / 3.0 * 1.6 / 3.0 + 2.0 / 3.0 * 1.4 / 3.0
    loss = moe_sequence_balance_loss(_stats(affinity, mask), alpha)
    np.testing.assert_allclose(float(loss), alpha * (seq0 + seq1) / 2, rtol=1e-12)
    # A single sequence may be passed as (S, E).
    np.testing.assert_allclose(float(moe_sequence_balance_loss(_stats(affinity[0], mask[0]), alpha)), alpha * seq0, rtol=1e-12)
    assert seq0 < 1.0  # alpha is the balanced value, not a lower bound (see the docstring)


def test_balance_loss_balanced_routing_is_alpha_and_unbalanced_is_more():
    alpha, num_experts, top_k, seq = 1e-4, 4, 2, 8
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 1.0, (seq, num_experts))
    # Balanced: every expert in the top-2 of exactly 4 of the 8 tokens; affinities arbitrary.
    balanced_mask = np.zeros((seq, num_experts))
    for t in range(seq):
        balanced_mask[t, [(t % 4), (t + 1) % 4]] = 1.0
    assert np.all(balanced_mask.sum(0) == seq * top_k / num_experts)
    affinity = raw / raw.sum(-1, keepdims=True)
    np.testing.assert_allclose(float(moe_sequence_balance_loss(_stats(affinity, balanced_mask), alpha)), alpha, rtol=1e-12)

    # Uniform affinities give alpha for any routing.
    uniform = np.full((seq, num_experts), 1.0 / num_experts)
    skewed_mask = np.zeros((seq, num_experts))
    skewed_mask[:, :2] = 1.0
    np.testing.assert_allclose(float(moe_sequence_balance_loss(_stats(uniform, skewed_mask), alpha)), alpha, rtol=1e-12)

    # Maximally unbalanced: every token routed to experts {0, 1}, which are its top-2 affinities.
    skewed = raw.copy()
    skewed[:, :2] += 1.0
    skewed_affinity = skewed / skewed.sum(-1, keepdims=True)
    assert np.all(np.argsort(-skewed_affinity, axis=-1)[:, :2].max(-1) <= 1)
    loss = float(moe_sequence_balance_loss(_stats(skewed_affinity, skewed_mask), alpha))
    assert loss > alpha
    expected = alpha * (num_experts / top_k) * skewed_affinity[:, :2].sum(-1).mean()
    np.testing.assert_allclose(loss, expected, rtol=1e-12)


def test_balance_loss_gradient_reaches_gate_only_through_affinity():
    moe = _make_moe(jnp.float32)
    x = _make_points(jax.random.key(13), 1.0, jnp.float32, scale=1.0)

    def loss_fn(model):
        _, stats = model(x, 1.0)
        return moe_sequence_balance_loss(stats, 1.0)

    loss, grads = nnx.value_and_grad(loss_fn)(moe)
    assert loss.shape == () and np.isfinite(float(loss))
    assert float(jnp.abs(grads["gate"]["kernel"][...]).sum()) > 0
    assert float(jnp.abs(grads["experts"]["w1"]["kernel"][...]).sum()) == 0.0
