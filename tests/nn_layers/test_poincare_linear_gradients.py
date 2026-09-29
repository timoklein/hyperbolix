"""Test that HypLinearPoincare layers produce finite gradients under various conditions.

This is a regression test for the NaN gradient issue that occurred when using
HypLinearPoincarePP with the [None, :]/[0] vmap pattern in downstream applications.

The root cause was improper weight initialization (std=1.0 instead of std=1/sqrt(fan_in)),
which caused layer outputs to saturate at the Poincaré ball boundary, leading to
gradient overflow in float32 when backpropagating through logmap_0.

The overflow the file guards is a **float32** failure mode, so every test that can
carry a dtype axis runs on both a float32 and a float64 manifold (the file used to
run float64 only, i.e. in the precision where the bug cannot reproduce). The
seed-swept ``..._vmap_pattern_...`` test keeps its original float64 form untouched
and gets a separate float32 twin.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import HypLinearPoincare, HypLinearPoincarePP

poincare = Poincare(dtype=jnp.float64)
poincare_f32 = Poincare(dtype=jnp.float32)

DTYPES = [jnp.float32, jnp.float64]
DTYPE_IDS = ["f32", "f64"]


def _manifold(dtype):
    return poincare_f32 if dtype == jnp.float32 else poincare


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_hyp_linear_poincare_single_layer_gradients_finite(dtype):
    """Single layer should produce finite gradients."""
    key = jax.random.PRNGKey(0)
    poincare = _manifold(dtype)
    layer = HypLinearPoincare(poincare, 10, 20, rngs=nnx.Rngs(key))

    x = jax.random.normal(key, (8, 10), dtype=dtype) * 0.1
    x_manifold = jax.vmap(poincare.expmap_0, in_axes=(0, None))(x, 1.0)

    def loss_fn(m):
        out = m(x_manifold, 1.0)
        return jnp.mean(jnp.sum(out**2, axis=-1))

    _loss, grads = nnx.value_and_grad(loss_fn)(layer)
    grad_state = nnx.state(grads, nnx.Param)

    for _, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
        assert jnp.all(jnp.isfinite(value)), "Gradients contain NaN or Inf"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_hyp_linear_poincare_chained_layers_gradients_finite(dtype):
    """Chained layers should produce finite gradients (regression test)."""
    key = jax.random.PRNGKey(0)
    poincare = _manifold(dtype)
    layer1 = HypLinearPoincare(poincare, 10, 20, rngs=nnx.Rngs(key), input_space="tangent")
    layer2 = HypLinearPoincare(
        poincare,
        20,
        10,
        rngs=nnx.Rngs(jax.random.PRNGKey(1)),
        input_space="manifold",
    )

    x_tangent = jax.random.normal(key, (8, 10), dtype=dtype) * 0.5

    def loss_fn(l1, l2):
        h = l1(x_tangent, 1.0)
        # Hyperbolic ReLU
        h_tangent = jax.vmap(poincare.logmap_0, in_axes=(0, None))(h, 1.0)
        h_relu = nnx.relu(h_tangent)
        h_back = jax.vmap(poincare.expmap_0, in_axes=(0, None))(h_relu, 1.0)
        out = l2(h_back, 1.0)
        return jnp.mean(jnp.sum(out**2, axis=-1))

    _loss, (g1, g2) = jax.value_and_grad(loss_fn, argnums=(0, 1))(layer1, layer2)
    gs1 = nnx.state(g1, nnx.Param)
    gs2 = nnx.state(g2, nnx.Param)

    for _, value in jax.tree_util.tree_flatten_with_path(gs1)[0]:
        assert jnp.all(jnp.isfinite(value)), "Layer 1 gradients contain NaN or Inf"

    for _, value in jax.tree_util.tree_flatten_with_path(gs2)[0]:
        assert jnp.all(jnp.isfinite(value)), "Layer 2 gradients contain NaN or Inf"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_hyp_linear_poincare_pp_single_layer_gradients_finite(dtype):
    """Single layer should produce finite gradients."""
    key = jax.random.PRNGKey(0)
    poincare = _manifold(dtype)
    layer = HypLinearPoincarePP(poincare, 10, 20, rngs=nnx.Rngs(key), input_space="tangent")

    x = jax.random.normal(key, (8, 10), dtype=dtype) * 0.5

    def loss_fn(m):
        out = m(x, 1.0)
        return jnp.mean(jnp.sum(out**2, axis=-1))

    _loss, grads = nnx.value_and_grad(loss_fn)(layer)
    grad_state = nnx.state(grads, nnx.Param)

    for _, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
        assert jnp.all(jnp.isfinite(value)), "Gradients contain NaN or Inf"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_hyp_linear_poincare_pp_chained_layers_gradients_finite(dtype):
    """Chained layers should produce finite gradients (regression test for issue)."""
    key = jax.random.PRNGKey(0)
    poincare = _manifold(dtype)
    layer1 = HypLinearPoincarePP(poincare, 12, 32, rngs=nnx.Rngs(key), input_space="tangent")
    layer2 = HypLinearPoincarePP(poincare, 32, 8, rngs=nnx.Rngs(jax.random.PRNGKey(1)), input_space="manifold")

    x_tangent = jax.random.normal(key, (12,), dtype=dtype)

    def loss_fn(l1, l2):
        h = l1(x_tangent[None, :], 1.0)[0]
        # Hyperbolic ReLU
        h_tangent = poincare.logmap_0(h, 1.0)
        h_relu = nnx.relu(h_tangent)
        h_back = poincare.expmap_0(h_relu, 1.0)
        out = l2(h_back[None, :], 1.0)[0]
        return jnp.sum(out**2)

    _loss, (g1, g2) = jax.value_and_grad(loss_fn, argnums=(0, 1))(layer1, layer2)
    gs1 = nnx.state(g1, nnx.Param)
    gs2 = nnx.state(g2, nnx.Param)

    for _, value in jax.tree_util.tree_flatten_with_path(gs1)[0]:
        assert jnp.all(jnp.isfinite(value)), "Layer 1 gradients contain NaN or Inf"

    for _, value in jax.tree_util.tree_flatten_with_path(gs2)[0]:
        assert jnp.all(jnp.isfinite(value)), "Layer 2 gradients contain NaN or Inf"


@pytest.mark.parametrize("seed", range(10))
def test_hyp_linear_poincare_pp_vmap_pattern_gradients_finite(seed):
    """Test the [None,:]/[0] vmap pattern that was reported in the issue.

    This pattern appeared in downstream world model code where per-example
    functions would unsqueeze inputs for the batch API then squeeze outputs.
    """
    key = jax.random.PRNGKey(seed)

    class TestDynamics(nnx.Module):
        def __init__(self, in_dim, out_dim, *, rngs, c=1.0):
            self.c = c
            self.layer = HypLinearPoincarePP(poincare, in_dim, out_dim, rngs=rngs, input_space="tangent")

        def __call__(self, x):
            # x: (in_dim,) per-example - unsqueeze for batch API, then squeeze
            return self.layer(x[None, :], self.c)[0]

    model = TestDynamics(6, 64, rngs=nnx.Rngs(key))
    batch = jax.random.normal(key, (8, 6))

    def loss_fn(m):
        def per_example(x):
            out = m(x)
            return jnp.sum(out**2)

        losses = jax.vmap(per_example)(batch)
        return jnp.mean(losses)

    _loss, grads = nnx.value_and_grad(loss_fn)(model)
    grad_state = nnx.state(grads, nnx.Param)

    for _, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
        assert jnp.all(jnp.isfinite(value)), f"Seed {seed}: Gradients contain NaN or Inf"


@pytest.mark.parametrize("seed", range(10))
def test_hyp_linear_poincare_pp_vmap_pattern_gradients_finite_f32(seed):
    """The seed sweep above, on a **float32** manifold.

    The reported failure was a float32 overflow in the ``logmap_0`` backward pass
    — the precision the sibling test never ran in. Same model, same seeds, same
    assertion; only the manifold dtype (and therefore the input dtype) changes.
    """
    key = jax.random.PRNGKey(seed)

    class TestDynamics(nnx.Module):
        def __init__(self, in_dim, out_dim, *, rngs, c=1.0):
            self.c = c
            self.layer = HypLinearPoincarePP(poincare_f32, in_dim, out_dim, rngs=rngs, input_space="tangent")

        def __call__(self, x):
            return self.layer(x[None, :], self.c)[0]

    model = TestDynamics(6, 64, rngs=nnx.Rngs(key))
    batch = jax.random.normal(key, (8, 6), dtype=jnp.float32)

    def loss_fn(m):
        def per_example(x):
            out = m(x)
            return jnp.sum(out**2)

        return jnp.mean(jax.vmap(per_example)(batch))

    _loss, grads = nnx.value_and_grad(loss_fn)(model)
    grad_state = nnx.state(grads, nnx.Param)

    for _, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
        assert jnp.all(jnp.isfinite(value)), f"Seed {seed}: float32 gradients contain NaN or Inf"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_hyp_linear_poincare_pp_layer_outputs_on_manifold(dtype):
    """Layer outputs should be within the Poincaré ball."""
    key = jax.random.PRNGKey(0)
    poincare = _manifold(dtype)
    layer = HypLinearPoincarePP(poincare, 12, 32, rngs=nnx.Rngs(key), input_space="tangent")

    x = jax.random.normal(key, (8, 12), dtype=dtype)
    outputs = layer(x, 1.0)
    norms = jnp.linalg.norm(outputs, axis=-1)

    # Verify outputs are on the manifold
    assert jnp.all(norms < 1.0), "All outputs should be within unit ball"


@pytest.mark.parametrize("out_dim", [10, 64])
@pytest.mark.parametrize("c", [0.1, 1.0])
def test_hyp_linear_poincare_pp_large_scores_stay_at_boundary_f32(c, out_dim):
    """Large MLR scores must lift to the ball's edge in float32, not collapse to the origin.

    Regression: the lift squared w = sinh(√c·v)/√c, so ``sum(w**2)`` overflowed once √c·v passed ~44,
    and the output became exactly the origin with an exactly-zero gradient and no NaN. Kernel x10
    puts max √c·v near 56, past that threshold. Kernel x1000 puts most outputs past the float32 sinh
    clip, and at out_dim = 64 even ‖sinh(√c·v)‖ exceeds FLT_MAX. The jitted c = 1 wide case also
    catches XLA folding an ``(s/D)/X`` spelling back into the overflowing ``s/(D·X)``.
    """
    layer = HypLinearPoincarePP(poincare_f32, 16, out_dim, rngs=nnx.Rngs(0))
    u_BI = jax.random.normal(jax.random.PRNGKey(0), (4, 16), dtype=jnp.float32)
    x_BI = 0.9 / jnp.sqrt(c) * u_BI / jnp.linalg.norm(u_BI, axis=-1, keepdims=True)  # 0.9 of the ball radius
    kernel_init_OI = layer.kernel[...]

    def loss_fn(m):
        return jnp.sum(m(x_BI, c))

    for scale in (10.0, 1000.0):
        layer.kernel[...] = kernel_init_OI * scale
        y_BO = nnx.jit(lambda m: m(x_BI, c))(layer)
        _loss, grads = nnx.jit(nnx.value_and_grad(loss_fn))(layer)
        radius_B = jnp.sqrt(c) * jnp.linalg.norm(y_BO, axis=-1)

        assert jnp.all(jnp.isfinite(y_BO)), f"scale={scale}: non-finite output"
        assert jnp.all(radius_B > 0.999), f"scale={scale}: rows left the ball's edge (√c·‖y‖ = {radius_B})"
        for _, value in jax.tree_util.tree_flatten_with_path(nnx.state(grads, nnx.Param))[0]:
            assert jnp.all(jnp.isfinite(value)), f"scale={scale}: gradients contain NaN or Inf"


@pytest.mark.parametrize("c", [0.1, 1.0, 2.5])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_hyp_linear_poincare_pp_origin_jacobian_matches_closed_form(dtype, c):
    """At x = origin with the zero-init bias the scores are exactly 0, where the lift's slope is I/2.

    HNN++ at x = 0 (conformal factor 2, bias 0) gives dv/dx = 4·Z, so dy/dx = 2·Z. This guards the
    lift's derivative at w = 0: a normalized spelling (``ŵ·tanh(asinh(√c‖w‖)/2)/√c`` with
    ``safe_normalize``) is value-correct there but has a zero Jacobian, and a floor on one side
    only scales it by a power of √c.
    """
    manifold = _manifold(dtype)
    layer = HypLinearPoincarePP(manifold, 6, 4, rngs=nnx.Rngs(3), param_dtype=dtype)
    x_I = jnp.zeros((6,), dtype=dtype)

    jac_OI = jax.jacrev(lambda x: layer(x[None, :], c)[0])(x_I)

    atol = 1e-6 if dtype == jnp.float32 else 1e-13
    assert jnp.allclose(jac_OI, 2.0 * layer.kernel[...], rtol=0.0, atol=atol), jnp.max(
        jnp.abs(jac_OI - 2.0 * layer.kernel[...])
    )


def _max_rel_err(a, b):
    """``max|a - b| / max|b|`` in float64 — the error relative to the largest reference entry."""
    a, b = jnp.asarray(a, jnp.float64), jnp.asarray(b, jnp.float64)
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def _tangent_rows(key, batch, in_dim, t, c):
    """Float64 rows of scaled norm ``√c‖v‖ = t`` in random directions."""
    u_BI = jax.random.normal(key, (batch, in_dim), dtype=jnp.float64)
    return t / jnp.sqrt(c) * u_BI / jnp.linalg.norm(u_BI, axis=-1, keepdims=True)


@pytest.mark.parametrize("c", [0.3, 1.0])
def test_hyp_linear_poincare_pp_tangent_input_matches_the_ball_route(c):
    """float64, t = √c‖v‖ ≤ 3: the tangent-input layer equals the route through the ball it replaced.

    That route lifted v first — expmap_0 → compute_mlr_pp → sinh → _poincare_sinh_lift → proj —
    and read the conformal factor back off the stored point. Outputs and input gradients agree to
    1e-12 relative to their largest entry.
    """
    from hyperbolix.nn_layers.poincare_linear import _poincare_sinh_lift
    from hyperbolix.utils.math_utils import sinh

    dtype, batch, in_dim, out_dim = jnp.float64, 32, 12, 8
    layer = HypLinearPoincarePP(poincare, in_dim, out_dim, rngs=nnx.Rngs(0), input_space="tangent", param_dtype=dtype)
    key_b, key_u, key_t, key_w = jax.random.split(jax.random.PRNGKey(1), 4)
    layer.bias[...] = 0.3 * jax.random.normal(key_b, (out_dim, 1), dtype=dtype)
    t_B1 = jax.random.uniform(key_t, (batch, 1), dtype=dtype, minval=0.05, maxval=3.0)
    v_BI = _tangent_rows(key_u, batch, in_dim, t_B1, c)
    w_BO = jax.random.normal(key_w, (batch, out_dim), dtype=dtype)

    def ball_route(v_BI):
        x_BI = jax.vmap(poincare.expmap_0, in_axes=(0, None))(v_BI, c)
        s_BO = sinh(jnp.sqrt(c) * poincare.compute_mlr_pp(x_BI, layer.kernel[...], layer.bias[...], c))
        return jax.vmap(poincare.proj, in_axes=(0, None))(_poincare_sinh_lift(s_BO, c), c)

    y_err = _max_rel_err(layer(v_BI, c), ball_route(v_BI))
    g_new_BI = jax.grad(lambda v: jnp.sum(w_BO * layer(v, c)))(v_BI)
    g_old_BI = jax.grad(lambda v: jnp.sum(w_BO * ball_route(v)))(v_BI)
    assert y_err < 1e-12, y_err
    assert _max_rel_err(g_new_BI, g_old_BI) < 1e-12, _max_rel_err(g_new_BI, g_old_BI)


@pytest.mark.parametrize("c", [0.3, 1.0])
def test_hyp_linear_poincare_pp_tangent_input_far_point_f32(c):
    """float32 at t = √c‖v‖ = 8: outputs and input gradients match float64 on the same inputs and parameters.

    The lift onto the ball capped this input at t ≈ 6.33: outputs 8.0e-2 … 9.0e-2 and input
    gradients 4.3e-2 off (probe_new.out, old-route columns); here 1.4e-6 / 3.3e-5 at most. The
    kernel is scaled by 0.1 so the outputs land well inside the ball: at the default init they sit
    on the float32 ``proj`` margin, which bounds the output's accuracy, not the input's. The
    gradient bound is not eps-sized for the reason given in ``test_regression_layers.py::
    test_hyp_regression_poincare_pp_tangent_input_far_point_f32``.
    """
    batch, in_dim, out_dim, t = 32, 16, 12, 8.0
    layer32, layer64 = (
        HypLinearPoincarePP(_manifold(dt), in_dim, out_dim, rngs=nnx.Rngs(0), input_space="tangent", param_dtype=dt)
        for dt in DTYPES
    )
    layer32.kernel[...] = 0.1 * layer32.kernel[...]
    layer32.bias[...] = 0.2 * jax.random.normal(jax.random.PRNGKey(2), (out_dim, 1), dtype=jnp.float32)
    layer64.kernel[...] = layer32.kernel[...].astype(jnp.float64)
    layer64.bias[...] = layer32.bias[...].astype(jnp.float64)
    v32_BI = _tangent_rows(jax.random.PRNGKey(0), batch, in_dim, t, c).astype(jnp.float32)
    w32_BO = jax.random.normal(jax.random.PRNGKey(1), (batch, out_dim), dtype=jnp.float32)

    def out_and_grad(layer, v_BI, w_BO):
        return layer(v_BI, c), jax.grad(lambda v: jnp.sum(w_BO * layer(v, c)))(v_BI)

    y32, g32 = out_and_grad(layer32, v32_BI, w32_BO)
    y64, g64 = out_and_grad(layer64, v32_BI.astype(jnp.float64), w32_BO.astype(jnp.float64))

    assert _max_rel_err(y32, y64) < 1e-5, _max_rel_err(y32, y64)
    assert _max_rel_err(g32, g64) < 3e-4, _max_rel_err(g32, g64)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_world_model_gradients_finite(dtype):
    """Full world model should produce finite gradients."""
    poincare = _manifold(dtype)

    class HyperbolicDynamicsHead(nnx.Module):
        def __init__(self, latent_dim, branching_factor, hidden_dim, *, rngs, curvature=1.0):
            self.c = curvature
            self.hyp_linear1 = HypLinearPoincarePP(
                poincare,
                latent_dim + branching_factor,
                hidden_dim,
                rngs=rngs,
                input_space="tangent",
            )
            self.hyp_linear2 = HypLinearPoincarePP(
                poincare,
                hidden_dim,
                latent_dim,
                rngs=rngs,
                input_space="manifold",
            )

        def __call__(self, z, action_onehot):
            z_tangent = poincare.logmap_0(z, self.c)
            dyn_input = jnp.concatenate([z_tangent, action_onehot])
            h = self.hyp_linear1(dyn_input[None, :], self.c)[0]
            h = poincare.expmap_0(nnx.relu(poincare.logmap_0(h, self.c)), self.c)
            z_next = self.hyp_linear2(h[None, :], self.c)[0]
            return z_next

    class SimpleEncoder(nnx.Module):
        def __init__(self, obs_dim, latent_dim, *, rngs):
            self.linear1 = nnx.Linear(obs_dim, 128, rngs=rngs)
            self.linear2 = nnx.Linear(128, latent_dim, rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.linear1(x))
            return self.linear2(x)

    class SimpleDecoder(nnx.Module):
        def __init__(self, latent_dim, obs_dim, *, rngs):
            self.linear1 = nnx.Linear(latent_dim, 128, rngs=rngs)
            self.linear2 = nnx.Linear(128, obs_dim, rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.linear1(x))
            return self.linear2(x)

    class HyperbolicWorldModel(nnx.Module):
        def __init__(self, obs_dim, latent_dim, branching_factor, hidden_dim, *, rngs, curvature=1.0):
            self.c = curvature
            self.encoder = SimpleEncoder(obs_dim, latent_dim, rngs=rngs)
            self.dynamics = HyperbolicDynamicsHead(latent_dim, branching_factor, hidden_dim, rngs=rngs, curvature=curvature)
            self.decoder = SimpleDecoder(latent_dim, obs_dim, rngs=rngs)

        def __call__(self, obs, action_onehot):
            z_euc = self.encoder(obs)
            z = poincare.expmap_0(z_euc, self.c)
            z_next = self.dynamics(z, action_onehot)
            z_next_tangent = poincare.logmap_0(z_next, self.c)
            obs_pred = self.decoder(z_next_tangent)
            return obs_pred, z, z_next

    key = jax.random.PRNGKey(0)
    model = HyperbolicWorldModel(32, 8, 4, 32, rngs=nnx.Rngs(key))

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(100), 3)
    obs_batch = jax.random.normal(k1, (8, 32), dtype=dtype)
    action_batch = jax.nn.one_hot(jax.random.randint(k2, (8,), 0, 4), 4, dtype=dtype)
    target_batch = jax.random.normal(k3, (8, 32), dtype=dtype)

    def loss_fn(m):
        def per_example(o, a, t):
            obs_pred, _, _ = m(o, a)
            return jnp.mean((obs_pred - t) ** 2)

        losses = jax.vmap(per_example)(obs_batch, action_batch, target_batch)
        return jnp.mean(losses)

    _loss, grads = nnx.value_and_grad(loss_fn)(model)
    grad_state = nnx.state(grads, nnx.Param)

    for path, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
        assert jnp.all(jnp.isfinite(value)), f"{path}: Gradients contain NaN or Inf"


def test_ganea_layers_curvature_tag_matches_call_time_c():
    """Legacy Ganea layers tag their manifold bias with the constructor curvature.

    Regression: the tag was hardcoded to 1.0 while __call__ accepted dynamic c,
    so Riemannian bias updates used the wrong curvature whenever c != 1.0.
    """
    from hyperbolix.nn_layers import HypRegressionPoincare
    from hyperbolix.optim import get_manifold_info

    manifold = Poincare()
    c = 0.1

    linear = HypLinearPoincare(manifold, 4, 3, rngs=nnx.Rngs(0), curvature=c)
    _, tag_c = get_manifold_info(linear.bias)
    assert tag_c == c

    regression = HypRegressionPoincare(manifold, 4, 3, rngs=nnx.Rngs(0), curvature=c)
    _, tag_c = get_manifold_info(regression.bias)
    assert tag_c == c

    # Callable curvature (learnable-c pattern) is resolved at read time.
    curv = nnx.Param(jnp.array(0.5))
    linear_learnable = HypLinearPoincare(manifold, 4, 3, rngs=nnx.Rngs(0), curvature=lambda: curv[...])
    _, tag_c = get_manifold_info(linear_learnable.bias)
    assert jnp.allclose(tag_c, 0.5)
