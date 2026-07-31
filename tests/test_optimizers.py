"""Tests for Riemannian optimizers.

This module tests the Riemannian SGD and Adam optimizers with:
1. Metadata detection and attachment
2. Simple convergence on Poincaré ball
3. Momentum transport verification
4. Mixed Euclidean/Riemannian parameters
5. Integration with nnx.Optimizer wrapper
6. Both expmap and retraction modes
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.optim import (
    ManifoldParam,
    get_manifold_info,
    has_manifold_params,
    mark_manifold_param,
    riemannian_adam,
    riemannian_sgd,
)
from hyperbolix.optim.riemannian_adam import RAdamState
from hyperbolix.optim.riemannian_sgd import RSGDState

poincare = Poincare(dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Independent reference arithmetic
#
# Deliberately transcribed from the papers in NumPy rather than obtained from
# ``Poincare.egrad2rgrad`` / ``Poincare.tangent_inner``: replacing ``egrad2rgrad``
# with the identity (dropping the metric factor 1/λ², i.e. the entire curvature
# dependence of the gradient) used to pass all 27 items of this file, because the
# Poincaré Riemannian gradient is a positive rescale of the Euclidean one and both
# optimizers' success criteria are scale-insensitive.
# ---------------------------------------------------------------------------


def _lam(x, c: float) -> float:
    """Conformal factor λ_x = 2/(1 - c‖x‖²) (Ganea et al. 2018, Eq. 1)."""
    x_np = np.asarray(x, dtype=np.float64)
    return 2.0 / (1.0 - c * float(np.dot(x_np, x_np)))


def _rgrad(g, x, c: float) -> jnp.ndarray:
    """Riemannian gradient of the conformal Poincaré metric: rgrad = g/λ_x²."""
    return jnp.asarray(np.asarray(g, dtype=np.float64) / _lam(x, c) ** 2)


def _tangent_sqnorm(v, x, c: float) -> float:
    """⟨v, v⟩_x = λ_x²·‖v‖² — the Riemannian norm RAdam accumulates in its second moment."""
    v_np = np.asarray(v, dtype=np.float64)
    return _lam(x, c) ** 2 * float(np.dot(v_np, v_np))


def test_mark_and_retrieve_metadata():
    """Test marking a parameter and retrieving metadata."""
    # Create a parameter and mark it
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, 0.2])),
        manifold=poincare,
        curvature=1.0,
    )

    # Retrieve metadata
    manifold_info = get_manifold_info(param)
    assert manifold_info is not None
    manifold_instance, c = manifold_info
    assert manifold_instance is poincare
    assert c == 1.0


def test_unmarked_parameter():
    """Test that unmarked parameters return None."""
    param = nnx.Param(jnp.array([0.1, 0.2]))
    manifold_info = get_manifold_info(param)
    assert manifold_info is None


def test_has_manifold_params_bare_param():
    """A bare ManifoldParam is detected.

    Regression: ``nnx.Variable`` is a registered pytree node whose child is the
    raw array, so an unguarded ``tree_leaves`` unwraps the ManifoldParam into an
    ArrayImpl and the detection returns False for *every* input.
    """
    param = ManifoldParam(jnp.array([0.1, 0.2]), manifold=poincare, curvature=1.0)
    assert has_manifold_params(param)


def test_has_manifold_params_in_containers():
    """Detection works through plain dict/list containers."""
    param = ManifoldParam(jnp.array([0.1, 0.2]), manifold=poincare, curvature=1.0)
    assert has_manifold_params({"bias": param})
    assert has_manifold_params([nnx.Param(jnp.zeros(3)), param])


def test_has_manifold_params_on_model_state():
    """The documented usage — nnx.state(model, nnx.Param) — and the model itself."""
    layer = HypLinearPoincare(
        poincare,
        in_dim=5,
        out_dim=3,
        rngs=nnx.Rngs(0),
    )

    assert has_manifold_params(nnx.state(layer, nnx.Param))
    assert has_manifold_params(layer)


def test_has_manifold_params_negative_cases():
    """No false positives on Euclidean params, raw arrays, or empty pytrees."""
    euclidean_layer = nnx.Linear(5, 3, rngs=nnx.Rngs(0))

    assert not has_manifold_params(nnx.state(euclidean_layer, nnx.Param))
    assert not has_manifold_params(nnx.Param(jnp.zeros(3)))
    assert not has_manifold_params({"kernel": jnp.zeros((5, 3))})
    assert not has_manifold_params({})


def test_callable_curvature():
    """Test marking with callable curvature."""
    c_value = jnp.array(2.0)

    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, 0.2])),
        manifold=poincare,
        curvature=lambda: c_value,
    )

    manifold_info = get_manifold_info(param)
    assert manifold_info is not None
    _, c = manifold_info
    assert c == 2.0


def test_layer_bias_has_metadata():
    """Test that HypLinearPoincare bias has manifold metadata."""
    layer = HypLinearPoincare(
        poincare,
        in_dim=5,
        out_dim=3,
        rngs=nnx.Rngs(0),
    )

    # Check that bias has manifold metadata
    bias_info = get_manifold_info(layer.bias)
    assert bias_info is not None
    manifold_instance, c = bias_info
    assert manifold_instance is poincare
    assert c == 1.0

    # Check that weight does NOT have manifold metadata
    weight_info = get_manifold_info(layer.kernel)
    assert weight_info is None


@pytest.mark.parametrize("use_expmap", [True, False])
def test_rsgd_single_step_equals_literal_riemannian_step(use_expmap):
    """One RSGD step equals ``move(-lr·g/λ_x², x)`` with λ transcribed independently.

    This is the file's only assertion that distinguishes a Riemannian step from a Euclidean
    one. Every other criterion here (loss decreased, point still in the ball, moments
    non-zero) survives ``egrad2rgrad → identity``, because on the Poincaré ball that
    mutation is a positive rescale of the descent direction — and for RAdam it is not even
    that, the step is *exactly* invariant to it.
    """
    c = 1.0
    lr = 0.1
    x0 = jnp.array([0.3, -0.4])
    g = jnp.array([0.7, 0.2])

    param = mark_manifold_param(nnx.Param(x0), manifold=poincare, curvature=c)
    tx = riemannian_sgd(learning_rate=lr, momentum=0.0, use_expmap=use_expmap)
    updates, _ = tx.update(g, tx.init(param), param)

    move = poincare.expmap if use_expmap else poincare.retraction
    expected = move(-lr * _rgrad(g, x0, c), x0, c)
    assert jnp.allclose(x0 + updates, expected, rtol=0, atol=1e-12)

    # And the mutation this pins is not a no-op: the Euclidean step is far away.
    euclidean_step = move(-lr * g, x0, c)
    assert not jnp.allclose(expected, euclidean_step, atol=1e-3)


@pytest.mark.parametrize("use_expmap", [True, False])
def test_rsgd_convergence_to_target(use_expmap):
    """Test RSGD can move a point toward a target on Poincaré ball."""
    # Target point on Poincaré ball
    target = jnp.array([0.3, 0.4])
    c = 1.0

    # Starting point
    x_init = jnp.array([0.1, 0.1])

    # Create a parameter and mark it as manifold
    param = mark_manifold_param(
        nnx.Param(x_init),
        manifold=poincare,
        curvature=c,
    )

    # Create optimizer
    tx = riemannian_sgd(learning_rate=0.1, momentum=0.0, use_expmap=use_expmap)
    opt_state = tx.init(param)

    # Loss: distance to target
    def loss_fn(x):
        return poincare.dist(x, target, c) ** 2

    # Run optimization steps
    initial_loss = loss_fn(param[...])
    for _ in range(50):
        # Compute gradient
        grad = jax.grad(loss_fn)(param[...])

        # Apply update
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates  # type: ignore[operator]

    # Check that loss decreased
    final_loss = loss_fn(param[...])
    assert final_loss < initial_loss * 0.1, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"

    # Check that point is still on manifold
    assert poincare.is_in_manifold(param[...], c)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_rsgd_momentum_transport(momentum):
    """Two RSGD steps reproduce a hand-computed, parallel-transported momentum buffer.

    Both parametrizations now assert. The old body ran ten steps and then executed *zero*
    assertions at ``momentum=0.0``; at ``0.9`` it only checked ``‖m‖ > 0``, which Poincaré
    ``ptransp`` (a positive rescale ``v ↦ (2/λ_y)·v``) cannot violate — deleting the transport
    call entirely passed.
    """
    target = jnp.array([0.3, 0.4])
    c = 1.0
    lr = 0.1
    x0 = jnp.array([0.1, 0.1])

    param = mark_manifold_param(nnx.Param(x0), manifold=poincare, curvature=c)
    tx = riemannian_sgd(learning_rate=lr, momentum=momentum, use_expmap=True)
    opt_state = tx.init(param)

    def loss_fn(x):
        return poincare.dist(x, target, c) ** 2

    for _ in range(2):
        grad = jax.grad(loss_fn)(param[...])
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates  # type: ignore[operator]

    # Two-step reference. riemannian_sgd only transports the buffer when momentum > 0
    # (riemannian_sgd.py:117), so at momentum = 0 the stored buffer is the raw second-step
    # Riemannian gradient — an exact equality, not a "should be non-zero".
    def transport(m, x_from, x_to):
        return poincare.ptransp(m, x_from, x_to, c) if momentum > 0.0 else m

    m1 = _rgrad(jax.grad(loss_fn)(x0), x0, c)
    x1 = poincare.expmap(-lr * m1, x0, c)
    m2 = momentum * transport(m1, x0, x1) + _rgrad(jax.grad(loss_fn)(x1), x1, c)
    x2 = poincare.expmap(-lr * m2, x1, c)

    assert jnp.allclose(param[...], x2, rtol=0, atol=1e-12)
    state = cast(RSGDState, opt_state)
    assert jnp.allclose(state.momentum, transport(m2, x1, x2), rtol=0, atol=1e-12)


def test_rsgd_mixed_parameters():
    """Each leaf of a mixed model takes its own exactly-predicted step.

    The old assertions were ``not allclose(param, initial)`` — satisfied by any non-zero
    garbage update, including applying the raw Euclidean gradient to the manifold leaf.
    """
    c = 1.0
    lr = 0.1

    class MixedModel(nnx.Module):
        def __init__(self, rngs):
            self.euclidean = nnx.Param(jnp.array([1.0, 2.0]))
            self.manifold = mark_manifold_param(
                nnx.Param(jnp.array([0.1, 0.1])),
                manifold=poincare,
                curvature=c,
            )

    model = MixedModel(rngs=nnx.Rngs(0))
    tx = riemannian_sgd(learning_rate=lr, momentum=0.0, use_expmap=True)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    def loss_fn(m):
        return jnp.sum(m.euclidean[...] ** 2) + jnp.sum(m.manifold[...] ** 2)

    initial_euclidean = model.euclidean[...].copy()
    initial_manifold = model.manifold[...].copy()

    _, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    # Euclidean leaf: plain SGD on ∂(Σx²)/∂x = 2x, no momentum ⇒ x ← x - lr·2x.
    assert jnp.allclose(model.euclidean[...], initial_euclidean - lr * 2.0 * initial_euclidean, rtol=0, atol=1e-12)

    # Manifold leaf: expmap of the *Riemannian* gradient, λ transcribed independently.
    g_manifold = 2.0 * initial_manifold
    expected = poincare.expmap(-lr * _rgrad(g_manifold, initial_manifold, c), initial_manifold, c)
    assert jnp.allclose(model.manifold[...], expected, rtol=0, atol=1e-12)
    assert not jnp.allclose(expected, poincare.expmap(-lr * g_manifold, initial_manifold, c), atol=1e-3)


@pytest.mark.parametrize("use_expmap", [True, False])
def test_radam_convergence_to_target(use_expmap):
    """Test RAdam can move a point toward a target on Poincaré ball."""
    target = jnp.array([0.3, 0.4])
    c = 1.0
    x_init = jnp.array([0.1, 0.1])

    param = mark_manifold_param(
        nnx.Param(x_init),
        manifold=poincare,
        curvature=c,
    )

    tx = riemannian_adam(learning_rate=0.1, use_expmap=use_expmap)
    opt_state = tx.init(param)

    def loss_fn(x):
        return poincare.dist(x, target, c) ** 2

    initial_loss = loss_fn(param[...])
    for _ in range(50):
        grad = jax.grad(loss_fn)(param[...])
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates  # type: ignore[operator]

    final_loss = loss_fn(param[...])
    assert final_loss < initial_loss * 0.1, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"
    assert poincare.is_in_manifold(param[...], c)


def test_radam_moment_transport():
    """Two RAdam steps reproduce hand-computed, parallel-transported moment buffers.

    The old body asserted ``‖m1‖ > 0`` and ``‖m2‖ > 0``, which removing the ``ptransp(m1, …)``
    call cannot violate — ``m1`` stays non-zero either way. Here the full recursion is written
    out: ``m1`` is transported to the new point after each step, ``m2`` accumulates the
    *Riemannian* inner product ⟨rgrad, rgrad⟩_x (a scalar broadcast, not elementwise g²) and is
    not transported.
    """
    target = jnp.array([0.3, 0.4])
    c = 1.0
    lr = 0.1
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    x0 = jnp.array([0.1, 0.1])

    param = mark_manifold_param(nnx.Param(x0), manifold=poincare, curvature=c)
    tx = riemannian_adam(learning_rate=lr, beta1=beta1, beta2=beta2, eps=eps, use_expmap=True)
    opt_state = tx.init(param)

    def loss_fn(x):
        return poincare.dist(x, target, c) ** 2

    for _ in range(2):
        grad = jax.grad(loss_fn)(param[...])
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates  # type: ignore[operator]

    # Two-step reference (Bécigneul & Ganea 2018 / riemannian_adam.py's documented recursion).
    x, m1, m2 = x0, jnp.zeros(2), jnp.zeros(2)
    for step in (1, 2):
        rgrad = _rgrad(jax.grad(loss_fn)(x), x, c)
        m1 = beta1 * m1 + (1 - beta1) * rgrad
        m2 = beta2 * m2 + (1 - beta2) * _tangent_sqnorm(rgrad, x, c)
        m1_hat = m1 / (1 - beta1**step)
        m2_hat = m2 / (1 - beta2**step)
        x_new = poincare.expmap(-lr * m1_hat / (jnp.sqrt(m2_hat) + eps), x, c)
        m1 = poincare.ptransp(m1, x, x_new, c)  # m1 is transported, m2 is not
        x = x_new

    state = cast(RAdamState, opt_state)
    assert jnp.allclose(param[...], x, rtol=0, atol=1e-12)
    assert jnp.allclose(state.m1, m1, rtol=0, atol=1e-12)
    assert jnp.allclose(state.m2, m2, rtol=0, atol=1e-12)
    assert state.count == 2


def test_radam_mixed_parameters():
    """Each leaf takes its own exactly-predicted step: optax.adam for Euclidean, the
    Riemannian recursion for the manifold leaf.

    The Euclidean oracle is an independent implementation (``optax.adam``), not a rewrite of
    ``euclidean_leaf_fn``; the old assertions were only ``not allclose(param, initial)``.
    """
    c = 1.0
    lr = 0.01
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    class MixedModel(nnx.Module):
        def __init__(self, rngs):
            self.euclidean = nnx.Param(jnp.array([1.0, 2.0]))
            self.manifold = mark_manifold_param(
                nnx.Param(jnp.array([0.1, 0.1])),
                manifold=poincare,
                curvature=c,
            )

    model = MixedModel(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, riemannian_adam(learning_rate=lr), wrt=nnx.Param)

    def loss_fn(m):
        return jnp.sum(m.euclidean[...] ** 2) + jnp.sum(m.manifold[...] ** 2)

    initial_euclidean = model.euclidean[...].copy()
    initial_manifold = model.manifold[...].copy()

    _, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    # Euclidean leaf must bit-match a plain optax.adam step on the same gradient.
    adam = optax.adam(lr)
    g_euclidean = 2.0 * initial_euclidean
    adam_update, _ = adam.update(g_euclidean, adam.init(initial_euclidean), initial_euclidean)
    assert jnp.allclose(model.euclidean[...], initial_euclidean + adam_update, rtol=0, atol=1e-12)

    # Manifold leaf: first RAdam step, bias corrections cancel to m1_hat = rgrad and
    # m2_hat = ⟨rgrad, rgrad⟩_x, so the step is expmap(-lr·rgrad/(√⟨rgrad,rgrad⟩_x + eps)).
    g_manifold = 2.0 * initial_manifold
    rgrad = _rgrad(g_manifold, initial_manifold, c)
    m1_hat = (1 - beta1) * rgrad / (1 - beta1)
    m2_hat = (1 - beta2) * _tangent_sqnorm(rgrad, initial_manifold, c) / (1 - beta2)
    expected = poincare.expmap(-lr * m1_hat / (jnp.sqrt(m2_hat) + eps), initial_manifold, c)
    assert jnp.allclose(model.manifold[...], expected, rtol=0, atol=1e-12)


def test_hyplinear_poincare_with_rsgd():
    """Test HypLinearPoincare with RSGD via nnx.Optimizer."""
    # Create layer
    layer = HypLinearPoincare(
        poincare,
        in_dim=5,
        out_dim=3,
        rngs=nnx.Rngs(0),
    )

    # Create optimizer (smaller LR for stability with scaled weight init)
    tx = riemannian_sgd(learning_rate=0.001, momentum=0.9)
    optimizer = nnx.Optimizer(layer, tx, wrt=nnx.Param)

    # Dummy input and loss. Scale down so features stay well inside the
    # Poincaré ball: an unscaled normal pushes the layer output onto the
    # boundary (‖y‖→1/√c), where ``proj`` clamps and the gradient vanishes,
    # leaving the loss frozen and the test passing or failing on init luck.
    x = jax.random.normal(jax.random.key(1), (8, 5)) * 0.1

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    # Initial loss
    initial_loss = loss_fn(layer)

    # Training steps
    for _ in range(100):
        _, grads = nnx.value_and_grad(loss_fn)(layer)
        optimizer.update(layer, grads)

    # Check loss decreased
    final_loss = loss_fn(layer)
    assert final_loss < initial_loss

    # Check bias is still on manifold
    assert poincare.is_in_manifold(layer.bias[...], 1.0)


def test_hyplinear_poincare_with_radam():
    """Test HypLinearPoincare with RAdam via nnx.Optimizer."""
    layer = HypLinearPoincare(
        poincare,
        in_dim=5,
        out_dim=3,
        rngs=nnx.Rngs(0),
    )

    # Smaller LR for stability with scaled weight init
    tx = riemannian_adam(learning_rate=0.001)
    optimizer = nnx.Optimizer(layer, tx, wrt=nnx.Param)

    x = jax.random.normal(jax.random.key(1), (8, 5))

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    initial_loss = loss_fn(layer)

    # Training steps
    for _ in range(100):
        _, grads = nnx.value_and_grad(loss_fn)(layer)
        optimizer.update(layer, grads)

    final_loss = loss_fn(layer)
    assert final_loss < initial_loss
    assert poincare.is_in_manifold(layer.bias[...], 1.0)


def test_jit_compilation():
    """Five jitted train steps land on the same parameters as five eager ones.

    ``assert isfinite(loss)`` (the old body) passes for an optimizer that returns zeros;
    ``nnx.jit`` is semantics-preserving, so without an eager arm the test only re-ran the
    eager path.
    """
    x = jax.random.normal(jax.random.key(1), (8, 5))

    def _build():
        layer = HypLinearPoincare(poincare, in_dim=5, out_dim=3, rngs=nnx.Rngs(0))
        return layer, nnx.Optimizer(layer, riemannian_sgd(learning_rate=0.01), wrt=nnx.Param)

    def loss_fn(m):
        return jnp.sum(m(x, c=1.0) ** 2)

    def step(model, opt):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    layer_jit, opt_jit = _build()
    layer_eager, opt_eager = _build()
    jit_step = nnx.jit(step)

    for _ in range(5):
        loss = jit_step(layer_jit, opt_jit)
        step(layer_eager, opt_eager)

    assert jnp.isfinite(loss)
    assert jnp.allclose(layer_jit.kernel[...], layer_eager.kernel[...], rtol=0, atol=1e-10)
    assert jnp.allclose(layer_jit.bias[...], layer_eager.bias[...], rtol=0, atol=1e-10)


def test_params_required_error():
    """Test that optimizers raise error when params not provided."""
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, 0.2])),
        manifold=poincare,
        curvature=1.0,
    )

    tx = riemannian_sgd(learning_rate=0.1)
    opt_state = tx.init(param)

    grad = jnp.array([0.1, 0.1])

    # Should raise error when params=None
    with pytest.raises(ValueError, match="requires params"):
        tx.update(grad, opt_state, params=None)


def test_no_nan_over_many_updates_near_boundary():
    """100 updates starting at ‖x‖ ≈ 0.7 stay finite and drive the loss down monotonically.

    Renamed from ``test_parameter_stays_on_manifold``: ``_expmap`` is documented as landing
    strictly inside the ball (‖second_term‖ = |tanh(·)|/√c < 1/√c) and ``_retraction`` ends in
    ``_proj``, so ``is_in_manifold`` after a move is true by construction for *any* direction
    and can only fire on NaN. That NaN guard near the boundary is the real content; the
    monotone loss is what makes the direction itself checkable.
    """
    c = 1.0
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.5, 0.5])),  # Start closer to boundary
        manifold=poincare,
        curvature=c,
    )

    tx = riemannian_sgd(learning_rate=0.1, use_expmap=True)
    opt_state = tx.init(param)

    target = jnp.array([0.7, 0.7])

    def loss_fn(x):
        return poincare.dist(x, target, c) ** 2

    losses = [float(loss_fn(param[...]))]
    for _ in range(100):
        grad = jax.grad(loss_fn)(param[...])
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates  # type: ignore[operator]

        assert jnp.all(jnp.isfinite(param[...])), f"Parameter went non-finite: {param[...]}"
        assert poincare.is_in_manifold(param[...], c), f"Parameter left manifold: {param[...]}"
        losses.append(float(loss_fn(param[...])))

    # Strictly monotone over the first 20 steps; past that the loss is at the float64 noise
    # floor (the step contracts the geodesic distance by a constant factor each iteration).
    assert all(losses[i + 1] < losses[i] for i in range(20)), f"Loss not monotone: {losses[:21]}"
    assert losses[-1] < 1e-9 * losses[0], f"Loss did not converge: {losses[0]} -> {losses[-1]}"


def test_zero_gradient():
    """Test optimizer handles zero gradients gracefully."""
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, 0.2])),
        manifold=poincare,
        curvature=1.0,
    )

    tx = riemannian_sgd(learning_rate=0.1)
    opt_state = tx.init(param)

    # Zero gradient
    grad = jnp.zeros_like(param[...])
    updates, opt_state = tx.update(grad, opt_state, param)

    # Parameter should not change
    new_value = param[...] + updates  # type: ignore[operator]
    assert jnp.allclose(new_value, param[...])


# ---------------------------------------------------------------------------
# Regression guards: manifold dispatch through nnx.Optimizer
#
# Flax's nnx.Optimizer (>= 0.12.5) strips Variables to raw arrays before the
# optax update, so isinstance(leaf, ManifoldParam) inside update_fn can never
# fire on that path. The metadata is now captured by init_fn instead. These
# tests pin the behaviors that the old in-update dispatch silently broke.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_tx", [lambda: riemannian_adam(0.1), lambda: riemannian_sgd(0.1, momentum=0.9)])
def test_nnx_optimizer_boundary_point_stays_on_manifold(make_tx):
    """A near-boundary point must take a Riemannian step through nnx.Optimizer.

    Regression guard: the old in-update isinstance dispatch never fired through
    nnx.Optimizer, so EVERY parameter silently took a Euclidean step — this
    exact setup sent ||x|| = 0.9 to ||x|| = 1.4, off the manifold, with no error.
    """
    c = 1.0

    class BoundaryModel(nnx.Module):
        def __init__(self):
            self.x = mark_manifold_param(
                nnx.Param(jnp.array([0.9, 0.0])),
                manifold=poincare,
                curvature=c,
            )

    model = BoundaryModel()
    optimizer = nnx.Optimizer(model, make_tx(), wrt=nnx.Param)
    target = jnp.array([0.0, 0.5])

    def loss_fn(m):
        return poincare.dist(m.x[...], target, c) ** 2

    grads = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads)

    assert jnp.linalg.norm(model.x[...]) < 1.0, f"Point left the ball: {model.x[...]}"
    assert poincare.is_in_manifold(model.x[...], c)


def test_nnx_optimizer_embedding_table():
    """(N, dim) manifold leaves work through nnx.Optimizer and stay on the ball.

    Regression guard: the manifold branch was single-point only — multi-point
    embedding tables crashed in egrad2rgrad with a matmul shape error.
    """
    c = 1.0

    class EmbModel(nnx.Module):
        def __init__(self):
            init = jax.random.normal(jax.random.key(0), (8, 4)) * 0.2
            init = jax.vmap(poincare.proj, in_axes=(0, None))(init, c)
            self.emb = mark_manifold_param(nnx.Param(init), manifold=poincare, curvature=c)
            self.head = nnx.Param(jnp.ones((4,)))  # mixed Euclidean param

    model = EmbModel()
    optimizer = nnx.Optimizer(model, riemannian_adam(learning_rate=0.05), wrt=nnx.Param)
    target = jnp.zeros(4).at[0].set(0.6)

    def loss_fn(m):
        dists = jax.vmap(poincare.dist, in_axes=(0, None, None))(m.emb[...], target, c)
        return jnp.sum(dists**2) + jnp.sum((m.head[...] - 2.0) ** 2)

    initial_loss = loss_fn(model)
    initial_head = model.head[...].copy()
    for _ in range(30):
        grads = nnx.grad(loss_fn)(model)
        optimizer.update(model, grads)

    assert loss_fn(model) < 0.2 * initial_loss
    assert not jnp.allclose(model.head[...], initial_head)  # Euclidean branch also updates
    on_ball = jax.vmap(lambda row: poincare.is_in_manifold(row, c))(model.emb[...])
    assert jnp.all(on_ball)


def test_raw_array_init_raises_instead_of_silent_euclidean():
    """Initializing with raw arrays (no Variable metadata) must fail loudly at
    update time rather than silently applying Euclidean updates to everything."""
    tx = riemannian_adam(learning_rate=0.1)
    raw = jnp.array([0.1, 0.2])
    opt_state = tx.init(raw)

    with pytest.raises(ValueError, match="no Variable metadata"):
        tx.update(raw * 0.1, opt_state, raw)
