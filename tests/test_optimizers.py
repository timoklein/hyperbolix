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
import pytest
from flax import nnx

from hyperbolix.manifolds import poincare as poincare_module
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.optim import (
    get_manifold_info,
    mark_manifold_param,
    riemannian_adam,
    riemannian_sgd,
)
from hyperbolix.optim.riemannian_adam import RAdamState
from hyperbolix.optim.riemannian_sgd import RSGDState

poincare = Poincare(dtype=jnp.float64)


def test_mark_and_retrieve_metadata():
    """Test marking a parameter and retrieving metadata."""
    # Create a parameter and mark it
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, 0.2])),
        manifold_type="poincare",
        curvature=1.0,
    )

    # Retrieve metadata
    manifold_info = get_manifold_info(param)
    assert manifold_info is not None
    manifold_module, c = manifold_info
    assert manifold_module is poincare_module
    assert c == 1.0


def test_unmarked_parameter():
    """Test that unmarked parameters return None."""
    param = nnx.Param(jnp.array([0.1, 0.2]))
    manifold_info = get_manifold_info(param)
    assert manifold_info is None


def test_callable_curvature():
    """Test marking with callable curvature."""
    c_value = jnp.array(2.0)

    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, 0.2])),
        manifold_type="poincare",
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
    manifold_module, c = bias_info
    assert manifold_module is poincare_module
    assert c == 1.0

    # Check that weight does NOT have manifold metadata
    weight_info = get_manifold_info(layer.weight)
    assert weight_info is None


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
        manifold_type="poincare",
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
        param[...] = param[...] + updates

    # Check that loss decreased
    final_loss = loss_fn(param[...])
    assert final_loss < initial_loss * 0.1, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"

    # Check that point is still on manifold
    assert poincare.is_in_manifold(param[...], c)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_rsgd_momentum_transport(momentum):
    """Test that momentum is parallel transported correctly."""
    target = jnp.array([0.3, 0.4])
    c = 1.0
    x_init = jnp.array([0.1, 0.1])

    param = mark_manifold_param(
        nnx.Param(x_init),
        manifold_type="poincare",
        curvature=c,
    )

    tx = riemannian_sgd(learning_rate=0.1, momentum=momentum, use_expmap=True)
    opt_state = tx.init(param)

    def loss_fn(x):
        return poincare.dist(x, target, c) ** 2

    # Run several steps
    for _ in range(10):
        grad = jax.grad(loss_fn)(param[...])
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates

    # With momentum > 0, the momentum state should be non-zero
    if momentum > 0:
        state = cast(RSGDState, opt_state)
        momentum_norm = jnp.linalg.norm(state.momentum)
        assert momentum_norm > 0, "Momentum should be non-zero"


def test_rsgd_mixed_parameters():
    """Test RSGD with mixed Euclidean and manifold parameters."""

    # Create a simple model with mixed parameters
    class MixedModel(nnx.Module):
        def __init__(self, rngs):
            self.euclidean = nnx.Param(jnp.array([1.0, 2.0]))
            self.manifold = mark_manifold_param(
                nnx.Param(jnp.array([0.1, 0.1])),
                manifold_type="poincare",
                curvature=1.0,
            )

    model = MixedModel(rngs=nnx.Rngs(0))

    # Create optimizer
    tx = riemannian_sgd(learning_rate=0.1, momentum=0.0, use_expmap=True)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Simple loss
    def loss_fn(m):
        return jnp.sum(m.euclidean[...] ** 2) + jnp.sum(m.manifold[...] ** 2)

    # Store initial values
    initial_euclidean = model.euclidean[...].copy()
    initial_manifold = model.manifold[...].copy()

    # Training step
    _, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    # Check both parameters were updated
    assert not jnp.allclose(model.euclidean[...], initial_euclidean)
    assert not jnp.allclose(model.manifold[...], initial_manifold)

    # Check manifold parameter is still on manifold
    assert poincare.is_in_manifold(model.manifold[...], 1.0)


@pytest.mark.parametrize("use_expmap", [True, False])
def test_radam_convergence_to_target(use_expmap):
    """Test RAdam can move a point toward a target on Poincaré ball."""
    target = jnp.array([0.3, 0.4])
    c = 1.0
    x_init = jnp.array([0.1, 0.1])

    param = mark_manifold_param(
        nnx.Param(x_init),
        manifold_type="poincare",
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
        param[...] = param[...] + updates

    final_loss = loss_fn(param[...])
    assert final_loss < initial_loss * 0.1, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"
    assert poincare.is_in_manifold(param[...], c)


def test_radam_moment_transport():
    """Test that moments are parallel transported correctly."""
    target = jnp.array([0.3, 0.4])
    c = 1.0
    x_init = jnp.array([0.1, 0.1])

    param = mark_manifold_param(
        nnx.Param(x_init),
        manifold_type="poincare",
        curvature=c,
    )

    tx = riemannian_adam(
        learning_rate=0.1,
        use_expmap=True,
    )
    opt_state = tx.init(param)

    def loss_fn(x):
        return poincare.dist(x, target, c) ** 2

    # Run several steps
    for _ in range(10):
        grad = jax.grad(loss_fn)(param[...])
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates

    # Check that moments are non-zero
    state = cast(RAdamState, opt_state)
    m1_norm = jnp.linalg.norm(state.m1)
    m2_norm = jnp.linalg.norm(state.m2)
    assert m1_norm > 0, "First moment should be non-zero"
    assert m2_norm > 0, "Second moment should be non-zero"

    # Check that step count increased
    assert state.count == 10


def test_radam_mixed_parameters():
    """Test RAdam with mixed Euclidean and manifold parameters."""

    # Create a simple model with mixed parameters
    class MixedModel(nnx.Module):
        def __init__(self, rngs):
            self.euclidean = nnx.Param(jnp.array([1.0, 2.0]))
            self.manifold = mark_manifold_param(
                nnx.Param(jnp.array([0.1, 0.1])),
                manifold_type="poincare",
                curvature=1.0,
            )

    model = MixedModel(rngs=nnx.Rngs(0))

    tx = riemannian_adam(learning_rate=0.01)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    def loss_fn(m):
        return jnp.sum(m.euclidean[...] ** 2) + jnp.sum(m.manifold[...] ** 2)

    # Store initial values
    initial_euclidean = model.euclidean[...].copy()
    initial_manifold = model.manifold[...].copy()

    # Training step
    _, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    assert not jnp.allclose(model.euclidean[...], initial_euclidean)
    assert not jnp.allclose(model.manifold[...], initial_manifold)
    assert poincare.is_in_manifold(model.manifold[...], 1.0)


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

    # Dummy input and loss
    x = jax.random.normal(jax.random.key(1), (8, 5))

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
    assert poincare.is_in_manifold(layer.bias[...].squeeze(0), 1.0)


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
    assert poincare.is_in_manifold(layer.bias[...].squeeze(0), 1.0)


def test_jit_compilation():
    """Test that optimizer works with JIT compilation."""
    layer = HypLinearPoincare(
        poincare,
        in_dim=5,
        out_dim=3,
        rngs=nnx.Rngs(0),
    )

    tx = riemannian_sgd(learning_rate=0.01)
    optimizer = nnx.Optimizer(layer, tx, wrt=nnx.Param)

    x = jax.random.normal(jax.random.key(1), (8, 5))

    @nnx.jit
    def train_step(model, opt, x):
        def loss_fn(m):
            y = m(x, c=1.0)
            return jnp.sum(y**2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    # Run a few JIT-compiled steps
    for _ in range(5):
        loss = train_step(layer, optimizer, x)

    # Check that it ran without errors
    assert jnp.isfinite(loss)


def test_params_required_error():
    """Test that optimizers raise error when params not provided."""
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, 0.2])),
        manifold_type="poincare",
        curvature=1.0,
    )

    tx = riemannian_sgd(learning_rate=0.1)
    opt_state = tx.init(param)

    grad = jnp.array([0.1, 0.1])

    # Should raise error when params=None
    with pytest.raises(ValueError, match="requires params"):
        tx.update(grad, opt_state, params=None)


def test_parameter_stays_on_manifold():
    """Test that parameters stay on manifold after many updates."""
    c = 1.0
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.5, 0.5])),  # Start closer to boundary
        manifold_type="poincare",
        curvature=c,
    )

    tx = riemannian_sgd(learning_rate=0.1, use_expmap=True)
    opt_state = tx.init(param)

    target = jnp.array([0.7, 0.7])

    def loss_fn(x):
        return poincare.dist(x, target, c) ** 2

    # Run many steps
    for _ in range(100):
        grad = jax.grad(loss_fn)(param[...])
        updates, opt_state = tx.update(grad, opt_state, param)
        param[...] = param[...] + updates

        # Check at each step
        assert poincare.is_in_manifold(param[...], c), f"Parameter left manifold: {param[...]}"


def test_zero_gradient():
    """Test optimizer handles zero gradients gracefully."""
    param = mark_manifold_param(
        nnx.Param(jnp.array([0.1, 0.2])),
        manifold_type="poincare",
        curvature=1.0,
    )

    tx = riemannian_sgd(learning_rate=0.1)
    opt_state = tx.init(param)

    # Zero gradient
    grad = jnp.zeros_like(param[...])
    updates, opt_state = tx.update(grad, opt_state, param)

    # Parameter should not change
    new_value = param[...] + updates
    assert jnp.allclose(new_value, param[...])
