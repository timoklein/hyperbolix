"""Tests for Poincaré ball activation functions.

Dimension key:
  B: batch size     C: channels (ball dim)
"""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers.poincare_activations import (
    poincare_leaky_relu,
    poincare_relu,
    poincare_tanh,
)

ACTIVATIONS = [poincare_relu, poincare_leaky_relu, poincare_tanh]


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=["relu", "leaky_relu", "tanh"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_activation_preserves_dtype(activation, dtype):
    """Output dtype equals input dtype.

    Regression guard: a module-level ``Poincare(dtype=jnp.float64)`` instance
    silently promoted float32 inputs to float64 under JAX_ENABLE_X64=1 (manifold
    methods cast inputs to the *manifold's* dtype, and nothing cast back).
    """
    x_BC = jnp.full((4, 3), 0.1, dtype=dtype)
    y_BC = activation(x_BC, c=1.0)
    assert y_BC.dtype == dtype
    assert y_BC.shape == x_BC.shape
    assert jnp.all(jnp.isfinite(y_BC))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_activation_manifold_module_override(dtype):
    """An explicit manifold_module controls the compute dtype."""
    x_BC = jnp.full((2, 3), 0.1, dtype=jnp.float32)
    y_BC = poincare_relu(x_BC, c=1.0, manifold_module=Poincare(dtype=dtype))
    assert y_BC.dtype == dtype


def test_relu_matches_tangent_space_composition():
    """poincare_relu equals exp_0 ∘ relu ∘ log_0 computed by hand."""
    manifold = Poincare(dtype=jnp.float64)
    c = 1.0
    x_BC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(jnp.array([[0.3, -0.2], [-0.1, 0.4]], dtype=jnp.float64), c)

    y_BC = poincare_relu(x_BC, c)

    t_BC = jax.vmap(manifold.logmap_0, in_axes=(0, None))(x_BC, c)
    expected_BC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(jax.nn.relu(t_BC), c)
    assert jnp.allclose(y_BC, expected_BC, atol=1e-10)


def test_activation_output_on_ball():
    """Activations return valid Poincaré ball points for 4D feature maps."""
    manifold = Poincare(dtype=jnp.float64)
    c = 1.0
    x_BC = jax.random.normal(jax.random.key(0), (2, 4, 4, 3), dtype=jnp.float64) * 0.3
    x_BC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(x_BC.reshape(-1, 3), c).reshape(2, 4, 4, 3)

    y_BC = poincare_tanh(x_BC, c)
    assert y_BC.shape == x_BC.shape
    flat = y_BC.reshape(-1, 3)
    on_ball = jax.vmap(lambda p: manifold.is_in_manifold(p, c, atol=1e-4))(flat)
    assert jnp.all(on_ball)
