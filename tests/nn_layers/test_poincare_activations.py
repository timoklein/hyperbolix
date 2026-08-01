"""Tests for Poincaré ball activation functions.

Dimension key:
  B: batch size     C: channels (ball dim)
"""

from functools import partial

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

# (id, Poincaré activation called as f(x, c), the Euclidean function it must apply
# in the tangent space at the origin). The two leaky_relu rows differ only in
# ``negative_slope``, which is what pins the argument actually reaching jax.nn.
TANGENT_ACTIVATIONS = [
    ("relu", poincare_relu, jax.nn.relu),
    ("leaky_relu_default", poincare_leaky_relu, lambda z: jax.nn.leaky_relu(z, 0.01)),
    ("leaky_relu_slope0.3", partial(poincare_leaky_relu, negative_slope=0.3), lambda z: jax.nn.leaky_relu(z, 0.3)),
    ("tanh", poincare_tanh, jnp.tanh),
]


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


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5], ids=["c0.3", "c1.0", "c2.5"])
@pytest.mark.parametrize("name,activation,euclidean_fn", TANGENT_ACTIVATIONS, ids=[a[0] for a in TANGENT_ACTIVATIONS])
def test_activation_equals_expmap_of_euclidean_fn(name, activation, euclidean_fn, c):
    """f_P = exp_0^c ∘ f ∘ log_0^c, with f the plain Euclidean activation.

    Built from a *known* tangent vector ``t`` so the reference never calls
    ``logmap_0`` on the layer's own output: ``x = exp_0(t)`` gives ``log_0(x) = t``
    exactly, and the expected output is ``exp_0(f(t))``.

    Parametrized over c because the exp/log pair is the only place the curvature
    enters — a dropped or hard-coded ``c`` is invisible at c = 1.0. The two
    negative-slope rows pin the ``negative_slope`` argument (a slope that never
    reaches ``jax.nn.leaky_relu`` makes both rows equal).
    """
    manifold = Poincare(dtype=jnp.float64)
    t_BC = jnp.array([[0.3, -0.2, 0.9], [-0.1, 0.4, -0.7]], dtype=jnp.float64)
    x_BC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(t_BC, c)

    y_BC = activation(x_BC, c)

    expected_BC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(euclidean_fn(t_BC), c)
    assert jnp.allclose(y_BC, expected_BC, atol=1e-12)
    # Non-vacuity: the activation must actually move the negative components.
    assert not jnp.allclose(y_BC, x_BC, atol=1e-4)


def test_leaky_relu_slopes_disagree_on_negative_components():
    """Two ``negative_slope`` values give different outputs (the argument is live)."""
    manifold = Poincare(dtype=jnp.float64)
    c = 1.0
    t_BC = jnp.array([[-0.5, 0.4], [0.2, -0.8]], dtype=jnp.float64)
    x_BC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(t_BC, c)

    y_small = poincare_leaky_relu(x_BC, c, negative_slope=0.01)
    y_large = poincare_leaky_relu(x_BC, c, negative_slope=0.3)
    assert not jnp.allclose(y_small, y_large, atol=1e-4)


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
