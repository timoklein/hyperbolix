"""Curvature utility helpers for learnable curvature workflows.

Provides ``learnable_curvature()`` and ``get_curvature()`` as the recommended
way to add trainable curvature to hyperbolic models. Store the parameter on
your ``nnx.Module`` and pass ``get_curvature(self.c_raw)`` to manifold methods.

Example::

    from hyperbolix import learnable_curvature, get_curvature
    from hyperbolix.manifolds import Hyperboloid

    class Model(nnx.Module):
        def __init__(self, rngs):
            self.manifold = Hyperboloid(c=1.0)
            self.c_raw = learnable_curvature(init_c=1.0)
            self.fc = FGGLinear(33, 65, rngs=rngs)

        def __call__(self, x):
            return self.fc(x, c=get_curvature(self.c_raw))
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx


def _inv_softplus(x: float) -> float:
    """Compute inv_softplus(x) = log(exp(x) - 1) in Python floats."""
    if x <= 0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}")
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


def learnable_curvature(init_c: float = 1.0) -> nnx.Param:
    """Create a learnable curvature parameter with softplus reparameterization.

    Store on your model and use ``get_curvature()`` to recover the positive value.
    Updated by standard Euclidean optimizers (optax.adam, etc.).

    Args:
        init_c: Initial curvature value (must be positive).

    Returns:
        An ``nnx.Param`` storing ``inv_softplus(init_c)`` as float32.
    """
    if init_c <= 0:
        raise ValueError(f"Learnable curvature requires init_c > 0, got {init_c}")
    return nnx.Param(jnp.array(_inv_softplus(init_c), dtype=jnp.float32))


def get_curvature(c_raw: jax.Array | nnx.Param) -> jax.Array:
    """Recover positive curvature value from a raw curvature parameter.

    Args:
        c_raw: Raw parameter (from ``learnable_curvature()``). Accepts either
            an ``nnx.Param`` or a bare ``jax.Array``.

    Returns:
        Positive curvature via softplus.
    """
    val = c_raw[...] if isinstance(c_raw, nnx.Variable) else c_raw
    return jax.nn.softplus(val)
