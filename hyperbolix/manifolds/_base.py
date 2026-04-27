"""Shared base class for manifold implementations."""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


def _inv_softplus(x: float) -> float:
    """Compute inv_softplus(x) = log(exp(x) - 1) in Python floats."""
    if x <= 0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}")
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class ManifoldBase(nnx.Module):
    """Base class providing shared __init__, _cast, and optional learnable curvature.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)
        c: Initial curvature value (default: 1.0). Must be positive.
        learnable: If True, curvature becomes a trainable ``nnx.Param``
            optimized via softplus reparameterization. Default: False.
    """

    def __init__(
        self,
        dtype: jnp.dtype = jnp.float32,
        *,
        c: float = 1.0,
        learnable: bool = False,
    ) -> None:
        self.dtype = dtype
        self._learnable = learnable
        if learnable:
            if c <= 0:
                raise ValueError(f"Learnable curvature requires c > 0, got {c}")
            self._c_raw = nnx.Param(jnp.array(_inv_softplus(c), dtype=jnp.float32))
        else:
            self._c_val = c

    @property
    def c(self) -> jax.Array | float:
        """Current curvature value. Positive when learnable (via softplus)."""
        if self._learnable:
            return jax.nn.softplus(self._c_raw[...])
        return self._c_val

    def _cast(self, x: Array) -> Array:
        """Cast array to target dtype if it's a floating-point array."""
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact):
            return x.astype(self.dtype)
        return x
