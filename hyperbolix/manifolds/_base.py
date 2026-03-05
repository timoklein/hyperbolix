"""Shared base class for manifold implementations."""

import jax
import jax.numpy as jnp
from jaxtyping import Array


class ManifoldBase:
    """Base class providing shared __init__ and _cast for all manifolds.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)
    """

    def __init__(self, dtype: jnp.dtype = jnp.float32) -> None:
        self.dtype = dtype

    def _cast(self, x: Array) -> Array:
        """Cast array to target dtype if it's a floating-point array."""
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact):
            return x.astype(self.dtype)
        return x
