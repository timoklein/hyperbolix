"""JAX manifold base definitions (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass
class Manifold:
    """Minimal interface mirror of the Torch `Manifold` for JAX.

    The geometry-specific operations remain unimplemented and will be
    provided during the core port in later phases. Methods raise
    ``NotImplementedError`` so the accompanying tests surface missing
    functionality while we iterate.
    """

    c: Any
    dtype: Any

    def __post_init__(self) -> None:
        self.c = jnp.asarray(self.c, dtype=self._resolve_dtype(self.dtype))
        self.dtype = self.c.dtype

    @staticmethod
    def _resolve_dtype(dtype: Any) -> jnp.dtype:
        if dtype in ("float32", "float64"):
            return jnp.dtype(dtype)
        return jnp.asarray(0, dtype=dtype).dtype

    # --- Geometry interface -------------------------------------------------
    def addition(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def scalar_mul(self, r: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def dist(self, x: jnp.ndarray, y: jnp.ndarray, *, version: str = "default") -> jnp.ndarray:
        raise NotImplementedError

    def dist_0(self, x: jnp.ndarray, *, version: str = "default") -> jnp.ndarray:
        raise NotImplementedError

    def expmap(self, v: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def expmap_0(self, v: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def retraction(self, v: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def logmap(self, y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def logmap_0(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def tangent_proj(self, v: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def proj(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def ptransp(self, v: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def ptransp_0(self, v: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def tangent_inner(self, u: jnp.ndarray, v: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def tangent_norm(self, v: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def is_in_manifold(self, x: jnp.ndarray) -> bool:
        raise NotImplementedError

    def is_in_tangent_space(self, v: jnp.ndarray, x: jnp.ndarray) -> bool:
        raise NotImplementedError

    def _create_origin_from_reference(self, reference: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def egrad2rgrad(self, grad: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def _gyration(self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
