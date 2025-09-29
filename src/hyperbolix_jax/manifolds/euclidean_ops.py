"""Pure functional operations for Euclidean manifold."""

import jax.numpy as jnp

from ..config import RuntimeConfig


Array = jnp.ndarray


def proj(
    x: Array,
    config: RuntimeConfig,
    axis: int = -1,
) -> Array:
    """Project point(s) onto Euclidean space (identity operation)."""
    return x


def dist(
    x: Array,
    y: Array,
    config: RuntimeConfig,
    axis: int = -1,
    backproject: bool = True,
) -> Array:
    """Compute Euclidean distance between points."""
    diff = x - y
    return jnp.linalg.norm(diff, axis=axis)


def dist_0(
    x: Array,
    config: RuntimeConfig,
    axis: int = -1,
) -> Array:
    """Compute Euclidean distance from origin."""
    return jnp.linalg.norm(x, axis=axis)


def expmap(
    v: Array,
    x: Array,
    config: RuntimeConfig,
    axis: int = -1,
    backproject: bool = True,
) -> Array:
    """Exponential map in Euclidean space (addition)."""
    return x + v


def logmap(
    y: Array,
    x: Array,
    config: RuntimeConfig,
    axis: int = -1,
    backproject: bool = True,
) -> Array:
    """Logarithmic map in Euclidean space (subtraction)."""
    return y - x


def ptransp(
    v: Array,
    x: Array,
    y: Array,
    config: RuntimeConfig,
    axis: int = -1,
    backproject: bool = True,
) -> Array:
    """Parallel transport in Euclidean space (identity)."""
    return v


def scalar_mul(
    r: Array,
    x: Array,
    config: RuntimeConfig,
    axis: int = -1,
    backproject: bool = True,
) -> Array:
    """Scalar multiplication in Euclidean space."""
    return r * x


def tangent_proj(
    v: Array,
    x: Array,
    config: RuntimeConfig,
    axis: int = -1,
) -> Array:
    """Project onto tangent space in Euclidean space (identity)."""
    return v