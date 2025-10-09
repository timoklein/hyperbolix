"""Math utils functions for hyperbolic operations with numerically stable limits.

Direct JAX port of PyTorch math_utils.py with type annotations using jaxtyping.
"""

import jax.nn as nn
import jax.numpy as jnp
from jaxtyping import Array, Float


def smooth_clamp_min(x: Float[Array, "..."], min_value: float, smoothing_factor: float = 50.0) -> Float[Array, "..."]:
    """Smoothly clamp array values to a minimum using softplus.

    Args:
        x: Input array of any shape
        min_value: Minimum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped above min_value
    """
    eps = jnp.finfo(x.dtype).eps
    shift = min_value + eps
    # Use JAX's numerically stable softplus: softplus_beta(x) = softplus(beta*x)/beta
    arg = smoothing_factor * (x - shift)
    x_clamped = shift + nn.softplus(arg) / smoothing_factor
    return jnp.where(x < shift, x_clamped, x)


def smooth_clamp_max(x: Float[Array, "..."], max_value: float, smoothing_factor: float = 50.0) -> Float[Array, "..."]:
    """Smoothly clamp array values to a maximum using softplus.

    Args:
        x: Input array of any shape
        max_value: Maximum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped below max_value
    """
    eps = jnp.finfo(x.dtype).eps
    shift = max_value - eps
    arg = smoothing_factor * (shift - x)
    x_clamped = shift - nn.softplus(arg) / smoothing_factor
    return jnp.where(x > shift, x_clamped, x)


def smooth_clamp(
    x: Float[Array, "..."], min_value: float, max_value: float, smoothing_factor: float = 50.0
) -> Float[Array, "..."]:
    """Smoothly clamp array values to a range [min_value, max_value].

    Args:
        x: Input array of any shape
        min_value: Minimum value to clamp to
        max_value: Maximum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped to [min_value, max_value]
    """
    x = smooth_clamp_max(x, max_value, smoothing_factor=smoothing_factor)
    return smooth_clamp_min(x, min_value, smoothing_factor=smoothing_factor)


def cosh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Hyperbolic cosine with overflow protection. Domain=(-inf, inf).

    Clamps input to safe ranges to prevent overflow based on dtype.
    Uses log(max) * 0.99 as safety margin.

    Args:
        x: Input array of any shape

    Returns:
        cosh(x) with overflow protection
    """
    # Safe limit based on dtype: cosh(x) ≈ exp(x)/2 for large x, so x < log(max)
    clamp = jnp.log(jnp.finfo(x.dtype).max) * 0.99
    x = smooth_clamp(x, -clamp, clamp)
    return jnp.cosh(x)


def sinh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Hyperbolic sine with overflow protection. Domain=(-inf, inf).

    Clamps input to safe ranges to prevent overflow based on dtype.
    Uses log(max) * 0.99 as safety margin.

    Args:
        x: Input array of any shape

    Returns:
        sinh(x) with overflow protection
    """
    # Safe limit based on dtype: sinh(x) ≈ exp(x)/2 for large x, so x < log(max)
    clamp = jnp.log(jnp.finfo(x.dtype).max) * 0.99
    x = smooth_clamp(x, -clamp, clamp)
    return jnp.sinh(x)


def acosh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse hyperbolic cosine with domain clamping. Domain=[1, inf).

    Args:
        x: Input array of any shape

    Returns:
        acosh(x) with domain protection (clamps x >= 1.0)
    """
    x = jnp.clip(x, 1.0, None)
    return jnp.acosh(x)


def asinh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse hyperbolic sine. Domain=(-inf, inf).

    Args:
        x: Input array of any shape

    Returns:
        asinh(x)
    """
    return jnp.asinh(x)


def atanh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse hyperbolic tangent with domain clamping. Domain=(-1, 1).

    Clamps input away from ±1 to avoid singularities.

    Args:
        x: Input array of any shape

    Returns:
        atanh(x) with domain protection
    """
    eps = jnp.finfo(x.dtype).eps
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return jnp.atanh(x)
