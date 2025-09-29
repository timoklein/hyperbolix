"""JAX math utils functions for hyperbolic operations with numerically stable limits."""

import jax.numpy as jnp
from jax import nn

from ..config import RuntimeConfig


Array = jnp.ndarray


def _get_array_eps(x: Array) -> float:
    """Get machine epsilon for array's dtype."""
    if x.dtype == jnp.float32:
        return jnp.finfo(jnp.float32).eps
    elif x.dtype == jnp.float64:
        return jnp.finfo(jnp.float64).eps
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")


def smooth_clamp_min(
    x: Array,
    min_value: float,
    smoothing_factor: float = 50.0
) -> Array:
    """Smoothly clamp array values to a minimum using softplus.

    Args:
        x: Input array
        min_value: Minimum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped above min_value
    """
    eps = _get_array_eps(x)
    shift = min_value + eps
    # JAX softplus doesn't have beta, so implement manually: softplus_beta(x) = log(1 + exp(beta*x))/beta
    arg = smoothing_factor * (x - shift)
    x_clamped = shift + jnp.log1p(jnp.exp(arg)) / smoothing_factor
    return jnp.where(x < shift, x_clamped, x)


def smooth_clamp_max(
    x: Array,
    max_value: float,
    smoothing_factor: float = 50.0
) -> Array:
    """Smoothly clamp array values to a maximum using softplus.

    Args:
        x: Input array
        max_value: Maximum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped below max_value
    """
    eps = _get_array_eps(x)
    shift = max_value - eps
    # JAX softplus doesn't have beta, so implement manually: softplus_beta(x) = log(1 + exp(beta*x))/beta
    arg = smoothing_factor * (shift - x)
    x_clamped = shift - jnp.log1p(jnp.exp(arg)) / smoothing_factor
    return jnp.where(x > shift, x_clamped, x)


def smooth_clamp(
    x: Array,
    min_value: float,
    max_value: float,
    smoothing_factor: float = 50.0
) -> Array:
    """Smoothly clamp array values to a range [min_value, max_value].

    Args:
        x: Input array
        min_value: Minimum value to clamp to
        max_value: Maximum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped to [min_value, max_value]
    """
    x = smooth_clamp_max(x, max_value, smoothing_factor=smoothing_factor)
    return smooth_clamp_min(x, min_value, smoothing_factor=smoothing_factor)


def safe_cosh(x: Array) -> Array:
    """Numerically stable hyperbolic cosine with domain clamping.

    Clamps input to safe ranges to prevent overflow:
    - float32: [-88, 88]
    - float64: [-709, 709]

    Args:
        x: Input array

    Returns:
        cosh(x) with overflow protection
    """
    # Safe limits as specified in SLEEF library
    clamp = 88.0 if x.dtype == jnp.float32 else 709.0
    x_clamped = smooth_clamp(x, -clamp, clamp)
    return jnp.cosh(x_clamped)


def safe_sinh(x: Array) -> Array:
    """Numerically stable hyperbolic sine with domain clamping.

    Clamps input to safe ranges to prevent overflow:
    - float32: [-88, 88]
    - float64: [-709, 709]

    Args:
        x: Input array

    Returns:
        sinh(x) with overflow protection
    """
    # Safe limits as specified in SLEEF library
    clamp = 88.0 if x.dtype == jnp.float32 else 709.0
    x_clamped = smooth_clamp(x, -clamp, clamp)
    return jnp.sinh(x_clamped)


def safe_acosh(x: Array) -> Array:
    """Numerically stable inverse hyperbolic cosine with domain clamping.

    Domain: [1, inf) - clamps input to be >= 1.0

    Args:
        x: Input array

    Returns:
        acosh(x) with domain protection
    """
    x_clamped = jnp.clip(x, 1.0, None)
    return jnp.acosh(x_clamped)


def safe_atanh(x: Array) -> Array:
    """Numerically stable inverse hyperbolic tangent with domain clamping.

    Domain: (-1, 1) - clamps input to avoid singularities at ±1

    Args:
        x: Input array

    Returns:
        atanh(x) with domain protection
    """
    eps = _get_array_eps(x)
    x_clamped = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return jnp.atanh(x_clamped)


# Configuration-aware versions that use RuntimeConfig tolerances
def smooth_clamp_with_config(
    x: Array,
    min_value: float,
    max_value: float,
    config: RuntimeConfig
) -> Array:
    """Smooth clamp using RuntimeConfig smoothing factor."""
    return smooth_clamp(x, min_value, max_value, smoothing_factor=config.smoothing_factor)


def get_safe_clamp_bounds(dtype: jnp.dtype) -> tuple[float, float]:
    """Get safe clamping bounds for cosh/sinh based on dtype."""
    if dtype == jnp.float32:
        return -88.0, 88.0
    elif dtype == jnp.float64:
        return -709.0, 709.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def safe_cosh_with_config(x: Array, config: RuntimeConfig) -> Array:
    """Safe cosh using RuntimeConfig for smoothing."""
    min_clamp, max_clamp = get_safe_clamp_bounds(x.dtype)
    x_clamped = smooth_clamp_with_config(x, min_clamp, max_clamp, config)
    return jnp.cosh(x_clamped)


def safe_sinh_with_config(x: Array, config: RuntimeConfig) -> Array:
    """Safe sinh using RuntimeConfig for smoothing."""
    min_clamp, max_clamp = get_safe_clamp_bounds(x.dtype)
    x_clamped = smooth_clamp_with_config(x, min_clamp, max_clamp, config)
    return jnp.sinh(x_clamped)