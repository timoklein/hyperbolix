"""JAX utilities for hyperbolix."""

from .math_utils import (
    smooth_clamp_min,
    smooth_clamp_max,
    smooth_clamp,
    safe_cosh,
    safe_sinh,
    safe_acosh,
    safe_atanh,
    smooth_clamp_with_config,
    safe_cosh_with_config,
    safe_sinh_with_config,
)

__all__ = [
    "smooth_clamp_min",
    "smooth_clamp_max",
    "smooth_clamp",
    "safe_cosh",
    "safe_sinh",
    "safe_acosh",
    "safe_atanh",
    "smooth_clamp_with_config",
    "safe_cosh_with_config",
    "safe_sinh_with_config",
]