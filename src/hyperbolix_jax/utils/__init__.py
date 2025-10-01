"""JAX utilities for hyperbolix."""

from .math_utils import (
    acosh,
    atanh,
    cosh,
    sinh,
    smooth_clamp,
    smooth_clamp_max,
    smooth_clamp_min,
)

__all__ = [
    "smooth_clamp_min",
    "smooth_clamp_max",
    "smooth_clamp",
    "cosh",
    "sinh",
    "acosh",
    "atanh",
]
