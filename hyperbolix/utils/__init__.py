"""JAX utilities for hyperbolix."""

from .curvature import LearnableCurvature
from .helpers import compute_hyperbolic_delta, compute_pairwise_distances, get_delta
from .math_utils import (
    acosh,
    atanh,
    capped_exp,
    cosh,
    safe_hypot,
    safe_norm,
    safe_normalize,
    sinh,
    smooth_clamp,
    smooth_clamp_max,
    smooth_clamp_min,
    tanh,
)

__all__ = [
    "LearnableCurvature",
    "acosh",
    "atanh",
    "capped_exp",
    "compute_hyperbolic_delta",
    "compute_pairwise_distances",
    "cosh",
    "get_delta",
    "safe_hypot",
    "safe_norm",
    "safe_normalize",
    "sinh",
    "smooth_clamp",
    "smooth_clamp_max",
    "smooth_clamp_min",
    "tanh",
]
