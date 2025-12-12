"""Probability distributions on hyperbolic manifolds."""

# For backwards compatibility, also expose as wrapped_normal
from . import wrapped_normal_hyperboloid
from . import wrapped_normal_hyperboloid as wrapped_normal
from . import wrapped_normal_poincare

__all__ = ["wrapped_normal", "wrapped_normal_hyperboloid", "wrapped_normal_poincare"]
