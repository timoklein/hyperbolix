"""Probability distributions on hyperbolic manifolds."""

from . import wrapped_normal_hyperboloid, wrapped_normal_poincare

# For backwards compatibility, also expose as wrapped_normal
from . import wrapped_normal_hyperboloid as wrapped_normal

__all__ = ["wrapped_normal", "wrapped_normal_hyperboloid", "wrapped_normal_poincare"]
