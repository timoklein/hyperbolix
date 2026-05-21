"""Hyperbolix JAX backend - pure functional implementation."""

from . import manifolds, utils
from .utils.curvature import LearnableCurvature

__all__ = [
    "LearnableCurvature",
    "manifolds",
    "utils",
]
