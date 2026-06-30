"""Hyperbolix JAX backend - pure functional implementation."""

from . import manifolds, utils
from .nn_layers import PoincareBatchNorm2D
from .utils.curvature import LearnableCurvature

__all__ = [
    "LearnableCurvature",
    "PoincareBatchNorm2D",
    "manifolds",
    "utils",
]
