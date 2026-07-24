"""Hyperbolix JAX backend - pure functional implementation."""

from . import decomposition, manifolds, utils
from .nn_layers import PoincareBatchNorm2D
from .utils.curvature import LearnableCurvature

__all__ = [
    "LearnableCurvature",
    "PoincareBatchNorm2D",
    "decomposition",
    "manifolds",
    "utils",
]
