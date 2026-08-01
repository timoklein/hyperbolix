"""Hyperbolix JAX backend - pure functional implementation."""

from . import decomposition, distributions, manifolds, nn_layers, optim, utils
from .utils.curvature import LearnableCurvature

__all__ = [
    "LearnableCurvature",
    "decomposition",
    "distributions",
    "manifolds",
    "nn_layers",
    "optim",
    "utils",
]
