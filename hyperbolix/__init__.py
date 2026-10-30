"""Hyperbolix JAX backend - pure functional implementation."""

from importlib.metadata import version as _version

from . import decomposition, distributions, manifolds, nn_layers, optim, utils
from .utils.curvature import LearnableCurvature

__version__ = _version("hyperbolix")

__all__ = [
    "LearnableCurvature",
    "decomposition",
    "distributions",
    "manifolds",
    "nn_layers",
    "optim",
    "utils",
]
