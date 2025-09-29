"""Hyperbolix JAX backend."""

from .config import RuntimeConfig, DEFAULT_CONFIG, FLOAT64_CONFIG, HIGH_PRECISION_CONFIG, create_config
from .manifolds import ManifoldBase, ManifoldOps, Manifold, Euclidean, create_euclidean, Hyperboloid, PoincareBall
from . import utils

__all__ = [
    "RuntimeConfig",
    "DEFAULT_CONFIG",
    "FLOAT64_CONFIG",
    "HIGH_PRECISION_CONFIG",
    "create_config",
    "ManifoldBase",
    "ManifoldOps",
    "Manifold",
    "Euclidean",
    "create_euclidean",
    "Hyperboloid",
    "PoincareBall",
    "utils",
]
