"""JAX manifold implementations - class-based approach with dtype control."""

# Import manifold classes
# Import manifold modules for backwards compatibility
from . import isometry_mappings
from .euclidean import Euclidean
from .hyperboloid import Hyperboloid
from .poincare import Poincare

__all__ = [
    "Euclidean",
    "Hyperboloid",
    "Poincare",
    "isometry_mappings",
]
