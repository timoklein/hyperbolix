"""JAX manifold implementations using Flax struct.dataclass."""

from .base import ManifoldBase, ManifoldOps
from .euclidean import Euclidean, create_euclidean
from .hyperboloid import Hyperboloid
from .poincare import PoincareBall

# Keep old import for compatibility
Manifold = ManifoldBase

__all__ = [
    "ManifoldBase",
    "ManifoldOps",
    "Manifold",  # alias for compatibility
    "Euclidean",
    "create_euclidean",
    "Hyperboloid",
    "PoincareBall",
]
