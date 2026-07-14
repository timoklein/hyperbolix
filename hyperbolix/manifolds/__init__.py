"""JAX manifold implementations - class-based approach with dtype control."""

from . import isometry_mappings
from .euclidean import Euclidean
from .hyperboloid import Hyperboloid
from .poincare import Poincare
from .product import ProductManifold
from .proper_velocity import ProperVelocity
from .protocol import Curvature, Manifold
from .stereographic import Stereographic

__all__ = [
    "Curvature",
    "Euclidean",
    "Hyperboloid",
    "Manifold",
    "Poincare",
    "ProductManifold",
    "ProperVelocity",
    "Stereographic",
    "isometry_mappings",
]
