"""JAX manifold implementations - class-based approach with dtype control."""

from . import isometry_mappings
from .euclidean import Euclidean
from .halfspace import HalfSpace
from .hyperboloid import Hyperboloid
from .klein import Klein
from .poincare import Poincare
from .product import ProductManifold
from .proper_velocity import ProperVelocity
from .protocol import Curvature, Manifold, ScalarCurvature
from .stereographic import Stereographic

__all__ = [
    "Curvature",
    "Euclidean",
    "HalfSpace",
    "Hyperboloid",
    "Klein",
    "Manifold",
    "Poincare",
    "ProductManifold",
    "ProperVelocity",
    "ScalarCurvature",
    "Stereographic",
    "isometry_mappings",
]
