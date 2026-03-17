"""Riemannian optimizers for hyperbolic neural networks.

This package provides Riemannian optimization algorithms (SGD, Adam) that work
seamlessly with Flax NNX and Optax. The optimizers automatically detect manifold
parameters via ``ManifoldParam`` and apply appropriate Riemannian operations.

Key Features:
- Automatic manifold detection via ``ManifoldParam`` (``nnx.Param`` subclass)
- Standard Optax GradientTransformation interface
- Compatible with nnx.Optimizer wrapper
- Supports mixed Euclidean/Riemannian parameters
- Momentum parallel transport for manifold parameters
- Both exponential map (exact) and retraction (fast approximation)

Example:
    >>> from flax import nnx
    >>> from hyperbolix.optim import riemannian_sgd
    >>> from hyperbolix.nn_layers import HypLinearPoincare
    >>> from hyperbolix.manifolds.poincare import Poincare
    >>>
    >>> # Create model with manifold parameters
    >>> manifold = Poincare(dtype=jnp.float64)
    >>> layer = HypLinearPoincare(manifold, 10, 5, rngs=nnx.Rngs(0))
    >>>
    >>> # Create Riemannian optimizer
    >>> tx = riemannian_sgd(learning_rate=0.01, momentum=0.9)
    >>> optimizer = nnx.Optimizer(layer, tx, wrt=nnx.Param)

References:
    Bécigneul, Gary, and Octavian-Eugen Ganea. "Riemannian adaptive optimization methods."
        arXiv preprint arXiv:1810.00760 (2018).
"""

from .manifold_metadata import (
    ManifoldParam,
    get_manifold_info,
    has_manifold_params,
    mark_manifold_param,
)
from .riemannian_adam import riemannian_adam
from .riemannian_sgd import riemannian_sgd

__all__ = [
    "ManifoldParam",
    "get_manifold_info",
    "has_manifold_params",
    # Metadata utilities
    "mark_manifold_param",
    "riemannian_adam",
    # Optimizers
    "riemannian_sgd",
]
