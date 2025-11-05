"""Riemannian optimizers for hyperbolic neural networks.

This package provides Riemannian optimization algorithms (SGD, Adam) that work
seamlessly with Flax NNX and Optax. The optimizers automatically detect manifold
parameters via metadata and apply appropriate Riemannian operations.

Key Features:
- Automatic manifold detection via parameter metadata
- Standard Optax GradientTransformation interface
- Compatible with nnx.Optimizer wrapper
- Supports mixed Euclidean/Riemannian parameters
- Momentum parallel transport for manifold parameters
- Both exponential map (exact) and retraction (fast approximation)

Example:
    >>> from flax import nnx
    >>> from hyperbolix_jax.optim import riemannian_sgd
    >>> from hyperbolix_jax.nn_layers import HypLinearPoincare
    >>> from hyperbolix_jax.manifolds import poincare
    >>>
    >>> # Create model with manifold parameters
    >>> layer = HypLinearPoincare(poincare, 10, 5, rngs=nnx.Rngs(0))
    >>>
    >>> # Create Riemannian optimizer
    >>> tx = riemannian_sgd(learning_rate=0.01, momentum=0.9)
    >>> optimizer = nnx.Optimizer(layer, tx, wrt=nnx.Param)

Metadata System:
    The metadata system uses Flax NNX's built-in Variable._var_metadata to store
    manifold information (type and curvature). This enables:
    - All parameters remain nnx.Param (single variable type)
    - Automatic serialization with checkpoints
    - Seamless Optax integration

Manifold Registry:
    Manifolds are registered by string identifiers for serialization-friendly metadata.
    The poincare and hyperboloid manifolds are automatically registered on import.

References:
    Bécigneul, Gary, and Octavian-Eugen Ganea. "Riemannian adaptive optimization methods."
        arXiv preprint arXiv:1810.00760 (2018).
"""

from .manifold_metadata import (
    get_manifold_info,
    get_manifold_module,
    has_manifold_params,
    mark_manifold_param,
    register_manifold,
)
from .riemannian_adam import riemannian_adam
from .riemannian_sgd import riemannian_sgd

# Register manifolds on import
# Import manifold modules and register them
try:
    from ..manifolds import hyperboloid, poincare

    register_manifold("poincare", poincare)
    register_manifold("hyperboloid", hyperboloid)
except ImportError as e:
    # If manifolds aren't available, warn but don't fail
    import warnings

    warnings.warn(f"Could not register manifolds: {e}")

__all__ = [
    "get_manifold_info",
    "get_manifold_module",
    "has_manifold_params",
    # Metadata utilities
    "mark_manifold_param",
    "register_manifold",
    "riemannian_adam",
    # Optimizers
    "riemannian_sgd",
]
