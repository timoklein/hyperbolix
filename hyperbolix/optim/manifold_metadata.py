"""Manifold metadata utilities for Riemannian optimization.

This module provides utilities to mark NNX parameters with manifold metadata
and extract that metadata during optimization. The metadata system uses
Flax NNX's built-in Variable._var_metadata attribute to store information
about which manifold a parameter lives on, enabling automatic detection
and appropriate handling during Riemannian optimization.

Design rationale:
- All trainable parameters remain nnx.Param (single variable type)
- Metadata is stored as part of the Variable, making it serialization-friendly
- String identifiers map to actual manifold modules via a registry
- Supports both static and callable curvature parameters

Example:
    >>> import jax.numpy as jnp
    >>> from flax import nnx
    >>> from hyperbolix.optim import mark_manifold_param, get_manifold_info
    >>>
    >>> # Create a parameter on the Poincaré manifold
    >>> bias_init = jnp.zeros((10,))
    >>> bias = mark_manifold_param(
    ...     nnx.Param(bias_init),
    ...     manifold_type='poincare',
    ...     curvature=1.0
    ... )
    >>>
    >>> # Later, in optimizer: extract manifold info
    >>> manifold_info = get_manifold_info(bias)
    >>> if manifold_info is not None:
    ...     manifold_module, c = manifold_info
    ...     # Apply Riemannian operations...
"""

from collections.abc import Callable
from typing import Any

from flax import nnx

# Global manifold registry: maps string identifiers to manifold modules
_MANIFOLD_REGISTRY: dict[str, Any] = {}


def register_manifold(name: str, manifold_module: Any) -> None:
    """Register a manifold module in the global registry.

    Parameters
    ----------
    name : str
        String identifier for the manifold (e.g., 'poincare', 'hyperboloid')
    manifold_module : Any
        The manifold module containing operations like expmap, egrad2rgrad, etc.

    Example
    -------
    >>> from hyperbolix.manifolds import poincare
    >>> register_manifold('poincare', poincare)
    """
    _MANIFOLD_REGISTRY[name] = manifold_module


def get_manifold_module(name: str) -> Any | None:
    """Retrieve a manifold module from the registry.

    Parameters
    ----------
    name : str
        String identifier for the manifold

    Returns
    -------
    manifold_module : Any or None
        The manifold module, or None if not found
    """
    return _MANIFOLD_REGISTRY.get(name)


def mark_manifold_param(
    param: nnx.Param,
    manifold_type: str,
    curvature: float | Callable[[], Any],
) -> nnx.Param:
    """Mark an nnx.Param with manifold metadata.

    This function attaches manifold information to a parameter's metadata,
    enabling Riemannian optimizers to automatically detect and handle
    manifold parameters appropriately.

    Parameters
    ----------
    param : nnx.Param
        The parameter to mark with manifold metadata
    manifold_type : str
        String identifier for the manifold (must be registered in the manifold registry)
    curvature : float or callable
        Either a static curvature value or a callable that returns the current curvature.
        Use a callable (e.g., lambda: self.c[...]) for learnable curvature.

    Returns
    -------
    param : nnx.Param
        The same parameter with manifold metadata attached

    Example
    -------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from flax import nnx
    >>>
    >>> # Static curvature
    >>> bias1 = mark_manifold_param(
    ...     nnx.Param(jax.random.normal(jax.random.key(0), (10,)) * 0.01),
    ...     manifold_type='poincare',
    ...     curvature=1.0
    ... )
    >>>
    >>> # Learnable curvature (callable)
    >>> class MyLayer(nnx.Module):
    ...     def __init__(self, rngs):
    ...         self.c = nnx.Param(jnp.array(1.0))
    ...         self.bias = mark_manifold_param(
    ...             nnx.Param(jax.random.normal(rngs.params(), (10,)) * 0.01),
    ...             manifold_type='poincare',
    ...             curvature=lambda: self.c.value
    ...         )

    Notes
    -----
    - The manifold_type must be registered via register_manifold() before use
    - For learnable curvature, use a callable to access the current value at runtime
    - The metadata is automatically serialized with the parameter during checkpointing
    """
    # Store manifold metadata in the Variable's metadata attribute
    if not hasattr(param, "_var_metadata") or param._var_metadata is None:
        param._var_metadata = {}

    param._var_metadata["manifold_type"] = manifold_type
    param._var_metadata["curvature"] = curvature

    return param


def get_manifold_info(param: nnx.Variable) -> tuple[Any, Any] | None:
    """Extract manifold information from a parameter's metadata.

    Parameters
    ----------
    param : nnx.Variable
        The parameter to extract manifold info from

    Returns
    -------
    manifold_info : tuple of (manifold_module, curvature) or None
        If the parameter has manifold metadata:
            - manifold_module: The manifold module from the registry
            - curvature: The current curvature value (evaluated if callable)
        If the parameter has no manifold metadata:
            - None

    Example
    -------
    >>> manifold_info = get_manifold_info(param)
    >>> if manifold_info is not None:
    ...     manifold_module, c = manifold_info
    ...     # param lives on a manifold, apply Riemannian operations
    ...     rgrad = manifold_module.egrad2rgrad(grad, param[...], c)
    ... else:
    ...     # param is Euclidean, apply standard operations
    ...     pass
    """
    # Check if parameter has manifold metadata
    if not hasattr(param, "_var_metadata") or param._var_metadata is None:
        return None

    metadata = param._var_metadata
    if "manifold_type" not in metadata:
        return None

    # Get manifold module from registry
    manifold_type = metadata["manifold_type"]
    manifold_module = get_manifold_module(manifold_type)
    if manifold_module is None:
        raise ValueError(
            f"Manifold type '{manifold_type}' not found in registry. Available manifolds: {list(_MANIFOLD_REGISTRY.keys())}"
        )

    # Get curvature value (evaluate if callable)
    curvature_value = metadata["curvature"]
    if callable(curvature_value):
        curvature_value = curvature_value()

    return (manifold_module, curvature_value)


def has_manifold_params(params_pytree: Any) -> bool:
    """Check if a parameter pytree contains any manifold parameters.

    Parameters
    ----------
    params_pytree : Any
        A pytree of parameters (typically from nnx.state(model, nnx.Param))

    Returns
    -------
    has_manifold : bool
        True if any parameter in the pytree has manifold metadata

    Example
    -------
    >>> import jax
    >>> from flax import nnx
    >>>
    >>> model = MyHyperbolicModel(rngs=nnx.Rngs(0))
    >>> params = nnx.state(model, nnx.Param)
    >>> if has_manifold_params(params):
    ...     print("Model contains manifold parameters")
    """
    from jax import tree_util

    def _check_param(x):
        if isinstance(x, nnx.Variable):
            return get_manifold_info(x) is not None
        return False

    # Flatten the pytree and check each leaf
    leaves = tree_util.tree_leaves(params_pytree)
    return any(_check_param(leaf) for leaf in leaves)
