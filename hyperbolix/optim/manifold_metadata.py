"""Manifold metadata utilities for Riemannian optimization.

This module provides utilities to mark NNX parameters with manifold metadata
and extract that metadata during optimization. The metadata system uses
Flax NNX's built-in Variable._var_metadata attribute to store information
about which manifold a parameter lives on, enabling automatic detection
and appropriate handling during Riemannian optimization.

Design rationale:
- All trainable parameters remain nnx.Param (single variable type)
- Metadata stores the manifold class instance directly (not a string identifier)
- The manifold instance carries dtype control via _cast(), so Riemannian
  optimizer operations automatically respect the layer's precision setting
- Supports both static and callable curvature parameters

Example:
    >>> import jax.numpy as jnp
    >>> from flax import nnx
    >>> from hyperbolix.manifolds.poincare import Poincare
    >>> from hyperbolix.optim import mark_manifold_param, get_manifold_info
    >>>
    >>> # Create a parameter on the Poincaré manifold
    >>> manifold = Poincare(dtype=jnp.float64)
    >>> bias_init = jnp.zeros((10,))
    >>> bias = mark_manifold_param(
    ...     nnx.Param(bias_init),
    ...     manifold=manifold,
    ...     curvature=1.0
    ... )
    >>>
    >>> # Later, in optimizer: extract manifold info
    >>> manifold_info = get_manifold_info(bias)
    >>> if manifold_info is not None:
    ...     manifold_instance, c = manifold_info
    ...     # Apply Riemannian operations via public methods...
"""

from collections.abc import Callable
from typing import Any

from flax import nnx

from hyperbolix.manifolds import Manifold


def mark_manifold_param[ParamT](
    param: nnx.Param[ParamT],
    manifold: Manifold,
    curvature: float | Callable[[], Any],
) -> nnx.Param[ParamT]:
    """Mark an nnx.Param with manifold metadata.

    This function attaches manifold information to a parameter's metadata,
    enabling Riemannian optimizers to automatically detect and handle
    manifold parameters appropriately.

    Parameters
    ----------
    param : nnx.Param
        The parameter to mark with manifold metadata
    manifold : Manifold
        A manifold class instance (e.g., ``Poincare(dtype=jnp.float64)``)
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
    >>> from hyperbolix.manifolds.poincare import Poincare
    >>>
    >>> manifold = Poincare(dtype=jnp.float64)
    >>>
    >>> # Static curvature
    >>> bias1 = mark_manifold_param(
    ...     nnx.Param(jax.random.normal(jax.random.key(0), (10,)) * 0.01),
    ...     manifold=manifold,
    ...     curvature=1.0
    ... )
    >>>
    >>> # Learnable curvature (callable)
    >>> class MyLayer(nnx.Module):
    ...     def __init__(self, rngs):
    ...         self.c = nnx.Param(jnp.array(1.0))
    ...         self.bias = mark_manifold_param(
    ...             nnx.Param(jax.random.normal(rngs.params(), (10,)) * 0.01),
    ...             manifold=manifold,
    ...             curvature=lambda: self.c.value
    ...         )

    Notes
    -----
    - For learnable curvature, use a callable to access the current value at runtime
    - The metadata is automatically serialized with the parameter during checkpointing
    """
    # Store manifold metadata in the Variable's metadata attribute
    if not hasattr(param, "_var_metadata") or param._var_metadata is None:
        param._var_metadata = {}

    param._var_metadata["manifold"] = manifold
    param._var_metadata["curvature"] = curvature

    return param


def get_manifold_info(param: nnx.Variable) -> tuple[Manifold, Any] | None:
    """Extract manifold information from a parameter's metadata.

    Parameters
    ----------
    param : nnx.Variable
        The parameter to extract manifold info from

    Returns
    -------
    manifold_info : tuple of (Manifold, curvature) or None
        If the parameter has manifold metadata:
            - manifold: The manifold class instance
            - curvature: The current curvature value (evaluated if callable)
        If the parameter has no manifold metadata:
            - None

    Example
    -------
    >>> manifold_info = get_manifold_info(param)
    >>> if manifold_info is not None:
    ...     manifold, c = manifold_info
    ...     # param lives on a manifold, apply Riemannian operations
    ...     rgrad = manifold.egrad2rgrad(grad, param[...], c)
    ... else:
    ...     # param is Euclidean, apply standard operations
    ...     pass
    """
    # Check if parameter has manifold metadata
    if not hasattr(param, "_var_metadata") or param._var_metadata is None:
        return None

    metadata = param._var_metadata
    if "manifold" not in metadata:
        return None

    manifold = metadata["manifold"]

    # Get curvature value (evaluate if callable)
    curvature_value = metadata["curvature"]
    if callable(curvature_value):
        curvature_value = curvature_value()

    return (manifold, curvature_value)


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
