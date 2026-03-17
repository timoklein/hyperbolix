"""Manifold metadata utilities for Riemannian optimization.

This module provides ``ManifoldParam``, an ``nnx.Param`` subclass that marks
parameters as living on a Riemannian manifold.  Riemannian optimizers detect
these parameters via ``isinstance(param, ManifoldParam)`` and apply the
appropriate exponential-map / retraction updates automatically.

Design rationale:
- ``ManifoldParam`` is a thin ``nnx.Param`` subclass — all NNX machinery
  (state extraction, serialization, JIT) works unchanged
- Manifold and curvature are stored as standard Variable metadata kwargs,
  accessible via attribute access (``param.manifold``, ``param.curvature``)
- The manifold instance carries dtype control via ``_cast()``, so Riemannian
  optimizer operations automatically respect the layer's precision setting
- Supports both static and callable curvature parameters

Example:
    >>> import jax.numpy as jnp
    >>> from flax import nnx
    >>> from hyperbolix.manifolds.poincare import Poincare
    >>> from hyperbolix.optim import ManifoldParam, get_manifold_info
    >>>
    >>> # Create a parameter on the Poincaré manifold
    >>> manifold = Poincare(dtype=jnp.float64)
    >>> bias = ManifoldParam(
    ...     jnp.zeros((10,)),
    ...     manifold=manifold,
    ...     curvature=1.0,
    ... )
    >>>
    >>> # In optimizer: extract manifold info
    >>> manifold_info = get_manifold_info(bias)
    >>> if manifold_info is not None:
    ...     manifold_instance, c = manifold_info
    ...     # Apply Riemannian operations via public methods...
"""

from collections.abc import Callable
from typing import Any

from flax import nnx

from hyperbolix.manifolds import Manifold


class ManifoldParam(nnx.Param):
    """``nnx.Param`` subclass for parameters living on a Riemannian manifold.

    Stores the manifold instance and curvature as Variable metadata kwargs,
    providing type-safe detection via ``isinstance(param, ManifoldParam)``.

    Parameters
    ----------
    value : array-like
        The parameter value (a JAX array).
    manifold : Manifold
        A manifold class instance (e.g., ``Poincare(dtype=jnp.float64)``).
    curvature : float or callable
        Either a static curvature value or a callable that returns the
        current curvature.  Use a callable (e.g., ``lambda: self.c[...]``)
        for learnable curvature.

    Example
    -------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from flax import nnx
    >>> from hyperbolix.manifolds.poincare import Poincare
    >>> from hyperbolix.optim import ManifoldParam
    >>>
    >>> manifold = Poincare(dtype=jnp.float64)
    >>>
    >>> # Static curvature
    >>> bias = ManifoldParam(
    ...     jax.random.normal(jax.random.key(0), (10,)) * 0.01,
    ...     manifold=manifold,
    ...     curvature=1.0,
    ... )
    >>>
    >>> # Learnable curvature (callable)
    >>> class MyLayer(nnx.Module):
    ...     def __init__(self, rngs):
    ...         self.c = nnx.Param(jnp.array(1.0))
    ...         self.bias = ManifoldParam(
    ...             jax.random.normal(rngs.params(), (10,)) * 0.01,
    ...             manifold=manifold,
    ...             curvature=lambda: self.c.value,
    ...         )
    """

    def __init__(
        self,
        value: Any,
        *,
        manifold: Manifold,
        curvature: float | Callable[[], Any],
        **metadata: Any,
    ) -> None:
        super().__init__(value, manifold=manifold, curvature=curvature, **metadata)


def mark_manifold_param(
    param: nnx.Param,
    manifold: Manifold,
    curvature: float | Callable[[], Any],
) -> ManifoldParam:
    """Create a ``ManifoldParam`` from an existing ``nnx.Param``.

    This is a convenience wrapper around ``ManifoldParam``.  Prefer using
    ``ManifoldParam`` directly in new code.

    Parameters
    ----------
    param : nnx.Param
        The parameter whose value will be copied into a new ``ManifoldParam``.
    manifold : Manifold
        A manifold class instance.
    curvature : float or callable
        Static curvature value or callable returning current curvature.

    Returns
    -------
    ManifoldParam
        A new ``ManifoldParam`` wrapping the same array value.
    """
    return ManifoldParam(param[...], manifold=manifold, curvature=curvature)


def get_manifold_info(param: nnx.Variable) -> tuple[Manifold, Any] | None:
    """Extract manifold information from a parameter.

    Parameters
    ----------
    param : nnx.Variable
        The parameter to extract manifold info from.

    Returns
    -------
    manifold_info : tuple of (Manifold, curvature) or None
        If the parameter is a ``ManifoldParam``:
            - manifold: The manifold class instance
            - curvature: The current curvature value (evaluated if callable)
        Otherwise ``None``.

    Example
    -------
    >>> manifold_info = get_manifold_info(param)
    >>> if manifold_info is not None:
    ...     manifold, c = manifold_info
    ...     rgrad = manifold.egrad2rgrad(grad, param[...], c)
    """
    if not isinstance(param, ManifoldParam):
        return None

    manifold = param.manifold
    curvature_value = param.curvature
    if callable(curvature_value):
        curvature_value = curvature_value()

    return (manifold, curvature_value)


def has_manifold_params(params_pytree: Any) -> bool:
    """Check if a parameter pytree contains any manifold parameters.

    Parameters
    ----------
    params_pytree : Any
        A pytree of parameters (typically from ``nnx.state(model, nnx.Param)``).

    Returns
    -------
    has_manifold : bool
        True if any parameter in the pytree is a ``ManifoldParam``.

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

    leaves = tree_util.tree_leaves(params_pytree)
    return any(isinstance(leaf, ManifoldParam) for leaf in leaves)
