"""Precision wrapper for manifold modules.

Provides automatic dtype casting at manifold function boundaries,
enabling float64 computations without modifying function signatures.
"""

import functools
import types
from typing import Any

import jax
import jax.numpy as jnp


class PrecisionWrapped:
    """Proxy object that auto-casts jax.Array arguments to a specified dtype.

    Wraps a manifold module so that all public callable attributes automatically
    cast ``jax.Array`` positional arguments to the target dtype before calling
    the underlying function. Non-array arguments (Python floats, ints, etc.)
    and non-callable attributes (constants) pass through unchanged.

    Args:
        module: The manifold module to wrap.
        dtype: The target JAX dtype (e.g. ``jnp.float64``).
    """

    def __init__(self, module: types.ModuleType, dtype: jnp.dtype) -> None:
        self._module = module
        self._dtype = dtype

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module, name)
        if callable(attr) and not name.startswith("_"):
            return self._wrap_fn(attr)
        return attr

    def _wrap_fn(self, fn: Any) -> Any:
        dtype = self._dtype

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cast_args = tuple(_cast_arg(a, dtype) for a in args)
            return fn(*cast_args, **kwargs)

        return wrapper


def _cast_arg(arg: Any, dtype: jnp.dtype) -> Any:
    """Cast a single argument to dtype if it is a jax Array."""
    if isinstance(arg, jax.Array):
        return arg.astype(dtype)
    return arg


def with_precision(manifold_module: types.ModuleType, dtype: jnp.dtype) -> PrecisionWrapped:
    """Wrap a manifold module to auto-cast array inputs to the given dtype.

    Returns a proxy object whose public functions automatically cast all
    ``jax.Array`` positional arguments to *dtype* before calling the
    underlying manifold function. Non-array arguments (Python scalars,
    version indices, etc.) pass through unchanged.

    Args:
        manifold_module: A manifold module (e.g. ``poincare``, ``hyperboloid``).
        dtype: Target JAX dtype (e.g. ``jnp.float64``).

    Returns:
        A ``PrecisionWrapped`` proxy with the same public API.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import poincare, with_precision
        >>>
        >>> poincare_f64 = with_precision(poincare, jnp.float64)
        >>> x = jnp.array([0.1, 0.2])  # float32
        >>> y = jnp.array([0.3, 0.4])  # float32
        >>> d = poincare_f64.dist(x, y, c=1.0)
        >>> d.dtype  # float64
    """
    return PrecisionWrapped(manifold_module, dtype)
