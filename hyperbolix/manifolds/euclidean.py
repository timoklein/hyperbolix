"""Euclidean manifold - class-based API with dtype control.

JAX port with vmap-native API. All functions operate on single points/vectors
with shape (dim,). Use jax.vmap for batch operations.

JIT Compilation & Batching
---------------------------
All functions work with single points and return scalars or vectors.
Use jax.vmap for batching:

    >>> import jax
    >>> import jax
import jax.numpy as jnp
    >>> from hyperbolix.manifolds.euclidean import Euclidean
    >>>
    >>> # Single point operations
    >>> x = jnp.array([1.0, 2.0])
    >>> y = jnp.array([3.0, 4.0])
    >>> manifold = Euclidean(dtype=jnp.float32)
    >>> distance = manifold.dist(x, y, c=0.0)  # Returns scalar
    >>>
    >>> # Batch operations with vmap
    >>> x_batch = jnp.array([[1.0, 2.0], [0.5, 1.5]])  # (batch, dim)
    >>> y_batch = jnp.array([[3.0, 4.0], [1.0, 2.0]])
    >>> dist_batched = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    >>> distances = dist_batched(x_batch, y_batch, 0.0)  # Returns (batch,)

See jax_migration.md for comprehensive usage patterns.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _proj(x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Project point onto Euclidean space (identity operation).

    Args:
        x: Point in Euclidean space, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Projected point (identity), shape (dim,)
    """
    return x


def _addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Add Euclidean points x and y.

    Args:
        x: Euclidean point, shape (dim,)
        y: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Sum x + y, shape (dim,)
    """
    return x + y


def _scalar_mul(r: float, x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Multiply Euclidean point x with scalar r.

    Args:
        r: Scalar factor
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Product r * x, shape (dim,)
    """
    return r * x


def _dist(
    x: Float[Array, "dim"],
    y: Float[Array, "dim"],
    c: float = 0.0,
) -> Float[Array, ""]:
    """Compute geodesic distance between Euclidean points x and y.

    Args:
        x: Euclidean point, shape (dim,)
        y: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Euclidean distance ||x - y||, scalar
    """
    return jnp.linalg.norm(x - y)


def _dist_0(x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, ""]:
    """Compute geodesic distance from Euclidean origin to x.

    Args:
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Euclidean distance ||x||, scalar
    """
    return jnp.linalg.norm(x)


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Exponential map: map tangent vector v at point x to manifold.

    In Euclidean space, this is simply addition.

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Point x + v, shape (dim,)
    """
    return x + v


def _expmap_0(v: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    In Euclidean space, this is identity.

    Args:
        v: Tangent vector at origin, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Point v, shape (dim,)
    """
    return v


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Retraction: first-order approximation of exponential map.

    In Euclidean space, retraction equals exponential map (addition).

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Point x + v, shape (dim,)
    """
    return x + v


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Logarithmic map: map point y to tangent space at point x.

    In Euclidean space, this is subtraction.

    Args:
        y: Euclidean point, shape (dim,)
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Tangent vector y - x, shape (dim,)
    """
    return y - x


def _logmap_0(y: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    In Euclidean space, this is identity.

    Args:
        y: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Tangent vector y, shape (dim,)
    """
    return y


def _ptransp(
    v: Float[Array, "dim"],
    x: Float[Array, "dim"],
    y: Float[Array, "dim"],
    c: float = 0.0,
) -> Float[Array, "dim"]:
    """Parallel transport tangent vector v from point x to point y.

    In Euclidean space, tangent spaces are identical everywhere (identity).

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Euclidean point (ignored), shape (dim,)
        y: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Tangent vector v (unchanged), shape (dim,)
    """
    return v


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Parallel transport tangent vector v from origin to point y.

    In Euclidean space, tangent spaces are identical everywhere (identity).

    Args:
        v: Tangent vector at origin, shape (dim,)
        y: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Tangent vector v (unchanged), shape (dim,)
    """
    return v


def _tangent_inner(
    u: Float[Array, "dim"],
    v: Float[Array, "dim"],
    x: Float[Array, "dim"],
    c: float = 0.0,
) -> Float[Array, ""]:
    """Compute inner product of tangent vectors u and v at point x.

    In Euclidean space, this is the standard dot product.

    Args:
        u: Tangent vector at x, shape (dim,)
        v: Tangent vector at x, shape (dim,)
        x: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Inner product <u, v>, scalar
    """
    return jnp.dot(u, v)


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, ""]:
    """Compute norm of tangent vector v at point x.

    In Euclidean space, this is the standard L2 norm.

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Norm ||v||, scalar
    """
    return jnp.linalg.norm(v)


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Convert Euclidean gradient to Riemannian gradient.

    In Euclidean space, these are identical.

    Args:
        grad: Euclidean gradient, shape (dim,)
        x: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Riemannian gradient (same as Euclidean gradient), shape (dim,)
    """
    return grad


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Project vector v onto tangent space at point x.

    In Euclidean space, tangent space is the entire space (identity).

    Args:
        v: Vector to project, shape (dim,)
        x: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Projected vector v (unchanged), shape (dim,)
    """
    return v


def _is_in_manifold(x: Float[Array, "dim"], c: float = 0.0) -> Array:
    """Check if point x lies in Euclidean manifold.

    In Euclidean space, all points are valid.

    Args:
        x: Point to check, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Always True
    """
    return jnp.array(True, dtype=bool)


def _is_in_tangent_space(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Array:
    """Check if vector v lies in tangent space at point x.

    In Euclidean space, all vectors are valid tangent vectors.

    Args:
        v: Vector to check, shape (dim,)
        x: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Always True
    """
    return jnp.array(True, dtype=bool)


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class Euclidean:
    """Euclidean manifold with automatic dtype casting.

    Provides all manifold operations with automatic casting of array inputs
    to the specified dtype.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds.euclidean import Euclidean
        >>>
        >>> manifold = Euclidean(dtype=jnp.float64)
        >>> x = jnp.array([1.0, 2.0])
        >>> y = jnp.array([3.0, 4.0])
        >>> d = manifold.dist(x, y)
    """

    def __init__(self, dtype: jnp.dtype = jnp.float32) -> None:
        self.dtype = dtype

    def _cast(self, x: Array) -> Array:
        """Cast array to target dtype if it's a floating-point array."""
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact):
            return x.astype(self.dtype)
        return x

    def proj(self, x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Project point onto Euclidean space (identity)."""
        return _proj(self._cast(x), c)

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Add Euclidean points."""
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float, x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Scalar multiplication."""
        return _scalar_mul(r, self._cast(x), c)

    def dist(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: float = 0.0) -> Float[Array, ""]:
        """Compute distance."""
        return _dist(self._cast(x), self._cast(y), c)

    def dist_0(self, x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, ""]:
        """Distance from origin."""
        return _dist_0(self._cast(x), c)

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Exponential map."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Exponential map from origin."""
        return _expmap_0(self._cast(v), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Retraction."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Logarithmic map."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Logarithmic map from origin."""
        return _logmap_0(self._cast(y), c)

    def ptransp(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: float = 0.0
    ) -> Float[Array, "dim"]:
        """Parallel transport."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim"], y: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Parallel transport from origin."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    def tangent_inner(
        self, u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0
    ) -> Float[Array, ""]:
        """Tangent inner product."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, ""]:
        """Tangent norm."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Euclidean to Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
        """Project onto tangent space."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim"], c: float = 0.0) -> Array:
        """Check if on manifold."""
        return _is_in_manifold(self._cast(x), c)

    def is_in_tangent_space(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Array:
        """Check if in tangent space."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c)
