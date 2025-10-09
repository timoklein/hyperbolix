"""Euclidean manifold - vmap-native pure functional implementation.

JAX port with vmap-native API. All functions operate on single points/vectors
with shape (dim,). Use jax.vmap for batch operations.

JIT Compilation & Batching
---------------------------
All functions work with single points and return scalars or vectors.
Use jax.vmap for batching:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix_jax.manifolds import euclidean
    >>>
    >>> # Single point operations
    >>> x = jnp.array([1.0, 2.0])
    >>> y = jnp.array([3.0, 4.0])
    >>> distance = euclidean.dist(x, y, c=0.0)  # Returns scalar
    >>>
    >>> # Batch operations with vmap
    >>> x_batch = jnp.array([[1.0, 2.0], [0.5, 1.5]])  # (batch, dim)
    >>> y_batch = jnp.array([[3.0, 4.0], [1.0, 2.0]])
    >>> dist_batched = jax.vmap(euclidean.dist, in_axes=(0, 0, None))
    >>> distances = dist_batched(x_batch, y_batch, 0.0)  # Returns (batch,)

See jax_migration.md for comprehensive usage patterns.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def proj(x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
    """Project point onto Euclidean space (identity operation).

    Args:
        x: Point in Euclidean space, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Projected point (identity), shape (dim,)
    """
    return x


def addition(
    x: Float[Array, "dim"], y: Float[Array, "dim"], c: float = 0.0, backproject: bool = True
) -> Float[Array, "dim"]:
    """Add Euclidean points x and y.

    Args:
        x: Euclidean point, shape (dim,)
        y: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Sum x + y, shape (dim,)
    """
    return x + y


def scalar_mul(
    r: float, x: Float[Array, "dim"], c: float = 0.0, backproject: bool = True
) -> Float[Array, "dim"]:
    """Multiply Euclidean point x with scalar r.

    Args:
        r: Scalar factor
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Product r * x, shape (dim,)
    """
    return r * x


def dist(
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


def dist_0(
    x: Float[Array, "dim"], c: float = 0.0
) -> Float[Array, ""]:
    """Compute geodesic distance from Euclidean origin to x.

    Args:
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Euclidean distance ||x||, scalar
    """
    return jnp.linalg.norm(x)


def expmap(
    v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0, backproject: bool = True
) -> Float[Array, "dim"]:
    """Exponential map: map tangent vector v at point x to manifold.

    In Euclidean space, this is simply addition.

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Point x + v, shape (dim,)
    """
    return x + v


def expmap_0(v: Float[Array, "dim"], c: float = 0.0, backproject: bool = True) -> Float[Array, "dim"]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    In Euclidean space, this is identity.

    Args:
        v: Tangent vector at origin, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Point v, shape (dim,)
    """
    return v


def retraction(
    v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0, backproject: bool = True
) -> Float[Array, "dim"]:
    """Retraction: first-order approximation of exponential map.

    In Euclidean space, retraction equals exponential map (addition).

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Point x + v, shape (dim,)
    """
    return x + v


def logmap(
    y: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0, backproject: bool = True
) -> Float[Array, "dim"]:
    """Logarithmic map: map point y to tangent space at point x.

    In Euclidean space, this is subtraction.

    Args:
        y: Euclidean point, shape (dim,)
        x: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector y - x, shape (dim,)
    """
    return y - x


def logmap_0(y: Float[Array, "dim"], c: float = 0.0, backproject: bool = True) -> Float[Array, "dim"]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    In Euclidean space, this is identity.

    Args:
        y: Euclidean point, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector y, shape (dim,)
    """
    return y


def ptransp(
    v: Float[Array, "dim"],
    x: Float[Array, "dim"],
    y: Float[Array, "dim"],
    c: float = 0.0,
    backproject: bool = True,
) -> Float[Array, "dim"]:
    """Parallel transport tangent vector v from point x to point y.

    In Euclidean space, tangent spaces are identical everywhere (identity).

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Euclidean point (ignored), shape (dim,)
        y: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector v (unchanged), shape (dim,)
    """
    return v


def ptransp_0(
    v: Float[Array, "dim"], y: Float[Array, "dim"], c: float = 0.0, backproject: bool = True
) -> Float[Array, "dim"]:
    """Parallel transport tangent vector v from origin to point y.

    In Euclidean space, tangent spaces are identical everywhere (identity).

    Args:
        v: Tangent vector at origin, shape (dim,)
        y: Euclidean point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector v (unchanged), shape (dim,)
    """
    return v


def tangent_inner(
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


def tangent_norm(
    v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0
) -> Float[Array, ""]:
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


def egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
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


def tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> Float[Array, "dim"]:
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


def is_in_manifold(x: Float[Array, "dim"], c: float = 0.0) -> bool:
    """Check if point x lies in Euclidean manifold.

    In Euclidean space, all points are valid.

    Args:
        x: Point to check, shape (dim,)
        c: Curvature (ignored, kept for consistency with other manifolds)

    Returns:
        Always True
    """
    return jnp.array(True, dtype=bool)


def is_in_tangent_space(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float = 0.0) -> bool:
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
