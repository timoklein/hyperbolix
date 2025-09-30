"""Euclidean manifold - pure functional implementation.

Direct JAX port of PyTorch euclidean.py with pure functions.
All operations are identity or simple linear operations.
"""

from jaxtyping import Array, Float
import jax.numpy as jnp


def proj(
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1
) -> Float[Array, "..."]:
    """Project point(s) onto Euclidean space (identity operation).

    Args:
        x: Point(s) in Euclidean space
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute projection (ignored, kept for consistency)

    Returns:
        Projected point(s) (identity)
    """
    return x


def addition(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Add Euclidean points x and y.

    Args:
        x: Euclidean point(s)
        y: Euclidean point(s)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute addition (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Sum x + y
    """
    return x + y


def scalar_mul(
    r: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Multiply Euclidean point(s) x with scalar(s) r.

    Args:
        r: Scalar factor(s)
        x: Euclidean point(s)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute multiplication (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Product r * x
    """
    return r * x


def dist(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    keepdim: bool = True,
    version: str = "default"
) -> Float[Array, "..."]:
    """Compute geodesic distance between Euclidean points x and y.

    Args:
        x: Euclidean point(s)
        y: Euclidean point(s)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute distance
        keepdim: Whether to keep the reduced dimension
        version: Version of distance (ignored, kept for consistency)

    Returns:
        Euclidean distance ||x - y||
    """
    return jnp.linalg.norm(x - y, axis=axis, keepdims=keepdim)


def dist_0(
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    keepdim: bool = True,
    version: str = "default"
) -> Float[Array, "..."]:
    """Compute geodesic distance from Euclidean origin to x.

    Args:
        x: Euclidean point(s)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute distance
        keepdim: Whether to keep the reduced dimension
        version: Version of distance (ignored, kept for consistency)

    Returns:
        Euclidean distance ||x||
    """
    return jnp.linalg.norm(x, axis=axis, keepdims=keepdim)


def expmap(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Exponential map: map tangent vector v at point x to manifold.

    In Euclidean space, this is simply addition.

    Args:
        v: Tangent vector(s) at x
        x: Euclidean point(s)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Point x + v
    """
    return x + v


def expmap_0(
    v: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    In Euclidean space, this is identity.

    Args:
        v: Tangent vector(s) at origin
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Point v
    """
    return v


def retraction(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Retraction: first-order approximation of exponential map.

    In Euclidean space, retraction equals exponential map (addition).

    Args:
        v: Tangent vector(s) at x
        x: Euclidean point(s)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Point x + v
    """
    return x + v


def logmap(
    y: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Logarithmic map: map point y to tangent space at point x.

    In Euclidean space, this is subtraction.

    Args:
        y: Euclidean point(s)
        x: Euclidean point(s)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector y - x
    """
    return y - x


def logmap_0(
    y: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    In Euclidean space, this is identity.

    Args:
        y: Euclidean point(s)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector y
    """
    return y


def ptransp(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Parallel transport tangent vector v from point x to point y.

    In Euclidean space, tangent spaces are identical everywhere (identity).

    Args:
        v: Tangent vector(s) at x
        x: Euclidean point(s) (ignored)
        y: Euclidean point(s) (ignored)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector v (unchanged)
    """
    return v


def ptransp_0(
    v: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Parallel transport tangent vector v from origin to point y.

    In Euclidean space, tangent spaces are identical everywhere (identity).

    Args:
        v: Tangent vector(s) at origin
        y: Euclidean point(s) (ignored)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector v (unchanged)
    """
    return v


def tangent_inner(
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    keepdim: bool = True
) -> Float[Array, "..."]:
    """Compute inner product of tangent vectors u and v at point x.

    In Euclidean space, this is the standard dot product.

    Args:
        u: Tangent vector(s) at x
        v: Tangent vector(s) at x
        x: Euclidean point(s) (ignored)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute inner product
        keepdim: Whether to keep the reduced dimension

    Returns:
        Inner product <u, v>
    """
    return jnp.sum(u * v, axis=axis, keepdims=keepdim)


def tangent_norm(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1,
    keepdim: bool = True
) -> Float[Array, "..."]:
    """Compute norm of tangent vector v at point x.

    In Euclidean space, this is the standard L2 norm.

    Args:
        v: Tangent vector(s) at x
        x: Euclidean point(s) (ignored)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute norm
        keepdim: Whether to keep the reduced dimension

    Returns:
        Norm ||v||
    """
    return jnp.linalg.norm(v, axis=axis, keepdims=keepdim)


def egrad2rgrad(
    grad: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1
) -> Float[Array, "..."]:
    """Convert Euclidean gradient to Riemannian gradient.

    In Euclidean space, these are identical.

    Args:
        grad: Euclidean gradient
        x: Euclidean point(s) (ignored)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)

    Returns:
        Riemannian gradient (same as Euclidean gradient)
    """
    return grad


def tangent_proj(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1
) -> Float[Array, "..."]:
    """Project vector v onto tangent space at point x.

    In Euclidean space, tangent space is the entire space (identity).

    Args:
        v: Vector(s) to project
        x: Euclidean point(s) (ignored)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to compute (ignored, kept for consistency)

    Returns:
        Projected vector v (unchanged)
    """
    return v


def is_in_manifold(
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1
) -> bool:
    """Check if point(s) x lie in Euclidean manifold.

    In Euclidean space, all points are valid.

    Args:
        x: Point(s) to check
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to check (ignored, kept for consistency)

    Returns:
        Always True
    """
    return True


def is_in_tangent_space(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float = 0.0,
    axis: int = -1
) -> bool:
    """Check if vector(s) v lie in tangent space at point x.

    In Euclidean space, all vectors are valid tangent vectors.

    Args:
        v: Vector(s) to check
        x: Euclidean point(s) (ignored)
        c: Curvature (ignored, kept for consistency with other manifolds)
        axis: Axis along which to check (ignored, kept for consistency)

    Returns:
        Always True
    """
    return True