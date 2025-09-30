"""Hyperboloid manifold - pure functional implementation.

Direct JAX port following the Lorentz/hyperboloid model.
Convention: -x₀² + ||x_rest||² = -1/c with c > 0, x₀ > 0, and sectional curvature -c.
"""

from jaxtyping import Array, Float
import jax.numpy as jnp
from ..utils.math_utils import acosh, atanh


# Default numerical parameters
MIN_NORM = 1e-15
MAX_NORM_EPS_F32 = 5e-06
MAX_NORM_EPS_F64 = 1e-08


def _get_max_norm_eps(x: Float[Array, "..."]) -> float:
    """Get maximum norm epsilon for array's dtype."""
    if x.dtype == jnp.float32:
        return MAX_NORM_EPS_F32
    elif x.dtype == jnp.float64:
        return MAX_NORM_EPS_F64
    else:
        return MAX_NORM_EPS_F32


def _minkowski_inner(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    axis: int = -1,
    keepdim: bool = True
) -> Float[Array, "..."]:
    """Compute Minkowski inner product ⟨x, y⟩_L = -x₀y₀ + ⟨x_rest, y_rest⟩.

    Args:
        x: Hyperboloid point(s)
        y: Hyperboloid point(s)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension

    Returns:
        Minkowski inner product
    """
    x0y0 = x[..., 0:1] * y[..., 0:1]
    x_rest_y_rest = jnp.sum(x[..., 1:] * y[..., 1:], axis=axis, keepdims=True)
    result = -x0y0 + x_rest_y_rest
    if not keepdim:
        result = jnp.squeeze(result, axis=-1)
    return result


def proj(
    x: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> Float[Array, "..."]:
    """Project point(s) onto hyperboloid by adjusting temporal component.

    Args:
        x: Point(s) to project
        c: Curvature (positive)
        axis: Axis along which to compute

    Returns:
        Projected point(s) with -x₀² + ||x_rest||² = -1/c, x₀ > 0
    """
    x_rest = x[..., 1:]
    x_rest_sqnorm = jnp.sum(x_rest ** 2, axis=axis, keepdims=True)
    x0_new = jnp.sqrt(jnp.maximum(1.0 / c + x_rest_sqnorm, MIN_NORM))
    return jnp.concatenate([x0_new, x_rest], axis=-1)


def addition(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Einstein gyrovector addition on hyperboloid.

    Args:
        x: Hyperboloid point(s)
        y: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to hyperboloid

    Returns:
        Einstein sum x ⊕ y

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    mink_inner_xy = _minkowski_inner(x, y, axis=axis, keepdim=True)

    # Einstein addition formula
    denom = jnp.maximum(1.0 - c * mink_inner_xy, MIN_NORM)
    gamma = 1.0 / denom

    res = x + gamma * (y + (c / (1.0 + sqrt_c)) * mink_inner_xy * x)

    if backproject:
        res = proj(res, c, axis=axis)
    return res


def scalar_mul(
    r: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Scalar multiplication r ⊗ x on hyperboloid.

    Args:
        r: Scalar factor(s)
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to hyperboloid

    Returns:
        Scaled point r ⊗ x

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # Map to tangent space, scale, map back
    v = logmap_0(x, c, axis=axis)
    v_scaled = r * v
    res = expmap_0(v_scaled, c, axis=axis, backproject=backproject)
    return res


def dist(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float,
    axis: int = -1,
    keepdim: bool = True,
    version: str = "default"
) -> Float[Array, "..."]:
    """Compute geodesic distance between hyperboloid points.

    Args:
        x: Hyperboloid point(s)
        y: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension
        version: Distance version ('default' or 'lorentzian')

    Returns:
        Geodesic distance d(x, y)

    References:
        Nickel & Kiela. "Poincaré embeddings for learning hierarchical representations." NeurIPS 2017.
    """
    sqrt_c = jnp.sqrt(c)
    mink_inner = -_minkowski_inner(x, y, axis=axis, keepdim=True)

    # Clamp to avoid numerical issues with acosh
    mink_inner = jnp.maximum(mink_inner, 1.0 + MIN_NORM)

    res = acosh(sqrt_c * mink_inner) / sqrt_c

    if not keepdim:
        res = jnp.squeeze(res, axis=axis)
    return res


def dist_0(
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    keepdim: bool = True,
    version: str = "default"
) -> Float[Array, "..."]:
    """Compute geodesic distance from hyperboloid origin.

    Args:
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension
        version: Distance version

    Returns:
        Geodesic distance d(origin, x)

    References:
        Nickel & Kiela. "Poincaré embeddings for learning hierarchical representations." NeurIPS 2017.
    """
    sqrt_c = jnp.sqrt(c)
    x0 = x[..., 0:1]

    # Clamp to avoid numerical issues
    x0_clamped = jnp.maximum(sqrt_c * x0, 1.0 + MIN_NORM)

    res = acosh(x0_clamped) / sqrt_c

    if not keepdim:
        res = jnp.squeeze(res, axis=axis)
    return res


def expmap(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Exponential map: map tangent vector v at point x to manifold.

    Args:
        v: Tangent vector(s) at x
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to hyperboloid

    Returns:
        Point exp_x(v)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    v_norm = jnp.maximum(
        jnp.sqrt(jnp.sum(v ** 2, axis=axis, keepdims=True)),
        MIN_NORM
    )

    res = jnp.cosh(sqrt_c * v_norm) * x + (jnp.sinh(sqrt_c * v_norm) / v_norm) * v

    if backproject:
        res = proj(res, c, axis=axis)
    return res


def expmap_0(
    v: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    Args:
        v: Tangent vector(s) at origin (spatial components only)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to hyperboloid

    Returns:
        Point exp_0(v)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    v_norm = jnp.maximum(
        jnp.sqrt(jnp.sum(v ** 2, axis=axis, keepdims=True)),
        MIN_NORM
    )

    # For hyperboloid with constraint -x₀² + ||x_rest||² = -1/c
    # exp_0([v]) = [cosh(√c||v||)/√c, sinh(√c||v||)/(√c||v||) * v]
    x0 = jnp.cosh(sqrt_c * v_norm) / sqrt_c
    x_rest = jnp.sinh(sqrt_c * v_norm) / (sqrt_c * v_norm) * v

    res = jnp.concatenate([x0, x_rest], axis=-1)

    if backproject:
        res = proj(res, c, axis=axis)
    return res


def retraction(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Retraction: first-order approximation of exponential map.

    Args:
        v: Tangent vector(s) at x
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to hyperboloid

    Returns:
        Point retr_x(v) ≈ exp_x(v)

    References:
        Bécigneul & Ganea. "Riemannian adaptive optimization." ICLR 2019.
    """
    res = x + v
    if backproject:
        res = proj(res, c, axis=axis)
    return res


def logmap(
    y: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Logarithmic map: map point y to tangent space at point x.

    Args:
        y: Hyperboloid point(s)
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector log_x(y)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    mink_inner = -_minkowski_inner(x, y, axis=axis, keepdim=True)
    mink_inner = jnp.maximum(mink_inner, 1.0 + MIN_NORM)

    alpha = acosh(sqrt_c * mink_inner) / sqrt_c
    alpha = jnp.maximum(alpha, MIN_NORM)

    v = y - mink_inner * x
    v_norm = jnp.maximum(
        jnp.sqrt(jnp.sum(v ** 2, axis=axis, keepdims=True)),
        MIN_NORM
    )

    res = alpha / v_norm * v
    return res


def logmap_0(
    y: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    Args:
        y: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector log_0(y) (spatial components only)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    y0 = y[..., 0:1]
    y_rest = y[..., 1:]

    # Inverse of expmap_0: v such that exp_0(v) = y
    # From exp_0: y0 = cosh(√c||v||)/√c and y_rest = sinh(√c||v||)/(√c||v||) * v
    # So: √c||v|| = acosh(√c·y0) and ||v|| = acosh(√c·y0)/√c
    y0_clamped = jnp.maximum(sqrt_c * y0, 1.0 + MIN_NORM)
    v_norm = acosh(y0_clamped) / sqrt_c

    # From y_rest = sinh(√c||v||)/(√c||v||) * v, we get:
    # v = y_rest * (√c||v||)/sinh(√c||v||)
    y_rest_norm = jnp.maximum(
        jnp.sqrt(jnp.sum(y_rest ** 2, axis=axis, keepdims=True)),
        MIN_NORM
    )

    # ||v|| * √c = acosh(√c·y0)
    sqrt_c_v_norm = sqrt_c * v_norm
    sinh_val = jnp.sinh(sqrt_c_v_norm)
    sinh_val = jnp.maximum(sinh_val, MIN_NORM)

    res = y_rest * sqrt_c_v_norm / sinh_val
    return res


def ptransp(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Parallel transport tangent vector v from point x to point y.

    Args:
        v: Tangent vector(s) at x
        x: Hyperboloid point(s)
        y: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Parallel transported tangent vector

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    mink_inner_xy = -_minkowski_inner(x, y, axis=axis, keepdim=True)
    mink_inner_xv = -_minkowski_inner(x, v, axis=axis, keepdim=True)

    mink_inner_xy = jnp.maximum(mink_inner_xy, 1.0 + MIN_NORM)

    res = v - mink_inner_xv / (mink_inner_xy + 1.0) * (x + y)
    return res


def ptransp_0(
    v: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Parallel transport tangent vector v from origin to point y.

    Args:
        v: Tangent vector(s) at origin
        y: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Parallel transported tangent vector

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    y0 = y[..., 0:1]

    # Create origin point [sqrt(1/c), 0, ..., 0]
    # For computing Minkowski inner product with origin
    # ⟨origin, v⟩_L = -sqrt(1/c) * v[0]

    v_norm = jnp.maximum(
        jnp.sqrt(jnp.sum(v ** 2, axis=axis, keepdims=True)),
        MIN_NORM
    )

    # Simplified formula for transport from origin
    res = jnp.sinh(sqrt_c * v_norm) / v_norm * v

    # Adjust for target point
    scale = 1.0 / (sqrt_c * y0 + 1.0)
    res = res * (1.0 + sqrt_c * scale)

    return res


def tangent_inner(
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    keepdim: bool = True
) -> Float[Array, "..."]:
    """Compute inner product of tangent vectors u and v at point x.

    Uses the Minkowski inner product restricted to tangent space.

    Args:
        u: Tangent vector(s) at x
        v: Tangent vector(s) at x
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension

    Returns:
        Riemannian inner product ⟨u, v⟩_x
    """
    return _minkowski_inner(u, v, axis=axis, keepdim=keepdim)


def tangent_norm(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    keepdim: bool = True
) -> Float[Array, "..."]:
    """Compute norm of tangent vector v at point x.

    Args:
        v: Tangent vector(s) at x
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension

    Returns:
        Riemannian norm ||v||_x
    """
    inner = tangent_inner(v, v, x, c, axis=axis, keepdim=True)
    res = jnp.sqrt(jnp.maximum(inner, MIN_NORM))

    if not keepdim:
        res = jnp.squeeze(res, axis=axis)
    return res


def egrad2rgrad(
    grad: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> Float[Array, "..."]:
    """Convert Euclidean gradient to Riemannian gradient.

    Projects Euclidean gradient onto tangent space.

    Args:
        grad: Euclidean gradient
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute

    Returns:
        Riemannian gradient

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # Project onto tangent space: grad - ⟨grad, x⟩_L · x
    mink_inner_grad_x = _minkowski_inner(grad, x, axis=axis, keepdim=True)
    rgrad = grad - mink_inner_grad_x * x
    return rgrad


def tangent_proj(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> Float[Array, "..."]:
    """Project vector v onto tangent space at point x.

    Args:
        v: Vector(s) to project
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute

    Returns:
        Projected vector onto tangent space
    """
    # Tangent space orthogonal to x in Minkowski metric
    mink_inner_vx = _minkowski_inner(v, x, axis=axis, keepdim=True)
    return v - mink_inner_vx * x


def is_in_manifold(
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    atol: float = 1e-5
) -> bool:
    """Check if point(s) x lie on hyperboloid.

    Args:
        x: Point(s) to check
        c: Curvature (positive)
        axis: Axis along which to check
        atol: Absolute tolerance

    Returns:
        True if -x₀² + ||x_rest||² = -1/c for all points and x₀ > 0
    """
    x0 = x[..., 0]
    x_rest = x[..., 1:]
    x_rest_sqnorm = jnp.sum(x_rest ** 2, axis=axis)

    # Check constraint: -x₀² + ||x_rest||² = -1/c
    constraint = -x0 ** 2 + x_rest_sqnorm + 1.0 / c

    # Check x₀ > 0
    valid_constraint = jnp.allclose(constraint, 0.0, atol=atol)
    valid_x0 = jnp.all(x0 > 0)

    return bool(valid_constraint and valid_x0)


def is_in_tangent_space(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    atol: float = 1e-5
) -> bool:
    """Check if vector(s) v lie in tangent space at point x.

    Tangent space is orthogonal to x in Minkowski metric: ⟨v, x⟩_L = 0

    Args:
        v: Vector(s) to check
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to check
        atol: Absolute tolerance

    Returns:
        True if ⟨v, x⟩_L ≈ 0 for all vectors
    """
    mink_inner = _minkowski_inner(v, x, axis=axis, keepdim=False)
    return bool(jnp.allclose(mink_inner, 0.0, atol=atol))