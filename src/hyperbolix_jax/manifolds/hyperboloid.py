"""Hyperboloid manifold - pure functional implementation.

Direct JAX port following the Lorentz/hyperboloid model.
Convention: -x₀² + ||x_rest||² = -1/c with c > 0, x₀ > 0, and sectional curvature -c.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import acosh, smooth_clamp_min

# Default numerical parameters
MIN_NORM = 1e-15


def _create_origin_from_reference(reference: Float[Array, "..."], c: float, axis: int = -1) -> Float[Array, "..."]:
    """Return the hyperboloid origin with the same shape/dtype as ``reference``."""
    sqrt_c = jnp.sqrt(c)
    origin = jnp.zeros_like(reference)
    axis = axis if axis >= 0 else reference.ndim + axis
    index = [slice(None)] * reference.ndim
    index[axis] = slice(0, 1)
    origin = origin.at[tuple(index)].set(1.0 / sqrt_c)
    return origin


def _minkowski_inner(
    x: Float[Array, "..."], y: Float[Array, "..."], axis: int = -1, keepdim: bool = True
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


def proj(x: Float[Array, "..."], c: float, axis: int = -1) -> Float[Array, "..."]:
    """Project point(s) onto hyperboloid by adjusting temporal component.

    Args:
        x: Point(s) to project
        c: Curvature (positive)
        axis: Axis along which to compute

    Returns:
        Projected point(s) with -x₀² + ||x_rest||² = -1/c, x₀ > 0
    """
    x_rest = x[..., 1:]
    x_rest_sqnorm = jnp.sum(x_rest**2, axis=axis, keepdims=True)
    x0_new = jnp.sqrt(jnp.maximum(1.0 / c + x_rest_sqnorm, MIN_NORM))
    return jnp.concatenate([x0_new, x_rest], axis=-1)


def addition(
    x: Float[Array, "..."], y: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True
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
    r: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True
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
    # Map to tangent space, scale geodesic length, map back
    v = logmap_0(x, c, axis=axis)
    v_sqnorm = _minkowski_inner(v, v, axis=axis, keepdim=True)
    v_norm = jnp.sqrt(jnp.maximum(v_sqnorm, MIN_NORM))
    unit_tangent = v / v_norm
    dist0 = dist_0(x, c, axis=axis, keepdim=True)
    tangent = r * dist0 * unit_tangent
    res = expmap_0(tangent, c, axis=axis, backproject=backproject)
    return res


def dist(
    x: Float[Array, "..."], y: Float[Array, "..."], c: float, axis: int = -1, keepdim: bool = True, version: str = "default"
) -> Float[Array, "..."]:
    """Compute geodesic distance between hyperboloid points.

    Args:
        x: Hyperboloid point(s)
        y: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension
        version: Distance version ('default' or 'smoothened')

    Returns:
        Geodesic distance d(x, y)

    References:
        Nickel & Kiela. "Poincaré embeddings for learning hierarchical representations." NeurIPS 2017.
    """
    sqrt_c = jnp.sqrt(c)
    lorentz_inner = _minkowski_inner(x, y, axis=axis, keepdim=True)

    arg = -c * lorentz_inner
    if version == "smoothened":
        arg = smooth_clamp_min(arg, 1.0)
    else:
        # Use hard clipping when explicitly requested (e.g., version="normal")
        arg = jnp.clip(arg, min=1.0)

    res = acosh(arg) / sqrt_c

    same = jnp.all(jnp.equal(x, y), axis=axis, keepdims=True)
    res = jnp.where(same, jnp.zeros_like(res), res)

    if not keepdim:
        same = jnp.squeeze(same, axis=axis)
        res = jnp.squeeze(res, axis=axis)
        res = jnp.where(same, jnp.zeros_like(res), res)
    return res


def dist_0(
    x: Float[Array, "..."], c: float, axis: int = -1, keepdim: bool = True, version: str = "default"
) -> Float[Array, "..."]:
    """Compute geodesic distance from hyperboloid origin.

    Args:
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension
        version: Distance version ('default' or 'smoothened')

    Returns:
        Geodesic distance d(origin, x)

    References:
        Nickel & Kiela. "Poincaré embeddings for learning hierarchical representations." NeurIPS 2017.
    """
    sqrt_c = jnp.sqrt(c)
    x0 = x[..., 0:1]

    arg = sqrt_c * x0
    if version == "smoothened":
        arg = smooth_clamp_min(arg, 1.0)
    else:
        arg = jnp.clip(arg, min=1.0)

    res = acosh(arg) / sqrt_c

    axis_index = axis if axis >= 0 else x.ndim + axis
    origin = jnp.zeros_like(x)
    selector = [slice(None)] * x.ndim
    selector[axis_index] = slice(0, 1)
    origin = origin.at[tuple(selector)].set(1.0 / sqrt_c)

    at_origin = jnp.all(jnp.equal(x, origin), axis=axis, keepdims=True)
    res = jnp.where(at_origin, jnp.zeros_like(res), res)

    if not keepdim:
        at_origin = jnp.squeeze(at_origin, axis=axis)
        res = jnp.squeeze(res, axis=axis)
        res = jnp.where(at_origin, jnp.zeros_like(res), res)
    return res


def expmap(
    v: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True
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
    v_sqnorm = jnp.clip(_minkowski_inner(v, v, axis=axis, keepdim=True), min=0.0)
    v_norm = jnp.sqrt(v_sqnorm)
    c_norm_prod = sqrt_c * v_norm

    denom = jnp.maximum(c_norm_prod, MIN_NORM)
    cosh_term = jnp.cosh(c_norm_prod) * x
    sinh_term = jnp.sinh(c_norm_prod) / denom * v

    res = cosh_term + sinh_term

    if backproject:
        res = proj(res, c, axis=axis)
    return res


def expmap_0(v: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True) -> Float[Array, "..."]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    Args:
        v: Tangent vector(s) at origin in ambient representation (n+1 dim, first component should be 0)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to hyperboloid

    Returns:
        Point exp_0(v) in ambient representation

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    v_sqnorm = jnp.clip(_minkowski_inner(v, v, axis=axis, keepdim=True), min=0.0)
    v_norm = jnp.sqrt(v_sqnorm)
    c_norm_prod = sqrt_c * v_norm

    denom = jnp.maximum(c_norm_prod, MIN_NORM)
    sinh_scale = jnp.sinh(c_norm_prod) / denom

    v0 = v[..., 0:1]
    v_rest = v[..., 1:]

    res0 = jnp.cosh(c_norm_prod) / sqrt_c + sinh_scale * v0
    res_rest = sinh_scale * v_rest

    res = jnp.concatenate([res0, res_rest], axis=-1)

    if backproject:
        res = proj(res, c, axis=axis)
    return res


def retraction(
    v: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True
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
    y: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True
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
    mink_inner = _minkowski_inner(x, y, axis=axis, keepdim=True)
    dist_xy = dist(x, y, c=c, axis=axis, keepdim=True)
    direction = y + c * mink_inner * x

    dir_sqnorm = _minkowski_inner(direction, direction, axis=axis, keepdim=True)
    dir_norm = jnp.sqrt(jnp.maximum(dir_sqnorm, MIN_NORM))
    res = dist_xy * direction / dir_norm

    if backproject:
        res = tangent_proj(res, x, c, axis=axis)

    return res


def logmap_0(y: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True) -> Float[Array, "..."]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    Args:
        y: Hyperboloid point(s) in ambient representation
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector log_0(y) in ambient representation (first component is 0)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    y_rest = y[..., 1:]
    y_rest_norm = jnp.linalg.norm(y_rest, axis=axis, keepdims=True)

    dist0 = dist_0(y, c=c, axis=axis, keepdim=True)
    scale = dist0 / jnp.maximum(y_rest_norm, MIN_NORM)

    v0 = jnp.zeros_like(dist0)
    v_rest = scale * y_rest
    res = jnp.concatenate([v0, v_rest], axis=-1)

    if backproject:
        origin = _create_origin_from_reference(res, c, axis=axis)
        res = tangent_proj(res, origin, c, axis=axis)

    return res


def ptransp(
    v: Float[Array, "..."], x: Float[Array, "..."], y: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True
) -> Float[Array, "..."]:
    """Parallel transport tangent vector v from point x to point y.

    Args:
        v: Tangent vector(s) at x
        x: Hyperboloid point(s)
        y: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject to tangent space at y

    Returns:
        Parallel transported tangent vector

    References:
        Aaron Lou, et al. "Differentiating through the fréchet mean."
            International conference on machine learning (2020).
    """
    # Compute Minkowski inner products
    vy = _minkowski_inner(v, y, axis=axis, keepdim=True)  # ⟨v, y⟩_L
    xy = _minkowski_inner(x, y, axis=axis, keepdim=True)  # ⟨x, y⟩_L

    # denom = 1/c - ⟨x, y⟩_L
    denom = 1.0 / c - xy
    denom = jnp.maximum(denom, MIN_NORM)  # Numerical stability

    # scale = ⟨v, y⟩_L / denom
    scale = vy / denom

    # res = v + scale * (x + y)
    res = v + scale * (x + y)

    if backproject:
        res = tangent_proj(res, y, c, axis=axis)

    return res


def ptransp_0(
    v: Float[Array, "..."], y: Float[Array, "..."], c: float, axis: int = -1, backproject: bool = True
) -> Float[Array, "..."]:
    """Parallel transport tangent vector v from origin to point y.

    Args:
        v: Tangent vector(s) at origin
        y: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject to tangent space at y

    Returns:
        Parallel transported tangent vector

    References:
        Aaron Lou, et al. "Differentiating through the fréchet mean."
            International conference on machine learning (2020).
    """
    # Create origin point [1/√c, 0, ..., 0] with same shape as y
    sqrt_c = jnp.sqrt(c)
    y0 = y[..., 0:1]

    # Build origin vector with appropriate shape
    origin = _create_origin_from_reference(y, c, axis=axis)

    # Compute Minkowski inner products
    vy = _minkowski_inner(v, y, axis=axis, keepdim=True)  # ⟨v, y⟩_L

    # denom = 1/c + y0/√c (from ⟨origin, y⟩_L = -y0/√c and denom = 1/c - ⟨origin, y⟩_L)
    denom = 1.0 / c + y0 / sqrt_c
    denom = jnp.maximum(denom, MIN_NORM)  # Numerical stability

    # scale = ⟨v, y⟩_L / denom
    scale = vy / denom

    # res = v + scale * (y + origin)
    res = v + scale * (y + origin)

    if backproject:
        res = tangent_proj(res, y, c, axis=axis)

    return res


def tangent_inner(
    u: Float[Array, "..."], v: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1, keepdim: bool = True
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
    v: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1, keepdim: bool = True
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
    res = jnp.sqrt(jnp.clip(inner, min=0.0))

    if not keepdim:
        res = jnp.squeeze(res, axis=axis)
    return res


def egrad2rgrad(grad: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1) -> Float[Array, "..."]:
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
    # In Lorentzian signature the temporal component carries a negative sign.
    # Flip it before projecting so we project the Riemannian gradient, matching PyTorch.
    grad_lorentz = grad.at[..., 0].set(-grad[..., 0])

    # Orthogonally project the Lorentzian gradient onto the tangent space.
    inner_xx = _minkowski_inner(x, x, axis=axis, keepdim=True)
    scale = jnp.sqrt(jnp.maximum(-c * inner_xx, MIN_NORM))
    x_normed = x / scale

    denom = _minkowski_inner(x_normed, x_normed, axis=axis, keepdim=True)
    coeff = _minkowski_inner(x_normed, grad_lorentz, axis=axis, keepdim=True) / denom
    return grad_lorentz - coeff * x_normed


def tangent_proj(v: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1) -> Float[Array, "..."]:
    """Project vector v onto tangent space at point x.

    Args:
        v: Vector(s) to project
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to compute

    Returns:
        Projected vector onto tangent space
    """
    # Normalize x w.r.t. measured Lorentz norm (robust in float32)
    inner_xx = _minkowski_inner(x, x, axis=axis, keepdim=True)
    scale = jnp.sqrt(jnp.maximum(-c * inner_xx, MIN_NORM))
    x_normed = x / scale

    denom = _minkowski_inner(x_normed, x_normed, axis=axis, keepdim=True)
    coeff = _minkowski_inner(x_normed, v, axis=axis, keepdim=True) / denom
    return v - coeff * x_normed


def is_in_manifold(x: Float[Array, "..."], c: float, axis: int = -1, atol: float = 1e-5) -> bool:
    """Check if point(s) x lie on hyperboloid.

    Args:
        x: Point(s) to check
        c: Curvature (positive)
        axis: Axis along which to check
        atol: Absolute tolerance

    Returns:
        True if -x₀² + ||x_rest||² = -1/c for all points and x₀ > 0
    """
    lorentz_norm = _minkowski_inner(x, x, axis=axis, keepdim=False)
    tol = max(atol, 1e-4)
    target = -1.0 / c

    valid_constraint = jnp.all(jnp.isclose(lorentz_norm, target, atol=tol, rtol=0.0))
    valid_x0 = jnp.all(x[..., 0] > 0)

    return bool(valid_constraint and valid_x0)


def is_in_tangent_space(
    v: Float[Array, "..."], x: Float[Array, "..."], c: float, axis: int = -1, atol: float | None = None
) -> bool:
    """Check if vector(s) v lie in tangent space at point x.

    Tangent space is orthogonal to x in Minkowski metric: ⟨v, x⟩_L = 0

    Args:
        v: Vector(s) to check
        x: Hyperboloid point(s)
        c: Curvature (positive)
        axis: Axis along which to check
        atol: Absolute tolerance (dtype-aware if None)

    Returns:
        True if ⟨v, x⟩_L ≈ 0 for all vectors
    """
    tol = 5e-4 if atol is None else atol
    mink_inner = _minkowski_inner(v, x, axis=axis, keepdim=False)
    return bool(jnp.all(jnp.abs(mink_inner) < tol))
