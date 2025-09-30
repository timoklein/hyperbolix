"""Poincaré Ball manifold - pure functional implementation.

Direct JAX port of PyTorch poincare.py with pure functions.
Convention: ||x||^2 < 1/c with c > 0 and sectional curvature -c.
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


def _conformal_factor(
    x: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> Float[Array, "..."]:
    """Compute conformal factor λ(x) = 2 / (1 - c||x||²).

    Args:
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute

    Returns:
        Conformal factor λ(x)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    x2 = jnp.sum(x ** 2, axis=axis, keepdims=True)
    max_norm_eps = _get_max_norm_eps(x)
    denom = jnp.maximum(
        1.0 - c * x2,
        2 * jnp.sqrt(c) * max_norm_eps - c * max_norm_eps ** 2
    )
    return 2.0 / denom


def _gyration(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    z: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> Float[Array, "..."]:
    """Compute gyration gyr[x,y]z to restore commutativity.

    Args:
        x: Poincaré ball point(s)
        y: Poincaré ball point(s)
        z: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute

    Returns:
        Gyration gyr[x,y]z

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    c2 = c ** 2
    x2 = jnp.sum(x ** 2, axis=axis, keepdims=True)
    y2 = jnp.sum(y ** 2, axis=axis, keepdims=True)
    xy = jnp.sum(x * y, axis=axis, keepdims=True)
    xz = jnp.sum(x * z, axis=axis, keepdims=True)
    yz = jnp.sum(y * z, axis=axis, keepdims=True)

    a = -c2 * xz * y2 + c * yz + 2 * c2 * xy * yz
    b = -c2 * yz * x2 - c * xz
    num = 2 * (a * x + b * y)
    denom = jnp.maximum(1 + 2 * c * xy + c2 * x2 * y2, MIN_NORM)

    return z + num / denom


def proj(
    x: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> Float[Array, "..."]:
    """Project point(s) onto Poincaré ball by clipping norm.

    Args:
        x: Point(s) to project
        c: Curvature (positive)
        axis: Axis along which to compute norm

    Returns:
        Projected point(s) with ||x|| < 1/√c
    """
    max_norm_eps = _get_max_norm_eps(x)
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    max_norm = (1.0 / jnp.sqrt(c)) - max_norm_eps
    cond = norm > max_norm
    return jnp.where(cond, x * (max_norm / jnp.maximum(norm, MIN_NORM)), x)


def addition(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Möbius gyrovector addition x ⊕ y.

    Non-commutative and non-associative!

    Args:
        x: Poincaré ball point(s)
        y: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to ball

    Returns:
        Möbius sum x ⊕ y

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    x2 = jnp.sum(x ** 2, axis=axis, keepdims=True)
    y2 = jnp.sum(y ** 2, axis=axis, keepdims=True)
    xy = jnp.sum(x * y, axis=axis, keepdims=True)

    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = jnp.maximum(1 + 2 * c * xy + c ** 2 * x2 * y2, MIN_NORM)
    res = num / denom

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
    """Scalar multiplication r ⊗ x on Poincaré ball.

    Args:
        r: Scalar factor(s)
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to ball

    Returns:
        Scaled point r ⊗ x

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    x_norm = jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), MIN_NORM)
    c_norm_prod = jnp.sqrt(c) * x_norm
    res = jnp.tanh(r * atanh(c_norm_prod)) / c_norm_prod * x

    if backproject:
        res = proj(res, c, axis=axis)
    return res


def dist(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    c: float,
    axis: int = -1,
    keepdim: bool = True,
    version: str = "mobius_direct"
) -> Float[Array, "..."]:
    """Compute geodesic distance between Poincaré ball points.

    Args:
        x: Poincaré ball point(s)
        y: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension
        version: Distance version ('mobius_direct', 'mobius', 'metric_tensor', 'lorentzian_proxy')

    Returns:
        Geodesic distance d(x, y)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
        Law et al. "Lorentzian distance learning." ICML 2019.
    """
    if version in ["mobius_direct", "default"]:
        # Symmetric Möbius distance
        sqrt_c = jnp.sqrt(c)
        x2y2 = jnp.sum(x ** 2, axis=axis, keepdims=True) * jnp.sum(y ** 2, axis=axis, keepdims=True)
        xy = jnp.sum(x * y, axis=axis, keepdims=True)
        num = jnp.linalg.norm(y - x, axis=axis, keepdims=True)
        denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c ** 2 * x2y2, MIN_NORM))
        xysum_norm = num / denom
        dist_c = atanh(sqrt_c * xysum_norm)
        res = 2 * dist_c / sqrt_c
    elif version == "mobius":
        # Möbius distance via addition
        sqrt_c = jnp.sqrt(c)
        diff = addition(-x, y, c, axis=axis, backproject=True)
        dist_c = atanh(sqrt_c * jnp.linalg.norm(diff, axis=axis, keepdims=True))
        res = 2 * dist_c / sqrt_c
    elif version == "metric_tensor":
        # Metric tensor induced distance
        x_sqnorm = jnp.sum(x ** 2, axis=axis, keepdims=True)
        y_sqnorm = jnp.sum(y ** 2, axis=axis, keepdims=True)
        xy_diff_sqnorm = jnp.sum((x - y) ** 2, axis=axis, keepdims=True)
        arg = 1 + 2 * c * xy_diff_sqnorm / ((1 - c * x_sqnorm) * (1 - c * y_sqnorm))
        condition = arg < 1 + MIN_NORM
        res = jnp.where(condition, jnp.zeros_like(arg), acosh(arg) / jnp.sqrt(c))
    elif version == "lorentzian_proxy":
        # Lorentzian proxy distance
        xy_prod = x * y
        xy0 = xy_prod[..., 0:1]
        xy_rem = jnp.sum(xy_prod[..., 1:], axis=axis, keepdims=True)
        xy_mink = xy_rem - xy0
        res = -2 / c - 2 * xy_mink
    else:
        raise ValueError(f"Unknown version: {version}")

    if not keepdim:
        res = jnp.squeeze(res, axis=axis)
    return res


def dist_0(
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    keepdim: bool = True,
    version: str = "mobius_direct"
) -> Float[Array, "..."]:
    """Compute geodesic distance from Poincaré ball origin.

    Args:
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension
        version: Distance version ('mobius_direct', 'mobius', 'metric_tensor', 'lorentzian_proxy')

    Returns:
        Geodesic distance d(0, x)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    if version in ["mobius_direct", "mobius", "default"]:
        sqrt_c = jnp.sqrt(c)
        dist_c = atanh(sqrt_c * jnp.linalg.norm(x, axis=axis, keepdims=True))
        res = 2 * dist_c / sqrt_c
    elif version == "metric_tensor":
        x_sqnorm = jnp.sum(x ** 2, axis=axis, keepdims=True)
        arg = 1 + 2 * c * x_sqnorm / (1 - c * x_sqnorm)
        condition = arg < 1 + MIN_NORM
        res = jnp.where(condition, jnp.zeros_like(arg), acosh(arg) / jnp.sqrt(c))
    elif version == "lorentzian_proxy":
        x0 = x[..., 0:1]
        res = -2 / c + 2 * x0 / jnp.sqrt(c)
    else:
        raise ValueError(f"Unknown version: {version}")

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
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to ball

    Returns:
        Point exp_x(v)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    v_norm = jnp.linalg.norm(v, axis=axis, keepdims=True)
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * v_norm, MIN_NORM)
    lambda_x = _conformal_factor(x, c, axis=axis)
    second_term = jnp.tanh(c_norm_prod * lambda_x / 2) / c_norm_prod * v

    if backproject:
        second_term = proj(second_term, c, axis=axis)
    res = addition(x, second_term, c, axis=axis, backproject=backproject)
    return res


def expmap_0(
    v: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    Args:
        v: Tangent vector(s) at origin
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to ball

    Returns:
        Point exp_0(v)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    v_norm = jnp.linalg.norm(v, axis=axis, keepdims=True)
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * v_norm, MIN_NORM)
    res = jnp.tanh(c_norm_prod) / c_norm_prod * v

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
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to project result back to ball

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
        y: Poincaré ball point(s)
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector log_x(y)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sub = addition(-x, y, c, axis=axis)
    x2y2 = jnp.sum(x ** 2, axis=axis, keepdims=True) * jnp.sum(y ** 2, axis=axis, keepdims=True)
    xy = jnp.sum(x * y, axis=axis, keepdims=True)
    num = jnp.linalg.norm(y - x, axis=axis, keepdims=True)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c ** 2 * x2y2, MIN_NORM))
    sub_norm = num / denom
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * sub_norm, MIN_NORM)
    lambda_x = _conformal_factor(x, c, axis=axis)
    res = 2 * atanh(c_norm_prod) / (c_norm_prod * lambda_x) * sub
    return res


def logmap_0(
    y: Float[Array, "..."],
    c: float,
    axis: int = -1,
    backproject: bool = True
) -> Float[Array, "..."]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    Args:
        y: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Tangent vector log_0(y)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    y_norm = jnp.linalg.norm(y, axis=axis, keepdims=True)
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * y_norm, MIN_NORM)
    res = atanh(c_norm_prod) / c_norm_prod * y
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
        x: Poincaré ball point(s)
        y: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Parallel transported tangent vector

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_x = _conformal_factor(x, c, axis=axis)
    lambda_y = _conformal_factor(y, c, axis=axis)
    return _gyration(y, -x, v, c, axis=axis) * (lambda_x / lambda_y)


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
        y: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        backproject: Whether to backproject (ignored, kept for consistency)

    Returns:
        Parallel transported tangent vector

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_y = _conformal_factor(y, c, axis=axis)
    return v / lambda_y


def tangent_inner(
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    keepdim: bool = True
) -> Float[Array, "..."]:
    """Compute inner product of tangent vectors u and v at point x.

    Args:
        u: Tangent vector(s) at x
        v: Tangent vector(s) at x
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension

    Returns:
        Riemannian inner product <u, v>_x

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_x = _conformal_factor(x, c, axis=axis)
    res = lambda_x ** 2 * jnp.sum(u * v, axis=axis, keepdims=keepdim)
    return res


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
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute
        keepdim: Whether to keep the reduced dimension

    Returns:
        Riemannian norm ||v||_x

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_x = _conformal_factor(x, c, axis=axis)
    if keepdim:
        res = lambda_x * jnp.linalg.norm(v, axis=axis, keepdims=True)
    else:
        res = lambda_x * jnp.linalg.norm(v, axis=axis, keepdims=False)
        res = jnp.squeeze(res, axis=axis if axis >= 0 else v.ndim + axis)
    return res


def egrad2rgrad(
    grad: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> Float[Array, "..."]:
    """Convert Euclidean gradient to Riemannian gradient.

    Args:
        grad: Euclidean gradient
        x: Poincaré ball point(s)
        c: Curvature (positive)
        axis: Axis along which to compute

    Returns:
        Riemannian gradient

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_x = _conformal_factor(x, c, axis=axis)
    return grad / (lambda_x ** 2)


def tangent_proj(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> Float[Array, "..."]:
    """Project vector v onto tangent space at point x.

    In Poincaré ball, tangent space equals ambient space (identity).

    Args:
        v: Vector(s) to project
        x: Poincaré ball point(s) (ignored)
        c: Curvature (ignored, kept for consistency)
        axis: Axis along which to compute (ignored, kept for consistency)

    Returns:
        Projected vector v (unchanged)
    """
    return v


def is_in_manifold(
    x: Float[Array, "..."],
    c: float,
    axis: int = -1,
    atol: float = 1e-5
) -> bool:
    """Check if point(s) x lie in Poincaré ball.

    Args:
        x: Point(s) to check
        c: Curvature (positive)
        axis: Axis along which to compute norm
        atol: Absolute tolerance

    Returns:
        True if ||x||² < 1/c for all points
    """
    x_sqnorm = jnp.sum(x ** 2, axis=axis)
    return bool(jnp.all(x_sqnorm < (1.0 / c) - atol))


def is_in_tangent_space(
    v: Float[Array, "..."],
    x: Float[Array, "..."],
    c: float,
    axis: int = -1
) -> bool:
    """Check if vector(s) v lie in tangent space at point x.

    In Poincaré ball, all vectors are valid tangent vectors.

    Args:
        v: Vector(s) to check
        x: Poincaré ball point(s) (ignored)
        c: Curvature (ignored, kept for consistency)
        axis: Axis along which to check (ignored, kept for consistency)

    Returns:
        Always True
    """
    return True