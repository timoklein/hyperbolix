"""Poincaré Ball manifold - vmap-native pure functional implementation.

JAX port with vmap-native API. All functions operate on single points/vectors
with shape (dim,). Use jax.vmap for batch operations.

Convention: ||x||^2 < 1/c with c > 0 and sectional curvature -c.

JIT Compilation & Batching
---------------------------
All functions work with single points and return scalars or vectors.
Use jax.vmap for batching and jax.jit for compilation:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix_jax.manifolds import poincare
    >>>
    >>> # Single point operations
    >>> x = jnp.array([0.1, 0.2])
    >>> y = jnp.array([0.3, 0.4])
    >>> distance = poincare.dist(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)
    >>>
    >>> # Batch operations with vmap
    >>> x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])  # (batch, dim)
    >>> y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45]])
    >>> dist_batched = jax.vmap(poincare.dist, in_axes=(0, 0, None, None))
    >>> distances = dist_batched(x_batch, y_batch, 1.0, poincare.VERSION_MOBIUS_DIRECT)
    >>>
    >>> # JIT compilation
    >>> dist_jit = jax.jit(poincare.dist, static_argnames=['version_idx'])
    >>> distance = dist_jit(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)

Version Constants:
    VERSION_MOBIUS_DIRECT (0): Direct Möbius distance formula (fastest)
    VERSION_MOBIUS (1): Möbius distance via addition
    VERSION_METRIC_TENSOR (2): Metric tensor induced distance
    VERSION_LORENTZIAN_PROXY (3): Lorentzian proxy distance

Note: Keep curvature parameter 'c' dynamic to support learnable curvature.
Use version_idx as static argument for JIT (static_argnames=['version_idx']).

Numerical Precision and Float32 Limitations
-------------------------------------------
Operations involving points near the boundary (||x|| ≈ 1/√c) can suffer from
numerical instability, especially with float32. The conformal factor λ(x) = 2/(1-c||x||²)
grows exponentially as points approach the boundary:

- At d(0,x) ≈ 5: λ(x) ≈ 100
- At d(0,x) ≈ 7: λ(x) ≈ 1,000
- At d(0,x) ≈ 10: λ(x) ≈ 10,000+

Float32 (~7 significant digits) loses precision in operations like:
- logmap/tangent_norm: divide by λ(x), then multiply by λ(x)
- expmap: multiplies by large λ(x) values
- addition: combines terms with vastly different scales

For numerical accuracy with large distances or near-boundary points:
- Use float64 when possible
- Expect ~3% relative error with float32 for distances > 10
- Consider projection after operations to maintain manifold constraints
"""

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import acosh, atanh

# Default numerical parameters
MIN_NORM = 1e-15

# Version selection constants for dist() and dist_0()
VERSION_MOBIUS_DIRECT = 0
VERSION_MOBIUS = 1
VERSION_METRIC_TENSOR = 2
VERSION_LORENTZIAN_PROXY = 3


def _get_max_norm_eps(x: Float[Array, "dim"]) -> float:
    """Get maximum norm epsilon for array's dtype.

    Uses eps^0.75 as empirically stable value that scales with precision.
    """
    return jnp.finfo(x.dtype).eps ** 0.75


def _conformal_factor(x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Compute conformal factor λ(x) = 2 / (1 - c||x||²).

    Args:
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Conformal factor λ(x), scalar

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    x2 = jnp.dot(x, x)
    max_norm_eps = _get_max_norm_eps(x)
    denom = jnp.maximum(1.0 - c * x2, 2 * jnp.sqrt(c) * max_norm_eps - c * max_norm_eps**2)
    return 2.0 / denom


def _gyration(x: Float[Array, "dim"], y: Float[Array, "dim"], z: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Compute gyration gyr[x,y]z to restore commutativity.

    Args:
        x: Poincaré ball point, shape (dim,)
        y: Poincaré ball point, shape (dim,)
        z: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Gyration gyr[x,y]z, shape (dim,)

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    c2 = c**2
    x2 = jnp.dot(x, x)
    y2 = jnp.dot(y, y)
    xy = jnp.dot(x, y)
    xz = jnp.dot(x, z)
    yz = jnp.dot(y, z)

    a = -c2 * xz * y2 + c * yz + 2 * c2 * xy * yz
    b = -c2 * yz * x2 - c * xz
    num = 2 * (a * x + b * y)
    denom = jnp.maximum(1 + 2 * c * xy + c2 * x2 * y2, MIN_NORM)

    return z + num / denom


def proj(x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Project point onto Poincaré ball by clipping norm.

    Args:
        x: Point to project, shape (dim,)
        c: Curvature (positive)

    Returns:
        Projected point with ||x|| < 1/√c, shape (dim,)
    """
    max_norm_eps = _get_max_norm_eps(x)
    norm = jnp.linalg.norm(x)
    max_norm = (1.0 / jnp.sqrt(c)) - max_norm_eps
    cond = norm > max_norm
    return jnp.where(cond, x * (max_norm / jnp.maximum(norm, MIN_NORM)), x)


def addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Möbius gyrovector addition x ⊕ y.

    Non-commutative and non-associative!

    Args:
        x: Poincaré ball point, shape (dim,)
        y: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Möbius sum x ⊕ y, shape (dim,)

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    x2 = jnp.dot(x, x)
    y2 = jnp.dot(y, y)
    xy = jnp.dot(x, y)

    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = jnp.maximum(1 + 2 * c * xy + c**2 * x2 * y2, MIN_NORM)
    res = num / denom
    res = proj(res, c)
    return res


def scalar_mul(r: float, x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Scalar multiplication r ⊗ x on Poincaré ball.

    Args:
        r: Scalar factor
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Scaled point r ⊗ x, shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    x_norm = jnp.maximum(jnp.linalg.norm(x), MIN_NORM)
    c_norm_prod = jnp.sqrt(c) * x_norm
    res = jnp.tanh(r * atanh(c_norm_prod)) / c_norm_prod * x
    res = proj(res, c)
    return res


# Distance implementations for lax.switch
def _dist_mobius_direct(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Direct Möbius distance formula (fastest)."""
    sqrt_c = jnp.sqrt(c)
    x2y2 = jnp.dot(x, x) * jnp.dot(y, y)
    xy = jnp.dot(x, y)
    num = jnp.linalg.norm(y - x)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c**2 * x2y2, MIN_NORM))
    xysum_norm = num / denom
    dist_c = atanh(sqrt_c * xysum_norm)
    return 2 * dist_c / sqrt_c


def _dist_mobius(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Möbius distance via addition."""
    sqrt_c = jnp.sqrt(c)
    diff = addition(-x, y, c)
    dist_c = atanh(sqrt_c * jnp.linalg.norm(diff))
    return 2 * dist_c / sqrt_c


def _dist_metric_tensor(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Metric tensor induced distance."""
    x_sqnorm = jnp.dot(x, x)
    y_sqnorm = jnp.dot(y, y)
    xy_diff_sqnorm = jnp.dot(x - y, x - y)
    arg = 1 + 2 * c * xy_diff_sqnorm / ((1 - c * x_sqnorm) * (1 - c * y_sqnorm))
    condition = arg < 1 + MIN_NORM
    return jnp.where(condition, 0.0, acosh(arg) / jnp.sqrt(c))


def _dist_lorentzian_proxy(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Lorentzian proxy distance."""
    xy_prod = x * y
    xy0 = xy_prod[0]
    xy_rem = jnp.sum(xy_prod[1:])
    xy_mink = xy_rem - xy0
    return -2 / c - 2 * xy_mink


def dist(
    x: Float[Array, "dim"],
    y: Float[Array, "dim"],
    c: float,
    version_idx: int = VERSION_MOBIUS_DIRECT,
) -> Float[Array, ""]:
    """Compute geodesic distance between Poincaré ball points.

    Args:
        x: Poincaré ball point, shape (dim,)
        y: Poincaré ball point, shape (dim,)
        c: Curvature (positive)
        version_idx: Distance version index (use VERSION_* constants)

    Returns:
        Geodesic distance d(x, y), scalar

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
        Law et al. "Lorentzian distance learning." ICML 2019.
    """
    return lax.switch(version_idx, [_dist_mobius_direct, _dist_mobius, _dist_metric_tensor, _dist_lorentzian_proxy], x, y, c)


# Distance from origin implementations for lax.switch
def _dist_0_mobius(x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Möbius distance from origin (mobius_direct and mobius use same formula)."""
    sqrt_c = jnp.sqrt(c)
    dist_c = atanh(sqrt_c * jnp.linalg.norm(x))
    return 2 * dist_c / sqrt_c


def _dist_0_metric_tensor(x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Metric tensor distance from origin."""
    x_sqnorm = jnp.dot(x, x)
    arg = 1 + 2 * c * x_sqnorm / (1 - c * x_sqnorm)
    condition = arg < 1 + MIN_NORM
    return jnp.where(condition, 0.0, acosh(arg) / jnp.sqrt(c))


def _dist_0_lorentzian_proxy(x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Lorentzian proxy distance from origin."""
    x0 = x[0]
    return -2 / c + 2 * x0 / jnp.sqrt(c)


def dist_0(x: Float[Array, "dim"], c: float, version_idx: int = VERSION_MOBIUS_DIRECT) -> Float[Array, ""]:
    """Compute geodesic distance from Poincaré ball origin.

    Args:
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)
        version_idx: Distance version index (use VERSION_* constants)
                     Note: VERSION_MOBIUS_DIRECT and VERSION_MOBIUS produce same result

    Returns:
        Geodesic distance d(0, x), scalar

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # mobius_direct and mobius use same implementation for dist_0
    return lax.switch(version_idx, [_dist_0_mobius, _dist_0_mobius, _dist_0_metric_tensor, _dist_0_lorentzian_proxy], x, c)


def expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Exponential map: map tangent vector v at point x to manifold.

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Point exp_x(v), shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    v_norm = jnp.linalg.norm(v)
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * v_norm, MIN_NORM)
    lambda_x = _conformal_factor(x, c)
    second_term = jnp.tanh(c_norm_prod * lambda_x / 2) / c_norm_prod * v
    second_term = proj(second_term, c)
    res = addition(x, second_term, c)
    return res


def expmap_0(v: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    Args:
        v: Tangent vector at origin, shape (dim,)
        c: Curvature (positive)

    Returns:
        Point exp_0(v), shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    v_norm = jnp.linalg.norm(v)
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * v_norm, MIN_NORM)
    res = jnp.tanh(c_norm_prod) / c_norm_prod * v
    res = proj(res, c)
    return res


def retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Retraction: first-order approximation of exponential map.

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Point retr_x(v) ≈ exp_x(v), shape (dim,)

    References:
        Bécigneul & Ganea. "Riemannian adaptive optimization." ICLR 2019.
    """
    res = x + v
    res = proj(res, c)
    return res


def logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Logarithmic map: map point y to tangent space at point x.

    Args:
        y: Poincaré ball point, shape (dim,)
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Tangent vector log_x(y), shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sub = addition(-x, y, c)
    x2y2 = jnp.dot(x, x) * jnp.dot(y, y)
    xy = jnp.dot(x, y)
    num = jnp.linalg.norm(y - x)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c**2 * x2y2, MIN_NORM))
    sub_norm = num / denom
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * sub_norm, MIN_NORM)
    lambda_x = _conformal_factor(x, c)
    res = 2 * atanh(c_norm_prod) / (c_norm_prod * lambda_x) * sub
    return res


def logmap_0(y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    Args:
        y: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Tangent vector log_0(y), shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    y_norm = jnp.linalg.norm(y)
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * y_norm, MIN_NORM)
    res = atanh(c_norm_prod) / c_norm_prod * y
    return res


def ptransp(v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Parallel transport tangent vector v from point x to point y.

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Poincaré ball point, shape (dim,)
        y: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Parallel transported tangent vector, shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_x = _conformal_factor(x, c)
    lambda_y = _conformal_factor(y, c)
    return _gyration(y, -x, v, c) * (lambda_x / lambda_y)


def ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Parallel transport tangent vector v from origin to point y.

    Args:
        v: Tangent vector at origin, shape (dim,)
        y: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Parallel transported tangent vector, shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_y = _conformal_factor(y, c)
    conformal_frac = 2 / lambda_y
    return conformal_frac * v


def tangent_inner(u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Compute inner product of tangent vectors u and v at point x.

    Args:
        u: Tangent vector at x, shape (dim,)
        v: Tangent vector at x, shape (dim,)
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Riemannian inner product <u, v>_x, scalar

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_x = _conformal_factor(x, c)
    return lambda_x**2 * jnp.dot(u, v)


def tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Compute norm of tangent vector v at point x.

    Args:
        v: Tangent vector at x, shape (dim,)
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Riemannian norm ||v||_x, scalar

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_x = _conformal_factor(x, c)
    return lambda_x * jnp.linalg.norm(v)


def egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Convert Euclidean gradient to Riemannian gradient.

    Args:
        grad: Euclidean gradient, shape (dim,)
        x: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Riemannian gradient, shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    lambda_x = _conformal_factor(x, c)
    return grad / (lambda_x**2)


def tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
    """Project vector v onto tangent space at point x.

    In Poincaré ball, tangent space equals ambient space (identity).

    Args:
        v: Vector to project, shape (dim,)
        x: Poincaré ball point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency)

    Returns:
        Projected vector v (unchanged), shape (dim,)
    """
    return v


def is_in_manifold(x: Float[Array, "dim"], c: float, atol: float = 1e-5) -> bool:
    """Check if point x lies in Poincaré ball.

    Args:
        x: Point to check, shape (dim,)
        c: Curvature (positive)
        atol: Absolute tolerance (kept for API consistency but not used)

    Returns:
        True if ||x||² < 1/c

    Notes:
        Matches PyTorch implementation which uses strict inequality with no tolerance.
        The projection function already ensures points are strictly inside the ball.
    """
    x_sqnorm = jnp.dot(x, x)
    return x_sqnorm < 1.0 / c


def is_in_tangent_space(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> bool:
    """Check if vector v lies in tangent space at point x.

    In Poincaré ball, all vectors are valid tangent vectors.

    Args:
        v: Vector to check, shape (dim,)
        x: Poincaré ball point (ignored), shape (dim,)
        c: Curvature (ignored, kept for consistency)

    Returns:
        Always True
    """
    return jnp.array(True, dtype=bool)
