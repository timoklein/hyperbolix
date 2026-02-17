"""Poincaré Ball manifold - class-based API with dtype control.

Provides a Poincare class for manifold operations with automatic dtype casting.
All operations work on single points with shape (dim,). Use jax.vmap for batching.

Convention: ||x||^2 < 1/c with c > 0 and sectional curvature -c.

JIT Compilation & Batching
---------------------------
Create a Poincare instance with desired dtype, then use its methods:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.manifolds.poincare import Poincare, VERSION_MOBIUS_DIRECT
    >>>
    >>> # Create manifold with float32 (default) or float64
    >>> manifold = Poincare(dtype=jnp.float32)
    >>>
    >>> # Single point operations
    >>> x = jnp.array([0.1, 0.2])
    >>> y = jnp.array([0.3, 0.4])
    >>> distance = manifold.dist(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)
    >>>
    >>> # Batch operations with vmap
    >>> x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])  # (batch, dim)
    >>> y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45]])
    >>> dist_batched = jax.vmap(manifold.dist, in_axes=(0, 0, None, None))
    >>> distances = dist_batched(x_batch, y_batch, 1.0, VERSION_MOBIUS_DIRECT)
    >>>
    >>> # JIT compilation
    >>> dist_jit = jax.jit(manifold.dist, static_argnames=['version_idx'])
    >>> distance = dist_jit(x, y, c=1.0, version_idx=VERSION_MOBIUS_DIRECT)

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
- Use Poincare(dtype=jnp.float64)
- Expect ~3% relative error with float32 for distances > 10
- Consider projection after operations to maintain manifold constraints
"""

import math

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import acosh, asinh, atanh, cosh, sinh, smooth_clamp

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
    return float(jnp.finfo(x.dtype).eps ** 0.75)


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


def _proj(x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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


def _addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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
    res = _proj(res, c)
    return res


def _scalar_mul(r: float, x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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
    res = _proj(res, c)
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
    diff = _addition(-x, y, c)
    dist_c = atanh(sqrt_c * jnp.linalg.norm(diff))
    return 2 * dist_c / sqrt_c


def _dist_metric_tensor(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Metric tensor induced distance."""
    x_sqnorm = jnp.dot(x, x)
    y_sqnorm = jnp.dot(y, y)
    xy_diff_sqnorm = jnp.dot(x - y, x - y)
    arg = 1 + 2 * c * xy_diff_sqnorm / ((1 - c * x_sqnorm) * (1 - c * y_sqnorm))
    condition = arg < 1 + MIN_NORM
    return jnp.where(condition, 0.0, acosh(arg) / jnp.sqrt(c))  # type: ignore[return-value]


def _dist_lorentzian_proxy(x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Lorentzian proxy distance."""
    xy_prod = x * y
    xy0 = xy_prod[0]
    xy_rem = jnp.sum(xy_prod[1:])
    xy_mink = xy_rem - xy0
    return -2 / c - 2 * xy_mink


def _dist(
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
    return jnp.where(condition, 0.0, acosh(arg) / jnp.sqrt(c))  # type: ignore[return-value]


def _dist_0_lorentzian_proxy(x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
    """Lorentzian proxy distance from origin."""
    x0 = x[0]
    return -2 / c + 2 * x0 / jnp.sqrt(c)


def _dist_0(x: Float[Array, "dim"], c: float, version_idx: int = VERSION_MOBIUS_DIRECT) -> Float[Array, ""]:
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


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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
    second_term = _proj(second_term, c)
    res = _addition(x, second_term, c)
    return res


def _expmap_0(v: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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
    res = _proj(res, c)
    return res


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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
    res = _proj(res, c)
    return res


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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
    sub = _addition(-x, y, c)
    x2y2 = jnp.dot(x, x) * jnp.dot(y, y)
    xy = jnp.dot(x, y)
    num = jnp.linalg.norm(y - x)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c**2 * x2y2, MIN_NORM))
    sub_norm = num / denom
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * sub_norm, MIN_NORM)
    lambda_x = _conformal_factor(x, c)
    res = 2 * atanh(c_norm_prod) / (c_norm_prod * lambda_x) * sub
    return res


def _logmap_0(y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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


def _ptransp(v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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


def _tangent_inner(u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
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


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
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


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
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


def _is_in_manifold(x: Float[Array, "dim"], c: float, atol: float = 1e-5) -> Array:
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


def _is_in_tangent_space(v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Array:
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


# ---------------------------------------------------------------------------
# Batch-compatible helpers (used by NN layers)
# ---------------------------------------------------------------------------


def _conformal_factor_batch(
    x: Float[Array, "... dim"],
    c: float,
) -> Float[Array, "... 1"]:
    """Numerically stable conformal factor lambda(x) = 2 / (1 - c||x||^2).

    Batch-compatible version that handles arbitrary leading dimensions.

    Args:
        x: Poincare ball point(s), shape (..., dim)
        c: Manifold curvature (positive)

    Returns:
        Conformal factor, shape (..., 1)
    """
    dtype = x.dtype
    c_arr = jnp.asarray(c, dtype=dtype)
    sqrt_c = jnp.sqrt(c_arr)
    # Use a single representative element for dtype-dependent eps
    max_norm_eps = jnp.asarray(_get_max_norm_eps(x.reshape(-1)[:1].squeeze()), dtype=dtype)
    x_norm_sq = jnp.sum(x**2, axis=-1, keepdims=True)
    denom_min = 2 * sqrt_c * max_norm_eps - c_arr * max_norm_eps**2
    denom = jnp.maximum(jnp.asarray(1.0, dtype=dtype) - c_arr * x_norm_sq, denom_min)
    return 2.0 / denom


def _compute_mlr_pp(
    x: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim"],
    r: Float[Array, "out_dim 1"],
    c: float,
    clamping_factor: float,
    smoothing_factor: float,
    min_enorm: float = 1e-15,
) -> Float[Array, "batch out_dim"]:
    """Compute HNN++ multinomial linear regression on the Poincare ball.

    Args:
        x: Poincare ball point(s), shape (batch, in_dim)
        z: Hyperplane tangent normals at origin, shape (out_dim, in_dim)
        r: Hyperplane translations, shape (out_dim, 1)
        c: Manifold curvature (positive)
        clamping_factor: Clamping value for the output
        smoothing_factor: Smoothing factor for the output
        min_enorm: Minimum norm to avoid division by zero

    Returns:
        MLR scores, shape (batch, out_dim)

    References:
        Shimizu et al. "Hyperbolic neural networks++." arXiv:2006.08210 (2020).
    """
    sqrt_c = jnp.sqrt(c)
    sqrt_c2r = 2 * sqrt_c * r.T  # (1, out_dim)
    z_norm = jnp.linalg.norm(z, ord=2, axis=-1, keepdims=True).clip(min=min_enorm)  # (out_dim, 1)

    lambda_x = _conformal_factor_batch(x, c)  # (batch, 1)

    z_unitx = jnp.einsum("bi,oi->bo", x, z / z_norm)  # (batch, out_dim)
    asinh_arg = (1 - lambda_x) * sinh(sqrt_c2r) + sqrt_c * lambda_x * cosh(sqrt_c2r) * z_unitx  # (batch, out_dim)

    eps = jnp.finfo(jnp.float32).eps if x.dtype == jnp.float32 else jnp.finfo(jnp.float64).eps
    clamp = clamping_factor * float(math.log(2 / eps))
    asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, smoothing_factor)  # (batch, out_dim)
    signed_dist2hyp = asinh(asinh_arg) / sqrt_c  # (batch, out_dim)
    res = 2 * z_norm.T * signed_dist2hyp  # (batch, out_dim)
    return res


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class Poincare:
    """Poincaré ball manifold with automatic dtype casting.

    Provides all manifold operations with automatic casting of array inputs
    to the specified dtype. This eliminates the need for manual casting and
    provides better numerical stability control.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds.poincare import Poincare, VERSION_MOBIUS_DIRECT
        >>>
        >>> # Create manifold with float64 for better precision
        >>> manifold = Poincare(dtype=jnp.float64)
        >>>
        >>> # Arrays are automatically cast to float64
        >>> x = jnp.array([0.1, 0.2], dtype=jnp.float32)
        >>> y = jnp.array([0.3, 0.4], dtype=jnp.float32)
        >>> d = manifold.dist(x, y, c=1.0)
        >>> d.dtype  # float64
    """

    VERSION_MOBIUS_DIRECT = VERSION_MOBIUS_DIRECT
    VERSION_MOBIUS = VERSION_MOBIUS
    VERSION_METRIC_TENSOR = VERSION_METRIC_TENSOR
    VERSION_LORENTZIAN_PROXY = VERSION_LORENTZIAN_PROXY

    def __init__(self, dtype: jnp.dtype = jnp.float32) -> None:
        self.dtype = dtype

    def _cast(self, x: Array) -> Array:
        """Cast array to target dtype if it's a floating-point array."""
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact):
            return x.astype(self.dtype)
        return x

    def proj(self, x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Project point onto Poincaré ball by clipping norm."""
        return _proj(self._cast(x), c)

    def gyration(
        self, x: Float[Array, "dim"], y: Float[Array, "dim"], z: Float[Array, "dim"], c: float
    ) -> Float[Array, "dim"]:
        """Compute gyration gyr[x,y]z to restore commutativity."""
        return _gyration(self._cast(x), self._cast(y), self._cast(z), c)

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Möbius gyrovector addition x ⊕ y."""
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float, x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Scalar multiplication r ⊗ x on Poincaré ball."""
        x = self._cast(x)
        r = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r, x, c)

    def dist(
        self,
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: float,
        version_idx: int = VERSION_MOBIUS_DIRECT,
    ) -> Float[Array, ""]:
        """Compute geodesic distance between Poincaré ball points."""
        return _dist(self._cast(x), self._cast(y), c, version_idx)

    def _dist(
        self,
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: float,
        version_idx: int = VERSION_MOBIUS_DIRECT,
    ) -> Float[Array, ""]:
        """Compatibility alias for legacy module-style API."""
        return self.dist(x, y, c, version_idx)

    def dist_0(self, x: Float[Array, "dim"], c: float, version_idx: int = VERSION_MOBIUS_DIRECT) -> Float[Array, ""]:
        """Compute geodesic distance from Poincaré ball origin."""
        return _dist_0(self._cast(x), c, version_idx)

    def _dist_0(self, x: Float[Array, "dim"], c: float, version_idx: int = VERSION_MOBIUS_DIRECT) -> Float[Array, ""]:
        """Compatibility alias for legacy module-style API."""
        return self.dist_0(x, c, version_idx)

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Exponential map: map tangent vector v at point x to manifold."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Exponential map from origin: map tangent vector v at origin to manifold."""
        return _expmap_0(self._cast(v), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Retraction: first-order approximation of exponential map."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Logarithmic map: map point y to tangent space at point x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Logarithmic map from origin: map point y to tangent space at origin."""
        return _logmap_0(self._cast(y), c)

    def ptransp(self, v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Parallel transport tangent vector v from point x to point y."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Parallel transport tangent vector v from origin to point y."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    def tangent_inner(
        self, u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: float
    ) -> Float[Array, ""]:
        """Compute inner product of tangent vectors u and v at point x."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
        """Compute norm of tangent vector v at point x."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Convert Euclidean gradient to Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Project vector v onto tangent space at point x."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim"], c: float, atol: float = 1e-5) -> Array:
        """Check if point x lies in Poincaré ball."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Array:
        """Check if vector v lies in tangent space at point x."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c)

    def conformal_factor(self, x: Float[Array, "... dim"], c: float) -> Float[Array, "... 1"]:
        """Numerically stable conformal factor lambda(x) = 2 / (1 - c||x||^2).

        Batch-compatible version that handles arbitrary leading dimensions.
        """
        return _conformal_factor_batch(self._cast(x), c)

    def compute_mlr_pp(
        self,
        x: Float[Array, "batch in_dim"],
        z: Float[Array, "out_dim in_dim"],
        r: Float[Array, "out_dim 1"],
        c: float,
        clamping_factor: float,
        smoothing_factor: float,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """Compute HNN++ multinomial linear regression on the Poincare ball."""
        return _compute_mlr_pp(self._cast(x), self._cast(z), self._cast(r), c, clamping_factor, smoothing_factor, min_enorm)
