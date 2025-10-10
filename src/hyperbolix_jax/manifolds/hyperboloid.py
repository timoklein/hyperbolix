"""Hyperboloid manifold - vmap-native pure functional implementation.

JAX port with vmap-native API. All functions operate on single points/vectors
in ambient (dim+1)-dimensional space. Use jax.vmap for batch operations.

Convention: -x₀² + ||x_rest||² = -1/c with c > 0, x₀ > 0, and sectional curvature -c.

JIT Compilation & Batching
---------------------------
All functions work with single points and return scalars or vectors.
Use jax.vmap for batching and jax.jit for compilation:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix_jax.manifolds import hyperboloid
    >>>
    >>> # Single point operations (points in ambient R^(dim+1))
    >>> x = jnp.array([1.0, 0.1, 0.2])  # Will be projected
    >>> y = jnp.array([1.0, 0.3, 0.4])
    >>> x = hyperboloid.proj(x, c=1.0)
    >>> y = hyperboloid.proj(y, c=1.0)
    >>> distance = hyperboloid.dist(x, y, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)
    >>>
    >>> # Batch operations with vmap
    >>> x_batch = jnp.array([[1.0, 0.1, 0.2], [1.0, 0.15, 0.25]])  # (batch, dim+1)
    >>> y_batch = jnp.array([[1.0, 0.3, 0.4], [1.0, 0.35, 0.45]])
    >>> dist_batched = jax.vmap(hyperboloid.dist, in_axes=(0, 0, None, None))
    >>> distances = dist_batched(x_batch, y_batch, 1.0, hyperboloid.VERSION_DEFAULT)
    >>>
    >>> # JIT compilation
    >>> dist_jit = jax.jit(hyperboloid.dist, static_argnames=['version_idx'])
    >>> distance = dist_jit(x, y, c=1.0, version_idx=hyperboloid.VERSION_DEFAULT)

Version Constants:
    VERSION_DEFAULT (0): Standard acosh distance with hard clipping
    VERSION_SMOOTHENED (1): Smoothened distance with soft clamping

Note: Keep curvature parameter 'c' dynamic to support learnable curvature.
Use version_idx as static argument for JIT (static_argnames=['version_idx']).
"""

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import acosh, smooth_clamp_min

# Default numerical parameters
MIN_NORM = 1e-15

# Version selection constants for dist() and dist_0()
VERSION_DEFAULT = 0
VERSION_SMOOTHENED = 1


def _create_origin(c: float, dim: int, dtype=jnp.float32) -> Float[Array, "dim_plus_1"]:
    """Create hyperboloid origin [1/√c, 0, ..., 0]."""
    sqrt_c = jnp.sqrt(c)
    origin = jnp.zeros(dim + 1, dtype=dtype)
    origin = origin.at[0].set(1.0 / sqrt_c)
    return origin


def _minkowski_inner(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"]) -> Float[Array, ""]:
    """Compute Minkowski inner product ⟨x, y⟩_L = -x₀y₀ + ⟨x_rest, y_rest⟩.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)

    Returns:
        Minkowski inner product, scalar
    """
    x0y0 = x[0] * y[0]
    x_rest_y_rest = jnp.dot(x[1:], y[1:])
    return -x0y0 + x_rest_y_rest


def proj(x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Project point onto hyperboloid by adjusting temporal component.

    Args:
        x: Point to project, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Projected point with -x₀² + ||x_rest||² = -1/c, x₀ > 0, shape (dim+1,)
    """
    x_rest = x[1:]
    x_rest_sqnorm = jnp.dot(x_rest, x_rest)
    x0_new = jnp.sqrt(jnp.maximum(1.0 / c + x_rest_sqnorm, MIN_NORM))
    return jnp.concatenate([x0_new[None], x_rest])


def addition(
    x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float, backproject: bool = True
) -> Float[Array, "dim_plus_1"]:
    """Einstein gyrovector addition on hyperboloid.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        backproject: Whether to project result back to hyperboloid

    Returns:
        Einstein sum x ⊕ y, shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    mink_inner_xy = _minkowski_inner(x, y)

    # Einstein addition formula
    denom = jnp.maximum(1.0 - c * mink_inner_xy, MIN_NORM)
    gamma = 1.0 / denom

    res = x + gamma * (y + (c / (1.0 + sqrt_c)) * mink_inner_xy * x)

    if backproject:
        res = proj(res, c)
    return res


def scalar_mul(r: float, x: Float[Array, "dim_plus_1"], c: float, backproject: bool = True) -> Float[Array, "dim_plus_1"]:
    """Scalar multiplication r ⊗ x on hyperboloid.

    Args:
        r: Scalar factor
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        backproject: Whether to project result back to hyperboloid

    Returns:
        Scaled point r ⊗ x, shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # Map to tangent space, scale geodesic length, map back
    v = logmap_0(x, c)
    v_sqnorm = _minkowski_inner(v, v)
    v_norm = jnp.sqrt(jnp.maximum(v_sqnorm, MIN_NORM))
    unit_tangent = v / v_norm
    dist0 = dist_0(x, c)
    tangent = r * dist0 * unit_tangent
    res = expmap_0(tangent, c, backproject=backproject)
    return res


# Distance implementations for lax.switch
def _dist_default(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
    """Standard acosh distance with hard clipping."""
    sqrt_c = jnp.sqrt(c)
    lorentz_inner = _minkowski_inner(x, y)
    arg = jnp.clip(-c * lorentz_inner, min=1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if points are identical
    same = jnp.all(jnp.equal(x, y))
    return jnp.where(same, 0.0, res)


def _dist_smoothened(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
    """Smoothened distance with soft clamping."""
    sqrt_c = jnp.sqrt(c)
    lorentz_inner = _minkowski_inner(x, y)
    arg = smooth_clamp_min(-c * lorentz_inner, 1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if points are identical
    same = jnp.all(jnp.equal(x, y))
    return jnp.where(same, 0.0, res)


def dist(
    x: Float[Array, "dim_plus_1"],
    y: Float[Array, "dim_plus_1"],
    c: float,
    version_idx: int = VERSION_DEFAULT,
) -> Float[Array, ""]:
    """Compute geodesic distance between hyperboloid points.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        version_idx: Distance version index (use VERSION_* constants)

    Returns:
        Geodesic distance d(x, y), scalar

    References:
        Nickel & Kiela. "Poincaré embeddings for learning hierarchical representations." NeurIPS 2017.
    """
    return lax.switch(version_idx, [_dist_default, _dist_smoothened], x, y, c)


# Distance from origin implementations for lax.switch
def _dist_0_default(x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
    """Standard acosh distance from origin with hard clipping."""
    sqrt_c = jnp.sqrt(c)
    x0 = x[0]
    arg = jnp.clip(sqrt_c * x0, min=1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if at origin
    origin = _create_origin(c, x.shape[0] - 1, x.dtype)
    at_origin = jnp.all(jnp.equal(x, origin))
    return jnp.where(at_origin, 0.0, res)


def _dist_0_smoothened(x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
    """Smoothened distance from origin with soft clamping."""
    sqrt_c = jnp.sqrt(c)
    x0 = x[0]
    arg = smooth_clamp_min(sqrt_c * x0, 1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if at origin
    origin = _create_origin(c, x.shape[0] - 1, x.dtype)
    at_origin = jnp.all(jnp.equal(x, origin))
    return jnp.where(at_origin, 0.0, res)


def dist_0(x: Float[Array, "dim_plus_1"], c: float, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
    """Compute geodesic distance from hyperboloid origin.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        version_idx: Distance version index (use VERSION_* constants)

    Returns:
        Geodesic distance d(origin, x), scalar

    References:
        Nickel & Kiela. "Poincaré embeddings for learning hierarchical representations." NeurIPS 2017.
    """
    return lax.switch(version_idx, [_dist_0_default, _dist_0_smoothened], x, c)


def expmap(
    v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float, backproject: bool = True
) -> Float[Array, "dim_plus_1"]:
    """Exponential map: map tangent vector v at point x to manifold.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        backproject: Whether to project result back to hyperboloid

    Returns:
        Point exp_x(v), shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    v_sqnorm = jnp.clip(_minkowski_inner(v, v), min=0.0)
    v_norm = jnp.sqrt(v_sqnorm)
    c_norm_prod = sqrt_c * v_norm

    denom = jnp.maximum(c_norm_prod, MIN_NORM)
    cosh_term = jnp.cosh(c_norm_prod) * x
    sinh_term = jnp.sinh(c_norm_prod) / denom * v

    res = cosh_term + sinh_term

    if backproject:
        res = proj(res, c)
    return res


def expmap_0(v: Float[Array, "dim_plus_1"], c: float, backproject: bool = True) -> Float[Array, "dim_plus_1"]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    Args:
        v: Tangent vector at origin in ambient representation, shape (dim+1,)
            (first component should be 0)
        c: Curvature (positive)
        backproject: Whether to project result back to hyperboloid

    Returns:
        Point exp_0(v) in ambient representation, shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    v_sqnorm = jnp.clip(_minkowski_inner(v, v), min=0.0)
    v_norm = jnp.sqrt(v_sqnorm)
    c_norm_prod = sqrt_c * v_norm

    denom = jnp.maximum(c_norm_prod, MIN_NORM)
    sinh_scale = jnp.sinh(c_norm_prod) / denom

    v0 = v[0]
    v_rest = v[1:]

    res0 = jnp.cosh(c_norm_prod) / sqrt_c + sinh_scale * v0
    res_rest = sinh_scale * v_rest

    res = jnp.concatenate([res0[None], res_rest])

    if backproject:
        res = proj(res, c)
    return res


def retraction(
    v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float, backproject: bool = True
) -> Float[Array, "dim_plus_1"]:
    """Retraction: first-order approximation of exponential map.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        backproject: Whether to project result back to hyperboloid

    Returns:
        Point retr_x(v) ≈ exp_x(v), shape (dim+1,)

    References:
        Bécigneul & Ganea. "Riemannian adaptive optimization." ICLR 2019.
    """
    res = x + v
    if backproject:
        res = proj(res, c)
    return res


def logmap(
    y: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float, backproject: bool = True
) -> Float[Array, "dim_plus_1"]:
    """Logarithmic map: map point y to tangent space at point x.

    Args:
        y: Hyperboloid point, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        backproject: Whether to backproject (project to tangent space)

    Returns:
        Tangent vector log_x(y), shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    mink_inner = _minkowski_inner(x, y)
    dist_xy = dist(x, y, c=c)
    direction = y + c * mink_inner * x

    dir_sqnorm = _minkowski_inner(direction, direction)
    dir_norm = jnp.sqrt(jnp.maximum(dir_sqnorm, MIN_NORM))
    res = dist_xy * direction / dir_norm

    if backproject:
        res = tangent_proj(res, x, c)

    return res


def logmap_0(y: Float[Array, "dim_plus_1"], c: float, backproject: bool = True) -> Float[Array, "dim_plus_1"]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    Args:
        y: Hyperboloid point in ambient representation, shape (dim+1,)
        c: Curvature (positive)
        backproject: Whether to backproject (project to tangent space)

    Returns:
        Tangent vector log_0(y) in ambient representation, shape (dim+1,)
        (first component is 0)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    y_rest = y[1:]
    y_rest_norm = jnp.linalg.norm(y_rest)

    dist0 = dist_0(y, c=c)
    scale = dist0 / jnp.maximum(y_rest_norm, MIN_NORM)

    v0 = jnp.array([0.0])
    v_rest = scale * y_rest
    res = jnp.concatenate([v0, v_rest])

    if backproject:
        origin = _create_origin(c, y.shape[0] - 1, y.dtype)
        res = tangent_proj(res, origin, c)

    return res


def ptransp(
    v: Float[Array, "dim_plus_1"],
    x: Float[Array, "dim_plus_1"],
    y: Float[Array, "dim_plus_1"],
    c: float,
    backproject: bool = True,
) -> Float[Array, "dim_plus_1"]:
    """Parallel transport tangent vector v from point x to point y.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        backproject: Whether to backproject to tangent space at y

    Returns:
        Parallel transported tangent vector, shape (dim+1,)

    References:
        Aaron Lou, et al. "Differentiating through the fréchet mean."
            International conference on machine learning (2020).
    """
    # Compute Minkowski inner products
    vy = _minkowski_inner(v, y)  # ⟨v, y⟩_L
    xy = _minkowski_inner(x, y)  # ⟨x, y⟩_L

    # denom = 1/c - ⟨x, y⟩_L
    denom = 1.0 / c - xy
    denom = jnp.maximum(denom, MIN_NORM)  # Numerical stability

    # scale = ⟨v, y⟩_L / denom
    scale = vy / denom

    # res = v + scale * (x + y)
    res = v + scale * (x + y)

    if backproject:
        res = tangent_proj(res, y, c)

    return res


def ptransp_0(
    v: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float, backproject: bool = True
) -> Float[Array, "dim_plus_1"]:
    """Parallel transport tangent vector v from origin to point y.

    Args:
        v: Tangent vector at origin, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        backproject: Whether to backproject to tangent space at y

    Returns:
        Parallel transported tangent vector, shape (dim+1,)

    References:
        Aaron Lou, et al. "Differentiating through the fréchet mean."
            International conference on machine learning (2020).
    """
    # Create origin point [1/√c, 0, ..., 0]
    sqrt_c = jnp.sqrt(c)
    y0 = y[0]

    # Build origin vector
    origin = _create_origin(c, y.shape[0] - 1, y.dtype)

    # Compute Minkowski inner products
    vy = _minkowski_inner(v, y)  # ⟨v, y⟩_L

    # denom = 1/c + y0/√c (from ⟨origin, y⟩_L = -y0/√c and denom = 1/c - ⟨origin, y⟩_L)
    denom = 1.0 / c + y0 / sqrt_c
    denom = jnp.maximum(denom, MIN_NORM)  # Numerical stability

    # scale = ⟨v, y⟩_L / denom
    scale = vy / denom

    # res = v + scale * (y + origin)
    res = v + scale * (y + origin)

    if backproject:
        res = tangent_proj(res, y, c)

    return res


def tangent_inner(
    u: Float[Array, "dim_plus_1"], v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float
) -> Float[Array, ""]:
    """Compute inner product of tangent vectors u and v at point x.

    Uses the Minkowski inner product restricted to tangent space.

    Args:
        u: Tangent vector at x, shape (dim+1,)
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Riemannian inner product ⟨u, v⟩_x, scalar
    """
    return _minkowski_inner(u, v)


def tangent_norm(v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
    """Compute norm of tangent vector v at point x.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Riemannian norm ||v||_x, scalar
    """
    inner = tangent_inner(v, v, x, c)
    return jnp.sqrt(jnp.clip(inner, min=0.0))


def egrad2rgrad(grad: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Convert Euclidean gradient to Riemannian gradient.

    Projects Euclidean gradient onto tangent space.

    Args:
        grad: Euclidean gradient, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Riemannian gradient, shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # In Lorentzian signature the temporal component carries a negative sign.
    # Flip it before projecting so we project the Riemannian gradient, matching PyTorch.
    grad_lorentz = grad.at[0].set(-grad[0])

    # Orthogonally project the Lorentzian gradient onto the tangent space.
    inner_xx = _minkowski_inner(x, x)
    scale = jnp.sqrt(jnp.maximum(-c * inner_xx, MIN_NORM))
    x_normed = x / scale

    denom = _minkowski_inner(x_normed, x_normed)
    coeff = _minkowski_inner(x_normed, grad_lorentz) / denom
    return grad_lorentz - coeff * x_normed


def tangent_proj(v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Project vector v onto tangent space at point x.

    Args:
        v: Vector to project, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Projected vector onto tangent space, shape (dim+1,)
    """
    # Normalize x w.r.t. measured Lorentz norm (robust in float32)
    inner_xx = _minkowski_inner(x, x)
    scale = jnp.sqrt(jnp.maximum(-c * inner_xx, MIN_NORM))
    x_normed = x / scale

    denom = _minkowski_inner(x_normed, x_normed)
    coeff = _minkowski_inner(x_normed, v) / denom
    return v - coeff * x_normed


def is_in_manifold(x: Float[Array, "dim_plus_1"], c: float, atol: float = 1e-5) -> bool:
    """Check if point x lies on hyperboloid.

    Args:
        x: Point to check, shape (dim+1,)
        c: Curvature (positive)
        atol: Absolute tolerance

    Returns:
        True if -x₀² + ||x_rest||² = -1/c and x₀ > 0
    """
    lorentz_norm = _minkowski_inner(x, x)
    tol = max(atol, 1e-4)
    target = -1.0 / c

    valid_constraint = jnp.isclose(lorentz_norm, target, atol=tol, rtol=0.0)
    valid_x0 = x[0] > 0

    return jnp.logical_and(valid_constraint, valid_x0)


def is_in_tangent_space(
    v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float, atol: float | None = None
) -> bool:
    """Check if vector v lies in tangent space at point x.

    Tangent space is orthogonal to x in Minkowski metric: ⟨v, x⟩_L = 0

    Args:
        v: Vector to check, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        atol: Absolute tolerance (dtype-aware if None)

    Returns:
        True if ⟨v, x⟩_L ≈ 0
    """
    tol = 5e-4 if atol is None else atol
    mink_inner = _minkowski_inner(v, x)
    return jnp.abs(mink_inner) < tol
