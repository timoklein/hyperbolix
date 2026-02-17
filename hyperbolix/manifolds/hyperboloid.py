"""Hyperboloid manifold - class-based API with dtype control.

JAX port with vmap-native API. All functions operate on single points/vectors
in ambient (dim+1)-dimensional space. Use jax.vmap for batch operations.

Convention: -x₀² + ||x_rest||² = -1/c with c > 0, x₀ > 0, and sectional curvature -c.

JIT Compilation & Batching
---------------------------
All functions work with single points and return scalars or vectors.
Use jax.vmap for batching and jax.jit for compilation:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.manifolds.hyperboloid import Hyperboloid, VERSION_DEFAULT
    >>>
    >>> # Single point operations (points in ambient R^(dim+1))
    >>> x = jnp.array([1.0, 0.1, 0.2])  # Will be projected
    >>> y = jnp.array([1.0, 0.3, 0.4])
    >>> manifold = Hyperboloid(dtype=jnp.float32)
    >>> x = manifold.proj(x, c=1.0)
    >>> y = manifold.proj(y, c=1.0)
    >>> distance = manifold.dist(x, y, c=1.0, version_idx=VERSION_DEFAULT)
    >>>
    >>> # Batch operations with vmap
    >>> x_batch = jnp.array([[1.0, 0.1, 0.2], [1.0, 0.15, 0.25]])  # (batch, dim+1)
    >>> y_batch = jnp.array([[1.0, 0.3, 0.4], [1.0, 0.35, 0.45]])
    >>> dist_batched = jax.vmap(manifold.dist, in_axes=(0, 0, None, None))
    >>> distances = dist_batched(x_batch, y_batch, 1.0, VERSION_DEFAULT)
    >>>
    >>> # JIT compilation
    >>> dist_jit = jax.jit(manifold.dist, static_argnames=['version_idx'])
    >>> distance = dist_jit(x, y, c=1.0, version_idx=VERSION_DEFAULT)

Version Constants:
    VERSION_DEFAULT (0): Standard acosh distance with hard clipping
    VERSION_SMOOTHENED (1): Smoothened distance with soft clamping

Note: Keep curvature parameter 'c' dynamic to support learnable curvature.
Use version_idx as static argument for JIT (static_argnames=['version_idx']).
"""

import math
from typing import cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import acosh, asinh, cosh, sinh, smooth_clamp, smooth_clamp_min

# Default numerical parameters
MIN_NORM = 1e-15

# Version selection constants for _dist() and _dist_0()
VERSION_DEFAULT = 0
VERSION_SMOOTHENED = 1


def _create_origin(c: Float[Array, ""] | float, dim: int, dtype=jnp.float32) -> Float[Array, "dim_plus_1"]:
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


def _embed_spatial_0(v_spatial: Float[Array, "... n"]) -> Float[Array, "... n_plus_1"]:
    """Embed spatial vector as tangent vector at origin by prepending zero.

    Creates tangent vector v = [0, v_bar] ∈ T_{μ₀}ℍⁿ from spatial vector v_bar ∈ ℝⁿ.
    This is used to embed Gaussian samples from spatial coordinates into the tangent
    space at the origin before parallel transport.

    Args:
        v_spatial: Spatial vector(s), shape (..., n)

    Returns:
        Tangent vector(s) at origin, shape (..., n+1)

    Examples:
        >>> v_spatial = jnp.array([0.1, 0.2])
        >>> v_tangent = _embed_spatial_0(v_spatial)
        >>> v_tangent
        Array([0. , 0.1, 0.2], dtype=float32)
    """
    zeros = jnp.zeros((*v_spatial.shape[:-1], 1), dtype=v_spatial.dtype)
    return jnp.concatenate([zeros, v_spatial], axis=-1)


def _proj(x: Float[Array, "dim_plus_1"], c: Float[Array, ""] | float) -> Float[Array, "dim_plus_1"]:
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


def _proj_batch(x: Float[Array, "... dim_plus_1"], c: Float[Array, ""] | float) -> Float[Array, "... dim_plus_1"]:
    """Project batched points onto hyperboloid by adjusting temporal component.

    Batch-compatible version of _proj() that handles arbitrary leading dimensions.

    Args:
        x: Points to project, shape (..., dim+1)
        c: Curvature (positive)

    Returns:
        Projected points with -x₀² + ||x_rest||² = -1/c, x₀ > 0, shape (..., dim+1)
    """
    x_rest = x[..., 1:]  # Shape: (..., dim)
    x_rest_sqnorm = jnp.sum(x_rest**2, axis=-1, keepdims=True)  # Shape: (..., 1)
    x0_new = jnp.sqrt(jnp.maximum(1.0 / c + x_rest_sqnorm, MIN_NORM))  # Shape: (..., 1)
    return jnp.concatenate([x0_new, x_rest], axis=-1)


def _addition(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Einstein gyrovector addition on hyperboloid.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

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
    res = _proj(res, c)
    return res


def _scalar_mul(r: float, x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Scalar multiplication r ⊗ x on hyperboloid.

    Args:
        r: Scalar factor
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Scaled point r ⊗ x, shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # Map to tangent space, scale geodesic length, map back
    v = _logmap_0(x, c)
    v_sqnorm = _minkowski_inner(v, v)
    v_norm = jnp.sqrt(jnp.maximum(v_sqnorm, MIN_NORM))
    unit_tangent = v / v_norm
    dist0 = _dist_0(x, c)
    tangent = r * dist0 * unit_tangent
    res = _expmap_0(tangent, c)
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
    return jnp.where(same, 0.0, res)  # type: ignore[return-value]


def _dist_smoothened(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
    """Smoothened distance with soft clamping."""
    sqrt_c = jnp.sqrt(c)
    lorentz_inner = _minkowski_inner(x, y)
    arg = smooth_clamp_min(-c * lorentz_inner, 1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if points are identical
    same = jnp.all(jnp.equal(x, y))
    return jnp.where(same, 0.0, res)  # type: ignore[return-value]


def _dist(
    x: Float[Array, "dim_plus_1"],
    y: Float[Array, "dim_plus_1"],
    c: Float[Array, ""] | float,
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
    return jnp.where(at_origin, 0.0, res)  # type: ignore[return-value]


def _dist_0_smoothened(x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
    """Smoothened distance from origin with soft clamping."""
    sqrt_c = jnp.sqrt(c)
    x0 = x[0]
    arg = smooth_clamp_min(sqrt_c * x0, 1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if at origin
    origin = _create_origin(c, x.shape[0] - 1, x.dtype)
    at_origin = jnp.all(jnp.equal(x, origin))
    return jnp.where(at_origin, 0.0, res)  # type: ignore[return-value]


def _dist_0(x: Float[Array, "dim_plus_1"], c: float, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
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


def _expmap(
    v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Float[Array, ""] | float
) -> Float[Array, "dim_plus_1"]:
    """Exponential map: map tangent vector v at point x to manifold.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

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
    res = _proj(res, c)
    return res


def _expmap_0(v: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    Args:
        v: Tangent vector at origin in ambient representation, shape (dim+1,)
            (first component should be 0)
        c: Curvature (positive)

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
    res = _proj(res, c)
    return res


def _retraction(v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Retraction: first-order approximation of exponential map.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Point retr_x(v) ≈ exp_x(v), shape (dim+1,)

    References:
        Bécigneul & Ganea. "Riemannian adaptive optimization." ICLR 2019.
    """
    res = x + v
    res = _proj(res, c)
    return res


def _logmap(
    y: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Float[Array, ""] | float
) -> Float[Array, "dim_plus_1"]:
    """Logarithmic map: map point y to tangent space at point x.

    Args:
        y: Hyperboloid point, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Tangent vector log_x(y), shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    mink_inner = _minkowski_inner(x, y)
    dist_xy = _dist(x, y, c=c)
    direction = y + c * mink_inner * x

    dir_sqnorm = _minkowski_inner(direction, direction)
    dir_norm = jnp.sqrt(jnp.maximum(dir_sqnorm, MIN_NORM))
    res = dist_xy * direction / dir_norm
    res = _tangent_proj(res, x, c)
    return res


def _logmap_0(y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    Args:
        y: Hyperboloid point in ambient representation, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Tangent vector log_0(y) in ambient representation, shape (dim+1,)
        (first component is 0)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    y_rest = y[1:]
    y_rest_norm = jnp.linalg.norm(y_rest)

    dist0 = _dist_0(y, c=c)
    scale = dist0 / jnp.maximum(y_rest_norm, MIN_NORM)

    v0 = jnp.zeros(1, dtype=y.dtype)
    v_rest = scale * y_rest
    res = jnp.concatenate([v0, v_rest])
    origin = _create_origin(c, y.shape[0] - 1, y.dtype)
    res = _tangent_proj(res, origin, c)
    return res


def _ptransp(
    v: Float[Array, "dim_plus_1"],
    x: Float[Array, "dim_plus_1"],
    y: Float[Array, "dim_plus_1"],
    c: float,
) -> Float[Array, "dim_plus_1"]:
    """Parallel transport tangent vector v from point x to point y.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

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
    res = _tangent_proj(res, y, c)
    return res


def _ptransp_0(v: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
    """Parallel transport tangent vector v from origin to point y.

    Args:
        v: Tangent vector at origin, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

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
    res = _tangent_proj(res, y, c)
    return res


def _tangent_inner(
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


def _tangent_norm(v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
    """Compute norm of tangent vector v at point x.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Riemannian norm ||v||_x, scalar
    """
    inner = _tangent_inner(v, v, x, c)
    return jnp.sqrt(jnp.clip(inner, min=0.0))


def _egrad2rgrad(grad: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
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


def _tangent_proj(
    v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Float[Array, ""] | float
) -> Float[Array, "dim_plus_1"]:
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


def _is_in_manifold(x: Float[Array, "dim_plus_1"], c: float, atol: float = 1e-5) -> Array:
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


def _is_in_tangent_space(
    v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float, atol: float | None = None
) -> Array:
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


def _lorentz_boost(
    x: Float[Array, "dim_plus_1"],
    v_raw: Float[Array, "dim"],
    c: float,
) -> Float[Array, "dim_plus_1"]:
    """Apply Lorentz boost to a single hyperboloid point.

    The Lorentz boost matrix is:
        B = [[gamma,       -gamma*v^T              ],
             [-gamma*v,     I + (gamma^2/(1+gamma))*v*v^T ]]

    where gamma = 1/sqrt(1 - ||v||^2), ||v|| < 1.

    Args:
        x: Single hyperboloid point in ambient space, shape (dim+1,)
            Satisfies: -x₀² + ‖x_rest‖² = -1/c
        v_raw: Velocity vector (unclipped), shape (dim,)
        c: Manifold curvature (positive)

    Returns:
        Boosted point on hyperboloid, shape (dim+1,)

    Notes:
        The velocity is projected to ensure ‖v‖ < 1 - ε for numerical stability.
        The output is projected onto the manifold to correct numerical drift.

        For batch operations, use jax.vmap:
            >>> lorentz_boost_batched = jax.vmap(lorentz_boost, in_axes=(0, None, None))
            >>> boosted = lorentz_boost_batched(x_batch, v, c)

    References:
        Ahmad Bdeir, et al. "Fully hyperbolic convolutional neural networks for computer vision."
            arXiv preprint arXiv:2303.15919 (2023).
    """
    # Numerical stability constant for velocity projection
    epsilon = 1e-5

    # Project velocity to ensure norm < 1 - epsilon
    # Formula: v = v_raw * min(1, (1 - epsilon) / |v_raw|)
    v_norm = jnp.linalg.norm(v_raw)
    scale = jnp.minimum(1.0, (1.0 - epsilon) / jnp.maximum(v_norm, MIN_NORM))
    v = v_raw * scale

    # Compute gamma = 1/√(1 - ‖v‖²)
    v_sqnorm = jnp.dot(v, v)
    gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - v_sqnorm, MIN_NORM))

    # Split input into time and spatial components
    x_t = x[0]  # scalar
    x_s = x[1:]  # Shape: (dim,)

    # Compute v·x_s
    v_dot_xs = jnp.dot(v, x_s)  # scalar

    # New time component: gamma*x_t - gamma*(v.x_s)
    new_t = gamma * x_t - gamma * v_dot_xs  # scalar

    # New spatial component: -gamma*v*x_t + x_s + (gamma^2/(1+gamma))*v*(v.x_s)
    new_s = -gamma * v * x_t + x_s + (gamma**2 / (1.0 + gamma)) * v * v_dot_xs  # Shape: (dim,)

    # Concatenate time and spatial components
    result = jnp.concatenate([new_t[None], new_s])  # Shape: (dim+1,)

    # Project onto manifold to correct numerical drift
    return _proj(result, c)


def _distance_rescale(
    x: Float[Array, "dim_plus_1"],
    c: float,
    x_t_max: float = 2000.0,
    slope: float = 1.0,
) -> Float[Array, "dim_plus_1"]:
    """Apply distance rescaling to bound hyperbolic distances (Eq. 2-3).

    Rescales distances from origin using:
        D_rescaled = m · tanh(D · atanh(0.99) / (s·m))

    where m = D_max (computed from x_t_max) and s = slope.

    Then reconstructs spatial components:
        x_s_rescaled = x_s · sinh(√c · D_rescaled) / sinh(√c · D)

    Args:
        x: Single hyperboloid point, shape (dim+1,)
        c: Manifold curvature (positive)
        x_t_max: Maximum time coordinate (default: 2000.0)
        slope: Rescaling slope parameter (default: 1.0)

    Returns:
        Rescaled point on hyperboloid, shape (dim+1,)

    Notes:
        For points at or very near the origin (D ≈ 0), the point is returned
        unchanged to avoid numerical instability from 0/0 in the scale computation.
        The output is projected onto the manifold to correct numerical drift.

        For batch operations, use jax.vmap:
            >>> distance_rescale_batched = jax.vmap(distance_rescale, in_axes=(0, None, None, None))
            >>> rescaled = distance_rescale_batched(x_batch, c, x_t_max, slope)

    References:
        Equation 2-3 from "Fully Hyperbolic CNNs" (Ahmad Bdeir et al., 2023).
    """
    sqrt_c = jnp.sqrt(c)

    # Extract time and spatial components
    x_t = x[0]  # scalar
    x_s = x[1:]  # Shape: (dim,)

    # Compute distance from origin: D = acosh(√c · x_t) / √c
    arg_acosh = smooth_clamp_min(sqrt_c * x_t, 1.0)
    D = acosh(arg_acosh) / sqrt_c  # scalar

    # Compute max distance: D_max = acosh(√c · x_t_max) / √c
    arg_acosh_max = smooth_clamp_min(sqrt_c * x_t_max, 1.0)
    D_max = acosh(arg_acosh_max) / sqrt_c

    # Apply Eq. 2: D_rescaled = D_max · tanh(D · atanh(0.99) / (slope · D_max))
    # Clamp to avoid division by zero
    D_max_safe = jnp.maximum(D_max, MIN_NORM)
    slope_safe = jnp.maximum(slope, MIN_NORM)

    atanh_val = jnp.atanh(0.99)
    arg_tanh = D * atanh_val / (slope_safe * D_max_safe)
    D_rescaled = D_max * jnp.tanh(arg_tanh)  # scalar

    # Rescale spatial components (Eq. 3): scale = sinh(√c · D_rescaled) / sinh(√c · D)
    sinh_rescaled = sinh(sqrt_c * D_rescaled)  # scalar
    sinh_original = sinh(sqrt_c * D)  # scalar

    # Handle origin edge case: when D ≈ 0, both sinh values are ≈ 0
    # Use L'Hopital's rule: lim(D->0) sinh(a*D_rescaled)/sinh(a*D) = D_rescaled/D
    # For D ≈ 0, D_rescaled ≈ D * atanh(0.99) / slope, so scale ≈ atanh(0.99) / slope
    # We use a smooth transition: when sinh_original is small, fall back to this limit
    is_near_origin = sinh_original < 1e-6
    scale_normal = sinh_rescaled / jnp.maximum(sinh_original, MIN_NORM)
    scale_origin = atanh_val / slope_safe  # Limit value at origin
    scale = cast(Array, jnp.where(is_near_origin, scale_origin, scale_normal))  # scalar

    x_s_rescaled = x_s * scale  # Shape: (dim,)

    # Reconstruct time component: x_t_rescaled = √(‖x_s_rescaled‖² + 1/c)
    x_s_sqnorm = jnp.dot(x_s_rescaled, x_s_rescaled)  # scalar
    x_t_rescaled = jnp.sqrt(jnp.maximum(x_s_sqnorm + 1.0 / c, MIN_NORM))  # scalar

    # Combine time and spatial components
    result = jnp.concatenate([x_t_rescaled[None], x_s_rescaled])  # Shape: (dim+1,)

    # Project onto manifold to correct numerical drift
    return _proj(result, c)


def _hcat(
    points: Float[Array, "N n"],
    c: float = 1.0,
) -> Float[Array, "dN_plus_1"]:
    """Lorentz direct concatenation for Hyperboloid points.

    Given N points on a d-dimensional Hyperboloid manifold (living in (d+1)-dimensional
    ambient space), concatenates them into a single point on a (dN)-dimensional Hyperboloid
    manifold (living in (dN+1)-dimensional ambient space).

    The formula is:
    y = [sqrt(sum(x_i[0]^2) - (N-1)/c), x_1[1:], ..., x_N[1:]]

    where x_i[0] is the time component and x_i[1:] are the space components.

    Args:
        points: N points in (d+1)-dimensional ambient space, shape (N, d+1).
                Each point satisfies: -x[0]^2 + sum(x[1:]^2) = -1/c
        c: Manifold curvature (positive)

    Returns:
        Single point in (dN+1)-dimensional ambient space, shape (dN+1,).
        - Time coordinate: sqrt(sum(x_i[0]^2) - (N-1)/c)
        - Space coordinates: concatenation of all input space components

    References:
        Qu, M., & Zou, J. (2022). Hyperbolic Hierarchical Knowledge Graph Embeddings for Link Prediction.
        Ahmad Bdeir, et al. "Fully hyperbolic convolutional neural networks for computer vision."
            arXiv preprint arXiv:2303.15919 (2023).

    Notes:
        The operation preserves the manifold structure: the output satisfies the Lorentz
        constraint for the (dN)-dimensional manifold.
    """
    N, _ambient_dim = points.shape

    # Extract time components (first coordinate of each point)
    time_components = points[:, 0]  # (N,)

    # Extract space components (remaining coordinates of each point)
    space_components = points[:, 1:]  # (N, d)

    # Compute new time coordinate using the formula
    # Note: MINUS (N-1)/c, not plus!
    # Numerical stability: clamp to MIN_NORM to handle points very close to origin
    time_sq_sum = jnp.sum(time_components**2) - (N - 1) / c
    time_new = jnp.sqrt(jnp.maximum(time_sq_sum, MIN_NORM))  # scalar

    # Concatenate all space components: [x_1[1:], x_2[1:], ..., x_N[1:]]
    space_concatenated = space_components.reshape(-1)  # (N*d,)

    # Combine: [time_new, space_concatenated]
    # Use time_new[None] instead of jnp.array([time_new]) to avoid extra allocation
    result = jnp.concatenate([time_new[None], space_concatenated])  # (1 + N*d,) = (dN+1,)

    return result


# ---------------------------------------------------------------------------
# Batch-compatible helpers (used by NN layers)
# ---------------------------------------------------------------------------


def _compute_mlr(
    x: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim_minus_1"],
    r: Float[Array, "out_dim 1"],
    c: float,
    clamping_factor: float,
    smoothing_factor: float,
    min_enorm: float = 1e-15,
) -> Float[Array, "batch out_dim"]:
    """Compute FHCNN multinomial linear regression on the hyperboloid.

    Args:
        x: Hyperboloid point(s), shape (batch, in_dim)
        z: Hyperplane tangent normals at origin (time coord omitted), shape (out_dim, in_dim-1)
        r: Hyperplane translations, shape (out_dim, 1)
        c: Manifold curvature (positive)
        clamping_factor: Clamping value for the output
        smoothing_factor: Smoothing factor for the output
        min_enorm: Minimum norm to avoid division by zero

    Returns:
        MLR scores, shape (batch, out_dim)

    References:
        Ahmad Bdeir et al. "Fully hyperbolic convolutional neural networks."
            arXiv:2303.15919 (2023).
    """
    sqrt_c = jnp.sqrt(c)
    sqrt_cr = sqrt_c * r.T  # (1, out_dim)
    z_norm = jnp.linalg.norm(z, ord=2, axis=-1, keepdims=True).clip(min=min_enorm).T  # (1, out_dim)
    x0 = x[:, 0:1]  # (batch, 1) - time coordinate
    x_rem = x[:, 1:]  # (batch, in_dim-1) - space coordinates
    zx_rem = jnp.einsum("bi,oi->bo", x_rem, z)  # (batch, out_dim)
    alpha = -x0 * sinh(sqrt_cr) * z_norm + cosh(sqrt_cr) * zx_rem  # (batch, out_dim)
    asinh_arg = sqrt_c * alpha / z_norm  # (batch, out_dim)

    eps = jnp.finfo(jnp.float32).eps if x.dtype == jnp.float32 else jnp.finfo(jnp.float64).eps
    clamp = clamping_factor * float(math.log(2 / eps))
    asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, smoothing_factor)  # (batch, out_dim)
    signed_dist2hyp = asinh(asinh_arg) / sqrt_c  # (batch, out_dim)
    res = z_norm * signed_dist2hyp  # (batch, out_dim)
    return res


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class Hyperboloid:
    """Hyperboloid manifold with automatic dtype casting.

    Provides all manifold operations with automatic casting of array inputs
    to the specified dtype. This eliminates the need for manual casting and
    provides better numerical stability control.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds.hyperboloid import Hyperboloid, VERSION_DEFAULT
        >>>
        >>> # Create manifold with float64 for better precision
        >>> manifold = Hyperboloid(dtype=jnp.float64)
        >>>
        >>> # Arrays are automatically cast to float64
        >>> x = jnp.array([1.0, 0.1, 0.2], dtype=jnp.float32)
        >>> x = manifold.proj(x, c=1.0)
        >>> x.dtype  # float64
    """

    VERSION_DEFAULT = VERSION_DEFAULT
    VERSION_SMOOTHENED = VERSION_SMOOTHENED

    def __init__(self, dtype: jnp.dtype = jnp.float32) -> None:
        self.dtype = dtype

    def _cast(self, x: Array) -> Array:
        """Cast array to target dtype if it's a floating-point array."""
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact):
            return x.astype(self.dtype)
        return x

    def create_origin(self, c: float, dim: int) -> Float[Array, "dim_plus_1"]:
        """Create hyperboloid origin [1/√c, 0, ..., 0]."""
        return _create_origin(c, dim, self.dtype)

    def minkowski_inner(self, x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"]) -> Float[Array, ""]:
        """Compute Minkowski inner product ⟨x, y⟩_L = -x₀y₀ + ⟨x_rest, y_rest⟩."""
        return _minkowski_inner(self._cast(x), self._cast(y))

    def proj(self, x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Project point onto hyperboloid."""
        return _proj(self._cast(x), c)

    def proj_batch(self, x: Float[Array, "... dim_plus_1"], c: float) -> Float[Array, "... dim_plus_1"]:
        """Project batched points onto hyperboloid (handles arbitrary leading dimensions)."""
        return _proj_batch(self._cast(x), c)

    def addition(self, x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Gyrovector addition on hyperboloid."""
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float, x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Scalar multiplication on hyperboloid."""
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r_cast, x, c)  # type: ignore[arg-type]

    def dist(
        self,
        x: Float[Array, "dim_plus_1"],
        y: Float[Array, "dim_plus_1"],
        c: float,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Compute geodesic distance between hyperboloid points."""
        return _dist(self._cast(x), self._cast(y), c, version_idx)

    def _dist(
        self,
        x: Float[Array, "dim_plus_1"],
        y: Float[Array, "dim_plus_1"],
        c: float,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Compatibility alias for legacy module-style API."""
        return self.dist(x, y, c, version_idx)

    def dist_0(self, x: Float[Array, "dim_plus_1"], c: float, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
        """Compute geodesic distance from hyperboloid origin."""
        return _dist_0(self._cast(x), c, version_idx)

    def _dist_0(self, x: Float[Array, "dim_plus_1"], c: float, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
        """Compatibility alias for legacy module-style API."""
        return self.dist_0(x, c, version_idx)

    def expmap(self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Exponential map: map tangent vector v at point x to manifold."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Exponential map from origin."""
        return _expmap_0(self._cast(v), c)

    def retraction(self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Retraction: first-order approximation of exponential map."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Logarithmic map: map point y to tangent space at point x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Logarithmic map from origin."""
        return _logmap_0(self._cast(y), c)

    def ptransp(
        self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float
    ) -> Float[Array, "dim_plus_1"]:
        """Parallel transport tangent vector v from point x to point y."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float) -> Float[Array, "dim_plus_1"]:
        """Parallel transport tangent vector v from origin to point y."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    def tangent_inner(
        self, u: Float[Array, "dim_plus_1"], v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float
    ) -> Float[Array, ""]:
        """Compute inner product of tangent vectors u and v at point x."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Float[Array, ""]:
        """Compute norm of tangent vector v at point x."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(
        self, grad: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float
    ) -> Float[Array, "dim_plus_1"]:
        """Convert Euclidean gradient to Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(
        self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float
    ) -> Float[Array, "dim_plus_1"]:
        """Project vector v onto tangent space at point x."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim_plus_1"], c: float, atol: float = 1e-4) -> Array:
        """Check if point x lies on hyperboloid."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: float) -> Array:
        """Check if vector v lies in tangent space at point x."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c)

    def lorentz_boost(
        self, x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: float
    ) -> Float[Array, "dim_plus_1 dim_plus_1"]:
        """Compute Lorentz boost matrix."""
        return _lorentz_boost(self._cast(x), self._cast(y), c)

    def hcat(
        self,
        points: Float[Array, "N n"],
        c: float = 1.0,
    ) -> Float[Array, "dN_plus_1"]:
        """Hyperbolic concatenation of N points into one point."""
        return _hcat(self._cast(points), c)

    def distance_rescale(self, x: Float[Array, "dim_plus_1"], c_in: float, c_out: float) -> Float[Array, "dim_plus_1"]:
        """Rescale distance from one curvature to another."""
        return _distance_rescale(self._cast(x), c_in, c_out)

    def embed_spatial_0(self, v_spatial: Float[Array, "... n"]) -> Float[Array, "... n_plus_1"]:
        """Embed spatial vector as tangent vector at origin."""
        return _embed_spatial_0(self._cast(v_spatial))

    def compute_mlr(
        self,
        x: Float[Array, "batch in_dim"],
        z: Float[Array, "out_dim in_dim_minus_1"],
        r: Float[Array, "out_dim 1"],
        c: float,
        clamping_factor: float,
        smoothing_factor: float,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """Compute multinomial linear regression on hyperboloid."""
        return _compute_mlr(self._cast(x), self._cast(z), self._cast(r), c, clamping_factor, smoothing_factor, min_enorm)
