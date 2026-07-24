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

import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jaxtyping import Array, Float

from ..utils.math_utils import acosh, cosh, sinh, smooth_clamp, smooth_clamp_min
from ._base import ManifoldBase
from .protocol import Curvature

# Default numerical parameters
MIN_NORM = 1e-15

# Version selection constants for _dist() and _dist_0()
VERSION_DEFAULT = 0
VERSION_SMOOTHENED = 1


def _create_origin(c: Curvature, dim: int, dtype=jnp.float32) -> Float[Array, "dim_plus_1"]:
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


def _proj(x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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


def _proj_batch(x: Float[Array, "... dim_plus_1"], c: Curvature) -> Float[Array, "... dim_plus_1"]:
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


def _addition(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
    """Lorentz gyrovector addition ``x ⊕ y`` on the Hyperboloid.

    Implements the gyroaddition of Chen et al. (2025b), adopted as the intrinsic Lorentz
    addition by Shi et al. (2026), Eq. (1)::

        x ⊕ y = Exp_x( PT_{0→x}( Log_0(y) ) )

    i.e. take the tangent vector at the origin that maps to ``y`` (``Log_0(y)``), parallel
    transport it from the origin to ``x``, then follow the geodesic from ``x``. This forms a
    gyrocommutative gyrogroup: ``0 ⊕ x = x ⊕ 0 = x`` and ``(⊖x) ⊕ x = x ⊕ (⊖x) = 0`` with
    inverse ``⊖x = (-1) ⊙ x = [x₀, -x_s]`` (see ``_scalar_mul``). Under the stereographic
    isometry it coincides with Möbius addition on the Poincaré ball.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Gyrovector sum x ⊕ y, shape (dim+1,)

    References:
        Chen et al. "Hyperbolic neural networks: gyrovector operations on the Lorentz model." 2025b.
        Shi et al. "Intrinsic Lorentz Neural Network." ICLR 2026, Eq. (1).
    """
    v0 = _logmap_0(y, c)  # tangent vector at the origin (first component 0)
    vx = _ptransp_0(v0, x, c)  # parallel transport origin → x; result is tangent at x
    res = _expmap(vx, x, c)  # geodesic step from x (already re-projected onto the manifold)
    return res


def _scalar_mul(r: float, x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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
def _dist_default(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Standard acosh distance with hard clipping."""
    sqrt_c = jnp.sqrt(c)
    lorentz_inner = _minkowski_inner(x, y)
    arg = jnp.clip(-c * lorentz_inner, min=1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if points are identical
    same = jnp.all(jnp.equal(x, y))
    return jnp.where(same, 0.0, res)  # type: ignore[return-value]


def _dist_smoothened(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
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
    c: Curvature,
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


def _sqdist(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Squared Lorentzian distance between hyperboloid points.

    Computes the squared Lorentzian distance of Law et al. (2019):

        d_L²(x, y) = -2/c - 2 * ⟨x, y⟩_L

    This is **not** the square of the geodesic distance :func:`_dist`. Since ⟨x,x⟩_L = ⟨y,y⟩_L =
    -1/c on the manifold, the two are related by a monotone closed form:

        d_L²(x, y) = (2/c) * (cosh(√c * d(x, y)) - 1) = (4/c) * sinh²(√c * d(x, y) / 2)

    so it is zero iff x == y, non-negative, symmetric, and strictly increasing in the geodesic
    distance d(x, y). That makes it a drop-in dissimilarity wherever a *monotone* distance proxy
    suffices (contrastive / commitment / k-NN / attention scores).

    Unlike :func:`_dist` it needs no ``acosh`` (no domain clamp, no infinite-gradient knee at
    x == y) and no coincidence reduction -- it is a single Minkowski inner product, which is both
    faster and numerically cleaner. The trade-off: it is not a true metric (no triangle
    inequality); use :func:`_dist` when the geodesic length itself is required.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Squared Lorentzian distance d_L²(x, y), scalar

    References:
        Law et al. "Lorentzian Distance Learning for Hyperbolic Representations." ICML 2019.
    """
    # d_L² = ||x - y||²_L = -2/c - 2⟨x,y⟩_L >= 0. Clip the float-rounding sliver below 0 at
    # coincidence (the true minimum is 0, where the gradient is 0 anyway).
    sqdist = -2.0 / c - 2.0 * _minkowski_inner(x, y)
    return jnp.clip(sqdist, min=0.0)


# Distance from origin implementations for lax.switch
def _dist_0_default(x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Standard acosh distance from origin with hard clipping."""
    sqrt_c = jnp.sqrt(c)
    x0 = x[0]
    arg = jnp.clip(sqrt_c * x0, min=1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if at origin
    origin = _create_origin(c, x.shape[0] - 1, x.dtype)
    at_origin = jnp.all(jnp.equal(x, origin))
    return jnp.where(at_origin, 0.0, res)  # type: ignore[return-value]


def _dist_0_smoothened(x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Smoothened distance from origin with soft clamping."""
    sqrt_c = jnp.sqrt(c)
    x0 = x[0]
    arg = smooth_clamp_min(sqrt_c * x0, 1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if at origin
    origin = _create_origin(c, x.shape[0] - 1, x.dtype)
    at_origin = jnp.all(jnp.equal(x, origin))
    return jnp.where(at_origin, 0.0, res)  # type: ignore[return-value]


def _dist_0(x: Float[Array, "dim_plus_1"], c: Curvature, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
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


def _expmap(v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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
    # Safe norm: +MIN_NORM² keeps sqrt's gradient finite at v=0
    # (sqrt'(0) = inf; the forward-only maximum below can't undo that NaN).
    v_norm = jnp.sqrt(v_sqnorm + MIN_NORM**2)
    c_norm_prod = sqrt_c * v_norm

    denom = jnp.maximum(c_norm_prod, MIN_NORM)
    cosh_term = cosh(c_norm_prod) * x
    sinh_term = sinh(c_norm_prod) / denom * v

    res = cosh_term + sinh_term
    res = _proj(res, c)
    return res


def _expmap_0(v: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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
    # Safe norm: +MIN_NORM² keeps sqrt's gradient finite at v=0 (see _expmap)
    v_norm = jnp.sqrt(v_sqnorm + MIN_NORM**2)
    c_norm_prod = sqrt_c * v_norm

    denom = jnp.maximum(c_norm_prod, MIN_NORM)
    sinh_scale = sinh(c_norm_prod) / denom

    v0 = v[0]
    v_rest = v[1:]

    res0 = cosh(c_norm_prod) / sqrt_c + sinh_scale * v0
    res_rest = sinh_scale * v_rest

    res = jnp.concatenate([res0[None], res_rest])
    res = _proj(res, c)
    return res


def _retraction(v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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


def _logmap(y: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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


def _logmap_0(y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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
    # Safe norm: +MIN_NORM² keeps the gradient finite at y = origin (y_rest = 0), where raw
    # jnp.linalg.norm has a 0/0 derivative. The forward-only jnp.maximum below cannot undo that
    # NaN (it gets multiplied into the VJP). This matters for the gyro-bias path of the PLFC /
    # Busemann FC layers, whose bias point is the origin at zero init.
    y_rest_norm = jnp.sqrt(jnp.sum(y_rest**2) + MIN_NORM**2)

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
    c: Curvature,
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


def _ptransp_0(v: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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
    u: Float[Array, "dim_plus_1"], v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature
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


def _tangent_norm(v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Compute norm of tangent vector v at point x.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Riemannian norm ||v||_x, scalar
    """
    inner = jnp.clip(_tangent_inner(v, v, x, c), min=0.0)
    # +MIN_NORM² keeps sqrt's gradient finite at v=0 (sqrt'(0)=inf); matches the safe norm in
    # _expmap. A bare clip(min=0.0) leaves the gradient as inf at the origin of the tangent space.
    return jnp.sqrt(inner + MIN_NORM**2)


def _egrad2rgrad(grad: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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


def _tangent_proj(v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
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


def _is_in_manifold(x: Float[Array, "dim_plus_1"], c: Curvature, atol: float = 1e-5) -> Array:
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
    v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature, atol: float | None = None
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


def _hcat(
    points: Float[Array, "N n"],
    c: Curvature = 1.0,
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

    time_N = points[:, 0]
    space_ND = points[:, 1:]

    # New time: sqrt(sum(x_i[0]^2) - (N-1)/c)
    time_sq_sum = jnp.sum(time_N**2) - (N - 1) / c
    time_new = jnp.sqrt(jnp.maximum(time_sq_sum, MIN_NORM))  # scalar

    space_flat_ND = space_ND.reshape(-1)  # (N*d,)

    result_A = jnp.concatenate([time_new[None], space_flat_ND])  # (1 + N*d,) = (dN+1,)

    return result_A


def _log_radius_concat(
    points: Float[Array, "N n"],
    c: Curvature = 1.0,
) -> Float[Array, "dN_plus_1"]:
    """Log-radius-preserving concatenation of N Hyperboloid points.

    A hyperboloid analog of the Poincaré β-concatenation (Shimizu et al. 2020), introduced by
    Shi et al. (2026, Sec. 4.3). Naively stacking the spatial parts of N blocks (as ``_hcat``
    does) biases the expected spatial radius upward with the post-concat dimension. To keep the
    expected *log* spatial radius invariant, each block's spatial part is rescaled by

        s = exp( ½ · ( ψ(n / 2) - ψ(nᵢ / 2) ) ),

    where ψ is the digamma function, ``nᵢ = d`` is the per-block spatial dimension and
    ``n = N · d`` is the total post-concat spatial dimension. The single time coordinate is then
    recomputed so the scaled point stays on the hyperboloid (here ``1/c`` equals the reference
    ``k = -1/K``):

        t' = sqrt( 1/c + s² · Σᵢ (tᵢ² - 1/c) ).

    When ``N = 1`` the scale is ``1`` and this reduces exactly to ``_hcat``.

    Args:
        points: N points in (d+1)-dimensional ambient space, shape (N, d+1).
                Each point satisfies -x[0]² + ||x[1:]||² = -1/c.
        c: Manifold curvature (positive).

    Returns:
        Single point in (dN+1)-dimensional ambient space, shape (dN+1,).
        - Time coordinate: sqrt(1/c + s² · Σ(tᵢ² - 1/c))
        - Space coordinates: each block's space part scaled by ``s``, then concatenated.

    References:
        Shi et al. "Intrinsic Lorentz Neural Network." ICLR 2026, Sec. 4.3.
        Shimizu et al. "Hyperbolic Neural Networks++." ICLR 2021 (β-concatenation analog).
    """
    N, ambient_dim = points.shape
    d = ambient_dim - 1  # per-block spatial dimension (nᵢ)
    n_total = N * d  # post-concat spatial dimension (n)

    # Digamma scale keeps E[log‖v_spatial‖] constant across the concat dim; s == 1 when N == 1.
    scale = jnp.exp(0.5 * (digamma(n_total / 2.0) - digamma(d / 2.0)))

    time_N = points[:, 0]  # (N,)
    space_ND = points[:, 1:]  # (N, d)

    space_scaled_flat = (scale * space_ND).reshape(-1)  # (N*d,)

    # Recompute the time coordinate so the scaled point lies on the (dN)-dim hyperboloid.
    time_sq = 1.0 / c + scale**2 * jnp.sum(time_N**2 - 1.0 / c)  # scalar
    time_new = jnp.sqrt(jnp.maximum(time_sq, MIN_NORM))  # scalar

    result_A = jnp.concatenate([time_new[None], space_scaled_flat])  # (1 + N*d,) = (dN+1,)
    return result_A


# ---------------------------------------------------------------------------
# Batch-compatible helpers (used by NN layers)
# ---------------------------------------------------------------------------


def _compute_mlr(
    x: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim_minus_1"],
    r: Float[Array, "out_dim 1"],
    c: Curvature,
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
    sqrt_cr_1P = sqrt_c * r.T  # r:(P,1) → r.T:(1,P)
    z_norm_1P = jnp.linalg.norm(z, ord=2, axis=-1, keepdims=True).clip(min=min_enorm).T  # (1,P)
    x0_B1 = x[:, 0:1]  # time coordinate
    x_rem_BD = x[:, 1:]  # space coordinates, D = in_dim-1
    zx_rem_BP = jnp.einsum("bi,oi->bo", x_rem_BD, z)
    alpha_BP = -x0_B1 * sinh(sqrt_cr_1P) * z_norm_1P + cosh(sqrt_cr_1P) * zx_rem_BP
    asinh_arg_BP = sqrt_c * alpha_BP / z_norm_1P

    eps = jnp.finfo(jnp.float32).eps if x.dtype == jnp.float32 else jnp.finfo(jnp.float64).eps
    clamp = clamping_factor * float(math.log(2 / eps))
    asinh_arg_BP = smooth_clamp(asinh_arg_BP, -clamp, clamp, smoothing_factor)
    signed_dist2hyp_BP = jnp.asinh(asinh_arg_BP) / sqrt_c
    res_BP = z_norm_1P * signed_dist2hyp_BP
    return res_BP


def _busemann(x: Float[Array, "dim_plus_1"], v: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Closed-form Lorentz Busemann function ``B^v(x)`` (point-to-horosphere coordinate).

    For a unit ideal direction ``v ∈ S^{n-1}`` (a *spatial* unit vector, dim ``d``) and a
    hyperboloid point ``x`` (ambient dim ``d+1``, time first), with curvature ``c = -K > 0``
    (Chen et al. 2026, Eq. 4)::

        B^v(x) = (1/√c) · log( √c · (x_t - ⟨x_s, v⟩) )

    with ``x_t = x[0]``, ``x_s = x[1:]``. The argument ``x_t - ⟨x_s, v⟩`` equals ``-⟨x, ω⟩_L``
    for the null lift ``ω = (1, v)`` and is strictly positive on the upper sheet
    (Cauchy-Schwarz: ``x_t = √(1/c + ‖x_s‖²) ≥ ‖x_s‖ ≥ ⟨x_s, v⟩``); it → 0 only as ``x``
    runs off to the ideal point ``v``, so the log argument is floored at ``MIN_NORM``.
    ``B^v(origin) = 0`` for unit ``v``.

    ``v`` is assumed unit-norm and is **not** normalized here — callers (the BMLR/BFC layers,
    or a downstream horospherical projection) normalize their direction set to the sphere.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        v: Unit ideal direction (spatial), shape (dim,)
        c: Curvature (positive)

    Returns:
        Busemann coordinate B^v(x), scalar

    References:
        Chen, Schölkopf, and Sebe. "Hyperbolic Busemann Neural Networks." 2026, Eq. 4.
    """
    sqrt_c = jnp.sqrt(c)
    arg = sqrt_c * (x[0] - jnp.dot(x[1:], v))  # = -sqrt_c * minkowski_inner(x, [1, v]); > 0 on the upper sheet
    return jnp.log(jnp.maximum(arg, MIN_NORM)) / sqrt_c


def _lorentz_boost(mu: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1 dim_plus_1"]:
    """Lorentz boost matrix ``B`` that sends ``mu`` to the origin (``B @ mu = origin``).

    Builds the pure Lorentz boost — a symmetric, proper, orthochronous Lorentz
    transformation — carrying the hyperboloid point ``mu`` to the manifold origin
    ``[1/√c, 0, ..., 0]``. With the Lorentz factor ``gamma = √c·mu₀`` (≥ 1) and the
    coordinate velocity ``v = mu_s / mu₀`` (``‖v‖ < 1``)::

        B = [[ gamma,       -gamma·vᵀ                    ],
             [-gamma·v,   I_d + gamma²/(1+gamma)·(v vᵀ)      ]]

    ``B`` is symmetric, so a batch of row-vector points is boosted by ``x_NA @ B.T``
    (equivalently ``x_NA @ B``); always follow with :func:`_proj_batch` to clear the
    float rounding off the constraint surface. The inverse boost is the boost of
    ``[mu₀, -mu_s]`` (i.e. ``B`` with ``v → -v``). The ``gamma → 1`` limit (``mu`` at the
    origin) is benign: ``v → 0`` and ``B → I``.

    Args:
        mu: Hyperboloid point to send to the origin, shape (dim+1,).
        c: Curvature (positive).

    Returns:
        Boost matrix ``B``, shape (dim+1, dim+1), with ``B @ mu = origin``.

    References:
        Chami et al. "HoroPCA: Hyperbolic dimensionality reduction via horospherical
            projections." ICML 2021 (Fréchet-mean centering).
    """
    sqrt_c = jnp.sqrt(c)
    mu_t = mu[0]  # time coordinate mu₀ ≥ 1/√c > 0
    mu_s = mu[1:]  # spatial coordinates
    dim = mu.shape[0] - 1

    gamma = sqrt_c * mu_t  # Lorentz factor gamma = √c·mu₀ ≥ 1
    v_D = mu_s / mu_t  # coordinate velocity, ‖v‖ < 1 (division by mu₀ > 0 is safe)

    top_row_A = jnp.concatenate([gamma[None], -gamma * v_D])  # (A,)
    bottom_left_D1 = (-gamma * v_D)[:, None]  # (D, 1)
    eye_DD = jnp.eye(dim, dtype=mu.dtype)  # (D, D)
    bottom_right_DD = eye_DD + (gamma**2 / (1.0 + gamma)) * jnp.outer(v_D, v_D)  # (D, D)
    bottom_DA = jnp.concatenate([bottom_left_D1, bottom_right_DD], axis=1)  # (D, A)
    boost_AA = jnp.concatenate([top_row_A[None, :], bottom_DA], axis=0)  # (A, A)
    return boost_AA


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class Hyperboloid(ManifoldBase):
    """Hyperboloid manifold with automatic dtype casting.

    Provides all manifold operations with automatic casting of array inputs
    to the specified dtype. This eliminates the need for manual casting and
    provides better numerical stability control.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)
        c: Curvature value (default: 1.0). Must be positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds.hyperboloid import Hyperboloid, VERSION_DEFAULT
        >>>
        >>> # Create manifold with float64 for better precision
        >>> manifold = Hyperboloid(dtype=jnp.float64)
        >>>
        >>> # Custom curvature
        >>> manifold = Hyperboloid(c=0.5)
        >>> c = manifold.c  # returns 0.5
    """

    VERSION_DEFAULT = VERSION_DEFAULT
    VERSION_SMOOTHENED = VERSION_SMOOTHENED

    def create_origin(self, c: Curvature, dim: int) -> Float[Array, "dim_plus_1"]:
        """Create hyperboloid origin [1/√c, 0, ..., 0]."""
        return _create_origin(c, dim, self.dtype)

    def minkowski_inner(self, x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"]) -> Float[Array, ""]:
        """Compute Minkowski inner product ⟨x, y⟩_L = -x₀y₀ + ⟨x_rest, y_rest⟩."""
        return _minkowski_inner(self._cast(x), self._cast(y))

    def proj(self, x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
        """Project point onto hyperboloid."""
        return _proj(self._cast(x), c)

    def proj_batch(self, x: Float[Array, "... dim_plus_1"], c: Curvature) -> Float[Array, "... dim_plus_1"]:
        """Project batched points onto hyperboloid (handles arbitrary leading dimensions)."""
        return _proj_batch(self._cast(x), c)

    def addition(
        self, x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature
    ) -> Float[Array, "dim_plus_1"]:
        """Lorentz gyrovector addition ``x ⊕ y = Exp_x(PT_{0→x}(Log_0(y)))``.

        The intrinsic Lorentz addition of Chen et al. (2025b) / Shi et al. (2026, Eq. 1).
        Forms a gyrocommutative gyrogroup with identity = origin and inverse
        ``⊖x = (-1) ⊙ x = [x₀, -x_s]``; matches Poincaré Möbius addition under the
        stereographic isometry. ``scalar_mul`` provides the companion gyro scaling (Eq. 2).
        """
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float, x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
        """Scalar multiplication on hyperboloid."""
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r_cast, x, c)  # type: ignore[arg-type]

    def dist(
        self,
        x: Float[Array, "dim_plus_1"],
        y: Float[Array, "dim_plus_1"],
        c: Curvature,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Compute geodesic distance between hyperboloid points."""
        return _dist(self._cast(x), self._cast(y), c, version_idx)

    def dist_0(self, x: Float[Array, "dim_plus_1"], c: Curvature, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
        """Compute geodesic distance from hyperboloid origin."""
        return _dist_0(self._cast(x), c, version_idx)

    def sqdist(self, x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
        """Squared Lorentzian distance ``d_L²(x, y) = -2/c - 2⟨x,y⟩_L`` (Law et al. 2019).

        A fast, ``acosh``-free dissimilarity that is monotone in the geodesic :meth:`dist`
        (``d_L² = (2/c)(cosh(√c·d) - 1)``) but is **not** the squared geodesic distance and **not**
        a true metric. Prefer it where a smooth monotone distance proxy suffices (contrastive /
        commitment / attention scores); use :meth:`dist` when the geodesic length is required.
        """
        return _sqdist(self._cast(x), self._cast(y), c)

    def expmap(self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
        """Exponential map: map tangent vector v at point x to manifold."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
        """Exponential map from origin."""
        return _expmap_0(self._cast(v), c)

    def retraction(
        self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature
    ) -> Float[Array, "dim_plus_1"]:
        """Retraction: first-order approximation of exponential map."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
        """Logarithmic map: map point y to tangent space at point x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
        """Logarithmic map from origin."""
        return _logmap_0(self._cast(y), c)

    def ptransp(
        self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature
    ) -> Float[Array, "dim_plus_1"]:
        """Parallel transport tangent vector v from point x to point y."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(
        self, v: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature
    ) -> Float[Array, "dim_plus_1"]:
        """Parallel transport tangent vector v from origin to point y."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    def tangent_inner(
        self, u: Float[Array, "dim_plus_1"], v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature
    ) -> Float[Array, ""]:
        """Compute inner product of tangent vectors u and v at point x."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
        """Compute norm of tangent vector v at point x."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(
        self, grad: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature
    ) -> Float[Array, "dim_plus_1"]:
        """Convert Euclidean gradient to Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(
        self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature
    ) -> Float[Array, "dim_plus_1"]:
        """Project vector v onto tangent space at point x."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim_plus_1"], c: Curvature, atol: float = 1e-4) -> Array:
        """Check if point x lies on hyperboloid."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature) -> Array:
        """Check if vector v lies in tangent space at point x."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c)

    def hcat(
        self,
        points: Float[Array, "N n"],
        c: Curvature = 1.0,
    ) -> Float[Array, "dN_plus_1"]:
        """Hyperbolic concatenation of N points into one point."""
        return _hcat(self._cast(points), c)

    def log_radius_concat(
        self,
        points: Float[Array, "N n"],
        c: Curvature = 1.0,
    ) -> Float[Array, "dN_plus_1"]:
        """Log-radius-preserving concatenation of N points (Shi et al. 2026, Sec. 4.3).

        Like :meth:`hcat`, but rescales each block's spatial part by a digamma factor so the
        expected log spatial radius is invariant to the number of concatenated blocks. Reduces
        to :meth:`hcat` when ``N == 1``.
        """
        return _log_radius_concat(self._cast(points), c)

    def embed_spatial_0(self, v_spatial: Float[Array, "... n"]) -> Float[Array, "... n_plus_1"]:
        """Embed spatial vector as tangent vector at origin."""
        return _embed_spatial_0(self._cast(v_spatial))

    def compute_mlr(
        self,
        x: Float[Array, "batch in_dim"],
        z: Float[Array, "out_dim in_dim_minus_1"],
        r: Float[Array, "out_dim 1"],
        c: Curvature,
        clamping_factor: float,
        smoothing_factor: float,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """Compute multinomial linear regression on hyperboloid."""
        return _compute_mlr(self._cast(x), self._cast(z), self._cast(r), c, clamping_factor, smoothing_factor, min_enorm)

    def busemann(self, x: Float[Array, "dim_plus_1"], v: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
        """Closed-form Lorentz Busemann function ``B^v(x) = (1/√c)·log(√c·(x_t - ⟨x_s, v⟩))``.

        Point-to-horosphere coordinate (Chen et al. 2026, Eq. 4). ``v`` must be a *unit*
        spatial direction (dim ``d``) — it is **not** normalized internally. Single point
        ``(d+1,)`` → scalar; use :func:`jax.vmap` for batching and over a direction set.
        ``B^v(origin) = 0``.

        See Also
        --------
        For an *asymmetric* quasimetric energy, compose this Busemann coordinate with an
        external Euclidean quasimetric (e.g. IQE/MRN). Do **not** reach for
        :meth:`Poincare.apollonian_dist` expecting asymmetry — it is a coboundary
        (symmetrizes to ``√c·dist``) and cannot deliver circulation.
        """
        return _busemann(self._cast(x), self._cast(v), c)

    def lorentz_boost(self, mu: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1 dim_plus_1"]:
        """Lorentz boost matrix ``B`` with ``B @ mu = origin`` (sends ``mu`` to the origin).

        Symmetric, proper, orthochronous Lorentz transformation. Boost a batch of
        row-vector points with ``x_NA @ B.T`` followed by :meth:`proj_batch`; the inverse
        boost is the boost of ``[mu₀, -mu_s]``. Used to Fréchet-mean-center data for
        HoroPCA (Chami et al. 2021). See :func:`_lorentz_boost`.
        """
        return _lorentz_boost(self._cast(mu), c)
