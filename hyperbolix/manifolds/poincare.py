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
import jax.scipy.special
from jaxtyping import Array, Float

from ..utils.math_utils import acosh, atanh, cosh, sinh, smooth_clamp
from ._base import ManifoldBase
from ._gyrovector_core import (
    MIN_NORM,
    _addition,
    _conformal_factor,
    _conformal_factor_batch,
    _gyration,
    _proj,
)
from .protocol import Curvature

# Version selection constants for dist() and dist_0()
VERSION_MOBIUS_DIRECT = 0
VERSION_MOBIUS = 1
VERSION_METRIC_TENSOR = 2


def _scalar_mul(r: float, x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
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
    # Safe norm: sqrt(||x||² + eps²) has finite gradients at x=0. The previous
    # maximum(norm, MIN_NORM) only guarded the forward — linalg.norm's VJP at 0
    # is 0/0 = NaN, and 0-cotangent · NaN is still NaN.
    x_norm = jnp.sqrt(jnp.sum(x**2) + MIN_NORM**2)
    c_norm_prod = jnp.sqrt(c) * x_norm
    res = jnp.tanh(r * atanh(c_norm_prod)) / c_norm_prod * x
    res = _proj(res, c)
    return res


def _embed_spatial_0(v_spatial: Float[Array, "... n"]) -> Float[Array, "... n"]:
    """Identity embedding for the Poincaré ball: no time coord to prepend.

    Kept for API parity with Hyperboloid/ProperVelocity so the gyro-normalization
    layers can treat every manifold uniformly (the bias-lift step). On the ball a
    tangent-at-origin vector *is* the spatial vector, so this is the identity.
    """
    return v_spatial


# Distance implementations for lax.switch
def _dist_mobius_direct(x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Direct Möbius distance formula (fastest)."""
    sqrt_c = jnp.sqrt(c)
    x2y2 = jnp.dot(x, x) * jnp.dot(y, y)
    xy = jnp.dot(x, y)
    # Safe norm: finite gradient at x == y (norm's VJP at 0 is 0/0 = NaN)
    num = jnp.sqrt(jnp.sum((y - x) ** 2) + MIN_NORM**2)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c**2 * x2y2, MIN_NORM))
    xysum_norm = num / denom
    dist_c = atanh(sqrt_c * xysum_norm)
    return 2 * dist_c / sqrt_c


def _dist_mobius(x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Möbius distance via addition."""
    sqrt_c = jnp.sqrt(c)
    diff = _addition(-x, y, c)
    # Safe norm: finite gradient at x == y (diff = 0)
    diff_norm = jnp.sqrt(jnp.sum(diff**2) + MIN_NORM**2)
    dist_c = atanh(sqrt_c * diff_norm)
    return 2 * dist_c / sqrt_c


def _dist_metric_tensor(x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Metric tensor induced distance."""
    xy_diff_sqnorm = jnp.dot(x - y, x - y)
    # 1 - c||x||² via the boundary-clamped conformal factor (= 2/λ(x)). A bare 1 - c||x||² hits 0
    # for an unprojected near-boundary point and blows up the divide; reuse the module-wide floor.
    one_minus_cx = 2.0 / _conformal_factor(x, c)
    one_minus_cy = 2.0 / _conformal_factor(y, c)
    arg = 1 + 2 * c * xy_diff_sqnorm / (one_minus_cx * one_minus_cy)
    condition = arg < 1 + MIN_NORM
    return jnp.where(condition, 0.0, acosh(arg) / jnp.sqrt(c))  # type: ignore[return-value]


def _apollonian_dist(x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Apollonian weak metric δ(x, y) on the Poincaré ball.

    A *weak metric*: δ(x, y) ≥ 0, δ(x, x) = 0 and the triangle inequality hold, but δ is
    NON-SYMMETRIC (δ(x, y) ≠ δ(y, x) in general). It is defined as the boundary supremum
    δ(x, y) = sup_{‖a‖=1/√c} log(‖x - a‖ / ‖y - a‖). Its symmetrization recovers the geodesic
    distance: δ(x, y) + δ(y, x) = √c · dist(x, y).

    Closed form (curvature-c, n-dimensional generalization of Papadopoulos & Troyanov, Thm 2):
        δ_c(x, y) = log( (√c‖x - y‖ + G) / (1 - c‖y‖²) )
        G = √(c²‖x‖²‖y‖² - 2c⟨x, y⟩ + 1)        (= |c·x·ȳ - 1| in the n=2 / C case)

    The paper's Theorem 2 covers the unit disk (c=1, n=2, x,y ∈ C). The complex term |x·ȳ - 1|
    expands to the real, dimension-free radical G; curvature enters via the similarity x ↦ √c·x
    (δ is a log of a *ratio* of distances, so similarity-invariant — paper Prop 4.3).

    Args:
        x: Poincaré ball point, shape (dim,)
        y: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Apollonian weak metric δ(x, y), scalar

    References:
        Papadopoulos & Troyanov. "Weak metrics on Euclidean domains." (Theorem 2.)
    """
    sqrt_c = jnp.sqrt(c)
    x2 = jnp.dot(x, x)
    y2 = jnp.dot(y, y)
    xy = jnp.dot(x, y)
    # G = |c·x·ȳ - 1| generalized to ℝⁿ, i.e. G² = c²‖x‖²‖y‖² - 2c⟨x,y⟩ + 1. We use the
    # Gram-determinant form (1 - c⟨x,y⟩)² + c²(‖x‖²‖y‖² - ⟨x,y⟩²): a sum of two non-negative
    # terms (Cauchy-Schwarz ⇒ Gram det ≥ 0), so no catastrophic cancellation near the boundary.
    # At x=y the Gram term is exactly 0, so G = 1 - c‖x‖² = denom and δ(x,x)=0 to machine precision.
    gram = x2 * y2 - xy**2  # ‖x‖²‖y‖² - ⟨x,y⟩² ≥ 0 (squared area of the x,y parallelogram)
    G = jnp.sqrt(jnp.maximum((1.0 - c * xy) ** 2 + c**2 * gram, MIN_NORM))
    num = sqrt_c * jnp.linalg.norm(x - y) + G
    # Denominator 1 - c‖y‖² = 2/λ(y); reuse the already-clamped conformal factor so the
    # near-boundary floor matches the rest of the module (δ → ∞ as y → ∂ball is expected).
    denom = 2.0 / _conformal_factor(y, c)
    return jnp.log(num / denom)


def _dist(
    x: Float[Array, "dim"],
    y: Float[Array, "dim"],
    c: Curvature,
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
    """
    return lax.switch(version_idx, [_dist_mobius_direct, _dist_mobius, _dist_metric_tensor], x, y, c)


# Distance from origin implementations for lax.switch
def _dist_0_mobius(x: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Möbius distance from origin (mobius_direct and mobius use same formula)."""
    sqrt_c = jnp.sqrt(c)
    # Safe norm: finite gradient at the origin
    x_norm = jnp.sqrt(jnp.sum(x**2) + MIN_NORM**2)
    dist_c = atanh(sqrt_c * x_norm)
    return 2 * dist_c / sqrt_c


def _dist_0_metric_tensor(x: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Metric tensor distance from origin."""
    x_sqnorm = jnp.dot(x, x)
    # 1 - c||x||² via the boundary-clamped conformal factor (= 2/λ(x)) so a near-boundary point
    # cannot drive the denominator to 0; consistent with _dist_metric_tensor and _apollonian_dist.
    one_minus_cx = 2.0 / _conformal_factor(x, c)
    arg = 1 + 2 * c * x_sqnorm / one_minus_cx
    condition = arg < 1 + MIN_NORM
    return jnp.where(condition, 0.0, acosh(arg) / jnp.sqrt(c))  # type: ignore[return-value]


def _dist_0(x: Float[Array, "dim"], c: Curvature, version_idx: int = VERSION_MOBIUS_DIRECT) -> Float[Array, ""]:
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
    return lax.switch(version_idx, [_dist_0_mobius, _dist_0_mobius, _dist_0_metric_tensor], x, c)


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
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
    # Safe norm: sqrt(||v||² + eps²) has well-defined gradients at v=0,
    # matching _expmap_0. The previous maximum(·, MIN_NORM) only guarded the
    # forward — linalg.norm's VJP at 0 is 0/0 = NaN, and 0·NaN is still NaN.
    v_norm = jnp.sqrt(jnp.sum(v**2) + MIN_NORM**2)
    c_norm_prod = jnp.sqrt(c) * v_norm
    lambda_x = _conformal_factor(x, c)
    # ||second_term|| = |tanh(·)|/√c < 1/√c, i.e. strictly inside the ball, so an explicit _proj
    # here is a no-op in the valid regime; _addition re-projects its output regardless. Skip it.
    second_term = jnp.tanh(c_norm_prod * lambda_x / 2) / c_norm_prod * v
    res = _addition(x, second_term, c)
    return res


def _expmap_0(v: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    Args:
        v: Tangent vector at origin, shape (dim,)
        c: Curvature (positive)

    Returns:
        Point exp_0(v), shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # Safe norm: sqrt(||v||² + eps²) has well-defined gradients at v=0,
    # unlike jnp.linalg.norm(v) which produces NaN gradients (0/0).
    # This matters when zero tangent vectors arise (e.g., all-black pixel patches).
    v_norm = jnp.sqrt(jnp.sum(v**2) + MIN_NORM**2)
    c_norm_prod = jnp.sqrt(c) * v_norm
    res = jnp.tanh(c_norm_prod) / c_norm_prod * v
    res = _proj(res, c)
    return res


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
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


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
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
    # Safe norm: finite gradient at x == y (raw norm's VJP at 0 is 0/0 = NaN). Identical quantity
    # and form to _dist_mobius_direct's num — keep the two consistent.
    num = jnp.sqrt(jnp.sum((y - x) ** 2) + MIN_NORM**2)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c**2 * x2y2, MIN_NORM))
    sub_norm = num / denom
    c_norm_prod = jnp.maximum(jnp.sqrt(c) * sub_norm, MIN_NORM)
    lambda_x = _conformal_factor(x, c)
    res = 2 * atanh(c_norm_prod) / (c_norm_prod * lambda_x) * sub
    return res


def _logmap_0(y: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    Args:
        y: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Tangent vector log_0(y), shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # Safe norm: sqrt(||y||² + eps²) has well-defined gradients at y=0.
    y_norm = jnp.sqrt(jnp.sum(y**2) + MIN_NORM**2)
    c_norm_prod = jnp.sqrt(c) * y_norm
    res = atanh(c_norm_prod) / c_norm_prod * y
    return res


def _ptransp(v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
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


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
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


def _tangent_inner(u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
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


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
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
    # Safe norm: sqrt(||v||² + eps²) keeps the gradient finite at v=0. A bare jnp.linalg.norm
    # has VJP 0/0 = NaN there (and 0-cotangent · NaN stays NaN), matching the _expmap/_proj idiom.
    return lambda_x * jnp.sqrt(jnp.sum(v**2) + MIN_NORM**2)


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
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


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
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


def _is_in_manifold(x: Float[Array, "dim"], c: Curvature, atol: float = 1e-5) -> Array:
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


def _is_in_tangent_space(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Array:
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


def _compute_mlr_pp(
    x: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim"],
    r: Float[Array, "out_dim 1"],
    c: Curvature,
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
    sqrt_c2r_1P = 2 * sqrt_c * r.T  # (1, P) — r is (P, 1), .T broadcasts

    # Safe norm: sqrt(sum(z²) + eps²) avoids NaN gradients at z=0
    z_norm_P1 = jnp.sqrt(jnp.sum(z**2, axis=-1, keepdims=True) + min_enorm**2)  # (P, 1)

    # Conformal factor lam(x) = 2 / (1 - c||x||²) per HNN++ Eq. 26 (boundary-clamped).
    # NOTE: van Spengler's poincare-resnet repo has 2*(1 - c||x||²) here — a
    # transcription bug the same author fixed in hypll. Do not "restore" it.
    lam_B1 = _conformal_factor_batch(x, c)  # (B, 1)

    z_unitx_BP = jnp.einsum("bi,oi->bo", x, z / z_norm_P1)  # (B, P)
    asinh_arg_BP = sqrt_c * lam_B1 * z_unitx_BP * cosh(sqrt_c2r_1P) - (lam_B1 - 1) * sinh(sqrt_c2r_1P)  # (B, P)

    eps = jnp.finfo(jnp.float32).eps if x.dtype == jnp.float32 else jnp.finfo(jnp.float64).eps
    clamp = clamping_factor * float(math.log(2 / eps))
    asinh_arg_BP = smooth_clamp(asinh_arg_BP, -clamp, clamp, smoothing_factor)  # (B, P)
    signed_dist2hyp_BP = jnp.asinh(asinh_arg_BP) / sqrt_c  # (B, P)
    res_BP = 2 * z_norm_P1.T * signed_dist2hyp_BP  # z_norm.T broadcasts (1, P) over (B, P)
    return res_BP


# ---------------------------------------------------------------------------
# Beta-concatenation (HNN++, Shimizu et al. 2020)
# ---------------------------------------------------------------------------


def _beta_concat(points: Float[Array, "M n_i"], c: Curvature) -> Float[Array, "n"]:
    """Beta-concatenation of M equal-dimensional Poincaré ball points.

    Concatenates M points in the tangent space at the origin with a scaling
    correction based on the Euler beta function, then maps back to the manifold.

    Args:
        points: M points on the Poincaré ball, shape (M, n_i). All points
                must have the same dimension n_i.
        c: Curvature (positive)

    Returns:
        Concatenated point on the Poincaré ball, shape (M * n_i,)

    References:
        Shimizu et al. "Hyperbolic neural networks++." arXiv:2006.08210 (2020).
    """
    M, n_i = points.shape
    n = M * n_i  # concatenated dimension

    # Euler beta function ratio: B(n/2, 1/2) / B(n_i/2, 1/2).
    # jax.scipy.special.beta returns a strongly-typed float64 scalar under
    # global jax_enable_x64 (unlike most scalar math, which stays weak-typed),
    # so without the cast the ratio would promote the computation to float64.
    beta_n = jax.scipy.special.beta(n / 2.0, 0.5)
    beta_ni = jax.scipy.special.beta(n_i / 2.0, 0.5)
    scale = jnp.asarray(beta_n / beta_ni, dtype=points.dtype)

    # Map all points to tangent space at origin
    tangent_MD = jax.vmap(_logmap_0, in_axes=(0, None))(points, c)  # (M, n_i)

    # Scale and concatenate in tangent space
    v_N = (scale * tangent_MD).reshape(n)  # (M*n_i,)

    # Map back to manifold
    return _expmap_0(v_N, c)  # (M*n_i,)


def _busemann(x: Float[Array, "dim"], v: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Closed-form Poincaré Busemann function ``B^v(x)`` (point-to-horosphere coordinate).

    For a unit ideal direction ``v ∈ S^{n-1}`` and a ball point ``x`` (``‖x‖ < 1/√c``), with
    curvature ``c = -K > 0`` (Chen et al. 2026, Eq. 3)::

        B^v(x) = (1/√c) · log( ‖v - √c·x‖² / (1 - c·‖x‖²) )

    Numerator and denominator are both strictly positive inside the ball; the denominator is
    floored exactly as in :func:`_conformal_factor`'s ``1 - c‖x‖²``. ``B^v(origin) = 0`` for
    unit ``v``. Under the Poincaré↔Hyperboloid isometry this agrees with the Lorentz Busemann
    function for the same ``v``.

    ``v`` is assumed unit-norm and is **not** normalized here — callers normalize their
    direction set to the sphere.

    Args:
        x: Poincaré ball point, shape (dim,)
        v: Unit ideal direction, shape (dim,)
        c: Curvature (positive)

    Returns:
        Busemann coordinate B^v(x), scalar

    References:
        Chen, Schölkopf, and Sebe. "Hyperbolic Busemann Neural Networks." 2026, Eq. 3.
    """
    sqrt_c = jnp.sqrt(c)
    num = jnp.sum((v - sqrt_c * x) ** 2)
    denom = jnp.maximum(1.0 - c * jnp.dot(x, x), MIN_NORM)
    return jnp.log(num / denom) / sqrt_c


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class Poincare(ManifoldBase):
    """Poincaré ball manifold with automatic dtype casting.

    Provides all manifold operations with automatic casting of array inputs
    to the specified dtype. This eliminates the need for manual casting and
    provides better numerical stability control.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)
        c: Curvature value (default: 1.0). Must be positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds.poincare import Poincare, VERSION_MOBIUS_DIRECT
        >>>
        >>> # Create manifold with float64 for better precision
        >>> manifold = Poincare(dtype=jnp.float64)
        >>>
        >>> # Static curvature
        >>> manifold = Poincare(c=0.1)
        >>> c = manifold.c  # returns 0.1
    """

    VERSION_MOBIUS_DIRECT = VERSION_MOBIUS_DIRECT
    VERSION_MOBIUS = VERSION_MOBIUS
    VERSION_METRIC_TENSOR = VERSION_METRIC_TENSOR

    def proj(self, x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Project point onto Poincaré ball by clipping norm."""
        return _proj(self._cast(x), c)

    def gyration(
        self, x: Float[Array, "dim"], y: Float[Array, "dim"], z: Float[Array, "dim"], c: Curvature
    ) -> Float[Array, "dim"]:
        """Compute gyration gyr[x,y]z to restore commutativity."""
        return _gyration(self._cast(x), self._cast(y), self._cast(z), c)

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Möbius gyrovector addition x ⊕ y."""
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float, x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Scalar multiplication r ⊗ x on Poincaré ball."""
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r_cast, x, c)  # type: ignore[arg-type]

    def dist(
        self,
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: Curvature,
        version_idx: int = VERSION_MOBIUS_DIRECT,
    ) -> Float[Array, ""]:
        """Compute geodesic distance between Poincaré ball points."""
        return _dist(self._cast(x), self._cast(y), c, version_idx)

    def dist_0(self, x: Float[Array, "dim"], c: Curvature, version_idx: int = VERSION_MOBIUS_DIRECT) -> Float[Array, ""]:
        """Compute geodesic distance from Poincaré ball origin."""
        return _dist_0(self._cast(x), c, version_idx)

    def apollonian_dist(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
        """Apollonian weak metric δ(x, y) — non-symmetric; symmetrizes to √c·dist(x, y).

        .. warning::
            Although ``δ`` is non-symmetric, its antisymmetric part is an exact **coboundary**
            (``δ(x, y) - δ(y, x)`` is a difference of a per-point potential), so it carries
            **no circulation** and is useless as a quasimetric energy. Do not reach for this
            expecting genuine asymmetry — use a :meth:`busemann` coordinate fed to an external
            quasimetric combinator (IQE/MRN) instead.
        """
        return _apollonian_dist(self._cast(x), self._cast(y), c)

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Exponential map: map tangent vector v at point x to manifold."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Exponential map from origin: map tangent vector v at origin to manifold."""
        return _expmap_0(self._cast(v), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Retraction: first-order approximation of exponential map."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Logarithmic map: map point y to tangent space at point x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Logarithmic map from origin: map point y to tangent space at origin."""
        return _logmap_0(self._cast(y), c)

    def ptransp(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature
    ) -> Float[Array, "dim"]:
        """Parallel transport tangent vector v from point x to point y."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Parallel transport tangent vector v from origin to point y."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    def tangent_inner(
        self, u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature
    ) -> Float[Array, ""]:
        """Compute inner product of tangent vectors u and v at point x."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
        """Compute norm of tangent vector v at point x."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Convert Euclidean gradient to Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
        """Project vector v onto tangent space at point x."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim"], c: Curvature, atol: float = 1e-5) -> Array:
        """Check if point x lies in Poincaré ball."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: Curvature) -> Array:
        """Check if vector v lies in tangent space at point x."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c)

    def conformal_factor(self, x: Float[Array, "... dim"], c: Curvature) -> Float[Array, "... 1"]:
        """Numerically stable conformal factor lambda(x) = 2 / (1 - c||x||^2).

        Batch-compatible version that handles arbitrary leading dimensions.
        """
        return _conformal_factor_batch(self._cast(x), c)

    def embed_spatial_0(self, v_spatial: Float[Array, "... n"]) -> Float[Array, "... n"]:
        """Identity embedding (the ball has no time coordinate). Kept for API parity."""
        return _embed_spatial_0(self._cast(v_spatial))

    def compute_mlr_pp(
        self,
        x: Float[Array, "batch in_dim"],
        z: Float[Array, "out_dim in_dim"],
        r: Float[Array, "out_dim 1"],
        c: Curvature,
        clamping_factor: float,
        smoothing_factor: float,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """Compute HNN++ multinomial linear regression on the Poincare ball."""
        return _compute_mlr_pp(self._cast(x), self._cast(z), self._cast(r), c, clamping_factor, smoothing_factor, min_enorm)

    def beta_concat(self, points: Float[Array, "M n_i"], c: Curvature) -> Float[Array, "n"]:
        """Beta-concatenation of M equal-dimensional Poincaré ball points."""
        return _beta_concat(self._cast(points), c)

    def busemann(self, x: Float[Array, "dim"], v: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
        """Closed-form Poincaré Busemann function ``B^v(x) = (1/√c)·log(‖v - √c·x‖²/(1 - c‖x‖²))``.

        Point-to-horosphere coordinate (Chen et al. 2026, Eq. 3). ``v`` must be a *unit*
        direction — it is **not** normalized internally. Single point ``(d,)`` → scalar; use
        :func:`jax.vmap` for batching and over a direction set. ``B^v(origin) = 0``, and it
        matches :meth:`Hyperboloid.busemann` under the Poincaré↔Hyperboloid isometry.

        See Also
        --------
        For an *asymmetric* quasimetric energy, compose this Busemann coordinate with an
        external Euclidean quasimetric (e.g. IQE/MRN). Do **not** reach for
        :meth:`apollonian_dist` expecting asymmetry — it is a coboundary (symmetrizes to
        ``√c·dist``) and cannot deliver circulation.
        """
        return _busemann(self._cast(x), self._cast(v), c)
