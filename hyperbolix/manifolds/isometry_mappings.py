"""Isometry mappings between hyperbolic manifold models.

This module implements distance-preserving transformations (isometries) between
different models of hyperbolic geometry. All functions operate on single points
and use JAX's vmap for batch operations.

Supported Models (curvature ``c > 0``, sectional curvature ``-c``):
    - Hyperboloid model (Lorentz model): Points in R^(d+1) satisfying ⟨x,x⟩_L = -1/c
    - Poincaré ball model: Points in R^d with ||y||² < 1/c
    - Proper Velocity (PV) model: Unconstrained points in R^d (Chen et al. 2026)

Provided maps (all exact, distance-preserving, mutually consistent):
    - Poincaré ↔ Hyperboloid: ``poincare_to_hyperboloid`` / ``hyperboloid_to_poincare``
      (curvature-aware stereographic projection through [-1/√c, 0, ..., 0]).
    - Poincaré ↔ PV: ``poincare_to_pv`` / ``pv_to_poincare``
      (the gyro-isomorphism of PVNN Eq. 4, an isometry by their Thm 4.2).
    - Hyperboloid ↔ PV: ``hyperboloid_to_pv`` / ``pv_to_hyperboloid``
      (direct map — PV coordinates are the space-like part of the 4-velocity).

JIT Compilation & Batching
---------------------------
All functions work with single points and return single points.
Use jax.vmap for batch operations:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.manifolds import isometry_mappings
    >>>
    >>> # Single point conversion
    >>> x_hyp = jnp.array([1.0, 0.1, 0.2])  # Hyperboloid point
    >>> y_poinc = isometry_mappings.hyperboloid_to_poincare(x_hyp, c=1.0)
    >>>
    >>> # Batch conversion with vmap
    >>> x_batch = jnp.array([[1.0, 0.1, 0.2], [1.1, 0.15, 0.25]])
    >>> convert_batch = jax.vmap(isometry_mappings.hyperboloid_to_poincare, in_axes=(0, None))
    >>> y_batch = convert_batch(x_batch, 1.0)

References:
    Wikipedia: Hyperboloid model
    https://en.wikipedia.org/wiki/Hyperboloid_model#Relation_to_other_models
    Chen et al. "Proper Velocity Neural Networks." ICLR 2026 (PV ↔ Poincaré, Eq. 4).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import MIN_NORM, floor_at, safe_hypot, safe_norm
from ..utils.precision import MATMUL_PRECISION
from .protocol import ScalarCurvature


def hyperboloid_to_poincare(
    x: Float[Array, "dim_plus_1"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert hyperboloid point to Poincaré ball via stereographic projection.

    Projects the hyperboloid point onto the hyperplane t = 0 by intersecting
    with a line through [-1/√c, 0, ..., 0]. This implements the canonical
    isometry between the two models (radius-1/√c Poincaré ball).

    Formula:
        y_i = x_i / (√c·t + 1)
        where x = [t, x_1, ..., x_n] on hyperboloid (t = x₀ ≥ 1/√c)

    Args:
        x: Point on hyperboloid, shape (dim+1,). Should satisfy ⟨x,x⟩_L = -1/c.
        c: Curvature (positive)

    Returns:
        Point in Poincaré ball, shape (dim,). Satisfies ||y||² < 1/c.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Convert hyperboloid origin to Poincaré origin
        >>> x_origin = jnp.array([1.0, 0.0, 0.0])  # c=1.0 origin
        >>> y = isometry_mappings.hyperboloid_to_poincare(x_origin, c=1.0)
        >>> bool(jnp.allclose(y, jnp.zeros(2)))
        True

    References:
        Wikipedia: Hyperboloid model - Relation to other models
    """
    sqrt_c = jnp.sqrt(c)
    t = x[0]  # Temporal component
    x_spatial = x[1:]  # Spatial components (x_1, ..., x_n)

    # Curvature-aware stereographic projection: y_i = x_i / (√c·t + 1).
    # Since t ≥ 1/√c on the hyperboloid, √c·t ≥ 1 and the denominator ≥ 2 — stable.
    denominator = floor_at(sqrt_c * t + 1.0, MIN_NORM)
    return x_spatial / denominator


def poincare_to_hyperboloid(
    y: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim_plus_1"]:
    """Convert Poincaré ball point to hyperboloid via inverse stereographic projection.

    Inverts the stereographic projection to map points from the Poincaré ball
    back to the hyperboloid. This implements the canonical isometry between
    the two models (radius-1/√c Poincaré ball).

    Formula:
        t   = (1 + c·||y||²) / ((1 - c·||y||²)·√c)
        x_i = 2·y_i / (1 - c·||y||²)
        where y = [y_1, ..., y_n] in Poincaré ball (||y||² < 1/c)

    Args:
        y: Point in Poincaré ball, shape (dim,). Should satisfy ||y||² < 1/c.
        c: Curvature (positive)

    Returns:
        Point on hyperboloid, shape (dim+1,). Satisfies ⟨x,x⟩_L = -1/c.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Convert Poincaré origin to hyperboloid origin
        >>> y_origin = jnp.array([0.0, 0.0])
        >>> x = isometry_mappings.poincare_to_hyperboloid(y_origin, c=1.0)
        >>> bool(jnp.allclose(x, jnp.array([1.0, 0.0, 0.0])))
        True

    References:
        Wikipedia: Hyperboloid model - Relation to other models
    """
    y_sqnorm = jnp.dot(y, y, precision=MATMUL_PRECISION)
    sqrt_c = jnp.sqrt(c)

    # Curvature-aware inverse stereographic projection. The spatial part scales
    # by the Poincaré conformal factor 1/(1 - c·||y||²); only the time component
    # carries the extra 1/√c, so the two denominators differ.
    one_minus = floor_at(1.0 - c * y_sqnorm, MIN_NORM)

    t = (1.0 + c * y_sqnorm) / (one_minus * sqrt_c)
    x_spatial = 2.0 * y / one_minus

    # Concatenate temporal and spatial components: [t, x_1, ..., x_n]
    return jnp.concatenate([t[None], x_spatial])


def pv_to_poincare(
    x: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a Proper Velocity point to the Poincaré ball.

    Implements the gyro-isomorphism π_{PV→P} of the PVNN paper (Chen et al.
    2026, Eq. 4), proven to be a Riemannian isometry (their Thm 4.2). With
    hyperbolix's c > 0 convention (K = -c):

    Formula:
        y = x / (1 + √(1 + c·||x||²))

    This is the numerically stable form of ``y = β_x/(1 + β_x)·x`` with the PV
    beta factor ``β_x = 1/√(1 + c·||x||²)``: dividing numerator and denominator
    by β_x turns it into ``x / (1 + 1/β_x)``. The denominator is ≥ 2, so the map
    never blows up — every finite PV point lands strictly inside the radius-1/√c
    ball.

    Args:
        x: Point in PV space (unconstrained R^n), shape (dim,).
        c: Curvature (positive).

    Returns:
        Point in the Poincaré ball, shape (dim,). Satisfies ||y||² < 1/c.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # PV origin (0) maps to Poincaré origin (0)
        >>> y = isometry_mappings.pv_to_poincare(jnp.zeros(2), c=1.0)
        >>> bool(jnp.allclose(y, jnp.zeros(2)))
        True

    References:
        Chen et al. "Proper Velocity Neural Networks." ICLR 2026, Eq. 4.
    """
    # √(1 + c·||x||²) as a two-leg hypot: `dot(x, x)` overflows float32 once ||x|| passes
    # 1.8e19/√c, and `x / (1 + inf)` then maps a far-out PV point to the *origin* instead of near
    # the ball boundary. `safe_hypot` never materialises the square.
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=x.dtype))
    beta_inv = safe_hypot(jnp.asarray(1.0, dtype=x.dtype), sqrt_c * safe_norm(x))  # 1/β_x
    return x / (1.0 + beta_inv)


def poincare_to_pv(
    y: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a Poincaré ball point to Proper Velocity space.

    Inverse of :func:`pv_to_poincare` — the map π_{P→PV} of the PVNN paper
    (Chen et al. 2026, Eq. 4). With hyperbolix's c > 0 convention (K = -c):

    Formula:
        x = 2·y / (1 - c·||y||²) = λ(y)·y

    where λ(y) = 2/(1 - c·||y||²) is the Poincaré conformal factor. As y
    approaches the ball boundary (||y||² → 1/c) the image grows without bound —
    expected, since PV is the *unconstrained* R^n model. The denominator is
    guarded by ``MIN_NORM`` to avoid division by zero at the boundary.

    Args:
        y: Point in the Poincaré ball, shape (dim,). Should satisfy ||y||² < 1/c.
        c: Curvature (positive).

    Returns:
        Point in PV space (unconstrained R^n), shape (dim,).

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Poincaré origin (0) maps to PV origin (0)
        >>> x = isometry_mappings.poincare_to_pv(jnp.zeros(2), c=1.0)
        >>> bool(jnp.allclose(x, jnp.zeros(2)))
        True

    References:
        Chen et al. "Proper Velocity Neural Networks." ICLR 2026, Eq. 4.
    """
    denominator = floor_at(1.0 - c * jnp.dot(y, y, precision=MATMUL_PRECISION), MIN_NORM)
    return 2.0 * y / denominator


def pv_to_hyperboloid(
    x: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim_plus_1"]:
    """Convert a Proper Velocity point to the hyperboloid model.

    Uses the direct relation "proper velocity is the spatial part of the
    (dimensionless) 4-velocity": the PV coordinates are exactly the space-like
    hyperboloid components, and the time component is reconstructed from the
    Lorentz constraint ⟨z,z⟩_L = -1/c.

    Formula:
        z = [√(1/c + ||x||²), x_1, ..., x_n]

    This is an exact isometry (it equals
    ``poincare_to_hyperboloid(pv_to_poincare(x, c), c)``) but avoids the
    near-boundary 1/(1 - c·||y||²) blow-up of the composed route — it is just a
    concatenation. The time component is ≥ 1/√c > 0, so the result is always a
    valid, numerically stable hyperboloid point.

    Args:
        x: Point in PV space (unconstrained R^n), shape (dim,).
        c: Curvature (positive).

    Returns:
        Point on the hyperboloid, shape (dim+1,). Satisfies ⟨z,z⟩_L = -1/c, z₀ > 0.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # PV origin maps to the hyperboloid origin [1/√c, 0, ...]
        >>> z = isometry_mappings.pv_to_hyperboloid(jnp.zeros(2), c=1.0)
        >>> bool(jnp.allclose(z, jnp.array([1.0, 0.0, 0.0])))
        True

    References:
        Chen et al. "Proper Velocity Neural Networks." ICLR 2026.
    """
    # √(1/c + ||x||²) as a two-leg hypot — the same shape as `Hyperboloid._proj`, and the same
    # fix: `dot(x, x)` overflows float32 past ||x|| = 1.8e19, returning an infinite time slot for
    # a point whose time slot is perfectly representable.
    time = safe_hypot(safe_norm(x), jnp.asarray(1.0, dtype=x.dtype) / jnp.sqrt(jnp.asarray(c, dtype=x.dtype)))
    return jnp.concatenate([time[None], x])


def hyperboloid_to_pv(
    x: Float[Array, "dim_plus_1"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a hyperboloid point to Proper Velocity space.

    Inverse of :func:`pv_to_hyperboloid`: the PV coordinates are exactly the
    space-like components of the hyperboloid point, so the map simply drops the
    time component.

    Formula:
        x = [z_1, ..., z_n]   (drop the time component z₀)

    The curvature ``c`` is unused (the relation is curvature-independent) but is
    kept in the signature for API symmetry with the other mappings.

    Args:
        x: Point on the hyperboloid, shape (dim+1,). Should satisfy ⟨x,x⟩_L = -1/c.
        c: Curvature (positive). Unused.

    Returns:
        Point in PV space (unconstrained R^n), shape (dim,).

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Hyperboloid origin [1/√c, 0, ...] maps to the PV origin (0)
        >>> x = isometry_mappings.hyperboloid_to_pv(jnp.array([1.0, 0.0, 0.0]), c=1.0)
        >>> bool(jnp.allclose(x, jnp.zeros(2)))
        True

    References:
        Chen et al. "Proper Velocity Neural Networks." ICLR 2026.
    """
    del c  # curvature-independent: PV coords are the hyperboloid spatial part
    return x[1:]
