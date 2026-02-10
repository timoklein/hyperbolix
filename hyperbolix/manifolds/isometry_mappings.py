"""Isometry mappings between hyperbolic manifold models.

This module implements distance-preserving transformations (isometries) between
different models of hyperbolic geometry. All functions operate on single points
and use JAX's vmap for batch operations.

Supported Models:
    - Hyperboloid model (Lorentz model): Points in R^(d+1) satisfying ⟨x,x⟩_L = -1/c
    - Poincaré ball model: Points in R^d with ||y||² < 1/c

The mappings are implemented via stereographic projection from the hyperboloid
to the Poincaré ball, projecting through the point [-1, 0, ..., 0].

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
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

# Default numerical parameter for safe division
MIN_DENOM = 1e-15


def hyperboloid_to_poincare(
    x: Float[Array, "dim_plus_1"],
    c: Float[Array, ""] | float,
) -> Float[Array, "dim"]:
    """Convert hyperboloid point to Poincaré ball via stereographic projection.

    Projects the hyperboloid point onto the hyperplane t = 0 by intersecting
    with a line through [-1, 0, ..., 0]. This implements the canonical isometry
    between the two models.

    Formula:
        y_i = x_i / (1 + t)
        where x = [t, x_1, ..., x_n] on hyperboloid

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
        >>> jnp.allclose(y, jnp.zeros(2))
        True

    References:
        Wikipedia: Hyperboloid model - Relation to other models
    """
    t = x[0]  # Temporal component
    x_spatial = x[1:]  # Spatial components (x_1, ..., x_n)

    # Stereographic projection: y_i = x_i / (1 + t)
    denominator = jnp.maximum(1.0 + t, MIN_DENOM)
    return x_spatial / denominator


def poincare_to_hyperboloid(
    y: Float[Array, "dim"],
    c: Float[Array, ""] | float,
) -> Float[Array, "dim_plus_1"]:
    """Convert Poincaré ball point to hyperboloid via inverse stereographic projection.

    Inverts the stereographic projection to map points from the Poincaré ball
    back to the hyperboloid. This implements the canonical isometry between
    the two models.

    Formula:
        (t, x_i) = ((1 + Σy_i²), 2y_i) / ((1 - Σy_i²) * √c)
        where y = [y_1, ..., y_n] in Poincaré ball

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
        >>> jnp.allclose(x, jnp.array([1.0, 0.0, 0.0]))
        True

    References:
        Wikipedia: Hyperboloid model - Relation to other models
    """
    y_sqnorm = jnp.dot(y, y)
    sqrt_c = jnp.sqrt(c)

    # Inverse stereographic projection with curvature scaling
    numerator = 1.0 + y_sqnorm
    denominator = jnp.maximum((1.0 - y_sqnorm) * sqrt_c, MIN_DENOM)

    t = numerator / denominator
    x_spatial = 2.0 * y / denominator

    # Concatenate temporal and spatial components: [t, x_1, ..., x_n]
    return jnp.concatenate([jnp.array([t]), x_spatial])
