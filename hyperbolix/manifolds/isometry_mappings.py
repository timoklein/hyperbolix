"""Isometry mappings between hyperbolic manifold models.

This module implements distance-preserving transformations (isometries) between
different models of hyperbolic geometry. All functions operate on single points
and use JAX's vmap for batch operations.

Supported Models (curvature ``c > 0``, sectional curvature ``-c``):
    - Hyperboloid model (Lorentz model): Points in R^(d+1) satisfying ⟨x,x⟩_L = -1/c
    - Poincaré ball model: Points in R^d with ||y||² < 1/c
    - Proper Velocity (PV) model: Unconstrained points in R^d (Chen et al. 2026)
    - Beltrami-Klein model: Points in R^d with ||k||² < 1/c (geodesics are straight chords)
    - Poincaré half-space model: Points in R^d with last coordinate x_n > 0 (origin e_n/√c)

Provided maps (all exact, distance-preserving, mutually consistent):
    - Poincaré ↔ Hyperboloid: ``poincare_to_hyperboloid`` / ``hyperboloid_to_poincare``
      (curvature-aware stereographic projection through [-1/√c, 0, ..., 0]).
    - Poincaré ↔ PV: ``poincare_to_pv`` / ``pv_to_poincare``
      (the gyro-isomorphism of PVNN Eq. 4, an isometry by their Thm 4.2).
    - Hyperboloid ↔ PV: ``hyperboloid_to_pv`` / ``pv_to_hyperboloid``
      (direct map — PV coordinates are the space-like part of the 4-velocity;
      the projection Π of Ferreira 2017, Sec. 2, built on the PV gyrogroup of
      Ungar 2005, Def. 3.40).
    - Klein ↔ Poincaré: ``klein_to_poincare`` / ``poincare_to_klein``
      (the Einstein half / Möbius double maps, k = 2 ⊗ p).
    - Klein ↔ Hyperboloid: ``klein_to_hyperboloid`` / ``hyperboloid_to_klein``
      (central projection from the ambient origin onto the plane x₀ = 1/√c).
    - Klein ↔ PV: ``klein_to_pv`` / ``pv_to_klein``
      (PV = Einstein velocity scaled by its Lorentz factor, x = gamma_k·k).
    - Half-space ↔ Poincaré: ``halfspace_to_poincare`` / ``poincare_to_halfspace``
      (the Cayley transform, origin e_n/√c ↦ 0 with differential ½·I).
    - Half-space ↔ Hyperboloid: ``halfspace_to_hyperboloid`` / ``hyperboloid_to_halfspace``.
    - Half-space ↔ Klein: ``halfspace_to_klein`` / ``klein_to_halfspace``.
    - Half-space ↔ PV: ``halfspace_to_pv`` / ``pv_to_halfspace``.

Klein chart precision: with scaled radius ``a = √c·d(0, ·)`` a Klein point has
``√c·||k|| = tanh(a)`` where a Poincaré point has ``tanh(a/2)``, so ``1 - c·||k||²``
reaches the ``_boundary_floor`` value at half the Poincaré radius — scaled radius
≈ 6.3 in float32 and ≈ 13.9 in float64 at c = 1 (Poincaré: 12.6 / 27.7). The maps
into Klein (``poincare_to_klein``, ``hyperboloid_to_klein``, ``pv_to_klein``) do not
project. A point farther out than that lands between the ``proj`` margin and the
boundary (float32, c = 1: ``||k|| = 0.9999983`` at a = 7 against the margin
0.99999356), and from a ≈ 10 on ``||k||`` rounds to exactly ``1/√c`` (at a = 12 for all
three maps). Klein operations floor ``1 - c·||k||²`` at the
``_boundary_floor`` value, so all such points read as a ≈ 6.32 (float32, c = 1). Call
``Klein.proj`` after mapping far points into Klein.

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
    Ferreira. "Harmonic Analysis on the Proper Velocity Gyrogroup." Banach J.
    Math. Anal. 11(1), 21-49, 2017 (PV ↔ Hyperboloid projection Π, Sec. 2).
    Ungar. "Analytic Hyperbolic Geometry: Mathematical Foundations and
    Applications." World Scientific, 2005 (PV gyrogroup, Def. 3.40; the
    Einstein ↔ Möbius gyrovector-space isomorphism behind Klein ↔ Poincaré).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import MIN_NORM, floor_at, safe_hypot_norm
from ..utils.precision import MATMUL_PRECISION
from ._gyrovector_core import _boundary_floor
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
    # `_boundary_floor` is the same dtype-aware floor `_conformal_factor` puts on this quantity —
    # the analytic minimum over projected points, below which any value is rounding noise.
    # `MIN_NORM = 1e-15` sits below that minimum in both dtypes, so it never bit.
    one_minus = floor_at(1.0 - c * y_sqnorm, _boundary_floor(y, c))

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
    # √(1 + c·||x||²) via `safe_hypot_norm`: `dot(x, x)` overflows float32 once ||x|| passes
    # 1.8e19/√c, and `x / (1 + inf)` then maps a far-out PV point to the *origin* instead of near
    # the ball boundary. `safe_hypot_norm` never materialises the square, and computes
    # `sqrt(1 + c||x||^2)` directly from the scaled input instead of rounding `safe_norm(x)` and
    # squaring it again (kept two-pass, unchanged from 1.2.0). Same form and the same measurement
    # as `ProperVelocity._beta_inv`; see there.
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=x.dtype))
    beta_inv = safe_hypot_norm(sqrt_c * x, jnp.asarray(1.0, dtype=x.dtype))  # 1/β_x
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
    floored at ``_boundary_floor(y, c)`` — the smallest value it can take on a
    projected ball point — to avoid division by zero at the boundary.

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
    # `_boundary_floor`: the dtype-aware floor `_conformal_factor` puts on `1 - c‖y‖²`; see
    # :func:`poincare_to_hyperboloid`.
    denominator = floor_at(1.0 - c * jnp.dot(y, y, precision=MATMUL_PRECISION), _boundary_floor(y, c))
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
        Ferreira. "Harmonic Analysis on the Proper Velocity Gyrogroup." Banach
        J. Math. Anal. 11(1), 21-49, 2017 — Sec. 2, the projection
        Π(x, √(t²+‖x‖²)) = x between the hyperboloid of radius t = 1/√c and
        PV space (t → 1/√c here).
        Ungar. "Analytic Hyperbolic Geometry: Mathematical Foundations and
        Applications." World Scientific, 2005 — Def. 3.40, the PV gyrogroup.
    """
    # √(1/c + ||x||²) via `safe_hypot_norm` — the same shape as `Hyperboloid._proj`, and the same
    # fix: `dot(x, x)` overflows float32 past ||x|| = 1.8e19, returning an infinite time slot for
    # a point whose time slot is perfectly representable. `safe_hypot_norm`, not
    # `safe_hypot(safe_norm(x), .)`: it keeps `sum(x**2)` intact instead of rounding the norm and
    # squaring it again, which is what the hyperboloid constraint check cancels against.
    time = safe_hypot_norm(x, jnp.asarray(1.0, dtype=x.dtype) / jnp.sqrt(jnp.asarray(c, dtype=x.dtype)))
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
        Ferreira. "Harmonic Analysis on the Proper Velocity Gyrogroup." Banach
        J. Math. Anal. 11(1), 21-49, 2017 — Sec. 2, the projection
        Π(x, √(t²+‖x‖²)) = x between the hyperboloid of radius t = 1/√c and
        PV space (t → 1/√c here).
        Ungar. "Analytic Hyperbolic Geometry: Mathematical Foundations and
        Applications." World Scientific, 2005 — Def. 3.40, the PV gyrogroup.
    """
    del c  # curvature-independent: PV coords are the hyperboloid spatial part
    return x[1:]


def _klein_gap(k: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """``g_k = 1 - c·||k||²`` floored at ``_boundary_floor(k, c)``, the inverse squared Lorentz factor 1/gamma_k².

    Same floor, and for the same reason, as the ``1 - c·||y||²`` of :func:`poincare_to_hyperboloid`:
    it is the analytic minimum on a projected ball point, so anything below it is rounding noise.
    On a Klein point ``g_k = sech²(a)`` at scaled radius ``a = √c·d(0, k)``, so the floor is
    reached at ``a ≈ 6.3`` (float32) / ``13.9`` (float64) — half the Poincaré chart's radius,
    whose gap is ``sech²(a/2)``. The cancellation in ``1 - c·||k||²`` costs a relative error of
    ``eps·cosh²(a)`` on ``g_k``, the Klein chart's own representation floor.
    """
    k_sqnorm = jnp.dot(k, k, precision=MATMUL_PRECISION)
    return floor_at(1.0 - c * k_sqnorm, _boundary_floor(k, c))


def klein_to_poincare(
    k: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a Beltrami-Klein point to the Poincaré ball.

    The Einstein half map ``p = (1/2) ⊗_E k``: with the Einstein Lorentz factor
    ``gamma_k = 1/√(1 - c·||k||²)``,

    Formula:
        p = gamma_k/(1 + gamma_k)·k = k / (1 + √(1 - c·||k||²))

    Both models use the radius-1/√c ball (curvature -c). The denominator is in
    [1, 2], so the map is well-conditioned; ``1 - c·||k||²`` is floored at
    ``_boundary_floor(k, c)`` (see :func:`_klein_gap`). A Klein point has
    ``√c·||k|| = tanh(a)`` at scaled radius ``a = √c·d(0, k)`` where a Poincaré
    point has ``tanh(a/2)``, so the Klein chart runs out of float precision at
    half the Poincaré radius (``a ≈ 6.3`` float32, ``13.9`` float64).

    Args:
        k: Point in the Klein ball, shape (dim,). Should satisfy ||k||² < 1/c.
        c: Curvature (positive).

    Returns:
        Point in the Poincaré ball, shape (dim,). Satisfies ||p||² < 1/c.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Klein origin maps to Poincaré origin
        >>> p = isometry_mappings.klein_to_poincare(jnp.zeros(2), c=1.0)
        >>> bool(jnp.allclose(p, jnp.zeros(2)))
        True

    References:
        Ungar. "Analytic Hyperbolic Geometry: Mathematical Foundations and
        Applications." World Scientific, 2005 — Einstein ↔ Möbius isomorphism.
    """
    c = jnp.asarray(c, dtype=k.dtype)
    # The gap is floored strictly positive, so the plain square root is safe and its derivative finite.
    return k / (1.0 + jnp.sqrt(_klein_gap(k, c)))


def poincare_to_klein(
    p: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a Poincaré ball point to the Beltrami-Klein model.

    Inverse of :func:`klein_to_poincare`: the Möbius doubling ``k = 2 ⊗_M p``.

    Formula:
        k = 2·p / (1 + c·||p||²)

    The denominator is in [1, 2], so no floor is needed. The result is not
    projected. Since ``√c·||k|| = tanh(a)`` with ``a = √c·d(0, p)``, a Poincaré point
    beyond scaled radius ≈ 6.3 (float32) / 13.9 (float64) at c = 1 lands past the
    ``Klein.proj`` margin, and Klein operations read it as radius ≈ 6.3 / 13.9 (see
    the module docstring). Call ``Klein.proj`` on the result for such points.

    Args:
        p: Point in the Poincaré ball, shape (dim,). Should satisfy ||p||² < 1/c.
        c: Curvature (positive).

    Returns:
        Point in the closed Klein ball, shape (dim,). In float32, ``||k||`` can round
        to ``1/√c`` beyond a ≈ 10; call ``Klein.proj`` before using such points.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Poincaré origin maps to Klein origin
        >>> k = isometry_mappings.poincare_to_klein(jnp.zeros(2), c=1.0)
        >>> bool(jnp.allclose(k, jnp.zeros(2)))
        True

    References:
        Ungar. "Analytic Hyperbolic Geometry: Mathematical Foundations and
        Applications." World Scientific, 2005 — Einstein ↔ Möbius isomorphism.
    """
    c = jnp.asarray(c, dtype=p.dtype)
    return 2.0 * p / (1.0 + c * jnp.dot(p, p, precision=MATMUL_PRECISION))


def klein_to_hyperboloid(
    k: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim_plus_1"]:
    """Convert a Beltrami-Klein point to the hyperboloid model.

    Inverse of the central projection from the ambient origin onto the plane
    ``x₀ = 1/√c`` (scaled by √c so the Klein ball has radius 1/√c).

    Formula:
        x_s = k / √(1 - c·||k||²) = gamma_k·k
        x₀  = 1 / (√c·√(1 - c·||k||²)) = √(1/c + ||x_s||²)

    The spatial part is :func:`klein_to_pv`; the time component is rebuilt from it
    with ``safe_hypot_norm`` exactly as :func:`pv_to_hyperboloid` does, so the
    result satisfies ``⟨x,x⟩_L = -1/c`` to rounding even where ``1 - c·||k||²`` is
    floored. The two time formulas agree in exact arithmetic.

    Args:
        k: Point in the Klein ball, shape (dim,). Should satisfy ||k||² < 1/c.
        c: Curvature (positive).

    Returns:
        Point on the hyperboloid, shape (dim+1,). Satisfies ⟨x,x⟩_L = -1/c, x₀ > 0.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Klein origin maps to the hyperboloid origin [1/√c, 0, ...]
        >>> x = isometry_mappings.klein_to_hyperboloid(jnp.zeros(2), c=1.0)
        >>> bool(jnp.allclose(x, jnp.array([1.0, 0.0, 0.0])))
        True

    References:
        Wikipedia: Hyperboloid model - Relation to other models
    """
    return pv_to_hyperboloid(klein_to_pv(k, c), c)


def hyperboloid_to_klein(
    x: Float[Array, "dim_plus_1"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a hyperboloid point to the Beltrami-Klein model.

    Central projection from the ambient origin onto the plane ``x₀ = 1/√c``.

    Formula:
        k = x_s / (√c·x₀)
        where x = [x₀, x_s] on the hyperboloid (x₀ ≥ 1/√c)

    ``hyperboloid_core.lorentz_scale`` writes the same map as ``φ_K(x) = x_s/x₀``:
    that is the unit-ball Klein coordinate ``√c·k``. Since ``√c·||k|| =
    ||x_s||/x₀ = tanh(a)`` at scaled radius ``a = √c·d(0, x)``, points beyond
    ``a ≈ 6.3`` (float32) / ``13.9`` (float64) at c = 1 land past the ``Klein.proj``
    margin (the result is not projected), and Klein operations read them as radius
    ≈ 6.3 / 13.9 (see the module docstring). Call ``Klein.proj`` on the result for
    such points.

    Args:
        x: Point on the hyperboloid, shape (dim+1,). Should satisfy ⟨x,x⟩_L = -1/c.
        c: Curvature (positive).

    Returns:
        Point in the closed Klein ball, shape (dim,). In float32, ``||k||`` can round
        to ``1/√c`` beyond a ≈ 10; call ``Klein.proj`` before using such points.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Hyperboloid origin maps to the Klein origin
        >>> k = isometry_mappings.hyperboloid_to_klein(jnp.array([1.0, 0.0, 0.0]), c=1.0)
        >>> bool(jnp.allclose(k, jnp.zeros(2)))
        True

    References:
        Wikipedia: Hyperboloid model - Relation to other models
    """
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=x.dtype))
    # √c·x₀ ≥ 1 on the hyperboloid; the floor only guards an off-manifold input, as in
    # :func:`hyperboloid_to_poincare`.
    return x[1:] / floor_at(sqrt_c * x[0], MIN_NORM)


def klein_to_pv(
    k: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a Beltrami-Klein point to Proper Velocity space.

    A Klein point is an Einstein (coordinate) velocity; its proper velocity is
    the velocity times its Lorentz factor.

    Formula:
        x = gamma_k·k = k / √(1 - c·||k||²)

    The image grows without bound as k approaches the boundary (PV is the
    unconstrained model); ``1 - c·||k||²`` is floored at ``_boundary_floor(k, c)``
    (see :func:`_klein_gap`), reached at scaled radius ``a ≈ 6.3`` (float32) /
    ``13.9`` (float64).

    Args:
        k: Point in the Klein ball, shape (dim,). Should satisfy ||k||² < 1/c.
        c: Curvature (positive).

    Returns:
        Point in PV space (unconstrained R^n), shape (dim,).

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Klein origin maps to the PV origin
        >>> x = isometry_mappings.klein_to_pv(jnp.zeros(2), c=1.0)
        >>> bool(jnp.allclose(x, jnp.zeros(2)))
        True

    References:
        Ungar. "Analytic Hyperbolic Geometry: Mathematical Foundations and
        Applications." World Scientific, 2005 — Einstein and PV gyrogroups.
    """
    c = jnp.asarray(c, dtype=k.dtype)
    # The gap is floored strictly positive, so the plain square root is safe and its derivative finite.
    return k / jnp.sqrt(_klein_gap(k, c))


def pv_to_klein(
    x: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a Proper Velocity point to the Beltrami-Klein model.

    Inverse of :func:`klein_to_pv`: the Einstein velocity is the proper velocity
    times the PV beta factor ``β_x = 1/√(1 + c·||x||²)``.

    Formula:
        k = β_x·x = x / √(1 + c·||x||²)

    Every finite PV point lands inside the Klein ball in exact arithmetic. The
    result is not projected: in floating point a point beyond scaled radius ≈ 6.3
    (float32) / 13.9 (float64) at c = 1 lands past the ``Klein.proj`` margin, and
    Klein operations read it as radius ≈ 6.3 / 13.9 (see the module docstring).
    Call ``Klein.proj`` on the result for such points.

    Args:
        x: Point in PV space (unconstrained R^n), shape (dim,).
        c: Curvature (positive).

    Returns:
        Point in the closed Klein ball, shape (dim,). In float32, ``||k||`` can round
        to ``1/√c`` beyond a ≈ 10; call ``Klein.proj`` before using such points.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # PV origin maps to the Klein origin
        >>> k = isometry_mappings.pv_to_klein(jnp.zeros(2), c=1.0)
        >>> bool(jnp.allclose(k, jnp.zeros(2)))
        True

    References:
        Ungar. "Analytic Hyperbolic Geometry: Mathematical Foundations and
        Applications." World Scientific, 2005 — Einstein and PV gyrogroups.
    """
    # √(1 + c·||x||²) via `safe_hypot_norm`, overflow-free past ||x|| = 1.8e19/√c; same form as
    # :func:`pv_to_poincare`.
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=x.dtype))
    beta_inv = safe_hypot_norm(sqrt_c * x, jnp.asarray(1.0, dtype=x.dtype))  # 1/β_x
    return x / beta_inv


# ---------------------------------------------------------------------------
# Half-space ↔ Poincaré
# ---------------------------------------------------------------------------


def halfspace_to_poincare(
    x: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a Poincaré half-space point to the Poincaré ball (the Cayley transform).

    With ``x = (x_s, x_n)``, ``x_n > 0`` (last coordinate), scaled coordinates
    ``s = √c·x`` and

    Formula:
        q   = c·||x_s||² + (√c·x_n - 1)(√c·x_n + 1)       (= c·||x||² - 1)
        den = ||s_s||² + (1 + s_n)²
        p   = (2·s_s, q) / (√c·den)

    The half-space origin ``e_n/√c`` maps to the ball origin with differential
    exactly ``½·I`` (no reflection); the positive ``x_n`` axis maps onto the
    ``e_n`` diameter, ``x_n → ∞`` to the north pole ``e_n/√c`` and ``x_n → 0`` to
    the boundary sphere.

    Numerics: ``den`` is a sum of non-negative terms, at least 1 on the half-space,
    so no floor is needed. ``q`` is written as a product so it does not cancel
    near the sphere ``c·||x||² = 1`` (the ball's equatorial plane ``p_n = 0``). The
    result is not projected: past the ball chart's ceiling (scaled radius
    ``a ≈ 12.6`` float32 / ``27.7`` float64) ``||p||`` rounds to ``1/√c`` or just
    beyond, as for every map into the ball — call ``Poincare.proj`` for such
    points. Past float32 ``|s| ≈ 1.8e19`` the squares overflow and the output is
    NaN, not a saturated point.

    Args:
        x: Point in the half-space, shape (dim,). Should satisfy x_n > 0.
        c: Curvature (positive).

    Returns:
        Point in the Poincaré ball, shape (dim,). Satisfies ||p||² < 1/c.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Half-space origin e_n/√c maps to the Poincaré origin
        >>> p = isometry_mappings.halfspace_to_poincare(jnp.array([0.0, 0.0, 2.0]), c=0.25)
        >>> bool(jnp.allclose(p, jnp.zeros(3)))
        True

    References:
        Ratcliffe. "Foundations of Hyperbolic Manifolds." Springer, 3rd ed.
        2019 — Ch. 4, the conformal ball and upper half-space models.
    """
    c = jnp.asarray(c, dtype=x.dtype)
    sqrt_c = jnp.sqrt(c)
    s_s = sqrt_c * x[:-1]
    s_n = sqrt_c * x[-1]
    s_s_sqnorm = jnp.dot(s_s, s_s, precision=MATMUL_PRECISION)

    # q = ||s||² - 1 as a product: the difference (s_n - 1) is exact near s_n = 1, so q keeps
    # its relative precision on the sphere ||s|| = 1 that maps to the equatorial plane p_n = 0.
    q = s_s_sqnorm + (s_n - 1.0) * (s_n + 1.0)
    den = s_s_sqnorm + (1.0 + s_n) ** 2  # ≥ 1 for s_n > 0: no floor
    return jnp.concatenate([2.0 * s_s, q[None]]) / (sqrt_c * den)


def poincare_to_halfspace(
    p: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a Poincaré ball point to the Poincaré half-space (inverse Cayley transform).

    Inverse of :func:`halfspace_to_poincare`. With scaled coordinates ``s = √c·p``,

    Formula:
        g   = 1 - ||s||²                                  (= 1 - c·||p||²)
        den = ||s_s||² + (s_n - 1)²                       (= ||s - e_n||²)
        x   = (2·s_s, g) / (√c·den)

    The ball origin maps to the half-space origin ``e_n/√c``; the north pole
    ``e_n/√c`` is the half-space's point at infinity and every other boundary
    point lands on ``x_n = 0``.

    Numerics: ``g`` is floored at ``_boundary_floor(p, c)`` exactly as
    :func:`poincare_to_hyperboloid` floors ``1 - c·||y||²`` — the analytic minimum
    on a ``Poincare.proj``-projected point, so anything below it is rounding
    noise. The ball chart ends at scaled radius ``a ≈ 12.6`` (float32) / ``27.7``
    (float64) at the ``proj`` margin; a point past it is wherever
    ``Poincare.proj`` put it, and its image is the half-space point at that
    capped radius. ``den`` is not floored: it vanishes only at the north pole,
    a point the half-space cannot hold, so the output there is ``inf``/NaN —
    loud, not a clamped finite point. On a projected point
    ``den ≥ (1 - ||s||)² ≥ eps**1.5`` (the ``proj`` margin squared), so the
    division is finite everywhere the ball is.

    Args:
        p: Point in the Poincaré ball, shape (dim,). Should satisfy ||p||² < 1/c.
        c: Curvature (positive).

    Returns:
        Point in the half-space, shape (dim,). Satisfies x_n > 0.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Poincaré origin maps to the half-space origin e_n/√c
        >>> x = isometry_mappings.poincare_to_halfspace(jnp.zeros(3), c=0.25)
        >>> bool(jnp.allclose(x, jnp.array([0.0, 0.0, 2.0])))
        True

    References:
        Ratcliffe. "Foundations of Hyperbolic Manifolds." Springer, 3rd ed.
        2019 — Ch. 4, the conformal ball and upper half-space models.
    """
    c = jnp.asarray(c, dtype=p.dtype)
    sqrt_c = jnp.sqrt(c)
    s_s = sqrt_c * p[:-1]
    s_n = sqrt_c * p[-1]
    s_s_sqnorm = jnp.dot(s_s, s_s, precision=MATMUL_PRECISION)

    # `_boundary_floor`: the dtype-aware floor `_conformal_factor` puts on `1 - c‖p‖²`; see
    # `poincare_to_hyperboloid`.
    gap = floor_at(1.0 - c * jnp.dot(p, p, precision=MATMUL_PRECISION), _boundary_floor(p, c))
    den = s_s_sqnorm + (s_n - 1.0) ** 2  # zero only at the north pole (x_n = ∞): no floor
    return jnp.concatenate([2.0 * s_s, gap[None]]) / (sqrt_c * den)


# ---------------------------------------------------------------------------
# Half-space ↔ Hyperboloid
# ---------------------------------------------------------------------------


def halfspace_to_hyperboloid(
    x: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim_plus_1"]:
    """Convert a Poincaré half-space point to the hyperboloid model.

    With ``x = (x_s, x_n)``, ``x_n > 0`` (last coordinate), metric ``||dx||²/(c·x_n²)`` and
    origin ``e_n/√c``, the map sends the origin to the hyperboloid origin ``[1/√c, 0, ..., 0]``
    and the vertical axis to the ``(X₀, X_n)`` plane. Let ``u_n = √c·x_n``.

    Formula:
        X_mid = x_s / (√c·x_n)                                    (n-1 entries)
        X_n   = (c·||x||² - 1) / (2c·x_n)
              = ||x_s||²/(2·x_n) + (u_n - 1)(u_n + 1)/(2c·x_n)
        X₀    = (c·||x||² + 1) / (2c·x_n) = √(1/c + ||X_mid||² + X_n²)
        X = [X₀, X_mid, X_n]

    Numerics: the spatial part ``X_s = (X_mid, X_n)`` carries the point, and ``X₀`` is rebuilt
    from it with ``safe_hypot_norm`` exactly as :func:`pv_to_hyperboloid` and
    :func:`klein_to_hyperboloid` do, so ``⟨X,X⟩_L = -1/c`` holds to one rounding of ``X₀``.
    ``X_n`` keeps ``c·x_n² - 1`` in the factored form ``(u_n - 1)(u_n + 1)``: the literal
    ``c·||x||² - 1`` rounds ``c·x_n²`` before subtracting and loses the relative accuracy of
    ``X_n`` near the origin (float32, 1e-3/√c around it, c ∈ {1, 4}: median relative error 3.5e-8
    against 1.6e-5 for the literal; at other ``c`` the rounding of ``√c`` bounds both). Where
    ``c·||x||² ≈ 1`` away from the axis (the hemisphere that maps to ``X_n = 0``) the subtraction
    cancels in every spelling; the absolute error there is ``~eps·X₀``, the hyperboloid's own
    storage floor. ``||x_s||²/(2·x_n)`` carries no factor of
    ``c``, and it overflows float32 only once ``||x_s||`` passes 1.8e19, where ``X_n ≥ ||x_s||``
    is already past the hyperboloid's float32 coordinate ceiling, so the overflow is where the
    hyperboloid itself returns ``inf``. ``x_n`` is not floored: every ``x_n > 0`` is a valid point,
    and ``x_n ≤ 0`` is off the model and returns ``inf``/NaN. The result is not projected.

    Args:
        x: Point in the half-space, shape (dim,). Should satisfy x_n = x[-1] > 0.
        c: Curvature (positive).

    Returns:
        Point on the hyperboloid, shape (dim+1,). Satisfies ⟨X,X⟩_L = -1/c, X₀ > 0.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Half-space origin e_n/√c maps to the hyperboloid origin [1/√c, 0, ...]
        >>> X = isometry_mappings.halfspace_to_hyperboloid(jnp.array([0.0, 2.0]), c=0.25)
        >>> bool(jnp.allclose(X, jnp.array([2.0, 0.0, 0.0])))
        True

    References:
        Cannon, Floyd, Kenyon, Parry. "Hyperbolic Geometry." Flavors of Geometry, MSRI Publ. 31,
        1997 — Sec. 7, the maps between the models at c = 1 (rescaled by 1/√c here).
    """
    c = jnp.asarray(c, dtype=x.dtype)
    sqrt_c = jnp.sqrt(c)
    x_s, x_n = x[:-1], x[-1]
    u_n = sqrt_c * x_n
    x_mid = x_s / u_n
    x_last = jnp.dot(x_s, x_s, precision=MATMUL_PRECISION) / (2.0 * x_n) + (u_n - 1.0) * ((u_n + 1.0) / (2.0 * c * x_n))
    x_spatial = jnp.concatenate([x_mid, x_last[None]])
    # √(1/c + ||X_s||²) in one overflow-free reduction; see :func:`pv_to_hyperboloid`.
    time = safe_hypot_norm(x_spatial, jnp.asarray(1.0, dtype=x.dtype) / sqrt_c)
    return jnp.concatenate([time[None], x_spatial])


def hyperboloid_to_halfspace(
    x: Float[Array, "dim_plus_1"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Convert a hyperboloid point to the Poincaré half-space model.

    Inverse of :func:`halfspace_to_hyperboloid`. With ``X = [X₀, X_mid, X_n]`` and
    ``Δ = X₀ - X_n > 0``:

    Formula:
        x_s = X_mid / (√c·Δ)
        x_n = 1 / (c·Δ)

    Numerics: written literally, ``Δ`` is a difference of two ``O(sinh a)`` numbers whose result
    is ``O(e^{-a})`` for points far up the vertical axis or out along the horosphere through the
    origin (``X_n ≈ X₀``). For ``X_n ≥ 0`` the sheet constraint gives the cancellation-free form

        Δ = (1/c + ||X_mid||²) / (X₀ + X_n),

    and for ``X_n < 0`` the literal ``X₀ - X_n`` is a sum of positive values. This is the spelling
    of ``hyperboloid._busemann_arg`` with ``v = e_n``: the numerator is built from the spatial
    part, and the stored ``X₀`` enters only the denominator (and the ``X_n < 0`` branch). The unused
    denominator is ``X₀ + where(X_n ≥ 0, X_n, 0)``, positive in both branches, so neither branch
    produces a NaN cotangent under reverse mode. ``Δ`` is positive on the sheet, so no floor is
    needed. The result is not projected. Measured in float32 against a 60-digit oracle, c ∈ {0.3,
    1, 4, 10}: median relative error 5e-8 far up the axis (``√c·x_n`` up to 1e6) and ≤ 7e-8 in
    every other family. The literal ``X₀ - X_n`` gives 0.2 far up the axis, first returns ``inf``
    at ``√c·x_n ≈ 6e3``, and returns it for almost every point past 1e4.

    Args:
        x: Point on the hyperboloid, shape (dim+1,). Should satisfy ⟨x,x⟩_L = -1/c.
        c: Curvature (positive).

    Returns:
        Point in the half-space, shape (dim,). Satisfies x_n > 0.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import isometry_mappings
        >>>
        >>> # Hyperboloid origin [1/√c, 0, ...] maps to the half-space origin e_n/√c
        >>> x = isometry_mappings.hyperboloid_to_halfspace(jnp.array([2.0, 0.0, 0.0]), c=0.25)
        >>> bool(jnp.allclose(x, jnp.array([0.0, 2.0])))
        True

    References:
        Cannon, Floyd, Kenyon, Parry. "Hyperbolic Geometry." Flavors of Geometry, MSRI Publ. 31,
        1997 — Sec. 7, the maps between the models at c = 1 (rescaled by 1/√c here).
    """
    c = jnp.asarray(c, dtype=x.dtype)
    sqrt_c = jnp.sqrt(c)
    x0, x_mid, x_last = x[0], x[1:-1], x[-1]
    nonnegative = x_last >= 0
    denominator = x0 + jnp.where(nonnegative, x_last, jnp.zeros_like(x_last))
    rationalized = (1.0 / c + jnp.dot(x_mid, x_mid, precision=MATMUL_PRECISION)) / denominator
    gap = jnp.where(nonnegative, rationalized, x0 - x_last)  # Δ = X₀ - X_n
    return jnp.concatenate([x_mid / (sqrt_c * gap), (1.0 / (c * gap))[None]])


# ---------------------------------------------------------------------------
# Half-space ↔ Klein
# ---------------------------------------------------------------------------

# (halfspace-klein maps go here)


# ---------------------------------------------------------------------------
# Half-space ↔ PV
# ---------------------------------------------------------------------------

# (halfspace-pv maps go here)
