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
    The same four slots select an arm of *both* the pairwise ``dist`` and the origin distance
    ``dist_0``; the two are listed separately below because they are different implementations.

    VERSION_DEFAULT (0):
        ``dist``: cancellation-free hyperbolic-haversine distance (hard floor at 0). Accurate at
        any representable radius — see the numerical-stability guide.
        ``dist_0``: ``arcsinh(√c·‖x_s‖)/√c``, read off the spatial part. Exact at every radius,
        no domain clamp, derivative bounded by 1.
    VERSION_SMOOTHENED (1):
        ``dist``: the same cancellation-free evaluation with a strictly-positive floor of
        ``2·arcsinh(10·eps)/√c`` applied in quadrature.
        ``dist_0``: VERSION_DEFAULT with the spatial radius floored in quadrature, giving a floor
        of ``arcsinh(20·eps)/√c`` — the same floor to first order.
    VERSION_LEGACY (2):
        ``dist``: pre-fix acosh-based distance with hard clipping, reproduced bit-for-bit for
        reproducibility. Routes through the Minkowski inner product and loses all precision once
        √c·(d₀(x) + d₀(y) - d(x, y)) exceeds ln(1/eps) — 15.9 (float32) / 36.0 (float64).
        ``dist_0``: pre-fix ``acosh(clip(√c·x₀, 1))/√c``. ``acosh``'s ``1 + 10·eps`` domain clamp
        makes every radius below ``sqrt(20·eps)/√c`` (1.5e-3 in float32) unrepresentable, and
        recovering the radius from ``x₀ = cosh(√c·d)/√c`` costs a further ``eps/(2·d²)`` relative.
        Use either only to match results computed before this fix.
    VERSION_LEGACY_SMOOTHENED (3): VERSION_LEGACY with soft clamping (``smooth_clamp_min``), and
        the same precision loss. For ``dist_0`` the softplus remainder additionally puts a
        ``0.16632/√c`` floor under *every* point that is not bitwise the origin, in both dtypes.

Note: Keep curvature parameter 'c' dynamic to support learnable curvature.
Use version_idx as static argument for JIT (static_argnames=['version_idx']).
"""

import math
from typing import NamedTuple

import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jaxtyping import Array, Float

from ..utils.math_utils import (
    MIN_NORM,
    acosh,
    cosh,
    safe_hypot,
    safe_norm,
    safe_normalize,
    sinh,
    smooth_clamp,
    smooth_clamp_min,
)
from ._base import ManifoldBase, default_atol
from .protocol import Curvature

# Version selection constants for _dist() and _dist_0()
VERSION_DEFAULT = 0
VERSION_SMOOTHENED = 1
VERSION_LEGACY = 2
VERSION_LEGACY_SMOOTHENED = 3


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
    # r ⊗ x = exp_0(r · log_0(x)) — the paper's Eq. 2 verbatim. ``_logmap_0`` already carries both
    # the geodesic radius (‖log_0 x‖ = d₀(x) by construction) and the direction, so the
    # normalize-then-rescale detour this replaced was an identity times a ``sqrt(maximum(‖v‖²,
    # MIN_NORM))`` floor. That floor pinned ‖log_0 x‖ at 3.16e-8 and made every point inside that
    # radius come back scaled by the wrong factor — invisible only while ``dist_0`` had its own,
    # larger acosh floor in front of it.
    v = _logmap_0(x, c)
    res = _expmap_0(r * v, c)
    return res


class _PolarFrame(NamedTuple):
    """Cancellation-free polar decomposition of a hyperboloid point pair (see :func:`_polar_frame`).

    Shared by ``_dist_stable``, ``_dist_stable_smoothened``, ``_sqdist`` and ``_logmap`` so the
    distance and the log map cannot drift apart. ``NamedTuple`` is a pytree, so this is jit- and
    vmap-clean and costs nothing at trace time.

    Dimension key:
        D: spatial dim    A: ambient dim (= D + 1, time coordinate first)
    """

    sqrt_c: Float[Array, ""]
    r_x: Float[Array, ""]  # ‖x_s‖ = sinh(a)/√c
    r_y: Float[Array, ""]  # ‖y_s‖ = sinh(b)/√c
    r_x_pos: Float[Array, ""]  # max(r_x, MIN_NORM) — see the floor note in _polar_frame
    r_y_pos: Float[Array, ""]
    x_hat_D: Float[Array, "dim"]  # x_s / r_x_pos (zero vector at the origin)
    y_hat_D: Float[Array, "dim"]
    x_time: Float[Array, ""]  # x₀ = cosh(a)/√c
    sinh_half_gap: Float[Array, ""]  # P = sinh((a - b)/2)
    q_angular: Float[Array, ""]  # q = ½·√c·√(r_x·r_y)·‖x̂ - ŷ‖
    sinh_half: Float[Array, ""]  # S = sinh(θ/2) = hypot(P, q),  θ = √c·d(x, y)
    cosh_half: Float[Array, ""]  # C = cosh(θ/2) = hypot(1, S)
    chord: Float[Array, ""]  # ‖x̂ - ŷ‖ = 2·sin(ψ/2)
    csum: Float[Array, ""]  # ‖x̂ + ŷ‖ = 2·cos(ψ/2)


def _polar_frame(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> _PolarFrame:
    """Hyperbolic haversine decomposition of the pair ``(x, y)``, free of catastrophic cancellation.

    **The problem.** Every ambient-chart formula built on ``⟨x, y⟩_L = -x₀y₀ + ⟨x_s, y_s⟩``
    subtracts two positive numbers of size ``e^(a+b)/(4c)`` to obtain ``cosh(θ)/c``, where
    ``a = √c·d₀(x)``, ``b = √c·d₀(y)`` and ``θ = √c·d(x, y)``. The surviving significand is
    ``e^(a + b - θ)`` times smaller than the operands — twice the Gromov product — so *all*
    precision is gone once ``a + b - θ`` exceeds ``ln(1/eps)``: 15.9 in float32, 36.0 in float64.
    Measured: float32 ``dist`` returns 0.0015 for a true distance of 1.0 at radius 10.

    **The fix.** The hyperbolic law of cosines, rewritten in half-angle (haversine) form, is a sum
    of two *non-negative* terms, so nothing cancels::

        sinh²(θ/2) = sinh²((a - b)/2) + (c/4)·r_x·r_y·‖x̂_s - ŷ_s‖²

    with ``r_x = ‖x_s‖``, ``x̂_s = x_s/r_x``. It follows from ``cosh θ = cosh a cosh b -
    sinh a sinh b cos ψ`` and ``‖x̂ - ŷ‖² = 2 - 2cos ψ`` via ``sinh²(t/2) = (cosh t - 1)/2``. Both
    roots are then taken through the on-manifold identities ``sinh a = √c·r_x``, ``cosh a = √c·x₀``,
    ``e^a = √c·(x₀ + r_x)``, which involve only additions of positive quantities:

    * ``P := sinh((a - b)/2) = ½·(u_x - u_y)/(√u_x·√u_y)`` with ``u_x = x₀ + r_x``.
    * ``q := ½·√c·√r_x·√r_y·chord``, so that ``q² = (c/4)·r_x·r_y·chord²``.
    * ``S := hypot(P, q) = sinh(θ/2)`` and ``C := hypot(1, S) = cosh(θ/2)``.

    **Operation orderings that are load-bearing** (each measured, do not "simplify"):

    * ``√u_x`` and ``√u_y`` are taken *separately*. ``√(u_x·u_y)`` overflows float32 as soon as
      ``a + b > 88``, while the quotient itself is perfectly representable.
    * ``√r_x·√r_y·chord``, never ``√(r_x·r_y)`` and never ``chord²·r_x·r_y``: both alternatives
      square a spatial radius, which leaves float32 at radius 44.
    * every norm goes through :func:`~hyperbolix.utils.math_utils.safe_norm`, whose max-scaling is
      what keeps a legitimate ``1e-34`` chord from being flushed to the ``MIN_NORM`` floor.
    * ``(u_x - u_y)`` is a difference, not a quotient of exponentials: for ``a ≈ b`` (small
      distances) the subtraction of nearby floats is exact (Sterbenz), whereas the algebraically
      equal ``½·(√(u_x/u_y) - √(u_y/u_x))`` loses all significance there.

    **The MIN_NORM floor on the radii is deliberate and must stay a floor** (``maximum``), not a
    ``where``-style exact-zero guard. At ``x`` exactly at the origin the two floors cancel *exactly*
    in the gradient: ``q ∝ √(r_x_pos) = √MIN_NORM`` while ``∂chord/∂x_s ∝ 1/r_x_pos = 1/MIN_NORM``,
    and the ``∂S/∂q = q/S`` factor restores the remaining ``√MIN_NORM``. The product is
    floor-independent and yields the analytically correct ``∇_{x_s} d = -ŷ_s`` with ``|∇| = 1``.
    Replacing the floor with an exact zero would make that gradient vanish and freeze every
    origin-initialized parameter.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        y: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        The shared :class:`_PolarFrame`.
    """
    sqrt_c = jnp.sqrt(c)
    x_time, y_time = x[0], y[0]
    x_s_D, y_s_D = x[1:], y[1:]

    r_x = safe_norm(x_s_D)
    r_y = safe_norm(y_s_D)
    r_x_pos = jnp.maximum(r_x, MIN_NORM)
    r_y_pos = jnp.maximum(r_y, MIN_NORM)
    x_hat_D = x_s_D / r_x_pos
    y_hat_D = y_s_D / r_y_pos

    # e^a = √c·u_x, so P = sinh((a-b)/2) = (u_x - u_y)/(2·√u_x·√u_y) — the √c cancels.
    # On the upper sheet u = x₀ + r_x >= x₀ >= 1/√c > 0, so the MIN_NORM floor is a no-op for every
    # valid point; it only stops a fully degenerate input (an all-zero "point", x₀ = 0) from
    # turning the quotient into 0/0 = NaN.
    u_x = jnp.maximum(x_time + r_x, MIN_NORM)
    u_y = jnp.maximum(y_time + r_y, MIN_NORM)
    # A point past the dtype's representable radius has x₀ = inf (the sqrt inside _proj
    # overflowed), where the exact quotient is inf/inf = NaN. Return ±inf instead: the geodesic
    # distance there really is infinite, and an inf stays visible downstream while a NaN silently
    # poisons every parameter it touches. Value- and gradient-identity whenever both are finite.
    representable = jnp.isfinite(u_x) & jnp.isfinite(u_y)
    sinh_half_gap = jnp.where(
        representable,
        0.5 * (u_x - u_y) / (jnp.sqrt(u_x) * jnp.sqrt(u_y)),
        jnp.where(u_x > u_y, jnp.inf, -jnp.inf),
    )

    chord = safe_norm(x_hat_D - y_hat_D)  # 2·sin(ψ/2)
    csum = safe_norm(x_hat_D + y_hat_D)  # 2·cos(ψ/2)
    # Grouped so the two radii meet each other first: float multiplication is commutative but not
    # associative, so ``(k·√r_x)·√r_y`` and ``(k·√r_y)·√r_x`` differ in the last ulp and ``dist``
    # would stop being *bitwise* symmetric under swapping x and y.
    q_angular = (0.5 * sqrt_c * chord) * (jnp.sqrt(r_x_pos) * jnp.sqrt(r_y_pos))

    sinh_half = safe_hypot(sinh_half_gap, q_angular)
    cosh_half = safe_hypot(jnp.ones_like(sinh_half), sinh_half)

    return _PolarFrame(
        sqrt_c=sqrt_c,
        r_x=r_x,
        r_y=r_y,
        r_x_pos=r_x_pos,
        r_y_pos=r_y_pos,
        x_hat_D=x_hat_D,
        y_hat_D=y_hat_D,
        x_time=x_time,
        sinh_half_gap=sinh_half_gap,
        q_angular=q_angular,
        sinh_half=sinh_half,
        cosh_half=cosh_half,
        chord=chord,
        csum=csum,
    )


# Distance implementations for lax.switch
def _dist_stable(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Cancellation-free geodesic distance, ``d = 2·arcsinh(sinh(θ/2))/√c`` (hyperbolic haversine).

    Reads ``S = sinh(θ/2)`` off the :func:`_polar_frame` decomposition, which builds it as a sum of
    non-negative terms instead of subtracting the two large halves of ``⟨x, y⟩_L``. No ``acosh``,
    no domain clamp, and no ``where(x == y, ...)`` coincidence guard: ``x == y`` gives ``P = 0`` and
    ``chord = 0`` exactly, hence ``S = 0``, ``d = 0`` and an exactly-zero gradient by construction.

    It also removes the float32 resolution floor the ``acosh`` form imposed: ``acosh``'s
    ``1 + 10·eps`` domain clamp made every distance below ~1.5e-3 unrepresentable, whereas
    ``arcsinh`` is exact near 0.

    ``jnp.arcsinh`` is used directly — measured ≤2.5 ulp in both dtypes with a clean derivative
    ``1/√(1 + S²)``, so no ``custom_jvp`` is needed anywhere in this path.
    """
    frame = _polar_frame(x, y, c)
    return 2.0 * jnp.arcsinh(frame.sinh_half) / frame.sqrt_c


def _dist_stable_smoothened(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """:func:`_dist_stable` with a strictly-positive floor: ``S`` is replaced by ``hypot(S, ε)``.

    ``ε = 10·eps`` puts the floor at ``2·arcsinh(ε)/√c`` ≈ 2.4e-6 (float32) / 4.4e-15 (float64),
    which is the smoothened arm's contract: a distance that is never exactly zero, with gradients
    that stay finite through coincidence. Because ``hypot`` is smooth and the floor enters in
    quadrature, the perturbation is ``O(ε²/S)`` — invisible at any distance above the floor.

    ``smooth_clamp_min`` is deliberately *not* used here (the legacy arm's approach). Its softplus
    remainder adds ``log(2)/β = 0.0139`` at the clamp point and never fully decays, so it would
    shift **every** distance in the working range by ~0.028 rather than only lifting zero.
    """
    frame = _polar_frame(x, y, c)
    eps = 10.0 * float(jnp.finfo(x.dtype).eps)
    sinh_half_floored = safe_hypot(frame.sinh_half, jnp.asarray(eps, dtype=x.dtype))
    return 2.0 * jnp.arcsinh(sinh_half_floored) / frame.sqrt_c


def _dist_legacy(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Standard acosh distance with hard clipping.

    Kept for reference and comparison only. It routes through :func:`_minkowski_inner` and so loses
    all precision once ``√c·(d₀(x) + d₀(y) - d(x, y))`` exceeds ``ln(1/eps)`` — see
    :func:`_polar_frame`. Prefer :func:`_dist_stable`.
    """
    sqrt_c = jnp.sqrt(c)
    lorentz_inner = _minkowski_inner(x, y)
    arg = jnp.clip(-c * lorentz_inner, min=1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if points are identical
    same = jnp.all(jnp.equal(x, y))
    return jnp.where(same, 0.0, res)  # type: ignore[return-value]


def _dist_legacy_smoothened(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Smoothened distance with soft clamping. Legacy — see :func:`_dist_legacy`."""
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
    return lax.switch(version_idx, [_dist_stable, _dist_stable_smoothened, _dist_legacy, _dist_legacy_smoothened], x, y, c)


def _sqdist(x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Squared Lorentzian distance between hyperboloid points.

    Computes the squared Lorentzian distance of Law et al. (2019):

        d_L²(x, y) = -2/c - 2 * ⟨x, y⟩_L

    evaluated **cancellation-free** as ``d_L² = (4/c)·sinh²(θ/2) = 4·S²/c`` off the
    :func:`_polar_frame` decomposition (``S = sinh(θ/2)``, ``θ = √c·d(x, y)``). The two expressions
    are algebraically identical on the manifold, but the literal one subtracts two positive numbers
    of size ``e^(a+b)/(2c)`` to obtain a result of size ``e^θ/c`` and so loses all precision past
    the ``ln(1/eps)`` Gromov-product threshold described in :func:`_polar_frame`. The haversine form
    is a sum of non-negative terms, which also makes the old ``clip(min=0)`` unnecessary: the result
    is non-negative by construction and exactly 0 at ``x == y``.

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
    # d_L² = ||x - y||²_L = (4/c)·sinh²(θ/2). Non-negative by construction — no clip needed.
    frame = _polar_frame(x, y, c)
    return 4.0 * frame.sinh_half**2 / c


# Distance from origin implementations for lax.switch
def _dist_0_stable(x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Geodesic radius read off the *spatial* part: ``d₀ = arcsinh(√c·‖x_s‖)/√c``.

    On the upper sheet ``√c·x₀ = sqrt(1 + c·‖x_s‖²)``, so ``acosh(√c·x₀) = arcsinh(√c·‖x_s‖)``
    exactly — the same number, obtained from the coordinate that still resolves it near the origin.
    The ``x₀`` route cannot: ``x₀ = cosh(√c·d)/√c ≈ (1 + c·d²/2)/√c`` stores ``d`` only to
    ``sqrt(eps)`` resolution (relative error ``eps/(2·c·d²)``), and ``acosh``'s ``1 + 10·eps``
    domain clamp then flattens every radius below ``sqrt(20·eps)/√c`` — 1.5e-3 in float32 — onto
    that one value. Measured at ``c = 1``, float32, dim 8 (median relative error, legacy → this):
    1.5e3 → 2.5e-9 at ``r = 1e-6``, 5.4e-1 → 9.8e-8 at 1e-3, 5.0e-4 → 2.7e-8 at 1e-2, 3.5e-6 →
    1.8e-8 at 0.1 (the ``eps/(2·c·d²)`` term), and ≤7.1e-8 → ≤5.2e-8 over ``r ∈ [1, 40]``
    (``logs/2026-09-02_submission-numerics/probe_hyperboloid_origin.py``, A100, jax 0.9.1).

    ``arcsinh`` needs no domain clamp (its argument is a norm, so the whole real half-line is
    valid) and its derivative ``1/sqrt(1 + u²)`` is bounded by 1 everywhere, so the singularity
    that motivated :func:`_dist_0_legacy_smoothened`'s soft clamp is gone rather than smoothed.
    ``safe_norm`` supplies the exactly-zero value *and* exactly-zero VJP at the origin, which is
    what keeps ``jax.grad(dist_0)`` finite there without a ``where``-guard.

    Reads only ``x_s``: an off-sheet input gets the radius of its :func:`_proj` projection, which
    is the same source of truth ``_proj`` itself uses (it reconstructs ``x₀`` from ``x_s``) and
    what ``ProperVelocity._dist_0`` already does.
    """
    sqrt_c = jnp.sqrt(c)
    return jnp.arcsinh(sqrt_c * safe_norm(x[1:])) / sqrt_c


def _dist_0_stable_smoothened(x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """:func:`_dist_0_stable` with a strictly-positive floor: the radius becomes ``hypot(u, ε)``.

    ``ε = 20·eps`` puts the floor at ``arcsinh(20·eps)/√c`` ≈ 2.4e-6 (float32) / 4.4e-15
    (float64) — to first order the pairwise smoothened arm's ``2·arcsinh(10·eps)/√c``, so the two
    smoothened arms agree on what "never exactly zero" means. Applied in quadrature like
    :func:`_dist_stable_smoothened`, the perturbation is ``O(ε²/u)`` and invisible above the floor.

    ``smooth_clamp_min`` (the legacy arm's approach) is deliberately not used: its ``log(2)/β``
    softplus remainder put a **0.16632/√c** floor under every non-origin point in *both* dtypes.
    """
    sqrt_c = jnp.sqrt(c)
    eps = 20.0 * float(jnp.finfo(x.dtype).eps)
    radius_floored = safe_hypot(sqrt_c * safe_norm(x[1:]), jnp.asarray(eps, dtype=x.dtype))
    return jnp.arcsinh(radius_floored) / sqrt_c


def _dist_0_legacy(x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Standard acosh distance from origin with hard clipping.

    Kept for reference and comparison only. It reads the radius off ``x₀`` through ``acosh``, whose
    domain clamp makes every radius below ``sqrt(20·eps)/√c`` (1.5e-3 float32, 6.7e-8 float64)
    unrepresentable, and the bitwise ``at_origin`` guard makes the gradient there exactly zero.
    Prefer :func:`_dist_0_stable`.
    """
    sqrt_c = jnp.sqrt(c)
    x0 = x[0]
    arg = jnp.clip(sqrt_c * x0, min=1.0)
    res = acosh(arg) / sqrt_c
    # Zero out if at origin
    origin = _create_origin(c, x.shape[0] - 1, x.dtype)
    at_origin = jnp.all(jnp.equal(x, origin))
    return jnp.where(at_origin, 0.0, res)  # type: ignore[return-value]


def _dist_0_legacy_smoothened(x: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
    """Smoothened distance from origin with soft clamping. Legacy — see :func:`_dist_0_legacy`.

    ``smooth_clamp_min(√c·x₀, 1.0)`` adds ``log(2)/β = 0.0139`` to the ``acosh`` argument, i.e. a
    floor of ``acosh(1 + log(2)/50)/√c = 0.16632/√c`` on every point that is not bitwise the
    origin, in both dtypes.
    """
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

    Slots 0/1 read the geodesic radius off the **spatial** part ``x_s`` and ignore ``x₀``; an
    off-sheet input therefore gets the radius of its :func:`_proj` projection. A raw
    ``jax.grad(dist_0)`` consequently has its support on ``x_s`` rather than on ``x₀``; the two
    differ by a multiple of the constraint normal, so after :func:`_egrad2rgrad` /
    :func:`_tangent_proj` the Riemannian gradient is identical and optimizers are unaffected.

    .. note::
       **Breaking change.** ``version_idx=2``/``3`` used to duplicate 0/1; they now select the
       pre-fix ``acosh`` arms (:func:`_dist_0_legacy`, :func:`_dist_0_legacy_smoothened`), matching
       what those slots already meant for :func:`_dist`.

    Args:
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)
        version_idx: Distance version index (use VERSION_* constants)

    Returns:
        Geodesic distance d(origin, x), scalar

    References:
        Nickel & Kiela. "Poincaré embeddings for learning hierarchical representations." NeurIPS 2017.
    """
    return lax.switch(
        version_idx, [_dist_0_stable, _dist_0_stable_smoothened, _dist_0_legacy, _dist_0_legacy_smoothened], x, c
    )


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
    """Logarithmic map: map point y to tangent space at **x** (the second argument is the base point).

    Built in an orthonormal geodesic frame at ``x`` instead of from ``y + c·⟨x, y⟩_L·x``. The
    textbook direction vector is a difference of two ambient vectors of size ``e^(a+b)``, so it
    inherits the ``_minkowski_inner`` cancellation described in :func:`_polar_frame` twice over
    (once in the inner product, once in the normalization) and returns NaN from radius ~10 in
    float32 / ~20 in float64.

    The frame is exactly tangent and exactly orthonormal *analytically*::

        e_rad = -√c·(r_x, x₀·x̂_s)         unit, Minkowski-orthogonal to x, pointing at the origin
        e_ang = (0, n̂),  n̂ = normalize(ŷ_s - ⟨x̂_s, ŷ_s⟩·x̂_s)
        log_x(y) = d·(cos φ · e_rad + sin φ · e_ang)

    (``⟨e_rad, e_rad⟩_L = c·(x₀² - r_x²) = 1`` and ``⟨e_rad, x⟩_L = 0`` are exact identities on the
    manifold.) The angle ``φ`` between the geodesic and the inward radial direction comes from

        cos φ·sinh θ = 2P·cosh((a - b)/2) + 2q²·coth a,   sin φ·sinh θ = sin ψ·sinh b

    which, using ``sinh θ = 2·S·C`` and ``sin ψ = chord·csum/2``, factor into products of
    **individually bounded** ratios of :func:`_polar_frame` quantities::

        cos φ = (P/S)·(hypot(1, P)/C) + (q/S)·(q/C)·(x₀/r_x)
        sin φ = (q/S)·(csum/2)·√r_y/(√r_x·C)

    with ``|P/S| ≤ 1``, ``q/S ≤ 1``, ``q/C ≤ 1``, ``hypot(1, P)/C ≤ 1`` and ``csum/2 ≤ 1``. The one
    unbounded factor, ``coth a = x₀/r_x``, is multiplied by ``q² = O(r_x)``, so the product is
    ``O(1)``; writing it as the ratio product above (rather than forming ``q²`` first) is what keeps
    it from overflowing at large radius, where ``q`` alone is ~1e18 in float32.

    The result is **not** passed through :func:`_tangent_proj`: that helper routes through
    :func:`_minkowski_inner` and would reintroduce exactly the NaN this rewrite removes. It is not
    needed — the frame is tangent by construction, with a measured relative residual
    ``|⟨u, x⟩_L|/(‖u‖∞·‖x‖∞)`` of ≤2.3e-7 (float32) / ≤2.9e-16 (float64). For the same reason
    ``‖log_x(y)‖_x = d(x, y)`` holds by construction: ``d`` is taken from the same frame.

    At ``x`` exactly at the origin the radial leg degenerates (``r_x = 0`` ⇒ ``e_rad = 0``), so the
    result falls back to :func:`_logmap_0`, which is exact there. Both branches of the ``where`` are
    finite — the ``MIN_NORM``-floored denominators guarantee it — so the ``where``'s VJP is NaN-free.

    Args:
        y: Hyperboloid point to map, shape (dim+1,)
        x: Hyperboloid point serving as the base of the tangent space, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Tangent vector log_x(y) at x, shape (dim+1,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    frame = _polar_frame(x, y, c)
    dist_xy = 2.0 * jnp.arcsinh(frame.sinh_half) / frame.sqrt_c

    # S = 0 exactly at x == y, where the direction is arbitrary; floor the denominator so the
    # discarded ratios stay finite (they are multiplied by dist_xy = 0 anyway).
    sinh_half_pos = jnp.maximum(frame.sinh_half, MIN_NORM)
    cos_phi = (frame.sinh_half_gap / sinh_half_pos) * (
        safe_hypot(jnp.ones_like(frame.sinh_half_gap), frame.sinh_half_gap) / frame.cosh_half
    ) + (frame.q_angular / sinh_half_pos) * (frame.q_angular / frame.cosh_half) * (frame.x_time / frame.r_x_pos)
    sin_phi = (
        (frame.q_angular / sinh_half_pos)
        * (frame.csum / 2.0)
        * jnp.sqrt(frame.r_y_pos)
        / (jnp.sqrt(frame.r_x_pos) * frame.cosh_half)
    )

    # Inward unit radial direction, exactly tangent at x.
    e_rad_A = -frame.sqrt_c * jnp.concatenate([frame.r_x[None], frame.x_time * frame.x_hat_D])
    # Unit angular direction: the component of ŷ_s orthogonal to x̂_s. Exactly the zero vector when
    # the two points share a ray (ψ = 0 or π), which is also where sin φ = 0.
    n_hat_D = safe_normalize(frame.y_hat_D - jnp.dot(frame.x_hat_D, frame.y_hat_D) * frame.x_hat_D)
    e_ang_A = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), n_hat_D])

    res = dist_xy * (cos_phi * e_rad_A + sin_phi * e_ang_A)
    return jnp.where(frame.r_x > 0, res, _logmap_0(y, c))


def _logmap_0(y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, "dim_plus_1"]:
    """Logarithmic map from origin: ``log_0(y) = [0, arcsinh(u)/u · y_s]`` with ``u = √c·‖y_s‖``.

    Built from the spatial part alone, the same source of truth :func:`_dist_0_stable` uses (see
    its docstring for why ``x₀`` cannot resolve a small radius). The scale ``arcsinh(u)/u`` *is*
    ``d₀(y)/‖y_s‖``, so ``‖log_0(y)‖ = d₀(y)`` holds by construction rather than by cancelling a
    floored numerator against an un-floored denominator — the arrangement this replaced, which
    returned an exactly-zero radial Jacobian entry below float32 radius 1.5e-3 (the ``acosh``
    clamp) and one ~1500 times too large at 1e-6 (the floored 1.5e-3 divided by 1e-6). Because ``arcsinh(u)/u → 1`` as ``u → 0`` and the
    ``MIN_NORM`` floor keeps ``u ≥ √c·1e-15``, the ratio is never 0/0.

    ``version_idx`` is intentionally not threaded through: there is nothing to switch on any more.

    The result is **not** passed through :func:`_tangent_proj`. At the origin that projection is
    the identity on a vector whose time component is already 0 (verified bitwise over magnitudes
    1e-20…1e10, with an identity VJP), and it routes through :func:`_minkowski_inner`, which turns
    an ``inf`` spatial input into an all-NaN result instead of leaving the time slot intact.

    Args:
        y: Hyperboloid point in ambient representation, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Tangent vector log_0(y) in ambient representation, shape (dim+1,)
        (first component is 0)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    sqrt_c = jnp.sqrt(c)
    y_rest = y[1:]
    # ``safe_norm``, not the file's older ``sqrt(sum(y_s²) + MIN_NORM²)``: the spatial part of a
    # float32 point at radius 45 is 1.6e19, whose sum of squares overflows to inf while the norm
    # itself is perfectly representable (the old expression then returned an all-zero tangent
    # vector there, and the arcsinh form would return NaN). ``safe_norm`` gives an exact 0 with an
    # exactly-zero VJP at the origin, so the ``MIN_NORM`` floor below is forward-only — it just
    # stops ``arcsinh(u)/u`` from evaluating 0/0; the Jacobian at y = origin is the identity either
    # way. That matters for the gyro-bias path of the PLFC / Busemann FC layers, whose bias point
    # is the origin at zero init.
    y_rest_norm = jnp.maximum(safe_norm(y_rest), MIN_NORM)

    u = sqrt_c * y_rest_norm
    scale = jnp.arcsinh(u) / u  # = d₀(y)/‖y_s‖, → 1 as u → 0

    v0 = jnp.zeros(1, dtype=y.dtype)
    v_rest = scale * y_rest
    return jnp.concatenate([v0, v_rest])


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
    """Riemannian norm ``‖v‖_x`` of a vector **assumed tangent at x**, computed without cancellation.

    ``v`` is required to satisfy ``⟨v, x⟩_L = 0``; the base point ``x`` is what makes that
    assumption usable. Eliminating ``v₀`` through it (``v₀·x₀ = ⟨v_s, x_s⟩``) turns the Lorentz
    norm into a sum of two non-negative terms::

        t = ⟨v_s, x̂_s⟩                                       (radial component of v_s)
        ‖v‖²_L = ‖v_s - t·x̂_s‖² + (t/(√c·x₀))²               (√c·x₀ = cosh a ≥ 1)

    The literal ambient form ``-v₀² + ‖v_s‖²`` instead subtracts two numbers of size
    ``(√c·x₀·‖v‖)²``, so its relative error grows like ``c·x₀²·eps``: measured on an *exactly unit*
    radial tangent vector it returns 0.87 at radius 8 and 1e-15 (i.e. the ``MIN_NORM`` floor, a
    100% error) at radius 10 in float32, and collapses the same way past radius 20 in float64.
    The form above errs like ``√c·x₀·eps`` instead — one power of ``x₀`` better, which roughly
    doubles the usable radius (float32: exact to radius ~15; float64: to ~25).

    That remaining ``√c·x₀·eps`` term is *not* an artifact of this formula but of the ambient chart:
    at radius ``a`` the tangent vector's ambient components are ``e^a`` times its Riemannian length,
    so one ulp of the representation already costs that much. No implementation reading only
    ``(v, x)`` can do better.

    ``safe_norm``/``safe_hypot`` compose the two terms, so nothing overflows at large radius and
    ``v = 0`` returns exactly 0 with an exactly-zero (hence finite) gradient — the ``MIN_NORM²``
    floor the previous implementation needed for that is no longer required.

    Args:
        v: Tangent vector at x, shape (dim+1,)
        x: Hyperboloid point, shape (dim+1,)
        c: Curvature (positive)

    Returns:
        Riemannian norm ||v||_x, scalar
    """
    sqrt_c = jnp.sqrt(c)
    x_s_D = x[1:]
    v_s_D = v[1:]
    x_hat_D = x_s_D / jnp.maximum(safe_norm(x_s_D), MIN_NORM)

    radial = jnp.dot(v_s_D, x_hat_D)
    perp_norm = safe_norm(v_s_D - radial * x_hat_D)
    # √c·x₀ = cosh a >= 1 on the upper sheet, so the floor is a no-op for every valid base point;
    # it only keeps a degenerate x (x₀ = 0) from dividing by zero.
    return safe_hypot(perp_norm, radial / jnp.maximum(sqrt_c * x[0], MIN_NORM))


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


def _is_in_manifold(x: Float[Array, "dim_plus_1"], c: Curvature, atol: float | None = None) -> Array:
    """Check if point x lies on hyperboloid.

    Args:
        x: Point to check, shape (dim+1,)
        c: Curvature (positive)
        atol: Absolute tolerance on the Lorentz-norm residual. ``None`` resolves to
            :func:`~hyperbolix.manifolds._base.default_atol` for ``x.dtype``.

    Returns:
        True if -x₀² + ||x_rest||² = -1/c (within ``atol``) and x₀ > 0
    """
    lorentz_norm = _minkowski_inner(x, x)
    tol = default_atol(x.dtype) if atol is None else atol
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
        atol: Absolute tolerance on ⟨v, x⟩_L. ``None`` resolves to
            :func:`~hyperbolix.manifolds._base.default_atol` for ``v.dtype``.

    Returns:
        True if ⟨v, x⟩_L ≈ 0
    """
    tol = default_atol(v.dtype) if atol is None else atol
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
    does) biases the expected spatial radius upward with the post-concat dimension: for
    Gaussian spatial parts ``‖v‖² ~ χ²_k``, so ``E[log‖v‖] = ½·(ψ(k/2) + log 2)`` grows with
    the dimension ``k``. To keep the expected *log* spatial radius invariant, each block's
    spatial part is rescaled by the ratio of the two chi radii,

        s = exp( ½ · ( ψ(nᵢ / 2) - ψ(n / 2) ) )   (≈ 1/√N, so s < 1 for N > 1),

    where ψ is the digamma function, ``nᵢ = d`` is the per-block spatial dimension and
    ``n = N · d`` is the total post-concat spatial dimension. The single time coordinate is then
    recomputed so the scaled point stays on the hyperboloid (here ``1/c`` equals the reference
    ``k = -1/K``):

        t' = sqrt( 1/c + s² · Σᵢ (tᵢ² - 1/c) ).

    When ``N = 1`` the scale is ``1`` and this reduces exactly to ``_hcat``.

    .. note::
       The digamma difference is a *shrink* (``s ≤ 1``). Both the Shi et al. 2026 reference
       implementation and hyperbolix ≤ 0.11 had the two arguments the other way round, which
       amplifies the spatial radius by ≈ √N per concatenation — worse than plain ``_hcat``,
       which it was meant to correct. Fixed 2026-07-31; see
       ``HypConv2DHyperboloidILNN.kernel_init_std`` for the coupled init change.

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
    scale = jnp.exp(0.5 * (digamma(d / 2.0) - digamma(n_total / 2.0)))

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
    VERSION_LEGACY = VERSION_LEGACY
    VERSION_LEGACY_SMOOTHENED = VERSION_LEGACY_SMOOTHENED

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
        """Geodesic distance from the origin, ``arcsinh(√c·‖x_s‖)/√c``.

        Read off the **spatial** part; ``x₀`` is ignored, so an off-sheet input gets the radius of
        its :meth:`proj` projection (the same convention ``proj`` itself follows). A raw
        ``jax.grad`` of this therefore has its support on ``x_s`` rather than ``x₀``; the two
        differ by a multiple of the constraint normal, so the Riemannian gradient after
        :meth:`egrad2rgrad` / :meth:`tangent_proj` is identical and optimizers are unaffected.

        ``version_idx`` 2/3 select the pre-fix ``acosh`` arms — see :func:`_dist_0`.
        """
        return _dist_0(self._cast(x), c, version_idx)

    def sqdist(self, x: Float[Array, "dim_plus_1"], y: Float[Array, "dim_plus_1"], c: Curvature) -> Float[Array, ""]:
        """Squared Lorentzian distance ``d_L²(x, y) = -2/c - 2⟨x,y⟩_L`` (Law et al. 2019).

        That is the mathematical definition; it is now evaluated cancellation-free as
        ``d_L² = 4·sinh²(√c·d(x,y)/2)/c`` off the :func:`_polar_frame` decomposition instead of the
        literal subtraction, which loses precision the same way :func:`_dist_legacy` does (see
        :func:`_sqdist`'s docstring for the full derivation).

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
        """Logarithmic map from the origin, ``[0, arcsinh(√c‖y_s‖)/(√c‖y_s‖) · y_s]``.

        Like :meth:`dist_0` it reads the radius off the **spatial** part and ignores ``y₀``, so an
        off-sheet input is treated as its :meth:`proj` projection. ``‖log_0(y)‖ = dist_0(y)`` holds
        by construction at every radius.
        """
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

    def is_in_manifold(self, x: Float[Array, "dim_plus_1"], c: Curvature, atol: float | None = None) -> Array:
        """Check if point x lies on hyperboloid (``atol`` default: :func:`default_atol`)."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(
        self, v: Float[Array, "dim_plus_1"], x: Float[Array, "dim_plus_1"], c: Curvature, atol: float | None = None
    ) -> Array:
        """Check if vector v lies in tangent space at point x (``atol`` default: :func:`default_atol`)."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c, atol)

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

        Like :meth:`hcat`, but shrinks each block's spatial part by the digamma factor
        ``s = exp(½·(ψ(d/2) - ψ(N·d/2))) ≈ 1/√N`` so the expected log spatial radius is
        invariant to the number of concatenated blocks. Reduces to :meth:`hcat` when
        ``N == 1``.
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
