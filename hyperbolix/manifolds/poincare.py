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

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.special
from jaxtyping import Array, Float

from ..utils.math_utils import MIN_NORM, asinh, atanh, cap_at, cosh, floor_at, safe_norm, safe_sqrt, sinh, tanh
from ..utils.precision import MATMUL_PRECISION
from ._base import ManifoldBase, default_atol
from ._gyrovector_core import (
    _addition,
    _boundary_floor,
    _conformal_factor,
    _conformal_factor_batch,
    _gyration,
    _max_norm,
    _mobius_denominator,
    _proj,
    _proj_batch,
)
from .protocol import ScalarCurvature

# Version selection constants for dist() and dist_0()
VERSION_MOBIUS_DIRECT = 0
VERSION_MOBIUS = 1
VERSION_METRIC_TENSOR = 2


def _scalar_mul(r: float | Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
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
    # `safe_sqrt`, not `safe_norm`: the argument is a **ball point** (or a difference of two),
    # so `sum(.**2) <= 4/c` and the max-scaling `safe_norm` pays a second reduction for is
    # unreachable here. `safe_sqrt`'s double-`where` supplies the same finite (zero) derivative
    # at 0 that the old `+ MIN_NORM**2` did, without its 1e-15 floor on the value.
    # The `floor_at` is the deliberate part -- `x_norm` is a *divisor* two lines down, so it must
    # stay bounded away from 0; above ~1e-14 the two forms agree to rounding.
    # `axis=-1, keepdims=True` rather than the whole-array reduction: for the single point this
    # function is contracted for it is the same value in a shape-(1,) box (bit-identical, same
    # kernel), but it makes the `/ c_norm_prod * x` below a *per-row* scale for the (B, dim)
    # inputs callers pass anyway, instead of one whole-array Frobenius norm.
    x_norm = floor_at(safe_sqrt(jnp.sum(x**2, axis=-1, keepdims=True)), MIN_NORM)
    c_norm_prod = jnp.sqrt(c) * x_norm
    # `tanh` is the math_utils wrapper (matching _expmap_0 and _expmap), not XLA's kernel: more
    # accurate in float32, and its ±(1 - 10·eps) output clip pairs with `atanh`'s own domain guard
    # so a round trip through the two cannot reach the pole. The trailing _proj still bounds the
    # result, so the clip is at worst redundant here.
    res = tanh(r * atanh(c_norm_prod)) / c_norm_prod * x
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
def _dist_mobius_direct(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Geodesic distance in the ``arcsinh`` form ``2·arcsinh(√t)/√c`` — version slots 0 **and** 2.

    The Möbius form (slot 0) is ``2·atanh(u)/√c`` with ``u = √c‖x - y‖/√D₋`` and
    ``D₋ = 1 - 2c⟨x,y⟩ + c²‖x‖²‖y‖²``. Writing ``B_x = 1 - c‖x‖²`` (``= 2/λ_x``), the identity
    ``D₋ = B_x·B_y + c‖x - y‖²`` gives ``1 - u² = B_x·B_y/D₋``, and ``atanh(u) = arcsinh(u/√(1 - u²))``
    turns the distance into::

        √t = √c‖x - y‖ / √(B_x·B_y),   d(x, y) = 2·arcsinh(√t)/√c

    The metric-tensor integral (slot 2, ``VERSION_METRIC_TENSOR``) gives ``acosh(1 + 2t)/√c`` with the
    same ``t = c‖x - y‖²/((1 - c‖x‖²)(1 - c‖y‖²))``, and the half-angle identity
    ``acosh(1 + 2t) = 2·arcsinh(√t)`` lands on the same expression, so both slots run this body. Each
    literal spelling had its own defect. The ``atanh`` one saturated for far pairs, where ``u → 1``
    and the float32 domain clip at ``1 - 10·eps`` took over: two points on opposite sides at scaled
    radius 7.2 each (true ``√c·d = 14.4``, c = 1) came back 14.333 with an exactly zero gradient, and
    at 9 each (true 18.0) still 14.333 — both pairs well inside the float32 chart. The ``acosh`` one
    put the whole separation into a perturbation of a leading 1: ``math_utils.acosh``'s ``1 + 10·eps``
    domain clamp pinned every pair with ``t < 5·eps`` to a constant floor with a zero gradient (in
    float32 closer than ``sqrt(5·eps/c)·(1 - c·r²)`` for two points at radius ``r``, 7.7e-4/√c at
    the origin). ``arcsinh`` has no domain to clip and a derivative bounded by 1: the far float32
    pairs now give 14.40011 and 17.99972 (float64 of the same inputs: 14.40006, 17.99968) with
    gradients within 2.8e-5 and 2.3e-5 relative. What is left is the chart's own floor, the float32
    rounding of ``B_x``, ``B_y`` (``eps/B`` relative).

    It is also the cheaper body: three reductions (``‖x - y‖²``, ``‖x‖²``, ``‖y‖²``) where the
    factored ``D₋`` took five with a traced ``c`` (its ``‖x‖²``, ``‖y‖²`` and both chord/sum
    reductions of ``_mobius_denominator``) and slot 2's own body four (``safe_norm``'s max-scaling
    pass, which a difference of two ball points does not need), plus a second rounding of each
    ``B`` taken as ``2/λ``.
    """
    sqrt_c = jnp.sqrt(c)
    # `safe_sqrt`, not `safe_norm`: the argument is a **ball point** (or a difference of two),
    # so `sum(.**2) <= 4/c` and the max-scaling `safe_norm` pays a second reduction for is
    # unreachable here. `safe_sqrt`'s double-`where` supplies the same finite (zero) derivative
    # at 0 that the old `+ MIN_NORM**2` did, without its 1e-15 floor on the value.
    # This is a numerator, not a divisor, so no floor is needed: the old `+ MIN_NORM**2` was purely
    # the sqrt-gradient guard and it cost a 1e-15 floor on every genuinely small separation.
    # `dist(x, x)` is therefore an exact 0 with an exactly-zero gradient.
    num = safe_sqrt(jnp.sum((y - x) ** 2))
    # B_x, B_y with `_conformal_factor`'s boundary clamp (the analytic minimum of 1 - c‖x‖² on a
    # projected point), so an unprojected near-boundary point cannot drive the divisor to 0.
    # Taken directly rather than as `2/λ`, which would round twice.
    floor_b = _boundary_floor(x, c)
    one_minus_cx = floor_at(1.0 - c * jnp.dot(x, x, precision=MATMUL_PRECISION), floor_b)
    one_minus_cy = floor_at(1.0 - c * jnp.dot(y, y, precision=MATMUL_PRECISION), floor_b)
    sqrt_t = sqrt_c * num / jnp.sqrt(one_minus_cx * one_minus_cy)
    return 2.0 * asinh(sqrt_t) / sqrt_c


def _dist_mobius(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Möbius distance via addition."""
    sqrt_c = jnp.sqrt(c)
    diff = _addition(-x, y, c)
    # `safe_sqrt`, not `safe_norm`: the argument is a **ball point** (or a difference of two),
    # so `sum(.**2) <= 4/c` and the max-scaling `safe_norm` pays a second reduction for is
    # unreachable here. `safe_sqrt`'s double-`where` supplies the same finite (zero) derivative
    # at 0 that the old `+ MIN_NORM**2` did, without its 1e-15 floor on the value.
    # `diff` is `_addition`'s projected output, so it is a ball point. Not a divisor, so no floor.
    diff_norm = safe_sqrt(jnp.sum(diff**2))
    dist_c = atanh(sqrt_c * diff_norm)
    return 2 * dist_c / sqrt_c


def _apollonian_dist(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
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
    diff_sqnorm = jnp.sum((x - y) ** 2)
    # 1 - c‖x‖² and 1 - c‖y‖² (= 2/λ) with `_conformal_factor`'s boundary clamp, so the
    # near-boundary floor matches the rest of the module (δ → ∞ as y → ∂ball is expected). Taken
    # directly rather than as `2/λ`, whose two roundings left δ(x, x) one ulp off 0 (2.2e-16 in
    # float64 at scaled radius 3, 1.2e-7 in float32 at 12).
    floor_b = _boundary_floor(x, c)
    one_minus_cx = floor_at(1.0 - c * jnp.dot(x, x, precision=MATMUL_PRECISION), floor_b)
    one_minus_cy = floor_at(1.0 - c * jnp.dot(y, y, precision=MATMUL_PRECISION), floor_b)
    # G = |c·x·ȳ - 1| generalized to ℝⁿ, i.e. G² = c²‖x‖²‖y‖² - 2c⟨x,y⟩ + 1, spelled as
    # (1 - c‖x‖²)(1 - c‖y‖²) + c‖x - y‖² (expand both to check): for c > 0 a sum of two
    # non-negative terms, nothing cancels. The Gram-determinant form (1 - c⟨x,y⟩)² +
    # c²(‖x‖²‖y‖² - ⟨x,y⟩²) used before only looked that way: its Gram determinant is itself a
    # difference of O(1) terms that cancels for a close pair, and near the boundary G is
    # O(1 - c‖x‖²), so float32 came back off by eps/(1 - c‖x‖²)² relative. Measured on a 0.05-nat
    # pair (|δ| ≈ 0.03, max over 20 directions, float32 vs float64): 9.5e-4 absolute at scaled
    # radius 5.7 and 6.7e-2 at 8 before, 5.3e-6 and 5.3e-5 after. It also reuses ‖x - y‖² and
    # drops the ⟨x,y⟩ reduction (three instead of four). The sum is at least the squared floor, so
    # the sqrt needs no guard, and at x = y the two factors are bitwise equal: G is exactly the
    # denominator and δ(x, x) an exact 0.
    G = jnp.sqrt(one_minus_cx * one_minus_cy + c * diff_sqnorm)
    # `safe_sqrt`, not `jnp.linalg.norm`: at x == y the latter's VJP is 0/0 = NaN, and 0*NaN
    # survives every downstream operation, so grad(delta)(x, x) was NaN. Both arguments are ball
    # points, so the sum of squares cannot overflow and no max-scaling is needed.
    num = sqrt_c * safe_sqrt(diff_sqnorm) + G
    return jnp.log(num / one_minus_cy)


def _dist(
    x: Float[Array, "dim"],
    y: Float[Array, "dim"],
    c: ScalarCurvature,
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
    # Slot 2 (metric tensor) is the same function as slot 0 and runs its body; see _dist_mobius_direct.
    return lax.switch(version_idx, [_dist_mobius_direct, _dist_mobius, _dist_mobius_direct], x, y, c)


# Distance from origin implementations for lax.switch
def _dist_0_mobius(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Möbius distance from origin (mobius_direct and mobius use same formula)."""
    sqrt_c = jnp.sqrt(c)
    # `safe_sqrt`, not `safe_norm`: the argument is a **ball point** (or a difference of two),
    # so `sum(.**2) <= 4/c` and the max-scaling `safe_norm` pays a second reduction for is
    # unreachable here. `safe_sqrt`'s double-`where` supplies the same finite (zero) derivative
    # at 0 that the old `+ MIN_NORM**2` did, without its 1e-15 floor on the value.
    # The old `+ MIN_NORM**2` put a hard floor of 1e-15 on the radius, so d_0 of a float32 point at
    # radius 1e-15 came back 40 % too large and every radius below that collapsed onto one value.
    x_norm = safe_sqrt(jnp.sum(x**2))
    dist_c = atanh(sqrt_c * x_norm)
    return 2 * dist_c / sqrt_c


def _dist_0_metric_tensor(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Metric-tensor distance from origin, in the ``arcsinh`` form: ``2·arcsinh(√t)/√c``.

    The metric-tensor integral gives ``acosh(1 + 2t)/√c`` with ``t = c‖x‖²/(1 - c‖x‖²)``, which is
    what this arm used to evaluate. That spelling has the same defect as the hyperboloid's old
    ``acosh(√c·x₀)`` route (see ``hyperboloid._dist_0_stable``): the whole radial signal sits in the
    ``2t ≈ 2c‖x‖²`` perturbation of a leading 1, so ``r`` is stored to ``sqrt(eps)`` resolution at
    best, and ``math_utils.acosh``'s ``1 + 10·eps`` domain clamp pinned every float32 radius
    below ``sqrt(20·eps/(2c)) = sqrt(10·eps/c)`` ≈ 1.1e-3/√c to a constant floor (killing the
    gradient). The extra ``arg < 1 + MIN_NORM`` guard zeroed a second, smaller band below
    ``sqrt(MIN_NORM/(2c))`` ≈ 2.2e-8/√c.

    The half-angle identity ``acosh(1 + 2t) = 2·arcsinh(√t)`` (both sides are ``2θ`` for the ``θ``
    with ``sinh²θ = t``, since ``cosh 2θ = 1 + 2 sinh²θ``) moves the radius out of the perturbed
    leading 1 and into an argument that is *linear* in ``r`` near the origin::

        √t = √c‖x‖ / sqrt(1 - c‖x‖²),   d₀(x) = 2·arcsinh(√t)/√c

    ``arcsinh`` needs no domain clamp (its argument is a norm over the whole real half-line) and its
    derivative is bounded by 1, so the acosh floor and the ``where``-guard are both gone rather than
    moved, and the arm reproduces slot 0's ``2·atanh(√c‖x‖)/√c`` to rounding at every radius. The
    two forms are algebraically the same function, so this is **not** a new version slot — slot 2 is
    the metric-tensor *distance*, and it is now computed accurately.

    The boundary guard is unchanged: ``1 - c‖x‖²`` still comes from the clamped conformal factor
    (= 2/λ(x)), so a point at or past the ball boundary gets the same floored denominator, and the
    representable ceiling stays 2·arcsinh(1/sqrt(floor))/√c ≈ 12.1/√c (float32) / 27.8/√c (float64).

    ``safe_norm`` supplies the exactly-zero value *and* exactly-zero VJP at the origin (the same
    reason ``hyperboloid._dist_0_stable`` uses it), so ``jax.grad`` is finite there without the
    ``where``-guard that used to pin it, and ``d₀(0) = 0`` stays exact.
    """
    sqrt_c = jnp.sqrt(c)
    x_norm = safe_norm(x)
    # 1 - c||x||² via the boundary-clamped conformal factor (= 2/λ(x)) so a near-boundary point
    # cannot drive the denominator to 0; the same floor as _dist_mobius_direct and _apollonian_dist.
    one_minus_cx = 2.0 / _conformal_factor(x, c)
    sqrt_t = sqrt_c * x_norm / jnp.sqrt(one_minus_cx)
    return 2.0 * asinh(sqrt_t) / sqrt_c


def _dist_0(x: Float[Array, "dim"], c: ScalarCurvature, version_idx: int = VERSION_MOBIUS_DIRECT) -> Float[Array, ""]:
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


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
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
    # One reduction: `safe_sqrt(sum(v**2))` inside the `floor_at`. `v` is a tangent vector and can
    # be exactly zero, so `safe_sqrt` (double-`where`) rather than a plain `sqrt`, whose infinite
    # derivative at 0 would meet a zero cotangent as NaN. The floor is deliberate and stays
    # *around* the sqrt (`c_norm_prod` divides two lines below, and `tanh(t)/t → 1` needs the same
    # floored t in numerator and denominator). `sum(v**2)` overflows float32 past coordinate
    # 1.8e19, and a tangent vector that long comes from an already-diverging network; there the
    # norm is `inf`, `tanh(inf)/inf * v` is the zero vector and `x ⊕ 0 = x`, so the result is the
    # base point `x` (documented in the changelog). `_expmap_0` keeps the two-pass norm, so it
    # stays finite past that coordinate and returns its boundary clamp instead.
    # `axis=-1, keepdims=True` keeps the norm
    # broadcastable against the `(..., dim)` operand it divides below (see _gyrovector_core._proj).
    v_norm = floor_at(safe_sqrt(jnp.sum(v**2, axis=-1, keepdims=True)), MIN_NORM)
    c_norm_prod = jnp.sqrt(c) * v_norm
    lambda_x = _conformal_factor(x, c)
    # ||second_term|| = |tanh(·)|/√c < 1/√c, i.e. strictly inside the ball, so an explicit _proj
    # here is a no-op in the valid regime; _addition re-projects its output regardless. Skip it.
    # `tanh` is the math_utils wrapper, matching _expmap_0 and _scalar_mul: it is more accurate
    # than XLA's kernel (expm1 form above the dtype seam, odd Maclaurin series below it in float32)
    # and its ±(1 - 10·eps) output clip only tightens the strict inequality above.
    second_term = tanh(c_norm_prod * lambda_x / 2) / c_norm_prod * v
    res = _addition(x, second_term, c)
    return res


def _expmap_0(v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Exponential map from origin: map tangent vector v at origin to manifold.

    Args:
        v: Tangent vector at origin, shape (dim,)
        c: Curvature (positive)

    Returns:
        Point exp_0(v), shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # `safe_norm` + `floor_at`. `v` is a *tangent* vector, so unlike the ball-point sites in this
    # module its magnitude is unbounded and the max-scaling earns its second reduction: the old
    # `sum(v**2)` overflowed float32 above coordinate 1.8e19. The floor is deliberate --
    # `c_norm_prod` divides three lines down, and `tanh(t)/t -> 1` needs numerator and denominator
    # to be the same floored quantity. `[..., None]` keeps the norm broadcastable against the
    # `(..., dim)` operand it divides below (see _gyrovector_core._proj). The two-pass form is not
    # free: in isolation the max-scaled `safe_norm` costs about 1.2-1.35x the one-pass
    # `safe_sqrt(sum(v**2))` (forward, float32, measured on an A100 with jax 0.9.1; no controlled
    # H100 A/B of this site exists), and at most ~1 % inside a training step for this site. It
    # stays because `v` is a tangent vector of unbounded magnitude, so the full-range guarantee is
    # worth a second reduction here, unlike the ball-point sites. `_expmap` is the one that
    # regressed and is converted.
    v_norm = floor_at(safe_norm(v)[..., None], MIN_NORM)
    sqrt_c = jnp.sqrt(c)
    c_norm_prod = sqrt_c * v_norm
    # Boundary clamp applied to the *scalar* instead of via _proj on the (dim,) result. The result
    # is ‖res‖ = (t/c_norm_prod)·‖v‖ ≤ t/√c because v_norm ≥ ‖v‖, so capping t at √c·max_norm
    # already guarantees ‖res‖ ≤ max_norm — exactly _proj's postcondition — while a trailing
    # _proj(res, c) would re-reduce ‖res‖ over the op's own output, i.e. one extra XLA reduction
    # kernel over (B, dim)-sized data under jit(vmap).
    # `tanh` is the clamped wrapper (input clip at ±0.5·log(2/eps), output clip at 1 - 10·eps). At
    # c ≈ 1 that output bound sits *above* √c·max_norm = 1 - √c·eps**0.75, so the `minimum` is what
    # binds; where it does not (very small c) the wrapper only shrinks t further, which can never
    # break the ‖res‖ ≤ max_norm bound. _max_norm takes c_norm_prod only for its dtype — the dtype
    # the removed _proj would have seen on `res`.
    t = cap_at(tanh(c_norm_prod), sqrt_c * _max_norm(c_norm_prod, c))
    res = t / c_norm_prod * v
    return res


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
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


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
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
    # `safe_sqrt`: exact 0 and exactly-zero VJP at x == y. Identical quantity and form to
    # _dist_mobius_direct's num -- keep the two consistent. No floor here: `c_norm_prod` below
    # already carries the explicit divisor floor.
    num = safe_sqrt(jnp.sum((y - x) ** 2))
    x_sqnorm = jnp.dot(x, x, precision=MATMUL_PRECISION)
    y_sqnorm = jnp.dot(y, y, precision=MATMUL_PRECISION)
    # ‖(-x) ⊕ y‖ = ‖y - x‖/√D₋ with the factored denominator: the normalization of `sub`. The
    # norms are reduced once and shared with it and with B_x, B_y below.
    denom = jnp.sqrt(_mobius_denominator(x, y, c, sign=-1, x_sqnorm=x_sqnorm, y_sqnorm=y_sqnorm))
    sub_norm = num / denom
    c_norm_prod = floor_at(jnp.sqrt(c) * sub_norm, MIN_NORM)
    # B_x = 1 - c‖x‖² = 2/λ_x and B_y, with `_conformal_factor`'s boundary clamp.
    floor_b = _boundary_floor(x, c)
    one_minus_cx = floor_at(1.0 - c * x_sqnorm, floor_b)
    one_minus_cy = floor_at(1.0 - c * y_sqnorm, floor_b)
    # The magnitude √c·d/2 = atanh(u), u = `c_norm_prod`, in its `asinh` form: 1 - u² = B_x·B_y/D₋
    # (because D₋ = B_x·B_y + c‖x - y‖²), so atanh(u) = asinh(u·√D₋/√(B_x·B_y)) — which for u
    # unfloored is asinh(√c‖y - x‖/√(B_x·B_y)), slot 0's `dist` argument. For far pairs u → 1 and
    # `atanh` hit its float32 domain clip: ‖logmap‖ came back 4.7e-3 (two points on opposite sides
    # at scaled radius 7.2 each, c = 1) and 0.20 (at 9 each) relative too short; 2.9e-5 and 2.6e-5
    # after. The argument is formed from the *floored* u, so the ratio asinh(·)/u below is 1 at
    # y = x, as atanh(u)/u was, and the Jacobian ∂logmap_x(y)/∂y there stays the identity. Built
    # from the unfloored √c‖y - x‖ instead, the value would still be right (0) but that Jacobian
    # would be 0.
    half_dist = asinh(c_norm_prod * denom / jnp.sqrt(one_minus_cx * one_minus_cy))
    # 2·half_dist/(u·λ_x) · sub with 2/λ_x = B_x.
    res = one_minus_cx * half_dist / c_norm_prod * sub
    return res


def _logmap_0(y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Logarithmic map from origin: map point y to tangent space at origin.

    Args:
        y: Poincaré ball point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Tangent vector log_0(y), shape (dim,)

    References:
        Ganea et al. "Hyperbolic neural networks." NeurIPS 2018.
    """
    # `safe_sqrt` + `floor_at`: `y` is a ball point (`sum(y**2) <= 1/c`), so the max-scaling
    # `safe_norm` would pay a second reduction for an overflow that cannot happen. The floor is
    # deliberate -- `c_norm_prod` divides on the next line, and `atanh(t)/t -> 1` needs numerator
    # and denominator to be the same floored quantity. `axis=-1, keepdims=True` keeps the norm
    # per-row and broadcastable against `y`; see _scalar_mul.
    y_norm = floor_at(safe_sqrt(jnp.sum(y**2, axis=-1, keepdims=True)), MIN_NORM)
    c_norm_prod = jnp.sqrt(c) * y_norm
    res = atanh(c_norm_prod) / c_norm_prod * y
    return res


def _ptransp(
    v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, "dim"]:
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


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
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


def _tangent_inner(
    u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, ""]:
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
    return lambda_x**2 * jnp.dot(u, v, precision=MATMUL_PRECISION)


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
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
    # `safe_norm`: exact 0 with an exactly-zero VJP at v = 0 (a bare jnp.linalg.norm has VJP
    # 0/0 = NaN there). Returned, not divided by, so no floor — ‖0‖_x is exactly 0. Kept on the
    # two-pass form: nothing measured slower here, so the full-range guarantee costs nothing.
    return lambda_x * safe_norm(v)


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
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


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
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


def _is_in_manifold(x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
    """Check if point x lies in Poincaré ball.

    The constraint is tested in its dimensionless form ``c‖x‖² < 1`` rather than as
    ``‖x‖² < 1/c``, so one tolerance means the same thing at every curvature (``1/c`` spans
    orders of magnitude; the residual ``c‖x‖² - 1`` does not).

    Args:
        x: Point to check, shape (dim,)
        c: Curvature (positive)
        atol: Absolute tolerance on the dimensionless residual ``c‖x‖² - 1``. ``None``
            resolves to :func:`~hyperbolix.manifolds._base.default_atol` for ``x.dtype``.

    Returns:
        True if ``c‖x‖² < 1 + atol``

    Notes:
        The slack is what makes the check agree with ``_proj``: projection clamps the norm to
        ``1/√c - eps**0.75``, and re-squaring that in float32 can land a hair above ``1/c``.
        A point genuinely outside the ball misses by far more than ``atol``.
    """
    x_sqnorm = jnp.dot(x, x, precision=MATMUL_PRECISION)
    tol = default_atol(x.dtype) if atol is None else atol
    return c * x_sqnorm < 1.0 + tol


def _is_in_tangent_space(
    v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
) -> Array:
    """Check if vector v lies in tangent space at point x.

    The ball is an open subset of R^d, so its tangent space at every point is all of R^d and
    the only thing to check is that ``v`` is finite. (This used to return the constant ``True``,
    which accepted NaN and Inf — the same defect fixed for ``Euclidean.is_in_manifold``.)

    Args:
        v: Vector to check, shape (dim,)
        x: Poincaré ball point (ignored — the tangent space does not depend on it), shape (dim,)
        c: Curvature (ignored, kept for consistency)
        atol: Accepted for signature uniformity across manifolds; a finiteness test has no
            tolerance to slacken, so it is unused.

    Returns:
        True iff every entry of v is finite.
    """
    del x, c, atol
    return jnp.all(jnp.isfinite(v))


# ---------------------------------------------------------------------------
# Batch-compatible helpers (used by NN layers)
# ---------------------------------------------------------------------------


def _compute_mlr_pp(
    x: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim"],
    r: Float[Array, "out_dim 1"],
    c: ScalarCurvature,
    min_enorm: float = 1e-15,
) -> Float[Array, "batch out_dim"]:
    """Compute HNN++ multinomial linear regression on the Poincare ball.

    Args:
        x: Poincare ball point(s), shape (batch, in_dim)
        z: Hyperplane tangent normals at origin, shape (out_dim, in_dim)
        r: Hyperplane translations, shape (out_dim, 1)
        c: Manifold curvature (positive)
        min_enorm: Minimum norm to avoid division by zero

    Returns:
        MLR scores, shape (batch, out_dim)

    References:
        Shimizu et al. "Hyperbolic neural networks++." arXiv:2006.08210 (2020).
    """
    sqrt_c = jnp.sqrt(c)
    sqrt_c2r_1P = 2 * sqrt_c * r.T  # (1, P) — r is (P, 1), .T broadcasts

    # `safe_norm` + `floor_at`: `z_norm_P1` is a divisor below (`z / z_norm_P1`), so the floor at
    # `min_enorm` is the deliberate part; the max-scaling removes the old spelling's float32
    # overflow and its 1e-15 floor on small hyperplane normals. Mirrors
    # `poincare_regression._compute_mlr`'s `floor_at(safe_norm(a)[:, None], min_enorm)`.
    z_norm_P1 = floor_at(safe_norm(z)[:, None], min_enorm)  # (P, 1)

    # Conformal factor lam(x) = 2 / (1 - c||x||²) per HNN++ Eq. 26 (boundary-clamped).
    # NOTE: van Spengler's poincare-resnet repo has 2*(1 - c||x||²) here — a
    # transcription bug the same author fixed in hypll. Do not "restore" it.
    lam_B1 = _conformal_factor_batch(x, c)  # (B, 1)

    # Pinned HIGHEST: the MLR logits are a decision quantity, and this dot enters the asinh
    # argument as a difference of a radial and an angular term. Measured f32-vs-f64 relative
    # error of compute_mlr_pp on an A100: 3.8e-5 … 1.4e-4 under TF32, 6.0e-8 … 2.2e-7 here.
    z_unitx_BP = jnp.einsum("bi,oi->bo", x, z / z_norm_P1, precision=MATMUL_PRECISION)  # (B, P)
    asinh_arg_BP = sqrt_c * lam_B1 * z_unitx_BP * cosh(sqrt_c2r_1P) - (lam_B1 - 1) * sinh(sqrt_c2r_1P)  # (B, P)

    # No clamp on the asinh argument — same reason as in `manifolds/hyperboloid._compute_mlr`;
    # see there. λ(x) puts this argument past the old float32 bound sooner than the hyperboloid
    # one does: λ ≈ 75 already at geodesic radius 5, c = 1.
    signed_dist2hyp_BP = asinh(asinh_arg_BP) / sqrt_c  # (B, P)
    res_BP = 2 * z_norm_P1.T * signed_dist2hyp_BP  # z_norm.T broadcasts (1, P) over (B, P)
    return res_BP


def _compute_mlr_pp_tangent(
    v: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim"],
    r: Float[Array, "out_dim 1"],
    c: ScalarCurvature,
    min_enorm: float = 1e-15,
) -> Float[Array, "batch out_dim"]:
    """:func:`_compute_mlr_pp` of ``x = expmap_0(v)``, evaluated on the tangent vector ``v`` itself.

    With ``t = √c‖v‖`` the point ``x`` sits at scaled radius ``2t``, and both conformal-factor
    terms of the HNN++ argument are closed-form in ``t``::

        λ_x·√c·⟨x, ẑ⟩ = sinh(2t)·⟨v, ẑ⟩/‖v‖,        λ_x - 1 = cosh(2t)

    so the ball point is never formed. That route read ``λ_x = 2/(1 - c‖x‖²)`` back off the stored
    point, a relative error ≈ ``eps·e^{2t}/4``, and in float32 its lift stopped at the ball's
    ceiling ``t ≈ 6.33``, past which the scores were constant with a zero radial gradient.
    Measured through ``HypRegressionPoincarePP`` (float32 vs float64 on the same inputs, relative
    to the largest entry, c ∈ {0.3, 1}): at t = 6 scores 1.1e-3 → 8.4e-7 and input gradients
    1.4e-3 → 1.3e-5; at t = 8 2.0e-1 → 1.5e-6 and 3.8e-1 → 2.8e-5. What remains is the one float32
    rounding of ``⟨v, ẑ⟩``, amplified by ≈ ``e^{2t}/2`` in a cell near its hyperplane — the
    point-representation floor, which the ball route shared.

    Args:
        v: Tangent vector(s) at the origin, shape (batch, in_dim)
        z: Hyperplane tangent normals at origin, shape (out_dim, in_dim)
        r: Hyperplane translations, shape (out_dim, 1)
        c: Manifold curvature (positive)
        min_enorm: Minimum norm to avoid division by zero

    Returns:
        MLR scores, shape (batch, out_dim)

    References:
        Shimizu et al. "Hyperbolic neural networks++." arXiv:2006.08210 (2020).
    """
    sqrt_c = jnp.sqrt(c)
    sqrt_c2r_1P = 2 * sqrt_c * r.T  # (1, P) — r is (P, 1), .T broadcasts
    z_norm_P1 = floor_at(safe_norm(z)[:, None], min_enorm)  # (P, 1), as in `_compute_mlr_pp`

    # The only reduction over the input. `safe_norm`, for the reason `_expmap_0` gives: `v` is a
    # tangent vector of unbounded magnitude. The route through the ball paid the same two passes
    # inside `_expmap_0` plus λ's `sum(x**2)`. The floor sits on `v_norm` and the sinhc below
    # divides by that same floored quantity, so `sinh(2t)/‖v‖ = 2√c` at v = 0 and the Jacobian
    # there is 4·z, as through the ball.
    v_norm_B1 = floor_at(safe_norm(v)[:, None], MIN_NORM)  # (B, 1)
    two_t_B1 = 2 * sqrt_c * v_norm_B1  # (B, 1): scaled radius of expmap_0(v)

    # Pinned HIGHEST: the decision quantity of `_compute_mlr_pp`'s einsum, now taken on v.
    z_unitv_BP = jnp.einsum("bi,oi->bo", v, z / z_norm_P1, precision=MATMUL_PRECISION)  # (B, P)
    asinh_arg_BP = (sinh(two_t_B1) / v_norm_B1) * z_unitv_BP * cosh(sqrt_c2r_1P) - cosh(two_t_B1) * sinh(sqrt_c2r_1P)  # (B, P)

    # No clamp on the asinh argument, as in `_compute_mlr_pp`. Both terms grow like
    # e^{2t + 2√c|r|}/4: their product overflows float32 once that exponent passes ≈ 89, far past
    # anything the float32 ball held (2t ≈ 12.7), and the score is then ±inf or NaN. Short of that,
    # `2t` past the sinh/cosh argument clip (87.8) saturates with a zero radial gradient.
    signed_dist2hyp_BP = asinh(asinh_arg_BP) / sqrt_c  # (B, P)
    res_BP = 2 * z_norm_P1.T * signed_dist2hyp_BP  # z_norm.T broadcasts (1, P) over (B, P)
    return res_BP


# ---------------------------------------------------------------------------
# Beta-concatenation (HNN++, Shimizu et al. 2020)
# ---------------------------------------------------------------------------


def _beta_concat(points: Float[Array, "M n_i"], c: ScalarCurvature) -> Float[Array, "n"]:
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


def _busemann(x: Float[Array, "dim"], v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
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
    # `_boundary_floor`, not `MIN_NORM`: this is the `1 - c‖x‖²` of `_conformal_factor`, and the
    # docstring above has always claimed the same floor. `MIN_NORM = 1e-15` sits below the analytic
    # minimum of this quantity in both dtypes (1.3e-5 float32 / 3.6e-12 float64 at c = 1), so it
    # never bit: a float32 point at the ball ceiling could reach the divisor with pure rounding
    # noise and return a Busemann coordinate tens of nats too large.
    denom = floor_at(1.0 - c * jnp.dot(x, x, precision=MATMUL_PRECISION), _boundary_floor(x, c))
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

    def proj(self, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Project point onto Poincaré ball by clipping norm."""
        return _proj(self._cast(x), c)

    def proj_batch(self, x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... dim"]:
        """Project batched points onto the ball (handles arbitrary leading dimensions).

        Batched sibling of :meth:`proj`, matching ``Hyperboloid.proj_batch``. Equivalent to
        ``jax.vmap(proj, in_axes=(0, None))`` on a 2D input, without materializing the vmap.
        """
        return _proj_batch(self._cast(x), c)

    def gyration(
        self, x: Float[Array, "dim"], y: Float[Array, "dim"], z: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, "dim"]:
        """Compute gyration gyr[x,y]z to restore commutativity."""
        return _gyration(self._cast(x), self._cast(y), self._cast(z), c)

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Möbius gyrovector addition x ⊕ y."""
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float | Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Scalar multiplication r ⊗ x on Poincaré ball."""
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r_cast, x, c)  # type: ignore[arg-type]

    def dist(
        self,
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: ScalarCurvature,
        version_idx: int = VERSION_MOBIUS_DIRECT,
    ) -> Float[Array, ""]:
        """Compute geodesic distance between Poincaré ball points."""
        return _dist(self._cast(x), self._cast(y), c, version_idx)

    def dist_0(self, x: Float[Array, "dim"], c: ScalarCurvature, version_idx: int = VERSION_MOBIUS_DIRECT) -> Float[Array, ""]:
        """Compute geodesic distance from Poincaré ball origin."""
        return _dist_0(self._cast(x), c, version_idx)

    def apollonian_dist(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
        """Apollonian weak metric δ(x, y) — non-symmetric; symmetrizes to √c·dist(x, y).

        .. warning::
            Although ``δ`` is non-symmetric, its antisymmetric part is an exact **coboundary**
            (``δ(x, y) - δ(y, x)`` is a difference of a per-point potential), so it carries
            **no circulation** and is useless as a quasimetric energy. Do not reach for this
            expecting genuine asymmetry — use a :meth:`busemann` coordinate fed to an external
            quasimetric combinator (IQE/MRN) instead.
        """
        return _apollonian_dist(self._cast(x), self._cast(y), c)

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map: map tangent vector v at point x to manifold."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map from origin: map tangent vector v at origin to manifold."""
        return _expmap_0(self._cast(v), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Retraction: first-order approximation of exponential map."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map: map point y to tangent space at point x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map from origin: map point y to tangent space at origin."""
        return _logmap_0(self._cast(y), c)

    def ptransp(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, "dim"]:
        """Parallel transport tangent vector v from point x to point y."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Parallel transport tangent vector v from origin to point y."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    def tangent_inner(
        self, u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, ""]:
        """Compute inner product of tangent vectors u and v at point x."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
        """Compute norm of tangent vector v at point x."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Convert Euclidean gradient to Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Project vector v onto tangent space at point x."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
        """Check if point x lies in Poincaré ball (``atol`` default: :func:`default_atol`)."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
    ) -> Array:
        """Check that v has finite entries (T_x B = R^d, so there is no other constraint)."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c, atol)

    def conformal_factor(self, x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... 1"]:
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
        c: ScalarCurvature,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """Compute HNN++ multinomial linear regression on the Poincare ball."""
        return _compute_mlr_pp(self._cast(x), self._cast(z), self._cast(r), c, min_enorm)

    def _compute_mlr_pp_tangent(
        self,
        v: Float[Array, "batch in_dim"],
        z: Float[Array, "out_dim in_dim"],
        r: Float[Array, "out_dim 1"],
        c: ScalarCurvature,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """``compute_mlr_pp(expmap_0(v), z, r, c)`` from the tangent vector ``v``, without the ball point.

        Private: the tangent-input path of the HNN++ layers (see :func:`_compute_mlr_pp_tangent`).
        """
        return _compute_mlr_pp_tangent(self._cast(v), self._cast(z), self._cast(r), c, min_enorm)

    def beta_concat(self, points: Float[Array, "M n_i"], c: ScalarCurvature) -> Float[Array, "n"]:
        """Beta-concatenation of M equal-dimensional Poincaré ball points."""
        return _beta_concat(self._cast(points), c)

    def busemann(self, x: Float[Array, "dim"], v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
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
