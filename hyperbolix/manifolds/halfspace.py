"""Poincaré upper half-space manifold - class-based API with dtype control.

Provides a HalfSpace class for manifold operations with automatic dtype casting.
All operations work on single points with shape (dim,). Use jax.vmap for batching.

Convention: points ``x = (x_s, x_n)`` with ``x_s ∈ R^{n-1}`` and height ``x_n > 0`` (the last coordinate),
sectional curvature ``-c``, and the conformal metric::

    g_x(u, v) = (u·v)/(c·x_n²)

The origin is ``o = e_n/√c`` (spelled ``o_n = 1/sqrt(c)`` by every ``_0`` op, and every ``_0`` op
is the general op evaluated at ``o``, so ``dist_0(o) = 0``, ``logmap_0(o) = 0`` and
``expmap_0(0) = o`` exactly). Geodesics are vertical lines and half-circles orthogonal to the
boundary ``x_n = 0``. In chart coordinates the exponential map, logarithmic map and parallel
transport do not depend on ``c``; only the origin, distances, norms and inner products do.

Isometry to the hyperboloid ``-X₀² + ‖X_s‖² + X_n² = -1/c``::

    X = ((‖x‖² + 1/c)/(2x_n), x_s/(√c·x_n), (‖x‖² - 1/c)/(2x_n)),   x_n = 1/(c(X₀ - X_n)),  x_s = √c·x_n·X_s

The gyrovector structure is the Möbius structure of the Poincaré ball carried over by the Cayley
transform (the ball isometry mapping ``0`` to ``o``): ``x ⊕ y = exp_x(PT_{o→x}(log_o y))``,
``r ⊗ x = exp_o(r·log_o x)``, and the gyro-inverse is ``⊖x = (-x_s, x_n)/(c‖x‖²)``.

Numerics
--------
Every two-point quantity is built from ``w = y - x`` (exact by Sterbenz for close points) and
the scaled chord ``r = (w/√x_n)/√y_n``, whose half-norm is ``sinh(√c·d/2)``. The divisions are
sequential and ``x_n·y_n`` or ``x_n²`` is never formed, so ``r`` itself neither under- nor overflows
for heights anywhere in the dtype's normal range. ``‖r‖²`` is a plain sum of squares, which sets the
pairwise ceiling: it overflows once the chord passes ``‖r‖ ≈ √(max float)`` — ``1.8e19`` in float32,
i.e. ``√c·d ≈ 88.7`` (float64: ``1.3e154``, ``√c·d ≈ 709.8``) — and past it ``dist`` returns ``inf``
and ``logmap``, ``gyro_difference`` (and ``ptransp`` once ``‖w_s‖/(x_n + y_n)`` overflows its square)
return inf/NaN, never a finite wrong value. Measured in float32 at ``c = 1`` against a 80-digit oracle
evaluated at the stored inputs, ``dist``, ``logmap`` and ``ptransp`` have median relative error
2-7e-8 at every radius up to 12 and separation down to 1e-5, and ``expmap`` stays within ~2.5
rounding floors for every step direction.

The literal forms of the reference implementation (HTorch) cancel. ``arcosh(1 + ‖w‖²/(2x_n y_n))``
returns 0 (relative error 1.0) at float32 separation 1e-5 and 2e-2 at 1e-3 — XLA:CPU's ``jit``
happens to rewrite it into an accurate form, XLA:GPU's does not. Its ``logmap``,
``x_n(θ·coth θ - (θ/sinh θ)(x_n/y_n))``, has median relative error 2e-3 to 1.2e-2 at float32
separation 1e-5 and is NaN at float64 separation 1e-9. Its ``expmap`` denominator
``θ·coth θ - p_n`` cancels for near-vertical upward steps: at float32 ``θ = 10`` it is NaN for
``φ ≤ 1e-4`` off vertical and 0.07 nats wrong at ``φ = 1e-2``. Its ``sqdist`` clamps ``d²`` at 50,
and its ``ptransp``/``mobius_add`` are wrong at any precision (:func:`_ptransp`, :func:`_addition`).

Storage floor: a stored point's own rounding, as a distance, is ``≈ 0.4·(eps/2)·cosh(√c·δ)/√c``
with ``cosh(√c·δ) = ‖x‖/x_n`` and ``δ`` the distance to the vertical geodesic through ``o``. On that
axis (``x_s = 0``) the floor stays at ``≈ 0.4·(eps/2)/√c`` for every height in the dtype's normal
range, so storage and the single-point ``_0`` ops (``dist_0``, ``logmap_0``) have no radius ceiling
along it (float32 heights ``1.2e-38`` to ``1.7e38``: relative error ≤ 1.3e-7, measured); two points on the axis are still
limited by the pairwise ceiling above (their separation ``ln(y_n/x_n)`` reaches 88.7 before the
heights leave the normal range). Off the axis the floor grows as ``cosh(√c·δ)``, like the
hyperboloid's ``eps·sinh(a)`` (float32 at ``√c·δ = 10``: ≈ 2.6e-4 nats).

Dimension key:
    D: manifold dimension (``dim``); the height is the last coordinate

References
----------
Cannon, Floyd, Kenyon, Parry. "Hyperbolic Geometry." Flavors of Geometry, MSRI Publ. 31, 1997.
Yu & De Sa. HTorch (github.com/ydtydr/HTorch), ``manifolds/halfspace.py``.
Ungar. "A Gyrovector Space Approach to Hyperbolic Geometry." 2009.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import MIN_NORM, asinh, cosh, floor_at, safe_hypot, safe_sqrt, sinh
from ..utils.precision import MATMUL_PRECISION
from ._base import ManifoldBase
from .hyperboloid import _asinhc
from .protocol import ScalarCurvature

# Version selection constant. HalfSpace has a single canonical implementation, kept for API
# consistency with Poincare / Hyperboloid.
VERSION_DEFAULT = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _origin_like(x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... dim"]:
    """The origin ``o = e_n/√c`` with the shape and dtype of ``x`` (the one spelling of ``o``)."""
    o_n = 1.0 / jnp.sqrt(jnp.asarray(c, dtype=x.dtype))
    return jnp.zeros_like(x).at[..., -1].set(o_n)


def _proj(x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
    """Floor the height at the smallest normal number; every valid point is left bit-identical.

    The reference clamps ``x_n`` at an absolute ``1e-7`` (float32) / ``1e-15`` (float64), which
    moves valid points: ``(0, 1e-8)`` sits 18.4 nats from ``o`` at ``c = 1``. A NaN height passes
    through (``floor_at`` compares with ``<``).
    """
    return x.at[..., -1].set(floor_at(x[..., -1], jnp.finfo(x.dtype).tiny))


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


@jax.custom_jvp
def _chord_sq(w: Float[Array, "dim"], x_n: Float[Array, "1"], y_n: Float[Array, "1"]) -> Float[Array, ""]:
    """Squared scaled chord ``‖r‖² = ‖w‖²/(x_n·y_n)``, as ``Σ r²`` with ``r = (w/√x_n)/√y_n`` (no ``x_n·y_n``).

    The JVP ``2·r·ṙ - ‖r‖²·(ẋ_n/x_n + ẏ_n/y_n)`` hands the heights their derivative through the
    reduced ``‖r‖²`` instead of through every ``r_i``, so reverse mode needs no second reduction over
    the row (autodiff of the plain spelling reduces ``r̄·r`` again for each height). Same value; the
    tangent is exactly 0 at ``w = 0``. Overflows to ``inf`` once ``‖r‖ > √(max float)``.
    """
    r_D = (w / jnp.sqrt(x_n)) / jnp.sqrt(y_n)
    return jnp.sum(r_D * r_D, axis=-1)


@_chord_sq.defjvp
def _chord_sq_jvp(primals, tangents):
    w, x_n, y_n = primals
    dw, dx_n, dy_n = tangents
    r_D = (w / jnp.sqrt(x_n)) / jnp.sqrt(y_n)
    r2 = jnp.sum(r_D * r_D, axis=-1)
    dr_D = (dw / jnp.sqrt(x_n)) / jnp.sqrt(y_n)
    return r2, 2.0 * jnp.sum(r_D * dr_D, axis=-1) - r2 * (dx_n / x_n + dy_n / y_n)[..., 0]


def _dist(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Geodesic distance ``(2/√c)·arsinh(‖r‖/2)``, ``r = ((y - x)/√x_n)/√y_n``.

    The same function as the literal ``arcosh(1 + ‖x - y‖²/(2x_n y_n))/√c`` via
    ``sinh(θ/2)² = (cosh θ - 1)/2``, but ``arsinh`` of the half-chord keeps full relative accuracy
    for close pairs, where the literal form's ``1 + tiny`` rounds the separation away. At ``x == y``
    ``r`` is exactly zero, and ``safe_sqrt`` gives ``0`` with a zero (finite) gradient.

    ``‖r‖²`` is a plain sum of squares (:func:`_chord_sq`): it overflows once the chord passes
    ``‖r‖ ≈ 1.8e19`` in float32 (``√c·d ≈ 88.6``; float64 ``1.3e154``, ``√c·d ≈ 709``), and past that
    the distance is ``inf`` rather than a finite wrong value.
    """
    c = jnp.asarray(c, dtype=x.dtype)
    return 2.0 * asinh(0.5 * safe_sqrt(_chord_sq(y - x, x[..., -1:], y[..., -1:]))) / jnp.sqrt(c)


def _dist_0(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Distance from the origin, :func:`_dist` at ``o``."""
    return _dist(_origin_like(x, c), x, c)


# ---------------------------------------------------------------------------
# Exp / log maps
# ---------------------------------------------------------------------------


@jax.custom_jvp
def _sq_over(v_s: Float[Array, "dim_s"], x_n: Float[Array, "1"]) -> Float[Array, "1"]:
    """``‖v_s/x_n‖²`` as a plain sum of squares, with a JVP that routes the height through the reduced value.

    JVP ``2·p_s·(v̇_s/x_n) - 2‖p_s‖²·ẋ_n/x_n`` (``p_s = v_s/x_n``): reverse mode needs no second
    reduction over the row for ``x̄_n``, and XLA no longer keeps ``p_s`` in memory between the forward
    and backward kernels (``expmap`` fwd+bwd on an A100: 1.2-1.5 → ~0.9-1.0 times the literal form).
    The tangent is exactly 0 at ``v_s = 0``.
    """
    p_s = v_s / x_n
    return jnp.sum(p_s * p_s, axis=-1, keepdims=True)


@_sq_over.defjvp
def _sq_over_jvp(primals, tangents):
    v_s, x_n = primals
    dv_s, dx_n = tangents
    p_s = v_s / x_n
    ps2 = jnp.sum(p_s * p_s, axis=-1, keepdims=True)
    return ps2, 2.0 * jnp.sum(p_s * (dv_s / x_n), axis=-1, keepdims=True) - 2.0 * ps2 * (dx_n / x_n)


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Exponential map, ``c``-independent in chart coordinates.

    With ``p = v/x_n``, ``θ = ‖p‖`` (the scaled step length ``√c·‖v‖_x``) and ``sc = sinh θ/θ``::

        E = cosh θ - sc·p_n,        exp_x(v) = (x_s + x_n·(sc·p_s)/E, x_n/E)

    (the reference's ``y_s = x_s + x_n·p_s/(θ·coth θ - p_n)`` is the same point). ``E`` cancels for
    near-vertical upward steps: ``E = cosh θ - sinh θ·cos φ`` with ``φ`` the angle to ``e_n``, which
    is ``e^{-θ}`` for ``φ = 0``. For ``p_n > 0`` and ``θ > 0.5`` it is evaluated as
    ``e^{-θ} + sinh θ·(1 - cos φ)`` with ``1 - cos φ = ‖p_s‖²/(θ(θ + p_n))``, a sum of non-negative
    terms. The ``θ > 0.5`` guard keeps tiny steps on the direct branch (where ``E`` has no
    cancellation, ``E ≥ e^{-0.5}``); without it the Jacobian at ``v = 1e-300·e_n`` is NaN in
    float64. The strict ``p_n > 0`` keeps ``v = 0`` off the rearranged branch. Double ``where`` on the
    denominator ``θ + p_n``.

    ``sc`` is evaluated from ``θ`` floored at ``MIN_NORM`` (value exactly 1 there), so ``v = 0``
    returns ``x`` with Jacobian exactly ``I`` in reverse mode and ``I`` to one ulp in forward mode
    (the tangent of ``p = v/x_n`` is ``fl(1/x_n)``, multiplied back by ``x_n``; the diagonal misses 1
    by an ulp at about one height in eight). The grouping ``(sc·p_s)/E`` matters: ``sc/E`` overflows
    float32 for an exactly vertical step from ``θ ≈ 50`` and ``inf·0`` would be NaN.

    ``θ² = ‖p_s‖² + p_n²`` with ``‖p_s‖²`` one plain sum of squares (:func:`_sq_over`; overflow only
    past ``θ ≈ 1.8e19`` in float32, far beyond ``sinh``'s own ``θ ≈ 89``), built from the same
    ``p_s = v_s/x_n`` the output uses, so ``θ`` and the step direction agree to the bit.
    """
    del c
    x_n = x[..., -1:]
    v_s = v[..., :-1]
    p_s = v_s / x_n
    p_n = v[..., -1:] / x_n
    ps2 = _sq_over(v_s, x_n)
    theta = safe_sqrt(ps2 + p_n * p_n)
    theta_f = floor_at(theta, MIN_NORM)
    sc = sinh(theta_f) / theta_f
    up = (p_n > 0) & (theta > 0.5)
    den = jnp.where(up, theta + p_n, jnp.ones_like(theta))
    e = jnp.where(up, jnp.exp(-theta) + sc * ps2 / den, cosh(theta) - sc * p_n)
    y_s = x[..., :-1] + x_n * ((sc * p_s) / e)
    return jnp.concatenate([y_s, x_n / e], axis=-1)


def _expmap_0(v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Exponential map from the origin, :func:`_expmap` at ``o``."""
    return _expmap(v, _origin_like(v, c), c)


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Retraction ``proj((x_s + v_s, x_n·exp(v_n/x_n)))``.

    A first-order retraction that never leaves the manifold and is exact for vertical steps (the
    vertical geodesic is ``x_n·e^t``). The plain ``x + v`` leaves the half-space for large downward
    steps (33% of random steps of length ``θ = 3``, measured).
    """
    del c
    x_n = x[..., -1:]
    y_n = x_n * jnp.exp(v[..., -1:] / x_n)
    return _proj(jnp.concatenate([x[..., :-1] + v[..., :-1], y_n], axis=-1))


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Logarithmic map, ``c``-independent in chart coordinates.

    With ``w = y - x``, ``r = (w/√x_n)/√y_n``, ``s = ‖r‖/2 = sinh(θ/2)`` and
    ``k = θ/sinh θ = arsinhc(s)/√(1 + s²)``::

        log_x(y) = ((k·(x_n/y_n))·w_s,  (k/2)·(x_n‖r_s‖² + w_n + (x_n/y_n)·w_n))

    The height component is the reference's ``x_n(θ·coth θ - (θ/sinh θ)(x_n/y_n))`` with
    ``cosh θ - x_n/y_n = w_n/y_n + ‖w‖²/(2x_n y_n)`` expanded, so it is built from ``w`` and never
    subtracts two O(1) terms. ``arsinhc`` is analytic at ``s = 0``, so ``y = x`` needs no special
    case: the Jacobian there is exactly ``I`` in ``y`` and ``-I`` in ``x``, in both forward and
    reverse mode. That exactness depends on the grouping: ``x_n`` enters only through the ratio
    ``x_n/y_n`` (exactly 1 at coincidence), never as ``x_n·(w/x_n)`` or ``(x_n·w)/y_n``, whose
    derivative rounds ``x_n·fl(1/x_n)`` and misses 1 by an ulp at ~12% of points.

    ``‖r‖² = ‖r_s‖² + r_n²`` is one plain reduction over the spatial slice, shared with the height
    component; past ``‖r‖ ≈ √(max float)`` (float32 ``√c·d ≈ 88.7``) it overflows and the result is NaN.
    """
    del c
    x_n = x[..., -1:]
    y_n = y[..., -1:]
    w_D = y - x
    w_s = w_D[..., :-1]
    w_n = w_D[..., -1:]
    r_s = (w_s / jnp.sqrt(x_n)) / jnp.sqrt(y_n)
    r_n = (w_n / jnp.sqrt(x_n)) / jnp.sqrt(y_n)
    rs2 = jnp.sum(r_s * r_s, axis=-1, keepdims=True)
    s = 0.5 * safe_sqrt(rs2 + r_n * r_n)
    k = _asinhc(s) / safe_hypot(jnp.ones_like(s), s)
    v_s = (k * (x_n / y_n)) * w_s
    v_n = 0.5 * k * (x_n * rs2 + w_n + (x_n / y_n) * w_n)
    return jnp.concatenate([v_s, v_n], axis=-1)


def _logmap_0(y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Logarithmic map at the origin, :func:`_logmap` at ``o``."""
    return _logmap(y, _origin_like(y, c), c)


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------


def _ptransp(
    v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, "dim"]:
    """Parallel transport along the geodesic from ``x`` to ``y``, ``c``-independent in chart coordinates.

    With ``ω = w_s/(x_n + y_n)``, ``k = 2/(1 + ‖ω‖²)`` and ``a = v_s·ω``::

        PT_{x→y}(v) = (y_n/x_n)·(v_s + k(v_n - a)·ω,  v_n - k(a + ‖ω‖²·v_n))

    the conformal scale ``y_n/x_n`` times the rotation by ``-2·atan(‖w_s‖/(x_n + y_n))`` in the
    vertical 2-plane through ``x`` and ``y`` (the angle between the geodesic's tangents at its two
    ends). It preserves the metric exactly and is the identity at ``x = y``.

    The reference (HTorch) transports with the ambient-hyperboloid formula
    ``v - (⟨log_x y, v⟩_x/d²)(log_x y + log_y x)`` applied to chart coordinates, which is not an
    isometry: at ``c = 1`` it maps ``(1, 0)`` at ``(0, 1)`` to ``(1, 0)`` at ``(0, 2)`` (norm 0.5;
    the transport is ``(2, 0)``).
    """
    del c
    x_n = x[..., -1:]
    y_n = y[..., -1:]
    omega = (y[..., :-1] - x[..., :-1]) / (x_n + y_n)
    o2 = jnp.sum(omega * omega, axis=-1, keepdims=True)
    k = 2.0 / (1.0 + o2)
    v_s = v[..., :-1]
    v_n = v[..., -1:]
    a = jnp.sum(v_s * omega, axis=-1, keepdims=True)
    out_s = v_s + (k * (v_n - a)) * omega
    out_n = v_n - k * (a + o2 * v_n)
    return (y_n / x_n) * jnp.concatenate([out_s, out_n], axis=-1)


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Parallel transport from the origin, :func:`_ptransp` at ``x = o``."""
    return _ptransp(v, _origin_like(y, c), y, c)


# ---------------------------------------------------------------------------
# Gyro-operations
# ---------------------------------------------------------------------------


def _addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Gyro addition ``x ⊕ y = exp_x(PT_{o→x}(log_o y))`` (Möbius addition through the Cayley map).

    The reference's ``mobius_add`` has this structure, but its transport is not an isometry
    (:func:`_ptransp`), which puts it off the Möbius sum: measured at ``c = 1`` in float64 against
    Cayley⁻¹(Möbius(Cayley)), 0.16-0.68 nats off for pairs within 1 nat of ``o`` and up to 5.3 nats
    (median 1.26) for random pairs.
    """
    o = _origin_like(x, c)
    return _expmap(_ptransp(_logmap(y, o, c), o, x, c), x, c)


def _gyro_difference(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Gyro difference ``(⊖x) ⊕ y = exp_o(PT_{x→o}(log_x y))``.

    Built from ``log_x y``, i.e. from ``w = y - x``, so two close points give a point close to ``o``
    without cancellation (the literal ``addition(⊖x, y)`` forms ``⊖x`` and loses the separation).
    """
    o = _origin_like(x, c)
    return _expmap(_ptransp(_logmap(y, x, c), x, o, c), o, c)


def _scalar_mul(r: Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Gyro scalar multiplication ``r ⊗ x = exp_o(r·log_o x)``."""
    o = _origin_like(x, c)
    return _expmap(r * _logmap(x, o, c), o, c)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def _tangent_inner(
    u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, ""]:
    """Metric ``g_x(u, v) = (u/x_n)·(v/x_n)/c`` (``x_n²`` is never formed, so it cannot under/overflow)."""
    c = jnp.asarray(c, dtype=x.dtype)
    x_n = x[-1]
    return jnp.dot(u / x_n, v / x_n, precision=MATMUL_PRECISION) / c


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """``‖v‖_x = ‖v/x_n‖/√c``: ``safe_sqrt`` of a plain sum of squares (exact 0 with zero VJP at ``v = 0``;
    ``inf`` once ``‖v/x_n‖`` passes ``√(max float)``, 1.8e19 in float32)."""
    c = jnp.asarray(c, dtype=x.dtype)
    p = v / x[..., -1:]
    return safe_sqrt(jnp.sum(p * p, axis=-1)) / jnp.sqrt(c)


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Riemannian gradient via the inverse metric, ``c·x_n·(x_n·grad)``."""
    c = jnp.asarray(c, dtype=x.dtype)
    x_n = x[..., -1:]
    return c * x_n * (x_n * grad)


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Identity: the half-space is an open subset of R^n, so ``T_x H = R^n``."""
    del x, c
    return v


def _is_in_manifold(x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
    """Every finite point with positive height lies on the half-space.

    ``atol`` is accepted for signature uniformity across manifolds; the constraint ``x_n > 0`` is
    open and scale-free, so there is no tolerance to slacken (as for ``ProperVelocity``).
    """
    del c, atol
    return jnp.all(jnp.isfinite(x)) & (x[-1] > 0)


def _is_in_tangent_space(
    v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
) -> Array:
    """``T_x H = R^n``, so the only check is that every entry of ``v`` is finite."""
    del x, c, atol
    return jnp.all(jnp.isfinite(v))


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class HalfSpace(ManifoldBase):
    """Poincaré upper half-space manifold (curvature ``-c``) with automatic dtype casting.

    Points are ``x = (x_s, x_n)`` with height ``x_n > 0`` in the last coordinate, metric
    ``‖dx‖²/(c·x_n²)`` and origin ``o = e_n/√c``. The gyrovector structure is Möbius addition carried
    over from the Poincaré ball by the Cayley transform.

    Precision: the pairwise ops (``dist``, ``logmap``, ``ptransp``, ``gyro_difference``) are built from
    ``w = y - x`` and are cancellation-free; ``expmap`` is cancellation-free for every step direction.
    What remains is the storage floor, a stored point's own rounding ``≈ 0.4·(eps/2)·cosh(√c·δ)/√c``
    as a distance, with ``cosh(√c·δ) = ‖x‖/x_n`` and ``δ`` the distance to the vertical geodesic
    through ``o``. Points on that axis are stored exactly across the dtype's normal range of heights;
    the pairwise ops return inf/NaN past a scaled distance ``√c·d ≈ 88.7`` (float32; float64 ``709.8``),
    where the squared chord overflows.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)
        c: Curvature value (default: 1.0). Must be positive.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import HalfSpace
        >>>
        >>> manifold = HalfSpace(dtype=jnp.float64)
        >>> x = jnp.array([0.1, 1.0])
        >>> y = jnp.array([0.3, 0.5])
        >>> d = manifold.dist(x, y, c=1.0)
        >>> dists = jax.vmap(manifold.dist, in_axes=(0, 0, None))(jnp.stack([x, y]), jnp.stack([y, x]), 1.0)

    References:
        Cannon, Floyd, Kenyon, Parry. "Hyperbolic Geometry." Flavors of Geometry, MSRI Publ. 31, 1997.
        Yu & De Sa. HTorch (github.com/ydtydr/HTorch), ``manifolds/halfspace.py``.
    """

    VERSION_DEFAULT = VERSION_DEFAULT

    # -- Projection ----------------------------------------------------------

    def proj(self, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Floor the height at the dtype's smallest normal number; valid points never move."""
        del c
        return _proj(self._cast(x))

    def proj_batch(self, x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... dim"]:
        """Project batched points (arbitrary leading dimensions); the same op as :meth:`proj`."""
        del c
        return _proj(self._cast(x))

    # -- Gyro-operations -----------------------------------------------------

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Gyro addition ``x ⊕ y = exp_x(PT_{o→x}(log_o y))`` (Möbius addition via the Cayley map)."""
        return _addition(self._cast(x), self._cast(y), c)

    def gyro_difference(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Gyro difference ``(⊖x) ⊕ y``, evaluated from ``log_x y`` without cancellation.

        Mathematically identical to ``addition(⊖x, y)`` with ``⊖x = (-x_s, x_n)/(c‖x‖²)``; use it when
        the result is expected much closer to the origin than the operands (centering, differences
        of two close points).
        """
        return _gyro_difference(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float | Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Gyro scalar multiplication ``r ⊗ x = exp_o(r·log_o x)`` (scales the distance to ``o`` by ``r``)."""
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r_cast, x, c)

    # -- Distance ------------------------------------------------------------

    def dist(
        self,
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: ScalarCurvature,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Geodesic distance ``(2/√c)·arsinh(‖r‖/2)`` (single implementation)."""
        del version_idx  # only one implementation
        return _dist(self._cast(x), self._cast(y), c)

    def dist_0(self, x: Float[Array, "dim"], c: ScalarCurvature, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
        """Geodesic distance from the origin ``o = e_n/√c``."""
        del version_idx
        return _dist_0(self._cast(x), c)

    # -- Exp / log maps ------------------------------------------------------

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map: tangent vector v at x to the manifold."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map from the origin."""
        return _expmap_0(self._cast(v), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Retraction ``proj((x_s + v_s, x_n·exp(v_n/x_n)))``."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map: point y to the tangent space at x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map at the origin."""
        return _logmap_0(self._cast(y), c)

    # -- Parallel transport --------------------------------------------------

    def ptransp(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, "dim"]:
        """Parallel transport of v from x to y."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Parallel transport of v from the origin to y."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    # -- Metric --------------------------------------------------------------

    def tangent_inner(
        self, u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, ""]:
        """Metric ``g_x(u, v) = (u·v)/(c·x_n²)``."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
        """Riemannian norm ``‖v‖_x = ‖v‖/(√c·x_n)``."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Convert a Euclidean gradient to the Riemannian gradient, ``c·x_n²·grad``."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Project v onto the tangent space at x (identity)."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
        """Check that x is finite with ``x_n > 0`` (``atol`` accepted and unused: the constraint is exact)."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
    ) -> Array:
        """Check that v has finite entries (``T_x H = R^n``, so there is no other constraint)."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c, atol)
