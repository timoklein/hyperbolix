"""Math utils functions for hyperbolic operations with numerically stable limits.

Direct JAX port of PyTorch math_utils.py with type annotations using jaxtyping.
"""

import functools
from collections.abc import Callable

import jax
import jax.nn as nn
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float


def _jit[**P, R](fun: Callable[P, R], **jit_kwargs) -> Callable[P, R]:
    """``jax.jit`` with the wrapped function's declared return type kept intact.

    ``jax.jit`` is annotated as returning ``jax.stages.Wrapped``, whose ``__call__`` returns
    ``Any``, so every ``@jax.jit``-decorated helper in this module used to hand its callers an
    untyped value. That is invisible where it happens and surfaces somewhere else: a
    ``jnp.where(cond, a, safe_norm(x))`` three modules away widens to
    ``Array | tuple[Array, ...]`` (pyright unions every overload when an argument is ``Any``)
    and the error lands on whatever declared type that union eventually reaches.

    This is a re-annotation and nothing else: it returns exactly the object ``jax.jit``
    returns, so tracing, the compilation cache, ``__name__``, ``__doc__`` and
    ``lower``/``trace`` are all untouched. ``**jit_kwargs`` is forwarded verbatim, which is
    what the ``static_argnames`` decorators below need.
    """
    return jax.jit(fun, **jit_kwargs)


# Canonical gradient-safety floor for norms and denominators, shared library-wide
# (`manifolds/`, `nn_layers/`, `distributions/`, `decomposition/` all import this name).
# Two uses, both about derivatives rather than values:
#   * `sqrt(sum(x**2) + MIN_NORM**2)` has a finite VJP at x = 0, unlike `linalg.norm`, whose
#     VJP there is 0/0 = NaN.
#   * `floor_at(denom, MIN_NORM)` keeps a division finite when the denominator collapses.
# 1e-15 is small enough to be invisible next to any float32 or float64 quantity the library
# works with (float64 eps is 2.2e-16), and `MIN_NORM**2 = 1e-30` is still normal in both
# dtypes (float32's smallest normal is 1.2e-38), so squaring it does not flush to zero.
MIN_NORM = 1e-15


def floor_at(x: Float[Array, "..."], min_value: ArrayLike) -> Float[Array, "..."]:
    """``max(x, min_value)`` written as a ``where``, for floors on differentiated paths.

    Same value as ``jnp.maximum(x, min_value)`` / ``jnp.clip(x, min_value, None)`` for **every**
    input, NaN and ±inf included: ``NaN < min_value`` is false, so a NaN ``x`` is selected and
    propagates exactly as ``maximum`` propagates it. The comparison direction is deliberately
    ``<`` for that reason — ``where(x > min_value, x, min_value)`` would silently replace a NaN
    with the floor.

    What changes is the **gradient**. ``jnp.maximum``'s JVP is jax's tie-breaking ``_balanced_eq``,
    ``g * [x == ans] / (1 + [min_value == ans])``: it tests the *operand* ``x`` for **bit**
    equality with the *result* ``ans``. That is sound only while the two are the same value in the
    compiled graph, and on XLA:GPU they need not be — the backward fusion may **recompute** ``x``
    (typically a reduction such as a norm) with a different emitter than the forward copy, chosen
    per process by the fusion autotuner. The two copies then differ by 1 ulp, ``[x == ans]`` is
    false, and the *whole* gradient branch through ``x`` is silently zeroed — no NaN, no warning,
    bit-identical within a process, different across launches. Measured on an A100 at
    ``manifolds/hyperboloid._compute_mlr``: 1.0e-2 relative gradient error, firing in ~2 of 3
    launches (see ``logs/2026-09-03_plfc_jit_grad/``).

    ``where`` compares against the *constant* instead, which no 1-ulp disagreement can flip. The
    only gradient difference is exactly at a tie ``x == min_value``, where ``maximum`` splits
    0.5/0.5 and this routes the full cotangent to ``x``.

    Args:
        x: Input array of any shape
        min_value: Lower bound (a Python/NumPy constant, or an array broadcastable against ``x``)

    Returns:
        ``x`` where it exceeds ``min_value``, ``min_value`` elsewhere
    """
    return jnp.where(x < min_value, min_value, x)


def cap_at(x: Float[Array, "..."], max_value: ArrayLike) -> Float[Array, "..."]:
    """``min(x, max_value)`` written as a ``where``. Mirror of :func:`floor_at`; same rationale.

    NaN-preserving for the same reason (``NaN > max_value`` is false, so ``x`` is selected).

    Args:
        x: Input array of any shape
        max_value: Upper bound (a Python/NumPy constant, or an array broadcastable against ``x``)

    Returns:
        ``x`` where it is below ``max_value``, ``max_value`` elsewhere
    """
    return jnp.where(x > max_value, max_value, x)


def clamp_to(x: Float[Array, "..."], min_value: ArrayLike, max_value: ArrayLike) -> Float[Array, "..."]:
    """``jnp.clip(x, min_value, max_value)`` written as two ``where``s. See :func:`floor_at`.

    Composed in ``clip``'s own order, ``min(max(x, lo), hi)``, so the two agree bit-for-bit even
    for the degenerate ``lo > hi`` (both return ``hi``).

    Args:
        x: Input array of any shape
        min_value: Lower bound
        max_value: Upper bound

    Returns:
        ``x`` clamped into ``[min_value, max_value]``
    """
    return cap_at(floor_at(x, min_value), max_value)


def _softplus_tail(u: Float[Array, "..."], smoothing_factor: float) -> Float[Array, "..."]:
    """``log(1 + exp(-beta*|u|)) / beta`` — the bounded remainder of the scaled softplus.

    Writing ``sp(u) = log(1 + exp(beta*u))/beta`` for the scaled softplus (``beta =
    smoothing_factor``), the exact hinge/remainder split is::

        sp(u) = max(u, 0) + _softplus_tail(u)

    The first term is the hard hinge; this one is the smooth remainder, and ``0 < tail <=
    log(2)/beta`` for **every** ``u``. All three clamps below are built from a hard ``jnp.clip`` /
    ``maximum`` / ``minimum`` (which is exactly representable and exactly in range) plus these
    bounded remainders, rather than from the scaled softplus directly. That is what makes the bound
    hold in floating point: evaluating ``min + sp(x - min) - sp(x - max)`` literally subtracts two
    nearly equal large numbers once ``x >> max``, and the cancellation error is unbounded relative
    to a narrow window (measured: float32, window ``[0, 0.01]``, beta=50 -> the literal form returns
    ``min`` at ``x = 1e6`` and overshoots ``max`` by 2.3e-7 elsewhere).

    ``-|u|`` is spelled ``minimum(u, -u)`` rather than ``-abs(u)`` deliberately. The two agree in
    value, but JAX's tie-breaking splits ``minimum``'s derivative evenly at ``u == 0``, so this term
    contributes exactly 0 there and the hinge's own 1/2 is the whole derivative — which is
    ``sigmoid(0) = 1/2``, the analytic value. ``-abs(u)`` has derivative ``+1`` at 0 in JAX, which
    would make the derivative *at the clamp boundary* wrong by 1/2.

    Caveat, first derivatives only: the split is exact in value and slope, not in curvature. At
    exactly ``x == min_value`` (or ``max_value``) a *second* derivative taken by autodiff sees the
    hinge's kink as flat and returns only the far bound's contribution, missing the near bound's
    ``beta/4``. Everywhere else — including one ulp away from the bound — it is exact.
    """
    return nn.softplus(smoothing_factor * jnp.minimum(u, -u)) / smoothing_factor


@functools.partial(_jit, static_argnames=["smoothing_factor"])
def smooth_clamp_min(x: Float[Array, "..."], min_value: float, smoothing_factor: float = 50.0) -> Float[Array, "..."]:
    """Smoothly clamp array values to a minimum using softplus. Range=(min_value, inf).

    Implements the gate-free lower clamp (``beta = smoothing_factor``)::

        smooth_clamp_min(x) = min_value + sp(x - min_value),   sp(u) = log(1 + exp(beta*u)) / beta

    evaluated in the numerically stable hinge + remainder form (see ``_softplus_tail``). Properties,
    for every ``beta > 0``:

    * **Bounded**: ``sp > 0`` everywhere, so the output is strictly above ``min_value``.
    * **Smooth and monotone**: ``C^inf`` with derivative ``sigmoid(beta*(x - min_value))`` in
      ``(0, 1)`` — including *at* ``min_value``, where it is exactly 1/2.
    * **Identity away from the bound**: above the bound the output exceeds ``x`` by
      ``log(1 + exp(-beta*d))/beta ~ exp(-beta*d)/beta`` with ``d = x - min_value``; at ``beta=50``
      that is below the float32 noise floor for ``d >~ 0.3``.

    This replaces a ``jnp.where(x < min_value + eps, softplus_branch, x)`` gate. That gate was
    discontinuous: ``sp(0) = log(2)/beta``, not 0, so the softplus branch did not glue onto the
    identity branch at the switch point but jumped by ``log(2)/beta`` (0.0139 at the default
    ``beta=50``). Composing the two gated one-sided clamps is what let the old ``smooth_clamp``
    leave its own window.

    Args:
        x: Input array of any shape
        min_value: Minimum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped above min_value
    """
    # max(x, min) + tail(x - min) == min + sp(x - min), without evaluating sp directly.
    return jnp.maximum(x, min_value) + _softplus_tail(x - min_value, smoothing_factor)


@functools.partial(_jit, static_argnames=["smoothing_factor"])
def smooth_clamp_max(x: Float[Array, "..."], max_value: float, smoothing_factor: float = 50.0) -> Float[Array, "..."]:
    """Smoothly clamp array values to a maximum using softplus. Range=(-inf, max_value).

    The mirror image of :func:`smooth_clamp_min` (``beta = smoothing_factor``)::

        smooth_clamp_max(x) = max_value - sp(max_value - x),  sp(u) = log(1 + exp(beta*u)) / beta

    so the output is strictly below ``max_value``, ``C^inf``, monotone increasing with derivative
    ``sigmoid(beta*(max_value - x))`` in ``(0, 1)``, and falls short of ``x`` below the bound by
    ``log(1 + exp(-beta*d))/beta`` with ``d = max_value - x``. Gate-free for the same reason as the
    min-side clamp — see :func:`smooth_clamp_min`.

    Args:
        x: Input array of any shape
        max_value: Maximum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped below max_value
    """
    # min(x, max) - tail(x - max) == max - sp(max - x); tail is even, so the same helper serves.
    return jnp.minimum(x, max_value) - _softplus_tail(x - max_value, smoothing_factor)


@functools.partial(_jit, static_argnames=["smoothing_factor"])
def smooth_clamp(
    x: Float[Array, "..."], min_value: float, max_value: float, smoothing_factor: float = 50.0
) -> Float[Array, "..."]:
    """Smoothly clamp array values to a range [min_value, max_value]. Range=(min_value, max_value).

    The two-sided clamp is a *difference* of scaled softplus terms, **not** a composition of
    :func:`smooth_clamp_min` and :func:`smooth_clamp_max` (``beta = smoothing_factor``)::

        smooth_clamp(x) = min_value + sp(x - min_value) - sp(x - max_value)

    with ``sp(u) = log(1 + exp(beta*u)) / beta``, evaluated in the stable hinge + remainder form
    (see ``_softplus_tail``). For every window width ``w = max_value - min_value > 0`` and every
    ``beta > 0``:

    * **Bounded**: ``sp`` is increasing so the output is above ``min_value``, and
      ``sp(u) - sp(u - beta*w) < w`` reduces to ``1 < exp(beta*w)`` — true for any positive window.
      The output is therefore strictly inside the window, with no constraint linking ``w`` to
      ``beta``.
    * **Smooth and monotone**: ``C^inf`` with derivative ``sigmoid(beta*(x - min_value)) -
      sigmoid(beta*(x - max_value))`` in ``(0, 1)``, exact at the two bounds as well.
    * **Identity in the interior**: the output differs from ``x`` by ``[exp(-beta*d_min) -
      exp(-beta*d_max)]/beta`` up to higher order, where ``d_min``/``d_max`` are the distances to
      the two bounds; at ``beta=50`` that is below the float32 noise floor once ``x`` is ~0.3 from
      both.

    The composition ``smooth_clamp_min(smooth_clamp_max(x))`` this replaces could return values
    **outside** the window whenever ``w < log(2)/beta`` (0.0139 at the default ``beta=50``), because
    each gated one-sided clamp displaced its boundary input by ``log(2)/beta`` rather than by ~0
    (see :func:`smooth_clamp_min`). Concretely, window ``[0, 0.01]`` at ``beta=50`` returned 0.01386
    for ``x = 0.0`` — 39% above ``max_value`` for an input that needed no clamping at all. Note that
    a *composition* of gate-free one-sided clamps does not fix this: it still overshoots by
    ``log(1 + exp(-beta*w))/beta``, which is of order the window itself in that regime. The
    difference form above is the one that is provably bounded.

    In floating point the strictness degrades only in the saturated tails, where the remainder terms
    underflow to 0 and the output equals ``min_value``/``max_value`` exactly. That is an
    *underflow* of ``exp(-beta*d)``, not an overflow: it happens past ``-log(smallest_subnormal)``,
    measured at ``beta*d`` ~104 in float32 and ~745 in float64. Inclusion in
    ``[min_value, max_value]`` always holds.

    Args:
        x: Input array of any shape
        min_value: Minimum value to clamp to
        max_value: Maximum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped to [min_value, max_value]
    """
    # clip(x) + tail(x - min) - tail(x - max) == min + sp(x - min) - sp(x - max). The clip supplies
    # the hinge part exactly (and exactly in range); the two remainders are each <= log(2)/beta.
    return (
        jnp.clip(x, min_value, max_value)
        + _softplus_tail(x - min_value, smoothing_factor)
        - _softplus_tail(x - max_value, smoothing_factor)
    )


def _pow2_divisor(m: Float[Array, "..."]) -> Float[Array, "..."]:
    """The rescaling divisor the safe-norm family shares: ``2**floor(log2 m)``, the binade floor of ``m``.

    A **power of two**, not ``m`` itself. That is the whole point: dividing and multiplying by a
    power of two only shifts the exponent, so both operations are exact and the rescale contributes
    **no rounding of its own** — while still never forming a quantity that can overflow. (It is not
    an unconditional bit-identity to ``sqrt(sum(v**2))``: XLA picks the association of a last-axis
    reduction per fusion, so the two programs agree exactly at most dimensions and differ by 1 to 2
    ulp at the rest. Measured mean ulp distance: 0.000 at dim 2/4/16/64 and 0.13-0.17 at dim 8,
    against 0.34-0.62 at *every* dim for the divide-by-max scaler.)

    Dividing by ``m`` is inexact, and the 1 to 2 ulp it puts into the result is visible where the
    caller's next step recomputes the same sum of squares: the
    hyperboloid time slot ``x₀ = sqrt(‖s‖² + 1/c)`` is checked against ``-x₀² + ‖s‖² = -1/c``, where
    at ``x₀ ≈ 34`` one ulp of ``x₀²`` is 1.2e-4 and the rounding no longer cancels.

    The divisor is built by **masking the mantissa bits off** ``m``, which leaves
    ``2**floor(log2 m)`` exactly on every backend. It is not built with ``jnp.ldexp``: ``ldexp``
    computes ``m * 2**e`` and lowers to an XLA ``power``, which CUDA evaluates as a transcendental
    accurate to ~1 ulp rather than exactly. In float64 on an A100 that is off by an ulp for 75 of
    the 600 binades ``k ∈ [-300, 300)`` — ``ldexp(1.0, -33)`` returns one ulp *below* ``2**-33`` —
    and an inexact divisor puts a rounding into every division it scales, which is the whole thing
    this helper exists to avoid. (float32 is exact on both backends; the measurements are in
    ``logs/2026-09-06_numerics_review_followup/gpu_norm_rescale/``.)

    ``2**floor(log2 m)``, never ``2 * 2**floor(log2 m)``: the larger choice is ``+inf`` for an ``m``
    in the top binade (float32 ``m ≥ 2**127``) while ``2**floor(log2 m) ≤ 2**127`` is always
    representable. Scaled magnitudes then land in ``[1, 2)`` rather than ``[0.5, 1)``, so the sum of
    squares sits in ``[1, 4n)`` — as overflow-free as before.

    The divisor is floored at the smallest **normal** power of two (``finfo.tiny``): a subnormal
    ``m`` has all its significant bits in the mantissa field, so masking leaves 0 and the division
    would be ``0/0``. Below that floor the scaled values are simply ``< 1``, which overflows
    nothing.

    ``m`` is sanitized to ``1.0`` where it is zero or non-finite, which reproduces the previous
    ``jnp.where((scale > 0) & isfinite(scale), scale, 1)`` guard exactly: 0 would give ``0/0`` and
    ``inf`` would give ``inf/inf``, whereas with divisor 1 the ``inf`` case flows through as
    ``sum(v**2) = inf -> inf``, the library's convention for keeping a degenerate point visible.

    Args:
        m: Largest magnitude of the operands, any shape. Callers pass it already
            ``stop_gradient``-ed; nothing here is differentiated.

    Returns:
        An exact power of two in ``[finfo.tiny, 2**(finfo.maxexp - 1)]``, same shape as ``m``
    """
    m_safe = jnp.where((m > 0.0) & jnp.isfinite(m), m, jnp.ones_like(m))
    uint_dtype = jnp.dtype(f"uint{8 * m.dtype.itemsize}")
    mantissa_mask = jnp.asarray((1 << jnp.finfo(m.dtype).nmant) - 1, dtype=uint_dtype)
    bits = jax.lax.bitcast_convert_type(m_safe, uint_dtype)
    truncated = jax.lax.bitcast_convert_type(bits & ~mantissa_mask, m.dtype)
    return jnp.maximum(truncated, jnp.asarray(jnp.finfo(m.dtype).tiny, dtype=m.dtype))


@_jit
def safe_norm(v: Float[Array, "... n"]) -> Float[Array, "..."]:
    """Euclidean norm over the last axis, computed on a rescaled vector. Domain=R^n, Range=[0, inf).

    Divides by the power of two just below ``max|v|`` (:func:`_pow2_divisor`) before squaring, so
    the sum of squares always sits in ``[1, 4n)`` no matter how large or small ``v`` is. The
    divisor being a power of two is what keeps this **bit-identical to** ``sqrt(sum(v**2))``
    wherever that neither overflows nor underflows -- see :func:`_pow2_divisor`. That is what the library's older
    ``sqrt(sum(v**2) + MIN_NORM**2)`` idiom cannot do, in *both* directions:

    * **Overflow**: ``sum(v**2)`` overflows float32 once ``|v| > 1.8e19`` — reached by the spatial
      part of a hyperboloid point at geodesic radius ~45 (``arcsinh(1.8e19) = 45.03`` at ``c = 1``;
      radius 44 is only coordinate ``6.4e18``), while the norm itself, ``1.8e19``, is perfectly
      representable.
    * **Underflow**: the ``+ MIN_NORM**2 = 1e-30`` floor *dominates* any genuinely small vector.
      A float32 chord of ``1e-34`` (a legitimate angular separation between two nearly parallel
      unit vectors) comes back as ``1e-15``, i.e. 19 orders of magnitude too large.

    The scale is wrapped in ``stop_gradient``, which makes the VJP exactly ``v/‖v‖`` — the analytic
    derivative, with no contribution from differentiating the scaling itself.

    At ``v == 0`` the result is exactly ``0`` and the VJP is exactly ``0`` (the derivative does not
    exist there; zero is the finite, direction-free choice). Both need the **double** ``where``
    below: sanitizing only the output leaves ``sqrt(0)``'s infinite derivative inside the VJP,
    where it meets the outer ``where``'s zero cotangent as ``0 * inf = NaN``. The first ``where``
    replaces the *argument* of the ``sqrt`` so the infinite derivative is never created.

    Non-finite inputs are passed through rather than turned into NaN: a vector holding an ``inf``
    returns ``inf``. Callers use that to keep an out-of-range point visibly degenerate instead of
    silently NaN-poisoning everything downstream.

    Args:
        v: Input array, norm taken over the last axis

    Returns:
        ``‖v‖₂`` over the last axis, shape ``v.shape[:-1]``
    """
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(v), axis=-1))
    is_zero = scale == 0.0
    divisor = _pow2_divisor(scale)
    sq = jnp.sum((v / divisor[..., None]) ** 2, axis=-1)
    # Double-where, first half: sqrt'(0) = inf would become 0*inf = NaN in the VJP below.
    sq_safe = jnp.where(is_zero, jnp.ones_like(sq), sq)
    # Double-where, second half: exact 0 forward, exactly-zero VJP at v = 0.
    return jnp.where(is_zero, jnp.zeros_like(sq), divisor * jnp.sqrt(sq_safe))


@_jit
def safe_hypot(p: Float[Array, "..."], q: Float[Array, "..."]) -> Float[Array, "..."]:
    """``sqrt(p² + q²)`` without intermediate overflow or underflow. Range=[0, inf).

    The two-argument form of :func:`safe_norm`, with the same power-of-two rescaling and the same double
    ``where``: exact ``0`` and exactly-zero gradient at ``p == q == 0``, no ``p**2`` materialized
    (so ``safe_hypot(1e30, 1.0)`` is finite in float32 even though ``1e30**2`` is not), and
    non-finite inputs pass through as ``inf`` rather than NaN.

    Used to build a hyperbolic ``sinh(θ/2)`` out of two non-negative contributions whose squares
    routinely straddle the dtype's whole exponent range.

    Args:
        p: First leg (any shape, broadcast against ``q``)
        q: Second leg

    Returns:
        ``sqrt(p² + q²)``
    """
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(p), jnp.abs(q)))
    is_zero = scale == 0.0
    divisor = _pow2_divisor(scale)
    sq = (p / divisor) ** 2 + (q / divisor) ** 2
    sq_safe = jnp.where(is_zero, jnp.ones_like(sq), sq)
    return jnp.where(is_zero, jnp.zeros_like(sq), divisor * jnp.sqrt(sq_safe))


@_jit
def safe_hypot_norm(v: Float[Array, "... n"], q: Float[Array, "..."]) -> Float[Array, "..."]:
    """``sqrt(‖v‖² + q²)`` over the last axis, in **one** reduction. Range=[0, inf).

    Not ``safe_hypot(safe_norm(v), q)``. That spelling rounds ``‖v‖`` to the dtype and then squares
    it again, so the ``sum(v**2)`` it started from is not recoverable: ``r = round(sqrt(S))`` gives
    ``r² = S(1 + 2δ)``, an error of up to one ulp *of S*. This form keeps the sum of squares intact
    and appends ``q²`` to it, exactly as the pre-``safe_norm`` idiom ``sqrt(sum(v**2) + q**2)`` did,
    so it is bit-identical to that idiom wherever it neither overflows nor underflows — the same
    guarantee :func:`_pow2_divisor` gives :func:`safe_norm`, and for the same reason.

    That matters because the library's callers **recompute the same sum of squares** right after.
    The hyperboloid time slot is ``x₀ = sqrt(‖x_s‖² + 1/c)`` and its own constraint is
    ``-x₀² + ‖x_s‖² = -1/c``: when ``x₀`` is built from the same ``sum(x_s**2)`` the check forms,
    the rounding cancels and the residual is one rounding of ``x₀``; when it is built from a
    re-squared ``‖x_s‖`` the two no longer cancel. At ``x₀ ≈ 34`` in float32 one ulp of ``x₀²`` is
    1.2e-4, which is larger than the 1e-4 tolerance the manifold checks use
    (``tests/nn_layers/test_hypformer.py::test_hrc_*_extreme_curvatures``).

    Overflow-free in the same way as :func:`safe_norm`: nothing is squared before the exact
    power-of-two rescale, so a spatial part at float32 radius 1e20 still returns a finite ``x₀``.
    Same double ``where``: exact ``0`` and an exactly-zero VJP where ``v`` and ``q`` are both zero,
    and a non-finite input passes through as ``inf`` rather than becoming NaN.

    Args:
        v: Vector leg, norm taken over the last axis
        q: Scalar leg, broadcast against ``v.shape[:-1]``

    Returns:
        ``sqrt(‖v‖₂² + q²)``, shape ``broadcast(v.shape[:-1], q.shape)``
    """
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(v), axis=-1), jnp.abs(q)))
    is_zero = scale == 0.0
    divisor = _pow2_divisor(scale)
    # `sum(...) + (q/divisor)**2`, not one reduction over the concatenation: this reproduces the
    # association of the idiom it replaces, `sum(v**2) + q**2`, which is what makes it bit-identical.
    sq = jnp.sum((v / divisor[..., None]) ** 2, axis=-1) + (q / divisor) ** 2
    sq_safe = jnp.where(is_zero, jnp.ones_like(sq), sq)
    return jnp.where(is_zero, jnp.zeros_like(sq), divisor * jnp.sqrt(sq_safe))


@_jit
def safe_sqrt(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """``sqrt(x)`` with an exactly-zero derivative at ``x == 0`` instead of ``inf``. Range=[0, inf).

    The scalar counterpart of :func:`safe_norm`, for the quantities that are already a squared
    magnitude when they arrive: a Minkowski or Riemannian quadratic form ``g_x(v, v)``, a sum of
    per-factor squared distances. Those cannot be routed through ``safe_norm`` (its input is the
    *vector*, and a Minkowski form is not a Euclidean norm), yet they hit the same wall: at
    ``x = 0`` the true derivative ``1/(2*sqrt(x))`` is infinite, and reverse-mode AD multiplies it
    by the incoming cotangent — which for the usual ``v = 0`` case is exactly ``0``, giving
    ``0 * inf = NaN`` for the whole row.

    The library's older answer was to add ``MIN_NORM**2`` under the ``sqrt``, which has the same
    two defects :func:`safe_norm` documents: the ``1e-30`` floor dominates a genuinely small ``x``
    (a form of ``1e-40`` comes back as ``1e-15``), and it does nothing about a large one.

    Same **double** ``where`` as :func:`safe_norm`: the first replaces the *argument* so the
    infinite derivative is never created, the second restores the exact ``0`` forward value. A
    negative ``x`` still yields NaN (as ``jnp.sqrt`` does) — callers that can produce a slightly
    negative form from rounding wrap the argument in ``floor_at(x, 0.0)`` first, which is the
    honest place for that decision. ``inf`` passes through as ``inf``.

    Args:
        x: Non-negative input array of any shape

    Returns:
        ``sqrt(x)``, with derivative ``0`` (not ``inf``) wherever ``x == 0``
    """
    is_zero = x == 0.0
    # Double-where, first half: sqrt'(0) = inf would become 0*inf = NaN in the VJP below.
    x_safe = jnp.where(is_zero, jnp.ones_like(x), x)
    # Double-where, second half: exact 0 forward, exactly-zero VJP at x = 0.
    return jnp.where(is_zero, jnp.zeros_like(x), jnp.sqrt(x_safe))


@_jit
def safe_normalize(v: Float[Array, "... n"]) -> Float[Array, "... n"]:
    """``v/‖v‖`` over the last axis, returning the **exact zero vector** at ``v == 0``.

    Same power-of-two-rescaled, double-``where`` construction as :func:`safe_norm`, so the result is a unit
    vector for every non-zero ``v`` (including magnitudes that would overflow or flush a
    sum-of-squares) and exactly ``0`` with a finite (zero) gradient at ``v == 0``.

    There is deliberately **no** ``maximum(‖v‖, MIN_NORM)`` floor on the denominator. This is used
    to build the *angular* unit vector of a geodesic frame, whose pre-normalization length is
    ``sin(ψ)`` for the angle ``ψ`` between two points; a ``1e-15`` floor against a genuine
    ``sin(ψ) = 1e-19`` would shrink the resulting direction by a factor of ``1e4`` instead of
    returning a unit vector. The exact-zero return at ``v == 0`` is the well-defined case
    (``ψ = 0``: no angular direction exists, and the frame's angular coefficient is zero there),
    which is why the floor is not needed.

    Args:
        v: Input array, normalized over the last axis

    Returns:
        ``v/‖v‖₂``, or the zero vector where ``v`` is zero
    """
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(v), axis=-1, keepdims=True))
    is_zero = scale == 0.0
    divisor = _pow2_divisor(scale)
    scaled = v / divisor
    sq = jnp.sum(scaled**2, axis=-1, keepdims=True)
    sq_safe = jnp.where(is_zero, jnp.ones_like(sq), sq)
    return jnp.where(is_zero, jnp.zeros_like(v), scaled / jnp.sqrt(sq_safe))


@_jit
def capped_exp(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Exponential with an overflow cap on the argument. Domain=(-inf, inf).

    Computes ``exp(minimum(x, 0.99*log(finfo.max)))`` (cap ≈ 87.8 for f32, ≈ 702.7 for f64), so the
    result cannot overflow to ``+inf``. Intended for ``exp`` of *unconstrained trainable parameters*
    (e.g. log-scale reparameterizations): a runaway parameter would otherwise produce an ``inf``
    that turns into NaN downstream (``inf - inf``, ``inf * 0``) and poisons every parameter within
    one optimizer step. With the cap, a runaway saturates at a huge-but-finite value with finite
    gradients, so the failure stays visible and recoverable instead of NaN-ing the run.

    The argument is capped rather than the output (``clip(exp(x), max)``) deliberately: an output
    clip still materializes ``exp(x) = inf`` in the forward pass and ``inf`` in its VJP, exactly
    the non-finite values the guard exists to prevent.

    Below the cap this is a bitwise value- and gradient-identity to ``jnp.exp``. Above the cap the
    gradient is exactly 0 (the ``minimum`` selects the constant side) — a parameter past the cap
    receives no gradient signal to come back down. ``LearnableCurvature``'s ``"log"``
    parameterization offers an opt-in straight-through backward for that regime; here the plain cap
    is kept because the straight-through form trades forward-value exactness for it (its
    ``stop_gradient`` arithmetic rounds through the raw parameter's magnitude), and a scale
    parameter past the cap (≈87.8 in float32, ≈702.7 in float64) means training has already
    diverged — the guard's job is containment.

    Args:
        x: Input array of any shape

    Returns:
        exp(x) with the argument capped so the result is always finite
    """
    clamp = jnp.log(jnp.finfo(x.dtype).max) * 0.99
    return jnp.exp(cap_at(x, clamp))


@jax.custom_jvp
def _cosh_stable(x):
    return 0.5 * (jnp.exp(x) + jnp.exp(-x))


@_cosh_stable.defjvp
def _cosh_stable_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return _cosh_stable(x), (0.5 * (jnp.expm1(x) - jnp.expm1(-x))) * t


@_jit
def cosh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Hyperbolic cosine with overflow protection. Domain=(-inf, inf).

    Hard-clips the input to ``±0.99*log(finfo.max)`` (≈±87.8 for f32, ±702.7 for f64) before
    computing the value so the result cannot overflow the dtype. This is a *pure overflow guard*:
    for any input that is not about to overflow the clip is a value- and gradient-identity, so the
    forward pass and the VJP match an unguarded ``cosh`` throughout the entire valid regime.

    A hard ``jnp.clip`` is used deliberately rather than a softplus ``smooth_clamp``: it matches the
    domain guards in ``acosh``/``atanh`` below, is free on accelerators, and avoids the ~2 extra
    ``exp`` per call that the smooth clamp evaluated on every element with no benefit. The only
    difference is in the saturated tail (|x| ≥ clamp), where the clip's gradient is 0 instead of a
    tiny nonzero value — acceptable, because that regime is already degenerate (the output is ~1e37)
    and you never want gradients pushing further into overflow.

    XLA's CPU ``jnp.cosh`` is inaccurate: ~17 ulps off for ``|x|`` in ``[16, 512]`` and ~496 ulps for
    ``[512, 710]`` (f64; the jump sits exactly at the power-of-two boundary 512 — NumPy's ``cosh`` is
    ≤1.8 ulps throughout), and ~24.5 ulps in f32. The exp-form identity ``cosh(x) = 0.5*(exp(x) +
    exp(-x))`` has no cancellation (both terms are positive) and measures ≤1.8 ulps f64 / ≤1.4 ulps
    f32 against an extended-precision reference. But its *naive* autodiff derivative is ``0.5*(exp(x) - exp(-x))``, which
    cancels near 0 (relative error ~eps/|x|) — the same failure mode the ``atanh`` rewrite above
    avoids for its own function. ``_cosh_stable`` fixes this with a ``custom_jvp`` that routes
    ``d/dx cosh`` to the accurate expm1-form ``sinh`` below (``0.5*(expm1(x) - expm1(-x))``) instead
    of differentiating the exp-form value expression.

    Args:
        x: Input array of any shape

    Returns:
        cosh(x) with overflow protection
    """
    # cosh(x) ≈ exp(x)/2 for large x, so the overflow boundary is x = log(max).
    clamp = jnp.log(jnp.finfo(x.dtype).max) * 0.99
    x = clamp_to(x, -clamp, clamp)
    # exp-form value (≤1.8 ulps f64 / ≤1.4 ulps f32 vs XLA CPU cosh's 17-496 / 24.5 ulps); the
    # custom_jvp above keeps the gradient on the accurate expm1-form sinh instead of the naive
    # cancelling exp-form derivative.
    return _cosh_stable(x)


@_jit
def sinh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Hyperbolic sine with overflow protection. Domain=(-inf, inf).

    Hard-clips the input to ``±0.99*log(finfo.max)`` (≈±87.8 for f32, ±702.7 for f64) before
    computing the value so the result cannot overflow the dtype. This is a *pure overflow guard*:
    for any input that is not about to overflow the clip is a value- and gradient-identity, so the
    forward pass and the VJP match an unguarded ``sinh`` throughout the entire valid regime.

    A hard ``jnp.clip`` is used deliberately rather than a softplus ``smooth_clamp``: it matches the
    domain guards in ``acosh``/``atanh`` below, is free on accelerators, and avoids the ~2 extra
    ``exp`` per call that the smooth clamp evaluated on every element with no benefit. The only
    difference is in the saturated tail (|x| ≥ clamp), where the clip's gradient is 0 instead of a
    tiny nonzero value — acceptable, because that regime is already degenerate (the output is ~1e37)
    and you never want gradients pushing further into overflow.

    XLA's CPU ``jnp.sinh`` is inaccurate the same way ``jnp.cosh`` is: ~17 ulps off for ``|x|`` in
    ``[16, 512]`` and ~496 ulps for ``[512, 710]`` (f64, jump at the power-of-two boundary 512;
    NumPy's ``sinh`` is ≤1.8 ulps throughout), and ~24.5 ulps in f32. The expm1-form identity
    ``sinh(x) = 0.5*(expm1(x) - expm1(-x))`` is cancellation-free *everywhere*, including near 0 —
    unlike the naive exp-form ``0.5*(exp(x) - exp(-x))``, the two ``expm1`` terms have opposite signs
    (``expm1(x) >= 0`` and ``-expm1(-x) >= 0`` for ``x >= 0``, and symmetrically for ``x < 0``), so
    the subtraction adds magnitudes instead of cancelling them. This measures ≤3.9 ulps f64 / ≤4.2
    ulps f32 against an extended-precision reference, everywhere, and is faster than ``jnp.sinh``. Unlike ``cosh`` above,
    ``sinh`` needs no ``custom_jvp``: its natural autodiff derivative is exactly the accurate exp-form
    ``0.5*(exp(x) + exp(-x))`` (``_cosh_stable``'s value expression), which has no cancellation to
    begin with.

    Args:
        x: Input array of any shape

    Returns:
        sinh(x) with overflow protection
    """
    # sinh(x) ≈ exp(x)/2 for large x, so the overflow boundary is x = log(max).
    clamp = jnp.log(jnp.finfo(x.dtype).max) * 0.99
    x = clamp_to(x, -clamp, clamp)
    # expm1-form: cancellation-free everywhere (opposite-signed terms), ≤3.9 ulps f64 vs XLA CPU
    # sinh's 17-496 ulps, and faster than the builtin. Autodiff of this expression is already the
    # accurate exp-form cosh, so no custom_jvp is needed here (contrast cosh above).
    return 0.5 * (jnp.expm1(x) - jnp.expm1(-x))


@_jit
def acosh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse hyperbolic cosine with domain clamping. Domain=[1, inf).

    Clamps to ``1 + 10*machine_eps`` — NOT exactly 1.0. ``acosh'(1) = inf``,
    so a hard clip at 1.0 lets inputs that land exactly on 1.0 (e.g. the
    distance argument at x == y) reach the singular derivative and produce
    NaN gradients; post-hoc ``jnp.where`` guards cannot remove them because
    the NaN cotangent already exists inside the VJP (0*inf = NaN). The
    margin bounds the derivative at ~1/sqrt(2*margin) and keeps the forward
    error sqrt(2*margin) below test tolerances (f32: ~1.5e-3, f64: ~6.6e-8).

    Args:
        x: Input array of any shape

    Returns:
        acosh(x) with domain and gradient protection
    """
    eps = 10.0 * float(jnp.finfo(x.dtype).eps)
    x = floor_at(x, 1.0 + eps)
    return jnp.acosh(x)


def _is_low_precision(dtype) -> bool:
    """True for float32 and narrower, False for float64. Static: ``dtype`` is known at trace time.

    Both accuracy rewrites below (``atanh``'s truncated series, ``tanh``'s ``expm1`` seam) are
    profitable only when the working precision is coarse enough to hide their extra error, so both
    gate on this. The 1e-10 threshold sits in the empty gap between float32's eps (1.2e-7) and
    float64's (2.2e-16); float16/bfloat16 land on the float32 side, which is the correct side for
    them (their eps is coarser still).
    """
    return float(jnp.finfo(dtype).eps) > 1e-10


# Below |x| = 1/8 the odd Maclaurin series `x + x^3/3 + ... + x^9/9` is exact to float32 rounding:
# the first dropped term is x^11/11, i.e. 1.1e-11 relative to x at x = 1/8 (float32 eps is 1.2e-7),
# and smaller for smaller x. It is NOT below float64 eps, hence the `_is_low_precision` gate.
_ATANH_SERIES_SEAM = 0.125


@_jit
def atanh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse hyperbolic tangent with domain clamping. Domain=(-1, 1).

    Clamps input to ``±(1 - 10*machine_eps)``. The factor 10 keeps the
    clamped value safely representable away from ±1.0 (where the float grid
    is coarsest) and bounds ``atanh'`` at ~1/(2*margin) instead of letting
    inputs ride the last representable value before the singularity.

    XLA evaluates ``jnp.atanh`` as ``0.5*(log1p(x) - log1p(-x))``, and XLA's CPU float64
    ``log1p`` is up to ~129 ulps off for arguments in ``[-0.53, -0.28]`` (NumPy's ``log1p`` is <1
    ulp there) — ``jnp.atanh`` inherits ~129 ulp error for ``x`` in ~``[0.28, 0.53]``, peaking at
    ``sqrt(2) - 1``. Float32 is unaffected. The single-log1p identity
    ``atanh(|x|) = 0.5*log1p(2|x|/(1 - |x|))``, computed on ``|x|`` with the sign restored via
    ``jnp.where``, keeps the ``log1p`` argument non-negative (outside the bad window) for either
    sign of ``x``. The plain (non-odd) rewrite ``0.5*log1p(2x/(1-x))`` is wrong for negative ``x``
    — its inner argument re-enters the bad ``log1p`` window and still errs by ~211 ulps; a
    ``jnp.sign(x)*...`` spelling is also wrong (its product-rule gradient at ``x == 0`` collapses
    to 0 instead of the analytic 1). This form measures at max 2.8 ulps (f64, both signs),
    bit-identical to the builtin at the clip boundary, exact gradients, and ~1.7x faster than
    ``jnp.atanh`` (one ``log1p`` instead of two). Upstream fixed the ``log1p`` hole in jax 0.11.1
    (openxla/xla#46765, jax-ml/jax#39707), but the rewrite is kept: it costs nothing, and it is
    what protects users still on jax <= 0.11.0.

    Below ``|x| = 1/8`` **in float32** the ``log1p`` form is replaced by the odd Maclaurin series
    ``x + x^3/3 + x^5/5 + x^7/7 + x^9/9`` (Horner in ``x**2``, evaluated on ``x`` itself so it is
    odd by construction). The series is exact to float32 rounding there — the first dropped term
    is ``x^11/11``, 1.1e-11 relative at ``x = 1/8`` against a float32 eps of 1.2e-7 — and it
    removes the rounding that the division ``2|x|/(1 - |x|)`` costs the ``log1p`` form near zero.
    Measured on 20k log-uniform float32 inputs in ``[1e-4, 0.125]``: max ulp 3 -> 1 (mean 0.48 ->
    0.33) on XLA GPU, 2 -> 1 (mean 0.28 -> 0.33) on XLA CPU. float64 keeps the ``log1p`` form on
    the whole domain: 1.1e-11 relative is ~5e4 float64 ulps, so the series is unusable there.

    Args:
        x: Input array of any shape

    Returns:
        atanh(x) with domain and gradient protection
    """
    eps = 10.0 * float(jnp.finfo(x.dtype).eps)
    x = clamp_to(x, -1.0 + eps, 1.0 - eps)
    # XLA evaluates jnp.atanh as 0.5*(log1p(x) - log1p(-x)), and XLA's float64 log1p is up to
    # ~129 ulps off for arguments in [-0.53, -0.28] (numpy's: <1 ulp) — jnp.atanh inherits this
    # for x in ~[0.28, 0.53], peaking at sqrt(2)-1. The single-log1p identity
    # atanh(|x|) = 0.5*log1p(2|x|/(1 - |x|)) keeps the log1p argument non-negative (outside the
    # bad window) for either sign of x; restoring the sign via where (not sign(x)*...) keeps
    # the gradient at x == 0 exact. Measured: 129 -> 2.8 ulps max (f64), value- and
    # gradient-identical at the clip boundary, and ~1.7x faster (one log1p instead of two).
    abs_x = jnp.abs(x)
    # The clip above IS the sanitised argument this branch needs: it leaves 1 - abs_x >= 10*eps
    # (1.19e-6 in f32, 2.2e-15 in f64), so the quotient, the log1p and their gradients are finite
    # for every real input — required because jnp.where below evaluates BOTH branches and their
    # gradients, and a NaN in the unselected one would leak into the selected one's cotangent.
    half_log1p = 0.5 * jnp.log1p(2.0 * abs_x / (1.0 - abs_x))
    out = jnp.where(x >= 0, half_log1p, -half_log1p)
    if not _is_low_precision(x.dtype):
        return out
    # Odd Maclaurin series through x**9, Horner in x**2. Needs no sanitising of its own: a
    # degree-9 polynomial on |x| <= 1 is bounded (|series| <= 1.788), as is its derivative.
    x2 = x * x
    series = x * (1.0 + x2 * (1.0 / 3.0 + x2 * (1.0 / 5.0 + x2 * (1.0 / 7.0 + x2 / 9.0))))
    return jnp.where(abs_x >= _ATANH_SERIES_SEAM, out, series)


# Where `tanh(x) = expm1(2x)/(expm1(2x) + 2)` takes over, per precision. Below the seam float32
# uses the odd Maclaurin series (see the comment under the constants) and float64 uses `jnp.tanh`.
# Measured with exact bit-pattern ulps on 20k log-uniform inputs per cell against a
# correctly-rounded reference (XLA GPU jax 0.9.1 and 0.11.0; XLA CPU), max ulp `jnp.tanh` ->
# `expm1` form:
#   float32  [1e-4, 0.0625] 4 -> 6 (CPU, a REGRESSION), [0.0625, 0.125] 4 -> 4,
#            [0.125, 0.25] 4 -> 4, [0.25, 0.5] 3 -> 2, [0.5, 0.9] 2 -> 1, [0.9, 7] 4 -> 1 (CPU);
#            [0.9, 8] 4 -> 2 (GPU). So float32 hands over at 1/8: the last bin that is a wash and
#            the first one above the regression.
#   float64  XLA's own tanh is already <= 1 ulp on GPU across the whole range, and the expm1 form
#            is 2 ulp below 0.5 there (a regression); from 0.5 up it ties on GPU (max 1, mean
#            0.055 -> 0.053 on [0.9, 7]) and strictly wins on CPU ([0.5, 0.9] 3 -> 2,
#            [0.9, 7] 5 -> 1). So float64 hands over at 1/2, not 1/8.
_TANH_EXPM1_SEAM_LOW_PRECISION = 0.125
_TANH_EXPM1_SEAM_FLOAT64 = 0.5

# Below the float32 seam neither kernel above is used: the odd Maclaurin series
# `x - x^3/3 + 2x^5/15 - 17x^7/315` is exact to float32 rounding there. The first dropped term is
# `62 x^9 / 2835`, i.e. 1.6e-10 absolute (1.3e-9 relative) at x = 1/8 against a float32 eps of
# 1.2e-7, and smaller for smaller x. It is NOT below float64 eps (1.3e-9 relative is ~6e6 float64
# ulps), hence the `_is_low_precision` gate, exactly as for `atanh`'s series above.


@_jit
def tanh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Hyperbolic tangent clamped so the output stays strictly inside (-1, 1). Domain=(-inf, inf).

    Clips the input to ``±0.5*log(2/machine_eps)`` (≈±8.3 for f32, ≈±18.4 for f64) AND clips the output to
    ``±(1 - 10*machine_eps)``. The input clip is the analytic point where ``1 - tanh`` reaches ``eps``; the
    output clip is required *in addition* because XLA's float32 ``tanh`` saturates to **exactly** ``1.0``
    slightly earlier (≈ x = 8), which the input clip alone does not prevent — and an exact ``1.0`` would make
    a downstream ``atanh`` singular. The output bound matches ``atanh``'s own ``±(1 - 10*eps)`` domain guard,
    so ``atanh(tanh(x))`` can never reach the pole. Outside the saturated tail the guards are a value- and
    gradient-identity (the tail's gradient is 0, as for ``cosh``/``sinh``). Mirrors the domain-guard
    philosophy of ``acosh``/``atanh`` above and the ``tanh(clamp)`` guard in the geoopt κ-stereographic
    reference.

    Away from zero the value comes from the algebraic identity ``tanh(x) = expm1(2x)/(expm1(2x) +
    2)`` rather than XLA's rational approximation to ``tanh``. Evaluated on ``|x|`` (sign restored
    by ``jnp.where``, so the result is odd to the last bit), the form is cancellation-free —
    numerator and denominator are both positive and the denominator never drops below 2 — and it
    keeps the saturating tail accurate, where the whole signal lives in
    ``1 - tanh(x) = 2/(expm1(2|x|) + 2)``. It is *worse* than the builtin near zero (where
    ``expm1(2x) ≈ 2x`` and the division adds roundings the builtin does not pay), so it is used
    only above a measured, precision-dependent seam: 1/8 in float32, 1/2 in float64 (see the
    ``_TANH_EXPM1_SEAM_*`` comment above for the ulp table behind both). Measured on 20k
    log-uniform float32 inputs, max ulp: ``[0.9, 7]`` 4 -> 2 on XLA GPU and 4 -> 1 on XLA CPU.

    Below ``|x| = 1/8`` **in float32** neither kernel is used: the value comes from the odd
    Maclaurin series ``x - x^3/3 + 2x^5/15 - 17x^7/315`` (Horner in ``x**2``, evaluated on
    ``|x|`` so the sign restoration below keeps it odd to the last bit), mirroring ``atanh``'s
    series branch above. The series is exact to float32 rounding on that range — the first
    dropped term is ``62 x^9/2835``, 1.6e-10 absolute (1.3e-9 relative) at ``x = 1/8`` against a
    float32 eps of 1.2e-7 — so it replaces XLA's 4 ulp rational approximation with a polynomial
    whose only error is its own rounding, and unlike the ``expm1`` form it does not pay a
    division near zero. Measured on 20k log-uniform float32 inputs in ``[1e-4, 0.125]``, max /
    mean ulp: 4 / 0.96 -> 1 / 0.16 on XLA CPU and 4 / 0.87 -> 1 / 0.16 on XLA GPU (the
    ``expm1`` form on that same range is 3 on GPU but **6** on CPU, which is why the seam does
    not simply move down). float64 keeps XLA's ``tanh`` below its own 1/2 seam: 1.3e-9 relative
    is ~6e6 float64 ulps, so the series is unusable there.

    Args:
        x: Input array of any shape

    Returns:
        tanh(x) clamped strictly inside (-1, 1)
    """
    # 1 - tanh(x) ≈ 2*exp(-2x); require it ≥ eps ⇒ x ≤ 0.5*log(2/eps) is the analytic bound.
    clamp = 0.5 * jnp.log(2.0 / jnp.finfo(x.dtype).eps)
    # This clip is also the sanitised argument the expm1 branch needs. jnp.where evaluates BOTH
    # branches and their gradients, and expm1(2x) overflows to inf for x > 44 (f32) / 355 (f64),
    # where inf/inf = NaN would leak into the *selected* branch's cotangent. Clipping first bounds
    # the branch argument at ±8.32 (f32) / ±18.4 (f64), so expm1(2x) tops out at 1.7e7 / 9.5e15 —
    # finite in both dtypes, for every real input including ±inf.
    x_safe = clamp_to(x, -clamp, clamp)
    seam = _TANH_EXPM1_SEAM_LOW_PRECISION if _is_low_precision(x.dtype) else _TANH_EXPM1_SEAM_FLOAT64
    # Evaluated on |x| with the sign restored via `where`, exactly as `atanh` above: `expm1(2x)`
    # and `expm1(-2x)` are not negatives of each other, so the raw quotient is not an odd function
    # to the last bit. Restoring the sign this way (rather than with `sign(x)*...`, whose
    # product-rule gradient at x == 0 would collapse to 0) makes tanh(-x) == -tanh(x) bitwise for
    # every input, and keeps the gradient at 0 exactly 1 (jnp.abs' VJP at 0 is +1). The two
    # below-the-seam branches are odd on |x| for the same reason: XLA's own tanh is already
    # bitwise odd, and the series' leading factor is abs_x.
    abs_x = jnp.abs(x_safe)
    t = jnp.expm1(2.0 * abs_x)
    if _is_low_precision(x.dtype):
        # Degree-7 odd Maclaurin series, Horner in abs_x**2. Needs no sanitising of its own: on
        # the clipped |x| <= 8.32 the polynomial and its derivative are bounded (~1.5e5 and
        # ~1.3e5), so the branch jnp.where does not select is still finite in value and cotangent.
        x2 = abs_x * abs_x
        below_seam = abs_x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 - x2 * (17.0 / 315.0))))
    else:
        below_seam = jnp.tanh(abs_x)
    magnitude = jnp.where(abs_x >= seam, t / (t + 2.0), below_seam)
    out = jnp.where(x_safe >= 0, magnitude, -magnitude)
    # Also clamp the output: XLA's float32 tanh reaches exactly 1.0 before the input bound bites.
    max_out = 1.0 - 10.0 * float(jnp.finfo(x.dtype).eps)
    return clamp_to(out, -max_out, max_out)
