"""Math utils functions for hyperbolic operations with numerically stable limits.

Direct JAX port of PyTorch math_utils.py with type annotations using jaxtyping.
"""

import functools

import jax
import jax.nn as nn
import jax.numpy as jnp
from jaxtyping import Array, Float

# Canonical gradient-safety floor for norms and denominators, shared library-wide
# (`manifolds/`, `nn_layers/`, `distributions/`, `decomposition/` all import this name).
# Two uses, both about derivatives rather than values:
#   * `sqrt(sum(x**2) + MIN_NORM**2)` has a finite VJP at x = 0, unlike `linalg.norm`, whose
#     VJP there is 0/0 = NaN.
#   * `maximum(denom, MIN_NORM)` keeps a division finite when the denominator collapses.
# 1e-15 is small enough to be invisible next to any float32 or float64 quantity the library
# works with (float64 eps is 2.2e-16), and `MIN_NORM**2 = 1e-30` is still normal in both
# dtypes (float32's smallest normal is 1.2e-38), so squaring it does not flush to zero.
MIN_NORM = 1e-15


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


@functools.partial(jax.jit, static_argnames=["smoothing_factor"])
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


@functools.partial(jax.jit, static_argnames=["smoothing_factor"])
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


@functools.partial(jax.jit, static_argnames=["smoothing_factor"])
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
    underflow to 0 and the output equals ``min_value``/``max_value`` exactly (``beta*d`` past ~88 in
    float32, ~745 in float64). Inclusion in ``[min_value, max_value]`` always holds.

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


@jax.jit
def cosh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Hyperbolic cosine with overflow protection. Domain=(-inf, inf).

    Hard-clips the input to ``±0.99*log(finfo.max)`` (≈±87.8 for f32, ±709 for f64) before
    ``jnp.cosh`` so the result cannot overflow the dtype. This is a *pure overflow guard*: for any
    input that is not about to overflow the clip is a value- and gradient-identity, so the forward
    pass and the VJP match an unguarded ``jnp.cosh`` throughout the entire valid regime.

    A hard ``jnp.clip`` is used deliberately rather than a softplus ``smooth_clamp``: it matches the
    domain guards in ``acosh``/``atanh`` below, is free on accelerators, and avoids the ~2 extra
    ``exp`` per call that the smooth clamp evaluated on every element with no benefit. The only
    difference is in the saturated tail (|x| ≥ clamp), where the clip's gradient is 0 instead of a
    tiny nonzero value — acceptable, because that regime is already degenerate (the output is ~1e37)
    and you never want gradients pushing further into overflow.

    Args:
        x: Input array of any shape

    Returns:
        cosh(x) with overflow protection
    """
    # cosh(x) ≈ exp(x)/2 for large x, so the overflow boundary is x = log(max).
    clamp = jnp.log(jnp.finfo(x.dtype).max) * 0.99
    x = jnp.clip(x, -clamp, clamp)
    return jnp.cosh(x)


@jax.jit
def sinh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Hyperbolic sine with overflow protection. Domain=(-inf, inf).

    Hard-clips the input to ``±0.99*log(finfo.max)`` (≈±87.8 for f32, ±709 for f64) before
    ``jnp.sinh`` so the result cannot overflow the dtype. This is a *pure overflow guard*: for any
    input that is not about to overflow the clip is a value- and gradient-identity, so the forward
    pass and the VJP match an unguarded ``jnp.sinh`` throughout the entire valid regime.

    A hard ``jnp.clip`` is used deliberately rather than a softplus ``smooth_clamp``: it matches the
    domain guards in ``acosh``/``atanh`` below, is free on accelerators, and avoids the ~2 extra
    ``exp`` per call that the smooth clamp evaluated on every element with no benefit. The only
    difference is in the saturated tail (|x| ≥ clamp), where the clip's gradient is 0 instead of a
    tiny nonzero value — acceptable, because that regime is already degenerate (the output is ~1e37)
    and you never want gradients pushing further into overflow.

    Args:
        x: Input array of any shape

    Returns:
        sinh(x) with overflow protection
    """
    # sinh(x) ≈ exp(x)/2 for large x, so the overflow boundary is x = log(max).
    clamp = jnp.log(jnp.finfo(x.dtype).max) * 0.99
    x = jnp.clip(x, -clamp, clamp)
    return jnp.sinh(x)


@jax.jit
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
    x = jnp.clip(x, 1.0 + eps, None)
    return jnp.acosh(x)


@jax.jit
def atanh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse hyperbolic tangent with domain clamping. Domain=(-1, 1).

    Clamps input to ``±(1 - 10*machine_eps)``. The factor 10 keeps the
    clamped value safely representable away from ±1.0 (where the float grid
    is coarsest) and bounds ``atanh'`` at ~1/(2*margin) instead of letting
    inputs ride the last representable value before the singularity.

    Args:
        x: Input array of any shape

    Returns:
        atanh(x) with domain and gradient protection
    """
    eps = 10.0 * float(jnp.finfo(x.dtype).eps)
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return jnp.atanh(x)


@jax.jit
def tanh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Hyperbolic tangent clamped so the output stays strictly inside (-1, 1). Domain=(-inf, inf).

    Clips the input to ``±0.5*log(2/machine_eps)`` (≈±8.3 for f32, ≈±18.4 for f64) AND clips the output to
    ``±(1 - 10*machine_eps)``. The input clip is the analytic point where ``1 - tanh`` reaches ``eps``; the
    output clip is required *in addition* because XLA's float32 ``tanh`` saturates to **exactly** ``1.0``
    slightly earlier (≈ x = 8), which the input clip alone does not prevent — and an exact ``1.0`` would make
    a downstream ``atanh`` singular. The output bound matches ``atanh``'s own ``±(1 - 10*eps)`` domain guard,
    so ``atanh(tanh(x))`` can never reach the pole. Within the non-saturated range this is a value- and
    gradient-identity to ``jnp.tanh``; only the saturated tail is affected (gradient 0 there, as for
    ``cosh``/``sinh``). Mirrors the domain-guard philosophy of ``acosh``/``atanh`` above and the
    ``tanh(clamp)`` guard in the geoopt κ-stereographic reference.

    Args:
        x: Input array of any shape

    Returns:
        tanh(x) clamped strictly inside (-1, 1)
    """
    # 1 - tanh(x) ≈ 2*exp(-2x); require it ≥ eps ⇒ x ≤ 0.5*log(2/eps) is the analytic bound.
    clamp = 0.5 * jnp.log(2.0 / jnp.finfo(x.dtype).eps)
    out = jnp.tanh(jnp.clip(x, -clamp, clamp))
    # Also clamp the output: XLA's float32 tanh reaches exactly 1.0 before the input bound bites.
    max_out = 1.0 - 10.0 * float(jnp.finfo(x.dtype).eps)
    return jnp.clip(out, -max_out, max_out)
