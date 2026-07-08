"""Math utils functions for hyperbolic operations with numerically stable limits.

Direct JAX port of PyTorch math_utils.py with type annotations using jaxtyping.
"""

import functools

import jax
import jax.nn as nn
import jax.numpy as jnp
from jaxtyping import Array, Float


@functools.partial(jax.jit, static_argnames=["smoothing_factor"])
def smooth_clamp_min(x: Float[Array, "..."], min_value: float, smoothing_factor: float = 50.0) -> Float[Array, "..."]:
    """Smoothly clamp array values to a minimum using softplus.

    Args:
        x: Input array of any shape
        min_value: Minimum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped above min_value
    """
    eps = jnp.finfo(x.dtype).eps
    shift = min_value + eps
    # Use JAX's numerically stable softplus: softplus_beta(x) = softplus(beta*x)/beta
    arg = smoothing_factor * (x - shift)
    x_clamped = shift + nn.softplus(arg) / smoothing_factor
    return jnp.where(x < shift, x_clamped, x)


@functools.partial(jax.jit, static_argnames=["smoothing_factor"])
def smooth_clamp_max(x: Float[Array, "..."], max_value: float, smoothing_factor: float = 50.0) -> Float[Array, "..."]:
    """Smoothly clamp array values to a maximum using softplus.

    Args:
        x: Input array of any shape
        max_value: Maximum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped below max_value
    """
    eps = jnp.finfo(x.dtype).eps
    shift = max_value - eps
    arg = smoothing_factor * (shift - x)
    x_clamped = shift - nn.softplus(arg) / smoothing_factor
    return jnp.where(x > shift, x_clamped, x)


@functools.partial(jax.jit, static_argnames=["smoothing_factor"])
def smooth_clamp(
    x: Float[Array, "..."], min_value: float, max_value: float, smoothing_factor: float = 50.0
) -> Float[Array, "..."]:
    """Smoothly clamp array values to a range [min_value, max_value].

    Args:
        x: Input array of any shape
        min_value: Minimum value to clamp to
        max_value: Maximum value to clamp to
        smoothing_factor: Beta parameter for softplus (higher = sharper transition)

    Returns:
        Array with values smoothly clamped to [min_value, max_value]
    """
    x = smooth_clamp_max(x, max_value, smoothing_factor=smoothing_factor)
    return smooth_clamp_min(x, min_value, smoothing_factor=smoothing_factor)


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

    Clips the input to ``±0.5*log(2/machine_eps)`` (≈±8.3 for f32, ≈±18.4 for f64) so ``|tanh(x)|``
    never rounds up to exactly 1.0 — which would make a downstream ``atanh`` singular. Within the
    un-clipped regime this is a value- and gradient-identity to ``jnp.tanh``: ``tanh`` saturates well
    before the clip bites, so the forward pass and the VJP match unguarded ``jnp.tanh`` throughout the
    entire non-saturated range. Mirrors the domain-guard philosophy of ``acosh``/``atanh`` above and
    the ``tanh(clamp)`` guard in the geoopt κ-stereographic reference.

    Args:
        x: Input array of any shape

    Returns:
        tanh(x) clamped strictly inside (-1, 1)
    """
    # 1 - tanh(x) ≈ 2*exp(-2x); require it ≥ eps ⇒ x ≤ 0.5*log(2/eps) keeps the output < 1 exactly.
    clamp = 0.5 * jnp.log(2.0 / jnp.finfo(x.dtype).eps)
    x = jnp.clip(x, -clamp, clamp)
    return jnp.tanh(x)


@jax.jit
def asinh(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse hyperbolic sine. Domain=(-inf, inf).

    ``asinh(x) = log(x + sqrt(1 + x^2))`` is already numerically stable over all reals — its derivative
    ``1/sqrt(1 + x^2) in (0, 1]`` has no singularity and the forward pass cannot overflow before its
    argument does — so no clamping is required. Provided as a thin wrapper for API symmetry with
    ``acosh``/``atanh`` and to give the κ-stereographic trig helpers a single stable-math source for the
    ``κ<0`` (hyperbolic) branch of ``arsin_κ``.

    Args:
        x: Input array of any shape

    Returns:
        asinh(x)
    """
    return jnp.asinh(x)
