"""Learnable curvature module for hyperbolic models.

Provides ``LearnableCurvature`` as the canonical way to add a trainable
curvature parameter. Instantiate one per distinct curvature in your model;
call the module at runtime to obtain the (optionally clamped) curvature —
positive for the ``softplus``/``log`` parameterizations, signed (spanning
hyperbolic/Euclidean/spherical, for the ``Stereographic`` manifold) for ``identity``.

Example::

    from hyperbolix import LearnableCurvature
    from hyperbolix.manifolds import Hyperboloid

    class Model(nnx.Module):
        def __init__(self, rngs):
            self.manifold = Hyperboloid(c=1.0)
            self.curvature = LearnableCurvature(init_c=1.0)
            self.fc = FGGLinear(33, 65, rngs=rngs)

        def __call__(self, x):
            return self.fc(x, c=self.curvature())

The raw parameter is Euclidean and updated by any ``nnx.Optimizer`` (no
Riemannian optimizer required).
"""

import math
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

Parameterization = Literal["softplus", "log", "identity"]


def _inv_softplus(x: float) -> float:
    """Compute inv_softplus(x) = log(exp(x) - 1) in Python floats."""
    if x <= 0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}")
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class _Auto:
    """Sentinel type: 'resolve this clamp bound from the parameterization' (distinct from ``None`` = disabled)."""


_AUTO = _Auto()

# Default clamp magnitudes. softplus/log use the positive window ``[_C_MIN_POS, _C_ABS_MAX]``; the signed
# ``identity`` parameterization uses the symmetric cap ``[-_C_ABS_MAX, +_C_ABS_MAX]``, which INCLUDES 0 (the
# Euclidean point) so it caps ``|c|`` against blow-up without ever forbidding the Euclidean/spherical half.
_C_MIN_POS = 0.1
_C_ABS_MAX = 10.0


class LearnableCurvature(nnx.Module):
    """Reparameterized learnable curvature parameter.

    Stores a single Euclidean ``nnx.Param`` whose value is mapped to a
    curvature on every forward call — positive for ``softplus``/``log``, signed
    for ``identity``. Three parameterizations are supported, with optional
    clamping of the recovered curvature to ``[c_min, c_max]`` for hard stability
    guarantees in compiled training loops; the ``log`` parameterization additionally
    caps its exponent so a large ``raw`` cannot overflow ``exp`` to a NaN gradient.

    Usage::

        self.curvature = LearnableCurvature(init_c=0.1)
        ...
        c = self.curvature()  # positive jax.Array

    Args:
        init_c: Initial curvature value. Must be positive for ``softplus``/``log``;
            any sign (including ``0.0``) for ``identity``. If clamp bounds are set,
            must also satisfy ``c_min <= init_c <= c_max``.
        parameterization: Reparameterization scheme.

            - ``"softplus"`` (default): ``c = softplus(raw)`` — strictly positive.
              Gradient bounded by ``sigmoid(raw) in (0, 1)``; smooth near zero;
              matches the van Spengler et al. 2023 Poincare ResNet convention.
            - ``"log"``: ``c = exp(raw)`` — strictly positive. Scale-invariant
              gradient (``dc/draw = c``); preferred when ``c`` may span orders of
              magnitude or for long compiled RL training loops. Matches the
              MERU convention.
            - ``"identity"``: ``c = raw`` — **signed**. Spans hyperbolic (``c>0``),
              Euclidean (``c=0``), and spherical (``c<0``) curvature for the
              :class:`~hyperbolix.manifolds.Stereographic` manifold; ``dc/draw = 1``,
              so it can cross zero. The only parameterization that reaches ``c<=0``.

        c_min: Lower clamp applied to the recovered ``c``. Default resolves per
            parameterization: ``0.1`` for ``softplus``/``log``, ``-10.0`` for the
            signed ``identity``. Pass ``None`` to disable, or a float to override.
        c_max: Upper clamp applied to the recovered ``c``. Default resolves to
            ``10.0``. Pass ``None`` to disable, or a float to override.
        straight_through_clamp: If ``True``, the clamp is gradient-transparent:
            the forward value is still clamped to ``[c_min, c_max]``, but the
            backward gradient is identity rather than zero, so ``raw`` can keep
            moving and ``c`` can re-enter the interval once the loss pulls the
            other way (default: ``False`` — plain ``jnp.clip``, see the
            gradient-dead note below).
        param_dtype: Storage dtype of the raw parameter (default:
            ``jnp.float32``), pinned so it does not become float64 under
            global ``jax_enable_x64``.

    Sharing note: Do **not** assign the same ``LearnableCurvature`` instance
    to multiple fields if you want independent learnable curvatures —
    instantiate one per location. Sharing creates a shared-reference
    pattern in the NNX pytree that breaks ``nnx.scan`` / ``nnx.fori_loop``
    (same root cause as the pre-refactor manifold bug).

    Gradient-dead clamp (default behavior): plain ``jnp.clip`` has zero
    gradient outside ``[c_min, c_max]``. If ``raw`` drifts far enough that the
    recovered ``c`` exits the clamp interval, the gradient to ``raw`` becomes
    permanently zero — ``c`` is pinned at the boundary and cannot re-enter the
    interval even if the loss would eventually pull it back. Monitor
    ``curvature.raw`` (or ``curvature()`` against the clamp bounds) in
    training logs: a curvature sitting exactly at ``c_min``/``c_max`` for many
    steps is "pinned", not "chosen". Pass ``straight_through_clamp=True`` to
    keep the forward safety guarantee while eliminating the ratchet.
    """

    def __init__(
        self,
        init_c: float = 1.0,
        *,
        parameterization: Parameterization = "softplus",
        c_min: float | None | _Auto = _AUTO,
        c_max: float | None | _Auto = _AUTO,
        straight_through_clamp: bool = False,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if parameterization not in ("softplus", "log", "identity"):
            raise ValueError(f"parameterization must be 'softplus', 'log', or 'identity', got {parameterization!r}")
        signed = parameterization == "identity"

        # Resolve sentinel clamp bounds from the parameterization: softplus/log keep the historical positive
        # window [0.1, 10]; identity uses a symmetric magnitude cap [-10, 10]. Explicit None still disables
        # the clamp; explicit numeric bounds are honored verbatim.
        if isinstance(c_min, _Auto):
            c_min = -_C_ABS_MAX if signed else _C_MIN_POS
        if isinstance(c_max, _Auto):
            c_max = _C_ABS_MAX

        # NaN slips through every range check below (all comparisons with NaN are False), silently storing
        # a NaN raw param that poisons the whole model — reject it up front.
        if not math.isfinite(init_c):
            raise ValueError(f"init_c must be finite, got {init_c}")
        # softplus/exp map onto (0, inf) and cannot represent c <= 0, so a non-positive init is a usage error
        # there. identity is signed and accepts any init_c (including 0.0 and negatives).
        if not signed and init_c <= 0:
            raise ValueError(f"LearnableCurvature requires init_c > 0 for parameterization {parameterization!r}, got {init_c}")
        if c_min is not None and c_max is not None and c_min > c_max:
            raise ValueError(f"c_min ({c_min}) must be <= c_max ({c_max})")
        if c_min is not None and init_c < c_min:
            raise ValueError(f"init_c ({init_c}) must be >= c_min ({c_min})")
        if c_max is not None and init_c > c_max:
            raise ValueError(f"init_c ({init_c}) must be <= c_max ({c_max})")

        self._parameterization = parameterization
        self._c_min = c_min
        self._c_max = c_max
        self._straight_through_clamp = straight_through_clamp

        if parameterization == "softplus":
            raw_init = _inv_softplus(init_c)
        elif parameterization == "log":
            raw_init = math.log(init_c)
        else:  # "identity"
            raw_init = init_c

        self.raw = nnx.Param(jnp.array(raw_init, dtype=param_dtype))

    def __call__(self) -> jax.Array:
        if self._parameterization == "softplus":
            c = jax.nn.softplus(self.raw[...])
        elif self._parameterization == "log":
            # Cap the exponent so exp() cannot overflow to +inf: an inf here makes the downstream clip's
            # out-of-range cotangent 0*inf = NaN (and, under straight_through_clamp, NaNs the forward value
            # via inf + (-inf)). Below the cap this is a value/grad identity; above it c is already pinned at
            # c_max by the clamp anyway, so nothing meaningful is lost.
            raw = self.raw[...]
            max_exp = 0.99 * math.log(float(jnp.finfo(raw.dtype).max))
            c = jnp.exp(jnp.minimum(raw, max_exp))
        else:  # "identity"
            c = self.raw[...]

        if self._c_min is not None or self._c_max is not None:
            c_clipped = jnp.clip(c, self._c_min, self._c_max)
            if self._straight_through_clamp:
                # Forward value stays clamped; backward gradient becomes identity instead of zero, so
                # `raw` can keep moving and `c` can re-enter the interval once the loss pulls the other way.
                # Numerically stable form: `c - stop_gradient(c)` is exactly 0, so the forward equals
                # c_clipped to full precision. The algebraically-equivalent `c + stop_gradient(c_clipped - c)`
                # cancels catastrophically when c ≫ c_clipped (e.g. log-param with a large raw → c ~ 1e38,
                # c_clipped = c_max: `c_max - c` loses c_max, and the sum collapses to 0).
                c = jax.lax.stop_gradient(c_clipped) + (c - jax.lax.stop_gradient(c))
            else:
                c = c_clipped

        return c
