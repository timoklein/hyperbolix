"""Learnable curvature module for hyperbolic models.

Provides ``LearnableCurvature`` as the canonical way to add a trainable
curvature parameter. Instantiate one per distinct curvature in your model;
call the module at runtime to obtain the (optionally clamped) curvature —
positive for the ``softplus``/``log`` parameterizations, signed (spanning
hyperbolic/Euclidean/spherical, for the ``Stereographic`` manifold) for ``identity``.

Example::

    from flax import nnx
    from hyperbolix import LearnableCurvature
    from hyperbolix.nn_layers import FGGLinear

    class Model(nnx.Module):
        def __init__(self, rngs):
            self.curvature = LearnableCurvature(init_c=1.0)
            self.fc = FGGLinear(33, 65, rngs=rngs)

        def __call__(self, x):
            return self.fc(x, c=self.curvature())

The raw parameter is Euclidean and updated by any ``nnx.Optimizer`` (no
Riemannian optimizer required).
"""

import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from .math_utils import cap_at, clamp_to

Parameterization = Literal["softplus", "log", "identity"]


def _inv_softplus(x: float) -> float:
    """Compute inv_softplus(x) = log(exp(x) - 1) in Python floats."""
    if x <= 0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}")
    if x > 20.0:
        return x + math.log1p(-math.exp(-x))
    return math.log(math.expm1(x))


class Auto:
    """Type of :data:`AUTO`, the default ``c_min``/``c_max`` of :class:`LearnableCurvature`.

    ``AUTO`` means "resolve this clamp bound from the parameterization" (see
    :class:`LearnableCurvature`), as opposed to ``None``, which disables the bound. Layers that
    build ``LearnableCurvature`` instances, such as :class:`~hyperbolix.nn_layers.LorentzMoE`,
    take ``AUTO`` as their default too and forward it. Test for it with
    ``isinstance(bound, Auto)``.
    """

    def __repr__(self) -> str:
        return "AUTO"


AUTO = Auto()
"""Default ``c_min``/``c_max`` of :class:`LearnableCurvature`: resolve the bound from the parameterization."""

# Default clamp bounds. softplus/log use the positive window ``[init_c / _SPAN, init_c * _SPAN]``, a decade either
# side of ``init_c`` (PhyCLIP's range); the signed ``identity`` parameterization uses the symmetric cap
# ``[-_C_ABS_MAX, +_C_ABS_MAX]``, which INCLUDES 0 (the Euclidean point) so it caps ``|c|`` against blow-up without
# ever forbidding the Euclidean/spherical half.
_SPAN = 10.0
_C_ABS_MAX = 10.0


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _clamp_keep_inward(c: jax.Array, c_min: float, c_max: float) -> jax.Array:
    """``c`` clipped to ``[c_min, c_max]``, with a backward that zeroes only a cotangent pointing out of the interval.

    The cotangent ``g`` passes unchanged where ``c`` is strictly inside the interval. On a bound or past it, ``g``
    passes only if a descent step moves ``c`` into the interval: at or below ``c_min`` a negative ``g``, at or
    above ``c_max`` a positive one. The other sign becomes exactly 0, which for a ``c`` exactly on a bound is
    the projected-gradient step. A NaN ``c`` or ``g`` passes through.
    """
    return clamp_to(c, c_min, c_max)


def _clamp_keep_inward_fwd(c: jax.Array, c_min: float, c_max: float) -> tuple[jax.Array, jax.Array]:
    return clamp_to(c, c_min, c_max), c


def _clamp_keep_inward_bwd(c_min: float, c_max: float, c: jax.Array, g: jax.Array) -> tuple[jax.Array]:
    outward = ((c <= c_min) & (g > 0)) | ((c >= c_max) & (g < 0))
    return (jnp.where(outward, 0.0, g),)


_clamp_keep_inward.defvjp(_clamp_keep_inward_fwd, _clamp_keep_inward_bwd)


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
            must also satisfy ``c_min <= init_c <= c_max``; it may sit on a bound.
        parameterization: Reparameterization scheme.

            - ``"log"`` (default): ``c = exp(raw)`` — strictly positive. Scale-invariant
              gradient (``dc/draw = c``); preferred when ``c`` may span orders of
              magnitude or for long compiled RL training loops. Matches the
              MERU convention.
            - ``"softplus"``: ``c = softplus(raw)`` — strictly positive. Gradient
              bounded by ``sigmoid(raw) in (0, 1)``; smooth near zero. The
              reparameterization itself (``raw`` = inverse-softplus of ``c``) is
              geoopt's ``Stereographic``/``PoincareBall`` convention, and is the
              code default (``learnable=True``) in the van Spengler et al. 2023
              Poincare ResNet reference implementation, which builds its manifold
              on geoopt. Note the paper's own reported curvature experiments
              (Sec. 4.2) sweep fixed values ``c in {1, 0.1, 0.01}`` and settle on
              ``c=0.1`` — not this learnable scheme.
            - ``"identity"``: ``c = raw`` — **signed**. Spans hyperbolic (``c>0``),
              Euclidean (``c=0``), and spherical (``c<0``) curvature for the
              :class:`~hyperbolix.manifolds.Stereographic` manifold; ``dc/draw = 1``,
              so it can cross zero. The only parameterization that reaches ``c<=0``.

        c_min: Lower clamp applied to the recovered ``c``. The default,
            :data:`AUTO`, resolves per parameterization: ``init_c / 10`` for
            ``softplus``/``log`` (a decade below the init, PhyCLIP's range),
            ``-10.0`` for the signed ``identity``. Pass ``None`` to disable, or a
            float to override.
        c_max: Upper clamp applied to the recovered ``c``. The default,
            :data:`AUTO`, resolves to ``init_c * 10`` for ``softplus``/``log``
            (``[0.1, 10]`` at the default ``init_c=1.0``), ``10.0`` for
            ``identity``. Pass ``None`` to disable, or a float to override.
        param_dtype: Storage dtype of the raw parameter (default:
            ``jnp.float32``), pinned so it does not become float64 under
            global ``jax_enable_x64``.

    Sharing note: Do **not** assign the same ``LearnableCurvature`` instance
    to multiple fields if you want independent learnable curvatures —
    instantiate one per location. Sharing creates a shared-reference
    pattern in the NNX pytree that breaks ``nnx.scan`` / ``nnx.fori_loop``
    (same root cause as the pre-refactor manifold bug).

    Clamp: the forward value is the recovered ``c`` clipped to
    ``[c_min, c_max]``. Strictly inside the interval the gradient is the full
    chain rule, ``dc/draw`` times the incoming gradient. On a bound or outside
    it, the gradient passes only if it points inside (at or below ``c_min``, one
    under which gradient descent raises ``c``; at or above ``c_max``, one under
    which it lowers ``c``); an outward gradient is set to exactly 0. This is
    projected gradient descent with the projection folded into the forward pass,
    so the train step needs no extra call: ``c`` rests on a bound while the loss
    pushes it outward and comes off it once the loss pulls it back, and a blocked
    gradient does not feed Adam's moments. ``raw`` itself is not clamped, but it
    does not drift: it leaves the interval only by the step that crossed the bound
    (plus the optimizer's decaying momentum). Coming back takes only the few steps
    that walk that overshoot back, each sized like an interior step at the bound,
    so no damping is needed. A ``c`` that sits on a bound for many steps is held
    there by the loss; widen the bound if that is unwanted. The clamp sees each
    call's gradient separately, so call the instance once per forward and reuse
    ``c``: in a toy fit where two calls of one instance pulled opposite ways at
    the floor, ``c`` hovered up to 0.21 % above it under Adam at ``1e-2`` (a
    level reached after about 10,000 steps and then held; 0.15 % under SGD),
    while one reused call held it on the floor.

    Forward mode: the clamp's backward is a custom VJP, so ``jax.jvp`` and
    ``jax.jacfwd`` through a clamped instance raise a ``TypeError``
    (``jax.hessian``, which is forward-over-reverse, works); with
    ``c_min=None, c_max=None`` it is plain autodiff.

    Init: ``raw`` is the inverse of ``init_c`` rounded to ``param_dtype``. For an
    ``init_c`` on a bound, that rounding can recover a ``c`` exactly on it, one
    float outside it (float32 ``exp(float32(log 0.1)) = 0.099999994``) or one
    float inside it (float64 ``exp(log 0.1) = 0.10000000000000002``). On or
    outside the bound, ``c`` starts on the bound, where from the first step the
    outward gradient is blocked and the inward one passes. One float inside,
    ``c`` starts as an interior point.
    """

    def __init__(
        self,
        init_c: float = 1.0,
        *,
        parameterization: Parameterization = "log",
        c_min: float | None | Auto = AUTO,
        c_max: float | None | Auto = AUTO,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if parameterization not in ("softplus", "log", "identity"):
            raise ValueError(f"parameterization must be 'softplus', 'log', or 'identity', got {parameterization!r}")
        signed = parameterization == "identity"

        # NaN slips through every range check below (all comparisons with NaN are False), silently storing
        # a NaN raw param that poisons the whole model — reject it up front.
        if not math.isfinite(init_c):
            raise ValueError(f"init_c must be finite, got {init_c}")
        # softplus/exp map onto (0, inf) and cannot represent c <= 0, so a non-positive init is a usage error
        # there. identity is signed and accepts any init_c (including 0.0 and negatives).
        if not signed and init_c <= 0:
            raise ValueError(f"LearnableCurvature requires init_c > 0 for parameterization {parameterization!r}, got {init_c}")

        # Resolve sentinel clamp bounds from the parameterization: softplus/log default to a decade either side
        # of init_c; identity uses a symmetric magnitude cap [-10, 10]. Explicit None still disables a bound;
        # explicit numeric bounds are honored verbatim.
        if isinstance(c_min, Auto):
            c_min = -_C_ABS_MAX if signed else init_c / _SPAN
        if isinstance(c_max, Auto):
            c_max = _C_ABS_MAX if signed else init_c * _SPAN

        if c_min is not None and c_max is not None and c_min > c_max:
            raise ValueError(f"c_min ({c_min}) must be <= c_max ({c_max})")
        if c_min is not None and init_c < c_min:
            raise ValueError(f"init_c ({init_c}) must be >= c_min ({c_min})")
        if c_max is not None and init_c > c_max:
            raise ValueError(f"init_c ({init_c}) must be <= c_max ({c_max})")

        self._parameterization = parameterization
        self._c_min = c_min
        self._c_max = c_max

        self.raw = nnx.Param(jnp.array(self._inverse(init_c), dtype=param_dtype))

    def _inverse(self, c: float) -> float:
        """The raw value the parameterization maps to ``c``, in Python floats."""
        if self._parameterization == "softplus":
            return _inv_softplus(c)
        if self._parameterization == "log":
            return math.log(c)
        return c  # "identity"

    def _unclamped(self, raw: jax.Array) -> jax.Array:
        """The recovered curvature before the clamp."""
        if self._parameterization == "softplus":
            return jax.nn.softplus(raw)
        if self._parameterization == "log":
            # Cap the exponent so exp() cannot overflow to +inf: an inf here turns the clamp's cotangent times
            # dc/draw into 0*inf = NaN. Below the cap this is a value/grad identity; above it c is ~1e38, far
            # past any sensible c_max, so nothing meaningful is lost.
            max_exp = 0.99 * math.log(float(jnp.finfo(raw.dtype).max))
            return jnp.exp(cap_at(raw, max_exp))
        return raw  # "identity"

    def __call__(self) -> jax.Array:
        c = self._unclamped(self.raw[...])
        if self._c_min is None and self._c_max is None:
            return c
        c_min = -math.inf if self._c_min is None else self._c_min
        c_max = math.inf if self._c_max is None else self._c_max
        return _clamp_keep_inward(c, c_min, c_max)
