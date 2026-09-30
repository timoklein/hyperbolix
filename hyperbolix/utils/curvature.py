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

from .math_utils import cap_at, floor_at

Parameterization = Literal["softplus", "log", "identity"]


def _inv_softplus(x: float) -> float:
    """Compute inv_softplus(x) = log(exp(x) - 1) in Python floats."""
    if x <= 0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}")
    if x > 20.0:
        return x + math.log1p(-math.exp(-x))
    return math.log(math.expm1(x))


class _Auto:
    """Sentinel type: 'resolve this clamp bound from the parameterization' (distinct from ``None`` = disabled)."""


_AUTO = _Auto()

# Default clamp magnitudes. softplus/log use the positive window ``[_C_MIN_POS, _C_ABS_MAX]``; the signed
# ``identity`` parameterization uses the symmetric cap ``[-_C_ABS_MAX, +_C_ABS_MAX]``, which INCLUDES 0 (the
# Euclidean point) so it caps ``|c|`` against blow-up without ever forbidding the Euclidean/spherical half.
_C_MIN_POS = 0.1
_C_ABS_MAX = 10.0


def _ulp_step(x: jax.Array, n: int) -> jax.Array:
    """The float ``n`` representable values above the finite 0-d ``x`` (below it for ``n < 0``).

    Steps along an integer key of the IEEE bit pattern that orders finite floats monotonically (both zeros
    share key 0), so a step of any size costs one evaluation. Saturates at the largest finite float: a step
    past it would leave the integer type (``LearnableCurvature(3e38, c_min=3e38, c_max=None)``, whose
    exponent cap makes ``c`` flat in ``raw``, gallops that far).
    """
    width = x.dtype.itemsize
    int_dtype = {2: jnp.int16, 4: jnp.int32, 8: jnp.int64}[width]
    sign_bit = 1 << (8 * width - 1)
    max_key = int(jax.lax.bitcast_convert_type(jnp.array(jnp.finfo(x.dtype).max, dtype=x.dtype), int_dtype))
    bits = int(jax.lax.bitcast_convert_type(x, int_dtype))
    key = (bits if bits >= 0 else -(bits + sign_bit)) + n
    key = max(-max_key, min(max_key, key))
    bits = key if key >= 0 else -key - sign_bit
    return jax.lax.bitcast_convert_type(jnp.array(bits, dtype=int_dtype), x.dtype)


class LearnableCurvature(nnx.Module):
    """Reparameterized learnable curvature parameter.

    Stores a single Euclidean ``nnx.Param`` whose value is mapped to a
    curvature on every forward call — positive for ``softplus``/``log``, signed
    for ``identity``. Three parameterizations are supported, with optional
    clamping of the recovered curvature to ``[c_min, c_max]`` for hard stability
    guarantees in compiled training loops; the ``log`` parameterization additionally
    caps its exponent so a large ``raw`` cannot overflow ``exp`` to a NaN gradient.

    Usage::

        self.curvature = LearnableCurvature(init_c=0.1, straight_through_clamp=True)
        ...
        c = self.curvature()  # positive jax.Array

    Args:
        init_c: Initial curvature value. Must be positive for ``softplus``/``log``;
            any sign (including ``0.0``) for ``identity``. If clamp bounds are set,
            must also satisfy ``c_min <= init_c <= c_max``. It may sit on a bound:
            ``raw`` is then nudged so the recovered ``c`` lies strictly inside, and
            ``c()`` differs from ``init_c`` by a few ulps (at most 4 at the default
            bounds; see the init note below).
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

        c_min: Lower clamp applied to the recovered ``c``. Default resolves per
            parameterization: ``0.1`` for ``softplus``/``log``, ``-10.0`` for the
            signed ``identity``. Pass ``None`` to disable, or a float to override.
        c_max: Upper clamp applied to the recovered ``c``. Default resolves to
            ``10.0``. Pass ``None`` to disable, or a float to override.
        straight_through_clamp: If ``True``, the clamp is gradient-transparent:
            the forward value is still clamped to ``[c_min, c_max]``, but the
            backward gradient flows instead of being zeroed, so ``raw`` can keep
            moving and ``c`` can re-enter the interval once the loss pulls the
            other way (default: ``False`` — a hard clamp, see the
            gradient-dead note below). Outside the interval that pass-through
            gradient is damped to ``O(1)``; see the re-entry step size note below.
        param_dtype: Storage dtype of the raw parameter (default:
            ``jnp.float32``), pinned so it does not become float64 under
            global ``jax_enable_x64``.

    Sharing note: Do **not** assign the same ``LearnableCurvature`` instance
    to multiple fields if you want independent learnable curvatures —
    instantiate one per location. Sharing creates a shared-reference
    pattern in the NNX pytree that breaks ``nnx.scan`` / ``nnx.fori_loop``
    (same root cause as the pre-refactor manifold bug).

    Gradient-dead clamp (default behavior): the clamp (``floor_at``/``cap_at``,
    a ``where`` against each bound) passes the full gradient on
    ``[c_min, c_max]``, an exact tie included, and zero gradient strictly
    outside it. If ``raw`` drifts far enough that the
    recovered ``c`` exits the clamp interval, the gradient to ``raw`` becomes
    permanently zero — ``c`` is pinned at the boundary and cannot re-enter the
    interval even if the loss would eventually pull it back. Monitor
    ``curvature.raw`` (or ``curvature()`` against the clamp bounds) in
    training logs: a curvature sitting exactly at ``c_min``/``c_max`` for many
    steps is "pinned", not "chosen". Pass ``straight_through_clamp=True`` to
    keep the forward safety guarantee while eliminating the ratchet.

    Init on a bound: rounding the inverse of ``init_c`` to the storage dtype can
    put the recovered ``c`` outside the clamp interval — in float32
    ``exp(float32(log 0.1)) = 0.099999994 < 0.1`` — which left the default
    ``LearnableCurvature(init_c=0.1)`` with a raw gradient of exactly 0 from
    step 0 and ``c`` frozen for the whole run (float64 ``init_c=10.0`` rounded
    above ``c_max`` the same way). ``raw`` is therefore nudged at
    construction, by as few ulps as needed, until the recovered ``c`` lies
    strictly inside ``(c_min, c_max)``, so ``dc/draw`` at init is the full
    analytic derivative (``c`` for ``log``, ``sigmoid(raw)`` for ``softplus``,
    ``1`` for ``identity``); an init already strictly inside keeps its ``raw``
    bit for bit. This only makes ``c`` live at step 0: under the default clamp a
    single step that pushes ``c`` below ``c_min`` still pins it there. When the
    loss may pull ``c`` below an init on ``c_min``, pass
    ``straight_through_clamp=True``.

    Re-entry step size under ``straight_through_clamp=True``: outside
    ``[c_min, c_max]`` the pass-through gradient is divided by
    ``max(dc/draw, 1)``, so ``d(c_out)/d(raw) ~= 1`` there and a re-entry step is
    sized by the loss gradient alone rather than by the parameterization's own
    gain. Without this, ``log`` (whose ``dc/draw = c`` is scale-invariant, not
    bounded) overshoots catastrophically: at ``raw = 15`` the un-damped
    pass-through gradient is ``2*(c_max - 1)*exp(15) ~= 5.9e7``, and a single
    plain-SGD step at ``lr = 0.05`` moves ``raw`` by ``~-2.9e6`` — clean over the
    whole interval, landing pinned at the *opposite* wall on step one (audit
    finding). Switching to ``optax.adam`` did not rescue that: its normalized step
    moves ``raw`` by only ``~lr`` per step, so it avoids the overshoot but needs
    ``~(raw - log(c_max)) / lr`` steps to walk back, and the 200-step repro stayed
    pinned at ``c_max``. The guard is deliberately one-sided: it only ever damps a
    gain above 1, never amplifies one below it. Per boundary and
    parameterization — ``log`` past ``c_max``: ``dc/draw = c >> 1``, damped to
    ``~1`` (the case this exists for); ``log`` below ``c_min``: ``dc/draw = c < 1``,
    left alone, since dividing by it would be the mirror-image blow-up;
    ``softplus`` at either boundary: ``dc/draw = sigmoid(raw) <= 1``, an exact
    no-op (and far below ``c_min`` ``sigmoid(raw) -> 0``, exactly where a
    two-sided rescale would explode); ``identity``: ``dc/draw = 1``, an exact
    no-op. Gradients while ``c`` is strictly *inside* the interval are untouched
    genuine chain-rule gradients, so ``log`` keeps its scale-invariant interior
    dynamics (the MERU convention) — only the fake straight-through signal
    outside the clamp is rescaled.
    """

    def __init__(
        self,
        init_c: float = 1.0,
        *,
        parameterization: Parameterization = "log",
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

        self.raw = nnx.Param(self._initial_raw(init_c, param_dtype))

    def _inverse(self, c: float) -> float:
        """The raw value the parameterization maps to ``c``, in Python floats."""
        if self._parameterization == "softplus":
            return _inv_softplus(c)
        if self._parameterization == "log":
            return math.log(c)
        return c  # "identity"

    def _initial_raw(self, init_c: float, param_dtype: DTypeLike) -> jax.Array:
        """The stored ``raw``: the inverse of ``init_c``, nudged by ulps so the clamp is live at init.

        Rounding the inverse to the storage dtype can put the recovered ``c`` on or outside a bound that
        ``init_c`` sits on: in float32, ``exp(float32(log 0.1)) = 0.099999994 < 0.1``, so the default
        ``LearnableCurvature(init_c=0.1)`` started strictly below ``c_min``, where the clamp's gradient is
        exactly 0, and never moved; float64 ``exp(float64(log 10))`` rounds above ``c_max = 10`` the same
        way. So when the recovered ``c`` is not strictly inside ``(c_min, c_max)``, ``raw`` is stepped toward
        the interior by the fewest ulps that bring ``c`` strictly inside (``c`` is increasing in ``raw`` for
        every parameterization): a galloping search over 1, 2, 4, ... ulps, then a bisection back to the first
        step that crosses. It searches ``raw`` itself rather than inverting
        nearby curvatures, so it does not rely on the inverse being accurate to the last ulp.

        Strictly inside, not on a bound: the clamp here passes the full gradient at a tie, but a
        ``jnp.clip``-style clamp halves it there, and a backend whose ``exp``/``softplus`` rounds one ulp
        differently from the construction backend could put a tie outside. ``c`` is computed by
        ``_unclamped``, the forward's own ops, and compared in the storage dtype, as ``floor_at``/``cap_at``
        compare it. An init already strictly inside keeps the plain inverse bit for bit.

        Runs eagerly under ``jax.ensure_compile_time_eval``, so a module built under ``nnx.jit`` or
        ``nnx.eval_shape`` still gets a concrete ``raw``. Bounds with no float strictly between them target
        the closed interval instead. If no ``raw`` reaches the target — bounds a few ulps apart, which one
        ``raw`` ulp can jump — the plain inverse is kept.
        """
        with jax.ensure_compile_time_eval():
            raw = jnp.array(self._inverse(init_c), dtype=param_dtype)
            if self._c_min is None and self._c_max is None:
                return raw
            dtype = raw.dtype  # float32 when x64 is off, whatever param_dtype asked for
            lo = jnp.array(-jnp.inf if self._c_min is None else self._c_min, dtype=dtype)
            hi = jnp.array(jnp.inf if self._c_max is None else self._c_max, dtype=dtype)
            # One representable value inside each bound; for bounds with no value strictly between them,
            # the closed interval (a tie is the best there is).
            lo_in, hi_in = jnp.nextafter(lo, hi), jnp.nextafter(hi, lo)
            if not bool(lo_in <= hi_in):
                lo_in, hi_in = lo, hi

            def inside(r: jax.Array) -> bool:
                c = self._unclamped(r)
                return bool((c >= lo_in) & (c <= hi_in))

            if inside(raw):
                return raw
            step = 1 if bool(self._unclamped(raw) < lo_in) else -1

            def crossed(k: int) -> bool:  # has a step of k ulps brought c past the violated bound?
                c = self._unclamped(_ulp_step(raw, step * k))
                return bool(c >= lo_in) if step > 0 else bool(c <= hi_in)

            # Largest step tried: 2**30 ulps in float32 (from raw = 0 that reaches raw = 2, far past any
            # crossing), 2**62 in float64.
            limit = 1 << (8 * dtype.itemsize - 2)
            short, long = 0, 1  # invariant: crossed(long) and not crossed(short)
            while not crossed(long):
                short, long = long, 2 * long
                if long > limit:
                    return raw
            while long - short > 1:
                mid = (short + long) // 2
                short, long = (short, mid) if crossed(mid) else (mid, long)
            candidate = _ulp_step(raw, step * long)
            return candidate if inside(candidate) else raw

    def _unclamped(self, raw: jax.Array) -> jax.Array:
        """The recovered curvature before the clamp."""
        if self._parameterization == "softplus":
            return jax.nn.softplus(raw)
        if self._parameterization == "log":
            # Cap the exponent so exp() cannot overflow to +inf: an inf here makes the downstream clip's
            # out-of-range cotangent 0*inf = NaN (and, under straight_through_clamp, NaNs the forward value
            # via inf + (-inf)). Below the cap this is a value/grad identity; above it c is already pinned at
            # c_max by the clamp anyway, so nothing meaningful is lost.
            max_exp = 0.99 * math.log(float(jnp.finfo(raw.dtype).max))
            capped = cap_at(raw, max_exp)
            if self._straight_through_clamp:
                # Honor the straight-through contract at the exponent cap too: forward still uses the
                # capped exponent (exp cannot overflow), but the backward is identity through the min, so
                # dc/draw = exp(capped) stays nonzero and a raw that drifted past the cap is not frozen
                # there forever (a plain minimum has zero gradient above the cap — permanent freeze).
                capped = jax.lax.stop_gradient(capped) + raw - jax.lax.stop_gradient(raw)
            return jnp.exp(capped)
        return raw  # "identity"

    def __call__(self) -> jax.Array:
        raw = self.raw[...]
        c = self._unclamped(raw)

        if self._c_min is not None or self._c_max is not None:
            c_clipped = c
            if self._c_min is not None:
                c_clipped = floor_at(c_clipped, self._c_min)
            if self._c_max is not None:
                c_clipped = cap_at(c_clipped, self._c_max)
            if self._straight_through_clamp:
                # Forward value stays clamped; backward gradient flows instead of being zeroed, so `raw` can
                # keep moving and `c` can re-enter the interval once the loss pulls the other way.
                # Numerically stable form: the pass-through term is exactly 0 in the forward (it is
                # `y - stop_gradient(y)`), so the forward equals c_clipped to full precision. The
                # algebraically-equivalent `c + stop_gradient(c_clipped - c)` cancels catastrophically when
                # c ≫ c_clipped (e.g. log-param with a large raw → c ~ 1e38, c_clipped = c_max: `c_max - c`
                # loses c_max, and the sum collapses to 0).
                c = jax.lax.stop_gradient(c_clipped) + self._pass_through(raw, c, c_clipped)
            else:
                c = c_clipped

        return c

    def _pass_through(self, raw: jax.Array, c: jax.Array, c_clipped: jax.Array) -> jax.Array:
        """Straight-through term: exactly ``0.0`` in the forward, carrying the gradient handed to ``raw``.

        Inside ``[c_min, c_max]`` (``c_clipped == c``) the gradient is the genuine chain rule ``dc/draw``,
        bit-identical to a plain clip's in-range gradient — in particular ``log`` keeps its scale-invariant
        ``dc/draw = c`` interior dynamics (the MERU convention, and the reason to pick it).

        Where the clamp is active the gradient is a fake signal anyway (the forward value is pinned), and
        passing ``dc/draw`` through unchanged makes the re-entry step scale with that gain: for ``log`` at
        ``raw = 15`` it is ``exp(15) ~= 3.3e6``, enough for one plain-SGD step to jump the entire interval
        and pin ``c`` at the opposite bound. So out there the pass-through gradient is divided by
        ``max(dc/draw, 1)``, giving ``d(c_out)/d(raw) = 1``.

        The ``max(., 1)`` is a one-sided guard: it only ever damps a gain above 1, never amplifies one
        below it. That matters at the *lower* bound, where every parameterization's gain shrinks toward 0
        (``exp(raw) -> 0`` for ``log``, ``sigmoid(raw) -> 0`` for ``softplus``) and a two-sided
        ``1 / (dc/draw)`` would be the mirror image of the blow-up it fixes. So ``softplus``
        (``dc/draw = sigmoid(raw) <= 1``) and ``identity`` (``dc/draw = 1``) are exact no-ops at both
        bounds, and only ``log`` past the upper bound is damped.

        The division is realized by *swapping the carrier* rather than by multiplying by ``1/gain``: where
        the damping applies, the gradient is carried by ``raw - stop_gradient(raw)`` (derivative exactly 1)
        instead of ``c - stop_gradient(c)`` (derivative ``dc/draw``). Both are exactly ``0.0``, so the
        forward is untouched either way, but the swap avoids ever forming ``1/gain`` — which underflows to
        a subnormal (and is flushed to 0 by XLA) exactly for the huge gains this exists to tame: at the
        ``log`` exponent cap ``gain ~ 1.4e38``, and the multiplicative form re-froze the gradient at 0.
        """
        if self._parameterization == "softplus":
            grad_gain = jax.nn.sigmoid(raw)  # d(softplus(raw))/draw, bounded by 1 → never damped
        elif self._parameterization == "log":
            grad_gain = c  # d(exp(raw))/draw = exp(raw) = c, the pre-clamp value computed above
        else:  # "identity"
            grad_gain = jnp.ones_like(c)  # d(raw)/draw = 1 → never damped
        damped = (c_clipped != c) & (grad_gain > 1.0)
        return jnp.where(damped, raw - jax.lax.stop_gradient(raw), c - jax.lax.stop_gradient(c))
