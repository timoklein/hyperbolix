"""κ-Stereographic manifold — class-based API with dtype control.

The κ-stereographic model (Bachmann, Bécigneul & Ganea, "Constant Curvature Graph Convolutional
Networks", 2020) is a *single* constant-curvature manifold that interpolates smoothly across zero
curvature. It unifies the Poincaré ball (hyperbolic), Euclidean space, and the stereographic
projection of the sphere (spherical) via curvature-generalized ("κ-") trigonometric functions.

All operations work on single points with shape ``(dim,)``. Use ``jax.vmap`` for batching.

Curvature convention (**signed** ``c``, sectional curvature ``= -c`` — extends the rest of hyperbolix
across zero):

======  =========================  ===================================================
 ``c``   sectional curvature        geometry
======  =========================  ===================================================
``> 0``  ``< 0``                    hyperbolic — **identical to** ``Poincare(c)``; open ball ‖x‖ < 1/√c
``= 0``  ``0``                      Euclidean (see the factor-2 note below)
``< 0``  ``> 0``                    spherical (projected sphere); no boundary, all of R^d
======  =========================  ===================================================

Internally the paper's curvature ``κ = -c``. Every private function sets ``k = -c`` once and then
follows the geoopt/Bachmann formulas verbatim. **This is sign-flipped from the paper/geoopt ``κ``**
(their ``κ > 0`` = spherical ↔ our ``c < 0`` = spherical): the sign is chosen so ``c`` matches every
other hyperbolix manifold and so ``Stereographic(c)`` reproduces ``Poincare(c)`` exactly for ``c > 0``.

Euclidean-limit factor of 2 (a classic gyrovector-space gotcha)
---------------------------------------------------------------
The metric is ``g^κ_x = (λ^κ_x)² I`` with the conformal factor ``λ^κ_x = 2 / (1 + κ‖x‖²) = 2 / (1 - c‖x‖²)``,
so ``λ^κ_0 = 2`` and the metric at ``c = 0`` is ``4·I``, **not** ``I``. Consequently, as ``c → 0``:

- ``addition``, ``expmap``, ``logmap`` reduce to the *bare* Euclidean ``x+y`` / ``x+v`` / ``y-x`` (factor 1);
- ``dist``, ``dist_0``, ``tangent_norm`` carry a **factor of 2**: ``d_0(x, y) = 2‖x - y‖`` (paper Thm. 3, Eq. 8),
  ``tangent_inner`` a factor of 4.

This matches hyperbolix's own Poincaré ``dist_0 = 2·atanh(√c‖x‖)/√c → 2‖x‖``. It therefore does **not**
equal the separate :class:`~hyperbolix.manifolds.Euclidean` manifold's ``dist`` (which uses the bare
metric ``I``). Use :class:`~hyperbolix.manifolds.Euclidean` for un-scaled Euclidean geometry; use
``Stereographic`` at ``c = 0`` only as the *continuous limit* of the curved family.

Numerical precision
-------------------
Bachmann et al. strongly recommend double precision. Prefer ``Stereographic(dtype=jnp.float64)`` for
distances ≳ 7 (hyperbolic boundary) or spherical points near the ``tan`` pole; float32 is fine for
moderate points. See :mod:`hyperbolix.manifolds.poincare` for the near-boundary conformal-factor caveats,
which apply identically to the ``c > 0`` regime here.

JIT / batching example::

    >>> import jax, jax.numpy as jnp
    >>> from hyperbolix.manifolds import Stereographic
    >>> m = Stereographic(dtype=jnp.float64)
    >>> x, y = jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4])
    >>> d_hyp = m.dist(x, y, c=1.0)     # hyperbolic  (== Poincare(c=1).dist)
    >>> d_sph = m.dist(x, y, c=-1.0)    # spherical
    >>> dist_batched = jax.vmap(m.dist, in_axes=(0, 0, None))   # batch over points

Note: ``c`` is kept dynamic (a traced value works) so learnable curvature via
:class:`~hyperbolix.utils.LearnableCurvature` and ``jax.grad`` w.r.t. ``c`` are supported. For a *signed*
learnable curvature that spans all three regimes, use ``LearnableCurvature(parameterization="identity")``
(``c = raw``, with a symmetric default clamp); the ``softplus``/``log`` parameterizations stay positive
(hyperbolic-only).

References:
    Bachmann, Bécigneul & Ganea. "Constant Curvature Graph Convolutional Networks." ICML 2020.
    (arXiv:1911.05076). Eq. numbers below refer to this paper.
    geoopt ``geoopt/manifolds/stereographic/math.py`` — the PyTorch reference this ports.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import MIN_NORM, asinh, atanh, clamp_to, floor_at, safe_norm, tanh
from ..utils.precision import MATMUL_PRECISION
from ._base import ManifoldBase, default_atol
from ._gyrovector_core import (
    _addition,
    _boundary_divisor_floor,
    _conformal_factor,
    _conformal_factor_batch,
    _gyration,
    _mobius_denominator,
    _proj,
)
from .protocol import ScalarCurvature

# Version selection constant. The κ-stereographic distance has a single implementation; the
# constant and the ``version_idx`` arguments on ``dist`` / ``dist_0`` exist so manifold-generic
# callers (e.g. ``hyperbolix.utils.helpers.compute_pairwise_distances``, which always forwards a
# ``version_idx``) work here. Same accept-and-ignore precedent as ``ProperVelocity``.
VERSION_DEFAULT = 0

# Switch the κ-trig functions to their (analytic, signed-κ) Taylor series when |κ| falls below this.
# For |κ| above it the closed forms use the true √|κ| (never the floor), so value AND gradient w.r.t.
# both the argument and κ are correct; the Taylor branch only carries the neighborhood of κ = 0 where
# the floored √|κ| would otherwise zero out the κ-gradient. Kept well above MIN_NORM (the √|κ| floor)
# and well below any realistic curvature so the series never diverges (needs |κ|·x² ≳ 1).
#
# The cutover is DTYPE-DEPENDENT. In float64 the closed forms stay accurate down to |κ| ~ 1e-9, but in
# float32 the closed-form κ-gradient `d/dκ [atanh(√|κ|·x)/√|κ|]` loses all significant digits to
# catastrophic cancellation for |κ| up to ~1e-5 (and is even wrong-SIGNED below ~1e-8). So float32 must
# hand over to the (cancellation-free polynomial) Taylor branch ~4 decades earlier — otherwise a signed
# `LearnableCurvature` crossing zero receives a corrupted curvature gradient in the library's DEFAULT
# dtype. `K_ZERO_EPS` is retained as the float64 value (and public/back-compat name).
K_ZERO_EPS = 1e-9  # float64
_K_ZERO_EPS_F32 = 1e-5  # float32 (and any dtype with ≤ 32 significand+exponent bits)

# The |κ| cutover alone is NOT a sufficient gate: the series' convergence variable is u = κ·x², and some
# callers feed x ~ 1/√|κ| (the antipode's half great-circle t = π/√|κ|; far-field spherical points, whose
# chart norm scales exactly as 1/√|κ|), making |u| ≳ 1 for EVERY |κ| below the cutover — there the series
# diverges (garbage values, even negative distances) and its x**-power terms overflow float32 into
# 0·inf = NaN. The gate therefore also requires |u| < `_TAYLOR_MAX_U`, routing those arguments to the
# closed forms — which are well-conditioned exactly there: the closed-form κ-gradient's catastrophic
# cancellation is confined to |u| ≪ 1 (relative error ~ 3·eps/|u|, i.e. ≲ 4e-5 in float32 at this gate),
# so the two hazard regions do not overlap. At |u| = 0.01 the order-5 truncation error (~|u|⁶/13 ≈ 8e-14
# relative) is below float32 resolution, keeping the u-seam value-continuous.
_TAYLOR_MAX_U = 0.01
# Clamp on u inside the series helpers: keeps the UNSELECTED Taylor branch of the jnp.where finite for
# any argument (u⁵ overflows float32 for |u| ≳ 5e7, and jnp.where's VJP turns a 0-cotangent times inf
# into a NaN that leaks into the SELECTED branch's gradient). Far above the gate, so selected values are
# never affected.
_TAYLOR_U_CLIP = 1.0


def _k_zero_eps(dtype: jnp.dtype) -> float:
    """Dtype-aware Taylor-cutover magnitude for the κ-trig functions (float32 hands over ~4 decades
    earlier than float64 to dodge catastrophic cancellation in the closed-form κ-gradient)."""
    return _K_ZERO_EPS_F32 if jnp.finfo(dtype).bits <= 32 else K_ZERO_EPS


# Clamp the argument of the spherical `tan` branch to a large finite value: prevents ±inf feeding `tan`
# (which would NaN the *unselected* branch's gradient), while still letting `tan` wrap through its poles
# as spherical geometry requires. Mirrors geoopt's `scaled_x.clamp_max(1e38)`.
_TAN_ARG_CLAMP = 1e30


# ---------------------------------------------------------------------------
# Curvature-generalized ("κ-") trigonometry
#
# Each `_*_k(x, k)` takes the paper's signed curvature `k = κ` and returns the κ-generalized function:
# a hyperbolic form for k < 0, a spherical form for k > 0, and the shared analytic Taylor series as
# k → 0. The Taylor series is written in the *signed* k and equals BOTH branches in the limit (that is
# exactly why the model is differentiable across zero — Bachmann et al. Thm. 3), so no |k| substitution
# is needed. `√|k|` is floored (`_sqrt_abs_k`) so `1/√|k|` stays finite in the un-selected branch at
# k = 0, keeping `jnp.where`'s two-sided gradient NaN-free.
# ---------------------------------------------------------------------------


def _sqrt_abs_k(k: ScalarCurvature) -> Float[Array, ""]:
    """``√|k|`` floored to ``√MIN_NORM`` so ``1/√|k|`` never diverges (NaN-safe unselected branch)."""
    return jnp.sqrt(floor_at(jnp.abs(jnp.asarray(k)), MIN_NORM))


def _tan_k_zero_taylor(x: Float[Array, "..."], k: ScalarCurvature) -> Float[Array, "..."]:
    """Order-5 Maclaurin series of ``tan_k`` in the signed ``u = k·x²`` (Horner form); equals both
    branches as ``k → 0``.

    Evaluating in ``u`` rather than powers of ``x`` keeps the polynomial exact at ``k = 0`` for ANY
    ``x`` (the x**-power form's ``x**11`` overflows float32 at ``x ≳ 3e3``, turning ``k⁵·x**11`` into
    ``0·inf = NaN`` — a value NaN in flat space). The clamp keeps even *discarded* evaluations finite
    under ``jnp.where``'s two-sided VJP; it sits far above the ``_TAYLOR_MAX_U`` gate, so selected
    values are never affected."""
    u = clamp_to(jnp.asarray(k) * x**2, -_TAYLOR_U_CLIP, _TAYLOR_U_CLIP)
    poly = 1.0 + u * (1.0 / 3.0 + u * (2.0 / 15.0 + u * (17.0 / 315.0 + u * (62.0 / 2835.0 + u * (1382.0 / 155925.0)))))
    return x * poly


def _artan_k_zero_taylor(x: Float[Array, "..."], k: ScalarCurvature) -> Float[Array, "..."]:
    """Order-5 Maclaurin series of ``artan_k`` in the signed ``u = k·x²`` (Horner form); equals both
    branches as ``k → 0``. Same u-form/clamp rationale as :func:`_tan_k_zero_taylor`."""
    u = clamp_to(jnp.asarray(k) * x**2, -_TAYLOR_U_CLIP, _TAYLOR_U_CLIP)
    poly = 1.0 - u * (1.0 / 3.0 - u * (1.0 / 5.0 - u * (1.0 / 7.0 - u * (1.0 / 9.0 - u * (1.0 / 11.0)))))
    return x * poly


def _use_taylor(x: Float[Array, "..."], k: ScalarCurvature) -> Array:
    """Gate for the κ→0 Taylor branch: ``|k|`` below the dtype cutover AND inside the series'
    convergence region ``|u| = |k|·x² < _TAYLOR_MAX_U`` (see the constants above for why the |k|
    condition alone is insufficient). Elementwise in ``x``."""
    k_arr = jnp.asarray(k)
    small_k = jnp.abs(k_arr) < _k_zero_eps(jnp.asarray(x).dtype)
    return small_k & (jnp.abs(k_arr * x**2) < _TAYLOR_MAX_U)


def _tan_k(x: Float[Array, "..."], k: ScalarCurvature) -> Float[Array, "..."]:
    """κ-tangent: ``tanh(√|k|·x)/√|k|`` (k<0), ``tan(√k·x)/√k`` (k>0), Taylor (k→0). Paper ``tan_κ``."""
    sqrt_abs_k = _sqrt_abs_k(k)
    scaled = sqrt_abs_k * x
    neg = tanh(scaled) / sqrt_abs_k
    pos = jnp.tan(clamp_to(scaled, -_TAN_ARG_CLAMP, _TAN_ARG_CLAMP)) / sqrt_abs_k
    nonzero = jnp.where(jnp.asarray(k) > 0, pos, neg)
    return jnp.where(_use_taylor(x, k), _tan_k_zero_taylor(x, k), nonzero)


def _artan_k(x: Float[Array, "..."], k: ScalarCurvature) -> Float[Array, "..."]:
    """κ-arctangent: ``atanh(√|k|·x)/√|k|`` (k<0), ``arctan(√k·x)/√k`` (k>0), Taylor (k→0). Paper ``tan_κ⁻¹``."""
    sqrt_abs_k = _sqrt_abs_k(k)
    scaled = sqrt_abs_k * x
    neg = atanh(scaled) / sqrt_abs_k
    pos = jnp.arctan(scaled) / sqrt_abs_k
    nonzero = jnp.where(jnp.asarray(k) > 0, pos, neg)
    return jnp.where(_use_taylor(x, k), _artan_k_zero_taylor(x, k), nonzero)


def _artan_k_pair(r: Float[Array, "..."], s: Float[Array, "..."], k: ScalarCurvature) -> Float[Array, "..."]:
    """``tan_κ⁻¹(r)`` of a Möbius-difference norm ``r = ‖(-x) ⊕_κ y‖``, far-pair safe for ``c > 0``.

    :func:`_artan_k` with its hyperbolic branch (``k < 0``, i.e. ``c > 0``) spelled
    ``asinh(√c·s)/√c``, where the caller passes ``s = ‖x - y‖/√(B_x·B_y)``, ``B = 1 - c‖·‖²``,
    formed from the two input points rather than from ``r``. With ``G²`` the Möbius denominator
    ``1 - 2c⟨x,y⟩ + c²‖x‖²‖y‖² = B_x·B_y + c‖x - y‖²``, the norm is ``r = ‖x - y‖/G``, so
    ``1 - c·r² = B_x·B_y/G²`` and ``s = r/√(1 - c·r²)`` exactly, at every sign of ``c``; that makes
    ``atanh(√c·r) = asinh(√c·s)``. With ``d`` the pair distance, ``√c·r = tanh(√c·d/2)`` is stuck
    below 1, and ``atanh`` of it needs the ``1 - √c·r`` that float32 stops resolving well inside the
    chart: at ``√c·d ≈ 12.66`` (c = 1) the Möbius difference reaches the ball's representation
    ceiling although both points are far from it. ``√c·s = sinh(√c·d/2)`` has no ceiling.

    The other two branches and the gate are :func:`_artan_k`'s, on ``r``, unchanged: the Taylor
    series near ``c = 0`` (κ-gradient rationale at ``_K_ZERO_EPS_F32`` and ``_TAYLOR_MAX_U``) and
    ``arctan(√|c|·r)/√|c|`` for ``c < 0``. There ``s = r/√(1 + |c|·r²)`` would give the ``asin``
    form, which loses accuracy near antipodal points, so the spherical branch keeps ``r``.

    Each branch is finite at every ``c``, so the ``where`` pair never meets a NaN or ``inf`` in the
    branch it discards, in value or gradient: ``√|k|`` is floored (:func:`_sqrt_abs_k`), the series
    clamps its ``u``, and ``s`` is finite because both ``B`` factors are floored positive for ``c > 0``
    and are ``≥ 1`` for ``c ≤ 0`` (:func:`_one_minus_c_sqnorm`).
    """
    sqrt_abs_k = _sqrt_abs_k(k)
    neg = asinh(sqrt_abs_k * s) / sqrt_abs_k
    pos = jnp.arctan(sqrt_abs_k * r) / sqrt_abs_k
    nonzero = jnp.where(jnp.asarray(k) > 0, pos, neg)
    return jnp.where(_use_taylor(r, k), _artan_k_zero_taylor(r, k), nonzero)


def _one_minus_c_sqnorm(sqnorm: Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """``B = 1 - c·sqnorm``, floored like :func:`_conformal_factor`'s divisor (``x`` gives the dtype).

    For ``c > 0`` the floor sits below the cap's rounding band (:func:`_boundary_divisor_floor`), so
    an unprojected point cannot drive the divisor to 0 while a capped one keeps its gradient; for
    ``c ≤ 0``, ``B ≥ 1`` and the ``MIN_NORM`` floor never binds.
    """
    floor_b = jnp.where(jnp.asarray(c) > 0, _boundary_divisor_floor(x, c), MIN_NORM)
    return floor_at(1.0 - c * sqnorm, floor_b)


def _sin_k_half_dist(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """``s = ‖x - y‖/√(B_x·B_y)``, floored at ``MIN_NORM``, for :func:`_artan_k_pair` in the ops that
    also build ``(-x) ⊕_κ y`` (:func:`_logmap`, :func:`_geodesic`).

    ``‖y - x‖²``, ``‖x‖²`` and ``‖y‖²`` are reduced exactly as :func:`_addition` reduces them for
    ``(-x) ⊕ y`` (the sum ``(-x) + y``, dots at ``MATMUL_PRECISION``), so XLA computes each once for
    both. A ``safe_norm`` of ``y - x`` would add two passes over the inputs, and neither of its
    guards matters here: ``_addition`` squares ``y - x`` anyway, and the floor below absorbs the
    underflow.

    The floor is there because the caller divides by the Möbius-difference norm, floored at
    ``MIN_NORM`` itself: at ``y = x`` both then sit on ``MIN_NORM``, the hyperbolic
    ``tan_κ⁻¹(r)/r`` is 1 as it was, and the Jacobian in ``y`` keeps its value (the identity for
    ``log_x``, ``t·I`` for the geodesic) instead of 0. It is applied to the square so that the
    square root never meets an exact 0, whose infinite derivative would reach the gradient as
    ``0·inf``.
    """
    neg_x = -x
    diff_D = neg_x + y
    diff_sqnorm = jnp.dot(diff_D, diff_D, precision=MATMUL_PRECISION)
    x_sqnorm = jnp.dot(neg_x, neg_x, precision=MATMUL_PRECISION)
    y_sqnorm = jnp.dot(y, y, precision=MATMUL_PRECISION)
    b_prod = _one_minus_c_sqnorm(x_sqnorm, x, c) * _one_minus_c_sqnorm(y_sqnorm, x, c)
    return jnp.sqrt(floor_at(diff_sqnorm / b_prod, MIN_NORM**2))


# ---------------------------------------------------------------------------
# Manifold operations (single point, shape (dim,)). Each takes signed `c` and bridges to k = -c.
# For c > 0 the shared gyrovector core (_addition/_gyration/_proj/_conformal_factor) is bit-identical to
# `hyperbolix.manifolds.poincare` (same function objects); the κ-trig-based maps (expmap/logmap/dist/…)
# agree with Poincaré's direct tanh forms only to float tolerance (clamped tanh, different association
# order), not bit-for-bit.
# ---------------------------------------------------------------------------


def _scalar_mul(r: float | Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """κ-scalar multiplication ``r ⊗_κ x = tan_κ(r·tan_κ⁻¹(‖x‖))·x/‖x‖`` (paper Eq. 3, ``κ = -c``)."""
    k = -c
    # `safe_norm` + `floor_at`: the floor is deliberate (`x_norm` divides on the next line), the
    # max-scaling replaces `sqrt(sum(x**2) + MIN_NORM**2)`, which overflows float32 past
    # coordinate 1.8e19 and floors every radius below 1e-15.
    x_norm = floor_at(safe_norm(x)[..., None], MIN_NORM)
    res = _tan_k(r * _artan_k(x_norm, k), k) * (x / x_norm)
    return _proj(res, c)


def _dist(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Geodesic distance ``d_κ(x, y) = 2·tan_κ⁻¹(‖(-x) ⊕_κ y‖)`` (paper Eq. 4, ``κ = -c``).

    Reduces to ``2‖x - y‖`` as ``c → 0`` (the metric is ``4·I`` at the origin — see module docstring).

    The Möbius difference is never formed as a point. Its norm is ``r = ‖x - y‖/G`` with ``G²``
    the Möbius denominator, and for ``c > 0`` the distance is ``2·asinh(√c‖x - y‖/√(B_x·B_y))/√c``,
    ``B = 1 - c‖·‖²`` (:func:`_artan_k_pair`), the expression ``Poincare``'s slot 0 evaluates. The
    point ``(-x) ⊕ y`` sits at the full pair distance from the origin, so for a far pair float32
    could not store it even with both inputs well inside the chart: two points at scaled radius 7.2
    on opposite sides (true ``√c·d = 14.4``, c = 1) came back 12.656 with a gradient norm of 0.037
    instead of 671, and at 9 each (true 18) 12.628. Now they give 14.39999 and 18.00020, as
    ``Poincare`` does; what is left is the chart's own floor, the float32 rounding of ``B_x`` and
    ``B_y`` (``eps/B`` relative). ``c ≤ 0`` and the Taylor band near ``c = 0`` keep
    :func:`_artan_k`'s forms, on ``r``.

    It is also cheaper: ``(-x) ⊕ y`` and its norm took seven reductions over the inputs with a
    static ``c > 0`` and eight with a traced ``c``; this takes four (``‖y - x‖`` as a ``safe_norm``,
    which is two passes, ``‖x‖²`` and ``‖y‖²``) and six (plus the chord and the sum of
    :func:`_mobius_denominator`, which only the ``r`` branches read). ``dist(x, x)`` is an exact 0
    with an exactly-zero gradient: ``‖y - x‖`` is a ``safe_norm`` and nothing is floored.
    """
    # `safe_norm`: exact 0 and exactly-zero VJP at x == y, and no overflow in the unbounded c < 0
    # chart; not a divisor, so no floor.
    num = safe_norm(y - x)
    x_sqnorm = jnp.dot(x, x, precision=MATMUL_PRECISION)
    y_sqnorm = jnp.dot(y, y, precision=MATMUL_PRECISION)
    # G from the factored denominator: no cancellation at either sign of c (for c < 0 it is 0 at the
    # antipode, where B_x·B_y + c‖x - y‖² would cancel), and the literal slope dG²/dc = -2⟨x,y⟩ at c = 0.
    denom = jnp.sqrt(_mobius_denominator(x, y, c, sign=-1, x_sqnorm=x_sqnorm, y_sqnorm=y_sqnorm))
    sqrt_bb = jnp.sqrt(_one_minus_c_sqnorm(x_sqnorm, x, c) * _one_minus_c_sqnorm(y_sqnorm, x, c))
    return 2.0 * _artan_k_pair(num / denom, num / sqrt_bb, -c)


def _dist_0(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Geodesic distance to the origin ``d_κ(0, x) = 2·tan_κ⁻¹(‖x‖)``. Reduces to ``2‖x‖`` as ``c → 0``."""
    k = -c
    # `safe_norm`: exact 0 at the origin, and no 1e-15 floor on a genuinely small radius.
    x_norm = safe_norm(x)
    return 2.0 * _artan_k(x_norm, k)


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Exponential map ``exp^κ_x(v) = x ⊕_κ (tan_κ(λ^κ_x‖v‖/2)·v/‖v‖)`` (paper Eq. 6, ``κ = -c``)."""
    k = -c
    # `safe_norm` + `floor_at`: `v_norm` divides below, so the floor stays; see _scalar_mul.
    v_norm = floor_at(safe_norm(v)[..., None], MIN_NORM)
    lam = _conformal_factor(x, c)
    second_term = _tan_k(lam * v_norm / 2.0, k) * (v / v_norm)
    return _addition(x, second_term, c)


def _expmap_0(v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Exponential map at the origin ``exp^κ_0(v) = tan_κ(‖v‖)·v/‖v‖``. Reduces to ``v`` as ``c → 0``."""
    k = -c
    # `safe_norm` + `floor_at`: `v` is a tangent vector (unbounded), and `v_norm` divides on the
    # same line, so both halves are needed. Unlike Poincare this module takes a *signed* curvature
    # -- for c < 0 the chart is the sphere minus a point, whose radius diverges near the antipode,
    # so no site here can assume a bounded input.
    v_norm = floor_at(safe_norm(v)[..., None], MIN_NORM)
    return _proj(_tan_k(v_norm, k) * (v / v_norm), c)


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """First-order retraction ``retr_x(v) = proj(x + v)`` (used by Euclidean-parameter optimizers)."""
    return _proj(x + v, c)


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Logarithmic map ``log^κ_x(y) = (2/λ^κ_x)·tan_κ⁻¹(‖s‖)·s/‖s‖`` with ``s = (-x) ⊕_κ y`` (paper Eq. 7).

    The direction is ``s/‖s‖`` from the stored point, as before. For ``c > 0`` the magnitude
    ``tan_κ⁻¹(‖s‖)`` is the ``asinh`` form of :func:`_artan_k_pair`, built from
    ``‖x - y‖/√(B_x·B_y)`` (:func:`_sin_k_half_dist`) rather than from ``‖s‖``: for a far pair ``s``
    lies past float32's representation ceiling and is capped there, which cost ``‖log_x(y)‖``
    12 % (two points at scaled radius 7.2 on opposite sides, c = 1) and 30 % (at 9 each). The cap
    keeps the direction of ``s``, so the direction is unaffected. ``c ≤ 0`` and the Taylor band
    near ``c = 0`` still read ``‖s‖``, unchanged.
    """
    k = -c
    sub = _addition(-x, y, c)
    # `safe_norm` + `floor_at`: `sub_norm` divides below, so the floor stays; see _scalar_mul.
    sub_norm = floor_at(safe_norm(sub)[..., None], MIN_NORM)
    lam = _conformal_factor(x, c)
    return 2.0 * _artan_k_pair(sub_norm, _sin_k_half_dist(x, y, c), k) * (sub / (lam * sub_norm))


def _logmap_0(y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Logarithmic map at the origin ``log^κ_0(y) = tan_κ⁻¹(‖y‖)·y/‖y‖``. Reduces to ``y`` as ``c → 0``."""
    k = -c
    # `safe_norm` + `floor_at`: divisor on the same line, and the chart radius is unbounded for
    # c < 0 (see _expmap_0).
    y_norm = floor_at(safe_norm(y)[..., None], MIN_NORM)
    return _artan_k(y_norm, k) * (y / y_norm)


def _ptransp(
    v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, "dim"]:
    """Parallel transport of ``v`` from ``x`` to ``y``: ``gyr[y, -x]v · λ^κ_x/λ^κ_y``."""
    lambda_x = _conformal_factor(x, c)
    lambda_y = _conformal_factor(y, c)
    return _gyration(y, -x, v, c) * (lambda_x / lambda_y)


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Parallel transport of ``v`` from the origin to ``y``: ``(2/λ^κ_y)·v = (1 - c‖y‖²)·v``."""
    lambda_y = _conformal_factor(y, c)
    return (2.0 / lambda_y) * v


def _tangent_inner(
    u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, ""]:
    """Riemannian inner product ``⟨u, v⟩_x = (λ^κ_x)²·⟨u, v⟩``."""
    lambda_x = _conformal_factor(x, c)
    return lambda_x**2 * jnp.dot(u, v, precision=MATMUL_PRECISION)


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Riemannian norm ``‖v‖_x = λ^κ_x·‖v‖``."""
    lambda_x = _conformal_factor(x, c)
    # `safe_norm`: exact 0 with an exactly-zero VJP at v = 0; returned, not divided by.
    return lambda_x * safe_norm(v)


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Euclidean → Riemannian gradient ``∇_x = ∇^E_x / (λ^κ_x)²``."""
    lambda_x = _conformal_factor(x, c)
    return grad / (lambda_x**2)


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Project ``v`` onto the tangent space at ``x`` (identity: tangent space = ambient space)."""
    return v


def _is_in_manifold(x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
    """Membership test: ``c‖x‖² < 1 + atol`` for ``c > 0`` (ball); finiteness for ``c ≤ 0`` (all of R^d).

    Written in the dimensionless form ``c‖x‖² < 1`` (like ``Poincare._is_in_manifold``) so a
    single tolerance means the same thing at every curvature. ``atol`` defaults to
    :func:`~hyperbolix.manifolds._base.default_atol` for ``x.dtype``; it has no effect on the
    ``c ≤ 0`` branch, which is unconstrained.
    """
    x2 = jnp.dot(x, x, precision=MATMUL_PRECISION)
    c_arr = jnp.asarray(c)
    tol = default_atol(x.dtype) if atol is None else atol
    finite = jnp.all(jnp.isfinite(x))
    inside_ball = c_arr * x2 < 1.0 + tol
    return jnp.where(c_arr > 0, inside_ball, finite)


def _is_in_tangent_space(
    v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
) -> Array:
    """Every finite vector is a valid tangent vector (tangent space = ambient space = R^d).

    ``atol`` is accepted for signature uniformity; a finiteness test has no tolerance to
    slacken. (This used to return the constant ``True``, which accepted NaN and Inf.)
    """
    del x, c, atol
    return jnp.all(jnp.isfinite(v))


def _geodesic(t: Float[Array, ""], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Point at time ``t`` on the geodesic ``x → y``: ``gamma(t) = x ⊕_κ (t ⊗_κ ((-x) ⊕_κ y))`` (paper Eq. 5).

    ``t ⊗ v = tan_κ(t·tan_κ⁻¹(‖v‖))·v/‖v‖`` for ``v = (-x) ⊕ y`` is :func:`_scalar_mul` written out,
    with the half distance ``tan_κ⁻¹(‖v‖)`` taken from :func:`_artan_k_pair`, whose ``c > 0``
    branch reads ``‖x - y‖/√(B_x·B_y)`` (:func:`_sin_k_half_dist`) instead of ``‖v‖``. ``v`` sits
    at the full pair distance ``d`` from the origin and is capped at float32's ceiling for a far
    pair, which put the midpoint of two points at scaled radius 7.2 on opposite sides (c = 1) 0.87
    away from the float64 one, and 2.7 at radius 9 each. The direction stays ``v/‖v‖``, which the
    cap does not change; ``c ≤ 0`` and the Taylor band near ``c = 0`` are unchanged.

    What remains is the intermediate ``t ⊗ v``, a point at radius ``t·d``: past the ceiling
    (``√c·t·d ≈ 12.6`` in float32) it is capped too, and no spelling of this formula can store
    it. ``t = 1/2`` stays inside for every pair of representable points; ``t`` near 1 or past it
    on a far pair does not.
    """
    k = -c
    v = _addition(-x, y, c)
    # `safe_norm` + `floor_at`, as in `_scalar_mul`: `v_norm` divides below, so the floor stays.
    v_norm = floor_at(safe_norm(v)[..., None], MIN_NORM)
    half_dist = _artan_k_pair(v_norm, _sin_k_half_dist(x, y, c), k)
    tv = _proj(_tan_k(t * half_dist, k) * (v / v_norm), c)
    return _addition(x, tv, c)


def _geodesic_unit(
    t: Float[Array, ""], x: Float[Array, "dim"], u: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, "dim"]:
    """Unit-speed geodesic ``gamma(t) = x ⊕_κ (tan_κ(t/2)·u/‖u‖)`` from ``x`` in direction ``u``."""
    k = -c
    # `safe_norm` + `floor_at`: `u_norm` divides on the next line, so the floor stays.
    u_norm = floor_at(safe_norm(u)[..., None], MIN_NORM)
    second_term = _tan_k(t / 2.0, k) * (u / u_norm)
    return _addition(x, second_term, c)


def _antipode(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Antipode. Spherical (``c < 0``): the point diametrically opposite ``x`` (distance ``π/√|κ|`` away).
    In the stereographic chart this is the closed-form inversion ``x/(c·‖x‖²) = -x/(κ‖x‖²)`` through the
    circle of radius ``R = 1/√|κ|`` — an exact involution whose sphere lift is the negated lift of ``x``.
    (The equivalent ``geodesic_unit(π·R, x, x/‖x‖)`` route evaluates ``tan`` at its pole and, for ``|c|``
    below the κ-trig Taylor cutover, fed the series an argument ``∝ 1/√|c|`` outside its convergence
    region — NaN/garbage antipodes.) Non-spherical (``c ≥ 0``): ``-x``.

    Note: ``dist(x, antipode(x))`` is numerically unreliable — antipodal points are the coordinate
    singularity of the stereographic chart where the geodesic-distance formula is an unavoidable ``0/0``
    (shared with the geoopt reference). The chart antipode of the ORIGIN is the point at infinity and is
    not representable; the safe-norm floor returns ``0`` there instead. As ``c → 0⁻`` the antipode
    genuinely diverges (the sphere flattens), so very small ``|c|`` yields correspondingly huge outputs."""
    c_arr = jnp.asarray(c)
    is_spherical = c_arr < 0
    # x/(c·‖x‖²) is evaluated below as (x/r)/(c·r) with r = ‖x‖: algebraically identical, but
    # nothing is squared, so the inversion of a far-out point no longer overflows float32 (the old
    # `sum(x**2) + MIN_NORM**2` returned inf past coordinate 1.8e19, i.e. antipode = 0). The
    # MIN_NORM floor is deliberate and unchanged: r is a divisor, and the chart antipode of the
    # ORIGIN is the point at infinity, which the floor renders as 0 (see the docstring).
    r = floor_at(safe_norm(x)[..., None], MIN_NORM)
    # Substitute a benign curvature in the DISCARDED inversion for c ≥ 0 so 1/(c·r²) cannot divide by
    # zero at c = 0 and leak a NaN gradient through the jnp.where into the selected -x branch.
    safe_c = jnp.where(is_spherical, c_arr, -jnp.ones_like(c_arr))
    spherical = (x / r) / (safe_c * r)
    return jnp.where(is_spherical, spherical, -x)


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class Stereographic(ManifoldBase):
    """κ-Stereographic manifold (Bachmann et al. 2020) with automatic dtype casting.

    A single constant-curvature manifold spanning hyperbolic, Euclidean, and spherical geometry via a
    **signed** curvature ``c`` (sectional curvature ``= -c``): ``c > 0`` hyperbolic (identical to
    :class:`~hyperbolix.manifolds.Poincare`), ``c = 0`` Euclidean (with the gyrovector factor-2 metric —
    see the module docstring), ``c < 0`` spherical. See the module docstring for the full convention
    table, the Euclidean-limit factor-2 gotcha, and precision notes.

    Args:
        dtype: Target JAX dtype for computations (default: ``jnp.float32``; float64 recommended).
        c: Default (signed) curvature stored on the instance (default: ``1.0``, i.e. hyperbolic). The
            geometry methods take ``c`` explicitly per call, so this is metadata only.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import Stereographic
        >>> m = Stereographic(dtype=jnp.float64)
        >>> x, y = jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4])
        >>> m.dist(x, y, c=1.0)      # hyperbolic
        >>> m.dist(x, y, c=-1.0)     # spherical
    """

    VERSION_DEFAULT = VERSION_DEFAULT

    def __init__(self, dtype: jnp.dtype = jnp.float32, *, c: float = 1.0) -> None:
        super().__init__(dtype, c=c)

    def proj(self, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Project point onto the manifold (identity for ``c ≤ 0``)."""
        return _proj(self._cast(x), c)

    def conformal_factor(self, x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... 1"]:
        """Conformal factor ``λ^κ_x = 2/(1 - c‖x‖²)``, batch-compatible over arbitrary leading dims."""
        return _conformal_factor_batch(self._cast(x), c)

    def gyration(
        self, x: Float[Array, "dim"], y: Float[Array, "dim"], z: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, "dim"]:
        """Gyration ``gyr[x, y]z``."""
        return _gyration(self._cast(x), self._cast(y), self._cast(z), c)

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """κ-Möbius gyrovector addition ``x ⊕_κ y`` (paper Eq. 2)."""
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float | Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """κ-scalar multiplication ``r ⊗_κ x`` (paper Eq. 3)."""
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r_cast, x, c)

    def dist(
        self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature, version_idx: int = VERSION_DEFAULT
    ) -> Float[Array, ""]:
        """Geodesic distance ``d_κ(x, y)`` (paper Eq. 4). Note ``→ 2‖x - y‖`` as ``c → 0``.

        ``version_idx`` is accepted and ignored — there is a single κ-stereographic distance.
        It exists so manifold-generic callers that always forward one (e.g.
        ``hyperbolix.utils.helpers.compute_pairwise_distances``) work here too.
        """
        del version_idx
        return _dist(self._cast(x), self._cast(y), c)

    def dist_0(self, x: Float[Array, "dim"], c: ScalarCurvature, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
        """Geodesic distance to the origin ``d_κ(0, x)`` (``version_idx`` accepted and ignored).

        Note ``→ 2‖x‖`` as ``c → 0``.
        """
        del version_idx
        return _dist_0(self._cast(x), c)

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map ``exp^κ_x(v)`` (paper Eq. 6)."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map at the origin ``exp^κ_0(v)``."""
        return _expmap_0(self._cast(v), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """First-order retraction ``proj(x + v)``."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map ``log^κ_x(y)`` (paper Eq. 7)."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map at the origin ``log^κ_0(y)``."""
        return _logmap_0(self._cast(y), c)

    def ptransp(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, "dim"]:
        """Parallel transport ``v`` from ``x`` to ``y``."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Parallel transport ``v`` from the origin to ``y``."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    def tangent_inner(
        self, u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, ""]:
        """Riemannian inner product ``⟨u, v⟩_x``."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
        """Riemannian norm ``‖v‖_x``."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Euclidean → Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Project ``v`` onto the tangent space at ``x`` (identity)."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
        """Check whether ``x`` lies on the manifold (``atol`` default: :func:`default_atol`)."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
    ) -> Array:
        """Check that ``v`` has finite entries (the tangent space is all of R^d)."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c, atol)

    def geodesic(
        self, t: Float[Array, ""], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, "dim"]:
        """Point at time ``t`` on the geodesic through ``x`` and ``y`` (paper Eq. 5)."""
        x = self._cast(x)
        t_cast = jnp.asarray(t, dtype=x.dtype)
        return _geodesic(t_cast, x, self._cast(y), c)

    def geodesic_unit(
        self, t: Float[Array, ""], x: Float[Array, "dim"], u: Float[Array, "dim"], c: ScalarCurvature
    ) -> Float[Array, "dim"]:
        """Point at time ``t`` on the unit-speed geodesic from ``x`` in direction ``u``."""
        x = self._cast(x)
        t_cast = jnp.asarray(t, dtype=x.dtype)
        return _geodesic_unit(t_cast, x, self._cast(u), c)

    def antipode(self, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Antipode of ``x`` (diametrically-opposite point for ``c < 0``; ``-x`` otherwise)."""
        return _antipode(self._cast(x), c)
