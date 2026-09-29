"""Shared gyrovector core — single source of truth for the curvature-generic Möbius algebra.

These operations are **sign-agnostic**: the same formulas serve the Poincaré ball (``c > 0``, hyperbolic)
and the κ-stereographic model (signed ``c`` — hyperbolic / Euclidean / spherical). Both
:mod:`hyperbolix.manifolds.poincare` and :mod:`hyperbolix.manifolds.stereographic` import from here so a
stability fix lands in exactly one place instead of silently diverging between two hand-mirrored copies.

Every function is curvature-generic (one formula at any sign of ``c``); ``_conformal_factor`` /
``_conformal_factor_batch`` / ``_proj`` carry a ``jnp.where(c > 0, …)`` / ``abs(c)`` generalization whose
``c > 0`` branch is exactly the historical Poincaré expression, bit-for-bit. The only theoretical
departure there is a ``√|c|`` floor at ``√MIN_NORM``: for ``0 < c < 1e-15`` (a Poincaré ball of radius
``1/√c > 3e7`` — never used) the boundary floor is marginally more conservative than a bare ``√c``.
``_addition`` and ``_gyration`` are *not* bit-for-bit the historical expressions: ``_addition`` regroups
the numerator and clamps on a scalar (see its implementation notes), and both take their denominator
from :func:`_mobius_denominator`, which forms ``1 ± 2c⟨x,y⟩ + c²‖x‖²‖y‖²`` as a sum of non-negative
terms instead of a cancelling difference. Both changes move the last ulps and are strictly more
accurate near the ball boundary.

All operations act on a single point of shape ``(dim,)``; batch with :func:`jax.vmap`. The
``_conformal_factor_batch`` helper is the exception — it broadcasts over arbitrary leading dims for the NN
layers.

Dimension key:
    dim: manifold (ambient == spatial for these models) dimension
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import MIN_NORM, floor_at, safe_sqrt
from ..utils.precision import MATMUL_PRECISION
from .protocol import ScalarCurvature


def _get_max_norm_eps(x: Float[Array, "dim"]) -> float:
    """Maximum-norm epsilon for the array's dtype (``eps**0.75`` — empirically stable, scales with precision)."""
    return float(jnp.finfo(x.dtype).eps ** 0.75)


def _max_norm(x: Float[Array, "..."], c: ScalarCurvature) -> Float[Array, ""]:
    """Largest row norm :func:`_proj` admits: ``1/√|c| - eps**0.75`` for ``c > 0``, else unbounded.

    Factored out of :func:`_proj` so a caller that already knows the norm it is about to produce can
    apply the same bound *on the scalar* instead of re-reducing over a ``(…, dim)`` result — see
    ``poincare._expmap_0``. Same ``√|c|`` floor at ``√MIN_NORM`` and same ``1e15`` stand-in for the
    boundary-free ``c ≤ 0`` case as the historical inline expression, so :func:`_proj` is unchanged
    bit-for-bit. Only ``x``'s dtype is read, never its values.
    """
    max_norm_eps = _get_max_norm_eps(x)
    sqrt_abs_c = jnp.sqrt(floor_at(jnp.abs(jnp.asarray(c)), MIN_NORM))
    return jnp.where(jnp.asarray(c) > 0, (1.0 / sqrt_abs_c) - max_norm_eps, jnp.asarray(1e15, dtype=x.dtype))


def _boundary_floor(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Smallest value ``1 - c‖x‖²`` can take on a :func:`_proj`-projected point, for ``c > 0``.

    ``_proj`` caps the radius at ``max_norm = 1/√c - eps**0.75`` (:func:`_max_norm`), so
    ``1 - c‖x‖² ≥ 1 - c·max_norm² = 2√c·eps**0.75 - c·eps**1.5`` — that is this expression. It is
    the *analytic* minimum of the quantity, so anything below it on a projected point is rounding
    noise, and that is what makes it the right floor rather than a dtype-blind constant. At
    ``c = 1`` it is 1.3e-5 (float32) / 3.6e-12 (float64); ``MIN_NORM = 1e-15`` sits below both (it
    corresponds to geodesic radius 36/√c, which no float64 ball point reaches), so on this
    quantity the old floor never bit and a float32 cancellation could reach the divisor.

    The direction flips once the value is **squared** into the Möbius denominator: 1.6e-10
    (float32) is above ``MIN_NORM`` but 1.3e-23 (float64) is eight orders below it, so there the
    old floor was the one clamping legitimate pairs — from radius ≈ 18/√c, well inside the 27.7/√c
    where the float64 chart itself ends. See :func:`_mobius_denominator`.

    Only ``x``'s dtype is read, never its values. Factored out of :func:`_conformal_factor`, whose
    historical spelling this is verbatim. No site floors at it any more: the floored divisors
    (:func:`_conformal_factor`, :func:`_mobius_denominator` squared, the Poincaré ``B``'s and
    ``poincare._busemann``), the Klein chart's gap and the gaps the isometry maps read off a
    Poincaré or Klein point take half of it, :func:`_boundary_divisor_floor`.
    """
    max_norm_eps = _get_max_norm_eps(x)
    abs_c = jnp.abs(jnp.asarray(c))
    sqrt_abs_c = jnp.sqrt(floor_at(abs_c, MIN_NORM))
    return 2.0 * sqrt_abs_c * max_norm_eps - abs_c * max_norm_eps**2


def _boundary_divisor_floor(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Floor for a ``1 - c‖x‖²`` that divides: half of :func:`_boundary_floor`, for ``c > 0``.

    :func:`_boundary_floor` is where a capped point sits in exact arithmetic, not where its
    *computed* ``1 - c‖x‖²`` lands. The cap (``_proj``, ``_expmap_0``'s scalar cap, ``_addition``'s
    clamp) rounds the stored coordinates, and the computed value of a capped point falls within
    about 6 eps of the analytic floor on either side: -6.2 to +6 eps over 20000 capped points per
    producer, dtype and c in {0.1, 0.3, 1, 2.5}, below it for 11-76 % of them (mean 39 %;
    ``logs/2026-09-29_cancellation-free/floorfix/probe_band.py``). A floor that binds returns a
    constant, so every derivative through it is zero. On those points the dominant radial term
    ``2c·x/(1 - c‖x‖²)`` of a distance gradient vanished, and ``dist``'s gradient came back with
    relative error 1.0. CO-SNE, which throws most points onto the float64 cap, then diverged
    between two runs 1e-15 apart (``tests/test_cosne.py``).

    Half the analytic floor sits ``√c·eps**0.75`` below the cap value, 54 eps (float32) / 8192 eps
    (float64) at c = 1. No projected point reaches it: with this floor the float64 ``dist``
    gradient at capped points is bit-identical to the unfloored one. Only a point outside the
    ball, one that was never projected, still meets the floor, which keeps the divisor positive
    for it. The float32 margin exceeds the measured band only for c ≳ 0.013; below that a capped
    point can reach this floor too.
    """
    return 0.5 * _boundary_floor(x, c)


def _conformal_factor(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Conformal factor ``λ_x = 2 / (1 - c‖x‖²)``.

    For ``c > 0`` the denominator → 0 at the ball boundary and is floored at
    :func:`_boundary_divisor_floor`, below every capped point's rounding band; for ``c ≤ 0`` the
    denominator is ``≥ 1`` and the ``MIN_NORM`` floor never bites. The floor used to be the analytic
    cap value ``_boundary_floor`` itself, which bound for 11-76 % of the capped points and zeroed
    ``∂λ/∂x = 4c·x/(1 - c‖x‖²)²`` there, the dominant term of every gradient that reads ``λ`` at a
    capped point: ``tangent_norm``, ``egrad2rgrad``, ``ptransp`` (relative error 1.0) and
    ``expmap`` (0.47, float64).

    ``λ`` grows like ``e^a`` at scaled radius ``a``, and every quantity that inherits it — the Möbius
    operations, the Poincaré MLR score, ``_conformal_factor_batch`` — shares this ball chart's
    representation ceiling (``a ≈ 12.6`` float32 / ``27.7`` float64); the library adds no guard
    beyond that floor, by design (loud divergence over silent saturation, see CLAUDE.md).
    """
    x2 = jnp.dot(x, x, precision=MATMUL_PRECISION)
    denom = floor_at(1.0 - c * x2, jnp.where(jnp.asarray(c) > 0, _boundary_divisor_floor(x, c), MIN_NORM))
    return 2.0 / denom


def _mobius_denominator(
    x_D: Float[Array, "dim"],
    y_D: Float[Array, "dim"],
    c: ScalarCurvature,
    sign: int,
    x_sqnorm: Float[Array, ""] | None = None,
    y_sqnorm: Float[Array, ""] | None = None,
) -> Float[Array, ""]:
    """The Möbius denominator ``1 + sign·2c⟨x,y⟩ + c²‖x‖²‖y‖²``, formed without cancellation.

    ``sign`` is a **static** Python ``+1``/``-1``: ``+1`` is the denominator of ``⊕`` and of
    ``gyr[x,y]``, ``-1`` the one under ``dist``/``logmap``'s ``(-x) ⊕ y``.

    Why not the literal spelling: with ``r_x = ‖x‖``, ``r_y = ‖y‖`` and ``x̂``, ``ŷ`` the unit
    directions, ``2(1 ± cos) = ‖x̂ ± ŷ‖²`` and ``1 + t² = (1 - t)² + 2t`` give, for ``t = |c|r_x r_y``::

        1 + sign·2c⟨x,y⟩ + c²r_x²r_y² = (1 - t)² + t·‖x̂ + τ·ŷ‖²,   τ = sign·sgn(c)

    (for ``c > 0`` the ``-`` denominator takes the chord ``‖x̂ - ŷ‖`` and the ``+`` denominator the
    sum ``‖x̂ + ŷ‖``; for ``c < 0`` the two swap, which is what ``τ`` encodes). Every term on the
    right is non-negative and the subtraction ``1 - t`` happens *before* the square, so an O(1)
    result stops being the difference of two O(e^{2a}) terms. Measured on a radial pair 0.1 apart
    at ``c = 1`` (``logs/2026-09-08_hyperboloid_tangent_primitives/step3_equivalence.py``): at
    geodesic radius 10 in float32 ``dist`` is 2.9e-2 wrong as-is and 4.3e-6 here. At radius 20 in
    float64, ``dist`` is 7.519e-02 wrong as-is and 5.367e-10 here
    (``probe_poincare_mobius_ebebd09.out``, table D.i).

    The floor is :func:`_boundary_divisor_floor` **squared**, a quarter of the analytic minimum of
    the denominator over projected points: both radii at the cap, with the directions that zero the
    ``‖x̂ + τŷ‖`` term (``x = y`` for the ``-`` denominator, antipodal for the ``+`` one). At
    ``c = 1`` that minimum is 1.6e-10 (float32) / 1.3e-23 (float64). The minimum itself sat inside
    the capped points' rounding band. It bound not only at exact coincidence but for capped pairs up
    to ``√c·d ≈ 0.29-0.45`` (float32) / ``0.02-0.04`` (float64) from coincidence or antipodality.
    There it dropped ``∂D/∂x`` from ``logmap``, ``ptransp`` and ``⊕``: float64 gradients off by
    1.5e-3 to 9.4e-3 relative, against a chart floor of 6e-5 to 1.1e-4
    (``logs/2026-09-29_cancellation-free/floorfix/``). The quartered floor is slack for every capped
    pair. For ``c ≤ 0`` the denominator legitimately reaches 0 (the sphere's antipode), so
    ``MIN_NORM`` is kept there.

    ``x_sqnorm``/``y_sqnorm`` let a caller that already reduced ``⟨x,x⟩``/``⟨y,y⟩`` for its
    numerator hand them over; the chord/sum is then the only extra reduction this costs.
    """
    c_arr = jnp.asarray(c)
    if x_sqnorm is None:
        x_sqnorm = jnp.dot(x_D, x_D, precision=MATMUL_PRECISION)
    if y_sqnorm is None:
        y_sqnorm = jnp.dot(y_D, y_D, precision=MATMUL_PRECISION)
    # `floor_at(safe_sqrt(·), MIN_NORM)`, the `_proj` idiom: the floor sits *outside* the sqrt
    # because `r_x` divides below, and `safe_sqrt` supplies the finite derivative at an exact 0
    # that the floor alone would not. The floored value is deliberately used for the product `t`
    # as well: at `x = 0` the 1e-15 in `t` cancels the 1e15 the normalization puts into
    # `d‖x̂ + τŷ‖²/dx`, so the gradient of the denominator at the origin stays the exact `2·sign·c·y`
    # instead of collapsing to 0 (the value there is 1 ∓ O(1e-15·|c|·r_y), i.e. 1 in float32).
    r_x = floor_at(safe_sqrt(x_sqnorm), MIN_NORM)
    r_y = floor_at(safe_sqrt(y_sqnorm), MIN_NORM)
    x_hat_D = x_D / r_x
    y_hat_D = y_D / r_y
    # `(r_x * r_y)` must keep its parentheses: `c_arr * r_x * r_y` associates as
    # `(c_arr * r_x) * r_y`, which is *not* invariant under swapping x and y, and `1 - t` turns
    # that 1-ulp asymmetry into a relative one of eps/(1 - t) — enough to break `d(x, y) ==
    # d(y, x)` in float32 past geodesic radius ~5 (test_dist_properties). Everything else here is
    # already swap-symmetric: `r_x*r_y` and `x̂ + ŷ` are commutative, `x̂ - ŷ` is the exact
    # negation of `ŷ - x̂`, and the floor reads only the dtype.
    # Keep both factorizations directly in signed `c`. Besides avoiding cancellation at either sign,
    # this gives both branches the literal origin slope d(denom)/dc = 2·sign·⟨x,y⟩. Using
    # `abs(c)` plus a sign-selected direction has the same values away from zero but differentiates
    # with the wrong one-sided sign at exactly c = 0.
    signed_t = c_arr * (r_x * r_y)
    positive_gap = 1.0 - signed_t
    positive_w_D = x_hat_D + sign * y_hat_D
    positive_denom = positive_gap * positive_gap + signed_t * jnp.sum(positive_w_D**2)

    negative_gap = 1.0 + signed_t
    negative_w_D = x_hat_D - sign * y_hat_D
    negative_denom = negative_gap * negative_gap - signed_t * jnp.sum(negative_w_D**2)
    denom = jnp.where(c_arr > 0, positive_denom, negative_denom)
    return floor_at(denom, jnp.where(c_arr > 0, _boundary_divisor_floor(x_D, c) ** 2, MIN_NORM))


def _proj(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Project onto the manifold. A boundary exists only for ``c > 0`` (``‖x‖ < 1/√c``); for ``c ≤ 0``
    (Euclidean / spherical) the space is all of R^d and this is the identity."""
    # One reduction: `safe_sqrt(sum(x**2))`, wrapped in the `floor_at`. This runs on every sample
    # of every Poincare/stereographic forward, so it reads `x` once.
    # `safe_sqrt`, not a plain `sqrt`: `x` can be exactly the origin, where `sqrt'(0) = inf` meets
    # the *untaken* `where` branch's zero cotangent as 0*inf = NaN.
    # The `floor_at` is deliberate and must stay *around* the sqrt: `norm` divides in the untaken
    # branch too, and `floor_at` under the sqrt would not stop that branch's infinite derivative.
    # `sum(x**2)` overflows float32 past coordinate 1.8e19 (`x` here is unprojected, so it is the
    # one site where an out-of-range input is conceivable); that is far outside the ball of any
    # curvature this library is used at, so a network feeding it is already diverging. There
    # `norm = inf` and the clamp `x * (max_norm / inf)` returns the ZERO VECTOR -- the pre-1.2.0
    # behaviour, documented in the changelog rather than guarded.
    # `axis=-1, keepdims=True` is what makes the clamp broadcast against `x`: the reduction removes
    # the last axis, so it must be re-added before the result multiplies a `(..., dim)` operand --
    # exactly as :func:`_proj_batch` does. For the single point this function is contracted for it
    # is a shape-(1,) scalar and the result is bit-identical either way; it is the (B, dim) inputs
    # that several call sites and tests pass anyway that need it, and they get the per-row clamp
    # instead of the pre-sweep whole-array Frobenius one.
    norm = floor_at(safe_sqrt(jnp.sum(x**2, axis=-1, keepdims=True)), MIN_NORM)
    max_norm = _max_norm(x, c)
    cond = norm > max_norm
    return jnp.where(cond, x * (max_norm / norm), x)


def _proj_batch(x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... dim"]:
    """Project onto the manifold over arbitrary leading dims (batched :func:`_proj`).

    Same clamp as :func:`_proj`, applied along the last axis, so
    ``_proj_batch(X, c)[i] == _proj(X[i], c)`` elementwise. Mirrors
    ``Hyperboloid._proj_batch`` and the ``_conformal_factor_batch`` helper below.

    The bound comes from :func:`_max_norm`, which is the expression this used to inline verbatim —
    it reads only ``x``'s dtype, so it is a scalar either way and the clamp is bit-identical (probed
    over a (64, 16) batch, both dtypes, ``c`` in {0.3, 1, 2.5}, rows inside and past the boundary).
    """
    # One reduction, `floor_at(safe_sqrt(sum(x**2)), MIN_NORM)` over the last axis; see
    # :func:`_proj` for why the sqrt is `safe_sqrt`, why the floor sits outside it, and what
    # happens past float32 coordinate 1.8e19.
    norm = floor_at(safe_sqrt(jnp.sum(x**2, axis=-1, keepdims=True)), MIN_NORM)  # (..., 1)
    max_norm = _max_norm(x, c)
    cond = norm > max_norm
    return jnp.where(cond, x * (max_norm / norm), x)


def _addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Möbius gyrovector addition ``x ⊕ y`` (curvature-generic; non-commutative, non-associative).

    Result is kept on the manifold by the same boundary clamp :func:`_proj` applies, but computed
    from reductions over the *inputs* — see the implementation notes below.

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    x2 = jnp.dot(x, x, precision=MATMUL_PRECISION)
    y2 = jnp.dot(y, y, precision=MATMUL_PRECISION)
    # s = x + y is one extra (dim,)-sized elementwise op; ``s2`` and ``xs`` are the two extra
    # *input* reductions that make ‖num‖ computable without ever touching the (dim,) output. That
    # is the whole point: the old `_proj(num/denom, c)` re-reduced the op's own result, which under
    # jit(vmap) forces XLA to materialise the unprojected (B, dim) array and read it back.
    s_D = x + y
    s2 = jnp.dot(s_D, s_D, precision=MATMUL_PRECISION)
    xs = jnp.dot(x, s_D, precision=MATMUL_PRECISION)

    # A - B = (1 + 2c·xy + c·y2) - (1 - c·x2) = c(x2 + 2xy + y2) = c‖x+y‖² *exactly*, so the
    # historical numerator A·x + B·y is identically B·s + (c·s2)·x. This grouping is the one that
    # survives near-boundary antipodal inputs (‖x‖ → 1/√c, y ≈ -x): there A·x and B·y each have
    # magnitude ε·‖x‖ (ε := 1 - c‖x‖²) but their sum is O(ε²), so fl(A·x) + fl(B·y) loses a factor
    # eps/ε of accuracy, while B·s and (c·s2)·x are individually of the same order as their sum.
    coef_b = 1 - c * x2  # B
    coef_g = c * s2  # A - B
    num_D = coef_b * s_D + coef_g * x
    # `_mobius_denominator` replaces `1 + 2c·xy + c²·x2·y2`: near-antipodal boundary operands make
    # that an O(ε²) difference of O(1) terms, which float32 cannot resolve at all (the whole value
    # is rounding noise from geodesic radius ≈ 8). It reuses `x2`/`y2` and drops the `xy` reduction
    # in exchange for the chord/sum one, so the op still costs five reductions over the inputs.
    denom = _mobius_denominator(x, y, c, sign=1, x_sqnorm=x2, y_sqnorm=y2)

    # ‖num‖² expanded in the same two coefficients that built `num_D`, so the clamp decision is
    # consistent with the vector it is applied to (an independently derived norm, e.g. the equally
    # valid ‖x⊕y‖ = ‖x+y‖/√denom, is not: near the boundary the two disagree by eps/ε and a row
    # can then be scaled to sit *outside* the ball).
    t_ss = coef_b * coef_b * s2
    t_sx = 2 * coef_b * coef_g * xs
    t_xx = coef_g * coef_g * x2
    norm2 = t_ss + t_sx + t_xx
    # `terms` is both the rounding-error scale of that sum and, since |Σt| ≤ Σ|t|, an upper bound
    # on ‖num‖². Two guards, in this order:
    #   `lost`       norm2 has no significant bits left (or went negative). Fall back to `terms`:
    #                over-clamping is safe, letting a row escape the ball is not.
    #   `degenerate` terms == 0, i.e. num_D is identically zero (y = -x). The `where` must sit
    #                *before* the sqrt — sqrt'(0) is infinite and would NaN the whole row's
    #                gradient, which `_proj`'s `sqrt(‖·‖² + MIN_NORM²)` used to prevent.
    terms = jnp.abs(t_ss) + jnp.abs(t_sx) + jnp.abs(t_xx)
    mach_eps = jnp.finfo(num_D.dtype).eps
    lost = norm2 <= 16 * mach_eps * terms
    degenerate = terms <= 0
    norm2_safe = jnp.where(degenerate, jnp.ones_like(norm2), jnp.where(lost, terms, norm2))
    norm = jnp.sqrt(norm2_safe) / denom
    max_norm = _max_norm(num_D, c)
    cond = jnp.logical_not(degenerate) & (norm > max_norm)
    # Rows that are not clamped are multiplied by an exact 1.0, i.e. they are exactly `num_D/denom`.
    scale = jnp.where(cond, max_norm / norm, jnp.ones_like(norm))
    return scale * (num_D / denom)


def _gyration(
    x: Float[Array, "dim"], y: Float[Array, "dim"], z: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, "dim"]:
    """Gyration ``gyr[x, y]z`` — restores the (broken) commutativity/associativity of ``⊕``.

    Curvature-generic simplified closed form ``z + 2(A·x + B·y)/D``, with
    ``A = -c²⟨x,z⟩‖y‖² + c⟨y,z⟩ + 2c²⟨x,y⟩⟨y,z⟩``, ``B = -c²⟨y,z⟩‖x‖² - c⟨x,z⟩`` and ``D`` the ``⊕``
    denominator; underlies parallel transport. The numerator is evaluated grouped in ``s = x + y``
    (see the implementation notes), which is the same function at every real ``c``.

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    x_sqnorm = jnp.dot(x, x, precision=MATMUL_PRECISION)  # scalar
    y_sqnorm = jnp.dot(y, y, precision=MATMUL_PRECISION)  # scalar
    # A·x + B·y = (A - B)·x + B·s with s = x + y, the device `_addition` uses. Substituting
    # ⟨x,z⟩ = ⟨s,z⟩ - ⟨y,z⟩ and ‖s‖² = ‖x‖² + 2⟨x,y⟩ + ‖y‖² gives, exactly and for either sign of c,
    #     A - B = c·(⟨s,z⟩·(1 - c‖y‖²) + c·‖s‖²·⟨y,z⟩),   B = -c·(⟨s,z⟩ - (1 - c‖x‖²)·⟨y,z⟩).
    # `ptransp(v, x, y)` is gyr[y, -x]v, so s = y - x: for two nearby points near the boundary every
    # term above is O(1 - c‖x‖²) on its own, where A·x and B·y were O(1) terms cancelling to that
    # size — and D is O((1 - c‖x‖²)²), so the rounding of the O(1) terms came back amplified by
    # eps/(1 - c‖x‖²)². Measured on a 0.05-nat step at scaled radius 8 (c = 1, float32 vs float64,
    # max over 20 directions): ptransp 1.9e-1 relative wrong before, 6.6e-5 after (the float32
    # floor eps/(1 - c‖x‖²) is 9e-5 there). At s = 0 the correction is an exact 0, so
    # ptransp(v, x, x) returns v bit-for-bit (was 8.4e-2 off in float32, 9.9e-11 in float64).
    # ⟨s,z⟩ and ‖s‖² must be dots on s itself: re-forming them from ⟨x,z⟩ + ⟨y,z⟩ or from
    # ‖x‖² + 2⟨x,y⟩ + ‖y‖² would bring the cancellation straight back.
    s_D = x + y  # (dim,)
    s_sqnorm = jnp.dot(s_D, s_D, precision=MATMUL_PRECISION)  # scalar
    sz = jnp.dot(s_D, z, precision=MATMUL_PRECISION)  # scalar
    yz = jnp.dot(y, z, precision=MATMUL_PRECISION)  # scalar

    coeff_x = c * (sz * (1 - c * y_sqnorm) + c * s_sqnorm * yz)  # A - B, scalar
    coeff_s = -c * (sz - (1 - c * x_sqnorm) * yz)  # B, scalar
    num_D = 2 * (coeff_x * x + coeff_s * s_D)  # (dim,)
    # Same denominator, and the same cancellation, as `_addition`; see `_mobius_denominator`. The
    # five reductions above replace ‖x‖², ‖y‖², ⟨x,y⟩, ⟨x,z⟩, ⟨y,z⟩ one for one, so the op still
    # costs the five plus the denominator's chord/sum.
    denom = _mobius_denominator(x, y, c, sign=1, x_sqnorm=x_sqnorm, y_sqnorm=y_sqnorm)  # scalar

    return z + num_D / denom


def _conformal_factor_batch(x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... 1"]:
    """Conformal factor ``λ_x = 2 / (1 - c‖x‖²)`` over arbitrary leading dims (for the NN layers).

    Floored as :func:`_conformal_factor` is. At the analytic cap value the floor bound for capped
    inputs and zeroed the ``λ`` term of the Poincaré MLR score's input gradient (relative error 1.0).
    """
    dtype = x.dtype
    c_arr = jnp.asarray(c, dtype=dtype)
    x2 = jnp.sum(x**2, axis=-1, keepdims=True)  # (..., 1)
    floor = jnp.where(c_arr > 0, _boundary_divisor_floor(x, c_arr), MIN_NORM)
    denom = floor_at(jnp.asarray(1.0, dtype=dtype) - c_arr * x2, floor)
    return 2.0 / denom
