"""Proper Velocity (PV) manifold - class-based API with dtype control.

JAX port with vmap-native API. All functions operate on single points/vectors
in R^n. Use jax.vmap for batch operations.

Convention
----------
Paper (Chen et al. 2026) uses curvature ``K < 0`` with
``β_x = 1/√(1 - K·||x||²)``. We keep hyperbolix's ``c > 0`` convention
(sectional curvature ``-c``), substituting ``K = -c``. All formulas below
are expressed in the ``c > 0`` form:

- PV beta factor: ``β_x = 1/√(1 + c·||x||²)``
- Riemannian metric: ``g_x(u, v) = ⟨u, v⟩ - c·β_x²·⟨x, u⟩·⟨x, v⟩``
- Origin: the zero vector ``0 ∈ R^n`` (PV has no time coordinate; unlike
  the Hyperboloid model, points are not constrained).

The PV space is an **unconstrained** ``R^n`` model of hyperbolic geometry
rooted in special relativity's proper velocity. It is algebraically a
gyrovector space (isomorphic to the Poincaré ball via
``π(x) = (β_x / (1 + β_x)) · x``) and carries a Riemannian metric that
makes that isomorphism an isometry.

JIT Compilation & Batching
---------------------------
All functions work on single points and return scalars or vectors.
Use jax.vmap for batching and jax.jit for compilation:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.manifolds.proper_velocity import ProperVelocity
    >>>
    >>> manifold = ProperVelocity(dtype=jnp.float32)
    >>> x = jnp.array([0.1, 0.2])
    >>> y = jnp.array([0.3, 0.4])
    >>> d = manifold.dist(x, y, c=1.0)
    >>>
    >>> # Batch operations via vmap
    >>> x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25]])
    >>> dist_batched = jax.vmap(manifold.dist, in_axes=(0, 0, None))
    >>> distances = dist_batched(x_batch, jnp.roll(x_batch, 1, axis=0), 1.0)

References
----------
Chen et al. "Proper Velocity Neural Networks." ICLR 2026.
Ungar. "A Gyrovector Space Approach to Hyperbolic Geometry." 2022.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import (
    MIN_NORM,
    cosh,
    floor_at,
    radial_perp_decomposition,
    safe_hypot_norm,
    safe_norm,
    sinh,
)
from ..utils.precision import MATMUL_PRECISION
from ._base import ManifoldBase
from ._gyrovector_core import _gyration
from .hyperboloid import _addition as _hyperboloid_addition
from .hyperboloid import _dist_stable as _hyperboloid_dist
from .hyperboloid import _expmap as _hyperboloid_expmap
from .hyperboloid import _logmap as _hyperboloid_logmap
from .isometry_mappings import pv_to_hyperboloid
from .protocol import ScalarCurvature

# Version selection constant. PV currently has a single canonical implementation,
# kept for API consistency with Poincare / Hyperboloid.
VERSION_DEFAULT = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _create_origin(c: ScalarCurvature, dim: int, dtype=jnp.float32) -> Float[Array, "dim"]:
    """Create PV origin: the zero vector in R^n."""
    del c  # curvature is irrelevant: the PV origin is always 0.
    return jnp.zeros(dim, dtype=dtype)


def _beta(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """PV beta factor β_x = 1/√(1 + c·||x||²). Reciprocal of :func:`_beta_inv`, see there."""
    return 1.0 / _beta_inv(x, c)


def _beta_inv(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Reciprocal of the PV beta factor: 1/β_x = √(1 + c·||x||²).

    Evaluated as ``safe_hypot_norm(√c·x, 1)`` rather than ``sqrt(1 + c·dot(x, x))``.
    Proper-velocity coordinates are unconstrained -- that is the point of the model -- so
    ``dot(x, x)`` genuinely reaches the float32 overflow at ``‖x‖ = 1.8e19`` (geodesic radius
    ``arcsinh(1.8e19) ≈ 45``), where the old form returned ``inf`` and ``_beta`` collapsed to 0,
    silently zeroing every coefficient built from it. ``safe_hypot_norm`` never materialises the
    square either, and unlike the two-leg ``safe_hypot(1, √c·safe_norm(x))`` it does not round
    ``‖x‖`` to the dtype and then square it again. Measured against an 80-bit reference over
    ``c ∈ {0.3, 1, 2.5}``, dims 2/8/16, both dtypes, radii 1e-3…1e3: mean 0.25 ulp and max 2,
    against 0.32 and max 3 for the two-leg form
    (``logs/2026-09-04_ci_regressions/probe_beta_inv_spellings.py``).

    ``√c·x``, not ``√c·‖x‖``: scaling the components cannot overflow where the result does not,
    because ``1/β_x ≥ √c·‖x‖ ≥ √c·max|xᵢ|``. The alternative that scales the scalar leg instead,
    ``√c·safe_hypot_norm(x, 1/√c)``, is exact at ``c = 1`` but loses to this one at ``c = 2.5``
    (mean 0.45 ulp in float64), where ``1/√c`` is itself inexact.
    """
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=x.dtype))
    return safe_hypot_norm(sqrt_c * x, jnp.asarray(1.0, dtype=x.dtype))


def _safe_norm(x: Float[Array, "dim"]) -> Float[Array, ""]:
    """Euclidean norm floored at ``MIN_NORM``, for the sites that divide by it.

    ``floor_at(safe_norm(x), MIN_NORM)``, not the old ``sqrt(sum(x**2) + MIN_NORM**2)``. The floor
    is what the callers need -- ``_scalar_mul``, ``_expmap_0`` and ``_logmap_0`` all form an
    ``f(arg)/arg`` whose limit is 1 only while the two are the same floored quantity --
    but making it multiplicative removes the old spelling's two defects: ``sum(x**2)`` overflowed
    float32 above coordinate 1.8e19, and the additive ``1e-30`` perturbed every value between
    ``MIN_NORM`` and ``10·MIN_NORM``. ``_dist_0`` does not divide by the norm and uses the
    unfloored ``safe_norm`` directly; ``_dist`` and ``_logmap`` take no norm of their own at all
    (they defer to the hyperboloid polar frame, see there).

    Returns the norm with the last axis **reduced away**. The callers that divide a ``(..., dim)``
    vector by it re-add the axis with ``[..., None]``, which is a no-op in value for the single
    point this module is contracted for and keeps a ``(B, dim)`` input on its per-row scale.
    """
    return floor_at(safe_norm(x), MIN_NORM)


def _dpi_x(x: Float[Array, "dim"], v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Differential of π: PV → Poincaré (paper Eq. 7 with K = -c).

    dπ_x(v) = β_x/(1+β_x)·v - c·β_x³/(1+β_x)²·⟨x, v⟩·x
    """
    beta_x = _beta(x, c)
    xv = jnp.dot(x, v, precision=MATMUL_PRECISION)
    one_plus_beta = 1.0 + beta_x
    term1 = (beta_x / one_plus_beta) * v
    term2 = (c * beta_x**3 / one_plus_beta**2) * xv * x
    return term1 - term2


# ---------------------------------------------------------------------------
# PV gyro-operations
# ---------------------------------------------------------------------------


def _addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """PV gyroaddition ``x ⊕_U y`` (paper Eq. 2 with K = -c), as the spatial part of the Lorentz boost.

    **The identity.** Ungar builds the proper-velocity gyrogroup out of the Lorentz boosts, and the
    two spellings are the same expression term for term, not merely the same map. Paper Eq. 2 is
    ``x ⊕ y = x + y + {(1/β_y - 1) + c·β_x/(1+β_x)·⟨x, y⟩}·x``;
    :func:`~hyperbolix.manifolds.hyperboloid._addition` is the transvection ``Λ_X Y``, whose spatial
    part is ``√c·x·Y₀ + y + c·⟨x, y⟩/(1 + √c·X₀)·x``. On the exact lift
    ``X = pv_to_hyperboloid(x, c) = (√(1/c + ‖x‖²), x)`` (see :func:`_expmap`) one has
    ``√c·X₀ = √(1 + c‖x‖²) = 1/β_x`` and ``√c·Y₀ = 1/β_y``, so the two agree coefficient by
    coefficient; the ambient form only splits ``(1/β_y)·x`` into ``x + (1/β_y - 1)·x``. Measured in
    float64 over ``c ∈ {0.1, 0.5, 1, 3}``, dims 2/5/64, five pair geometries and 5 seeds:
    ≤1.4e-13 relative at ``a ≤ 3`` and ≤5.8e-12 at ``a ≤ 6`` — the two forms' own rounding on the
    anti-parallel case, not a difference of law
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_addition_is_the_boost.out``).

    **Why the boost.** It is the same result read off the hyperboloid, so the PV gyro-ops and
    :func:`_dist`/:func:`_logmap`/:func:`_expmap` now all speak through one lift instead of two
    parallel ambient spellings, and :func:`_expmap` needs the lift anyway; the boost's own
    conditioning is documented at :func:`~hyperbolix.manifolds.hyperboloid._addition`. Accuracy is
    a wash to slightly better: against an 80-bit ``np.longdouble`` reference on the same grid the
    worst case is the anti-parallel pair, 8.8e-14 for this form against 1.6e-13 for the ambient one
    at ``a ≤ 3`` and 2.4e-11 against 2.9e-11 at ``a ≤ 6``; every other geometry is at 1e-15 for both
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_expmap_equivalence.out``,
    sections A and B).

    ``x ⊕ 0`` and ``0 ⊕ y`` stay exact up to one ulp: with ``y = 0`` the boost's second and third
    terms vanish and the first is ``(√c·Y₀)·x`` with ``√c·Y₀ = √c·√(1/c)``, which is 1 only to
    within the curvature round trip — measured 0 ulp in float64 and 1 ulp in float32 over
    ``c ∈ {0.1, 0.5, 1, 3}``, against 0 ulp for the ambient form, which reached ``x`` by adding an
    exactly-zero coefficient. With ``x = 0`` every term carrying ``x_s`` is exactly zero and the
    result is ``y`` bit for bit in both dtypes.

    **The near-identity limit is not covered here.** ``(⊖x) ⊕ y`` with ``y ≈ x`` cancels the boost's
    three ``O(e^{2a})`` terms identically — the configuration
    :func:`~hyperbolix.manifolds.hyperboloid._gyro_difference` exists for. The ambient PV spelling
    cancels in exactly the same place, so this is unchanged by the rewrite, and PV has no
    ``gyro_difference`` of its own; callers that centre a batch (``ProperVelocityGyroBatchNorm``)
    inherit that limit.
    """
    return _hyperboloid_addition(pv_to_hyperboloid(x, c), pv_to_hyperboloid(y, c), c)[1:]


def _scalar_mul(t: Float[Array, ""] | float, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """PV scalar multiplication t ⊗_U x (paper Eq. 3 with K = -c).

    t ⊗ x = sinh(t · asinh(√c·||x||)) · x / (√c·||x||),     t ⊗ 0 = 0
    """
    sqrt_c = jnp.sqrt(c)
    x_norm = _safe_norm(x)[..., None]
    arg = sqrt_c * x_norm  # √c·||x||, never exactly zero
    # sinh is the overflow-protected variant; jnp.asinh is stable on all of R.
    scale = sinh(t * jnp.asinh(arg)) / arg
    return scale * x


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def _dist(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Geodesic distance d(x, y) on PV (paper Eq. 13), evaluated through the exact hyperboloid lift.

    Paper Eq. 13 is ``(1/√c)·asinh(√c·‖z‖)`` with the gyro-difference ``z = (⊖x) ⊕ y``,
    algebraically equal to the atanh form ``(2/√c)·atanh(√c·‖π(z)‖)`` by the sinh/tanh half-angle
    identity. **Neither is evaluated here.**

    **Why the gyro-difference form cancels.** ``z = -x + y + {(1/β_y - 1) + c·β_x/(1+β_x)·⟨-x, y⟩}·(-x)``
    is a sum of three terms each of size ``e^(a+b)/√c`` -- with ``a = √c·d(0, x)``, ``b = √c·d(0, y)``
    -- that has to cancel down to ``sinh(θ)/√c``, ``θ = √c·d(x, y)``. The surviving significand is
    ``e^(a + b - θ)`` times smaller than the operands, i.e. twice the Gromov product, so all
    precision is gone once ``a + b - θ`` passes ``ln(1/eps)``: 15.9 in float32, 36.0 in float64.
    This is the same failure :func:`~hyperbolix.manifolds.hyperboloid._polar_frame` documents for
    ``⟨x, y⟩_L``, and it is the same failure for a concrete reason: PV coordinates *are* the
    hyperboloid spatial part. Measured at ``c = 0.5``, dim 16, on two points a true 0.1 apart along
    a coordinate axis (which survives the float32 cast exactly, so only the arithmetic is measured),
    float32 returned **0.0599 at a = 8**, **4.383 at a = 10** and **10.56 at a = 12**.

    **Why the lift is exact.** ``pv_to_hyperboloid`` is an isometry that appends the on-sheet time
    slot ``√(1/c + ‖x‖²)`` -- a sum of positives, no cancellation of its own -- and leaves the
    coordinates alone, so ``d_PV(x, y) = d_H(X, Y)`` holds *exactly*, not to some tolerance. The
    hyperboloid distance runs in the polar (haversine) frame, where ``sinh²(θ/2)`` is a sum of two
    non-negative terms. Same construction, same points: **≤1.0e-6 relative at all three radii**,
    against a float32 *storage* floor for the pair itself of 9.2e-7 -- there is nothing left to win
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_accuracy.out``).

    In float64 the new and old spellings agree to ≤8.1e-14 absolute at ``a ≤ 3`` over dims 2/5/64
    and ``c ∈ {0.1, 0.5, 1, 3}``, which pins the change to a re-spelling rather than a redefinition.
    At ``a ≤ 6`` they part by up to 3.6e-11, and an 80-bit reference attributes all of it to the old
    arm: against it the new form is within 7.1e-15 everywhere on that grid, the old within 3.6e-11
    (``step2c_pv_equivalence.out``, section B).

    ``d(x, y) == 0`` stays *exactly* zero at ``x == y`` with an exactly-zero gradient, for the same
    structural reason it did before: identical inputs give identical spatial radii and an identical
    unit direction, so the frame's radial gap and chord are both exactly 0 and
    ``2·arcsinh(hypot(0, 0))/√c`` is 0 through a ``safe_hypot`` whose VJP at the origin is zero.
    No coincidence ``where`` is needed on either side.
    """
    return _hyperboloid_dist(pv_to_hyperboloid(x, c), pv_to_hyperboloid(y, c), c)


def _dist_0(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Geodesic distance from the PV origin (paper Thm 4.3 simplified).

    d(0, x) = (1/√c) · asinh(√c · ||x||)

    Uses ``safe_norm`` for the same reason as :func:`_dist`: ``jnp.linalg.norm``'s VJP at
    ``x = 0`` is ``0/0 = NaN``, and the origin is exactly where this function is most often
    differentiated (it is the PV analogue of the wrapped-normal NaN-at-the-mean bug). Not the
    floored ``_safe_norm``: the norm is not a divisor here, so ``d(0, 0)`` is exactly 0 and a
    small radius is reported exactly rather than floored at ``1e-15``.
    """
    sqrt_c = jnp.sqrt(c)
    x_norm = safe_norm(x)
    return jnp.asinh(sqrt_c * x_norm) / sqrt_c


# ---------------------------------------------------------------------------
# Exp / log maps
# ---------------------------------------------------------------------------


def _expmap_0(v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Exponential map from the origin (paper Thm 4.3 simplified).

    exp_0(v) = sinh(√c·||v||) · v / (√c·||v||)

    Written via sinh/arg with a safe-norm substitution so the v=0 case gives 0.
    """
    sqrt_c = jnp.sqrt(c)
    v_norm = _safe_norm(v)[..., None]
    arg = sqrt_c * v_norm
    # sinh(arg)/arg has limit 1 as arg → 0, which the safe_norm preserves.
    scale = sinh(arg) / arg
    return scale * v


def _logmap_0(y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Logarithmic map to the origin (paper Thm 4.3 simplified).

    log_0(y) = asinh(√c·||y||) · y / (√c·||y||)
    """
    sqrt_c = jnp.sqrt(c)
    y_norm = _safe_norm(y)[..., None]
    arg = sqrt_c * y_norm
    # asinh(arg)/arg has limit 1 as arg → 0.
    scale = jnp.asinh(arg) / arg
    return scale * y


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Exponential map at x (paper Eq. 10 with K = -c), as the spatial part of the hyperboloid map.

    Paper Eq. 10, simplified with ``(1+β_x)/β_x·‖dπ_x(v)‖ = √g_x(v, v)``, reads
    ``exp_x(v) = x ⊕ ((1+β_x)/β_x · sinhc(√c·‖v‖_x) · dπ_x(v))`` with ``sinhc(t) = sinh(t)/t``.
    **That form is not evaluated here.** Its geodesic *length* was already cancellation-free (it
    comes from :func:`_tangent_norm`), but the *direction* is not: for a radial ``v`` the two terms
    of ``dπ_x(v) = β_x/(1+β_x)·v - c·β_x³/(1+β_x)²·⟨x, v⟩·x`` have radial coefficients
    ``β_x/(1+β_x)`` and ``β_x(1-β_x)/(1+β_x)``, which cancel down to ``β_x²/(1+β_x)`` — a factor
    ``β_x ≈ e^{-a}`` of the operands, so the relative error of the direction grows like ``eps·e^a``.
    The landing ``x ⊕ ·`` then adds its own ``eps·e^{2a}`` (the boost's three-term conditioning,
    see :func:`_addition`). Measured at ``c = 0.5``, dim 16, on a radial step of true length 0.1:
    the ambient spelling's ``dπ`` is 5.9e-5 relative off at ``a = 8`` and 5.9e-4 at ``a = 10`` in
    float32 (``logs/2026-09-08_hyperboloid_tangent_primitives/step6b_pv_expmap_isolation.out``), and
    the step it landed measured 0.100011573 at ``a = 8``, 0.099952025 on the CPU and 0.099870228 on
    an A100 at ``a = 10``, and 0.098831346 at ``a = 12`` — the ``a = 10`` figure is 1.3x the
    ``rel=1e-3`` bound of
    ``test_pv_expmap_step_length_matches_the_tangent_norm_at_large_radius[10.0]``, which is how the
    cancellation surfaced. The output stayed finite and plausible — a step length, positive, of the
    right order — so the assert is on accuracy against float64, not on finiteness.

    **The lift.** ``pv_to_hyperboloid`` sends ``x ↦ X = (√(1/c + ‖x‖²), x)``, exactly and without
    cancellation of its own, and a PV tangent vector ``v`` at ``x`` lifts to the unique hyperboloid
    tangent ``V = (⟨x, v⟩/X₀, v)`` — the time slot is forced by ``⟨V, X⟩_L = -X₀V₀ + ⟨x, v⟩ = 0``.
    The lift is an isometry and its differential is exact on spatial parts, so

        exp^PV_x(v) = exp^H_X(V)[1:]

    holds *exactly*, and :func:`~hyperbolix.manifolds.hyperboloid._expmap` lands
    ``cosh(√c‖V‖_X)·X + sinhc(√c‖V‖_X)·V`` — a two-term combination of the base point and the
    tangent vector, with no ``O(e^{a})`` cancellation in the direction — before rebuilding the time
    slot with :func:`~hyperbolix.manifolds.hyperboloid._proj`. Its ``‖V‖_X`` is
    :func:`~hyperbolix.manifolds.hyperboloid._tangent_norm`, which reads the same radial/perp split
    as this module's :func:`_tangent_norm` with ``√c·X₀ = 1/β_x``, so the step *length* is the one
    this map already had.

    ``V₀`` does not reach the output: ``_tangent_norm`` eliminates ``v₀`` through tangency and the
    closing ``_proj`` rebuilds the time slot from the spatial part, so a zero time slot returns a
    bit-identical answer (measured 0 ulp over ``c ∈ {0.1, 0.5, 1, 3}``, dims 2/5/64, both dtypes,
    radii ≤ 6 and step lengths 1e-3…50). XLA does not eliminate it for us — the optimized HLO keeps
    the extra ``dot`` — and it is paid for: ``vmap(expmap)`` over 8192 float32 points at dim 64
    costs 4.27 ms with the exact slot against 3.47 ms with a zero one, both under the 5.14 ms of the
    ambient spelling this replaces (``step2c_pv_expmap_time_slot.out``). It is spelled out anyway,
    because ``V₀ = ⟨x, v⟩/X₀`` is what makes ``V`` *the* lift of ``v`` rather than an arbitrary
    extension of it, and because the ``v₀``-blind contract is the hyperboloid map's to keep, not
    this one's to assume.

    **Measured.** Float32 landing error ``|d(x, exp_x(v)) - ‖v‖_x| / ‖v‖_x`` against a float64
    yardstick, ``c = 0.5``, dim 16, radial tangent of Riemannian length 0.1, worst over 4 random
    base directions per radius, CPU / A100 (``step2c_pv_expmap_landing_{cpu,gpu}.out``)::

        a       ambient            this form          float32 storage floor of the pair
        8       3.4e-4 / 2.3e-4    3.2e-7 / 3.2e-7    2.5e-7
        10      1.2e-3 / 4.9e-4    2.7e-5 / 3.1e-5    2.8e-5
        12      1.5e-2 / 6.1e-3    1.7e-3 / 1.4e-3    6.0e-4

    The floor is the geodesic error of the float32-*stored* exact landing point, which no arithmetic
    can remove; this form sits within 2.9x of it at every radius, where the ambient one was 25x to
    1400x above it. On a *generic* tangent direction the two are indistinguishable (both at the
    floor): the ``dπ_x`` cancellation is radial-only, and a generic step's storage floor is ``e^a``
    times larger relative to the step, so it swamps the arithmetic.

    In float64 the new and old spellings agree to ≤3.2e-14 relative at ``a ≤ 3`` over
    ``c ∈ {0.1, 0.5, 1, 3}``, dims 2/5/64, five tangent geometries and step lengths 1e-3…1, which
    pins this to a re-spelling; at ``a ≤ 6`` an 80-bit reference puts this form at 6.0e-15 and the
    ambient one at 3.9e-14 (``step2c_pv_expmap_equivalence.out``, sections A and B). Its float32
    gradients track float64 to ≤1.3e-6 relative with no non-finite entry, against ≤6.4e-6 for the
    ambient form (``step2c_pv_expmap_gradients.out``).

    The float64 round trip ``exp_x(log_x(y)) = y`` is *not* what this improves: 1.6e-13 at ``a ≤ 3``
    and 7.5e-10 at ``a ≤ 6``, against 2.5e-13 and 6.5e-10 for the ambient form. That composition is
    limited by neither map's spelling but by the conditioning of ``exp_x`` for a *long* step, and
    the same 80-bit check leaves 2.2e-13 when the exponential map is taken out of it — see
    ``tests/test_pv_manifold.py::test_pv_logmap_carries_the_distance_and_inverts_expmap_in_float64``.

    ``exp_x(0) = x`` stays bit-exact (measured 0 ulp, both dtypes): ``‖V‖_X`` is floored at
    ``MIN_NORM``, ``cosh(√c·1e-15)`` is exactly 1 in both dtypes and the ``sinhc`` term is
    multiplied by an exactly-zero ``V``, so the spatial part is ``1.0·x`` and ``_proj`` leaves it
    alone.
    """
    x_H = pv_to_hyperboloid(x, c)
    # Tangency ⟨V, X⟩_L = -X₀·V₀ + ⟨x, v⟩ = 0 fixes the time slot; X₀ ≥ 1/√c > 0, so no floor.
    v0 = jnp.dot(x, v, precision=MATMUL_PRECISION) / x_H[0]
    v_H = jnp.concatenate([v0[None], v])
    return _hyperboloid_expmap(v_H, x_H, c)[1:]


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Logarithmic map at x (paper Eq. 11 with K = -c), as the spatial part of the hyperboloid log.

    Paper Eq. 11, simplified with the half-angle identity
    ``2·atanh(√c·‖π(z)‖) = asinh(√c·‖z‖)``, reads
    ``log_x(y) = asinhc(√c·‖z‖)·(z + (β_x·c/(1+β_x))·⟨x, z⟩·x)`` with ``z = (⊖x) ⊕ y`` and
    ``asinhc(t) = asinh(t)/t``. **That form is not evaluated here**: it is built on the same
    ``z`` as :func:`_dist`, so it inherits the same three-way ``O(e^(a+b))`` cancellation described
    there in full, once in ``‖z‖`` and once in the direction vector.

    **The lift.** ``pv_to_hyperboloid`` lifts ``x ↦ X = (√(1/c + ‖x‖²), x)`` and its differential is
    the identity on spatial parts (``dX = (⟨x, dx⟩/X₀, dx)``), so the hyperboloid tangent vector at
    ``X`` carries the PV tangent vector as its spatial part and

        log^PV_x(y) = log^H_X(Y)[1:]

    holds *exactly*. :func:`~hyperbolix.manifolds.hyperboloid._logmap` builds that vector in an
    orthonormal geodesic frame at ``X`` out of individually bounded ratios of the polar frame, never
    from an ambient difference. Its ``r_x == 0`` branch is :func:`~hyperbolix.manifolds.hyperboloid._logmap_0`,
    whose spatial part is exactly this module's :func:`_logmap_0`, so the PV origin keeps the value
    and the gradient it had.

    Measured at ``c = 0.5``, dim 16, on two points a true 0.1 apart along a coordinate axis, float32
    ``‖log_x(y)‖_x``: the old spelling returned **0.0599 at a = 8**, **4.383 at a = 10** and
    **10.56 at a = 12**; this one is within **≤1.0e-6 relative** at all three, against a float32
    storage floor of 9.2e-7 (``step2c_pv_accuracy.out``). In float64 the two agree to ≤8.1e-14 in
    PV tangent norm at ``a ≤ 3``, dims 2/5/64, ``c ∈ {0.1, 0.5, 1, 3}``, and against an 80-bit
    reference at ``a ≤ 6`` the new form is within 4.4e-14 *relative* and the old within 2.7e-11
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2c_pv_equivalence.out``).

    ``‖log_x(y)‖_x = d(x, y)`` now holds by construction rather than by two independent asinh
    evaluations agreeing: both read the same polar frame. Measured ≤1.3e-15 relative in float64.

    **The collinear gradient.** When ``x`` and ``y`` lie on *exactly* the same ray through the
    origin, the angular unit vector ``n̂ = normalize(ŷ_s - ⟨x̂_s, ŷ_s⟩·x̂_s)`` that
    :func:`~hyperbolix.manifolds.hyperboloid._logmap_direction` used to build normalized a vector
    that is zero up to rounding, so its derivative was arbitrary; multiplied by a ``sin φ`` that is
    itself only ``O(rounding)`` rather than exactly 0, it left an ``O(1)`` error in the *gradient*
    while the forward value stayed correct. Measured on ``∇ ‖log_x(y)‖²`` against an 80-bit finite
    difference, that spelling was 0.12 relative in float64 and 11 in float32 on a same-ray pair,
    against 2.5e-11 / 4.2e-6 for the pre-lift PV form (``step2c_pv_equivalence.out``, section F).
    The helper now returns the unnormalized ``perp_y`` instead, which is smooth on the whole
    collinear set. Re-measured with both arms against the *same* 80-bit reference on the same grid
    (``c ∈ {0.5, 1}``, dims 5/64, ``a ≤ 3``, same-ray and anti-ray): the lift is **3.3e-10** in
    float64 and **2.2e-5** in float32, the pre-lift ambient form 3.3e-10 and 4.1e-5 — identical in
    float64, where both sit on the reference's own floor, and 1.9x better in float32
    (``logs/2026-09-08_hyperboloid_tangent_primitives/step2e_gradients.out``, section J).
    """
    return _hyperboloid_logmap(pv_to_hyperboloid(y, c), pv_to_hyperboloid(x, c), c)[1:]


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Euclidean retraction x + v. PV is unconstrained, so this is exact.

    Kept separate from expmap to match the other manifolds' APIs; use ``expmap``
    for the exact Riemannian map.
    """
    del c  # PV is unconstrained -- no projection needed.
    return x + v


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Parallel transport from the origin to y (paper Thm 4.3, K = -c).

    PT_{0→y}(v) = v + c·β_y/(1+β_y) · ⟨y, v⟩ · y
    """
    beta_y = _beta(y, c)
    yv = jnp.dot(y, v, precision=MATMUL_PRECISION)
    coef = c * beta_y / (1.0 + beta_y)
    return v + coef * yv * y


def _ptransp(
    v: Float[Array, "dim"],
    x: Float[Array, "dim"],
    y: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, "dim"]:
    """Parallel transport from T_x PV to T_y PV (paper Eq. 12 with K = -c).

    PT_{x→y}(v) = (1+β_x)/β_x · ṽ + c·(1+β_x)·β_y/((1+β_y)·β_x) · ⟨y, ṽ⟩ · y

    where ṽ = gyr_M[ȳ, -x̄](dπ_x(v)) is the Möbius gyration in the Poincaré
    ball acting on dπ_x(v), with x̄ = β_x/(1+β_x)·x and ȳ = β_y/(1+β_y)·y.
    """
    beta_x = _beta(x, c)
    beta_y = _beta(y, c)

    # Poincaré-ball images of x and y under π.
    x_bar = (beta_x / (1.0 + beta_x)) * x
    y_bar = (beta_y / (1.0 + beta_y)) * y

    # Differential of π at x applied to v.
    dpi_v = _dpi_x(x, v, c)

    # Möbius gyration in the Poincaré ball (the shared curvature-generic gyrovector core).
    v_tilde = _gyration(y_bar, -x_bar, dpi_v, c)

    # Eq. 12 (K = -c flips the sign of the second term).
    yv = jnp.dot(y, v_tilde, precision=MATMUL_PRECISION)
    coef1 = (1.0 + beta_x) / beta_x
    coef2 = c * (1.0 + beta_x) * beta_y / ((1.0 + beta_y) * beta_x)
    return coef1 * v_tilde + coef2 * yv * y


# ---------------------------------------------------------------------------
# Tangent space
# ---------------------------------------------------------------------------


def _tangent_inner(
    u: Float[Array, "dim"],
    v: Float[Array, "dim"],
    x: Float[Array, "dim"],
    c: ScalarCurvature,
) -> Float[Array, ""]:
    """Riemannian inner product ⟨u, v⟩_x (paper Eq. 1 with K = -c), free of cancellation.

    The literal ``g_x(u, v) = ⟨u, v⟩ - c·β_x²·⟨x, u⟩·⟨x, v⟩`` subtracts two ``O(‖u‖·‖v‖)`` terms
    that are individually as large as ``‖x‖²·β_x²`` times the answer's radial part, so at scaled
    geodesic radius ``a`` it loses ``cosh²(a)·eps`` of relative accuracy — exactly the failure
    ``Hyperboloid._tangent_inner`` documents, and for the same reason: PV coordinates *are* the
    hyperboloid spatial part, with ``β_x² = 1/(1 + c‖x‖²) = 1/(c·x₀²)``.

    Splitting ``u`` and ``v`` along and across ``x`` makes the metric diagonal, hence a sum of
    products of same-signed factors. With ``x̂ = x/‖x‖``, ``radial(u) = ⟨u, x̂⟩`` and
    ``perp(u) = u - radial(u)·x̂``, ``⟨x, u⟩ = ‖x‖·radial(u)`` gives::

        g_x(u, v) = ⟨perp(u), perp(v)⟩ + radial(u)·radial(v)·(1 - c·β_x²·‖x‖²)
                  = ⟨perp(u), perp(v)⟩ + (β_x·radial(u))·(β_x·radial(v))

    using ``1 - c‖x‖²/(1 + c‖x‖²) = β_x²``. Nothing is subtracted, and ``β_x`` is applied as a
    *division by* ``1/β_x = √(1 + c‖x‖²)`` (:func:`_beta_inv`) rather than as a squared factor, so
    no ``‖x‖²`` is ever materialized — it overflows float32 at ``‖x‖ = 1.8e19``, geodesic radius
    ~45, while the quantity itself stays representable. ``1/β_x ≥ 1`` always, so it needs no floor.

    The one guard is the ``MIN_NORM`` floor under ``x̂`` inside
    :func:`~hyperbolix.utils.math_utils.radial_perp_decomposition`: at ``x = 0`` it returns the
    exact zero vector, so ``radial = 0`` and ``perp = u`` — the right limit, where the metric is
    the Euclidean one.

    Args:
        u: Tangent vector at x, shape (dim,)
        v: Tangent vector at x, shape (dim,)
        x: PV point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Riemannian inner product ⟨u, v⟩_x, scalar
    """
    radial_u, perp_u_D = radial_perp_decomposition(u, x)
    radial_v, perp_v_D = radial_perp_decomposition(v, x)
    beta_inv = _beta_inv(x, c)  # 1/β_x = √(1 + c‖x‖²) ≥ 1, so no floor is needed
    return jnp.dot(perp_u_D, perp_v_D, precision=MATMUL_PRECISION) + (radial_u / beta_inv) * (radial_v / beta_inv)


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Riemannian norm ``‖v‖_x = √g_x(v, v)``, computed as a hypotenuse of two non-negative legs.

    The radial/perp split of :func:`_tangent_inner` applied to ``v`` twice makes the metric norm
    ``√(‖perp(v)‖² + (β_x·radial(v))²)``, which :func:`~hyperbolix.utils.math_utils.safe_hypot_norm`
    evaluates in one reduction without materializing either square. That replaces the old
    ``safe_sqrt(floor_at(g_x(v, v), 0))``: the form it took the root of was the literal difference,
    whose relative error grows like ``cosh²(a)·eps`` — measured on an *exactly unit* radial tangent
    vector at ``c = 1``, float32, it returned a norm off by 0.59 at scaled radius ``a = 8``, 3.9 at
    9 and 30 at 10, against ≤1.4e-5 for this form.

    Both of the old guards go with it. ``floor_at(., 0.0)`` clipped a form that had rounded
    *negative*; a sum of two squares cannot. ``safe_sqrt``'s job — a finite (zero) derivative at
    ``v = 0``, where ``sqrt'`` is infinite and reverse-mode AD forms ``0 * inf = NaN`` — is what
    ``safe_hypot_norm``'s own double-``where`` already provides, with an exact ``0`` forward value.

    Args:
        v: Tangent vector at x, shape (dim,)
        x: PV point, shape (dim,)
        c: Curvature (positive)

    Returns:
        Riemannian norm ‖v‖_x, scalar
    """
    radial, perp_D = radial_perp_decomposition(v, x)
    return safe_hypot_norm(perp_D, radial / _beta_inv(x, c))


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Tangent-space projection. T_x PV = R^n, so the projection is identity."""
    del x, c
    return v


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Convert Euclidean gradient to Riemannian gradient under the PV metric.

    From g_x(u, v) = ⟨u, A(x) v⟩ with A(x) = I - c·β_x²·x xᵀ, the Riemannian
    gradient satisfies A(x) rgrad = grad. Sherman-Morrison gives
    A(x)⁻¹ = I + c·x xᵀ (the 1 - c·β_x²·||x||² factor cancels with β_x²), so

        rgrad = grad + c · ⟨x, grad⟩ · x.
    """
    xg = jnp.dot(x, grad, precision=MATMUL_PRECISION)
    return grad + c * xg * x


# ---------------------------------------------------------------------------
# Projection & validation
# ---------------------------------------------------------------------------


def _proj(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Projection onto PV. PV is R^n; we only replace non-finite entries."""
    del c
    return jnp.nan_to_num(x)


def _is_in_manifold(x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
    """Every finite point in R^n lies on the PV manifold.

    ``atol`` is accepted for signature uniformity across manifolds; PV is unconstrained, so
    the check is exact finiteness and there is no tolerance to slacken.
    """
    del c, atol
    return jnp.all(jnp.isfinite(x))


def _is_in_tangent_space(
    v: Float[Array, "dim"],
    x: Float[Array, "dim"],
    c: ScalarCurvature,
    atol: float | None = None,
) -> Array:
    """Every finite vector in R^n is a tangent vector at any PV point (``atol`` unused, as above)."""
    del x, c, atol
    return jnp.all(jnp.isfinite(v))


def _embed_spatial_0(v_spatial: Float[Array, "... n"]) -> Float[Array, "... n"]:
    """Identity embedding for PV: no time coord to prepend (kept for API parity)."""
    return v_spatial


# ---------------------------------------------------------------------------
# Batch-compatible helpers (used by NN layers)
# ---------------------------------------------------------------------------


def _compute_mlr(
    x: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim"],
    r: Float[Array, "out_dim 1"],
    c: ScalarCurvature,
    min_enorm: float = 1e-15,
) -> Float[Array, "batch out_dim"]:
    """PV multinomial logistic regression (paper Thm 5.2, Eq. 19 with K = -c).

    For each class k with parameters ``(z_k, r_k)`` the signed margin to the
    PV hyperplane is

        v_k(x) = (||z_k|| / √c) · asinh(
                    cosh(√c·r_k) · √c/||z_k|| · ⟨x, z_k⟩
                    - sinh(√c·r_k) · √(1 + c·||x||²)
                 )

    The asinh argument is unclamped, matching the convention used by the
    Poincaré and Hyperboloid MLR helpers.

    Args:
        x: PV points, shape (batch, in_dim).
        z: Per-class spatial directions, shape (out_dim, in_dim).
        r: Per-class scalar offsets, shape (out_dim, 1).
        c: Curvature (positive).
        min_enorm: Multiplicative floor on ‖z‖ when normalizing z.

    Returns:
        MLR scores, shape (batch, out_dim).
    """
    sqrt_c = jnp.sqrt(c)
    sr_1P = sqrt_c * r.T  # (1, P)

    # `safe_norm` + `floor_at`: `z_norm_P1` divides below, so the floor at `min_enorm` is the
    # deliberate part; the max-scaling removes the old spelling's float32 overflow and its
    # `min_enorm` floor on genuinely small direction rows.
    z_norm_P1 = floor_at(safe_norm(z)[:, None], min_enorm)  # (P, 1)

    # sqrt(1 + c*||x||^2) via `safe_hypot_norm`: no `sum(x**2)` to overflow float32 at
    # ||x|| > 1.8e19/sqrt(c), where the old spelling returned inf and drove the score to inf, and
    # one reduction rather than a rounded `safe_norm(x)` that is squared again. Same form and same
    # measurement as `_beta_inv`; see there.
    beta_inv_x_B1 = safe_hypot_norm(sqrt_c * x, jnp.asarray(1.0, dtype=x.dtype))[:, None]  # (B, 1)

    # Pinned HIGHEST: the MLR logits are a decision quantity, and this dot enters the asinh
    # argument as a difference of a radial and an angular term (see hyperbolix.utils.precision).
    xz_BP = jnp.einsum("bi,oi->bo", x, z, precision=MATMUL_PRECISION)  # (B, P)

    # Eq. 19 asinh argument, in (B, P).
    term_A_BP = cosh(sr_1P) * (sqrt_c / z_norm_P1.T) * xz_BP
    term_B_BP = sinh(sr_1P) * beta_inv_x_B1
    asinh_arg_BP = term_A_BP - term_B_BP

    # No clamp on the asinh argument — same reason as in `manifolds/hyperboloid._compute_mlr`;
    # see there. PV coordinates are unconstrained, so this argument grows like `sinh(r)` with no
    # bounded factor in front of it.
    return (z_norm_P1.T / sqrt_c) * jnp.asinh(asinh_arg_BP)


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class ProperVelocity(ManifoldBase):
    """Proper Velocity (PV) manifold with automatic dtype casting.

    PV is an unconstrained representation of hyperbolic geometry rooted in
    special relativity's proper velocity (Ungar 2022, Ch. 10). Points live
    in R^n without any manifold constraint, which gives better numerical
    stability for large radii than the bounded Poincaré ball or the
    constrained hyperboloid (Chen et al. 2026, Tables 1-3).

    Args:
        dtype: Target JAX dtype for computations (default: ``jnp.float32``).
        c: Curvature value (default: 1.0). Must be positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds.proper_velocity import ProperVelocity
        >>>
        >>> manifold = ProperVelocity(dtype=jnp.float64)
        >>> x = jnp.array([0.1, 0.2], dtype=jnp.float32)
        >>> y = jnp.array([0.3, 0.4], dtype=jnp.float32)
        >>> d = manifold.dist(x, y, c=1.0)
        >>> d.dtype  # float64
    """

    VERSION_DEFAULT = VERSION_DEFAULT

    # -- Structural helpers --------------------------------------------------

    def create_origin(self, c: ScalarCurvature, dim: int) -> Float[Array, "dim"]:
        """Create the PV origin (zero vector in R^n)."""
        return _create_origin(c, dim, self.dtype)

    def beta(self, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
        """PV beta factor β_x = 1/√(1 + c·||x||²)."""
        return _beta(self._cast(x), c)

    # -- Gyro-operations -----------------------------------------------------

    def proj(self, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Projection onto PV (replaces non-finite values; PV is unconstrained)."""
        return _proj(self._cast(x), c)

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """PV gyroaddition x ⊕_U y."""
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float | Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """PV scalar multiplication r ⊗_U x."""
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r_cast, x, c)  # type: ignore[arg-type]

    # -- Distance ------------------------------------------------------------

    def dist(
        self,
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: ScalarCurvature,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Geodesic distance between PV points."""
        del version_idx  # only one implementation currently
        return _dist(self._cast(x), self._cast(y), c)

    def dist_0(self, x: Float[Array, "dim"], c: ScalarCurvature, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
        """Geodesic distance from the PV origin."""
        del version_idx
        return _dist_0(self._cast(x), c)

    # -- Exp / log maps ------------------------------------------------------

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map at x."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map from the origin."""
        return _expmap_0(self._cast(v), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map at x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map to the origin."""
        return _logmap_0(self._cast(y), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Euclidean retraction (exact for PV)."""
        return _retraction(self._cast(v), self._cast(x), c)

    # -- Parallel transport --------------------------------------------------

    def ptransp(
        self,
        v: Float[Array, "dim"],
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: ScalarCurvature,
    ) -> Float[Array, "dim"]:
        """Parallel transport v from T_x PV to T_y PV."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Parallel transport v from T_0 PV to T_y PV."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    # -- Tangent space -------------------------------------------------------

    def tangent_inner(
        self,
        u: Float[Array, "dim"],
        v: Float[Array, "dim"],
        x: Float[Array, "dim"],
        c: ScalarCurvature,
    ) -> Float[Array, ""]:
        """Riemannian inner product ⟨u, v⟩_x."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
        """Riemannian norm ||v||_x."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Tangent-space projection (identity for PV)."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Convert Euclidean gradient to Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    # -- Validation ----------------------------------------------------------

    def is_in_manifold(self, x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
        """Check that all entries are finite (PV has no constraint, so ``atol`` is unused)."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
    ) -> Array:
        """Check that v has finite entries (T_x PV = R^n, so ``atol`` is unused)."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c, atol)

    def embed_spatial_0(self, v_spatial: Float[Array, "... n"]) -> Float[Array, "... n"]:
        """Identity embedding (no time coordinate). Kept for API parity."""
        return _embed_spatial_0(self._cast(v_spatial))

    # -- Batch helpers -------------------------------------------------------

    def compute_mlr(
        self,
        x: Float[Array, "batch in_dim"],
        z: Float[Array, "out_dim in_dim"],
        r: Float[Array, "out_dim 1"],
        c: ScalarCurvature,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """PV multinomial logistic regression (paper Thm 5.2)."""
        return _compute_mlr(self._cast(x), self._cast(z), self._cast(r), c, min_enorm)
