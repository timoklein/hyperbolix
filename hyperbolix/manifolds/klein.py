"""Beltrami-Klein ball manifold - class-based API with dtype control.

Provides a Klein class for manifold operations with automatic dtype casting.
All operations work on single points with shape (dim,). Use jax.vmap for batching.

Convention: ||x||^2 < 1/c with c > 0 and sectional curvature -c. Geodesics are the straight
chords of the ball, and the metric is **not** conformal::

    g_x(u, v) = (u·v)/g_x + c(x·u)(x·v)/g_x²,     g_x := 1 - c‖x‖²  (the boundary gap)

Isometry to the hyperboloid ``-X₀² + ‖X_s‖² = -1/c``::

    X = (1/(√c·√g_x), x/√g_x),      x = X_s/(√c·X₀)

and ``1/√g_x`` is the Einstein Lorentz factor ``gamma_x``. The gyrovector structure is Ungar's
Einstein addition ``⊕_E``; Einstein scalar multiplication and ``expmap_0`` are the same formulas
as their Möbius/Poincaré counterparts and are imported from :mod:`.poincare`.

Numerics
--------
Every two-point quantity is built from ``w = y - x`` and the pair quantity::

    N  = g_x‖w‖² + c(x·w)²,          S² = c·N/(g_x·g_y) = sinh²(√c·d(x, y))

a sum of non-negative terms. It never forms ``1 - c x·y`` or ``‖x‖²‖y‖² - (x·y)²``, the two
cancelling differences of the literal formulas. ``S²`` follows from Zhang et al. 2026 (App.
Eqs. 12 & 14) via the Lagrange identity ``‖x‖²‖y‖² - (x·y)² = ‖x‖²‖w‖² - (x·w)²`` and the
Einstein gamma identity ``1 - c‖(-x) ⊕_E y‖² = g_x g_y/(1 - c x·y)²``.

Dimension key:
    N: number of points (``einstein_midpoint`` only)
    D: manifold dimension (``dim``)

References
----------
Zhang et al. "Klein Hyperbolic Metric Learning." ICML 2026.
Mao et al. "Klein Model for Hyperbolic Neural Networks." arXiv:2410.16813 (2024).
Ungar. "A Gyrovector Space Approach to Hyperbolic Geometry." 2009.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import asinh, floor_at, safe_hypot_norm, safe_norm, safe_sqrt, tanh
from ..utils.precision import MATMUL_PRECISION
from ._base import ManifoldBase, default_atol
from ._gyrovector_core import _boundary_floor, _proj, _proj_batch
from .hyperboloid import _asinhc
from .poincare import _expmap_0 as _poincare_expmap_0
from .poincare import _scalar_mul as _poincare_scalar_mul
from .protocol import ScalarCurvature

# Version selection constant. Klein has a single canonical implementation, kept for API
# consistency with Poincare / Hyperboloid.
VERSION_DEFAULT = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _gap(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Boundary gap ``g_x = 1 - c‖x‖² = 1/gamma_x²``, floored at its analytic minimum on projected points.

    The Klein chart is the same Euclidean ball as Poincaré's with the same ``eps**0.75`` margin
    (:func:`_gyrovector_core._proj`), so :func:`_gyrovector_core._boundary_floor` is the same
    analytic floor here.
    """
    x2 = jnp.dot(x, x, precision=MATMUL_PRECISION)
    return floor_at(1.0 - c * x2, _boundary_floor(x, c))


def _pair(
    x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, "dim"], Float[Array, ""]]:
    """Shared pair quantities ``(g_x, g_y, w, S²)`` of :func:`_dist`, :func:`_logmap`, :func:`_ptransp`.

    ``S² = c·N/(g_x·g_y) = sinh²(√c·d)`` with ``N = g_x‖w‖² + c(x·w)² = g_x²·‖w‖_x²``, ``w = y - x``.
    Both terms of ``N`` are non-negative and ``w`` is formed first (exact by Sterbenz for close
    points), so the separation signal never sits in a difference of O(1) terms. At ``x == y``,
    ``w`` is exactly zero and so is ``S²``.
    """
    g_x = _gap(x, c)
    g_y = _gap(y, c)
    w_D = y - x
    w2 = jnp.dot(w_D, w_D, precision=MATMUL_PRECISION)
    xw = jnp.dot(x, w_D, precision=MATMUL_PRECISION)
    n = g_x * w2 + c * xw * xw
    s2 = c * n / (g_x * g_y)
    return g_x, g_y, w_D, s2


def _xcothx(theta: Float[Array, "..."]) -> Float[Array, "..."]:
    """``θ·coth θ`` — even, analytic, value ``1`` and slope ``0`` at ``θ = 0``.

    Below ``(945·eps/2)^(1/6)`` — 0.196 in float32, 6.9e-3 in float64, where the first dropped
    term ``2θ⁶/945`` falls under one rounding — the series ``1 + θ²/3 - θ⁴/45`` is used. Double
    ``where``: the direct branch is ``0/0`` at ``θ = 0``, so its *argument* is sanitised too, or the
    NaN derivative would leak into the selected branch's cotangent.
    """
    threshold = (945.0 / 2.0 * float(jnp.finfo(theta.dtype).eps)) ** (1.0 / 6.0)
    small = theta < threshold
    theta_safe = jnp.where(small, jnp.ones_like(theta), theta)
    t2 = theta * theta
    series = 1.0 + t2 / 3.0 - t2 * t2 / 45.0
    # `tanh` is the math_utils wrapper: its saturated tail returns 1 - 10·eps with zero slope,
    # so for large θ this is θ/(1 - 10·eps) with derivative ≈ 1, the true limit.
    return jnp.where(small, series, theta_safe / tanh(theta_safe))


# ---------------------------------------------------------------------------
# Gyro-operations
# ---------------------------------------------------------------------------


def _addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Einstein gyrovector addition ``x ⊕_E y`` (non-commutative, non-associative).

    Literal form (Ungar 2009; Mao et al. 2024), ``gamma_x = 1/√g_x``::

        x ⊕_E y = (x + y/gamma_x + c·gamma_x/(1 + gamma_x)·(x·y)·x) / (1 + c x·y)

    Evaluated with ``s = x + y`` as ``(√g_x·s + c(x·s)·x/(1 + √g_x)) / (g_x + c(x·s))``: the
    coefficient of ``x`` collapses via ``√g_x + c‖x‖²/(1 + √g_x) = 1``, so at ``y ≈ -x`` nothing
    subtracts two O(1) terms. Algebraically identical to the literal form.
    """
    g_x = _gap(x, c)
    sqrt_g_x = jnp.sqrt(g_x)
    s_D = x + y
    xs = jnp.dot(x, s_D, precision=MATMUL_PRECISION)
    num_D = sqrt_g_x * s_D + (c * xs / (1.0 + sqrt_g_x)) * x
    return _proj(num_D / (g_x + c * xs), c)


def _gyro_difference(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Einstein gyro-difference ``(⊖x) ⊕_E y`` with ``⊖x = -x``.

    :func:`_addition` at ``(-x, y)`` with ``s = w = y - x`` formed first::

        (√g_x·w + c(x·w)·x/(1 + √g_x)) / (g_x - c(x·w))

    so two close points give a small result without cancellation (the literal ``addition(-x, y)``
    forms ``1 - c x·y`` near the boundary). Its norm is ``tanh(√c·d(x, y))/√c``.
    """
    g_x = _gap(x, c)
    sqrt_g_x = jnp.sqrt(g_x)
    w_D = y - x
    xw = jnp.dot(x, w_D, precision=MATMUL_PRECISION)
    num_D = sqrt_g_x * w_D + (c * xw / (1.0 + sqrt_g_x)) * x
    return _proj(num_D / (g_x - c * xw), c)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def _dist(x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Geodesic distance ``d(x, y) = arsinh(S)/√c``, ``S² = c·N/(g_x·g_y)``.

    The same function as Zhang et al. 2026 Eq. 4, ``artanh(√c‖(-x) ⊕_E y‖)/√c``, and as the
    hyperboloid ``arcosh(-c⟨X, Y⟩_L)/√c``; see the module docstring for the derivation. ``arsinh``
    needs no domain clamp and ``S`` is linear in ``‖w‖``, so short distances keep full relative
    accuracy. ``safe_sqrt`` gives ``d(x, x) = 0`` exactly with a zero (finite) gradient.
    """
    _, _, _, s2 = _pair(x, y, c)
    return asinh(safe_sqrt(s2)) / jnp.sqrt(c)


def _dist_0(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Distance from the origin, ``arsinh(√c‖x‖/√g_x)/√c`` (``= artanh(√c‖x‖)/√c``, :func:`_dist` at ``0``)."""
    sqrt_c = jnp.sqrt(c)
    return asinh(sqrt_c * safe_norm(x) / jnp.sqrt(_gap(x, c))) / sqrt_c


# ---------------------------------------------------------------------------
# Exp / log maps
# ---------------------------------------------------------------------------


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Exponential map ``exp_x(v) = x + v / (θ·coth θ + c(x·v)/g_x)``, ``θ = √c·‖v‖_x``.

    Klein geodesics are chords, so ``exp_x(v)`` lies on the ray ``x + t·v``; the scalar follows
    from the hyperboloid geodesic pulled back through the isometry. ``θ·coth θ`` is evaluated by
    :func:`_xcothx`, so ``v = 0`` gives ``x`` with Jacobian exactly the identity. The denominator is
    positive: ``θ·coth θ > θ ≥ c|x·v|/g_x``.

    Mao et al.'s reference ``_klein_expmap`` omits the factor ``c`` in the denominator's second
    term (exact only at ``c = 1``).

    ``g_x‖v‖²`` overflows float32 past tangent coordinate ~1.8e19 (a diverging network); there
    ``θ = inf`` and the result is the base point ``x``, as for ``Poincare.expmap``.
    """
    g_x = _gap(x, c)
    xv = jnp.dot(x, v, precision=MATMUL_PRECISION)
    v2 = jnp.dot(v, v, precision=MATMUL_PRECISION)
    theta = jnp.sqrt(c) * safe_sqrt(g_x * v2 + c * xv * xv) / g_x
    return _proj(x + v / (_xcothx(theta) + c * xv / g_x), c)


def _expmap_0(v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """``exp_0(v) = tanh(√c‖v‖)·v/(√c‖v‖)``.

    Identical to Poincaré's ``exp_0``: the Klein metric at the origin is the identity and a Klein
    point at distance ``d`` has ``√c‖x‖ = tanh(√c·d)``, while the Poincaré metric there is 4 times the
    identity and a Poincaré point has ``√c‖x‖ = tanh(√c·d/2)`` — the two factors of 2 cancel.
    """
    return _poincare_expmap_0(v, c)


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Retraction ``proj(x + v)`` — the first-order approximation of :func:`_expmap`."""
    return _proj(x + v, c)


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Logarithmic map ``log_x(y) = d(x, y)·w/‖w‖_x = arsinhc(S)·√(g_x/g_y)·w``.

    The direction is the chord ``w = y - x`` (geodesics are straight) and ``‖w‖_x = √N/g_x``, so
    ``d/‖w‖_x`` collapses to ``arsinh(S)/S·√(g_x/g_y)``. ``_asinhc`` is analytic at ``S = 0``, so
    no origin or coincidence special case is needed and the gradient at ``y = x`` is finite.
    """
    g_x, g_y, w_D, s2 = _pair(x, y, c)
    return _asinhc(safe_sqrt(s2)) * jnp.sqrt(g_x / g_y) * w_D


def _logmap_0(y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """``log_0(y) = arsinhc(s)·y/√g_y``, ``s = √c‖y‖/√g_y`` (:func:`_logmap` at ``x = 0``)."""
    sqrt_g_y = jnp.sqrt(_gap(y, c))
    s = jnp.sqrt(c) * safe_norm(y) / sqrt_g_y
    return _asinhc(s) * y / sqrt_g_y


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------


def _ptransp(
    v: Float[Array, "dim"], x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, "dim"]:
    """Parallel transport along the chord from ``x`` to ``y``: ``√(g_y/g_x)·(v - κ·w)``.

    ``κ = [c(x·v)/g_x + c(y·v)/√(g_x·g_y)] / (1 + cosh(√c·d))`` with ``cosh(√c·d) = √(1 + S²)``;
    the hyperboloid transport ``V + ⟨Y, V⟩_L/(1/c - ⟨X, Y⟩_L)·(X + Y)`` pulled back through the
    isometry. Preserves the Klein metric.
    """
    g_x, g_y, w_D, s2 = _pair(x, y, c)
    xv = jnp.dot(x, v, precision=MATMUL_PRECISION)
    yv = jnp.dot(y, v, precision=MATMUL_PRECISION)
    kappa = (c * xv / g_x + c * yv / jnp.sqrt(g_x * g_y)) / (1.0 + jnp.sqrt(1.0 + s2))
    return jnp.sqrt(g_y / g_x) * (v - kappa * w_D)


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Transport from the origin: ``√g_y·v - c(y·v)·√g_y/(1 + √g_y)·y`` (:func:`_ptransp` at ``x = 0``)."""
    sqrt_g_y = jnp.sqrt(_gap(y, c))
    yv = jnp.dot(y, v, precision=MATMUL_PRECISION)
    return sqrt_g_y * v - (c * yv * sqrt_g_y / (1.0 + sqrt_g_y)) * y


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def _tangent_inner(
    u: Float[Array, "dim"], v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature
) -> Float[Array, ""]:
    """Klein metric ``g_x(u, v) = [g_x(u·v) + c(x·u)(x·v)] / g_x²``."""
    g_x = _gap(x, c)
    uv = jnp.dot(u, v, precision=MATMUL_PRECISION)
    xu = jnp.dot(x, u, precision=MATMUL_PRECISION)
    xv = jnp.dot(x, v, precision=MATMUL_PRECISION)
    return (g_x * uv + c * xu * xv) / (g_x * g_x)


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """``‖v‖_x = sqrt(g_x‖v‖² + c(x·v)²)/g_x``.

    ``v`` is a tangent vector of unbounded magnitude, so the square root goes through
    ``safe_hypot_norm`` (overflow-free, one reduction, exact 0 with zero VJP at ``v = 0``), as
    ``Poincare.tangent_norm`` keeps its two-pass ``safe_norm``.
    """
    g_x = _gap(x, c)
    xv = jnp.dot(x, v, precision=MATMUL_PRECISION)
    return safe_hypot_norm(jnp.sqrt(g_x) * v, jnp.sqrt(c) * xv) / g_x


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Riemannian gradient via the inverse Klein metric: ``g_x·(grad - c(x·grad)·x)``."""
    g_x = _gap(x, c)
    xg = jnp.dot(x, grad, precision=MATMUL_PRECISION)
    return g_x * (grad - (c * xg) * x)


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Identity: the ball is an open subset of R^d, so ``T_x K = R^d``."""
    del x, c
    return v


def _is_in_manifold(x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
    """``c‖x‖² < 1 + atol`` — the dimensionless form of Poincaré's check (same ball, same ``_proj``)."""
    x_sqnorm = jnp.dot(x, x, precision=MATMUL_PRECISION)
    tol = default_atol(x.dtype) if atol is None else atol
    return c * x_sqnorm < 1.0 + tol


def _is_in_tangent_space(
    v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
) -> Array:
    """``T_x K = R^d``, so the only check is that every entry of ``v`` is finite."""
    del x, c, atol
    return jnp.all(jnp.isfinite(v))


# ---------------------------------------------------------------------------
# Einstein extras
# ---------------------------------------------------------------------------


def _lorentz_factor(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Einstein Lorentz factor ``gamma_x = 1/√(1 - c‖x‖²)`` (the hyperboloid ``√c·X₀``)."""
    return 1.0 / jnp.sqrt(_gap(x, c))


def _einstein_midpoint(
    x_ND: Float[Array, "N dim"], weights_N: Float[Array, "N"] | None, c: ScalarCurvature
) -> Float[Array, "dim"]:
    """Weighted Einstein midpoint ``Σ wᵢ·gammaᵢ·xᵢ / Σ wᵢ·gammaᵢ`` of N Klein points (Ungar 2009).

    Equal to the normalized weighted Lorentz centroid ``Σ wᵢXᵢ / (√c‖Σ wᵢXᵢ‖_L)`` mapped to Klein:
    the Klein coordinate ``X_s/(√c·X₀)`` of any positive multiple of ``Σ wᵢXᵢ`` is this ratio.
    With non-negative weights every term is non-negative, so nothing cancels.
    """
    if weights_N is None:
        weights_N = jnp.ones(x_ND.shape[0], dtype=x_ND.dtype)
    x2_N = jnp.einsum("nd,nd->n", x_ND, x_ND, precision=MATMUL_PRECISION)  # (N,)
    g_N = floor_at(1.0 - c * x2_N, _boundary_floor(x_ND, c))  # (N,)
    wg_N = weights_N / jnp.sqrt(g_N)  # (N,) wᵢ·gammaᵢ
    num_D = jnp.einsum("n,nd->d", wg_N, x_ND, precision=MATMUL_PRECISION)  # (D,)
    return _proj(num_D / jnp.sum(wg_N), c)


# ---------------------------------------------------------------------------
# Class-based manifold API
# ---------------------------------------------------------------------------


class Klein(ManifoldBase):
    """Beltrami-Klein ball manifold (curvature ``-c``, radius ``1/√c``) with automatic dtype casting.

    Points are Euclidean ball points ``c‖x‖² < 1`` as for :class:`~hyperbolix.manifolds.Poincare`,
    but geodesics are straight chords and the metric
    ``g_x(u, v) = (u·v)/g_x + c(x·u)(x·v)/g_x²`` (``g_x = 1 - c‖x‖²``) is not conformal. The
    gyrovector structure is Einstein addition; ``scalar_mul`` and ``expmap_0`` coincide with
    Poincaré's formulas.

    Precision: the chart shares Poincaré's ``eps**0.75`` boundary margin, but a Klein point at
    scaled radius ``a = √c·d`` has ``√c‖x‖ = tanh(a)`` where a Poincaré point has ``tanh(a/2)``,
    so the representable ceiling is **half** Poincaré's: ``a ≈ 6.3`` in float32 and ``≈ 13.9`` in
    float64 (at ``c = 1``). Inside it the pairwise ops (``dist``, ``logmap``, ``ptransp``,
    ``gyro_difference``) are cancellation-free; what remains is the storage floor of the gap,
    ``1 - c‖x‖²`` known to relative ``eps·cosh²(a)``.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)
        c: Curvature value (default: 1.0). Must be positive.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from hyperbolix.manifolds import Klein
        >>>
        >>> manifold = Klein(dtype=jnp.float64)
        >>> x = jnp.array([0.1, 0.2])
        >>> y = jnp.array([0.3, 0.4])
        >>> d = manifold.dist(x, y, c=1.0)
        >>> dists = jax.vmap(manifold.dist, in_axes=(0, 0, None))(jnp.stack([x, y]), jnp.stack([y, x]), 1.0)

    References:
        Zhang et al. "Klein Hyperbolic Metric Learning." ICML 2026.
        Mao et al. "Klein Model for Hyperbolic Neural Networks." arXiv:2410.16813 (2024).
        Ungar. "A Gyrovector Space Approach to Hyperbolic Geometry." 2009.
    """

    VERSION_DEFAULT = VERSION_DEFAULT

    # -- Projection ----------------------------------------------------------

    def proj(self, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Project a point onto the Klein ball by clipping its norm (same clamp as Poincaré)."""
        return _proj(self._cast(x), c)

    def proj_batch(self, x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... dim"]:
        """Project batched points onto the ball (arbitrary leading dimensions)."""
        return _proj_batch(self._cast(x), c)

    # -- Gyro-operations -----------------------------------------------------

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Einstein gyrovector addition ``x ⊕_E y``."""
        return _addition(self._cast(x), self._cast(y), c)

    def gyro_difference(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Einstein gyro-difference ``(⊖x) ⊕_E y``, evaluated without the ``1 - c x·y`` cancellation.

        Mathematically identical to ``addition(-x, y)``; use it when the result is expected much
        closer to the origin than the operands (centering, differences of two close points).
        """
        return _gyro_difference(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float | Float[Array, ""], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Einstein scalar multiplication ``r ⊗ x = tanh(r·artanh(√c‖x‖))·x/(√c‖x‖)``.

        The same formula as Möbius scalar multiplication (Ungar 2009): both scale the origin
        distance by ``r``, and each model's radial coordinate is ``tanh`` of its own multiple of
        that distance.
        """
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _poincare_scalar_mul(r_cast, x, c)  # type: ignore[arg-type]

    # -- Distance ------------------------------------------------------------

    def dist(
        self,
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: ScalarCurvature,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Geodesic distance ``arsinh(S)/√c`` between Klein points (single implementation)."""
        del version_idx  # only one implementation
        return _dist(self._cast(x), self._cast(y), c)

    def dist_0(self, x: Float[Array, "dim"], c: ScalarCurvature, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
        """Geodesic distance from the origin."""
        del version_idx
        return _dist_0(self._cast(x), c)

    # -- Exp / log maps ------------------------------------------------------

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map: tangent vector v at x to the manifold."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Exponential map from the origin (the Poincaré formula; see :func:`_expmap_0`)."""
        return _expmap_0(self._cast(v), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Retraction ``proj(x + v)``."""
        return _retraction(self._cast(v), self._cast(x), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map: point y to the tangent space at x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Logarithmic map from the origin."""
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
        """Klein metric ``g_x(u, v)``."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
        """Riemannian norm ``‖v‖_x``."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Convert a Euclidean gradient to the Riemannian gradient (inverse Klein metric)."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
        """Project v onto the tangent space at x (identity)."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def is_in_manifold(self, x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None) -> Array:
        """Check ``c‖x‖² < 1 + atol`` (``atol`` default: :func:`default_atol`)."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(
        self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: ScalarCurvature, atol: float | None = None
    ) -> Array:
        """Check that v has finite entries (``T_x K = R^d``, so there is no other constraint)."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c, atol)

    # -- Einstein extras -----------------------------------------------------

    def lorentz_factor(self, x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
        """Einstein Lorentz factor ``gamma_x = 1/√(1 - c‖x‖²)``, scalar."""
        return _lorentz_factor(self._cast(x), c)

    def einstein_midpoint(
        self, x_ND: Float[Array, "N dim"], weights_N: Float[Array, "N"] | None, c: ScalarCurvature
    ) -> Float[Array, "dim"]:
        """Weighted Einstein midpoint ``Σ wᵢ·gammaᵢ·xᵢ / Σ wᵢ·gammaᵢ`` of N points.

        Args:
            x_ND: Klein points, shape (N, dim)
            weights_N: Non-negative weights, shape (N,); ``None`` for uniform weights
            c: Curvature (positive)

        Returns:
            Midpoint, shape (dim,) — the normalized weighted Lorentz centroid mapped to Klein.
        """
        weights = None if weights_N is None else self._cast(weights_N)
        return _einstein_midpoint(self._cast(x_ND), weights, c)
