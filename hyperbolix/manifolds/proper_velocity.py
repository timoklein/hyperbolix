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

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import cosh, sinh, smooth_clamp
from ._base import ManifoldBase

# Default numerical parameters
MIN_NORM = 1e-15

# Version selection constant. PV currently has a single canonical implementation,
# kept for API consistency with Poincare / Hyperboloid.
VERSION_DEFAULT = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _create_origin(c: Float[Array, ""] | float, dim: int, dtype=jnp.float32) -> Float[Array, "dim"]:
    """Create PV origin: the zero vector in R^n."""
    del c  # curvature is irrelevant: the PV origin is always 0.
    return jnp.zeros(dim, dtype=dtype)


def _beta(x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, ""]:
    """PV beta factor β_x = 1/√(1 + c·||x||²)."""
    x_sqnorm = jnp.dot(x, x)
    return 1.0 / jnp.sqrt(1.0 + c * x_sqnorm)


def _beta_inv(x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, ""]:
    """Reciprocal of the PV beta factor: 1/β_x = √(1 + c·||x||²)."""
    x_sqnorm = jnp.dot(x, x)
    return jnp.sqrt(1.0 + c * x_sqnorm)


def _safe_norm(x: Float[Array, "dim"]) -> Float[Array, ""]:
    """Numerically safe Euclidean norm √(||x||² + MIN_NORM²) with smooth gradient at x=0."""
    return jnp.sqrt(jnp.sum(x**2) + MIN_NORM**2)


def _dpi_x(x: Float[Array, "dim"], v: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Differential of π: PV → Poincaré (paper Eq. 7 with K = -c).

    dπ_x(v) = β_x/(1+β_x)·v - c·β_x³/(1+β_x)²·⟨x, v⟩·x
    """
    beta_x = _beta(x, c)
    xv = jnp.dot(x, v)
    one_plus_beta = 1.0 + beta_x
    term1 = (beta_x / one_plus_beta) * v
    term2 = (c * beta_x**3 / one_plus_beta**2) * xv * x
    return term1 - term2


def _mobius_gyration(
    a: Float[Array, "dim"],
    b: Float[Array, "dim"],
    z: Float[Array, "dim"],
    c: Float[Array, ""] | float,
) -> Float[Array, "dim"]:
    """Möbius gyration gyr_M[a, b]z on the Poincaré ball (self-contained copy).

    Same formula as ``poincare._gyration`` but inlined here so the PV module
    does not depend on the Poincaré implementation.
    """
    c2 = c**2
    a_sqnorm = jnp.dot(a, a)
    b_sqnorm = jnp.dot(b, b)
    ab = jnp.dot(a, b)
    az = jnp.dot(a, z)
    bz = jnp.dot(b, z)

    coef_a = -c2 * az * b_sqnorm + c * bz + 2 * c2 * ab * bz
    coef_b = -c2 * bz * a_sqnorm - c * az
    num = 2 * (coef_a * a + coef_b * b)
    denom = jnp.maximum(1 + 2 * c * ab + c2 * a_sqnorm * b_sqnorm, MIN_NORM)
    return z + num / denom


# ---------------------------------------------------------------------------
# PV gyro-operations
# ---------------------------------------------------------------------------


def _addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """PV gyroaddition x ⊕_U y (paper Eq. 2 with K = -c).

    x ⊕ y = x + y + {(1 - β_y)/β_y + c·β_x/(1+β_x)·⟨x, y⟩}·x
    """
    beta_x = _beta(x, c)
    xy = jnp.dot(x, y)

    # Numerically robust form for (1 - β_y)/β_y = 1/β_y - 1 = √(1+c||y||²) - 1.
    # This avoids catastrophic cancellation for ||y|| ≈ 0.
    beta_y_inv_minus_one = _beta_inv(y, c) - 1.0
    coef = beta_y_inv_minus_one + c * beta_x / (1.0 + beta_x) * xy
    return x + y + coef * x


def _scalar_mul(t: Float[Array, ""] | float, x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """PV scalar multiplication t ⊗_U x (paper Eq. 3 with K = -c).

    t ⊗ x = sinh(t · asinh(√c·||x||)) · x / (√c·||x||),     t ⊗ 0 = 0
    """
    sqrt_c = jnp.sqrt(c)
    x_norm = _safe_norm(x)
    arg = sqrt_c * x_norm  # √c·||x||, never exactly zero
    # sinh is the overflow-protected variant; jnp.asinh is stable on all of R.
    scale = sinh(t * jnp.asinh(arg)) / arg
    return scale * x


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def _dist(x: Float[Array, "dim"], y: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, ""]:
    """Geodesic distance d(x, y) on PV (paper Eq. 13, asinh form).

    The atanh form ``(2/√c)·atanh(√c·||π(-x⊕y)||)`` is algebraically equal to
    ``(1/√c)·asinh(√c·||-x⊕y||)`` (both follow from the tanh/sinh half-angle
    identities). We use the asinh form because jnp.asinh is stable over all
    of R while atanh requires boundary clamping.
    """
    sqrt_c = jnp.sqrt(c)
    z = _addition(-x, y, c)
    z_norm = jnp.linalg.norm(z)
    return jnp.asinh(sqrt_c * z_norm) / sqrt_c


def _dist_0(x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, ""]:
    """Geodesic distance from the PV origin (paper Thm 4.3 simplified).

    d(0, x) = (1/√c) · asinh(√c · ||x||)
    """
    sqrt_c = jnp.sqrt(c)
    x_norm = jnp.linalg.norm(x)
    return jnp.asinh(sqrt_c * x_norm) / sqrt_c


# ---------------------------------------------------------------------------
# Exp / log maps
# ---------------------------------------------------------------------------


def _expmap_0(v: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Exponential map from the origin (paper Thm 4.3 simplified).

    exp_0(v) = sinh(√c·||v||) · v / (√c·||v||)

    Written via sinh/arg with a safe-norm substitution so the v=0 case gives 0.
    """
    sqrt_c = jnp.sqrt(c)
    v_norm = _safe_norm(v)
    arg = sqrt_c * v_norm
    # sinh(arg)/arg has limit 1 as arg → 0, which the safe_norm preserves.
    scale = sinh(arg) / arg
    return scale * v


def _logmap_0(y: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Logarithmic map to the origin (paper Thm 4.3 simplified).

    log_0(y) = asinh(√c·||y||) · y / (√c·||y||)
    """
    sqrt_c = jnp.sqrt(c)
    y_norm = _safe_norm(y)
    arg = sqrt_c * y_norm
    # asinh(arg)/arg has limit 1 as arg → 0.
    scale = jnp.asinh(arg) / arg
    return scale * y


def _expmap(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Exponential map at x (paper Eq. 10 with K = -c).

    Uses the simplified form

        exp_x(v) = x ⊕ ((1+β_x)/β_x · sinhc(√c·√g_x(v,v)) · dπ_x(v))

    with sinhc(t) = sinh(t)/t. This exploits the identity
    ``(1+β_x)/β_x · ||dπ_x(v)|| = √g_x(v, v)`` (proved by expanding both sides
    against the Riemannian metric), which avoids dividing by ||dπ_x(v)||
    — potentially tiny — and keeps the result well-defined at v = 0.
    """
    sqrt_c = jnp.sqrt(c)
    beta_x = _beta(x, c)

    dpi_v = _dpi_x(x, v, c)

    # g_x(v, v) = ⟨v, v⟩ - c·β_x²·⟨x, v⟩²
    xv = jnp.dot(x, v)
    g_vv = jnp.dot(v, v) - c * beta_x**2 * xv**2
    g_vv_safe = jnp.maximum(g_vv, 0.0) + MIN_NORM**2
    g_norm = jnp.sqrt(g_vv_safe)
    arg = sqrt_c * g_norm

    # sinhc(arg) = sinh(arg)/arg; the safe norm guarantees arg > 0.
    sinhc = sinh(arg) / arg

    coef = (1.0 + beta_x) / beta_x * sinhc
    return _addition(x, coef * dpi_v, c)


def _logmap(y: Float[Array, "dim"], x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Logarithmic map at x (paper Eq. 11 with K = -c).

    Uses the simplified form

        log_x(y) = asinhc(√c·||z||) · (z + (β_x·c / (1+β_x)) · ⟨x, z⟩ · x)

    with z = -x ⊕ y and asinhc(t) = asinh(t)/t. This follows from the
    identity ``2·atanh(√c·||π(z)||) = asinh(√c·||z||)`` (tanh/sinh half-angle
    identity applied to ``β_z/(1+β_z)·√c·||z||``), which collapses the paper's
    sigma, tau coefficients into a single scalar scaling applied after a vector
    combination. Avoids explicitly computing π(z).
    """
    sqrt_c = jnp.sqrt(c)
    beta_x = _beta(x, c)

    z = _addition(-x, y, c)
    xz = jnp.dot(x, z)
    z_norm = _safe_norm(z)
    arg = sqrt_c * z_norm

    # asinhc(arg) = asinh(arg)/arg, limit 1 as arg → 0.
    asinhc = jnp.asinh(arg) / arg

    coef_x = beta_x * c / (1.0 + beta_x)
    direction = z + coef_x * xz * x
    return asinhc * direction


def _retraction(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Euclidean retraction x + v. PV is unconstrained, so this is exact.

    Kept separate from expmap to match the other manifolds' APIs; use ``expmap``
    for the exact Riemannian map.
    """
    del c  # PV is unconstrained -- no projection needed.
    return x + v


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------


def _ptransp_0(v: Float[Array, "dim"], y: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Parallel transport from the origin to y (paper Thm 4.3, K = -c).

    PT_{0→y}(v) = v + c·β_y/(1+β_y) · ⟨y, v⟩ · y
    """
    beta_y = _beta(y, c)
    yv = jnp.dot(y, v)
    coef = c * beta_y / (1.0 + beta_y)
    return v + coef * yv * y


def _ptransp(
    v: Float[Array, "dim"],
    x: Float[Array, "dim"],
    y: Float[Array, "dim"],
    c: Float[Array, ""] | float,
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

    # Möbius gyration in the Poincaré ball.
    v_tilde = _mobius_gyration(y_bar, -x_bar, dpi_v, c)

    # Eq. 12 (K = -c flips the sign of the second term).
    yv = jnp.dot(y, v_tilde)
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
    c: Float[Array, ""] | float,
) -> Float[Array, ""]:
    """Riemannian inner product ⟨u, v⟩_x (paper Eq. 1 with K = -c).

    g_x(u, v) = ⟨u, v⟩ - c·β_x²·⟨x, u⟩·⟨x, v⟩
    """
    beta_x = _beta(x, c)
    uv = jnp.dot(u, v)
    xu = jnp.dot(x, u)
    xv = jnp.dot(x, v)
    return uv - c * beta_x**2 * xu * xv


def _tangent_norm(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, ""]:
    """Riemannian norm ||v||_x = √g_x(v, v)."""
    inner = _tangent_inner(v, v, x, c)
    return jnp.sqrt(jnp.maximum(inner, 0.0))


def _tangent_proj(v: Float[Array, "dim"], x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Tangent-space projection. T_x PV = R^n, so the projection is identity."""
    del x, c
    return v


def _egrad2rgrad(grad: Float[Array, "dim"], x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Convert Euclidean gradient to Riemannian gradient under the PV metric.

    From g_x(u, v) = ⟨u, A(x) v⟩ with A(x) = I - c·β_x²·x xᵀ, the Riemannian
    gradient satisfies A(x) rgrad = grad. Sherman-Morrison gives
    A(x)⁻¹ = I + c·x xᵀ (the 1 - c·β_x²·||x||² factor cancels with β_x²), so

        rgrad = grad + c · ⟨x, grad⟩ · x.
    """
    xg = jnp.dot(x, grad)
    return grad + c * xg * x


# ---------------------------------------------------------------------------
# Projection & validation
# ---------------------------------------------------------------------------


def _proj(x: Float[Array, "dim"], c: Float[Array, ""] | float) -> Float[Array, "dim"]:
    """Projection onto PV. PV is R^n; we only replace non-finite entries."""
    del c
    return jnp.nan_to_num(x)


def _is_in_manifold(x: Float[Array, "dim"], c: Float[Array, ""] | float, atol: float = 1e-5) -> Array:
    """Every finite point in R^n lies on the PV manifold."""
    del c, atol
    return jnp.all(jnp.isfinite(x))


def _is_in_tangent_space(
    v: Float[Array, "dim"],
    x: Float[Array, "dim"],
    c: Float[Array, ""] | float,
    atol: float | None = None,
) -> Array:
    """Every finite vector in R^n is a tangent vector at any PV point."""
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
    c: Float[Array, ""] | float,
    clamping_factor: float,
    smoothing_factor: float,
    min_enorm: float = 1e-15,
) -> Float[Array, "batch out_dim"]:
    """PV multinomial logistic regression (paper Thm 5.2, Eq. 19 with K = -c).

    For each class k with parameters ``(z_k, r_k)`` the signed margin to the
    PV hyperplane is

        v_k(x) = (||z_k|| / √c) · asinh(
                    cosh(√c·r_k) · √c/||z_k|| · ⟨x, z_k⟩
                    - sinh(√c·r_k) · √(1 + c·||x||²)
                 )

    The asinh argument is smoothly clamped to ``±clamping_factor · log(2/eps)``
    for numerical stability, matching the convention used by the Poincaré and
    Hyperboloid MLR helpers.

    Args:
        x: PV points, shape (batch, in_dim).
        z: Per-class spatial directions, shape (out_dim, in_dim).
        r: Per-class scalar offsets, shape (out_dim, 1).
        c: Curvature (positive).
        clamping_factor: Scales the dtype-dependent clamp bound.
        smoothing_factor: Softness of the smooth clamp transition.
        min_enorm: Lower bound added under the sqrt when normalizing z.

    Returns:
        MLR scores, shape (batch, out_dim).
    """
    sqrt_c = jnp.sqrt(c)
    sr_1P = sqrt_c * r.T  # (1, P)

    # Safe norm on z so that z=0 rows do not create NaNs.
    z_norm_P1 = jnp.sqrt(jnp.sum(z**2, axis=-1, keepdims=True) + min_enorm**2)  # (P, 1)

    beta_inv_x_B1 = jnp.sqrt(1.0 + c * jnp.sum(x**2, axis=-1, keepdims=True))  # (B, 1)

    xz_BP = jnp.einsum("bi,oi->bo", x, z)  # (B, P)

    # Eq. 19 asinh argument, in (B, P).
    term_A_BP = cosh(sr_1P) * (sqrt_c / z_norm_P1.T) * xz_BP
    term_B_BP = sinh(sr_1P) * beta_inv_x_B1
    asinh_arg_BP = term_A_BP - term_B_BP

    eps = jnp.finfo(x.dtype).eps
    clamp = clamping_factor * float(math.log(2.0 / eps))
    asinh_arg_BP = smooth_clamp(asinh_arg_BP, -clamp, clamp, smoothing_factor)

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

    def create_origin(self, c: float, dim: int) -> Float[Array, "dim"]:
        """Create the PV origin (zero vector in R^n)."""
        return _create_origin(c, dim, self.dtype)

    def beta(self, x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
        """PV beta factor β_x = 1/√(1 + c·||x||²)."""
        return _beta(self._cast(x), c)

    # -- Gyro-operations -----------------------------------------------------

    def proj(self, x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Projection onto PV (replaces non-finite values; PV is unconstrained)."""
        return _proj(self._cast(x), c)

    def addition(self, x: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """PV gyroaddition x ⊕_U y."""
        return _addition(self._cast(x), self._cast(y), c)

    def scalar_mul(self, r: float, x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """PV scalar multiplication r ⊗_U x."""
        x = self._cast(x)
        r_cast = jnp.asarray(r, dtype=x.dtype)
        return _scalar_mul(r_cast, x, c)  # type: ignore[arg-type]

    # -- Distance ------------------------------------------------------------

    def dist(
        self,
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: float,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Geodesic distance between PV points."""
        del version_idx  # only one implementation currently
        return _dist(self._cast(x), self._cast(y), c)

    def dist_0(self, x: Float[Array, "dim"], c: float, version_idx: int = VERSION_DEFAULT) -> Float[Array, ""]:
        """Geodesic distance from the PV origin."""
        del version_idx
        return _dist_0(self._cast(x), c)

    # -- Exp / log maps ------------------------------------------------------

    def expmap(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Exponential map at x."""
        return _expmap(self._cast(v), self._cast(x), c)

    def expmap_0(self, v: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Exponential map from the origin."""
        return _expmap_0(self._cast(v), c)

    def logmap(self, y: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Logarithmic map at x."""
        return _logmap(self._cast(y), self._cast(x), c)

    def logmap_0(self, y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Logarithmic map to the origin."""
        return _logmap_0(self._cast(y), c)

    def retraction(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Euclidean retraction (exact for PV)."""
        return _retraction(self._cast(v), self._cast(x), c)

    # -- Parallel transport --------------------------------------------------

    def ptransp(
        self,
        v: Float[Array, "dim"],
        x: Float[Array, "dim"],
        y: Float[Array, "dim"],
        c: float,
    ) -> Float[Array, "dim"]:
        """Parallel transport v from T_x PV to T_y PV."""
        return _ptransp(self._cast(v), self._cast(x), self._cast(y), c)

    def ptransp_0(self, v: Float[Array, "dim"], y: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Parallel transport v from T_0 PV to T_y PV."""
        return _ptransp_0(self._cast(v), self._cast(y), c)

    # -- Tangent space -------------------------------------------------------

    def tangent_inner(
        self,
        u: Float[Array, "dim"],
        v: Float[Array, "dim"],
        x: Float[Array, "dim"],
        c: float,
    ) -> Float[Array, ""]:
        """Riemannian inner product ⟨u, v⟩_x."""
        return _tangent_inner(self._cast(u), self._cast(v), self._cast(x), c)

    def tangent_norm(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, ""]:
        """Riemannian norm ||v||_x."""
        return _tangent_norm(self._cast(v), self._cast(x), c)

    def tangent_proj(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Tangent-space projection (identity for PV)."""
        return _tangent_proj(self._cast(v), self._cast(x), c)

    def egrad2rgrad(self, grad: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Float[Array, "dim"]:
        """Convert Euclidean gradient to Riemannian gradient."""
        return _egrad2rgrad(self._cast(grad), self._cast(x), c)

    # -- Validation ----------------------------------------------------------

    def is_in_manifold(self, x: Float[Array, "dim"], c: float, atol: float = 1e-5) -> Array:
        """Check that all entries are finite (PV has no constraint)."""
        return _is_in_manifold(self._cast(x), c, atol)

    def is_in_tangent_space(self, v: Float[Array, "dim"], x: Float[Array, "dim"], c: float) -> Array:
        """Check that v has finite entries (T_x PV = R^n)."""
        return _is_in_tangent_space(self._cast(v), self._cast(x), c)

    def embed_spatial_0(self, v_spatial: Float[Array, "... n"]) -> Float[Array, "... n"]:
        """Identity embedding (no time coordinate). Kept for API parity."""
        return _embed_spatial_0(self._cast(v_spatial))

    # -- Batch helpers -------------------------------------------------------

    def compute_mlr(
        self,
        x: Float[Array, "batch in_dim"],
        z: Float[Array, "out_dim in_dim"],
        r: Float[Array, "out_dim 1"],
        c: float,
        clamping_factor: float,
        smoothing_factor: float,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """PV multinomial logistic regression (paper Thm 5.2)."""
        return _compute_mlr(
            self._cast(x),
            self._cast(z),
            self._cast(r),
            c,
            clamping_factor,
            smoothing_factor,
            min_enorm,
        )
