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
``_addition`` is the one function that is *not* bit-for-bit the historical expression: it regroups the
numerator and clamps on a scalar (see its implementation notes), which changes the last ulps and is
strictly more accurate near the ball boundary.

All operations act on a single point of shape ``(dim,)``; batch with :func:`jax.vmap`. The
``_conformal_factor_batch`` helper is the exception — it broadcasts over arbitrary leading dims for the NN
layers.

Dimension key:
    dim: manifold (ambient == spatial for these models) dimension
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import MIN_NORM, floor_at, safe_norm
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


def _conformal_factor(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, ""]:
    """Conformal factor ``λ_x = 2 / (1 - c‖x‖²)``.

    For ``c > 0`` the denominator → 0 at the ball boundary and is floored with a dtype-eps margin (the
    historical Poincaré behavior); for ``c ≤ 0`` the denominator is ``≥ 1`` and the floor never bites.
    """
    x2 = jnp.dot(x, x, precision=MATMUL_PRECISION)
    max_norm_eps = _get_max_norm_eps(x)
    abs_c = jnp.abs(jnp.asarray(c))
    sqrt_abs_c = jnp.sqrt(floor_at(abs_c, MIN_NORM))
    boundary_floor = 2.0 * sqrt_abs_c * max_norm_eps - abs_c * max_norm_eps**2
    denom = floor_at(1.0 - c * x2, jnp.where(jnp.asarray(c) > 0, boundary_floor, MIN_NORM))
    return 2.0 / denom


def _proj(x: Float[Array, "dim"], c: ScalarCurvature) -> Float[Array, "dim"]:
    """Project onto the manifold. A boundary exists only for ``c > 0`` (``‖x‖ < 1/√c``); for ``c ≤ 0``
    (Euclidean / spherical) the space is all of R^d and this is the identity."""
    # `safe_norm` + `floor_at`. The floor is deliberate: `norm` divides in the *untaken* branch
    # of the `where` too, and 0-cotangent times that branch's inf is NaN. The max-scaling is the
    # fix -- `x` here is by definition unprojected, and `sum(x**2)` overflows float32 above
    # coordinate 1.8e19. That mattered more than anywhere else in the library: with `norm = inf`
    # the clamp `x * (max_norm / inf)` is the ZERO VECTOR, so the farthest representable point was
    # projected onto the origin instead of onto the boundary (measured: ||proj(x)|| = 0.0 at
    # float32 radius 1e20, now 0.99999).
    # The trailing `[..., None]` is what makes the clamp broadcast against `x`: `safe_norm`
    # reduces the last axis, so it must be re-added before the result multiplies a `(..., dim)`
    # operand -- exactly as :func:`_proj_batch` does. For the single point this function is
    # contracted for it is a shape-(1,) scalar and the result is bit-identical either way; it is
    # the (B, dim) inputs that several call sites and tests pass anyway that need it, and they
    # now get the per-row clamp instead of the pre-sweep whole-array Frobenius one.
    norm = floor_at(safe_norm(x)[..., None], MIN_NORM)
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
    # `safe_norm` + `floor_at` over the last axis; see :func:`_proj` for both halves.
    norm = floor_at(safe_norm(x)[..., None], MIN_NORM)  # (..., 1)
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
    xy = jnp.dot(x, y, precision=MATMUL_PRECISION)
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
    denom = floor_at(1 + 2 * c * xy + c**2 * x2 * y2, MIN_NORM)

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

    Curvature-generic simplified closed form; underlies parallel transport.

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    c2 = c**2
    x_sqnorm = jnp.dot(x, x, precision=MATMUL_PRECISION)  # scalar
    y_sqnorm = jnp.dot(y, y, precision=MATMUL_PRECISION)  # scalar
    xy = jnp.dot(x, y, precision=MATMUL_PRECISION)  # scalar
    xz = jnp.dot(x, z, precision=MATMUL_PRECISION)  # scalar
    yz = jnp.dot(y, z, precision=MATMUL_PRECISION)  # scalar

    coeff_x = -c2 * xz * y_sqnorm + c * yz + 2 * c2 * xy * yz  # scalar
    coeff_y = -c2 * yz * x_sqnorm - c * xz  # scalar
    num_D = 2 * (coeff_x * x + coeff_y * y)  # (dim,)
    denom = floor_at(1 + 2 * c * xy + c2 * x_sqnorm * y_sqnorm, MIN_NORM)  # scalar

    return z + num_D / denom


def _conformal_factor_batch(x: Float[Array, "... dim"], c: ScalarCurvature) -> Float[Array, "... 1"]:
    """Conformal factor ``λ_x = 2 / (1 - c‖x‖²)`` over arbitrary leading dims (for the NN layers)."""
    dtype = x.dtype
    c_arr = jnp.asarray(c, dtype=dtype)
    max_norm_eps = jnp.asarray(float(jnp.finfo(dtype).eps ** 0.75), dtype=dtype)
    x2 = jnp.sum(x**2, axis=-1, keepdims=True)  # (..., 1)
    abs_c = jnp.abs(c_arr)
    sqrt_abs_c = jnp.sqrt(floor_at(abs_c, MIN_NORM))
    boundary_floor = 2.0 * sqrt_abs_c * max_norm_eps - abs_c * max_norm_eps**2
    denom = floor_at(jnp.asarray(1.0, dtype=dtype) - c_arr * x2, jnp.where(c_arr > 0, boundary_floor, MIN_NORM))
    return 2.0 / denom
