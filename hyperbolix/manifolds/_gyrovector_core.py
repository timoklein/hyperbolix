"""Shared gyrovector core — single source of truth for the curvature-generic Möbius algebra.

These operations are **sign-agnostic**: the same formulas serve the Poincaré ball (``c > 0``, hyperbolic)
and the κ-stereographic model (signed ``c`` — hyperbolic / Euclidean / spherical). Both
:mod:`hyperbolix.manifolds.poincare` and :mod:`hyperbolix.manifolds.stereographic` import from here so a
stability fix lands in exactly one place instead of silently diverging between two hand-mirrored copies.

For ``c > 0`` every function reproduces the historical Poincaré implementation **bit-for-bit**:
``_addition`` / ``_gyration`` are curvature-generic (identical formula at any ``c``), and
``_conformal_factor`` / ``_conformal_factor_batch`` / ``_proj`` carry a ``jnp.where(c > 0, …)`` /
``abs(c)`` generalization whose ``c > 0`` branch is exactly the Poincaré expression. The only theoretical
departure is a ``√|c|`` floor at ``√MIN_NORM``: for ``0 < c < 1e-15`` (a Poincaré ball of radius
``1/√c > 3e7`` — never used) the boundary floor is marginally more conservative than a bare ``√c``.

All operations act on a single point of shape ``(dim,)``; batch with :func:`jax.vmap`. The
``_conformal_factor_batch`` helper is the exception — it broadcasts over arbitrary leading dims for the NN
layers.

Dimension key:
    dim: manifold (ambient == spatial for these models) dimension
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from .protocol import Curvature

# Smallest safe denominator / squared-norm floor shared across the manifold modules.
MIN_NORM = 1e-15


def _get_max_norm_eps(x: Float[Array, "dim"]) -> float:
    """Maximum-norm epsilon for the array's dtype (``eps**0.75`` — empirically stable, scales with precision)."""
    return float(jnp.finfo(x.dtype).eps ** 0.75)


def _conformal_factor(x: Float[Array, "dim"], c: Curvature) -> Float[Array, ""]:
    """Conformal factor ``λ_x = 2 / (1 - c‖x‖²)``.

    For ``c > 0`` the denominator → 0 at the ball boundary and is floored with a dtype-eps margin (the
    historical Poincaré behavior); for ``c ≤ 0`` the denominator is ``≥ 1`` and the floor never bites.
    """
    x2 = jnp.dot(x, x)
    max_norm_eps = _get_max_norm_eps(x)
    abs_c = jnp.abs(jnp.asarray(c))
    sqrt_abs_c = jnp.sqrt(jnp.maximum(abs_c, MIN_NORM))
    boundary_floor = 2.0 * sqrt_abs_c * max_norm_eps - abs_c * max_norm_eps**2
    denom = jnp.maximum(1.0 - c * x2, jnp.where(jnp.asarray(c) > 0, boundary_floor, MIN_NORM))
    return 2.0 / denom


def _proj(x: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
    """Project onto the manifold. A boundary exists only for ``c > 0`` (``‖x‖ < 1/√c``); for ``c ≤ 0``
    (Euclidean / spherical) the space is all of R^d and this is the identity."""
    max_norm_eps = _get_max_norm_eps(x)
    # Safe norm: sqrt(||x||² + eps²) avoids NaN gradients at x=0.
    norm = jnp.sqrt(jnp.sum(x**2) + MIN_NORM**2)
    sqrt_abs_c = jnp.sqrt(jnp.maximum(jnp.abs(jnp.asarray(c)), MIN_NORM))
    max_norm = jnp.where(jnp.asarray(c) > 0, (1.0 / sqrt_abs_c) - max_norm_eps, jnp.asarray(1e15, dtype=x.dtype))
    cond = norm > max_norm
    return jnp.where(cond, x * (max_norm / norm), x)


def _addition(x: Float[Array, "dim"], y: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
    """Möbius gyrovector addition ``x ⊕ y`` (curvature-generic; non-commutative, non-associative).

    Result is projected back onto the manifold (a no-op for ``c ≤ 0``).

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    x2 = jnp.dot(x, x)
    y2 = jnp.dot(y, y)
    xy = jnp.dot(x, y)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = jnp.maximum(1 + 2 * c * xy + c**2 * x2 * y2, MIN_NORM)
    return _proj(num / denom, c)


def _gyration(x: Float[Array, "dim"], y: Float[Array, "dim"], z: Float[Array, "dim"], c: Curvature) -> Float[Array, "dim"]:
    """Gyration ``gyr[x, y]z`` — restores the (broken) commutativity/associativity of ``⊕``.

    Curvature-generic simplified closed form; underlies parallel transport.

    References:
        Ungar. "A gyrovector space approach to hyperbolic geometry." 2022.
    """
    c2 = c**2
    x_sqnorm = jnp.dot(x, x)  # scalar
    y_sqnorm = jnp.dot(y, y)  # scalar
    xy = jnp.dot(x, y)  # scalar
    xz = jnp.dot(x, z)  # scalar
    yz = jnp.dot(y, z)  # scalar

    coeff_x = -c2 * xz * y_sqnorm + c * yz + 2 * c2 * xy * yz  # scalar
    coeff_y = -c2 * yz * x_sqnorm - c * xz  # scalar
    num_D = 2 * (coeff_x * x + coeff_y * y)  # (dim,)
    denom = jnp.maximum(1 + 2 * c * xy + c2 * x_sqnorm * y_sqnorm, MIN_NORM)  # scalar

    return z + num_D / denom


def _conformal_factor_batch(x: Float[Array, "... dim"], c: Curvature) -> Float[Array, "... 1"]:
    """Conformal factor ``λ_x = 2 / (1 - c‖x‖²)`` over arbitrary leading dims (for the NN layers)."""
    dtype = x.dtype
    c_arr = jnp.asarray(c, dtype=dtype)
    max_norm_eps = jnp.asarray(float(jnp.finfo(dtype).eps ** 0.75), dtype=dtype)
    x2 = jnp.sum(x**2, axis=-1, keepdims=True)  # (..., 1)
    abs_c = jnp.abs(c_arr)
    sqrt_abs_c = jnp.sqrt(jnp.maximum(abs_c, MIN_NORM))
    boundary_floor = 2.0 * sqrt_abs_c * max_norm_eps - abs_c * max_norm_eps**2
    denom = jnp.maximum(jnp.asarray(1.0, dtype=dtype) - c_arr * x2, jnp.where(c_arr > 0, boundary_floor, MIN_NORM))
    return 2.0 / denom
