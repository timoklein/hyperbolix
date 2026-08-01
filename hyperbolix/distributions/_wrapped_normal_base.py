"""Shared utilities for wrapped normal distributions on Poincaré ball and hyperboloid.

Contains the common Gaussian log-probability, log-det-Jacobian formula,
and batched vmap transform logic used by both distribution implementations.

Dimension key:
  S: sample dimensions (from sample_shape)
  B: batch dimensions (from mu batch shape)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# Taylor gate for ``log(sinh(x)/x)``: the switch point is set where the two branches' error
# curves cross, which depends on the working precision, so it is derived from the dtype's eps
# rather than fixed.
#
#   Taylor truncation (2 terms kept)  ~ x^6 / 2835      (first dropped series term)
#   direct-branch cancellation        ~ eps * |log x|   (two O(|log x|) terms differencing to
#                                                        an O(x^2/6) result)
#
# Balancing them gives a threshold of the form (K * eps)**(1/6). The constant was fitted by
# sweeping x over [1e-6, 0.6] against a 60-digit Decimal oracle (value *and* d/dr, both dtypes)
# and picking the K that minimises the worst relative error on either side of the gate:
# K = 1e3 lands at x = 0.222 (float32) and x = 7.8e-3 (float64), with worst-case relative error
# 2.1e-5 / 1.5e-5 (value / grad) in float32 and 3.2e-11 / 2.1e-11 in float64. Shrinking K walks
# the gate into the direct branch's cancellation zone; growing it walks into Taylor truncation.
_TAYLOR_GATE_EPS_FACTOR = 1.0e3


def _taylor_gate_threshold(dtype: jnp.dtype) -> float:
    """Return the x = sqrt(c)*r below which the Taylor branch beats the direct formula.

    Args:
        dtype: Floating dtype the log-det is being evaluated in.

    Returns:
        Python float threshold (static — safe to compare against a traced array under jit).
    """
    return float(_TAYLOR_GATE_EPS_FACTOR * jnp.finfo(dtype).eps) ** (1.0 / 6.0)


def _log_det_jacobian_from_r(
    r: Float[Array, "..."],
    c: float,
    n: int,
) -> Float[Array, "..."]:
    """Compute log determinant of projection Jacobian given Riemannian norm r.

    Formula: log det = (n-1) * log(sinh(sqrt(c)*r) / (sqrt(c)*r))

    For small x = sqrt(c)*r the direct form ``log(sinh(x)) - log(x)`` differences two
    O(|log x|) terms down to an O(x^2/6) result and loses every significant digit, so a
    two-term Taylor expansion takes over below a dtype-dependent threshold:
        log(sinh(x)/x) = x^2/6 - x^4/180 + O(x^6)

    Args:
        r: Riemannian norm of tangent vector, shape (...)
        c: Curvature (positive scalar)
        n: Spatial/manifold dimension

    Returns:
        Log determinant of Jacobian, shape (...)
    """
    sqrt_c = jnp.sqrt(c)
    sqrt_c_r = sqrt_c * r

    # Static in dtype, so the comparison against the traced array stays jit-friendly.
    threshold = _taylor_gate_threshold(sqrt_c_r.dtype)
    small = sqrt_c_r < threshold

    # Standard: log(sinh(x)/x) = log(sinh(x)) - log(x).
    # Double-where: at r = 0 the standard branch is log(0) - log(0) = NaN, and reverse-mode AD
    # propagates that NaN through the *unselected* side of a single jnp.where. Feeding the
    # standard branch a dummy x = 1 wherever the Taylor branch wins keeps the values identical
    # (the branch is discarded there) while making the gradient finite.
    sqrt_c_r_safe = jnp.where(small, jnp.ones_like(sqrt_c_r), sqrt_c_r)
    log_ratio_standard = jnp.log(jnp.sinh(sqrt_c_r_safe)) - jnp.log(sqrt_c_r_safe)

    # Taylor: log(sinh(x)/x) = x^2/6 - x^4/180 + O(x^6). A plain polynomial in x^2, so this
    # branch is NaN-free for every finite r and its gradient is exactly 0 at r = 0.
    x2 = sqrt_c_r * sqrt_c_r
    log_ratio_taylor = x2 * (1.0 / 6.0 - x2 / 180.0)

    log_ratio = jnp.where(small, log_ratio_taylor, log_ratio_standard)

    return (n - 1) * log_ratio


def _vmap_sample_and_batch(fn, n_sample_dims: int, n_batch_dims: int):
    """Lift a single-point ``fn(x_dim, mu_dim)`` to inputs shaped ``(*S, *B, dim)`` / ``(*B, dim)``.

    The mean's own batch axes ``B`` pair elementwise with the trailing axes of ``x``; the
    leading sample axes ``S`` broadcast the mean. Building the vmaps inside-out (batch first,
    then samples) is what makes the outermost vmap correspond to the outermost axis — the same
    order :func:`_batched_transform` uses on the sampling side.

    Args:
        fn: Single-point function ``(x_dim, mu_dim) -> out``.
        n_sample_dims: Number of leading sample axes S (mean broadcast over these).
        n_batch_dims: Number of mean batch axes B (paired).

    Returns:
        A callable ``(x, mu) -> out`` accepting the batched shapes.
    """
    for _ in range(n_batch_dims):
        fn = jax.vmap(fn)
    for _ in range(n_sample_dims):
        fn = jax.vmap(fn, in_axes=(0, None))
    return fn


def _batched_transform(
    transform_single,
    v: Float[Array, "..."],
    mu: Float[Array, "..."],
    sample_shape: tuple[int, ...],
    mu_batch_shape: tuple[int, ...],
) -> Float[Array, "..."]:
    """Apply transform_single(v, mu) with appropriate vmap batching.

    Handles three cases:
    - No batching: direct call
    - Batch mu only: vmap over batch dims
    - Samples + batch: vmap over batch dims, then sample dims (broadcast mu)

    Args:
        transform_single: Function (v_single, mu_single) -> z_single
        v: Tangent vectors, shape (*S, *B, dim)
        mu: Mean points, shape (*B, dim)
        sample_shape: Sample dimensions S
        mu_batch_shape: Batch dimensions B from mu

    Returns:
        Transformed points, shape (*S, *B, dim)
    """
    if len(sample_shape) == 0 and len(mu_batch_shape) == 0:
        return transform_single(v, mu)

    if len(sample_shape) == 0:
        vmapped_fn = transform_single
        for _ in mu_batch_shape:
            vmapped_fn = jax.vmap(vmapped_fn)
        return vmapped_fn(v, mu)

    # sample_shape > 0: vmap over batch dims (both), then sample dims (v only)
    vmapped_fn = transform_single
    for _ in mu_batch_shape:
        vmapped_fn = jax.vmap(vmapped_fn)
    for _ in sample_shape:
        vmapped_fn = jax.vmap(vmapped_fn, in_axes=(0, None))
    return vmapped_fn(v, mu)
