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


def _log_det_jacobian_from_r(
    r: Float[Array, "..."],
    c: float,
    n: int,
) -> Float[Array, "..."]:
    """Compute log determinant of projection Jacobian given Riemannian norm r.

    Formula: log det = (n-1) * log(sinh(sqrt(c)*r) / (sqrt(c)*r))

    Uses Taylor expansion for small sqrt(c)*r to avoid 0/0:
        log(sinh(x)/x) ~ x^2/6  for x -> 0

    Args:
        r: Riemannian norm of tangent vector, shape (...)
        c: Curvature (positive scalar)
        n: Spatial/manifold dimension

    Returns:
        Log determinant of Jacobian, shape (...)
    """
    sqrt_c = jnp.sqrt(c)
    sqrt_c_r = sqrt_c * r

    threshold = 1e-3
    small = sqrt_c_r < threshold

    # Standard: log(sinh(x)/x) = log(sinh(x)) - log(x).
    # Double-where: at r = 0 the standard branch is log(0) - log(0) = NaN, and reverse-mode AD
    # propagates that NaN through the *unselected* side of a single jnp.where. Feeding the
    # standard branch a dummy x = 1 wherever the Taylor branch wins keeps the values identical
    # (the branch is discarded there) while making the gradient finite.
    sqrt_c_r_safe = jnp.where(small, jnp.ones_like(sqrt_c_r), sqrt_c_r)
    log_ratio_standard = jnp.log(jnp.sinh(sqrt_c_r_safe)) - jnp.log(sqrt_c_r_safe)

    # Taylor: log(sinh(x)/x) ~ x^2/6
    log_ratio_taylor = (c * r**2) / 6.0

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
