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

    # Standard: log(sinh(x)/x) = log(sinh(x)) - log(x)
    log_ratio_standard = jnp.log(jnp.sinh(sqrt_c_r)) - jnp.log(sqrt_c_r)

    # Taylor: log(sinh(x)/x) ~ x^2/6
    log_ratio_taylor = (c * r**2) / 6.0

    log_ratio = jnp.where(sqrt_c_r < threshold, log_ratio_taylor, log_ratio_standard)

    return (n - 1) * log_ratio


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
