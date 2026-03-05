"""Common utilities for wrapped normal distributions.

Dimension key:
  N: spatial/manifold dimension
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Float, PRNGKeyArray


def sigma_to_cov(
    sigma: Float[Array, "..."] | float,
    n: int,
    dtype,
) -> Float[Array, "n n"]:
    """Convert sigma parameterization to covariance matrix.

    Args:
        sigma: Covariance parameterization. Can be:
            - Scalar: isotropic covariance sigma^2 I (n x n)
            - 1D array of length n: diagonal covariance diag(sigma_1^2, ..., sigma_n^2)
            - 2D array (n, n): full covariance matrix (must be SPD)
        n: Spatial dimension
        dtype: Output dtype

    Returns:
        Covariance matrix of shape (n, n)

    Raises:
        ValueError: If sigma dimensions don't match expected dimension n
    """
    sigma_arr = jnp.asarray(sigma, dtype=dtype)

    if sigma_arr.ndim == 0:  # Scalar -> isotropic covariance sigma^2 I
        return (sigma_arr**2) * jnp.eye(n, dtype=dtype)
    elif sigma_arr.ndim == 1:  # Vector -> diagonal covariance
        if sigma_arr.shape[0] != n:
            raise ValueError(f"Diagonal sigma must have length {n}, got {sigma_arr.shape[0]}")
        return jnp.diag(sigma_arr**2)
    else:  # Matrix -> full covariance
        if sigma_arr.shape[-2:] != (n, n):
            raise ValueError(f"Covariance matrix must be ({n}, {n}), got {sigma_arr.shape}")
        return sigma_arr


def sample_gaussian(
    key: PRNGKeyArray,
    cov: Float[Array, "n n"],
    sample_shape: tuple[int, ...] = (),
    dtype=None,
) -> Float[Array, "..."]:
    """Sample from zero-mean multivariate normal with given covariance.

    Args:
        key: JAX random key
        cov: Covariance matrix, shape (n, n)
        sample_shape: Shape of samples to draw, prepended to output
        dtype: Output dtype

    Returns:
        Samples from N(0, cov), shape sample_shape + (n,)
    """
    n = cov.shape[0]
    mean_N = jnp.zeros(n, dtype=dtype or cov.dtype)
    return jax.random.multivariate_normal(key, mean_N, cov, shape=sample_shape, dtype=dtype)


def gaussian_log_prob(
    v: Float[Array, "... n"],
    sigma: Float[Array, "..."] | float,
    n: int,
    dtype,
) -> Float[Array, "..."]:
    """Compute log probability of zero-mean Gaussian.

    For v ~ N(0, Sigma), computes:
    log p(v) = -n/2 * log(2pi) - 1/2 * log|Sigma| - 1/2 * v^T Sigma^(-1) v

    Args:
        v: Vector(s) in tangent space, shape (..., n)
        sigma: Covariance parameterization (scalar, 1D, or 2D)
        n: Dimension
        dtype: Data type

    Returns:
        Log probability, shape (...)
    """
    cov_DD = sigma_to_cov(sigma, n, dtype)
    mean_D = jnp.zeros(n, dtype=dtype)
    return jnp.asarray(multivariate_normal.logpdf(v, mean_D, cov_DD))
