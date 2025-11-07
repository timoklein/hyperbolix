"""Common utilities for wrapped normal distributions."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def sigma_to_cov(
    sigma: Float[Array, "..."] | float,
    n: int,
    dtype,
) -> Float[Array, "n n"]:
    """Convert sigma parameterization to covariance matrix.

    Args:
        sigma: Covariance parameterization. Can be:
            - Scalar: isotropic covariance σ² I (n × n)
            - 1D array of length n: diagonal covariance diag(σ₁², ..., σₙ²)
            - 2D array (n, n): full covariance matrix (must be SPD)
        n: Spatial dimension
        dtype: Output dtype

    Returns:
        Covariance matrix of shape (n, n)

    Raises:
        ValueError: If sigma dimensions don't match expected dimension n
    """
    sigma_array = jnp.asarray(sigma, dtype=dtype)

    if sigma_array.ndim == 0:  # Scalar -> isotropic covariance σ²I
        return (sigma_array**2) * jnp.eye(n, dtype=dtype)
    elif sigma_array.ndim == 1:  # Vector -> diagonal covariance
        if sigma_array.shape[0] != n:
            raise ValueError(f"Diagonal sigma must have length {n}, got {sigma_array.shape[0]}")
        return jnp.diag(sigma_array**2)
    else:  # Matrix -> full covariance
        if sigma_array.shape[-2:] != (n, n):
            raise ValueError(f"Covariance matrix must be ({n}, {n}), got {sigma_array.shape}")
        return sigma_array


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
    mean = jnp.zeros(n, dtype=dtype or cov.dtype)
    return jax.random.multivariate_normal(key, mean, cov, shape=sample_shape, dtype=dtype)
