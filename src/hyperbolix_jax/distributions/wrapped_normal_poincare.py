"""Wrapped normal distribution on Poincaré ball.

Simpler implementation than hyperboloid - no parallel transport needed!
Uses exponential map and Möbius addition.

References:
    Mathieu et al. "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
    NeurIPS 2019. https://arxiv.org/abs/1901.06033
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._common import sample_gaussian, sigma_to_cov
from ..manifolds import poincare


def sample(
    key: PRNGKeyArray,
    mu: Float[Array, "... n"],
    sigma: Float[Array, "..."] | float,
    c: Float[Array, "..."] | float,
    sample_shape: tuple[int, ...] = (),
    dtype=None,
) -> Float[Array, "..."]:
    """Sample from wrapped normal distribution on Poincaré ball.

    Simpler than hyperboloid version - no parallel transport needed!

    Algorithm:
    1. Sample v ~ N(0, Σ) ∈ R^n (directly in tangent space, no embedding)
    2. Map to ball at origin: z_0 = exp_0(v)
    3. Move to mean: z = μ ⊕ z_0 (Möbius addition)

    Args:
        key: JAX random key
        mu: Mean point on Poincaré ball, shape (..., n)
        sigma: Covariance parameterization. Can be:
            - Scalar: isotropic covariance σ² I (n × n)
            - 1D array of length n: diagonal covariance diag(σ₁², ..., σₙ²)
            - 2D array (n, n): full covariance matrix (must be SPD)
        c: Curvature (positive scalar or array broadcastable with mu's batch shape)
        sample_shape: Shape of samples to draw, prepended to output. Default: ()
        dtype: Output dtype. Default: infer from mu

    Returns:
        Samples from wrapped normal distribution, shape sample_shape + mu.shape

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from hyperbolix_jax.distributions import wrapped_normal_poincare
        >>>
        >>> # Single sample with isotropic covariance
        >>> key = jax.random.PRNGKey(0)
        >>> mu = jnp.array([0.0, 0.0])  # Origin in Poincaré ball
        >>> sigma = 0.1  # Isotropic
        >>> z = wrapped_normal_poincare.sample(key, mu, sigma, c=1.0)
        >>> z.shape
        (2,)
        >>>
        >>> # Multiple samples with diagonal covariance
        >>> sigma_diag = jnp.array([0.1, 0.2])  # Diagonal
        >>> z = wrapped_normal_poincare.sample(key, mu, sigma_diag, c=1.0, sample_shape=(5,))
        >>> z.shape
        (5, 2)
        >>>
        >>> # Batch of means
        >>> mu_batch = jnp.array([[0.0, 0.0], [0.1, 0.1]])  # (2, 2)
        >>> z = wrapped_normal_poincare.sample(key, mu_batch, 0.1, c=1.0)
        >>> z.shape
        (2, 2)
    """
    # Determine output dtype
    if dtype is None:
        dtype = mu.dtype

    # Extract dimension
    n = mu.shape[-1]  # Dimension of Poincaré ball


    # Step 1: Sample v ~ N(0, Σ) ∈ R^n (directly in tangent space at origin)
    cov = sigma_to_cov(sigma, n, dtype)
    v = sample_gaussian(key, cov, sample_shape=sample_shape, dtype=dtype)

    # Step 2: Map to ball at origin: z_0 = exp_0(v)
    # Step 3: Move to mean: z = μ ⊕ z_0

    # Handle sample_shape and batch dimensions
    if len(sample_shape) > 0:
        # Define the transformation for a single sample
        def transform_single_sample(v_single):
            # Map to ball at origin
            if mu.ndim == 1:
                # Single mean point
                z_0 = poincare.expmap_0(v_single, c)
                # Move to mean using Möbius addition
                z = poincare.addition(mu, z_0, c)
            else:
                # Batched mu, need to vmap
                z_0 = jax.vmap(lambda vs: poincare.expmap_0(vs, c))(v_single)
                z = jax.vmap(lambda m, z0: poincare.addition(m, z0, c))(mu, z_0)

            return z

        # vmap over sample dimensions
        for _ in sample_shape:
            transform_single_sample = jax.vmap(transform_single_sample)

        z = transform_single_sample(v)
    else:
        # No sample_shape, just handle potential batching in mu
        if mu.ndim == 1:
            # Single point
            # Map to ball at origin
            z_0 = poincare.expmap_0(v, c)
            # Move to mean using Möbius addition
            z = poincare.addition(mu, z_0, c)
        else:
            # Batched mu, need to vmap
            z_0 = jax.vmap(lambda vs: poincare.expmap_0(vs, c))(v)
            z = jax.vmap(lambda m, z0: poincare.addition(m, z0, c))(mu, z_0)

    return z
