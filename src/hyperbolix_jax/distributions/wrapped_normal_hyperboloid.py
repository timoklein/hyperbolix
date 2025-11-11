"""Wrapped normal distribution on hyperboloid manifold.

Implementation of the wrapped normal distribution that wraps a Gaussian from the
tangent space at the origin onto the hyperboloid via parallel transport and exponential map.

References:
    Mathieu et al. "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
    NeurIPS 2019. https://arxiv.org/abs/1901.06033
"""

import jax
from jaxtyping import Array, Float, PRNGKeyArray

from ._common import sample_gaussian, sigma_to_cov
from ..manifolds import hyperboloid


def sample(
    key: PRNGKeyArray,
    mu: Float[Array, "... n_plus_1"],
    sigma: Float[Array, "..."] | float,
    c: Float[Array, "..."] | float,
    sample_shape: tuple[int, ...] = (),
    dtype=None,
) -> Float[Array, "..."]:
    """Sample from wrapped normal distribution on hyperboloid.

    Implements Algorithm 1 from the paper:
    1. Sample v_bar ~ N(0, Σ) ∈ R^n
    2. Embed as tangent vector v = [0, v_bar] ∈ T_{μ₀}ℍⁿ at origin
    3. Parallel transport to mean: u = PT_{μ₀→μ}(v) ∈ T_μℍⁿ
    4. Map to manifold: z = exp_μ(u) ∈ ℍⁿ

    Args:
        key: JAX random key
        mu: Mean point on hyperboloid, shape (..., n+1) in ambient coordinates
        sigma: Covariance parameterization in spatial coordinates. Can be:
            - Scalar: isotropic covariance sigma^2 I (n x n)
            - 1D array of length n: diagonal covariance diag(sigma_1^2, ..., sigma_n^2)
            - 2D array (n, n): full covariance matrix (must be SPD)
        c: Curvature (positive scalar or array broadcastable with mu's batch shape)
        sample_shape: Shape of samples to draw, prepended to output. Default: ()
        dtype: Output dtype. Default: infer from mu

    Returns:
        Samples from wrapped normal distribution, shape sample_shape + mu.shape

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from hyperbolix_jax.distributions import wrapped_normal_hyperboloid
        >>>
        >>> # Single sample with isotropic covariance
        >>> key = jax.random.PRNGKey(0)
        >>> mu = jnp.array([1.0, 0.0, 0.0])  # Origin in H^2
        >>> sigma = 0.1  # Isotropic
        >>> z = wrapped_normal_hyperboloid.sample(key, mu, sigma, c=1.0)
        >>> z.shape
        (3,)
        >>>
        >>> # Multiple samples with diagonal covariance
        >>> sigma_diag = jnp.array([0.1, 0.2])  # Diagonal
        >>> z = wrapped_normal_hyperboloid.sample(key, mu, sigma_diag, c=1.0, sample_shape=(5,))
        >>> z.shape
        (5, 3)
        >>>
        >>> # Batch of means
        >>> mu_batch = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.1]])  # (2, 3)
        >>> z = wrapped_normal_hyperboloid.sample(key, mu_batch, 0.1, c=1.0)
        >>> z.shape
        (2, 3)
    """
    # Determine output dtype
    if dtype is None:
        dtype = mu.dtype

    # Extract spatial dimension
    n = mu.shape[-1] - 1  # Spatial dimension (n for H^n in R^(n+1))


    # Step 1: Sample v_bar ~ N(0, Σ) ∈ R^n
    cov = sigma_to_cov(sigma, n, dtype)
    v_spatial = sample_gaussian(key, cov, sample_shape=sample_shape, dtype=dtype)

    # Step 2: Embed as tangent vector v = [0, v_bar] ∈ T_{μ₀}ℍⁿ at origin
    v = hyperboloid.embed_spatial_0(v_spatial)

    # Handle sample_shape and batch dimensions for parallel transport and expmap
    # v has shape: sample_shape + (possibly batch from sigma) + (n+1)
    # mu has shape: (possibly batch) + (n+1)
    # We need to broadcast and apply operations element-wise

    # If we have sample_shape, we need to vmap over it
    if len(sample_shape) > 0:
        # Define the transformation for a single sample
        def transform_single_sample(v_single):
            # v_single has shape: batch_shape + (n+1)
            # mu has shape: batch_shape + (n+1)

            # Step 3: Parallel transport from origin to mu
            # ptransp_0 expects single point, so we may need to vmap over batch dims
            if mu.ndim == 1:
                # Single point, no batching
                u = hyperboloid.ptransp_0(v_single, mu, c)
                # Step 4: Exponential map at mu
                z = hyperboloid.expmap(u, mu, c)
            else:
                # Batched mu, need to vmap
                # Broadcast v_single to match mu's batch shape if needed
                u = jax.vmap(lambda m, vs: hyperboloid.ptransp_0(vs, m, c))(mu, v_single)
                z = jax.vmap(lambda uu, m: hyperboloid.expmap(uu, m, c))(u, mu)

            return z

        # vmap over sample dimensions
        for _ in sample_shape:
            transform_single_sample = jax.vmap(transform_single_sample)

        z = transform_single_sample(v)
    else:
        # No sample_shape, just handle potential batching in mu
        if mu.ndim == 1:
            # Single point
            # Step 3: Parallel transport from origin to mu
            u = hyperboloid.ptransp_0(v, mu, c)
            # Step 4: Exponential map at mu
            z = hyperboloid.expmap(u, mu, c)
        else:
            # Batched mu, need to vmap
            u = jax.vmap(lambda m, vs: hyperboloid.ptransp_0(vs, m, c))(mu, v)
            z = jax.vmap(lambda uu, m: hyperboloid.expmap(uu, m, c))(u, mu)

    return z
