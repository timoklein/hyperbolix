"""Wrapped normal distribution on hyperboloid manifold.

Implementation of the wrapped normal distribution that wraps a Gaussian from the
tangent space at the origin onto the manifold via parallel transport and exponential map.

References:
    Mathieu et al. "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
    NeurIPS 2019. https://arxiv.org/abs/1901.06033
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..manifolds import hyperboloid


def _sample_gaussian(
    key: PRNGKeyArray,
    sigma: Float[Array, "..."] | float,
    n: int,
    sample_shape: tuple[int, ...] = (),
    dtype=None,
) -> Float[Array, "..."]:
    """Sample from Gaussian in spatial coordinates with flexible covariance.

    Args:
        key: JAX random key
        sigma: Covariance parameterization. Can be:
            - Scalar: isotropic covariance σ² I
            - 1D array of length n: diagonal covariance diag(σ₁², ..., σₙ²)
            - 2D array (n, n): full covariance matrix (must be SPD)
        n: Spatial dimension (number of coordinates, excluding temporal)
        sample_shape: Shape of samples to draw, prepended to output
        dtype: Output dtype (default: infer from sigma or use float32)

    Returns:
        Gaussian samples v_bar ~ N(0, Σ) of shape sample_shape + batch_shape + (n,)
        where batch_shape comes from broadcasting sigma's batch dimensions
    """
    # Determine dtype
    if dtype is None:
        if isinstance(sigma, (int, float)):
            dtype = jnp.float32
        else:
            dtype = sigma.dtype

    # Handle different covariance parameterizations
    sigma_array = jnp.asarray(sigma, dtype=dtype)

    # Isotropic: scalar -> σ² I
    if sigma_array.ndim == 0 or (sigma_array.ndim == 1 and sigma_array.shape[0] == 1):
        scalar_sigma = sigma_array.item() if sigma_array.ndim == 1 else sigma_array
        cov = jnp.eye(n, dtype=dtype) * (scalar_sigma**2)
        mean = jnp.zeros(n, dtype=dtype)
        return jax.random.multivariate_normal(key, mean, cov, shape=sample_shape, dtype=dtype)

    # Diagonal: 1D vector -> diag(σ₁², ..., σₙ²)
    elif sigma_array.ndim == 1:
        # sigma_array has shape (..., n) potentially with batch dims
        if sigma_array.shape[-1] != n:
            raise ValueError(f"Diagonal sigma must have spatial dimension {n}, got {sigma_array.shape[-1]}")

        # For diagonal covariance with batching, we need to handle each batch separately
        # Extract batch shape
        batch_shape = sigma_array.shape[:-1]

        if len(batch_shape) == 0:
            # No batch dimensions
            cov = jnp.diag(sigma_array**2)
            mean = jnp.zeros(n, dtype=dtype)
            return jax.random.multivariate_normal(key, mean, cov, shape=sample_shape, dtype=dtype)
        else:
            # With batch dimensions, need to vmap
            def sample_single(key_i, sigma_i):
                cov_i = jnp.diag(sigma_i**2)
                mean_i = jnp.zeros(n, dtype=dtype)
                return jax.random.multivariate_normal(key_i, mean_i, cov_i, shape=sample_shape, dtype=dtype)

            # Split keys for each batch element
            n_batch = int(jnp.prod(jnp.array(batch_shape)))
            keys = jax.random.split(key, n_batch)
            sigma_flat = sigma_array.reshape(n_batch, n)

            # vmap over batch dimension
            samples = jax.vmap(sample_single)(keys, sigma_flat)
            # Reshape to sample_shape + batch_shape + (n,)
            return samples.reshape(sample_shape + batch_shape + (n,))

    # Full: 2D matrix -> use directly as covariance
    elif sigma_array.ndim == 2:
        if sigma_array.shape[-2:] != (n, n):
            raise ValueError(f"Full covariance must be ({n}, {n}), got {sigma_array.shape[-2:]}")

        mean = jnp.zeros(n, dtype=dtype)
        return jax.random.multivariate_normal(key, mean, sigma_array, shape=sample_shape, dtype=dtype)

    # Full with batch: 3D or higher
    else:
        # sigma_array has shape (..., n, n) with batch dimensions
        if sigma_array.shape[-2:] != (n, n):
            raise ValueError(f"Full covariance must end with ({n}, {n}), got {sigma_array.shape[-2:]}")

        batch_shape = sigma_array.shape[:-2]

        def sample_single(key_i, sigma_i):
            mean_i = jnp.zeros(n, dtype=dtype)
            return jax.random.multivariate_normal(key_i, mean_i, sigma_i, shape=sample_shape, dtype=dtype)

        # Split keys for each batch element
        n_batch = int(jnp.prod(jnp.array(batch_shape)))
        keys = jax.random.split(key, n_batch)
        sigma_flat = sigma_array.reshape(n_batch, n, n)

        # vmap over batch dimension
        samples = jax.vmap(sample_single)(keys, sigma_flat)
        # Reshape to sample_shape + batch_shape + (n,)
        return samples.reshape(sample_shape + batch_shape + (n,))


def _embed_in_tangent_space(v_spatial: Float[Array, "... n"]) -> Float[Array, "... n_plus_1"]:
    """Embed spatial vector in tangent space at origin by prepending zero.

    Creates tangent vector v = [0, v_bar] ∈ T_{μ₀}ℍⁿ from spatial sample v_bar ∈ ℝⁿ.

    Args:
        v_spatial: Spatial sample(s), shape (..., n)

    Returns:
        Tangent vector(s) at origin, shape (..., n+1)
    """
    # Prepend zero for temporal component
    zeros = jnp.zeros(v_spatial.shape[:-1] + (1,), dtype=v_spatial.dtype)
    return jnp.concatenate([zeros, v_spatial], axis=-1)


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
        >>> from hyperbolix_jax.distributions import wrapped_normal
        >>>
        >>> # Single sample with isotropic covariance
        >>> key = jax.random.PRNGKey(0)
        >>> mu = jnp.array([1.0, 0.0, 0.0])  # Origin in H^2
        >>> sigma = 0.1  # Isotropic
        >>> z = wrapped_normal.sample(key, mu, sigma, c=1.0)
        >>> z.shape
        (3,)
        >>>
        >>> # Multiple samples with diagonal covariance
        >>> sigma_diag = jnp.array([0.1, 0.2])  # Diagonal
        >>> z = wrapped_normal.sample(key, mu, sigma_diag, c=1.0, sample_shape=(5,))
        >>> z.shape
        (5, 3)
        >>>
        >>> # Batch of means
        >>> mu_batch = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.1]])  # (2, 3)
        >>> z = wrapped_normal.sample(key, mu_batch, 0.1, c=1.0)
        >>> z.shape
        (2, 3)
    """
    # Determine output dtype
    if dtype is None:
        dtype = mu.dtype

    # Extract spatial dimension
    n = mu.shape[-1] - 1  # Spatial dimension (n for H^n in R^(n+1))

    # Ensure curvature is array for consistency
    c_array = jnp.asarray(c, dtype=dtype)

    # Step 1: Sample v_bar ~ N(0, Σ) ∈ R^n
    v_spatial = _sample_gaussian(key, sigma, n, sample_shape=sample_shape, dtype=dtype)

    # Step 2: Embed as tangent vector v = [0, v_bar] ∈ T_{μ₀}ℍⁿ
    v = _embed_in_tangent_space(v_spatial)

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
                u = hyperboloid.ptransp_0(v_single, mu, c_array)
                # Step 4: Exponential map at mu
                z = hyperboloid.expmap(u, mu, c_array)
            else:
                # Batched mu, need to vmap
                # Broadcast v_single to match mu's batch shape if needed
                u = jax.vmap(lambda m, vs: hyperboloid.ptransp_0(vs, m, c_array))(mu, v_single)
                z = jax.vmap(lambda uu, m: hyperboloid.expmap(uu, m, c_array))(u, mu)

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
            u = hyperboloid.ptransp_0(v, mu, c_array)
            # Step 4: Exponential map at mu
            z = hyperboloid.expmap(u, mu, c_array)
        else:
            # Batched mu, need to vmap
            u = jax.vmap(lambda m, vs: hyperboloid.ptransp_0(vs, m, c_array))(mu, v)
            z = jax.vmap(lambda uu, m: hyperboloid.expmap(uu, m, c_array))(u, mu)

    return z
