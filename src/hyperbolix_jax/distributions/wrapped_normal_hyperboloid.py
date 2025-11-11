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
    c: float,
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
        c: Curvature (positive scalar)
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


def _gaussian_log_prob(
    v_spatial: Float[Array, "... n"],
    sigma: Float[Array, "..."] | float,
    n: int,
    dtype,
) -> Float[Array, "..."]:
    """Compute log probability of zero-mean Gaussian.

    For v ~ N(0, Σ), computes:
    log p(v) = -n/2 * log(2π) - 1/2 * log|Σ| - 1/2 * v^T Σ^(-1) v

    Args:
        v_spatial: Spatial vector(s), shape (..., n)
        sigma: Covariance parameterization (scalar, 1D, or 2D)
        n: Spatial dimension
        dtype: Data type

    Returns:
        Log probability, shape (...)
    """
    import jax.numpy as jnp
    from jax.scipy.stats import multivariate_normal

    # Convert sigma parameterization to covariance matrix
    cov = sigma_to_cov(sigma, n, dtype)

    # Zero mean
    mean = jnp.zeros(n, dtype=dtype)

    # Compute log probability using JAX built-in
    # Cast to Array to satisfy type checker
    return jnp.asarray(multivariate_normal.logpdf(v_spatial, mean, cov))


def _log_det_jacobian(
    v: Float[Array, "... n_plus_1"],
    c: float,
    n: int,
) -> Float[Array, "..."]:
    """Compute log determinant of projection Jacobian.

    Computes log det(∂proj_μ(v)/∂v) = (n-1) * log(sinh(r) / r)
    where r = ||v||_L is the Minkowski norm.

    For numerical stability, uses Taylor expansion for small r:
    log(sinh(r) / r) ≈ r²/6 for r → 0

    Args:
        v: Tangent vector at origin, shape (..., n+1)
        c: Curvature (positive scalar)
        n: Spatial dimension (manifold dimension)

    Returns:
        Log determinant of Jacobian, shape (...)
    """
    import jax.numpy as jnp

    # Compute Minkowski norm: r = sqrt(⟨v, v⟩_L)
    # For tangent vector at origin: v = [0, v_bar], so ⟨v, v⟩_L = -v₀² + ||v_rest||²
    # Since v = [0, v_bar], we have ⟨v, v⟩_L = ||v_bar||² = ||v[..., 1:]||²
    v_spatial = v[..., 1:]  # Extract spatial components
    v_minkowski_sq = jnp.sum(v_spatial**2, axis=-1)  # ||v_bar||²
    r = jnp.sqrt(jnp.maximum(v_minkowski_sq, 1e-15))  # Clip for numerical stability

    # Threshold for switching to Taylor expansion
    r_threshold = 1e-3

    # Standard computation: log(sinh(r) / r) = log(sinh(r)) - log(r)
    log_sinh_r_over_r_standard = jnp.log(jnp.sinh(r)) - jnp.log(r)

    # Taylor expansion for small r: log(sinh(r) / r) ≈ r²/6
    log_sinh_r_over_r_taylor = (r**2) / 6.0

    # Use Taylor expansion when r < r_threshold
    log_sinh_r_over_r = jnp.where(
        r < r_threshold,
        log_sinh_r_over_r_taylor,
        log_sinh_r_over_r_standard
    )

    # log det = (n-1) * log(sinh(r) / r)
    log_det = (n - 1) * log_sinh_r_over_r

    return log_det


def log_prob(
    z: Float[Array, "... n_plus_1"],
    mu: Float[Array, "... n_plus_1"],
    sigma: Float[Array, "..."] | float,
    c: float,
) -> Float[Array, "..."]:
    """Compute log probability of wrapped normal distribution.

    Implements Algorithm 2 from the paper:
    1. Map z to u = exp_μ⁻¹(z) ∈ T_μℍⁿ (logarithmic map)
    2. Move u to v = PT_{μ→μ₀}(u) ∈ T_{μ₀}ℍⁿ (parallel transport to origin)
    3. Calculate log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)

    Args:
        z: Sample point(s) on hyperboloid, shape (..., n+1)
        mu: Mean point on hyperboloid, shape (..., n+1)
        sigma: Covariance parameterization. Can be:
            - Scalar: isotropic covariance sigma^2 I (n x n)
            - 1D array of length n: diagonal covariance diag(sigma_1^2, ..., sigma_n^2)
            - 2D array (n, n): full covariance matrix (must be SPD)
        c: Curvature (positive scalar)

    Returns:
        Log probability, shape (...) (manifold dimension removed)

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from hyperbolix_jax.distributions import wrapped_normal_hyperboloid
        >>>
        >>> # Compute log probability of samples
        >>> key = jax.random.PRNGKey(0)
        >>> mu = jnp.array([1.0, 0.0, 0.0])
        >>> sigma = 0.1
        >>> z = wrapped_normal_hyperboloid.sample(key, mu, sigma, c=1.0)
        >>> log_p = wrapped_normal_hyperboloid.log_prob(z, mu, sigma, c=1.0)
        >>> log_p.shape
        ()
        >>>
        >>> # Batch computation
        >>> z_batch = wrapped_normal_hyperboloid.sample(key, mu, sigma, c=1.0, sample_shape=(10,))
        >>> log_p_batch = wrapped_normal_hyperboloid.log_prob(z_batch, mu, sigma, c=1.0)
        >>> log_p_batch.shape
        (10,)
    """
    # Determine dtype
    dtype = z.dtype

    # Extract spatial dimension
    n = mu.shape[-1] - 1  # Spatial dimension

    # Handle batching with vmap if needed
    # For now, assume z and mu have compatible shapes
    # If z has extra leading dimensions (samples), we need to vmap

    # Step 1: Map z to tangent space at mu using logarithmic map
    # u = log_μ(z) = exp_μ⁻¹(z)
    if z.ndim > mu.ndim:
        # z has sample dimensions, need to vmap over them
        # Figure out how many sample dimensions
        n_sample_dims = z.ndim - mu.ndim

        # Create vmapped version of logmap
        logmap_fn = hyperboloid.logmap
        for _ in range(n_sample_dims):
            logmap_fn = jax.vmap(logmap_fn, in_axes=(0, None, None))

        u = logmap_fn(z, mu, c)
    elif z.ndim == mu.ndim and mu.ndim > 1:
        # Both are batched, vmap over batch dimension
        u = jax.vmap(lambda zz, mm: hyperboloid.logmap(zz, mm, c))(z, mu)
    else:
        # Single point
        u = hyperboloid.logmap(z, mu, c)

    # Step 2: Parallel transport from mu to origin
    # v = PT_{μ→μ₀}(u)
    mu_0 = hyperboloid._create_origin(c, n, dtype)

    if u.ndim > 1:
        # Batched, need to vmap
        if mu.ndim > 1:
            # mu is also batched
            v = jax.vmap(lambda uu, mm: hyperboloid.ptransp(uu, mm, mu_0, c))(u, mu)
        else:
            # Only u is batched (from sample_shape)
            v = jax.vmap(lambda uu: hyperboloid.ptransp(uu, mu, mu_0, c))(u)
    else:
        # Single point
        v = hyperboloid.ptransp(u, mu, mu_0, c)

    # Step 3: Extract spatial components from v (remove temporal component)
    # v = [0, v_bar] at origin, so v_spatial = v[..., 1:]
    v_spatial = v[..., 1:]

    # Step 4: Compute log p(v) where v ~ N(0, Σ)
    log_p_v = _gaussian_log_prob(v_spatial, sigma, n, dtype)

    # Step 5: Compute log det Jacobian
    log_det_jac = _log_det_jacobian(v, c, n)

    # Step 6: Compute final log probability
    # log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)
    log_p_z = log_p_v - log_det_jac

    return log_p_z
