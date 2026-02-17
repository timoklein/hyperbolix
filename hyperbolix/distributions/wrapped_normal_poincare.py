"""Wrapped normal distribution on Poincaré ball.

Simpler implementation than hyperboloid - no parallel transport needed!
Uses exponential map and Möbius addition.

References:
    Mathieu et al. "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
    NeurIPS 2019. https://arxiv.org/abs/1901.06033
"""

import jax
from jaxtyping import Array, Float, PRNGKeyArray

from ._common import sample_gaussian, sigma_to_cov


def sample(
    key: PRNGKeyArray,
    mu: Float[Array, "... n"],
    sigma: Float[Array, "..."] | float,
    c: float,
    sample_shape: tuple[int, ...] = (),
    dtype=None,
    manifold_module=None,
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
        >>> from hyperbolix.distributions import wrapped_normal_poincare
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
    # Use provided manifold module or default class instance
    if manifold_module is not None:
        manifold = manifold_module
    else:
        from ..manifolds.poincare import Poincare

        _dtype = dtype if dtype is not None else mu.dtype
        manifold = Poincare(dtype=_dtype)

    # Determine output dtype
    if dtype is None:
        dtype = mu.dtype

    # Extract dimension
    n = mu.shape[-1]  # Dimension of Poincaré ball

    # Determine batch shape from mu (all dims except the last one)
    # mu.shape = batch_shape + (n,)
    mu_batch_shape = mu.shape[:-1]

    # Step 1: Sample v ~ N(0, Σ) ∈ R^n (directly in tangent space at origin)
    # Need to sample enough noise vectors for sample_shape AND batch dimensions of mu
    # Full noise shape: sample_shape + mu_batch_shape + (n,)
    cov = sigma_to_cov(sigma, n, dtype)
    full_sample_shape = sample_shape + mu_batch_shape
    v = sample_gaussian(key, cov, sample_shape=full_sample_shape, dtype=dtype)

    # Step 2: Map to ball at origin: z_0 = exp_0(v)
    # Step 3: Move to mean: z = μ ⊕ z_0

    def transform_single(v_single, mu_single):
        """Transform a single (v, mu) pair."""
        # Map to ball at origin
        z_0 = manifold.expmap_0(v_single, c)
        # Move to mean using Möbius addition
        z = manifold.addition(mu_single, z_0, c)
        return z

    if len(sample_shape) == 0 and len(mu_batch_shape) == 0:
        # Single point, no batching
        z = transform_single(v, mu)
    elif len(sample_shape) == 0:
        # No sample_shape but batched mu
        # v has shape mu_batch_shape + (n,), mu has shape mu_batch_shape + (n,)
        # vmap over all batch dimensions
        vmapped_fn = transform_single
        for _ in mu_batch_shape:
            vmapped_fn = jax.vmap(vmapped_fn)
        z = vmapped_fn(v, mu)
    else:
        # Have sample_shape (and possibly batched mu)
        # v has shape sample_shape + mu_batch_shape + (n,)
        # mu has shape mu_batch_shape + (n,)
        # Need to vmap over sample_shape dims (broadcasting mu) then over batch dims

        # First, vmap over mu_batch_shape dimensions (both v and mu have these)
        vmapped_fn = transform_single
        for _ in mu_batch_shape:
            vmapped_fn = jax.vmap(vmapped_fn)

        # Then vmap over sample_shape dimensions (only v has these, mu is broadcast)
        for _ in sample_shape:
            vmapped_fn = jax.vmap(vmapped_fn, in_axes=(0, None))

        z = vmapped_fn(v, mu)

    return z


def _gaussian_log_prob(
    v: Float[Array, "... n"],
    sigma: Float[Array, "..."] | float,
    n: int,
    dtype,
) -> Float[Array, "..."]:
    """Compute log probability of zero-mean Gaussian.

    For v ~ N(0, Σ), computes:
    log p(v) = -n/2 * log(2π) - 1/2 * log|Σ| - 1/2 * v^T Σ^(-1) v

    Args:
        v: Vector(s) in tangent space, shape (..., n)
        sigma: Covariance parameterization (scalar, 1D, or 2D)
        n: Dimension
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
    return jnp.asarray(multivariate_normal.logpdf(v, mean, cov))


def _log_det_jacobian(
    v: Float[Array, "... n"],
    c: float,
    n: int,
) -> Float[Array, "..."]:
    """Compute log determinant of projection Jacobian for Poincaré ball.

    Computes log det(∂proj_μ(v)/∂v) = (n-1) * log(sinh(√c·r) / (√c·r))
    where r = ||v|| is the Euclidean norm of the tangent vector at origin.

    For numerical stability, uses Taylor expansion for small √c·r:
    log(sinh(√c·r) / (√c·r)) ≈ (c·r²)/6 for √c·r → 0

    Args:
        v: Tangent vector at origin, shape (..., n)
        c: Curvature (positive scalar)
        n: Dimension of Poincaré ball

    Returns:
        Log determinant of Jacobian, shape (...)
    """
    import jax.numpy as jnp

    # Compute Euclidean norm of tangent vector
    r = jnp.sqrt(jnp.sum(v**2, axis=-1))

    # Scale by √c
    sqrt_c = jnp.sqrt(c)
    sqrt_c_r = sqrt_c * r

    # Threshold for switching to Taylor expansion
    threshold = 1e-3

    # Standard computation: log(sinh(√c·r) / (√c·r)) = log(sinh(√c·r)) - log(√c·r)
    log_sinh_over_arg_standard = jnp.log(jnp.sinh(sqrt_c_r)) - jnp.log(sqrt_c_r)

    # Taylor expansion for small √c·r: log(sinh(x) / x) ≈ x²/6
    # Here x = √c·r, so x² = c·r²
    log_sinh_over_arg_taylor = (c * r**2) / 6.0

    # Use Taylor expansion when √c·r < threshold
    log_sinh_over_arg = jnp.where(sqrt_c_r < threshold, log_sinh_over_arg_taylor, log_sinh_over_arg_standard)

    # log det = (n-1) * log(sinh(√c·r) / (√c·r))
    log_det = (n - 1) * log_sinh_over_arg

    return log_det


def log_prob(
    z: Float[Array, "... n"],
    mu: Float[Array, "... n"],
    sigma: Float[Array, "..."] | float,
    c: float,
    manifold_module=None,
) -> Float[Array, "..."]:
    """Compute log probability of wrapped normal distribution on Poincaré ball.

    Implements Algorithm 2 from the paper adapted for Poincaré ball:
    1. Map z to u = log_μ(z) ∈ T_μB^n (logarithmic map)
    2. Move u to v = PT_{μ→0}(u) ∈ T_0B^n (parallel transport to origin)
    3. Calculate log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)

    Args:
        z: Sample point(s) on Poincaré ball, shape (..., n)
        mu: Mean point on Poincaré ball, shape (..., n)
        sigma: Covariance parameterization. Can be:
            - Scalar: isotropic covariance sigma^2 I (n x n)
            - 1D array of length n: diagonal covariance diag(sigma_1^2, ..., sigma_n^2)
            - 2D array (n, n): full covariance matrix (must be SPD)
        c: Curvature (positive scalar)

    Returns:
        Log probability, shape (...) (spatial dimension removed)

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from hyperbolix.distributions import wrapped_normal_poincare
        >>>
        >>> # Compute log probability of samples
        >>> key = jax.random.PRNGKey(0)
        >>> mu = jnp.array([0.0, 0.0])
        >>> sigma = 0.1
        >>> z = wrapped_normal_poincare.sample(key, mu, sigma, c=1.0)
        >>> log_p = wrapped_normal_poincare.log_prob(z, mu, sigma, c=1.0)
        >>> log_p.shape
        ()
        >>>
        >>> # Batch computation
        >>> z_batch = wrapped_normal_poincare.sample(key, mu, sigma, c=1.0, sample_shape=(10,))
        >>> log_p_batch = wrapped_normal_poincare.log_prob(z_batch, mu, sigma, c=1.0)
        >>> log_p_batch.shape
        (10,)
    """
    import jax.numpy as jnp

    # Use provided manifold module or default class instance
    if manifold_module is not None:
        manifold = manifold_module
    else:
        from ..manifolds.poincare import Poincare

        manifold = Poincare(dtype=z.dtype)

    # Determine dtype
    dtype = z.dtype

    # Extract dimension
    n = mu.shape[-1]  # Dimension of Poincaré ball

    # Step 1: Map z to tangent space at mu using logarithmic map
    # u = log_μ(z)
    if z.ndim > mu.ndim:
        # z has sample dimensions, need to vmap over them
        n_sample_dims = z.ndim - mu.ndim

        # Create vmapped version of logmap
        logmap_fn = manifold.logmap
        for _ in range(n_sample_dims):
            logmap_fn = jax.vmap(logmap_fn, in_axes=(0, None, None))

        u = logmap_fn(z, mu, c)
    elif z.ndim == mu.ndim and mu.ndim > 1:
        # Both are batched, vmap over batch dimension
        u = jax.vmap(lambda zz, mm: manifold.logmap(zz, mm, c))(z, mu)
    else:
        # Single point
        u = manifold.logmap(z, mu, c)

    # Step 2: Parallel transport from mu to origin
    # v = PT_{μ→0}(u)
    mu_0 = jnp.zeros(n, dtype=dtype)  # Origin is zero vector in Poincaré ball

    if u.ndim > 1:
        # Batched, need to vmap
        if mu.ndim > 1:
            # mu is also batched
            v = jax.vmap(lambda uu, mm: manifold.ptransp(uu, mm, mu_0, c))(u, mu)
        else:
            # Only u is batched (from sample_shape)
            v = jax.vmap(lambda uu: manifold.ptransp(uu, mu, mu_0, c))(u)
    else:
        # Single point
        v = manifold.ptransp(u, mu, mu_0, c)

    # Step 3: Compute log p(v) where v ~ N(0, Σ)
    # Note: v is already in R^n, no need to extract spatial components like in hyperboloid
    log_p_v = _gaussian_log_prob(v, sigma, n, dtype)

    # Step 4: Compute log det Jacobian
    log_det_jac = _log_det_jacobian(v, c, n)

    # Step 5: Compute final log probability
    # log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)
    log_p_z = log_p_v - log_det_jac

    return log_p_z
