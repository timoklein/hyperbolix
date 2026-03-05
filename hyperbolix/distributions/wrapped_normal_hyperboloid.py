"""Wrapped normal distribution on hyperboloid manifold.

Implementation of the wrapped normal distribution that wraps a Gaussian from the
tangent space at the origin onto the hyperboloid via parallel transport and exponential map.

Dimension key:
  S: sample dimensions (from sample_shape)
  B: batch dimensions (from mu batch shape)
  D: spatial dimension (n)
  A: ambient dimension (n+1, hyperboloid time+space)

References:
    Mathieu et al. "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
    NeurIPS 2019. https://arxiv.org/abs/1901.06033
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Float, PRNGKeyArray

from ..manifolds.hyperboloid import Hyperboloid
from ._common import sample_gaussian, sigma_to_cov


def sample(
    key: PRNGKeyArray,
    mu: Float[Array, "... n_plus_1"],
    sigma: Float[Array, "..."] | float,
    c: float,
    sample_shape: tuple[int, ...] = (),
    dtype=None,
    manifold_module: Hyperboloid | None = None,
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
        >>> from hyperbolix.distributions import wrapped_normal_hyperboloid
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
    # Use provided manifold module or default class instance
    if manifold_module is not None:
        manifold = manifold_module
    else:
        # Determine output dtype first for class instantiation
        _dtype = dtype if dtype is not None else mu.dtype
        manifold = Hyperboloid(dtype=_dtype)

    # Determine output dtype
    if dtype is None:
        dtype = mu.dtype

    # Extract spatial dimension
    n = mu.shape[-1] - 1  # Spatial dimension (n for H^n in R^(n+1))

    # Determine batch shape from mu (all dims except the last one)
    # mu.shape = batch_shape + (n+1,)
    mu_batch_shape = mu.shape[:-1]

    # Step 1: Sample v_bar ~ N(0, Σ) ∈ R^n
    # Full noise shape: sample_shape + mu_batch_shape + (n,) = (*S, *B, D)
    cov_DD = sigma_to_cov(sigma, n, dtype)
    full_sample_shape = sample_shape + mu_batch_shape
    v_spatial_SBD = sample_gaussian(key, cov_DD, sample_shape=full_sample_shape, dtype=dtype)

    # Step 2: Embed as tangent vector v = [0, v_bar] ∈ T_{μ₀}ℍⁿ at origin
    # Shape: (*S, *B, A) where A = n+1
    v_SBA = manifold.embed_spatial_0(v_spatial_SBD)

    # Step 3 & 4: Parallel transport and exponential map
    # We need to apply ptransp_0 and expmap element-wise, matching v and mu

    def transform_single(v_A, mu_A):
        """Transform a single (v, mu) pair."""
        u_A = manifold.ptransp_0(v_A, mu_A, c)  # Step 3: parallel transport to mu
        z_A = manifold.expmap(u_A, mu_A, c)  # Step 4: exponential map at mu
        return z_A

    if len(sample_shape) == 0 and len(mu_batch_shape) == 0:
        # Single point, no batching
        z_SBA = transform_single(v_SBA, mu)
    elif len(sample_shape) == 0:
        # No sample_shape but batched mu: v_SBA is (*B, A), mu is (*B, A)
        vmapped_fn = transform_single
        for _ in mu_batch_shape:
            vmapped_fn = jax.vmap(vmapped_fn)
        z_SBA = vmapped_fn(v_SBA, mu)
    else:
        # v_SBA is (*S, *B, A), mu is (*B, A)
        # First, vmap over mu_batch_shape dimensions (both v and mu have these)
        vmapped_fn = transform_single
        for _ in mu_batch_shape:
            vmapped_fn = jax.vmap(vmapped_fn)

        # Then vmap over sample_shape dimensions (only v has these, mu is broadcast)
        for _ in sample_shape:
            vmapped_fn = jax.vmap(vmapped_fn, in_axes=(0, None))

        z_SBA = vmapped_fn(v_SBA, mu)

    return z_SBA


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

    cov_DD = sigma_to_cov(sigma, n, dtype)
    mean_D = jnp.zeros(n, dtype=dtype)
    return jnp.asarray(multivariate_normal.logpdf(v_spatial, mean_D, cov_DD))


def _log_det_jacobian(
    v: Float[Array, "... n_plus_1"],
    c: float,
    n: int,
) -> Float[Array, "..."]:
    """Compute log determinant of projection Jacobian.

    Computes log det(∂proj_μ(v)/∂v) = (n-1) * log(sinh(√c·r) / (√c·r))
    where r = ||v||_L is the Minkowski norm (= Euclidean norm of spatial components at origin).

    For numerical stability, uses Taylor expansion for small √c·r:
    log(sinh(√c·r) / (√c·r)) ≈ c·r²/6 for √c·r → 0

    Args:
        v: Tangent vector at origin, shape (..., n+1)
        c: Curvature (positive scalar)
        n: Spatial dimension (manifold dimension)

    Returns:
        Log determinant of Jacobian, shape (...)
    """

    # Compute Minkowski norm: r = sqrt(⟨v, v⟩_L)
    # For tangent vector at origin: v = [0, v_bar], so ⟨v, v⟩_L = ||v_bar||²
    v_spatial_SBD = v[..., 1:]  # Extract spatial components, (..., D)
    v_minkowski_sq_SB = jnp.sum(v_spatial_SBD**2, axis=-1)  # (...,) — sum over D
    r_SB = jnp.sqrt(jnp.maximum(v_minkowski_sq_SB, 1e-15))

    # Scale by √c — expmap uses cosh(√c·r)/sinh(√c·r), so Jacobian uses the same
    sqrt_c_r_SB = jnp.sqrt(c) * r_SB

    # Threshold for switching to Taylor expansion
    r_threshold = 1e-3

    # Standard computation: log(sinh(√c·r) / (√c·r)) = log(sinh(√c·r)) - log(√c·r)
    log_ratio_standard_SB = jnp.log(jnp.sinh(sqrt_c_r_SB)) - jnp.log(sqrt_c_r_SB)

    # Taylor expansion for small √c·r: log(sinh(x) / x) ≈ x²/6 where x = √c·r
    log_ratio_taylor_SB = (c * r_SB**2) / 6.0

    # Use Taylor expansion when √c·r < threshold
    log_ratio_SB = jnp.where(sqrt_c_r_SB < r_threshold, log_ratio_taylor_SB, log_ratio_standard_SB)

    # log det = (n-1) * log(sinh(√c·r) / (√c·r))
    log_det_SB = (n - 1) * log_ratio_SB

    return log_det_SB


def log_prob(
    z: Float[Array, "... n_plus_1"],
    mu: Float[Array, "... n_plus_1"],
    sigma: Float[Array, "..."] | float,
    c: float,
    manifold_module: Hyperboloid | None = None,
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
        >>> from hyperbolix.distributions import wrapped_normal_hyperboloid
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
    # Use provided manifold module or default class instance
    if manifold_module is not None:
        manifold = manifold_module
    else:
        manifold = Hyperboloid(dtype=z.dtype)

    # Determine dtype
    dtype = z.dtype

    # Extract spatial dimension
    n = mu.shape[-1] - 1  # Spatial dimension

    # Handle batching with vmap if needed
    # For now, assume z and mu have compatible shapes
    # If z has extra leading dimensions (samples), we need to vmap

    # Step 1: Map z to tangent space at mu: u = log_μ(z), shape (..., A)
    if z.ndim > mu.ndim:
        n_sample_dims = z.ndim - mu.ndim
        logmap_fn = manifold.logmap
        for _ in range(n_sample_dims):
            logmap_fn = jax.vmap(logmap_fn, in_axes=(0, None, None))
        u_SBA = logmap_fn(z, mu, c)
    elif z.ndim == mu.ndim and mu.ndim > 1:
        u_SBA = jax.vmap(lambda zz, mm: manifold.logmap(zz, mm, c))(z, mu)
    else:
        u_SBA = manifold.logmap(z, mu, c)

    # Step 2: Parallel transport from mu to origin: v = PT_{μ→μ₀}(u)
    mu_0_A = manifold.create_origin(c, n)

    if u_SBA.ndim > 1:
        if mu.ndim > 1:
            v_SBA = jax.vmap(lambda uu, mm: manifold.ptransp(uu, mm, mu_0_A, c))(u_SBA, mu)
        else:
            v_SBA = jax.vmap(lambda uu: manifold.ptransp(uu, mu, mu_0_A, c))(u_SBA)
    else:
        v_SBA = manifold.ptransp(u_SBA, mu, mu_0_A, c)

    # Step 3: Extract spatial components: v = [0, v_bar] at origin
    v_spatial_SBD = v_SBA[..., 1:]

    # Step 4: Compute log p(v) where v ~ N(0, Σ)
    log_p_v_SB = _gaussian_log_prob(v_spatial_SBD, sigma, n, dtype)

    # Step 5: Compute log det Jacobian
    log_det_jac_SB = _log_det_jacobian(v_SBA, c, n)

    # Step 6: log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)
    log_p_z_SB = log_p_v_SB - log_det_jac_SB

    return log_p_z_SB
