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

from jaxtyping import Array, Float, PRNGKeyArray

from ..manifolds.hyperboloid import Hyperboloid
from ..utils.math_utils import safe_norm
from ._common import gaussian_log_prob, sample_gaussian, sigma_to_cov
from ._wrapped_normal_base import _batched_transform, _log_det_jacobian_from_r, _vmap_sample_and_batch


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
        manifold_module: Optional Hyperboloid instance. Default: Hyperboloid(dtype)

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

    def transform_single(v_A, mu_A):
        """Transform a single (v, mu) pair."""
        u_A = manifold.ptransp_0(v_A, mu_A, c)  # Step 3: parallel transport to mu
        z_A = manifold.expmap(u_A, mu_A, c)  # Step 4: exponential map at mu
        return z_A

    return _batched_transform(transform_single, v_SBA, mu, sample_shape, mu_batch_shape)


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
        manifold_module: Optional Hyperboloid instance. Default: Hyperboloid(z.dtype)

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

    # Batching layout: z is (*S, *B, A) and mu is (*B, A). The mean's batch axes pair with z's
    # trailing axes; the leading sample axes broadcast the mean. Steps 1 and 2 must use the
    # *same* layout — vmapping u and mu together over axis 0 (as step 2 used to) pairs a sample
    # axis with a batch axis and fails on sample()'s own output whenever both are present.
    n_sample_dims = max(z.ndim - mu.ndim, 0)
    n_batch_dims = mu.ndim - 1

    # Step 1: Map z to tangent space at mu: u = log_μ(z), shape (..., A)
    def _logmap_single(z_A, mu_A):
        return manifold.logmap(z_A, mu_A, c)

    u_SBA = _vmap_sample_and_batch(_logmap_single, n_sample_dims, n_batch_dims)(z, mu)

    # Step 2: Parallel transport from mu to origin: v = PT_{μ→μ₀}(u)
    mu_0_A = manifold.create_origin(c, n)

    def _ptransp_single(u_A, mu_A):
        return manifold.ptransp(u_A, mu_A, mu_0_A, c)

    v_SBA = _vmap_sample_and_batch(_ptransp_single, n_sample_dims, n_batch_dims)(u_SBA, mu)

    # Step 3: Extract spatial components: v = [0, v_bar] at origin
    v_spatial_SBD = v_SBA[..., 1:]

    # Step 4: Compute log p(v) where v ~ N(0, Σ)
    log_p_v_SB = gaussian_log_prob(v_spatial_SBD, sigma, n, dtype)

    # Step 5: Compute log det Jacobian
    # Minkowski norm at origin: r = ||v_spatial|| (since v = [0, v_bar]). `safe_norm` gives the
    # finite (zero) gradient at z == mu that the old `floor_at(sum(v**2), MIN_NORM)` was there
    # for, without flooring r at 3.16e-8 and without `sum(v**2)` overflowing float32.
    # `_log_det_jacobian_from_r` is already NaN-free and stationary at r = 0.
    r_SB = safe_norm(v_spatial_SBD)
    log_det_jac_SB = _log_det_jacobian_from_r(r_SB, c, n)

    # Step 6: log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)
    log_p_z_SB = log_p_v_SB - log_det_jac_SB

    return log_p_z_SB
