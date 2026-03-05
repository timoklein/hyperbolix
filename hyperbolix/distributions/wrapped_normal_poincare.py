"""Wrapped normal distribution on Poincaré ball.

Simpler implementation than hyperboloid - no parallel transport needed!
Uses exponential map and Möbius addition.

Dimension key:
  S: sample dimensions (from sample_shape)
  B: batch dimensions (from mu batch shape)
  D: spatial/manifold dimension (n)

References:
    Mathieu et al. "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
    NeurIPS 2019. https://arxiv.org/abs/1901.06033
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Float, PRNGKeyArray

from hyperbolix.manifolds import Manifold

from ._common import sample_gaussian, sigma_to_cov


def sample(
    key: PRNGKeyArray,
    mu: Float[Array, "... n"],
    sigma: Float[Array, "..."] | float,
    c: float,
    sample_shape: tuple[int, ...] = (),
    dtype=None,
    manifold_module: Manifold | None = None,
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
    # Full noise shape: sample_shape + mu_batch_shape + (n,) = (*S, *B, D)
    cov_DD = sigma_to_cov(sigma, n, dtype)
    full_sample_shape = sample_shape + mu_batch_shape
    v_SBD = sample_gaussian(key, cov_DD, sample_shape=full_sample_shape, dtype=dtype)

    # Scale Euclidean tangent vector to Riemannian tangent space coordinates.
    # At origin, the conformal factor λ(0) = 2/(1 - c·0) = 2, so Riemannian
    # coordinates require dividing by λ(0). This matches the reference (Mathieu et al.
    # 2019) which does v = v / lambda_x(zero) before the exponential map.
    v_SBD = v_SBD / 2.0

    # Step 2: Map to ball at origin: z_0 = exp_0(v)
    # Step 3: Move to mean: z = μ ⊕ z_0

    def transform_single(v_D, mu_D):
        """Transform a single (v, mu) pair."""
        z_0_D = manifold.expmap_0(v_D, c)
        z_D = manifold.addition(mu_D, z_0_D, c)
        return z_D

    if len(sample_shape) == 0 and len(mu_batch_shape) == 0:
        # Single point, no batching
        z_SBD = transform_single(v_SBD, mu)
    elif len(sample_shape) == 0:
        # No sample_shape but batched mu: v_SBD is (*B, D), mu is (*B, D)
        vmapped_fn = transform_single
        for _ in mu_batch_shape:
            vmapped_fn = jax.vmap(vmapped_fn)
        z_SBD = vmapped_fn(v_SBD, mu)
    else:
        # v_SBD is (*S, *B, D), mu is (*B, D)
        # First, vmap over mu_batch_shape dimensions (both v and mu have these)
        vmapped_fn = transform_single
        for _ in mu_batch_shape:
            vmapped_fn = jax.vmap(vmapped_fn)

        # Then vmap over sample_shape dimensions (only v has these, mu is broadcast)
        for _ in sample_shape:
            vmapped_fn = jax.vmap(vmapped_fn, in_axes=(0, None))

        z_SBD = vmapped_fn(v_SBD, mu)

    return z_SBD


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

    cov_DD = sigma_to_cov(sigma, n, dtype)
    mean_D = jnp.zeros(n, dtype=dtype)
    return jnp.asarray(multivariate_normal.logpdf(v, mean_D, cov_DD))


def _log_det_jacobian(
    v: Float[Array, "... n"],
    c: float,
    n: int,
) -> Float[Array, "..."]:
    """Compute log determinant of projection Jacobian for Poincaré ball.

    Computes log det(∂proj_μ(v)/∂v) = (n-1) * log(sinh(√c·r) / (√c·r))
    where r = λ(0)·||v||_E = 2·||v||_E is the Riemannian norm of v at the origin.

    The Riemannian norm is needed because the Jacobian formula uses the geodesic
    distance, which equals the Riemannian norm of the log-map tangent vector.

    For numerical stability, uses Taylor expansion for small √c·r:
    log(sinh(√c·r) / (√c·r)) ≈ (c·r²)/6 for √c·r → 0

    Args:
        v: Tangent vector at origin in Euclidean coordinates, shape (..., n)
        c: Curvature (positive scalar)
        n: Dimension of Poincaré ball

    Returns:
        Log determinant of Jacobian, shape (...)
    """

    # Compute Riemannian norm of tangent vector at origin.
    # ||v||_g = λ(0) · ||v||_E = 2 · ||v||_E, since the conformal factor at
    # the origin is λ(0) = 2. The Jacobian formula uses the Riemannian norm.
    r_euclid_B = jnp.sqrt(jnp.sum(v**2, axis=-1))  # (...,) — sum over D
    r_B = 2.0 * r_euclid_B  # Riemannian norm

    # Scale by √c
    sqrt_c = jnp.sqrt(c)
    sqrt_c_r_B = sqrt_c * r_B

    # Threshold for switching to Taylor expansion
    threshold = 1e-3

    # Standard computation: log(sinh(√c·r) / (√c·r)) = log(sinh(√c·r)) - log(√c·r)
    log_ratio_standard_B = jnp.log(jnp.sinh(sqrt_c_r_B)) - jnp.log(sqrt_c_r_B)

    # Taylor expansion for small √c·r: log(sinh(x) / x) ≈ x²/6
    # Here x = √c·r_riem, so x² = c·r_riem² = c·(2·r_euclid)² = 4·c·r_euclid²
    log_ratio_taylor_B = (c * r_B**2) / 6.0

    # Use Taylor expansion when √c·r < threshold
    log_ratio_B = jnp.where(sqrt_c_r_B < threshold, log_ratio_taylor_B, log_ratio_standard_B)

    # log det = (n-1) * log(sinh(√c·r) / (√c·r))
    log_det_B = (n - 1) * log_ratio_B

    return log_det_B


def log_prob(
    z: Float[Array, "... n"],
    mu: Float[Array, "... n"],
    sigma: Float[Array, "..."] | float,
    c: float,
    manifold_module: Manifold | None = None,
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

    # Step 1: Map z to tangent space at mu: u = log_μ(z), shape (..., D)
    if z.ndim > mu.ndim:
        n_sample_dims = z.ndim - mu.ndim
        logmap_fn = manifold.logmap
        for _ in range(n_sample_dims):
            logmap_fn = jax.vmap(logmap_fn, in_axes=(0, None, None))
        u_SBD = logmap_fn(z, mu, c)
    elif z.ndim == mu.ndim and mu.ndim > 1:
        u_SBD = jax.vmap(lambda zz, mm: manifold.logmap(zz, mm, c))(z, mu)
    else:
        u_SBD = manifold.logmap(z, mu, c)

    # Step 2: Parallel transport from mu to origin: v = PT_{μ→0}(u)
    mu_0_D = jnp.zeros(n, dtype=dtype)

    if u_SBD.ndim > 1:
        if mu.ndim > 1:
            v_SBD = jax.vmap(lambda uu, mm: manifold.ptransp(uu, mm, mu_0_D, c))(u_SBD, mu)
        else:
            v_SBD = jax.vmap(lambda uu: manifold.ptransp(uu, mu, mu_0_D, c))(u_SBD)
    else:
        v_SBD = manifold.ptransp(u_SBD, mu, mu_0_D, c)

    # Step 3: Compute log p(v) where v ~ N(0, Σ)
    # Scale to Riemannian coordinates: v_riem = λ(0) · v_euclid = 2 · v
    v_riem_SBD = v_SBD * 2.0
    log_p_v_SB = _gaussian_log_prob(v_riem_SBD, sigma, n, dtype)

    # Step 4: Compute log det Jacobian
    log_det_jac_SB = _log_det_jacobian(v_SBD, c, n)

    # Step 5: log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)
    log_p_z_SB = log_p_v_SB - log_det_jac_SB

    return log_p_z_SB
