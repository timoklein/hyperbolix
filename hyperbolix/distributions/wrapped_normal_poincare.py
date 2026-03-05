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
from jaxtyping import Array, Float, PRNGKeyArray

from hyperbolix.manifolds import Manifold

from ._common import gaussian_log_prob, sample_gaussian, sigma_to_cov
from ._wrapped_normal_base import _batched_transform, _log_det_jacobian_from_r


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

    return _batched_transform(transform_single, v_SBD, mu, sample_shape, mu_batch_shape)


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
    log_p_v_SB = gaussian_log_prob(v_riem_SBD, sigma, n, dtype)

    # Step 4: Compute log det Jacobian
    # Riemannian norm r = λ(0) · ||v||_E = 2 · ||v||_E
    r_SB = 2.0 * jnp.sqrt(jnp.sum(v_SBD**2, axis=-1))
    log_det_jac_SB = _log_det_jacobian_from_r(r_SB, c, n)

    # Step 5: log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)
    log_p_z_SB = log_p_v_SB - log_det_jac_SB

    return log_p_z_SB
