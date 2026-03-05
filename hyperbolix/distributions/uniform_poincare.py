"""Riemannian-uniform distribution on a geodesic ball in the Poincaré model.

Samples points uniformly (w.r.t. the Riemannian volume element) within a
geodesic ball B(center, R) of finite radius R. Uses geodesic polar
coordinates: sample a direction on S^{n-1}, sample a radius from the
hyperbolic radial density, form a tangent vector, and map to the ball.

The radial density is p(r) ∝ sinh^{n-1}(√c·r) on [0, R].  A substitution
u = cosh(√c·r) - 1 simplifies sampling:
  - n = 2: u is uniform on [0, cosh(√c·R) - 1]  (closed-form)
  - n ≥ 3: rejection sampling with acceptance ∝ (u·(u+2))^{(n-2)/2}

Dimension key:
  S: sample dimensions (from sample_shape)
  D: spatial/manifold dimension (n)
  Q: quadrature points (64 for GL)
  T: total flattened samples (for rejection sampling)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hyperbolix.manifolds import Manifold


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature (64 nodes/weights on [-1, 1])
# Precomputed via numpy.polynomial.legendre.leggauss(64).
# ---------------------------------------------------------------------------
def _gl_nodes_weights_64() -> tuple[Float[Array, "64"], Float[Array, "64"]]:
    """Return 64-point Gauss-Legendre nodes and weights on [-1, 1]."""
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(64)
    return jnp.asarray(nodes, dtype=jnp.float64), jnp.asarray(weights, dtype=jnp.float64)


# Cache at module level so they are computed once.
_GL_NODES, _GL_WEIGHTS = _gl_nodes_weights_64()


# ---------------------------------------------------------------------------
# Volume of geodesic ball
# ---------------------------------------------------------------------------
def volume(c: float, n: int, R: float) -> Float[Array, ""]:
    """Riemannian volume of a geodesic ball B^n_c(R) in n-dim hyperbolic space.

    Vol = ω_{n-1} / c^{(n-1)/2} · ∫₀ᴿ sinh^{n-1}(√c·r) dr

    where ω_{n-1} is the surface area of the unit (n-1)-sphere.

    Computed via 64-point Gauss-Legendre quadrature.

    Args:
        c: Positive curvature parameter.
        n: Ambient dimension of the Poincaré ball.
        R: Geodesic radius of the ball.

    Returns:
        Scalar volume.
    """
    # ω_{n-1} = 2 π^{n/2} / Γ(n/2)
    omega = 2.0 * jnp.pi ** (n / 2.0) / jnp.exp(jax.lax.lgamma(n / 2.0))

    sqrt_c = jnp.sqrt(jnp.float64(c))

    # Map GL nodes from [-1, 1] to [0, R]: r = R/2 · (t + 1)
    r_nodes_Q = (R / 2.0) * (_GL_NODES + 1.0)
    integrand_Q = jnp.sinh(sqrt_c * r_nodes_Q) ** (n - 1)
    integral = (R / 2.0) * jnp.sum(_GL_WEIGHTS * integrand_Q)

    vol = omega / sqrt_c ** (n - 1) * integral
    return vol


# ---------------------------------------------------------------------------
# Direction sampling (Muller method)
# ---------------------------------------------------------------------------
def _sample_uniform_direction(
    key: PRNGKeyArray,
    n: int,
    shape: tuple[int, ...],
    dtype,
) -> Float[Array, "... n"]:
    """Sample directions uniformly on S^{n-1} via the Muller method."""
    z_SD = jax.random.normal(key, shape=(*shape, n), dtype=dtype)
    norm_S1 = jnp.sqrt(jnp.sum(z_SD**2, axis=-1, keepdims=True))  # (*S, 1)
    norm_S1 = jnp.maximum(norm_S1, 1e-15)
    return z_SD / norm_S1


# ---------------------------------------------------------------------------
# Radial sampling
# ---------------------------------------------------------------------------
def _sample_radial_n2(
    key: PRNGKeyArray,
    c: float,
    R: float,
    shape: tuple[int, ...],
    dtype,
) -> Float[Array, "..."]:
    """Closed-form radial sampling for n = 2.

    u = cosh(√c·r) - 1 is uniform on [0, u_max] when n = 2.
    """
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=dtype))
    u_max = jnp.cosh(sqrt_c * R) - 1.0
    u_S = jax.random.uniform(key, shape=shape, dtype=dtype, minval=0.0, maxval=u_max)
    r_S = jnp.acosh(u_S + 1.0) / sqrt_c
    return r_S


def _sample_radial_rejection(
    key: PRNGKeyArray,
    c: float,
    n: int,
    R: float,
    shape: tuple[int, ...],
    dtype,
) -> Float[Array, "..."]:
    """Rejection sampling for radial component when n ≥ 3.

    Proposal: u ~ Uniform[0, u_max] where u = cosh(√c·r) - 1.
    Acceptance: (u·(u+2) / u_max²)^{(n-2)/2}.

    Uses jax.lax.while_loop for JIT compatibility.
    """
    sqrt_c = jnp.asarray(jnp.sqrt(c), dtype=dtype)
    u_max = jnp.cosh(sqrt_c * R) - 1.0
    # Maximum of u*(u+2) over [0, u_max] is at u = u_max
    ref_val = u_max * (u_max + 2.0)
    exponent = (n - 2) / 2.0

    # Flatten shape for the while_loop, then reshape at the end
    total = 1
    for s in shape:
        total *= s

    def body_fn(state):
        accepted_T, samples_T, loop_key = state
        k1, k2, loop_key = jax.random.split(loop_key, 3)
        u_T = jax.random.uniform(k1, shape=(total,), dtype=dtype, minval=0.0, maxval=u_max)
        alpha_T = jax.random.uniform(k2, shape=(total,), dtype=dtype)
        accept_prob_T = ((u_T * (u_T + 2.0)) / ref_val) ** exponent
        accept_mask_T = alpha_T < accept_prob_T
        # Fill in not-yet-accepted positions
        new_mask_T = accept_mask_T & ~accepted_T
        samples_T = jnp.where(new_mask_T, u_T, samples_T)
        accepted_T = accepted_T | accept_mask_T
        return accepted_T, samples_T, loop_key

    def cond_fn(state):
        accepted_T, _, _ = state
        return ~jnp.all(accepted_T)

    init_state = (
        jnp.zeros(total, dtype=jnp.bool_),
        jnp.zeros(total, dtype=dtype),
        key,
    )
    _, u_accepted_T, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    u_accepted_T = jnp.asarray(u_accepted_T, dtype=dtype)  # narrow type for pyright

    r_T = jnp.acosh(u_accepted_T + 1.0) / sqrt_c
    return r_T.reshape(shape)


def _sample_radial(
    key: PRNGKeyArray,
    c: float,
    n: int,
    R: float,
    shape: tuple[int, ...],
    dtype,
) -> Float[Array, "..."]:
    """Sample geodesic radii from p(r) ∝ sinh^{n-1}(√c·r) on [0, R]."""
    if n == 2:
        return _sample_radial_n2(key, c, R, shape, dtype)
    else:
        return _sample_radial_rejection(key, c, n, R, shape, dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def sample(
    key: PRNGKeyArray,
    n: int,
    c: float,
    R: float,
    sample_shape: tuple[int, ...] = (),
    center: Float[Array, "n"] | None = None,
    dtype=None,
    manifold_module: Manifold | None = None,
) -> Float[Array, "... n"]:
    """Sample uniformly from a geodesic ball in the Poincaré model.

    Draws points that are Riemannian-uniform within B(center, R).

    Algorithm:
        1. Sample direction u ~ Uniform(S^{n-1})
        2. Sample geodesic radius r ~ p(r) ∝ sinh^{n-1}(√c·r) on [0, R]
        3. Form tangent vector t = (r/2)·u  (the /2 accounts for λ(0)=2)
        4. Map to ball: x₀ = expmap_0(t, c)
        5. Move to center: x = center ⊕ x₀  (Möbius addition)

    Args:
        key: JAX PRNG key.
        n: Dimension of the Poincaré ball.
        c: Positive curvature parameter.
        R: Geodesic radius of the ball.
        sample_shape: Batch shape of samples. Default: () → single sample.
        center: Center of the geodesic ball, shape (n,). Default: origin.
        dtype: Output dtype. Default: float64.
        manifold_module: Optional Manifold instance. Default: Poincare(dtype).

    Returns:
        Samples on the Poincaré ball, shape ``sample_shape + (n,)``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from hyperbolix.distributions import uniform_poincare
        >>>
        >>> key = jax.random.PRNGKey(0)
        >>> x = uniform_poincare.sample(key, n=2, c=1.0, R=1.0, sample_shape=(100,))
        >>> x.shape
        (100, 2)
    """
    if manifold_module is not None:
        manifold = manifold_module
    else:
        from ..manifolds.poincare import Poincare

        _dtype = dtype if dtype is not None else jnp.float64
        manifold = Poincare(dtype=_dtype)

    if dtype is None:
        dtype = jnp.float64

    k1, k2 = jax.random.split(key)

    # 1. Direction on S^{n-1}
    directions_SD = _sample_uniform_direction(k1, n, sample_shape, dtype)

    # 2. Geodesic radii
    radii_S = _sample_radial(k2, c, n, R, sample_shape, dtype)

    # 3. Tangent vectors: t = (r/2) · u
    # The /2 compensates for expmap_0 mapping ||v|| → geodesic distance 2·||v||
    tangents_SD = (radii_S[..., None] / 2.0) * directions_SD  # (*S, 1) * (*S, D)

    # 4-5. Map to ball and translate to center
    def _map_single(t_D):
        x0_D = manifold.expmap_0(t_D, c)
        if center is not None:
            return manifold.addition(center, x0_D, c)
        return x0_D

    # vmap over all sample dimensions
    mapped_fn = _map_single
    for _ in sample_shape:
        mapped_fn = jax.vmap(mapped_fn)

    if sample_shape:
        result_SD = mapped_fn(tangents_SD)
    else:
        result_SD = _map_single(tangents_SD)

    return result_SD


def log_prob(
    x: Float[Array, "... n"],
    c: float,
    R: float,
    center: Float[Array, "n"] | None = None,
    manifold_module: Manifold | None = None,
) -> Float[Array, "..."]:
    """Log-probability of the Riemannian-uniform distribution on B(center, R).

    Returns -log Vol(B^n_c(R)) for points inside the geodesic ball, -∞ outside.

    Args:
        x: Point(s) on the Poincaré ball, shape (..., n).
        c: Positive curvature parameter.
        R: Geodesic radius of the ball.
        center: Center of the geodesic ball, shape (n,). Default: origin.
        manifold_module: Optional Manifold instance. Default: Poincare(dtype).

    Returns:
        Log-probability, shape (...).
    """
    if manifold_module is not None:
        manifold = manifold_module
    else:
        from ..manifolds.poincare import Poincare

        manifold = Poincare(dtype=x.dtype)

    n = x.shape[-1]

    # Compute geodesic distance from center
    if center is not None:
        if x.ndim > 1:
            dist_fn = jax.vmap(lambda xi: manifold.dist(xi, center, c))
            d_SB = dist_fn(x)
        else:
            d_SB = manifold.dist(x, center, c)
    else:
        if x.ndim > 1:
            dist_fn = jax.vmap(lambda xi: manifold.dist_0(xi, c))
            d_SB = dist_fn(x)
        else:
            d_SB = manifold.dist_0(x, c)

    log_vol = jnp.log(volume(c, n, R))

    # -log(vol) inside ball, -inf outside
    inside_SB = d_SB <= R
    return jnp.where(inside_SB, -log_vol, -jnp.inf)
