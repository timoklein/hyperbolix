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
  T: total flattened points (rejection sampling, and log_prob's flattened leading axes)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hyperbolix.manifolds import Manifold
from hyperbolix.utils.math_utils import MIN_NORM

# ---------------------------------------------------------------------------
# 64-point Gauss-Legendre quadrature nodes/weights on [-1, 1].
# Precomputed with numpy.polynomial.legendre.leggauss(64).
#
# Stored as plain Python tuples, not ``jnp.array(..., dtype=float64)``: a module-level
# float64 array warns (and silently truncates to float32) on every import when
# ``jax_enable_x64`` is off, and it pins the quadrature — and therefore ``volume``'s
# return dtype — to one precision regardless of what the caller asked for. ``volume``
# materializes them at its own working dtype instead.
# fmt: off
# ---------------------------------------------------------------------------
_GL_NODES = (
    -9.993050417357721704e-01, -9.963401167719552198e-01, -9.910133714767442870e-01, -9.833362538846259771e-01,
    -9.733268277899109755e-01, -9.610087996520537690e-01, -9.464113748584027652e-01, -9.295691721319395695e-01,
    -9.105221370785028245e-01, -8.893154459951141400e-01, -8.659993981540927699e-01, -8.406292962525803159e-01,
    -8.132653151227975385e-01, -7.839723589433413853e-01, -7.528199072605319397e-01, -7.198818501716108820e-01,
    -6.852363130542332703e-01, -6.489654712546573112e-01, -6.111553551723932776e-01, -5.718956462026340004e-01,
    -5.312794640198945650e-01, -4.894031457070529556e-01, -4.463660172534640869e-01, -4.022701579639916258e-01,
    -3.572201583376681255e-01, -3.113228719902109698e-01, -2.646871622087674236e-01, -2.174236437400070832e-01,
    -1.696444204239928311e-01, -1.214628192961205444e-01, -7.299312178779904237e-02, -2.435029266342442905e-02,
    2.435029266342442905e-02, 7.299312178779904237e-02, 1.214628192961205444e-01, 1.696444204239928311e-01,
    2.174236437400070832e-01, 2.646871622087674236e-01, 3.113228719902109698e-01, 3.572201583376681255e-01,
    4.022701579639916258e-01, 4.463660172534640869e-01, 4.894031457070529556e-01, 5.312794640198945650e-01,
    5.718956462026340004e-01, 6.111553551723932776e-01, 6.489654712546573112e-01, 6.852363130542332703e-01,
    7.198818501716108820e-01, 7.528199072605319397e-01, 7.839723589433413853e-01, 8.132653151227975385e-01,
    8.406292962525803159e-01, 8.659993981540927699e-01, 8.893154459951141400e-01, 9.105221370785028245e-01,
    9.295691721319395695e-01, 9.464113748584027652e-01, 9.610087996520537690e-01, 9.733268277899109755e-01,
    9.833362538846259771e-01, 9.910133714767442870e-01, 9.963401167719552198e-01, 9.993050417357721704e-01,
)

_GL_WEIGHTS = (
    1.783280721694139931e-03, 4.147033260564499287e-03, 6.504457968978502418e-03, 8.846759826363397028e-03,
    1.116813946013102757e-02, 1.346304789671786024e-02, 1.572603047602503037e-02, 1.795171577569728422e-02,
    2.013482315353008756e-02, 2.227017380838296895e-02, 2.435270256871120004e-02, 2.637746971505491173e-02,
    2.833967261425953538e-02, 3.023465707240255429e-02, 3.205792835485149483e-02, 3.380516183714179362e-02,
    3.547221325688230259e-02, 3.705512854024008845e-02, 3.855015317861563984e-02, 3.995374113272054384e-02,
    4.126256324262357611e-02, 4.247351512365361154e-02, 4.358372452932354757e-02, 4.459055816375656622e-02,
    4.549162792741818367e-02, 4.628479658131447183e-02, 4.696818281621007590e-02, 4.754016571483041936e-02,
    4.799938859645842132e-02, 4.834476223480299595e-02, 4.857546744150351148e-02, 4.869095700913981389e-02,
    4.869095700913981389e-02, 4.857546744150351148e-02, 4.834476223480299595e-02, 4.799938859645842132e-02,
    4.754016571483041936e-02, 4.696818281621007590e-02, 4.628479658131447183e-02, 4.549162792741818367e-02,
    4.459055816375656622e-02, 4.358372452932354757e-02, 4.247351512365361154e-02, 4.126256324262357611e-02,
    3.995374113272054384e-02, 3.855015317861563984e-02, 3.705512854024008845e-02, 3.547221325688230259e-02,
    3.380516183714179362e-02, 3.205792835485149483e-02, 3.023465707240255429e-02, 2.833967261425953538e-02,
    2.637746971505491173e-02, 2.435270256871120004e-02, 2.227017380838296895e-02, 2.013482315353008756e-02,
    1.795171577569728422e-02, 1.572603047602503037e-02, 1.346304789671786024e-02, 1.116813946013102757e-02,
    8.846759826363397028e-03, 6.504457968978502418e-03, 4.147033260564499287e-03, 1.783280721694139931e-03,
)
# fmt: on


def _default_float_dtype() -> jnp.dtype:
    """JAX's current default float dtype — float64 when ``jax_enable_x64`` is set, else float32.

    Used as the last fallback when neither an explicit ``dtype``, a ``center`` nor a
    ``manifold_module`` says what precision the caller wants.
    """
    return jnp.dtype(jnp.result_type(float))


# ---------------------------------------------------------------------------
# Volume of geodesic ball
# ---------------------------------------------------------------------------
def volume(c: float, n: int, R: float, dtype=None) -> Float[Array, ""]:
    """Riemannian volume of a geodesic ball B^n_c(R) in n-dim hyperbolic space.

    Vol = ω_{n-1} / c^{(n-1)/2} · ∫₀ᴿ sinh^{n-1}(√c·r) dr

    where ω_{n-1} is the surface area of the unit (n-1)-sphere.

    Computed via 64-point Gauss-Legendre quadrature.

    Args:
        c: Positive curvature parameter.
        n: Ambient dimension of the Poincaré ball.
        R: Geodesic radius of the ball.
        dtype: Working/output dtype. Default: JAX's default float dtype.

    Returns:
        Scalar volume, of dtype ``dtype``.
    """
    working_dtype = jnp.dtype(dtype) if dtype is not None else _default_float_dtype()

    # ω_{n-1} = 2 π^{n/2} / Γ(n/2)
    half_n = jnp.asarray(n / 2.0, dtype=working_dtype)
    omega = 2.0 * jnp.asarray(jnp.pi, dtype=working_dtype) ** half_n / jnp.exp(jax.lax.lgamma(half_n))

    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=working_dtype))
    R_arr = jnp.asarray(R, dtype=working_dtype)

    # Map GL nodes from [-1, 1] to [0, R]: r = R/2 · (t + 1)
    nodes_Q = jnp.asarray(_GL_NODES, dtype=working_dtype)
    weights_Q = jnp.asarray(_GL_WEIGHTS, dtype=working_dtype)
    r_nodes_Q = (R_arr / 2.0) * (nodes_Q + 1.0)
    integrand_Q = jnp.sinh(sqrt_c * r_nodes_Q) ** (n - 1)
    integral = (R_arr / 2.0) * jnp.sum(weights_Q * integrand_Q)

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
    norm_S1 = jnp.maximum(norm_S1, MIN_NORM)
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
        dtype: Output dtype. Default: inferred, in order of preference, from ``center``,
            then ``manifold_module``, then JAX's default float dtype.
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
    # Resolve the working dtype the same way the wrapped-normal siblings do: an explicit
    # ``dtype`` wins, otherwise it is inferred from the caller's own array (``center`` here,
    # ``mu`` there). Only when the caller supplies neither does a default apply — the manifold's
    # dtype if one was passed, else JAX's default float dtype. Hardcoding float64 here silently
    # upcast float32 pipelines and warned whenever ``jax_enable_x64`` was off.
    if dtype is not None:
        dtype = jnp.dtype(dtype)
    elif center is not None:
        dtype = jnp.dtype(center.dtype)
    elif manifold_module is not None:
        dtype = jnp.dtype(manifold_module.dtype)
    else:
        dtype = _default_float_dtype()

    if manifold_module is not None:
        manifold = manifold_module
    else:
        from ..manifolds.poincare import Poincare

        manifold = Poincare(dtype=dtype)

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

    # vmap over all sample dimensions. With an empty ``sample_shape`` the loop leaves
    # ``mapped_fn is _map_single``, so the single-sample case needs no branch of its own.
    mapped_fn = _map_single
    for _ in sample_shape:
        mapped_fn = jax.vmap(mapped_fn)

    return mapped_fn(tangents_SD)


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
        manifold_module: Optional Manifold instance. Default: Poincare(x.dtype).

    Returns:
        Log-probability, shape (...), in ``x``'s dtype.
    """
    if manifold_module is not None:
        manifold = manifold_module
    else:
        from ..manifolds.poincare import Poincare

        manifold = Poincare(dtype=x.dtype)

    n = x.shape[-1]
    leading_shape = x.shape[:-1]

    def _dist_single(x_D):
        if center is not None:
            return manifold.dist(x_D, center, c)
        return manifold.dist_0(x_D, c)

    # ``manifold.dist``/``dist_0`` are single-point ops, so one vmap covers exactly one leading
    # axis. Flattening every leading axis into a single one (and restoring the shape afterwards)
    # is what makes the documented ``(..., n) -> (...)`` contract hold for ndim > 2 as well; the
    # old two-branch form raised a dot_general shape error there.
    x_flat_TD = x.reshape(-1, n)
    d_flat_T = jax.vmap(_dist_single)(x_flat_TD)
    d_S = d_flat_T.reshape(leading_shape)

    # Quadrature at ``x``'s precision: a float64 log-volume would upcast the whole result and
    # hand a float32 caller a float64 array.
    log_vol = jnp.log(volume(c, n, R, dtype=x.dtype))

    # -log(vol) inside ball, -inf outside
    inside_S = d_S <= R
    return jnp.where(inside_S, -log_vol, jnp.asarray(-jnp.inf, dtype=x.dtype))
