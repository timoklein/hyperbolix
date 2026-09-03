"""HoroPCA — hyperbolic dimensionality reduction via horospherical projections.

Port of the curvature-general HoroPCA variant (Chami et al., ICML 2021) to JAX. All
computation runs on the hyperboloid; Poincaré input is mapped in via the exact isometry.
K ideal points (horospheres) are jointly optimized to maximize the variance of the
horospherically projected data (measured as the mean squared pairwise geodesic distance),
using Adam with gradient clipping. Projecting onto the K-dimensional ideal span and reading
off Busemann-style ball coordinates yields the low-dimensional embedding.

This module provides a pure functional core (independently usable and JIT-friendly) plus a
thin sklearn-style :class:`HoroPCA` wrapper.

Dimension key:
  N: number of points
  D: spatial dim
  A: ambient dim (D + 1)
  K: number of components (ideal points)
  S: number of fit steps (max_steps)

References:
    Chami, Gu, Nguyen, Ré. "HoroPCA: Hyperbolic Dimensionality Reduction via Horospherical
        Projections." ICML 2021.
"""

import functools

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from ..manifolds import Hyperboloid, Manifold, Poincare
from ..manifolds._gyrovector_core import _proj_batch as _proj_batch_ball
from ..manifolds.hyperboloid import (
    VERSION_DEFAULT,
    VERSION_SMOOTHENED,
    _busemann,
    _create_origin,
    _dist,
    _embed_spatial_0,
    _expmap,
    _expmap_0,
    _minkowski_inner,
    _proj,
    _proj_batch,
)
from ..manifolds.isometry_mappings import hyperboloid_to_poincare, poincare_to_hyperboloid
from ..manifolds.protocol import Curvature
from ..utils.helpers import compute_pairwise_distances
from ..utils.math_utils import MIN_NORM, floor_at
from ..utils.precision import MATMUL_PRECISION
from .frechet import frechet_mean

# -------------------------------------------------------------------------------------
# Functional core
# -------------------------------------------------------------------------------------


def lift_ideals(q_KD: Float[Array, "K D"]) -> Float[Array, "K A"]:
    """Lift spatial ideal directions to null vectors on the light cone.

    Prepends a ones column: ``p_k = [1, q_k]``. For unit ``q_k`` these are null vectors
    (``⟨p_k, p_k⟩_L = -1 + ‖q_k‖² = 0``), the ideal-point lifts used by the horospherical
    projection. Curvature-free (the null lift does not depend on ``c``).

    Args:
        q_KD: Spatial ideal directions (rows), shape (K, D). Assumed unit-norm.

    Returns:
        Null lifts, shape (K, A) with A = D + 1.
    """
    ones_K1 = jnp.ones((q_KD.shape[0], 1), dtype=q_KD.dtype)
    return jnp.concatenate([ones_K1, q_KD], axis=1)


def orthonormalize_rows(q_KD: Float[Array, "K D"]) -> Float[Array, "K D"]:
    """Orthonormalize the rows of ``q_KD`` via a reduced QR of its transpose.

    Returns rows that are orthonormal in the Euclidean metric (needs ``D >= K``). The QR
    sign is deterministic per input but frame-dependent (flipping a row flips the
    corresponding ideal point) — compare only geometric invariants across runs, never the
    components themselves.

    Args:
        q_KD: Component directions (rows), shape (K, D) with D >= K.

    Returns:
        Row-orthonormal directions, shape (K, D).
    """
    q_ortho_DK, _ = jnp.linalg.qr(q_KD.T)  # reduced QR: (D, K) with orthonormal columns
    return q_ortho_DK.T


def horo_projection(
    x_A: Float[Array, "A"],
    q_ortho_KD: Float[Array, "K D"],
    c: Curvature,
    version_idx: int = VERSION_DEFAULT,
) -> Float[Array, "A"]:
    """Horospherical projection of a single hyperboloid point onto the K ideal directions.

    Projects ``x`` onto the geodesic submanifold spanned by the K ideal points while
    preserving every Busemann coordinate ``B^{q_k}(x)`` (Chami et al. 2021). Dispatches on
    the static number of components ``K``:

    - ``K = 1``: closed form ``π(x) = expmap_0(-B_q(x)·[0, q])`` (the reference's
      Sherman-Morrison inverse is singular at K=1). The sign convention matches
      :func:`~hyperbolix.manifolds.hyperboloid._busemann`.
    - ``K ≥ 2``: (1) Minkowski-project ``x`` onto ``span(p_k)`` using the closed-form
      inverse of the null-lift Gram ``I - 𝟙𝟙ᵀ`` (Sherman-Morrison, exact for K ≥ 2);
      normalize to the manifold to get the *spine* point; (2) build the unit tangent at the
      spine pointing toward the origin (orthogonal to the ideal span); (3) walk a geodesic of
      length ``d(x, spine)`` along it.

    Args:
        x_A: Hyperboloid point, shape (A,), A = D + 1.
        q_ortho_KD: Row-orthonormal ideal directions, shape (K, D).
        c: Curvature (positive).
        version_idx: Distance version used for the geodesic step length (default
            ``VERSION_DEFAULT``; the loss passes ``VERSION_SMOOTHENED`` for smooth grads).

    Returns:
        Projected hyperboloid point, shape (A,).
    """
    num_components = q_ortho_KD.shape[0]
    dim = q_ortho_KD.shape[1]
    dtype = x_A.dtype

    if num_components == 1:
        # Closed form: preserve the single Busemann coordinate B^q(x).
        q_D = q_ortho_KD[0]  # unit spatial ideal direction
        busemann = _busemann(x_A, q_D, c)  # B^q(x)
        tangent_A = -busemann * _embed_spatial_0(q_D)  # tangent at origin: -B·[0, q]
        return _proj(_expmap_0(tangent_A, c), c)

    p_KA = lift_ideals(q_ortho_KD)  # null lifts [1, q_k], shape (K, A)

    def _span_coeffs(y_A: Float[Array, "A"]) -> Float[Array, "K"]:
        # Minkowski inner products ⟨y, p_k⟩_L = -y_0 + q_k · y_s, then apply the closed-form
        # inverse of the null-lift Gram G = I - 𝟙𝟙ᵀ: G⁻¹ = I + 𝟙𝟙ᵀ/(1-K) (Sherman-Morrison).
        by_K = -y_A[0] + jnp.matmul(q_ortho_KD, y_A[1:], precision=MATMUL_PRECISION)  # (K,)
        return by_K + (jnp.sum(by_K) / (1.0 - num_components)) * jnp.ones(num_components, dtype=dtype)

    # (1) Minkowski projection of x onto span(p_k), normalized onto the manifold (the spine).
    coeffs_K = _span_coeffs(x_A)
    mp_A = jnp.matmul(coeffs_K, p_KA, precision=MATMUL_PRECISION)  # (A,) projection of x onto the ideal span
    mp_inner = _minkowski_inner(mp_A, mp_A)  # < 0 (timelike) for a valid spine
    spine_A = mp_A / jnp.sqrt(floor_at(-c * mp_inner, MIN_NORM))
    # Sheet hygiene BEFORE _proj: _proj rebuilds a positive time from the spatial part and
    # cannot itself flip a lower-sheet point back up, so reflect first (spine_0 != 0).
    spine_A = spine_A * jnp.sign(spine_A[0])
    spine_A = _proj(spine_A, c)

    # (2) Unit tangent at the spine pointing toward the origin (⊥ span(P) ⇒ tangent at spine).
    origin_A = _create_origin(c, dim, dtype)
    origin_coeffs_K = _span_coeffs(origin_A)
    # projection of the origin onto the ideal span
    proj_span_o_A = jnp.matmul(origin_coeffs_K, p_KA, precision=MATMUL_PRECISION)
    tangent_A = origin_A - proj_span_o_A  # spacelike, ⊥ span(P)
    tangent_inner = _minkowski_inner(tangent_A, tangent_A)  # > 0 (spacelike)
    unit_tangent_A = tangent_A / jnp.sqrt(floor_at(tangent_inner, MIN_NORM))

    # (3) Geodesic walk of length d(x, spine) along the unit tangent.
    step = _dist(x_A, spine_A, c, version_idx)
    return _proj(_expmap(step * unit_tangent_A, spine_A, c), c)


def horopca_loss(q_KD: Float[Array, "K D"], x_NA: Float[Array, "N A"], c: Curvature) -> Float[Array, ""]:
    """Negative mean squared pairwise (smoothened) distance of the projected points.

    Orthonormalizes ``q_KD`` internally, horospherically projects every point (with the
    smoothened distance version, differentiable at coincident points), then returns
    ``-mean(d(π(x_i), π(x_j))²)`` over the full NxN matrix (diagonal included, matching the
    reference). Maximizing projected variance ⇔ minimizing this loss.

    Args:
        q_KD: Component directions (rows, pre-orthonormalization), shape (K, D).
        x_NA: Hyperboloid points, shape (N, A).
        c: Curvature (positive).

    Returns:
        Scalar loss.
    """
    q_ortho_KD = orthonormalize_rows(q_KD)
    proj_fn = functools.partial(horo_projection, version_idx=VERSION_SMOOTHENED)
    proj_NA = jax.vmap(proj_fn, in_axes=(0, None, None))(x_NA, q_ortho_KD, c)

    dist_fn = functools.partial(_dist, version_idx=VERSION_SMOOTHENED)
    pairwise_NN = jax.vmap(jax.vmap(dist_fn, in_axes=(None, 0, None)), in_axes=(0, None, None))(proj_NA, proj_NA, c)
    return -jnp.mean(pairwise_NN**2)


def fit_horopca(
    x_NA: Float[Array, "N A"],
    c: Curvature,
    key: PRNGKeyArray,
    *,
    n_components: int,
    lr: float = 1e-3,
    max_steps: int = 100,
) -> tuple[Float[Array, "K D"], Float[Array, "S"]]:
    """Fit HoroPCA components on hyperboloid data by Adam + gradient clipping.

    Draws a Gaussian ``(K, D)`` component matrix, then runs ``max_steps`` of Adam (with
    global-norm clipping at 1e5) on :func:`horopca_loss` via ``lax.scan``. Returns the
    final orthonormalized components and the per-step loss trace.

    Args:
        x_NA: Hyperboloid points (Fréchet-centered for best results), shape (N, A).
        c: Curvature (positive).
        key: PRNG key for the Gaussian component init.
        n_components: Number of components K (static; needs 1 <= K <= D).
        lr: Adam learning rate (default 1e-3).
        max_steps: Number of optimization steps S (default 100).

    Returns:
        Tuple ``(q_ortho_KD, losses_S)``: row-orthonormal components (K, D) and the loss
        trace (S,).
    """
    dim = x_NA.shape[1] - 1
    q_KD = jax.random.normal(key, (n_components, dim), dtype=x_NA.dtype)

    tx = optax.chain(optax.clip_by_global_norm(1e5), optax.adam(lr))
    opt_state = tx.init(q_KD)

    def step_fn(carry: tuple[Array, optax.OptState], _: None) -> tuple[tuple[Array, optax.OptState], Array]:
        q, state = carry
        loss, grad = jax.value_and_grad(horopca_loss)(q, x_NA, c)
        updates, new_state = tx.update(grad, state, q)
        new_q = optax.apply_updates(q, updates)
        return (new_q, new_state), loss

    (q_final_KD, _), losses_S = jax.lax.scan(step_fn, (q_KD, opt_state), None, length=max_steps)
    return orthonormalize_rows(q_final_KD), losses_S


def transform_horopca(
    x_NA: Float[Array, "N A"],
    q_ortho_KD: Float[Array, "K D"],
    c: Curvature,
) -> tuple[Float[Array, "N A"], Float[Array, "N K"]]:
    """Project data and read off the K-dimensional Poincaré ball coordinates.

    Horospherically projects each point (default distance version), maps the projections to
    the Poincaré ball, and reads the K ball coordinates as the component-frame coefficients
    ``y @ q_orthoᵀ`` (an isometry onto the K-ball, since the projected spatial parts lie in
    the span of the orthonormal rows).

    Args:
        x_NA: Hyperboloid points, shape (N, A).
        q_ortho_KD: Row-orthonormal components, shape (K, D).
        c: Curvature (positive).

    Returns:
        Tuple ``(proj_NA, ball_NK)``: the projected hyperboloid points (N, A) and the
        low-dimensional Poincaré ball coordinates (N, K).
    """
    proj_NA = jax.vmap(horo_projection, in_axes=(0, None, None, None))(x_NA, q_ortho_KD, c, VERSION_DEFAULT)
    ball_ND = jax.vmap(hyperboloid_to_poincare, in_axes=(0, None))(proj_NA, c)  # (N, D) full ball coords
    ball_NK = jnp.matmul(ball_ND, q_ortho_KD.T, precision=MATMUL_PRECISION)  # (N, K) coordinates in the component frame
    return proj_NA, ball_NK


# Module-level jitted singletons (shared across HoroPCA instances → no per-instance recompiles).
# Dynamic: x, c, key, lr. Static: n_components (also drives the K=1 vs K≥2 Python dispatch and the
# component shape) and max_steps (the scan length).
_fit_jit = jax.jit(fit_horopca, static_argnames=("n_components", "max_steps"))
_transform_jit = jax.jit(transform_horopca)


# -------------------------------------------------------------------------------------
# sklearn-style class wrapper
# -------------------------------------------------------------------------------------


class HoroPCA:
    """HoroPCA — hyperbolic PCA via horospherical projections (Chami et al. 2021).

    A thin sklearn-style wrapper over the functional core. Fit K ideal points to maximize the
    variance of the horospherically projected data, then transform points to K-dimensional
    ball coordinates. Input/output models follow the manifold: Poincaré input ``(N, D)`` maps
    to ``(N, K)`` ball coordinates; Hyperboloid input ``(N, A)`` maps to ``(N, K+1)``
    hyperboloid points (the ball coordinates lifted back through the isometry).

    Args:
        manifold: A ``Poincare`` or ``Hyperboloid`` instance (sets the I/O model and dtype).
        n_components: Number of components K (needs 1 <= K <= D, the spatial dim).
        lr: Adam learning rate (default 1e-3).
        max_steps: Number of optimization steps (default 100).
        center_data: Fréchet-mean-center the data before fitting (default True — the
            algorithm assumes mean-zero data). When False the boost is the identity.
        frechet_step_size: Karcher step size for the Fréchet mean (default 1.0).
        frechet_tol: Karcher convergence tolerance (default 1e-8).
        frechet_max_iters: Karcher iteration cap (default 100).

    Attributes (set after :meth:`fit`, sklearn trailing-underscore convention):
        components_: Row-orthonormal ideal directions, shape (K, D).
        mean_: Fréchet mean on the hyperboloid, shape (A,) (origin if not centering).
        boost_: Lorentz boost sending ``mean_`` to the origin, shape (A, A) (identity if not
            centering).
        c_: Curvature used at fit time.
        losses_: Per-step loss trace, shape (S,).
        total_variance_: Mean squared pairwise (default) distance of the working points.
        explained_variance_: Mean squared pairwise (default) distance of the projections.
        explained_variance_ratio_: ``explained_variance_ / total_variance_``.

    Note:
        ``total_variance_`` / ``explained_variance_`` use the *pairwise* variance matching
        the objective (mean squared pairwise geodesic distance), not point-to-mean scatter.
    """

    def __init__(
        self,
        manifold: Manifold,
        n_components: int,
        *,
        lr: float = 1e-3,
        max_steps: int = 100,
        center_data: bool = True,
        frechet_step_size: float = 1.0,
        frechet_tol: float = 1e-8,
        frechet_max_iters: int = 100,
    ) -> None:
        if isinstance(manifold, Hyperboloid):
            self._is_hyperboloid = True
        elif isinstance(manifold, Poincare):
            self._is_hyperboloid = False
        else:
            raise ValueError(f"HoroPCA supports 'Poincare' or 'Hyperboloid' manifolds, got {type(manifold).__name__}.")

        self.manifold = manifold
        self.n_components = n_components
        self.lr = lr
        self.max_steps = max_steps
        self.center_data = center_data
        self.frechet_step_size = frechet_step_size
        self.frechet_tol = frechet_tol
        self.frechet_max_iters = frechet_max_iters

        # All hyperboloid working-space ops share one internal Hyperboloid at the I/O dtype.
        self._hyperboloid = Hyperboloid(dtype=manifold.dtype)

        # Fitted state (populated by fit()).
        self.components_: Array | None = None
        self.mean_: Array | None = None
        self.boost_: Array | None = None
        self.c_: Curvature | None = None
        self.losses_: Array | None = None
        self.total_variance_: Array | None = None
        self.explained_variance_: Array | None = None
        self.explained_variance_ratio_: Array | None = None

    def _spatial_dim(self, x_ND: Float[Array, "N R"]) -> int:
        """Spatial dim D from an input array (ambient - 1 for hyperboloid, as-is for ball)."""
        return x_ND.shape[1] - 1 if self._is_hyperboloid else x_ND.shape[1]

    def _to_hyperboloid(self, x_ND: Float[Array, "N R"], c: Curvature) -> Float[Array, "N A"]:
        """Convert/clean input into on-manifold hyperboloid points (N, A)."""
        x_cast = self._hyperboloid._cast(x_ND)
        if self._is_hyperboloid:
            return _proj_batch(x_cast, c)  # projection hygiene
        # Ball hygiene BEFORE the lift: poincare_to_hyperboloid floors 1 - c‖y‖² instead of
        # erroring, so an on/outside-boundary point (routine under float32 saturation) would lift
        # to a time coordinate ~1e15 and silently NaN the Fréchet mean.
        x_ball = _proj_batch_ball(x_cast, c)
        return jax.vmap(poincare_to_hyperboloid, in_axes=(0, None))(x_ball, c)

    def fit(self, x_ND: Float[Array, "N R"], c: Curvature, key: PRNGKeyArray) -> "HoroPCA":
        """Fit the components on ``x_ND`` at curvature ``c``.

        Args:
            x_ND: Input points, shape (N, D) for Poincaré or (N, A) for Hyperboloid.
            c: Curvature (positive) — bound into the fitted state.
            key: PRNG key for the Gaussian component init.

        Returns:
            ``self``.
        """
        if x_ND.ndim != 2:
            raise ValueError(f"Expected a 2D array (n_points, dim), got shape {x_ND.shape}.")
        dim = self._spatial_dim(x_ND)
        if not (1 <= self.n_components <= dim):
            raise ValueError(f"n_components must satisfy 1 <= n_components <= {dim} (spatial dim), got {self.n_components}.")

        x_hyp_NA = self._to_hyperboloid(x_ND, c)

        if self.center_data:
            mean_A = frechet_mean(
                x_hyp_NA,
                self._hyperboloid,
                c,
                step_size=self.frechet_step_size,
                tol=self.frechet_tol,
                max_iters=self.frechet_max_iters,
            )
            boost_AA = self._hyperboloid.lorentz_boost(mean_A, c)
            x_work_NA = _proj_batch(jnp.matmul(x_hyp_NA, boost_AA.T, precision=MATMUL_PRECISION), c)
        else:
            ambient = x_hyp_NA.shape[1]
            mean_A = _create_origin(c, ambient - 1, self.manifold.dtype)
            boost_AA = jnp.eye(ambient, dtype=self.manifold.dtype)
            x_work_NA = x_hyp_NA

        components_KD, losses_S = _fit_jit(
            x_work_NA, c, key, n_components=self.n_components, lr=self.lr, max_steps=self.max_steps
        )
        proj_work_NA, _ = _transform_jit(x_work_NA, components_KD, c)

        # Pairwise variance (mean squared pairwise geodesic distance), matching the objective.
        work_dists_NN = compute_pairwise_distances(x_work_NA, self._hyperboloid, c, VERSION_DEFAULT)
        proj_dists_NN = compute_pairwise_distances(proj_work_NA, self._hyperboloid, c, VERSION_DEFAULT)
        total_variance = jnp.mean(work_dists_NN**2)
        explained_variance = jnp.mean(proj_dists_NN**2)

        self.components_ = components_KD
        self.mean_ = mean_A
        self.boost_ = boost_AA
        self.c_ = c
        self.losses_ = losses_S
        self.total_variance_ = total_variance
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance / floor_at(total_variance, MIN_NORM)
        return self

    def transform(self, x_ND: Float[Array, "N R"]) -> Float[Array, "N out"]:
        """Project ``x_ND`` onto the fitted components and return the low-dim embedding.

        Reuses the stored curvature and boost from :meth:`fit` (the fitted state is tied to
        the fit-time ``c``).

        Args:
            x_ND: Input points, shape (N, D) for Poincaré or (N, A) for Hyperboloid.

        Returns:
            Poincaré ball coordinates ``(N, K)`` for Poincaré input, or hyperboloid points
            ``(N, K+1)`` for Hyperboloid input.
        """
        if self.components_ is None or self.c_ is None or self.boost_ is None:
            raise ValueError("HoroPCA must be fitted before calling transform().")

        c = self.c_
        x_hyp_NA = self._to_hyperboloid(x_ND, c)
        x_work_NA = _proj_batch(jnp.matmul(x_hyp_NA, self.boost_.T, precision=MATMUL_PRECISION), c)
        _, ball_NK = _transform_jit(x_work_NA, self.components_, c)

        if self._is_hyperboloid:
            return jax.vmap(poincare_to_hyperboloid, in_axes=(0, None))(ball_NK, c)  # (N, K+1)
        return ball_NK  # (N, K)

    def fit_transform(self, x_ND: Float[Array, "N R"], c: Curvature, key: PRNGKeyArray) -> Float[Array, "N out"]:
        """Fit on ``x_ND`` then return its embedding (equivalent to ``fit(...).transform(...)``)."""
        return self.fit(x_ND, c, key).transform(x_ND)
