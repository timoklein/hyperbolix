"""CO-SNE — dimensionality reduction and visualization for hyperbolic data.

Port of CO-SNE (Guo, Guo, Yu — CVPR 2022) to JAX. CO-SNE is the hyperbolic analogue of
t-SNE: high-dimensional similarities are the perplexity-calibrated conditional/joint
probabilities of a hyperbolic normal on **squared** geodesic distances (paper Eq. 2), and
low-dimensional similarities are a heavy-tailed Student-t / Cauchy kernel on geodesic
distances in a low-dimensional Poincaré ball (paper Eq. 9 / the author's code). On top of
the t-SNE KL divergence, CO-SNE adds a **magnitude loss** ``H = (1/N)·Σᵢ(‖xᵢ‖² - ‖yᵢ‖²)²``
(paper Eq. 10) that preserves each point's distance-to-origin — the hierarchy depth that a
plain KL objective would otherwise wash out.

Optimization is projected gradient descent on the ball with momentum 0: the KL gradient is
Riemannian-scaled by ``(1 - c‖y‖²)²/4`` (paper §3.7 / reference), the magnitude gradient
stays Euclidean, and the run is split into two fixed-length stages — an early-exaggeration
KL-only exploration stage, then a KL + magnitude stage.

This module provides a pure functional core (independently usable and JIT-friendly) plus a
thin sklearn-style :class:`CoSNE` wrapper. All embedding-space work runs on the Poincaré
ball; Hyperboloid input/output is mapped through the exact isometry.

Deliberate deviations from the reference implementation
-------------------------------------------------------
- **Exact autodiff gradients** of the declared losses (``jax.grad``), not the reference's
  hand-derived gradients. The reference gradients are demonstrably buggy (an unsquared
  ``alpha = 1 - ‖y‖`` where paper Eq. 13 needs ``1 - ‖y‖²``; a missing ``1/gamma²`` chain factor; a
  magnitude gradient dropping the loss's own ``1/N``). We keep the reference's *structure*
  (Riemannian scaling on the KL gradient only, Euclidean magnitude gradient, projected GD,
  momentum 0, two-stage schedule) but with correct gradients. Consequence: the reference
  learning rates do not transfer — the defaults here are calibrated (see :class:`CoSNE`).
- **Ball re-projection** via :func:`hyperbolix.manifolds._gyrovector_core._proj` (norm
  clip). The reference subtracts ``1e-5`` per coordinate, which can land *outside* the ball
  when ``Σyᵢ < 0``.
- **Fixed-length ``lax.scan`` stages** — no early stopping, progress checks, or per-point
  gains (the reference sets momentum 0 and never uses gains; fixed lengths are JIT-friendly).
- **Curvature-general**: a single ``c`` for both spaces, passed at fit time (the reference
  hardcodes ``c = 1``). Same-``c`` balls have equal radius ``1/√c``, so the magnitude loss
  compares like for like.
- **No out-of-sample ``transform``** — t-SNE is non-parametric (sklearn's ``TSNE`` has none).

Dimension key:
  N: number of points
  D: input spatial dim
  K: number of components (embedding dim)
  S: number of iterations (n_iter)

References:
    Guo, Guo, Yu. "CO-SNE: Dimensionality Reduction and Visualization for Hyperbolic Data."
        CVPR 2022.
    van der Maaten, Hinton. "Visualizing Data using t-SNE." JMLR 2008.
"""

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..manifolds import Hyperboloid, Manifold, Poincare
from ..manifolds._gyrovector_core import _proj
from ..manifolds.hyperboloid import _proj_batch
from ..manifolds.isometry_mappings import hyperboloid_to_poincare, poincare_to_hyperboloid
from ..manifolds.poincare import VERSION_MOBIUS_DIRECT, _dist, _egrad2rgrad
from ..manifolds.protocol import Curvature
from ..utils.helpers import compute_pairwise_distances

# Probability floor / sum guard for P and Q — sklearn's MACHINE_EPSILON semantics: always
# float64 eps, independent of the compute dtype. Flooring at float32 eps (1.19e-7) instead
# would put most of a large-N joint P at the floor (measured ~80% of off-diagonal entries at
# N=1000, ΣP ≈ 1.08), injecting spurious uniform attraction mass. 2.2e-16 is representable in
# float32 and keeps log(P), log(Q) finite.
_MACHINE_EPS = float(jnp.finfo(jnp.float64).eps)

# -------------------------------------------------------------------------------------
# Functional core — high-dimensional similarities (paper Eq. 2)
# -------------------------------------------------------------------------------------


def conditional_probabilities(
    dist_NN: Float[Array, "N N"],
    perplexity: Float[Array, ""] | float,
    *,
    n_steps: int = 100,
) -> Float[Array, "N N"]:
    """Row-stochastic conditional similarities ``p_{j|i}`` (paper Eq. 2, t-SNE style).

    Takes the **plain** hyperbolic distance matrix and squares it internally (matching the
    reference's ``square_distances=True`` flow), then per-row calibrates a precision
    ``βᵢ = 1/(2sigmaᵢ²)`` so the entropy of row ``i`` equals ``ln(perplexity)`` nats (the sklearn
    ``_binary_search_perplexity`` algorithm). The bisection is vectorized over all rows and
    carried through a fixed ``n_steps``-iteration ``lax.fori_loop`` (no early-exit tolerance;
    100 bisections ≫ converged). Degenerate all-equal-distance rows converge to the uniform
    distribution — acceptable.

    Args:
        dist_NN: Plain (not squared) pairwise geodesic distance matrix, shape (N, N).
        perplexity: Target perplexity (effective number of neighbors); must be < N.
        n_steps: Number of bisection iterations (static; default 100).

    Returns:
        Row-stochastic conditional similarities ``P``, shape (N, N); rows sum to 1, diagonal
        is 0.
    """
    dtype = dist_NN.dtype
    eps = _MACHINE_EPS
    n_points = dist_NN.shape[0]
    d2_NN = dist_NN**2  # Eq. 2 uses squared hyperbolic distances
    offdiag_NN = ~jnp.eye(n_points, dtype=bool)
    log_perp = jnp.log(jnp.asarray(perplexity, dtype=dtype))  # target entropy (nats)

    inf = jnp.asarray(jnp.inf, dtype=dtype)
    beta_N = jnp.ones((n_points,), dtype=dtype)
    beta_min_N = jnp.full((n_points,), -inf, dtype=dtype)
    beta_max_N = jnp.full((n_points,), inf, dtype=dtype)

    def _row_entropy(beta_N: Float[Array, "N"]) -> tuple[Float[Array, "N N"], Float[Array, "N"], Float[Array, "N"]]:
        # Unnormalized similarities exp(-d²·β), diagonal masked to 0.
        p_NN = jnp.where(offdiag_NN, jnp.exp(-d2_NN * beta_N[:, None]), 0.0)
        sum_p_N = jnp.maximum(jnp.sum(p_NN, axis=1), eps)
        # H(Pᵢ) = ln(Σ pᵢ) + βᵢ·Σ(d²·pᵢ)/Σ pᵢ  (Shannon entropy of the normalized row, nats).
        entropy_N = jnp.log(sum_p_N) + beta_N * jnp.sum(d2_NN * p_NN, axis=1) / sum_p_N
        return p_NN, sum_p_N, entropy_N

    def _body(_: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        beta_N, beta_min_N, beta_max_N = carry
        _, _, entropy_N = _row_entropy(beta_N)
        # Entropy too high ⇒ row too diffuse ⇒ sharpen by increasing β.
        too_high_N = entropy_N > log_perp
        new_beta_min_N = jnp.where(too_high_N, beta_N, beta_min_N)
        new_beta_max_N = jnp.where(too_high_N, beta_max_N, beta_N)
        # Clamp the doubling: with coincident points the target entropy can be unreachable from
        # above, so β doubles every step — unclamped it overflows to inf and 0·inf = NaN on the
        # zero-distance entries (float32 reaches inf at ~128 doublings).
        beta_capped_N = jnp.minimum(beta_N * 2.0, jnp.asarray(jnp.finfo(dtype).max / 4.0, dtype=dtype))
        beta_up_N = jnp.where(jnp.isinf(beta_max_N), beta_capped_N, (beta_N + beta_max_N) / 2.0)
        beta_down_N = jnp.where(jnp.isinf(beta_min_N), beta_N / 2.0, (beta_N + beta_min_N) / 2.0)
        new_beta_N = jnp.where(too_high_N, beta_up_N, beta_down_N)
        return new_beta_N, new_beta_min_N, new_beta_max_N

    beta_N, _, _ = jax.lax.fori_loop(0, n_steps, _body, (beta_N, beta_min_N, beta_max_N))
    p_NN, sum_p_N, _ = _row_entropy(beta_N)
    return p_NN / sum_p_N[:, None]


def joint_probabilities(
    dist_NN: Float[Array, "N N"],
    perplexity: Float[Array, ""] | float,
) -> Float[Array, "N N"]:
    """Symmetric joint similarities ``P`` (paper Eq. 2, symmetrized t-SNE affinities).

    Symmetrizes the conditional similarities (``P = cond + condᵀ``), normalizes them to sum to
    1, and floors the off-diagonal at machine epsilon (the diagonal is forced to 0) — matching
    sklearn's ``_joint_probabilities`` including the eps floor.

    Args:
        dist_NN: Plain (not squared) pairwise geodesic distance matrix, shape (N, N).
        perplexity: Target perplexity; must be < N.

    Returns:
        Symmetric joint similarity matrix ``P``, shape (N, N); ``ΣP ≈ 1``, off-diagonal
        ``≥ eps``, diagonal 0.
    """
    eps = _MACHINE_EPS
    n_points = dist_NN.shape[0]
    cond_NN = conditional_probabilities(dist_NN, perplexity)
    p_NN = cond_NN + cond_NN.T
    p_NN = p_NN / jnp.maximum(jnp.sum(p_NN), eps)
    offdiag_NN = ~jnp.eye(n_points, dtype=bool)
    return jnp.where(offdiag_NN, jnp.maximum(p_NN, eps), 0.0)


# -------------------------------------------------------------------------------------
# Functional core — low-dimensional similarities (paper Eq. 9 / reference kernel)
# -------------------------------------------------------------------------------------


def low_dim_probabilities(
    y_NK: Float[Array, "N K"],
    gamma: Float[Array, ""] | float,
    c: Curvature,
    *,
    exact_cauchy: bool = False,
) -> Float[Array, "N N"]:
    """Heavy-tailed low-dimensional similarities ``Q`` on the Poincaré ball (paper Eq. 9).

    Computes pairwise ball distances (direct Möbius formula), forms a Student-t / Cauchy
    kernel ``(1 + u/dof)^(-(dof+1)/2)`` with ``dof = max(K-1, 1)``, and normalizes to a joint
    distribution. Two kernel forms select the argument ``u``:

    - ``exact_cauchy=False`` (default, the author's-code form that produced the published
      figures and is used by HycoCLIP): ``u = d/gamma²`` (**plain** geodesic distance).
    - ``exact_cauchy=True`` (paper Eq. 9 Cauchy): ``u = d²/gamma²`` (**squared** distance).

    At ``K = 2`` (``dof = 1``) the kernel reduces to ``(1 + u)⁻¹``. The diagonal is zeroed
    before normalization; the result is eps-floored off-diagonal for a stable ``log q`` in the
    KL divergence.

    Args:
        y_NK: Low-dimensional Poincaré ball coordinates, shape (N, K).
        gamma: Kernel scale gamma (paper default 0.1).
        c: Curvature (positive).
        exact_cauchy: Use the paper Eq. 9 squared-distance kernel instead of the default
            plain-distance kernel.

    Returns:
        Symmetric low-dimensional similarity matrix ``Q``, shape (N, N); off-diagonal
        ``≥ eps``, diagonal 0.
    """
    dtype = y_NK.dtype
    eps = _MACHINE_EPS
    n_points, n_components = y_NK.shape

    dist_fn = functools.partial(_dist, c=c, version_idx=VERSION_MOBIUS_DIRECT)
    d_NN = jax.vmap(jax.vmap(dist_fn, in_axes=(None, 0)), in_axes=(0, None))(y_NK, y_NK)

    gamma2 = jnp.asarray(gamma, dtype=dtype) ** 2
    u_NN = d_NN**2 / gamma2 if exact_cauchy else d_NN / gamma2
    dof = jnp.asarray(max(n_components - 1, 1), dtype=dtype)  # Student-t dof (static shape ⇒ concrete)
    kernel_NN = (1.0 + u_NN / dof) ** (-(dof + 1.0) / 2.0)

    offdiag_NN = ~jnp.eye(n_points, dtype=bool)
    kernel_NN = jnp.where(offdiag_NN, kernel_NN, 0.0)  # zero diagonal BEFORE normalizing
    sum_kernel = jnp.maximum(jnp.sum(kernel_NN), eps)
    return jnp.where(offdiag_NN, jnp.maximum(kernel_NN / sum_kernel, eps), 0.0)


# -------------------------------------------------------------------------------------
# Functional core — losses
# -------------------------------------------------------------------------------------


def kl_divergence_loss(
    y_NK: Float[Array, "N K"],
    p_NN: Float[Array, "N N"],
    gamma: Float[Array, ""] | float,
    c: Curvature,
    *,
    exact_cauchy: bool = False,
) -> Float[Array, ""]:
    """t-SNE KL divergence ``Σ_{i≠j} p·ln(p/q)`` between high- and low-dim similarities.

    ``q`` is recomputed from ``y_NK`` via :func:`low_dim_probabilities`. The diagonal is
    masked out; both ``p`` and ``q`` are replaced by 1.0 on the diagonal *inside* the ``log``
    so the log argument is strictly positive everywhere (the masked terms then contribute a
    hard zero with a finite — not ``0·∞`` — gradient). Off-diagonal ``p`` and ``q`` are already
    eps-floored by :func:`joint_probabilities` / :func:`low_dim_probabilities`.

    During early exaggeration this is called with ``p = P·early_exaggeration`` (scaling the
    unnormalized joint P is the standard t-SNE exaggeration trick).

    Args:
        y_NK: Low-dimensional Poincaré ball coordinates, shape (N, K).
        p_NN: High-dimensional (possibly exaggerated) joint similarities, shape (N, N).
        gamma: Kernel scale gamma.
        c: Curvature (positive).
        exact_cauchy: Kernel-form flag forwarded to :func:`low_dim_probabilities`.

    Returns:
        Scalar KL divergence.
    """
    n_points = y_NK.shape[0]
    q_NN = low_dim_probabilities(y_NK, gamma, c, exact_cauchy=exact_cauchy)
    offdiag_NN = ~jnp.eye(n_points, dtype=bool)
    p_safe_NN = jnp.where(offdiag_NN, p_NN, 1.0)  # 1.0 on the diagonal ⇒ log finite, gradient safe
    q_safe_NN = jnp.where(offdiag_NN, q_NN, 1.0)
    log_ratio_NN = jnp.log(p_safe_NN) - jnp.log(q_safe_NN)
    kl_NN = jnp.where(offdiag_NN, p_NN * log_ratio_NN, 0.0)
    return jnp.sum(kl_NN)


def magnitude_loss(
    y_NK: Float[Array, "N K"],
    x_sqnorms_N: Float[Array, "N"],
) -> Float[Array, ""]:
    """Magnitude (distance-to-origin) preservation loss ``(1/N)·Σᵢ(‖xᵢ‖² - ‖yᵢ‖²)²`` (Eq. 10).

    The CO(-ordinate) term of CO-SNE: it pins each embedded point's squared Euclidean ball
    norm to the input point's squared ball norm, preserving the hierarchy depth that the KL
    term alone would discard. Includes the ``1/N`` the reference gradient drops.

    Args:
        y_NK: Low-dimensional Poincaré ball coordinates, shape (N, K).
        x_sqnorms_N: Squared Euclidean ball norms ``‖xᵢ‖²`` of the input points, shape (N,).

    Returns:
        Scalar magnitude loss.
    """
    y_sqnorms_N = jnp.sum(y_NK**2, axis=1)
    return jnp.mean((x_sqnorms_N - y_sqnorms_N) ** 2)


# -------------------------------------------------------------------------------------
# Functional core — fit
# -------------------------------------------------------------------------------------


def fit_cosne(
    dist_NN: Float[Array, "N N"],
    x_sqnorms_N: Float[Array, "N"],
    c: Curvature,
    key: PRNGKeyArray,
    *,
    n_components: int,
    perplexity: Float[Array, ""] | float,
    gamma: Float[Array, ""] | float,
    learning_rate: Float[Array, ""] | float,
    learning_rate_h: Float[Array, ""] | float,
    early_exaggeration: Float[Array, ""] | float,
    n_iter: int,
    exploration_n_iter: int,
    exact_cauchy: bool = False,
) -> tuple[Float[Array, "N K"], Float[Array, "S"], Float[Array, "S"], Float[Array, ""]]:
    """Fit a CO-SNE embedding from the two sufficient statistics of the input data.

    Takes the input distance matrix and squared ball norms (the two statistics CO-SNE needs —
    the HycoCLIP flow) and runs two-stage projected gradient descent on the Poincaré ball:

    1. **Exploration** (length ``min(exploration_n_iter, n_iter)``): early-exaggerated joint P
       (``p = P·early_exaggeration``), KL gradient only (magnitude learning rate 0).
    2. **Convergence** (remaining iterations): unexaggerated P, KL + magnitude gradients.

    Each step (both stages) applies a single Euclidean update then re-projects onto the ball:
    the KL gradient is Riemannian-scaled by ``(1 - c‖y‖²)²/4`` (via ``_egrad2rgrad``), the
    magnitude gradient stays Euclidean (paper §3.7 / reference structure — deliberate).

    Note:
        Stage-1 KL-trace entries are measured against the *exaggerated* P, so a visible
        discontinuity at the stage boundary is expected. ``final_kl`` re-evaluates the KL on
        the unexaggerated P (one extra eval inside JIT) for a clean final value even when
        ``n_iter ≤ exploration_n_iter``.

    Args:
        dist_NN: Input-space pairwise geodesic distance matrix (Poincaré ball), shape (N, N).
        x_sqnorms_N: Squared Euclidean ball norms of the input points, shape (N,).
        c: Curvature (positive), shared by input and embedding spaces.
        key: PRNG key for the ``0.01·N(0, I)`` embedding init.
        n_components: Embedding dimension K (static).
        perplexity: Target perplexity for the high-dimensional similarities.
        gamma: Low-dimensional kernel scale gamma.
        learning_rate: KL-gradient step size.
        learning_rate_h: Magnitude-gradient step size (applied in stage 2 only).
        early_exaggeration: Exaggeration factor for the joint P during stage 1.
        n_iter: Total number of iterations S (static; the scan length).
        exploration_n_iter: Exploration-stage length (static; clipped to ``n_iter``).
        exact_cauchy: Kernel-form flag forwarded to the low-dimensional similarities.

    Returns:
        Tuple ``(y_NK, kl_trace_S, h_trace_S, final_kl)``: the embedding (N, K), the per-step
        KL and magnitude traces (each length S), and the final KL on the unexaggerated P.
    """
    dtype = dist_NN.dtype
    n_points = dist_NN.shape[0]

    p_NN = joint_probabilities(dist_NN, perplexity)

    # Reference init: 0.01·randn projected onto the ball.
    y0_NK = 0.01 * jax.random.normal(key, (n_points, n_components), dtype=dtype)
    y0_NK = jax.vmap(_proj, in_axes=(0, None))(y0_NK, c)

    kl_loss_fn = functools.partial(kl_divergence_loss, exact_cauchy=exact_cauchy)

    def _make_step(p_eff_NN: Float[Array, "N N"], lr_h_eff: Float[Array, ""]):
        def step_fn(y_NK: Float[Array, "N K"], _: None) -> tuple[Array, tuple[Array, Array]]:
            kl, grad_kl_NK = jax.value_and_grad(kl_loss_fn)(y_NK, p_eff_NN, gamma, c)
            rgrad_kl_NK = jax.vmap(_egrad2rgrad, in_axes=(0, 0, None))(grad_kl_NK, y_NK, c)  # KL grad only
            h, grad_h_NK = jax.value_and_grad(magnitude_loss)(y_NK, x_sqnorms_N)  # Euclidean magnitude grad
            y_new_NK = y_NK - learning_rate * rgrad_kl_NK - lr_h_eff * grad_h_NK
            y_new_NK = jax.vmap(_proj, in_axes=(0, None))(y_new_NK, c)
            return y_new_NK, (kl, h)

        return step_fn

    explore = min(exploration_n_iter, n_iter)  # static

    # Stage 1: exaggerated P, magnitude off (lr_h = 0).
    step1_fn = _make_step(p_NN * early_exaggeration, jnp.asarray(0.0, dtype=dtype))
    y_NK, (kl1_S, h1_S) = jax.lax.scan(step1_fn, y0_NK, None, length=explore)

    # Stage 2: unexaggerated P, magnitude on.
    step2_fn = _make_step(p_NN, jnp.asarray(learning_rate_h, dtype=dtype))
    y_NK, (kl2_S, h2_S) = jax.lax.scan(step2_fn, y_NK, None, length=n_iter - explore)

    kl_trace_S = jnp.concatenate([kl1_S, kl2_S])
    h_trace_S = jnp.concatenate([h1_S, h2_S])
    final_kl = kl_loss_fn(y_NK, p_NN, gamma, c)  # unexaggerated P
    return y_NK, kl_trace_S, h_trace_S, final_kl


# Module-level jitted singleton (shared across CoSNE instances → no per-instance recompiles).
# Dynamic: dist_NN, x_sqnorms_N, c, key, perplexity, gamma, learning_rate, learning_rate_h,
# early_exaggeration. Static: n_components, n_iter, exploration_n_iter (scan lengths + init
# shape) and exact_cauchy (the kernel Python branch).
_fit_jit = jax.jit(
    fit_cosne,
    static_argnames=("n_components", "n_iter", "exploration_n_iter", "exact_cauchy"),
)


# -------------------------------------------------------------------------------------
# sklearn-style class wrapper
# -------------------------------------------------------------------------------------


class CoSNE:
    """CO-SNE — hyperbolic t-SNE with magnitude preservation (Guo et al. 2022).

    A thin sklearn-style wrapper over the functional core. Fits a low-dimensional Poincaré
    ball embedding of hyperbolic data that preserves both pairwise similarities (t-SNE KL
    term) and each point's distance-to-origin (the CO magnitude term). Input/output follow the
    manifold: Poincaré input ``(N, D)`` embeds to ``(N, K)`` ball coordinates; Hyperboloid
    input ``(N, A)`` embeds to ``(N, K+1)`` hyperboloid points (the ball coordinates lifted
    back through the isometry).

    Being non-parametric (like sklearn's ``TSNE``), it has no out-of-sample ``transform`` —
    only ``fit`` / ``fit_transform``. For a precomputed-distance workflow (e.g. HycoCLIP), call
    :func:`fit_cosne` directly with the distance matrix and squared ball norms.

    .. note::
        Every step forms dense ``O(N²)`` matrices, so this shares the ``compute_pairwise_
        distances`` caveat: budget memory carefully beyond a few thousand points. Fitting is
        most reliable in ``dtype=jnp.float64``.

    Args:
        manifold: A ``Poincare`` or ``Hyperboloid`` instance (sets the I/O model and dtype).
        n_components: Embedding dimension K (default 2).
        perplexity: Target perplexity — the effective neighbor count (default 30.0, sklearn).
            Must be < N.
        gamma: Low-dimensional kernel scale gamma (default 0.1, paper).
        learning_rate: KL-gradient step size (default 0.5). **Calibrated**, not inherited: the
            exact-autodiff gradients differ in scale from the reference's buggy hand gradients
            (the missing ``1/gamma²`` factor alone is ~100x at gamma = 0.1), so the reference learning
            rates do not transfer. Swept over {0.1, 0.5, 1, 5, 10} on the synthetic-cluster
            recovery test: values ≥ 1 *overshoot* (the final KL rises with lr and cluster
            recovery degrades); lr = 0.1 minimizes the KL best but makes the magnitude loss
            redundant (KL alone already preserves norms, so the CO ablation gap collapses into
            noise, ~+0.003 Spearman); lr = 0.5 is the balance point — stable KL, strong norm
            preservation (Spearman ~0.83), and a robust magnitude-loss ablation gap (~+0.10 to
            +0.17 Spearman) that shows the CO term doing its job.
        learning_rate_h: Magnitude-gradient step size (default 0.01, paper λ₂). Applied in the
            convergence stage only.
        early_exaggeration: Joint-P exaggeration factor during exploration (default 12.0,
            sklearn).
        n_iter: Total number of iterations (default 1000).
        exploration_n_iter: Exploration-stage length (default 500, reference / paper's "first
            500 iterations"); clipped to ``n_iter``.
        exact_cauchy: Use the paper Eq. 9 squared-distance kernel ``(1 + d²/gamma²)⁻¹`` instead of
            the default author's-code plain-distance kernel ``(1 + d/gamma²)⁻¹`` (default False).

    Attributes (set after :meth:`fit`, sklearn trailing-underscore convention):
        embedding_: Low-dimensional embedding — ``(N, K)`` ball coordinates for Poincaré
            input, ``(N, K+1)`` hyperboloid points for Hyperboloid input.
        kl_divergence_: Final KL divergence on the unexaggerated P (scalar).
        losses_kl_: Per-step KL trace, shape (S,). Stage-1 entries use the exaggerated P (a
            discontinuity at the stage boundary is expected).
        losses_h_: Per-step magnitude-loss trace, shape (S,) (recorded in both stages;
            applied only in stage 2).
        c_: Curvature used at fit time.
    """

    def __init__(
        self,
        manifold: Manifold,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
        gamma: float = 0.1,
        learning_rate: float = 0.5,
        learning_rate_h: float = 0.01,
        early_exaggeration: float = 12.0,
        n_iter: int = 1000,
        exploration_n_iter: int = 500,
        exact_cauchy: bool = False,
    ) -> None:
        if isinstance(manifold, Hyperboloid):
            self._is_hyperboloid = True
        elif isinstance(manifold, Poincare):
            self._is_hyperboloid = False
        else:
            raise ValueError(f"CoSNE supports 'Poincare' or 'Hyperboloid' manifolds, got {type(manifold).__name__}.")

        self.manifold = manifold
        self.n_components = n_components
        self.perplexity = perplexity
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_h = learning_rate_h
        self.early_exaggeration = early_exaggeration
        self.n_iter = n_iter
        self.exploration_n_iter = exploration_n_iter
        self.exact_cauchy = exact_cauchy

        # All embedding-space work runs on the ball at the I/O dtype.
        self._poincare = Poincare(dtype=manifold.dtype)

        # Fitted state (populated by fit()).
        self.embedding_: Array | None = None
        self.kl_divergence_: Array | None = None
        self.losses_kl_: Array | None = None
        self.losses_h_: Array | None = None
        self.c_: Curvature | None = None

    def _to_ball(self, x_ND: Float[Array, "N R"], c: Curvature) -> Float[Array, "N D"]:
        """Convert/clean input into Poincaré ball coordinates (N, D)."""
        x_cast = self._poincare._cast(x_ND)
        if self._is_hyperboloid:
            x_hyp = _proj_batch(x_cast, c)  # hyperboloid hygiene before the isometry (mirrors HoroPCA)
            return jax.vmap(hyperboloid_to_poincare, in_axes=(0, None))(x_hyp, c)
        return jax.vmap(_proj, in_axes=(0, None))(x_cast, c)  # ball hygiene

    def fit(self, x_ND: Float[Array, "N R"], c: Curvature, key: PRNGKeyArray) -> "CoSNE":
        """Fit the embedding on ``x_ND`` at curvature ``c``.

        Args:
            x_ND: Input points, shape (N, D) for Poincaré or (N, A) for Hyperboloid.
            c: Curvature (positive) — bound into the fitted state.
            key: PRNG key for the embedding init.

        Returns:
            ``self``.
        """
        if x_ND.ndim != 2:
            raise ValueError(f"Expected a 2D array (n_points, dim), got shape {x_ND.shape}.")
        n_points = x_ND.shape[0]
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if n_points <= 1:
            raise ValueError(f"Need more than one point to embed, got {n_points}.")
        if self.perplexity >= n_points:
            raise ValueError(f"perplexity must be < n_points ({n_points}), got {self.perplexity}.")

        x_ball_ND = self._to_ball(x_ND, c)
        dist_NN = compute_pairwise_distances(x_ball_ND, self._poincare, c, VERSION_MOBIUS_DIRECT)
        x_sqnorms_N = jnp.sum(x_ball_ND**2, axis=1)

        y_NK, kl_trace_S, h_trace_S, final_kl = _fit_jit(
            dist_NN,
            x_sqnorms_N,
            c,
            key,
            n_components=self.n_components,
            perplexity=self.perplexity,
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            learning_rate_h=self.learning_rate_h,
            early_exaggeration=self.early_exaggeration,
            n_iter=self.n_iter,
            exploration_n_iter=self.exploration_n_iter,
            exact_cauchy=self.exact_cauchy,
        )

        if self._is_hyperboloid:
            embedding = jax.vmap(poincare_to_hyperboloid, in_axes=(0, None))(y_NK, c)  # (N, K+1)
        else:
            embedding = y_NK  # (N, K)

        self.embedding_ = embedding
        self.kl_divergence_ = final_kl
        self.losses_kl_ = kl_trace_S
        self.losses_h_ = h_trace_S
        self.c_ = c
        return self

    def fit_transform(self, x_ND: Float[Array, "N R"], c: Curvature, key: PRNGKeyArray) -> Float[Array, "N out"]:
        """Fit on ``x_ND`` then return its embedding (equivalent to ``fit(...).embedding_``)."""
        return self.fit(x_ND, c, key).embedding_
