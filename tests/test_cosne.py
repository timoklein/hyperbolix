"""Tests for CO-SNE (``hyperbolix.decomposition.cosne``).

Correctness anchors use independent oracles — the defining property of each piece, never a
recomputation with the code under test:

- ``conditional_probabilities`` hits the target perplexity per row (``exp(H(Pᵢ)) ≈ perplexity``,
  the algorithm's defining invariant); rows are stochastic with a zero diagonal.
- ``joint_probabilities`` is symmetric, sums to 1, is non-negative, and equals ``1/6`` on every
  off-diagonal of an equilateral triple (hand literal).
- ``low_dim_probabilities`` matches a hand-computed kernel on collinear-through-origin points
  where the ball distance has the closed form ``d(0, r) = (2/√c)·atanh(√c·r)`` — both kernel
  forms.
- ``kl_divergence_loss`` gradients match central finite differences (including at a coincident
  embedding pair); ``magnitude_loss`` matches a hand literal and its transcribed closed-form
  gradient ``-(4/N)(‖x‖² - ‖y‖²)·y``.
- The fit stays strictly inside the ball with finite traces, decreases the stage-2 KL, is
  invariant to ``learning_rate_h`` when the exploration stage fills the whole run, matches
  across Poincaré/Hyperboloid input, reuses its JIT cache, validates its arguments, and
  recovers planted clusters while preserving the norm hierarchy (the CO selling point).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.decomposition import (
    CoSNE,
    conditional_probabilities,
    joint_probabilities,
    kl_divergence_loss,
    magnitude_loss,
)
from hyperbolix.decomposition.cosne import _fit_jit, low_dim_probabilities
from hyperbolix.distributions import wrapped_normal_poincare as wn
from hyperbolix.manifolds import Euclidean, Hyperboloid, Poincare
from hyperbolix.manifolds import isometry_mappings as iso
from hyperbolix.utils.helpers import compute_pairwise_distances

SEEDS = [10, 11, 12]
CURVATURES = [0.3, 1.0, 2.5]


def _ball_points(seed: int, n: int, dim: int, c: float, sigma: float, dtype) -> jnp.ndarray:
    """Wrapped-normal samples on the Poincaré ball (mean at origin), shape (n, dim)."""
    P = Poincare(dtype=dtype)
    mu0 = jnp.zeros(dim, dtype=dtype)
    return wn.sample(jax.random.PRNGKey(seed), mu0, jnp.asarray(sigma, dtype=dtype), c, sample_shape=(n,), manifold_module=P)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks; no scipy)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _knn_accuracy(embedding: np.ndarray, labels: np.ndarray, c: float, k: int = 5) -> float:
    """k-NN classification accuracy in the embedding (geodesic ball distance, majority vote)."""
    P = Poincare(dtype=jnp.float64)
    d = np.array(compute_pairwise_distances(jnp.asarray(embedding), P, c, 0))
    np.fill_diagonal(d, np.inf)
    correct = sum(np.bincount(labels[np.argsort(d[i])[:k]]).argmax() == labels[i] for i in range(len(labels)))
    return correct / len(labels)


# --- 1. conditional probabilities hit the target perplexity ----------------------------
@pytest.mark.parametrize("perplexity", [5.0, 30.0])
@pytest.mark.parametrize("c", CURVATURES)
def test_conditional_probabilities_hit_perplexity(perplexity, c):
    """Every row's realized perplexity ``exp(H(Pᵢ))`` matches the target (defining invariant)."""
    dtype = jnp.float64
    x = _ball_points(10, 64, 5, c, 0.4, dtype)
    dist_NN = compute_pairwise_distances(x, Poincare(dtype=dtype), c, 0)
    cond = np.array(conditional_probabilities(dist_NN, perplexity))

    assert np.allclose(cond.sum(axis=1), 1.0, atol=1e-9)  # rows stochastic
    assert np.allclose(np.diag(cond), 0.0, atol=1e-12)  # zero diagonal
    for i in range(cond.shape[0]):
        p = cond[i]
        nz = p > 0
        entropy = -np.sum(p[nz] * np.log(p[nz]))  # H(Pᵢ) in nats
        assert np.exp(entropy) == pytest.approx(perplexity, rel=1e-3)


# --- 2. joint probabilities: symmetric / normalized / non-negative / zero diagonal ------
def test_joint_probabilities_properties():
    dtype = jnp.float64
    c = 1.0
    x = _ball_points(11, 40, 5, c, 0.4, dtype)
    dist_NN = compute_pairwise_distances(x, Poincare(dtype=dtype), c, 0)
    p = np.array(joint_probabilities(dist_NN, 20.0))

    assert np.allclose(p, p.T, atol=1e-12)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert (p >= 0).all()
    assert np.allclose(np.diag(p), 0.0, atol=1e-12)


# --- 3. joint probabilities on an equilateral triple → 1/6 off-diagonal (hand literal) --
@pytest.mark.parametrize("c", CURVATURES)
def test_joint_probabilities_equidistant_literal(c):
    """Three mutually-equidistant points give ``p = 1/6`` on every off-diagonal entry.

    Each conditional row is uniform ``[·, 1/2, 1/2]`` (equal distances ⇒ β-independent), so the
    symmetrized-and-normalized joint is ``(1/2 + 1/2)/6 = 1/6`` on all six off-diagonal cells.
    """
    dtype = jnp.float64
    r = 0.4
    angles = jnp.asarray([0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0], dtype=dtype)
    pts = r * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)  # equilateral around origin
    dist_NN = compute_pairwise_distances(pts, Poincare(dtype=dtype), c, 0)
    p = np.array(joint_probabilities(dist_NN, 2.0))

    offdiag = ~np.eye(3, dtype=bool)
    assert np.allclose(p[offdiag], 1.0 / 6.0, atol=1e-9)
    assert np.allclose(np.diag(p), 0.0, atol=1e-12)


# --- 4. low-dim probabilities on collinear points (closed-form distance, hand kernel) ---
@pytest.mark.parametrize("exact_cauchy", [False, True])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.5])
def test_low_dim_probabilities_literal(exact_cauchy, c):
    """Q matches a hand-computed kernel on ``[-a, 0], [0, 0], [b, 0]`` (K=2 ⇒ dof=1)."""
    dtype = jnp.float64
    a, b, gamma = 0.3, 0.5, 0.2
    y = jnp.asarray([[-a, 0.0], [0.0, 0.0], [b, 0.0]], dtype=dtype)

    sqrt_c = np.sqrt(c)
    d01 = (2.0 / sqrt_c) * np.arctanh(sqrt_c * a)  # d(0, a)
    d12 = (2.0 / sqrt_c) * np.arctanh(sqrt_c * b)  # d(0, b)
    d02 = d01 + d12  # collinear through origin ⇒ distances add
    dmat = np.array([[0.0, d01, d02], [d01, 0.0, d12], [d02, d12, 0.0]])

    u = dmat**2 / gamma**2 if exact_cauchy else dmat / gamma**2
    kernel = 1.0 / (1.0 + u)  # dof=1 ⇒ exponent -(dof+1)/2 = -1
    np.fill_diagonal(kernel, 0.0)
    q_expected = kernel / kernel.sum()

    q = np.array(low_dim_probabilities(y, gamma, c, exact_cauchy=exact_cauchy))
    assert np.allclose(q, q_expected, atol=1e-9)
    assert q.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(np.diag(q), 0.0, atol=1e-12)


# --- 5. KL gradient vs finite differences (incl. coincident pair) -----------------------
@pytest.mark.parametrize("exact_cauchy", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
def test_kl_grad_matches_finite_differences(exact_cauchy, duplicate):
    """``jax.grad(kl_divergence_loss)`` matches central FD — the kernel + normalization chain.

    Uses ``gamma=0.3`` (not the steeper default 0.1) purely to condition the finite-difference
    oracle: the squared-distance kernel ``1/(1 + d²/gamma²)`` at gamma=0.1 has a large third derivative,
    inflating central-FD truncation error past a tight tolerance. A factor-level gradient bug
    (e.g. a missing ``1/gamma²``) would fail at any gamma.
    """
    dtype = jnp.float64
    c, gamma = 1.0, 0.3
    x = _ball_points(12, 8, 5, c, 0.4, dtype)
    p_NN = joint_probabilities(compute_pairwise_distances(x, Poincare(dtype=dtype), c, 0), 4.0)

    y = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (8, 2), dtype=dtype)
    y = jax.vmap(lambda yy: Poincare(dtype=dtype).proj(yy, c))(y)
    if duplicate:
        y = y.at[1].set(y[0])  # coincident embedding pair (safe-norm gradient path)

    def loss(yy):
        return kl_divergence_loss(yy, p_NN, gamma, c, exact_cauchy=exact_cauchy)

    grad = np.array(jax.grad(loss)(y))
    assert np.isfinite(grad).all()

    h = 1e-6
    y_np = np.array(y)
    fd = np.zeros_like(y_np)
    for idx in np.ndindex(y_np.shape):
        yp, ym = y_np.copy(), y_np.copy()
        yp[idx] += h
        ym[idx] -= h
        fd[idx] = (float(loss(jnp.asarray(yp))) - float(loss(jnp.asarray(ym)))) / (2.0 * h)
    assert np.allclose(grad, fd, atol=1e-5, rtol=1e-4)


# --- 6. magnitude loss: hand literal + transcribed closed-form gradient -----------------
def test_magnitude_loss_literal_and_grad():
    dtype = jnp.float64
    y = jnp.asarray([[0.1, 0.2], [0.3, 0.0], [0.0, 0.5]], dtype=dtype)  # N=3
    x_sqnorms = jnp.asarray([0.1, 0.4, 0.6], dtype=dtype)

    y_sq = np.array([0.1**2 + 0.2**2, 0.3**2, 0.5**2])
    x_sq = np.array([0.1, 0.4, 0.6])
    loss_expected = np.mean((x_sq - y_sq) ** 2)
    assert float(magnitude_loss(y, x_sqnorms)) == pytest.approx(loss_expected, rel=1e-12)

    # Eq. 14 with the 1/N restored: ∂H/∂yᵢ = -(4/N)(‖xᵢ‖² - ‖yᵢ‖²)·yᵢ.
    grad = np.array(jax.grad(magnitude_loss)(y, x_sqnorms))
    grad_expected = -(4.0 / 3.0) * (x_sq - y_sq)[:, None] * np.array(y)
    assert np.allclose(grad, grad_expected, atol=1e-12)


# --- 7. fit stays strictly inside the ball with finite traces --------------------------
@pytest.mark.parametrize("lr", [0.1, 1.0, 20.0])
def test_fit_stays_in_ball_and_finite(lr):
    dtype = jnp.float64
    c = 1.0
    x = _ball_points(13, 30, 5, c, 0.4, dtype)
    model = CoSNE(Poincare(dtype=dtype), 2, perplexity=10.0, learning_rate=lr, n_iter=200, exploration_n_iter=100)
    emb = np.array(model.fit_transform(x, c, jax.random.PRNGKey(0)))

    assert (c * np.sum(emb**2, axis=1) < 1.0).all()  # strictly inside the ball
    assert np.isfinite(emb).all()
    assert np.isfinite(np.array(model.losses_kl_)).all()
    assert np.isfinite(np.array(model.losses_h_)).all()


# --- 8. fit shapes and stage-2 KL decrease ---------------------------------------------
def test_fit_shapes_and_loss_decrease():
    dtype = jnp.float64
    c = 1.0
    n = 60
    x = _ball_points(14, n, 5, c, 0.4, dtype)
    model = CoSNE(Poincare(dtype=dtype), 2, n_iter=600, exploration_n_iter=300).fit(x, c, jax.random.PRNGKey(0))

    assert model.embedding_.shape == (n, 2)
    assert model.losses_kl_.shape == (600,)
    assert model.losses_h_.shape == (600,)

    # Stage-2 KL progress, measured against the post-exploration baseline ``s2[0]`` (both on the
    # unexaggerated P — the exaggeration discontinuity lives at the stage-1→2 boundary, not here).
    # We compare a late window to ``s2[0]`` rather than an early window to a late one, because the
    # active magnitude loss makes the KL dip then partly rebound mid-stage (KL vs norm trade-off),
    # so an "early window" is not a clean baseline.
    stage2 = np.array(model.losses_kl_)[300:]
    assert stage2[-50:].mean() < stage2[0]


# --- 9. stagewise contract: lr_h is inert when exploration fills the whole run ----------
def test_stagewise_contract():
    """With ``n_iter == exploration_n_iter`` the magnitude loss never fires, so lr_h is a no-op."""
    dtype = jnp.float64
    c = 1.0
    x = _ball_points(15, 30, 5, c, 0.4, dtype)
    key = jax.random.PRNGKey(0)
    emb0 = CoSNE(
        Poincare(dtype=dtype), 2, perplexity=10.0, learning_rate_h=0.0, n_iter=200, exploration_n_iter=200
    ).fit_transform(x, c, key)
    emb1 = CoSNE(
        Poincare(dtype=dtype), 2, perplexity=10.0, learning_rate_h=10.0, n_iter=200, exploration_n_iter=200
    ).fit_transform(x, c, key)
    assert np.allclose(np.array(emb0), np.array(emb1), atol=1e-12)


# --- 10. Poincaré / hyperboloid input equivalence --------------------------------------
def test_hyperboloid_input_equivalence():
    """Hyperboloid input embeds isometrically to the Poincaré-input embedding (same key)."""
    dtype = jnp.float64
    c = 1.0
    x_ball = _ball_points(16, 24, 5, c, 0.4, dtype)
    x_hyp = jax.vmap(iso.poincare_to_hyperboloid, in_axes=(0, None))(x_ball, c)
    key = jax.random.PRNGKey(0)

    emb_ball = CoSNE(Poincare(dtype=dtype), 2, perplexity=10.0, n_iter=200, exploration_n_iter=100).fit_transform(
        x_ball, c, key
    )
    emb_hyp = CoSNE(Hyperboloid(dtype=dtype), 2, perplexity=10.0, n_iter=200, exploration_n_iter=100).fit_transform(
        x_hyp, c, key
    )
    emb_hyp_ball = jax.vmap(iso.hyperboloid_to_poincare, in_axes=(0, None))(emb_hyp, c)  # (N, 3) → (N, 2)

    # atol 1e-5 (matching HoroPCA's equivalence test): the two paths are identical in exact
    # arithmetic, but the ``h2p ∘ p2h`` round-trip on the hyperboloid input injects ~1e-15
    # relative error that grows to ~1e-6 over 200 near-boundary optimization steps.
    assert np.allclose(np.array(emb_ball), np.array(emb_hyp_ball), atol=1e-5)


# --- 11. end-to-end synthetic-cluster recovery + norm preservation (paper §4.1) ---------
def test_end_to_end_synthetic_clusters():
    """Recover 5 planted clusters in 2-D while preserving the input norm ordering.

    Deviates from the plan's ``5-NN > 0.8``: these exact means/sigma overlap heavily near the
    origin — three clusters have mean ball-norm ~0.26/0.33/0.26 — so the **input-space** 5-NN
    ceiling is only ~0.74 and 0.8 is geometrically unreachable in *any* embedding. Threshold
    relaxed to 0.5 (the calibrated default lr=0.5 attains ~0.61 here; sanctioned tolerance
    deviation). (b) the norm-Spearman and (c) the strict magnitude-loss ablation hold as
    planned — the magnitude loss provably lifts norm preservation.
    """
    dtype = jnp.float64
    c = 1.0
    means = jnp.asarray(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    P = Poincare(dtype=dtype)
    key = jax.random.PRNGKey(10)
    clusters, labels = [], []
    for i, mu in enumerate(means):
        key, sub = jax.random.split(key)
        clusters.append(wn.sample(sub, mu, jnp.asarray(0.25, dtype=dtype), c, sample_shape=(20,), manifold_module=P))
        labels.extend([i] * 20)
    x = jnp.concatenate(clusters, axis=0)
    labels = np.array(labels)
    x_norm = np.linalg.norm(np.array(x), axis=1)

    fit_key = jax.random.PRNGKey(0)
    emb = np.array(CoSNE(P, 2).fit_transform(x, c, fit_key))

    # (a) clusters recovered (relaxed threshold; see docstring).
    assert _knn_accuracy(emb, labels, c) > 0.5
    # (b) norm hierarchy preserved (the CO selling point).
    sp_on = _spearman(x_norm, np.linalg.norm(emb, axis=1))
    assert sp_on > 0.5
    # (c) the magnitude loss is what preserves the norm ordering.
    emb_off = np.array(CoSNE(P, 2, learning_rate_h=0.0).fit_transform(x, c, fit_key))
    sp_off = _spearman(x_norm, np.linalg.norm(emb_off, axis=1))
    assert sp_off < sp_on


# --- 12. JIT cache reuse ---------------------------------------------------------------
def test_jit_cache_reuse():
    """A second ``_fit_jit`` call with identical statics/shapes/dtypes does not recompile."""
    # Asserted, not skipped: removing the jax.jit wrapper from _fit_jit — the exact
    # regression this test names — makes the attribute vanish, and a skip would report
    # that as a green run.
    assert hasattr(_fit_jit, "_cache_size"), "_fit_jit is no longer a jax.jit-wrapped callable"
    dtype = jnp.float64
    c = 1.0
    x = _ball_points(17, 32, 5, c, 0.4, dtype)
    dist_NN = compute_pairwise_distances(x, Poincare(dtype=dtype), c, 0)
    x_sqnorms = jnp.sum(x**2, axis=1)
    statics = dict(
        n_components=2,
        perplexity=10.0,
        gamma=0.1,
        learning_rate=0.5,
        learning_rate_h=0.01,
        early_exaggeration=12.0,
        n_iter=50,
        exploration_n_iter=25,
        exact_cauchy=False,
    )
    _fit_jit(dist_NN, x_sqnorms, c, jax.random.PRNGKey(0), **statics)
    size_after_first = _fit_jit._cache_size()
    _fit_jit(dist_NN, x_sqnorms, c, jax.random.PRNGKey(1), **statics)
    assert _fit_jit._cache_size() == size_after_first


# --- 13. argument validation -----------------------------------------------------------
def test_validation_errors():
    dtype = jnp.float64
    c = 1.0
    P = Poincare(dtype=dtype)
    x = _ball_points(18, 10, 5, c, 0.4, dtype)
    key = jax.random.PRNGKey(0)

    with pytest.raises(ValueError, match="perplexity"):
        CoSNE(P, 2, perplexity=10.0).fit(x, c, key)  # perplexity 10 >= N=10
    with pytest.raises(ValueError, match="2D"):
        CoSNE(P, 2).fit(x[:, 0], c, key)  # 1-D input
    with pytest.raises(ValueError, match="n_components"):
        CoSNE(P, 0).fit(x, c, key)  # n_components < 1
    with pytest.raises(ValueError, match=r"Poincare|Hyperboloid"):
        CoSNE(Euclidean(dtype=dtype), 2)  # unsupported manifold


# --- 14. dtype preservation ------------------------------------------------------------
def test_dtype_preserved():
    dtype = jnp.float32
    c = 1.0
    x = _ball_points(19, 24, 5, c, 0.4, dtype)
    model = CoSNE(Poincare(dtype=dtype), 2, perplexity=10.0, n_iter=100, exploration_n_iter=50).fit(
        x, c, jax.random.PRNGKey(0)
    )

    assert model.embedding_.dtype == jnp.float32
    assert model.losses_kl_.dtype == jnp.float32
    assert model.losses_h_.dtype == jnp.float32
    assert model.kl_divergence_.dtype == jnp.float32


# --- 15. f32 joint P is not floor-dominated (sklearn float64-eps floor semantics) --------
def test_joint_probabilities_f32_sum_close_to_one():
    """In float32 the P floor is float64 eps (sklearn's MACHINE_EPSILON), so ``ΣP ≈ 1``.

    Regression: flooring at float32 eps (1.19e-7) put most off-diagonal entries of a large-N
    joint P at the floor (measured ~80% at N=1000, ``ΣP ≈ 1.08``) — spurious uniform attraction
    mass that silently compresses f32 embeddings relative to f64.
    """
    dtype = jnp.float32
    c = 1.0
    x = _ball_points(20, 512, 5, c, 0.4, dtype)
    dist_NN = compute_pairwise_distances(x, Poincare(dtype=dtype), c, 0)
    p = np.array(joint_probabilities(dist_NN, 10.0))
    assert p.sum() == pytest.approx(1.0, abs=2e-3)


# --- 16. bisection stays finite with exact-zero distances and large n_steps -------------
def test_conditional_probabilities_exact_zero_distances_finite():
    """β doubling is clamped: exact-zero off-diagonal distances must not produce NaN.

    A precomputed distance matrix (the ``fit_cosne`` workflow) can contain exact zeros for
    duplicate points. A duplicate pair bounds the row entropy below by ln 2 > ln 1.5, so the
    target is unreachable and β doubles every step. Regression: unclamped, float32 β overflowed
    to inf within ~128 doublings and ``exp(-0·inf)`` produced NaN rows. Tie-dominated rows may
    legitimately normalize slightly under 1 (denormal-precision division near the underflow
    boundary) or underflow to all-zero — the contract is finiteness and no mass blow-up.
    """
    dtype = jnp.float32
    c = 1.0
    x = _ball_points(21, 12, 3, c, 0.4, dtype)
    dist_np = np.array(compute_pairwise_distances(x, Poincare(dtype=dtype), c, 0))
    dist_np[0, 1] = dist_np[1, 0] = 0.0  # exact duplicate pair, as a precomputed matrix would have
    cond = np.array(conditional_probabilities(jnp.asarray(dist_np, dtype=dtype), 1.5, n_steps=200))

    assert np.isfinite(cond).all()
    row_sums = cond.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-5).all() and (row_sums >= 0.0).all()
    # The duplicate rows themselves stay properly normalized (the pair dominates at the β cap).
    assert row_sums[0] == pytest.approx(1.0, abs=1e-5)
    assert row_sums[1] == pytest.approx(1.0, abs=1e-5)
