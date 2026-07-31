"""Tests for HoroPCA (``hyperbolix.decomposition.horopca``).

Correctness anchors are geometric invariants with independent oracles — never the components
themselves, which are frame-dependent (QR sign) and vary across seeds/BLAS:

- ``lift_ideals`` produces exact null vectors; the Lorentz boost sends the mean to the origin
  and is an isometry.
- **Defining invariant**: the horospherical projection preserves every Busemann coordinate
  (oracle: the independent ``Hyperboloid.busemann``).
- The projection lands on the manifold, is idempotent, fixes origin-side geodesic points, and
  its spatial part lies in the component span.
- The loss has finite gradients (including at coincident pairs) and decreases; the fit JIT is
  reused; input/output models match; Poincaré and hyperboloid inputs give isometric outputs;
  centering preserves pairwise variance; a planted 2-D submanifold is recovered end-to-end.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.decomposition import HoroPCA, fit_horopca, frechet_mean, horo_projection, horopca_loss
from hyperbolix.decomposition.horopca import _fit_jit, lift_ideals, orthonormalize_rows
from hyperbolix.distributions import wrapped_normal_hyperboloid as wn
from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.manifolds import isometry_mappings as iso
from hyperbolix.manifolds.hyperboloid import VERSION_DEFAULT
from hyperbolix.utils.helpers import compute_pairwise_distances

# One seed suffices for the geometric identities below: each item already checks the identity
# on 16-24 generic points, and ``B @ mean == origin`` / ``⟨p_k, p_k⟩_L = 0`` / Busemann
# preservation are algebraic properties of the constructed boost/lift, not data-dependent ones.
# The dtype and curvature axes stay (tolerance path / curvature scaling).
SEEDS = [10]
DIMS = [2, 5, 10]
DIMS_GE3 = [5, 10]
CURVATURES = [0.3, 1.0, 2.5]


def _tol(dtype) -> float:
    return 4e-3 if dtype == jnp.float32 else 1e-7


def _hyp_points(seed: int, n: int, dim: int, c: float, sigma: float, dtype) -> jnp.ndarray:
    """Wrapped-normal samples on the hyperboloid, shape (n, dim+1)."""
    H = Hyperboloid(dtype=dtype)
    mu0 = H.create_origin(c, dim)
    return wn.sample(jax.random.PRNGKey(seed), mu0, jnp.asarray(sigma, dtype=dtype), c, sample_shape=(n,), manifold_module=H)


def _ortho_q(seed: int, k: int, dim: int, dtype) -> jnp.ndarray:
    """Row-orthonormal ideal directions, shape (k, dim)."""
    q = jax.random.normal(jax.random.PRNGKey(seed), (k, dim), dtype=dtype)
    return orthonormalize_rows(q)


def _busemann_coords(H: Hyperboloid, pts_NA, q_KD, c) -> jnp.ndarray:
    """All Busemann coordinates B^{q_k}(x_i), shape (N, K)."""
    return jax.vmap(lambda p: jax.vmap(H.busemann, in_axes=(None, 0, None))(p, q_KD, c))(pts_NA)


# --- 6. lift_ideals --------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("dim", [5])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_lift_ideals_are_null_vectors(dtype, dim, k):
    """After orthonormalization, lifted ideals [1, q_k] are null (⟨p_k, p_k⟩_L = 0).

    ``-1 + ‖q‖² = 0`` does not depend on ``dim`` beyond ``orthonormalize_rows`` succeeding,
    so a single ambient dimension is enough.
    """
    atol = _tol(dtype)
    H = Hyperboloid(dtype=dtype)
    q = _ortho_q(0, k, dim, dtype)
    p = lift_ideals(q)
    inners = jax.vmap(lambda pk: H.minkowski_inner(pk, pk))(p)
    assert jnp.allclose(inners, 0.0, atol=atol)


# --- 7. boost sends mean to origin -----------------------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("seed", SEEDS)
def test_boost_sends_mean_to_origin(dtype, c, dim, seed):
    """B @ mean == origin."""
    atol = _tol(dtype)
    H = Hyperboloid(dtype=dtype)
    x = _hyp_points(seed, 24, dim, c, sigma=0.4, dtype=dtype)
    mean = frechet_mean(x, H, c)
    boost = H.lorentz_boost(mean, c)
    origin = H.create_origin(c, dim)
    assert jnp.allclose(boost @ mean, origin, atol=atol)


# --- 8. boost is an isometry -----------------------------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("seed", SEEDS)
def test_boost_is_isometry(dtype, c, seed):
    """Boosting preserves pairwise geodesic distances and keeps points on the manifold."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    H = Hyperboloid(dtype=dtype)
    x = _hyp_points(seed, 20, 5, c, sigma=0.5, dtype=dtype)
    mean = frechet_mean(x, H, c)
    boost = H.lorentz_boost(mean, c)
    x_boosted = H.proj_batch(x @ boost.T, c)

    d_before = compute_pairwise_distances(x, H, c, VERSION_DEFAULT)
    d_after = compute_pairwise_distances(x_boosted, H, c, VERSION_DEFAULT)
    assert jnp.allclose(d_before, d_after, atol=atol)
    assert bool(jax.vmap(H.is_in_manifold, in_axes=(0, None))(x_boosted, c).all())


# --- 9. defining invariant: Busemann coordinates preserved -----------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", DIMS_GE3)
@pytest.mark.parametrize("seed", SEEDS)
def test_horo_projection_preserves_busemann(dtype, c, dim, seed):
    """B^{q_k}(π(x)) == B^{q_k}(x) for every component (oracle: independent Hyperboloid.busemann)."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    H = Hyperboloid(dtype=dtype)
    x = _hyp_points(seed, 16, dim, c, sigma=0.5, dtype=dtype)
    for k in (2, 3):
        q = _ortho_q(seed + k, k, dim, dtype)
        proj = jax.vmap(horo_projection, in_axes=(0, None, None, None))(x, q, c, VERSION_DEFAULT)
        b_x = _busemann_coords(H, x, q, c)
        b_proj = _busemann_coords(H, proj, q, c)
        assert jnp.allclose(b_x, b_proj, atol=atol)


# --- 10-11-13. projection properties (shared setup) ------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_horo_projection_properties(dtype, c, k):
    """π lands on the manifold, is idempotent, and its spatial part lies in span(Q).

    The three properties shared one setup (``_hyp_points`` + ``_ortho_q`` + a vmapped
    ``horo_projection``) across three separate tests; merged here to run that setup once.
    """
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    H = Hyperboloid(dtype=dtype)
    dim = 6
    x = _hyp_points(10, 16, dim, c, sigma=0.5, dtype=dtype)
    q = _ortho_q(1, k, dim, dtype)
    proj = jax.vmap(horo_projection, in_axes=(0, None, None, None))(x, q, c, VERSION_DEFAULT)

    # (a) π(x) is a valid hyperboloid point.
    assert bool(jax.vmap(H.is_in_manifold, in_axes=(0, None))(proj, c).all())

    # (b) π(π(x)) == π(x).
    proj2 = jax.vmap(horo_projection, in_axes=(0, None, None, None))(proj, q, c, VERSION_DEFAULT)
    assert jnp.allclose(proj, proj2, atol=atol)

    # (c) The spatial part lies in the row span of q (q rows orthonormal ⇒ residual ≈ 0).
    spatial = proj[:, 1:]  # (N, D)
    residual = spatial - (spatial @ q.T) @ q
    assert jnp.allclose(residual, 0.0, atol=atol)


# --- 12. fixed points on the origin side -----------------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
def test_horo_projection_fixes_origin_side_geodesic(dtype, c):
    """Points built as expmap(s·t̂, spine, c) with s>0 (origin side) are fixed by π.

    ``t̂`` is the origin-ward unit tangent at the spine, orthogonal to the ideal span; the
    spine sits on the geodesic g = H ∩ span(p). Only the origin side is fixed — the far side
    (s<0) mirrors across the spine (the two-component caveat), so we test s>0 only.
    """
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    H = Hyperboloid(dtype=dtype)
    dim = 5
    q = _ortho_q(4, 2, dim, dtype)
    p = lift_ideals(q)  # (2, A)

    # Spine on g: an all-positive combination in span(p) is timelike; normalize onto H.
    coeff = jnp.asarray([0.7, 1.3], dtype=dtype)
    mp = coeff @ p
    spine = mp / jnp.sqrt(-c * H.minkowski_inner(mp, mp))
    spine = spine * jnp.sign(spine[0])

    # Origin-ward unit tangent (independent reimplementation of the projection's step 2).
    origin = H.create_origin(c, dim)
    bo = -origin[0] + q @ origin[1:]
    coeffs_o = bo + (jnp.sum(bo) / (1.0 - 2)) * jnp.ones(2, dtype=dtype)
    t = origin - coeffs_o @ p
    t_hat = t / jnp.sqrt(H.minkowski_inner(t, t))

    for s in (0.2, 0.5, 1.0):
        candidate = H.expmap(jnp.asarray(s, dtype=dtype) * t_hat, spine, c)
        proj = horo_projection(candidate, q, c, VERSION_DEFAULT)
        assert jnp.allclose(proj, candidate, atol=atol)


# --- 14. K=1 closed form ---------------------------------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
def test_horo_projection_k1(dtype, c):
    """K=1: Busemann coordinate preserved and the projected spatial part is parallel to q."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    H = Hyperboloid(dtype=dtype)
    dim = 5
    x = _hyp_points(13, 16, dim, c, sigma=0.5, dtype=dtype)
    q = _ortho_q(5, 1, dim, dtype)  # (1, dim)
    proj = jax.vmap(horo_projection, in_axes=(0, None, None, None))(x, q, c, VERSION_DEFAULT)

    q_vec = q[0]
    b_x = jax.vmap(H.busemann, in_axes=(0, None, None))(x, q_vec, c)
    b_proj = jax.vmap(H.busemann, in_axes=(0, None, None))(proj, q_vec, c)
    assert jnp.allclose(b_x, b_proj, atol=atol)

    # Spatial part ∝ q: the component orthogonal to q vanishes.
    spatial = proj[:, 1:]
    ortho = spatial - jnp.outer(spatial @ q_vec, q_vec)
    assert jnp.allclose(ortho, 0.0, atol=atol)


# --- 15. loss gradients finite (incl. coincident pair) and loss decreases --------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_loss_grad_finite_and_decreases(dtype, k):
    """Loss gradient is finite with a duplicated row (coincident-pair path) and the fit decreases it."""
    c = 1.0
    dim = 5
    x = _hyp_points(14, 16, dim, c, sigma=0.5, dtype=dtype)
    x = x.at[1].set(x[0])  # exact duplicate → coincident projected pair

    q0 = jax.random.normal(jax.random.PRNGKey(7), (k, dim), dtype=dtype)
    grad = jax.grad(horopca_loss)(q0, x, c)
    assert bool(jnp.isfinite(grad).all())

    _, losses = fit_horopca(x, c, jax.random.PRNGKey(8), n_components=k, lr=1e-2, max_steps=40)
    assert bool(jnp.isfinite(losses).all())
    assert float(losses[-1]) < float(losses[0])


# --- 16. fit JIT is reused -------------------------------------------------------------
def test_fit_jit_cache_reuse():
    """A second _fit_jit call with identical statics/shapes/dtypes does not recompile."""
    # Asserted, not skipped: removing the jax.jit wrapper from _fit_jit — the exact
    # regression this test names — makes the attribute vanish, and a skip would report
    # that as a green run.
    assert hasattr(_fit_jit, "_cache_size"), "_fit_jit is no longer a jax.jit-wrapped callable"
    dtype = jnp.float64
    c = 1.0
    x = _hyp_points(15, 32, 5, c, sigma=0.5, dtype=dtype)
    key = jax.random.PRNGKey(0)
    _fit_jit(x, c, key, n_components=2, max_steps=20)
    size_after_first = _fit_jit._cache_size()
    _fit_jit(x, c, jax.random.PRNGKey(1), n_components=2, max_steps=20)
    assert _fit_jit._cache_size() == size_after_first


# --- 17. shapes and output models ------------------------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("k", [1, 2])
def test_transform_shapes_and_models(dtype, k):
    """Hyperboloid in → (N, K+1) on-manifold; Poincaré in → (N, K) inside the ball."""
    c = 1.0
    dim = 5
    n = 20
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    x_hyp = _hyp_points(16, n, dim, c, sigma=0.5, dtype=dtype)
    x_ball = jax.vmap(iso.hyperboloid_to_poincare, in_axes=(0, None))(x_hyp, c)
    key = jax.random.PRNGKey(0)

    z_hyp = HoroPCA(H, k, max_steps=30).fit_transform(x_hyp, c, key)
    assert z_hyp.shape == (n, k + 1)
    assert bool(jax.vmap(H.is_in_manifold, in_axes=(0, None))(z_hyp, c).all())

    z_ball = HoroPCA(P, k, max_steps=30).fit_transform(x_ball, c, key)
    assert z_ball.shape == (n, k)
    radius = 1.0 / jnp.sqrt(jnp.asarray(c, dtype=dtype))
    assert bool((jnp.linalg.norm(z_ball, axis=1) < radius).all())


# --- 18. Poincaré / hyperboloid input equivalence --------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
def test_input_model_equivalence(dtype, c):
    """Fitting the same data as Poincaré vs hyperboloid gives isometric embeddings."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-5
    dim = 5
    H, P = Hyperboloid(dtype=dtype), Poincare(dtype=dtype)
    x_hyp = _hyp_points(17, 24, dim, c, sigma=0.5, dtype=dtype)
    x_ball = jax.vmap(iso.hyperboloid_to_poincare, in_axes=(0, None))(x_hyp, c)
    key = jax.random.PRNGKey(0)

    z_hyp = HoroPCA(H, 2, max_steps=50).fit_transform(x_hyp, c, key)
    z_ball = HoroPCA(P, 2, max_steps=50).fit_transform(x_ball, c, key)

    d_hyp = compute_pairwise_distances(z_hyp, H, c, VERSION_DEFAULT)
    d_ball = compute_pairwise_distances(z_ball, P, c, VERSION_DEFAULT)
    assert jnp.allclose(d_hyp, d_ball, atol=atol)


# --- 19. centering flow ----------------------------------------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", CURVATURES)
def test_center_data_flow(dtype, c):
    """mean_ satisfies the Karcher condition; total_variance_ is centering-invariant; transform matches fit_transform."""
    karcher_atol = 2e-3 if dtype == jnp.float32 else 1e-6
    var_rtol = 4e-3 if dtype == jnp.float32 else 1e-6
    out_atol = 4e-3 if dtype == jnp.float32 else 1e-6
    H = Hyperboloid(dtype=dtype)
    dim = 5
    x = _hyp_points(18, 24, dim, c, sigma=0.5, dtype=dtype)
    key = jax.random.PRNGKey(0)

    model = HoroPCA(H, 2, max_steps=50, center_data=True).fit(x, c, key)

    # mean_ is a Karcher mean of the (on-manifold) input.
    logs = jax.vmap(H.logmap, in_axes=(0, None, None))(x, model.mean_, c)
    residual = float(H.tangent_norm(jnp.mean(logs, axis=0), model.mean_, c))
    assert residual < karcher_atol

    # Pairwise variance is invariant under the isometric centering boost.
    model_nc = HoroPCA(H, 2, max_steps=50, center_data=False).fit(x, c, key)
    assert float(model.total_variance_) == pytest.approx(float(model_nc.total_variance_), rel=var_rtol)

    # transform(train) reproduces fit_transform.
    z_transform = model.transform(x)
    z_fit = HoroPCA(H, 2, max_steps=50, center_data=True).fit_transform(x, c, key)
    assert jnp.allclose(z_transform, z_fit, atol=out_atol)


# --- 20. end-to-end recovery of a planted 2-D submanifold ------------------------------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_end_to_end_recovery(dtype):
    """A wrapped-normal K=2 submanifold zero-padded to D=5 and rotated is recovered by HoroPCA.

    Optimization-limited, single seed: relaxed documented tolerances (the fit is a non-convex
    Adam run). Deviates from the plan's max_steps=300 (which plateaus at EVR≈0.91): 1000 steps
    are needed to reach the global optimum, matching the plan's target rel≈1e-2 (f64).
    """
    c = 1.0
    n, dim = 128, 5
    H = Hyperboloid(dtype=dtype)

    # Planted 2-D submanifold: wrapped normal on H², spatial-padded to 5, rotated (an isometry).
    mu0 = H.create_origin(c, 2)
    x2 = wn.sample(jax.random.PRNGKey(7), mu0, jnp.asarray(0.35, dtype=dtype), c, sample_shape=(n,), manifold_module=H)
    spatial = jnp.pad(x2[:, 1:], ((0, 0), (0, dim - 2)))
    time = jnp.sqrt(1.0 / c + jnp.sum(spatial**2, axis=1, keepdims=True))
    x_hi = jnp.concatenate([time, spatial], axis=1)
    rot, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(3), (dim, dim), dtype=dtype))
    x_rot = jax.vmap(H.proj, in_axes=(0, None))(x_hi.at[:, 1:].set(x_hi[:, 1:] @ rot.T), c)

    model = HoroPCA(H, 2, max_steps=1000, lr=1e-2).fit(x_rot, c, jax.random.PRNGKey(9))
    z = model.transform(x_rot)  # (n, 3) hyperboloid

    # Pairwise distances of the recovered 2-D embedding match the original 2-D points.
    d_orig = np.asarray(compute_pairwise_distances(x2, H, c, VERSION_DEFAULT))
    d_out = np.asarray(compute_pairwise_distances(z, H, c, VERSION_DEFAULT))
    iu = np.triu_indices(n, 1)
    rel = np.abs(d_out[iu] - d_orig[iu]) / (np.abs(d_orig[iu]) + 1e-9)

    if dtype == jnp.float32:
        assert float(model.explained_variance_ratio_) > 0.95
        assert float(np.median(rel)) < 5e-2
    else:
        assert float(model.explained_variance_ratio_) > 0.99
        assert float(np.median(rel)) < 1e-2


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_boundary_input_is_finite(dtype):
    """A Poincaré input containing an exactly on-boundary point must not NaN the fit.

    Regression: the Poincaré branch of ``_to_hyperboloid`` used to lift without ball hygiene;
    ``poincare_to_hyperboloid`` floors ``1 - c‖y‖²`` instead of erroring, so an on-boundary
    point (routine under float32 saturation) lifted to a time coordinate ~1e15 and silently
    NaN'd the Fréchet mean — and with it every fitted attribute.
    """
    c = 1.0
    P = Poincare(dtype=dtype)
    x_hyp = _hyp_points(5, 32, 5, c, sigma=0.3, dtype=dtype)
    x_ball = jax.vmap(iso.hyperboloid_to_poincare, in_axes=(0, None))(x_hyp, c)
    boundary_D = jnp.zeros(5, dtype=dtype).at[0].set(1.0 / jnp.sqrt(jnp.asarray(c, dtype=dtype)))
    x_ball = x_ball.at[0].set(boundary_D)  # exactly on the ball boundary

    model = HoroPCA(P, 2, max_steps=30).fit(x_ball, c, jax.random.PRNGKey(0))
    z = model.transform(x_ball)

    assert np.isfinite(np.asarray(model.mean_)).all()
    assert np.isfinite(np.asarray(model.components_)).all()
    assert np.isfinite(np.asarray(model.losses_)).all()
    assert np.isfinite(np.asarray(z)).all()
