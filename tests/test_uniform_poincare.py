"""Tests for Riemannian-uniform distribution on Poincaré geodesic ball.

Dimension key:
  N: number of samples    D: spatial/manifold dimension (n)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from hyperbolix.distributions import uniform_poincare
from hyperbolix.distributions.uniform_poincare import _sample_uniform_direction
from hyperbolix.manifolds.poincare import Poincare


# ---------------------------------------------------------------------------
# Volume tests
#
# No dtype axis here: ``volume`` casts ``c`` to float64 internally and takes no array
# input, so the float32 and float64 parametrizations ran byte-identical code.
# ---------------------------------------------------------------------------
def test_volume_n2_c1_exact():
    """n=2, c=1: exact formula Vol = 2π(cosh(R) - 1)."""
    R = 2.0
    vol = uniform_poincare.volume(c=1.0, n=2, R=R)
    expected = 2.0 * jnp.pi * (jnp.cosh(R) - 1.0)
    assert jnp.allclose(vol, expected, atol=1e-7), f"vol={vol}, expected={expected}"


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_volume_n3_exact(c):
    """n=3: exact formula Vol = (π/c^{3/2})·(sinh(2√c·R) - 2√c·R), including c ≠ 1.

    Every other value-level volume assertion runs at c = 1, where the ``1/√c^{n-1}``
    curvature scaling and (at n = 2) the ``ω_{n-1} = 2π^{n/2}/Γ(n/2)`` sphere area are both
    exactly 1 — deleting either factor from ``volume`` passed the whole file.
    """
    R = 1.5
    sqrt_c = jnp.sqrt(jnp.float64(c))
    expected = jnp.pi / c**1.5 * (jnp.sinh(2.0 * sqrt_c * R) - 2.0 * sqrt_c * R)
    vol = uniform_poincare.volume(c=c, n=3, R=R)
    assert jnp.allclose(vol, expected, rtol=1e-9), f"vol={vol}, expected={expected}"


def test_volume_monotone_in_R():
    """Volume increases with R."""
    vols = [float(uniform_poincare.volume(c=1.0, n=3, R=r)) for r in [0.5, 1.0, 2.0, 4.0]]
    for i in range(len(vols) - 1):
        assert vols[i] < vols[i + 1], f"Volume not monotone: {vols}"


# ---------------------------------------------------------------------------
# Sample shape tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5])
def test_sample_shape(n, dtype):
    """Output shape matches sample_shape + (n,)."""
    key = jax.random.PRNGKey(42)
    samples = uniform_poincare.sample(key, n=n, c=1.0, R=1.5, sample_shape=(50,), dtype=dtype)
    assert samples.shape == (50, n)
    assert samples.dtype == dtype


def test_sample_single(dtype):
    """Single sample (no sample_shape) returns shape (n,); jit reproduces the eager draw."""
    key = jax.random.PRNGKey(0)
    manifold = Poincare(dtype=dtype)
    x = uniform_poincare.sample(key, n=3, c=1.0, R=1.0, dtype=dtype, manifold_module=manifold)
    assert x.shape == (3,)

    # Folded in from the former standalone test_sample_jit, which only re-asserted the shape.
    @jax.jit
    def _sample(k):
        return uniform_poincare.sample(k, n=3, c=1.0, R=1.0, dtype=dtype, manifold_module=manifold)

    assert jnp.allclose(_sample(key), x, atol=1e-6)


def test_sample_batch_shape(dtype):
    """Multi-dimensional sample_shape works."""
    key = jax.random.PRNGKey(1)
    samples = uniform_poincare.sample(key, n=2, c=1.0, R=1.0, sample_shape=(4, 5), dtype=dtype)
    assert samples.shape == (4, 5, 2)


# ---------------------------------------------------------------------------
# Manifold validity tests
# ---------------------------------------------------------------------------
def test_samples_in_poincare_ball(dtype):
    """All samples satisfy ||x|| < 1/√c."""
    key = jax.random.PRNGKey(7)
    c = 0.5
    samples = uniform_poincare.sample(key, n=3, c=c, R=2.0, sample_shape=(500,), dtype=dtype)
    norms = jnp.sqrt(jnp.sum(samples**2, axis=-1))
    ball_radius = 1.0 / jnp.sqrt(c)
    assert jnp.all(norms < ball_radius), f"Max norm {jnp.max(norms)} >= ball radius {ball_radius}"


@pytest.mark.parametrize("c", [0.1, 0.5, 1.0, 2.0])
def test_samples_within_geodesic_ball(c, dtype):
    """All samples have geodesic distance ≤ R from center, across curvatures.

    Absorbs the former ``test_sample_multiple_curvatures``, which asserted the same
    containment plus a shape already pinned by ``test_sample_shape``.
    """
    key = jax.random.PRNGKey(8)
    manifold = Poincare(dtype=dtype)
    R = 1.5
    samples = uniform_poincare.sample(key, n=3, c=c, R=R, sample_shape=(500,), dtype=dtype, manifold_module=manifold)
    assert samples.shape == (500, 3)
    dists = jax.vmap(lambda x: manifold.dist_0(x, c))(samples)
    # Allow small numerical tolerance
    assert jnp.all(dists <= R + 1e-5), f"Max dist {jnp.max(dists)} > R={R}"


def test_samples_within_geodesic_ball_nonorigin_center(dtype):
    """Samples around non-origin center stay within geodesic ball."""
    key = jax.random.PRNGKey(9)
    manifold = Poincare(dtype=dtype)
    c, R = 1.0, 1.0
    center = jnp.array([0.3, 0.1], dtype=dtype)
    samples = uniform_poincare.sample(
        key, n=2, c=c, R=R, sample_shape=(500,), center=center, dtype=dtype, manifold_module=manifold
    )
    dists = jax.vmap(lambda x: manifold.dist(x, center, c))(samples)
    assert jnp.all(dists <= R + 1e-5), f"Max dist {jnp.max(dists)} > R={R}"


# ---------------------------------------------------------------------------
# Empirical uniformity (n=2): radial CDF check
# ---------------------------------------------------------------------------
def test_radial_cdf_n2(dtype, tolerance):
    """Radial CDF matches theoretical F(r) = (cosh(√c·r)-1)/(cosh(√c·R)-1)."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    c, R = 1.0, 2.0
    n_samples = 5000

    samples = uniform_poincare.sample(key, n=2, c=c, R=R, sample_shape=(n_samples,), dtype=dtype, manifold_module=manifold)
    dists = jax.vmap(lambda x: manifold.dist_0(x, c))(samples)

    # Theoretical CDF: F(r) = (cosh(√c·r) - 1) / (cosh(√c·R) - 1)
    sqrt_c = jnp.sqrt(c)
    normalizer = jnp.cosh(sqrt_c * R) - 1.0

    # Check at several quantile points
    for q in [0.25, 0.5, 0.75]:
        # Theoretical quantile: r_q where F(r_q) = q
        # cosh(√c·r_q) = 1 + q·(cosh(√c·R) - 1)
        r_q = jnp.acosh(1.0 + q * normalizer) / sqrt_c
        empirical_cdf = jnp.mean(dists <= r_q)
        # KS-style tolerance: allow some slack for finite samples
        assert abs(float(empirical_cdf) - q) < 0.05, f"CDF mismatch at q={q}: empirical={float(empirical_cdf):.3f}"


# ---------------------------------------------------------------------------
# log_prob tests
# ---------------------------------------------------------------------------
def test_log_prob_constant_inside(dtype, tolerance):
    """log_prob is constant for all points inside the ball."""
    key = jax.random.PRNGKey(99)
    manifold = Poincare(dtype=dtype)
    c, R = 1.0, 1.5
    samples = uniform_poincare.sample(key, n=3, c=c, R=R, sample_shape=(100,), dtype=dtype, manifold_module=manifold)
    lps = uniform_poincare.log_prob(samples, c=c, R=R, manifold_module=manifold)
    # All values should be identical
    atol, _ = tolerance
    assert jnp.allclose(lps, lps[0], atol=atol), f"log_prob not constant: std={jnp.std(lps)}"


def test_log_prob_equals_neg_log_volume(dtype, tolerance):
    """log_prob inside ball equals -log(volume)."""
    atol, _ = tolerance
    manifold = Poincare(dtype=dtype)
    c, R, n = 1.0, 1.5, 3
    x = jnp.zeros(n, dtype=dtype)  # Origin is inside the ball
    lp = uniform_poincare.log_prob(x, c=c, R=R, manifold_module=manifold)
    vol = uniform_poincare.volume(c=c, n=n, R=R)
    expected = -jnp.log(vol)
    assert jnp.allclose(lp, expected, atol=atol), f"lp={lp}, expected={expected}"

    # Folded in from the former standalone test_log_prob_jit, which only asserted isfinite.
    lp_jit = jax.jit(lambda xi: uniform_poincare.log_prob(xi, c=c, R=R, manifold_module=manifold))(x)
    assert jnp.allclose(lp_jit, lp, atol=atol)


def test_log_prob_neg_inf_outside(dtype):
    """log_prob is -inf for points outside the geodesic ball."""
    manifold = Poincare(dtype=dtype)
    c, R = 1.0, 0.5
    # A point far from origin (but still in Poincaré ball)
    x = jnp.array([0.8, 0.0], dtype=dtype)
    lp = uniform_poincare.log_prob(x, c=c, R=R, manifold_module=manifold)
    assert lp == -jnp.inf, f"Expected -inf, got {lp}"


def test_log_prob_nonorigin_center(dtype, tolerance):
    """log_prob works with non-origin center."""
    atol, _ = tolerance
    manifold = Poincare(dtype=dtype)
    c, R = 1.0, 1.0
    center = jnp.array([0.3, 0.0], dtype=dtype)

    # Center itself should be inside
    lp_center = uniform_poincare.log_prob(center, c=c, R=R, center=center, manifold_module=manifold)
    vol = uniform_poincare.volume(c=c, n=2, R=R)
    assert jnp.allclose(lp_center, -jnp.log(vol), atol=atol)

    # A point far from center should be outside
    far_point = jnp.array([-0.8, 0.0], dtype=dtype)
    lp_far = uniform_poincare.log_prob(far_point, c=c, R=R, center=center, manifold_module=manifold)
    assert lp_far == -jnp.inf


def test_log_prob_batch(dtype, tolerance):
    """log_prob handles batched inputs."""
    atol, _ = tolerance
    manifold = Poincare(dtype=dtype)
    c, R = 1.0, 1.5
    key = jax.random.PRNGKey(55)
    samples = uniform_poincare.sample(key, n=2, c=c, R=R, sample_shape=(20,), dtype=dtype, manifold_module=manifold)
    lps = uniform_poincare.log_prob(samples, c=c, R=R, manifold_module=manifold)
    assert lps.shape == (20,)
    vol = uniform_poincare.volume(c=c, n=2, R=R)
    expected = -jnp.log(vol)
    assert jnp.allclose(lps, expected, atol=atol)


# =============================================================================================
# Radial and angular laws (audit M2-04, M2-11, M2-14)
#
# ``test_radial_cdf_n2`` above only touches the n = 2 closed-form branch, at three quantiles.
# The rejection branch (n >= 3), the non-origin-center path and the direction sampler had no
# distributional check at all: a sampler that returned the *proposal* instead of the accepted
# value, ignored ``center``, or drew directions from a lattice would pass every containment,
# shape and dtype assertion in this file.
#
# Fixed seeds and float64 throughout: the assertions are one-sided KS bounds, not p-value
# lotteries.
# =============================================================================================

_F64 = jnp.float64
_KS_N = 8000
_KS_MAX_D = 0.04
"""One-sided KS acceptance bound. At N = 8000 the observed null statistics are 0.008-0.016 and
0.04 is the p ~ 1e-11 point, while the nearest *wrong* law in each family (the n+1 radial
density, the Beta with a+1) gives 0.05-0.14 — a comfortable margin on both sides."""


def _radial_cdf_reference(r_N: np.ndarray, c: float, R: float, n: int) -> np.ndarray:
    """CDF of the geodesic radius under the Riemannian-uniform law: ∫ sinh^{n-1}(√c t) dt, normalized.

    Independent dense-trapezoid transcription of the radial density (the library computes its
    volume by 64-point Gauss-Legendre and never forms this CDF).
    """
    grid_Q = np.linspace(0.0, R, 40001)
    density_Q = np.sinh(np.sqrt(c) * grid_Q) ** (n - 1)
    cumulative_Q = np.concatenate([[0.0], np.cumsum(0.5 * (density_Q[1:] + density_Q[:-1]) * np.diff(grid_Q))])
    return np.interp(np.clip(r_N, 0.0, R), grid_Q, cumulative_Q / cumulative_Q[-1])


@pytest.mark.parametrize(("n", "c", "R"), [(3, 1.0, 2.0), (5, 0.4, 2.5), (8, 2.0, 1.5)])
def test_radial_law_matches_reference_cdf_rejection_branch(n: int, c: float, R: float) -> None:
    """n >= 3 (rejection sampler): the geodesic radii follow p(r) ∝ sinh^{n-1}(√c·r) on [0, R].

    Discriminating case: the acceptance exponent ``(n-2)/2``. Dropping it turns the radial law
    into the n = 2 one, which this KS bound rejects by an order of magnitude, while every other
    test in the file (containment, shape, dtype, log_prob) stays green.
    """
    manifold = Poincare(dtype=_F64)
    samples_ND = uniform_poincare.sample(
        jax.random.PRNGKey(20), n=n, c=c, R=R, sample_shape=(_KS_N,), dtype=_F64, manifold_module=manifold
    )
    radii_N = np.asarray(jax.vmap(lambda x_D: manifold.dist_0(x_D, c))(samples_ND))
    statistic = scipy.stats.kstest(radii_N, lambda r: _radial_cdf_reference(r, c, R, n)).statistic
    assert statistic < _KS_MAX_D, f"KS statistic {statistic:.4f} against p(r) ∝ sinh^{n - 1}(√c·r)"


@pytest.mark.parametrize(("n", "c", "R"), [(2, 1.0, 1.0), (3, 0.5, 2.0)])
def test_radial_law_around_nonorigin_center(n: int, c: float, R: float) -> None:
    """With a non-origin center the radii *measured from that center* follow the same law.

    The Möbius translation ``center ⊕ ·`` is an isometry, so it must move the ball without
    reshaping the radial profile. The origin-referenced control below is what makes this a real
    check of ``center``: a sampler that ignored the argument would still pass the recentered
    bound if the center were near the origin, but its origin-referenced statistic would stay
    small instead of blowing up.
    """
    manifold = Poincare(dtype=_F64)
    center_D = jnp.asarray([0.3, -0.15] + [0.1] * (n - 2), dtype=_F64) / np.sqrt(c)
    samples_ND = uniform_poincare.sample(
        jax.random.PRNGKey(21),
        n=n,
        c=c,
        R=R,
        sample_shape=(_KS_N,),
        center=center_D,
        dtype=_F64,
        manifold_module=manifold,
    )

    radii_N = np.asarray(jax.vmap(lambda x_D: manifold.dist(x_D, center_D, c))(samples_ND))
    statistic = scipy.stats.kstest(radii_N, lambda r: _radial_cdf_reference(r, c, R, n)).statistic
    assert statistic < _KS_MAX_D, f"KS statistic {statistic:.4f} for radii around the center"

    radii_from_origin_N = np.asarray(jax.vmap(lambda x_D: manifold.dist_0(x_D, c))(samples_ND))
    origin_statistic = scipy.stats.kstest(radii_from_origin_N, lambda r: _radial_cdf_reference(r, c, R, n)).statistic
    assert origin_statistic > 0.1, f"origin-referenced radii also match ({origin_statistic:.4f}): center had no effect"


@pytest.mark.parametrize("n", [3, 5, 10])
def test_direction_marginal_matches_beta(n: int) -> None:
    """Directions are uniform on S^{n-1}: (u₁+1)/2 follows Beta((n-1)/2, (n-1)/2).

    A moment oracle is provably vacuous here — E[u] = 0 and E[u uᵀ] = I/n hold for *any*
    sign-symmetric direction law, including one that only ever returns coordinate axes — so the
    exact first-coordinate marginal is what pins the distribution. Density of u₁ on [-1, 1] is
    ∝ (1 - u₁²)^{(n-3)/2}; substituting t = (u₁+1)/2 gives the symmetric Beta above.
    """
    directions_ND = np.asarray(_sample_uniform_direction(jax.random.PRNGKey(22), n, (_KS_N,), _F64))
    assert np.max(np.abs(np.linalg.norm(directions_ND, axis=-1) - 1.0)) < 1e-12

    shape_param = (n - 1) / 2.0
    first_coordinate_N = (directions_ND[:, 0] + 1.0) / 2.0
    statistic = scipy.stats.kstest(first_coordinate_N, "beta", args=(shape_param, shape_param)).statistic
    assert statistic < _KS_MAX_D, f"KS statistic {statistic:.4f} against Beta({shape_param}, {shape_param})"


# =============================================================================================
# Rejection-sampler termination (audit M2-15)
# =============================================================================================


@pytest.mark.parametrize(("n", "R"), [(3, 8.0), (20, 4.0), (50, 8.0), (50, 1.0)])
def test_rejection_sampler_terminates_in_high_dimension(n: int, R: float) -> None:
    """``_sample_radial_rejection`` finishes and returns valid radii up to n = 50, R = 8.

    The loop has no iteration cap: it runs until *every* position is filled, with per-proposal
    acceptance ``(u(u+2)/(u_max(u_max+2)))^{(n-2)/2}`` (~1/(n-1)). A tightened acceptance rule,
    or a reference bound the numerator can never reach, would hang the caller rather than
    raising — so termination itself is the assertion. The loose KS bound rules out the
    degenerate escapes (returning the proposal, or a single accepted value broadcast).
    """
    c = 1.0
    n_samples = 256
    manifold = Poincare(dtype=_F64)
    samples_ND = uniform_poincare.sample(
        jax.random.PRNGKey(23), n=n, c=c, R=R, sample_shape=(n_samples,), dtype=_F64, manifold_module=manifold
    )
    assert samples_ND.shape == (n_samples, n)
    assert bool(jnp.isfinite(samples_ND).all())

    radii_N = np.asarray(jax.vmap(lambda x_D: manifold.dist_0(x_D, c))(samples_ND))
    assert np.all(radii_N <= R + 1e-9), f"max radius {radii_N.max()} exceeds R={R}"
    assert np.all(radii_N > 0.0)
    # Loose distributional sanity: at 256 samples the null KS statistic is ~0.05 and 0.20 is
    # roughly the p = 1e-3 point. (The sharp version of this check is the 8000-sample test above;
    # here the point is termination, so the bound only has to exclude a degenerate output.)
    statistic = scipy.stats.kstest(radii_N, lambda r: _radial_cdf_reference(r, c, R, n)).statistic
    assert statistic < 0.2, f"KS statistic {statistic:.4f}: radii do not follow the target law"
