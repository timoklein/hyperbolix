"""Tests for Riemannian-uniform distribution on Poincaré geodesic ball."""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.distributions import uniform_poincare
from hyperbolix.manifolds.poincare import Poincare


# ---------------------------------------------------------------------------
# Volume tests
# ---------------------------------------------------------------------------
def test_volume_n2_c1_exact(dtype, tolerance):
    """n=2, c=1: exact formula Vol = 2π(cosh(R) - 1)."""
    atol, _ = tolerance
    R = 2.0
    vol = uniform_poincare.volume(c=1.0, n=2, R=R)
    expected = 2.0 * jnp.pi * (jnp.cosh(R) - 1.0)
    assert jnp.allclose(vol, expected, atol=atol), f"vol={vol}, expected={expected}"


def test_volume_monotone_in_R(dtype):
    """Volume increases with R."""
    vols = [float(uniform_poincare.volume(c=1.0, n=3, R=r)) for r in [0.5, 1.0, 2.0, 4.0]]
    for i in range(len(vols) - 1):
        assert vols[i] < vols[i + 1], f"Volume not monotone: {vols}"


def test_volume_positive(dtype):
    """Volume is positive for R > 0."""
    for n in [2, 3, 5]:
        for c in [0.1, 1.0, 2.0]:
            vol = uniform_poincare.volume(c=c, n=n, R=1.0)
            assert vol > 0, f"Volume non-positive for n={n}, c={c}: {vol}"


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
    """Single sample (no sample_shape) returns shape (n,)."""
    key = jax.random.PRNGKey(0)
    x = uniform_poincare.sample(key, n=3, c=1.0, R=1.0, dtype=dtype)
    assert x.shape == (3,)


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


def test_samples_within_geodesic_ball(dtype):
    """All samples have geodesic distance ≤ R from center."""
    key = jax.random.PRNGKey(8)
    manifold = Poincare(dtype=dtype)
    c, R = 1.0, 1.5
    samples = uniform_poincare.sample(key, n=3, c=c, R=R, sample_shape=(500,), dtype=dtype, manifold_module=manifold)
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
# Multiple curvatures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c", [0.1, 0.5, 1.0, 2.0])
def test_sample_multiple_curvatures(c, dtype):
    """Sampling works across curvature values."""
    key = jax.random.PRNGKey(10)
    manifold = Poincare(dtype=dtype)
    R = 1.0
    samples = uniform_poincare.sample(key, n=3, c=c, R=R, sample_shape=(100,), dtype=dtype, manifold_module=manifold)
    assert samples.shape == (100, 3)
    dists = jax.vmap(lambda x: manifold.dist_0(x, c))(samples)
    assert jnp.all(dists <= R + 1e-5)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------
def test_sample_jit(dtype):
    """sample is JIT-compatible."""
    manifold = Poincare(dtype=dtype)

    @jax.jit
    def _sample(key):
        return uniform_poincare.sample(key, n=2, c=1.0, R=1.0, sample_shape=(10,), dtype=dtype, manifold_module=manifold)

    samples = _sample(jax.random.PRNGKey(0))
    assert samples.shape == (10, 2)


def test_log_prob_jit(dtype):
    """log_prob is JIT-compatible."""
    manifold = Poincare(dtype=dtype)

    @jax.jit
    def _lp(x):
        return uniform_poincare.log_prob(x, c=1.0, R=1.0, manifold_module=manifold)

    x = jnp.zeros(2, dtype=dtype)
    lp = _lp(x)
    assert jnp.isfinite(lp)


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
