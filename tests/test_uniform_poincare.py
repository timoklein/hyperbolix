"""Tests for Riemannian-uniform distribution on Poincaré geodesic ball."""

import jax
import jax.numpy as jnp
import pytest

from hyperbolix.distributions import uniform_poincare
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
