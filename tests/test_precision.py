"""Module-constant surface, singular-point gradient guards, and MLR closed forms.

Historical note (audit A3-03/04/05): this file used to open with ~20 "auto-casting" tests whose
inputs were annotated ``# float32`` but were in fact float64 — ``tests/conftest.py`` enables x64
globally, so ``jnp.array([0.1, 0.2])`` is f64 and ``ManifoldBase._cast`` was never exercised
(a ``_cast → return x`` mutation left all 42 items passing). Their replacement, with explicit
``.astype`` inputs in both directions, is
``tests/test_class_based_manifolds.py::test_dtype_policy_casts_inputs_to_manifold_dtype``.
Also removed: two ``test_dist_values_match`` tests that compared ``_dist(x.astype(f64), …)`` to
itself, and ``TestVmapJitCompat``, a byte-identical clone (same literal points) of
``tests/test_class_based_manifolds.py``'s vmap/jit block, which asserts strictly more.
"""

import jax
import jax.numpy as jnp

from hyperbolix.manifolds import hyperboloid as hyperboloid_module
from hyperbolix.manifolds import poincare as poincare_module
from hyperbolix.manifolds import proper_velocity as proper_velocity_module
from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.manifolds.proper_velocity import ProperVelocity


class TestPoincareClassConstants:
    """Test that module constants are still accessible."""

    def test_module_constants(self):
        """Verify constants are available from module."""
        assert hasattr(poincare_module, "VERSION_MOBIUS_DIRECT")
        assert hasattr(poincare_module, "VERSION_MOBIUS")
        assert hasattr(poincare_module, "MIN_NORM")

    def test_hyperboloid_module_constants(self):
        """Verify hyperboloid constants are available from module."""
        assert hasattr(hyperboloid_module, "VERSION_DEFAULT")
        assert hasattr(hyperboloid_module, "VERSION_SMOOTHENED")
        assert hasattr(hyperboloid_module, "MIN_NORM")

    def test_proper_velocity_module_constants(self):
        """Verify PV constants are available from module."""
        assert hasattr(proper_velocity_module, "VERSION_DEFAULT")
        assert hasattr(proper_velocity_module, "MIN_NORM")


class TestSingularPointGradients:
    """Gradients at geometric singular points must be finite.

    Regression guards for the NaN-gradient family: acosh'(1) = inf reached
    through hard clips at exactly 1.0, and linalg.norm / sqrt VJPs at zero
    vectors (0/0 = NaN) that post-hoc ``jnp.where`` masking cannot remove.
    """

    def test_poincare_dist_grad_at_coincident_points(self):
        for dtype in [jnp.float32, jnp.float64]:
            manifold = Poincare(dtype=dtype)
            x = jnp.array([0.3, -0.4], dtype=dtype)
            for version_idx in (0, 1, 2):
                g = jax.grad(lambda a, m=manifold, y=x, v=version_idx: m.dist(a, y, 1.0, version_idx=v))(x)
                assert jnp.all(jnp.isfinite(g)), f"NaN grad: dist v{version_idx} {dtype}"
            g0 = jax.grad(lambda a, m=manifold: m.dist_0(a, 1.0))(jnp.zeros(2, dtype=dtype))
            assert jnp.all(jnp.isfinite(g0))

    def test_poincare_expmap_grad_at_zero_tangent(self):
        for dtype in [jnp.float32, jnp.float64]:
            manifold = Poincare(dtype=dtype)
            x = jnp.array([0.3, -0.4], dtype=dtype)
            g = jax.grad(lambda v, m=manifold, p=x: jnp.sum(m.expmap(v, p, 1.0)))(jnp.zeros(2, dtype=dtype))
            assert jnp.all(jnp.isfinite(g))

    def test_poincare_scalar_mul_grad_at_origin(self):
        for dtype in [jnp.float32, jnp.float64]:
            manifold = Poincare(dtype=dtype)
            g = jax.grad(lambda a, m=manifold: jnp.sum(m.scalar_mul(0.7, a, 1.0)))(jnp.zeros(2, dtype=dtype))
            assert jnp.all(jnp.isfinite(g))

    def test_hyperboloid_dist_grad_at_coincident_points(self):
        for dtype in [jnp.float32, jnp.float64]:
            manifold = Hyperboloid(dtype=dtype)
            x = manifold.proj(jnp.array([1.5, 0.3, -0.4], dtype=dtype), 1.0)
            origin = jnp.array([1.0, 0.0, 0.0], dtype=dtype)
            for version_idx in (0, 1):
                g = jax.grad(lambda a, m=manifold, y=x, v=version_idx: m.dist(a, y, 1.0, version_idx=v))(x)
                assert jnp.all(jnp.isfinite(g)), f"NaN grad: dist v{version_idx} {dtype}"
            g0 = jax.grad(lambda a, m=manifold: m.dist_0(a, 1.0))(origin)
            assert jnp.all(jnp.isfinite(g0))

    def test_hyperboloid_expmap_grad_at_zero_tangent(self):
        for dtype in [jnp.float32, jnp.float64]:
            manifold = Hyperboloid(dtype=dtype)
            x = manifold.proj(jnp.array([1.5, 0.3, -0.4], dtype=dtype), 1.0)
            v0 = jnp.zeros(3, dtype=dtype)
            g = jax.grad(lambda v, m=manifold, p=x: jnp.sum(m.expmap(v, p, 1.0)))(v0)
            assert jnp.all(jnp.isfinite(g))
            g0 = jax.grad(lambda v, m=manifold: jnp.sum(m.expmap_0(v, 1.0)))(v0)
            assert jnp.all(jnp.isfinite(g0))


class TestMLRFunctions:
    """Test that MLR functions work correctly via class methods."""

    def test_poincare_conformal_factor(self):
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        cf = manifold.conformal_factor(x, c=1.0)
        assert cf.shape == (2, 1)
        assert jnp.all(cf > 0)
        assert cf.dtype == jnp.float64

    def test_poincare_compute_mlr_pp(self):
        manifold = Poincare(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (4, 3)) * 0.3  # batch=4, in_dim=3
        x = jax.vmap(manifold.proj, in_axes=(0, None))(x, 1.0)
        z = jax.random.normal(k2, (5, 3)) * 0.1  # out_dim=5
        r = jax.random.normal(k3, (5, 1)) * 0.01

        result = manifold.compute_mlr_pp(x, z, r, c=1.0, clamping_factor=1.0, smoothing_factor=50.0)
        assert result.shape == (4, 5)
        assert jnp.all(jnp.isfinite(result))
        assert result.dtype == jnp.float64

    def test_poincare_compute_mlr_pp_matches_geodesic_distance(self):
        """|logit| / (2‖z‖) equals the geodesic distance to the HNN++ hyperplane.

        Ground truth 0.915745 was obtained by brute-force minimisation of
        d(x, exp_q(s·w)) over the hyperplane through q = exp_0(r·ẑ) with normal
        a = PT_{0→q}(z), w ⊥ a (matches hypll / official HNN++; van Spengler's
        poincare-resnet has a λ-transcription bug giving 0.064 here).
        """
        manifold = Poincare(dtype=jnp.float64)
        x = jnp.array([[0.35, 0.42]])
        z = jnp.array([[0.7, -0.3]])
        r = jnp.array([[0.4]])

        logits = manifold.compute_mlr_pp(x, z, r, c=1.0, clamping_factor=10.0, smoothing_factor=50.0)
        dist = jnp.abs(logits[0, 0]) / (2.0 * jnp.linalg.norm(z))
        assert jnp.allclose(dist, 0.915745, atol=1e-4)

    def test_poincare_compute_mlr_pp_zero_on_hyperplane(self):
        """The hyperplane base point q = exp_0(r·ẑ) lies on the decision
        boundary, so its logit must vanish. This discriminates the conformal
        factor from the buggy 2(1 - c‖x‖²) form at any curvature — the two only
        agree at the origin."""
        for c in (0.5, 1.0):
            manifold = Poincare(dtype=jnp.float64)
            z = jnp.array([[0.7, -0.3]])
            r = jnp.array([[0.4]])
            z_hat = z[0] / jnp.linalg.norm(z[0])
            q = manifold.expmap_0(r[0, 0] * z_hat, c)

            logit = manifold.compute_mlr_pp(q[None], z, r, c=c, clamping_factor=10.0, smoothing_factor=50.0)
            assert jnp.allclose(logit[0, 0], 0.0, atol=1e-10)

    def test_hyperboloid_compute_mlr(self):
        manifold = Hyperboloid(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (4, 4))  # batch=4, in_dim=4 (ambient)
        x = jax.vmap(manifold.proj, in_axes=(0, None))(x, 1.0)
        z = jax.random.normal(k2, (5, 3)) * 0.1  # out_dim=5, in_dim-1=3
        r = jax.random.normal(k3, (5, 1)) * 0.01

        result = manifold.compute_mlr(x, z, r, c=1.0, clamping_factor=1.0, smoothing_factor=50.0)
        assert result.shape == (4, 5)
        assert jnp.all(jnp.isfinite(result))
        assert result.dtype == jnp.float64

    def test_proper_velocity_compute_mlr(self):
        manifold = ProperVelocity(dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (4, 3)) * 0.3  # batch=4, in_dim=3
        z = jax.random.normal(k2, (5, 3)) * 0.1  # out_dim=5, in_dim=3 (PV has no time coord)
        r = jax.random.normal(k3, (5, 1)) * 0.01

        result = manifold.compute_mlr(x, z, r, c=1.0, clamping_factor=1.0, smoothing_factor=50.0)
        assert result.shape == (4, 5)
        assert jnp.all(jnp.isfinite(result))
        assert result.dtype == jnp.float64
