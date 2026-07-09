"""Tests for curvature helpers and manifold-as-plain-class behavior."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Euclidean, Hyperboloid, Poincare, ProperVelocity

jax.config.update("jax_enable_x64", True)


HYPERBOLIC_MANIFOLDS = [
    lambda **kw: Poincare(**kw),
    lambda **kw: Hyperboloid(**kw),
    lambda **kw: ProperVelocity(**kw),
]


# ===========================================================================
# 1. Manifolds are plain classes, not nnx.Module
# ===========================================================================


class TestManifoldIsPlainClass:
    @pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
    def test_not_nnx_module(self, make_manifold):
        m = make_manifold()
        assert not isinstance(m, nnx.Module)

    def test_euclidean_not_nnx_module(self):
        m = Euclidean()
        assert not isinstance(m, nnx.Module)

    @pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
    def test_no_learnable_attribute(self, make_manifold):
        m = make_manifold()
        assert not hasattr(m, "_learnable")
        assert not hasattr(m, "_c_raw")


# ===========================================================================
# 2. Static curvature on manifolds
# ===========================================================================


@pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
class TestStaticCurvature:
    def test_default_c_value(self, make_manifold):
        m = make_manifold()
        assert m.c == 1.0

    def test_custom_c_value(self, make_manifold):
        m = make_manifold(c=0.5)
        assert m.c == 0.5

    def test_c_is_float(self, make_manifold):
        m = make_manifold(c=2.0)
        assert isinstance(m.c, float)


class TestEuclidean:
    def test_fixed_c_zero(self):
        m = Euclidean()
        assert m.c == 0.0

    def test_not_nnx_module(self):
        m = Euclidean()
        assert not isinstance(m, nnx.Module)


# ===========================================================================
# 3. LearnableCurvature module
# ===========================================================================


class TestLearnableCurvatureInit:
    @pytest.mark.parametrize("init_c", [0.1, 0.5, 1.0, 5.0, 10.0])
    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_init_recovery_within_default_bounds(self, init_c, parameterization):
        c = LearnableCurvature(init_c, parameterization=parameterization)
        assert jnp.allclose(c(), init_c, atol=1e-5)

    @pytest.mark.parametrize("init_c", [0.01, 50.0, 100.0])
    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_init_recovery_with_disabled_clamp(self, init_c, parameterization):
        c = LearnableCurvature(init_c, parameterization=parameterization, c_min=None, c_max=None)
        assert jnp.allclose(c(), init_c, atol=1e-4)

    def test_raw_is_nnx_param(self):
        c = LearnableCurvature(0.1)
        assert isinstance(c.raw, nnx.Param)

    def test_is_nnx_module(self):
        c = LearnableCurvature(1.0)
        assert isinstance(c, nnx.Module)

    @pytest.mark.parametrize("bad_c", [-1.0, 0.0])
    def test_nonpositive_init_raises(self, bad_c):
        with pytest.raises(ValueError, match="init_c > 0"):
            LearnableCurvature(bad_c)

    def test_init_below_c_min_raises(self):
        with pytest.raises(ValueError, match=r"init_c.*c_min"):
            LearnableCurvature(0.05, c_min=0.1, c_max=10.0)

    def test_init_above_c_max_raises(self):
        with pytest.raises(ValueError, match=r"init_c.*c_max"):
            LearnableCurvature(20.0, c_min=0.1, c_max=10.0)

    def test_c_min_greater_than_c_max_raises(self):
        with pytest.raises(ValueError, match=r"c_min.*c_max"):
            LearnableCurvature(1.0, c_min=10.0, c_max=1.0)

    def test_unknown_parameterization_raises(self):
        with pytest.raises(ValueError, match="parameterization"):
            LearnableCurvature(1.0, parameterization="quadratic")  # type: ignore[arg-type]

    @pytest.mark.parametrize("parameterization", ["softplus", "log", "identity"])
    def test_nan_init_c_raises(self, parameterization):
        # NaN slips through every range check (all NaN comparisons are False); reject it up front rather
        # than silently store a NaN raw param. `identity` accepts negatives/zero, so it needs this too.
        with pytest.raises(ValueError, match="finite"):
            LearnableCurvature(float("nan"), parameterization=parameterization)


class TestLearnableCurvatureClamping:
    """Clamping applies to the recovered c, not the raw param."""

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_default_clamp_upper_bound(self, parameterization):
        c = LearnableCurvature(1.0, parameterization=parameterization)
        # Push raw to a huge value; default c_max=10.0 must hold.
        c.raw[...] = jnp.array(100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(10.0)

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_default_clamp_lower_bound(self, parameterization):
        c = LearnableCurvature(1.0, parameterization=parameterization)
        # Push raw to a very negative value; default c_min=0.1 must hold.
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(0.1)

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_disabled_clamp_allows_extremes(self, parameterization):
        c = LearnableCurvature(1.0, parameterization=parameterization, c_min=None, c_max=None)
        c.raw[...] = jnp.array(20.0, dtype=jnp.float32)
        # No clamp: softplus(20) ≈ 20, exp(20) ≈ 4.85e8 — both well above 10.
        assert float(c()) > 10.0

    def test_custom_clamp_bounds(self):
        c = LearnableCurvature(0.5, parameterization="log", c_min=0.2, c_max=2.0)
        c.raw[...] = jnp.array(10.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(2.0)
        c.raw[...] = jnp.array(-10.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(0.2)


class TestLearnableCurvatureGradients:
    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_gradient_flow(self, parameterization):
        class Holder(nnx.Module):
            def __init__(self):
                self.curvature = LearnableCurvature(0.5, parameterization=parameterization)

        holder = Holder()

        def loss_fn(h):
            return h.curvature() ** 2

        _loss, grads = nnx.value_and_grad(loss_fn)(holder)
        grad_val = grads.curvature.raw[...]
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0

    def test_log_parameterization_scale_invariance(self):
        """For c = exp(raw), dc/draw = c — scale-invariant gradient."""
        c = LearnableCurvature(2.0, parameterization="log", c_min=None, c_max=None)

        def fn(m):
            return m()

        _val, grads = nnx.value_and_grad(fn)(c)
        # d(exp(raw))/draw = exp(raw) = c
        assert jnp.allclose(grads.raw[...], 2.0, atol=1e-5)

    def test_softplus_parameterization_sigmoid_gradient(self):
        """For c = softplus(raw), dc/draw = sigmoid(raw) ∈ (0, 1)."""
        c = LearnableCurvature(1.0, parameterization="softplus", c_min=None, c_max=None)
        raw_val = float(c.raw[...])

        def fn(m):
            return m()

        _val, grads = nnx.value_and_grad(fn)(c)
        expected = float(jax.nn.sigmoid(jnp.array(raw_val)))
        assert jnp.allclose(grads.raw[...], expected, atol=1e-5)

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_default_clamp_is_gradient_dead_past_boundary(self, parameterization):
        """Documents current default behavior: plain jnp.clip zeroes the gradient
        once c exits [c_min, c_max] -- c is pinned, not chosen."""
        c = LearnableCurvature(1.0, parameterization=parameterization)
        # raw=15.0 pushes c well past c_max=10.0 for both parameterizations (softplus(15)~=15,
        # exp(15)~=3.3e6). (The former exp() overflow at large raw is now guarded — see
        # test_log_parameterization_no_nan_gradient_on_exp_overflow.)
        c.raw[...] = jnp.array(15.0, dtype=jnp.float32)

        def fn(m):
            return m()

        _val, grads = nnx.value_and_grad(fn)(c)
        assert float(grads.raw[...]) == 0.0

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_straight_through_clamp_keeps_gradient_nonzero_past_boundary(self, parameterization):
        """straight_through_clamp=True fixes the gradient-dead ratchet: the forward
        value stays clamped, but the gradient keeps flowing past the boundary."""
        c = LearnableCurvature(1.0, parameterization=parameterization, straight_through_clamp=True)
        # raw=15.0 pushes c well past c_max=10.0 for both parameterizations (softplus(15)~=15,
        # exp(15)~=3.3e6). (The former exp() overflow at large raw is now guarded — see
        # test_log_parameterization_no_nan_gradient_on_exp_overflow.)
        c.raw[...] = jnp.array(15.0, dtype=jnp.float32)

        def fn(m):
            return m()

        val, grads = nnx.value_and_grad(fn)(c)
        assert float(val) == pytest.approx(10.0)  # forward value still clamped
        assert jnp.isfinite(grads.raw[...])
        assert float(grads.raw[...]) != 0.0

    @pytest.mark.parametrize("straight_through", [False, True])
    def test_log_parameterization_no_nan_gradient_on_exp_overflow(self, straight_through):
        # Regression: exp(raw) overflowed float32 to +inf for large raw, making the clip's out-of-range
        # cotangent 0*inf = NaN (and, under straight_through, NaN-ing the forward via inf + (-inf)). The
        # exponent cap + numerically-stable straight-through must keep the forward pinned at c_max and the
        # gradient finite even where exp() would have overflowed.
        c = LearnableCurvature(1.0, parameterization="log", straight_through_clamp=straight_through)
        c.raw[...] = jnp.array(100.0, dtype=jnp.float32)  # exp(100) overflows float32

        def fn(m):
            return m()

        val, grads = nnx.value_and_grad(fn)(c)
        assert jnp.isfinite(val) and float(val) == pytest.approx(10.0)  # forward pinned at c_max, not NaN/0
        assert jnp.isfinite(grads.raw[...])


# ===========================================================================
# 4. Vmap compatibility (manifolds as plain classes)
# ===========================================================================


@pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
class TestVmapCompatibility:
    def test_vmap_with_fixed_c(self, make_manifold):
        m = make_manifold(c=0.5)
        x = jnp.array([[0.1, 0.2], [0.05, 0.15]], dtype=jnp.float32)
        y = jnp.array([[0.3, 0.1], [0.2, 0.05]], dtype=jnp.float32)

        if isinstance(m, Hyperboloid):
            x = jax.vmap(m.proj, in_axes=(0, None))(jnp.concatenate([jnp.ones((2, 1)), x], axis=-1), 0.5)
            y = jax.vmap(m.proj, in_axes=(0, None))(jnp.concatenate([jnp.ones((2, 1)), y], axis=-1), 0.5)

        dists = jax.vmap(m.dist, in_axes=(0, 0, None))(x, y, m.c)
        assert dists.shape == (2,)
        assert jnp.all(jnp.isfinite(dists))

    def test_vmap_expmap_0(self, make_manifold):
        m = make_manifold(c=0.5)
        if isinstance(m, Hyperboloid):
            v = jnp.array([[0.0, 0.1, 0.2], [0.0, 0.05, 0.15]], dtype=jnp.float32)
        else:
            v = jnp.array([[0.1, 0.2], [0.05, 0.15]], dtype=jnp.float32)

        result = jax.vmap(m.expmap_0, in_axes=(0, None))(v, m.c)
        assert result.shape == v.shape
        assert jnp.all(jnp.isfinite(result))


# ===========================================================================
# 5. Training integration with LearnableCurvature
# ===========================================================================


PARAMETERIZATIONS = ["softplus", "log"]


class TestTrainingIntegration:
    @pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
    def test_poincare_curvature_trains(self, parameterization):
        from hyperbolix.nn_layers import HypLinearPoincarePP, HypRegressionPoincarePP

        manifold = Poincare(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.curvature = LearnableCurvature(init_c=1.0, parameterization=parameterization)
                self.fc = HypLinearPoincarePP(m, 4, 3, rngs=rngs)
                self.head = HypRegressionPoincarePP(m, 3, 2, rngs=rngs)

            def __call__(self, x):
                c = self.curvature()
                h = self.fc(x, c)
                return self.head(h, c)

        model = Model(manifold, nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (16, 4), dtype=jnp.float32) * 0.1
        target = jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float32)

        def loss_fn(m):
            logits = m(x)
            return jnp.mean((logits - target) ** 2)

        c_before = float(model.curvature())
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(model.curvature())
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    @pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
    def test_hyperboloid_curvature_trains(self, parameterization):
        from hyperbolix.nn_layers import FGGLinear, FGGLorentzMLR

        manifold = Hyperboloid(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.curvature = LearnableCurvature(init_c=1.0, parameterization=parameterization)
                self.fc = FGGLinear(5, 4, rngs=rngs, activation=jax.nn.relu)
                self.head = FGGLorentzMLR(4, 3, rngs=rngs)

            def __call__(self, x):
                c = self.curvature()
                h = self.fc(x, c)
                return self.head(h, c)

        model = Model(manifold, nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

        key = jax.random.PRNGKey(0)
        x_spatial = jax.random.normal(key, (16, 4), dtype=jnp.float32) * 0.1
        x = manifold.proj_batch(
            jnp.concatenate([jnp.ones((16, 1), dtype=jnp.float32), x_spatial], axis=-1),
            1.0,
        )
        target = jax.random.randint(jax.random.PRNGKey(1), (16,), 0, 3)

        def loss_fn(m):
            logits = m(x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()

        c_before = float(model.curvature())
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(model.curvature())
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    @pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
    def test_pv_curvature_trains(self, parameterization):
        from hyperbolix.nn_layers import HypLinearPV, HypRegressionPV

        manifold = ProperVelocity(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.curvature = LearnableCurvature(init_c=1.0, parameterization=parameterization)
                self.fc = HypLinearPV(m, 4, 3, rngs=rngs)
                self.head = HypRegressionPV(m, 3, 2, rngs=rngs)

            def __call__(self, x):
                c = self.curvature()
                h = self.fc(x, c=c)
                return self.head(h, c=c)

        model = Model(manifold, nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (16, 4), dtype=jnp.float32) * 0.3
        target = jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float32)

        def loss_fn(m):
            y = m(x)
            return jnp.mean((y - target) ** 2)

        c_before = float(model.curvature())
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(model.curvature())
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0


# ===========================================================================
# 6. Scan compatibility (regression test for the original bug)
# ===========================================================================


class TestScanCompatibility:
    @pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
    def test_fori_loop_with_shared_manifold_and_learnable_c(self, parameterization):
        """Regression test: shared manifold + LearnableCurvature in nnx.fori_loop.

        This was the original bug: when ManifoldBase was an nnx.Module with a
        learnable _c_raw param, sharing the manifold across layers caused
        NNX graph deduplication to fail inside fori_loop with
        'ValueError: Dict key mismatch'. Now that manifolds are plain classes,
        they're static graphdef attributes — no deduplication issues. The
        LearnableCurvature instance lives once on the model so there is no
        shared-reference aliasing.
        """
        from hyperbolix.nn_layers import HypLinearPoincarePP

        manifold = Poincare(c=0.1)

        class Model(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.manifold = manifold
                self.curvature = LearnableCurvature(init_c=0.1, parameterization=parameterization)
                self.l1 = HypLinearPoincarePP(manifold, 4, 4, rngs=nnx.Rngs(0))
                self.l2 = HypLinearPoincarePP(manifold, 4, 4, rngs=nnx.Rngs(1))

            def __call__(self, x):
                c = self.curvature()
                h = self.l1(x, c)
                return jnp.sum(self.l2(h, c))

        model = Model(nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

        x = jnp.ones((4, 4), dtype=jnp.float32) * 0.1

        def train_step(i, carry):
            model, optimizer = carry
            _, grads = nnx.value_and_grad(lambda m: m(x))(model)
            optimizer.update(model, grads)
            return model, optimizer

        model, optimizer = nnx.fori_loop(0, 3, train_step, (model, optimizer))
        loss = model(x)
        assert jnp.isfinite(loss)
        assert float(model.curvature()) > 0


# ===========================================================================
# 7. Signed `identity` parameterization (Stereographic manifold)
# ===========================================================================


class TestLearnableCurvatureIdentity:
    """The signed ``identity`` parameterization (``c = raw``) for the Stereographic manifold.

    Unlike softplus/log (strictly positive), identity spans hyperbolic (``c>0``), Euclidean
    (``c=0``), and spherical (``c<0``); its default clamp is the symmetric magnitude cap
    ``[-10, 10]`` (which *includes* 0), not the positive ``[0.1, 10]`` window.
    """

    @pytest.mark.parametrize("init_c", [-2.0, -0.5, 0.0, 1.5, 9.0])
    def test_identity_recovers_signed_init(self, init_c):
        c = LearnableCurvature(init_c, parameterization="identity")
        assert jnp.allclose(c(), init_c, atol=1e-6)

    @pytest.mark.parametrize("init_c", [-1.0, 0.0, -0.05])
    def test_identity_accepts_nonpositive_init(self, init_c):
        # softplus/log would raise here (they cannot represent c <= 0); identity must not.
        c = LearnableCurvature(init_c, parameterization="identity")
        assert jnp.isfinite(c())

    def test_identity_signed_default_clamp(self):
        c = LearnableCurvature(0.0, parameterization="identity")
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(-10.0)  # symmetric lower bound, NOT the softplus +0.1
        c.raw[...] = jnp.array(100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(10.0)

    def test_identity_default_clamp_includes_zero(self):
        # The Euclidean point c=0 must be reachable; a naive c_min=0.1 carryover would forbid it.
        c = LearnableCurvature(0.0, parameterization="identity")
        c.raw[...] = jnp.array(0.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(0.0)

    def test_identity_disabled_clamp(self):
        c = LearnableCurvature(-2.0, parameterization="identity", c_min=None, c_max=None)
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(-100.0)

    def test_identity_custom_signed_bounds(self):
        c = LearnableCurvature(0.0, parameterization="identity", c_min=-3.0, c_max=3.0)
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(-3.0)
        c.raw[...] = jnp.array(100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(3.0)

    def test_identity_below_signed_c_min_raises(self):
        # init below the symmetric default lower bound (-10) is still range-checked.
        with pytest.raises(ValueError, match=r"init_c.*c_min"):
            LearnableCurvature(-20.0, parameterization="identity")

    def test_identity_gradient_is_one(self):
        """For c = raw, dc/draw = 1 everywhere — no zero-crossing obstruction (the whole point)."""
        c = LearnableCurvature(1.5, parameterization="identity", c_min=None, c_max=None)

        def fn(m):
            return m()

        _val, grads = nnx.value_and_grad(fn)(c)
        assert jnp.allclose(grads.raw[...], 1.0, atol=1e-6)

    def test_identity_gradient_finite_at_zero(self):
        """The zero-crossing point c=0 has a finite, well-defined gradient (== 1)."""
        c = LearnableCurvature(0.0, parameterization="identity", c_min=None, c_max=None)

        def fn(m):
            return m()

        val, grads = nnx.value_and_grad(fn)(c)
        assert float(val) == pytest.approx(0.0)
        assert jnp.isfinite(grads.raw[...])
        assert jnp.allclose(grads.raw[...], 1.0, atol=1e-6)

    def test_identity_grad_through_stereographic_at_zero(self):
        """Grad w.r.t. curvature is finite through Stereographic.dist at c=0 — guards the
        κ-trig Taylor seam from the learnable-curvature side."""
        from hyperbolix.manifolds import Stereographic

        manifold = Stereographic(dtype=jnp.float32)
        x = jnp.array([0.1, 0.2, -0.05], dtype=jnp.float32)
        y = jnp.array([-0.2, 0.15, 0.1], dtype=jnp.float32)
        c = LearnableCurvature(0.0, parameterization="identity", c_min=None, c_max=None)

        def fn(m):
            return manifold.dist(x, y, m())

        val, grads = nnx.value_and_grad(fn)(c)
        assert jnp.isfinite(val)
        assert jnp.isfinite(grads.raw[...])

    def test_identity_straight_through_clamp_keeps_gradient(self):
        c = LearnableCurvature(0.0, parameterization="identity", straight_through_clamp=True)
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)  # past the -10 lower bound

        def fn(m):
            return m()

        val, grads = nnx.value_and_grad(fn)(c)
        assert float(val) == pytest.approx(-10.0)  # forward value still clamped
        assert jnp.isfinite(grads.raw[...])
        assert float(grads.raw[...]) != 0.0


class TestStereographicCurvatureTraining:
    def test_signed_curvature_trains(self):
        """A learnable signed curvature drives a Stereographic distance objective: the curvature
        updates, stays finite across steps, and is free to be negative (spherical)."""
        from hyperbolix.manifolds import Stereographic

        manifold = Stereographic(dtype=jnp.float32)

        class Model(nnx.Module):
            def __init__(self):
                self.manifold = manifold
                self.curvature = LearnableCurvature(init_c=-1.0, parameterization="identity")
                self.point = nnx.Param(jnp.array([0.1, 0.2, -0.05], dtype=jnp.float32))

            def __call__(self, target):
                c = self.curvature()
                return self.manifold.dist(self.point[...], target, c) ** 2

        model = Model()
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
        target = jnp.array([-0.15, 0.1, 0.2], dtype=jnp.float32)

        def loss_fn(m):
            return m(target)

        c_before = float(model.curvature())
        losses = []
        for _ in range(25):
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            losses.append(float(loss))

        c_after = float(model.curvature())
        assert all(jnp.isfinite(jnp.array(loss_val)) for loss_val in losses)
        assert c_before != c_after, "signed curvature did not update"
        assert jnp.isfinite(jnp.array(c_after))
