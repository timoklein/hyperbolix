"""Tests for curvature helpers and manifold-as-plain-class behavior."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix import get_curvature, learnable_curvature
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
# 3. Curvature helpers
# ===========================================================================


class TestCurvatureHelpers:
    @pytest.mark.parametrize("init_c", [0.01, 0.1, 1.0, 5.0, 20.0])
    def test_init_recovery(self, init_c):
        c_raw = learnable_curvature(init_c)
        recovered = get_curvature(c_raw)
        assert jnp.allclose(recovered, init_c, atol=1e-5)

    def test_positivity_zeroed_raw(self):
        c_raw = learnable_curvature(1.0)
        c_raw[...] = jnp.array(0.0, dtype=jnp.float32)
        assert float(get_curvature(c_raw)) > 0

    def test_positivity_negative_raw(self):
        c_raw = learnable_curvature(1.0)
        c_raw[...] = jnp.array(-10.0, dtype=jnp.float32)
        assert float(get_curvature(c_raw)) >= 0

    def test_is_nnx_param(self):
        c_raw = learnable_curvature(0.1)
        assert isinstance(c_raw, nnx.Param)

    def test_negative_init_raises(self):
        with pytest.raises(ValueError, match="init_c > 0"):
            learnable_curvature(-1.0)

    def test_zero_init_raises(self):
        with pytest.raises(ValueError, match="init_c > 0"):
            learnable_curvature(0.0)

    def test_gradient_flow(self):
        c_raw = learnable_curvature(0.5)

        class Holder(nnx.Module):
            def __init__(self, c):
                self.c_raw = c

        holder = Holder(c_raw)

        def loss_fn(h):
            return get_curvature(h.c_raw) ** 2

        _loss, grads = nnx.value_and_grad(loss_fn)(holder)
        grad_val = grads.c_raw[...]
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0

    def test_roundtrip_with_bare_array(self):
        c_raw = learnable_curvature(1.5)
        val = get_curvature(c_raw[...])
        assert jnp.allclose(val, 1.5, atol=1e-5)


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
# 5. Training integration with learnable_curvature helpers
# ===========================================================================


class TestTrainingIntegration:
    def test_poincare_curvature_trains(self):
        from hyperbolix.nn_layers import HypLinearPoincarePP, HypRegressionPoincarePP

        manifold = Poincare(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.c_raw = learnable_curvature(init_c=1.0)
                self.fc = HypLinearPoincarePP(m, 4, 3, rngs=rngs)
                self.head = HypRegressionPoincarePP(m, 3, 2, rngs=rngs)

            def __call__(self, x):
                c = get_curvature(self.c_raw)
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

        c_before = float(get_curvature(model.c_raw))
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(get_curvature(model.c_raw))
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    def test_hyperboloid_curvature_trains(self):
        from hyperbolix.nn_layers import FGGLinear, FGGLorentzMLR

        manifold = Hyperboloid(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.c_raw = learnable_curvature(init_c=1.0)
                self.fc = FGGLinear(5, 4, rngs=rngs, activation=jax.nn.relu)
                self.head = FGGLorentzMLR(4, 3, rngs=rngs)

            def __call__(self, x):
                c = get_curvature(self.c_raw)
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

        c_before = float(get_curvature(model.c_raw))
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(get_curvature(model.c_raw))
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    def test_pv_curvature_trains(self):
        from hyperbolix.nn_layers import HypLinearPV, HypRegressionPV

        manifold = ProperVelocity(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.c_raw = learnable_curvature(init_c=1.0)
                self.fc = HypLinearPV(m, 4, 3, rngs=rngs)
                self.head = HypRegressionPV(m, 3, 2, rngs=rngs)

            def __call__(self, x):
                c = get_curvature(self.c_raw)
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

        c_before = float(get_curvature(model.c_raw))
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(get_curvature(model.c_raw))
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0


# ===========================================================================
# 6. Scan compatibility (regression test for the original bug)
# ===========================================================================


class TestScanCompatibility:
    def test_fori_loop_with_shared_manifold_and_learnable_c(self):
        """Regression test: shared manifold + learnable curvature param in nnx.fori_loop.

        This was the original bug: when ManifoldBase was an nnx.Module with a
        learnable _c_raw param, sharing the manifold across layers caused
        NNX graph deduplication to fail inside fori_loop with
        'ValueError: Dict key mismatch'. Now that manifolds are plain classes,
        they're static graphdef attributes — no deduplication issues.
        """
        from hyperbolix.nn_layers import HypLinearPoincarePP

        manifold = Poincare(c=0.1)

        class Model(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.manifold = manifold
                self.c_raw = learnable_curvature(init_c=0.1)
                self.l1 = HypLinearPoincarePP(manifold, 4, 4, rngs=nnx.Rngs(0))
                self.l2 = HypLinearPoincarePP(manifold, 4, 4, rngs=nnx.Rngs(1))

            def __call__(self, x):
                c = get_curvature(self.c_raw)
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
        assert float(get_curvature(model.c_raw)) > 0
