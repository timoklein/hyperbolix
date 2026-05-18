"""Tests for learnable curvature on manifold classes."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds import Euclidean, Hyperboloid, Poincare, ProperVelocity
from hyperbolix.optim import ManifoldParam

jax.config.update("jax_enable_x64", True)


HYPERBOLIC_MANIFOLDS = [
    lambda **kw: Poincare(**kw),
    lambda **kw: Hyperboloid(**kw),
    lambda **kw: ProperVelocity(**kw),
]


@pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
class TestDefaultNonLearnable:
    def test_default_c_value(self, make_manifold):
        m = make_manifold()
        assert m.c == 1.0

    def test_no_params_in_state(self, make_manifold):
        m = make_manifold()
        state = nnx.state(m, nnx.Param)
        flat = nnx.to_flat_state(state)
        assert len(flat) == 0


@pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
class TestLearnableCurvature:
    @pytest.mark.parametrize("init_c", [0.01, 0.1, 1.0, 5.0, 20.0])
    def test_init_recovery(self, make_manifold, init_c):
        m = make_manifold(c=init_c, learnable=True)
        assert jnp.allclose(m.c, init_c, atol=1e-5)

    def test_positivity_zeroed_raw(self, make_manifold):
        m = make_manifold(c=1.0, learnable=True)
        m._c_raw = nnx.Param(jnp.array(0.0, dtype=jnp.float32))
        assert float(m.c) > 0

    def test_positivity_negative_raw(self, make_manifold):
        m = make_manifold(c=1.0, learnable=True)
        m._c_raw = nnx.Param(jnp.array(-10.0, dtype=jnp.float32))
        # softplus(-10) ≈ 4.5e-5, still positive
        assert float(m.c) >= 0

    def test_curvature_is_euclidean_param(self, make_manifold):
        m = make_manifold(c=0.1, learnable=True)
        assert isinstance(m._c_raw, nnx.Param)
        assert not isinstance(m._c_raw, ManifoldParam)

    def test_state_extraction(self, make_manifold):
        m = make_manifold(c=0.1, learnable=True)
        state = nnx.state(m, nnx.Param)
        flat = nnx.to_flat_state(state)
        assert len(flat) == 1
        path, _ = next(iter(flat))
        assert "_c_raw" in path


@pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
class TestGradientFlowGeneric:
    def test_gradient_through_curvature(self, make_manifold):
        m = make_manifold(c=0.1, learnable=True)

        def loss_fn(manifold):
            return manifold.c**2

        _loss, grads = nnx.value_and_grad(loss_fn)(m)
        grad_val = grads._c_raw[...]
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0


class TestGradientFlowDist:
    def test_gradient_through_poincare_dist(self):
        m = Poincare(c=0.5, learnable=True)
        x = jnp.array([0.1, 0.2], dtype=jnp.float32)
        y = jnp.array([0.3, 0.1], dtype=jnp.float32)

        def loss_fn(manifold):
            return manifold.dist(x, y, manifold.c)

        loss, grads = nnx.value_and_grad(loss_fn)(m)
        grad_val = grads._c_raw[...]
        assert jnp.isfinite(grad_val)
        assert jnp.isfinite(loss)

    def test_gradient_through_hyperboloid_dist(self):
        m = Hyperboloid(c=0.5, learnable=True)
        x = m.proj(jnp.array([1.5, 0.1, 0.2], dtype=jnp.float32), 0.5)
        y = m.proj(jnp.array([1.5, 0.3, 0.1], dtype=jnp.float32), 0.5)

        def loss_fn(manifold):
            return manifold.dist(x, y, manifold.c)

        loss, grads = nnx.value_and_grad(loss_fn)(m)
        grad_val = grads._c_raw[...]
        assert jnp.isfinite(grad_val)
        assert jnp.isfinite(loss)

    def test_gradient_through_pv_dist(self):
        m = ProperVelocity(c=0.5, learnable=True)
        x = jnp.array([0.1, 0.2], dtype=jnp.float32)
        y = jnp.array([0.3, 0.1], dtype=jnp.float32)

        def loss_fn(manifold):
            return manifold.dist(x, y, manifold.c)

        loss, grads = nnx.value_and_grad(loss_fn)(m)
        grad_val = grads._c_raw[...]
        assert jnp.isfinite(grad_val)
        assert jnp.isfinite(loss)


@pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
class TestVmapCompatibility:
    def test_vmap_with_learnable_c(self, make_manifold):
        m = make_manifold(c=0.5, learnable=True)
        x = jnp.array([[0.1, 0.2], [0.05, 0.15]], dtype=jnp.float32)
        y = jnp.array([[0.3, 0.1], [0.2, 0.05]], dtype=jnp.float32)

        if isinstance(m, Hyperboloid):
            x = jax.vmap(m.proj, in_axes=(0, None))(jnp.concatenate([jnp.ones((2, 1)), x], axis=-1), 0.5)
            y = jax.vmap(m.proj, in_axes=(0, None))(jnp.concatenate([jnp.ones((2, 1)), y], axis=-1), 0.5)

        dists = jax.vmap(m.dist, in_axes=(0, 0, None))(x, y, m.c)
        assert dists.shape == (2,)
        assert jnp.all(jnp.isfinite(dists))

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
        m = make_manifold(c=0.5, learnable=True)
        if isinstance(m, Hyperboloid):
            v = jnp.array([[0.0, 0.1, 0.2], [0.0, 0.05, 0.15]], dtype=jnp.float32)
        else:
            v = jnp.array([[0.1, 0.2], [0.05, 0.15]], dtype=jnp.float32)

        result = jax.vmap(m.expmap_0, in_axes=(0, None))(v, m.c)
        assert result.shape == v.shape
        assert jnp.all(jnp.isfinite(result))


class TestJitCompatibility:
    @pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
    def test_jit_with_learnable(self, make_manifold):
        m = make_manifold(c=0.5, learnable=True)
        x = jnp.array([0.1, 0.2], dtype=jnp.float32)
        y = jnp.array([0.3, 0.1], dtype=jnp.float32)

        if isinstance(m, Hyperboloid):
            x = m.proj(jnp.array([1.5, 0.1, 0.2], dtype=jnp.float32), 0.5)
            y = m.proj(jnp.array([1.5, 0.3, 0.1], dtype=jnp.float32), 0.5)

        @jax.jit
        def compute(c_val):
            return m.dist(x, y, c_val)

        eager = m.dist(x, y, m.c)
        jitted = compute(m.c)
        assert jnp.allclose(eager, jitted, atol=1e-6)


class TestSharedManifold:
    def test_shared_manifold_gradient(self):
        from hyperbolix.nn_layers import HypLinearPoincarePP

        m = Poincare(c=0.1, learnable=True)
        l1 = HypLinearPoincarePP(m, 4, 3, rngs=nnx.Rngs(0))
        l2 = HypLinearPoincarePP(m, 3, 2, rngs=nnx.Rngs(1))

        class Model(nnx.Module):
            def __init__(self, manifold, layer1, layer2):
                self.manifold = manifold
                self.l1 = layer1
                self.l2 = layer2

            def __call__(self, x):
                c = self.manifold.c
                h = self.l1(x, c)
                return jnp.sum(self.l2(h, c))

        model = Model(m, l1, l2)
        x = jnp.ones((2, 4), dtype=jnp.float32) * 0.1
        loss, grads = nnx.value_and_grad(lambda mdl: mdl(x))(model)
        assert jnp.isfinite(loss)
        # NNX deduplicates shared modules — find _c_raw in flat grad state
        flat_grads = nnx.to_flat_state(nnx.state(grads, nnx.Param))
        c_grads = [val[...] for path, val in flat_grads if "_c_raw" in path]
        assert len(c_grads) == 1
        assert jnp.isfinite(c_grads[0])


class TestEuclidean:
    def test_fixed_c_zero(self):
        m = Euclidean()
        assert m.c == 0.0

    def test_no_learnable_params(self):
        m = Euclidean()
        state = nnx.state(m, nnx.Param)
        flat = nnx.to_flat_state(state)
        assert len(flat) == 0


class TestTrainingIntegration:
    """Verify curvature actually updates during optimizer steps."""

    def test_poincare_curvature_trains(self):
        from hyperbolix.nn_layers import HypLinearPoincarePP, HypRegressionPoincarePP

        manifold = Poincare(c=1.0, learnable=True)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.fc = HypLinearPoincarePP(m, 4, 3, rngs=rngs)
                self.head = HypRegressionPoincarePP(m, 3, 2, rngs=rngs)

            def __call__(self, x):
                c = self.manifold.c
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

        c_before = float(manifold.c)
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(manifold.c)
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    def test_hyperboloid_curvature_trains(self):
        from hyperbolix.nn_layers import FGGLinear, FGGLorentzMLR

        manifold = Hyperboloid(c=1.0, learnable=True)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.fc = FGGLinear(5, 4, rngs=rngs, activation=jax.nn.relu)
                self.head = FGGLorentzMLR(4, 3, rngs=rngs)

            def __call__(self, x):
                c = self.manifold.c
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

        c_before = float(manifold.c)
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(manifold.c)
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    def test_pv_curvature_trains(self):
        from hyperbolix.nn_layers import HypLinearPV, HypRegressionPV

        manifold = ProperVelocity(c=1.0, learnable=True)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.fc = HypLinearPV(m, 4, 3, rngs=rngs)
                self.head = HypRegressionPV(m, 3, 2, rngs=rngs)

            def __call__(self, x):
                c = self.manifold.c
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

        c_before = float(manifold.c)
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(manifold.c)
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    def test_shared_curvature_trains(self):
        """Two layers sharing one manifold — curvature updates once per step."""
        from hyperbolix.nn_layers import HypLinearPoincarePP

        manifold = Poincare(c=0.1, learnable=True)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.l1 = HypLinearPoincarePP(m, 4, 3, rngs=nnx.Rngs(0))
                self.l2 = HypLinearPoincarePP(m, 3, 2, rngs=nnx.Rngs(1))

            def __call__(self, x):
                c = self.manifold.c
                h = self.l1(x, c)
                return jnp.sum(self.l2(h, c))

        model = Model(manifold, nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

        x = jnp.ones((8, 4), dtype=jnp.float32) * 0.1

        def loss_fn(m):
            return m(x)

        c_before = float(manifold.c)
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(manifold.c)
        assert c_before != c_after, f"Shared curvature did not change: {c_before}"
        assert c_after > 0


class TestValidation:
    def test_learnable_negative_c_raises(self):
        with pytest.raises(ValueError, match="c > 0"):
            Poincare(c=-1.0, learnable=True)

    def test_learnable_zero_c_raises(self):
        with pytest.raises(ValueError, match="c > 0"):
            Poincare(c=0.0, learnable=True)
