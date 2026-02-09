"""Test that HypLinearPoincare layers produce finite gradients under various conditions.

This is a regression test for the NaN gradient issue that occurred when using
HypLinearPoincarePP with the [None, :]/[0] vmap pattern in downstream applications.

The root cause was improper weight initialization (std=1.0 instead of std=1/sqrt(fan_in)),
which caused layer outputs to saturate at the Poincaré ball boundary, leading to
gradient overflow in float32 when backpropagating through logmap_0.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import poincare
from hyperbolix.nn_layers import HypLinearPoincare, HypLinearPoincarePP


class TestHypLinearPoincareGradients:
    """Test gradient computation for HypLinearPoincare layer."""

    def test_single_layer_gradients_finite(self):
        """Single layer should produce finite gradients."""
        key = jax.random.PRNGKey(0)
        layer = HypLinearPoincare(poincare, 10, 20, rngs=nnx.Rngs(key))

        x = jax.random.normal(key, (8, 10)) * 0.1
        x_manifold = jax.vmap(poincare.expmap_0, in_axes=(0, None))(x, 1.0)

        def loss_fn(m):
            out = m(x_manifold, 1.0)
            return jnp.mean(jnp.sum(out**2, axis=-1))

        _loss, grads = nnx.value_and_grad(loss_fn)(layer)
        grad_state = nnx.state(grads, nnx.Param)

        for _, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
            assert jnp.all(jnp.isfinite(value)), "Gradients contain NaN or Inf"

    def test_chained_layers_gradients_finite(self):
        """Chained layers should produce finite gradients (regression test)."""
        key = jax.random.PRNGKey(0)
        layer1 = HypLinearPoincare(poincare, 10, 20, rngs=nnx.Rngs(key), input_space="tangent")
        layer2 = HypLinearPoincare(poincare, 20, 10, rngs=nnx.Rngs(jax.random.PRNGKey(1)), input_space="manifold")

        x_tangent = jax.random.normal(key, (8, 10)) * 0.5

        def loss_fn(l1, l2):
            h = l1(x_tangent, 1.0)
            # Hyperbolic ReLU
            h_tangent = jax.vmap(poincare.logmap_0, in_axes=(0, None))(h, 1.0)
            h_relu = nnx.relu(h_tangent)
            h_back = jax.vmap(poincare.expmap_0, in_axes=(0, None))(h_relu, 1.0)
            out = l2(h_back, 1.0)
            return jnp.mean(jnp.sum(out**2, axis=-1))

        _loss, (g1, g2) = jax.value_and_grad(loss_fn, argnums=(0, 1))(layer1, layer2)
        gs1 = nnx.state(g1, nnx.Param)
        gs2 = nnx.state(g2, nnx.Param)

        for _, value in jax.tree_util.tree_flatten_with_path(gs1)[0]:
            assert jnp.all(jnp.isfinite(value)), "Layer 1 gradients contain NaN or Inf"

        for _, value in jax.tree_util.tree_flatten_with_path(gs2)[0]:
            assert jnp.all(jnp.isfinite(value)), "Layer 2 gradients contain NaN or Inf"


class TestHypLinearPoincarePPGradients:
    """Test gradient computation for HypLinearPoincarePP layer."""

    def test_single_layer_gradients_finite(self):
        """Single layer should produce finite gradients."""
        key = jax.random.PRNGKey(0)
        layer = HypLinearPoincarePP(poincare, 10, 20, rngs=nnx.Rngs(key), input_space="tangent")

        x = jax.random.normal(key, (8, 10)) * 0.5

        def loss_fn(m):
            out = m(x, 1.0)
            return jnp.mean(jnp.sum(out**2, axis=-1))

        _loss, grads = nnx.value_and_grad(loss_fn)(layer)
        grad_state = nnx.state(grads, nnx.Param)

        for _, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
            assert jnp.all(jnp.isfinite(value)), "Gradients contain NaN or Inf"

    def test_chained_layers_gradients_finite(self):
        """Chained layers should produce finite gradients (regression test for issue)."""
        key = jax.random.PRNGKey(0)
        layer1 = HypLinearPoincarePP(poincare, 12, 32, rngs=nnx.Rngs(key), input_space="tangent")
        layer2 = HypLinearPoincarePP(poincare, 32, 8, rngs=nnx.Rngs(jax.random.PRNGKey(1)), input_space="manifold")

        x_tangent = jax.random.normal(key, (12,))

        def loss_fn(l1, l2):
            h = l1(x_tangent[None, :], 1.0)[0]
            # Hyperbolic ReLU
            h_tangent = poincare.logmap_0(h, 1.0)
            h_relu = nnx.relu(h_tangent)
            h_back = poincare.expmap_0(h_relu, 1.0)
            out = l2(h_back[None, :], 1.0)[0]
            return jnp.sum(out**2)

        _loss, (g1, g2) = jax.value_and_grad(loss_fn, argnums=(0, 1))(layer1, layer2)
        gs1 = nnx.state(g1, nnx.Param)
        gs2 = nnx.state(g2, nnx.Param)

        for _, value in jax.tree_util.tree_flatten_with_path(gs1)[0]:
            assert jnp.all(jnp.isfinite(value)), "Layer 1 gradients contain NaN or Inf"

        for _, value in jax.tree_util.tree_flatten_with_path(gs2)[0]:
            assert jnp.all(jnp.isfinite(value)), "Layer 2 gradients contain NaN or Inf"

    @pytest.mark.parametrize("seed", range(10))
    def test_vmap_pattern_gradients_finite(self, seed):
        """Test the [None,:]/[0] vmap pattern that was reported in the issue.

        This pattern appeared in downstream world model code where per-example
        functions would unsqueeze inputs for the batch API then squeeze outputs.
        """
        key = jax.random.PRNGKey(seed)

        class TestDynamics(nnx.Module):
            def __init__(self, in_dim, out_dim, *, rngs, c=1.0):
                self.c = c
                self.layer = HypLinearPoincarePP(poincare, in_dim, out_dim, rngs=rngs, input_space="tangent")

            def __call__(self, x):
                # x: (in_dim,) per-example - unsqueeze for batch API, then squeeze
                return self.layer(x[None, :], self.c)[0]

        model = TestDynamics(6, 64, rngs=nnx.Rngs(key))
        batch = jax.random.normal(key, (8, 6))

        def loss_fn(m):
            def per_example(x):
                out = m(x)
                return jnp.sum(out**2)

            losses = jax.vmap(per_example)(batch)
            return jnp.mean(losses)

        _loss, grads = nnx.value_and_grad(loss_fn)(model)
        grad_state = nnx.state(grads, nnx.Param)

        for _, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
            assert jnp.all(jnp.isfinite(value)), f"Seed {seed}: Gradients contain NaN or Inf"

    def test_layer_outputs_on_manifold(self):
        """Layer outputs should be within the Poincaré ball."""
        key = jax.random.PRNGKey(0)
        layer = HypLinearPoincarePP(poincare, 12, 32, rngs=nnx.Rngs(key), input_space="tangent")

        x = jax.random.normal(key, (8, 12))
        outputs = layer(x, 1.0)
        norms = jnp.linalg.norm(outputs, axis=-1)

        # Verify outputs are on the manifold
        assert jnp.all(norms < 1.0), "All outputs should be within unit ball"


class TestWorldModelGradients:
    """Test gradient computation for a simplified world model architecture.

    This replicates the architecture pattern from the original issue report.
    """

    def test_world_model_gradients_finite(self):
        """Full world model should produce finite gradients."""

        class HyperbolicDynamicsHead(nnx.Module):
            def __init__(self, latent_dim, branching_factor, hidden_dim, *, rngs, curvature=1.0):
                self.c = curvature
                self.hyp_linear1 = HypLinearPoincarePP(
                    poincare,
                    latent_dim + branching_factor,
                    hidden_dim,
                    rngs=rngs,
                    input_space="tangent",
                )
                self.hyp_linear2 = HypLinearPoincarePP(poincare, hidden_dim, latent_dim, rngs=rngs, input_space="manifold")

            def __call__(self, z, action_onehot):
                z_tangent = poincare.logmap_0(z, self.c)
                dyn_input = jnp.concatenate([z_tangent, action_onehot])
                h = self.hyp_linear1(dyn_input[None, :], self.c)[0]
                h = poincare.expmap_0(nnx.relu(poincare.logmap_0(h, self.c)), self.c)
                z_next = self.hyp_linear2(h[None, :], self.c)[0]
                return z_next

        class SimpleEncoder(nnx.Module):
            def __init__(self, obs_dim, latent_dim, *, rngs):
                self.linear1 = nnx.Linear(obs_dim, 128, rngs=rngs)
                self.linear2 = nnx.Linear(128, latent_dim, rngs=rngs)

            def __call__(self, x):
                x = nnx.relu(self.linear1(x))
                return self.linear2(x)

        class SimpleDecoder(nnx.Module):
            def __init__(self, latent_dim, obs_dim, *, rngs):
                self.linear1 = nnx.Linear(latent_dim, 128, rngs=rngs)
                self.linear2 = nnx.Linear(128, obs_dim, rngs=rngs)

            def __call__(self, x):
                x = nnx.relu(self.linear1(x))
                return self.linear2(x)

        class HyperbolicWorldModel(nnx.Module):
            def __init__(self, obs_dim, latent_dim, branching_factor, hidden_dim, *, rngs, curvature=1.0):
                self.c = curvature
                self.encoder = SimpleEncoder(obs_dim, latent_dim, rngs=rngs)
                self.dynamics = HyperbolicDynamicsHead(
                    latent_dim, branching_factor, hidden_dim, rngs=rngs, curvature=curvature
                )
                self.decoder = SimpleDecoder(latent_dim, obs_dim, rngs=rngs)

            def __call__(self, obs, action_onehot):
                z_euc = self.encoder(obs)
                z = poincare.expmap_0(z_euc, self.c)
                z_next = self.dynamics(z, action_onehot)
                z_next_tangent = poincare.logmap_0(z_next, self.c)
                obs_pred = self.decoder(z_next_tangent)
                return obs_pred, z, z_next

        key = jax.random.PRNGKey(0)
        model = HyperbolicWorldModel(32, 8, 4, 32, rngs=nnx.Rngs(key))

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(100), 3)
        obs_batch = jax.random.normal(k1, (8, 32))
        action_batch = jax.nn.one_hot(jax.random.randint(k2, (8,), 0, 4), 4)
        target_batch = jax.random.normal(k3, (8, 32))

        def loss_fn(m):
            def per_example(o, a, t):
                obs_pred, _, _ = m(o, a)
                return jnp.mean((obs_pred - t) ** 2)

            losses = jax.vmap(per_example)(obs_batch, action_batch, target_batch)
            return jnp.mean(losses)

        _loss, grads = nnx.value_and_grad(loss_fn)(model)
        grad_state = nnx.state(grads, nnx.Param)

        for path, value in jax.tree_util.tree_flatten_with_path(grad_state)[0]:
            assert jnp.all(jnp.isfinite(value)), f"{path}: Gradients contain NaN or Inf"
