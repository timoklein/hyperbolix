# Mixture of Curvature Experts

The feed-forward block of HELM-MiCE (He et al. 2025): a DeepSeek-style mixture of
experts on the hyperboloid. Each routed expert works on its own hyperboloid sheet with
its own (optionally learnable) curvature, and the routed and shared expert outputs are
merged by one weighted Lorentzian centroid. `LorentzMoE` returns the block output and
the routing statistics; the Lorentzian residual with the block input is left to the
caller (see [`LorentzResidual`](positional-encoding.md#lorentzian-residual-connection)).
Attention for the same architecture is [`LorentzMLA`](attention.md#latent-attention-helm).

Load balancing follows DeepSeek-V3: add `moe_sequence_balance_loss(stats, alpha)` to the
training loss and call `update_bias(stats.mask, speed)` after each optimizer step. Its
default `rule="sign"` is the DeepSeek-V3 rule the HELM paper cites; `rule="proportional"` is
the rule of the unused `update_bias` in HELM's released code. The routing bias is a `RoutingBias` variable, not an `nnx.Param`, so gradient transforms and
`nnx.Optimizer(..., wrt=nnx.Param)` skip it.

## Mixture layer

::: hyperbolix.nn_layers.LorentzMoE
    options:
      heading_level: 3

## Components

::: hyperbolix.nn_layers.LorentzMoEGate
    options:
      heading_level: 3

::: hyperbolix.nn_layers.LorentzSwiGLU
    options:
      heading_level: 3

## Load balancing

::: hyperbolix.nn_layers.moe_sequence_balance_loss
    options:
      heading_level: 3

::: hyperbolix.nn_layers.MoERoutingStats
    options:
      heading_level: 3

::: hyperbolix.nn_layers.RoutingBias
    options:
      heading_level: 3

## Example

```python
import jax, jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import LorentzMoE, moe_sequence_balance_loss

dim, c = 9, 1.0  # 8 spatial dims + 1 time
hyperboloid = Hyperboloid()
tangent = jnp.zeros((2, 5, dim)).at[..., 1:].set(jax.random.normal(jax.random.PRNGKey(0), (2, 5, dim - 1)) * 0.1)
x = jax.vmap(jax.vmap(hyperboloid.expmap_0, in_axes=(0, None)), in_axes=(0, None))(tangent, c)  # (2, 5, 9)

moe = LorentzMoE(dim, inter_dim=16, num_routed=4, num_shared=1, top_k=2, rngs=nnx.Rngs(0))
out, stats = moe(x, c)                          # out: (2, 5, 9) on the hyperboloid
aux_loss = moe_sequence_balance_loss(stats, alpha=1e-4)
moe.update_bias(stats.mask, speed=0.005)        # after the optimizer step
print(out.shape, stats.mask.shape)  # (2, 5, 9) (2, 5, 4)
```
