# Attention & Transformer

Three hyperbolic attention variants from the Hypformer paper (Yang et al. 2025,
Section 4.3). All operate on hyperboloid points and support independent curvatures for
input (`c_in`), attention computation (`c_attn`), and output (`c_out`), plus **causal
(autoregressive) masking** via `causal=True`. For complexity/use-case trade-offs see
the [NN Layers guide](../../user-guide/nn-layers.md#attention-normalization-positional-encoding).

The supporting utilities (`spatial_to_hyperboloid`, `lorentz_midpoint`,
`focus_transform`) are documented under [Primitives](primitives.md). The Hypformer
transformation component `HTCLinear` (used for MLP sublayers) is on the
[Linear](linear.md) page; HRC normalization is on the
[Normalization](normalization.md) page.

## Attention modules

::: hyperbolix.nn_layers.HyperbolicLinearAttention
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HyperbolicSoftmaxAttention
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HyperbolicFullAttention
    options:
      heading_level: 3

## Causal (autoregressive) masking

All three variants support `causal=True`: position `n` attends only to positions
`m ≤ n`, required for autoregressive tasks. Masking is JIT-compatible.

!!! info "How causal masking is implemented"
    - **`HyperbolicSoftmaxAttention`** / **`HyperbolicFullAttention`**: lower-triangular
      `-inf` mask on the score matrix before softmax — O(N²) in both modes.
    - **`HyperbolicLinearAttention`**: a cumulative-sum recurrence (`jax.lax.scan`,
      Katharopoulos et al. 2020): `S_i = Σ_{j≤i} φ(K_j) V_jᵀ`, O(1) per step →
      **O(N) total**, well-suited to long autoregressive sequences.

## Example

```python
import jax, jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HyperbolicLinearAttention

hyperboloid = Hyperboloid()
B, N, A_in, D_out = 4, 8, 9, 8  # 8-dim spatial + 1 time
spatial = jax.random.normal(jax.random.PRNGKey(0), (B, N, A_in - 1)) * 0.1
time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
x = jnp.concatenate([time, spatial], axis=-1)  # (B, N, A_in) on the hyperboloid

attn = HyperbolicLinearAttention(in_features=A_in, out_features=D_out, num_heads=2, power=2.0, rngs=nnx.Rngs(0))
y = attn(x, c_in=1.0, c_attn=1.0, c_out=1.0)          # bidirectional
y_causal = attn(x, c_in=1.0, c_attn=1.0, c_out=1.0, causal=True)
print(y.shape)  # (4, 8, 9) — D_out spatial + 1 time
```

See the [NN Layers guide](../../user-guide/nn-layers.md#composition-patterns)
(Pattern 3) for a full hyperbolic transformer block.
