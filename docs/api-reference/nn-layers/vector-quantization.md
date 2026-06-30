# Vector Quantization

Hyperbolic VQ-VAE quantizer bottlenecks for the Poincaré ball. Both take encoder
features (Euclidean tangent vectors at the origin) and return a `PoincareVQOutput`
with the straight-through quantized vector for the decoder, the discrete code indices,
an auxiliary loss, and the codebook perplexity. The encoder/decoder stay in user code
— these layers are *only* the quantization step. For when to pick each, see the
[NN Layers guide](../../user-guide/nn-layers.md#vector-quantization-poincare).

## Embedding VQ (HVQ-VAE, EMA codebook)

The codebook is an `nnx.Variable` **buffer**, so `nnx.Optimizer(..., wrt=nnx.Param)`
ignores it — it moves only through `ema_update(z, indices, c)`, the hyperbolic moving
average of GGBall (Bu et al. 2026). Call `ema_update` in the train step **after**
`optimizer.update`. Set `dead_code_revival=True` (with a `reset_key`) to replace stale
codes with random encoder points.

::: hyperbolix.nn_layers.HypVQEmbeddingPoincare
    options:
      heading_level: 3

## MLR VQ (HyperVQ, implicit codebook)

The codebook is the rows of an internal `HypRegressionPoincarePP`, trained by plain
`optax.adam`. Selection is a Gumbel-Softmax straight-through sample over the MLR
scores; putting the STE on the categorical weights (not on `z_q`) is what lets the
gradient reach the MLR parameters. A `deterministic` flag (toggled by
`model.eval()` / `model.train()`) switches to an argmax MAP estimate at inference.

::: hyperbolix.nn_layers.HypVQMLRPoincare
    options:
      heading_level: 3

## Output type

::: hyperbolix.nn_layers.PoincareVQOutput
    options:
      heading_level: 3

The GGBall weighted gyromidpoint that backs the EMA centroid is a reusable helper —
see `poincare_weighted_midpoint` under [Primitives](primitives.md).

## Example

```python
import jax, jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypVQEmbeddingPoincare

manifold = Poincare(dtype=jnp.float64)  # manifold ops in f64; quantized decoder input is f32
h = jax.random.normal(jax.random.PRNGKey(1), (256, 64)) * 0.3  # encoder tangent features

vq = HypVQEmbeddingPoincare(manifold, num_codes=128, code_dim=64, rngs=nnx.Rngs(0),
                            commitment_weight=0.5, ema_decay=0.99, dead_code_revival=True)
out = vq(h, c=1.0)
print(out.quantized.shape, out.indices.shape, float(out.loss))  # (256,64) (256,) + commitment loss

# In the train step, AFTER optimizer.update(model, grads):
vq.ema_update(out.z, out.indices, c=1.0, reset_key=jax.random.key(7))
```
