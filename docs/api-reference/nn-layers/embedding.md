# Embedding

`LorentzEmbedding` is a token embedding table whose rows are points on the hyperboloid
(the `LorentzEmbeddings` table of HELM, He et al. 2025). Two parameterizations are
available: a `ManifoldParam` table of ambient points trained with
[`riemannian_adam`](../optimizers.md), or a Euclidean table of spatial coordinates whose
time coordinate is rebuilt in the forward pass and trained with any Euclidean optimizer.
`lorentz_embedding_init` builds the manifold-parameterized table.

::: hyperbolix.nn_layers.LorentzEmbedding
    options:
      heading_level: 3

::: hyperbolix.nn_layers.lorentz_embedding_init
    options:
      heading_level: 3

## Example

```python
import jax.numpy as jnp
import optax
from flax import nnx
from hyperbolix.nn_layers import LorentzEmbedding
from hyperbolix.optim import riemannian_adam

emb = LorentzEmbedding(num_embeddings=100, features=9, rngs=nnx.Rngs(0), c=1.0)
ids = jnp.array([[3, 7, 7], [1, 0, 42]])
points = emb(ids)                       # (2, 3, 9) on the hyperboloid at c = 1
points_c2 = emb(ids, c_out=2.0)         # the same rows mapped onto the c = 2 sheet

# The manifold table takes a Riemannian step; other nnx.Params take a plain Adam step.
optimizer = nnx.Optimizer(emb, riemannian_adam(1e-3), wrt=nnx.Param)
```
