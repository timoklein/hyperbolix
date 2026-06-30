# Positional Encoding

Position-aware layers for hyperbolic Transformers on the hyperboloid. **HOPE**
(`hope` / `HyperbolicRoPE`) is a deterministic rotary encoding with no learnable
parameters that preserves relative position (`⟨HOPE(q,i), HOPE(k,j)⟩_L` depends only
on `i-j`); **`HypformerPositionalEncoding`** is learnable (HTCLinear + Lorentzian
residual). For when to prefer each, see the
[NN Layers guide](../../user-guide/nn-layers.md#attention-normalization-positional-encoding).

The `lorentz_residual` / `lorentz_scale` functions that back the residual skip
connection are documented under [Primitives](primitives.md); the `LorentzResidual`
NNX module wrapper is here.

## HOPE (Hyperbolic Rotary Positional Encoding)

::: hyperbolix.nn_layers.hope
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HyperbolicRoPE
    options:
      heading_level: 3

## Hypformer Positional Encoding

::: hyperbolix.nn_layers.HypformerPositionalEncoding
    options:
      heading_level: 3

!!! warning "Why `epsilon` is not learnable"
    The Hypformer reference keeps `epsilon` a plain (non-trainable) tensor fixed at
    1.0. Making it a parameter is unsafe: gradient descent can drive it below -1,
    where `x + epsilon·p` leaves the upper hyperboloid sheet and the `abs()` in the
    Lorentzian residual normalizer silently masks the violation instead of raising.
    For the same reason, `lorentz_residual`'s `w_y` must always be non-negative.

## Lorentzian residual connection

::: hyperbolix.nn_layers.LorentzResidual
    options:
      heading_level: 3

## Example

```python
import jax, jax.numpy as jnp
from flax import nnx
from hyperbolix.nn_layers import hope, HyperbolicRoPE
from hyperbolix.manifolds import Hyperboloid

batch, seq_len, d = 4, 16, 8
spatial = jax.random.normal(jax.random.PRNGKey(42), (batch, seq_len, d)) * 0.1
time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
z = jnp.concatenate([time, spatial], axis=-1)  # (4, 16, 9)
positions = jnp.arange(seq_len)

z_enc = hope(z, positions, c=1.0)                          # functional interface
rope = HyperbolicRoPE(dim=d, max_seq_len=64, base=10000.0)  # or the NNX module
z_enc = rope(z, positions, c=1.0)
print(z_enc.shape)  # (4, 16, 9)
```
