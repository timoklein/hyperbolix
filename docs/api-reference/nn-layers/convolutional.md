# Convolutional Layers

2D hyperbolic convolutions and pooling for the Hyperboloid, Poincaré, and Proper
Velocity models. For **which** conv to pick see the
[NN Layers guide](../../user-guide/nn-layers.md#convolutional).

!!! warning "Dimensional growth is internal, not in the output shape"
    `HypConv2DHyperboloid` / `HypConv2DHyperboloidFHNN` / `FGGConv2D`
    HCat-concatenate a `d+1`-ambient input over the `N`-pixel receptive field
    into a `(d×N)+1`-ambient intermediate, which the kernel projects down to
    `out_channels`. The output ambient width is always `out_channels`; a
    larger kernel costs parameters and compute, not output width.
    `LorentzConv2D` does no HCat, and its output is also `out_channels`.

## Hyperboloid

`HypConv2DHyperboloid` (the robust default) uses HCat patch extraction + Lorentz FC
(Bdeir et al. 2023). `HypConv2DHyperboloidILNN` is the intrinsic Lorentz conv (Shi
et al. 2026): LogCat (log-radius-preserving concatenation) + PLFC, with origin
padding. `LorentzConv2D` is the HRC-based variant (Euclidean conv on the spatial part; faster,
lower accuracy — legacy/benchmarking).

::: hyperbolix.nn_layers.HypConv2DHyperboloid
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypConv2DHyperboloidFHNN
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypConv2DHyperboloidILNN
    options:
      heading_level: 3

::: hyperbolix.nn_layers.FGGConv2D
    options:
      heading_level: 3

::: hyperbolix.nn_layers.LorentzConv2D
    options:
      heading_level: 3

## Poincaré

`HypConv2DPoincare` extracts patches, applies beta-concatenation (HNN++, Shimizu
et al. 2020) over the receptive field, then a `HypLinearPoincarePP` layer. The
concatenated `K²·C_in` patch is mapped to `C_out`. Input is tangent by default
(`input_space="manifold"` accepts ball points); output is always tangent vectors.

::: hyperbolix.nn_layers.HypConv2DPoincare
    options:
      heading_level: 3

## Proper Velocity

`HypConv2DPV` (Chen et al. 2026, Sec 5.3). Because PV geometry is unconstrained
$\mathbb{R}^n$, patch concatenation coincides with Euclidean concatenation — **no
beta-scaling step**; the output width is `out_channels`. Outputs live on the PV manifold, so
Euclidean activations apply directly between conv layers.

::: hyperbolix.nn_layers.HypConv2DPV
    options:
      heading_level: 3

## Pooling & flattening (conv → FC bridge)

Two ways to turn an `(B, H', W', C)` hyperboloid feature map into one point per
sample before a classification head. `hyp_avg_pool2d` averages the spatial parts over
the grid and keeps the width (`C` in → `C` out). `hyp_flatten2d` keeps every pixel by
LogCat-concatenating them, so the width grows to `H'·W'·(C−1) + 1`.

!!! warning "Do not flatten a feature map with a plain `reshape`"
    Concatenating `N = H'·W'` hyperboloid points without the LogCat digamma rescale
    inflates the flattened point's spatial radius by `≈ √(H'·W')` — the same
    dimension-widening bias `Hyperboloid.log_radius_concat` exists to correct inside
    `HypConv2DHyperboloidILNN`'s per-patch concat, but with `N` = the whole feature
    map instead of one 3×3 receptive field. Use `hyp_flatten2d`; see the
    [numerical-stability guide](../../user-guide/numerical-stability.md#logcat-flatten).

::: hyperbolix.nn_layers.hyp_avg_pool2d
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hyp_flatten2d
    options:
      heading_level: 3

## Example

```python
import jax, jax.numpy as jnp
from flax import nnx
from hyperbolix.nn_layers import HypConv2DHyperboloid
from hyperbolix.manifolds import Hyperboloid

conv = HypConv2DHyperboloid(
    manifold_module=Hyperboloid(), in_channels=17, out_channels=32,
    kernel_size=(3, 3), stride=(1, 1), rngs=nnx.Rngs(0),
)
x = jax.random.normal(jax.random.PRNGKey(1), (8, 28, 28, 16))
x_amb = jnp.concatenate([jnp.sqrt(jnp.sum(x**2, -1, keepdims=True) + 1.0), x], -1)  # (8,28,28,17)
output = conv(x_amb, c=1.0)
print(output.shape)  # (8, 28, 28, 32) — output width is out_channels, independent of kernel size
```
