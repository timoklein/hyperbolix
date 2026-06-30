# Linear Layers

Fully-connected hyperbolic layers across the Poincaré, Hyperboloid, and Proper
Velocity models. For **which** linear layer to pick (defaults, speed/expressiveness
trade-offs) see the [NN Layers guide](../../user-guide/nn-layers.md#linear-fully-connected).

!!! note "Channel convention"
    Hyperboloid linear layers (`HTCLinear`, `FGGLinear`, `HypLinearHyperboloid*`) take
    **ambient** `in_features = d+1`. Poincaré and PV layers take **spatial**
    `in_dim = d`. `HTCLinear` is the exception whose `out_features` is *spatial* — its
    output shape is `(B, out_features + 1)`.

## Poincaré

::: hyperbolix.nn_layers.HypLinearPoincare
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypLinearPoincarePP
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypLinearPoincareBusemann
    options:
      heading_level: 3

## Hyperboloid

`HypLinearHyperboloidPLFC` is the point-to-hyperplane Lorentz FC (Shi et al. 2026),
the Lorentz analog of the HNN++ formulation; the Busemann FC decides with
point-to-*horosphere* distances. `HTCLinear` (the robust default) is documented here;
the foundational `htc()` function lives in [Primitives](primitives.md).

::: hyperbolix.nn_layers.HTCLinear
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypLinearHyperboloidFHCNN
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypLinearHyperboloidFHNN
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypLinearHyperboloidPLFC
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypLinearHyperboloidBusemann
    options:
      heading_level: 3

### FGG Linear (Klis et al. 2026)

`FGGLinear` achieves linear (not logarithmic) growth of hyperbolic distance via the
sinh/arcsinh cancellation in the Lorentzian activation chain. It defaults to a
norm-preserving `fan_out` init suited to unnormalized stacks — see the
[init-scales note](../../user-guide/nn-layers.md#initialization-scales).

::: hyperbolix.nn_layers.FGGLinear
    options:
      heading_level: 3

## Proper Velocity

`HypLinearPV` implements the PV fully-connected layer from Chen et al. 2026
(Thm 5.3 / Eq. 22): $y_k = (1/\sqrt{c}) \cdot \sinh(\sqrt{c} \cdot v_k(x))$ where
$v_k(x)$ is the PV multinomial-logistic-regression signed margin. PV is an
unconstrained $\mathbb{R}^n$ model, so inputs and outputs are Euclidean with no
projection step.

::: hyperbolix.nn_layers.HypLinearPV
    options:
      heading_level: 3

## Example

```python
import jax
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.manifolds import Poincare

poincare = Poincare()
layer = HypLinearPoincare(manifold_module=poincare, in_dim=32, out_dim=16, rngs=nnx.Rngs(0))

x = jax.random.normal(jax.random.PRNGKey(1), (10, 32)) * 0.3
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)
output = layer(x_proj, c=1.0)
print(output.shape)  # (10, 16)
```

See the [NN Layers guide](../../user-guide/nn-layers.md#composition-patterns) for
multi-layer composition patterns (HTC classifier, hybrid CNN + Poincaré head).
