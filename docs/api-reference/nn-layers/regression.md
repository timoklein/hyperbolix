# Regression & MLR

Single-layer hyperbolic classifiers (multinomial logistic regression heads). For
**which** head to pair with which backbone, see the
[NN Layers guide](../../user-guide/nn-layers.md#regression-classification-head).

Two decision geometries are available: **point-to-hyperplane** heads
(Ganea/Shimizu/Bdeir lineage — the `HypRegression*` and `FGGLorentzMLR` layers) and
**point-to-horosphere** Busemann heads (Chen et al. 2026), whose logits are
$u_k(x) = -\alpha_k B^{v_k}(x) + b_k$ from the closed-form Busemann function
(`Hyperboloid.busemann` / `Poincare.busemann`).

## Poincaré

::: hyperbolix.nn_layers.HypRegressionPoincare
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypRegressionPoincarePP
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypRegressionPoincareBusemann
    options:
      heading_level: 3

## Hyperboloid

`FGGLorentzMLR` is the FGG-family head (Klis et al. 2026); pair it with an FGG linear
stack.

::: hyperbolix.nn_layers.HypRegressionHyperboloid
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HypRegressionHyperboloidBusemann
    options:
      heading_level: 3

::: hyperbolix.nn_layers.FGGLorentzMLR
    options:
      heading_level: 3

## Proper Velocity

`HypRegressionPV` (Chen et al. 2026, Thm 5.2 / Eq. 19) returns the Euclidean signed
margin to each PV hyperplane — feed to a standard softmax cross-entropy loss.

::: hyperbolix.nn_layers.HypRegressionPV
    options:
      heading_level: 3

## Example

```python
import jax
from flax import nnx
from hyperbolix.nn_layers import HypRegressionPoincare
from hyperbolix.manifolds import Poincare

poincare = Poincare()
head = HypRegressionPoincare(manifold_module=poincare, in_dim=32, out_dim=10, rngs=nnx.Rngs(0))

x = jax.random.normal(jax.random.PRNGKey(1), (64, 32)) * 0.3
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)
logits = head(x_proj, c=1.0)            # (64, 10)
probs = jax.nn.softmax(logits, axis=-1)
```
