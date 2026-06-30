# Activation Functions

Hyperbolic activations that preserve manifold constraints. Three families:
**curvature-preserving** `hyp_*` (HRC pattern, fixed `c`), **Poincaré** tangent-space
wrappers, and **curvature-changing** `hrc_*` (`c_in → c_out`).

!!! info "How the HRC-pattern activations work"
    The `hyp_*` / `hrc_*` activations operate on the Hyperboloid: (1) extract space
    components `x_s = x[..., 1:]`, (2) apply the activation `y_s = f(x_s)`, (3) scale
    for curvature change `y_s = sqrt(c_in/c_out)·y_s`, (4) reconstruct time
    `y_t = sqrt(‖y_s‖² + 1/c_out)`. This avoids exp/log maps while preserving
    geometry. The Poincaré activations instead use `logmap_0 → activation → expmap_0`.

## Curvature-preserving (Hyperboloid)

::: hyperbolix.nn_layers.hyp_relu
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hyp_leaky_relu
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hyp_tanh
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hyp_swish
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hyp_gelu
    options:
      heading_level: 3

## Poincaré (tangent-space wrappers)

::: hyperbolix.nn_layers.poincare_relu
    options:
      heading_level: 3

::: hyperbolix.nn_layers.poincare_leaky_relu
    options:
      heading_level: 3

::: hyperbolix.nn_layers.poincare_tanh
    options:
      heading_level: 3

## Curvature-changing (HRC-based)

::: hyperbolix.nn_layers.hrc_relu
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hrc_leaky_relu
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hrc_tanh
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hrc_swish
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hrc_gelu
    options:
      heading_level: 3

## Example

```python
import jax, jax.numpy as jnp
from hyperbolix.nn_layers import hyp_relu, hrc_relu

x = jax.random.normal(jax.random.PRNGKey(0), (10, 5))
x_amb = jnp.concatenate([jnp.sqrt(jnp.sum(x**2, -1, keepdims=True) + 1.0), x], -1)  # (10, 6)

y = hyp_relu(x_amb, c=1.0)              # curvature-preserving
y2 = hrc_relu(x_amb, c_in=1.0, c_out=2.0)  # 1.0 → 2.0
print(y.shape)  # (10, 6) — same shape, still on the hyperboloid
```
