# Optimizers API

Riemannian optimization algorithms for training neural networks with hyperbolic parameters.

## Overview

Hyperbolix provides two Riemannian optimizers for manifold-valued parameters:

- **Riemannian SGD (RSGD)**: Stochastic gradient descent with momentum
- **Riemannian Adam**: Adaptive learning rates with moment transport

Both are Optax `GradientTransformation`s that work with `nnx.Optimizer`. They
detect manifold-valued parameters (tagged with `ManifoldParam`) and give every
other parameter a Euclidean update, so one optimizer covers a mixed model. Most
layers keep all their weights Euclidean and need neither; see the
[Riemannian Optimizers guide](../user-guide/optimizers.md) for when you do.

## Riemannian SGD

::: hyperbolix.optim.riemannian_sgd
    options:
      show_source: true
      heading_level: 3

### Example

```python
import jax.numpy as jnp
from flax import nnx
from hyperbolix.optim import riemannian_sgd
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Create model with hyperbolic parameters
model = HypLinearPoincare(
    manifold_module=poincare,
    in_dim=32,
    out_dim=16,
    rngs=nnx.Rngs(0)
)

# Create Riemannian SGD optimizer
optimizer = nnx.Optimizer(
    model,
    riemannian_sgd(learning_rate=0.01, momentum=0.9),
    wrt=nnx.Param
)

# Training step (use nnx.jit, not jax.jit, if you compile it:
# a jax.jit step drops the in-place parameter and optimizer updates)
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        pred = model(x, c=1.0)
        return jnp.mean((pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss
```

## Riemannian Adam

::: hyperbolix.optim.riemannian_adam
    options:
      show_source: true
      heading_level: 3

### Example

```python
from hyperbolix.optim import riemannian_adam

# Create Riemannian Adam optimizer
optimizer = nnx.Optimizer(
    model,
    riemannian_adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ),
    wrt=nnx.Param
)

# Use in training loop (same as RSGD example, call optimizer.update(model, grads))
```

## Manifold Metadata System

`ManifoldParam`, an `nnx.Param` subclass, tags a parameter that lives on a
manifold; the optimizers detect it with `isinstance`. For a tagged parameter
they convert the gradient to a Riemannian one, move along the manifold and
transport the momentum or moments; every plain `nnx.Param` gets the Euclidean
update. Riemannian math runs in the manifold's dtype; updates and moments keep
each parameter's storage dtype (see
[Storage vs. Compute Dtype](../user-guide/numerical-stability.md#storage-vs-compute-dtype)).

::: hyperbolix.optim.manifold_metadata
    options:
      show_source: true
      heading_level: 3
      members:
        - ManifoldParam
        - mark_manifold_param
        - get_manifold_info
        - has_manifold_params

## Expmap vs Retraction

Both optimizers take a `use_expmap` flag. The default, `True`, moves a manifold
parameter with the manifold's `expmap`; `False` uses its `retraction`, a
first-order approximation.

```python
opt = riemannian_adam(learning_rate=0.001)                    # expmap (default)
opt = riemannian_adam(learning_rate=0.001, use_expmap=False)  # retraction
```

## Mixed Optimization

On a model that mixes `nnx.Linear` layers with `HypLinearPoincare`,
`riemannian_adam` will:

- Apply standard Adam updates to the `nnx.Linear` parameters
- Apply Riemannian Adam updates to `HypLinearPoincare.bias` (a `ManifoldParam`)
- Apply Euclidean Adam updates to `HypLinearPoincare.kernel` (a plain `nnx.Param`)

The [user guide](../user-guide/optimizers.md#the-legacy-ganea-exception) covers these
layers and tagging your own manifold-valued parameters.

## References

The Riemannian optimizers are based on:

- Bécigneul, G., & Ganea, O. (2019). "Riemannian Adaptive Optimization Methods." ICLR 2019.
- Bonnabel, S. (2013). "Stochastic gradient descent on Riemannian manifolds." IEEE TAC.
