# Optimizers API

Riemannian optimization algorithms for training neural networks with hyperbolic parameters.

## Overview

Hyperbolix provides two Riemannian optimizers that extend standard Euclidean optimizers to manifold-valued parameters:

- **Riemannian SGD (RSGD)**: Stochastic gradient descent with momentum
- **Riemannian Adam**: Adaptive learning rates with moment transport

Both optimizers:

- Follow the standard Optax `GradientTransformation` interface
- Automatically detect manifold parameters via metadata
- Support mixed Euclidean/Riemannian parameter optimization
- Are compatible with `nnx.Optimizer` wrapper

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

# Training step
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

Hyperbolix uses `ManifoldParam`, an `nnx.Param` subclass, to tag parameters that live on a Riemannian manifold. The optimizers detect these via `isinstance` checks.

### How It Works

```python
from hyperbolix.optim import ManifoldParam

# Layer tags hyperbolic parameters using ManifoldParam
class HypLinearPoincare(nnx.Module):
    def __init__(self, manifold_module, in_dim, out_dim, *, rngs, curvature=1.0):
        self.manifold = manifold_module

        # Kernel is Euclidean (plain nnx.Param)
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (out_dim, in_dim)) * 0.01
        )

        # Bias lives on Poincaré ball (ManifoldParam)
        self.bias = ManifoldParam(
            jnp.zeros(out_dim),
            manifold=manifold_module,
            curvature=curvature,  # the optimizer's bias update uses this, not the call-time c
        )
```

The optimizer automatically:

1. Detects `ManifoldParam` instances
2. Applies Riemannian gradient updates (expmap/retraction)
3. Performs parallel transport for momentum/adaptive moments
4. Falls back to Euclidean updates for plain `nnx.Param` parameters

**Dtype contract:** Riemannian math runs in the manifold's dtype; updates and
moments keep each parameter's storage dtype. See the
[Numerical Stability](../user-guide/numerical-stability.md#storage-vs-compute-dtype).

### API Reference

::: hyperbolix.optim.manifold_metadata
    options:
      show_source: true
      heading_level: 4
      members:
        - ManifoldParam
        - mark_manifold_param
        - get_manifold_info
        - has_manifold_params

## Expmap vs Retraction

Both optimizers support two update modes:

- **Exponential map** (default): `expmap(x, -lr * grad)`
  - Exact geodesic following
  - Numerically stable for large steps
  - Slightly slower

- **Retraction**: `proj(x - lr * grad)`
  - First-order approximation
  - Faster computation
  - Can be less stable for large learning rates

### Choosing Update Mode

```python
opt = riemannian_adam(learning_rate=0.001)                    # expmap (default)
opt = riemannian_adam(learning_rate=0.001, use_expmap=False)  # retraction
```

`riemannian_sgd` takes the same `use_expmap` flag.

In practice, exponential maps provide better stability and convergence, especially for hyperbolic neural networks.

## Mixed Optimization

The optimizers seamlessly handle models with both Euclidean and hyperbolic parameters:

```python
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.optim import riemannian_adam

poincare = Poincare()

class MixedModel(nnx.Module):
    def __init__(self, rngs):
        # Euclidean linear layer
        self.fc1 = nnx.Linear(32, 64, rngs=rngs)

        # Hyperbolic layer (bias has manifold metadata)
        self.hyp = HypLinearPoincare(
            manifold_module=poincare,
            in_dim=64,
            out_dim=16,
            rngs=rngs
        )

        # Another Euclidean layer
        self.fc2 = nnx.Linear(16, 10, rngs=rngs)

model = MixedModel(nnx.Rngs(0))

# Optimizer handles all parameter types automatically
optimizer = nnx.Optimizer(model, riemannian_adam(learning_rate=0.001), wrt=nnx.Param)
```

The optimizer will:

- Apply standard Adam updates to `fc1` and `fc2` parameters
- Apply Riemannian Adam updates to `hyp.bias` (tagged with metadata)
- Apply Euclidean Adam updates to `hyp.kernel` (no metadata)

## Performance Considerations

!!! tip "JIT Compilation"
    Both optimizers are JIT-compatible. Use `nnx.jit`, not `jax.jit`: a
    `jax.jit` step drops the in-place parameter and optimizer updates.

    ```python
    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            return compute_loss(model, x, y)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss
    ```

## References

The Riemannian optimizers are based on:

- Bécigneul, G., & Ganea, O. (2019). "Riemannian Adaptive Optimization Methods." ICLR 2019.
- Bonnabel, S. (2013). "Stochastic gradient descent on Riemannian manifolds." IEEE TAC.

See the [User Guide](../user-guide/optimizers.md) for detailed explanations and best practices.
