# Riemannian Optimizer Implementation Plan

## Key Insights

1. **Most parameters are Euclidean**: Only Poincaré bias terms in non-PP layers live on manifolds
2. **Hyperboloid never needs manifold params**: No vector addition, so no bias on manifold
3. **Minimal scope**: Only 2 layer types need updates (non-PP Poincaré layers)

## Architecture Overview

### Three Components

1. **Manifold Metadata Wrapper** (`src/hyperbolix_jax/optim/manifold_param.py`)
   - Lightweight pytree node to mark manifold parameters
   - Stores: manifold module reference + curvature
   - Default: all params are Euclidean unless explicitly marked

2. **Riemannian Optimizers** (`src/hyperbolix_jax/optim/`)
   - `riemannian_sgd()`: Momentum + parallel transport
   - `riemannian_adam()`: First/second moment + parallel transport
   - Implemented as Optax `GradientTransformation`
   - Auto-detect metadata, apply `egrad2rgrad` to manifold params only

3. **Layer Annotations** (minimal changes)
   - Update only: `HypLinearPoincare`, `HypRegressionPoincare` (non-PP)
   - Mark bias parameters with manifold metadata
   - PP variants and Hyperboloid: no changes (all Euclidean)

## Implementation Algorithm

### Riemannian SGD
```
For each parameter:
  1. If manifold param: grad = manifold.egrad2rgrad(grad, param, c)
  2. Apply momentum: m = momentum * m + grad
  3. If manifold param:
     - new_param = manifold.expmap(-lr * m, param, c)  # or retraction
     - m = manifold.ptransp(m, param, new_param, c)
  4. Else: new_param = param - lr * m
```

### Riemannian Adam
```
For each parameter:
  1. If manifold param: grad = manifold.egrad2rgrad(grad, param, c)
  2. Update moments: m1 = beta1*m1 + (1-beta1)*grad
                     m2 = beta2*m2 + (1-beta2)*grad^2
  3. Bias correction + compute update
  4. If manifold param:
     - new_param = manifold.expmap(-lr * update, param, c)
     - m1 = manifold.ptransp(m1, param, new_param, c)
     - m2 = manifold.ptransp(m2, param, new_param, c)  # element-wise
  5. Else: new_param = param - lr * update
```

## File Structure

```
src/hyperbolix_jax/optim/
├── __init__.py                    # NEW: exports riemannian_sgd, riemannian_adam
├── manifold_param.py              # NEW: ManifoldMetadata pytree wrapper
├── riemannian_sgd.py              # NEW: RSGD implementation
└── riemannian_adam.py             # NEW: RAdam implementation

src/hyperbolix_jax/nn_layers/
├── poincare_linear.py             # MODIFY: annotate HypLinearPoincare.bias
└── poincare_regression.py         # MODIFY: annotate HypRegressionPoincare.bias

tests/jax/
└── test_optimizers.py             # NEW: port from tests/test_optimizers.py
```

## Testing Strategy

Port existing PyTorch tests (`tests/test_optimizers.py`):
- Simple convergence: move point toward target on manifold
- Test both `expmap_update=True/False`
- Verify momentum/moment transport
- No extensive convergence comparisons needed initially

## Design Decisions

1. **Idiomatic JAX**: Pure functions, pytrees, Optax patterns
2. **Opt-in complexity**: Metadata only where needed (rare)
3. **Composable**: Works with Optax schedules, clipping, etc.
4. **Minimal invasiveness**: Only 2 layer files need updates
5. **Support both expmap and retraction**: User choice (exact vs fast)

## Usage Example

```python
import jax
from flax import nnx
import hyperbolix_jax.manifolds.poincare as poincare
from hyperbolix_jax.optim import riemannian_sgd
from hyperbolix_jax.nn_layers import HypLinearPoincare

# Create layer (bias auto-annotated with manifold metadata)
layer = HypLinearPoincare(poincare, in_dim=10, out_dim=5, rngs=nnx.Rngs(0))

# Create optimizer
optimizer = riemannian_sgd(learning_rate=0.01, momentum=0.9)
opt_state = optimizer.init(nnx.state(layer))

# Training step
def loss_fn(model, x):
    return jnp.sum(model(x, c=1.0) ** 2)

loss, grads = nnx.value_and_grad(loss_fn)(layer, x)
updates, opt_state = optimizer.update(grads, opt_state, nnx.state(layer))
nnx.update(layer, optax.apply_updates(nnx.state(layer), updates))
```

## Open Questions

1. **Pytree metadata approach**: Use `jax.tree_util.register_pytree_with_keys` or custom wrapper?
2. **Moment transport for Adam**: Transport m2 element-wise or as vector?
3. **Default to expmap or retraction**: Which should be default for updates?
4. **Learning rate schedules**: How to compose with Optax schedules?
