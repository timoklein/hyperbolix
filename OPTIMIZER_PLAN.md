# Riemannian Optimizer Implementation Plan

## Key Insights

1. **Most parameters are Euclidean**: Only Poincaré bias terms in non-PP layers live on manifolds
2. **Hyperboloid never needs manifold params**: No vector addition, so no bias on manifold
3. **Minimal scope**: Only 2 layer types need updates (non-PP Poincaré layers)

## Design Decision: Use NNX Variable Metadata (Not Inheritance)

Instead of creating custom Variable types or inheriting from `nnx.Param`, we leverage **Flax NNX's built-in metadata system**. This is more elegant because:

1. **All params remain `nnx.Param`** - single variable type throughout
2. **Metadata is the designed pattern** - `Variable.__slots__` includes `_var_metadata` for this exact use case
3. **Seamless Optax integration** - works naturally with standard `GradientTransformation` interface
4. **Minimal invasiveness** - only 2 layer files need updates
5. **Serialization-friendly** - store string identifiers instead of module references
6. **Composable** - works with Optax combinators (chain, schedules, etc.)

## Architecture Overview

### Three Components

1. **Manifold Metadata Utilities** (`src/hyperbolix_jax/optim/manifold_metadata.py`)
   - Helper function to mark parameters with manifold info via metadata
   - Manifold registry mapping string IDs to manifold modules
   - Utility to extract metadata from parameter pytrees
   - Example: `mark_manifold_param(param, 'poincare', curvature=1.0)`

2. **Riemannian Optimizers** (`src/hyperbolix_jax/optim/`)
   - `riemannian_sgd()`: Returns standard Optax `GradientTransformation`
   - `riemannian_adam()`: Returns standard Optax `GradientTransformation`
   - Both detect metadata and apply appropriate updates automatically
   - Compatible with `nnx.Optimizer` wrapper
   - Auto-detect metadata, apply `egrad2rgrad` to manifold params only

3. **Layer Annotations** (minimal changes)
   - Update only: `HypLinearPoincare`, `HypRegressionPoincare` (non-PP)
   - Mark bias parameters with manifold metadata using helper function
   - PP variants and Hyperboloid: no changes (all Euclidean)

## Implementation Details

### Manifold Metadata System

```python
# Mark parameter as manifold param (stores string ID + curvature in metadata)
self.bias = mark_manifold_param(
    nnx.Param(init_value),
    manifold_type='poincare',  # String identifier
    curvature=1.0  # or callable: lambda: self.c.value
)

# In optimizer: detect metadata and extract manifold info
manifold_info = get_manifold_info(param)
if manifold_info is not None:
    manifold, c = manifold_info  # Get manifold module from registry
    # Apply Riemannian operations...
```

### Riemannian SGD Algorithm
```
For each parameter:
  1. Check metadata: manifold_info = get_manifold_info(param)
  2. If manifold param:
     - grad = manifold.egrad2rgrad(grad, param, c)
     - m = momentum * m + grad
     - new_param = manifold.expmap(-lr * m, param, c)  # or retraction
     - m = manifold.ptransp(m, param, new_param, c)
  3. Else (Euclidean):
     - m = momentum * m + grad
     - new_param = param - lr * m
```

### Riemannian Adam Algorithm
```
For each parameter:
  1. Check metadata: manifold_info = get_manifold_info(param)
  2. If manifold param:
     - grad = manifold.egrad2rgrad(grad, param, c)
     - m1 = beta1*m1 + (1-beta1)*grad
     - m2 = beta2*m2 + (1-beta2)*grad^2
     - (bias correction + compute direction)
     - new_param = manifold.expmap(-lr * direction, param, c)
     - m1 = manifold.ptransp(m1, param, new_param, c)
     - m2 = manifold.ptransp(m2, param, new_param, c)  # optional
  3. Else (Euclidean):
     - Standard Adam update
```

## File Structure

```
src/hyperbolix_jax/optim/
├── __init__.py                    # NEW: exports + manifold registry setup
├── manifold_metadata.py           # NEW: metadata helpers + manifold registry
├── riemannian_sgd.py              # NEW: RSGD as Optax GradientTransformation
└── riemannian_adam.py             # NEW: RAdam as Optax GradientTransformation

src/hyperbolix_jax/nn_layers/
├── poincare_linear.py             # MODIFY: mark HypLinearPoincare.bias with metadata
└── poincare_regression.py         # MODIFY: mark HypRegressionPoincare.bias with metadata

tests/jax/
└── test_optimizers.py             # NEW: test metadata system + convergence
```

## Testing Strategy

**Test Cases**:
1. **Metadata Detection**: Test `mark_manifold_param()` attaches metadata correctly
2. **Simple Convergence (RSGD)**: Move point toward target on Poincaré ball
3. **Simple Convergence (RAdam)**: Same with adaptive learning rates
4. **Momentum Transport**: Verify momentum is parallel transported correctly
5. **Mixed Parameters**: Model with both Euclidean and manifold params
6. **Integration**: End-to-end with `nnx.Optimizer` wrapper

**Key Tests**:
- Test both `use_expmap=True/False` (expmap vs retraction)
- Verify parameters stay on manifold after updates
- No extensive convergence comparisons needed initially

## Design Rationale

1. **Why Metadata Over Custom Variable Types?**
   - Single source of truth: all trainable params are `nnx.Param`
   - Optax compatibility: `GradientTransformation` interface unchanged
   - Automatic serialization: metadata checkpointed with parameters
   - Type safety: no need to update `wrt` filters everywhere

2. **Why String Identifiers for Manifolds?**
   - Serialization: can't pickle module references easily
   - Flexibility: easy to add new manifolds via registry
   - Testing: can mock manifolds for tests

3. **Why Standard Optax GradientTransformation?**
   - Composable: works with `optax.chain()`, schedules, clipping
   - Familiar API: no learning curve for users
   - Ecosystem: leverages existing Optax utilities

4. **Minimal Invasiveness**: Only 2 layer files need updates
5. **User Choice**: Support both expmap (exact) and retraction (fast)

## Usage Example

```python
import jax
import jax.numpy as jnp
from flax import nnx

# Import manifold and layers
from hyperbolix_jax.manifolds import poincare
from hyperbolix_jax.nn_layers import HypLinearPoincare

# Import optimizer
from hyperbolix_jax.optim import riemannian_sgd

# 1. Create model with manifold parameters
layer = HypLinearPoincare(
    poincare,
    in_dim=10,
    out_dim=5,
    rngs=nnx.Rngs(0)
)
# Note: layer.bias is automatically marked with manifold metadata

# 2. Create Riemannian optimizer (standard Optax interface)
tx = riemannian_sgd(learning_rate=0.01, momentum=0.9, use_expmap=True)
optimizer = nnx.Optimizer(layer, tx, wrt=nnx.Param)

# 3. Training step
x = jax.random.normal(jax.random.key(1), (32, 10))

def loss_fn(model, x):
    y = model(x, c=1.0)
    return jnp.sum(y ** 2)

# Compute gradients
grads = nnx.grad(loss_fn)(layer, x)

# Update (automatically detects manifold params via metadata)
optimizer.update(layer, grads)

# The optimizer automatically:
# - Detects layer.bias has manifold metadata
# - Applies egrad2rgrad conversion
# - Updates via exponential map
# - Transports momentum
# - Leaves layer.weight (Euclidean) unchanged
```

### Advanced: Learnable Curvature

```python
class HypLinearPoincare(nnx.Module):
    def __init__(self, manifold_module, in_dim, out_dim, *, rngs, ...):
        # Learnable curvature
        self.c = nnx.Param(jnp.array(1.0))

        # Weight (Euclidean)
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)))

        # Bias with callable curvature (reads current value each step)
        bias_init = jax.random.normal(rngs.params(), (1, out_dim)) * 0.01
        self.bias = mark_manifold_param(
            nnx.Param(bias_init),
            manifold_type='poincare',
            curvature=lambda: self.c.value,  # Callable!
        )
```

## Open Questions & Future Work

1. **Second moment transport in Adam**:
   - Current: transport both m1 and m2 for symmetry
   - Alternative: only transport m1 (per Bécigneul & Ganea 2019)
   - Make configurable: `transport_second_moment=True/False`?

2. **Numerical stability**:
   - Add projection after updates?
   - Tolerance checks for manifold constraints?
   - Gradient clipping when curvature is learnable?

3. **Performance profiling**:
   - Overhead vs standard optimizers
   - Expmap vs retraction performance tradeoff

4. **Composability testing**:
   - Test with `optax.chain()` for LR schedules, weight decay
   - Verify with `optax.apply_if_finite()` for gradient clipping
