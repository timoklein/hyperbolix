# JAX & Flax Migration Plan (Simplified)

## Objectives
- Replace PyTorch with JAX-based equivalents (JAX, Flax NNX, Optax) with a straightforward port-and-test approach
- Preserve public API semantics where practical
- Maintain numerical stability and leverage JIT/VMAP for performance

## Scope Reality Check
This is a **~5.6k LOC codebase** (11 core Python modules + 8 test files), not an enterprise system. The previous 10-phase plan was over-engineered. This simplified plan focuses on direct porting with integrated testing.

## Simple 3-Phase Approach

### Phase 1: Core Geometry (1-2 days)
**Port manifolds + math utilities** - the foundation everything else depends on.

Files to port:
- `src/manifolds/euclidean.py` → `src/hyperbolix_jax/manifolds/euclidean.py`
- `src/manifolds/hyperboloid.py` → `src/hyperbolix_jax/manifolds/hyperboloid.py`
- `src/manifolds/poincare.py` → `src/hyperbolix_jax/manifolds/poincare.py`
- `src/utils/math_utils.py` → `src/hyperbolix_jax/utils/math_utils.py`

Migration approach:
- Replace `torch.tensor` → `jnp.array`, `torch.nn.Parameter` → function arguments
- Replace `torch.jit.script` decorators with `@jax.jit` (add `static_argnums` as needed)
- Remove in-place ops (`.copy_()`, `.clamp_()`) → pure functional versions
- Add `@jax.jit` and `jax.vmap` decorators after correctness is verified

Testing:
- Port existing pytest tests, run side-by-side comparisons
- Focus on: distance computation, exp/log maps, gradient flow
- Use `jnp.allclose` with reasonable tolerances (1e-6 for float32, 1e-12 for float64)

### Phase 2: Optimizers (1 day)
**Port Riemannian optimizers** - state management as pytrees.

Files to port:
- `src/optim/riemannian_sgd.py` → `src/hyperbolix_jax/optim/rsgd.py`
- `src/optim/riemannian_adam.py` → `src/hyperbolix_jax/optim/radam.py`

Migration approach:
- Use Optax's state management patterns (pytree dictionaries)
- Momentum/Adam state becomes nested dicts: `{'momentum': ..., 'step': ...}`
- Implement `init(params)` and `update(grads, state, params)` functions
- Support both exponential map and retraction-based updates

Testing:
- Simple optimization loops on toy problems
- Verify state persistence and gradient updates match PyTorch within tolerance

### Phase 3: Layers & Utils (1-2 days)
**Port neural network layers and supporting utilities**.

Files to port:
- `src/nn_layers/*.py` → `src/hyperbolix_jax/nn_layers/` (using Flax NNX)
- `src/utils/horo_pca.py` → `src/hyperbolix_jax/utils/horo_pca.py`
- `src/utils/vis_utils.py` → keep as-is (visualization works with both)

Migration approach:
- Use Flax NNX for stateful layers (simpler than Linen for PyTorch users)
- Basic pattern: `class HypLinear(nnx.Module)` with `__init__` and `__call__`
- Port high-value layers first (linear layers, HNN), defer RL variants if complex

Testing:
- Forward pass tests with known inputs/outputs
- Gradient checks using `jax.grad`
- Small training loops to verify end-to-end functionality

## Implementation Notes

**Manifolds**:
- Keep class-based API similar to PyTorch version
- Add dtype/tolerance as constructor args (simpler than elaborate config systems)
- Use `@partial(jax.jit, static_argnums=(...))` for args like `keepdim`, `axis`

**Optimizers**:
- Follow Optax conventions: `(params, opt_state) → (updates, new_opt_state)`
- Provide thin wrappers if you want PyTorch-style `.step()` API

**Layers**:
- Flax NNX is Pythonic and close to PyTorch's `nn.Module`
- Use `nnx.Param` for trainable weights, standard JAX arrays for buffers

**Testing**:
- Don't build elaborate fixture infrastructure - just test inline
- Use `JAX_ENABLE_X64=1` env var when you need float64 precision
- Parametrize tests over dtypes if needed: `@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])`

## Timeline
- **Day 1-2**: Phase 1 (manifolds working with tests passing)
- **Day 3**: Phase 2 (optimizers functional)
- **Day 4-5**: Phase 3 (layers ported, end-to-end validation)

Total: **~1 focused week** for a complete port with testing.

## What We're NOT Doing
- ❌ Elaborate dual-backend testing infrastructure
- ❌ Separate "performance validation" phases with benchmarking frameworks
- ❌ Weight conversion utilities (just retrain if needed - it's hyperbolic geometry research code)
- ❌ Complex config management systems (simple constructor args suffice)
- ❌ Separate `*_ops.py` modules (keep operations with their classes)

---

## JIT Compilation Strategy

### Overview
All manifold operations in `src/hyperbolix_jax/manifolds/` are **pure functions** designed to be JIT-compatible. However, several parameters must be marked as **static** when applying `jax.jit` to enable optimal compilation.

### Static Arguments Required

For successful JIT compilation, the following parameters **must** be static (i.e., known at compile time):

#### All Manifold Operations
- **`axis`**: Used for array indexing and reductions. Different values produce different array shapes.
- **`keepdim`**: Controls whether reduced dimensions are kept. Affects output shape.
- **`backproject`**: Boolean flag controlling conditional projection operations (if/else branches).
- **`version`**: String parameter selecting algorithm variants (if/elif chains).

#### Dynamic Arguments (Can Vary at Runtime)
- **`c`**: Curvature parameter. Keep dynamic to support learnable curvature in neural networks.
- **`x, y, v`**: Input arrays (points, tangent vectors). These are the data being transformed.

### JIT Compilation Examples

#### Basic Usage Pattern
```python
from functools import partial
import jax
import hyperbolix_jax.manifolds.poincare as poincare

# Create JIT-compiled version with static arguments
poincare_dist_jit = jax.jit(
    poincare.dist,
    static_argnames=['axis', 'keepdim', 'version']
)

# Use like normal function
x = jnp.array([[0.1, 0.2]])
y = jnp.array([[0.3, 0.4]])
distance = poincare_dist_jit(x, y, c=1.0, axis=-1, keepdim=True, version='mobius_direct')
```

#### Pre-Configured JIT Functions
```python
# For convenience, create partially applied versions
from functools import partial

# Poincaré ball operations with sensible defaults
poincare_ops = {
    'dist': jax.jit(poincare.dist, static_argnames=['axis', 'keepdim', 'version']),
    'expmap': jax.jit(poincare.expmap, static_argnames=['axis', 'backproject']),
    'logmap': jax.jit(poincare.logmap, static_argnames=['axis', 'backproject']),
    'proj': jax.jit(poincare.proj, static_argnames=['axis']),
}

# Use in training loop
def loss_fn(params, x, y, c):
    pred = poincare_ops['expmap'](params['v'], x, c=c, axis=-1, backproject=True)
    return poincare_ops['dist'](pred, y, c=c, axis=-1, keepdim=False, version='mobius_direct')

# JIT the entire loss function
loss_jit = jax.jit(loss_fn)
```

#### Handling Multiple Versions
```python
# Compile separate functions for each version if needed at runtime
dist_mobius = jax.jit(poincare.dist, static_argnames=['axis', 'keepdim', 'version'])
dist_metric_tensor = jax.jit(poincare.dist, static_argnames=['axis', 'keepdim', 'version'])

# JAX will compile different versions based on static argument values
d1 = dist_mobius(x, y, c=1.0, version='mobius_direct', axis=-1, keepdim=True)
d2 = dist_metric_tensor(x, y, c=1.0, version='metric_tensor', axis=-1, keepdim=True)
```

### Validation Functions

**Important:** The following functions return Python `bool` and **cannot** be JIT-compiled directly:
- `is_in_manifold(x, c, axis)`
- `is_in_tangent_space(v, x, c, axis)`

These are intended for **testing and validation only**, not for use inside JIT-compiled training loops.

**Usage Pattern:**
```python
# ✓ Correct: Use for validation before/after JIT-compiled code
x = initialize_points(...)
assert poincare.is_in_manifold(x, c=1.0, axis=-1), "Invalid manifold points"

loss_jit = jax.jit(loss_fn)
result = loss_jit(x, ...)

# ✗ Incorrect: Do not use inside JIT-compiled functions
@jax.jit
def bad_function(x, c):
    if poincare.is_in_manifold(x, c, axis=-1):  # Will fail during tracing!
        return x
```

### Known JIT Compatibility Issues

#### 1. Control Flow on Non-Static Arguments (58 occurrences)
All `if backproject:`, `if version == "..."`, and axis-dependent indexing require static arguments.

**Solution:** Use `static_argnames` as shown in examples above.

#### 2. Assertions in NN Layers (17 assertions)
Layers in `nn_layers/` contain assertions like `assert axis == -1`. These will fail during JIT compilation if the asserted value is not statically known.

**Solution:** For now, ensure assertions are satisfied. Future work: Move to validation functions in `__init__`.

#### 3. Dtype-Dependent Logic (10 locations)
Functions like `_get_max_norm_eps(x)` branch on `x.dtype`. This works as long as dtype is known at trace time (typical case).

**Solution:** No action needed unless you dynamically change dtypes at runtime (rare).

### Performance Considerations

1. **First Call Cost:** First invocation triggers compilation (may take 1-10 seconds depending on function complexity).
2. **Cached Compilation:** Subsequent calls with same static arguments are fast (compiled code is cached).
3. **Recompilation Triggers:** Changing any static argument triggers recompilation.
4. **Memory:** Each unique combination of static arguments creates a new compiled function (binary size increases).

### Recommended Workflow

```python
# 1. Import manifold operations
import hyperbolix_jax.manifolds.poincare as poincare
import hyperbolix_jax.manifolds.hyperboloid as hyperboloid

# 2. Create JIT-compiled versions at module level (compile once)
poincare_dist_jit = jax.jit(poincare.dist, static_argnames=['axis', 'keepdim', 'version'])
poincare_expmap_jit = jax.jit(poincare.expmap, static_argnames=['axis', 'backproject'])

# 3. Compose into larger JIT-compiled functions
@jax.jit
def training_step(params, batch_x, batch_y, c):
    embeddings = forward_pass(params, batch_x, c)  # Uses manifold ops internally
    distances = poincare_dist_jit(embeddings, batch_y, c=c, axis=-1, keepdim=False, version='mobius_direct')
    return jnp.mean(distances ** 2)

# 4. Use value_and_grad for optimization
loss_and_grad = jax.jit(jax.value_and_grad(training_step))
```

### Testing JIT Compatibility

Verify your usage is JIT-compatible:
```python
import jax

# Test basic JIT compilation
@jax.jit
def test_jit(x, y, c):
    return poincare.dist(x, y, c=c, axis=-1, keepdim=True, version='mobius_direct')

# If this runs without errors, JIT compilation works
x = jnp.array([[0.1, 0.2]])
y = jnp.array([[0.3, 0.4]])
result = test_jit(x, y, c=1.0)
print(result)

# Test gradient computation
grad_fn = jax.jit(jax.grad(lambda x: jnp.sum(poincare.dist(x, y, c=1.0, axis=-1, keepdim=True, version='mobius_direct'))))
gradients = grad_fn(x)
print(gradients)
```

### Future Work

- [ ] Create `manifolds/jit_wrappers.py` with pre-configured JIT functions
- [ ] Refactor validation functions to return JAX arrays (optional)
- [ ] Add `checkify` for runtime assertions in JIT-compiled code
- [ ] Comprehensive JIT performance benchmarks vs non-JIT
