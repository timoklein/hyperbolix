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
