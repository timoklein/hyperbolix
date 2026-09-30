# Batching & JIT Guide

Efficient JAX patterns for hyperbolic deep learning with vmap-native APIs and JIT compilation.

## Overview

Hyperbolix adopts a **vmap-native API design** where all manifold functions operate on single points/vectors. This design provides maximum flexibility and composability with JAX's transformation system.

!!! success "Key Design Principles"
    - Functions operate on **single points** with shape `(dim,)` or `(dim+1,)` (ambient)
    - Use `jax.vmap` for batch operations
    - Use `jax.jit` for compilation with appropriate static arguments
    - No built-in `axis` or `keepdim` parameters — compose transformations explicitly

## The vmap-Native API

### Single Point Operations

All manifold methods work with individual points:

```python
import jax.numpy as jnp
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Single points (intrinsic coordinates)
x = jnp.array([0.1, 0.2])  # Shape: (2,)
y = jnp.array([0.3, 0.4])  # Shape: (2,)

# Compute distance between two points
distance = poincare.dist(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)
print(distance)  # Scalar

# Exponential map from origin
v = jnp.array([0.5, 0.0])  # Tangent vector at origin
point = poincare.expmap_0(v, c=1.0)
print(point.shape)  # (2,)
```

### Batching with vmap

Use `jax.vmap` to process batches efficiently:

```python
import jax

poincare = Poincare()

# Batch of points
x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25], [0.05, 0.1]])  # (3, 2)
y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45], [0.2, 0.3]])   # (3, 2)

# Option 1: Explicit vmap
dist_fn = jax.vmap(poincare.dist, in_axes=(0, 0, None, None))
distances = dist_fn(x_batch, y_batch, 1.0, poincare.VERSION_MOBIUS_DIRECT)
print(distances.shape)  # (3,)

# Option 2: Inline vmap
distances = jax.vmap(
    lambda x, y: poincare.dist(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)
)(x_batch, y_batch)
```

### Understanding in_axes

The `in_axes` parameter specifies which axes to map over:

```python
# in_axes=(0, 0, None, None) means:
# - Map over axis 0 of first argument (x_batch)
# - Map over axis 0 of second argument (y_batch)
# - Don't map over curvature (c) — use same value for all
# - Don't map over version_idx (it indexes a fixed set of implementations, not data)
```

Common patterns:

```python
poincare = Poincare()

# Project batch of points
x_batch = jax.random.normal(jax.random.PRNGKey(0), (100, 16))
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x_batch, 1.0)

# Compute distances from single point to batch
origin = jnp.zeros(16)
x_batch = jax.random.normal(jax.random.PRNGKey(0), (100, 16)) * 0.3
distances = jax.vmap(
    lambda x: poincare.dist(origin, x, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)
)(x_batch)
print(distances.shape)  # (100,)

# Exponential map with batch of tangent vectors
v_batch = jax.random.normal(jax.random.PRNGKey(0), (100, 16))
base_point = jnp.zeros(16)
points = jax.vmap(
    lambda v: poincare.expmap(v, base_point, c=1.0)
)(v_batch)
print(points.shape)  # (100, 16)
```

## JIT Compilation

### Basic JIT Usage

Use `jax.jit` to compile functions and make repeated calls fast:

```python
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Without JIT
distance = poincare.dist(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)

# With JIT (version_idx marked static here for compile size; a dynamic value also works via lax.switch)
dist_jit = jax.jit(poincare.dist, static_argnames=['version_idx'])
distance = dist_jit(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)
```

### Static vs Dynamic Arguments

**Static arguments** are known at compile time and trigger recompilation if changed:

```python
# version_idx is static (integer constant)
dist_jit = jax.jit(poincare.dist, static_argnames=['version_idx'])

# These compile once and reuse:
d1 = dist_jit(x1, y1, c=1.0, version_idx=0)
d2 = dist_jit(x2, y2, c=1.5, version_idx=0)  # Reuses compilation

# This triggers recompilation (different version_idx):
d3 = dist_jit(x3, y3, c=1.0, version_idx=1)
```

**Dynamic arguments** can change without recompilation:

```python
# Curvature 'c' is dynamic (can vary)
d1 = dist_jit(x1, y1, c=1.0, version_idx=0)
d2 = dist_jit(x2, y2, c=2.5, version_idx=0)  # No recompilation needed
```

A new input shape or a new static `version_idx` triggers a recompile; a new `c` value does not.

!!! warning "Learnable Curvature"
    Don't mark `c` static: it recompiles for every value and blocks gradients. Learnable curvature comes from a `LearnableCurvature` module called inside the model (see [Manifolds](manifolds.md#working-with-curvature)).

### Combining vmap and jit

Put `jax.jit` outermost: vmap inside, jit around it (or around the whole train step).

```python
from functools import partial

from hyperbolix.manifolds import Poincare

poincare = Poincare()

x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25], [0.05, 0.1]])  # (3, 2)
y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45], [0.2, 0.3]])   # (3, 2)

# version_idx is bound before jit, so it stays static
dist_batched = jax.jit(
    jax.vmap(
        partial(poincare.dist, version_idx=poincare.VERSION_MOBIUS_DIRECT),
        in_axes=(0, 0, None),
    )
)
distances = dist_batched(x_batch, y_batch, 1.0)
print(distances.shape)  # (3,)
```

!!! tip "Why jit goes outermost"
    `jax.vmap(jax.jit(f))` gives the same numbers but traces the vmapped
    function again on every call. On CPU (Poincaré `dist`, dim 16) it was
    about 20x slower at batch 32 and 2.3x slower at batch 4096 than
    `jax.jit(jax.vmap(f))`.

## Neural Network Patterns

### Forward Pass

Flax NNX layers automatically handle batching:

```python
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Create layer
layer = HypLinearPoincare(
    manifold_module=poincare,
    in_dim=128,
    out_dim=64,
    rngs=nnx.Rngs(0)
)

# Batch input: (batch_size, in_dim)
x_batch = jax.random.normal(jax.random.PRNGKey(1), (32, 128)) * 0.3
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x_batch, 1.0)

# Forward pass handles batching internally
output = layer(x_proj, c=1.0)
print(output.shape)  # (32, 64)
```

### Activations

`hyp_relu` and the other hyperboloid activations take ambient points
`(..., dim+1)` with any leading batch axes, so they need no `vmap`. They are
for hyperboloid points only: don't apply them to Poincaré ball points.

```python
from hyperbolix.nn_layers import hyp_relu

# Single point (ambient coordinates, d+1 dims for hyperboloid)
x = jnp.array([1.5, 0.2, 0.3, 0.1])  # Ambient coordinates (4,)
activated = hyp_relu(x, c=1.0)

# Batch of points: no vmap needed
x_batch = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
activated_batch = hyp_relu(x_batch, c=1.0)
print(activated_batch.shape)  # (32, 4)
```

For a complete model, see [Training Workflows](training-workflows.md).

## Training Loop Patterns

### Efficient Training Step

Use `nnx.jit`, not `jax.jit`, for a train step: a `jax.jit` step drops the
in-place parameter and optimizer updates, so the model never trains.

```python
import optax
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincarePP

model = HypLinearPoincarePP(manifold_module=poincare, in_dim=16, out_dim=4, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, x_batch, y_batch, c):
    def loss_fn(model):
        preds = model(x_batch, c)
        return jnp.mean((preds - y_batch) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


x_batch = jax.vmap(poincare.expmap_0, in_axes=(0, None))(
    0.1 * jax.random.normal(jax.random.PRNGKey(0), (32, 16)), 1.0
)
y_batch = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (32, 4))
for step in range(5):
    loss = train_step(model, optimizer, x_batch, y_batch, 1.0)
    print(f"step {step}: loss = {loss:.4f}")  # decreases step by step
```

For a complete loop, see [Training Workflows](training-workflows.md).

## Performance Optimization Tips

### 1. Profile Before Optimizing

```python
import time

# Warmup JIT compilation
_ = dist_jit(x, y, c=1.0, version_idx=0)

# Time subsequent calls
start = time.time()
for _ in range(1000):
    _ = dist_jit(x, y, c=1.0, version_idx=0)
elapsed = time.time() - start
print(f"Time per call: {elapsed/1000*1e6:.2f} µs")
```

### 2. Memory vs Computation Trade-offs

```python
# Memory-efficient: Process in chunks
def process_large_batch(x_batch, chunk_size=1000):
    n = len(x_batch)
    results = []
    for i in range(0, n, chunk_size):
        chunk = x_batch[i:i+chunk_size]
        results.append(jax.vmap(some_fn)(chunk))
    return jnp.concatenate(results)

# Compute-efficient: Process all at once (may OOM)
def process_all_at_once(x_batch):
    return jax.vmap(some_fn)(x_batch)
```

## Common Pitfalls

### Pitfall 1: Shape Mismatches with vmap

```python
# WRONG: Incompatible in_axes
x_batch = jnp.array([[0.1, 0.2]])  # (1, 2)
y_batch = jnp.array([[0.3, 0.4]])  # (1, 2)
c_batch = jnp.array([1.0, 1.5])    # (2,)

distances = jax.vmap(poincare.dist, in_axes=(0, 0, 0))(
    x_batch, y_batch, c_batch  # Shape mismatch: (1,) vs (2,)
)

# CORRECT: Broadcast curvature or use same value
distances = jax.vmap(poincare.dist, in_axes=(0, 0, None))(
    x_batch, y_batch, 1.0
)
```

## Why JIT and vmap Help

`jax.jit` removes per-call Python dispatch and tracing overhead by compiling a function to XLA once and reusing the compiled kernel on subsequent calls. `jax.vmap` replaces an explicit Python loop over single-point manifold operations with a single batched XLA kernel, avoiding per-element Python overhead. Both effects are largest for small, frequently-called per-point operations — exactly the vmap-native functions in this library — and matter less as the per-call workload (batch size, dimension) grows and Python overhead becomes a smaller fraction of total runtime. Actual speedups depend on hardware, batch size, and dimensionality; profile your own workload with a `jax.jit` warmup (see "Profile Before Optimizing" above) rather than assuming a fixed multiplier.

## See Also

- [Manifolds API](../api-reference/manifolds.md): Manifold function signatures
- [NN Layers API](../api-reference/nn-layers/index.md): Layer implementations
- [Training Workflows](training-workflows.md): Complete training examples
- [Numerical Stability](numerical-stability.md): Float precision considerations
