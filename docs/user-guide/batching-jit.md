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
# - Don't map over version_idx — static argument
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

Use `jax.jit` to compile functions for 10-100x speedup:

```python
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Without JIT
distance = poincare.dist(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)

# With JIT (version_idx is static since it controls which kernel to run)
dist_jit = jax.jit(poincare.dist, static_argnames=['version_idx'])
distance = dist_jit(x, y, c=1.0, version_idx=poincare.VERSION_MOBIUS_DIRECT)
```

!!! tip "JIT Performance"
    - **First call**: Slow (compilation overhead, 100ms-1s)
    - **Subsequent calls**: Fast (10-100x speedup)
    - Most beneficial for large batches (1000+) and high dimensions (128+)

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

!!! warning "Learnable Curvature"
    Keep curvature parameter `c` **dynamic** (not static) to support gradient-based learning of curvature values during training.

### Combining vmap and jit

The order matters for performance:

```python
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Pattern 1: JIT then vmap (RECOMMENDED)
@jax.jit
def distance_fn(x, y, c):
    return poincare.dist(x, y, c, version_idx=poincare.VERSION_MOBIUS_DIRECT)

distances = jax.vmap(distance_fn, in_axes=(0, 0, None))(x_batch, y_batch, 1.0)

# Pattern 2: vmap then JIT
dist_batched = jax.vmap(poincare.dist, in_axes=(0, 0, None, None))
distances = jax.jit(dist_batched, static_argnames=['version_idx'])(
    x_batch, y_batch, 1.0, poincare.VERSION_MOBIUS_DIRECT
)

# Pattern 3: Combined (one-liner)
distances = jax.jit(
    jax.vmap(poincare.dist, in_axes=(0, 0, None, None)),
    static_argnames=['version_idx']
)(x_batch, y_batch, 1.0, poincare.VERSION_MOBIUS_DIRECT)
```

!!! tip "Best Practice"
    JIT the inner function and vmap the outer function for best performance and flexibility.

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

### Activations with vmap

Hyperbolic activations are functional and need explicit batching:

```python
from hyperbolix.nn_layers import hyp_relu

# Single point (ambient coordinates, d+1 dims for hyperboloid)
x = jnp.array([1.5, 0.2, 0.3, 0.1])  # Ambient coordinates (4,)
activated = hyp_relu(x, c=1.0)

# Batch of points - use vmap
x_batch = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
activated_batch = jax.vmap(lambda x: hyp_relu(x, c=1.0))(x_batch)
print(activated_batch.shape)  # (32, 4)
```

### Complete Model with JIT

```python
from hyperbolix.nn_layers import HypLinearPoincare, hyp_relu
from hyperbolix.manifolds import Poincare

poincare = Poincare()

class HyperbolicClassifier(nnx.Module):
    def __init__(self, rngs):
        self.layer1 = HypLinearPoincare(poincare, 784, 256, rngs=rngs)
        self.layer2 = HypLinearPoincare(poincare, 256, 128, rngs=rngs)
        self.layer3 = HypLinearPoincare(poincare, 128, 10, rngs=rngs)

    def __call__(self, x, c=1.0):
        x = self.layer1(x, c)
        # vmap activation over batch
        x = jax.vmap(lambda xi: hyp_relu(xi, c))(x)

        x = self.layer2(x, c)
        x = jax.vmap(lambda xi: hyp_relu(xi, c))(x)

        x = self.layer3(x, c)
        return x

# Create model
model = HyperbolicClassifier(rngs=nnx.Rngs(0))

# JIT the forward pass
@jax.jit
def forward(model, x, c):
    return model(x, c)

# Use with batch
x_batch = jax.random.normal(jax.random.PRNGKey(1), (32, 784)) * 0.1
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x_batch, 1.0)
logits = forward(model, x_proj, c=1.0)
print(logits.shape)  # (32, 10)
```

## Training Loop Patterns

### Efficient Training Step

```python
from flax import nnx
from hyperbolix.manifolds import Poincare
from hyperbolix.optim import riemannian_adam

poincare = Poincare()

@jax.jit
def train_step(model, optimizer, x_batch, y_batch, c):
    """Single training step with JIT compilation."""
    def loss_fn(model):
        preds = model(x_batch, c)
        return jnp.mean((preds - y_batch) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # Project to manifold
        x_batch = jax.vmap(poincare.proj, in_axes=(0, None))(x_batch, 1.0)

        # Single JIT-compiled step
        loss = train_step(model, optimizer, x_batch, y_batch, c=1.0)

        print(f"Loss: {loss:.4f}")
```

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

### 2. Minimize Recompilation

```python
# BAD: Different shapes trigger recompilation
d1 = dist_jit(x1, y1, c=1.0, version_idx=0)  # Compile for shape (16,)
d2 = dist_jit(x2, y2, c=1.0, version_idx=0)  # Recompile for shape (32,)

# GOOD: Use consistent shapes
x_batch = jnp.array([[0.1, 0.2], [0.3, 0.4]])
distances = jax.vmap(dist_jit, in_axes=(0, 0, None, None))(
    x_batch[:, 0], x_batch[:, 1], 1.0, poincare.VERSION_MOBIUS_DIRECT
)
```

### 3. Use Static Arguments Appropriately

```python
# GOOD: Keep curvature dynamic
@jax.jit
def process_batch(x_batch, c):
    return jax.vmap(
        lambda x: poincare.proj(x, c)  # Simple projection
    )(x_batch)

# BAD: Making everything static reduces flexibility
@jax.jit
def process_batch_bad(x_batch):  # c=1.0 hardcoded
    return jax.vmap(
        lambda x: poincare.proj(x, c=1.0)
    )(x_batch)  # Can't change curvature without recompilation
```

### 4. Batch Size Considerations

```python
# Small batches: Less JIT benefit
x_small = jax.random.normal(jax.random.PRNGKey(0), (10, 128))
# ~10-20x speedup

# Large batches: Maximum JIT benefit
x_large = jax.random.normal(jax.random.PRNGKey(0), (1000, 128))
# ~50-100x speedup
```

### 5. Memory vs Computation Trade-offs

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

### Pitfall 1: Forgetting to vmap Activations

```python
# WRONG: Activation expects single point
x = layer(x_batch, c=1.0)  # (batch, dim+1) for hyperboloid
activated = hyp_relu(x, c=1.0)  # May work but semantics unclear

# CORRECT: Explicit vmap
activated = jax.vmap(lambda xi: hyp_relu(xi, c=1.0))(x)

# ALSO CORRECT: hyp_relu handles batches
activated = hyp_relu(x, c=1.0)  # Directly works on (batch, dim+1)
```

### Pitfall 2: Shape Mismatches with vmap

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

### Pitfall 3: Static Curvature

```python
# WRONG: Can't learn curvature
@jax.jit
def model_forward(x, c=1.0):  # c fixed at compile time
    return poincare.proj(x, c)

# CORRECT: Keep c dynamic
@jax.jit
def model_forward(x, c):  # c can vary
    return poincare.proj(x, c)
```

## Benchmark Results

Typical speedups on M1/M2 Mac or modern GPU:

| Operation | Batch Size | No JIT | With JIT | Speedup |
|-----------|------------|--------|----------|---------|
| Distance (dim=128) | 100 | 12 ms | 0.8 ms | 15x |
| Distance (dim=128) | 1000 | 120 ms | 1.5 ms | 80x |
| Expmap (dim=256) | 100 | 18 ms | 1.2 ms | 15x |
| Linear layer forward | 1000 | 45 ms | 2.1 ms | 21x |
| Full model (3 layers) | 1000 | 150 ms | 6.5 ms | 23x |

Run benchmarks yourself:

```bash
uv run pytest benchmarks/ --benchmark-only -v
```

## See Also

- [Manifolds API](../api-reference/manifolds.md): Manifold function signatures
- [NN Layers API](../api-reference/nn-layers.md): Layer implementations
- [Training Workflows](training-workflows.md): Complete training examples
- [Numerical Stability](numerical-stability.md): Float precision considerations
