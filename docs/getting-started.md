# Getting Started

This guide will help you install Hyperbolix and run your first examples.

## Installation

### Requirements

- Python 3.12 or higher
- JAX 0.4.20+ (with CPU or GPU support)
- Flax NNX 0.12.0+

### Install from Source

```bash
git clone https://github.com/hyperbolix/hyperbolix.git
cd hyperbolix
uv sync  # or pip install -e .
```

For GPU support, install JAX with CUDA:

```bash
uv pip install "jax[cuda12]>=0.4.20"
```

## Quick Start: Distance Computation

Let's compute distances on the Poincaré ball:

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import poincare

# Create two points
x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, -0.1])
c = 1.0  # Curvature

# Project to manifold (ensures points lie on Poincaré ball)
x_proj = poincare.proj(x, c)
y_proj = poincare.proj(y, c)

# Compute hyperbolic distance
distance = poincare.dist(x_proj, y_proj, c, version_idx=0)
print(f"Distance: {distance:.4f}")
```

## Batching with vmap

Hyperbolix uses a **vmap-native API**: functions operate on single points, and you use `jax.vmap` for batching:

```python
# Batch of 100 points
key = jax.random.PRNGKey(0)
x_batch = jax.random.normal(key, (100, 2)) * 0.3
y_batch = jax.random.normal(jax.random.PRNGKey(1), (100, 2)) * 0.3

# Project each point (batched operation)
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x_batch, c)
y_proj = jax.vmap(poincare.proj, in_axes=(0, None))(y_batch, c)

# Compute pairwise distances
distances = jax.vmap(poincare.dist, in_axes=(0, 0, None, None))(
    x_proj, y_proj, c, 0
)
print(f"Distances shape: {distances.shape}")  # (100,)
```

## Key Concepts

### Manifolds

Hyperbolix provides three manifold types:

- **Euclidean**: Flat space (baseline)
- **Poincaré Ball**: Conformal model (angles preserved)
- **Hyperboloid**: Lorentz model (natural for convolutions)

### Curvature Parameter

The curvature `c` controls the "amount of hyperbolicity":

- `c = 0`: Euclidean space (flat)
- `c = 1`: Unit curvature (standard hyperbolic space)
- `c > 1`: Higher curvature (more curved)

Pass `c` at call time for maximum flexibility:

```python
# Different curvatures
dist_c1 = poincare.dist(x, y, c=1.0, version_idx=0)
dist_c2 = poincare.dist(x, y, c=2.0, version_idx=0)
```

### Version Parameter

Many manifold functions accept a `version` parameter for numerical stability:

```python
# Poincaré distance has 4 versions
dist_v0 = poincare.dist(x, y, c, version_idx=0)  # Fastest
dist_v1 = poincare.dist(x, y, c, version_idx=1)  # Metric tensor
dist_v2 = poincare.dist(x, y, c, version_idx=2)  # Lorentzian proxy
dist_v3 = poincare.dist(x, y, c, version_idx=3)  # Conformal factor
```

## Building a Neural Network

Here's a simple 2-layer hyperbolic network:

```python
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.manifolds import poincare

class SimpleHypNet(nnx.Module):
    def __init__(self, rngs):
        self.layer1 = HypLinearPoincare(
            manifold_module=poincare,
            in_dim=32,
            out_dim=16,
            rngs=rngs
        )
        self.layer2 = HypLinearPoincare(
            manifold_module=poincare,
            in_dim=16,
            out_dim=8,
            rngs=rngs
        )

    def __call__(self, x, c=1.0):
        x = self.layer1(x, c)
        x = self.layer2(x, c)
        return x

# Create model
model = SimpleHypNet(rngs=nnx.Rngs(0))

# Forward pass
x = jax.random.normal(nnx.Rngs(1).params(), (10, 32)) * 0.3
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

output = model(x_proj, c=1.0)
print(output.shape)  # (10, 8)
```

## Next Steps

- **[Batching & JIT Guide](user-guide/batching-jit.md)**: Learn efficient JAX patterns
- **[Tutorials](tutorials/basic-manifold-ops.ipynb)**: Hands-on Jupyter notebooks
- **[API Reference](api-reference/manifolds.md)**: Complete function documentation
- **[Training Workflows](user-guide/training-workflows.md)**: Full training examples

## Common Issues

### Import Errors

If you get import errors, ensure you've installed all dependencies:

```bash
uv sync --dev
```

### Float32 Precision

If you see `NaN` or `inf` values, try using float64:

```python
from jax import config
config.update("jax_enable_x64", True)
```

See [Numerical Stability](user-guide/numerical-stability.md) for details.

### JIT Compilation Issues

If JIT compilation fails, check that curvature `c` is not being traced:

```python
# Wrong: c is traced
@jax.jit
def forward(x, c):
    return poincare.dist(x, y, c, version_idx=0)

# Better: use partial
from functools import partial

@partial(jax.jit, static_argnums=(2,))
def forward(x, y, c):
    return poincare.dist(x, y, c, version_idx=0)
```
