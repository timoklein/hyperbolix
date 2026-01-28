# Hyperbolix

**Hyperbolic Deep Learning in JAX**

Hyperbolix is a pure JAX implementation of hyperbolic deep learning, providing manifold operations, neural network layers, and Riemannian optimizers for hyperbolic geometry. Built with Flax NNX and Optax for modern JAX workflows.

## Features

- **3 Manifolds**: Euclidean, Poincaré Ball, and Hyperboloid with complete geometric operations
- **Neural Network Layers**: 13+ hyperbolic layers including linear, convolutional, and regression layers
- **Activation Functions**: 4 hyperbolic activations (ReLU, Leaky ReLU, Tanh, Swish)
- **Riemannian Optimizers**: RAdam and RSGD with automatic manifold parameter detection
- **Wrapped Normal Distributions**: For probabilistic modeling on hyperbolic manifolds
- **Pure JAX/Flax NNX**: No PyTorch dependency, fully compatible with JAX ecosystem
- **vmap-native API**: Efficient batching through JAX's functional paradigm
- **JIT-compatible**: All operations support JIT compilation for performance
- **Comprehensive Test Suite**: 1,400+ tests with 100% pass rate

## Quick Example

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import poincare

# Create points on the Poincaré ball
x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, -0.1])
c = 1.0  # Curvature parameter

# Compute distance (single point operation)
distance = poincare.dist(x, y, c, version=0)
print(f"Distance: {distance}")

# Batch operations with vmap
x_batch = jax.random.normal(jax.random.PRNGKey(0), (100, 2)) * 0.3
y_batch = jax.random.normal(jax.random.PRNGKey(1), (100, 2)) * 0.3

# Project to manifold and compute pairwise distances
x_proj = jax.vmap(poincare.proj, in_axes=(0, None, None))(x_batch, c, None)
y_proj = jax.vmap(poincare.proj, in_axes=(0, None, None))(y_batch, c, None)
distances = jax.vmap(poincare.dist, in_axes=(0, 0, None, None))(x_proj, y_proj, c, 0)
```

## Installation

Install from source:

```bash
git clone https://github.com/hyperbolix/hyperbolix.git
cd hyperbolix
uv sync  # or pip install -e .
```

Requirements: Python 3.12+, JAX, Flax NNX, Optax

## Architecture

Hyperbolix follows a **pure functional design**:

```python
# Pure functions, no classes
import hyperbolix.manifolds.poincare as poincare
distance = poincare.dist(x, y, c, version)

# Neural network layers as Flax NNX modules
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincare

model = HypLinearPoincare(
    manifold_module=poincare,
    in_dim=32,
    out_dim=16,
    rngs=nnx.Rngs(0)
)
output = model(input_data, c=1.0)
```

## Project Status

**All core functionality is complete and production-ready!**

- ✅ Phase 1: Manifolds (978 passing tests)
- ✅ Phase 2: Riemannian Optimizers (20 passing tests)
- ✅ Phase 3a: Neural Network Layers (44 passing tests)
- ✅ Phase 3b: Regression Layers (22 passing tests)
- ✅ Hyperboloid Convolutions (68 passing tests)
- ✅ Lorentz Convolutions (66 passing tests)
- ✅ Hyperboloid Activations (86 passing tests)
- ✅ CI/CD Pipeline with benchmarking
- ✅ Clean, unified codebase structure

## Key Concepts

### vmap-native API

Functions operate on **single points** by design. Use `jax.vmap` for batching:

```python
# Single point operation
result = poincare.expmap(x, v, c)

# Batched operation
batch_result = jax.vmap(poincare.expmap, in_axes=(0, 0, None))(x_batch, v_batch, c)
```

This design enables efficient JIT compilation and clear semantics.

### Curvature Parameter

The curvature `c` is passed at **call time**, not stored in objects:

```python
# Different curvatures for different calls
dist_c1 = poincare.dist(x, y, c=1.0, version=0)
dist_c2 = poincare.dist(x, y, c=2.0, version=0)
```

This allows for learnable curvature in neural networks.

### Manifold Operations

Each manifold provides:

- **proj**: Project points onto the manifold
- **dist**: Compute distances (multiple versions for numerical stability)
- **expmap/logmap**: Exponential and logarithmic maps
- **ptransp**: Parallel transport
- **egrad2rgrad**: Convert Euclidean to Riemannian gradients

## Next Steps

- [Getting Started](getting-started.md): Installation and first examples
- [User Guide](user-guide/manifolds.md): Core concepts and patterns
- [Tutorials](tutorials/basic-manifold-ops.ipynb): Hands-on learning
- [API Reference](api-reference/manifolds.md): Complete API documentation

## Citation

If you use Hyperbolix in your research, please cite:

```bibtex
@software{hyperbolix2026,
  title = {Hyperbolix: Hyperbolic Deep Learning in JAX},
  author = {Klein, Timo and Lang, Thomas and Shkabrii, Andrii},
  year = {2026},
  url = {https://github.com/hyperbolix/hyperbolix}
}
```

## License

MIT License. See LICENSE for details.

## Acknowledgments

This library implements methods from several research papers:

- Ganea et al. (2018): "Hyperbolic Neural Networks"
- Bécigneul & Ganea (2019): "Riemannian Adaptive Optimization Methods"
- Bdeir et al. (2023): "Fully Hyperbolic Convolutional Neural Networks"
- And many others (see references in individual modules)
