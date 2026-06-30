# Hyperbolix

**Hyperbolic Deep Learning in JAX**

Hyperbolix is a pure JAX implementation of hyperbolic deep learning, providing manifold operations, neural network layers, and Riemannian optimizers for hyperbolic geometry. Built with Flax NNX and Optax for modern JAX workflows.

## Features

- **5 Manifolds**: Euclidean, Poincaré Ball, Hyperboloid, Proper Velocity, and Product Manifold (mixed-curvature composition) — all with complete geometric operations
- **Learnable Curvature**: `LearnableCurvature` module bundles parameter + reparameterization (softplus or log/exp) + optional clamp; works with any `nnx.Optimizer`
- **Neural Network Layers**: 20+ hyperbolic layers including linear, convolutional, regression, attention, and PV layers
- **Activation Functions**: 4 hyperbolic activations (ReLU, Leaky ReLU, Tanh, Swish)
- **Riemannian Optimizers**: RAdam and RSGD with automatic manifold parameter detection
- **Wrapped Normal Distributions**: For probabilistic modeling on hyperbolic manifolds
- **Pure JAX/Flax NNX**: No PyTorch dependency, fully compatible with JAX ecosystem
- **vmap-native API**: Efficient batching through JAX's functional paradigm
- **JIT-compatible**: All operations support JIT compilation for performance
- **Comprehensive Test Suite**: 4,400+ tests (parametrized across seeds, dtypes, manifolds) with 100% pass rate

## Quick Example

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import Poincare

# Create manifold (float32 by default, float64 for higher precision)
poincare = Poincare()

# Create points on the Poincaré ball
x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, -0.1])
c = 1.0  # Curvature parameter

# Compute distance (single point operation)
distance = poincare.dist(x, y, c)
print(f"Distance: {distance}")

# Batch operations with vmap
x_batch = jax.random.normal(jax.random.PRNGKey(0), (100, 2)) * 0.3
y_batch = jax.random.normal(jax.random.PRNGKey(1), (100, 2)) * 0.3

# Project to manifold and compute pairwise distances
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x_batch, c)
y_proj = jax.vmap(poincare.proj, in_axes=(0, None))(y_batch, c)
distances = jax.vmap(poincare.dist, in_axes=(0, 0, None))(x_proj, y_proj, c)
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

Hyperbolix follows a **class-based manifold design** with functional transformations:

```python
# Manifold classes with automatic dtype casting
from hyperbolix.manifolds import Poincare
import jax.numpy as jnp

poincare = Poincare(dtype=jnp.float64)  # Optional float64 precision
distance = poincare.dist(x, y, c=1.0)

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

**Active development — current release: v0.6.0.** Core functionality is complete and stable; new layers and manifolds are added incrementally. See the [Changelog](changelog.md) for the full release history.

| Capability | Status | Shipped in |
|---|---|---|
| Class-based manifolds (Euclidean, Poincaré, Hyperboloid) | ✅ Stable | v0.3.0 |
| Riemannian optimizers (RSGD, RAdam) | ✅ Stable | v0.1.4 |
| Hyperbolic linear / convolutional layers | ✅ Stable | v0.1.4 |
| Hyperboloid attention + hyperbolic transformer blocks | ✅ Stable | v0.2.0 |
| Isometry mappings (Poincaré ↔ Hyperboloid) | ✅ Stable | v0.2.0 |
| FGG-LNN layers (Klis et al. 2026) | ✅ Stable | v0.3.0 |
| Poincaré BatchNorm2d, FHCNN layers, hyperbolic avg-pool | ✅ Stable | v0.5.x |
| Proper Velocity manifold + PV layers (Chen et al. 2026) | ✅ Stable | v0.5.3 |
| Learnable curvature (softplus-reparametrized `nnx.Param`) | ✅ Stable | v0.6.0 |
| Product manifolds (Gu et al. 2019, mixed-curvature composition) | 🚧 Unreleased | next |
| CI/CD pipeline with benchmarking | ✅ Stable | v0.1.4 |

## Key Concepts

### vmap-native API

Methods operate on **single points** by design. Use `jax.vmap` for batching:

```python
poincare = Poincare()

# Single point operation
result = poincare.expmap(x, v, c)

# Batched operation
batch_result = jax.vmap(poincare.expmap, in_axes=(0, 0, None))(x_batch, v_batch, c)
```

This design enables efficient JIT compilation and clear semantics.

### Curvature Parameter

The curvature `c` is passed at **call time**, not stored in the manifold:

```python
poincare = Poincare()

# Different curvatures for different calls
dist_c1 = poincare.dist(x, y, c=1.0)
dist_c2 = poincare.dist(x, y, c=2.0)
```

### Learnable Curvature

Curvature can be made trainable via the `LearnableCurvature` module.
Instantiate one per distinct curvature in your model and call it at runtime:

```python
from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Hyperboloid

class Model(nnx.Module):
    def __init__(self, rngs):
        self.manifold = Hyperboloid(c=1.0)
        self.curvature = LearnableCurvature(init_c=1.0)  # nnx.Module on the model
        self.fc = FGGLinear(33, 65, rngs=rngs)

    def __call__(self, x):
        c = self.curvature()                             # positive, clamped to [0.1, 10.0]
        return self.fc(x, c=c)
```

Pick `parameterization="log"` for compiled RL loops or when `c` spans
orders of magnitude; the default `"softplus"` is best for supervised
training. See the [Manifolds guide](user-guide/manifolds.md#choosing-softplus-vs-log)
for details.

### Mixed-Curvature Product Spaces

Compose heterogeneous factors (each with its own curvature) into a single
product manifold (Gu et al. 2019). Curvature is supplied at call time as a
per-factor sequence — pass `product.curvatures` for static factors, or
build the sequence from `LearnableCurvature` calls for trainable factors:

```python
from hyperbolix.manifolds import ProductManifold, Hyperboloid, Poincare, Euclidean

product = ProductManifold(
    (Hyperboloid(c=1.0), 5),   # hyperbolic factor
    (Poincare(c=0.1), 3),      # Poincaré factor
    (Euclidean(), 4),          # flat factor
)
c = product.curvatures               # (1.0, 0.1, 0.0)
d = product.dist(x, y, c)            # sqrt(sum d_i^2) over factors
```

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
- [Training Workflows](user-guide/training-workflows.md): Hands-on training examples
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
