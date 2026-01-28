# Hyperbolix

## Hyperbolic Deep Learning in JAX

[![Tests](https://img.shields.io/badge/tests-1400%2B%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![JAX](https://img.shields.io/badge/JAX-compatible-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Pure JAX implementation of hyperbolic deep learning with manifold operations, neural network layers, and Riemannian optimizers. Built with Flax NNX and Optax.

## Features

> [!WARNING]
> **Project Status**: This is a 100% Vibe-coded project. While we have extensive test coverage, bugs and errors should be expected. Use with caution in production environments.

- 🌐 **3 Manifolds**: Euclidean, Poincaré Ball, Hyperboloid
- 🧠 **13+ Neural Network Layers**: Linear, convolutional (2D/3D), regression
- ⚡ **4 Hyperbolic Activations**: ReLU, Leaky ReLU, Tanh, Swish
- 📈 **Riemannian Optimizers**: RAdam and RSGD with automatic manifold detection
- 🚀 **Pure JAX/Flax NNX**: vmap-native API, JIT-compatible (10-100x speedup)
- ✅ **1,400+ tests passing** with comprehensive benchmark suite

## Quick Start

```python
import jax.numpy as jnp
from hyperbolix.manifolds import poincare
from hyperbolix.nn_layers import HypLinearPoincare
from flax import nnx

# Manifold operations
x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, -0.1])
distance = poincare.dist(x, y, c=1.0, version_idx=0)

# Neural network layer
layer = HypLinearPoincare(
    manifold_module=poincare,
    in_dim=128,
    out_dim=64,
    rngs=nnx.Rngs(0)
)
output = layer(x_batch, c=1.0)
```

## Installation

```bash
git clone https://github.com/hyperbolix/hyperbolix.git
cd hyperbolix
uv sync  # or: pip install -e .
```

**Requirements**: Python 3.12+, JAX 0.4.20+, Flax 0.8.0+, Optax 0.1.7+

## Documentation

📖 **[Full Documentation](https://hyperbolix.github.io/hyperbolix/)** (coming soon)

- **[Getting Started](docs/getting-started.md)** - Installation and first examples
- **[User Guides](docs/user-guide/)** - Manifolds, layers, optimizers, batching, numerical stability
- **[API Reference](docs/api-reference/)** - Complete API documentation
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Development setup and workflows

Build docs locally: `uv run mkdocs serve`

## Key Concepts

**Pure functional design**: No stateful classes, curvature passed at call time

```python
import hyperbolix.manifolds.poincare as poincare
dist = poincare.dist(x, y, c=1.0, version_idx=0)  # (dim,) → scalar
```

**vmap-native API**: Functions operate on single points, use `jax.vmap` for batching

```python
# Batch operations
distances = jax.vmap(poincare.dist, in_axes=(0, 0, None, None))(
    x_batch, y_batch, 1.0, 0
)
```

## Citation

```bibtex
@software{hyperbolix2026,
  title = {Hyperbolix: Hyperbolic Deep Learning in JAX},
  author = {Klein, Timo and Lang, Thomas},
  year = {2026},
  url = {https://github.com/hyperbolix/hyperbolix}
}
```

## References

Implements methods from:

- Ganea et al. (2018): Hyperbolic Neural Networks
- Bécigneul & Ganea (2019): Riemannian Adaptive Optimization
- Nagano et al. (2019): Wrapped Normal Distribution on Hyperbolic Space
- Shimizu et al. (2020): Hyperbolic Neural Networks++
- Bdeir et al. (2023): Fully Hyperbolic CNNs
- Bdeir et al. (2025): Robust Hyperbolic Learning

See individual module docstrings for detailed references.

## Contributing

Contributions welcome! See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for setup and guidelines.

For bugs or questions, [open an issue](https://github.com/hyperbolix/hyperbolix/issues).

## License

MIT License. See LICENSE for details.
