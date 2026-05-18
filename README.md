# Hyperbolix

## Hyperbolic Deep Learning in JAX

[![Tests](https://img.shields.io/badge/tests-4400%2B%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![JAX](https://img.shields.io/badge/JAX-compatible-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Pure JAX implementation of hyperbolic deep learning with manifold operations, neural network layers, and Riemannian optimizers. Built with Flax NNX and Optax.

## Features

- 🌐 **5 Manifolds**: Euclidean, Poincaré Ball, Hyperboloid, Proper Velocity, and Product Manifold (mixed-curvature composition)
- 🎛️ **Learnable Curvature**: Optional softplus-reparametrized `nnx.Param` curvature per manifold; with per-factor control in product spaces
- 🧠 **20+ Neural Network Layers**: Linear, convolutional, regression, attention, positional encoding, PV
- ⚡ **5 Hyperbolic Activations**: ReLU, Leaky ReLU, Tanh, Swish, GELU
- 📈 **Riemannian Optimizers**: RAdam and RSGD with automatic manifold detection
- 🚀 **Pure JAX/Flax NNX**: vmap-native API, JIT-compatible (10-100x speedup)
- ✅ **4,400+ tests passing** (618 test functions, parametrized across seeds, dtypes, manifolds) with comprehensive benchmark suite

## Quick Start

```python
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypLinearPoincare

# Class-based manifold instance (optionally float64 or learnable curvature)
poincare = Poincare()  # add dtype=jnp.float64 / learnable=True as needed

# Manifold operations (single-point; use jax.vmap for batches)
x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, -0.1])
distance = poincare.dist(x, y, c=1.0)

# Neural network layer
layer = HypLinearPoincare(
    manifold_module=poincare,
    in_dim=128,
    out_dim=64,
    rngs=nnx.Rngs(0),
)
output = layer(x_batch, c=1.0)
```

### Mixed-Curvature Product Spaces

```python
from hyperbolix.manifolds import ProductManifold, Hyperboloid, Poincare, Euclidean

# H^5 (learnable c) × P^3 (fixed c=0.1) × E^4 — points live in R^12
product = ProductManifold(
    (Hyperboloid(c=1.0, learnable=True), 5),
    (Poincare(c=0.1), 3),
    (Euclidean(), 4),
)
d = product.dist(x, y)  # sqrt(sum d_i^2) over factors
```

## Installation

```bash
git clone https://github.com/hyperbolix/hyperbolix.git
cd hyperbolix
uv sync  # or: pip install -e .
```

**Requirements**: Python 3.12+, JAX 0.4.20+, Flax 0.8.0+, Optax 0.1.7+

## Documentation

📖 **[Full Documentation](https://hyperbolix.github.io/hyperbolix/)**

- **[Getting Started](docs/getting-started.md)** - Installation and first examples
- **[User Guides](docs/user-guide/)** - Manifolds, layers, optimizers, batching, numerical stability
- **[API Reference](docs/api-reference/)** - Complete API documentation
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Development setup and workflows

Build docs locally: `uv run mkdocs serve`

## Key Concepts

**Class-based manifolds, curvature passed at call time:** Each manifold is a `Manifold`-protocol-conformant class with automatic dtype casting; the curvature `c` is supplied per call so it can be static, dynamic, or even a learnable `nnx.Param`.

```python
from hyperbolix.manifolds import Poincare
poincare = Poincare()  # add dtype=jnp.float64 / learnable=True as needed
dist = poincare.dist(x, y, c=1.0)  # (dim,) → scalar
```

**vmap-native API:** Methods operate on single points; use `jax.vmap` for batching.

```python
distances = jax.vmap(poincare.dist, in_axes=(0, 0, None))(
    x_batch, y_batch, 1.0
)
```

**Learnable curvature:** Pass `learnable=True` to any base manifold to expose `c` as a trainable `nnx.Param` via softplus reparameterization. In a `ProductManifold`, use the 5-tuple form of `from_signature` to learn some factors' curvatures while freezing others.

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
- Gu et al. (2019): Learning Mixed-Curvature Representations in Product Spaces
- Nagano et al. (2019): Wrapped Normal Distribution on Hyperbolic Space
- Shimizu et al. (2020): Hyperbolic Neural Networks++
- Bdeir et al. (2023): Fully Hyperbolic CNNs
- Bdeir et al. (2025): Robust Hyperbolic Learning
- Klis et al. (2026): Fast and Geometrically Grounded Lorentz Neural Networks
- Chen et al. (2026): Proper Velocity Neural Networks

See individual module docstrings for detailed references.

## Contributing

Contributions welcome! See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for setup and guidelines.

For bugs or questions, [open an issue](https://github.com/hyperbolix/hyperbolix/issues).

## License

MIT License. See LICENSE for details.
