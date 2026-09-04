# Hyperbolix

## Hyperbolic Deep Learning in JAX

[![Tests](https://github.com/timoklein/hyperbolix/actions/workflows/ci.yaml/badge.svg)](https://github.com/timoklein/hyperbolix/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![JAX](https://img.shields.io/badge/JAX-compatible-orange)](https://jax.dev)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Pure JAX implementation of hyperbolic deep learning with manifold operations, neural network layers, and Riemannian optimizers. Built with Flax NNX and Optax.

## Features

- 🌐 **6 Manifolds**: Euclidean, Poincaré Ball, Hyperboloid, Proper Velocity, κ-Stereographic (signed curvature — hyperbolic, flat, and spherical in one manifold), and Product Manifold (mixed-curvature composition)
- 🎛️ **Learnable Curvature**: `LearnableCurvature` module bundles parameter + reparameterization (softplus, log/exp, or signed identity) + optional clamp. Works with any `nnx.Optimizer` — no Riemannian optimizer needed
- 🧠 **40+ Neural Network Layers**: Linear, convolutional, regression, attention, normalization, positional encoding, PV
- ⚡ **5 Hyperbolic Activations**: ReLU, Leaky ReLU, Tanh, Swish, GELU
- 📈 **Riemannian Optimizers**: RAdam and RSGD with automatic manifold detection
- 🚀 **Pure JAX/Flax NNX**: vmap-native API, JIT-compatible
- ✅ **4,000+ tests passing** (959 test functions, parametrized across dtypes, dimensions, manifolds) checked against independently transcribed NumPy/SciPy oracles

## Quick Start

```python
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypLinearPoincare

# Plain Python manifold class (optionally float64; pass `c=` for fixed curvature)
poincare = Poincare()  # add dtype=jnp.float64 as needed

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

# H^5 × P^3 × E^4 — points live in R^12, each factor keeps its own curvature
product = ProductManifold(
    (Hyperboloid(c=1.0), 5),
    (Poincare(c=0.1), 3),
    (Euclidean(), 4),
)
cs = product.curvatures        # (1.0, 0.1, 0.0) — pass per-factor at call time
d = product.dist(x, y, cs)     # sqrt(sum d_i^2) over factors
```

To make any factor's curvature trainable, store one `LearnableCurvature`
instance per factor on your model and call it to obtain `c` for per-factor
operations (see "Learnable curvature" below).

## Installation

```bash
git clone https://github.com/timoklein/hyperbolix.git
cd hyperbolix
uv sync  # or: pip install -e .
```

**Requirements**: Python 3.12+, JAX 0.9+, Flax 0.12+, Optax 0.2.6+

## Documentation

📖 **[Full Documentation](https://timoklein.github.io/hyperbolix/)**

- **[Getting Started](docs/getting-started.md)** - Installation and first examples
- **[User Guides](docs/user-guide/)** - Manifolds, layers, optimizers, batching, numerical stability
- **[API Reference](docs/api-reference/)** - Complete API documentation
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Development setup and workflows

Build docs locally: `uv run mkdocs serve`

## Key Concepts

**Plain-class manifolds, curvature passed at call time:** Each manifold is a plain Python class (not an `nnx.Module`) with automatic dtype casting; the curvature `c` is supplied per call so it can be static, dynamic, or a traced `jax.Array` driven by a learnable parameter on your model.

```python
from hyperbolix.manifolds import Poincare
poincare = Poincare()  # add dtype=jnp.float64 as needed
dist = poincare.dist(x, y, c=1.0)  # (dim,) → scalar
```

**vmap-native API:** Methods operate on single points; use `jax.vmap` for batching.

```python
distances = jax.vmap(poincare.dist, in_axes=(0, 0, None))(
    x_batch, y_batch, 1.0
)
```

**Learnable curvature:** Use the `LearnableCurvature` module — assign one instance per distinct curvature in your model and call it to obtain a positive (optionally clamped) value. The manifold itself stays a fixed plain Python class, which keeps it out of the NNX state pytree (safe to share the same instance across layers and inside `nnx.scan` / `nnx.fori_loop`). The default clamp `[0.1, 10.0]` matches published reference ranges; pass `c_min=None, c_max=None` to disable. Use `parameterization="log"` (MERU-style) when `c` may span orders of magnitude or for long compiled training loops; the default `"softplus"` matches van Spengler 2023.

```python
from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Hyperboloid

class Model(nnx.Module):
    def __init__(self, rngs):
        self.manifold = Hyperboloid(c=1.0)               # static, shared
        self.curvature = LearnableCurvature(init_c=1.0)  # one per distinct c
        self.fc = HypLinearHyperboloidPLFC(self.manifold, 33, 65, rngs=rngs)

    def __call__(self, x):
        c = self.curvature()                              # positive, clamped
        return self.fc(x, c=c)

# Updated by any standard Euclidean optimizer — no Riemannian optimizer needed.
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
```

## Citation

```bibtex
@software{hyperbolix2026,
  title = {Hyperbolix: Hyperbolic Deep Learning in JAX},
  author = {Klein, Timo and Lang, Thomas},
  year = {2026},
  url = {https://github.com/timoklein/hyperbolix}
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

For bugs or questions, [open an issue](https://github.com/timoklein/hyperbolix/issues).

## License

MIT License. See LICENSE for details.
