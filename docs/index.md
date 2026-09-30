<p align="center" markdown="1">
  ![Hyperbolix](assets/hyperbolix-light.png#only-light){ width="360" }
  ![Hyperbolix](assets/hyperbolix-dark.png#only-dark){ width="360" }
</p>

<p align="center"><strong>Hyperbolic Deep Learning in JAX</strong></p>

Hyperbolix is a pure JAX implementation of hyperbolic deep learning, providing manifold operations, neural network layers, and Riemannian optimizers for hyperbolic geometry. Built with Flax NNX and Optax for modern JAX workflows.

**Status:** Stable, v1.3.0; see the [Changelog](changelog.md).

## Features

- **8 Manifolds**: Euclidean, Poincaré Ball, Hyperboloid, Proper Velocity, κ-Stereographic (signed curvature — hyperbolic, flat, and spherical in one manifold), Klein (Beltrami–Klein ball), Half-Space (Poincaré upper half-space), and Product Manifold (mixed-curvature composition)
- **Learnable Curvature**: `LearnableCurvature` module bundles parameter + reparameterization (softplus, log/exp, or signed identity) + optional clamp; works with any `nnx.Optimizer`
- **Neural Network Layers**: 40+ hyperbolic layers including linear, convolutional, regression, attention, normalization, and PV layers
- **Activation Functions**: 5 hyperbolic activations (ReLU, Leaky ReLU, Tanh, Swish, GELU)
- **Riemannian Optimizers**: RAdam and RSGD with automatic manifold parameter detection
- **Wrapped Normal Distributions**: For probabilistic modeling on hyperbolic manifolds
- **Dimensionality Reduction**: HoroPCA (Chami et al. 2021), CO-SNE (Guo et al. 2022, hyperbolic t-SNE), and the Fréchet-mean centering primitive
- **Pure JAX/Flax NNX**: No PyTorch dependency; operations work on single points, batch with `jax.vmap`, and support JIT compilation
- **Test Suite**: 7,000+ tests (parametrized across dtypes, dimensions, manifolds) checked against independently transcribed NumPy/SciPy oracles

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
git clone https://github.com/timoklein/hyperbolix.git
cd hyperbolix
uv sync  # or pip install -e .
```

Requirements: Python 3.12+, JAX 0.9+, Flax 0.12+, Optax 0.2.6+

## Next Steps

- [Getting Started](getting-started.md): Installation and first examples
- [User Guide](user-guide/manifolds.md): Core concepts and patterns, including learnable curvature and product manifolds
- [Training Workflows](user-guide/training-workflows.md): Hands-on training examples
- [API Reference](api-reference/manifolds.md): Complete API documentation

## Citation

If you use Hyperbolix in your research, please cite:

```bibtex
@software{hyperbolix2026,
  title = {Hyperbolix: Hyperbolic Deep Learning in JAX},
  author = {Klein, Timo and Lang, Thomas},
  year = {2026},
  url = {https://github.com/timoklein/hyperbolix}
}
```

## License

MIT License. See LICENSE for details.

## Acknowledgments

This library implements methods from several research papers, among them Ganea et al. (2018), "Hyperbolic Neural Networks"; Bécigneul & Ganea (2019), "Riemannian Adaptive Optimization Methods"; and Bdeir et al. (2023), "Fully Hyperbolic Convolutional Neural Networks". See the references in individual modules for the others.
