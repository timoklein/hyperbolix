# Changelog

All notable changes to Hyperbolix will be documented in this file.

## [Unreleased]

### Added
- MkDocs Material documentation system
- Complete API reference documentation
- Getting Started guide
- CI/CD workflow for documentation builds
- Positional encoding layers for hyperbolic Transformers:
    - `lorentz_residual`: Lorentzian midpoint-based residual connection
    - `hope`: Hyperbolic Rotary Positional Encoding (functional)
    - `HyperbolicRoPE`: NNX module wrapper for HOPE
    - `HypformerPositionalEncoding`: Learnable positional encoding with HTCLinear

## [0.1.0] - 2026-01

### Added
- Pure JAX implementation of hyperbolic manifolds (Euclidean, Poincaré, Hyperboloid)
- 13+ neural network layers (linear, convolutional, regression)
- 4 hyperbolic activation functions (ReLU, Leaky ReLU, Tanh, Swish)
- Riemannian optimizers (RSGD, RAdam) with automatic manifold detection
- Wrapped normal distributions for VAEs
- HoroPCA for dimensionality reduction
- Comprehensive test suite (1,400+ tests)
- CI/CD pipeline with benchmarking
- vmap-native functional API design

### Changed
- Migrated from PyTorch to pure JAX/Flax NNX
- Unified package structure: `hyperbolix_jax` → `hyperbolix`

### References
- Based on research by Ganea et al. (2018), Bécigneul & Ganea (2019), Bdeir et al. (2023)

[Unreleased]: https://github.com/hyperbolix/hyperbolix/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hyperbolix/hyperbolix/releases/tag/v0.1.0
