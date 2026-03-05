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
- Class-based manifold API with automatic dtype casting (`Poincare`, `Hyperboloid`, `Euclidean`)
- Isometry mappings between Poincaré ball and hyperboloid models
- `Manifold` structural protocol for type-safe manifold dispatch
- **Causal attention masking** (`causal=True`) for all three hyperbolic attention variants:
    - `HyperbolicSoftmaxAttention`: lower-triangular `-inf` mask before softmax
    - `HyperbolicFullAttention`: lower-triangular `-inf` mask on Lorentzian similarity scores
    - `HyperbolicLinearAttention`: O(N) cumulative-sum recurrence via `jax.lax.scan` (Katharopoulos et al. 2020), keeping O(N) complexity in causal mode
- Tiny Shakespeare character-level benchmark (`benchmarks/bench_shakespeare_attention.py`) comparing all four model variants (Euclidean + 3 hyperbolic) with causal attention

### Changed
- **Breaking**: Manifold public functions renamed to private (`dist()` → `_dist()`); use class methods instead
- Replaced `with_precision()` wrapper with `Poincare(dtype=jnp.float64)` pattern

## [0.1.4] - 2026-02

### Added
- Pure JAX implementation of hyperbolic manifolds (Euclidean, Poincaré, Hyperboloid)
- 13+ neural network layers (linear, convolutional, regression)
- Hypformer components: HTC/HRC with curvature-change support
- 4 hyperbolic activation functions (ReLU, Leaky ReLU, Tanh, Swish)
- Riemannian optimizers (RSGD, RAdam) with automatic manifold detection
- Wrapped normal distributions for VAEs
- Comprehensive test suite (1,400+ tests)
- CI/CD pipeline with benchmarking
- vmap-native API design

### Changed
- Migrated from PyTorch to pure JAX/Flax NNX
- Unified package structure: `hyperbolix_jax` → `hyperbolix`

### References
- Based on research by Ganea et al. (2018), Bécigneul & Ganea (2019), Bdeir et al. (2023)

[Unreleased]: https://github.com/timoklein/hyperbolix/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/timoklein/hyperbolix/releases/tag/v0.1.0
