# Changelog

All notable changes to Hyperbolix will be documented in this file.

## [Unreleased]

### Changed
- **Manifolds are now plain Python classes** (not `nnx.Module`). This structurally prevents shared-manifold bugs in `nnx.scan`/`nnx.fori_loop` — manifolds become static graphdef attributes with no state pytree entries
- **`learnable=True` removed** from manifold constructors. Use the `LearnableCurvature` module instead — assign one instance per distinct curvature on your model. See the [Manifolds User Guide](user-guide/manifolds.md#static-vs-learnable)
- **`ProductManifold` curvature API redesigned** — every geometry method (`dist`, `expmap`, `logmap`, `proj`, `origin`, `tangent_inner`, etc.) now takes a positional `c` argument that must be a sequence of length `n_factors` (one curvature per factor) instead of the silently-ignored scalar `c`. There is no default and no scalar broadcast: pass `product.curvatures` for static curvatures, or build the sequence from `LearnableCurvature` calls for trainable ones. The protocol-level `Curvature` type was widened to `ScalarCurvature | Sequence[ScalarCurvature]` so `ProductManifold` satisfies the `Manifold` protocol — `isinstance(product, Manifold)` is `True` and generic code typed against `Manifold` accepts product instances. The product still has no `c` attribute (use `product.curvatures`). **Breaking change** — call sites that passed `c=0.0` (or any scalar) must now pass a per-factor sequence `(c_0, c_1, …)`
- **`ProductManifold`** is now a plain class; `from_signature` accepts 3- and 4-tuple specs only (5-tuple `learnable` override removed)
- **`learnable_curvature()` / `get_curvature()` functional helpers replaced** by the `LearnableCurvature(nnx.Module)` class. The class bundles the raw parameter, reparameterization scheme, and clamp bounds in one object, making accidental init/recovery mismatches structurally impossible. Call sites change from `c = get_curvature(self.c_raw)` to `c = self.curvature()`. **Breaking change** — no deprecation shim

### Added
- **`LearnableCurvature(nnx.Module)`** — canonical module for trainable curvature in `hyperbolix.utils.curvature`, also exported at top level (`from hyperbolix import LearnableCurvature`). Supports two reparameterizations:
    - `parameterization="softplus"` (default): `c = softplus(raw)`, bounded gradient via sigmoid (van Spengler 2023 convention)
    - `parameterization="log"`: `c = exp(raw)`, scale-invariant gradient `dc/draw = c` (MERU convention, preferred for compiled RL loops)

    Default clamp `[0.1, 10.0]` is applied to the recovered `c` (not the raw parameter) as a hard stability guard for long compiled training loops; pass `c_min=None, c_max=None` to disable. Updated by any standard Euclidean `nnx.Optimizer` — no Riemannian optimizer needed
- **Product manifold** (`hyperbolix.manifolds.ProductManifold`) — heterogeneous-curvature composition $M_1 \times M_2 \times \dots \times M_n$ where each factor may be any base manifold with its own curvature (Gu et al. 2019). Points are flat concatenated arrays of shape `(total_dim,)`; geometry methods take a per-factor curvature sequence as positional `c` at call time. Provides Pythagorean L2 geodesic distance plus auxiliary `dist_l1` / `dist_min` / `component_dist` reductions, full per-factor decomposition of `expmap`/`logmap`/`ptransp`/`proj`/`egrad2rgrad`/`tangent_inner`, an `origin(c)` helper, and a `from_signature` factory accepting 3- and 4-tuple specs
- **Proper Velocity (PV) manifold** (`hyperbolix.manifolds.ProperVelocity`) — unconstrained $\mathbb{R}^n$ model of hyperbolic geometry from Chen et al. (2026), with complete geometric operations: `addition`, `scalar_mul`, `dist`, `expmap`/`logmap` (at origin and arbitrary base points), `ptransp`/`ptransp_0`, `egrad2rgrad`, and Riemannian inner product
- **Proper Velocity neural-network layers**:
    - `HypLinearPV`: PV fully-connected layer (Thm 5.3 / Eq. 22)
    - `HypConv2DPV`: PV 2D convolution with raw Euclidean patch concatenation (Sec 5.3) — no beta-scaling, dimension-preserving
    - `HypRegressionPV`: PV multinomial-logistic-regression head (Thm 5.2 / Eq. 19)
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
