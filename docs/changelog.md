# Changelog

All notable changes to Hyperbolix will be documented in this file.

## [Unreleased]

### Changed
- **Manifolds are now plain Python classes** (not `nnx.Module`). This structurally prevents shared-manifold bugs in `nnx.scan`/`nnx.fori_loop` — manifolds become static graphdef attributes with no state pytree entries
- **`learnable=True` removed** from manifold constructors. Use the `LearnableCurvature` module instead — assign one instance per distinct curvature on your model. See the [Manifolds User Guide](user-guide/manifolds.md#static-vs-learnable)
- **`ProductManifold` curvature API redesigned** — every geometry method (`dist`, `expmap`, `logmap`, `proj`, `origin`, `tangent_inner`, etc.) now takes a positional `c` argument that must be a sequence of length `n_factors` (one curvature per factor) instead of the silently-ignored scalar `c`. There is no default and no scalar broadcast: pass `product.curvatures` for static curvatures, or build the sequence from `LearnableCurvature` calls for trainable ones. The protocol-level `Curvature` type was widened to `ScalarCurvature | Sequence[ScalarCurvature]` so `ProductManifold` satisfies the `Manifold` protocol — `isinstance(product, Manifold)` is `True` and generic code typed against `Manifold` accepts product instances. The product still has no `c` attribute (use `product.curvatures`). **Breaking change** — call sites that passed `c=0.0` (or any scalar) must now pass a per-factor sequence `(c_0, c_1, …)`
- **`ProductManifold`** is now a plain class; `from_signature` accepts 3- and 4-tuple specs only (5-tuple `learnable` override removed)
- **`learnable_curvature()` / `get_curvature()` functional helpers replaced** by the `LearnableCurvature(nnx.Module)` class. The class bundles the raw parameter, reparameterization scheme, and clamp bounds in one object, making accidental init/recovery mismatches structurally impossible. Call sites change from `c = get_curvature(self.c_raw)` to `c = self.curvature()`. **Breaking change** — no deprecation shim
- **`Hyperboloid.addition` now implements the Lorentz gyrovector addition** `x ⊕ y = Exp_x(PT_{0→x}(Log_0(y)))` (Chen et al. 2025b; Shi et al. 2026, Eq. 1), and `ProductManifold.addition` delegates it per factor. It forms a gyrocommutative gyrogroup (identity = origin, inverse `⊖x = (-1) ⊙ x = [x₀, -x_s]`) and coincides with Poincaré Möbius addition under the stereographic isometry. This replaces the earlier coordinate-wise formula, which was incorrect at *every* curvature (e.g. `origin ⊕ y ≠ y`) and failed silently via a trailing projection. `Hyperboloid.scalar_mul` (geodesic scaling, = Eq. 2) is unchanged. **Breaking change** — `addition` no longer raises `NotImplementedError`

### Added
- **`LearnableCurvature(nnx.Module)`** — canonical module for trainable curvature in `hyperbolix.utils.curvature`, also exported at top level (`from hyperbolix import LearnableCurvature`). Supports two reparameterizations:
    - `parameterization="softplus"` (default): `c = softplus(raw)`, bounded gradient via sigmoid (van Spengler 2023 convention)
    - `parameterization="log"`: `c = exp(raw)`, scale-invariant gradient `dc/draw = c` (MERU convention, preferred for compiled RL loops)

    Default clamp `[0.1, 10.0]` is applied to the recovered `c` (not the raw parameter) as a hard stability guard for long compiled training loops; pass `c_min=None, c_max=None` to disable. Updated by any standard Euclidean `nnx.Optimizer` — no Riemannian optimizer needed
- **Product manifold** (`hyperbolix.manifolds.ProductManifold`) — heterogeneous-curvature composition $M_1 \times M_2 \times \dots \times M_n$ where each factor may be any base manifold with its own curvature (Gu et al. 2019). Points are flat concatenated arrays of shape `(total_dim,)`; geometry methods take a per-factor curvature sequence as positional `c` at call time. Provides Pythagorean L2 geodesic distance plus auxiliary `dist_l1` / `dist_min` / `component_dist` reductions, full per-factor decomposition of `expmap`/`logmap`/`ptransp`/`proj`/`egrad2rgrad`/`tangent_inner`, an `origin(c)` helper, and a `from_signature` factory accepting 3- and 4-tuple specs
- **`Hyperboloid.log_radius_concat`** — log-radius-preserving concatenation of N Lorentz points (Shi et al. 2026, Sec. 4.3), the hyperboloid analog of Poincaré β-concatenation. Rescales each block's spatial part by `exp(½·(ψ(N·d/2) − ψ(d/2)))` (digamma `ψ`) so the expected log spatial radius stays invariant to the post-concat dimension, then recomputes the time coordinate to keep the result on the manifold. Reduces to `hcat` when `N = 1`; convolutional layers keep using the unscaled `hcat` by default
- **Proper Velocity (PV) manifold** (`hyperbolix.manifolds.ProperVelocity`) — unconstrained $\mathbb{R}^n$ model of hyperbolic geometry from Chen et al. (2026), with complete geometric operations: `addition`, `scalar_mul`, `dist`, `expmap`/`logmap` (at origin and arbitrary base points), `ptransp`/`ptransp_0`, `egrad2rgrad`, and Riemannian inner product
- **Proper Velocity neural-network layers**:
    - `HypLinearPV`: PV fully-connected layer (Thm 5.3 / Eq. 22)
    - `HypConv2DPV`: PV 2D convolution with raw Euclidean patch concatenation (Sec 5.3) — no beta-scaling, dimension-preserving
    - `HypRegressionPV`: PV multinomial-logistic-regression head (Thm 5.2 / Eq. 19)
- **Poincaré vector quantization layers** (`hyperbolix.nn_layers`):
    - `HypVQEmbeddingPoincare`: HVQ-VAE quantizer (Chen et al. 2025) — explicit on-ball codebook held as a non-parameter `nnx.Variable` buffer, geodesic nearest-neighbour selection, copy-gradient straight-through estimator, and a hyperbolic-EMA codebook update (GGBall, Bu et al. 2026, Eqs. 41-43) with optional dead-code revival via the `ema_update` method (called after `optimizer.update`; the optimizer never touches the codebook, only the commitment loss trains the encoder)
    - `HypVQMLRPoincare`: HyperVQ quantizer (Goswami et al. 2025) — implicit codebook = the rows of an internal `HypRegressionPoincarePP`, Gumbel-Softmax straight-through selection on the categorical weights (so plain `optax.adam` trains it), with a `deterministic` flag (toggled by `model.eval()` / `model.train()`) for a deterministic argmax MAP estimate at inference
    - `PoincareVQOutput`: shared pytree-friendly `NamedTuple` return type `(quantized, indices, loss, perplexity, z)`; `quantized` is cast to float32 at the manifold→decoder boundary
    - `poincare_weighted_midpoint`: reusable Poincaré weighted gyromidpoint (GGBall Eq. 41), the Poincaré analog of `lorentz_midpoint` and a weighted generalization of `poincare_midpoint`
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
- Isometry mappings now span all three models — Poincaré ↔ Hyperboloid, Poincaré ↔ Proper Velocity, and Hyperboloid ↔ Proper Velocity. New functions: `pv_to_poincare`, `poincare_to_pv`, `pv_to_hyperboloid`, `hyperboloid_to_pv` (in `hyperbolix.manifolds.isometry_mappings`). All six maps are exact, curvature-correct Riemannian isometries (PVNN Eq. 4; Chen et al. 2026). The direct `pv_to_hyperboloid` (PV coords are the hyperboloid's spatial part; time reconstructed from `⟨z,z⟩_L = -1/c`) avoids the near-boundary blow-up of composing through the ball
- `Manifold` structural protocol for type-safe manifold dispatch
- **Causal attention masking** (`causal=True`) for all three hyperbolic attention variants:
    - `HyperbolicSoftmaxAttention`: lower-triangular `-inf` mask before softmax
    - `HyperbolicFullAttention`: lower-triangular `-inf` mask on Lorentzian similarity scores
    - `HyperbolicLinearAttention`: O(N) cumulative-sum recurrence via `jax.lax.scan` (Katharopoulos et al. 2020), keeping O(N) complexity in causal mode
- Tiny Shakespeare character-level benchmark (`benchmarks/bench_shakespeare_attention.py`) comparing all four model variants (Euclidean + 3 hyperbolic) with causal attention

### Changed
- **Breaking**: Manifold public functions renamed to private (`dist()` → `_dist()`); use class methods instead
- Replaced `with_precision()` wrapper with `Poincare(dtype=jnp.float64)` pattern

### Fixed
- **Poincaré ↔ Hyperboloid isometry maps were only correct at `c=1`.** `poincare_to_hyperboloid` / `hyperboloid_to_poincare` used unit-ball stereographic formulas missing their `√c` factors, so round-trips and distance preservation silently failed at any other curvature (the recommended default is `c=0.1`). Both are now curvature-correct at all `c`, verified by round-trip, isometry, commutative-diagram, and extreme-curvature tests across dtypes
- **Test fixture `uniform_points` generated hyperboloid points at the wrong geodesic radius for `c ≠ 1`** (an extra `/√c` on the spatial coordinate, masked by a trailing projection). Corrected so generated points are the exact Poincaré images at any curvature

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
