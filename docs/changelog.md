# Changelog

All notable changes to Hyperbolix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **HoroPCA** (`hyperbolix.decomposition.HoroPCA`) — hyperbolic dimensionality reduction via horospherical projections (Chami et al. 2021). A thin sklearn-style class (`fit` / `transform` / `fit_transform`) over a pure, independently usable and JIT-friendly functional core (`fit_horopca`, `transform_horopca`, `horo_projection`, `horopca_loss`). All computation runs on the hyperboloid (Poincaré input is mapped in via the exact isometry); K ideal points are jointly optimized with Adam + gradient clipping to maximize the pairwise-distance variance of the horospherically projected data, which preserves every Busemann coordinate. Poincaré input `(N, D)` → `(N, K)` ball coordinates; Hyperboloid input `(N, A)` → `(N, K+1)`. `K = 1` is supported via a closed form (the reference's Sherman–Morrison inverse is singular there). Exposes `components_`, `mean_`, `boost_`, `losses_`, and pairwise-variance-based `explained_variance_` / `total_variance_` / `explained_variance_ratio_`. Double precision recommended for fitting
- **`hyperbolix.decomposition.frechet_mean`** — manifold-generic Karcher (Fréchet) mean via fixed-point iteration in a `lax.while_loop` (Lorentz-centroid init on the hyperboloid, projected-mean init on the Poincaré ball). The data-centering primitive HoroPCA builds on; configurable `step_size` / `tol` / `max_iters`
- **`Hyperboloid.lorentz_boost(mu, c)`** — the exact Lorentz boost matrix `B` with `B @ mu = origin` (symmetric, proper, orthochronous), used to Fréchet-mean-center data before fitting. Closes a docs–code gap (the method was already advertised in the manifolds API reference)

### Changed
- **`HTCLinear` default init is now fan-in-aware and norm-preserving.** `init_bound` accepts `None` (the new default), which resolves to `sqrt(3 / in_features)` — per-layer input-Jacobian gain ≈ 1, since `htc` applies no nonlinearity to the matmul output. The previous fixed `U(-0.02, 0.02)` was width-dependent **contractive** (gain `0.02/sqrt(3)·sqrt(in_features)` ≈ 0.09 at fan-in 65): stacks of ≥ 2 layers collapsed input variation below the float32 noise floor, becoming a constant map with ≈0 gradients — training silently frozen from step 0 (observed in goal-conditioned RL: depth-2 HTC head, c=0.5, frozen through 3.1M steps; escape time monotone in the bound). The `0.02` was an in-house convention inherited from the FHNN/FHCNN family, **not** the Hypformer reference — the official Hypformer code initializes with Xavier-uniform gain `sqrt(2)` (`bound = 2·sqrt(3/(in+out))`, ~15× larger than 0.02 at fan-in 65), intended for its ReLU + LayerNorm regime; recover it by passing that bound explicitly. The attention layers (`HyperbolicLinearAttention`, `HyperbolicSoftmaxAttention`, `HyperbolicFullAttention`) and `HypformerPositionalEncoding` now default `init_bound=None` and inherit the fan-in-aware init for their HTC projections (matching the reference, whose Wq/Wk/Wv and positional encoding use the same Xavier init). FHNN/FHCNN/PLFC layers are unchanged — their `0.02` matches their references (verified against the official Chen 2021, Bdeir 2023, and Shi 2026 code). **Behavior change** — existing HTC models pick up the new init unless they pass `init_bound=0.02` explicitly (bit-for-bit reproduction of the old init)

## [0.11.0] - 2026-07-14

### Added
- **κ-Stereographic manifold** (`hyperbolix.manifolds.Stereographic`) — a single constant-curvature manifold spanning hyperbolic, Euclidean, and spherical geometry via a **signed** curvature `c` (Bachmann et al. 2020): `c > 0` is hyperbolic and reproduces `Poincare(c)` exactly (both now build on a shared internal `_gyrovector_core` module, so Möbius addition, gyration, conformal factor, and projection are bit-identical), `c = 0` is the flat limit (carrying the conformal factor-2 metric — see the API reference), and `c < 0` is the stereographic projection of the sphere. Note the **sign flip vs the paper/geoopt** (`κ = -c`), chosen so `c` matches every other hyperbolix manifold. Implements the full `Manifold` protocol plus `conformal_factor`, `gyration`, `geodesic`, `geodesic_unit`, and `antipode`, using curvature-generalized trig functions with a dtype-aware Taylor branch near `c = 0` — gated on both `|κ|` and the series argument `|κ|·‖x‖²`, so extreme chart radii (e.g. near-antipodal spherical points) always keep the exact closed form — and an exact closed-form antipode `x/(c‖x‖²)`. Verified compatible with `ProductManifold` factors and the Riemannian optimizers (`ManifoldParam` + `riemannian_adam`/`riemannian_sgd`). The κ-GCN neural-network layers (`mobius_matvec`, weighted gyromidpoint, `dist2plane`, `sproj`) are not yet included. Double precision recommended when training a signed curvature that may cross zero, per the reference
- **`hyperbolix.utils.tanh`** — clamped hyperbolic tangent whose output stays strictly inside `(-1, 1)`: clips the input at the analytic saturation bound `±0.5·log(2/eps)` *and* the output at `±(1 − 10·eps)`, because XLA's float32 `tanh` saturates to exactly `1.0` slightly before the input bound bites. Matches `atanh`'s own domain guard, so `atanh(tanh(x))` can never reach the pole. Used by the κ-stereographic maps in the hyperbolic regime
- **Signed `identity` reparameterization for `LearnableCurvature`** (`parameterization="identity"`, `c = raw`) — **signed**, crossing zero smoothly with gradient 1, for learning the *sign* of curvature with the `Stereographic` manifold; `init_c` may be zero or negative, and the default clamp resolves to a symmetric `[-10.0, 10.0]` (deliberately including 0) instead of the positive `[0.1, 10.0]` used by `softplus`/`log`. The `log` scheme's exponent is now capped at `0.99·log(finfo.max)`, so a drifting raw parameter yields a large finite curvature instead of `inf`; the cap is gradient-transparent under `straight_through_clamp=True`

## [0.10.2] - 2026-07-02

### Changed
- **Legacy Ganea layers (`HypLinearPoincare`, `HypRegressionPoincare`) accept a `curvature` constructor argument** (float or callable, default 1.0) for their manifold-valued `bias` tag instead of hardcoding 1.0. The Riemannian optimizer uses this tag for the bias update, so it must match the `c` passed at call time — previously any `c ≠ 1.0` silently applied the wrong Riemannian correction to the bias

### Fixed
- Kernel-init corrections across hyperbolic layers: `HypRegressionHyperboloid` now uses the fan-scaled kernel init matching `HypRegressionPoincarePP`, and `HypConv2DHyperboloid` gains an opt-in `reset_params="fan_in"` for BatchNorm-free encoders
- `GyroBatchNormBase` gains a `min_var` floor that prevents collapsed-variance saturation
- `FGGConv2D` docstring stated the default weight init is `"kaiming"`; the actual default is `"lorentz_kaiming"`
- `PoincareBatchNorm2D` now documents that the learned variance is an unconstrained scalar under a sqrt (NaN if driven negative) — kept as-is to match the van Spengler reference, with a workaround noted

## [0.10.1] - 2026-06-30

### Added
- **`GyroBatchNorm` and `GyroRMSNorm`** normalization layers for the Hyperboloid and Proper Velocity manifolds

## [0.10.0] - 2026-06-29

### Added
- **LResNet `lorentz_scale` (He et al. 2025, Eq. 10)** — Klein-geodesic output rescaling that slides a hyperboloid point along the geodesic ray from the origin to control output norm while staying on the manifold — and a constrained `LorentzResidual` module built on it
- **Busemann MLR and BFC layers** (Chen et al. 2026)

### Changed
- **FGG layer init defaults changed to norm-preserving.** `FGGLinear` and `FGGConv2D` now default to `reset_params="fan_out"` (Gaussian, `std=sqrt(1/out_spatial)`) with `init_bias=0.0` and a new `gain=1.0` multiplier, so the output spatial norm tracks the input (`‖z‖ ≈ gain·‖x_spatial‖`) instead of growing as `sqrt(out_channels)` from step 0. This is a **deliberate deviation from the Klis et al. 2026 classification reference** (which normalizes magnitude away with BatchNorm/`FGGMeanOnlyBatchNorm` after each layer): it is motivated by *unnormalized* stacks — e.g. an RL backbone feeding a bounded Poincaré-ball projection, where the fan-in default saturated the projection. The new `gain` knob sets the effective column std to `gain/sqrt(out_spatial)`; it is a no-op for `reset_params="eye"` and is renormalized away under `use_weight_norm=True`. Restore the previous reference-style init with `reset_params="eye", init_bias=0.5` (`FGGLinear`) or `reset_params="lorentz_kaiming", init_bias=0.5` (`FGGConv2D`). `FGGLorentzMLR` (a terminal Euclidean-logits head) is intentionally unchanged. **Behavior change** — existing FGG models pick up the new init unless they pass the old `reset_params`/`init_bias` explicitly

## [0.9.0] - 2026-06-23

### Changed
- Performance improvements across the hyperboloid layers
- Unified the patch-extraction path into a shared `hyperboloid_core._extract_patches` helper (deduplication, no behavior change)

## [0.8.2] - 2026-06-12

### Changed
- **Renamed `HypConv2DHyperboloidPP` → `HypConv2DHyperboloidILNN`** — the Shi et al. 2026 Lorentz convolution (LogCat via `log_radius_concat` + PLFC), with origin padding (`pad_mode="origin"`) and ILNN improvements

## [0.8.1] - 2026-06-12

### Changed
- **Renamed `HypLinearHyperboloidPP` → `HypLinearHyperboloidPLFC`** with PLFC (parametrized Lorentz fully-connected) improvements (Shi et al. 2026)

## [0.8.0] - 2026-06-11

### Added
- **Poincaré vector quantization layers** (`hyperbolix.nn_layers`):
    - `HypVQEmbeddingPoincare`: HVQ-VAE quantizer (Chen et al. 2025) — explicit on-ball codebook held as a non-parameter `nnx.Variable` buffer, geodesic nearest-neighbour selection, copy-gradient straight-through estimator, and a hyperbolic-EMA codebook update (GGBall, Bu et al. 2026, Eqs. 41-43) with optional dead-code revival via the `ema_update` method (called after `optimizer.update`; the optimizer never touches the codebook, only the commitment loss trains the encoder)
    - `HypVQMLRPoincare`: HyperVQ quantizer (Goswami et al. 2025) — implicit codebook = the rows of an internal `HypRegressionPoincarePP`, Gumbel-Softmax straight-through selection on the categorical weights (so plain `optax.adam` trains it), with a `deterministic` flag (toggled by `model.eval()` / `model.train()`) for a deterministic argmax MAP estimate at inference
    - `PoincareVQOutput`: shared pytree-friendly `NamedTuple` return type `(quantized, indices, loss, perplexity, z)`; `quantized` is cast to float32 at the manifold→decoder boundary
    - `poincare_weighted_midpoint`: reusable Poincaré weighted gyromidpoint (GGBall Eq. 41), the Poincaré analog of `lorentz_midpoint` and a weighted generalization of `poincare_midpoint`
    - **`squared_commitment` flag on `HypVQEmbeddingPoincare`** (default `False`). The cited references genuinely disagree on the commitment penalty: HVQ-VAE uses the plain geodesic distance `d(z, sg(q))`, GGBall's L_HVQVAE uses `d²` (the Euclidean VQ-VAE convention). The flag selects between them without touching library code
- **`Hyperboloid.log_radius_concat`** — log-radius-preserving concatenation of N Lorentz points (Shi et al. 2026, Sec. 4.3), the hyperboloid analog of Poincaré β-concatenation. Rescales each block's spatial part by `exp(½·(ψ(N·d/2) − ψ(d/2)))` (digamma `ψ`) so the expected log spatial radius stays invariant to the post-concat dimension, then recomputes the time coordinate to keep the result on the manifold. Reduces to `hcat` when `N = 1`; convolutional layers keep using the unscaled `hcat` by default
- **Isometry mappings now span all three models** — Poincaré ↔ Hyperboloid, Poincaré ↔ Proper Velocity, and Hyperboloid ↔ Proper Velocity. New functions: `pv_to_poincare`, `poincare_to_pv`, `pv_to_hyperboloid`, `hyperboloid_to_pv` (in `hyperbolix.manifolds.isometry_mappings`). All six maps are exact, curvature-correct Riemannian isometries (PVNN Eq. 4; Chen et al. 2026). The direct `pv_to_hyperboloid` (PV coords are the hyperboloid's spatial part; time reconstructed from `⟨z,z⟩_L = -1/c`) avoids the near-boundary blow-up of composing through the ball
- Self-hosted MathJax with a CI external-host allowlist guard

### Changed
- **All NN layers now take a `param_dtype` constructor argument (default `jnp.float32`)** controlling the *storage* dtype of every trainable parameter and persistent state (batch-norm statistics, VQ codebooks, scalar scales/temperatures). The manifold's `dtype` controls *compute* precision only. Under global `jax_enable_x64`, parameters previously materialized as float64 (2× parameter/optimizer memory, float64 checkpoints); they now stay float32 unless `param_dtype=jnp.float64` is passed explicitly. The Riemannian optimizers complete the contract: `egrad2rgrad`/`expmap`/`ptransp` run in the manifold dtype, but returned updates and momentum buffers are cast back to the parameter's storage dtype (previously a float32 `ManifoldParam` next to a float64 manifold silently promoted its optimizer state to float64 after one step). See [Numerical Stability — Storage vs. Compute Dtype](user-guide/numerical-stability.md#storage-vs-compute-dtype). **Breaking changes:**
    - `FGGMeanOnlyBatchNorm(dtype=...)` renamed to `param_dtype=...`
    - `PoincareBatchNorm2D` and `HypVQEmbeddingPoincare` state is no longer pinned to `manifold.dtype` — it defaults to float32; pass `param_dtype=manifold.dtype` to restore the old behavior
    - Checkpoints saved under global x64 before this change carry float64 leaves and will dtype-mismatch on restore into float32-storage modules — re-initialize or cast the state tree on load
- **`Hyperboloid.addition` now implements the Lorentz gyrovector addition** `x ⊕ y = Exp_x(PT_{0→x}(Log_0(y)))` (Chen et al. 2025b; Shi et al. 2026, Eq. 1), and `ProductManifold.addition` delegates it per factor. It forms a gyrocommutative gyrogroup (identity = origin, inverse `⊖x = (-1) ⊙ x = [x₀, -x_s]`) and coincides with Poincaré Möbius addition under the stereographic isometry. This replaces the earlier coordinate-wise formula, which was incorrect at *every* curvature (e.g. `origin ⊕ y ≠ y`) and failed silently via a trailing projection. `Hyperboloid.scalar_mul` (geodesic scaling, = Eq. 2) is unchanged. **Breaking change** — `addition` no longer raises `NotImplementedError`
- **Breaking: `VERSION_LORENTZIAN_PROXY` removed from the Poincaré manifold.** The formula treated Poincaré ball coordinate 0 as a Lorentz time component, so it was only meaningful for hyperboloid inputs and returned `dist(x, x) = -2/c ≠ 0` for ball points; the docs additionally mis-advertised it as "best near boundary". The reference implementation never had it. `dist`/`dist_0` now select between 3 versions (`VERSION_MOBIUS_DIRECT`, `VERSION_MOBIUS`, `VERSION_METRIC_TENSOR`). For a true Lorentzian (squared Minkowski) proxy, convert to the hyperboloid via `isometry_mappings.poincare_to_hyperboloid` first

### Fixed
- **Poincaré ↔ Hyperboloid isometry maps were only correct at `c=1`.** `poincare_to_hyperboloid` / `hyperboloid_to_poincare` used unit-ball stereographic formulas missing their `√c` factors, so round-trips and distance preservation silently failed at any other curvature (the recommended default is `c=0.1`). Both are now curvature-correct at all `c`, verified by round-trip, isometry, commutative-diagram, and extreme-curvature tests across dtypes
- **Test fixture `uniform_points` generated hyperboloid points at the wrong geodesic radius for `c ≠ 1`** (an extra `/√c` on the spatial coordinate, masked by a trailing projection). Corrected so generated points are the exact Poincaré images at any curvature
- **Silent float64 promotion under x64 in hyperboloid layers.** Runtime buffers created without an explicit dtype (`FGGConv2D` SAME/origin padding buffer, causal linear-attention scan carries, full-attention uniform head-averaging weights) now derive their dtype from the input, and the scalar attention parameters (`temperature`, `scale`, `attn_bias`) are cast to the compute dtype at use. Previously, with `jax_enable_x64` active, these defaulted to float64 and promoted float32 activations through the rest of the layer

## [0.7.2] - 2026-05-22

### Changed
- **`ProductManifold` curvature API redesigned** — every geometry method (`dist`, `expmap`, `logmap`, `proj`, `origin`, `tangent_inner`, etc.) now takes a positional `c` argument that must be a sequence of length `n_factors` (one curvature per factor) instead of the silently-ignored scalar `c`. There is no default and no scalar broadcast: pass `product.curvatures` for static curvatures, or build the sequence from `LearnableCurvature` calls for trainable ones. The protocol-level `Curvature` type was widened to `ScalarCurvature | Sequence[ScalarCurvature]` so `ProductManifold` satisfies the `Manifold` protocol — `isinstance(product, Manifold)` is `True` and generic code typed against `Manifold` accepts product instances. The product still has no `c` attribute (use `product.curvatures`). **Breaking change** — call sites that passed `c=0.0` (or any scalar) must now pass a per-factor sequence `(c_0, c_1, …)`

### Fixed
- Corrected the `ProductManifold` API examples and the protocol-conformance claim in the documentation

## [0.7.1] - 2026-05-21

### Added
- **`LearnableCurvature(nnx.Module)`** — canonical module for trainable curvature in `hyperbolix.utils.curvature`, also exported at top level (`from hyperbolix import LearnableCurvature`). Introduces the `softplus` (default, `c = softplus(raw)`, bounded gradient via sigmoid, van Spengler 2023 convention) and `log` (`c = exp(raw)`, scale-invariant gradient `dc/draw = c`, MERU convention, preferred for compiled RL loops) reparameterizations. The default clamp is applied to the recovered `c` (not the raw parameter) as a hard stability guard for long compiled training loops — `[0.1, 10.0]` for both schemes; pass `c_min=None, c_max=None` to disable. Updated by any standard Euclidean `nnx.Optimizer` — no Riemannian optimizer needed

### Changed
- **Manifolds are now plain Python classes** (not `nnx.Module`). This structurally prevents shared-manifold bugs in `nnx.scan`/`nnx.fori_loop` — manifolds become static graphdef attributes with no state pytree entries
- **`learnable=True` removed** from manifold constructors. Use the `LearnableCurvature` module instead — assign one instance per distinct curvature on your model. See the [Manifolds User Guide](user-guide/manifolds.md#static-vs-learnable)
- **`ProductManifold`** is now a plain class; `from_signature` accepts 3- and 4-tuple specs only (5-tuple `learnable` override removed)
- **`learnable_curvature()` / `get_curvature()` functional helpers replaced** by the `LearnableCurvature(nnx.Module)` class. The class bundles the raw parameter, reparameterization scheme, and clamp bounds in one object, making accidental init/recovery mismatches structurally impossible. Call sites change from `c = get_curvature(self.c_raw)` to `c = self.curvature()`. **Breaking change** — no deprecation shim

## [0.7.0] - 2026-05-18

### Added
- **Product manifold** (`hyperbolix.manifolds.ProductManifold`) — heterogeneous-curvature composition $M_1 \times M_2 \times \dots \times M_n$ where each factor may be any base manifold with its own curvature (Gu et al. 2019). Points are flat concatenated arrays of shape `(total_dim,)`; geometry methods take a per-factor curvature sequence as positional `c` at call time. Provides Pythagorean L2 geodesic distance plus auxiliary `dist_l1` / `dist_min` / `component_dist` reductions, full per-factor decomposition of `expmap`/`logmap`/`ptransp`/`proj`/`egrad2rgrad`/`tangent_inner`, an `origin(c)` helper, and a `from_signature` factory accepting 3- and 4-tuple specs
- **Learnable curvature via an `nnx.Module`** — first-class trainable curvature assigned per distinct curvature on the model (later canonicalized into `LearnableCurvature` in v0.7.1)

## [0.6.0] - 2026-04-20

### Added
- **Proper Velocity (PV) manifold** (`hyperbolix.manifolds.ProperVelocity`) — unconstrained $\mathbb{R}^n$ model of hyperbolic geometry from Chen et al. (2026), with complete geometric operations: `addition`, `scalar_mul`, `dist`, `expmap`/`logmap` (at origin and arbitrary base points), `ptransp`/`ptransp_0`, `egrad2rgrad`, and Riemannian inner product
- **Proper Velocity neural-network layers**:
    - `HypLinearPV`: PV fully-connected layer (Thm 5.3 / Eq. 22)
    - `HypConv2DPV`: PV 2D convolution with raw Euclidean patch concatenation (Sec 5.3) — no beta-scaling, dimension-preserving
    - `HypRegressionPV`: PV multinomial-logistic-regression head (Thm 5.2 / Eq. 19)

## [0.5.3] - 2026-04-14

### Added
- Hyperbolic average-pooling layer

## [0.5.2] - 2026-04-02

### Added
- `HypLinearHyperboloidFHNN` and `HypConv2DHyperboloidFHNN` layers (fully-hyperbolic neural network, Bdeir et al. 2023)

## [0.5.1] - 2026-03-27

### Added
- **`PoincareBatchNorm2D`** — Poincaré ball BatchNorm between conv layers (van Spengler et al. 2023)

## [0.5.0] - 2026-03-26

### Removed
- `HypRegressionPoincareHDRL` layer and its tests

## [0.4.2] - 2026-03-24

### Changed
- Removed `HypConv3DHyperboloid`; `beta_scale` is now wrapped in `float`

## [0.4.1] - 2026-03-19

### Changed
- Flattened conv-layer parameter paths via forward-function extraction (cleaner param trees, no behavior change)

## [0.4.0] - 2026-03-17

### Added
- MkDocs Material documentation system
- Complete API reference documentation
- Getting Started guide
- CI/CD workflow for documentation builds

### Changed
- Adopted Flax NNX layer naming convention (`kernel`/`bias`) across all layers

## [0.3.0] - 2026-03-10

### Added
- **FGG-LNN layers** (`FGGLinear`, `FGGConv2D`, `FGGLorentzMLR`) — fully-generalized Lorentz layers (Klis et al. 2026)

## [0.2.1] - 2026-03-06

### Changed
- Migrated benchmarks to Grain data loading

## [0.2.0] - 2026-03-05

### Added
- Class-based manifold API with automatic dtype casting (`Poincare`, `Hyperboloid`, `Euclidean`)
- `Manifold` structural protocol for type-safe manifold dispatch
- Poincaré convolution layers
- Positional encoding layers for hyperbolic Transformers:
    - `lorentz_residual`: Lorentzian midpoint-based residual connection
    - `hope`: Hyperbolic Rotary Positional Encoding (functional)
    - `HyperbolicRoPE`: NNX module wrapper for HOPE
    - `HypformerPositionalEncoding`: Learnable positional encoding with HTCLinear
- **Causal attention masking** (`causal=True`) for all three hyperbolic attention variants:
    - `HyperbolicSoftmaxAttention`: lower-triangular `-inf` mask before softmax
    - `HyperbolicFullAttention`: lower-triangular `-inf` mask on Lorentzian similarity scores
    - `HyperbolicLinearAttention`: O(N) cumulative-sum recurrence via `jax.lax.scan` (Katharopoulos et al. 2020), keeping O(N) complexity in causal mode
- Tiny Shakespeare character-level benchmark (`benchmarks/bench_shakespeare_attention.py`) comparing all four model variants (Euclidean + 3 hyperbolic) with causal attention

### Changed
- **Breaking**: Manifold public functions renamed to private (`dist()` → `_dist()`); use class methods instead
- Replaced `with_precision()` wrapper with the `Poincare(dtype=jnp.float64)` pattern
- **Breaking: `HypformerPositionalEncoding.epsilon` is no longer a learnable `nnx.Param`** — it is a fixed, non-negative constructor float (default 1.0), matching the Hypformer reference (which keeps it a plain non-trainable tensor). A learnable epsilon could be driven below -1 by gradient descent, where `x + epsilon * p` leaves the upper hyperboloid sheet and the `abs()` in the `lorentz_residual` normalizer silently masks the violation. `lorentz_residual` now documents the `w_y >= 0` requirement

## [0.1.4] - 2026-02-10

### Added
- Pure JAX implementation of hyperbolic manifolds (Euclidean, Poincaré, Hyperboloid)
- 13+ neural network layers (linear, convolutional, regression)
- Hypformer components: HTC/HRC with curvature-change support
- 4 hyperbolic activation functions (ReLU, Leaky ReLU, Tanh, Swish)
- Riemannian optimizers (RSGD, RAdam) with automatic manifold detection
- Isometry mappings between the Poincaré ball and hyperboloid models
- Wrapped normal distributions for VAEs
- Comprehensive test suite (1,400+ tests)
- CI/CD pipeline with benchmarking
- vmap-native API design

### Changed
- Migrated from PyTorch to pure JAX/Flax NNX
- Unified package structure: `hyperbolix_jax` → `hyperbolix`

### References
- Based on research by Ganea et al. (2018), Bécigneul & Ganea (2019), Bdeir et al. (2023)

[Unreleased]: https://github.com/timoklein/hyperbolix/compare/v0.11.1...HEAD
[0.11.1]: https://github.com/timoklein/hyperbolix/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/timoklein/hyperbolix/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/timoklein/hyperbolix/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/timoklein/hyperbolix/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/timoklein/hyperbolix/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/timoklein/hyperbolix/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/timoklein/hyperbolix/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/timoklein/hyperbolix/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/timoklein/hyperbolix/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/timoklein/hyperbolix/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/timoklein/hyperbolix/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/timoklein/hyperbolix/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/timoklein/hyperbolix/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/timoklein/hyperbolix/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/timoklein/hyperbolix/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/timoklein/hyperbolix/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/timoklein/hyperbolix/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/timoklein/hyperbolix/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/timoklein/hyperbolix/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/timoklein/hyperbolix/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/timoklein/hyperbolix/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/timoklein/hyperbolix/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/timoklein/hyperbolix/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/timoklein/hyperbolix/releases/tag/v0.1.4
