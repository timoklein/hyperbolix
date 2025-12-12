# Hyperbolix - Project Handoff

**Hyperbolix** is a pure JAX implementation of hyperbolic deep learning, providing manifold operations, neural network layers, and Riemannian optimizers for hyperbolic geometry. Built with Flax NNX and Optax for modern JAX workflows.

## Current Status

**Phase 1 (Manifolds) ✅** | **Phase 2 (Optimizers) ✅** | **Phase 3a & 3b (NN Layers) ✅** | **Idiomatic JAX Refactor ✅** | **CI/Tooling ✅** | **Code Cleanup ✅**

### Test Results
- **Manifolds**: 978 passing, 72 skipped (100% non-skipped)
- **NN Layers**: 44/44 passing (100%)
- **Hyperboloid Convolution**: 68/68 passing (100%) - includes 2D (44 tests) and 3D (24 tests)
- **Lorentz Convolution**: 66/66 passing (100%) - LorentzConv2D and LorentzConv3D layers
- **Hyperboloid Activations**: 86/86 passing (100%) - hyp_relu, hyp_leaky_relu, hyp_tanh, hyp_swish
- **Math Utils**: 8/8 passing (100%)
- **Helper Utils**: 38/38 passing (100%)
- **HoroPCA**: 25/25 passing (100%)
- **Optimizers**: 20/20 passing (100%)
- **Benchmarks**: 168 test cases passing (100%)

---

## What's Complete

### Phase 1: Core Geometry (Manifolds)

**Location**: `hyperbolix/manifolds/`

**Implemented**:
- ✅ `euclidean.py` - Flat Euclidean space
- ✅ `poincare.py` - Poincaré ball with Möbius operations
- ✅ `hyperboloid.py` - Hyperboloid with Lorentz/Minkowski geometry
- ✅ `*_checked.py` - Checkify error handling modules for all manifolds

**API**:
- Pure functional design (no classes, no state)
- vmap-native API: functions operate on single points, no `axis`/`keepdim` parameters
- Integer version indices with `lax.switch` for JIT optimization
- Operations: proj, addition, scalar_mul, dist, expmap, logmap, retraction, ptransp, tangent operations, egrad2rgrad, validation

**Math Utilities** (`hyperbolix/utils/math_utils.py`):
- ✅ JIT-compiled hyperbolic functions (cosh, sinh, acosh, atanh)
- ✅ Numerically stable smooth clamping with static `smoothing_factor`

**Helper Utilities** (`hyperbolix/utils/helpers.py`):
- ✅ `compute_pairwise_distances`: Efficient pairwise distance computation using vmap
- ✅ `compute_hyperbolic_delta`: Delta-hyperbolicity metric based on Gromov 4-point condition
- ✅ `get_delta`: Combined delta, diameter, and relative delta computation with subsampling

**HoroPCA** (`hyperbolix/utils/horo_pca.py`):
- ✅ `compute_frechet_mean`: Gradient descent Fréchet mean on hyperboloid
- ✅ `center_data`: Lorentz transformation centering via Lorentz boost
- ✅ `HoroPCA`: Flax NNX module for hyperbolic dimensionality reduction via horospherical projections
- ✅ Supports Poincaré & Hyperboloid manifolds, rank-1 special case, pinv for stability

### Phase 3a: Linear Neural Network Layers

**Location**: `hyperbolix/nn_layers/`

**Implemented**:
- ✅ Standard layers: Expmap, Logmap, Proj, TanProj, Retraction, HyperbolicActivation
- ✅ Poincaré: HypLinearPoincare, HypLinearPoincarePP
- ✅ Hyperboloid: HypLinearHyperboloid, FHNN, FHCNN variants
- ✅ Hyperboloid Convolution: HypConv2DHyperboloid, HypConv3DHyperboloid with Lorentz direct concatenation (HCat)
  - `HypConvHyperboloid` is backward-compatible alias for `HypConv2DHyperboloid`
- ✅ Lorentz Convolution: LorentzConv2D, LorentzConv3D implementing "Fully Hyperbolic CNNs" (Bdeir et al., 2023)
  - Pipeline: RotationConv → DistanceRescaling → LorentzBoost
  - `lorentz_boost()` and `distance_rescale()` operations in hyperboloid.py
  - Norm-preserving rotation convolution with automatic Algorithm 3 condition checking
  - Optional distance rescaling and Lorentz boost transformations
- ✅ Hyperboloid Activations: hyp_relu, hyp_leaky_relu, hyp_tanh, hyp_swish
  - Functional implementations that apply activation to space components
  - Reconstructs time component using manifold constraint
  - Avoids frequent exp/log maps for better numerical stability
  - Works on arrays of any shape (similar to jax.nn.relu)

**Architecture**: Flax NNX modules storing manifold module references, curvature `c` passed at call time (layers); pure functions for activations

### Phase 3b: Regression Neural Network Layers

**Implemented**:
- ✅ Poincaré: HypRegressionPoincare, HypRegressionPoincarePP
- ✅ Hyperboloid: HypRegressionHyperboloid
- ✅ RL: HypRegressionPoincareHDRL (standard & rs versions)
- ✅ Helpers: compute_mlr_poincare_pp, compute_mlr_hyperboloid, safe_conformal_factor

### Phase 2: Riemannian Optimizers

**Location**: `hyperbolix/optim/`

**Implemented**:
- ✅ `manifold_metadata.py` - Metadata system using NNX Variable._var_metadata
- ✅ `riemannian_sgd.py` - RSGD with momentum & parallel transport
- ✅ `riemannian_adam.py` - RAdam with adaptive rates & moment transport
- ✅ Layer annotations: HypLinearPoincare, HypRegressionPoincare bias marked as manifold params
- ✅ Multi-parameter pytree handling & PyTorch-style second-moment accumulation (all tests green)

**Architecture**: Standard Optax GradientTransformation, automatic manifold detection, supports expmap/retraction modes

---

## Recent Improvements

### Project Structure Cleanup (2025-12-12)
- ✅ Removed PyTorch legacy code: cleaned up old PyTorch implementation
- ✅ Unified directory structure: `src/hyperbolix_jax/` → `hyperbolix/`, `tests/jax/` → `tests/`
- ✅ Updated package name: `hyperbolix_jax` → `hyperbolix` across all imports
- ✅ Updated all configuration files: pyproject.toml, CI workflows, pre-commit hooks
- ✅ Updated all documentation: DEVELOPER_GUIDE.md, handoff.md, code examples
- ✅ Updated all docstrings: 30+ files with corrected import paths
- ✅ Verified: 0 references to old paths remaining

### Lorentz Convolution Implementation (2025-12-11)
- ✅ Implemented LorentzConv2D and LorentzConv3D from "Fully Hyperbolic CNNs" paper
- ✅ Added `lorentz_boost()` and `distance_rescale()` to hyperboloid.py manifold
- ✅ Fixed 9 critical bugs from implementation review:
  - Fixed velocity projection formula in lorentz_boost (prevented gamma=inf)
  - Added near-zero convolution output handling for numerical stability
  - Fixed distance_rescale origin edge case with L'Hopital limit
  - Added dtype parameter support (float32/float64) throughout layer
  - Added manifold projection after boost/rescale operations
  - Replaced no-op Algorithm 3 check with meaningful condition warning at init
  - Documented SAME padding norm pooling behavior (edge value weighting)
- ✅ All 66 Lorentz convolution tests passing (shape, manifold constraint, gradients, JIT, curvature)

### Idiomatic JAX Refactor (2025-10-09)
- ✅ vmap-native API: All manifolds refactored to single-point operations
- ✅ `lax.switch` for version selection with integer indices
- ✅ `jnp.finfo(x.dtype)` for dtype-aware epsilon
- ✅ Checkify modules for runtime validation
- ✅ NN layers updated: assertions in `__init__`, vmap API

### CI & Tooling Hardening (2025-10-16)
- ✅ CI pipeline: split into lint, type-check, test matrix (4 parallel suites), benchmark jobs
- ✅ Pre-commit hooks: Ruff linting/formatting, YAML/TOML validation, merge conflict detection
- ✅ Pyright type checking: configured for `src` with Python 3.12+ syntax
- ✅ JIT benchmarks: 168 parametrized tests across manifolds & NN layers
- ✅ Benchmark regression detection: fails on >10% slowdown
- ✅ Legacy code modernized: built-in generics, f-strings, Python 3.12+

### Developer Workflow (2025-10-15)
- ✅ Pre-commit: automatic formatting & validation
- ✅ DEVELOPER_GUIDE.md with comprehensive workflow reference
- ✅ benchmarks/README.md with usage guide
- ✅ UV dependency caching for faster CI

---

## Architecture

### Pure Functional Design
```python
# Old PyTorch (stateful class)
manifold = Hyperboloid(c=1.0)
result = manifold.dist(x, y)

# New JAX (pure functions, vmap-native)
import hyperbolix.manifolds.hyperboloid as hyperboloid
result = hyperboloid.dist(x, y, c=1.0)  # (dim,) -> scalar
```

### NN Layer Pattern
```python
class HypLinearPoincare(nnx.Module):
    def __init__(self, manifold_module, in_dim, out_dim, *, rngs):
        self.manifold = manifold_module
        self.weight = nnx.Param(...)
        self.bias = nnx.Param(...)

    def __call__(self, x, c=1.0):  # x: (batch, in_dim)
        # Forward pass with manifold operations
        ...
```

---

## Test Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test suites
uv run pytest tests/test_manifolds.py -v
uv run pytest tests/test_nn_layers.py -v
uv run pytest tests/test_regression_layers.py -v

# Run benchmarks
uv run pytest benchmarks/ -v

# Run pre-commit checks
uv run pre-commit run --all-files
```

---

## Key Files

### Manifolds
- `hyperbolix/manifolds/{euclidean,poincare,hyperboloid}.py` (includes `hcat` in hyperboloid)
- `hyperbolix/manifolds/{euclidean,poincare,hyperboloid}_checked.py`
- `hyperbolix/utils/math_utils.py`
- `hyperbolix/utils/helpers.py`

### NN Layers
- `hyperbolix/nn_layers/standard_layers.py`
- `hyperbolix/nn_layers/{poincare,hyperboloid}_linear.py`
- `hyperbolix/nn_layers/hyperboloid_conv.py` - Hyperboloid convolution with HCat
- `hyperbolix/nn_layers/lorentz_conv.py` - Lorentz convolution (LorentzConv2D/3D)
- `hyperbolix/nn_layers/hyperboloid_activations.py` - Hyperboloid activation functions (hyp_relu, hyp_leaky_relu, hyp_tanh, hyp_swish)
- `hyperbolix/nn_layers/{poincare,hyperboloid}_regression.py`
- `hyperbolix/nn_layers/poincare_rl.py`
- `hyperbolix/nn_layers/helpers.py`

### Optimizers
- `hyperbolix/optim/manifold_metadata.py`
- `hyperbolix/optim/riemannian_{sgd,adam}.py`
- `OPTIMIZER_PLAN.md` - Design document

### Tests
- `tests/test_manifolds.py` (912 parametrized tests)
- `tests/test_nn_layers.py` (22 tests)
- `tests/test_hyperboloid_conv.py` (68 tests: HCat operation + 2D/3D conv layers)
  - HCat: 5 tests (manifold constraint, dimensionality, time coordinate formula, space concatenation)
  - 2D Conv: 39 tests (shape, manifold constraint, stride, curvature, tangent input)
  - 3D Conv: 24 tests (shape, manifold constraint, stride, curvature, tangent input, anisotropic kernels)
- `tests/test_lorentz_conv.py` (66 tests: LorentzConv2D and LorentzConv3D layers)
  - 2D Conv: shape, manifold constraint, stride, input_space, gradients, JIT, curvature, boost/rescaling flags
  - 3D Conv: shape, manifold constraint, stride, curvature, gradients, JIT, boost/rescaling flags
- `tests/test_hyperboloid_activations.py` (86 tests: hyp_relu, hyp_leaky_relu, hyp_tanh, hyp_swish)
  - Manifold constraint tests (single point, batch, multi-dim batches)
  - Shape preservation tests (different dtypes, dimensions, batch sizes)
  - Correctness tests (formula verification, activation behavior)
  - Gradient tests (finite gradients for all activations)
  - JIT compatibility tests
  - Curvature tests (different c values)
  - Edge case tests (zero inputs, moderate magnitudes)
- `tests/test_regression_layers.py` (22 tests)
- `tests/test_optimizers.py` (20/20 passing; covers metadata, mixed params, NNX integration)
- `tests/test_math_utils.py` (8 tests)
- `tests/test_helpers.py` (38 tests)
- `tests/test_horo_pca.py` (25 tests: Fréchet mean, centering, fit/transform, rank-1)

### Documentation
- `DEVELOPER_GUIDE.md` - Development workflow
- `benchmarks/README.md` - Benchmark usage
- `.github/workflows/ci.yaml` - CI pipeline
- `pyproject.toml` - Project config, Pyright settings

---

## Project Status

**All core functionality is complete and production-ready!** The project includes:
- ✅ Full manifold operations (Euclidean, Poincaré, Hyperboloid)
- ✅ Neural network layers (linear, convolutional, regression)
- ✅ Riemannian optimizers (RSGD, RAdam)
- ✅ Probability distributions (wrapped normal)
- ✅ Utility functions (HoroPCA, delta-hyperbolicity)
- ✅ Comprehensive test suite (100% passing)
- ✅ CI/CD pipeline with benchmarking
- ✅ Clean, unified codebase structure

## Next Steps

1. **End-to-end examples** - Training loops demonstrating JAX/NNX usage
2. **Documentation** - API docs, usage examples, JIT best practices
3. **Distribution package** - Publish to PyPI as `hyperbolix`

## Known Issues

- None currently tracked for optimizers; flag new regressions in CI.

## Edge Cases & Considerations

### Hyperboloid Convolutional Layers (2D & 3D)

1. **Padding Strategy**
   - Uses `mode="edge"` (replicates border pixels/voxels) instead of zero-padding
   - **Rationale**: Zero vectors don't lie on the hyperboloid manifold; edge replication preserves valid manifold points
   - **Implication**: Different behavior from standard Euclidean CNNs at boundaries
   - **Applies to**: Both 2D and 3D convolutions

2. **Dimensional Growth in Multi-Layer Architectures**
   - HCat operation increases dimensionality: input ambient dim `d+1` → output ambient dim `(d×N)+1`
   - **2D Example**: `N = kernel_h × kernel_w`, e.g., 3-dim input with 3×3 kernel → 3×9+1 = 28-dim output
   - **3D Example**: `N = kernel_d × kernel_h × kernel_w`, e.g., 3-dim input with 2×2×2 kernel → 3×8+1 = 25-dim output
   - **Implication**: Dimension grows rapidly in deep networks; consider small kernels or dimensionality reduction between layers

3. **3D Convolution Design**
   - **Separate class**: `HypConv3DHyperboloid` (not a generic `ndim` parameter)
   - **Rationale**: Optimal JIT performance, type safety with explicit 5D shapes, matches PyTorch/TF conventions
   - **Input shape**: `(batch, depth, height, width, in_channels)`
   - **Kernel/stride**: 3-tuples `(d, h, w)` or scalar (expanded to cubic)
   - **Anisotropic kernels**: Supported, e.g., `kernel_size=(2, 3, 2)` for different spatial scales

4. **Numerical Stability in HCat**
   - Formula: `sqrt(sum(x_i[0]^2) - (N-1)/c)` requires `sum(x_i[0]^2) ≥ (N-1)/c`
   - **Valid for**: Properly initialized hyperboloid points where `x[0] ≥ 1/sqrt(c)`
   - **Risk**: Low under normal conditions; NaN possible with invalid inputs or extreme curvatures
   - **Mitigation**: Input validation via `is_in_manifold` checks

5. **Jaxtyping Annotations**
   - Ruff linter reports F722/F821 errors on shape specifications (`"N n"`, `"dim_plus_1"`, etc.)
   - **Status**: False positives - jaxtyping uses string literals for runtime shape checking
   - **Action**: Ignore these specific Ruff errors; pattern used consistently throughout codebase
