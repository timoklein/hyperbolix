# Hyperbolix JAX Migration - Project Handoff

## Current Status

**Phase 1 (Manifolds) ✅** | **Phase 2 (Optimizers) 🚧** | **Phase 3a & 3b (NN Layers) ✅** | **Idiomatic JAX Refactor ✅** | **CI/Tooling ✅**

### Test Results
- **Manifolds**: 978 passing, 72 skipped (100% non-skipped)
- **NN Layers**: 44/44 passing (100%)
- **Math Utils**: 8/8 passing (100%)
- **Helper Utils**: 38/38 passing (100%)
- **HoroPCA**: 25/25 passing (100%)
- **Optimizers**: 15/20 passing (75%) - 5 multi-parameter tests need tree handling fix
- **Benchmarks**: 168 test cases passing (100%)

---

## What's Complete

### Phase 1: Core Geometry (Manifolds)

**Location**: `src/hyperbolix_jax/manifolds/`

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

**Math Utilities** (`src/hyperbolix_jax/utils/math_utils.py`):
- ✅ JIT-compiled hyperbolic functions (cosh, sinh, acosh, atanh)
- ✅ Numerically stable smooth clamping with static `smoothing_factor`

**Helper Utilities** (`src/hyperbolix_jax/utils/helpers.py`):
- ✅ `compute_pairwise_distances`: Efficient pairwise distance computation using vmap
- ✅ `compute_hyperbolic_delta`: Delta-hyperbolicity metric based on Gromov 4-point condition
- ✅ `get_delta`: Combined delta, diameter, and relative delta computation with subsampling

**HoroPCA** (`src/hyperbolix_jax/utils/horo_pca.py`):
- ✅ `compute_frechet_mean`: Gradient descent Fréchet mean on hyperboloid
- ✅ `center_data`: Lorentz transformation centering via Lorentz boost
- ✅ `HoroPCA`: Flax NNX module for hyperbolic dimensionality reduction via horospherical projections
- ✅ Supports Poincaré & Hyperboloid manifolds, rank-1 special case, pinv for stability

### Phase 3a: Linear Neural Network Layers

**Location**: `src/hyperbolix_jax/nn_layers/`

**Implemented**:
- ✅ Standard layers: Expmap, Logmap, Proj, TanProj, Retraction, HyperbolicActivation
- ✅ Poincaré: HypLinearPoincare, HypLinearPoincarePP
- ✅ Hyperboloid: HypLinearHyperboloid, FHNN, FHCNN variants

**Architecture**: Flax NNX modules storing manifold module references, curvature `c` passed at call time

### Phase 3b: Regression Neural Network Layers

**Implemented**:
- ✅ Poincaré: HypRegressionPoincare, HypRegressionPoincarePP
- ✅ Hyperboloid: HypRegressionHyperboloid
- ✅ RL: HypRegressionPoincareHDRL (standard & rs versions)
- ✅ Helpers: compute_mlr_poincare_pp, compute_mlr_hyperboloid, safe_conformal_factor

### Phase 2: Riemannian Optimizers 🚧

**Location**: `src/hyperbolix_jax/optim/`

**Implemented**:
- ✅ `manifold_metadata.py` - Metadata system using NNX Variable._var_metadata
- ✅ `riemannian_sgd.py` - RSGD with momentum & parallel transport
- ✅ `riemannian_adam.py` - RAdam with adaptive rates & moment transport
- ✅ Layer annotations: HypLinearPoincare, HypRegressionPoincare bias marked as manifold params
- ✅ Multi-parameter pytree handling & PyTorch-style second-moment accumulation (all tests green)

**Architecture**: Standard Optax GradientTransformation, automatic manifold detection, supports expmap/retraction modes

---

## Recent Improvements

### Idiomatic JAX Refactor (2025-10-09)
- ✅ vmap-native API: All manifolds refactored to single-point operations
- ✅ `lax.switch` for version selection with integer indices
- ✅ `jnp.finfo(x.dtype)` for dtype-aware epsilon
- ✅ Checkify modules for runtime validation
- ✅ NN layers updated: assertions in `__init__`, vmap API

### CI & Tooling Hardening (2025-10-16)
- ✅ CI pipeline: split into lint, type-check, test matrix (4 parallel suites), benchmark jobs
- ✅ Pre-commit hooks: Ruff linting/formatting, YAML/TOML validation, merge conflict detection
- ✅ Pyright type checking: configured for `src/hyperbolix_jax` with Python 3.12+ syntax
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
import hyperbolix_jax.manifolds.hyperboloid as hyperboloid
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
# Run all JAX tests
uv run pytest tests/jax/ -v

# Run specific test suites
uv run pytest tests/jax/test_manifolds.py -v
uv run pytest tests/jax/test_nn_layers.py -v
uv run pytest tests/jax/test_regression_layers.py -v

# Run benchmarks
uv run pytest benchmarks/ -v

# Run pre-commit checks
uv run pre-commit run --all-files
```

---

## Key Files

### Manifolds
- `src/hyperbolix_jax/manifolds/{euclidean,poincare,hyperboloid}.py`
- `src/hyperbolix_jax/manifolds/{euclidean,poincare,hyperboloid}_checked.py`
- `src/hyperbolix_jax/utils/math_utils.py`
- `src/hyperbolix_jax/utils/helpers.py`

### NN Layers
- `src/hyperbolix_jax/nn_layers/standard_layers.py`
- `src/hyperbolix_jax/nn_layers/{poincare,hyperboloid}_linear.py`
- `src/hyperbolix_jax/nn_layers/{poincare,hyperboloid}_regression.py`
- `src/hyperbolix_jax/nn_layers/poincare_rl.py`
- `src/hyperbolix_jax/nn_layers/helpers.py`

### Optimizers
- `src/hyperbolix_jax/optim/manifold_metadata.py`
- `src/hyperbolix_jax/optim/riemannian_{sgd,adam}.py`
- `OPTIMIZER_PLAN.md` - Design document

### Tests
- `tests/jax/test_manifolds.py` (912 parametrized tests)
- `tests/jax/test_nn_layers.py` (22 tests)
- `tests/jax/test_regression_layers.py` (22 tests)
- `tests/jax/test_optimizers.py` (20/20 passing; covers metadata, mixed params, NNX integration)
- `tests/jax/test_math_utils.py` (8 tests)
- `tests/jax/test_helpers.py` (38 tests)
- `tests/jax/test_horo_pca.py` (25 tests: Fréchet mean, centering, fit/transform, rank-1)

### Documentation
- `DEVELOPER_GUIDE.md` - Development workflow
- `benchmarks/README.md` - Benchmark usage
- `.github/workflows/ci.yaml` - CI pipeline
- `pyproject.toml` - Project config, Pyright settings

---

## Next Steps

1. **End-to-end examples** - Training loops demonstrating JAX/NNX usage
2. **Documentation** - API docs, usage examples, JIT best practices

## Known Issues

- None currently tracked for optimizers; flag new regressions in CI.
