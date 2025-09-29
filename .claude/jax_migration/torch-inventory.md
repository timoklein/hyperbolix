# Torch Usage Inventory for JAX Migration

## Summary Statistics
- **Total files with torch imports**: 18
- **Total nn.Module subclasses**: 14
- **Primary modules**: manifolds (4), nn_layers (9), optim (2), utils (4)

## Torch Imports by Module

### src/manifolds/
| File | Torch Usage | Migration Complexity |
|------|-------------|---------------------|
| `manifold.py:1` | `import torch` + `torch.nn.Module` base class | HIGH - Core interface |
| `euclidean.py:1` | `import torch` + dtype handling | MEDIUM |
| `hyperboloid.py:1` | `import torch` + dtype + `torch.finfo` precision warnings | MEDIUM |
| `poincare.py:1` | `import torch` + dtype + `torch.finfo` precision warnings | MEDIUM |

### src/nn_layers/
| File | Torch Usage | Migration Complexity |
|------|-------------|---------------------|
| `helpers.py:2` | `import torch` + dtype utilities (`get_torch_dtype`, `DTYPE_MAP`) | HIGH - Shared utilities |
| `hyperbolic_standard_layers.py:1` | `import torch` + 8 `torch.nn.Module` subclasses | HIGH - Core layers |
| `hyperboloid_linear_layers.py:1` | `import torch` + 3 `torch.nn.Module` subclasses + `torch.finfo` | HIGH |
| `hyperboloid_regression_layers.py:1` | `import torch` + 1 `torch.nn.Module` subclass + `torch.finfo` | MEDIUM |
| `poincare_linear_layers.py:1` | `import torch` + 2 `torch.nn.Module` subclasses + `torch.finfo` | HIGH |
| `poincare_regression_layers.py:2` | `import torch` + 2 `torch.nn.Module` subclasses + `torch.finfo` | MEDIUM |
| `poincare_rl_layers.py:1` | `import torch` + 1 `torch.nn.Module` subclass + `torch.finfo` | MEDIUM |

### src/optim/
| File | Torch Usage | Migration Complexity |
|------|-------------|---------------------|
| `riemannian_sgd.py:1` | `import torch` (tensor math, no nn.Module) | MEDIUM - Pure tensor ops |
| `riemannian_adam.py:1` | `import torch` (tensor math, no nn.Module) | MEDIUM - Pure tensor ops |

### src/utils/
| File | Torch Usage | Migration Complexity |
|------|-------------|---------------------|
| `helpers.py:1` | `import torch` + `torch.finfo` for eps handling | LOW - Utilities |
| `math_utils.py:3` | `import torch` + `torch.finfo` for eps constants | LOW - Math utilities |
| `vis_utils.py:2` | `import torch` for visualization | LOW - Optional features |
| `horo_pca.py:1,2` | `import torch` + `torch.nn` + `torch.linalg.qr` | MEDIUM - Linear algebra |

## nn.Module Subclasses Inventory

### Core Manifold Classes (1)
1. `manifolds/manifold.py:7` - `Manifold(torch.nn.Module)` - **Base class for all manifolds**

### Standard Hyperbolic Layers (8)
1. `nn_layers/hyperbolic_standard_layers.py:6` - `Expmap(torch.nn.Module)`
2. `nn_layers/hyperbolic_standard_layers.py:28` - `Expmap_0(torch.nn.Module)`
3. `nn_layers/hyperbolic_standard_layers.py:50` - `Retraction(torch.nn.Module)`
4. `nn_layers/hyperbolic_standard_layers.py:72` - `Logmap(torch.nn.Module)`
5. `nn_layers/hyperbolic_standard_layers.py:94` - `Logmap_0(torch.nn.Module)`
6. `nn_layers/hyperbolic_standard_layers.py:116` - `Proj(torch.nn.Module)`
7. `nn_layers/hyperbolic_standard_layers.py:135` - `TanProj(torch.nn.Module)`
8. `nn_layers/hyperbolic_standard_layers.py:155` - `HyperbolicActivation(torch.nn.Module)`

### Hyperboloid Layers (3)
1. `nn_layers/hyperboloid_linear_layers.py:9` - `HyperbolicLinearHyperboloid(torch.nn.Module)`
2. `nn_layers/hyperboloid_linear_layers.py:105` - `HyperbolicLinearHyperboloidFHNN(torch.nn.Module)`
3. `nn_layers/hyperboloid_linear_layers.py:224` - `HyperbolicLinearHyperboloidFHCNN(torch.nn.Module)`

### Hyperboloid Regression (1)
1. `nn_layers/hyperboloid_regression_layers.py:7` - `HyperbolicRegressionHyperboloid(torch.nn.Module)`

### Poincaré Layers (2)
1. `nn_layers/poincare_linear_layers.py:8` - `HyperbolicLinearPoincare(torch.nn.Module)`
2. `nn_layers/poincare_linear_layers.py:92` - `HyperbolicLinearPoincarePP(torch.nn.Module)`

### Poincaré Regression (2)
1. `nn_layers/poincare_regression_layers.py:9` - `HyperbolicRegressionPoincare(torch.nn.Module)`
2. `nn_layers/poincare_regression_layers.py:138` - `HyperbolicRegressionPoincarePP(torch.nn.Module)`

### Poincaré RL (1)
1. `nn_layers/poincare_rl_layers.py:8` - `HyperbolicRegressionPoincareHDRL(torch.nn.Module)`

## Dtype and Precision Handling

### Global Settings
- **No `torch.set_default_dtype` usage found** - precision handled per-instance
- **No global torch configuration** - dtype resolved through individual classes

### Dtype Utilities
- `nn_layers/helpers.py:10-25` - `DTYPE_MAP` and `get_torch_dtype()` function
- Maps string dtype names to `torch.dtype` objects
- Used extensively across nn_layers for parameter initialization

### Precision Warnings Pattern
Found in multiple files using `torch.finfo(dtype).eps` comparisons:
- `manifolds/poincare.py:34` - curvature vs manifold dtype precision
- `manifolds/hyperboloid.py:32` - curvature vs manifold dtype precision
- All nn_layers classes check `params_dtype` vs `manifold.dtype` precision
- `utils/math_utils.py:9-10` - eps constants for float32/float64

### Epsilon Handling
- `torch.finfo(torch.float32).eps` / `torch.finfo(torch.float64).eps` used for numerical stability
- Pattern: conditional epsilon selection based on manifold dtype
- Critical for hyperbolic geometry numerical stability

## Third-Party Dependencies
- **No GeoOpt patterns detected** in current usage
- **Pure PyTorch implementation** - no external hyperbolic libraries
- **Linear algebra**: `torch.linalg.qr` in `utils/horo_pca.py`

## Migration Priority Assessment

### Phase 2 - Critical (Manifolds)
- `Manifold` base class - **BLOCKER** for all other work
- Precision/dtype handling in curved manifolds

### Phase 3 - High Priority (Core Layers)
- `helpers.py` dtype utilities - **SHARED DEPENDENCY**
- Standard hyperbolic layers (Expmap, Logmap, Proj, etc.)
- Linear layers (both Hyperboloid and Poincaré variants)

### Phase 4 - Medium Priority (Optimizers)
- Riemannian SGD/Adam - pure tensor operations
- No nn.Module dependencies - easier to migrate

### Phase 5 - Lower Priority
- Regression layers
- RL layers
- Visualization utilities
- PCA utilities

## Notes
- **Consistent patterns**: Most classes follow similar dtype/precision warning patterns
- **Modular structure**: Clear separation between manifolds, layers, and optimizers
- **No stateful globals**: All precision handling is instance-based
- **Testing surface**: All nn.Module classes will need JAX dataclass equivalents