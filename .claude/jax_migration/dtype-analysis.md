# Dtype and Precision Analysis for JAX Migration

## Summary
The hyperbolix codebase uses **per-instance dtype management** with sophisticated precision handling patterns. No global torch defaults are used, making migration more predictable.

## Dtype Management Patterns

### 1. Manifold Dtype System
Each manifold class manages its own dtype through constructor parameters:

```python
# Pattern in manifolds/poincare.py:17, hyperboloid.py:17, euclidean.py:15
def __init__(self, dtype: str | torch.dtype = "float32"):
    if dtype == "float32" or dtype == torch.float32:
        self.dtype = torch.float32
        self.min_enorm = 1e-15
        self.max_enorm_eps = 5e-06
    elif dtype == "float64" or dtype == torch.float64:
        self.dtype = torch.float64
        self.min_enorm = 1e-15
        self.max_enorm_eps = 1e-08
```

**Key Features:**
- Accepts both string (`"float32"`) and torch.dtype arguments
- Sets precision-specific numerical tolerances
- Raises ValueError for unsupported dtypes

### 2. Precision Hierarchy Warning System
Critical pattern in `manifolds/poincare.py:34` and `hyperboloid.py:32`:

```python
if torch.finfo(c.dtype).eps < torch.finfo(self.dtype).eps:
    print(f"Warning: self.c.dtype is {c.dtype}, but self.dtype is {self.dtype}. "
          f"All manifold operations will be performed in precision {c.dtype}!")
    self.dtype = c.dtype
```

**Purpose:**
- Ensures curvature parameter `c` doesn't have higher precision than manifold
- Automatically promotes manifold dtype to match curvature precision
- Prevents silent precision loss in hyperbolic geometry calculations

### 3. NN Layer Parameter Dtype System
Pattern across all `nn_layers/` classes using `helpers.py:15`:

```python
# In each layer constructor
from .helpers import get_torch_dtype
self.params_dtype = get_torch_dtype(params_dtype)

# Precision validation in each layer
if torch.finfo(self.params_dtype).eps < torch.finfo(manifold.dtype).eps:
    print(f"Warning: params_dtype is {self.params_dtype}, but manifold dtype is {manifold.dtype}...")
```

**DTYPE_MAP in helpers.py:**
```python
DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}
```

### 4. Epsilon-Based Numerical Stability
Two main patterns for epsilon handling:

#### A. Dynamic Epsilon Selection (`utils/math_utils.py:6-17`)
```python
@torch.jit.script
def _get_tensor_eps(
    x: torch.Tensor,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
```

#### B. Conditional Epsilon in Compute Functions (`helpers.py:85,132`)
```python
eps = torch.finfo(torch.float32).eps if manifold.dtype == torch.float32 else torch.finfo(torch.float64).eps
clamp = clamping_factor * float(math.log(2 / eps))
```

## Precision Constants and Tolerances

### Manifold-Specific Tolerances
| Manifold | dtype | min_enorm | max_enorm_eps |
|----------|-------|-----------|---------------|
| PoincareBall | float32 | 1e-15 | 5e-06 |
| PoincareBall | float64 | 1e-15 | 1e-08 |
| Hyperboloid | float32 | 1e-15 | 5e-06 |
| Hyperboloid | float64 | 1e-15 | 1e-08 |
| Euclidean | float32 | 1e-15 | 5e-06 |
| Euclidean | float64 | 1e-15 | 1e-08 |

### Standard Epsilon Values
```python
# From torch.finfo
torch.float32.eps ≈ 1.19e-07
torch.float64.eps ≈ 2.22e-16
```

## JAX Migration Strategy

### 1. Central Config Approach ✅ IMPLEMENTED
Created `hyperbolix_jax/config.py` with:
- Global dtype management via `config.set_default_dtype()`
- Automatic x64 precision control
- Tolerance configuration
- Device management

### 2. Dtype Conversion Utilities NEEDED
```python
# JAX equivalent to get_torch_dtype
JAX_DTYPE_MAP = {
    "float32": jnp.float32,
    "float64": jnp.float64,
}

def get_jax_dtype(dtype_str: str) -> jnp.dtype:
    # Mirror helpers.py:15 functionality
```

### 3. Epsilon Handling Conversion
```python
# JAX equivalent to torch.finfo
def get_jax_eps(dtype: jnp.dtype) -> float:
    return jnp.finfo(dtype).eps

# Mirror _get_tensor_eps functionality
def _get_array_eps(x: jnp.ndarray) -> float:
    return jnp.finfo(x.dtype).eps
```

### 4. Precision Warning System
Need to implement similar hierarchy checking:
```python
def check_precision_hierarchy(param_dtype: jnp.dtype, manifold_dtype: jnp.dtype):
    if jnp.finfo(param_dtype).eps < jnp.finfo(manifold_dtype).eps:
        warnings.warn(f"Parameter dtype {param_dtype} has higher precision...")
```

## Critical Migration Points

### 1. **No Global State** ✅
- No `torch.set_default_dtype()` usage found
- All precision managed per-instance
- JAX config.py handles global settings

### 2. **String-to-Dtype Conversion** 🔄
- `helpers.py:15` `get_torch_dtype()` needs JAX equivalent
- Used extensively across all nn_layers

### 3. **Precision Validation** 🔄
- Parameter vs manifold dtype checking pattern repeated everywhere
- Need centralized JAX validation function

### 4. **Numerical Stability** 🔄
- Dynamic epsilon selection based on array dtype
- Critical for hyperbolic geometry operations
- JIT-compiled in PyTorch (`@torch.jit.script`)

## Testing Requirements

### Dtype Parity Tests Needed
1. **Precision preservation**: Ensure JAX operations maintain same precision as PyTorch
2. **Epsilon accuracy**: Verify `jnp.finfo().eps` matches `torch.finfo().eps`
3. **Tolerance compliance**: Check numerical operations respect configured tolerances
4. **Warning behavior**: Ensure precision hierarchy warnings fire correctly

### Edge Cases to Test
1. Mixed precision scenarios (float32 manifold, float64 parameters)
2. Curvature parameter precision promotion
3. Epsilon-based clamping with different dtypes
4. JIT compilation of dtype-dependent functions

## Notes for Implementation
- **Priority 1**: Implement JAX dtype utilities before any manifold migration
- **Priority 2**: Create precision validation helpers
- **Priority 3**: Test epsilon-based numerical stability
- **Pattern consistency**: All manifolds and layers follow identical dtype patterns
- **JIT compatibility**: Many functions use `@torch.jit.script` - need JAX JIT equivalents