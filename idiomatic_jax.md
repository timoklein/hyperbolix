# Making Hyperbolix JAX Code More Idiomatic

**Status:** Analysis phase - no implementation yet
**Date:** 2025-10-09
**Context:** Current code is functional, correct, and well-tested. This document explores potential improvements to make the code more JAX-idiomatic.

---

## Overview

The current implementation follows a **NumPy/PyTorch style** with axis-based batch operations. This analysis explores how to make it more **JAX-native** using patterns like `vmap`, while considering tradeoffs between idiomaticity and practicality.

**Key Metrics:**
- 60+ functions with `axis` and `keepdim` parameters
- 23 `jnp.sum(..., axis=axis)` operations
- 14 `jnp.linalg.norm(..., axis=axis)` operations
- 10 dtype-dependent control flow branches
- 8 string-based version selections
- 4 `raise` statements in computational paths

---

## Issue 1: Axis-Based Operations vs. vmap

### Current Pattern (NumPy/PyTorch style)
```python
def dist(x, y, c, axis=-1, keepdim=True, version="mobius_direct"):
    """x, y can be batched: shape (batch, dim) or (dim,)"""
    x2y2 = jnp.sum(x**2, axis=axis, keepdims=True) * jnp.sum(y**2, axis=axis, keepdims=True)
    xy = jnp.sum(x * y, axis=axis, keepdims=True)
    num = jnp.linalg.norm(y - x, axis=axis, keepdims=True)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c**2 * x2y2, MIN_NORM))
    xysum_norm = num / denom
    dist_c = atanh(sqrt_c * xysum_norm)
    res = 2 * dist_c / sqrt_c
    if not keepdim:
        res = jnp.squeeze(res, axis=axis)
    return res
```

**Prevalence:**
- All 60+ manifold operations use this pattern
- Every reduction operation includes axis/keepdim logic
- Adds ~5-10 lines of boilerplate per function

### JAX-Idiomatic Alternative (vmap style)
```python
def dist_single(x, y, c):
    """x, y are single points: shape (dim,)"""
    x2 = jnp.dot(x, x)
    y2 = jnp.dot(y, y)
    xy = jnp.dot(x, y)
    num = jnp.linalg.norm(y - x)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c**2 * x2 * y2, MIN_NORM))
    xysum_norm = num / denom
    dist_c = atanh(sqrt_c * xysum_norm)
    return 2 * dist_c / sqrt_c  # Returns scalar

# Batching via vmap
dist = jax.vmap(dist_single, in_axes=(0, 0, None))
```

### Pros of vmap Approach
- ✅ **Clearer semantics**: Functions describe single-item operations
- ✅ **More composable**: Easy to nest `vmap`, combine with `scan`, etc.
- ✅ **Better performance**: JAX optimizes vmap better than manual axis ops
- ✅ **Simpler signatures**: No axis/keepdim clutter (~40% fewer parameters)
- ✅ **Natural gradients**: `grad(vmap(f))` is more intuitive than `grad(f_with_axis)`
- ✅ **Easier debugging**: Test single-item functions in isolation
- ✅ **More JAX-native**: Matches JAX documentation examples
- ✅ **Eliminates keepdim logic**: Returns natural shapes

### Cons of vmap Approach
- ❌ **Breaking API change**: All user code would need updates
- ❌ **Learning curve**: Users must understand vmap
- ❌ **Less familiar**: NumPy/PyTorch users expect axis parameters
- ❌ **Backward incompatibility**: Can't drop-in replace PyTorch version
- ❌ **Migration effort**: ~60 functions × 2 hours = ~120 hours refactoring
- ❌ **Two-tier API**: Would need both single-item and batched versions
- ❌ **Documentation burden**: Must teach vmap to all users

---

## Issue 2: Manual keepdim Logic

### Current Pattern
```python
def tangent_norm(v, x, c, axis=-1, keepdim=True):
    lambda_x = _conformal_factor(x, c, axis=axis)
    res = lambda_x * jnp.linalg.norm(v, axis=axis, keepdims=True)
    if not keepdim:
        res = jnp.squeeze(res, axis=axis if axis >= 0 else v.ndim + axis)
    return res
```

**Prevalence:**
- ~20 functions with keepdim conditional logic
- Axis normalization needed for negative indices
- 3-5 lines of boilerplate per function

### Problem
- Increases code complexity
- Easy to introduce shape bugs
- Duplicates NumPy behavior that vmap eliminates naturally

### Solution with vmap
```python
def tangent_norm_single(v, x, c):
    lambda_x = _conformal_factor_single(x, c)
    return lambda_x * jnp.linalg.norm(v)  # Returns scalar, no keepdim needed
```

---

## Issue 3: Dtype-Dependent Control Flow

### Current Pattern (10 locations)
```python
def _get_max_norm_eps(x):
    if x.dtype == jnp.float32:
        return MAX_NORM_EPS_F32
    elif x.dtype == jnp.float64:
        return MAX_NORM_EPS_F64
    else:
        return MAX_NORM_EPS_F32

def cosh(x):
    clamp = 88.0 if x.dtype == jnp.float32 else 709.0
    x = smooth_clamp(x, -clamp, clamp)
    return jnp.cosh(x)
```

**Files affected:**
- `math_utils.py`: 3 functions
- `poincare.py`: 1 function
- `helpers.py`: 3 functions

### Problem
Works fine if dtype is known at trace time (typical case), but:
- Not idiomatic JAX (should avoid Python conditionals on traced values)
- Could fail in edge cases with dynamic dtype changes
- Less clear to readers unfamiliar with JAX tracing

### JAX-Idiomatic Alternatives

**Option A: Use jax.lax.cond (JIT-safe but verbose)**
```python
def _get_max_norm_eps(x):
    return jax.lax.cond(
        x.dtype == jnp.float32,
        lambda: MAX_NORM_EPS_F32,
        lambda: MAX_NORM_EPS_F64
    )
```

**Option B: Rely on JAX's dtype promotion (simpler)**
```python
def _get_max_norm_eps(x):
    return jnp.finfo(x.dtype).eps * 1e3  # Appropriate scaling
```

**Option C: Static type selection (best for JIT)**
```python
@partial(jax.jit, static_argnames=['dtype'])
def cosh(x, dtype=jnp.float32):
    clamp = {jnp.float32: 88.0, jnp.float64: 709.0}[dtype]
    x = smooth_clamp(x, -clamp, clamp)
    return jnp.cosh(x)
```

**Recommendation:** Option B where possible (simplest), Option C for functions where dtype matters for control flow.

---

## Issue 4: String-Based Version Selection

### Current Pattern (8 locations)
```python
def dist(x, y, c, axis=-1, keepdim=True, version="mobius_direct"):
    if version in ["mobius_direct", "default"]:
        # ... implementation A (30 lines)
    elif version == "mobius":
        # ... implementation B (20 lines)
    elif version == "metric_tensor":
        # ... implementation C (25 lines)
    elif version == "lorentzian_proxy":
        # ... implementation D (15 lines)
    else:
        raise ValueError(f"Unknown version: {version}")
```

**Locations:**
- `poincare.dist()`: 4 versions
- `poincare.dist_0()`: 4 versions
- `hyperboloid.dist()`: 2 versions
- `hyperboloid.dist_0()`: 2 versions

### Problem
- 90+ lines of branching code in single functions
- Hard to test individual implementations
- String argument must be static for JIT (already documented)
- Violates single responsibility principle

### JAX-Idiomatic Alternatives

**Option A: Separate functions (most explicit, recommended)**
```python
def dist_mobius_direct(x, y, c):
    """Direct Möbius distance formula (fastest)."""
    sqrt_c = jnp.sqrt(c)
    # ... 10-15 lines

def dist_mobius(x, y, c):
    """Möbius distance via addition (most accurate near boundary)."""
    # ... 10-15 lines

def dist_metric_tensor(x, y, c):
    """Metric tensor induced distance (geometric interpretation)."""
    # ... 10-15 lines

# Default to most common
dist = dist_mobius_direct
```

**Benefits:**
- Easier to test individual implementations
- Clear documentation of when to use which
- Can JIT-compile separately
- Users can import specific version they need

**Option B: Function registry (flexible)**
```python
_DIST_IMPLEMENTATIONS = {
    "mobius_direct": dist_mobius_direct,
    "mobius": dist_mobius,
    "metric_tensor": dist_metric_tensor,
    "lorentzian_proxy": dist_lorentzian_proxy,
}

def dist(x, y, c, version="mobius_direct"):
    """Dispatch to version-specific implementation."""
    return _DIST_IMPLEMENTATIONS[version](x, y, c)
```

**Option C: jax.lax.switch (JIT-optimized but complex)**
```python
def dist(x, y, c, version_idx=0):
    """version_idx: 0=mobius_direct, 1=mobius, 2=metric_tensor, 3=lorentzian_proxy"""
    return jax.lax.switch(
        version_idx,
        [dist_mobius_direct, dist_mobius, dist_metric_tensor, dist_lorentzian_proxy],
        x, y, c
    )
```

**Recommendation:** Option A for clarity. Keep Option B wrapper for backward compatibility.

---

## Issue 5: Error Handling with raise

### Current Pattern (4 locations)
```python
# In poincare.py:216, 254
if version not in valid_versions:
    raise ValueError(f"Unknown version: {version}")

# In math_utils.py:18
if x.dtype not in [jnp.float32, jnp.float64]:
    raise RuntimeError(f"Expected floating-point, got {x.dtype}")

# In helpers.py:71
if dtype_str not in DTYPE_MAP:
    raise ValueError(f"Unsupported dtype: {dtype_str}")
```

### Problem
`raise` statements force tracer to evaluate values during JIT compilation. This:
- Breaks JIT compilation if condition depends on traced values
- Currently works because version/dtype are static, but fragile
- Not idiomatic JAX error handling

### JAX-Idiomatic Alternatives

**Option A: Use checkify (JAX 0.4.13+, recommended for runtime checks)**
```python
from jax.experimental import checkify

@checkify.checkify
def dist(x, y, c, version="mobius_direct"):
    checkify.check(version in VALID_VERSIONS, "Invalid version: {}", version)
    # ... rest of function
```

**Option B: Remove runtime checks (trust static_argnames)**
```python
# No runtime check - version is static anyway via static_argnames
def dist(x, y, c, version="mobius_direct"):
    # Assume version is valid (enforced at JIT compilation time)
    if version in ["mobius_direct", "default"]:
        # ...
```

**Option C: Validation layer (separate from computation, recommended)**
```python
def validate_dist_inputs(x, y, c, version):
    """Non-JIT validation for user input."""
    if version not in VALID_VERSIONS:
        raise ValueError(f"Invalid version: {version}. Choose from {VALID_VERSIONS}")
    if not jnp.isfinite(c) or c <= 0:
        raise ValueError(f"Curvature must be positive finite, got {c}")
    # ... other checks

# User-facing API with validation
def dist_safe(x, y, c, version="mobius_direct"):
    validate_dist_inputs(x, y, c, version)
    return _dist_jit(x, y, c, version)  # JIT-compilable core

# Direct JIT-compiled version (no validation overhead)
_dist_jit = jax.jit(_dist_impl, static_argnames=['version'])
```

**Recommendation:** Option C - separate validation from computation. Provide both safe (validated) and fast (direct) APIs.

---

## Issue 6: Assertions in NN Layers

### Current Pattern (17 assertions)
```python
# In poincare_linear.py, poincare_regression.py, etc.
def __call__(self, x, c=1.0, axis=-1, backproject=True):
    assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
    assert axis == -1, "axis must be -1, reshape your tensor accordingly."
    assert x.shape[axis] == self.in_dim, f"Expected input dim {self.in_dim}, got {x.shape[axis]}"
    # ... rest of method
```

**Locations:**
- `poincare_linear.py`: 4 assertions
- `poincare_regression.py`: 4 assertions
- `poincare_rl.py`: 3 assertions
- `hyperboloid_linear.py`: 6 assertions

### Problem
- Assertions fail during JIT tracing if values aren't statically known
- Python `assert` is removed with `-O` optimization flag
- Not idiomatic for JAX (should use checkify or move to `__init__`)

### JAX-Idiomatic Alternatives

**Option A: Move checks to `__init__` (best for config validation)**
```python
class HypLinearPoincare(nnx.Module):
    def __init__(self, manifold_module, in_dim, out_dim, *, rngs, input_space="tangent"):
        # Validate at construction time
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be 'tangent' or 'manifold', got {input_space}")

        self.manifold = manifold_module
        self.input_space = input_space
        # ... rest of init

    def __call__(self, x, c=1.0, axis=-1, backproject=True):
        # No assertions needed - validated at construction
        # axis can be static_argnames if JIT-compiled
```

**Option B: Use checkify for runtime checks**
```python
@checkify.checkify
def __call__(self, x, c=1.0, axis=-1, backproject=True):
    checkify.check(x.shape[-1] == self.in_dim, "Shape mismatch: {} != {}", x.shape[-1], self.in_dim)
    # ...
```

**Option C: Remove assertions, document preconditions**
```python
def __call__(self, x, c=1.0, axis=-1, backproject=True):
    """Apply hyperbolic linear transformation.

    Args:
        x: Input tensor with x.shape[-1] == self.in_dim
        axis: Must be -1 for this implementation

    Note: Preconditions are not checked for performance. Use in debug mode if needed.
    """
    # ... implementation without checks
```

**Recommendation:** Option A for config checks (input_space), Option C for shape checks in production code (with Option B available via debug flag).

---

## Issue 7: Helper Functions and Global Constants

### Current Pattern
```python
# Module-level globals
MIN_NORM = 1e-15
MAX_NORM_EPS_F32 = 5e-06
MAX_NORM_EPS_F64 = 1e-08

def _conformal_factor(x, c, axis=-1):
    """Helper function using global constants."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=True)
    max_norm_eps = _get_max_norm_eps(x)  # Reads globals
    denom = jnp.maximum(
        1.0 - c * x2,
        2 * jnp.sqrt(c) * max_norm_eps - c * max_norm_eps**2
    )
    return 2.0 / denom
```

### Consideration
This is actually **fine** for numerical constants. JAX traces treat module-level constants correctly.

### Alternative (if desired for explicitness)

**Option A: Configuration pytree**
```python
from dataclasses import dataclass

@dataclass
class NumericalConfig:
    """Numerical stability parameters."""
    min_norm: float = 1e-15
    max_norm_eps_f32: float = 5e-6
    max_norm_eps_f64: float = 1e-8

def _conformal_factor(x, c, config: NumericalConfig, axis=-1):
    # Makes dependencies explicit, easier to tune per-application
    max_norm_eps = config.max_norm_eps_f32 if x.dtype == jnp.float32 else config.max_norm_eps_f64
    # ...
```

**Option B: Pass constants as kwargs (too verbose)**
```python
def _conformal_factor(x, c, axis=-1, min_norm=1e-15):
    # Too many parameters, not worth it
```

**Recommendation:** Keep current approach. Global constants for numerical stability are fine and conventional.

---

## Issue 8: Unnecessary Intermediate Variables

### Example
```python
def dist(x, y, c, axis=-1, keepdim=True, version="mobius_direct"):
    sqrt_c = jnp.sqrt(c)
    x2y2 = jnp.sum(x**2, axis=axis, keepdims=True) * jnp.sum(y**2, axis=axis, keepdims=True)
    xy = jnp.sum(x * y, axis=axis, keepdims=True)
    num = jnp.linalg.norm(y - x, axis=axis, keepdims=True)
    denom = jnp.sqrt(jnp.maximum(1 - 2 * c * xy + c**2 * x2y2, MIN_NORM))
    xysum_norm = num / denom
    dist_c = atanh(sqrt_c * xysum_norm)
    res = 2 * dist_c / sqrt_c
    return res
```

### Assessment: **This is Good!**

Intermediate variables:
- ✅ Improve readability dramatically
- ✅ Match mathematical notation in papers
- ✅ Make debugging easier (can print intermediate values)
- ✅ JAX optimizes these away (no performance penalty)
- ✅ Align with CLAUDE.md principle: "Prefer readable code over clever code"

**Recommendation:** Keep current style. This is idiomatic *Python*, and JAX handles it perfectly.

---

## Proposed Strategies

### Strategy A: Evolutionary (Low Risk, Low Disruption) ⭐ **RECOMMENDED**

**Approach:**
1. Keep current axis-based API as primary interface (backward compatibility)
2. Add supplementary `single.py` modules with vmap-friendly versions
3. Document vmap patterns prominently in examples
4. Selectively refactor internal implementations where vmap simplifies code

**Timeline:** 2-3 weeks
- Week 1: Create `manifolds/*/single.py` modules
- Week 2: Add documentation and examples
- Week 3: Testing and performance benchmarks

**Pros:**
- ✅ Non-breaking changes
- ✅ Users choose their preferred style
- ✅ Gradual adoption path
- ✅ Maintains backward compatibility
- ✅ Low risk

**Cons:**
- ⚠️ Code duplication (2 implementations)
- ⚠️ Two APIs to maintain
- ⚠️ Doesn't fully commit to JAX idioms

**Example structure:**
```
manifolds/
├── poincare.py              # Current axis-based API
├── poincare_single.py       # New vmap-native API
├── hyperboloid.py           # Current axis-based API
├── hyperboloid_single.py    # New vmap-native API
└── jit_wrappers.py          # Wrappers for both styles
```

---

### Strategy B: Revolutionary (High Risk, High Reward)

**Approach:**
1. Refactor all functions to single-item operations
2. Provide convenience vmap wrappers for batching
3. Breaking change requiring major version bump (2.0.0)
4. Comprehensive migration guide

**Timeline:** 6-8 weeks
- Week 1-2: Refactor euclidean + poincare
- Week 3-4: Refactor hyperboloid
- Week 5-6: Update all tests and nn_layers
- Week 7-8: Documentation, migration guide, examples

**Pros:**
- ✅ Clean, idiomatic JAX code
- ✅ Better long-term maintenance
- ✅ Smaller function signatures
- ✅ Natural vmap composition
- ✅ Easier to extend

**Cons:**
- ❌ Breaks all existing code
- ❌ Significant effort (120+ hours)
- ❌ User friction and migration pain
- ❌ Need to maintain 1.x branch
- ❌ Documentation overhead

---

### Strategy C: Hybrid (Balanced)

**Approach:**
1. Keep current public API unchanged
2. Refactor internal implementations to use vmap where beneficial
3. Expose both APIs through different import paths

**Timeline:** 3-4 weeks
- Week 1: Internal refactoring of helpers
- Week 2-3: Test both paths thoroughly
- Week 4: Documentation

**Pros:**
- ✅ Internal benefits of vmap
- ✅ Maintains external compatibility
- ✅ Performance improvements

**Cons:**
- ⚠️ Complex internal implementation
- ⚠️ Need to translate axis → vmap
- ⚠️ Doesn't fully solve the problem

---

### Strategy D: Incremental Improvements (Minimal Risk) ⭐ **ALSO RECOMMENDED**

**Approach:**
1. Keep current API completely unchanged
2. Apply selective improvements from Issues 3-7:
   - Remove `raise` statements → validation layer
   - Refactor version strings → separate functions
   - Move assertions to `__init__`
   - Improve dtype handling

**Timeline:** 1 week
- Day 1-2: Issue 5 (error handling)
- Day 3-4: Issue 4 (version selection)
- Day 5: Issue 6 (assertions)

**Pros:**
- ✅ Immediate improvements
- ✅ Zero breaking changes
- ✅ Minimal effort (40 hours)
- ✅ Low risk

**Cons:**
- ⚠️ Doesn't address main vmap issue
- ⚠️ Still axis-based architecture

---

## Concrete Recommendations

### Short Term (This Week)
**Strategy D: Incremental Improvements**

1. **Refactor version selection** (Issue 4)
   - Split `dist()` into `dist_mobius_direct()`, `dist_mobius()`, etc.
   - Keep wrapper for backward compatibility
   - **Impact:** Cleaner code, easier testing
   - **Effort:** 8 hours

2. **Improve error handling** (Issue 5)
   - Move `raise` to validation layer
   - Provide both validated and fast APIs
   - **Impact:** Better JIT compatibility
   - **Effort:** 4 hours

3. **Move assertions to __init__** (Issue 6)
   - Validate config at construction time
   - Remove runtime assertions
   - **Impact:** JIT-safe, faster execution
   - **Effort:** 6 hours

**Total: 18 hours, zero breaking changes**

---

### Medium Term (Next Month)
**Strategy A: Add vmap-native alternatives**

4. **Create `*_single.py` modules** (Issue 1, 2)
   - `poincare_single.py`, `hyperboloid_single.py`, `euclidean_single.py`
   - Clean single-item implementations
   - Provide vmap wrappers
   - **Impact:** JAX-idiomatic option available
   - **Effort:** 60 hours

5. **Update documentation**
   - Add vmap usage examples
   - Performance comparison
   - Migration guide
   - **Impact:** Better user experience
   - **Effort:** 20 hours

**Total: 80 hours, backward compatible**

---

### Long Term (Consider for 2.0)
**Strategy B: Full refactor** (only if user feedback demands it)

6. **Deprecate axis-based API**
   - Announce deprecation in 1.x
   - Provide automated migration tools
   - Release 2.0 with vmap-native API

**Decision point:** Wait for user feedback on `*_single.py` modules first.

---

## Decision Criteria

### Choose Strategy D (Incremental) if:
- ✅ Backward compatibility is critical
- ✅ Users are happy with current API
- ✅ Limited development time available
- ✅ Low risk tolerance

### Choose Strategy A (Evolutionary) if:
- ✅ Want JAX idioms available as option
- ✅ Can invest 3-4 weeks
- ✅ Want to test user adoption before committing
- ✅ Willing to maintain dual APIs temporarily

### Choose Strategy B (Revolutionary) if:
- ✅ Ready for major version bump
- ✅ User base is small/controlled
- ✅ Can invest 2+ months
- ✅ Want clean slate for long-term

---

## Open Questions

1. **API stability:** How important is backward compatibility? Are breaking changes acceptable?

2. **User base:**
   - Are users expected to know vmap?
   - Are they NumPy/PyTorch converts or JAX natives?
   - How many downstream projects would break?

3. **Scope:** Should this be:
   - Quick improvements (Strategy D)?
   - Gradual migration (Strategy A)?
   - Complete rewrite (Strategy B)?

4. **Timeline:**
   - Quick wins needed (1 week)?
   - Can invest a month?
   - Long-term vision (3+ months)?

5. **Performance:**
   - Are current axis operations a bottleneck?
   - Would vmap provide meaningful speedups?
   - Have we benchmarked?

---

## Next Steps

**Recommended path:**

1. **Immediate** (this week): Implement Strategy D incremental improvements
   - Low risk, immediate benefit
   - Test ground for more changes

2. **Short term** (next month): Prototype Strategy A with one manifold
   - Create `poincare_single.py` as proof of concept
   - Benchmark performance difference
   - Gather user feedback

3. **Medium term** (2-3 months): Based on feedback:
   - If positive → complete Strategy A for all manifolds
   - If negative → stop at Strategy D improvements
   - If very positive → consider Strategy B for 2.0

4. **Long term**: Let user adoption guide decision on 2.0 API

---

## References

- JAX docs on vmap: https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html
- JAX transformation composability: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html
- Checkify docs: https://jax.readthedocs.io/en/latest/debugging/checkify.html
- Project CLAUDE.md: Prefer readable code over clever code

---

**Author:** Claude
**Review Status:** Pending human review
**Implementation Status:** Analysis only - no code changes made
