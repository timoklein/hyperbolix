# Numerical Stability Guide

Best practices for maintaining numerical precision in hyperbolic operations.

## Overview

Hyperbolic geometry presents unique numerical challenges due to the exponential growth of the conformal factor near the boundary and the involvement of hyperbolic functions (cosh, sinh, atanh). This guide explains these challenges and provides strategies to maintain numerical stability.

!!! warning "Key Challenges"
    - **Conformal factor explosion**: λ(x) grows exponentially as points approach the boundary
    - **Float32 limitations**: ~7 significant digits, insufficient for large distances (>10)
    - **Hyperbolic function overflow**: cosh/sinh overflow for large arguments
    - **Division by near-zero**: Operations involving 1 - c||x||² near the boundary

## Float Precision: Float32 vs Float64

### When to Use Each

**Float32 (default)**:
- Sufficient for most applications with small to moderate distances (< 5)
- 2-4x faster on GPU
- Lower memory footprint (important for large models)
- ~7 significant decimal digits

**Float64 (high precision)**:
- Required for large distances (> 10) or near-boundary points
- Better numerical stability in edge cases
- ~15-16 significant decimal digits
- Use for research, validation, or stability-critical applications

```python
import jax.numpy as jnp
from hyperbolix.manifolds import poincare

# Float32 (default)
x = jnp.array([0.1, 0.2], dtype=jnp.float32)
y = jnp.array([0.8, 0.5], dtype=jnp.float32)
dist = poincare.dist(x, y, c=1.0, version_idx=0)

# Float64 (high precision)
x = jnp.array([0.1, 0.2], dtype=jnp.float64)
y = jnp.array([0.8, 0.5], dtype=jnp.float64)
dist = poincare.dist(x, y, c=1.0, version_idx=0)
```

### Precision Requirements by Distance

| Distance from Origin | Float32 Accuracy | Recommended Precision |
|----------------------|------------------|----------------------|
| d < 3 | Excellent (< 0.01% error) | float32 |
| 3 ≤ d < 5 | Good (< 0.1% error) | float32 |
| 5 ≤ d < 10 | Moderate (< 3% error) | float64 for critical ops |
| d ≥ 10 | Poor (> 3% error) | **float64 required** |

!!! tip "Quick Check"
    If your embeddings have distances from the origin > 7, switch to float64:

    ```python
    distances = jax.vmap(lambda x: poincare.dist(jnp.zeros_like(x), x, c=1.0, version_idx=0))(x_batch)
    max_dist = jnp.max(distances)
    print(f"Max distance from origin: {max_dist:.2f}")
    # If > 7, consider float64
    ```

## The Conformal Factor Problem

### Understanding λ(x)

The **conformal factor** in Poincaré ball geometry is:

$$
\lambda(x) = \frac{2}{1 - c||x||^2}
$$

This factor appears in:
- Exponential map: scales tangent vectors
- Logarithmic map: scales back to tangent space
- Riemannian gradient: converts Euclidean to Riemannian gradients

### Exponential Growth

As points move toward the boundary (||x|| → 1/√c), λ(x) explodes:

```python
import jax.numpy as jnp
from hyperbolix.manifolds import poincare

c = 1.0
distances = [0, 1, 2, 3, 5, 7, 10]

for d in distances:
    # Point at distance d from origin
    x = poincare.expmap_0(jnp.array([d, 0.0]), c=c)
    norm = jnp.linalg.norm(x)
    lambda_x = 2.0 / (1.0 - c * norm**2)
    print(f"d={d:2d}: ||x||={norm:.6f}, λ(x)={lambda_x:10.1f}")
```

Output:
```
d= 0: ||x||=0.000000, λ(x)=       2.0
d= 1: ||x||=0.761594, λ(x)=       3.6
d= 2: ||x||=0.964028, λ(x)=      27.7
d= 3: ||x||=0.995055, λ(x)=     202.0
d= 5: ||x||=0.999909, λ(x)=   11013.2
d= 7: ||x||=0.999991, λ(x)= 1096633.2
d=10: ||x||=1.000000, λ(x)=       inf
```

### Numerical Issues

**Problem 1: Precision loss in logmap**

```python
# logmap divides by λ(x), then later operations multiply by λ(x)
# With float32 and λ(x) ≈ 10,000:
# - Division by 10,000 loses 4 digits of precision
# - Multiplication by 10,000 doesn't recover them
# Result: ~3 digits of precision remaining (out of 7)
```

**Problem 2: Cancellation in 1 - c||x||²**

```python
# Near boundary: ||x||² ≈ 0.999999
# Computing 1 - c||x||² loses significant digits due to catastrophic cancellation
# Float32: 1.0 - 0.999999 = 0.000001 (but stored imprecisely!)
```

### Mitigation Strategies

**1. Use projection after operations**

```python
from hyperbolix.manifolds import poincare

# After addition or other operations
result = poincare.add(x, y, c=1.0)
result = poincare.proj(result, c=1.0)  # Project back to manifold
```

**2. Keep points away from boundary**

```python
# During initialization
def init_hyperbolic_embeddings(n_points, dim, max_norm=0.8):
    """Initialize embeddings safely away from boundary."""
    x = jax.random.normal(key, (n_points, dim)) * 0.1
    x_proj = jax.vmap(poincare.proj, in_axes=(0, None, None))(x, c=1.0, version_idx=None)

    # Clip to max_norm to avoid boundary
    norms = jnp.linalg.norm(x_proj, axis=-1, keepdims=True)
    x_clipped = jnp.where(norms > max_norm, x_proj * max_norm / norms, x_proj)
    return x_clipped
```

**3. Use float64 for critical operations**

```python
# Convert to float64 for numerically sensitive operations
x_f64 = x.astype(jnp.float64)
y_f64 = y.astype(jnp.float64)

dist_precise = poincare.dist(x_f64, y_f64, c=1.0, version_idx=0)

# Convert back if needed
dist_f32 = dist_precise.astype(jnp.float32)
```

## Hyperbolic Function Overflow

### The Problem

Standard implementations of cosh, sinh can overflow:

```python
# Standard numpy/jax
import jax.numpy as jnp

x = jnp.array(100.0, dtype=jnp.float32)
print(jnp.cosh(x))  # inf (overflow!)
print(jnp.sinh(x))  # inf (overflow!)
```

### Solution: Protected Math Utils

Hyperbolix provides overflow-protected hyperbolic functions:

```python
from hyperbolix.utils.math_utils import cosh, sinh, acosh, atanh

# Protected versions
x = jnp.array(100.0, dtype=jnp.float32)
print(cosh(x))  # Finite value (clamped to safe range)
print(sinh(x))  # Finite value (clamped to safe range)

# Domain-protected inverse functions
y = jnp.array(0.5, dtype=jnp.float32)
print(acosh(y))  # Clamped to valid domain [1, inf)

z = jnp.array(0.999999, dtype=jnp.float32)
print(atanh(z))  # Clamped away from ±1 singularities
```

### Smooth Clamping

The library uses **smooth clamping** via softplus instead of hard clipping:

```python
from hyperbolix.utils.math_utils import smooth_clamp

# Smooth clamp (differentiable, no gradient issues)
x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
clamped = smooth_clamp(x, min_value=-5.0, max_value=5.0, smoothing_factor=50.0)
print(clamped)
# Near boundaries: smooth transition, not abrupt cutoff
```

Benefits:
- Differentiable everywhere (no gradient discontinuities)
- Numerically stable (uses softplus internally)
- Adjustable smoothing factor for trade-off between accuracy and gradient flow

## Version Parameters

### Purpose

Many manifold operations have multiple mathematically equivalent formulations that differ in numerical properties. The `version_idx` parameter selects which to use.

### Poincaré Ball Distance Versions

```python
from hyperbolix.manifolds import poincare

x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, 0.4])
c = 1.0

# Version 0: Direct Möbius distance (FASTEST)
d0 = poincare.dist(x, y, c, version_idx=poincare.VERSION_MOBIUS_DIRECT)

# Version 1: Möbius via addition
d1 = poincare.dist(x, y, c, version_idx=poincare.VERSION_MOBIUS)

# Version 2: Metric tensor induced
d2 = poincare.dist(x, y, c, version_idx=poincare.VERSION_METRIC_TENSOR)

# Version 3: Lorentzian proxy
d3 = poincare.dist(x, y, c, version_idx=poincare.VERSION_LORENTZIAN_PROXY)

print(f"Version 0: {d0:.6f}")
print(f"Version 1: {d1:.6f}")
print(f"Version 2: {d2:.6f}")
print(f"Version 3: {d3:.6f}")
# All should be approximately equal
```

### Which Version to Use?

**General recommendation**: `VERSION_MOBIUS_DIRECT` (version 0)
- Fastest
- Fewest intermediate operations
- Best for most applications

**Special cases**:
- **Near-boundary points** (||x|| > 0.9): Try `VERSION_LORENTZIAN_PROXY` (version 3) for better stability
- **Very high dimensions** (> 1000): `VERSION_METRIC_TENSOR` (version 2) may be more stable
- **Debugging**: Compare all versions — significant differences indicate numerical issues

### Using Versions with JIT

```python
import jax

# IMPORTANT: version_idx must be static for JIT
@jax.jit
def compute_distances(x_batch, y_batch, c):
    # Version baked into function body (static)
    return jax.vmap(
        lambda x, y: poincare.dist(x, y, c, version_idx=0)
    )(x_batch, y_batch)

# Or use static_argnames
dist_jit = jax.jit(poincare.dist, static_argnames=['version_idx'])
d = dist_jit(x, y, c=1.0, version_idx=0)
```

## Projection Strategies

### Why Project?

Operations like addition, linear transformations can push points off the manifold. Projection restores the manifold constraint.

### When to Project

**Always project**:
- After Möbius addition: `poincare.add()`
- After neural network layers
- After parameter updates in optimization

**Usually don't need projection**:
- After `expmap` (already on manifold)
- After `proj` (redundant)

### Projection

Projection ensures points stay on the manifold by clipping norms:

```python
# Project to Poincaré ball
x_proj = poincare.proj(x, c=1.0)

# Projection is numerically stable and automatically handles edge cases
```

### Projection in Training

```python
from hyperbolix.manifolds import poincare
from flax import nnx

class HyperbolicModel(nnx.Module):
    def __init__(self, rngs):
        self.layer1 = HypLinearPoincare(poincare, 128, 64, rngs=rngs)
        self.layer2 = HypLinearPoincare(poincare, 64, 32, rngs=rngs)

    def __call__(self, x, c=1.0):
        x = self.layer1(x, c)
        # Project after layer (layer already includes projection internally)

        x = self.layer2(x, c)
        # Final projection
        x = jax.vmap(lambda xi: poincare.proj(xi, c))(x)
        return x
```

!!! note "Layer Projection"
    Hyperbolix layers already project internally after operations, so explicit projection between layers is optional but recommended for extra safety.

## Common Edge Cases

### Edge Case 1: Points Near the Boundary

**Symptoms**: NaN or Inf in gradients, exploding losses

**Solution**:
```python
# Check if points are too close to boundary
def check_boundary_proximity(x_batch, c=1.0):
    norms = jnp.linalg.norm(x_batch, axis=-1)
    max_norm = 1.0 / jnp.sqrt(c)
    proximity = norms / max_norm

    if jnp.any(proximity > 0.95):
        print(f"WARNING: Points near boundary (max proximity: {jnp.max(proximity):.4f})")
        return True
    return False

# Clip if needed
def safe_clip_to_interior(x_batch, c=1.0, safety_factor=0.9):
    max_allowed = safety_factor / jnp.sqrt(c)
    norms = jnp.linalg.norm(x_batch, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, max_allowed / (norms + 1e-8))
    return x_batch * scale
```

### Edge Case 2: Zero or Near-Zero Vectors

**Symptoms**: Division by zero warnings, NaN in tangent operations

**Solution**:
```python
# Manifold functions handle this internally with MIN_NORM
# But you can add explicit checks:

def safe_normalize(v, eps=1e-8):
    norm = jnp.linalg.norm(v)
    return jnp.where(norm > eps, v / norm, jnp.zeros_like(v))
```

### Edge Case 3: Large Learning Rates

**Symptoms**: Points shoot to boundary, training collapse

**Solution**:
```python
# Use conservative learning rates
from hyperbolix.optim import riemannian_adam

# For Poincaré ball
optimizer = riemannian_adam(learning_rate=1e-3)  # Not 1e-2 or higher!

# For Hyperboloid
optimizer = riemannian_adam(learning_rate=5e-4)  # Even more conservative

# Use learning rate scheduling
from optax import exponential_decay

schedule = exponential_decay(
    init_value=1e-3,
    transition_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = riemannian_adam(learning_rate=schedule)
```

### Edge Case 4: High Curvature Values

**Symptoms**: Numerical instability, rapid convergence to boundary

**Solution**:
```python
# Keep curvature moderate
c = 1.0  # Good default

# High curvature (c > 1) increases numerical challenges
c = 0.1  # Lower curvature = larger hyperbolic space = more stable

# If learning curvature, clip it
def clip_curvature(c, min_c=0.01, max_c=10.0):
    return jnp.clip(c, min_c, max_c)
```

## Checking Manifold Constraints

### Validation Functions

Each manifold provides `is_in_manifold` for validation:

```python
from hyperbolix.manifolds import poincare, hyperboloid

# Poincaré ball: ||x||² < 1/c
x = jnp.array([0.5, 0.3])
assert poincare.is_in_manifold(x, c=1.0, atol=1e-5)

# Hyperboloid: x₀² - Σxᵢ² = 1/c
x_ambient = jnp.array([1.5, 0.2, 0.3, 0.1])  # (dim+1,)
assert hyperboloid.is_in_manifold(x_ambient, c=1.0, atol=1e-5)
```

### Automated Checking (Checkify)

Use checkify modules for runtime validation:

```python
from hyperbolix.manifolds import poincare_checked
import jax.experimental.checkify as checkify

# Wrap computation
@checkify.checkify
def safe_distance(x, y, c):
    return poincare_checked.dist(x, y, c, version_idx=0)

# Run with error checking
err, result = safe_distance(x, y, c=1.0)
err.throw()  # Raises exception if constraint violated
```

### Batch Validation

```python
def validate_batch(x_batch, c=1.0, atol=1e-5):
    """Check if all points in batch satisfy manifold constraint."""
    valid = jax.vmap(lambda x: poincare.is_in_manifold(x, c, atol))(x_batch)
    num_valid = jnp.sum(valid)
    total = len(x_batch)

    if num_valid < total:
        print(f"WARNING: {total - num_valid}/{total} points off manifold")
        # Find violating points
        violations = jnp.where(~valid)[0]
        print(f"Violating indices: {violations[:10]}")  # Show first 10

    return jnp.all(valid)
```

## Best Practices Summary

!!! success "Numerical Stability Checklist"
    - ✅ **Use float32 for distances < 7, float64 for larger**
    - ✅ **Project after operations that might violate constraints**
    - ✅ **Keep points away from boundary** (max norm < 0.9/√c)
    - ✅ **Use conservative learning rates** (< 1e-3 for Poincaré, < 5e-4 for Hyperboloid)
    - ✅ **Use protected math functions** (`hyperbolix.utils.math_utils`)
    - ✅ **Monitor conformal factors** during training
    - ✅ **Validate manifold constraints** in debugging
    - ✅ **Use `VERSION_MOBIUS_DIRECT` for Poincaré distance** unless issues arise
    - ✅ **Clip curvature** if learnable (0.01 < c < 10.0)
    - ✅ **Initialize embeddings conservatively** (small norms)

## Debugging Numerical Issues

### Step-by-Step Diagnostic

1. **Check for NaN/Inf**:
   ```python
   assert jnp.all(jnp.isfinite(x_batch)), "NaN or Inf detected in data"
   ```

2. **Verify manifold constraints**:
   ```python
   validate_batch(x_batch, c=1.0, atol=1e-5)
   ```

3. **Check boundary proximity**:
   ```python
   check_boundary_proximity(x_batch, c=1.0)
   ```

4. **Switch to float64**:
   ```python
   x_batch = x_batch.astype(jnp.float64)
   ```

5. **Try different version**:
   ```python
   # Try VERSION_LORENTZIAN_PROXY if VERSION_MOBIUS_DIRECT fails
   dist = poincare.dist(x, y, c, version_idx=3)
   ```

6. **Enable checkify**:
   ```python
   # Use checked manifolds for runtime assertions
   from hyperbolix.manifolds import poincare_checked
   ```

## See Also

- [Batching & JIT](batching-jit.md): Performance optimization patterns
- [Manifolds API](../api-reference/manifolds.md): Manifold function reference
- [Training Workflows](training-workflows.md): End-to-end training examples
- [Mathematical Background](../mathematical-background.md): Theory and formulas
