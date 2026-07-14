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
from hyperbolix.manifolds import Poincare

# Float32 (default)
poincare_f32 = Poincare()
x = jnp.array([0.1, 0.2])
y = jnp.array([0.8, 0.5])
dist = poincare_f32.dist(x, y, c=1.0)

# Float64 (high precision) — inputs are automatically cast
poincare_f64 = Poincare(dtype=jnp.float64)
dist = poincare_f64.dist(x, y, c=1.0)  # returns float64
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
    from hyperbolix.manifolds import Poincare

    poincare = Poincare()
    distances = jax.vmap(lambda x: poincare.dist_0(x, c=1.0))(x_batch)
    max_dist = jnp.max(distances)
    print(f"Max distance from origin: {max_dist:.2f}")
    # If > 7, create Poincare(dtype=jnp.float64) instead
    ```

## Storage vs. Compute Dtype

Hyperbolix separates two dtype concerns that are easy to conflate:

- **Compute precision** — the dtype in which manifold operations (`dist`,
  `expmap`, `logmap`, …) run. Controlled by the manifold's `dtype` attribute
  (e.g. `Poincare(dtype=jnp.float64)`). Manifold methods cast their array
  arguments to this dtype on entry.
- **Storage dtype** — the dtype in which a layer's trainable parameters and
  persistent state (batch-norm statistics, VQ codebooks) are kept. Controlled
  by the `param_dtype` constructor argument on every NN layer
  (default: `jnp.float32`, following the Flax convention).

The two are decoupled: a float32-stored parameter that enters a float64
manifold operation is promoted to float64 *for that computation only*; the
parameter itself stays float32. The Riemannian optimizers follow the same
contract — `egrad2rgrad`/`expmap`/`ptransp` run in the manifold dtype, but the
returned updates and the momentum buffers are cast back to the parameter's
storage dtype.

**The recommended high-precision recipe** is therefore float64 *compute* with
float32 *storage* — full precision where the geometry needs it, at half the
parameter/optimizer-state memory and with float32 checkpoints:

```python
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypLinearPoincarePP

# Requires global x64 (JAX_ENABLE_X64=1) for the float64 compute path.
manifold = Poincare(dtype=jnp.float64)            # compute: float64
layer = HypLinearPoincarePP(manifold, 64, 32, rngs=nnx.Rngs(0))  # storage: float32 (default)

# Fully-float64 networks are an explicit opt-in:
layer_f64 = HypLinearPoincarePP(manifold, 64, 32, rngs=nnx.Rngs(0), param_dtype=jnp.float64)
```

!!! note "Parameter dtype rarely matters"
    Float32 parameter storage costs essentially nothing in accuracy: precision
    in hyperbolic networks is consumed by the manifold operations (conformal
    factors, `atanh`/`acosh` near their singularities), not by where the
    weights are stored. Reach for `param_dtype=jnp.float64` only for
    reproducibility studies or numerical debugging.

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
from hyperbolix.manifolds import Poincare

poincare = Poincare()
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
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# After Möbius addition or other operations
result = poincare.addition(x, y, c=1.0)
result = poincare.proj(result, c=1.0)  # Project back to manifold
```

**2. Keep points away from boundary**

```python
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# During initialization
def init_hyperbolic_embeddings(key, n_points, dim, max_norm=0.8):
    """Initialize embeddings safely away from boundary."""
    x = jax.random.normal(key, (n_points, dim)) * 0.1
    x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

    # Clip to max_norm to avoid boundary
    norms = jnp.linalg.norm(x_proj, axis=-1, keepdims=True)
    x_clipped = jnp.where(norms > max_norm, x_proj * max_norm / norms, x_proj)
    return x_clipped
```

**3. Use float64 manifold for critical operations**

```python
from hyperbolix.manifolds import Poincare
import jax.numpy as jnp

# Create a float64 manifold — inputs are automatically cast
poincare_f64 = Poincare(dtype=jnp.float64)
dist_precise = poincare_f64.dist(x, y, c=1.0)  # returns float64
```

## Proper Velocity: An Unconstrained Alternative

The Proper Velocity (PV) model (Chen et al. 2026) sidesteps the conformal-factor and boundary problems above by representing hyperbolic geometry in **unconstrained $\mathbb{R}^n$**. Points carry no norm constraint, so there is no boundary to drift toward and no $\lambda(x) \to \infty$ singularity.

Use `ProperVelocity` when your features or embeddings reach large geodesic distances from the origin and float32 precision must be preserved.

### Why PV Stays Stable at Large Radii

| Issue (Poincaré / Hyperboloid) | PV behavior |
|--------------------------------|-------------|
| $\lambda(x) = 2/(1 - c\|x\|^2) \to \infty$ near boundary | $\beta_x = 1/\sqrt{1 + c\|x\|^2}$, bounded in $(0, 1]$, smooth everywhere |
| Catastrophic cancellation in $1 - c\|x\|^2$ | No boundary; $1 + c\|x\|^2$ grows monotonically |
| Hyperboloid constraint drift after Euclidean update | PV is $\mathbb{R}^n$ — any finite vector is a valid point |
| `atanh` clamp required at the boundary | Geodesic distance uses `asinh`, stable on all of $\mathbb{R}$ |

The PV distance formula
$$
d(0, x) = \frac{1}{\sqrt{c}} \cdot \mathrm{asinh}(\sqrt{c}\,\|x\|)
$$
remains finite and accurate in float32 for $\|x\|$ up to at least $10^2$ — covered by `test_pv_stability_at_large_norms` in the test suite.

### Example

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import ProperVelocity

pv = ProperVelocity()
c = 1.0

# PV tolerates large-norm inputs where Poincaré would hit the boundary.
x_large = jnp.array([50.0, 0.0, 0.0])
d = pv.dist_0(x_large, c)      # ~ 4.61 — finite, accurate
y = pv.logmap_0(x_large, c)    # finite tangent vector
x_rec = pv.expmap_0(y, c)      # round-trips to x_large
```

### Choosing a Manifold for Stability

- **Poincaré ball**: compact, bounded — fine for small distances ($<5$) and visualization; clamp or use float64 past that.
- **Hyperboloid**: unbounded radius, but the constraint $\langle x, x\rangle_L = -1/c$ must be maintained and can drift under Euclidean updates.
- **Proper Velocity**: unconstrained $\mathbb{R}^n$, stable at large radii, exact Euclidean retraction (plain `optax.adam` / SGD trains PV layers without a Riemannian wrapper). Preferred when embeddings naturally grow large.
- **κ-Stereographic**: identical numerics to the Poincaré ball for $c > 0$ (they share the same gyrovector core); adds the flat and spherical regimes and a Taylor-series switchover near $c = 0$ — see the [dedicated section below](#stereographic-near-zero-curvature).

!!! note "Training PV layers"
    `HypLinearPV`, `HypConv2DPV`, and `HypRegressionPV` store their weights as plain `nnx.Param` (not `ManifoldParam`). Use a standard `nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)` — no `riemannian_adam` / `riemannian_sgd` wrapper is required.

## κ-Stereographic: Numerics Near Zero Curvature {#stereographic-near-zero-curvature}

The `Stereographic` manifold's signed curvature introduces one numerical regime the other manifolds don't have: the neighborhood of $c = 0$, where every closed-form expression becomes $0/0$ and the implementation switches to Taylor series. The switching logic is internal, but its consequences matter when you train a *signed* learnable curvature that may cross zero.

For $c > 0$ nothing here is new — `Stereographic` shares its gyrovector core with `Poincare`, so every boundary/conformal-factor consideration above applies verbatim. See the [κ-Stereographic API reference](../api-reference/manifolds.md) for the sign convention and the factor-2 flat limit.

### The Taylor Cutover Is dtype-Dependent

The curvature-generalized trig functions ($\tan_\kappa$, $\tan_\kappa^{-1}$) use their exact closed forms away from zero and a truncated Taylor series (degree 5 in $\kappa\lVert x\rVert^2$) near zero. The cutover differs by precision:

| dtype | Taylor branch used when | Why |
|-------|-------------------------|-----|
| float64 | $\lvert\kappa\rvert < 10^{-9}$ | closed forms accurate down to ~$10^{-9}$ |
| float32 | $\lvert\kappa\rvert < 10^{-5}$ | catastrophic cancellation in the closed-form **curvature gradient** below this |

In float32 the *values* stay accurate well below $10^{-5}$; it is $\partial(\cdot)/\partial c$ computed through the closed forms that degrades. The wider float32 window trades a small seam error (worst-case measured relative error of the curvature gradient just above the cutover: ~2.4%, sign always correct) for finite, well-behaved gradients everywhere.

The Taylor branch is additionally gated on its convergence region $\lvert\kappa\rvert\,\lVert x\rVert^2 < 0.01$: points at extreme chart radii ($\lVert x\rVert \sim 1/\sqrt{\lvert\kappa\rvert}$, e.g. spherical points far from the chart origin) always keep the exact closed form, no matter how small $\lvert\kappa\rvert$ is.

### Spherical Regime ($c < 0$) Cautions

- **The chart has no boundary, but it has a pole.** Stereographic coordinates cover the sphere minus one point; near-antipodal pairs have chart norms $\sim 1/\sqrt{\lvert c\rvert}$ and the metric shrinks accordingly. `antipode` itself is exact (closed form $x/(c\lVert x\rVert^2)$), but *optimizing through* near-antipodal configurations concentrates precision loss the same way the Poincaré boundary does.
- **Distances saturate at $\pi R = \pi/\sqrt{\lvert c\rvert}$.** Gradients of `dist` vanish as a pair approaches antipodal, analogous to `atanh` saturation in the hyperbolic regime.

### Recommendations

- Use `Stereographic(dtype=jnp.float64)` when training a signed curvature that may cross zero, per Bachmann et al. (2020). Float32 is fine at fixed moderate curvature ($\lvert c\rvert \gtrsim 10^{-4}$) and moderate radii.
- With `LearnableCurvature(parameterization="identity")`, the default clamp $[-10, 10]$ includes 0 by design — a curvature crossing zero is a *feature* (the geometry interpolates hyperbolic → flat → spherical smoothly), not an error state.

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
from hyperbolix.manifolds import Poincare
import jax.numpy as jnp

poincare = Poincare()
x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, 0.4])
c = 1.0

# Version 0: Direct Möbius distance (FASTEST, default)
d0 = poincare.dist(x, y, c, version_idx=poincare.VERSION_MOBIUS_DIRECT)

# Version 1: Möbius via addition
d1 = poincare.dist(x, y, c, version_idx=poincare.VERSION_MOBIUS)

# Version 2: Metric tensor induced
d2 = poincare.dist(x, y, c, version_idx=poincare.VERSION_METRIC_TENSOR)

print(f"Version 0: {d0:.6f}")
print(f"Version 1: {d1:.6f}")
print(f"Version 2: {d2:.6f}")
# All should be approximately equal
```

### Which Version to Use?

**General recommendation**: `VERSION_MOBIUS_DIRECT` (version 0)
- Fastest
- Fewest intermediate operations
- Best for most applications

**Special cases**:
- **Near-boundary points** (||x|| > 0.9): Use `Poincare(dtype=jnp.float64)`, or convert to the
  hyperboloid via `isometry_mappings.poincare_to_hyperboloid` and use `Hyperboloid.dist`
  (the hyperboloid is unbounded, so there is no boundary to saturate)
- **Very high dimensions** (> 1000): `VERSION_METRIC_TENSOR` (version 2) may be more stable
- **Debugging**: Compare all versions — significant differences indicate numerical issues

### Using Versions with JIT

```python
import jax
from hyperbolix.manifolds import Poincare

poincare = Poincare()

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
- After Möbius addition: `poincare.addition(x, y, c)`
- After neural network layers
- After parameter updates in optimization

**Usually don't need projection**:
- After `expmap` (already on manifold)
- After `proj` (redundant)

### Projection

Projection ensures points stay on the manifold by clipping norms:

```python
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Project to Poincaré ball
x_proj = poincare.proj(x, c=1.0)

# Projection is numerically stable and automatically handles edge cases
```

### Projection in Training

```python
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypLinearPoincare
from flax import nnx

poincare = Poincare()

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
from hyperbolix.manifolds import Poincare, Hyperboloid
import jax.numpy as jnp

poincare = Poincare()
hyperboloid = Hyperboloid()

# Poincaré ball: ||x||² < 1/c
x = jnp.array([0.5, 0.3])
assert poincare.is_in_manifold(x, c=1.0, atol=1e-5)

# Hyperboloid: -x₀² + Σxᵢ² = -1/c  (with x₀ > 0)
x_ambient = jnp.array([1.5, 0.2, 0.3, 0.1])  # (dim+1,)
assert hyperboloid.is_in_manifold(x_ambient, c=1.0, atol=1e-5)
```

### Batch Validation

```python
from hyperbolix.manifolds import Poincare

poincare = Poincare()

def validate_batch(x_batch, c=1.0, atol=1e-5):
    """Check if all points in batch satisfy manifold constraint."""
    valid = jax.vmap(lambda x: poincare.is_in_manifold(x, c, atol))(x_batch)
    num_valid = jnp.sum(valid)
    total = len(x_batch)

    if num_valid < total:
        print(f"WARNING: {total - num_valid}/{total} points off manifold")
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
    - ✅ **Prefer `ProperVelocity` for large-radius features** — unconstrained $\mathbb{R}^n$ avoids the boundary entirely and trains with plain `optax.adam`

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
   # Try VERSION_METRIC_TENSOR if VERSION_MOBIUS_DIRECT fails
   from hyperbolix.manifolds import Poincare
   poincare = Poincare()
   dist = poincare.dist(x, y, c, version_idx=poincare.VERSION_METRIC_TENSOR)
   ```

6. **Use float64 manifold**:
   ```python
   from hyperbolix.manifolds import Poincare
   import jax.numpy as jnp
   poincare_f64 = Poincare(dtype=jnp.float64)
   dist = poincare_f64.dist(x, y, c)
   ```

## See Also

- [Batching & JIT](batching-jit.md): Performance optimization patterns
- [Manifolds API](../api-reference/manifolds.md): Manifold function reference
- [Training Workflows](training-workflows.md): End-to-end training examples
