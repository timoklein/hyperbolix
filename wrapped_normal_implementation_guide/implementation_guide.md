# Wrapped Normal Distribution Implementation Guide

## Overview

Implementation of the wrapped normal distribution on hyperbolic space (hyperboloid model) in JAX. This distribution wraps a Gaussian from the tangent space at the origin onto the manifold via parallel transport and exponential map.

## Mathematical Background

### Sampling (Algorithm 1)
1. Sample `v_bar ~ N(0, Σ) ∈ R^n`
2. Embed as tangent vector `v = [0, v_bar] ∈ T_{μ₀}ℍⁿ` at origin `μ₀ = (1, 0, ..., 0)ᵀ`
3. Parallel transport to mean: `u = PT_{μ₀→μ}(v) ∈ T_μℍⁿ`
4. Map to manifold: `z = exp_μ(u) ∈ ℍⁿ`

### Log-PDF (Algorithm 2)
1. Map to tangent space: `u = log_μ(z) = exp_μ⁻¹(z)`
2. Transport to origin: `v = PT_{μ→μ₀}(u)`
3. Compute: `log p(z) = log p(v) - log det(∂proj_μ(v)/∂v)`
   - Where: `det(∂proj_μ(v)/∂v) = (sinh r / r)^(n-1)`
   - And: `r = ||v||_L` (Minkowski norm)

## Key Design Decisions

### 1. Framework Choice
**Decision:** JAX only (in `src/hyperbolix_jax/`)

**Rationale:** Focus on single framework for initial implementation. PyTorch version can be added later if needed.

### 2. API Design: Pure Functional
**Decision:** Pure functional interface (no classes, no NamedTuples)

```python
# Sampling
z = wrapped_normal.sample(key, mu, sigma, c, sample_shape=(), dtype=None)

# Log probability
log_p = wrapped_normal.log_prob(z, mu, sigma, c)
```

**Rationale:**
- **Simplest:** No overhead of classes, constructors, or type definitions
- **Most JAX idiomatic:** Matches `jax.random.*`, `jax.nn.*`, and existing `hyperboloid.*` functions
- **Familiar:** Mirrors `jax.random.multivariate_normal`'s `sample_shape`/`dtype` interface so callers can adopt it drop-in
- **Best jit/vmap integration:** Explicit control over which arguments get vmapped
- **Consistent:** All existing manifold operations are pure functions
- **Flexible:** Parameters come from neural network outputs, not stored in distribution object

**Alternatives considered:**
- ❌ `flax.nnx.Module`: Overkill when parameters aren't learnable in the distribution itself
- ❌ `NamedTuple`: Bundles 3 parameters but adds complexity without clear benefit

### 3. Module Organization
**Decision:** Separate `distributions/` module

```
src/hyperbolix_jax/
├── manifolds/          # Geometric primitives
├── distributions/      # Probability distributions (NEW)
│   ├── __init__.py
│   └── wrapped_normal.py
├── nn_layers/
├── optim/
└── utils/
```

**Rationale:**
- **Separation of concerns:** Geometry vs. probability
- **Extensibility:** Room for future distributions (wrapped Cauchy, Riemannian normal, etc.)
- **Conceptual clarity:** Distributions BUILD ON manifolds, not part of manifold definition
- **Matches conventions:** PyTorch (`torch.nn` vs `torch.distributions`), SciPy (`scipy.spatial` vs `scipy.stats`)

## Implementation Details

### Covariance Support
**Support all three parameterizations (converted before calling `jax.random.multivariate_normal`):**
1. **Isotropic:** `sigma` is scalar standard deviation → expand to `σ² I`
2. **Diagonal:** `sigma` is n-dimensional vector of standard deviations → expand to `diag(σ₁², ..., σₙ²)`
3. **Full:** `sigma` is n×n covariance matrix (already SPD)

**Primary use case:** Diagonal covariance (most common). `_sample_gaussian` delegates to `jax.random.multivariate_normal` so SPD/dtype checks happen upstream; the helper only standardizes parameterizations and applies broadcasting.

### Batch & Shape Semantics
- `sample` mirrors `jax.random.multivariate_normal`: accepts `sample_shape=()` and `dtype` arguments, broadcasts `mu`, `sigma`, and optional curvature arrays across shared batch dimensions, and materializes leading `sample_shape` axes before parameter axes.
- `mu` has shape `(..., n + 1)` (hyperboloid coordinates). Spatial dimension `n = mu.shape[-1] - 1` determines covariance size.
- `sigma` matches the parameterization shape for the spatial dimension but supports any number of leading batch axes that broadcast with `mu`.
- Curvature `c` can be a Python float or an array; arrays must match (or be broadcastable to) `mu.shape[:-1]` to avoid ambiguous broadcasting. When batched, `c` should line up exactly with the batch axes of `mu`/`sigma`.
- `log_prob` mirrors broadcasting rules as well: output shape is `sample_shape + batch_shape` (no manifold dimension).
- Helper functions should raise informative shape errors before calling JAX primitives when broadcasting fails.

### Dtype Propagation
- Default to `dtype or mu.dtype` and explicitly cast `sigma`, intermediate Gaussian samples, and curvature-dependent computations to that dtype.
- Respect `jax_enable_x64`; allow callers to request float64 draws while keeping parameters in float32.

### Dimension Handling
- Covariance `Σ` is **n × n** for spatial components only
- Sample `v_bar ~ N(0, Σ) ∈ R^n`
- Construct tangent vector: `v = [0, v_bar] ∈ R^(n+1)` (prepend zero for temporal component)
- This places the sample in the tangent space at origin `μ₀`

### Parallel Transport Direction
- **Sampling:** Use `ptransp_0(v, μ, c)` to transport FROM origin TO μ
- **Log-prob:** Use `ptransp(u, μ, μ₀, c)` to transport FROM μ TO origin
  - Where `μ₀ = [1/√c, 0, ..., 0]ᵀ` (created via `_create_origin`)

### Inverse Exponential Map
**Implementation:** `exp_μ⁻¹(z) = log_μ(z)`

Use existing `logmap(z, μ, c)` from `hyperboloid.py:316-340`

### Existing Operations
All required primitives already exist in `hyperboloid.py`:
- `expmap(v, x, c)` and `expmap_0(v, c)` - exponential maps
- `logmap(y, x, c)` and `logmap_0(y, c)` - logarithmic maps
- `ptransp(v, x, y, c)` and `ptransp_0(v, y, c)` - parallel transport
- `_create_origin(c, dim, dtype)` - origin point construction
- `_minkowski_inner(x, y)` - Minkowski inner product

## Implementation Plan

### Phase 1: Core Sampling Function
1. Create `src/hyperbolix_jax/distributions/` directory
2. Create `__init__.py` to export public API
3. Implement `wrapped_normal.py`:
   - Helper: `_sample_gaussian(key, sigma, n)` - handles isotropic/diagonal/full covariance
   - Helper: `_embed_in_tangent_space(v_spatial)` - prepends zero to create `[0, v_bar]`
   - Main: `sample(key, mu, sigma, c, sample_shape=(), dtype=None)` - full sampling algorithm mirroring `jax.random.multivariate_normal` semantics

### Phase 2: Log-Probability Function
4. Implement log-prob components:
   - Helper: `_gaussian_log_prob(v, sigma)` - log probability of Gaussian
   - Helper: `_log_det_jacobian(v, c)` - compute `log((sinh r / r)^(n-1))` with a dedicated `r → 0` Taylor branch
   - Main: `log_prob(z, mu, sigma, c)` - full log-probability

### Phase 3: Testing
5. Create `tests/jax/test_wrapped_normal.py`:
   - **Test 1:** Basic functionality + `sample_shape` semantics (`sample_shape=(5, 3)` with batched `mu`)
   - **Test 2:** Manifold constraint (sampled points satisfy hyperboloid equation)
   - **Test 3:** Euclidean limit sanity check (`c → 0` / mean at origin) by comparing against `jax.random.multivariate_normal`
   - **Test 4:** Gradient flow + `jit/vmap` compatibility (ensure differentiability for learning)
   - **Test 5:** All covariance types (isotropic, diagonal, full) and dtype propagation (float32 vs float64)

### Phase 4: Validation & Documentation
6. Lightweight validation helper: `validate_params(mu, sigma, c)` guarding shape agreement and curvature broadcast issues (optionally toggled via a `debug` flag)
7. Add docstrings with math notation and usage examples
8. Update main package `__init__.py` to expose distributions module
9. Add usage examples in docstrings

## Testing Strategy

### Test 1: Basic Functionality
```python
# Ensure functions execute without errors and respect sample_shape/dtype
z = wrapped_normal.sample(key, mu, sigma, c, sample_shape=(4,), dtype=jnp.float64)
lp = wrapped_normal.log_prob(z, mu, sigma, c)
assert z.shape == (4,) + mu.shape
assert lp.shape == (4,) + mu.shape[:-1]
assert z.dtype == jnp.float64
```

### Test 2: Manifold Constraint
```python
# Verify samples lie on hyperboloid
z = wrapped_normal.sample(key, mu, sigma, c)
assert hyperboloid.is_in_manifold(z, c, atol=1e-5)
```

### Test 3: Euclidean Limit Consistency
```python
# With curvature near zero, wrapped samples should match Euclidean MVN statistics.
flat = jax.random.multivariate_normal(key, mean=jnp.zeros(n), cov=jnp.eye(n), shape=(1024,))
wrapped = wrapped_normal.sample(key, origin_mu, sigma, c=jnp.array(1e-5), sample_shape=(1024,))
assert jnp.allclose(flat.mean(axis=0), wrapped[..., 1:].mean(axis=0), atol=1e-2)
```

### Test 4: Gradient Flow
```python
# Ensure differentiability inside jit/vmap pipelines
def loss(mu):
    z = wrapped_normal.sample(key, mu, sigma, c, sample_shape=(2,))
    return wrapped_normal.log_prob(z, mu, sigma, c).sum()

grad_mu = jax.grad(loss)(mu)
assert not jnp.any(jnp.isnan(grad_mu))
```

### Test 5: Covariance Types & Dtypes
```python
# Test isotropic
z1 = wrapped_normal.sample(key, mu, 0.1, c)

# Test diagonal
z2 = wrapped_normal.sample(key, mu, jnp.array([0.1, 0.2]), c)

# Test full
z3 = wrapped_normal.sample(key, mu, jnp.eye(2) * 0.1, c)

# Test dtype propagation
z_float64 = wrapped_normal.sample(key, mu, sigma, c, dtype=jnp.float64)
assert z_float64.dtype == jnp.float64
```

## Notes

- Keep functions pure and stateless for JAX compatibility
- Use `jaxtyping` for type hints (Array, Float, PRNGKeyArray)
- Follow existing code style (line length 127, black formatting)
- Numerical stability: use same conventions as `hyperboloid.py` (MIN_NORM, clipping, etc.)
- Consider adding `jax.jit` decorators for performance after validation
