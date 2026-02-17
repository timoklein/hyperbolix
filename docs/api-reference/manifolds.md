# Manifolds API

This page documents the core manifold operations in Hyperbolix. Each manifold is a class that provides geometric operations and automatic dtype casting.

## Overview

Hyperbolix provides three manifold classes:

- **Euclidean**: Flat Euclidean space (baseline)
- **Poincaré Ball**: Conformal model of hyperbolic space
- **Hyperboloid**: Lorentz/Minkowski model of hyperbolic space

All manifolds share a common interface defined by the `Manifold` protocol and support:

- **Automatic dtype casting**: Pass `dtype=jnp.float64` for higher precision
- **vmap-native methods**: Methods operate on single points; use `jax.vmap` for batching
- **JIT compatibility**: All methods are JIT-compilable

## Manifold Protocol

::: hyperbolix.manifolds.protocol.Manifold
    options:
      show_source: true
      heading_level: 3

## Euclidean

Flat Euclidean space (identity operations).

::: hyperbolix.manifolds.euclidean.Euclidean
    options:
      show_source: true
      heading_level: 3

## Poincaré Ball

The Poincaré ball model with Möbius operations.

!!! note "Distance Versions"
    The Poincaré `dist` method has a `version_idx` parameter selecting between 4 formulations:

    - `VERSION_MOBIUS_DIRECT` (0): Möbius addition formula (default, fastest)
    - `VERSION_MOBIUS` (1): Möbius via addition
    - `VERSION_METRIC_TENSOR` (2): Direct metric tensor integration
    - `VERSION_LORENTZIAN_PROXY` (3): Lorentzian model proxy (best near boundary)

    Constants are available as `poincare.VERSION_MOBIUS_DIRECT` etc., or from
    `hyperbolix.manifolds.poincare`.

::: hyperbolix.manifolds.poincare.Poincare
    options:
      show_source: true
      heading_level: 3

## Hyperboloid

The hyperboloid (Lorentz) model with Minkowski geometry.

!!! note "Lorentz Operations"
    The Hyperboloid class includes specialized operations for convolutional layers:

    - `lorentz_boost`: Lorentz boost transformation
    - `distance_rescale`: Distance-based rescaling
    - `hcat`: Lorentz direct concatenation for convolutions

::: hyperbolix.manifolds.hyperboloid.Hyperboloid
    options:
      show_source: true
      heading_level: 3

## Isometry Mappings

Distance-preserving maps between Poincaré ball and hyperboloid models.

::: hyperbolix.manifolds.isometry_mappings
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Basic Distance Computation

```python
import jax.numpy as jnp
from hyperbolix.manifolds import Poincare

poincare = Poincare()

x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, -0.1])
c = 1.0

# Compute distance (default: VERSION_MOBIUS_DIRECT)
distance = poincare.dist(x, y, c)
```

### Float64 Precision

```python
from hyperbolix.manifolds import Poincare
import jax.numpy as jnp

# High-precision manifold
poincare_f64 = Poincare(dtype=jnp.float64)

x = jnp.array([0.1, 0.2])  # float32 input
distance = poincare_f64.dist(x, y, c=1.0)  # automatically cast to float64
print(distance.dtype)  # float64
```

### Batched Operations with vmap

```python
import jax
from hyperbolix.manifolds import Hyperboloid

hyperboloid = Hyperboloid()
c = 1.0

# Batch of ambient points (d+1 dimensions)
x_batch = jax.random.normal(jax.random.PRNGKey(0), (100, 4))
y_batch = jax.random.normal(jax.random.PRNGKey(1), (100, 4))

# Project to hyperboloid
x_proj = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x_batch, c)
y_proj = jax.vmap(hyperboloid.proj, in_axes=(0, None))(y_batch, c)

# Compute distances
distances = jax.vmap(hyperboloid.dist, in_axes=(0, 0, None))(x_proj, y_proj, c)
```

### Exponential and Logarithmic Maps

```python
from hyperbolix.manifolds import Poincare
import jax.numpy as jnp

poincare = Poincare()

# Point on manifold
x = poincare.proj(jnp.array([0.2, 0.3]), c=1.0)

# Tangent vector
v = jnp.array([0.1, -0.05])

# Exponential map (move along geodesic)
y = poincare.expmap(v, x, c=1.0)

# Logarithmic map (inverse operation)
v_recovered = poincare.logmap(y, x, c=1.0)
```

### Isometry Mappings

```python
from hyperbolix.manifolds import isometry_mappings
import jax.numpy as jnp

# Hyperboloid point (ambient coordinates, d+1 dims)
x_hyperboloid = jnp.array([1.5, 0.5, 0.3])  # Must satisfy Lorentz constraint

# Map to Poincaré ball (intrinsic coordinates, d dims)
x_poincare = isometry_mappings.hyperboloid_to_poincare(x_hyperboloid, c=1.0)

# Map back (round-trip)
x_hyperboloid_recovered = isometry_mappings.poincare_to_hyperboloid(x_poincare, c=1.0)
```

## Numerical Considerations

!!! warning "Float32 Precision"
    Float32 can cause numerical issues, especially in the Poincaré ball near the boundary. Use `Poincare(dtype=jnp.float64)` for:

    - High curvature values (`c > 1.0`)
    - Points near manifold boundaries
    - Deep neural networks with many layers

See the [Numerical Stability](../user-guide/numerical-stability.md) guide for details.
