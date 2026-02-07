# Manifolds API

This page documents the core manifold operations in Hyperbolix. Each manifold module provides a consistent API for geometric operations on hyperbolic and Euclidean spaces.

## Overview

Hyperbolix provides three manifold implementations:

- **Euclidean**: Flat Euclidean space (baseline)
- **Poincaré Ball**: Conformal model of hyperbolic space
- **Hyperboloid**: Lorentz/Minkowski model of hyperbolic space

All manifolds follow a **pure functional design** with vmap-native operations.

## Common Operations

Each manifold provides these core operations:

- `proj`: Project points onto the manifold
- `dist`: Compute distances between points
- `expmap`: Exponential map (tangent → manifold)
- `logmap`: Logarithmic map (manifold → tangent)
- `ptransp`: Parallel transport of tangent vectors
- `egrad2rgrad`: Convert Euclidean to Riemannian gradients

## Euclidean

Flat Euclidean space (identity operations).

::: hyperbolix.manifolds.euclidean
    options:
      show_source: true
      heading_level: 3
      members:
        - proj
        - dist
        - expmap
        - logmap
        - ptransp
        - egrad2rgrad
        - inner
        - norm

## Poincaré Ball

The Poincaré ball model with Möbius operations.

!!! note "Distance Versions"
    The Poincaré ball provides **4 distance computation methods** via the `version` parameter:

    - `version=0`: Möbius addition formula (fastest)
    - `version=1`: Direct metric tensor integration
    - `version=2`: Lorentzian model proxy
    - `version=3`: Conformal factor integration

    Different versions offer trade-offs between speed and numerical stability.

::: hyperbolix.manifolds.poincare
    options:
      show_source: true
      heading_level: 3
      members:
        - proj
        - dist
        - expmap
        - logmap
        - ptransp
        - egrad2rgrad
        - mobius_add
        - mobius_scalar_mul
        - conformal_factor
        - inner
        - norm
        - gyration

## Hyperboloid

The hyperboloid (Lorentz) model with Minkowski geometry.

!!! note "Lorentz Operations"
    The hyperboloid module includes specialized operations for convolutional layers:

    - `lorentz_boost`: Lorentz boost transformation
    - `distance_rescale`: Distance-based rescaling
    - `hcat`: Lorentz direct concatenation for convolutions

::: hyperbolix.manifolds.hyperboloid
    options:
      show_source: true
      heading_level: 3
      members:
        - proj
        - dist
        - expmap
        - logmap
        - ptransp
        - egrad2rgrad
        - lorentz_add
        - lorentz_scalar_mul
        - lorentz_boost
        - distance_rescale
        - hcat
        - inner
        - norm


## Usage Examples

### Basic Distance Computation

```python
import jax.numpy as jnp
from hyperbolix.manifolds import poincare

x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, -0.1])
c = 1.0

# Compute distance
distance = poincare.dist(x, y, c, version_idx=0)
```

### Batched Operations with vmap

```python
import jax
from hyperbolix.manifolds import hyperboloid

# Batch of points
x_batch = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
y_batch = jax.random.normal(jax.random.PRNGKey(1), (100, 3))

# Project to hyperboloid
x_proj = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x_batch, c)
y_proj = jax.vmap(hyperboloid.proj, in_axes=(0, None))(y_batch, c)

# Compute distances
distances = jax.vmap(hyperboloid.dist, in_axes=(0, 0, None, None))(
    x_proj, y_proj, c, 0
)
```

### Exponential and Logarithmic Maps

```python
from hyperbolix.manifolds import poincare

# Point on manifold
x = poincare.proj(jnp.array([0.2, 0.3]), c=1.0)

# Tangent vector
v = jnp.array([0.1, -0.05])

# Exponential map (move along geodesic)
y = poincare.expmap(x, v, c=1.0)

# Logarithmic map (inverse operation)
v_recovered = poincare.logmap(x, y, c=1.0)
```

## Numerical Considerations

!!! warning "Float32 Precision"
    Float32 can cause numerical issues, especially in the Poincaré ball near the boundary. Consider using float64 for:

    - High curvature values (`c > 1.0`)
    - Points near manifold boundaries
    - Deep neural networks with many layers

See the [Numerical Stability](../user-guide/numerical-stability.md) guide for details.
