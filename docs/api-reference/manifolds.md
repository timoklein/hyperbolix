# Manifolds API

This page documents the core manifold operations in Hyperbolix. Each manifold is a class that provides geometric operations and automatic dtype casting.

## Overview

Hyperbolix provides four base manifold classes plus a composition class:

- **Euclidean**: Flat Euclidean space (baseline)
- **Poincaré Ball**: Conformal model of hyperbolic space
- **Hyperboloid**: Lorentz/Minkowski model of hyperbolic space
- **Proper Velocity**: Unconstrained $\mathbb{R}^n$ model from special relativity (Chen et al. 2026)
- **Product Manifold**: Heterogeneous-curvature product spaces $M_1 \times M_2 \times \dots \times M_n$ (Gu et al. 2019)

All manifolds share a common interface defined by the `Manifold` protocol and support:

- **Automatic dtype casting**: Pass `dtype=jnp.float64` for higher precision
- **vmap-native methods**: Methods operate on single points; use `jax.vmap` for batching
- **JIT compatibility**: All methods are JIT-compilable
- **Learnable curvature**: Use the `LearnableCurvature` module to add trainable curvature to any model (softplus or log/exp reparameterization, optional clamping)

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
    The Poincaré `dist` method has a `version_idx` parameter selecting between 3 formulations:

    - `VERSION_MOBIUS_DIRECT` (0): Möbius addition formula (default, fastest)
    - `VERSION_MOBIUS` (1): Möbius via addition
    - `VERSION_METRIC_TENSOR` (2): Direct metric tensor integration

    Constants are available as `poincare.VERSION_MOBIUS_DIRECT` etc., or from
    `hyperbolix.manifolds.poincare`.

!!! note "Apollonian weak metric"
    `apollonian_dist(x, y, c)` is the **non-symmetric** Apollonian weak metric $\delta$
    (Papadopoulos & Troyanov, *Weak metrics on Euclidean domains*, Thm 2) — a *weak metric*,
    not a geodesic distance:

    $$\delta_c(x,y) = \log\!\left(\frac{\sqrt{c}\,\lVert x-y\rVert + \sqrt{c^2\lVert x\rVert^2\lVert y\rVert^2 - 2c\langle x,y\rangle + 1}}{1-c\lVert y\rVert^2}\right)$$

    It satisfies $\delta(x,x)=0$, $\delta\ge 0$ and the triangle inequality, but
    $\delta(x,y) \neq \delta(y,x)$ in general. Its symmetrization recovers the geodesic distance:
    $\delta(x,y) + \delta(y,x) = \sqrt{c}\cdot$ `dist(x, y, c)`.

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
    - `log_radius_concat`: log-radius–preserving concatenation (digamma-scaled `hcat`; Shi et al. 2026, Sec. 4.3)

::: hyperbolix.manifolds.hyperboloid.Hyperboloid
    options:
      show_source: true
      heading_level: 3

## Proper Velocity

The Proper Velocity (PV) model — an **unconstrained** $\mathbb{R}^n$ representation of hyperbolic geometry rooted in special relativity's proper velocity (Ungar 2022, Ch. 10). PV is algebraically a gyrovector space isomorphic to the Poincaré ball via $\pi(x) = (\beta_x / (1 + \beta_x)) \cdot x$, and carries a Riemannian metric making that isomorphism an isometry.

!!! note "Why Proper Velocity?"
    Unlike the bounded Poincaré ball (points must stay in the open unit ball) and the constrained hyperboloid (points must satisfy $\langle x, x \rangle_L = -1/c$), PV points live in all of $\mathbb{R}^n$ with no constraint. This gives:

    - **No projection step** after updates — the manifold *is* $\mathbb{R}^n$
    - **Better numerical stability** for large radii (no boundary collapse, no Lorentz constraint drift)
    - **Drop-in with Euclidean optimizers** — the retraction reduces to $x + v$

    The metric $g_x(u,v) = \langle u, v\rangle - c\beta_x^2\langle x,u\rangle\langle x,v\rangle$ still gives sectional curvature $-c$; only the coordinates change.

!!! info "Convention"
    The paper uses curvature $K < 0$ with $\beta_x = 1/\sqrt{1 - K\|x\|^2}$. Hyperbolix keeps the $c > 0$ convention (sectional curvature $-c$), substituting $K = -c$, so $\beta_x = 1/\sqrt{1 + c\|x\|^2}$.

::: hyperbolix.manifolds.proper_velocity.ProperVelocity
    options:
      show_source: true
      heading_level: 3

## Product Manifold

Heterogeneous-curvature product space $P = M_1 \times M_2 \times \dots \times M_n$ where each factor $M_i$ can be any base manifold (Poincaré, Hyperboloid, Euclidean, Proper Velocity) with its own curvature $c_i$. Points are represented as flat concatenated arrays of shape `(total_dim,)`.

The geodesic distance on a product Riemannian manifold is Pythagorean over component distances:

$$d_P(x, y) \;=\; \sqrt{\sum_{i=1}^{n} d_{M_i}(x_i, y_i)^2}$$

where $x_i$, $y_i$ are the per-factor slices of the flat points.

!!! note "Per-factor `c` argument"
    Every geometry method (`dist`, `expmap`, `logmap`, `proj`, `origin`, …) takes a positional `c` argument that must be a sequence of length `n_factors` — one curvature per factor. There is no scalar fallback and no broadcast: pass `product.curvatures` for static curvatures, or a tuple built from `LearnableCurvature` calls for trainable ones. `ProductManifold` satisfies the `Manifold` protocol — the protocol-level `Curvature` type unions scalar and sequence-of-scalars, so `isinstance(product, Manifold)` is `True` and generic code typed against `Manifold` accepts product instances. The product itself has **no `c` attribute** — read factor-stored values via `product.curvatures`.

!!! tip "Static vs learnable curvature"
    Factor instances may carry an initial `c` (`Hyperboloid(c=1.0)`), but `ProductManifold` never reads it in its geometry methods — it is exposed via `product.curvatures` as a convenience default. For learnable curvature, instantiate one `LearnableCurvature` per factor on your `nnx.Module` and pass `c=(self.curv_a(), self.curv_b(), ...)` to the product. See the [Manifolds User Guide — Curvature in ProductManifold](../user-guide/manifolds.md#curvature-in-productmanifold) for the full pattern.

::: hyperbolix.manifolds.product.ProductManifold
    options:
      show_source: true
      heading_level: 3

## Isometry Mappings

Distance-preserving maps between the Poincaré ball, hyperboloid, and Proper
Velocity (PV) models — all coordinate models of the same hyperbolic space.
Provides Poincaré ↔ Hyperboloid, Poincaré ↔ PV (PVNN Eq. 4), and the direct
Hyperboloid ↔ PV map (PV coordinates are the space-like part of the 4-velocity).

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

### Proper Velocity Operations

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import ProperVelocity

pv = ProperVelocity()
c = 1.0

# Points live in unconstrained R^n — no projection step needed
x = jnp.array([0.3, 0.5])
y = jnp.array([-0.2, 0.4])

# Geodesic distance (asinh form — stable over all of R^n)
d = pv.dist(x, y, c)

# Exp/log maps at the origin
v = jnp.array([0.1, -0.2])
y_moved = pv.expmap_0(v, c)
v_recovered = pv.logmap_0(y_moved, c)

# Euclidean gradient -> Riemannian gradient under the PV metric
grad_euc = jnp.array([1.0, 0.0])
grad_riem = pv.egrad2rgrad(grad_euc, x, c)

# Retraction is exact Euclidean addition (PV is unconstrained)
x_next = pv.retraction(v, x, c)
assert jnp.allclose(x_next, x + v)
```

### Product Manifolds (Mixed Curvature)

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import (
    ProductManifold, Hyperboloid, Poincare, Euclidean,
)

# Build P = H^5(c=1.0) x P^3(c=0.1) x E^4
product = ProductManifold(
    (Hyperboloid(c=1.0), 5),  # 5 = ambient dim (d+1) for hyperboloid
    (Poincare(c=0.1), 3),     # 3 = spatial dim for poincaré
    (Euclidean(), 4),         # 4 = standard euclidean dim
)

# Per-factor curvatures must be passed at call time as a sequence.
c = product.curvatures             # (1.0, 0.1, 0.0) — static default
o = product.origin(c)              # shape (12,)

# Pythagorean product distance: d_P = sqrt(sum d_i^2)
x = product.origin(c)
y = product.origin(c)  # generated elsewhere; here we use o for illustration
d_l2 = product.dist(x, y, c)               # scalar
d_per_factor = product.component_dist(x, y, c)  # shape (3,) per-factor distances

# Batch with vmap: broadcast c with None across the batch.
dist_batch = jax.vmap(product.dist, in_axes=(0, 0, None))
# distances = dist_batch(xs, ys, c)

# Repeated-factor construction via from_signature
mixed = ProductManifold.from_signature(
    (Hyperboloid, 5, 4, 1.0),   # 4 copies of H^4(c=1.0)
    (Poincare,    3, 2, 0.1),   # 2 copies of P^3(c=0.1)
    (Euclidean,   4, 1),         # 1 copy of E^4
)
# For learnable curvature, build the sequence from LearnableCurvature calls on
# your nnx.Module:
#   c = (self.curv_h(), self.curv_p(), 0.0)
#   d = product.dist(x, y, c)
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
