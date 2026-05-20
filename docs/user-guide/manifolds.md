# Manifolds User Guide

Synthesis content for working with manifolds across the library — choosing a
manifold, conventions you have to get right, curvature workflows, and the
patterns that aren't obvious from any single docstring.

For per-method signatures and full API surface, see the
[Manifolds API reference](../api-reference/manifolds.md).

## Choosing a Manifold

| Use case | Recommended manifold | Why |
|---|---|---|
| Tree / hierarchy with bounded depth | `Poincare` (small `c`) | Bounded ball matches bounded data; conformal model is intuitive for visualization |
| Continuous-depth or unbounded radii | `Hyperboloid` or `ProperVelocity` | No boundary collapse for large norms |
| Heterogeneous structure (mixed tree + cycles + flat) | `ProductManifold` | Mixed curvatures fit mixed-structure data (Gu et al. 2019) |
| Cross-curvature transformations (`c_in != c_out`) | `Hyperboloid` + `HTCLinear` | Native cross-curvature support in HTC layers |
| Drop-in numerical stability | `ProperVelocity` | Unconstrained $\mathbb{R}^n$; no projection or constraint drift |
| You don't know which to pick | `Hyperboloid` (or `ProperVelocity`) | Robust at `c=1.0`; PV adds no-projection convenience |

!!! tip "Single best default"
    Start with `Hyperboloid(c=1.0)` for new hyperbolic models. It's well-behaved
    at the default curvature, has the fastest layers (`FGGLinear`, `LorentzConv2D`),
    and avoids the boundary-collapse issues of Poincaré at `c=1.0`.

## Convention Cheat-Sheet

The single biggest source of layer-construction bugs in Hyperbolix is the
**ambient vs. spatial dimension** distinction. Different layer families take
different conventions:

| Layer / Op family | Channel arg | Convention | Example: 32 spatial dims |
|---|---|---|---|
| `FGGLinear`, `LorentzConv2D`, `HTCLinear`, `HypLinearHyperboloid*` | `in_features` | **Ambient (d+1)** — includes time | `33` |
| `HRCBatchNorm`, `HRCLayerNorm` (Hyperboloid normalization) | `num_features` | **Spatial (d)** — excludes time | `32` |
| `HypLinearPoincare*`, `HypConv2DPoincare`, `HypRegressionPoincare*` | `in_dim` | **Spatial (d)** — Poincaré has no time | `32` |
| `HypLinearPV`, `HypConv2DPV`, `HypRegressionPV` | `in_dim` | **Spatial (d)** | `32` |
| `hyp_avg_pool2d` (Hyperboloid global pool) | NHWC channels | **Ambient (d+1)** | `33` |
| `ProductManifold` factor `dim` | per-factor | **Same as the factor's layer** (ambient for Hyperboloid, spatial otherwise) | `33` for `Hyperboloid`, `32` for `Poincare` |

!!! warning "HRC vs HTC normalization"
    Both `HRCBatchNorm` and the `HTC`-flavored normalizers exist, but they
    take **different conventions**:

    - **HRC** ops see only spatial components `x[..., 1:]` — pass spatial dim.
    - **HTC** ops see the full ambient point — pass ambient dim.

    Reconstructing a valid Hyperboloid point is handled internally for both;
    you only have to get the channel argument right at construction.

Curvature convention is uniform across all manifolds: `c > 0` means sectional
curvature $-c$ (so larger `c` → more curved). `Euclidean` ignores `c` entirely.

## Working with Curvature

### Static vs. learnable

Manifolds are **pure geometric utilities** (plain Python classes, not `nnx.Module`).
They hold a fixed curvature value. For **learnable curvature**, use the
`learnable_curvature()` / `get_curvature()` helpers and store the parameter
on your `nnx.Module`:

```python
from hyperbolix import learnable_curvature, get_curvature
from hyperbolix.manifolds import Hyperboloid, Poincare

# Fixed curvature (default) — manifold.c is a Python float
manifold = Hyperboloid(c=1.0)
manifold = Poincare(c=0.1)

# Learnable curvature: store the nnx.Param on your model
class Model(nnx.Module):
    def __init__(self, rngs):
        self.manifold = Hyperboloid(c=1.0)       # static geometric utility
        self.c_raw = learnable_curvature(1.0)     # nnx.Param on the model
        self.fc = FGGLinear(33, 65, rngs=rngs)

    def __call__(self, x):
        c = get_curvature(self.c_raw)             # softplus → positive
        return self.fc(x, c=c)
```

The curvature parameter is Euclidean and works with any standard `nnx.Optimizer`
(no Riemannian optimizer required):

```python
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
# self.c_raw is optimized alongside other params automatically.
```

### When `c=1.0` works and when it doesn't

| Manifold | `c=1.0` default behavior | Notes |
|---|---|---|
| `Hyperboloid` | Stable across most workloads | Unbounded; no boundary collapse |
| `ProperVelocity` | Stable | Unconstrained $\mathbb{R}^n$; PV's safe-norm formulation tolerates wide ranges |
| `Poincare` | **Often too aggressive for deep nets** | Conformal factor $\lambda = 2/(1 - c\|x\|^2)$ collapses near boundary, killing MLR signal |

For Poincaré in deep networks, the **van Spengler et al. (2023)** convention
is `init_c=0.1` with learnable per-layer curvatures:

```python
from hyperbolix import learnable_curvature, get_curvature

class HypResNetBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.manifold = Poincare(c=0.1)
        self.c_a = learnable_curvature(init_c=0.1)
        self.c_b = learnable_curvature(init_c=0.1)
        self.conv_a = HypConv2DPoincare(self.manifold, ..., rngs=rngs)
        self.conv_b = HypConv2DPoincare(self.manifold, ..., rngs=rngs)

    def __call__(self, x):
        h = self.conv_a(x, get_curvature(self.c_a))
        return self.conv_b(h, get_curvature(self.c_b))
```

### Curvature in `ProductManifold`

A `ProductManifold` has **no single `c`** — accessing `.c` raises `TypeError`.
Use one of:

```python
product.curvatures           # tuple of all factor curvatures
product.factors[i].c         # specific factor's curvature
product.component_dist(x, y) # per-factor distance vector (before reduction)
```

For learnable per-factor curvatures, store the parameters on your model:

```python
from hyperbolix import learnable_curvature, get_curvature
from hyperbolix.manifolds import Hyperboloid, Poincare, ProductManifold

class Model(nnx.Module):
    def __init__(self, rngs):
        self.pm = ProductManifold(
            (Hyperboloid(c=1.0), 3),
            (Poincare(c=0.5), 2),
        )
        self.c_h = learnable_curvature(init_c=1.0)
        self.c_p = learnable_curvature(init_c=0.5)

    def __call__(self, x, y):
        c_h = get_curvature(self.c_h)
        c_p = get_curvature(self.c_p)
        x_parts = self.pm.split(x)
        y_parts = self.pm.split(y)
        d_h = self.pm.factors[0].dist(x_parts[0], y_parts[0], c_h)
        d_p = self.pm.factors[1].dist(x_parts[1], y_parts[1], c_p)
        return d_h + d_p
```

### Custom curvature parameterization

The built-in `learnable_curvature()` uses softplus reparameterization, which
guarantees positivity without clamping. If you need a different
parameterization (e.g., the log/exp scheme from
[MERU](https://arxiv.org/abs/2304.09172)), manage the curvature parameter
yourself and pass it to manifold operations:

```python
import math
import jax.numpy as jnp
from flax import nnx

class MERUStyleModel(nnx.Module):
    def __init__(self, init_c: float = 1.0):
        self.manifold = Hyperboloid()  # fixed c, unused
        self._log_c = nnx.Param(jnp.array(math.log(init_c)))
        self._c_bounds = (math.log(init_c / 10), math.log(init_c * 10))

    @property
    def c(self):
        return jnp.exp(jnp.clip(self._log_c, *self._c_bounds))

    def __call__(self, x, y):
        return self.manifold.dist(x, y, self.c)
```

All manifold methods accept `c` as a dynamic argument, so the manifold's
own `.c` value is bypassed entirely.

## Going Euclidean → Manifold

A frequent confusion: there are several ways to map a Euclidean feature
vector onto a hyperbolic manifold, and they are **not interchangeable**.

| Pattern | When to use | Caveats |
|---|---|---|
| `manifold.expmap_0(v, c)` | Small-norm Euclidean features near the origin | `expmap_0` involves $\sinh$/$\cosh$; large norms blow up exponentially |
| Constraint projection (Hyperboloid only): `[sqrt(\|\|x\|\|² + 1/c), x]` | Large-norm features, CNN feature maps, ImageNet-scale | Not a geodesic; just enforces the Lorentz constraint. Use when expmap_0 would saturate |
| `manifold.proj(x, c)` | Cleaning up an already-near-manifold point (numerical drift) | Identity for Euclidean; clamp-to-ball for Poincaré; constraint enforcement for Hyperboloid |
| `manifold.expmap(v, x, c)` | Moving along a geodesic from an existing manifold point `x` | Requires `x` already on-manifold |

```python
# Pattern A: small-norm features (typical for an embedding layer or MLP head)
x_euclidean = nnx.Linear(input_dim, 32, rngs=rngs)(x)  # small-norm
x_manifold = jax.vmap(lambda v: hyperboloid.expmap_0(jnp.concatenate([jnp.zeros(1), v]), c))(x_euclidean)

# Pattern B: large-norm features (typical for a CNN backbone)
features = cnn_stem(images)                            # large activations
time_coord = jnp.sqrt(jnp.sum(features**2, axis=-1, keepdims=True) + 1.0 / c)
x_manifold = jnp.concatenate([time_coord, features], axis=-1)
```

Both patterns appear in the MNIST benchmark (`benchmarks/bench_mnist_hyperboloid.py`):
`FHCNNHybrid` uses Pattern A after a small Euclidean embedding;
`FullyHyperbolicCNN_*` uses Pattern B per-pixel from raw image values.

## Hyperboloid ↔ Poincaré: Use the Isometry

When you need to switch models (e.g., move from a Hyperboloid CNN backbone
to a Poincaré classifier head), **do not** route through `logmap_0 → expmap_0`
on the other manifold. Use the direct isometry:

```python
from hyperbolix.manifolds import isometry_mappings

# Hyperboloid (d+1) → Poincaré (d) — distance preserving, ~10x faster than logmap/expmap
x_poincare = isometry_mappings.hyperboloid_to_poincare(x_hyperboloid, c)

# Reverse direction
x_hyperboloid = isometry_mappings.poincare_to_hyperboloid(x_poincare, c)
```

The `logmap → expmap` route is lossy (tangent-space round-trip accumulates
numerical error) and slower. The isometry is exact.

## Common Pitfalls

### 1. Raw `jnp.acosh` / `jnp.atanh` instead of the hyperbolix versions

```python
# ❌ NaN at domain boundaries (inner_product < 1 for acosh, |x| >= 1 for atanh)
d = jnp.acosh(inner_product)

# ✅ Clamped and stable
from hyperbolix.utils.math_utils import acosh, atanh
d = acosh(inner_product)
```

Always use `hyperbolix.utils.math_utils` (`acosh`, `atanh`, `sinh`, `cosh`)
when implementing custom hyperbolic ops.

### 2. Re-implementing distance from scratch

```python
# ❌ Hand-rolled — likely numerically unstable
norm_sq = jnp.sum((x - y) ** 2, axis=-1)
d = some_formula(norm_sq)

# ✅ Use the manifold's vetted, dtype-aware implementation
d = poincare.dist(x, y, c)
```

### 3. Riemannian optimizer for layers that don't need one

Most modern layers (`FGG*`, `*PP`, `HRC*`, `HTC*`, `*PV`) parameterize weights
in **Euclidean space** internally. They do NOT need a Riemannian optimizer:

```python
# ✅ Standard Euclidean Adam works for all modern layers
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
```

Only use `riemannian_adam` / `riemannian_sgd` when parameters live **directly
on the manifold** — typically hyperbolic embedding tables wrapped in
`ManifoldParam(value, manifold=..., curvature=...)`. The legacy
`HypLinearPoincare` (Ganea-style) is the only NN layer whose weights are
manifold-valued; prefer `HypLinearPoincarePP` / `FGGLinear` to avoid the
need entirely.

### 4. Skipping `proj` after manual point construction

If you build a hyperboloid point by hand from spatial coordinates
(`[sqrt(...), spatial...]`), it's correct in float64 but may drift in float32.
After several training steps, accumulated drift can violate
`<x, x>_L = -1/c`. Periodic re-projection via `manifold.proj` is cheap and
keeps points on-manifold:

```python
x = manifold.proj(x, c)  # cheap; idempotent on already-valid points
```

### 5. Picking a slow layer when a fast equivalent exists

Within each layer family, prefer the variant with the highest reported speed
in the [API reference](../api-reference/nn-layers.md):

| Family | Slow | Fast |
|---|---|---|
| Hyperboloid linear | `HypLinearHyperboloid*` | `FGGLinear` (~3× faster), `HTCLinear` (cross-curvature) |
| Hyperboloid convolution | `HypConv2DHyperboloid` (HCat) | `LorentzConv2D` (~2.5× faster), `FGGConv2D` |
| Poincaré convolution | `HypConv2DPoincare` | (no faster variant; this is the standard) |

## See Also

- **[API Reference: Manifolds](../api-reference/manifolds.md)** — full method signatures and docstrings.
- **[Numerical Stability Guide](numerical-stability.md)** — when to use float64, conformal factor pitfalls, clamping strategies.
- **[Batching & JIT Guide](batching-jit.md)** — `jax.vmap` patterns, JIT static arguments, version_idx static-ness.
- **[Training Workflows](training-workflows.md)** *(WIP)* — end-to-end training examples.
