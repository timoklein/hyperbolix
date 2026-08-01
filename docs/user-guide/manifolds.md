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
| Spherical / cyclic data, or learning the *sign* of curvature | `Stereographic` | One signed-`c` manifold spans hyperbolic (`c>0`), Euclidean (`c=0`), and spherical (`c<0`), differentiable across zero (Bachmann et al. 2020) |
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
| `FGGLinear`, `HTCLinear` | `in_features` | **Ambient (d+1)** — includes time | `33` |
| `LorentzConv2D` | `in_channels` | **Ambient (d+1)** — includes time | `33` |
| `HypLinearHyperboloid*` | `in_dim` | **Ambient (d+1)** — includes time | `33` |
| `HRCBatchNorm`, `HRCLayerNorm` (Hyperboloid normalization) | `num_features` | **Spatial (d)** — excludes time | `32` |
| `HypLinearPoincare*`, `HypConv2DPoincare`, `HypRegressionPoincare*` | `in_dim` | **Spatial (d)** — Poincaré has no time | `32` |
| `HypLinearPV`, `HypConv2DPV`, `HypRegressionPV` | `in_dim` | **Spatial (d)** | `32` |
| `hyp_avg_pool2d` (Hyperboloid global pool) | NHWC channels | **Ambient (d+1)** | `33` |
| `hyp_flatten2d` (Hyperboloid LogCat flatten) | NHWC channels | **Ambient (d+1)** in, `H·W·d + 1` out | `33` in, `H·W·32 + 1` out |
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
The one exception is `Stereographic`, which takes a **signed** `c` that *extends*
this same convention across zero: `c > 0` hyperbolic (identical to `Poincare(c)`),
`c = 0` Euclidean, `c < 0` spherical. See the
[κ-Stereographic API reference](../api-reference/manifolds.md) for the factor-2
Euclidean-limit caveat and the sign-flip relative to the paper's $\kappa$.

## Working with Curvature

### Static vs. learnable

Manifolds are **pure geometric utilities** (plain Python classes, not `nnx.Module`).
They hold a fixed curvature value. For **learnable curvature**, use the
`LearnableCurvature` module and assign one instance per distinct curvature
in your model:

```python
from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Hyperboloid, Poincare

# Fixed curvature (default) — manifold.c is a Python float
manifold = Hyperboloid(c=1.0)
manifold = Poincare(c=0.1)

# Learnable curvature: one LearnableCurvature per distinct c on your model
class Model(nnx.Module):
    def __init__(self, rngs):
        self.manifold = Hyperboloid(c=1.0)               # static geometric utility
        self.curvature = LearnableCurvature(init_c=1.0)  # nnx.Module on the model
        self.fc = FGGLinear(33, 65, rngs=rngs)

    def __call__(self, x):
        c = self.curvature()                              # softplus → positive, clamped
        return self.fc(x, c=c)
```

The underlying raw parameter is Euclidean and works with any standard
`nnx.Optimizer` (no Riemannian optimizer required):

```python
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
# self.curvature.raw is optimized alongside other params automatically.
```

### Choosing a parameterization

| Parameterization | Formula | Range | Gradient w.r.t. raw | When to prefer |
|---|---|---|---|---|
| `"softplus"` (default) | `c = softplus(raw)` | `c > 0` | `sigmoid(raw) ∈ (0, 1)` — bounded | Supervised training; van Spengler 2023 convention |
| `"log"` | `c = exp(raw)` | `c > 0` | `c` — scale-invariant | RL/compiled loops; `c` spans orders of magnitude; MERU convention |
| `"identity"` | `c = raw` | **signed** (`c ⋛ 0`) | `1` — crosses zero | `Stereographic` manifold: learn hyperbolic/Euclidean/spherical from data |

```python
self.curvature = LearnableCurvature(init_c=1.0, parameterization="log")
# Signed curvature for the Stereographic manifold (c<0 spherical, 0 Euclidean, c>0 hyperbolic):
self.kappa = LearnableCurvature(init_c=-1.0, parameterization="identity")
```

The positive parameterizations apply the default clamp `[0.1, 10.0]` to the
recovered `c` (not the raw parameter), giving a hard stability guard.
`"identity"` instead uses a symmetric magnitude cap `[-10.0, 10.0]` that
*includes* `0`, so it never forbids the Euclidean/spherical half. Pass
`c_min=None, c_max=None` to disable, or set tighter bounds to fit your workload.

### When `c=1.0` works and when it doesn't

| Manifold | `c=1.0` default behavior | Notes |
|---|---|---|
| `Hyperboloid` | Stable across most workloads | Unbounded; no boundary collapse |
| `ProperVelocity` | Stable | Unconstrained $\mathbb{R}^n$; PV's safe-norm formulation tolerates wide ranges |
| `Poincare` | **Often too aggressive for deep nets** | Conformal factor $\lambda = 2/(1 - c\|x\|^2)$ collapses near boundary, killing MLR signal |

For Poincaré in deep networks, the **van Spengler et al. (2023)** convention
is `init_c=0.1` with learnable per-layer curvatures (one `LearnableCurvature`
per layer — do **not** share a single instance across layers, see the
compiled-loops note below):

```python
from hyperbolix import LearnableCurvature

class HypResNetBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.manifold = Poincare(c=0.1)
        self.curv_a = LearnableCurvature(init_c=0.1)
        self.curv_b = LearnableCurvature(init_c=0.1)
        self.conv_a = HypConv2DPoincare(self.manifold, ..., rngs=rngs)
        self.conv_b = HypConv2DPoincare(self.manifold, ..., rngs=rngs)

    def __call__(self, x):
        h = self.conv_a(x, self.curv_a())
        return self.conv_b(h, self.curv_b())
```

### Curvature in `ProductManifold`

A `ProductManifold` has **no single `c`**: it has one curvature per factor.
Every geometry method takes a positional `c` argument that must be a sequence
of length `n_factors` — there is no scalar fallback, no default, and no `.c`
attribute on the product. This is intentional: it forces the curvature choice
to be explicit at every call site, and it makes static and learnable
curvatures look identical to readers. The protocol-level `Curvature` type
unions the scalar shape (used by `Poincare`/`Hyperboloid`/`ProperVelocity`/
`Euclidean`) with the sequence shape (used by `ProductManifold`), so
`ProductManifold` satisfies the `Manifold` protocol and generic code typed
against `Manifold` accepts the product too.

```python
product.curvatures            # tuple of factor-stored curvatures — pass as c when static
product.factors[i].c          # specific factor's stored curvature
product.dist(x, y, c)         # c: sequence of length n_factors
product.component_dist(x, y, c)  # per-factor distance vector before reduction
```

For **static** factor curvatures, pass `product.curvatures`:

```python
product = ProductManifold((Hyperboloid(c=1.0), 5), (Poincare(c=0.1), 3))
c = product.curvatures                   # (1.0, 0.1)
d = product.dist(x, y, c)
```

For **learnable** per-factor curvatures, instantiate one `LearnableCurvature`
per factor on your model and build the sequence in `__call__`:

```python
from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Hyperboloid, Poincare, ProductManifold

class Model(nnx.Module):
    def __init__(self, rngs):
        self.pm = ProductManifold(
            (Hyperboloid(c=1.0), 3),
            (Poincare(c=0.5), 2),
        )
        self.curv_h = LearnableCurvature(init_c=1.0)
        self.curv_p = LearnableCurvature(init_c=0.5)

    @property
    def c(self):
        return (self.curv_h(), self.curv_p())

    def __call__(self, x, y):
        return self.pm.dist(x, y, self.c)
```

The curvature tuple is a JAX pytree, so `jax.jit(self.pm.dist)(x, y, c)` and
`jax.vmap(self.pm.dist, in_axes=(0, 0, None))(xs, ys, c)` work without any
`static_argnames` — broadcast the whole tuple with `None` to vmap over batched
points but constant curvatures.

!!! note "Factor `c` is an initial value only"
    The `c=...` you pass to a factor at construction (`Hyperboloid(c=1.0)`)
    is stored on the factor and exposed via `product.curvatures`, but
    `ProductManifold` never reads it in its geometry methods. Treat it as a
    *default that you choose to thread through via `product.curvatures`* —
    not as a value the product silently uses.

### Learnable curvature in compiled training loops (`nnx.scan` / `nnx.fori_loop`)

Long compiled training loops (RL agents, episode rollouts, multi-step
training kernels) have two stability concerns that motivate the
`LearnableCurvature` defaults:

1. **Sharing rule**: Assigning the **same** `LearnableCurvature` instance
   to multiple fields creates a shared reference in the NNX pytree, which
   breaks `nnx.scan` / `nnx.fori_loop` with `ValueError: Dict key mismatch`.
   This is the same failure mode that motivated the manifold refactor.
   Always instantiate a fresh `LearnableCurvature` per location where you
   want a distinct learnable `c`. (The manifold itself is a plain Python
   class with no NNX state, so sharing the manifold across layers is
   always safe.)

2. **Clamp guard**: Over millions of gradient steps, an unclamped curvature
   can drift to `c → 0` (effectively Euclidean) or `c → ∞` (numerical
   blow-up). The default `[0.1, 10.0]` bounds — applied directly to `c`,
   not to the raw parameter — cover the entire useful hyperbolic geometry
   range without losing expressivity.

The recommended pattern for a compiled RL loop:

```python
from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypLinearPoincarePP

manifold = Poincare(c=0.1)  # shared across layers — safe (plain class)

class HypPolicy(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.manifold = manifold
        # Log parameterization: scale-invariant gradient (dc/draw = c).
        # Default clamp [0.1, 10.0] is the stability guard.
        self.curvature = LearnableCurvature(
            init_c=0.1, parameterization="log",
        )
        self.l1 = HypLinearPoincarePP(manifold, 4, 4, rngs=rngs)
        self.l2 = HypLinearPoincarePP(manifold, 4, 4, rngs=rngs)

    def __call__(self, x):
        c = self.curvature()
        h = self.l1(x, c)
        return self.l2(h, c)

# Standard Euclidean optimizer — self.curvature.raw is updated like any param.
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# Training loop survives nnx.fori_loop / nnx.scan because:
# 1. The manifold is a plain class (not in the pytree).
# 2. LearnableCurvature lives at exactly one path on the model.
# 3. The clamp prevents pathological values from accumulating.
```

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

## Switching Models: Use the Isometry

When you need to switch models (e.g., move from a Hyperboloid CNN backbone
to a Poincaré classifier head, or lift Euclidean features into the unconstrained
PV space), **do not** route through `logmap_0 → expmap_0` on the other manifold.
Use the direct isometries — they are exact and distance-preserving:

```python
from hyperbolix.manifolds import isometry_mappings

# Hyperboloid (d+1) ↔ Poincaré (d) — ~10x faster than logmap/expmap
x_poincare = isometry_mappings.hyperboloid_to_poincare(x_hyperboloid, c)
x_hyperboloid = isometry_mappings.poincare_to_hyperboloid(x_poincare, c)

# Proper Velocity (d) ↔ Poincaré (d)   (PVNN Eq. 4)
x_pv = isometry_mappings.poincare_to_pv(x_poincare, c)
x_poincare = isometry_mappings.pv_to_poincare(x_pv, c)

# Proper Velocity (d) ↔ Hyperboloid (d+1) — direct: PV coords are the
# space-like part of the 4-velocity, so this is just a concat / slice.
x_hyperboloid = isometry_mappings.pv_to_hyperboloid(x_pv, c)        # add time = √(1/c + ‖x‖²)
x_pv = isometry_mappings.hyperboloid_to_pv(x_hyperboloid, c)        # drop the time component
```

All single-point functions; batch with `jax.vmap(fn, in_axes=(0, None))`.

The `logmap → expmap` route is lossy (tangent-space round-trip accumulates
numerical error) and slower. The isometries are exact and mutually consistent —
`pv_to_hyperboloid` equals `poincare_to_hyperboloid ∘ pv_to_poincare`.

!!! tip "Why PV for numerically hard regimes"
    The Poincaré ball is bounded (`‖y‖² < 1/c`) and the hyperboloid time
    component grows exponentially with distance, so both can be unstable far from
    the origin. PV space is unconstrained `ℝⁿ`, so converting *to* PV
    (`poincare_to_pv` / `hyperboloid_to_pv`) is a stable way to operate on points
    near the boundary — at the cost of an unbounded coordinate, which is expected.

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
`HypLinearPoincare` and `HypRegressionPoincare` (Ganea-style) are the only NN
layers with a manifold-valued **bias** — their kernels stay Euclidean;
prefer `HypLinearPoincarePP` / `HypRegressionPoincarePP` / `FGGLinear` to
avoid the need entirely.

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
in the [API reference](../api-reference/nn-layers/index.md):

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
