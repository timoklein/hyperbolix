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
| No projection or constraint drift | `ProperVelocity` | Unconstrained $\mathbb{R}^n$; pairwise ops go through the hyperboloid, so the accuracy is the hyperboloid's |
| Spherical / cyclic data, or learning the *sign* of curvature | `Stereographic` | One signed-`c` manifold spans hyperbolic (`c>0`), Euclidean (`c=0`), and spherical (`c<0`), differentiable across zero (Bachmann et al. 2020) |
| Straight-line geodesics, Einstein midpoint aggregation, or porting Klein-model code (Mao et al. 2024; Zhang et al. 2026) | `Klein` | Geodesics are chords of the ball; `einstein_midpoint` is a closed-form weighted mean. Scaled radius $\sqrt{c}\,d_0$ is limited to 6.32 in float32 (13.86 in float64) at `c=1`, half the Poincaré ball's — see the [numerical-stability guide](numerical-stability.md#klein-numerics) |
| Horosphere / vertical-geodesic structure (horospheres centred at infinity are the planes $x_n = \text{const}$), or porting half-space code (HTorch) | `HalfSpace` | Conformal metric $\lVert dx\rVert^2/(c\,x_n^2)$ with the height as the last coordinate and origin $e_n/\sqrt{c}$. Pairwise ops are cancellation-free, and points on the vertical axis through the origin are stored to full precision at every height in the dtype's normal range. Pairwise `dist`/`logmap` return `inf`/NaN past scaled distance $\sqrt{c}\,d = 88.7$ in float32 (709.8 in float64) — see the [numerical-stability guide](numerical-stability.md#halfspace-numerics) |
| You don't know which to pick | `Hyperboloid` (or `ProperVelocity`) | Robust at `c=1.0`; PV adds no-projection convenience |

!!! tip "Single best default"
    Start with `Hyperboloid()` and `c = 1.0` passed at call time for new hyperbolic models. It's well-behaved
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
| `HypLinearPoincare*`, `HypConv2DPoincare`, `HypRegressionPoincare*` | `in_dim` (linear/regression), `in_channels` (conv) | **Spatial (d)** — Poincaré has no time | `32` |
| `HypLinearPV`, `HypConv2DPV`, `HypRegressionPV` | `in_dim` (linear/regression), `in_channels` (conv) | **Spatial (d)** | `32` |
| `Klein` points and the `klein_to_*` / `*_to_klein` maps (no Klein layers) | point shape | **Spatial (d)** — a ball point like Poincaré | `32` |
| `HalfSpace` points and the `halfspace_to_*` / `*_to_halfspace` maps (no half-space layers) | point shape | **Spatial (d)** — the height $x_n > 0$ is the **last** coordinate; the origin is $e_n/\sqrt{c}$, not 0 | `32` |
| `hyp_avg_pool2d` (Hyperboloid global pool) | NHWC channels | **Ambient (d+1)** | `33` |
| `hyp_flatten2d` (Hyperboloid LogCat flatten) | NHWC channels | **Ambient (d+1)** in, `H·W·d + 1` out | `33` in, `H·W·32 + 1` out |
| `ProductManifold` factor `dim` | per-factor | **Same as the factor's layer** (ambient for Hyperboloid, spatial otherwise) | `33` for `Hyperboloid`, `32` for `Poincare` |

!!! warning "HRC vs HTC normalization"
    `HRCBatchNorm`/`HRCLayerNorm`/`HRCRMSNorm` take the full hyperboloid point
    but normalize only `x[..., 1:]`, so `num_features` is the spatial `d`.
    `HTCLinear` acts on the full point, so `in_features` is `d+1`. Both rebuild
    the time coordinate internally.

Curvature convention is uniform across all manifolds: `c > 0` means sectional
curvature $-c$ (so larger `c` → more curved). `Euclidean` ignores `c` entirely.
The one exception is `Stereographic`, which takes a **signed** `c` that *extends*
this same convention across zero: `c > 0` hyperbolic (the `Poincare(c)` geometry,
with some ops equal only to rounding), `c = 0` Euclidean, `c < 0` spherical. See the
[κ-Stereographic API reference](../api-reference/manifolds.md) for the factor-2
Euclidean-limit caveat and the sign-flip relative to the paper's $\kappa$.

## Working with Curvature

### Static vs. learnable

Manifolds are **pure geometric utilities** (plain Python classes, not `nnx.Module`).
They hold a fixed curvature value. For **learnable curvature**, use the
`LearnableCurvature` module and assign one instance per distinct curvature
in your model:

```python
from flax import nnx

from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.nn_layers import FGGLinear

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
        c = self.curvature()                              # exp → positive, clamped
        return self.fc(x, c=c)
```

The underlying raw parameter is Euclidean and works with any standard
`nnx.Optimizer` (no Riemannian optimizer required):

```python
import optax

model = Model(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
# self.curvature.raw is optimized alongside other params automatically.
```

### Choosing a parameterization

| Parameterization | Formula | Range | Gradient w.r.t. raw | When to prefer |
|---|---|---|---|---|
| `"log"` (default) | `c = exp(raw)` | `c > 0` | `c` — scale-invariant | RL/compiled loops; `c` spans orders of magnitude; MERU convention |
| `"softplus"` | `c = softplus(raw)` | `c > 0` | `sigmoid(raw) ∈ (0, 1)` — bounded | geoopt `Stereographic`/`PoincareBall` convention; also the code default in the van Spengler et al. 2023 Poincare ResNet reference implementation |
| `"identity"` | `c = raw` | **signed** (`c ⋛ 0`) | `1` — crosses zero | `Stereographic` manifold: learn hyperbolic/Euclidean/spherical from data |

```python
self.curvature = LearnableCurvature(init_c=1.0, parameterization="log")
# Signed curvature for the Stereographic manifold (c<0 spherical, 0 Euclidean, c>0 hyperbolic):
self.kappa = LearnableCurvature(init_c=-1.0, parameterization="identity")
```

The positive parameterizations clamp the recovered `c` (not the raw parameter)
to a decade either side of `init_c`, `[init_c / 10, init_c * 10]` (`[0.1, 10]`
at the default `init_c=1.0`), giving a hard stability guard. `"identity"`
instead uses a symmetric magnitude cap `[-10.0, 10.0]` that *includes* `0`, so
it never forbids the Euclidean/spherical half. Pass `c_min`/`c_max` to set other
bounds, or `None` to disable one. The clamp does not freeze `c` on a bound:
it blocks only a gradient that would push `c` out, so `c` comes off the bound
once the loss pulls it back inside (projected gradient descent, with no extra
call in the train step).

### When `c=1.0` works and when it doesn't

| Manifold | `c=1.0` default behavior | Notes |
|---|---|---|
| `Hyperboloid` | Stable across most workloads | Unbounded; no boundary collapse |
| `ProperVelocity` | Stable | Unconstrained $\mathbb{R}^n$; PV's safe-norm formulation tolerates wide ranges |
| `Poincare` | **Often too aggressive for deep nets** | The conformal factor $\lambda = 2/(1 - c\|x\|^2)$ grows without bound near the boundary, and the ball chart stops at scaled radius ≈ 12.6 (float32) |

For Poincaré in deep networks, **van Spengler et al. (2023)** report `c=0.1`
as their best *fixed* curvature (Sec. 4.2 sweeps `c ∈ {1, 0.1, 0.01}`); their
released code instead defaults to a *learnable* per-layer curvature via
geoopt's softplus reparameterization. To reproduce that learnable-curvature
setup at the same scale (one `LearnableCurvature` per layer — do **not**
share a single instance across layers, see the compiled-loops note below).
geoopt does not clamp `c`; pass `c_min=None, c_max=None` to match it exactly.

```python
from flax import nnx

from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypConv2DPoincare

class HypResNetBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.manifold = Poincare(c=0.1)
        self.curv_a = LearnableCurvature(init_c=0.1, parameterization="softplus")
        self.curv_b = LearnableCurvature(init_c=0.1, parameterization="softplus")
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
unions the scalar shape (used by every single manifold: `Poincare`,
`Hyperboloid`, `ProperVelocity`, `Klein`, `HalfSpace`, `Stereographic`,
`Euclidean`) with the sequence shape (used by `ProductManifold`).
`isinstance(product, Manifold)` is `True`, but `Manifold` is typed with a
scalar `c`; annotate code that passes a per-factor sequence as
`ProductManifold`.

```python
product.curvatures            # tuple of factor-stored curvatures — pass as c when static
product.factors[i].c          # specific factor's stored curvature
product.dist(x, y, c)         # c: sequence of length n_factors
product.component_dist(x, y, c)  # per-factor distance vector before reduction
```

For **static** factor curvatures, pass `product.curvatures`:

```python
from hyperbolix.manifolds import Hyperboloid, Poincare, ProductManifold

product = ProductManifold((Hyperboloid(c=1.0), 5), (Poincare(c=0.1), 3))
c = product.curvatures                   # (1.0, 0.1)
x = y = product.origin(c)                # flat points of shape (8,)
d = product.dist(x, y, c)
```

For **learnable** per-factor curvatures, instantiate one `LearnableCurvature`
per factor on your model and build the sequence in `__call__`:

```python
from flax import nnx

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

Assigning the **same** `LearnableCurvature` instance to multiple fields
creates a shared reference in the NNX pytree, which breaks `nnx.scan` /
`nnx.fori_loop` with `ValueError: Dict key mismatch`. Instantiate a fresh
`LearnableCurvature` per location where you want a distinct learnable `c`.
The manifold is a plain Python class with no NNX state, so sharing it across
layers is always safe. The default clamp (see
[Choosing a parameterization](#choosing-a-parameterization)) keeps `c` from
drifting over long runs.

The recommended pattern for a compiled RL loop:

```python
import optax
from flax import nnx

from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypLinearPoincarePP

manifold = Poincare(c=0.1)  # shared across layers — safe (plain class)

class HypPolicy(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.manifold = manifold
        # Log parameterization: scale-invariant gradient (dc/draw = c).
        # The default clamp [0.01, 1.0], a decade either side of init_c, is the stability guard.
        self.curvature = LearnableCurvature(init_c=0.1, parameterization="log")
        # l1 takes the Euclidean observation as a tangent vector; l2 takes l1's ball point.
        self.l1 = HypLinearPoincarePP(manifold, 4, 4, rngs=rngs, input_space="tangent")
        self.l2 = HypLinearPoincarePP(manifold, 4, 4, rngs=rngs)

    def __call__(self, x):  # x: (B, 4) Euclidean observations
        c = self.curvature()
        h = self.l1(x, c)
        return self.l2(h, c)

model = HypPolicy(nnx.Rngs(0))
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
| Constraint projection (Hyperboloid only): `[sqrt(\|\|x\|\|² + 1/c), x]` | Large-norm features, CNN feature maps, ImageNet-scale | Not a geodesic; just enforces the Lorentz constraint. Use when expmap_0 would saturate (this is `pv_to_hyperboloid`: the features are read as PV coordinates) |
| `manifold.proj(x, c)` | Cleaning up an already-near-manifold point (numerical drift) | Identity for Euclidean and ProperVelocity; clamp to the ball for Poincaré and Klein; height floor for HalfSpace; time coordinate rebuilt from the spatial part for Hyperboloid |
| `manifold.expmap(v, x, c)` | Moving along a geodesic from an existing manifold point `x` | Requires `x` already on-manifold |

```python
# Pattern A: small-norm features (typical for an embedding layer or MLP head)
x_euclidean = nnx.Linear(input_dim, 32, rngs=rngs)(x)  # small-norm
x_manifold = jax.vmap(lambda v: hyperboloid.expmap_0(hyperboloid.embed_spatial_0(v), c))(x_euclidean)

# Pattern B: large-norm features (typical for a CNN backbone)
features = cnn_stem(images)                            # large activations
# same as isometry_mappings.pv_to_hyperboloid per pixel
time_coord = jnp.sqrt(jnp.sum(features**2, axis=-1, keepdims=True) + 1.0 / c)
x_manifold = jnp.concatenate([time_coord, features], axis=-1)
```

Both patterns in practice: a hybrid CNN (Euclidean stem → hyperbolic head) uses
Pattern A after a small Euclidean embedding; a fully hyperbolic CNN uses
Pattern B per-pixel from raw image values.

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

# Proper Velocity (d) ↔ Hyperboloid (d+1): add / drop the time coordinate
x_hyperboloid = isometry_mappings.pv_to_hyperboloid(x_pv, c)
x_pv = isometry_mappings.hyperboloid_to_pv(x_hyperboloid, c)

# Klein (d) ↔ Poincaré (d): Einstein addition in Klein is Möbius addition in Poincaré
x_poincare = isometry_mappings.klein_to_poincare(x_klein, c)
x_klein = isometry_mappings.poincare_to_klein(x_poincare, c)

# Klein (d) ↔ Hyperboloid (d+1) and ↔ Proper Velocity (d)
x_hyperboloid = isometry_mappings.klein_to_hyperboloid(x_klein, c)
x_klein = isometry_mappings.hyperboloid_to_klein(x_hyperboloid, c)
x_pv = isometry_mappings.klein_to_pv(x_klein, c)
x_klein = isometry_mappings.pv_to_klein(x_pv, c)

# Half-space (d, height last) ↔ Poincaré (d): the Cayley transform, origin e_n/√c ↦ 0
x_poincare = isometry_mappings.halfspace_to_poincare(x_halfspace, c)
x_halfspace = isometry_mappings.poincare_to_halfspace(x_poincare, c)

# Half-space (d) ↔ Hyperboloid (d+1), ↔ Klein (d) and ↔ Proper Velocity (d)
x_hyperboloid = isometry_mappings.halfspace_to_hyperboloid(x_halfspace, c)
x_halfspace = isometry_mappings.hyperboloid_to_halfspace(x_hyperboloid, c)
x_klein = isometry_mappings.halfspace_to_klein(x_halfspace, c)
x_halfspace = isometry_mappings.klein_to_halfspace(x_klein, c)
x_pv = isometry_mappings.halfspace_to_pv(x_halfspace, c)
x_halfspace = isometry_mappings.pv_to_halfspace(x_pv, c)
```

All single-point functions; batch with `jax.vmap(fn, in_axes=(0, None))`.

The `logmap → expmap` route is lossy (tangent-space round-trip accumulates
numerical error) and slower. The isometries are exact and mutually consistent —
`pv_to_hyperboloid` equals `poincare_to_hyperboloid ∘ pv_to_poincare`,
`klein_to_hyperboloid` is `pv_to_hyperboloid ∘ klein_to_pv`, and `halfspace_to_pv` is
the spatial part of `halfspace_to_hyperboloid`.

!!! warning "Mapping into Klein halves the representable radius"
    Mapping into Klein halves the representable radius: a point past scaled radius 6.32
    (float32) / 13.86 (float64), at $c = 1$, lands outside `Klein.proj`'s margin, and the
    maps into Klein do not project. Call `Klein.proj` after mapping far points in. Details:
    [Klein chart ceiling](numerical-stability.md#klein-chart-ceiling).

!!! note "The half-space model: height last, origin at $e_n/\sqrt{c}$"
    A `HalfSpace` point keeps its height $x_n > 0$ in the last coordinate, and its origin is
    $e_n/\sqrt{c}$, not 0. Code written for the zero origin of `Poincare`, `Klein` or
    `ProperVelocity` should use `expmap_0`/`logmap_0`, which start from $e_n/\sqrt{c}$. None of the maps
    into the half-space projects; `HalfSpace.proj` floors the height at the dtype's smallest
    normal number. A worked example is in the
    [API reference](../api-reference/manifolds.md#halfspace-operations).

## Common Pitfalls

### 1. Textbook `acosh` / `atanh` forms instead of the manifold's primitives

```python
# ❌ Loses small separations in float32, even with a domain clamp
d = jnp.arccosh(-c * hyperboloid.minkowski_inner(x, y)) / jnp.sqrt(c)

# ✅ Cancellation-free: use the manifold's own primitive
d = hyperboloid.dist(x, y, c)
```

Build custom ops from the manifold's own primitives (`dist`, `logmap`,
`gyro_difference`, `Hyperboloid.dist`), or from `asinh` forms. `acosh(1 + t)`
and `atanh` near 1 lose small separations in float32 even with the
`math_utils` domain clamp. Use `math_utils.acosh`/`atanh` only where the
argument is far from 1.

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

### 4. Skipping `proj` after updating all ambient coordinates

Points drift off the hyperboloid when all ambient coordinates are updated,
e.g. by a manual update of an embedding table. `manifold.proj(x, c)` rebuilds
$x_0 = \sqrt{1/c + \lVert x_s\rVert^2}$ from the spatial part. A point you built
that way yourself is already projected.

```python
x = manifold.proj(x, c)  # cheap; idempotent on already-valid points
```

### 5. Picking a slow layer when a fast equivalent exists

Within each layer family, prefer the variant with the highest reported speed
in the [API reference](../api-reference/nn-layers/index.md):

| Family | Slow | Fast |
|---|---|---|
| Hyperboloid linear | `HypLinearHyperboloid*` | `FGGLinear`, `HTCLinear` (cross-curvature) |
| Hyperboloid convolution | `HypConv2DHyperboloid` (HCat) | `LorentzConv2D` (~2.5× faster), `FGGConv2D` |
| Poincaré convolution | `HypConv2DPoincare` | (no faster variant; this is the standard) |

## See Also

- **[API Reference: Manifolds](../api-reference/manifolds.md)** — full method signatures and docstrings.
- **[Numerical Stability Guide](numerical-stability.md)** — when to use float64, the conformal factor, and where each model's chart stops.
- **[Batching & JIT Guide](batching-jit.md)** — `jax.vmap` patterns, JIT static arguments, version_idx as a static vs. dynamic argument.
- **[Training Workflows](training-workflows.md)** — end-to-end training examples.
