# Neural Network Layers User Guide

Synthesis content for building hyperbolic networks — choosing among the 20+
layers, the boundary between Euclidean and hyperbolic computation, and the
composition patterns that aren't obvious from any single layer's docstring.

For per-layer signatures, init defaults, and call semantics, see the
[NN Layers API reference](../api-reference/nn-layers.md).

## Choosing a Layer

Three axes matter, in this order: **which manifold**, **fully-hyperbolic vs.
hybrid**, and **which speed/expressiveness trade-off** within that family. The
tables below collapse this into per-task decisions.

### Linear / Fully Connected

| Layer | Manifold | When to pick |
|---|---|---|
| `HTCLinear` | Hyperboloid | **Default** for hyperboloid FC — simple, robust, just works. Also supports cross-curvature (`c_in != c_out`) for Hypformer blocks |
| `FGGLinear` | Hyperboloid | Advanced — ~3× faster but trickier (init-sensitive, requires the `0.5·eye` default and `lorentz_kaiming` to converge). Use once you have a working HTC baseline |
| `HypLinearHyperboloidPP` | Hyperboloid | Deep hyperboloid networks following the HNN++ formulation |
| `HypLinearPoincarePP` | Poincaré | **Default** for Poincaré FC — Euclidean-parameterized weights, works with `optax.adam` |
| `HypLinearPoincare` | Poincaré | Legacy Ganea 2018 — manifold-valued weights, requires `riemannian_adam`. Prefer `PP` |
| `HypLinearPV` | Proper Velocity | PV networks — Euclidean weights, He init |

### Convolutional

| Layer | Manifold | When to pick |
|---|---|---|
| `HypConv2DHyperboloid` | Hyperboloid | **Default** for hyperboloid conv — well-established HCat formulation, robust across configurations |
| `HypConv2DHyperboloidPP` / `HypConv2DHyperboloidFHNN` | Hyperboloid | Variants of the HCat conv following specific paper formulations (HNN++, FHNN); reach for these when reproducing those papers |
| `FGGConv2D` | Hyperboloid | Advanced — HCat expressiveness with FGG's speed, but inherits FGG's init sensitivity |
| `HypConv2DPoincare` | Poincaré | Standard for Poincaré CNNs (no faster variant exists) |
| `HypConv2DPV` | Proper Velocity | PV CNNs — raw Euclidean patch concatenation, no β-scaling |
| `LorentzConv2D` | Hyperboloid | Legacy / benchmarking only — HRC-based hack kept for comparing against older code. Not recommended for new work |

### Regression / Classification Head

| Layer | When to pick |
|---|---|
| `HypRegressionHyperboloid` | **Default** for hyperboloid classification — well-established MLR over Lorentzian hyperplanes, just works |
| `FGGLorentzMLR` | FGG-style MLR head — pair with FGG linear stack if you've already adopted the FGG family |
| `HypRegressionPoincarePP` | **Default** for Poincaré classification — HNN++ formulation |
| `HypRegressionPV` | PV classification head — small `std=1e-2` init |
| `HypRegressionPoincare` | Legacy Ganea — prefer `PP` |

### Attention, Normalization, Positional Encoding

| Use case | Layer |
|---|---|
| Long sequences, linear-complexity attention | `HyperbolicLinearAttention` (O(N)) |
| Most geometrically faithful attention | `HyperbolicFullAttention` (O(N²)) |
| Softmax attention with hyperbolic queries/keys | `HyperbolicSoftmaxAttention` |
| Normalization between hyperboloid layers | `HRCLayerNorm`, `HRCRMSNorm`, `HRCBatchNorm` |
| Normalization between Poincaré conv layers | `PoincareBatchNorm2D` |
| Dropout on hyperboloid features | `HRCDropout` |
| Rotary positional encoding (hyperbolic) | `HyperbolicRoPE` / `hope` |
| Learnable positional encoding (Hypformer-style) | `HypformerPositionalEncoding` |

!!! tip "Single best default"
    For a new hyperbolic classifier:
    `HTCLinear` → `HRCLayerNorm` → `HTCLinear` → `HypRegressionHyperboloid` on
    Hyperboloid at `c=1.0`. The HTC/HRC family is the most robust starting
    point — simple, well-tuned defaults, and converges across a wide range of
    configurations. Move to FGG only if you need the speedup and are willing
    to babysit the init.

## Layer Families at a Glance

The four families differ in what space the *weights* live in, which controls
optimizer choice:

| Family | Weight space | Optimizer | Examples | Role |
|---|---|---|---|---|
| **HRC / HTC** (Hypformer) | Euclidean | `optax.adam` | `HTCLinear`, `HRC*`, normalization | **Robust starting family** |
| **HCat** (Bdeir 2023) | Euclidean | `optax.adam` | `HypConv2DHyperboloid*` | **Robust conv family** |
| **HNN++** (Shimizu 2020 / van Spengler 2023) | Euclidean | `optax.adam` | `*PP` variants on both manifolds | Standard for Poincaré |
| **PV** (Chen et al. 2026) | Euclidean | `optax.adam` | `HypLinearPV`, `HypConv2DPV` | Use when you want PV's unconstrained $\mathbb{R}^n$ |
| **FGG** (Klis et al. 2026) | Euclidean | `optax.adam` | `FGGLinear`, `FGGConv2D`, `FGGLorentzMLR` | Advanced — fastest, but init-sensitive |
| **Ganea (legacy)** | On Poincaré | `riemannian_adam` | `HypLinearPoincare`, `HypRegressionPoincare` | Legacy — prefer `PP` |

**Bottom line:** every modern layer parameterizes weights in Euclidean space.
Standard `optax.adam` works. See the [Riemannian Optimizers guide](optimizers.md)
*(WIP)* for the rare cases where manifold-valued parameters appear.

## Channel Conventions

The single most common bug in layer construction is passing the wrong channel
count — ambient (d+1) vs. spatial (d). The full table lives in the
[Manifolds guide](manifolds.md#convention-cheat-sheet); the short form:

- **Hyperboloid layers** (`FGGLinear`, `LorentzConv2D`, `HTCLinear`, `HypLinearHyperboloid*`) take **ambient (d+1)**.
- **HRC normalization** (`HRCLayerNorm`, `HRCBatchNorm`, etc.) takes **spatial (d)**.
- **Poincaré and PV layers** take **spatial (d)** (no time component).

Example: a hyperboloid network with 32 latent spatial dims:

```python
manifold = Hyperboloid(c=1.0)
fc1 = FGGLinear(in_features=33, out_features=33, rngs=rngs)   # ambient
norm = HRCLayerNorm(num_features=32, rngs=rngs)               # spatial
fc2 = FGGLinear(in_features=33, out_features=33, rngs=rngs)   # ambient
```

## Initialization Scales

Standard Euclidean inits (He, Xavier) are **too large for hyperbolic layers**:
they push the first-layer output toward the Poincaré boundary or far up the
hyperboloid, where distances and gradients explode. Each family ships with a
hyperbolic-aware default — keep it unless you have a reason to change it.

| Family | Default init | Rationale |
|---|---|---|
| `FGGLinear` | `lorentz_kaiming`, `std=sqrt(1/in_features)` (ambient); default initial weight is `0.5·eye`, bias `0.5` | Keeps the spacelike V matrix near identity; preserves initial point magnitudes |
| `HypLinear*PP` | Standard normal `std=1.0` (Shimizu 2020 reference) | Tuned empirically for HNN++ formulation |
| `HypLinearPoincare*` | Scaled normal `std=(2·in·out)^{-0.5}` (van Spengler 2023) | Keeps row norms small so outputs stay away from the boundary |
| `HypLinearHyperboloidFHCNN`, `HTCLinear` | Small uniform `U(-0.02, 0.02)` | Keeps points near the apex `[1/sqrt(c), 0, …]` initially |
| `HypLinear*PV` | He init | PV is unconstrained $\mathbb{R}^n$; standard scales work |
| `HypRegressionPV` | `std=1e-2` | MLR head: small scores at init |

!!! warning "Don't override init with He/Xavier on hyperbolic layers"
    If you wrap a hyperbolic layer in code that auto-applies a default Flax
    init, you will see `NaN` losses within the first few steps. The
    constructor's `kernel_init` argument exists for tuning, not for swapping
    in a Euclidean default.

## The Euclidean ↔ Hyperbolic Boundary

Most networks aren't *fully* hyperbolic. The question is **where you cross the
boundary** between Euclidean and hyperbolic computation:

| Pattern | Boundary location | Typical use case |
|---|---|---|
| **Fully hyperbolic** | Inputs are already on-manifold (e.g. embedding table on Poincaré) | Knowledge graph embeddings, hierarchy learning |
| **Hyperbolic head** | After a Euclidean backbone (CNN/Transformer) → `expmap_0` or constraint projection → hyperbolic classifier | ImageNet-scale CNNs with hyperbolic MLR (van Spengler 2023) |
| **Hyperbolic backbone** | At input via `expmap_0` per-pixel, then fully-hyperbolic through to a Euclidean logits layer | FullyHyperbolicCNN on MNIST |
| **Hybrid (sandwiched)** | Euclidean stem → small Euclidean embed → `expmap_0` → hyperbolic block → `logmap_0` → Euclidean head | When you want hyperbolic geometry only mid-network |

The boundary lift itself is covered in the
[Manifolds guide](manifolds.md#going-euclidean-manifold) (Pattern A vs B).
For the hybrid case on Hyperboloid, `HyperPPFeatureScaling` is the canonical
recipe to prepare Euclidean features before `expmap_0`:

```python
from hyperbolix.nn_layers import HyperPPFeatureScaling

scale = HyperPPFeatureScaling(
    dim=feature_dim, manifold_module=hyperboloid, rngs=rngs,
)
x_euclidean = scale(x_euclidean)               # RMSNorm + activation + dim scaling
x_manifold = jax.vmap(lambda v: hyperboloid.expmap_0(
    jnp.concatenate([jnp.zeros(1), v]), c
))(x_euclidean)
```

### Proper Velocity: when to use `expmap_0` (and when not to)

PV has its own rule because PV points live in unconstrained $\mathbb{R}^n$ —
there's no "outside the manifold" to project from. Whether you apply
`expmap_0` at the boundary depends on the rest of the architecture:

| Architecture | Apply `expmap_0` at input? | Why |
|---|---|---|
| **Fully hyperbolic PV** (PV layers all the way through) | ✅ **Yes**, once at the beginning | Establishes the proper-velocity coordinate frame; downstream PV layers assume their inputs were lifted from Euclidean tangent vectors |
| **Hybrid PV** (Euclidean backbone → PV head, or Euclidean ↔ PV alternating) | ❌ **No** — pass Euclidean features directly to PV layers | The PV layer's metric already accounts for the geometry of its inputs; an explicit `expmap_0` here is redundant and can hurt training |

In other words: `expmap_0` is the **once-per-network** entry into the PV
coordinate frame, not a per-layer adapter. If your network has a Euclidean
stem feeding a PV classifier, hand the raw Euclidean activations to the PV
layer; if your entire network is PV, lift once at the input and stay in PV
coordinates from there on.

## Composition Patterns

### Pattern 1: HTC hyperboloid classifier (recommended starter)

The HTC/HRC family is the most robust default — well-established, forgiving of
init, and converges across a wide range of configurations.

```python
class HTCClassifier(nnx.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, *, rngs: nnx.Rngs):
        # in_dim, hidden are AMBIENT (d+1)
        self.manifold = Hyperboloid(c=1.0)
        self.fc1 = HTCLinear(in_features=in_dim, out_features=hidden, rngs=rngs)
        self.norm = HRCLayerNorm(num_features=hidden - 1, rngs=rngs)  # SPATIAL
        self.fc2 = HTCLinear(in_features=hidden, out_features=hidden, rngs=rngs)
        self.head = HypRegressionHyperboloid(
            manifold_module=self.manifold,
            in_features=hidden, out_features=num_classes, rngs=rngs,
        )

    def __call__(self, x_BAi: jax.Array, c: float = 1.0) -> jax.Array:
        h = self.fc1(x_BAi, c)
        h = self.norm(h, c)
        h = self.fc2(h, c)
        return self.head(h, c)  # (B, num_classes) Euclidean logits
```

### Pattern 2: Hybrid CNN backbone + Poincaré head

```python
from hyperbolix import LearnableCurvature

class HybridCNN(nnx.Module):
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.stem = nnx.Conv(3, 64, kernel_size=(3, 3), rngs=rngs)  # Euclidean
        self.pool = lambda x: jnp.mean(x, axis=(1, 2))               # GAP
        self.poincare = Poincare(c=0.1)
        self.curvature = LearnableCurvature(init_c=0.1)              # per van Spengler
        self.head = HypRegressionPoincarePP(
            manifold_module=self.poincare,
            in_dim=64, out_dim=num_classes, rngs=rngs,
        )

    def __call__(self, images: jax.Array) -> jax.Array:
        c = self.curvature()
        features = self.pool(jax.nn.relu(self.stem(images)))  # (B, 64) Euclidean
        x_poincare = jax.vmap(self.poincare.expmap_0, in_axes=(0, None))(
            features, c,
        )
        return self.head(x_poincare, c)
```

### Pattern 3: Hyperbolic transformer block

```python
class HypTransformerBlock(nnx.Module):
    def __init__(self, dim_ambient: int, n_heads: int, *, rngs: nnx.Rngs):
        d_spatial = dim_ambient - 1
        self.attn_norm = HRCLayerNorm(num_features=d_spatial, rngs=rngs)
        self.attn = HyperbolicSoftmaxAttention(
            in_features=dim_ambient, n_heads=n_heads, rngs=rngs,
        )
        self.mlp_norm = HRCLayerNorm(num_features=d_spatial, rngs=rngs)
        self.mlp_in = HTCLinear(in_features=dim_ambient,
                                out_features=4 * dim_ambient, rngs=rngs)
        self.mlp_out = HTCLinear(in_features=4 * dim_ambient,
                                 out_features=dim_ambient, rngs=rngs)

    def __call__(self, x_BLAi: jax.Array, c: float) -> jax.Array:
        h = self.attn(self.attn_norm(x_BLAi, c), c)
        x_BLAi = lorentz_residual(x_BLAi, h, c)              # Möbius-style residual
        h = self.mlp_out(self.mlp_in(self.mlp_norm(x_BLAi, c), c), c)
        return lorentz_residual(x_BLAi, h, c)
```

### Pattern 4: Per-layer learnable curvature (deep Poincaré nets)

When stacking many Poincaré layers, give each block its own learnable curvature
to avoid the conformal-factor collapse near the boundary:

```python
from hyperbolix import LearnableCurvature

class HypResBlock(nnx.Module):
    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.manifold = Poincare(c=0.1)
        self.curv1 = LearnableCurvature(init_c=0.1)
        self.curv2 = LearnableCurvature(init_c=0.1)
        self.conv1 = HypConv2DPoincare(self.manifold, channels, channels,
                                       kernel_size=(3, 3), rngs=rngs)
        self.bn1 = PoincareBatchNorm2D(self.manifold, channels, rngs=rngs)
        self.conv2 = HypConv2DPoincare(self.manifold, channels, channels,
                                       kernel_size=(3, 3), rngs=rngs)
        self.bn2 = PoincareBatchNorm2D(self.manifold, channels, rngs=rngs)
```

## Common Pitfalls

### 1. Wrong channel count (ambient vs. spatial)

By far the most common construction bug. If a hyperboloid layer raises an
incomprehensible shape error during the first call, check whether you passed
spatial dim (`d`) where it wanted ambient (`d+1`), or vice versa.

### 2. Reaching for a Riemannian optimizer

Modern layers don't need one. The Euclidean defaults — `optax.adam`,
`optax.adamw` — work for FGG, HNN++, HRC/HTC, and PV layers. Use
`riemannian_adam` only when parameters live directly on a manifold (typically a
hyperbolic embedding table); see the [Optimizers guide](optimizers.md) *(WIP)*.

### 3. Forgetting `version_idx` is static

Several Poincaré ops (`dist`, `expmap`, `logmap`) take a `version_idx` selecting
between multiple formulations. Under JIT, this argument must be **static**:

```python
# ❌ Traces a new graph on every call
jit_fn = jax.jit(lambda x, y, idx: poincare.dist(x, y, c=1.0, version_idx=idx))

# ✅ Bind the variant before JIT-ing
from functools import partial
dist_v0 = partial(poincare.dist, version_idx=0)
jit_fn = jax.jit(dist_v0)
```

### 4. Mixing layer families incoherently

Stacking `HypLinearPoincare` (manifold-valued weights, expects
`riemannian_adam`) on top of `HypLinearPoincarePP` (Euclidean weights, expects
`optax.adam`) gives you a model where one optimizer is wrong for half the
parameters. Pick one family per network and stay in it; `mark_manifold_param`
lets the Riemannian optimizer auto-dispatch correctly if you genuinely need a
mix, but the simpler fix is to migrate the legacy layers to their `PP`
equivalents.

### 5. Using a Euclidean `Dropout` / `LayerNorm` on hyperboloid points

A point on the hyperboloid satisfies $\langle x, x \rangle_L = -1/c$ —
elementwise zeroing or affine normalization breaks the constraint and produces
silent NaNs downstream. Use the manifold-aware variants: `HRCDropout`,
`HRCLayerNorm`, `HRCRMSNorm`, `HRCBatchNorm`. (Poincaré has its own
`PoincareBatchNorm2D` for conv stacks.)

### 6. Forgetting to project the input

If you build a hyperboloid point by hand (e.g. constraint projection from a
Euclidean backbone) and feed it to a hyperbolic layer, float32 drift can violate
the Lorentz constraint after a few training steps. Cheap insurance:

```python
x_BAi = jax.vmap(self.manifold.proj, in_axes=(0, None))(x_BAi, c)
# ... feed into hyperbolic layers
```

`proj` is idempotent on valid points and adds negligible cost.

## See Also

- **[API Reference: NN Layers](../api-reference/nn-layers.md)** — full constructor and call signatures
- **[Manifolds Guide](manifolds.md)** — convention cheat-sheet, Euclidean→manifold lifts, isometry mappings
- **[Numerical Stability Guide](numerical-stability.md)** — when to use float64, clamping, safe norms
- **[Riemannian Optimizers Guide](optimizers.md)** *(WIP)* — when (rarely) you need Riemannian optimization
