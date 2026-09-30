# Neural Network Layers User Guide

Synthesis content for building hyperbolic networks — choosing among the 20+
layers, the boundary between Euclidean and hyperbolic computation, and the
composition patterns that aren't obvious from any single layer's docstring.

For per-layer signatures, init defaults, and call semantics, see the
[NN Layers API reference](../api-reference/nn-layers/index.md).

## Choosing a Layer

Three axes matter, in this order: **which manifold**, **fully-hyperbolic vs.
hybrid**, and **which speed/expressiveness trade-off** within that family. The
tables below collapse this into per-task decisions.

### Linear / Fully Connected

| Layer | Manifold | When to pick |
|---|---|---|
| `HTCLinear` | Hyperboloid | **Default** for hyperboloid FC — simple, robust, just works. Also supports cross-curvature (`c_in != c_out`) for Hypformer blocks |
| `FGGLinear` | Hyperboloid | Advanced — ~3× faster but init-sensitive. Defaults to the norm-preserving `fan_out` init (`init_bias=0.0`), which suits unnormalized stacks feeding a bounded projection; pass `reset_params="eye", init_bias=0.5` for the reference (BatchNorm-regime) init. Use once you have a working HTC baseline |
| `HypLinearHyperboloidPLFC` | Hyperboloid | Deep hyperboloid networks — point-to-hyperplane Lorentz FC (PLFC, Shi et al. 2026), the Lorentz analog of the HNN++ formulation. Optional intrinsic gyro-bias via `use_gyro_bias=True` |
| `HypLinearPoincarePP` | Poincaré | **Default** for Poincaré FC — Euclidean-parameterized weights, works with `optax.adam` |
| `HypLinearPoincare` | Poincaré | Legacy Ganea 2018 — manifold-valued bias, requires `riemannian_adam`. Prefer `PP` |
| `HypLinearPV` | Proper Velocity | PV networks — Euclidean weights, He init |

### Convolutional

| Layer | Manifold | When to pick |
|---|---|---|
| `HypConv2DHyperboloid` | Hyperboloid | **Default** for hyperboloid conv — well-established HCat formulation, robust across configurations |
| `HypConv2DHyperboloidILNN` | Hyperboloid | Intrinsic Lorentz conv (Shi et al. 2026) — log-radius-preserving concatenation (LogCat) + PLFC channel mixing, origin padding, optional gyro-bias |
| `HypConv2DHyperboloidFHNN` | Hyperboloid | HCat conv following the FHNN paper formulation; reach for it when reproducing that paper |
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

### Vector Quantization (Poincaré)

| Layer | When to pick |
|---|---|
| `HypVQEmbeddingPoincare` | Explicit on-ball codebook with a geometric **EMA** update (GGBall, Bu et al. 2026). No Riemannian optimizer — the codebook is a buffer moved by `ema_update` (called after `optimizer.update`); only the commitment loss trains the encoder. Optional dead-code revival |
| `HypVQMLRPoincare` | Codebook-free — quantization as Poincaré-MLR classification with Gumbel-Softmax. Plain `optax.adam`, reconstruction-only loss, deterministic argmax at eval (`model.eval()`) |

Both are quantizer *bottlenecks*: feed them encoder tangent features, add `output.loss` to your reconstruction loss, and decode `output.quantized` (returned as float32). See the [VQ API reference](../api-reference/nn-layers/vector-quantization.md).

### Attention, Normalization, Positional Encoding

| Use case | Layer |
|---|---|
| Long sequences, linear-complexity attention | `HyperbolicLinearAttention` (O(N)) |
| Most geometrically faithful attention | `HyperbolicFullAttention` (O(N²)) |
| Softmax attention with hyperbolic queries/keys | `HyperbolicSoftmaxAttention` |
| Language-model attention: low-rank latent keys/values, decoupled HOPE slice, causal/segment/padding masks (HELM) | `LorentzMLA`. Aggregates values with the single-GEMM centroid by default (`centroid_form="gemm"`, as HELM does); pass `"variance"` for short sequences or values far from the origin ([numerical stability](numerical-stability.md#attention-score-floor)) |
| Feed-forward block of a language model: top-k mixture of experts, one curvature per expert (HELM-MiCE) | `LorentzMoE` (with `LorentzMoEGate`, `LorentzSwiGLU`); balance with `moe_sequence_balance_loss` and `update_bias` |
| Token embedding on the hyperboloid | `LorentzEmbedding` — a `ManifoldParam` table trained with `riemannian_adam` (`parameterization="manifold"`), or a Euclidean spatial table (`"spatial"`); `c_out` maps the output onto another sheet |
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
| **Ganea (legacy)** | Bias on Poincaré, kernel Euclidean | `riemannian_adam` | `HypLinearPoincare`, `HypRegressionPoincare` | Legacy — prefer `PP` |

**Bottom line:** every modern layer parameterizes weights in Euclidean space,
and even the legacy Ganea layers keep their *kernel* Euclidean — only the
*bias* is manifold-valued. Standard `optax.adam` works for everything except
that one bias. See the [Riemannian Optimizers guide](optimizers.md) *(WIP)*
for the rare cases where a manifold-valued parameter appears.

## Channel Conventions

The single most common bug in layer construction is passing the wrong channel
count — ambient (d+1) vs. spatial (d). The full table lives in the
[Manifolds guide](manifolds.md#convention-cheat-sheet); the short form:

- **Hyperboloid layers** (`FGGLinear`, `LorentzConv2D`, `HTCLinear`, `HypLinearHyperboloid*`) take **ambient (d+1)** for their input channel arg. Exception: `HTCLinear.out_features` is **spatial (d)** — its output is `(B, out_features + 1)` ambient.
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
| `FGGLinear` / `FGGConv2D` | `fan_out`, `std=sqrt(1/out_spatial)`, bias `0.0`, `gain=1.0` | Norm-preserving (`‖z‖ ≈ gain·‖x_spatial‖`) so deep *unnormalized* stacks don't saturate a bounded projection — a deliberate deviation from the Klis et al. BatchNorm-regime reference. Restore the reference with `reset_params="eye"`/`"lorentz_kaiming"` + `init_bias=0.5` |
| `HypConv2DHyperboloidILNN` | Fan-out normal `std=sqrt(1/out_spatial)` (`kernel_init_std=None`) | Norm-preserving: the PLFC chain linearizes to `y_spatial ≈ W @ u_spatial` near the origin, and the fixed LogCat hands over the per-pixel spatial radius, so this holds gain ≈ 1. `kernel_init_std=0.02` restores the Shi et al. 2026 regime bit-for-bit |
| `HypLinearHyperboloidPLFC` | Small normal `std=0.02`, gyro-bias zeros (Shi et al. 2026 PLFC reference init) | `kernel_init_std=1.0` recovers the old HNN++-style init (Shimizu et al. 2020) |
| `HypLinearPoincare` / `HypLinearPoincarePP` | Fan-in normal `std=1/sqrt(in_dim)` | Keeps row norms small so outputs stay away from the boundary |
| `HypRegressionPoincare` / `HypRegressionPoincarePP` | Scaled normal `std=(2·in·out)^{-0.5}` (van Spengler 2023) | Sibling-scaled: an unscaled `N(0,1)` kernel gives row norms ≈ `sqrt(in_dim)`, which overwhelms the MLR output scaling |
| `HTCLinear` | Fan-in uniform `U(-√(3/in), √(3/in))` | Norm-preserving (per-layer Jacobian gain ≈ 1), though the time coordinate still feeds the kernel, so a stack's spatial norm grows ≈√2 per layer and its geodesic radius by ≈0.35 nats at large radius (log√2; measured mean 0.36–0.45) (see [TF32 and depth](numerical-stability.md#tf32-on-ampere-and-hopper-gpus)): `htc` applies no nonlinearity, so a fixed bound is contractive at realistic widths — the old `U(-0.02, 0.02)` froze depth-≥2 stacks. The Hypformer reference (Xavier·√2, `bound = 2√(3/(in+out))`) assumes ReLU + LayerNorm between layers; pass it explicitly to recover |
| `HypLinearHyperboloidFHCNN` | Small uniform `U(-0.02, 0.02)` | Keeps points near the apex `[1/sqrt(c), 0, …]` initially (matches the Bdeir et al. 2023 reference) |
| `HypLinear*PV` | He init | PV is unconstrained $\mathbb{R}^n$; standard scales work |
| `HypRegressionPV` | `std=1e-2` | MLR head: small scores at init |

!!! warning "Don't override init with He/Xavier on hyperbolic layers"
    If you wrap a hyperbolic layer in code that auto-applies a default Flax
    init, you will see `NaN` losses within the first few steps. The
    constructor's `kernel_init` argument exists for tuning, not for swapping
    in a Euclidean default.

!!! warning "Too small is as fatal as too large — and quieter"
    A too-small init doesn't NaN; it *freezes*. If a layer's per-layer gain
    `std·√fan_in` is below 1, stacking compounds the contraction as
    `gain^depth`, and pairwise output distances fall below the float32
    resolution of the distance computation: the stack becomes a constant map
    with ≈0 gradients from step 0, and training silently never starts. See
    the [numerical stability guide](numerical-stability.md#init-scale-vs-depth)
    for the mechanism.

## The Euclidean ↔ Hyperbolic Boundary

Most networks aren't *fully* hyperbolic. The question is **where you cross the
boundary** between Euclidean and hyperbolic computation:

| Pattern | Boundary location | Typical use case |
|---|---|---|
| **Fully hyperbolic** | Inputs are already on-manifold (e.g. embedding table on Poincaré) | Knowledge graph embeddings, hierarchy learning |
| **Hyperbolic head** | After a Euclidean backbone (CNN/Transformer) → `expmap_0` or constraint projection → hyperbolic classifier; a Poincaré PP or Busemann head takes the features directly with `input_space="tangent"` ([Pattern 2](#pattern-2-hybrid-cnn-backbone-poincare-head)) | ImageNet-scale CNNs with hyperbolic MLR (van Spengler 2023) |
| **Hyperbolic backbone** | At input via `expmap_0` per-pixel, then fully-hyperbolic through to a Euclidean logits layer | FullyHyperbolicCNN on MNIST |
| **Hybrid (sandwiched)** | Euclidean stem → small Euclidean embed → `expmap_0` → hyperbolic block → `logmap_0` → Euclidean head | When you want hyperbolic geometry only mid-network |

The boundary lift itself is covered in the
[Manifolds guide](manifolds.md#going-euclidean-manifold) (Pattern A vs B).
For the hybrid case on Hyperboloid, `HyperPPFeatureScaling` is the canonical
recipe to prepare Euclidean features before `expmap_0`:

```python
from hyperbolix.nn_layers import HyperPPFeatureScaling

scale = HyperPPFeatureScaling(dim=feature_dim, rngs=rngs)
x_euclidean = scale(x_euclidean, c)            # RMSNorm + activation + dim scaling
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
        # in_dim is AMBIENT (d+1); hidden is SPATIAL — HTCLinear.out_features
        # is the one exception to "Hyperboloid layers take ambient" (see
        # Channel Conventions above), so its output is ambient (hidden + 1)
        self.manifold = Hyperboloid(c=1.0)
        self.fc1 = HTCLinear(in_features=in_dim, out_features=hidden, rngs=rngs)
        self.norm = HRCLayerNorm(num_features=hidden, rngs=rngs)  # SPATIAL
        self.fc2 = HTCLinear(in_features=hidden + 1, out_features=hidden, rngs=rngs)
        self.head = HypRegressionHyperboloid(
            manifold_module=self.manifold,
            in_dim=hidden + 1, out_dim=num_classes, rngs=rngs,
        )

    def __call__(self, x_BAi: jax.Array, c: float = 1.0) -> jax.Array:
        h = self.fc1(x_BAi, c)
        h = self.norm(h, c)
        h = self.fc2(h, c)
        return self.head(h, c)  # (B, num_classes) Euclidean logits
```

### Pattern 2: Hybrid CNN backbone + Poincaré head

Pass the Euclidean features to the head as they are, with `input_space="tangent"`, instead of
lifting them with `expmap_0` first. The head then scores the point `expmap_0` would give, in closed
form, without storing that point. A lifted float32 point stops at the ball's ceiling: a longer
feature vector is scored as if it sat on the ceiling, with a zero radial gradient. The tangent path
stays accurate past it; at $\sqrt{c}\,\lVert v\rVert = 8$ the float32 error of the scores drops from
2.0e-1 to 1.5e-6 (see
[Tangent Inputs to the HNN++ and Busemann Layers](numerical-stability.md#poincare-tangent-input)).
`HypLinearPoincarePP`, `HypRegressionPoincareBusemann` and `HypLinearPoincareBusemann` take the same
flag. Keep the default `input_space="manifold"` when the input is already a ball point, such as the
output of an earlier Poincaré layer. `HypConv2DPoincare` already defaults to `input_space="tangent"`
and always returns tangent vectors, so a stack of these convs takes the tangent path without
setting the flag.

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
            input_space="tangent",  # takes the Euclidean features, no expmap_0
        )

    def __call__(self, images: jax.Array) -> jax.Array:
        c = self.curvature()
        features = self.pool(jax.nn.relu(self.stem(images)))  # (B, 64) Euclidean
        return self.head(features, c)                          # (B, num_classes) logits
```

### Pattern 3: Hyperbolic transformer block

```python
class HypTransformerBlock(nnx.Module):
    def __init__(self, dim_ambient: int, n_heads: int, *, rngs: nnx.Rngs):
        d_spatial = dim_ambient - 1
        self.attn_norm = HRCLayerNorm(num_features=d_spatial, rngs=rngs)
        self.attn = HyperbolicSoftmaxAttention(
            in_features=dim_ambient, out_features=d_spatial, num_heads=n_heads, rngs=rngs,
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

### Pattern 3b: HELM-style decoder block

The block of HELM (He et al. 2025): pre-norm, latent attention, a mixture of curvature experts, and a
Lorentzian residual after each. `hidden` is ambient, `x_BSA` has shape `(B, S, hidden)`.

```python
import math
from hyperbolix.nn_layers import HRCRMSNorm, LorentzMLA, LorentzMoE, LorentzResidual

class HELMBlock(nnx.Module):
    def __init__(self, hidden: int, *, rngs: nnx.Rngs):
        self.attn_norm = HRCRMSNorm(hidden - 1, rngs=rngs)              # SPATIAL
        self.attn = LorentzMLA(
            hidden, num_heads=8, kv_lora_rank=64, qk_nope_head_dim=32,
            qk_rope_head_dim=17, v_head_dim=33, rngs=rngs,              # rope/v dims are AMBIENT
        )
        self.moe_norm = HRCRMSNorm(hidden - 1, rngs=rngs)
        self.moe = LorentzMoE(hidden, inter_dim=512, num_routed=8, num_shared=1, top_k=2, rngs=rngs)
        # HELM's residual: raw learnable weight, fixed output scale sqrt(hidden)
        self.attn_res = LorentzResidual(weight_parameterization="identity", scale=True, init_gamma=math.sqrt(hidden))
        self.moe_res = LorentzResidual(weight_parameterization="identity", scale=True, init_gamma=math.sqrt(hidden))

    def __call__(self, x_BSA: jax.Array, c: float):
        h = self.attn(self.attn_norm(x_BSA, c, c), c)                   # causal by default
        x_BSA = self.attn_res(x_BSA, h, c=c)
        h, stats = self.moe(self.moe_norm(x_BSA, c, c), c)
        return self.moe_res(x_BSA, h, c=c), stats
```

Add `moe_sequence_balance_loss(stats, alpha=1e-4)` to the loss, and call
`block.moe.update_bias(stats.mask, speed)` after each optimizer step (`RoutingBias` is not trained by
gradient). The token table is a `LorentzEmbedding(vocab, hidden, rngs=rngs, c=c)`; train it with
`riemannian_adam` inside the same `nnx.Optimizer` as the Euclidean weights. The expert curvatures start
at `linspace(0.1, 2.0, E)`; `expert_curvatures=[1.0] * E` reproduces the released HELM checkpoint.

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
        self.bn1 = PoincareBatchNorm2D(self.manifold, channels)
        self.conv2 = HypConv2DPoincare(self.manifold, channels, channels,
                                       kernel_size=(3, 3), rngs=rngs)
        self.bn2 = PoincareBatchNorm2D(self.manifold, channels)
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

### 3. Leaving `version_idx` dynamic under JIT

Several Poincaré ops (`dist`, `expmap`, `logmap`) take a `version_idx` selecting
between multiple formulations. `jax.lax.switch` accepts a traced index, so this
argument does not have to be static. Marking it static is still recommended:
it compiles only the selected variant instead of every arm.

```python
# Works, but compiles every variant into one lax.switch
jit_fn = jax.jit(lambda x, y, idx: poincare.dist(x, y, c=1.0, version_idx=idx))

# Recommended: bind the variant before JIT-ing, compiles only that arm
from functools import partial
dist_v0 = partial(poincare.dist, version_idx=0)
jit_fn = jax.jit(dist_v0)
```

### 4. Mixing layer families incoherently

Stacking `HypLinearPoincare` (manifold-valued bias, expects
`riemannian_adam`) on top of `HypLinearPoincarePP` (fully Euclidean weights,
expects `optax.adam`) gives you a model where one optimizer is wrong for part
of the parameters. Pick one family per network and stay in it;
`riemannian_adam` auto-dispatches correctly across a mix of `ManifoldParam`
and plain `nnx.Param` if you genuinely need one, but the simpler fix is to
migrate the legacy layers to their `PP` equivalents.

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

- **[API Reference: NN Layers](../api-reference/nn-layers/index.md)** — full constructor and call signatures
- **[Manifolds Guide](manifolds.md)** — convention cheat-sheet, Euclidean→manifold lifts, isometry mappings
- **[Numerical Stability Guide](numerical-stability.md)** — when to use float64, clamping, safe norms
- **[Riemannian Optimizers Guide](optimizers.md)** *(WIP)* — when (rarely) you need Riemannian optimization
