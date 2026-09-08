# Numerical Stability Guide

Best practices for maintaining numerical precision in hyperbolic operations.

## Overview

Hyperbolic geometry presents unique numerical challenges due to the exponential growth of the conformal factor near the boundary and the involvement of hyperbolic functions (cosh, sinh, atanh). This guide explains these challenges and provides strategies to maintain numerical stability.

!!! warning "Key Challenges"
    - **Conformal factor explosion**: λ(x) grows exponentially as points approach the boundary
    - **Float32 limitations**: ~7 significant digits, insufficient for large distances (>10)
    - **Hyperbolic function overflow**: cosh/sinh overflow for large arguments
    - **Division by near-zero**: Operations involving 1 - c||x||² near the boundary

    These challenges are specific to the Poincaré ball; see [Hyperboloid](#the-hyperboloids-two-point-cancellation-failure-mode) below for operations that are accurate at any representable radius in float32.

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

*Table scoped to the Poincaré ball. `Hyperboloid.dist`/`logmap`/`sqdist`/`tangent_norm`/`expmap`/`ptransp`/`tangent_proj`/`tangent_inner`/`egrad2rgrad`/gyro `addition`/`busemann` under `VERSION_DEFAULT` are accurate to the point-representation floor at any radius in float32 — see [Hyperboloid](#the-hyperboloids-two-point-cancellation-failure-mode) below.*

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

### TF32 on Ampere and Hopper GPUs

The dtype is not the only thing setting your float32 accuracy on a modern NVIDIA card.
XLA:GPU runs float32 matmuls in **TF32** by default, whose 10-bit mantissa carries ~1e-3
relative error against float32's ~1e-7 — three of your seven significant digits, silently.

hyperbolix splits its float32 dot products in two:

- **Geometry is pinned** to `jax.lax.Precision.HIGHEST` and is not configurable: the manifold
  vector dots, `lorentz_midpoint` and `poincare_weighted_midpoint`, the conv patch extraction,
  the attention score and aggregation einsums and the spatial residual projection added to
  that aggregate, and the point-to-hyperplane kernel einsums — the MLR heads, and with them the
  PLFC, Poincaré++ and proper-velocity linear and conv layers, whose weight GEMM *is* that
  einsum applied to their kernel. These are the cancellations the geometry is built on, and they are not
  where the throughput is.
- **The other layer weight GEMMs follow JAX** — `HTCLinear` (the attention Q/K/V projections
  included), `FGGLinear`/`FGGConv2D`, the FHCNN/FHNN linears, `HypLinearPoincare`, the VQ
  codebook matmul.
  On an Ampere or Hopper card these run in TF32 unless you say otherwise. The FGG hidden dot
  `x @ V` is a deliberate exception: it absorbs the Minkowski metric into `V`, so its
  cancellation happens inside one accumulation, and it is left on the default anyway (see the
  depth numbers below).

To run everything in full float32, set JAX's own knob:

```python
jax.config.update("jax_default_matmul_precision", "highest")

# or scoped to a block (both spellings are jit-cache aware — changing them re-traces):
with jax.default_matmul_precision("highest"):
    logits = model(x)
```

A deep, fully hyperbolic stack is a good reason to do so: a TF32 error introduced in an early
layer's weight GEMM is carried by every layer after it. Measured on an A100 (jax 0.9.1,
ambient dim 33, batch 64, float32 against a float64 reference), the relative error of
`FGGLinear` is **2.6e-4** at the TF32 default against **6.8e-8** under `HIGHEST`, and
`FGGLorentzMLR` **1.3e-4 … 3.6e-4** against **3.6e-8 … 9.6e-8**. The cost is real but modest:
`HIGHEST` replaces one TF32 pass with a three-pass float32 emulation, measured **+5.5 %** on a
jitted hyperbolic attention forward.

**Depth is what decides whether you need it.** At depth 2 you do not: an independent
teacher-student comparison (5 seeds per arm, $D = 128$, $B = 256$, $c = 1$, 2 000 Adam updates,
independent A100 measurement, jax 0.9.1) found no mean heldout-loss difference between the
default and global `HIGHEST` — 0.27736 against 0.27738 for an `HTCLinear` stack, 0.28610
against 0.28614 for an `FGGLinear` one, against a seed spread of ~0.015–0.017. The
float32-vs-float64 gradient error does grow with depth. At depth 16, $c = 1$, input geodesic
radius 5 (same measurement), the `HTCLinear` stack's parameter gradient is **1.539 %**
relative L2 off a float64 reference under the default against **2.5e-6** under global
`HIGHEST`, and its input gradient **2.013 %** against **1.2e-5**; the `FGGLinear` stack at that
same cell is **0.199 %** against **8.0e-7** and **0.274 %** against **1.0e-6** — which is why
the FGG hidden dot is left on the default despite its cancellation. That cell is the *worst* of
a 24-cell grid (depths 2/8/16, $c \in \{0.1, 1\}$, input radius $\{0.5, 5\}$), one
initialisation and one input draw per cell and no seeds: these are grid maxima at $n = 1$, not
means.

The two stacks differ in the radius they reach. `HTCLinear` feeds the whole ambient point — time
coordinate included — through its Euclidean kernel, so at the default init the curvature-scaled
geodesic radius $\sqrt{c}\,r$ climbs by ≈0.35 nats per layer (measured mean +0.36 … +0.38 at depths
8 and 16, the input→layer-1 step excluded), while an `FGGLinear` stack stays at its input radius
(−0.011 … +0.002). Across the four depth-16 cells the `HTCLinear` stack's TF32 parameter-gradient
error is **2.3–7.8×** the `FGGLinear` stack's, on that one initialisation and one input draw per
cell. Whether the radius climb is *what* costs the precision is a **hypothesis**: rescaling the HTC
kernel by $1/\sqrt{2}$ to flatten the climb cuts the depth-16 error 3.9× in a CPU rounding
emulation, but doubling the climb did not raise it and the emulation does not reproduce the GPU
gap. At depth 2 the HTC/FGG ratios are 0.87–0.97, which is not an ordering. For a deep or
gradient-sensitive stack, set the global knob.

`HIGHEST` is a no-op on CPU (there is no TF32 path) and for float64 anywhere.

### The Hyperboloid's Two-Point Cancellation Failure Mode

The distance-from-origin table above governs single-point operations (`dist_0`, `logmap_0`,
`expmap_0`), which read the geodesic radius off the **spatial** part of the point and have their
own near-origin story, described in [The Hyperboloid Origin Chart](#hyperboloid-origin-chart)
below. Point-to-point operations (`dist`, `logmap`, `sqdist`, `tangent_norm`) are governed by a
different, two-point quantity — being far from the origin is not itself the problem; two points
far from the origin **and close together** is.

Every one of these operations used to go through the Minkowski inner product
$\langle x, y\rangle_L = -x_0y_0 + \langle x_s, y_s\rangle$, which for two hyperboloid points is a
subtraction of two positive terms each roughly $e^{\sqrt{c}\,(d_0(x) + d_0(y))}$ in size, leaving a
result proportional to $e^{\sqrt{c}\,d(x,y)}$. The number of significant digits lost is set by the
**Gromov-product-like quantity**
$$
\sqrt{c}\,\bigl(d_0(x) + d_0(y) - d(x, y)\bigr),
$$
and once it exceeds $\ln(1/\epsilon)$ — 15.9 for float32, 36.0 for float64 — every digit of the
result is cancellation noise. Two nearby points that are each individually far from the origin hit
this constantly (e.g. points sampled along a shared geodesic ray, or clustered leaf embeddings in a
deep hierarchy), while a single point far from the origin, or two points that are merely far from
each other, does not.

Concretely, before this fix: float32 `dist` returned `0.0015` for a true distance of `1.0` for two
points at radius 10 from the origin — not a large relative error but a *complete loss of
information*, since the correct value could have been anything below the float32 noise floor.
`logmap` returned `NaN` from radius ~10 (float32) / ~20 (float64); `tangent_norm` returned ~0 on
tangent vectors of exactly unit length past radius 8 (float32) / 20 (float64) — a 100% error with
no warning. Deep metric-learning embeddings routinely sit at radius 30–60, so this was not an edge
case in practice.

As of this fix, `dist`, `logmap`, `sqdist`, and `tangent_norm` under the default `version_idx`
(`VERSION_DEFAULT` / `VERSION_SMOOTHENED`) are evaluated through a cancellation-free "hyperbolic
haversine" decomposition and are accurate at any representable radius — see
[Hyperboloid Distance Versions](#hyperboloid-distance-versions) below for the version constants,
and [Known Limitations](#hyperboloid-known-limitations) for the remaining two-point primitives —
which are now also cancellation-free — and the handful of places that still lose accuracy at
large radius for other reasons.

#### Known Limitations {#hyperboloid-known-limitations}

The two-point cancellation fix originally covered `dist`, `logmap`, `sqdist`, and `tangent_norm`.
A second pass extended the same pattern — read the radius off the spatial part, subtract before
squaring, write the result as a sum of non-negative terms — to the remaining two-point and
tangent-space primitives: `expmap` (the two-point form), `ptransp`/`ptransp_0`, `tangent_proj`,
`tangent_inner`, `egrad2rgrad`, the gyro `addition` (the Lorentz boost — this is what
`Hyperboloid.scalar_mul` and a `use_gyro_bias=True` gyro-bias on `HypLinearHyperboloidPLFC`,
`HypConv2DHyperboloidILNN`, and `HypLinearHyperboloidBusemann` call), and `busemann`. None of
these route through the literal `⟨x,y⟩_L = -x_0 y_0 + ⟨x_s,y_s⟩` any more, and none of them are
limited to radius ~7-9 in float32. Radii below are the **scaled geodesic radius**
`a = √c·d`. Measurements are 4-seed medians, float32 unless noted, from
`logs/2026-09-08_hyperboloid_tangent_primitives/`.

`tangent_norm`/`tangent_inner` are exact to `a ≈ 15` (float32) / `a ≈ 25` (float64) — one power
of `cosh` better than the ambient chart's own point-representation floor, `eps·sinh(a)/√c`, which
sits at `a ≈ 16.6` in float32 (measured B.iv: `|⟨v,v⟩-1|` on an exactly-unit radial tangent goes
from 1.100e+01 as-is to 1.132e-06 fixed at `a = 10`, and the ProperVelocity twin from 8.000e+00
to 1.132e-06). Origin-chart operations (`dist_0`, `logmap_0`, `expmap_0`) never routed through the
Minkowski inner product, so none of this applies to them; they had a different problem at the
*small*-radius end, fixed separately and described next.

**Gyro-addition and the PLFC gyro-bias.** `x ⊕ exp_0(b)` at `c = 0.5`, `‖b‖ = 0.5`
(`probe_addition_{46abd2b,ebebd09}.out`, table A.1a): forward geodesic error 4.181e-04 → 1.366e-05
at `a = 6`, 1.527e-02 → 9.767e-05 at `a = 8`, 6.557e-02 → 6.802e-04 at `a = 10`, 5.670e-01 →
4.565e-03 at `a = 12`; 4 of 32 float32 seeds were non-finite as-is across the table, none are
fixed. Gradients improve further: `d/db` relative error 7.496e-02 → 1.315e-06 at `a = 8`,
8.467e-01 → 4.616e-07 at `a = 10`; `d/dx` 2.164e-02 → 9.752e-08 at `a = 8`, 4.752e+00 → 8.233e-08
at `a = 12`.

**Transport, tangent projection, gradient conversion, and expmap.** End-to-end
`riemannian_adam` on a `ManifoldParam`, 30 steps (`probe_optimizer_*.out`, table B.i): as-is loses
both seeds to NaN by `a = 9` (first non-finite step between 1 and 9 of 30); fixed stays finite
through `a = 16`, descending the expected loss ~0.29 through `a = 14` (0.290 at `a=9`, 0.291 at
`a=12`, 0.305 at `a=14`). The parallel-transport isometry `‖PT v‖_y/‖v‖_x` (table B.ii) goes from
non-finite on 1-2 of 4 seeds from `a = 9` on (max residual 6.6e+14 at `a=10`) to finite everywhere,
`|ratio-1|` at 1.192e-07 (`a=6`), 1.132e-06 (`a=8`), 7.557e-05 (`a=12`), 1.333e-04 (`a=14`).
`egrad2rgrad` (table B.iii) goes from a relative error that reaches 1.0 (the whole gradient lost,
non-finite on half the seeds) at `a = 14` to a flat ~5e-8 with no radius dependence through
`a = 14` — it needs no division by a measured quantity at all, since `⟨x,x⟩_L = -1/c` holds
exactly on the sheet, so it is not floor-limited the way the others are.

`expmap`'s landing accuracy (`dist(x, expmap(v,x))/‖v‖_x`, table B.v) is the one cell in this pass
that is not a clean win: the `a = 12` outlier is gone (max geodesic landing error 1.651e-01 →
5.312e-03), but at `a = 14` the fixed landing error is *worse* than as-is — 3.500e-02 vs 2.410e-02
median (1.45×), 4.571e-02 vs 2.917e-02 max (1.57×). Both revisions are within the `tangent_norm`
floor's own uncertainty at that radius, so this reads as the `tangent_norm` floor showing through
the exponential map rather than a new cancellation.

**`is_in_manifold` / `is_in_tangent_space`.** Both now compare the stored `x₀` (or `v₀`) directly
against the value the spatial part implies, instead of testing the Lorentz form against `-1/c` (or
`0`). The old residual was the honest time-slot discrepancy scaled by `2·x₀ = 2·cosh(a)/√c` *and*
obtained as a difference of two `O(cosh²a)` numbers; reading the time slot directly removes both
factors, so a genuinely on-sheet point now passes a fixed `atol` far past the `a ≈ 7` (float32) /
`a ≈ 11` (float64) ceiling the old check imposed — see [The `atol` Convention](#the-atol-convention).

#### Known limits

Three places still lose accuracy at large radius, for reasons the fix above does not remove:

1. **`ptransp`'s direction below the float32 angular resolution.** A transported direction that
   differs from the identity by less than the point-representation floor `eps·sinh(a)/√c` is
   input-limited: the two endpoints do not disagree by more than storage rounding, so no formula
   recovers a direction that was never represented in the first place. This is a different limit
   from the transport *isometry* fixed above (table B.ii), which holds the vector's length, not
   the smallest resolvable direction.
2. **Attention scores through a GEMM.** The Lorentzian similarity score behind
   `HyperbolicFullAttention` still forms `2 + 2⟨Q,K⟩_L` from a matrix product, and a GEMM cannot be
   made cancellation-free the way a single pairwise `dist` can — see
   [Full Attention's Float32 Score Floor](#attention-score-floor) below.
3. **The wrapped-normal `log_prob` on the hyperboloid.** Its density involves the same
   `⟨x,x⟩_L`-style terms, so float32 and float64 diverge by roughly `eps·cosh(a)·‖v‖²/σ²` at large
   radius — qualitative, no probe in this pass isolates it.

Two more items that used to be on this list — gyro-centering with a near-identity partner, and
`ProperVelocity.dist`/`logmap` between nearby points — were fixed in a later pass; see
[Gyro-Difference and GyroBatchNorm Centering at Large Radius](#gyro-difference) and
[ProperVelocity `dist`/`logmap` Through the Exact Lift](#pv-dist-lift) below.

#### Full Attention's Float32 Score Floor {#attention-score-floor}

`HyperbolicFullAttention`'s scores are `2 + 2⟨Q,K⟩_L`, a difference of two Minkowski terms each of
size `cosh(a_q)·cosh(a_k)/c`. Unlike the primitives above, there is no GEMM-compatible
cancellation-free spelling for it: any matrix product of the ambient coordinates returns the Gram
matrix to absolute `eps`, so the angular term alone already carries an error of
`eps·sinh(a_q)·sinh(a_k)/c`, the same order as the literal form. The absolute error on one score is
therefore `eps·cosh(a_q)·cosh(a_k)/(c·scale)`, which with float32's `eps ≈ 1.19e-7` (`c = 1`,
`scale = 1`) is about 4.8e-3 at `a = 6`, 0.26 at `a = 8`, and 2.0 at `a = 9`: from `a ≈ 8` the error
exceeds the score spread softmax is meant to resolve, and the weights come out wrong while staying
perfectly finite. The remedy is `score_dtype=jnp.float64`, which runs the similarity and the
`2 + 2⟨Q,K⟩_L` / scale / bias / causal-mask arithmetic as a float64 island and casts back before
the softmax; it fixes the arithmetic only, so activations that are themselves stored in float32
past `a ≈ 8` still carry a floor of the same order, and a `HyperboloidGyroRMSNorm` in front of the
layer (or a smaller `c`) is the cheaper first remedy.

### The Hyperboloid Origin Chart {#hyperboloid-origin-chart}

`dist_0` and `logmap_0` used to recover the geodesic radius from the ambient **time** coordinate
$x_0$, and that coordinate cannot resolve a small radius. On the sheet

$$
x_0 = \frac{\cosh(\sqrt{c}\,d)}{\sqrt{c}} \approx \frac{1 + c\,d^2/2}{\sqrt{c}},
$$

so $d$ is stored only to $\sqrt{\varepsilon}$ resolution (relative error $\varepsilon/(2cd^2)$),
and `acosh`'s `1 + 10·eps` domain clamp flattened every float32 radius below
$\sqrt{20\varepsilon}/\sqrt{c} = 1.54\text{e-}3$ onto exactly zero. Both operations now read the
radius off the **spatial** part instead, where the same number is available exactly:

$$
d_0(x) = \frac{\operatorname{arcsinh}(\sqrt{c}\,\lVert x_s\rVert)}{\sqrt{c}},
\qquad
\log_0(y) = \Bigl[0,\; \frac{\operatorname{arcsinh}(u)}{u}\, y_s\Bigr],\quad u = \sqrt{c}\,\lVert y_s\rVert .
$$

`arcsinh` needs no domain clamp (its argument is a norm) and its derivative is bounded by 1, so
$\lVert \log_0(y)\rVert = d_0(y)$ now holds by construction at every radius. Median relative
error at $c = 1$, dim 8, before → after (A100, jax 0.9.1; the CPU backend and jax 0.11.0 agree in
every floored cell):

| radius | float32 `dist_0` | float32 `log_0(exp_0(v))` | float64 `dist_0` | float64 round trip |
| --- | --- | --- | --- | --- |
| 1e-6 | 1.5e3 → 2.5e-9 | 1.5e3 → 0 | 4.4e-5 → 0 | 4.4e-5 → 1.3e-16 |
| 1e-3 | 5.4e-1 → 9.8e-8 | 5.4e-1 → 0 | 5.9e-11 → 0 | 1.5e-10 → 1.2e-16 |
| 1e-2 | 5.0e-4 → 2.7e-8 | 5.2e-4 → 0 | 1.0e-12 → 1.7e-16 | 8.3e-13 → 1.2e-16 |
| 0.1 | 3.5e-6 → 1.8e-8 | 1.4e-7 → 3.9e-8 | 7.6e-15 → 1.4e-16 | 9.5e-15 → 1.1e-16 |
| 1 to 40 | ≤7.1e-8 → ≤5.2e-8 | ≤1.0e-7 → ≤8.8e-8 | ≤1.3e-16 → ≤1.9e-16 | ≤1.4e-16 → ≤1.4e-16 |

!!! warning "A zero-initialised hyperboloid gyro-bias could not train"
    The old `dist_0` returned a constant `0` under a bitwise `at_origin` guard, so
    $\partial(x \oplus b)/\partial b$ at $b = $ origin was the exact zero matrix in **both**
    dtypes. Gyro addition inherits that guard through `logmap_0`, so the bias of any
    `HypLinearHyperboloidPLFC`, `HypConv2DHyperboloidILNN`, or `HypLinearHyperboloidBusemann`
    built with `use_gyro_bias=True` sat at the origin, received an exactly-zero gradient, and
    never moved. Measured gyro-bias gradient L2 at init (c = 1, dim 8, float32): `0.0` before,
    `1.37` / `4.24` / `3.07` after. The Poincaré `HypLinearPoincareBusemann` bias uses Möbius
    addition and was never affected. A model trained with one of the three hyperboloid biases was
    trained with it pinned to the origin.

Operations that inherit the fix without any change of their own: gyro `addition`, `scalar_mul`,
`logmap`'s origin fallback, and `HyperboloidGyroRMSNorm`, which divides by `dist_0` and so
mis-normalised every float32 sample inside radius 1.5e-3.

!!! note "An infinitely far point gives an infinite tangent vector, not NaN"
    A spatial entry that is `inf` passes through the radius as `inf` on purpose, so an out-of-range
    point stays visibly degenerate instead of silently NaN-poisoning everything downstream. That made
    `logmap_0`'s scale $\operatorname{arcsinh}(u)/u$ an $\infty/\infty$ NaN, which then multiplied
    *every* entry — the time slot included. It is now
    `where(isfinite(u), arcsinh(u)/u, 1)`: the infinite entries come back as $\pm\infty$ with their
    signs, the finite entries keep their values, and the time slot stays exactly 0. This is the same
    $\pm\infty$-not-NaN convention the pairwise `dist`/`logmap` already follow through
    `_polar_frame`'s `isfinite` guard. The finite path is unchanged: the forward value is bit-identical on
    both backends and the gradient is bit-identical on XLA:CPU, moving by at most 2 ulps on XLA:GPU
    (where the extra `where` changes the VJP's fusion) against a 4-ulp CPU-vs-GPU spread the unchanged
    code already had.

### Norms: One Reduction, Gradient-Safe at Zero {#safe-norms}

On the operations that 1.2.0 made slower, the per-sample Euclidean norm is a **single pass over
the data** again: `safe_sqrt(sum(v**2))` where the input can be exactly zero, a plain
`sqrt(const + sum(v**2))` where a strictly positive constant is added, and
`floor_at(..., MIN_NORM)` wrapped *around* either where the norm is a divisor. These run on every
sample of every step, so the norm they use is judged by what it costs an SGD run — and a second
read of the same `(B, dim)` array is a real cost, while the failure it guards against is not one
training reaches.

A site is converted only where 1.2.0 measured **slower** (hyperboloid `expmap_0` 1.28x, `HTCLinear`
forward+backward 1.12x, on both A100 and H100) or where PR #75 raised the compiled kernel /
reduction count. That is the whole list:

| Where | Operations |
|---|---|
| `Hyperboloid` | `proj`, `proj_batch`, `dist_0` (both version slots) |
| `Poincare` / `Stereographic` | `proj`, `proj_batch` (the shared gyrovector core), and `Poincare.expmap` |
| Layers | `spatial_to_hyperboloid` (so `HTCLinear`), the FHCNN and FGG linear forwards |

Every other per-sample norm keeps the max-scaled two-pass form, which is full-range safe:
hyperboloid `logmap_0` (which measured *faster* in 1.2.0), the pairwise `dist`/`logmap` polar
frame, hyperboloid and Poincaré `tangent_norm`, `Poincare.expmap_0` (no measured change either
way), the FHNN linear forward, the linear-attention `focus_transform` (converting it measured no
speedup of its own), every `ProperVelocity` operation and the proper-velocity isometry
maps, the Poincaré metric-tensor distances, the rest of `Stereographic`, `Euclidean` and
`ProductManifold`, the wrapped-normal `log_prob`, and every weight norm. Where nothing was
measured slower there is no cost to trade the full-range guarantee against.

Neither of the two older idioms is used on that path any more. The **additive**
`sqrt(sum(x**2) + MIN_NORM**2)` was a gradient guard: it gives `sqrt` a finite derivative at
$x = 0$, which `jnp.linalg.norm` does not (its VJP there is $0/0 =$ NaN). It paid for that with a
$\texttt{MIN\_NORM}^2 = 10^{-30}$ floor that dominates any genuinely small vector, relative
residual $\texttt{MIN\_NORM}^2/(2r^2)$: $5.0\times10^{-15}$ for a float64 point at radius
$10^{-8}$, and 41 % for a float32 point at radius $10^{-15}$, where `dist_0` returned
$2.83\times10^{-15}$ for a true $2.00\times10^{-15}$. `safe_sqrt`'s double-`where` supplies the
same finite (in fact exactly zero) derivative with no floor on the value at all.

The **max-scaled** `safe_norm`/`safe_hypot_norm` reads the input twice — once for
$\max_i |x_i|$, once for the sum — to keep $\sum_i x_i^2$ from overflowing float32, which happens
once a coordinate passes $1.8\times10^{19}$. That is geodesic radius $\approx 45$ at $c = 1$ and
$\approx 139$ at $c = 0.1$, in float32; a network that far out is already diverging, and `inf` is
the signal you want. So the converted sites take the single reduction and give up their
answer past that coordinate. What they give instead was measured, float32 at $c = 1$, rather than
argued — and it is **not** uniform:

| Converted operation | Past coordinate $1.8\times10^{19}$ (measured, float32, $c = 1$) |
|---|---|
| `Hyperboloid.proj`, `proj_batch` | $x_0 = \infty$, spatial part unchanged, no NaN |
| `Hyperboloid.dist_0` (both version slots) | $\infty$ |
| `spatial_to_hyperboloid` (`HTCLinear`), FHCNN linear forward | $x_0 = \infty$, no NaN |
| FGG linear forward | non-finite (NaN or $\infty$, depending on the input) |
| Poincaré and $\kappa$-stereographic `proj` (the boundary clamp) | the **origin** — a finite point, with a finite (exactly zero) gradient |
| `Poincare.expmap` | the **base point** — a finite point (the origin, when the base is the origin) |
| everything on the two-pass form | unchanged from 1.2.0 |

So two of the converted sites do saturate to a finite point, and that is the clamp's documented
behaviour rather than an accident: with $\lVert x\rVert = \infty$ the boundary clamp
$x \cdot (r_{\max}/\lVert x\rVert)$ *is* the zero vector, which is the pre-1.2.0 behaviour, and
`expmap` inherits it through the same clamp. Nothing is added to guard a regime SGD cannot reach.
(`Poincare.expmap_0`, which kept the two-pass norm, still returns the correct boundary point
there.)

One consequence worth stating plainly: for the converted origin-chart operations the largest
representable geodesic radius drops. It used to be set by $x_0 = \cosh(a)/\sqrt{c}$ fitting the
dtype, i.e. $a = \ln(2\,\texttt{finfo.max})$ — radius $\approx 89$ in float32, $\approx 710$ in
float64 at $c = 1$ (89.416 / 710.476; the same number as $\operatorname{arcsinh}(\texttt{finfo.max})$,
since $\sinh(a)$ and $\cosh(a)$ leave the dtype together). For `proj` and `dist_0` it is now set by
the **coordinate**, $\sinh(a)/\sqrt{c} < \sqrt{\texttt{finfo.max}}$ — radius $\approx 45$ in
float32 and $\approx 356$ in float64. `logmap_0` and the pairwise `dist`/`logmap` keep the
two-pass norm and therefore keep both their finite result and a ceiling of that order: `logmap_0`
reads only $\lVert y_s\rVert$ and keeps the old $\approx 89$ / $\approx 710$ exactly, while the
pairwise pair binds a fraction of a radius earlier, at $e^a/\sqrt{c} \le \texttt{finfo.max}$
($a = \ln(\texttt{finfo.max})$: 88.72 / 709.78), because `_polar_frame` composes
$u = x_0 + \lVert x_s\rVert$ and returns $\pm\infty$ once that overflows. Every hyperbolic model in
the literature lives four to five orders of magnitude inside every one of these limits.

What has **not** changed is how the constant is folded in. The hyperboloid time slot is
$x_0 = \sqrt{1/c + \sum_i (x_s)_i^2}$ — the constant added to the sum *under* the square root, in
the same reduction — and never `hypot(‖x_s‖, 1/√c)`, which rounds $\lVert x_s\rVert$ to the dtype
and then squares it again. The library's own constraint check is
$-x_0^2 + \lVert x_s\rVert^2 = -1/c$: when $x_0$ is built from the same $\sum_i (x_s)_i^2$ that
check forms, the rounding of the sum cancels against itself and the residual is one rounding of
$x_0$; when $x_0$ is built from a re-squared norm the two no longer cancel, and at
$x_0 \approx 34$ one float32 ulp of $x_0^2$ is $1.2\times10^{-4}$. The constant is spelled `1/c`
rather than `(1/√c)**2`, which is one rounding fewer.

All the primitives stay public in `hyperbolix.utils.math_utils`, and each has its own job.
`safe_norm`, `safe_hypot_norm` and `safe_normalize` are the max-scaled, full-range ones: they hold
every per-sample site that was not converted, and the **weight** norms — computed once per forward
over an `(in, out)` kernel, where a second reduction is not on the per-sample path. They also
remain the right choice in user code that genuinely needs the full float32 exponent range.
`safe_hypot` is different: it is a two-leg *scalar* composer $\sqrt{p^2+q^2}$, and it is on the hot
path — the hyperboloid `_polar_frame` builds its radial and angular legs with it. `safe_sqrt` is
the scalar counterpart used at the converted sites.

All of them return an exact `0` with an **exactly zero** VJP at the zero vector — the finite,
direction-free choice at a point where the derivative does not exist — and pass a non-finite input
through as `inf` rather than turning it into NaN. Where a norm is a *divisor* the floor is still
there, as `floor_at(safe_sqrt(sum(x**2)), MIN_NORM)`: a multiplicative floor, exact everywhere
above itself, instead of an additive one that perturbs every value. It has to sit **around** the
square root, not under it — `floor_at` under the `sqrt` still leaves `sqrt'(0) = inf` in the
untaken branch of the surrounding `where`, and $0 \times \infty$ is NaN. One further subtlety when
the floored norm feeds a $\sinh(t)/t$: the floor has to be applied to $t$ itself and not only to
the denominator, or the ratio is $0$ at $t = 0$ instead of $1$.

!!! note "The pairwise `dist` reads the same radial gap off the spatial part"
    `dist` is a separate code path, and it used to carry its own $O(\varepsilon/r)$ relative error
    near the origin: its polar decomposition needs $u_x - u_y$ with $u = x_0 + \lVert x_s\rVert$,
    and forming that difference directly throws away the $\varepsilon\,x_0$ that $x_0$ is stored
    with. In float32 that was 4.6e-2 relative at radius 1e-6 and 1.4e-4 at 1e-4. It now uses the
    on-sheet identity $x_0 - y_0 = (\lVert x_s\rVert^2 - \lVert y_s\rVert^2)/(x_0 + y_0)$, i.e.

    $$
    u_x - u_y = (\lVert x_s\rVert - \lVert y_s\rVert)
                \Bigl(1 + \frac{\lVert x_s\rVert + \lVert y_s\rVert}{x_0 + y_0}\Bigr),
    $$

    which subtracts only the spatial radii. Float32 `dist(origin, x)` is now within 3.1e-7 of
    `dist_0(x)` at every radius from 1e-8 up, and two float32 points at radius 1e-3 separated by
    1e-3 agree with the float64 answer to 1.5e-7 (was 3.3e-5). `dist_0(x, c)` remains marginally
    cheaper and marginally tighter than `dist(origin, x, c)` — at $y$ exactly at the origin the
    `MIN_NORM` floor on $\lVert y_s\rVert$ leaves a $\tfrac12\,$`MIN_NORM`$/\lVert x_s\rVert$
    relative residual in the pairwise arm — but the two no longer disagree in any digit float32
    can see.

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

### The Round-Trip Ceiling {#poincare-roundtrip-ceiling}

`proj` keeps points inside $1/\sqrt{c}$ by a margin of `eps**0.75`
(`_gyrovector_core._get_max_norm_eps`), which caps the largest tangent vector `expmap_0` can
represent at $\|v\| = \mathrm{atanh}(1 - \varepsilon^{0.75})/\sqrt{c}$ — a geodesic radius of
$d_0 = 2\,\mathrm{atanh}(1 - \varepsilon^{0.75})/\sqrt{c}$, i.e. $12.65/\sqrt{c}$ in float32 and
$27.7/\sqrt{c}$ in float64 (the factor 2 is the Poincaré metric's). Past that `expmap_0`
saturates and `logmap_0(expmap_0(v))` hands back the ceiling instead of `v`. Measured ceilings
(median returned $\|v\|$ on a round trip, in units of $1/\sqrt{c}$; double them for the geodesic
radius):

| library | float32 | float64 | boundary margin |
| --- | --- | --- | --- |
| hyperbolix | 6.32 | 13.86 | `eps**0.75` |
| geoopt, hypLL | 3.11 | 6.10 | fixed 4e-3, fixed 1e-5 |
| unguarded closed form | 8.66 | 18.72 | none |

The margin is deliberate. It stops short of the unguarded limit so the conformal factor stays
under ~3e5 (float32) / ~1e12 (float64), and it still leaves twice the radius the fixed margins
used elsewhere allow. If your embeddings need larger radii, switch to the hyperboloid or to
`ProperVelocity` rather than shrinking the margin.

### The Factored Möbius Denominator {#factored-mobius-denominator}

Every gyrovector op that needs $x \oplus y$ — `dist`, `logmap`, `addition`, `gyr` — divides by the
Möbius denominator $1 + \mathrm{sign}\cdot 2c\langle x,y\rangle + c^2\lVert x\rVert^2\lVert
y\rVert^2$, and the literal spelling is a difference of two terms that both grow like the ball
chart's own ceiling squared as $x, y$ approach the boundary. `_mobius_denominator` instead uses
$1 + t^2 = (1-t)^2 + 2t$ with $t = \lvert c\rvert\,r_x r_y$: the subtraction $1 - t$ happens
*before* the square, so an $O(1)$ result stops being the difference of two large numbers. This is
what a pairwise `dist`/`logmap` on the Poincaré ball uses at every radius; it is what the
[chart ceiling](#poincare-roundtrip-ceiling) above ultimately bounds. Radii below are the geodesic
distance from the origin ($c = 1$, so $a = d$).

Measured on a radial pair with a geodesic gap of 0.1 (`probe_poincare_mobius_ebebd09.out`, table
D.i, medians over 4 seeds): in float32, `dist` error goes from 2.816e-03 to 2.086e-06 at $d = 8$,
6.506e-02 to 1.958e-05 at $d = 10$, and 1.423e+01 to 1.044e-04 at $d = 12$ (the as-is value at
$d = 12$ is the whole geodesic gap, lost); `‖logmap‖` error goes from 5.523e-03 to 8.799e-07 at
$d = 8$ and 2.866e-01 to 3.206e-04 at $d = 12$. In float64, `dist` error goes from 1.101e-03 to
1.584e-10 at $d = 18$, 7.519e-02 to 5.367e-10 at $d = 20$, and 9.664e-02 to 2.105e-09 at $d = 22$
(again, the as-is value at $d = 20$ is essentially the whole 0.1 gap); `‖logmap‖` error goes from
2.187e-03 to 2.324e-10 at $d = 18$ and 9.384e-02 to 1.243e-09 at $d = 20$. Both dtypes' fixed
columns are still climbing as $d$ approaches the chart ceiling from below — the factoring removes
the denominator's own cancellation, not the ball's inability to represent a point past
$12.65/\sqrt{c}$ (float32) / $27.7/\sqrt{c}$ (float64) at all.

### Float32 Accuracy Near the Origin Depends on the Backend

XLA's float32 transcendental kernels are a few ulps off, and near the origin that is the whole
error budget of a Poincaré round trip. Exact bit-pattern ulp error against a float64 reference
(20k inputs in $[10^{-4}, 0.9]$, max / mean):

| function | XLA GPU | XLA CPU | torch CPU | torch CUDA |
| --- | --- | --- | --- | --- |
| `tanh` | 4 / 0.85 | 4 / 0.90 | 1 / 0.01 | 2 / 0.17 |
| `atanh` | 3 / 0.45 | 2 / 0.25 | 1 / 0.00 | 3 / 0.45 |
| `arcsinh` | 2 | 2 | 2 | 2 |
| `expm1` | 1 | 5 | n/a | n/a |

Consequence: float32 `logmap_0(expmap_0(v))` at radius $10^{-3}$ (dim 32, median relative error)
was 2.4e-7 on the CPU backend with raw XLA kernels and exactly 0 on GPU; hyperbolix's own `tanh`
and `atanh` wrappers (series below 1/8, `expm1` form above) bring the CPU backend to exactly 0 as
well, see the changelog. A torch-based library reaches ~1.5e-8 on CPU with raw kernels,
because torch's CPU `tanh`/`atanh` are correctly rounded; on CUDA it has no such edge (torch's
CUDA `atanh` is bit-identical to XLA's). The closed forms are the same in both cases, so this is
a kernel difference rather than a formula difference, but it does mean that a near-origin float32
accuracy number is only meaningful with its backend quoted.

## Init Scale vs. Depth

Weight-init failures on hyperbolic layers come in two flavors, and only one of
them is loud. A **too-large** init pushes first-layer outputs toward the
Poincaré boundary or far up the hyperboloid — distances and gradients explode
and you see `NaN` within a few steps. A **too-small** init fails *silently*:
for a linear-in-the-matmul layer (e.g. `HTCLinear`, whose `htc` tail applies no
nonlinearity), the per-layer input-Jacobian gain is

$$
g \approx \sigma_w \cdot \sqrt{\text{fan\_in}},
$$

and a stack compounds it as $g^{\text{depth}}$. When $g < 1$, pairwise
distances between outputs shrink geometrically until they fall below the
float32 resolution of the distance computation itself: near the origin the
hyperboloid distance passes through $\mathrm{acosh}(1 + c\,d^2/2)$, and once
$c\,d^2/2 < \varepsilon_{f32} \approx 1.19 \times 10^{-7}$ the computed
distance quantizes to exactly zero. The stack is then a constant map —
gradients are ≈0 from step 0 and training never starts, with no `NaN` or
warning to point at.

!!! warning "Fixed bounds cannot be width-independent"
    A hard-coded init bound bakes in a width: `U(-0.02, 0.02)` has
    $g \approx 0.09$ at fan-in 65 (frozen by depth 2) but $g \approx 1$ only
    near fan-in 7,500. Hold $g \approx 1$ instead by scaling with fan-in —
    e.g. `HTCLinear`'s default `init_bound = sqrt(3 / in_features)`. Layers
    followed by normalization (LayerNorm/BatchNorm absorb magnitude) tolerate
    $g > 1$; unnormalized stacks — typical in RL — do not.

See the [initialization scales table](nn-layers.md#initialization-scales) for
each layer family's default and how to recover reference inits.

## Flattening a Conv Feature Map: Use LogCat, Not `reshape` {#logcat-flatten}

At the **conv → FC boundary** of a hyperboloid CNN you have an `(B, H', W', C)`
feature map — one hyperboloid point per pixel — and need one point per sample for
the classification head. The reflex from Euclidean code is
`x.reshape(B, H' * W' * C)`, and on the hyperboloid that is wrong twice over: it
concatenates `H'·W'` time coordinates as if they were features, and even the
correct-by-construction version (`Hyperboloid.hcat`, which stacks only the spatial
parts and rebuilds one time coordinate) inflates the radius.

The inflation is a dimension effect, not a bug in `hcat`. For Gaussian-ish spatial
parts $\|v\|^2 \sim \chi^2_k$, so

$$
\mathbb{E}[\log \|v\|] = \tfrac{1}{2}\left(\psi(k/2) + \log 2\right),
$$

which **grows with the dimension $k$**. Concatenating $N = H' \cdot W'$ blocks widens
$k$ from $d$ to $N\,d$ and lifts the expected log spatial radius by $\approx \tfrac12 \log N$
— a radius inflation of $\approx \sqrt{H' \cdot W'}$. LogCat
(`Hyperboloid.log_radius_concat`, Shi et al. 2026 Sec. 4.3) cancels it by shrinking
every block first:

$$
s = \exp\!\left(\tfrac{1}{2}\left(\psi(d/2) - \psi(N d/2)\right)\right) \approx \frac{1}{\sqrt{N}},
$$

then recomputing the time coordinate so the result stays on the (widened) hyperboloid.

!!! warning "Why this bites harder at the FC boundary than inside a conv"
    `HypConv2DHyperboloidILNN` already applies LogCat to each receptive field, where
    $N = 9$ for a 3×3 kernel. At the flatten, $N$ is the **entire feature map** —
    tens to low hundreds — so the naive flatten hands the head a point whose radius
    is an order of magnitude past what its weights were initialized for. The
    observed symptom is an MLR head sitting at 100% saturation-cap occupancy at
    step 0 (logits pinned, gradients ≈ 0), which clears when the flatten uses LogCat.

Use `hyp_flatten2d`, which reshapes the grid to the per-sample point sequence and
applies LogCat for you:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HypConv2DHyperboloidILNN, HypRegressionHyperboloid, hyp_flatten2d

hyperboloid, c = Hyperboloid(), 1.0
conv = HypConv2DHyperboloidILNN(
    manifold_module=hyperboloid, in_channels=17, out_channels=9,
    kernel_size=3, stride=2, rngs=nnx.Rngs(0),
)
head = HypRegressionHyperboloid(
    manifold_module=hyperboloid, in_dim=4 * 4 * 8 + 1, out_dim=10, rngs=nnx.Rngs(1),
)

v = 0.1 * jax.random.normal(jax.random.PRNGKey(2), (8, 8, 8, 17))
x = jax.vmap(jax.vmap(jax.vmap(hyperboloid.expmap_0, in_axes=(0, None)), in_axes=(0, None)), in_axes=(0, None))(
    v.at[..., 0].set(0.0), c
)                                          # (8, 8, 8, 17) on-manifold feature map

feat = conv(x, c)                          # (8, 4, 4, 9)  — 9 ambient = 8 spatial + time
flat = hyp_flatten2d(feat, hyperboloid, c)  # (8, 129)     — 4*4*8 spatial + one time
logits = head(flat, c)                     # (8, 10)
```

**Width bookkeeping.** `hyp_flatten2d` grows the ambient dimension from `A` per pixel
to `H'·W'·(A − 1) + 1`, so size the head accordingly (`in_dim = 4*4*8 + 1 = 129`
above). If that width is impractical, use `hyp_avg_pool2d` instead — it averages the
spatial parts over the grid and keeps the width at `A`, at the cost of discarding
spatial layout. Both are documented on the
[convolutional API page](../api-reference/nn-layers/convolutional.md#pooling-flattening-conv-fc-bridge).

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
- **Hyperboloid**: unbounded radius, and `dist`/`logmap`/`sqdist`/`tangent_norm`/`expmap`/`ptransp`/`tangent_proj`/`tangent_inner`/`egrad2rgrad`/gyro `addition`/`busemann` are all cancellation-free, accurate to the point-representation floor at any representable radius (see [above](#the-hyperboloids-two-point-cancellation-failure-mode)). The constraint $\langle x, x\rangle_L = -1/c$ must still be maintained and can drift under Euclidean updates — see [The `atol` Convention](#the-atol-convention) — and a handful of places still lose accuracy for reasons the fix does not remove, listed under [Known Limitations](#hyperboloid-known-limitations).
- **Proper Velocity**: unconstrained $\mathbb{R}^n$, stable at large radii, exact Euclidean retraction (plain `optax.adam` / SGD trains PV layers without a Riemannian wrapper). Preferred when embeddings naturally grow large. Its tangent-space metric shares the hyperboloid's fix (see [below](#pv-tangent-metric)), and `PV.dist`/`logmap` between two nearby points at large radius now go through the exact hyperboloid lift — see [below](#pv-dist-lift).
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

## Lorentz Residual and Midpoint at Large Radius {#lorentz-residual-midpoint}

`lorentz_residual` (the two-point combination behind `LorentzResidual` and
`HypformerPositionalEncoding`) and `lorentz_midpoint` (the weighted aggregation behind
`HyperbolicFullAttention`, `HyperboloidGyroRMSNorm`, and the hyperboloid Fréchet
mean) both form a raw ambient vector $h = (h_0, h_s)$ — time coordinate $h_0$, spatial
part $h_s$ — and pull it back onto the sheet by dividing by
$\sqrt{\lvert\langle h,h\rangle_L\rvert}$. Computing that Minkowski square directly,

$$
\langle h, h\rangle_L = -h_0^2 + \lVert h_s\rVert^2,
$$

cancels catastrophically. On the sheet $h_0 \approx \lVert h_s \rVert$, so both terms
are of size $\lVert s\rVert^2$ — writing $\lVert s\rVert$ for the spatial radius of the
inputs — while their difference is only $O(1/c)$. The float32 relative error of the
result therefore grows with the radius as
$\varepsilon_{32}\, c\, \lVert s\rVert^2$ with $\varepsilon_{32} \approx 1.19\times10^{-7}$,
which crosses a target error `err` at

$$
\lVert s\rVert \approx \sqrt{\mathtt{err} \,/\, (\varepsilon_{32}\, c)}.
$$

In plain terms: at $c = 0.1$, a relative error of $10^{-3}$ arrives once the spatial
radius reaches about 290, and gradients hit that level roughly three times sooner in
$\lVert s\rVert$ than the forward value does, because the normalizer's Jacobian cancels
a second time. Past $\lVert s\rVert \sim 10^4$ the computed square flips sign outright.

Hyperbolix instead evaluates $\langle h,h\rangle_L$ from exact identities, valid for any
weights as long as the inputs are on the sheet ($\langle x,x\rangle_L = -1/c$). For the
residual $h = x + w\,y$ with scalar weight $w$:

$$
\langle h,h\rangle_L = -\frac{(1+w)^2}{c} - w\,\langle x-y,\; x-y\rangle_L .
$$

For the midpoint of $M$ points $x_1,\dots,x_M$ with weights $w_1,\dots,w_M$, the normalizer is
built instead from the **variance form**. Write $r_m = \lVert x_{m,s}\rVert$ for each point's
spatial radius, $\hat x_m = x_{m,s}/r_m$ its unit direction, $u_m = x_{m,0}+r_m$, and, with
$R = \sum_m w_m r_m$ the weighted spatial radius and $\omega_m = w_m r_m/R$ (so $\sum_m \omega_m = 1$):

$$
\bar m = \sum_m \omega_m \hat x_m, \qquad
V = \sum_m \omega_m \lVert \hat x_m - \bar m\rVert^2 = 1 - \lVert\bar m\rVert^2 ,
$$

the last equality the variance identity, exact for any weights summing to 1 and any unit
$\hat x_m$. The identity is used only in the derivation above: the code evaluates $V$ as the
direct weighted sum over the $(\ldots, N, M, D)$ differences $\hat x_m - \bar m$, never as
$1 - \lVert\bar m\rVert^2$, which cancels for a tight cluster — measured max $V$ 1.19e-07 for that
spelling against 8.42e-15 for the direct sum on a radial float32 cluster (`step2d_equivalence.out`).
On the sheet $x_{m,0} - r_m = 1/(c\,u_m)$, so with $W = \sum_m w_m$:

$$
\mathrm{gap} = \sum_m \frac{w_m}{c\,u_m} = h_0 - R, \qquad
\mathrm{small} = \mathrm{gap} + \frac{R\,V}{1+\lVert\bar m\rVert} = h_0 - \lVert h_s\rVert, \qquad
\mathrm{big} = \mathrm{gap} + R\,(1+\lVert\bar m\rVert) = h_0 + \lVert h_s\rVert,
$$

$$
-c\langle h,h\rangle_L = c\cdot \mathrm{small}\cdot\mathrm{big} .
$$

`gap` is a sum of positives (every $u_m > 0$ on the sheet), and `small`/`big` are built from `gap`
plus $R\,V$ / $R\,(1+\lVert\bar m\rVert)$, both non-negative for non-negative weights — nothing is
subtracted anywhere, and the $O(e^{2a})$ scale of $h_0$ and $\lVert h_s\rVert$ never enters the
normalizer at all. The identity is exact for arbitrary weights (it reduces algebraically to the
literal $h_0^2-\lVert h_s\rVert^2$), and for non-negative weights every quantity above is
non-negative.

It costs $O(N\cdot M\cdot D)$ — one $(\ldots,N,M,D)$ broadcast-reduce for $V$ — in place of an
earlier $O(M^2)$ key-Gram form's $(\ldots,M,M,D)$ pairwise chord. Two earlier forms of the same
identity were tried and neither shipped: a pivot-relative decomposition (commit `ebebd09`, the
residual's identity above generalised with $x_1$ as a shared reference point) reached the same
$O(M)$ cost but regressed a cloud spread mostly in *angle* at one common radius — there the pivot
gap itself scales with the radius, so both sides of that decomposition's difference return to
$O(e^{4a})$; a key-Gram form (`9093ea7`/`4f8057a`) fixed the angular regression by summing the
pairwise Minkowski Gram matrix directly ($-c\langle h,h\rangle_L = \sum_{mn} w_m w_n \cosh\theta_{mn}$),
exact but $O(M^2)$ per key set. The variance form (`10b1e6c`) is exact like the key-Gram form and
linear like the pivot form.

Measured (`probe_midpoint_horopca_busemann_pv_fd0c1d7.out`, table C.i; $M = 16$, $c = 0.5$,
uniform weights, $a = \sqrt{c}\,d$ the scaled radius, medians over 4 seeds; as-is numbers from
`probe_midpoint_horopca_busemann_pv_46abd2b.out`): a *radial* cloud (spread 0.3 in $a$ along one
direction) goes from 5.199e-04 / 1.068e-02 / 4.142e-01 / 2.998e+00 as-is to 1.452e-05 / 1.022e-04
/ 3.165e-04 / 4.988e-03 fixed at $a = 6/8/9/12$ — roughly 36 to 1300× better. An *angular* cloud
(spread 0.3 rad at one common radius — the attention/aggregation case) sits at 3.632e-07 /
4.580e-07 / 3.601e-07 / 3.454e-07 fixed across the same four radii, essentially unchanged from the
3.632e-07 / 4.580e-07 / 3.619e-07 / 3.556e-07 as-is, because this cloud has almost nothing to
cancel in the literal form either — every point shares the same time coordinate. A *mixed* cloud
(both spreads at once) goes from 4.381e-07 / 7.099e-07 / 4.271e-07 / 4.092e-07 as-is to
3.214e-07 / 3.573e-07 / 3.612e-07 / 3.676e-07 fixed. The radial rows are the one case that does not
sit flat, and it is not the normalizer: a relative coordinate error of $2^{-24}$ is an angular
error of the same size, and a geodesic at radius $a$ amplifies that by $\sinh a$ —
$6\text{e-}8 \cdot \sinh 12 \approx 5\text{e-}3$ is the whole $a = 12$ radial entry, i.e. the
float32 representation floor of the *inputs*, not the aggregation. In float64 the medians run
4.142e-14 (radial $a=6$) to 6.355e-11 (radial $a=12$), and 5.7e-16 to 7.8e-16 for every angular and
mixed row. Cost is linear in the point count; the exact throughput numbers arrive with the cost
tables in a later pass.

The residual's own identity is unaffected by the angular-cloud regression the pivot form hit — with
only two points there is no third point to reintroduce that cancellation — so its measured float32
error against float64 at $\lVert s\rVert = 10^4$ stays $\approx 2\times10^{-5}$ on the value and
$\approx 6\times10^{-4}$ on the gradient, and the limit that remains for it is the difference
$x - y$ itself: when two points share a direction at $\lVert s\rVert \gtrsim 10^4$, float32
rounding swallows the subtraction before the formula sees it, and that regime needs float64.

!!! note "The rest of the hyperboloid is covered too"
    `Hyperboloid.dist`'s two slots, the remaining tangent-space and two-point primitives
    (`expmap`, `ptransp`, `tangent_proj`, `tangent_inner`, `egrad2rgrad`, gyro `addition`,
    `gyro_difference`, `busemann`), and this midpoint normalizer all use the same cancellation-free pattern and are
    accurate to the point-representation floor at any representable radius — see
    [Known Limitations](#hyperboloid-known-limitations) for the handful of places that still lose
    accuracy for other reasons, and [The `atol` Convention](#the-atol-convention) for why merely
    *storing* a point past distance ~11 in float64 can already need an explicit `atol` on
    `is_in_manifold`.

### Busemann Coordinates at Large Radius {#busemann-large-radius}

`Hyperboloid.busemann(x, v, c)` (the horosphere coordinate behind
`HypLinearHyperboloidBusemann`/`HypRegressionHyperboloidBusemann` and HoroPCA) is
`log(√c·arg)/√c` with `arg = x_0 - ⟨x_s, v⟩`. Along the branch aligned with the ideal
direction $v$, `x_0` and `⟨x_s,v⟩` are both $O(\cosh a)$ and nearly equal, so the literal
subtraction cancels the same way the other Minkowski forms above do; `arg` is rewritten as the sum
of two positives, $\mathrm{arg} = 1/(c(x_0+r)) + r\lVert\hat x - v\rVert^2/2$ with $r = \lVert x_s\rVert$,
algebraically identical to the original but without the cancellation. Measured
(`probe_midpoint_horopca_busemann_pv_9093ea7.out`, table C.iii; $c = 1$, 4-seed medians): on the
aligned branch ($\psi = 0$, the one that actually cancels) the relative error goes from 1.096e-02
to 1.585e-09 at $a = 8$ and from 1.878e+00 to 2.274e-06 at $a = 12$; as-is pinned the value on the
`MIN_NORM` floor ($\log(10^{-15})/\sqrt{c} = -34.54$) on 1 of 4 seeds at $a = 10$ and 3 of 4 at
$a = 12$, with an identically-zero gradient there — fixed pins none and returns a real gradient
(7.364e-03 at $a = 12$).

The batched attention-style head, `nn_layers.busemann_core._busemann_score` (and the vmapped
`busemann` it shares with the Busemann MLR/FC layers), uses the same exact sum-of-positives form.
Measured on a batch where every point sits within 1e-3 rad of one of $K = 4$ directions,
$a = 10$ (table C.iii(b)): the median relative error is unchanged at 3.509e-08 (only the
near-aligned pairs cancel at all), but the max drops from 4.006e-02 to 5.366e-06 — 7,470× — since
it is exactly those near-aligned pairs that the old form lost. For a use case that needs float64
throughout without paying for it everywhere, the recipe is a float64 island around just the score:
cast $\hat x$ and the class directions to float64, run the one similarity GEMM there,
compute $\mathrm{arg} = 1/(c(x_0+r)) + r(1-g)$ and its `log` in float64, then cast back — this is a
documented pattern, not a library option; `HyperbolicFullAttention`'s `score_dtype` (see
[above](#attention-score-floor)) is the shipped version of the same idea for the Lorentzian
similarity score.

### HoroPCA Projection at Large Radius

`horo_projection`, the ideal-point projection behind `hyperbolix.decomposition.horopca`, inherits
the same Busemann cancellation: it is defined to preserve every Busemann coordinate of its input
exactly, so any drift in `busemann` itself shows up as drift in the projected point. Measured
(`probe_midpoint_horopca_busemann_pv_9093ea7.out`, table C.ii; a point 1e-3 rad off the first ideal
direction, $K$ ideal directions, `proj err` against the library's own float64 `horo_projection`):
at $K = 2$ the projection error goes from 1.413e-01 to 1.086e-04 at $a = 8$ and from 3.660e+00 to
8.633e-03 at $a = 12$; Busemann-coordinate preservation (`|dB|`, the true value of which is 0) goes
from 8.170e-02 to 8.836e-05 at $a = 8$ and from 7.731e-02 to 1.575e-04 at $a = 12$.

### Gyro-Difference and GyroBatchNorm Centering at Large Radius {#gyro-difference}

`Hyperboloid.gyro_difference(x, y, c)` computes $(\ominus x)\oplus y$, the operation
`HyperboloidGyroBatchNorm` needs to center a batch on its mean and any layer needs for a difference
of two far points. The general gyro-addition `addition(neg(x), y)` (the Lorentz boost
$\Lambda_{\ominus x}\,y$ — see [above](#the-hyperboloids-two-point-cancellation-failure-mode)) is
accurate almost everywhere *except* here: at $y \approx x$ the boost's spatial part is a sum of
three terms, each $O(e^{2a})$, that cancel identically at $y = x$, so its absolute error is
`eps·cosh²(a)/√c` no matter how close $y$ is to $x$ — the near-identity gyro-addition has no
ambient-spelling fix.

`gyro_difference` instead uses the isometry $\Lambda_x^{-1}$, the transvection carrying $x$ back to
the origin, whose differential along the geodesic $x \to 0$ *is* parallel transport:

$$
(\ominus x) \oplus y = \Lambda_x^{-1} y = \mathrm{Exp}_0\big(\mathrm{PT}_{x\to 0}(\mathrm{Log}_x y)\big) .
$$

In the polar frame the transport is free: the inward radial leg of $\mathrm{Log}_x y$ transports to
the outward direction $-\hat x$ continuing the same geodesic past the origin, and the in-plane
angular leg is untouched, so the result is read straight off the frame with no cancellation and no
transcendental beyond the frame's own.

**The floor.** What limits `gyro_difference` at $y \approx x$ is not the arithmetic but the
operands: two float32 points whose directions are $\psi$ radians apart pin the *direction* of their
difference only to `eps32·√D/ψ` relative — that is the accuracy of the result regardless of
formula. When the result is instead far from the origin, the binding floor is one float32 ulp of
its own spatial radius, `eps32·sinh(√c·d₀)/√c`.

Measured (`step2c_gyro_difference_accuracy.out`; $c = 0.5$, $D = 64$, $a \in \{9, 12\}$, $y$ a
$\psi$-radian rotation of $x$, medians/maxima over 4 seeds; `true d0` is the true radius of the
result):

| $a$ | $\psi$ | true $d_0$ | boost abs err (med) | `gyro_difference` abs err (max) | direction floor | radius floor |
|---|---|---|---|---|---|---|
| 9 | 1e-2 | 1.047e+01 | 8.904e-01 | 3.203e-04 | 9.987e-04 | 1.385e-04 |
| 9 | 1e-4 | 5.691e-01 | 2.560e+00 | 7.147e-04 | 5.428e-03 | 6.969e-08 |
| 9 | 1e-6 | 5.748e-03 | 2.053e+00 | 1.047e-03 | 5.482e-03 | 6.852e-10 |
| 12 | 1e-2 | 1.896e+01 | 1.593e+01 | 2.986e-02 | 1.808e-03 | 5.582e-02 |
| 12 | 1e-4 | 5.972e+00 | 1.704e+01 | 7.407e-03 | 5.695e-02 | 5.748e-06 |
| 12 | 1e-6 | 1.150e-01 | 1.072e+01 | 1.916e-02 | 1.097e-01 | 1.372e-08 |

`gyro_difference`'s worst error over these six cells is 0.535× the larger of the two floors —
input-limited everywhere — against the boost's absolute error, which tracks $a$ and not $\psi$ at
all: 0.89 to 2.05 nats whether the true answer is 10.5 nats from the origin or 5.7e-3.

`HyperboloidGyroBatchNorm` now centers with `gyro_difference`. Measured on 32 points at $a = 9$,
$c = 0.5$, $D = 16$, $\gamma = 1.3$ (`step2c_gyro_bn_centering.out`, float32 vs float64 on
bit-identical inputs; `err` = max geodesic distance between the two legs' layer output; `grad rel` =
max relative error of $\partial\text{loss}/\partial\text{bias}$):

| target sep | mean sep | before err | after err | ratio | before grad | after grad |
|---|---|---|---|---|---|---|
| 0.3 | 0.556 | 5.220e+00 | 3.996e-03 | 1306.3× | 2.835e+01 | 4.133e-03 |
| 0.6 | 1.101 | 3.960e+00 | 1.869e-03 | 2118.5× | 1.329e+01 | 2.073e-03 |
| 1.0 | 1.797 | 2.142e+00 | 8.996e-04 | 2381.4× | 2.099e+00 | 1.278e-03 |
| 2.0 | 3.388 | 4.218e-01 | 4.766e-04 | 884.9× | 7.551e-02 | 2.193e-04 |
| 4.0 | 6.250 | 6.311e-02 | 1.052e-04 | 600.0× | 4.921e-02 | 8.333e-05 |
| 8.0 | 11.749 | 1.525e-03 | 1.791e-05 | 85.2× | 2.384e-04 | 2.172e-05 |

The `before` column is finite and plausible at every row — that is the failure mode, not a NaN.

`gyro_difference` is verified against `addition(neg(x), y)` in float64 over dims 2/5/64,
$c \in \{0.1, 0.5, 1, 3\}$ and random/parallel/anti-parallel/perpendicular/degenerate operand
pairs: worst relative disagreement 2.34e-15; against a `np.longdouble` reference the result stays
inside the float64 representation floor of its own radius (worst 0.62×)
(`step2c_gyro_difference_equivalence.out`).

### The Geodesic Frame Has No Normalize {#geodesic-frame}

`_polar_frame`'s angular leg used to come from a helper, `_logmap_direction`, that normalized a
vector which is exactly zero on the whole collinear set — a pair $(x, y)$ sharing a ray, $\psi = 0$
or $\pi$, $y = x$ included. It returned $(\cos\varphi, \sin\varphi, \hat n)$ with
$\hat n = \mathrm{normalize}(\hat y_s - \langle \hat x_s, \hat y_s\rangle\, \hat x_s)$; on a
collinear pair that argument is zero up to rounding, so the normalization's derivative is
arbitrary, and multiplied by a $\sin\varphi$ that is itself only $O(\text{rounding})$ rather than
exactly 0, it left an $O(1)$ error in the *gradient* of every consumer while the forward value
stayed correct. Four public functions share the helper and inherited the defect:
`Hyperboloid.logmap`, `.ptransp`, `.gyro_difference`, and, through the exact lift onto the
hyperboloid, `ProperVelocity.logmap`. Pre-existing since the polar-frame `logmap` of 1.1.2
(2026-08-25).

**The fix.** `_logmap_direction` now returns $(S\cos\varphi, \mathrm{perp}_y)$ with no $1/S$ and no
$1/\lVert\mathrm{perp}_y\rVert$ anywhere:

$$
\mathrm{perp}_y = r_y\Big[\tfrac{\mathrm{csum}^2}{4}(\hat y_s - \hat x_s) +
\tfrac{\mathrm{chord}^2}{4}(\hat x_s + \hat y_s)\Big],
$$

with $\mathrm{chord}$ and $\mathrm{csum}$ the frame's own norms of $\hat y_s - \hat x_s$ and
$\hat x_s + \hat y_s$. Each consumer's own prefactor cancels the remaining $S$:

$$
\log_x(y) = \mathrm{asinhc}(S)\cdot\tfrac{2}{\sqrt c}\cdot(S\cos\varphi)\cdot e_{\mathrm{rad}} +
\tfrac{\mathrm{asinhc}(S)}{C}\cdot(0, \mathrm{perp}_y), \qquad
\mathrm{asinhc}(S) = \frac{\mathrm{arcsinh}(S)}{S},
$$

$$
\mathrm{ptransp\ scale} = -\frac{S\cos\varphi}{C}\cdot\frac{\mathrm{radial}(v)}{x_0} +
\frac{c}{2}\cdot\frac{\langle \mathrm{perp}_y, \mathrm{perp}(v)\rangle}{C^2}, \qquad
(\ominus x)\oplus y = -\frac{2}{\sqrt c}\cdot C\cdot(S\cos\varphi)\cdot\hat x + \mathrm{perp}_y .
$$

**Two design points.**

- $S\cos\varphi$ is formed as a sum of individually bounded ratios —
  $P\cdot\mathrm{hypot}(1,P)/C + q\cdot(q/C)\cdot(x_0/r_x)$ — never as the unbounded product $q^2$
  first, since at large radius $q$ alone is $\sim 10^{18}$ in float32. This keeps the float32 worst
  case at $a \in \{9, 12\}$ to 1.7e-7 against 2.3e-7 for the previous spelling
  (`step2e_equivalence.out`).
- $\mathrm{perp}_y$ is spelled as the non-negative combination of the two orthogonal vectors
  $\hat y_s - \hat x_s$ and $\hat x_s + \hat y_s$ above, not as
  $r_y(\hat y_s - \langle\hat x_s,\hat y_s\rangle\hat x_s)$: the dot-product form loses
  $\mathrm{eps}/\sin\psi$, while the orthogonal-vector form is exactly zero on both degenerate rays
  ($\psi = 0$ zeroes $\hat y_s - \hat x_s$ and $\mathrm{chord}$ together, $\psi = \pi$ zeroes
  $\hat x_s + \hat y_s$ and $\mathrm{csum}$ together) and cannot cancel at any other $\psi$.

**Measured** (`step2e_gradients.out`, $a \le 3$, old $\to$ new):

| quantity | dtype | old | new |
|---|---|---|---|
| $\nabla\lVert\log_x(y)\rVert^2$ / $\nabla\langle w,\log\rangle$, collinear + $y=x$ | float64 | 2.81 | 1.8e-10 |
| Jacobian of $\log_x(\cdot)$ at $y=x$ vs. the identity | float64 | 1.00 | 5.0e-16 |
| `ptransp` $\nabla\langle w, \mathrm{PT}\,v\rangle$, f32-vs-f64 relative error | float32/64 | 3.29 | 4.0e-6 |
| `gyro_difference` gradient, f32-vs-f64 relative error | float32/64 | 5.94 | 9.6e-7 |
| PV $\nabla\lVert\log\rVert^2$, collinear | float32 | 4.1e-5 | 2.2e-5 |
| PV $\nabla\lVert\log\rVert^2$, collinear | float64 | 3.3e-10 | 3.3e-10 |

The PV float32 row is 1.9× better than even the pre-lift ambient PV spelling on the same reference,
and the float64 row is unchanged either way. Every gradient in this table is finite in both
dtypes at every radius tested, before and after the fix.

Forward values are unaffected: float64 new-vs-old equivalence at $a \le 6$ over dims 2/5/64,
$c \in \{0.1, 0.5, 1, 3\}$, 5 geometries, 5 seeds — `logmap` 4.7e-16, `ptransp` 5.8e-14,
`gyro_difference` 3.2e-15, PV `logmap` 1.5e-15, all against a 1e-12 budget
(`step2e_equivalence.out`); `dist`, `sqdist` and `_polar_frame` are bitwise unchanged.

### ProperVelocity's Tangent-Space Metric at Large Radius {#pv-tangent-metric}

The proper-velocity `tangent_inner`/`tangent_norm` share the hyperboloid's
`radial_perp_decomposition` helper and its fix. Measured on the exactly-unit radial tangent, $c =
0.5$ (`probe_midpoint_horopca_busemann_pv_9093ea7.out`, table C.iv): $\lvert\langle v,v\rangle -
1\rvert$ goes from 6.250e-01 to 2.384e-07 at $a = 8$ and from 1.536e+03 to 5.192e-05 at $a = 12$;
the library's own float64 path on the same input goes from 9.313e-10 to 2.220e-16 at $a = 8$.

### ProperVelocity `dist`/`logmap` Through the Exact Lift {#pv-dist-lift}

`ProperVelocity._dist` and `_logmap` go through the exact lift onto the hyperboloid,

$$
X = \Big(\sqrt{1/c + \lVert x\rVert^2},\; x\Big) ,
$$

and then the hyperboloid's own cancellation-free polar-frame `dist`/`logmap` — PV coordinates *are*
the hyperboloid's spatial part, so the lift is exact, not an approximation. This closes the gap the
tangent-space metric fix above did not reach: `PV.dist`/`logmap` between two nearby points at large
radius used to cancel the same way the ambient hyperboloid primitives did before their own fix.

Measured (`step2c_pv_accuracy.out`, $c = 0.5$, dim 16, a radial step of true Riemannian length 0.1,
landing point computed in float64, coordinate-axis direction): old `dist` 5.988811e-02 /
4.382994e+00 / 1.056287e+01 at $a = 8/10/12$ (true value 0.1 in every case), new relative error
9.54e-07 / 8.05e-07 / 4.62e-07 — inside the float32 storage floor of the same input pairs,
9.24e-07 / 7.22e-07 / 4.92e-07. The full float32 chain (`expmap` in float32 too, so its own error
is included) used to return 3.167524e-01 at $a = 8$ and 4.369660e+00 at $a = 10$ for a true step of
0.1; it now returns 1.000014e-01 and 9.999371e-02.

Equivalence (`step2c_pv_equivalence.out`): float64 new vs old at $a \le 3$ (random, parallel,
antiparallel, perpendicular, coincident pairs) agree to 8.09e-14 max; against an 80-bit reference at
$a \le 6$, new is 7.11e-15 max against old's 3.64e-11; `dist(x, x)` and `logmap(x, x)` are exactly 0
in both dtypes, with a finite gradient there.

Callers that inherit the fix with no change of their own: `utils.helpers.compute_pairwise_distances`,
`decomposition/frechet.py`, `nn_layers/poincare_batchnorm.py` (when built with a PV manifold), and
`manifolds/product.py`.

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

#### Slot 2 Reads the Radius Through `arcsinh` {#poincare-metric-tensor-dist-0}

Both metric-tensor arms — the origin distance `dist_0` and the pairwise `dist` — used to have the
[hyperboloid origin chart's defect](#hyperboloid-origin-chart) in Poincaré coordinates. Taking
`dist_0` first, the metric-tensor integral is

$$
d_0(x) = \frac{1}{\sqrt{c}}\operatorname{acosh}\!\left(1 + \frac{2c\lVert x\rVert^2}{1 - c\lVert x\rVert^2}\right),
$$

and the whole radial signal sits in that $2t$ perturbation of a leading 1. `acosh`'s
$1 + 10\varepsilon$ domain clamp therefore flattened every float32 radius below
$\sqrt{10\varepsilon/c} \approx 1.1\text{e-}3/\sqrt{c}$ onto exactly zero, and a second
`arg < 1 + MIN_NORM` short-circuit zeroed the band below
$\sqrt{\texttt{MIN\_NORM}/2c} \approx 2.2\text{e-}8/\sqrt{c}$ in **both** dtypes. The arm now uses
the half-angle identity $\operatorname{acosh}(1 + 2t) = 2\operatorname{arcsinh}(\sqrt{t})$, whose
argument is *linear* in the radius near the origin:

$$
d_0(x) = \frac{2}{\sqrt{c}}\operatorname{arcsinh}\!\left(\frac{\sqrt{c}\,\lVert x\rVert}{\sqrt{1 - c\lVert x\rVert^2}}\right).
$$

Same function, so slot 2 still means "metric-tensor distance" — nothing was moved to a new slot,
and the boundary clamp (via the conformal factor) is unchanged. Median relative error against a
60-digit `decimal` reference, $c = 1$, dim 8, before → after:

| radius | float32 | float64 |
| --- | --- | --- |
| 1e-8 | 7.7e4 → 1.4e-8 | 1.0 → 1.7e-16 |
| 1e-6 | 7.7e2 → 1.4e-8 | 1.1e-5 → 0 |
| 1e-4 | 6.7 → 2.9e-8 | 2.5e-9 → 1.4e-16 |
| 1e-3 | 6.6e-3 → 5.3e-8 | 2.8e-12 → 0 |
| 1e-2 | 3.3e-5 → 8.6e-8 | 1.4e-13 → 0 |
| 0.1 | 4.9e-7 → 3.4e-8 | 2.2e-15 → 0 |
| 0.9/√c | 6.5e-8 → 3.5e-8 | 1.5e-16 → 1.5e-16 |

!!! warning "The floored band had a zero gradient, not just a wrong value"
    `jax.grad` of slot 2 returned exactly `0` for float32 radii ≤ 1e-6 at every curvature (≤ 1e-3
    at $c = 0.3$) and float64 radii ≤ 1e-8, instead of the analytic $2/(1 - c r^2) \to 2$. Anything
    optimised through slot 2 near the origin received no gradient at all. Slots 0 and 1
    (`VERSION_MOBIUS_DIRECT` / `VERSION_MOBIUS`, both $2\operatorname{atanh}(\sqrt{c}\lVert x\rVert)/\sqrt{c}$)
    were never affected, and remain the default.

The pairwise `dist` slot 2 is the same story with a separation in place of a radius:

$$
d(x,y) = \frac{1}{\sqrt{c}}\operatorname{acosh}\!\left(1 + 2t\right)
       = \frac{2}{\sqrt{c}}\operatorname{arcsinh}\!\left(\sqrt{t}\right),
\qquad
t = \frac{c\lVert x - y\rVert^2}{(1 - c\lVert x\rVert^2)(1 - c\lVert y\rVert^2)} .
$$

The `acosh` clamp zeroed every pair with $t < 5\varepsilon$ and the `MIN_NORM` short-circuit a
second band below $t = \texttt{MIN\_NORM}/2$. The two $(1 - c r^2)$ factors scale the threshold
with the pair's radius, so at $c = 1$ the float32 separation floor is 1.2e-3 at the origin and
9.2e-4 at radius 0.5. Median relative error against a 60-digit `decimal` reference routed through
Möbius addition (independent of the formula under test), $c = 1$, dim 8, before → after:

| radius | separation | float32 | float64 |
| --- | --- | --- | --- |
| 1e-2 | 1e-8 | 7.7e4 → 3.0e-8 | 1.0 → 1.7e-16 |
| 1e-2 | 1e-6 | 7.7e2 → 2.6e-8 | 5.4e-8 → 2.1e-16 |
| 1e-2 | 1e-4 | 6.7 → 2.3e-8 | 2.5e-9 → 6.8e-17 |
| 1e-2 | 1e-2 | 6.7e-5 → 1.0e-7 | 1.9e-13 → 0 |
| 0.5 | 1e-8 | 3.5e4 → 3.6e-8 | 1.0 → 1.2e-16 |
| 0.5 | 1e-6 | 5.8e2 → 3.3e-8 | 6.3e-6 → 0 |
| 0.5 | 1e-4 | 4.8 → 4.5e-8 | 3.6e-10 → 0 |
| 0.5 | 1e-2 | 2.6e-6 → 6.1e-8 | 5.1e-14 → 1.3e-16 |

!!! warning "Two points inside the floor had no gradient pulling them apart"
    `‖∂d(x,y)/∂y‖` must equal the conformal factor $\lambda(y) = 2/(1 - c\lVert y\rVert^2)$ — the
    unit-speed statement in Euclidean coordinates. Inside the floored band the old arm returned
    exactly `0` instead: 23 of 60 probed gradient cells, across both dtypes, three curvatures and
    both radii. A loss separating two nearby points through slot 2 received no gradient at all. At
    $y = x$ the derivative does not exist and the convention is the finite, direction-free 0, which
    `safe_sqrt`'s double-`where` now supplies (it also keeps `dist(x, x)` exactly 0).

### Which Version to Use?

**General recommendation**: `VERSION_MOBIUS_DIRECT` (version 0)
- Fastest
- Fewest intermediate operations
- Best for most applications

**Special cases**:
- **Near-boundary points** (||x|| > 0.9): Use `Poincare(dtype=jnp.float64)`, or convert to the
  hyperboloid via `isometry_mappings.poincare_to_hyperboloid` and use `Hyperboloid.dist` —
  `dist`/`logmap` are now genuinely safe there at any representable radius (see [above](
  #the-hyperboloids-two-point-cancellation-failure-mode)). The contrast that motivates this
  advice: the Poincaré ball itself cannot even *represent* a point past $d_0 \approx 12.65/\sqrt{c}$
  (float32) / $27.7/\sqrt{c}$ (float64) — `proj`'s boundary clamp saturates there — while the
  hyperboloid chart's representable ceiling is $\sqrt{c}\,d = \ln(2\,\texttt{finfo.max}) \approx 89$
  (float32), where $\cosh$ overflows — with `dist` itself giving up a fraction of a radius earlier,
  at $\ln(\texttt{finfo.max}) \approx 88.7$, where the polar frame's $x_0 + \lVert x_s\rVert = e^a$
  does.
- **Very high dimensions** (> 1000): `VERSION_METRIC_TENSOR` (version 2) may be more stable
- **Debugging**: Compare all versions — significant differences indicate numerical issues

### Hyperboloid Distance Versions {#hyperboloid-distance-versions}

The Hyperboloid manifold has its own two-way `version_idx`, orthogonal to the Poincaré versions
above. The same two slots select an arm of both the pairwise `dist` and the origin distance
`dist_0`, which are different implementations. A `version_idx` outside {0, 1} raises `ValueError`:

| slot | `dist` arm | `dist_0` arm | floor |
| --- | --- | --- | --- |
| `VERSION_DEFAULT` (0) | cancellation-free hyperbolic haversine | `arcsinh(√c·‖x_s‖)/√c` | none: exactly 0 at coincidence / at the origin |
| `VERSION_SMOOTHENED` (1) | the same, floored in quadrature | the same, with `‖x_s‖` floored in quadrature | `2·arcsinh(10·eps)/√c` and `arcsinh(20·eps)/√c`, equal to first order: ≈2.4e-6/√c (float32), ≈4.4e-15/√c (float64) |

```python
from hyperbolix.manifolds import Hyperboloid
import jax.numpy as jnp

hyperboloid = Hyperboloid()
x = hyperboloid.proj(jnp.array([1.0, 0.1, 0.2]), c=1.0)
y = hyperboloid.proj(jnp.array([1.0, 0.3, 0.4]), c=1.0)
c = 1.0

# VERSION_DEFAULT (0): cancellation-free hyperbolic-haversine distance — the default.
d0 = hyperboloid.dist(x, y, c, version_idx=hyperboloid.VERSION_DEFAULT)

# VERSION_SMOOTHENED (1): same evaluation, with a strictly-positive floor at coincidence.
d1 = hyperboloid.dist(x, y, c, version_idx=hyperboloid.VERSION_SMOOTHENED)
```

**When to use which**:

- `VERSION_DEFAULT` — the default, and the right choice for new code. Accurate at any
  representable radius (see [above](#the-hyperboloids-two-point-cancellation-failure-mode)).
- `VERSION_SMOOTHENED` — same numerics, but coincident points return a small positive distance
  with a well-defined gradient instead of exactly 0. Useful when a downstream `1/dist` or `log
  dist` would otherwise divide by zero. The floor is tiny: $2\,\mathrm{arcsinh}(10\epsilon)/\sqrt{c}
  \approx 2.4\text{e-}6/\sqrt{c}$ in float32 (vs. float64's $\approx 4.4\text{e-}15/\sqrt{c}$) — a
  large drop from the old softplus floor of $\mathrm{acosh}(1 + \ln 2/\beta)/\sqrt{c}
  \approx 0.166/\sqrt{c}$, which shifted *every* distance in the working range, not just
  coincident ones.

### Using Versions with JIT

Manifold operations are single-point functions: batch them with `jax.vmap` and compile the result
with `jax.jit` yourself. An uncompiled `vmap` re-traces dozens of primitives on every call and
runs 10-100x slower than the compiled path; that is the expected cost of the calling convention,
not a bug.

`version_idx` does **not** have to be a static argument. The version switch is a `lax.switch`,
which accepts a traced index, so a dynamic `version_idx` compiles and runs. Making it static
(baking it into the function body, or `static_argnames`) is a compile-size optimization: it
lets XLA drop the branches you do not use.

```python
import jax
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Recommended: bake the version into the function body, then jit the batched call.
@jax.jit
def compute_distances(x_batch, y_batch, c):
    return jax.vmap(
        lambda x, y: poincare.dist(x, y, c, version_idx=0)
    )(x_batch, y_batch)

# Or mark it static explicitly
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
# Manifold functions handle this internally: every norm on a differentiated path is
# safe_sqrt(sum(v**2)), which returns an exact 0 with an exactly-zero gradient at v = 0
# (see "Norms: One Reduction, Gradient-Safe at Zero" above). If you need the same in your
# own code, use the library's primitives rather than rolling a floor:
from hyperbolix.utils import safe_normalize, safe_sqrt
import jax.numpy as jnp

norm = safe_sqrt(jnp.sum(v**2, axis=-1))  # 0 at v = 0, gradient 0, one reduction
unit = safe_normalize(v)                  # exact zero vector at v = 0, unit vector otherwise
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
def clip_curvature(c, min_c=0.1, max_c=10.0):
    return jnp.clip(c, min_c, max_c)
```

## Checking Manifold Constraints

### Validation Functions

Each manifold provides `is_in_manifold` for validation:

```python
from hyperbolix.manifolds import Poincare, Hyperboloid
from hyperbolix.nn_layers import spatial_to_hyperboloid
import jax.numpy as jnp

poincare = Poincare()
hyperboloid = Hyperboloid()

# Poincaré ball: c·||x||² < 1
x = jnp.array([0.5, 0.3])
assert poincare.is_in_manifold(x, c=1.0)

# Hyperboloid: -x₀² + Σxᵢ² = -1/c  (with x₀ > 0). Building the ambient point
# from its spatial part is the only way to land on the sheet exactly:
# x₀ = sqrt(||x_spatial||² + 1/c).
x_ambient = spatial_to_hyperboloid(jnp.array([0.2, 0.3, 0.1]), 1.0, 1.0)  # (dim+1,)
assert hyperboloid.is_in_manifold(x_ambient, c=1.0)
```

### The `atol` Convention

`is_in_manifold` and `is_in_tangent_space` take `atol: float | None = None`.
Left as `None`, every manifold resolves it through
`hyperbolix.manifolds._base.default_atol(dtype) = sqrt(finfo(dtype).eps)` —
`3.45e-4` in float32, `1.49e-8` in float64. An explicit value is used as given:
it is never floored, clamped, or ignored (through v1.0.0 the hyperboloid floored
it at `1e-4`, so no caller could tighten it, and the Poincaré ball dropped it
entirely). Ball membership tests
the dimensionless residual `c||x||² - 1`, so one tolerance means the same thing
at every curvature.

The float64 default is strict enough to matter at large radii: a genuinely
on-sheet hyperboloid point past hyperbolic distance ~11 accumulates more than
`1.49e-8` of Lorentz residual just from storing `x₀`, so validate far-out points
with an explicit `atol` rather than assuming the default is a bug.

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
    - ✅ **On the Poincaré ball, use float32 for distances < 7, float64 for larger; on the
      hyperboloid, `dist`/`logmap`/`sqdist`/`tangent_norm`/`expmap`/`ptransp`/`tangent_proj`/
      `tangent_inner`/`egrad2rgrad`/gyro `addition`/`busemann` are accurate to the
      point-representation floor at any radius in float32 — see
      [Known Limitations](#hyperboloid-known-limitations) for the exceptions**
    - ✅ **Project after operations that might violate constraints**
    - ✅ **Keep points away from boundary** (max norm < 0.9/√c)
    - ✅ **Use conservative learning rates** (< 1e-3 for Poincaré, < 5e-4 for Hyperboloid)
    - ✅ **Use protected math functions** (`hyperbolix.utils.math_utils`)
    - ✅ **Monitor conformal factors** during training
    - ✅ **Validate manifold constraints** in debugging
    - ✅ **Use `VERSION_MOBIUS_DIRECT` for Poincaré distance** unless issues arise
    - ✅ **Clip curvature** if learnable (0.1 < c < 10.0)
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
