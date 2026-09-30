# Numerical Stability Guide

Best practices for maintaining numerical precision in hyperbolic operations.

## Overview

Hyperbolic geometry presents unique numerical challenges due to the exponential growth of the conformal factor near the boundary and the involvement of hyperbolic functions (cosh, sinh, atanh). This guide explains these challenges and provides strategies to maintain numerical stability.

!!! warning "Key Challenges"
    - **Conformal factor explosion**: λ(x) grows exponentially as points approach the boundary
    - **Float32 limitations**: ~7 significant digits, not enough for critical operations past scaled radius $a = \sqrt{c}\,d \approx 10$ from the origin (see the [table](#precision-requirements-by-distance) below)
    - **Hyperbolic function overflow**: cosh/sinh overflow for large arguments
    - **Division by near-zero**: Operations involving 1 - c||x||² near the boundary

    These challenges are specific to the Poincaré ball. The hyperboloid is covered [below](#the-hyperboloids-two-point-cancellation-failure-mode); Klein and the half-space model have their own limits, see [Klein](#klein-numerics) and [Half-Space](#halfspace-numerics).

Each model's pairwise operations hold up to a scaled radius $a = \sqrt{c}\,d$ (at $c = 1$). Past it
the chart cannot represent the point; that is a property of the chart, not a bug.

| Model | Good to $a$ (float32 / float64) | What sets the limit |
|---|---|---|
| Hyperboloid | 16.6 / much further | the storage floor $\varepsilon\sinh(a)/\sqrt{c}$ of a stored point |
| Poincaré | 12.65 / 27.7 (13.80 / 28.88 at $c = 0.1$) | the ball chart's ceiling, see [The Round-Trip Ceiling](#poincare-roundtrip-ceiling) |
| Klein | 6.32 / 13.86 | half the Poincaré radius, and the floor $\varepsilon\cosh^2(a)$, see [Klein](#klein-chart-ceiling) |
| HalfSpace | `inf`/NaN past 88.7 / 709.8 | a stored point's rounding, which grows with the distance to the vertical axis, see [Half-Space](#halfspace-numerics) |

## Float Precision: Float32 vs Float64

### When to Use Each

**Float32 (default)**:

- Sufficient up to scaled radius $a = \sqrt{c}\,d \approx 7$, and up to $a \approx 10$ outside critical operations
- Much faster on GPU: float64 runs at half rate on A100/H100 and 1/32–1/64 rate on consumer cards
- Lower memory footprint (important for large models)
- ~7 significant decimal digits

**Float64 (high precision)**:

- Recommended for critical operations from $a \approx 10$; required past the float32 chart ceiling, $a \approx 12.65$ at $c = 1$ (13.80 at $c = 0.1$)
- Better numerical stability in edge cases
- ~15-16 significant decimal digits
- Use for research, validation, or stability-critical applications

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import Poincare

# Float32 (default)
poincare_f32 = Poincare()
x = jnp.array([0.1, 0.2])
y = jnp.array([0.8, 0.5])
dist = poincare_f32.dist(x, y, c=1.0)

# Float64 (high precision) — inputs are automatically cast
jax.config.update("jax_enable_x64", True)  # required for float64
poincare_f64 = Poincare(dtype=jnp.float64)
dist = poincare_f64.dist(x, y, c=1.0)  # returns float64
```

### Precision Requirements by Distance

| Scaled radius $a = \sqrt{c}\,d$ | Float32 relative error, median / max | Recommended Precision |
|----------------------|------------------|----------------------|
| 1 | 6.1e-8 / 2.5e-7 | float32 |
| 3 | 1.2e-7 / 5.1e-7 | float32 |
| 5 | 2.7e-7 / 2.4e-6 | float32 |
| 7 | 5.8e-6 / 1.5e-5 | float32 |
| 10 | 4.6e-5 / 2.2e-4 | float32; float64 for critical ops |
| 12 | 4.8e-4 / 1.1e-3 | float64 for critical ops |
| 12.5 | 8.2e-4 / 2.5e-3 | float64 for critical ops |
| ≥ 12.65 | past the float32 chart ceiling: the ball cannot store the point | **float64 required** |

*Poincaré ball only: float32 against float64 for `dist_0`, `logmap_0`, `expmap_0` and their
round trip, the worst of the four and of $c \in \{0.1, 1\}$. The last row is the ceiling at
$c = 1$ (13.8 at $c = 0.1$; see [The Round-Trip Ceiling](#poincare-roundtrip-ceiling)). For the
hyperboloid, see [The Hyperboloid Origin Chart](#hyperboloid-origin-chart) and the
[two-point section](#the-hyperboloids-two-point-cancellation-failure-mode) below.*

!!! tip "Quick Check"
    Compute the largest scaled radius $a = \sqrt{c}\,d_0$ of your embeddings with your own $c$.
    From $a \approx 10$, use float64 for critical operations. A float32 point cannot sit past the
    chart ceiling $2\,\mathrm{atanh}(1 - \sqrt{c}\,\varepsilon^{0.75})$, 12.65 at $c = 1$ and
    13.80 at $c = 0.1$ (14.33 below $c \approx 0.035$, where the `atanh` clip binds first; see
    [The Round-Trip Ceiling](#poincare-roundtrip-ceiling)), so a maximum within ~0.1 of it means
    the ball has capped points, and float64 is required:

    ```python
    import math

    import jax
    import jax.numpy as jnp
    from hyperbolix.manifolds import Poincare

    c = 0.1  # the curvature your model uses
    poincare = Poincare()
    v_batch = jax.random.normal(jax.random.PRNGKey(0), (256, 16))  # stand-in for your features
    x_batch = jax.vmap(lambda v: poincare.expmap_0(v, c))(v_batch)  # ... and embeddings

    a = jnp.sqrt(c) * jax.vmap(lambda x: poincare.dist_0(x, c))(x_batch)  # scaled radius
    eps = float(jnp.finfo(jnp.float32).eps)
    # proj margin √c·eps**0.75; below c ≈ 0.035 the tanh/atanh clip 10·eps binds first
    margin = max(math.sqrt(c) * eps**0.75, 10 * eps)
    a_ceiling = 2 * math.atanh(1 - margin)  # float32: 12.65 at c = 1, 13.80 at c = 0.1
    print(f"max a = {float(a.max()):.2f}, float32 ceiling = {a_ceiling:.2f}")
    # max a > 10: create Poincare(dtype=jnp.float64) for critical ops;
    # max a within ~0.1 of the ceiling: the ball has capped points, float64 is required
    ```

### TF32 on Ampere and Hopper GPUs

XLA:GPU runs float32 matmuls in **TF32** by default. Its 10-bit mantissa carries ~1e-3 relative
error against float32's ~1e-7, so it silently costs three of your seven significant digits.

hyperbolix splits its float32 dot products in two:

- **Geometry is pinned** to `jax.lax.Precision.HIGHEST` and is not configurable: the manifold
  vector dots, the Lorentz and Poincaré midpoints, the conv patch extraction, the attention
  score and aggregation einsums, and the point-to-hyperplane einsums of the MLR heads and of the
  PLFC, Poincaré++ and proper-velocity linear and conv layers. These dots cancel, and they are
  not where the throughput is.
- **The other layer weight GEMMs follow JAX**: `HTCLinear` (the attention Q/K/V projections
  included), `FGGLinear`/`FGGConv2D`, the FHCNN/FHNN linears, `HypLinearPoincare`, the VQ
  codebook matmul. On an Ampere or Hopper card these run in TF32 unless you say otherwise.

To run everything in full float32, set JAX's own knob:

```python
jax.config.update("jax_default_matmul_precision", "highest")

# or scoped to a block (both spellings are jit-cache aware — changing them re-traces):
with jax.default_matmul_precision("highest"):
    logits = model(x)
```

On an A100 (float32 against a float64 reference), the relative error of `FGGLinear` is
**2.6e-4** at the TF32 default against **6.8e-8** under `HIGHEST`. `HIGHEST` replaces one TF32
pass with a three-pass float32 emulation, measured **+5.5 %** on a jitted hyperbolic attention
forward.

**Depth decides whether you need it.** At depth 2 the default and `HIGHEST` train identically; the
float32 gradient error grows with depth (1.5 % at depth 16 for an `HTCLinear` stack against 2.5e-6
under `HIGHEST`). For a deep or gradient-sensitive stack, set the global knob. Details:
[`hyperbolix.utils.precision`](../api-reference/utils.md#matmul-precision).

`HIGHEST` is a no-op on CPU (there is no TF32 path) and for float64 anywhere.

### The Hyperboloid's Two-Point Cancellation Failure Mode

The table above is for the Poincaré ball. For the hyperboloid, being far from the origin is not
the problem by itself; two points far from the origin **and close together** is. The literal
Minkowski inner product $\langle x, y\rangle_L = -x_0y_0 + \langle x_s, y_s\rangle$ subtracts two
terms of size about $e^{\sqrt{c}\,(d_0(x) + d_0(y))}$ to leave a result of size
$e^{\sqrt{c}\,d(x,y)}$. The digits lost are set by the **Gromov-product-like quantity**

$$
\sqrt{c}\,\bigl(d_0(x) + d_0(y) - d(x, y)\bigr),
$$

and once it exceeds $\ln(1/\epsilon)$ (15.9 for float32, 36.0 for float64) every digit of the
result is cancellation noise. Nearby points on a shared geodesic ray, or clustered leaf embeddings
in a deep hierarchy, reach it easily.

hyperbolix never forms that product in its two-point primitives. `dist`, `logmap`, `sqdist` and
`tangent_norm` are cancellation-free under both version slots (`VERSION_DEFAULT` /
`VERSION_SMOOTHENED`, see [Hyperboloid Distance Versions](#hyperboloid-distance-versions)). The
single-point operations are covered in [The Hyperboloid Origin Chart](#hyperboloid-origin-chart),
and the places that still lose accuracy are listed below.

#### Known Limitations {#hyperboloid-known-limitations}

`expmap`, `ptransp`, `tangent_proj`, `tangent_inner`, `egrad2rgrad`, gyro `addition` and
`busemann` are also cancellation-free. `is_in_manifold` and `is_in_tangent_space` compare the
stored time slot with the value the spatial part implies, so an on-sheet point passes a fixed
`atol` far out (see [The `atol` Convention](#the-atol-convention)).

Five places still lose accuracy at large radius:

1. **`ptransp`'s direction below the float32 angular resolution.** A transport step shorter than
   the point-representation floor `eps·sinh(a)/√c` is lost in the endpoints' storage rounding, and
   no formula recovers it.
2. **Attention scores through a GEMM.** The Lorentzian similarity score behind
   `HyperbolicFullAttention` forms `2 + 2⟨Q,K⟩_L` from a matrix product, and a GEMM cannot be
   made cancellation-free the way a single pairwise `dist` can — see
   [Full Attention's Float32 Score Floor](#attention-score-floor) below.
3. **Transport to the origin from a far point.** The wrapped-normal `log_prob` uses its own
   transport to the origin and is accurate up to the float32 storage of its tangent vector $u$
   (the ambient chart stores a tangent vector's components at $\cosh a$ times its length). The
   generic `Hyperboloid.ptransp(v, far_point, origin)` still cancels: its radial component has a
   float32 error of 1.95e-3 at scaled radius 10.
4. **The MLR score, when the hyperplane itself sits far from the origin.** The reference Lorentz
   MLR subtracts two terms of size `e^(a+ρ)`, with `ρ` the scaled hyperplane offset, so a large
   bias costs as many digits as a large input radius — see
   [The MLR Score at a Large Hyperplane Offset](#mlr-large-bias) below.
5. **Busemann gradients at a far stored point.** When the Busemann layers score a stored float32
   point, from scaled radius $a \approx 29.8$ at $c = 1$ ($29.4$ at $c = 0.3$) the gradients of the
   scores with $\langle x_s, \omega\rangle \ge 0$ can be up to 0.9 off relative to their largest
   entry while the scores stay right (measured on CPU; see
   [Tangent Inputs to the HNN++ and Busemann Layers](#poincare-tangent-input)).

See also [Gyro-Difference and GyroBatchNorm Centering at Large Radius](#gyro-difference) and
[ProperVelocity `dist`/`logmap` Through the Exact Lift](#pv-dist-lift) below.

#### Full Attention's Float32 Score Floor {#attention-score-floor}

`HyperbolicFullAttention`'s scores are `2 + 2⟨Q,K⟩_L`, a difference of two Minkowski terms each of
size `cosh(a_q)·cosh(a_k)/c`. A matrix product returns the Gram matrix only to absolute `eps`, so
no GEMM spelling avoids the cancellation. The absolute error on one score is about
`eps·cosh(a_q)·cosh(a_k)/(c·scale)`: in float32 (`c = 1`, `scale = 1`) 4.8e-3 at `a = 6`, 0.26 at
`a = 8`, and 2.0 at `a = 9`. From `a ≈ 8` it exceeds the score spread softmax is meant to resolve,
and the weights come out wrong while staying finite.

The cheaper first remedy is a `HyperboloidGyroRMSNorm` in front of the layer, or a smaller `c`.
`score_dtype=jnp.float64` runs the score arithmetic as a float64 island and casts back before the
softmax; it does not help activations that are themselves stored in float32 past `a ≈ 8`.

The value aggregation has its own form, `lorentz_midpoint(form=...)`, set by `centroid_form` on
`HyperbolicFullAttention` and `LorentzMLA`. `"variance"` (the default of `lorentz_midpoint` and
`HyperbolicFullAttention`) is accurate for far-out values but costs `O(N·M·D)` work and compiles
slowly on an A100: `LorentzMLA` at S = 512 takes 104 s against 9.6 s and did not finish at S = 2048
in 10 min; `HyperbolicFullAttention` at N = 512 takes 15.9 s against 4.0 s. `"gemm"` (the default of
`LorentzMLA`, as in HELM) is one GEMM, but in float32 it loses the centroid once the *values* pass
`a ≈ 6`, even when the scores are accurate: values at `a ≈ 9` give 11.4 nats of error against 6.4e-4
for `"variance"`. bfloat16 activations (`eps` 2^16 times float32's) make both worse.

#### The MLR Score at a Large Hyperplane Offset {#mlr-large-bias}

The hyperplane MLR heads (`HypRegressionHyperboloid`, `HypRegressionPV`, `HypRegressionPoincarePP`)
evaluate the reference Lorentz MLR score of Bdeir et al. 2023,

$$
\alpha = -x_0\sinh(\sqrt{c}\,r)\,\lVert z\rVert + \cosh(\sqrt{c}\,r)\,\langle z, x_s\rangle,
$$

for a hyperplane with normal `z` and offset `r`. Writing `a = √c·d` for the scaled radius of the
input point, `ρ = √c·r` for the scaled offset, and `θ` for the angle between the point and the
hyperplane normal, that is

$$
\sinh(a)\cosh(\rho)\cos\theta \;-\; \cosh(a)\sinh(\rho),
$$

two terms of size `e^(a+ρ)` whose difference is `O(1)` for a point near the hyperplane, so the
rounding error is `eps·e^(a+ρ)`: **a hyperplane far from the origin costs as many digits as an
input far from the origin.** The same expression sits in the PLFC / ILNN layers, the Poincaré++
MLR (where `ρ = 2√c·r`), and `FGGLinear`'s spacelike-`V` GEMM.

Median relative error of the float32 gradient with respect to the point, hyperboloid, points
exactly on the hyperplane:

| `a` \ `ρ` | 0.5 | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- |
| 8 | 8e-8 | 1e-7 | 4e-7 | 8e-6 | n/a |
| 10 | 2e-7 | 4e-7 | 5e-6 | 3e-4 | 0.52 |
| 12 | 1e-5 | 5e-5 | 3e-4 | 0.013 | 0.84 |
| 14 | 9e-4 | 1.3e-3 | 0.013 | 0.32 | 0.98 |
| 16 | 0.042 | 0.040 | 0.39 | 0.86 | 1.0 |

`n/a` marks `ρ ≥ a`, where the hyperplane cannot cross the point. The rule of thumb: **the float32
gradient is about 0.1 % wrong at `a + ρ ≈ 15`, about 1 % at 16, roughly 40 % by 18, and entirely
wrong by 22**. At `ρ = 8` the *score* itself can take the wrong sign. Float64's limit is about
`a + ρ ≈ 36` (extrapolated).

It takes an input far from the origin, **and** a hyperplane far from the origin, **and** points
close to that hyperplane. With normalized features (`a ≲ 8`) and an `O(1)` bias none of this
arises. Remedies: run the head in float64, or keep `a + ρ` under ≈ 15 by bounding the trunk's
radius and weight-decaying the head's bias. A cancellation-free half-angle form was measured at
1.8–2.5× the cost and not adopted. `FGGLinear`'s spacelike `V` has a related `eps` floor on its
column norm, which matters only for a small weight column with a large bias.

#### Input Overflow {#input-overflow-fingerprint}

A hyperboloid point whose spatial part passes float32's `1.84e19` has an `inf` time coordinate
(scaled radius `a ≈ 45` at `c = 1`; see [Norms](#safe-norms) for where this ceiling comes from).
`HypLinearHyperboloidPLFC`, `HypConv2DHyperboloidILNN` and `HypLinearHyperboloidBusemann` pass the
resulting non-finite score through their sinh lift, so the loss is NaN at the minibatch where it
happens. In 1.3.0 and earlier the lift clipped it to a finite, saturated point: the loss stayed
finite and the next kernel and bias gradient was 100 % NaN.

hyperbolix adds no input-side bound here, so as not to hide the divergence. Monitor `dist_0` (or
the time coordinate) at the layer's input and bound the trunk that feeds it in your model.

#### Where Float32 Overflow Still Collapses Silently {#silent-overflow-sites}

The library lets non-finite values propagate so that a diverging model fails loudly. At the sites
below, a float32 sum of squares or a clip still turns an absurdly large finite input (or, for the
Poincaré lifts, an `inf` score) into a finite point. Each needs an input at or near the float32
coordinate ceiling `√FLT_MAX ≈ 1.84e19` (see [Norms](#safe-norms)), which only an already-diverging
model reaches, so the code is left as it is.

- **Poincaré and κ-stereographic `proj`:** a finite `x` with `‖x‖ > 1.84e19` becomes the zero vector.
- **`Poincare.expmap`, `Klein.expmap`:** `‖v‖ > 1.84e19` returns the base point.
- **`Hyperboloid.expmap_0`:** `‖v‖ > 1.84e19` returns the origin; a shorter `v` whose image passes the ceiling gets an `inf` time slot, as intended.
- **`HyperPPFeatureScaling`:** the mean of squares in its Flax `RMSNorm` overflows for a row with `‖x‖ > 1.84e19`; the row becomes zero, i.e. the origin after `expmap_0`.
- **`HRCBatchNorm` (train mode):** a feature whose batch sum of squares overflows equals its BatchNorm bias in every row, and its running variance becomes `inf` for good, so eval mode does the same. This starts below the ceiling, at `1.84e19/√N` for `N` rows near one value (1.15e18 at `N = 256`).
- **HRC (`HRCLayerNorm`, `HRCRMSNorm`, …):** reads only the spatial part, so the `inf` time slot of a point past the ceiling is dropped; the LayerNorm/RMSNorm mean of squares then overflows (`‖x_s‖ > 1.84e19`) and the row becomes the LayerNorm bias or the origin.
- **`lorentz_residual` (`w_y ≤ 1`):** the `4h²/c` in its normalizer, `h = sinh(√c·d(x, y)/2)`, overflows once `h > √(min(c, 1)·FLT_MAX)/2` (9.2e18 at `c = 1`), and the output is the origin — e.g. two points at spatial radius 1.5e19, 120° apart.
- **`lorentz_midpoint` with `c > 1`:** its normalizer, about `c·x₀²` times the weighted variance of the directions `x_s/x₀` (at most 1), overflows in a widely spread cloud from time coordinates of about `1.84e19/√c`, and the output is the origin. For `c ≤ 1` that is past the ceiling.
- **FHCNN `normalize=True`:** a linear output whose spatial norm exceeds 1.84e19 gets spatial part 0 and a finite time slot — a finite point off the hyperboloid; its spatial part has zero gradient (the time slot's sigmoid gate still gets one).
- **`HypLinearPoincarePP`, `HypLinearPoincareBusemann`:** an `inf` score is clipped — by the `sinh` argument clip at `±0.99·ln FLT_MAX ≈ ±87.8`, or by `v_max` — to a finite point at the ball's edge, where the hyperboloid lift above passes it through.
- **`HypConv2DPoincare`:** the same `sinh` clip; an `inf` score comes out as a finite tangent vector along its channel with `√c‖out‖ = 43.9`, half the clip.

### The Hyperboloid Origin Chart {#hyperboloid-origin-chart}

The time coordinate $x_0 = \cosh(\sqrt{c}\,d)/\sqrt{c} \approx (1 + c\,d^2/2)/\sqrt{c}$ cannot
resolve a small radius. `dist_0` and `logmap_0` therefore read the radius off the **spatial**
part:

$$
d_0(x) = \frac{\operatorname{arcsinh}(\sqrt{c}\,\lVert x_s\rVert)}{\sqrt{c}},
\qquad
\log_0(y) = \Bigl[0,\; \frac{\operatorname{arcsinh}(u)}{u}\, y_s\Bigr],\quad u = \sqrt{c}\,\lVert y_s\rVert .
$$

$\lVert \log_0(y)\rVert = d_0(y)$ holds by construction at every radius, with float32 median
relative error ≤ 1e-7 from radius 1e-6 to 40 ($c = 1$). The pairwise `dist` reads its radial gap
off the spatial part too, so float32 `dist(origin, x)` agrees with `dist_0(x)` to 3.1e-7 at every
radius from 1e-8 up; `dist_0` stays marginally cheaper and tighter. An `inf` spatial entry gives an
infinite tangent vector from `logmap_0`, not NaN.

### Norms: One Reduction, Gradient-Safe at Zero {#safe-norms}

The per-sample Euclidean norms in hyperbolix are either `safe_sqrt(sum(v**2))`, a single pass over
the data, or the max-scaled two-pass `safe_norm`. Both return an exact `0` with an **exactly zero**
VJP at the zero vector, where `jnp.linalg.norm`'s VJP is NaN, and both pass a non-finite input
through as `inf` rather than NaN.

The single reduction overflows once a coordinate passes $\sqrt{\texttt{finfo.max}}$
($1.84\times10^{19}$ in float32). On the hyperboloid that is scaled radius $a \approx 45$
(float32) / $356$ (float64) for `proj` and `dist_0`; `logmap_0` uses the two-pass norm and reaches
$\approx 89$ / $710$. Typical embeddings sit far inside these limits. Past the ceiling
(float32, $c = 1$) the single-reduction sites return:

| Operation | Past coordinate $1.84\times10^{19}$ |
|---|---|
| `Hyperboloid.proj`, `proj_batch` | $x_0 = \infty$, spatial part unchanged, no NaN |
| `Hyperboloid.dist_0` (both version slots) | $\infty$ |
| `spatial_to_hyperboloid` (`HTCLinear`), FHCNN linear forward | $x_0 = \infty$, no NaN |
| FGG linear forward | non-finite (NaN or $\infty$, depending on the input) |
| Poincaré and $\kappa$-stereographic `proj` (the boundary clamp) | the **origin**, a finite point with a finite (exactly zero) gradient |
| `Poincare.expmap` | the **base point**, a finite point (the origin, when the base is the origin) |

Where a norm is a *divisor*, floor it multiplicatively **around** the square root,
`floor_at(safe_sqrt(sum(x**2)), MIN_NORM)`, not under it: under the `sqrt`, the untaken branch of
the surrounding `where` still has `sqrt'(0) = inf`, and $0 \times \infty$ is NaN. When the floored
norm feeds a $\sinh(t)/t$, floor $t$ itself, not only the denominator, or the ratio is $0$ at
$t = 0$ instead of $1$.

In your own code (all primitives are in `hyperbolix.utils.math_utils`), use `safe_norm` where you
need the full float32 exponent range, and `safe_sqrt(sum(v**2))` on hot per-sample paths.

## Storage vs. Compute Dtype

- **Compute precision** is the dtype manifold operations run in, set by the
  manifold's `dtype` (e.g. `Poincare(dtype=jnp.float64)`). Manifold methods
  cast their array arguments to it on entry.
- **Storage dtype** is the dtype of a layer's parameters and persistent state
  (batch-norm statistics, VQ codebooks), set by the `param_dtype` argument on
  every NN layer (default `jnp.float32`, as in Flax).

A float32 parameter entering a float64 manifold operation is promoted for that
computation only. The Riemannian optimizers run in the manifold dtype and cast
updates and momentum buffers back to the parameter's dtype.

**The recommended high-precision recipe** is float64 *compute* with float32
*storage*: full precision where the geometry needs it, at half the
parameter/optimizer-state memory:

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
    Precision in hyperbolic networks is lost in the manifold operations, not
    in weight storage. Use `param_dtype=jnp.float64` only for reproducibility
    studies or numerical debugging.

## The Conformal Factor Problem

The **conformal factor** of the Poincaré ball, $\lambda(x) = 2/(1 - c\lVert x\rVert^2)$, scales
tangent vectors in the exponential and logarithmic maps and converts Euclidean gradients to
Riemannian ones.

### Exponential Growth

As points move toward the boundary ($\lVert x\rVert \to 1/\sqrt{c}$), λ(x) explodes. A point at
scaled radius $a = \sqrt{c}\,d_0$ has $\sqrt{c}\,\lVert x\rVert = \tanh(a/2)$, so
$\lambda(x) = 2\cosh^2(a/2) = 1 + \cosh a$, which grows like $e^a/2$:

```python
import jax.numpy as jnp
from hyperbolix.manifolds import Poincare

poincare = Poincare()
c = 1.0  # at c = 1 the scaled radius a is the distance from the origin

for a in [0, 2, 4, 6, 8, 10, 12, 14, 20]:
    # expmap_0(v) lands at distance 2‖v‖ from the origin, so ask for ‖v‖ = a/2
    x = poincare.expmap_0(jnp.array([a / 2, 0.0]), c=c)
    norm = jnp.linalg.norm(x)
    lambda_x = 2.0 / (1.0 - c * norm**2)
    print(f"a={a:2d}: ||x||={norm:.6f}, λ(x)={lambda_x:9.1f}, dist_0(x)={poincare.dist_0(x, c):.4f}")
```

Output:
```
a= 0: ||x||=0.000000, λ(x)=      2.0, dist_0(x)=0.0000
a= 2: ||x||=0.761594, λ(x)=      4.8, dist_0(x)=2.0000
a= 4: ||x||=0.964028, λ(x)=     28.3, dist_0(x)=4.0000
a= 6: ||x||=0.995055, λ(x)=    202.7, dist_0(x)=6.0000
a= 8: ||x||=0.999329, λ(x)=   1491.6, dist_0(x)=8.0000
a=10: ||x||=0.999909, λ(x)=  11008.7, dist_0(x)=9.9995
a=12: ||x||=0.999988, λ(x)=  81442.8, dist_0(x)=12.0008
a=14: ||x||=0.999994, λ(x)= 155344.6, dist_0(x)=12.6465
a=20: ||x||=0.999994, λ(x)= 155344.6, dist_0(x)=12.6465
```

Each step of 2 in $a$ multiplies λ by about $e^2 \approx 7.4$. Past the float32 [chart ceiling](#poincare-roundtrip-ceiling),
$a \approx 12.65$ at $c = 1$, `expmap_0` caps the point, so $a = 14$ and $a = 20$ give the same
point.

Pick the dtype from the scaled radius ([table above](#precision-requirements-by-distance)); for
float64, enable `jax_enable_x64` and use `Poincare(dtype=jnp.float64)`. `addition`, `expmap` and
the layers already project their outputs.

Near the origin, float32 accuracy depends on the backend's transcendental kernels, which are a few
ulps off; hyperbolix's own `tanh`/`atanh` wrappers (float32: odd series below 1/8, `expm1`/`log1p`
forms above) make the CPU backend match the GPU there.

### The Round-Trip Ceiling {#poincare-roundtrip-ceiling}

`proj` keeps points inside $1/\sqrt{c}$ by an absolute margin of `eps**0.75`
(`_gyrovector_core._max_norm`). A stored point therefore has
$\sqrt{c}\,\lVert x\rVert \le 1 - \sqrt{c}\,\varepsilon^{0.75}$, and the ceiling in scaled radius
is

$$
a_{\max} = \sqrt{c}\,d_0 = 2\,\mathrm{atanh}\bigl(1 - \sqrt{c}\,\varepsilon^{0.75}\bigr).
$$

| $c$ | float32 | float64 |
| --- | --- | --- |
| 1 | 12.65 | 27.73 |
| 0.1 | 13.80 | 28.88 |

Past the ceiling `expmap_0` saturates, and `logmap_0(expmap_0(v))` returns the ceiling instead of
`v`. In float32 below $c \approx 0.035$ the `tanh`/`atanh` wrappers' clip at $1 - 10\varepsilon$
binds first, and `expmap_0`, `logmap_0` and `dist_0` stop at $a \approx 14.33$.

The margin caps the conformal factor near $1/(\sqrt{c}\,\varepsilon^{0.75})$: 1.6e5 in float32 and
5.5e11 in float64 at $c = 1$. If your embeddings need larger radii, switch to the hyperboloid or
to `ProperVelocity` rather than shrinking the margin.

### Pairwise Operations Near the Boundary {#factored-mobius-denominator}

<a id="poincare-far-pairs"></a>

The Poincaré pairwise operations are cancellation-free up to the chart ceiling; past it the ball
cannot store the point.

- **Distance.** `dist` slots 0 and 2 (the default and the metric-tensor slot) evaluate, with
  $B_x = 1 - c\lVert x\rVert^2$,

    $$
    d(x, y) = \frac{2}{\sqrt{c}}\operatorname{arcsinh}\!\left(\frac{\sqrt{c}\,\lVert x - y\rVert}{\sqrt{B_x B_y}}\right),
    $$

    which has no domain to clip. Two float32 points on opposite sides at scaled radius 7.2 each
    (true $\sqrt{c}\,d = 14.4$, $c = 1$) give 14.400112. Slot 1 (`VERSION_MOBIUS`) forms
    $(-x) \oplus y$, a point at the full pair distance, and saturates at the ceiling: 12.637 for a
    true 14.4, with gradient relative error 1.0.
- **Möbius denominator.** `logmap`, `addition` and `gyr` divide by
  $1 + 2sc\langle x, y\rangle + c^2\lVert x\rVert^2\lVert y\rVert^2$. `_mobius_denominator`
  rewrites $1 + t^2$ as $(1 - t)^2 + 2t$ with $t = c\,r_x r_y$, so the subtraction $1 - t$ happens
  before the square. `logmap` takes its length from the same `arcsinh` argument as `dist`.
- **Transport.** `ptransp` uses the gyration regrouped in $s = x + y$ ($= y - x$ for the transport), where each coefficient is
  $O(1 - c\lVert x\rVert^2)$ on its own; `ptransp(v, x, x)` returns `v` exactly.
- **`apollonian_dist`** writes its $G^2$ as $B_x B_y + c\lVert x - y\rVert^2$, a sum of two
  non-negative terms.

### Gradients at Capped Points {#poincare-divisor-floors}

Divisors built from $B = 1 - c\lVert x\rVert^2$ are floored at half the value `proj` leaves at the
cap (`_boundary_divisor_floor`), which no projected point reaches, so the floor does not zero the
gradients of capped points. Some float32 gradients at capped points stay wrong, with relative error
up to 1.0 (1.3 for Klein `addition`): Poincaré `expmap` with respect to its base point, Poincaré `addition` with respect to
$y$, Klein `expmap` with respect to its base point, and Klein `addition` on pairs of one capped and
one free point. The floor is not the cause.

### Tangent Inputs to the HNN++ and Busemann Layers {#poincare-tangent-input}

`HypRegressionPoincarePP` and `HypLinearPoincarePP` with `input_space="tangent"`,
`HypConv2DPoincare` (tangent in, tangent out), and the four Busemann layers with
`input_space="tangent"` score the tangent vector in closed form and never form the ball point.
For the Poincaré layers, with $t = \sqrt{c}\,\lVert v\rVert$, the scores use
$\lambda_x\sqrt{c}\,\langle x, \hat{z}\rangle = \sinh(2t)\,\langle v, \hat{z}\rangle/\lVert v\rVert$
and $\lambda_x - 1 = \cosh(2t)$, and `HypConv2DPoincare` maps its output back with `logmap_0` of
the lift in closed form.

These layers are therefore not bound by the ball's ceiling,
$t = \mathrm{atanh}(1 - \sqrt{c}\,\varepsilon^{0.75})$ ≈ 6.33 at $c = 1$: past it the scores keep
growing, and the conv output's norm is not capped. At $t = 8$ the float32
outputs match float64 to ≤ 1.6e-6 (max-abs error over max-abs value).

The Hyperboloid Busemann layers place $v = (0, v_s)$ at scaled radius $t = \sqrt{c}\,\lVert v_s\rVert$
and score $\sqrt{c}\,B^\omega = \log(\cosh t - \sinh t\,\langle\omega, \hat v\rangle)$. For random $\omega$ their float32 outputs and gradients stay within 1.2e-6 of float64 from
$t = 1$ to $t = 50$. One limit remains on every route: for $\omega$ within float32 rounding of
$\hat v$, the score depends on the misalignment of $\omega$ and $\hat v$ times $\sinh t$, which
float32 cannot resolve at large $t$ (at $t = 20$, $\sqrt{c}\,B$ is off by 1.7 to 1.9).

The Busemann layers check the manifold at construction and raise a `TypeError` for a wrong one:
`HypRegressionPoincareBusemann` and `HypLinearPoincareBusemann` accept only a `Poincare`,
`HypRegressionHyperboloidBusemann` and `HypLinearHyperboloidBusemann` only a `Hyperboloid`.

### `PoincareBatchNorm2D`'s Batch Mean at Large Radius {#poincare-batchnorm-mean}

`PoincareBatchNorm2D` averages the batch in Klein coordinates, $\sum\lambda x/\sum(\lambda - 1)$,
and projects that average with `proj`, so the batch mean cannot sit farther out than the Klein
chart's ceiling: 6.33 at $c = 1$ and 6.90 at $c = 0.1$ in float32 (13.86 and 14.44 in float64).
For a cluster at scaled radius 7 ($c = 1$), the float32 batch mean reads 6.31–6.33 and lies 0.68
from the float64 one. In float32, keep this layer's inputs inside scaled radius ≈ 6, or run it in
float64.

## Init Scale vs. Depth

Weight-init failures on hyperbolic layers come in two flavors, and only one of
them is loud. A **too-large** init pushes outputs toward the Poincaré
boundary or far up the hyperboloid, and you see `NaN` within a few steps. A **too-small** init fails *silently*:
for a linear-in-the-matmul layer (e.g. `HTCLinear`, whose `htc` tail applies no
nonlinearity), the per-layer input-Jacobian gain is

$$
g \approx \sigma_w \cdot \sqrt{\text{fan\_in}},
$$

and a stack compounds it as $g^{\text{depth}}$. When $g < 1$, the input variation the outputs
carry shrinks geometrically until float32 rounds it away. The stack is then a constant map:
gradients are ≈0 from step 0 and training never starts, with no `NaN` or warning to point at. This
freeze was observed: `HTCLinear` with `init_bound=0.02` froze stacks of depth ≥ 2. The distance is not the cause
(`Hyperboloid.dist` resolves a spatial gap of 1e-6 in float32). A likely cause, not yet measured:
`htc` multiplies the full point, time coordinate included, so every output carries an $O(1)$
offset from $x_0 \approx 1/\sqrt{c}$, and variation smaller than $\varepsilon$ times that offset is
rounded away.

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

At the **conv → FC boundary** of a hyperboloid CNN, an `(B, H', W', C)` feature map (one point
per pixel) must become one point per sample. `x.reshape(B, H' * W' * C)` concatenates `H'·W'`
time coordinates as if they were features. `Hyperboloid.hcat`, which stacks only the spatial parts
and rebuilds one time coordinate, stays on the manifold but inflates the radius.

The inflation is a dimension effect. For Gaussian-ish spatial parts
$\|v\|^2 \sim \chi^2_k$, so

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
    Inside `HypConv2DHyperboloidILNN`, LogCat covers a receptive field of $N = 9$. At the
    flatten, $N$ is the **entire feature map** (tens to low hundreds), so a naive flatten
    hands the head a point an order of magnitude farther out than its init expects. The
    observed symptom, from before the MLR `asinh` clamp was removed, was an MLR head
    whose logits all sat at that clamp at step 0 with near-zero gradients. The
    symptom cleared when the flatten used LogCat.

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
to `H'·W'·(A − 1) + 1`; size the head for it (`in_dim = 129` above). `hyp_avg_pool2d` keeps
the width at `A` but discards the spatial layout. Both are on the
[convolutional API page](../api-reference/nn-layers/convolutional.md#pooling-flattening-conv-fc-bridge).

## Proper Velocity: The Hyperboloid Without a Constraint

The Proper Velocity (PV) model (Chen et al. 2026) stores a point as the spatial part of a
hyperboloid point, in **unconstrained $\mathbb{R}^n$**: any finite vector is a valid point. It has
the hyperboloid's accuracy, and no better: `dist` and `logmap` go through the exact hyperboloid
lift (see [below](#pv-dist-lift)), and its tangent-space metric shares the hyperboloid's
formulation (see [below](#pv-tangent-metric)). What PV adds is convenience:

- no projection and no constraint drift after an update;
- a bounded factor $\beta_x = 1/\sqrt{1 + c\|x\|^2} \in (0, 1]$ in place of a conformal factor
  that blows up at a boundary;
- plain `optax` optimizers train PV layers without a Riemannian wrapper.

The PV distance from the origin, $d(0, x) = \mathrm{asinh}(\sqrt{c}\,\|x\|)/\sqrt{c}$, stays finite
in float32 for $\|x\|$ up to at least $10^2$ (checked by `test_pv_stability_at_large_norms`).

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import ProperVelocity

pv = ProperVelocity()
c = 1.0

# Any finite vector is a PV point; there is no boundary to project onto.
x_large = jnp.array([50.0, 0.0, 0.0])
d = pv.dist_0(x_large, c)      # asinh(50) ≈ 4.61
y = pv.logmap_0(x_large, c)    # finite tangent vector
x_rec = pv.expmap_0(y, c)      # round-trips to x_large
```

!!! note "Training PV layers"
    `HypLinearPV`, `HypConv2DPV`, and `HypRegressionPV` store their weights as plain `nnx.Param` (not `ManifoldParam`). Use a standard `nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)` — no `riemannian_adam` / `riemannian_sgd` wrapper is required.

### Where Each Model Stops

Pairwise operations hold to the scaled radius $a = \sqrt{c}\,d$ below ($c = 1$). Past it the chart
cannot represent the point, which is not a bug. To choose a manifold by use case, see
[Choosing a manifold](manifolds.md#choosing-a-manifold).

| Model | Good to $a$ (float32 / float64) | Limit |
| --- | --- | --- |
| Hyperboloid | 16.6 / much further | the storage floor $\varepsilon\sinh(a)/\sqrt{c}$; the constraint can drift under Euclidean updates ([The `atol` Convention](#the-atol-convention), [Known Limitations](#hyperboloid-known-limitations)) |
| ProperVelocity | as the hyperboloid | the hyperboloid's storage floor |
| Poincaré | 12.65 / 27.73 (13.80 / 28.88 at $c = 0.1$) | the [ball chart's ceiling](#poincare-roundtrip-ceiling); float32 is fine to $a \approx 7$ ([table](#precision-requirements-by-distance)) |
| κ-Stereographic, $c > 0$ | as Poincaré | plus the Taylor band near $c = 0$ ([below](#stereographic-near-zero-curvature)) |
| Klein | 6.33 / 13.86 | half the Poincaré radius, and the floor $\varepsilon\cosh^2(a)$ ([below](#klein-numerics)) |
| HalfSpace | `inf`/NaN past 88.7 / 709.8 | the floor $\approx 0.4\,(\varepsilon/2)\cosh(\sqrt{c}\,\delta)/\sqrt{c}$ ([below](#halfspace-numerics)) |

## κ-Stereographic: Numerics Near Zero Curvature {#stereographic-near-zero-curvature}

Signed curvature adds a regime the other manifolds lack: near $c = 0$ every closed form becomes
$0/0$, and `Stereographic` switches to Taylor series. This matters when you train a *signed*
learnable curvature that may cross zero.

For $c > 0$ the Poincaré sections above apply: `addition`, `gyration`, `proj` and the conformal
factor are the functions `Poincare` calls, and `dist`, `logmap` and `geodesic` use the
[asinh form](#poincare-far-pairs) of Poincaré's default `dist`. `dist` has no counterpart of
Poincaré's saturating slot 1. `geodesic(t, x, y)` stores $t \otimes ((-x) \oplus y)$, a point at
radius $t\,d$ that float32 caps once $\sqrt{c}\,t\,d$ passes the
[chart ceiling](#poincare-roundtrip-ceiling); $t = 1/2$ stays inside for any two points the ball
stores. See the [κ-Stereographic API reference](../api-reference/manifolds.md) for the sign
convention and the factor-2 flat limit.

### The Taylor Cutover Is dtype-Dependent

The curvature-generalized trig functions ($\tan_\kappa$, $\tan_\kappa^{-1}$) use their exact closed forms away from zero and a truncated Taylor series (degree 5 in $\kappa\lVert x\rVert^2$) near zero. The cutover differs by precision:

| dtype | Taylor branch used when | Why |
|-------|-------------------------|-----|
| float64 | $\lvert\kappa\rvert < 10^{-9}$ | closed forms accurate down to ~$10^{-9}$ |
| float32 | $\lvert\kappa\rvert < 10^{-5}$ | catastrophic cancellation in the closed-form **curvature gradient** below this |

In float32 the *values* stay accurate well below $10^{-5}$; it is $\partial(\cdot)/\partial c$
computed through the closed forms that degrades. The wider float32 window gives finite,
well-behaved gradients below the cutover at the cost of a seam just above it: there the curvature
gradient has relative error of order $\varepsilon/(\lvert\kappa\rvert\,\lVert x\rVert^2)$, with
median 1–2 % and worst case 8–30 % over 200 points with $\lVert x\rVert \approx 0.85$, and
`logmap`'s gradient changed sign on 1–2 of them.

The Taylor branch is additionally gated on its convergence region $\lvert\kappa\rvert\,\lVert x\rVert^2 < 0.01$: points at extreme chart radii ($\lVert x\rVert \sim 1/\sqrt{\lvert\kappa\rvert}$, e.g. spherical points far from the chart origin) always keep the exact closed form, no matter how small $\lvert\kappa\rvert$ is.

At exactly $c = 0$ the Möbius denominator's signed factorizations have the literal curvature slope
$2s\langle x, y\rangle$ for nonzero operands, so a signed learnable curvature gets a gradient of
the right sign on its flat step.

### Spherical Regime ($c < 0$) Cautions

- **The chart has no boundary, but it has a pole.** Stereographic coordinates cover the sphere minus one point; near-antipodal pairs have chart norms $\sim 1/\sqrt{\lvert c\rvert}$ and the metric shrinks accordingly. `antipode` itself is exact (closed form $x/(c\lVert x\rVert^2)$), but *optimizing through* near-antipodal configurations concentrates precision loss the same way the Poincaré boundary does.
- **Distances saturate at $\pi R = \pi/\sqrt{\lvert c\rvert}$.** Gradients of `dist` vanish as a pair approaches antipodal, analogous to `atanh` saturation in the hyperbolic regime.

### Recommendations

- Use `Stereographic(dtype=jnp.float64)` when training a signed curvature that may cross zero, per Bachmann et al. (2020). Float32 is fine at fixed moderate curvature ($\lvert c\rvert \gtrsim 10^{-4}$) and moderate radii.
- With `LearnableCurvature(parameterization="identity")`, the default clamp $[-10, 10]$ includes 0 by design — a curvature crossing zero is a *feature* (the geometry interpolates hyperbolic → flat → spherical smoothly), not an error state.

## Klein: Cancellation-Free Distance on a Half-Radius Chart {#klein-numerics}

`Klein` stores points in the same Euclidean ball as `Poincare`, with the same `proj`, and its
two-point operations (`dist`, `logmap`, `ptransp`, `gyro_difference`) are cancellation-free. The
chart limits it: a Klein point reaches the boundary at half the Poincaré radius, and
$g_x = 1 - c\lVert x\rVert^2$, which every formula divides by, is only as accurate as its own
rounding. Below, $a = \sqrt{c}\,d_0$ is the scaled radius and
$\varepsilon$ the machine epsilon (1.19e-7 in float32, 2.22e-16 in float64).

### The Distance Without Cancellation

The textbook spellings lose digits: the hyperboloid form lifted to Klein takes `acosh` of a number
close to 1 for close points, and the Einstein-sum form of Zhang et al. (2026), Eq. 4, builds
$(-x)\oplus_E y$ from $O(1)$ terms that cancel. `Klein.dist` follows Zhang et al. (2026), Appendix
Eqs. 12 and 14. With $w = y - x$ (exact for close points, by Sterbenz's lemma), the Lagrange and
Einstein gamma identities give

$$
N = g_x\lVert w\rVert^2 + c\,\langle x, w\rangle^2, \qquad
S^2 = \sinh^2(\sqrt{c}\,d) = \frac{c\,N}{g_x\, g_y}, \qquad
d(x, y) = \frac{\operatorname{arsinh}(S)}{\sqrt{c}}.
$$

$N$ is a sum of two non-negative terms, and no `acosh` or `atanh` is evaluated near 1. At $x = y$,
$w$ is exactly zero, so `dist` is exactly 0 with a finite zero gradient. `logmap` and `ptransp` are
built from the same $N$.

### Measured Accuracy

Float32 relative error of `dist` (median / max over 200 pairs, dimension 8, $c = 1$), for $y$ a
step of 1 % of $x$'s Euclidean gap to the boundary, against a high-precision evaluation at the
stored points:

| $a$ | `Klein.dist` | literal `acosh` | Zhang et al. Eq. 4 (`artanh`) | $\varepsilon\cosh^2(a)$ |
|---|---|---|---|---|
| 0.5 | 3.35e-08 / 1.48e-07 | 1.18e-03 / 4.29e-03 | 5.44e-07 / 2.57e-06 | 1.5e-07 |
| 2 | 1.94e-07 / 8.29e-07 | 7.73e-02 / 1.00e+00 | 1.42e-04 / 9.33e-04 | 1.7e-06 |
| 4 | 1.19e-05 / 5.15e-05 | 1.00e+00 / 4.23e+01 | 4.33e-02 / 3.58e-01 | 8.9e-05 |
| 6 | 7.48e-04 / 3.01e-03 | 1.00e+00 / 1.04e+03 | 9.97e-01 / 9.21e+01 | 4.9e-03 |

`Klein.dist`'s error does not depend on the separation, and it stays at or below the chart
floor $\varepsilon\cosh^2(a)$ (last column, explained [below](#klein-chart-ceiling)). `Klein.logmap`'s error is the
same size. `Klein.dist` forward+backward also runs in about half the
time of the Eq. 4 spelling on an A100.

### The Chart Ceiling and Floor {#klein-chart-ceiling}

**Ceiling.** `proj` caps $\lVert x\rVert$ at $1/\sqrt{c} - \varepsilon^{0.75}$, the Poincaré
ball's [margin](#poincare-roundtrip-ceiling). A Klein point has
$\sqrt{c}\,\lVert x\rVert = \tanh(a)$ where a Poincaré point has $\tanh(a/2)$, so at $c = 1$ the
ceiling is $a = \operatorname{atanh}(1 - \varepsilon^{0.75})$ = **6.33** in float32 and
**13.86** in float64, half the Poincaré ball's 12.65 / 27.73.

`poincare_to_klein`, `hyperboloid_to_klein` and `pv_to_klein` do not project. Because the Klein operations floor the gap $g_k$
at half its value on the margin, a mapped-in point reads at its own radius up to $a \approx 6.67$
and as 6.67 beyond (float32, $c = 1$; float64: 14.21). **Call `Klein.proj` after
mapping far points in**, so that the stored point is the one the operations use.

**Floor.** $g_x = 1 - c\lVert x\rVert^2 = \operatorname{sech}^2(a)$. Computing it subtracts
$c\lVert x\rVert^2 \approx 1$ from 1, so $g_x$ carries an absolute error of about $\varepsilon$,
a relative error of $\varepsilon/g_x = \varepsilon\cosh^2(a)$. The pairwise formulas and the metric
divide by $g_x$ or its square root, so this is the chart's relative error floor. It scales as
$e^{2a}$, where the Poincaré ball's and the hyperboloid's scale as $e^{a}$.

In practice, float32 `Klein` is accurate to about 1e-5 at $a = 4$ and 1e-3 at $a = 6$. For larger
radii use `Klein(dtype=jnp.float64)`, or the hyperboloid.

`Klein.expmap` evaluates its denominator as a sum of non-negative terms, so inward steps also stay near
the chart floor. It keeps the factor $c$ that the reference implementation's `_klein_expmap` omits,
so it is correct for every $c$, not only $c = 1$.

## Half-Space: Cancellation-Free Pairwise Operations {#halfspace-numerics}

`HalfSpace` (height $x_n > 0$ in the last coordinate, metric $\lVert dx\rVert^2/(c\,x_n^2)$,
origin $o = e_n/\sqrt{c}$) builds its two-point operations from $w = y - x$, which is exact in
floating point for close points. The textbook distance
$\operatorname{arcosh}\!\big(1 + \lVert y - x\rVert^2/(2x_n y_n)\big)/\sqrt{c}$ takes `acosh` of a
number close to 1 for close pairs and loses the separation. `HalfSpace.dist` evaluates the same
function as $(2/\sqrt{c})\operatorname{arsinh}(\lVert r\rVert/2)$ with
$r = \big((y - x)/\sqrt{x_n}\big)/\sqrt{y_n}$, and `logmap` builds $\theta/\sinh\theta$ from the
same $r$. In float32 at scaled separation $10^{-5}$, the
literal `acosh` has median relative error 1.00 (eager, or jitted on GPU), and `HalfSpace.dist` a median of at most 4.24e-8
at scaled radii from 0.5 to 12, $c = 1$. (The reference implementation, HTorch, uses the literal
`acosh`, and its `ptransp` is not an isometry; hyperbolix implements the correct transport.)

The remaining error is a stored point's own rounding, about
$0.4\,(\varepsilon/2)\cosh(\sqrt{c}\,\delta)/\sqrt{c}$, with $\delta$ the distance to the vertical
geodesic through $o$; along that axis it does not grow with the height.

Three cases return `inf`/NaN instead of a finite wrong value:

- the squared chord $\lVert r\rVert^2$ overflows past a scaled distance $\sqrt{c}\,d$ of 88.72 in
  float32 (709.78 in float64); past it `dist` returns `inf`, and `logmap`, `ptransp` and
  `gyro_difference` return non-finite values;
- `logmap` can return a non-finite vector earlier, once $x_n e^{\sqrt{c}\,d}$ passes the largest
  float;
- an exactly vertical upward `expmap` step longer than $\theta = \ln(1/\text{tiny})$, where tiny is
  the smallest normal number (87.34 in float32, 708.40 in float64), returns an infinite height with
  NaN horizontal coordinates. The gyro operations inherit this limit through $\exp_o$.

## The `math_utils` Wrappers {#hyperbolic-function-overflow}

The manifolds call the transcendentals in `hyperbolix.utils.math_utils`, not the `jnp` builtins.
Each has a guard that changes neither value nor gradient inside its valid range:

- `cosh` and `sinh` clip their argument at $\pm 0.99\ln(\texttt{finfo.max})$ (≈ ±87.8 in float32,
  ±702.7 in float64), so the value cannot overflow. Past the clip the value is constant and the
  gradient is exactly 0.
- `acosh` floors its argument at $1 + 10\,\varepsilon$, and `atanh` clamps it to
  $\pm(1 - 10\,\varepsilon)$, where the builtin derivatives are infinite.

An argument in the clipped range means the model has already diverged. The clip only keeps that
one value finite; it is not a regime to train in.

### `asinh` and `acosh`: The Derivative Overflows Before the Value Does {#asinh-acosh-wrappers}

JAX's derivative rules for `asinh` and `acosh` are `g·rsqrt(x² + 1)` and `g·rsqrt(x² − 1)`. In
float32, `x²` overflows once $|x| > 1.84\times10^{19}$, and `rsqrt(inf)` is 0, so the derivative
comes back exactly 0.0 while the forward value is correct and no warning is raised: at
`x = 1e22`, `jax.grad(jnp.arcsinh)` returns `0.0` instead of `1e-22`. A parameter downstream of an
`asinh` whose argument grows exponentially stops moving.

The wrappers keep `jnp.arcsinh`'s forward value bit for bit and spell the derivatives as
`1/hypot(1, x)` and `1/(√(x−1)·√(x+1))`, which never form `x²`. Every
library `asinh` call site — the manifolds' `dist`/`dist_0`/`logmap` forms, `asinhc`, the
proper-velocity operations, the samplers and every MLR head — goes through the wrapper. The
hyperboloid MLR reaches this regime at a large hyperplane offset (see
[The MLR Score at a Large Hyperplane Offset](#mlr-large-bias)). In your own code, use
`math_utils.asinh` wherever the argument can grow exponentially.

This is an upstream bug, reported as
[jax-ml/jax#40634](https://github.com/jax-ml/jax/issues/40634). Once JAX stops squaring the
argument, the wrappers can go and the call sites can return to `jnp.arcsinh`.

## Lorentz Residual and Midpoint at Large Radius {#lorentz-residual-midpoint}

`lorentz_residual` (the two-point combination behind `LorentzResidual` and
`HypformerPositionalEncoding`) and `lorentz_midpoint` (the weighted aggregation behind
`HyperbolicFullAttention`, `HyperboloidGyroBatchNorm`/`ProperVelocityGyroBatchNorm`'s batch mean,
and the hyperboloid Fréchet mean) both form a raw ambient vector $h = (h_0, h_s)$ — time
coordinate $h_0$, spatial part $h_s$ — and pull it back onto the sheet by dividing by
$\sqrt{-c\,\langle h,h\rangle_L}$. The midpoint takes the absolute value
$\sqrt{c\,\lvert\langle h,h\rangle_L\rvert}$; the residual does not, so a negative weight that
makes $h$ spacelike gives NaN rather than a wrong point on the sheet. Computing that Minkowski
square directly,

$$
\langle h, h\rangle_L = -h_0^2 + \lVert h_s\rVert^2,
$$

cancels. On the sheet both terms are of size $\lVert s\rVert^2$, the squared spatial radius of the
inputs, while their difference is only $O(1/c)$, so the float32 relative error grows as
$\varepsilon_{32}\, c\, \lVert s\rVert^2$ ($\varepsilon_{32} \approx 1.19\times10^{-7}$). At
$c = 0.1$ it reaches $10^{-3}$ at a spatial radius of about 290, gradients about three times
sooner, and past $\lVert s\rVert \sim 10^4$ the computed square flips sign.

Hyperbolix instead evaluates $\langle h,h\rangle_L$ from exact identities, valid for any weights
as long as the inputs are on the sheet. For the residual $h = x + w\,y$ with scalar weight $w$:

$$
\langle h,h\rangle_L = -\frac{(1+w)^2}{c} - w\,\langle x-y,\; x-y\rangle_L .
$$

For the midpoint of $M$ points, write $h=\sum_m w_m x_m$, $t_m=x_{m,0}$, $z_m=x_{m,s}/t_m$,
$T=h_0$ and $\bar z=h_s/T$. The normalizer is

$$
D^2 = T\sum_m w_m\left(\frac{1}{t_m}+c\,t_m\lVert z_m-\bar z\rVert^2\right)
     = c\left(T^2-\lVert h_s\rVert^2\right),
$$

where the equality follows from the sheet identity $1-\lVert z_m\rVert^2=1/(c t_m^2)$. The
variance is evaluated from coordinate differences, so every term is non-negative for non-negative
weights. Negative weights are unsupported; an all-zero weight row returns the origin.

The residual's float32 error against float64 at $\lVert s\rVert = 10^4$ is about
$2\times10^{-5}$ on the value and $6\times10^{-4}$ on the gradient ($c = 0.1$, $x \approx y$). The
limit that remains is the difference $x - y$ itself: when two points share a direction at
$\lVert s\rVert \gtrsim 10^4$, float32 rounding swallows the subtraction before the formula sees
it, and that regime needs float64.

### Busemann Coordinates at Large Radius {#busemann-large-radius}

`Hyperboloid.busemann(x, v, c)` (the horosphere coordinate behind
`HypLinearHyperboloidBusemann`/`HypRegressionHyperboloidBusemann` and HoroPCA) is
`log(√c·arg)/√c` with `arg = x_0 - ⟨x_s, v⟩`. Along the branch aligned with the ideal direction
$v$, `x_0` and `⟨x_s,v⟩` are both $O(\cosh a)$ and nearly equal, so the literal subtraction
cancels. With $q=\langle x_s,v\rangle$ and $p=x_s-qv$, the library evaluates

$$
\mathrm{arg}=\begin{cases}
(1/c+\lVert p\rVert^2)/(x_0+q),&q\ge0,\\
x_0-q,&q<0.
\end{cases}
$$

The direction $v$ must be a unit vector: normalize learned directions before the call. At the
origin the spatial gradient is exactly $-v$ in exact arithmetic. The batched score
`nn_layers.busemann_core._busemann_score`, used by the Busemann MLR/FC layers, applies the same
formula.

For a model that needs float64 only in the score, use a float64 island: cast $\hat x$ and the class
directions to float64, run the one similarity GEMM there, compute
$\mathrm{arg} = 1/(c(x_0+r)) + r(1-g)$, with $r = \lVert x_s\rVert$ and
$g = \langle \hat x_s, v\rangle$, and its `log` in float64, then cast back. This is a pattern, not
a library option; `HyperbolicFullAttention`'s `score_dtype` (see
[above](#attention-score-floor)) is the library's version of the same idea for the Lorentzian
score.

### HoroPCA Projection at Large Radius

`horo_projection`, the ideal-point projection behind `hyperbolix.decomposition.horopca`, preserves
every Busemann coordinate of its input, so it has the accuracy of `busemann` above. `HoroPCA`
centres its data with `Hyperboloid.gyro_difference` rather than a Lorentz-boost matrix product,
which cancels for points close to a far centre: at $a = 8$ in float32, the largest error of the
centred points is 3.7e-4 scaled nats, against 0.58 with the boost.

### Gyro-Difference and GyroBatchNorm Centering at Large Radius {#gyro-difference}

`Hyperboloid.gyro_difference(x, y, c)` computes $(\ominus x)\oplus y$, the operation
`HyperboloidGyroBatchNorm` needs to center a batch on its mean and any layer needs for the
difference of two far points. The general `addition(neg(x), y)` (the Lorentz boost
$\Lambda_{\ominus x}\,y$ — see [above](#the-hyperboloids-two-point-cancellation-failure-mode)) is
accurate almost everywhere except here: at $y \approx x$ the boost's spatial part is a sum of three
terms, each $O(e^{2a})$, that cancel at $y = x$, so its absolute error is `eps·cosh²(a)/√c` however
close $y$ is to $x$.

`gyro_difference` instead uses the transvection carrying $x$ back to the origin,

$$
(\ominus x) \oplus y = \Lambda_x^{-1} y = \mathrm{Exp}_0\big(\mathrm{PT}_{x\to 0}(\mathrm{Log}_x y)\big) ,
$$

which the polar frame evaluates with no cancellation. What limits it at $y \approx x$ is the
inputs: two float32 points whose directions are $\psi$ radians apart fix the direction of their
difference only to `eps32·√D/ψ` relative, and a result far from the origin to one ulp of its own
radius, `eps32·sinh(√c·d₀)/√c`. Measured at $a \in \{9, 12\}$, `gyro_difference` stays within
these floors, while the boost's error at $a = 9$ is 0.9 to 2.6 nats whatever the true answer.

`HyperboloidGyroBatchNorm` and `ProperVelocityGyroBatchNorm` center with `gyro_difference`. The
boost's error there is finite and plausible, not a NaN: centering 32 points at $a = 9$ with it
gives output errors up to 5.2 nats, against 4e-3 with `gyro_difference`.

### Origin derivatives and the Cartesian chart {#origin-derivatives}

<a id="geodesic-frame"></a>

`gyro_difference` and `ptransp` use Cartesian formulas when either endpoint's scaled spatial
radius is at most 1e-1: the difference is the inverse Lorentz boost with reconstructed time, and
transport is the closed-form geodesic transport, with the input tangent's time coordinate taken as
$v_0=\langle x_s,v_s\rangle/x_0$. Otherwise both use the stable geodesic frame.

`logmap` uses the Cartesian expression when either endpoint is exactly the origin and the polar
frame otherwise. For close pairs, with the frame's separation $S=\sinh(\sqrt c\,d(x,y)/2)$, it
uses the equivalent spatial displacement
$\operatorname{asinhc}(S)((y_s-x_s)-2S^2x_s)/\sqrt{1+S^2}$ when $S\le0.5$. At coincidence the
Jacobians through spatial lifts are $+I$ for the target and $-I$ for the base.

The polar frame never normalizes a direction that vanishes on a collinear pair (two points on one
ray, $y = x$ included). Its angular leg is a non-negative combination of $\hat y_s - \hat x_s$ and
$\hat x_s + \hat y_s$, which is exactly zero on both degenerate rays, so `logmap`, `ptransp` and
`gyro_difference` have correct gradients there as well as correct values.

PV `logmap`, `gyro_difference` and `ptransp` inherit this through their exact lifts (below). The
tests check these derivatives against analytic or finite-difference references.

### ProperVelocity Through the Hyperboloid Lift {#pv-operation-lifts}

<a id="pv-dist-lift"></a>
<a id="pv-tangent-metric"></a>

Proper-velocity coordinates are the hyperboloid's spatial part, so the lift

$$
X = \Big(\sqrt{1/c + \lVert x\rVert^2},\; x\Big)
$$

is exact, and a PV tangent $v$ lifts to $(\langle x,v\rangle/X_0,\,v)$. PV `dist`, `logmap`,
`expmap`, gyro `addition`, `gyro_difference` and `ptransp` lift their inputs, run the hyperboloid
operation, and return its spatial part; no Poincaré chart conversion is involved.
`tangent_inner`/`tangent_norm` share the hyperboloid's `radial_perp_decomposition`. So PV has the
hyperboloid's accuracy and its remaining limits, including transport of directions below float32
angular resolution.

Measured at $c = 0.5$: the float32 `dist` of a radial step of length 0.1 at $a = 10$ has relative
error 8e-7 along a coordinate axis, about the storage floor of the input pair (1.1e-4 for one
random direction, whose floor is 1.4e-5), and the unit radial tangent's
$\lvert\langle v,v\rangle - 1\rvert$ is 5.2e-5 at $a = 12$. `dist(x, x)` and `logmap(x, x)` are
exactly 0, with a finite gradient. Code built on PV `dist` inherits this, e.g.
`utils.helpers.compute_pairwise_distances` and `nn_layers.poincare_batchnorm.frechet_variance`
(called by `ProperVelocityGyroBatchNorm`).

## Version Parameters

Several manifold operations have more than one formula for the same quantity. The `version_idx`
argument selects which one runs.

### Poincaré Ball Distance Versions

```python
from hyperbolix.manifolds import Poincare
import jax.numpy as jnp

poincare = Poincare()
x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, 0.4])
c = 1.0

# Version 0: Direct Möbius distance (default)
d0 = poincare.dist(x, y, c, version_idx=poincare.VERSION_MOBIUS_DIRECT)

# Version 1: Möbius via addition
d1 = poincare.dist(x, y, c, version_idx=poincare.VERSION_MOBIUS)

# Version 2: metric-tensor form; `dist` runs version 0's body (identical values)
d2 = poincare.dist(x, y, c, version_idx=poincare.VERSION_METRIC_TENSOR)

print(f"Version 0: {d0:.6f}")
print(f"Version 1: {d1:.6f}")
print(f"Version 2: {d2:.6f}")
# 0 and 2 are identical; 1 agrees here but saturates on far pairs
```

| slot | `dist` | `dist_0` |
| --- | --- | --- |
| `VERSION_MOBIUS_DIRECT` (0), default | $\frac{2}{\sqrt c}\operatorname{arcsinh}\big(\sqrt c\lVert x-y\rVert/\sqrt{B_x B_y}\big)$, $B_x = 1 - c\lVert x\rVert^2$ | $\frac{2}{\sqrt c}\operatorname{atanh}(\sqrt c\lVert x\rVert)$ |
| `VERSION_MOBIUS` (1) | the norm of $(-x)\oplus y$; saturates on far pairs | same as slot 0 |
| `VERSION_METRIC_TENSOR` (2) | runs slot 0's body | the `arcsinh` form below |

#### Slot 2 Reads the Radius Through `arcsinh` {#poincare-metric-tensor-dist-0}

The metric-tensor distance is
$d_0(x) = \frac{1}{\sqrt{c}}\operatorname{acosh}\big(1 + \frac{2c\lVert x\rVert^2}{1 - c\lVert x\rVert^2}\big)$,
whose whole radial signal is a small perturbation of a leading 1: through `acosh`'s domain floor,
every float32 radius below $\approx 1.1\times10^{-3}/\sqrt{c}$ would round to a distance of 0, with
zero gradient. Slot 2 therefore evaluates it with the half-angle identity
$\operatorname{acosh}(1 + 2t) = 2\operatorname{arcsinh}(\sqrt{t})$,

$$
d_0(x) = \frac{2}{\sqrt{c}}\operatorname{arcsinh}\!\left(\frac{\sqrt{c}\,\lVert x\rVert}{\sqrt{1 - c\lVert x\rVert^2}}\right),
$$

whose argument is linear in the radius near the origin. Its median float32 relative error against
a 60-digit reference is below 1e-7 from radius 1e-8 to $0.9/\sqrt c$ ($c = 1$), and it equals
slot 0 to rounding. The pairwise `dist` uses the same identity with
$t = c\lVert x - y\rVert^2/\big((1 - c\lVert x\rVert^2)(1 - c\lVert y\rVert^2)\big)$; that is
slot 0's body. `dist(x, x)` is exactly 0, with gradient 0.

### Which Version to Use? {#which-version-to-use}

Use `VERSION_MOBIUS_DIRECT` (version 0), the default.

- `VERSION_METRIC_TENSOR` (version 2): for `dist` it runs slot 0's body and returns the same
  values; `dist_0`'s slot 2 is the `arcsinh` form, equal to slot 0 to rounding (see
  [above](#poincare-metric-tensor-dist-0)).
- `VERSION_MOBIUS` (version 1) saturates on far pairs: in float32 it returns 12.637 for a true
  distance of 14.4 (two points at scaled radius 7.2 on opposite sides, $c = 1$), with a gradient
  relative error of 1.0. In float64 it saturates at $\sqrt{c}\,d \approx 27.7$.
- Far points: no version helps. From $a = \sqrt{c}\,d \approx 10$, use float64 or the hyperboloid
  (see the [table](#precision-requirements-by-distance)); the ball cannot store a point past
  $a = 12.65$ (float32) / 27.73 (float64) at $c = 1$ (see
  [The Round-Trip Ceiling](#poincare-roundtrip-ceiling)). The hyperboloid holds float32 accuracy to
  $a \approx 16.6$ (the storage floor) and stores points up to $a \approx 45$ at $c = 1$, where the
  time coordinate overflows (see [Input Overflow](#input-overflow-fingerprint)). Convert with
  `isometry_mappings.poincare_to_hyperboloid`.

### Hyperboloid Distance Versions {#hyperboloid-distance-versions}

The Hyperboloid has its own two-way `version_idx`. The same two slots select an arm of both the
pairwise `dist` and the origin distance `dist_0`, which are different implementations. A
`version_idx` outside {0, 1} raises `ValueError`:

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

Use `VERSION_DEFAULT` for new code. Use `VERSION_SMOOTHENED` when a downstream `1/dist` or
`log dist` would otherwise divide by zero: coincident points then return the floor above, with a
well-defined gradient.

### Using Versions with JIT

Keep `version_idx` static (bake it into the function, or `static_argnames`): a traced index still
compiles, but every variant ends up in one `lax.switch`. Batch the single-point operations with
`jax.vmap` and compile with `jax.jit`; see [Batching & JIT](batching-jit.md#combining-vmap-and-jit)
for the patterns.

## Projection {#projection-strategies}

The library applies `proj` where its operations need it: Poincaré `addition` clamps its output,
layers such as `HypLinearPoincare` end in that `addition`, and the Riemannian optimizers move
parameters by `expmap` or retraction. Call `proj` yourself only on points you build: raw
parameters, data loaded from disk, and hand-made ambient vectors.

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

### The `atol` Convention {#the-atol-convention}

`is_in_manifold` and `is_in_tangent_space` take `atol: float | None = None`. Left as `None`, every
manifold resolves it through
`hyperbolix.manifolds._base.default_atol(dtype) = sqrt(finfo(dtype).eps)` — `3.45e-4` in float32,
`1.49e-8` in float64. An explicit value is used as given: it is never floored, clamped or ignored.
Ball membership tests the dimensionless residual `c||x||² - 1`, so one tolerance means the same
thing at every curvature. On the hyperboloid the check compares the stored `x₀` with
`√(1/c + ‖x_s‖²)` and uses the tolerance as both `rtol` and `atol`, so on-sheet points pass the
default at any radius (see [above](#hyperboloid-known-limitations)).

### Batch Validation

```python
import jax
import jax.numpy as jnp
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
    - **Choose the dtype by the scaled radius** $a = \sqrt{c}\,d$. On the Poincaré ball, float32
      is fine to $a \approx 10$; use float64 for critical operations from there, and float64 is
      required past the float32 chart ceiling ($a \approx 12.65$ at $c = 1$; see the
      [table](#precision-requirements-by-distance)).
    - **For large radii, use the hyperboloid or `ProperVelocity`.** The hyperboloid holds float32
      accuracy to $a \approx 16.6$ (see [Known Limitations](#hyperboloid-known-limitations));
      `ProperVelocity` lives in unconstrained $\mathbb{R}^n$ and trains with plain `optax.adam`.
    - **Use `VERSION_MOBIUS_DIRECT` for Poincaré distance** (the default). `VERSION_MOBIUS`
      saturates on far pairs, past $\sqrt{c}\,d \approx 12.6$ in float32 and 27.7 in float64 at
      $c = 1$.
    - **Learn curvature with `LearnableCurvature`** (default log parameterization, bounds
      $[\texttt{init\_c}/10,\ \texttt{init\_c}\cdot10]$; its clamp passes only gradients that
      point back inside). Don't `jnp.clip` the curvature yourself: past the bound the gradient of
      `jnp.clip` is 0, and the curvature stops moving.
    - **Keep the default initializations.** A too-small init freezes a deep stack without any NaN;
      see [Initialization Scales](nn-layers.md#initialization-scales).
    - **A NaN loss is the intended signal of divergence.** Don't clip or clamp points, inputs or
      curvature to keep values finite; find where the value diverged instead (below).

## Debugging Numerical Issues

1. **Check for NaN/Inf**:
   ```python
   assert jnp.all(jnp.isfinite(x_batch)), "NaN or Inf detected in data"
   ```

2. **Verify manifold constraints** with `validate_batch` from above:
   ```python
   validate_batch(x_batch, c=1.0, atol=1e-5)
   ```

3. **Check the scaled radius** of your points against the model's limit (the
   [Quick Check](#precision-requirements-by-distance) computes it for the Poincaré ball).

4. **Switch to float64**: enable `jax.config.update("jax_enable_x64", True)` and build the
   manifold with `dtype=jnp.float64` — casting only the data does nothing, since a float32
   manifold casts its inputs back. Switching the Poincaré `version_idx` does not help (see
   [Which Version to Use?](#which-version-to-use)):
   ```python
   import jax
   import jax.numpy as jnp
   from hyperbolix.manifolds import Poincare

   jax.config.update("jax_enable_x64", True)
   poincare_f64 = Poincare(dtype=jnp.float64)
   x, y = jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4])
   dist = poincare_f64.dist(x, y, 1.0)
   ```

## See Also

- [Batching & JIT](batching-jit.md): Performance optimization patterns
- [Manifolds API](../api-reference/manifolds.md): Manifold function reference
- [Training Workflows](training-workflows.md): End-to-end training examples
