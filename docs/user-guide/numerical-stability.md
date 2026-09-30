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
`HyperbolicFullAttention` and `LorentzMLA`. `"variance"` (the `lorentz_midpoint` default, and the
default of `HyperbolicFullAttention`) is accurate for far-out values but costs `O(N·M·D)` work, and on
an A100 it compiles slowly: `LorentzMLA` at S = 512 takes 104 s against 9.6 s, S = 2048 did not finish
in 10 min, and `HyperbolicFullAttention` takes 15.9 s against 4.0 s at N = 512. `"gemm"` (the default of
`LorentzMLA`, as in HELM) is one GEMM, but in float32 it loses the centroid once the *values* pass scaled
radius `a ≈ 6`, even when the scores are accurate: values at `a ≈ 9` give 11.4 nats of error against
6.4e-4 for `"variance"`. bfloat16 activations make both the scores and the `"gemm"` centroid worse,
since bfloat16's `eps` is 2^16 times float32's.

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

As points move toward the boundary (||x|| → 1/√c), λ(x) explodes. A point at scaled radius
$a = \sqrt{c}\,d_0$ has $\sqrt{c}\,\lVert x\rVert = \tanh(a/2)$, so
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

From $a = 6$ to $a = 12$, each step of 2 in $a$ multiplies λ by about $e^2 \approx 7.4$. Float32
misses the exact $1 + \cosh a$ by 5e-4 at $a = 10$ and 8e-4 at $a = 12$ (11014.2 and 81378.4):
the stored point's own rounding, which $1 - c\lVert x\rVert^2$ amplifies (below). Past the float32
[chart ceiling](#poincare-roundtrip-ceiling), $a \approx 12.65$ at $c = 1$, `expmap_0` caps the
point at the `proj` margin: $a = 14$ and $a = 20$ give the same point, with λ = 155344.6 and
`dist_0` = 12.6465 (`logs/2026-09-29_cancellation-free/docs_b3/run_doc_snippets.out`).

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
(`_gyrovector_core._max_norm`). The margin is absolute, not a fraction of $1/\sqrt{c}$, so a
stored point has $\sqrt{c}\,\lVert x\rVert \le 1 - \sqrt{c}\,\varepsilon^{0.75}$, the largest
tangent vector `expmap_0` can represent has
$\sqrt{c}\,\lVert v\rVert = \mathrm{atanh}(1 - \sqrt{c}\,\varepsilon^{0.75})$, and the ceiling in
scaled radius is

$$
a_{\max} = \sqrt{c}\,d_0 = 2\,\mathrm{atanh}\bigl(1 - \sqrt{c}\,\varepsilon^{0.75}\bigr)
$$

(the factor 2 is the Poincaré metric's): 12.65 in float32 and 27.73 in float64 at $c = 1$, 13.80
and 28.88 at $c = 0.1$. Because it depends on $c$,
$2\,\mathrm{atanh}(1 - \varepsilon^{0.75})/\sqrt{c}$ is the geodesic radius of the ceiling only at
$c = 1$: at $c = 0.1$ it gives 40.0 in float32, where the ball reaches 43.6. Past the ceiling
`expmap_0` saturates and `logmap_0(expmap_0(v))` hands back the ceiling instead of `v`. In
float32 below $c \approx 0.035$, where $\sqrt{c}\,\varepsilon^{0.75} < 10\varepsilon$, the
`tanh`/`atanh` wrappers' clip at $1 - 10\varepsilon$ binds first: `expmap_0` and `logmap_0` stop
at $a \approx 14.33$, and the default `dist_0` reads every point that `proj` capped (out to
$a \approx 14.9$ at $c = 0.01$) as 14.33. Measured ceilings (median returned
$\sqrt{c}\,\lVert v\rVert$ on a round trip, at $c = 1$ unless marked; double them for $a$):

| library | float32 | float64 | boundary margin |
| --- | --- | --- | --- |
| hyperbolix | 6.32 | 13.86 | `eps**0.75` |
| hyperbolix, $c = 0.1$ | 6.90 | 14.44 | `eps**0.75` |
| geoopt, hypLL | 3.11 | 6.10 | fixed 4e-3, fixed 1e-5 |
| unguarded closed form | 8.66 | 18.72 | none |

The margin is deliberate. It stops short of the unguarded limit, so the conformal factor tops out
near $1/(\sqrt{c}\,\varepsilon^{0.75})$ — 1.6e5 (float32) and 5.5e11 (float64) at $c = 1$, 4.9e5
and 1.7e12 at $c = 0.1$ — and it still leaves twice the radius the fixed margins used elsewhere
allow. If your embeddings need larger radii, switch to the hyperboloid or to `ProperVelocity`
rather than shrinking the margin. The ceilings, the conformal factors and the hyperbolix rows of
the table are from `logs/2026-09-29_cancellation-free/docs_b3/probe_ceiling.out`.

### The Factored Möbius Denominator {#factored-mobius-denominator}

Every gyrovector op that needs $x \oplus y$ — `logmap`, `addition`, `gyr`, and `dist` in slot 1 —
divides by the Möbius denominator $1 + \mathrm{sign}\cdot 2c\langle x,y\rangle + c^2\lVert
x\rVert^2\lVert y\rVert^2$, and the literal spelling is a difference of two terms that both grow
like the ball chart's own ceiling squared as $x, y$ approach the boundary. `_mobius_denominator`
instead uses $1 + t^2 = (1-t)^2 + 2t$ with $t = \lvert c\rvert\,r_x r_y$: the subtraction $1 - t$
happens *before* the square, so an $O(1)$ result stops being the difference of two large numbers.
The pairwise `logmap` uses it at every radius, and so did the default `dist` (slot 0) until the
[asinh form](#poincare-far-pairs) below replaced it; it is what the
[chart ceiling](#poincare-roundtrip-ceiling) above ultimately bounds. Radii below are the geodesic
distance from the origin ($c = 1$, so $a = d$).

Measured on a radial pair with a geodesic gap of 0.1 (`probe_poincare_mobius_ebebd09.out`, table
D.i, medians over 4 seeds; `dist` still in the factored form): in float32, `dist` error goes
from 2.816e-03 to 2.086e-06 at $d = 8$, 6.506e-02 to 1.958e-05 at $d = 10$, and 1.423e+01 to 1.044e-04 at $d = 12$ (the as-is value at
$d = 12$ is the whole geodesic gap, lost); `‖logmap‖` error goes from 5.523e-03 to 8.799e-07 at
$d = 8$ and 2.866e-01 to 3.206e-04 at $d = 12$. In float64, `dist` error goes from 1.101e-03 to
1.584e-10 at $d = 18$, 7.519e-02 to 5.367e-10 at $d = 20$, and 9.664e-02 to 2.105e-09 at $d = 22$
(again, the as-is value at $d = 20$ is essentially the whole 0.1 gap); `‖logmap‖` error goes from
2.187e-03 to 2.324e-10 at $d = 18$ and 9.384e-02 to 1.243e-09 at $d = 20$. Both dtypes' fixed
columns are still climbing as $d$ approaches the chart ceiling from below — the factoring removes
the denominator's own cancellation, but it cannot make the ball represent a point past its
ceiling, $a \approx 12.65$ (float32) / $27.73$ (float64) at $c = 1$.

**Far pairs: `dist` and `logmap` in the asinh form.**{#poincare-far-pairs} The factoring fixes a
close pair, not a far one. Slot 0 evaluated $d = 2\,\mathrm{atanh}(u)/\sqrt{c}$ with
$u = \sqrt{c}\,\lVert x - y\rVert/\sqrt{D_-}$, $D_-$ the Möbius denominator above with sign $-1$,
and for a far pair $u \to 1$, where float32
`atanh`'s domain clip at $1 - 10\varepsilon$ took over. With $B_x = 1 - c\lVert x\rVert^2$, the
identity $D_- = B_x B_y + c\lVert x - y\rVert^2$ gives $1 - u^2 = B_x B_y/D_-$, so

$$
d(x, y) = \frac{2}{\sqrt{c}}\operatorname{arcsinh}\!\left(\frac{\sqrt{c}\,\lVert x - y\rVert}{\sqrt{B_x B_y}}\right),
$$

which has no domain to clip. Two points on opposite sides at scaled radius 7.2 each (true
$\sqrt{c}\,d = 14.4$, $c = 1$) came back 14.333 in float32 with an exactly zero gradient; they now
give 14.400112. At 9 each (true 18.0) the old value was still 14.333; it is now 17.999722
(`logs/2026-09-29_cancellation-free/1a/probe_old.out`,
`logs/2026-09-29_cancellation-free/2_evidence/probes/1a_merged.out`). This is the metric-tensor
distance of [slot 2](#poincare-metric-tensor-dist-0), the same function, so slots 0 and 2 now run
one body (see [Which Version to Use?](#which-version-to-use)). `logmap` keeps the factored
denominator for its direction and takes its length from the same `arcsinh` argument: the float32
relative error of $\lVert\log_x(y)\rVert$ on those pairs goes from 4.7e-3 to 2.9e-5 at 7.2 and from
0.20 to 2.6e-5 at 9.

**Transport: the regrouped gyration.** `ptransp(v, x, y)` is $\mathrm{gyr}[y, -x]\,v$ scaled by
$\lambda_x/\lambda_y$, and the gyration's numerator $A\,x + B\,y$ was two $O(1)$ terms cancelling
down to $O(1 - c\lVert x\rVert^2)$ for two nearby points near the boundary, then divided by a
denominator of size $O((1 - c\lVert x\rVert^2)^2)$. `_gyration` now writes it as
$(A - B)\,x + B\,s$ with $s = x + y$ ($= y - x$ for the transport), where each coefficient is
$O(1 - c\lVert x\rVert^2)$ on its own. On a 0.05-nat step ($c = 1$, float32 against float64, max
over 20 directions) the relative error goes from 1.9e-1 to 6.6e-5 at scaled radius 8 — the float32
floor $\varepsilon/(1 - c\lVert x\rVert^2) = \varepsilon\cosh^2(4)$ is 8.9e-5 there — and from 6.6
to 1.1e-3 at 10. At $s = 0$ the correction is exactly zero, so `ptransp(v, x, x)` returns `v`
bit for bit; it was 8.4e-2 off in float32.

**The Apollonian $G$.** `apollonian_dist` needs
$G = \sqrt{c^2\lVert x\rVert^2\lVert y\rVert^2 - 2c\langle x, y\rangle + 1}$. It now uses the same
identity, $G^2 = B_x B_y + c\lVert x - y\rVert^2$, a sum of two non-negative terms, where the
Gram-determinant spelling it replaced cancelled for a close pair. On a 0.05-nat pair (float32
against float64, max over 20 directions) the absolute error goes from 9.5e-4 to 5.3e-6 at scaled
radius 5.7 and from 6.7e-2 to 5.3e-5 at 8.

**Reductions.** The asinh form reads three reductions over the dimension —
$\lVert x - y\rVert^2$, $\lVert x\rVert^2$, $\lVert y\rVert^2$ — where slot 0's factored form took
five and slot 2's old body four (the max-scaled norm's second pass, which a difference of two ball
points does not need); the Apollonian $G$ takes three instead of four. Numbers in this and the two
paragraphs above: `logs/2026-09-29_cancellation-free/1a/probe_old.out` (before) and
`logs/2026-09-29_cancellation-free/2_evidence/probes/1a_merged.out` (after).

### Divisor Floors Below the Cap's Rounding Band {#poincare-divisor-floors}

`proj` caps a point at $\lVert x\rVert = 1/\sqrt{c} - \varepsilon^{0.75}$, where
$B = 1 - c\lVert x\rVert^2$ takes the analytic value `_boundary_floor`, and every divisor built
from $B$ — the $B_x$, $B_y$ above, $\lambda_x = 2/B_x$, the Busemann divisor, and the Möbius
denominator (at the square of that value) — used to be floored at exactly that value. Over float32
and float64 and $c \in \{0.1, 0.3, 1, 2.5\}$, the *computed* $B$ of a capped point lands from
$6.2\,\varepsilon$ below that value to $6\,\varepsilon$ above it (Poincaré `expmap` at $c \le 0.3$
also returns points farther inside, up to $17\,\varepsilon$ above in float32), and below it for
11–76 % of the capped points (mean 39 %; `logs/2026-09-29_cancellation-free/floorfix/probe_band.out`,
`logs/2026-09-29_cancellation-free/floorfix/summarize_band.out`). A floor that binds returns a
constant, so every derivative through it is zero: the float32 `dist` gradient at capped points came
back with relative error up to 1.0 (measured at the intermediate commit 9b16f31, where `dist`
already had the asinh form). The floors now sit at half the cap value
(`_boundary_divisor_floor`; the Möbius denominator's squared floor at a quarter),
$\sqrt{c}\,\varepsilon^{0.75}$ below the cap — $54\sqrt{c}\,\varepsilon$ in float32,
$8192\sqrt{c}\,\varepsilon$ in float64 — which no projected point reaches. Against a float64
unfloored reference, the float32 `dist` gradient error at capped points is now 1.6e-2 in $x$ and
1.8e-2 in $y$, and in float64 the gradient is bit-identical to the unfloored one
(`logs/2026-09-29_cancellation-free/floorfix/probe_grad_9b16f31.summary`). The float32 margin
covers the measured band only for $c \gtrsim (6.16/54)^2 \approx 0.013$; below that a capped point
can reach the floor again. Measured at smaller $c$ in float32, the half floor first binds at
$c = 0.005$, for 0.03 % of the capped points; at $c = 0.02$–$0.05$ up to 97–99 % of them sit below
the analytic value, where the old floor would have bound, while the half floor stays slack
(`logs/2026-09-29_cancellation-free/wave4/audit_F/probe_small_c.out`). A point outside the ball,
never projected, still meets the floor, which keeps its divisor positive. The Klein chart's gap
$g_x$ and the gaps the isometry maps read off a Poincaré or Klein point take the same half floor;
that is why a far Klein point now reads as $a \approx 6.67$ rather than at the `proj` ceiling 6.32
(float32, $c = 1$; see [The Chart Ceiling and Floor](#klein-chart-ceiling)).

Some float32 gradients at capped points stay wrong: Poincaré `expmap` with respect to its base
point and Poincaré `addition` with respect to $y$ keep relative errors up to 1.0, Klein `expmap`
with respect to its base point up to 1.0 (above 0.1 on 40–48 % of the pairs), and Klein
`addition` up to 1.3 on pairs of one capped and one free point (on pairs of two capped points its
gradient with respect to $x$ is fixed, 1.0 → 2.8e-3 at $c = 1$ and 7.8e-3 at $c = 0.3$, while the
one with respect to $y$ stays at 6.9e-2 and 1.5e-2, as before). The floor is not the cause: the gradients with the half floor and with no
floor are identical (`logs/2026-09-29_cancellation-free/floorfix/probe_grad_commit2.out`,
`logs/2026-09-29_cancellation-free/floorfix2/summary_grad2_fix.out`,
`logs/2026-09-29_cancellation-free/docs_b4b/check_floor_not_cause.out`).

### Tangent Inputs to the HNN++ and Busemann Layers {#poincare-tangent-input}

`HypRegressionPoincarePP` and `HypLinearPoincarePP` with `input_space="tangent"`, and
`HypConv2DPoincare` (tangent in, tangent out), used to lift the tangent input with `expmap_0` and
read the conformal factor back off the stored ball point; in float32 that lift stops at the
ball's ceiling, $t = \sqrt{c}\,\lVert v\rVert = \mathrm{atanh}(1 - \sqrt{c}\,\varepsilon^{0.75})$,
≈ 6.33 at $c = 1$ and ≈ 6.63 at $c = 0.3$ (the conv's old output map measured 6.3233 and 6.6256;
`logs/2026-09-29_cancellation-free/docs_a1/probe_conv_inf_score.out`), and
past it the scores were those of the ceiling, with a zero radial gradient. They now score the
tangent vector in closed form,
$\lambda_x\sqrt{c}\,\langle x, \hat{z}\rangle = \sinh(2t)\,\langle v, \hat{z}\rangle/\lVert v\rVert$
and $\lambda_x - 1 = \cosh(2t)$, and `HypConv2DPoincare` also maps its output back with `logmap_0`
of the lift in closed form, so neither ball point is formed. **This changes the output past
that ceiling**, and the conv output's norm is no longer capped. At $t = 8$, float32 against
float64 on the same inputs (max-abs error over max-abs value, worst of $c = 0.3$ and $c = 1$;
`logs/2026-09-29_cancellation-free/2_evidence/probes/1b_merged.out`): the regression head goes from
2.0e-1 to 1.5e-6, `HypLinearPoincarePP` (kernel scaled by 0.1, so its output stays off the ball's
edge) from 9.0e-2 to 9.7e-7, and the conv from 2.3e-1 (identity weights) and 6.9e-1 (random
weights) to ≤ 1.6e-6; the conv output's largest $\sqrt{c}\,\lVert\text{out}\rVert$ at $c = 1$ is now 20.2,
where the old route capped it at the ceiling (6.3279 in that probe). In
float64 the new and old routes agree to ≤ 8.8e-11 for $t \le 8$, except the conv with random
weights from $t = 5$ on, whose output ($\sqrt{c}\,\lVert\text{out}\rVert \ge 12.28$) is near or
past the float64 ball's ceiling. The Poincaré Busemann layers
(`HypRegressionPoincareBusemann`, `HypLinearPoincareBusemann`) score a tangent input the same way,
through `Poincare._busemann_tangent`: at $t = 8$, with random $\omega$, the regression scores go from
2.2e-1 to 2.0e-7 and the input gradients from 1.0 to ≤ 2.6e-7
(`logs/2026-09-29_cancellation-free/bz/probe_layers_v3.out`). For $\omega$ within float32 rounding of
$\hat v$, the input and kernel gradients are off by 0.1 or more from $t = 5$ on both routes
(`logs/2026-09-29_cancellation-free/poincare_small_t/probe_branch.out`, `probe_main.out`).

The Hyperboloid Busemann layers (`HypRegressionHyperboloidBusemann`, `HypLinearHyperboloidBusemann`)
with `input_space="tangent"` now score in closed form too, through `Hyperboloid._busemann_tangent`,
which shares its closed form with `Poincare._busemann_tangent`. The hyperboloid's `expmap_0` places
$v = (0, v_s)$ at scaled radius $t = \sqrt{c}\,\lVert v_s\rVert$, half the radius the ball's
`expmap_0` reaches for the same norm, so for a unit $\omega$ the score is
$\sqrt{c}\,B^\omega = \log(\cosh t - \sinh t\,\langle\omega, \hat v\rangle)$, the Poincaré tangent
score with $2t$ replaced by $t$. The time slot $v_0$ is not read; for a tangent vector at the origin
it is 0. The layers used to lift $v$ with `expmap_0` and call `busemann` on the stored point. The
float32 `expmap_0` is accurate up to the coordinate ceiling, so the scores were right, but the
gradients were not. For random directions $\omega$ the float32 input and kernel gradients broke
from $t \approx 29.4$ at $c = 0.3$ and $t \approx 29.8$ at $c = 1$; at $t = 30$ and 40 they were off
by 5.3e-2 to 1.07 relative to their largest entry, against ≤ 6.9e-7 up to $t = 20$. Only scores
with $\langle x_s, \omega\rangle \ge 0$ break: there `_busemann_arg` divides, and the backward pass
of that division multiplies a cotangent of size about $e^{-t}$ by the squared inverse denominator,
about $e^{-2t}$. The product falls below float32's smallest normal number, and XLA:CPU flushes it
to zero. Past the coordinate ceiling, $t \approx 45$, the lifted point overflows and the scores are
NaN. The closed form has neither limit: for random $\omega$ the layers' float32 outputs and
gradients stay within 1.2e-6 of float64 from $t = 1$ to $t = 50$, and in float64 they match the old
route's to ≤ 2.2e-15 (`logs/2026-09-29_cancellation-free/hbz/probe_old.out`, `probe_new.out`,
`compare_f64.out`, `scan_old_grad.out`;
`logs/2026-09-29_cancellation-free/hbz_docs/scan_onset_jit.out`). All of this was measured on CPU;
a GPU that keeps subnormal numbers may move the onset later.

For $\omega$ within float32 rounding of $\hat v$, both routes still err at large $t$. Written as
$\sqrt{c}\,B^\omega = \log\bigl(e^{-t} + (1 - \langle\omega, \hat v\rangle)\sinh t\bigr)$, the score
takes the misalignment of $\omega$ and $\hat v$ times $\sinh t$, next to a term of size $e^{-t}$,
and both routes read that misalignment off float32 directions, so its rounding limits either one.
The closed form removes the part of the error that came from the stored point. At $t = 12$ the
float32 $\sqrt{c}\,B$ is 8.6e-5 to 1.2e-4 off float64 in closed form and 3.0e-4 to 5.2e-4 through
the lift, 2.6 to 6 times more, and the layer outputs are 2.4 to 3.2 times more accurate. At
$t = 20$ the gain is about 2: $\sqrt{c}\,B$ is 1.7 to 1.9 off against 3.6 to 3.7 ($c = 0.3$ and 1;
`probe_new.out`, `probe_old.out`).

At small $t$ the float32 kernel gradient of both Busemann tangent paths has a relative error that
grows like $1/t$. Through the Hyperboloid layers it is 1.0e-3 (regression head) and 8.1e-4 (linear
layer) at $t = 10^{-3}$, and 4.6e-6 and 5.4e-6 at $t = 0.1$, against 1.5e-7 to 2.5e-7 through the old
lift (`logs/2026-09-29_cancellation-free/hbz/probe_small_t.out`). The closed form's
$\omega$-gradient is $2(\omega - w)/\lVert\omega - w\rVert^2$, where $w$ is $\sqrt{c}$ times the
point's image in the ball, of norm $\tanh(t/2)$ here. At small $t$ it points almost along $\omega$.
The layers' kernel normalization projects that component out, which leaves a part of size about $t$
next to a rounding error of size about $\varepsilon$. The Poincaré layers evaluate the same closed
form and give 4.8e-4 and 5.9e-4 at $t = \sqrt{c}\,\lVert v\rVert = 10^{-3}$; the ball route they
replaced gave 4.8e-4 and 4.4e-4, since its $\omega$-gradient has the same form, so there the error
is not new (`logs/2026-09-29_cancellation-free/hbz_docs/probe_small_t_poincare.out`).

The four Busemann layers check the manifold at construction, and a wrong one raises a `TypeError`.
The first check is by the names of the methods they call. `Klein`, `HalfSpace`, `Stereographic`,
`ProperVelocity` and `Euclidean` have neither `busemann` nor `_busemann_tangent`, so it rejects
them. It cannot tell `Poincare` from `Hyperboloid`, which have both, so each layer also checks the
class: the Poincaré Busemann layers (`HypRegressionPoincareBusemann`, `HypLinearPoincareBusemann`)
accept only a `Poincare`, and the Hyperboloid ones (`HypRegressionHyperboloidBusemann`,
`HypLinearHyperboloidBusemann`) only a `Hyperboloid`. Before that check, the other model passed
construction and failed at the first call with a shape error, except at `in_dim = 2`: there a
one-entry spatial part broadcasts against the kernel rows, and some of these calls ran without an
error (`logs/2026-09-29_cancellation-free/hbz_docs/check_busemann_validation.out`; after the check:
`logs/2026-09-29_cancellation-free/busemann_type/check_busemann_validation_new.out`).

### `PoincareBatchNorm2D`'s Batch Mean at Large Radius {#poincare-batchnorm-mean}

`PoincareBatchNorm2D` averages the batch in Klein coordinates, $\sum\lambda x/\sum(\lambda - 1)$,
and projects that average with `proj` before the Möbius half-scaling, so the batch mean cannot sit
farther out than the Klein chart's ceiling, $\operatorname{atanh}(1 - \sqrt{c}\,\varepsilon^{0.75})$:
6.3250 at $c = 1$ and 6.9006 at $c = 0.1$ in float32 (13.86 and 14.44 in float64). Measured on
clusters at scaled radius $a \in \{5, 6, 7, 8\}$ (spread 0.3, $N = 256$, $C = 16$, max over 4
seeds; float32 against float64 and an independent longdouble Lorentz-centroid reference;
`logs/2026-09-29_cancellation-free/docs/probe_batchnorm_midpoint.out`): at $c = 1$ the batch-mean
error $\sqrt{c}\,d(\mu_{32}, \mu_{64})$ is 7.3e-4, 6.0e-3, 0.683 and 1.688 at $a = 5, 6, 7, 8$ —
at $a = 7$ and 8 the float32 mean reads 6.31–6.33, the cap. The relative error of the Fréchet
variance is 3.9e-6, 4.2e-4, 5.3 and 32, and that of the layer output 1.9e-3, 1.4e-2, 1.1 and 1.4;
the output's batch mean sits 0.89 ($a = 7$) and 0.99 ($a = 8$) from the learned mean, against
2.6e-3 in float64. At $c = 0.1$ the float32 mean reads 6.86–6.92 at $a = 7, 8$ and the mean errors
are 8.2e-4, 8.2e-3, 0.142 and 1.10. The float64 path matches the independent reference to
≤ 1.5e-9. In float32, keep this layer's inputs inside scaled radius ≈ 6, or run it in float64.

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

**`uniform_poincare.sample` at small and extreme radii.** The sampler draws
$u = \cosh(\sqrt{c}\,r) - 1$ and inverts it, and it now evaluates both directions in half-angle
forms, $2\sinh^2(\sqrt{c}\,r/2)$ and $2\operatorname{arcsinh}(\sqrt{u/2})/\sqrt{c}$: at
$\sqrt{c}\,R = 10^{-3}$ every old float32 sample fell outside $R$ (max $1.54R$) and none does now,
and at $10^{-4}$, where the old $n = 3$ rejection loop did not finish within 120 s, the new
samples give a Kolmogorov–Smirnov statistic of 0.0062 against the null's ≈ 0.006 ($N = 20000$;
old: `logs/2026-09-29_cancellation-free/1d/probe_old.out`, new:
`logs/2026-09-29_cancellation-free/2_evidence/probes/fixup_merged.out`). Where that loop cannot
accept any draw — float32 $\sqrt{c}\,R$ below 2.168e-19 or from 45.0546 on, float64 below
2.983e-154 or from 355.5845 on — the radius now comes from the closed-form flat limit or
exponential tail of its density instead of the loop, which used to hang there
(`logs/2026-09-29_cancellation-free/samplerhang/thresholds.out`).

### Cost {#poincare-pass-cost}

Old `3a43701` against new `9b16f31`, time ratios new/old
(`logs/2026-09-29_cancellation-free/2_evidence/timing/final_table.out`). The new snapshot predates
three changes described on this page: the Busemann tangent scores, the sampler's closed-form
fallback and the half floors. Every row is an AOT-compiled float32 function at $c = 1$, and each
ratio is the median over 8 old/new process pairs. On the CPU each process is pinned to one core
with single-threaded XLA. There are two runs, plus a third (re-time) of the rows that came out more
than 6 % slower on the CPU or 10 % on the GPU.

- **Slower.** `ptransp` forward is about 11 % slower on the CPU (1.112 and 1.117, re-time 1.114),
  forward+backward 1.104 and 1.086 (re-time 1.046). On the A100 the `ptransp` forward is not
  resolved: 1.120, 1.199 and 0.984 in three runs, while an A/A run with identical code on both
  sides gave ratios from 0.814 to 1.113 on the same small Poincaré rows
  (`logs/2026-09-29_cancellation-free/2_evidence/timing/analyze_all.out`). The $n = 2$ sampler is
  about 8 % slower on the CPU (1.063, 1.081, re-time 1.091), and `logmap` forward+backward
  measured 1.031 and 1.027.
- **Faster on the CPU.** Slot 0 `dist` forward 0.671 / 0.651 and forward+backward 0.335 / 0.292;
  slot 2 `dist` forward 0.676 / 0.635 and forward+backward 0.782 / 0.786; `apollonian_dist`
  forward 0.777 / 0.770 and forward+backward 0.758 / 0.748. Forward+backward:
  `HypRegressionPoincarePP` with tangent input 0.802 / 0.793, `HypLinearPoincarePP` with tangent
  input 0.938 / 0.929, `HypConv2DPoincare` 0.837 / 0.840, the wrapped-normal `log_prob`
  0.787 / 0.844, and `PoincareBatchNorm2D` in training mode 0.905 / 0.906. `HoroPCA.fit`
  0.957 / 0.963.

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

- **Poincaré ball**: compact, bounded — float32 is fine up to scaled radius $a = \sqrt{c}\,d \approx 7$ and for visualization; use float64 for critical operations from $a \approx 10$, and float64 is required past the float32 chart ceiling ($a \approx 12.65$ at $c = 1$; see the [table](#precision-requirements-by-distance)).
- **Hyperboloid**: unbounded radius, and `dist`/`logmap`/`sqdist`/`tangent_norm`/`expmap`/`ptransp`/`tangent_proj`/`tangent_inner`/`egrad2rgrad`/gyro `addition`/`busemann` are all cancellation-free, designed to avoid the identified cancellation, with accuracy limited by the operation and stored inputs (see [above](#the-hyperboloids-two-point-cancellation-failure-mode)). The constraint $\langle x, x\rangle_L = -1/c$ must still be maintained and can drift under Euclidean updates — see [The `atol` Convention](#the-atol-convention) — and a handful of places still lose accuracy for reasons the fix does not remove, listed under [Known Limitations](#hyperboloid-known-limitations).
- **Proper Velocity**: unconstrained $\mathbb{R}^n$, stable at large radii, exact Euclidean retraction (plain `optax.adam` / SGD trains PV layers without a Riemannian wrapper). Preferred when embeddings naturally grow large. Its tangent-space metric shares the hyperboloid's fix (see [below](#pv-tangent-metric)), and `PV.dist`/`logmap` between two nearby points at large radius now go through the exact hyperboloid lift — see [below](#pv-dist-lift).
- **κ-Stereographic**: the Poincaré ball's numerics for $c > 0$: it shares the gyrovector core (`addition`, `gyration`, `proj`, the conformal factor), and `dist`, `logmap` and `geodesic` use the far-pair asinh form of Poincaré's default `dist`. It adds the flat and spherical regimes and a Taylor-series switchover near $c = 0$ — see the [dedicated section below](#stereographic-near-zero-curvature).
- **Klein**: the pairwise operations are cancellation-free, but the chart reaches its boundary at half the Poincaré radius (scaled radius 6.32 in float32, 13.86 in float64, at $c = 1$) and its error floor grows as $\varepsilon\cosh^2(a)$ — see the [dedicated section below](#klein-numerics).
- **HalfSpace**: no boundary at finite distance, and the pairwise operations are cancellation-free. The error floor is a stored point's rounding, which grows as $\cosh(\sqrt{c}\,\delta)$ with the distance $\delta$ to the vertical geodesic through the origin and stays at its minimum on that geodesic at every height. The pairwise operations return `inf`/NaN past a scaled distance of 88.7 in float32 (709.8 in float64) — see the [dedicated section below](#halfspace-numerics).

!!! note "Training PV layers"
    `HypLinearPV`, `HypConv2DPV`, and `HypRegressionPV` store their weights as plain `nnx.Param` (not `ManifoldParam`). Use a standard `nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)` — no `riemannian_adam` / `riemannian_sgd` wrapper is required.

## κ-Stereographic: Numerics Near Zero Curvature {#stereographic-near-zero-curvature}

The `Stereographic` manifold's signed curvature introduces one numerical regime the other manifolds don't have: the neighborhood of $c = 0$, where every closed-form expression becomes $0/0$ and the implementation switches to Taylor series. The switching logic is internal, but its consequences matter when you train a *signed* learnable curvature that may cross zero.

For $c > 0$ the Poincaré sections above apply. `addition`, `gyration`, `proj` and the conformal
factor are the functions `Poincare` calls, and `ptransp` (with the
[regrouped gyration](#factored-mobius-denominator)), `ptransp_0`, `tangent_inner`, `tangent_norm`,
`egrad2rgrad`, `retraction` and `dist_0` return `Poincare`'s bits (checked at $c = 0.3$, 1 and 2.5
in both dtypes). `dist`, `logmap` and `geodesic` take the half distance in the
[asinh form](#poincare-far-pairs) of Poincaré's default `dist`, not from the norm of
$(-x) \oplus y$: that point lies at the full pair distance from the origin, so float32 caps it at
the ball's ceiling while both inputs are still well inside. Two float32 points at scaled radius 7.2
on opposite sides ($c = 1$, true $\sqrt{c}\,d = 14.4$) gave `dist` 12.656 with a gradient norm of
0.037; they now give 14.39999 with 670.7, as `Poincare` does. `dist`, `logmap`, `expmap`,
`expmap_0`, `logmap_0` and `scalar_mul` agree with `Poincare`'s to rounding, but not always bit for
bit (float32: `logmap` within 1.1e-5 relative on pairs at scaled radius 6 to 11, the rest within
3.7e-7). Two things differ.
`geodesic(t, x, y)` still stores $t \otimes ((-x) \oplus y)$, a point at radius $t\,d$ that float32
caps once $\sqrt{c}\,t\,d$ passes the [chart ceiling](#poincare-roundtrip-ceiling) (12.65 at $c = 1$,
13.80 at $c = 0.1$); $t = 1/2$ stays inside for any two points the ball stores.
And `dist` has no counterpart of Poincaré's saturating slot 1. For $c \le 0$ and in the Taylor band
near $c = 0$ the formulas are unchanged: `logmap` and `geodesic` return the same bits as before, and
`dist`, which now takes the norm of $(-x) \oplus y$ from the Möbius denominator without forming the
point, moves by at most 3.1e-7 relative in float32 (5.6e-16 in float64). Measured in
`logs/2026-09-29_cancellation-free/stereo_fix/` (`probe_old.out`, `probe_new.out`,
`diff_old_new.out`) and `logs/2026-09-29_cancellation-free/docs_b1/probe_stereo_vs_poincare.out`.
See the [κ-Stereographic API reference](../api-reference/manifolds.md) for the sign convention and
the factor-2 flat limit.

### The Taylor Cutover Is dtype-Dependent

The curvature-generalized trig functions ($\tan_\kappa$, $\tan_\kappa^{-1}$) use their exact closed forms away from zero and a truncated Taylor series (degree 5 in $\kappa\lVert x\rVert^2$) near zero. The cutover differs by precision:

| dtype | Taylor branch used when | Why |
|-------|-------------------------|-----|
| float64 | $\lvert\kappa\rvert < 10^{-9}$ | closed forms accurate down to ~$10^{-9}$ |
| float32 | $\lvert\kappa\rvert < 10^{-5}$ | catastrophic cancellation in the closed-form **curvature gradient** below this |

In float32 the *values* stay accurate well below $10^{-5}$; it is $\partial(\cdot)/\partial c$ computed through the closed forms that degrades. The wider float32 window trades a small seam error (worst-case measured relative error of the curvature gradient just above the cutover: ~2.4%, sign always correct) for finite, well-behaved gradients everywhere.

The Taylor branch is additionally gated on its convergence region $\lvert\kappa\rvert\,\lVert x\rVert^2 < 0.01$: points at extreme chart radii ($\lVert x\rVert \sim 1/\sqrt{\lvert\kappa\rvert}$, e.g. spherical points far from the chart origin) always keep the exact closed form, no matter how small $\lvert\kappa\rvert$ is.

### Möbius Denominators at Exactly Zero Curvature

The stable Möbius denominator has separate signed factorizations for $c>0$
and $c\le0$. Both equal the literal polynomial
$1+2s c\langle x,y\rangle+c^2\lVert x\rVert^2\lVert y\rVert^2$, where $s$
is the operation's sign. For nonzero operands the selected branch has the literal
curvature slope $2s\langle x,y\rangle$ at $c=0$. If an operand is exactly zero,
the existing `MIN_NORM` radial floors leave a bounded residual instead of a
machine-exact identity. The earlier `abs(c)` factorization gave the wrong
one-sided slope at zero, which could send a signed learnable curvature in the
wrong direction on its flat step even though fixed-curvature values looked correct.

### Spherical Regime ($c < 0$) Cautions

- **The chart has no boundary, but it has a pole.** Stereographic coordinates cover the sphere minus one point; near-antipodal pairs have chart norms $\sim 1/\sqrt{\lvert c\rvert}$ and the metric shrinks accordingly. `antipode` itself is exact (closed form $x/(c\lVert x\rVert^2)$), but *optimizing through* near-antipodal configurations concentrates precision loss the same way the Poincaré boundary does.
- **Distances saturate at $\pi R = \pi/\sqrt{\lvert c\rvert}$.** Gradients of `dist` vanish as a pair approaches antipodal, analogous to `atanh` saturation in the hyperbolic regime.

### Recommendations

- Use `Stereographic(dtype=jnp.float64)` when training a signed curvature that may cross zero, per Bachmann et al. (2020). Float32 is fine at fixed moderate curvature ($\lvert c\rvert \gtrsim 10^{-4}$) and moderate radii.
- With `LearnableCurvature(parameterization="identity")`, the default clamp $[-10, 10]$ includes 0 by design — a curvature crossing zero is a *feature* (the geometry interpolates hyperbolic → flat → spherical smoothly), not an error state.

## Klein: Cancellation-Free Distance on a Half-Radius Chart {#klein-numerics}

`Klein` stores points in the same Euclidean ball as `Poincare` and uses the same `proj`, but
its two-point operations (`dist`, `logmap`, `ptransp`, `gyro_difference`) are written so that no
digits cancel. What limits it is the chart: a Klein point reaches the boundary at half the
Poincaré radius, and the one quantity every formula divides by, $g_x = 1 - c\lVert x\rVert^2$,
cannot be computed more accurately than its own rounding. Below, $a = \sqrt{c}\,d_0$ is the
scaled radius (geodesic distance from the origin times $\sqrt{c}$), and $\varepsilon$ is the
machine epsilon (1.19e-7 in float32, 2.22e-16 in float64). Measurements are from
`logs/2026-09-28_klein-manifold/probe_klein.out`.

### The Distance Without Cancellation

Two textbook spellings of the Klein distance lose digits:

- The hyperboloid form lifted to Klein, $\cosh(\sqrt{c}\,d) = (1 - c\langle x, y\rangle)/\sqrt{g_x g_y}$,
  takes `acosh` of a number close to 1 when the points are close — the hyperboloid's
  [two-point cancellation](#the-hyperboloids-two-point-cancellation-failure-mode) again.
- Zhang et al. (2026), Eq. 4, $d = \operatorname{artanh}\!\big(\sqrt{c}\,\lVert(-x)\oplus_E y\rVert\big)/\sqrt{c}$,
  builds the Einstein sum from $O(1)$ terms that cancel when $x$ and $y$ are close, and divides by
  $1 - c\langle x, y\rangle$, which is itself a small difference of $O(1)$ terms near the boundary.

`Klein.dist` follows the derivation in Zhang et al. (2026), Appendix Eqs. 12 and 14. Let
$w = y - x$ (exact in floating point for close points, by Sterbenz's lemma). Two identities do
the work: the Lagrange identity
$\lVert x\rVert^2\lVert y\rVert^2 - \langle x, y\rangle^2 = \lVert x\rVert^2\lVert w\rVert^2 - \langle x, w\rangle^2$,
and the Einstein gamma identity
$1 - c\lVert(-x)\oplus_E y\rVert^2 = g_x g_y/(1 - c\langle x, y\rangle)^2$. The second says
$\operatorname{sech}^2(\sqrt{c}\,d) = g_x g_y/(1 - c\langle x, y\rangle)^2$, so
$\sinh^2(\sqrt{c}\,d) = \big[(1 - c\langle x, y\rangle)^2 - g_x g_y\big]/(g_x g_y)$. Expanding the
bracket gives $c\lVert w\rVert^2 - c^2\big(\lVert x\rVert^2\lVert y\rVert^2 - \langle x, y\rangle^2\big)$,
and the Lagrange identity turns it into $c\,N$ with

$$
N = g_x\lVert w\rVert^2 + c\,\langle x, w\rangle^2, \qquad
S^2 = \sinh^2(\sqrt{c}\,d) = \frac{c\,N}{g_x\, g_y}, \qquad
d(x, y) = \frac{\operatorname{arsinh}(S)}{\sqrt{c}}.
$$

$N$ is a sum of two non-negative terms built from $w$, the factor $1 - c\langle x, y\rangle$ has
cancelled out, and no `acosh` or `atanh` is evaluated near 1. At $x = y$, $w$ is exactly zero,
so `dist` is exactly 0 and its gradient is a finite zero. The same $N$ gives the other pairwise
operations in closed form:

- tangent norm of the chord: $\lVert w\rVert_x = \sqrt{N}/g_x$
- `logmap`: $\log_x(y) = \operatorname{arsinhc}(S)\,\sqrt{g_x/g_y}\;w$, with
  $\operatorname{arsinhc}(s) = \operatorname{arsinh}(s)/s$ (analytic at $s = 0$)
- `ptransp`: $\sqrt{g_y/g_x}\,(v - \kappa\, w)$, with
  $\kappa = \big[c\langle x, v\rangle/g_x + c\langle y, v\rangle/\sqrt{g_x g_y}\big]/\big(1 + \sqrt{1 + S^2}\big)$

In float64, against independent NumPy oracles (dimension 5, 200 pairs per curvature, error
$\lVert a - b\rVert/\max(\lVert b\rVert, 1)$, maximum over pairs), all 30 checks pass the 1e-11
threshold; for example `dist` against the hyperboloid `acosh` is 4.510e-15, `logmap` against the
hyperboloid log 4.493e-15, `ptransp` against the hyperboloid transport 1.310e-15, and
`einstein_midpoint` against the normalized Lorentz centroid 4.415e-16.

### Measured Accuracy

The accuracy probe places $x$ at scaled radius $a$, $x = \tanh(a)/\sqrt{c}\cdot\hat{x}$, and
$y = x + t\,(1 - \tanh a)/\sqrt{c}\cdot\hat{v}$ with random unit $\hat{x}, \hat{v}$ (dimension 8,
200 pairs per cell). So $t$ is the step as a fraction of $x$'s Euclidean gap to the boundary: at
$c = 1$, $a = 4$, the gap is $1 - \tanh 4 = 6.7\times10^{-4}$ and $t = 10^{-2}$ moves $y$ by
$6.7\times10^{-6}$. The reference is a 90-digit `Decimal` evaluation at the **stored**
(already rounded) points, so the table measures the error of the formula, not the error of
storing the points. Median / max relative error of `dist`, $c = 1$, $t = 10^{-2}$:

| dtype | $a$ | `Klein.dist` | literal `acosh` | Zhang et al. Eq. 4 (`artanh`) | `Klein.logmap` | $\varepsilon\cosh^2(a)$ |
|---|---|---|---|---|---|---|
| float32 | 0.5 | 3.35e-08 / 1.48e-07 | 1.18e-03 / 4.29e-03 | 5.44e-07 / 2.57e-06 | 3.79e-08 / 1.74e-07 | 1.5e-07 |
| float32 | 2 | 1.94e-07 / 8.29e-07 | 7.73e-02 / 1.00e+00 | 1.42e-04 / 9.33e-04 | 2.27e-07 / 1.15e-06 | 1.7e-06 |
| float32 | 4 | 1.19e-05 / 5.15e-05 | 1.00e+00 / 4.23e+01 | 4.33e-02 / 3.58e-01 | 1.41e-05 / 5.20e-05 | 8.9e-05 |
| float32 | 6 | 7.48e-04 / 3.01e-03 | 1.00e+00 / 1.04e+03 | 9.97e-01 / 9.21e+01 | 6.93e-04 / 3.16e-03 | 4.9e-03 |
| float64 | 0.5 | 5.87e-17 / 2.77e-16 | 1.78e-12 / 7.10e-12 | 1.88e-15 / 8.15e-15 | 8.14e-17 / 2.94e-16 | 2.8e-16 |
| float64 | 2 | 3.31e-16 / 1.84e-15 | 1.56e-10 / 1.15e-09 | 2.44e-13 / 1.43e-12 | 3.97e-16 / 1.86e-15 | 3.1e-15 |
| float64 | 4 | 1.97e-14 / 9.33e-14 | 2.58e-08 / 2.04e-06 | 4.50e-11 / 3.88e-10 | 2.47e-14 / 1.25e-13 | 1.7e-13 |
| float64 | 6 | 1.24e-12 / 4.92e-12 | 7.01e-07 / 3.62e-04 | 2.15e-09 / 4.28e-08 | 1.51e-12 / 5.49e-12 | 9.0e-12 |
| float64 | 10 | 3.74e-09 / 1.82e-08 | 2.55e-03 / 1.01e+00 | 7.99e-06 / 3.08e-03 | 3.34e-09 / 1.71e-08 | 2.7e-08 |
| float64 | 13 | 1.54e-06 / 5.58e-06 | 1.00e+00 / 6.89e+02 | 3.11e-03 / 1.87e-01 | 1.74e-06 / 5.81e-06 | 1.1e-05 |

The last column is the chart's floor, explained [below](#klein-chart-ceiling). The full grid in
the probe output also covers $c = 0.1$ and $t \in \{10^{-4}, 0.3\}$. What it shows:

- **`Klein.dist` does not depend on the separation.** At float32, $c = 1$, $a = 4$ the median
  is 1.17e-05, 1.19e-05 and 1.25e-05 for $t = 10^{-4}, 10^{-2}, 0.3$. The literal `acosh`
  loses the whole distance on close pairs (median 1.00e+00 at $t = 10^{-4}$ for every float32 $a$), and
  the Eq. 4 form has median 2.97e+00 in the same float32, $a = 4$, $t = 10^{-4}$ cell.
- **Its error is the chart floor.** From $a = 2$ on, the maximum `Klein.dist` error is below
  $\varepsilon\cosh^2(a)$ in 47 of the 48 cells (both dtypes, both $c$, all three $t$); the
  exception is float32, $c = 0.1$, $a = 6$, $t = 10^{-4}$ at 5.26e-03, 1.08× the floor. At
  $a = 0.5$ the maximum is under $2\varepsilon$ (2.11e-07 in float32, 3.81e-16 in float64).
  `logmap`'s error is the same size as `dist`'s in every cell except float32, $a = 6$,
  $t = 10^{-4}$ (both $c$), described next.
- At float32, $a = 6$, $t = 10^{-4}$, the step $10^{-4}(1 - \tanh 6) = 1.2\times10^{-9}$ is
  below the float32 spacing of the coordinates, and 137 of 200 stored $y$ equal $x$ at $c = 1$
  (131 at $c = 0.1$); those pairs have no relative error and are skipped.

### The Chart Ceiling and Floor {#klein-chart-ceiling}

**Ceiling.** `proj` caps $\lVert x\rVert$ at $1/\sqrt{c} - \varepsilon^{0.75}$, the Poincaré
ball's [margin](#poincare-roundtrip-ceiling). A Klein point has
$\sqrt{c}\,\lVert x\rVert = \tanh(a)$ where a Poincaré point has $\tanh(a/2)$, so at $c = 1$ the
projection ceiling is $a = \operatorname{atanh}(1 - \varepsilon^{0.75})$
= **6.32** in float32 and **13.86** in float64 — half the Poincaré ball's 12.65 / 27.7.

`poincare_to_klein`, `hyperboloid_to_klein` and `pv_to_klein` do not project. A point farther
out than the ceiling lands between the `proj` margin and the boundary, not on the margin: in
float32 at $c = 1$ the margin is $\lVert k\rVert = 1 - \varepsilon^{0.75} = 0.99999356$, and the
three maps give $\lVert k\rVert$ = 0.9999983 at $a = 7$ and 0.99999976 at $a = 8$. From $a = 10$
on, $\lVert k\rVert$ rounds to exactly $1/\sqrt{c}$ (at $a = 10$ for `poincare_to_klein` and
`pv_to_klein`, at $a = 12$ for all three). Klein operations floor the gap $g_k$ at half its
value on the margin: the computed gap of a point that `proj` capped lands within a few $\varepsilon$
of that value on either side, and a floor there would zero the gradient of every point it binds
on. So a mapped-in point reads at its own radius up to $a \approx 6.67$ and as $a \approx 6.67$
beyond: `Klein.dist` to the origin is 6.51 at $a = 6.5$ and 6.67 for all three maps at
$a = 7, 8, 10, 12$ (float64: 14.21 from $a = 15$ on;
`logs/2026-09-29_cancellation-free/floorfix2/probe_far_reading.out`). The radius beyond that is
lost either way; call `Klein.proj` after mapping far points in, so that the stored point is the
one the operations actually use.

**Floor.** $g_x = 1 - c\lVert x\rVert^2 = \operatorname{sech}^2(a)$. Computing it subtracts
$c\lVert x\rVert^2 \approx 1$ from 1, so $g_x$ carries an absolute error of about $\varepsilon$,
a relative error of $\varepsilon/g_x = \varepsilon\cosh^2(a)$. The pairwise formulas and the
metric divide by $g_x$ or its square root, so this is the relative error floor of the chart. Worked example,
float32 at $a = 4$: $1.19\times10^{-7}\cdot\cosh^2 4 = 1.19\times10^{-7}\cdot 745.7 = 8.9\times10^{-5}$,
against a measured `Klein.dist` median of 1.19e-05 and maximum of 5.15e-05 ($c = 1$,
$t = 10^{-2}$). Since $\cosh^2(a) \approx e^{2a}/4$, the Klein radial floor scales as $e^{2a}$,
where the floor of the Poincaré ball and the hyperboloid scales as $e^{a}$ (constants dropped).
The rewrite above removes every other cancellation, so the rounding of $g_x$ is the error that
remains.

In practice: float32 `Klein` is accurate to about 1e-5 at $a = 4$ and 1e-3 at $a = 6$ (medians
1.19e-05 and 7.48e-04 above). For larger radii use `Klein(dtype=jnp.float64)`, or the
hyperboloid, whose floor scales as $e^{a}$.

### Cost

`jit(vmap(dist))` over $10^6$ pairs, float32, $c = 1$, radius $\le 0.9/\sqrt{c}$, median of 25 runs
after warm-up (jax 0.9.1, NVIDIA A100-PCIE-40GB, shared with other processes during the runs);
the ratio is to the Eq. 4 form, the reference Einstein `artanh` spelling. One run
(`logs/2026-09-28_klein-manifold/probe_klein.out`):

| dim | mode | `Klein.dist` ms | Eq. 4 `artanh` ms | literal `acosh` ms |
|---|---|---|---|---|
| 16 | fwd | 0.383 (0.94×) | 0.407 (1.00×) | 0.363 (0.89×) |
| 16 | fwd+bwd | 0.535 (0.53×) | 1.016 (1.00×) | 0.519 (0.51×) |
| 128 | fwd | 1.051 (0.58×) | 1.812 (1.00×) | 1.024 (0.56×) |
| 128 | fwd+bwd | 2.493 (0.42×) | 5.925 (1.00×) | 2.426 (0.41×) |

Over five runs (this one, `probe_klein_after_logmap0_fix.out`, and three in
`logs/2026-09-28_klein_audit/timing_rerun.out`), `Klein.dist` forward+backward takes
0.48–0.53× the Eq. 4 time at dim 16 and 0.42–0.45× at dim 128, and the dim-128 forward takes
0.55–0.59×. The dim-16 forward ratio ranged from 0.91× to 1.44×, within run-to-run noise on
the shared GPU, so no speed difference is claimed there.

### The Reference `_klein_expmap` Is Exact Only at $c = 1$

`Klein.expmap` is $\exp_x(v) = x + v/\big(\theta\coth\theta + c\langle x, v\rangle/g_x\big)$ with
$\theta = \sqrt{c}\,\lVert v\rVert_x$. The reference implementation
(github.com/sc-zyl/Klein_hml, `Hyperbolic/hmath.py`, `_klein_expmap`) omits the factor $c$ in the
second denominator term. The two agree at $c = 1$. For $c \ne 1$ the reference still moves $x$
along $\pm v$, but by the wrong amount. In a float64 check at $c = 0.1$ with three random samples
(`logs/2026-09-28_klein-manifold/probe_ref_expmap.out`,
`logs/2026-09-28_klein_audit/audit_ref_expmap_nan.out`), one result landed outside the ball
($c\lVert\cdot\rVert^2 = 1.055$, NaN distance, where $\lVert v\rVert_x = 2.117$), one landed on
the far side of $x$ ($t = -0.61$ along $v$, distance 1.368 instead of 1.749), and one landed at
distance 0.317 instead of 0.504. All three $c = 1$ samples matched. `Klein.expmap` matches the hyperboloid
exponential map to 1.748e-15 in the float64 oracle check above.

On an inward step ($u = c\langle x, v\rangle/g_x < 0$) the two denominator terms have opposite
signs, and `Klein.expmap` now evaluates the denominator as a sum of non-negative terms for either
direction, $\theta\coth\theta + u = 2\theta e^{-2\theta}/(1 - e^{-2\theta}) + (\theta + u)$ with
$\theta + u = c\lVert v\rVert^2/(g_x(\theta - u))$ on an inward step: a radial inward step
$\theta = 8$ from $a = 4$ ($c = 1$, float32 against the float64 hyperboloid `expmap`) landed 0.85
scaled nats off before and lands 1.4e-4 off now, against the chart floor
$\varepsilon\cosh^2(4) = 8.9\text{e-}5$
(`logs/2026-09-29_cancellation-free/1d/probe_old.out`,
`logs/2026-09-29_cancellation-free/2_evidence/probes/fixup_merged.out`).

## Half-Space: Cancellation-Free Pairwise Operations {#halfspace-numerics}

`HalfSpace` (height $x_n > 0$ in the last coordinate, metric $\lVert dx\rVert^2/(c\,x_n^2)$,
origin $o = e_n/\sqrt{c}$) builds its two-point operations from $w = y - x$, which is exact in
floating point for close points. The textbook distance
$\operatorname{arcosh}\!\big(1 + \lVert y - x\rVert^2/(2x_n y_n)\big)/\sqrt{c}$, which the
reference implementation (HTorch) uses, takes `acosh` of a number close to 1 for close pairs
and loses the separation. `HalfSpace.dist` evaluates the same function as
$(2/\sqrt{c})\operatorname{arsinh}(\lVert r\rVert/2)$ with
$r = \big((y - x)/\sqrt{x_n}\big)/\sqrt{y_n}$, and `logmap` builds $\theta/\sinh\theta$ from the
same $r$. `expmap` evaluates its denominator in a non-cancelling form for near-vertical upward
steps. `ptransp` is a rational formula: the conformal scale $y_n/x_n$ times a rotation by
$-2\arctan\big(\lVert w_s\rVert/(x_n + y_n)\big)$. In float32 at scaled separation $10^{-5}$, the
literal `acosh` has median relative error 1.00 (evaluated eagerly, or jitted on XLA:GPU; XLA:CPU's
`jit` rewrites it into an accurate form), and `HalfSpace.dist` a median of at most 4.24e-8
at scaled radii from 0.5 to 12, $c = 1$ (`logs/2026-09-28_halfspace-manifold/timing_pass/probe_final.out`).

The error that remains is a stored point's own rounding, about
$0.4\,(\varepsilon/2)\cosh(\sqrt{c}\,\delta)/\sqrt{c}$ as a distance, with $\varepsilon$ the machine
epsilon and $\delta$ the distance to the vertical geodesic through $o$, so along that axis it does
not grow with the height.

Three cases return `inf`/NaN instead of a finite wrong value. The squared chord
$\lVert r\rVert^2$ overflows past a scaled distance $\sqrt{c}\,d$ of 88.72 in float32 (709.78 in
float64); past it `dist` returns `inf`, and `logmap`, `ptransp` and `gyro_difference` return
non-finite values (`timing_pass/ceiling_check.out`). `logmap` can return a non-finite vector
earlier, once $x_n e^{\sqrt{c}\,d}$ passes the largest float
(`test_fixes/logmap_overflow_repro_plain.out`). An exactly vertical upward `expmap` step
longer than $\theta = \ln(1/\text{tiny})$, where tiny is the smallest normal number (87.34 in
float32, 708.40 in float64), returns an infinite height with NaN horizontal coordinates: the
denominator $e^{-\theta}$ underflows to 0 while the true height $x_n e^{\theta}$ is still
representable (`docs_checks/docs_checks.out`, CPU). The gyro operations inherit this limit
through $\exp_o$.

HTorch's `ptransp` applies the ambient-hyperboloid transport formula to chart coordinates, which
is not an isometry: it transports $(1, 0)$ from $(0, 1)$ to $(0, 2)$ as $(1, 0)$, whose norm at
the endpoint is 0.5, where the parallel transport is $(2, 0)$ (`timing_pass/probe_final.out`).

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
$\sqrt{\lvert\langle h,h\rangle_L\rvert}$. Computing that Minkowski square directly,

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
