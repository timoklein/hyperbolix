# Numerical Stability Guide

Best practices for maintaining numerical precision in hyperbolic operations.

## Overview

Hyperbolic geometry presents unique numerical challenges due to the exponential growth of the conformal factor near the boundary and the involvement of hyperbolic functions (cosh, sinh, atanh). This guide explains these challenges and provides strategies to maintain numerical stability.

!!! warning "Key Challenges"
    - **Conformal factor explosion**: λ(x) grows exponentially as points approach the boundary
    - **Float32 limitations**: ~7 significant digits, insufficient for large distances (>10)
    - **Hyperbolic function overflow**: cosh/sinh overflow for large arguments
    - **Division by near-zero**: Operations involving 1 - c||x||² near the boundary

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
and [Known Limitations](#hyperboloid-known-limitations) for the operations that still route
through the Minkowski inner product and remain unsafe past the threshold above. `VERSION_LEGACY` /
`VERSION_LEGACY_SMOOTHENED` reproduce the old acosh-based arms bit-for-bit, for reproducing results
computed before this fix.

#### Known Limitations {#hyperboloid-known-limitations}

The two-point cancellation fix covers `dist`, `logmap`, `sqdist`, and `tangent_norm`. The
following Hyperboloid operations still route through the Minkowski inner product and remain
unsafe past the `ln(1/eps)` threshold above:

- `expmap` (the two-point form — `expmap_0` is unaffected)
- `ptransp`
- `tangent_proj`
- `tangent_inner`
- `egrad2rgrad`
- `is_in_manifold`

`tangent_norm` is exact to radius ~15 (float32) / ~25 (float64) — one power of `cosh` better than
before this fix (was accurate only to radius ~8 float32 / ~20 float64, then floored to ~0 past
that on exactly-unit tangent vectors), but not unlimited like `dist`/`logmap`/`sqdist`.
Origin-chart operations (`dist_0`, `logmap_0`, `expmap_0`) never routed through the Minkowski
inner product, so the threshold above does not apply to them. They had a different problem at the
*small*-radius end, fixed separately and described next.

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
    `safe_norm` passes an `inf` spatial entry through as `inf` on purpose, so an out-of-range point
    stays visibly degenerate instead of silently NaN-poisoning everything downstream. That made
    `logmap_0`'s scale $\operatorname{arcsinh}(u)/u$ an $\infty/\infty$ NaN, which then multiplied
    *every* entry — the time slot included. It is now
    `where(isfinite(u), arcsinh(u)/u, 1)`: the infinite entries come back as $\pm\infty$ with their
    signs, the finite entries keep their values, and the time slot stays exactly 0. This is the same
    $\pm\infty$-not-NaN convention the pairwise `dist`/`logmap` already follow through
    `_polar_frame`'s `isfinite` guard. The finite path is unchanged: the forward value is bit-identical on
    both backends and the gradient is bit-identical on XLA:CPU, moving by at most 2 ulps on XLA:GPU
    (where the extra `where` changes the VJP's fusion) against a 4-ulp CPU-vs-GPU spread the unchanged
    code already had.

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
(`_gyrovector_core._get_max_norm_eps`), which caps the largest geodesic radius the ball can
represent at $\mathrm{atanh}(1 - \varepsilon^{0.75})/\sqrt{c}$. Past that radius `expmap_0`
saturates and `logmap_0(expmap_0(v))` hands back the ceiling instead of `v`. Measured ceilings
(median returned radius on a round trip, in units of $1/\sqrt{c}$):

| library | float32 | float64 | boundary margin |
| --- | --- | --- | --- |
| hyperbolix | 6.32 | 13.86 | `eps**0.75` |
| geoopt, hypLL | 3.11 | 6.10 | fixed 4e-3, fixed 1e-5 |
| unguarded closed form | 8.66 | 18.72 | none |

The margin is deliberate. It stops short of the unguarded limit so the conformal factor stays
under ~3e5 (float32) / ~1e12 (float64), and it still leaves twice the radius the fixed margins
used elsewhere allow. If your embeddings need larger radii, switch to the hyperboloid or to
`ProperVelocity` rather than shrinking the margin.

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
- **Hyperboloid**: unbounded radius, and `dist`/`logmap`/`sqdist`/`tangent_norm` are now cancellation-free at any representable radius (see [above](#the-hyperboloids-two-point-cancellation-failure-mode)). The constraint $\langle x, x\rangle_L = -1/c$ must still be maintained and can drift under Euclidean updates, and `expmap`, `ptransp`, `tangent_proj`, `tangent_inner`, and `egrad2rgrad` still route through the Minkowski inner product — see [Known Limitations](#hyperboloid-known-limitations).
- **Proper Velocity**: unconstrained $\mathbb{R}^n$, stable at large radii, exact Euclidean retraction (plain `optax.adam` / SGD trains PV layers without a Riemannian wrapper). Preferred when embeddings naturally grow large.
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

For the midpoint of points $x_1,\dots,x_M$ with weights $w_1,\dots,w_M$, taking the first
point $p = x_1$ as reference and setting $\delta_m = x_m - p$, $W = \sum_m w_m$,
$\Delta = \sum_m w_m \delta_m$:

$$
\langle h,h\rangle_L = -\frac{W^2}{c} - W \sum_m w_m \langle \delta_m, \delta_m\rangle_L
+ \langle \Delta, \Delta\rangle_L .
$$

Every rounded quantity is now a difference between nearby points, of size
$O(\lVert x-y\rVert^2)$ rather than $O(\lVert s\rVert^2)$, so both operations are
radius-agnostic in float32: the measured float32 error against float64 at
$\lVert s\rVert = 10^4$ is $\approx 2\times10^{-5}$ on the value and $\approx 6\times10^{-4}$
on the gradient. The one limit that remains is the difference $x - y$ itself: when two
points share a direction at $\lVert s\rVert \gtrsim 10^4$, float32 rounding swallows the
subtraction before either formula sees it — neither the old nor the new form is
trustworthy there, and that regime needs float64.

!!! warning "The rest of the hyperboloid is still radius-limited"
    This buys accuracy in these two operations only. `Hyperboloid.dist`'s default and
    smoothened slots (`VERSION_DEFAULT`/`VERSION_SMOOTHENED`) use the same cancellation-free
    hyperbolic-haversine form and stay accurate at any representable radius; only the legacy
    slots (`VERSION_LEGACY`/`VERSION_LEGACY_SMOOTHENED`) route through the Minkowski inner
    product via `acosh` and still lose digits past hyperbolic distance ~7 in float32 (see the
    [precision table](#precision-requirements-by-distance)), and merely *storing* a point
    past distance ~11 accumulates more Lorentz residual than the float64 default
    tolerance allows — see [The `atol` Convention](#the-atol-convention).

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
    `safe_norm` now supplies (it also keeps `dist(x, x)` exactly 0).

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
  hyperboloid chart's representable ceiling is $\sqrt{c}\,d \approx 88$ (float32), where `cosh`
  overflows.
- **Very high dimensions** (> 1000): `VERSION_METRIC_TENSOR` (version 2) may be more stable
- **Debugging**: Compare all versions — significant differences indicate numerical issues

### Hyperboloid Distance Versions {#hyperboloid-distance-versions}

The Hyperboloid manifold has its own four-way `version_idx`, orthogonal to the Poincaré versions
above. The same four slots select an arm of both the pairwise `dist` and the origin distance
`dist_0`, which are different implementations:

| slot | `dist` arm | `dist_0` arm | floor |
| --- | --- | --- | --- |
| `VERSION_DEFAULT` (0) | cancellation-free hyperbolic haversine | `arcsinh(√c·‖x_s‖)/√c` | none: exactly 0 at coincidence / at the origin |
| `VERSION_SMOOTHENED` (1) | the same, floored in quadrature | the same, with `‖x_s‖` floored in quadrature | `2·arcsinh(10·eps)/√c` and `arcsinh(20·eps)/√c`, equal to first order: ≈2.4e-6/√c (float32), ≈4.4e-15/√c (float64) |
| `VERSION_LEGACY` (2) | pre-fix `acosh` form, hard clip | `acosh(clip(√c·x₀, 1))/√c` | 1.54e-3/√c in float32, 6.7e-8/√c in float64 (the `acosh` domain clamp) |
| `VERSION_LEGACY_SMOOTHENED` (3) | pre-fix `acosh` form, soft clamp | `acosh(smooth_clamp_min(√c·x₀, 1))/√c` | 0.16632/√c, in **both** dtypes, on every point that is not bitwise the origin |

!!! warning "Breaking: slots 2 and 3 of `dist_0` changed meaning"
    `dist_0(..., version_idx=2)` and `version_idx=3` used to duplicate slots 0 and 1. They now
    select the pre-fix `acosh` arms, matching what those slots have meant for the pairwise `dist`
    since 1.1.2. Code that passed 2 or 3 to `dist_0` and expected the default behavior must pass
    `VERSION_DEFAULT` (0) or `VERSION_SMOOTHENED` (1).

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

# VERSION_LEGACY (2): pre-fix acosh-based distance, reproduced bit-for-bit.
d2 = hyperboloid.dist(x, y, c, version_idx=hyperboloid.VERSION_LEGACY)

# VERSION_LEGACY_SMOOTHENED (3): VERSION_LEGACY with soft clamping.
d3 = hyperboloid.dist(x, y, c, version_idx=hyperboloid.VERSION_LEGACY_SMOOTHENED)
```

**When to use which**:

- `VERSION_DEFAULT` — the default, and the right choice for new code. Accurate at any
  representable radius (see [above](#the-hyperboloids-two-point-cancellation-failure-mode)).
- `VERSION_SMOOTHENED` — same numerics, but coincident points return a small positive distance
  with a well-defined gradient instead of exactly 0. Useful when a downstream `1/dist` or `log
  dist` would otherwise divide by zero. The floor is tiny: $2\,\mathrm{arcsinh}(10\epsilon)/\sqrt{c}
  \approx 2.4\text{e-}6/\sqrt{c}$ in float32 (vs. float64's $\approx 4.4\text{e-}15/\sqrt{c}$) — a
  large drop from the legacy smoothened floor of $\mathrm{acosh}(1 + \ln 2/\beta)/\sqrt{c}
  \approx 0.166/\sqrt{c}$, which shifted *every* distance in the working range, not just
  coincident ones.
- `VERSION_LEGACY` / `VERSION_LEGACY_SMOOTHENED` — reproduce the pre-fix acosh-based arms
  bit-for-bit. Use these only to match results computed before this fix; they lose all precision
  past the cancellation threshold described [above](
  #the-hyperboloids-two-point-cancellation-failure-mode).

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
# Manifold functions handle this internally with MIN_NORM
# But you can add explicit checks:

def safe_normalize(v, eps=1e-8):
    norm = jnp.linalg.norm(v)
    return jnp.where(norm > eps, v / norm, jnp.zeros_like(v))
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
def clip_curvature(c, min_c=0.01, max_c=10.0):
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
    - ✅ **Use float32 for distances < 7, float64 for larger**
    - ✅ **Project after operations that might violate constraints**
    - ✅ **Keep points away from boundary** (max norm < 0.9/√c)
    - ✅ **Use conservative learning rates** (< 1e-3 for Poincaré, < 5e-4 for Hyperboloid)
    - ✅ **Use protected math functions** (`hyperbolix.utils.math_utils`)
    - ✅ **Monitor conformal factors** during training
    - ✅ **Validate manifold constraints** in debugging
    - ✅ **Use `VERSION_MOBIUS_DIRECT` for Poincaré distance** unless issues arise
    - ✅ **Clip curvature** if learnable (0.01 < c < 10.0)
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
