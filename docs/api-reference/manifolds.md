# Manifolds API

This page documents the core manifold operations in Hyperbolix. Each manifold is a class that provides geometric operations and automatic dtype casting.

## Overview

Hyperbolix provides six base manifold classes plus a composition class:

- **Euclidean**: Flat Euclidean space (baseline)
- **Poincaré Ball**: Conformal model of hyperbolic space
- **Hyperboloid**: Lorentz/Minkowski model of hyperbolic space
- **Proper Velocity**: Unconstrained $\mathbb{R}^n$ model from special relativity (Chen et al. 2026)
- **κ-Stereographic**: Signed-curvature model unifying hyperbolic, Euclidean, and spherical geometry in one manifold (Bachmann et al. 2020)
- **Klein**: Beltrami–Klein ball model — straight-chord geodesics and Einstein gyrovector operations (Mao et al. 2024; Zhang et al. 2026)
- **Product Manifold**: Heterogeneous-curvature product spaces $M_1 \times M_2 \times \dots \times M_n$ (Gu et al. 2019)

All manifolds share a common interface defined by the `Manifold` protocol and support:

- **Automatic dtype casting**: Pass `dtype=jnp.float64` for higher precision
- **vmap-native methods**: Methods operate on single points; use `jax.vmap` for batching
- **JIT compatibility**: All methods are JIT-compilable
- **Learnable curvature**: Use the `LearnableCurvature` module to add trainable curvature to any model (positive `softplus`/`log` or signed `identity` reparameterization, optional clamping)

## Manifold Protocol

!!! note "The `Curvature` type"
    Manifold methods take the curvature as a positional `c: Curvature` argument
    (`hyperbolix.manifolds.Curvature`). It is the union
    `ScalarCurvature | Sequence[ScalarCurvature]`, where `ScalarCurvature = float |
    jax.Array`: single manifolds (`Poincare`, `Hyperboloid`, `ProperVelocity`,
    `Euclidean`) take a **scalar** `c`, while `ProductManifold` takes a **sequence**
    of per-factor scalars. Passing a traced `jax.Array` (e.g. the value returned by a
    `LearnableCurvature` call) makes the curvature differentiable.

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

    Slot 2 of **both** `dist` and `dist_0` evaluates the metric-tensor distance through the
    half-angle identity $\operatorname{acosh}(1 + 2t) = 2\operatorname{arcsinh}(\sqrt{t})$ rather
    than the `acosh` form directly — the same function, without the `acosh` domain clamp that used
    to floor every small radius (`dist_0`) and every small separation (`dist`). See
    [slot 2 reads the radius through `arcsinh`](
    ../user-guide/numerical-stability.md#poincare-metric-tensor-dist-0).

!!! note "Apollonian weak metric"
    `apollonian_dist(x, y, c)` is the **non-symmetric** Apollonian weak metric $\delta$
    (Papadopoulos & Troyanov, *Weak metrics on Euclidean domains*, Thm 2) — a *weak metric*,
    not a geodesic distance:

    $$\delta_c(x,y) = \log\!\left(\frac{\sqrt{c}\,\lVert x-y\rVert + \sqrt{c^2\lVert x\rVert^2\lVert y\rVert^2 - 2c\langle x,y\rangle + 1}}{1-c\lVert y\rVert^2}\right)$$

    It satisfies $\delta(x,x)=0$, $\delta\ge 0$ and the triangle inequality, but
    $\delta(x,y) \neq \delta(y,x)$ in general. Its symmetrization recovers the geodesic distance:
    $\delta(x,y) + \delta(y,x) = \sqrt{c}\cdot$ `dist(x, y, c)`.

    !!! warning
        The antisymmetric part of $\delta$ is an exact **coboundary** (a difference of a per-point
        potential), so it carries no circulation and is useless as an asymmetric quasimetric energy.
        For that, use the `busemann` coordinate below with an external quasimetric combinator.

!!! note "Busemann function (Chen et al. 2026)"
    `busemann(x, v, c)` is the closed-form **point-to-horosphere** coordinate $B^v(x)$ for a unit
    ideal direction $v\in\mathbb{S}^{n-1}$ — the horospherical analog of the point-to-hyperplane
    `compute_mlr`/`compute_mlr_pp`. `v` must be unit-norm (not normalized internally). It is an
    intrinsic quantity, so `Poincare.busemann` and `Hyperboloid.busemann` agree under
    `poincare_to_hyperboloid`, and $B^v(\text{origin})=0$. Backs the `*Busemann` MLR/FC layers.

    $$\mathbb{P}^n:\ B^v(x) = \tfrac{1}{\sqrt c}\log\!\frac{\lVert v-\sqrt c\,x\rVert^2}{1-c\lVert x\rVert^2}
    \qquad\quad \mathbb{L}^n:\ B^v(x) = \tfrac{1}{\sqrt c}\log\!\big(\sqrt c\,(x_t-\langle x_s,v\rangle)\big)$$

::: hyperbolix.manifolds.poincare.Poincare
    options:
      show_source: true
      heading_level: 3

## κ-Stereographic

A single constant-curvature manifold spanning **hyperbolic, Euclidean, and spherical** geometry via a **signed** curvature `c` (Bachmann et al. 2020). It generalizes the Poincaré ball across zero curvature using curvature-generalized ("$\kappa$-") trigonometric functions ($\tan_\kappa$, $\tan_\kappa^{-1}$), enabling a network to learn the *sign* of curvature from data.

!!! info "Signed-curvature convention (sectional curvature $= -c$)"
    Unlike the other manifolds (which require $c > 0$), `Stereographic` takes a **signed** $c$:

    | `c` | sectional curvature | geometry |
    |---|---|---|
    | $> 0$ | $< 0$ | hyperbolic — **identical to `Poincare(c)`** |
    | $= 0$ | $0$ | Euclidean (factor-2 limit; see below) |
    | $< 0$ | $> 0$ | spherical (stereographic projection of the sphere) |

    Internally the paper's $\kappa = -c$. This is **sign-flipped from the paper/geoopt $\kappa$** (their $\kappa > 0$ = spherical), chosen so `c` matches every other hyperbolix manifold and so `Stereographic(c)` reproduces `Poincare(c)` exactly for $c > 0$.

!!! warning "The Euclidean limit carries a factor of 2"
    The conformal factor is $\lambda^\kappa_x = 2/(1 - c\lVert x\rVert^2)$, so $\lambda^\kappa_0 = 2$ and the metric at $c = 0$ is $4\cdot I$, **not** $I$. As $c \to 0$: `addition`/`expmap`/`logmap` reduce to the *bare* Euclidean $x{+}y$ / $x{+}v$ / $y{-}x$, but `dist` $\to 2\lVert x-y\rVert$ and `tangent_norm` $\to 2\lVert v\rVert$ (paper Thm. 3). This matches Poincaré's own `dist_0` $\to 2\lVert x\rVert$, and therefore does **not** equal the separate `Euclidean` manifold's `dist` (bare metric $I$). Use `Euclidean` for un-scaled flat geometry; use `Stereographic` at $c=0$ only as the *continuous limit* of the curved family.

!!! note "Curvature derivatives at zero"
    The shared Möbius denominator evaluates separate signed factorizations on the
    $c>0$ and $c\le0$ sides. For nonzero operands they match the literal
    polynomial value and curvature derivative at $c=0$; an exactly zero operand
    retains only the bounded residual from the existing `MIN_NORM` radial floor.
    The previous `abs(c)` factorization selected the wrong one-sided denominator
    slope at zero, affecting signed-curvature gradients through Möbius operations.

!!! note "Scope of this release"
    Provides the complete core Riemannian manifold — the `Manifold` protocol plus `conformal_factor`, `gyration`, `geodesic`, `geodesic_unit`, and `antipode`. The $\kappa$-GCN neural-network layers and building blocks (`mobius_matvec`, weighted gyromidpoint, `dist2plane`, `sproj`/`inv_sproj`) are not yet included. Signed learnable curvature — spanning hyperbolic/Euclidean/spherical — is available via `LearnableCurvature(parameterization="identity")` (with a symmetric default clamp around zero); the `softplus`/`log` parameterizations remain positive-only. Double precision (`dtype=jnp.float64`) is strongly recommended, per Bachmann et al.

::: hyperbolix.manifolds.stereographic.Stereographic
    options:
      show_source: true
      heading_level: 3

## Hyperboloid

The hyperboloid (Lorentz) model with Minkowski geometry.

!!! note "Distance Versions"
    The Hyperboloid `dist` method has a `version_idx` parameter selecting between 2 formulations:

    - `VERSION_DEFAULT` (0): cancellation-free hyperbolic-haversine distance (default; see the
      measured range and remaining input-representation limits in the numerical-stability guide)
    - `VERSION_SMOOTHENED` (1): same evaluation, with a strictly-positive floor at coincidence

    The same two slots select an arm of `dist_0`, which is a separate implementation:

    - `VERSION_DEFAULT` (0): `arcsinh(√c·‖x_s‖)/√c`, read off the spatial part (default,
      with no domain clamp)
    - `VERSION_SMOOTHENED` (1): the same with `‖x_s‖` floored in quadrature, giving a floor of
      `arcsinh(20·eps)/√c` (≈2.4e-6/√c float32, ≈4.4e-15/√c float64)

    A `version_idx` outside {0, 1} raises `ValueError`.

    Constants are available as `hyperboloid.VERSION_DEFAULT` etc., or from
    `hyperbolix.manifolds.hyperboloid`. See the [numerical-stability guide](
    ../user-guide/numerical-stability.md#hyperboloid-distance-versions) for when to use each, the
    cancellation failure mode `VERSION_DEFAULT` fixes, and the [origin-chart rewrite](
    ../user-guide/numerical-stability.md#hyperboloid-origin-chart) behind the `dist_0` arms.

!!! note "Lorentz Operations"
    The Hyperboloid class includes specialized operations for convolutional layers:

    - `lorentz_boost`: Lorentz boost transformation
    - `distance_rescale`: Distance-based rescaling
    - `hcat`: Lorentz direct concatenation for convolutions
    - `log_radius_concat`: log-radius–preserving concatenation (digamma-scaled `hcat`; Shi et al. 2026, Sec. 4.3)

!!! note "Origin derivatives"
    `gyro_difference` and `ptransp` use Cartesian formulas when either endpoint's
    scaled spatial radius is at most 1e-1, and retain the stable polar frame
    otherwise. `logmap` uses the regular Cartesian expression at an exact origin
    endpoint and the stable polar frame otherwise. Ordinary autodiff preserves derivatives with respect to
    an origin endpoint; earlier value-only origin fallbacks erased them. `busemann` uses a projected-coordinate
    branch whose value and constrained first derivative agree at its branch surface.
    See [Origin derivatives and the Cartesian chart](
    ../user-guide/numerical-stability.md#origin-derivatives).

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

!!! note "Exact hyperboloid bridges"
    `dist`, `logmap`, `expmap`, gyro `addition`, `gyro_difference`, and `ptransp`
    evaluate through the direct isometry
    $x\mapsto(\sqrt{1/c+\lVert x\rVert^2},x)$. For tangent operations,
    $v\in T_x$ lifts to $(\langle x,v\rangle/X_0,v)$. The PV result is the
    spatial part of the hyperboloid result; no Poincaré chart conversion is used.
    `gyro_difference`, `ptransp`, and `logmap` inherit the hyperboloid's exact-origin
    Cartesian branch and its repaired origin derivatives.
    `ProperVelocityGyroBatchNorm` centers with `gyro_difference`. See
    [ProperVelocity operation bridges](
    ../user-guide/numerical-stability.md#pv-operation-lifts).

::: hyperbolix.manifolds.proper_velocity.ProperVelocity
    options:
      show_source: true
      heading_level: 3

## Klein

The Beltrami–Klein model (the Klein model) uses the same open ball $\lVert x\rVert < 1/\sqrt{c}$ as the Poincaré ball, with sectional curvature $-c$, but its geodesics are the **straight chords** of the ball. The metric is not conformal:

$$g_x(u, v) = \frac{\langle u, v\rangle}{g_x} + \frac{c\,\langle x, u\rangle\langle x, v\rangle}{g_x^2}, \qquad g_x = 1 - c\lVert x\rVert^2.$$

The gyrovector structure is Ungar's Einstein gyrovector space (Ungar 2009): `addition` is Einstein addition $\oplus_E$ and `scalar_mul` is Einstein scalar multiplication $\otimes_E$. `Klein` conforms to the scalar-`c` `Manifold` protocol.

!!! info "Relation to the Poincaré ball and the hyperboloid"
    A Klein point at geodesic distance $d_0$ from the origin has $\sqrt{c}\,\lVert x\rVert = \tanh(\sqrt{c}\,d_0)$, where a Poincaré point has $\tanh(\sqrt{c}\,d_0/2)$. The isometries in `isometry_mappings` are the Einstein half and the Möbius double:

    - `klein_to_poincare(k)` $= k/(1 + \sqrt{g_k})$, which is $\tfrac12 \otimes_E k$
    - `poincare_to_klein(p)` $= 2p/(1 + c\lVert p\rVert^2)$, which is $2 \otimes_M p$
    - `klein_to_hyperboloid(k)` $= \big(1/(\sqrt{c}\sqrt{g_k}),\ k/\sqrt{g_k}\big)$, `hyperboloid_to_klein(x)` $= x_s/(\sqrt{c}\,x_0)$
    - `klein_to_pv(k)` $= k/\sqrt{g_k}$, `pv_to_klein(x)` $= x/\sqrt{1 + c\lVert x\rVert^2}$

    Under the first pair, Einstein addition corresponds to Möbius addition. Einstein and Möbius scalar multiplication have the same formula, and so do the two models' `expmap_0`; `Klein` reuses Poincaré's code for both.

!!! note "Distance: one formulation, cancellation-free"
    `dist` has a single implementation; `version_idx` is accepted and ignored, as for `ProperVelocity`. With $w = y - x$,

    $$d(x, y) = \frac{1}{\sqrt{c}}\,\operatorname{arsinh}\sqrt{\frac{c\,\big[g_x\lVert w\rVert^2 + c\,\langle x, w\rangle^2\big]}{g_x\, g_y}},$$

    whose numerator is a sum of non-negative terms. It never forms $1 - c\langle x, y\rangle$ and needs no `acosh`/`atanh` near 1. `logmap` and `ptransp` are built from the same quantity. The derivation and measured errors are in the [numerical-stability guide](../user-guide/numerical-stability.md#klein-numerics).

!!! note "Einstein extras"
    Beyond the protocol, `Klein` provides:

    - `gyro_difference(x, y, c)`: $(\ominus x) \oplus_E y$, evaluated without the $1 - c\langle x, y\rangle$ cancellation
    - `lorentz_factor(x, c)`: $\gamma_x = 1/\sqrt{g_x}$, which is the hyperboloid's $\sqrt{c}\,X_0$
    - `einstein_midpoint(x_ND, weights_N, c)`: $\sum_i w_i \gamma_i x_i / \sum_i w_i \gamma_i$ (`weights_N=None` for uniform weights), equal to the normalized weighted Lorentz centroid mapped to the Klein ball

!!! warning "Half the Poincaré ball's radius range"
    `proj` uses the Poincaré ball's `eps**0.75` boundary margin, but because a Klein point sits at $\tanh(a)$ rather than $\tanh(a/2)$ ($a = \sqrt{c}\,d_0$, the scaled radius), the projection ceiling at $c = 1$ is $a = \operatorname{atanh}(1 - \varepsilon^{0.75})$ = **6.32** in float32 and **13.86** in float64, half of Poincaré's 12.65 / 27.7. In float32, Klein points can be stored farther out (at $c = 1$, a point at $a = 8$ has $\lVert x\rVert = 0.99999976$, above the margin $0.99999356$), but `proj` and the floor on $g_x$ cap every such point at $a = 6.32$. The margin $\varepsilon^{0.75}$ is absolute, so for general $c$ the ceiling is $a = \operatorname{atanh}(1 - \sqrt{c}\,\varepsilon^{0.75})$ and depends mildly on $c$: at $c = 0.1$ it is 6.90 in float32 and 14.44 in float64. The maps into Klein (`poincare_to_klein`, `hyperboloid_to_klein`, `pv_to_klein`) do not project; call `proj` after mapping far points in. See [the chart ceiling and floor](../user-guide/numerical-stability.md#klein-chart-ceiling).

::: hyperbolix.manifolds.klein.Klein
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

Distance-preserving maps between the Poincaré ball, hyperboloid, Proper
Velocity (PV), and Klein models — all coordinate models of the same hyperbolic space.
Provides Poincaré ↔ Hyperboloid, Poincaré ↔ PV (PVNN Eq. 4), the direct
Hyperboloid ↔ PV map (PV coordinates are the space-like part of the 4-velocity),
and Klein ↔ Poincaré / Hyperboloid / PV (`klein_to_poincare`, `poincare_to_klein`,
`klein_to_hyperboloid`, `hyperboloid_to_klein`, `klein_to_pv`, `pv_to_klein`; a Klein
point is an Einstein velocity, and its proper velocity is `k/√g_k`).

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

### Klein Operations

```python
import jax.numpy as jnp
from hyperbolix.manifolds import Klein, isometry_mappings

klein = Klein()
c = 1.0

x = jnp.array([0.3, 0.5])   # points in the ball c‖x‖² < 1
y = jnp.array([-0.2, 0.4])

d = klein.dist(x, y, c)          # cancellation-free arsinh form
v = klein.logmap(y, x, c)        # parallel to the chord y - x
y_rec = klein.expmap(v, x, c)    # back to y

gamma = klein.lorentz_factor(x, c)                         # 1/√(1 - c‖x‖²)
m = klein.einstein_midpoint(jnp.stack([x, y]), None, c)    # = Lorentz centroid

# Einstein addition in Klein = Möbius addition in Poincaré, through the isometry
p = isometry_mappings.klein_to_poincare(x, c)
x_back = isometry_mappings.poincare_to_klein(p, c)
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
