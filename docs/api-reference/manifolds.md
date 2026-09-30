# Manifolds API

This page documents the core manifold operations in Hyperbolix. Each manifold is a class that provides geometric operations and automatic dtype casting.

## Overview

Hyperbolix provides seven base manifold classes plus a composition class:

- **Euclidean**: Flat Euclidean space (baseline)
- **Poincaré Ball**: Conformal model of hyperbolic space
- **Hyperboloid**: Lorentz/Minkowski model of hyperbolic space
- **Proper Velocity**: Unconstrained $\mathbb{R}^n$ model from special relativity (Chen et al. 2026)
- **κ-Stereographic**: Signed-curvature model unifying hyperbolic, Euclidean, and spherical geometry in one manifold (Bachmann et al. 2020)
- **Klein**: Beltrami–Klein ball model — straight-chord geodesics and Einstein gyrovector operations (Mao et al. 2024; Zhang et al. 2026)
- **HalfSpace**: Poincaré upper half-space model — the height is the last coordinate, and the gyrovector operations are Möbius operations carried over from the ball by the Cayley transform
- **Product Manifold**: Heterogeneous-curvature product spaces $M_1 \times M_2 \times \dots \times M_n$ (Gu et al. 2019)

All manifolds share a common interface defined by the `Manifold` protocol and support:

- **Automatic dtype casting**: Pass `dtype=jnp.float64` for higher precision (needs `jax_enable_x64`)
- **vmap-native methods**: Methods operate on single points; use `jax.vmap` for batching
- **JIT compatibility**: All methods are JIT-compilable
- **Learnable curvature**: Use the `LearnableCurvature` module to add trainable curvature to any model (positive `softplus`/`log` or signed `identity` reparameterization, clamped by default to `[init_c/10, init_c·10]`, or `[-10, 10]` for identity)

## Manifold Protocol

!!! note "The `Curvature` type"
    Manifold methods take the curvature as a positional `c: Curvature` argument
    (`hyperbolix.manifolds.Curvature`). It is the union
    `ScalarCurvature | Sequence[ScalarCurvature]`, where `ScalarCurvature = float |
    jax.Array`: every single manifold (`Poincare`, `Hyperboloid`, `ProperVelocity`,
    `Klein`, `HalfSpace`, `Stereographic`, `Euclidean`) takes a **scalar** `c`, while `ProductManifold` takes a **sequence**
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

    - `VERSION_MOBIUS_DIRECT` (0): direct Möbius distance formula (default)
    - `VERSION_MOBIUS` (1): Möbius via addition
    - `VERSION_METRIC_TENSOR` (2): Direct metric tensor integration

    Constants are available as `poincare.VERSION_MOBIUS_DIRECT` etc., or from
    `hyperbolix.manifolds.poincare`.

    Slots 0 and 2 compute the same `asinh` distance, accurate to the ball chart's ceiling; slot 1
    saturates there:

    $$d(x, y) = \frac{2}{\sqrt{c}}\operatorname{arcsinh}\!\left(\frac{\sqrt{c}\,\lVert x-y\rVert}{\sqrt{(1-c\lVert x\rVert^2)(1-c\lVert y\rVert^2)}}\right)$$

    Measured values: [Which Version to Use?](../user-guide/numerical-stability.md#which-version-to-use).

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
    | $> 0$ | $< 0$ | hyperbolic — the `Poincare(c)` geometry (see below) |
    | $= 0$ | $0$ | Euclidean (factor-2 limit; see below) |
    | $< 0$ | $> 0$ | spherical (stereographic projection of the sphere) |

    Internally the paper's $\kappa = -c$. This is **sign-flipped from the paper/geoopt $\kappa$** (their $\kappa > 0$ = spherical), chosen so `c` matches every other hyperbolix manifold and so `Stereographic(c)` matches `Poincare(c)` for $c > 0$. Most ops return `Poincare`'s bits; `dist`, `logmap`, `expmap`, `expmap_0`, `logmap_0` and `scalar_mul` agree only to rounding (see [κ-Stereographic numerics](../user-guide/numerical-stability.md#stereographic-near-zero-curvature)).

!!! warning "The Euclidean limit carries a factor of 2"
    The conformal factor is $\lambda^\kappa_x = 2/(1 - c\lVert x\rVert^2)$, so $\lambda^\kappa_0 = 2$ and the metric at $c = 0$ is $4\cdot I$, **not** $I$. As $c \to 0$: `addition`/`expmap`/`logmap` reduce to the *bare* Euclidean $x{+}y$ / $x{+}v$ / $y{-}x$, but `dist` $\to 2\lVert x-y\rVert$ and `tangent_norm` $\to 2\lVert v\rVert$ (paper Thm. 3). This matches Poincaré's own `dist_0` $\to 2\lVert x\rVert$, and therefore does **not** equal the separate `Euclidean` manifold's `dist` (bare metric $I$). Use `Euclidean` for un-scaled flat geometry; use `Stereographic` at $c=0$ only as the *continuous limit* of the curved family.

!!! note "Curvature derivatives at zero"
    The shared Möbius denominator evaluates separate signed factorizations on the
    $c>0$ and $c\le0$ sides. For nonzero operands they match the literal
    polynomial value and curvature derivative at $c=0$; an exactly zero operand
    retains only the bounded residual from the existing `MIN_NORM` radial floor.

!!! note "Scope"
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
    - `hcat`: Lorentz direct concatenation for convolutions
    - `log_radius_concat`: log-radius–preserving concatenation (digamma-scaled `hcat`; Shi et al. 2026, Sec. 4.3)

!!! note "Origin derivatives"
    `gyro_difference` and `ptransp` use Cartesian formulas when either endpoint's
    scaled spatial radius is at most 1e-1, and retain the stable polar frame
    otherwise. `logmap` uses the regular Cartesian expression at an exact origin
    endpoint and the stable polar frame otherwise. Ordinary autodiff preserves derivatives with respect to
    an origin endpoint. `busemann` uses a projected-coordinate
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
    - **No boundary and no constraint to drift from**; accuracy equals the hyperboloid's, since the pairwise ops go through it
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
    Cartesian branch and its origin derivatives.
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
    `proj` uses the Poincaré ball's `eps**0.75` boundary margin, but a Klein point sits at $\tanh(a)$ rather than $\tanh(a/2)$ ($a = \sqrt{c}\,d_0$, the scaled radius), so the projection ceiling at $c = 1$ is $a = \operatorname{atanh}(1 - \varepsilon^{0.75})$ = **6.32** in float32 and **13.86** in float64, half of Poincaré's 12.65 / 27.7. The maps into Klein (`poincare_to_klein`, `hyperboloid_to_klein`, `pv_to_klein`) do not project; call `proj` after mapping far points in. See [the chart ceiling and floor](../user-guide/numerical-stability.md#klein-chart-ceiling).

::: hyperbolix.manifolds.klein.Klein
    options:
      show_source: true
      heading_level: 3

## HalfSpace

The Poincaré upper half-space model stores a point as $x = (x_s, x_n)$ with the height $x_n > 0$ in the **last** coordinate. The sectional curvature is $-c$ and the metric is conformal:

$$g_x(u, v) = \frac{\langle u, v\rangle}{c\,x_n^2}.$$

The origin is $o = e_n/\sqrt{c}$, not 0, and every `_0` method is the general method evaluated at $o$. Geodesics are vertical lines and half-circles orthogonal to the boundary $x_n = 0$. In chart coordinates `expmap`, `logmap` and `ptransp` do not depend on $c$, because a constant factor on the metric leaves the Christoffel symbols unchanged; only the origin, distances, norms and inner products carry $c$. The convention follows HTorch (github.com/ydtydr/HTorch, `manifolds/halfspace.py`). `HalfSpace` conforms to the scalar-`c` `Manifold` protocol.

!!! info "Relation to the Poincaré ball and the hyperboloid"
    The Cayley transform `halfspace_to_poincare` maps $o$ to the ball origin with differential $\tfrac12 I$, the vertical axis through $o$ onto the $e_n$ diameter, $x_n \to \infty$ to the north pole $e_n/\sqrt{c}$, and $x_n \to 0$ to the boundary sphere. The isometries in `isometry_mappings`, with $g_k = 1 - c\lVert k\rVert^2$:

    - `halfspace_to_poincare(x)` $= \big(2x_s,\ (c\lVert x\rVert^2 - 1)/\sqrt{c}\big)\big/\big(c\lVert x_s\rVert^2 + (1 + \sqrt{c}\,x_n)^2\big)$, `poincare_to_halfspace(p)` $= \big(2p_s,\ (1 - c\lVert p\rVert^2)/\sqrt{c}\big)\big/\big(c\lVert p_s\rVert^2 + (\sqrt{c}\,p_n - 1)^2\big)$
    - `halfspace_to_hyperboloid(x)` $= \big(c\lVert x\rVert^2 + 1,\ 2\sqrt{c}\,x_s,\ c\lVert x\rVert^2 - 1\big)/(2c\,x_n)$ (time coordinate first), `hyperboloid_to_halfspace(X)` $= \big(X_{\mathrm{mid}},\ 1/\sqrt{c}\big)\big/\big(\sqrt{c}\,(X_0 - X_n)\big)$
    - `halfspace_to_klein(x)` $= \big(2x_s,\ (c\lVert x\rVert^2 - 1)/\sqrt{c}\big)/(1 + c\lVert x\rVert^2)$, `klein_to_halfspace(k)` $= \big(k_s,\ \sqrt{g_k}/\sqrt{c}\big)/(1 - \sqrt{c}\,k_n)$
    - `halfspace_to_pv(x)` $= \big(2\sqrt{c}\,x_s,\ c\lVert x\rVert^2 - 1\big)/(2c\,x_n)$, the spatial part of the hyperboloid point; `pv_to_halfspace(u)` $= \big(u_s,\ 1/\sqrt{c}\big)\big/\big(\sqrt{c}\,(\sqrt{1/c + \lVert u\rVert^2} - u_n)\big)$

    These are the formulas; the code rearranges the differences that cancel. For example, $c\lVert x\rVert^2 - 1$ is evaluated as $c\lVert x_s\rVert^2 + (\sqrt{c}\,x_n - 1)(\sqrt{c}\,x_n + 1)$, and for $X_n \ge 0$ the gap $X_0 - X_n$ as $(1/c + \lVert X_{\mathrm{mid}}\rVert^2)/(X_0 + X_n)$. None of the maps projects its output.

!!! note "Gyrovector structure: Möbius operations through the Cayley transform"
    - `addition(x, y)` $= x \oplus y = \exp_x\!\big(P_{o\to x}(\log_o y)\big)$
    - `gyro_difference(x, y)` $= (\ominus x) \oplus y = \exp_o\!\big(P_{x\to o}(\log_x y)\big)$, built from $\log_x y$, so two close points give a point close to $o$ without cancellation
    - `scalar_mul(r, x)` $= r \otimes x = \exp_o(r\,\log_o x)$
    - the gyro-inverse is $\ominus x = (-x_s,\ x_n)/(c\lVert x\rVert^2)$

    Each equals the Möbius operation of the Poincaré ball conjugated by `halfspace_to_poincare`. Unlike `Poincare`, the identity element is $o = e_n/\sqrt{c}$, and $\ominus x \ne -x$.

!!! note "Pairwise operations: cancellation-free"
    `dist` has a single implementation; `version_idx` is accepted and ignored. With $r = \big((y - x)/\sqrt{x_n}\big)/\sqrt{y_n}$, divided in sequence so that $x_n y_n$ is never formed,

    $$d(x, y) = \frac{2}{\sqrt{c}}\,\operatorname{arsinh}\!\Big(\tfrac12\lVert r\rVert\Big).$$

    This is the same function as the textbook $\operatorname{arcosh}\!\big(1 + \lVert y - x\rVert^2/(2x_n y_n)\big)/\sqrt{c}$, without its `acosh` of a number close to 1 for close pairs. `logmap` builds $\theta/\sinh\theta$ from the same $r$. `expmap` evaluates its denominator in a second, non-cancelling form for near-vertical upward steps. `ptransp` is a rational formula with no transcendental functions. The derivations and measured errors are in the [numerical-stability guide](../user-guide/numerical-stability.md#halfspace-numerics).

!!! warning "Storage floor and the float ceilings"
    The error that remains is a stored point's own rounding, about $0.4\,(\varepsilon/2)\cosh(\sqrt{c}\,\delta)/\sqrt{c}$ as a distance, where $\delta$ is the distance to the vertical geodesic through $o$ and $\cosh(\sqrt{c}\,\delta) = \lVert x\rVert/x_n$. On that axis the floor does not grow with the height. Two cases return `inf`/NaN at a finite radius: the pairwise operations past a scaled distance $\sqrt{c}\,d = 88.72$ in float32 (709.78 in float64), where $\lVert r\rVert^2$ overflows, and `expmap` for an exactly vertical upward step longer than $\theta = 87.34$ in float32 (708.40 in float64), where $e^{-\theta}$ underflows. See [the half-space numerics](../user-guide/numerical-stability.md#halfspace-numerics).

!!! note "Projection, retraction and membership"
    `proj` floors the height at the dtype's smallest normal number, `jnp.finfo(dtype).tiny`, so a valid point is returned unchanged. `retraction` is `proj((x_s + v_s, x_n·exp(v_n/x_n)))`: first order, exact for vertical steps, and never outside the half-space, where the plain `x + v` leaves it for large downward steps. `is_in_manifold` checks that `x` is finite with `x_n > 0`; `atol` is accepted and unused, as for `ProperVelocity`.

::: hyperbolix.manifolds.halfspace.HalfSpace
    options:
      show_source: true
      heading_level: 3

## Product Manifold

Heterogeneous-curvature product space $P = M_1 \times M_2 \times \dots \times M_n$ where each factor $M_i$ can be any single manifold (Poincaré, Hyperboloid, Proper Velocity, Klein, HalfSpace, Stereographic, Euclidean) with its own curvature $c_i$. Points are represented as flat concatenated arrays of shape `(total_dim,)`.

The geodesic distance on a product Riemannian manifold is Pythagorean over component distances:

$$d_P(x, y) \;=\; \sqrt{\sum_{i=1}^{n} d_{M_i}(x_i, y_i)^2}$$

where $x_i$, $y_i$ are the per-factor slices of the flat points.

!!! note "Per-factor `c` argument"
    Every geometry method takes a positional `c` that must be a sequence of length `n_factors`: pass `product.curvatures` for static curvatures, or a tuple of `LearnableCurvature` calls for trainable ones. `isinstance(product, Manifold)` is `True`, but `Manifold` is typed with a scalar `c`; annotate code that passes a per-factor sequence as `ProductManifold`. See the [Manifolds User Guide — Curvature in ProductManifold](../user-guide/manifolds.md#curvature-in-productmanifold) for the full pattern.

::: hyperbolix.manifolds.product.ProductManifold
    options:
      show_source: true
      heading_level: 3

## Isometry Mappings

Distance-preserving maps between the Poincaré ball, hyperboloid, Proper
Velocity (PV), Klein, and half-space models — all coordinate models of the same hyperbolic space.
Provides Poincaré ↔ Hyperboloid, Poincaré ↔ PV (PVNN Eq. 4), the direct
Hyperboloid ↔ PV map (PV coordinates are the space-like part of the 4-velocity),
Klein ↔ Poincaré / Hyperboloid / PV (`klein_to_poincare`, `poincare_to_klein`,
`klein_to_hyperboloid`, `hyperboloid_to_klein`, `klein_to_pv`, `pv_to_klein`; a Klein
point is an Einstein velocity, and its proper velocity is `k/√g_k`), and half-space ↔
Poincaré / Hyperboloid / Klein / PV (`halfspace_to_poincare`, `poincare_to_halfspace`,
`halfspace_to_hyperboloid`, `hyperboloid_to_halfspace`, `halfspace_to_klein`,
`klein_to_halfspace`, `halfspace_to_pv`, `pv_to_halfspace`; the first pair is the Cayley
transform, and `halfspace_to_pv` is the spatial part of `halfspace_to_hyperboloid`).

None of the half-space maps projects its output. `poincare_to_halfspace` and
`klein_to_halfspace` floor the ball's gap $1 - c\lVert\cdot\rVert^2$, as the other maps out
of a ball do, so the height they return is positive. `klein_to_halfspace` maps every point
of the closed Klein ball, the north pole included, to a finite point;
`poincare_to_halfspace` returns `inf`/NaN at the north pole $e_n/\sqrt{c}$, the half-space's
point at infinity, which a `Poincare.proj`-projected point never reaches.
`hyperboloid_to_halfspace` and `pv_to_halfspace` do not floor $x_n$. `klein_to_halfspace`
is limited by the Klein chart's own rounding, relative error $\varepsilon\cosh^2(a)$ at
scaled radius $a$, in every spelling (see
[the Klein chart's floor](../user-guide/numerical-stability.md#klein-chart-ceiling)).

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
import jax

jax.config.update("jax_enable_x64", True)  # float64 needs x64 enabled

from hyperbolix.manifolds import Poincare
import jax.numpy as jnp

# High-precision manifold
poincare_f64 = Poincare(dtype=jnp.float64)

x = jnp.array([0.1, 0.2], dtype=jnp.float32)  # float32 input
y = jnp.array([0.3, -0.1], dtype=jnp.float32)
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

### HalfSpace Operations

```python
import jax.numpy as jnp
from hyperbolix.manifolds import HalfSpace, Poincare, isometry_mappings

halfspace = HalfSpace()
c = 1.0

x = jnp.array([0.1, 1.0])   # (x_s, x_n): the height x_n > 0 is the last coordinate
y = jnp.array([0.3, 0.5])

d = halfspace.dist(x, y, c)                # (2/√c)·asinh(‖r‖/2), cancellation-free
v = halfspace.logmap(y, x, c)              # tangent_norm(v, x, c) equals d
y_rec = halfspace.expmap(v, x, c)          # back to y
o = halfspace.expmap_0(jnp.zeros(2), c)    # the origin e_n/√c = [0, 1]

# The Cayley transform to the Poincaré ball preserves the distance
p_x = isometry_mappings.halfspace_to_poincare(x, c)
p_y = isometry_mappings.halfspace_to_poincare(y, c)
d_ball = Poincare().dist(p_x, p_y, c)                     # equals d
x_back = isometry_mappings.poincare_to_halfspace(p_x, c)  # round trip to x
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
x_hyperboloid = isometry_mappings.pv_to_hyperboloid(jnp.array([0.5, 0.3]), c=1.0)  # on the hyperboloid

# Map to Poincaré ball (intrinsic coordinates, d dims)
x_poincare = isometry_mappings.hyperboloid_to_poincare(x_hyperboloid, c=1.0)

# Map back (round-trip)
x_hyperboloid_recovered = isometry_mappings.poincare_to_hyperboloid(x_poincare, c=1.0)
```

## Numerical Considerations

!!! warning "Float32 Precision"
    Each model holds pairwise ops to a fixed scaled radius $\sqrt{c}\,d$ in float32 (Hyperboloid 16.6, Poincaré 12.6, Klein 6.32 at $c = 1$). Past it, use float64 (`jax_enable_x64` plus `dtype=jnp.float64`).

See the [Numerical Stability](../user-guide/numerical-stability.md) guide for details.
