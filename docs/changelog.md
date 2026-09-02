# Changelog

All notable changes to Hyperbolix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Every zero-initialised hyperboloid gyro-bias was frozen at its init value: `∂(x ⊕ b)/∂b` at `b = origin` was exactly the zero matrix, in both dtypes.** `dist_0` short-circuited on a bitwise `at_origin` guard (`jnp.where(at_origin, 0.0, res)`); `logmap_0` scales the spatial part by `dist_0(y)/‖y_s‖` and inherited it; Lorentz gyro addition `x ⊕ y = Exp_x(PT_{0→x}(Log_0(y)))` inherited it in turn. A bias parameter that starts at the origin, which is what `use_gyro_bias=True` does, therefore received an exactly-zero gradient at step 0 and never left. Affected layers: `HypLinearHyperboloidPLFC`, `HypConv2DHyperboloidILNN`, `HypLinearHyperboloidBusemann`. Measured gyro-bias gradient L2 at init (c = 1, dim 8, batch 4, `dist_0` loss, before → after): float32 `0.0 → 1.37` (PLFC), `0.0 → 4.24` (ILNN conv), `0.0 → 3.07` (Busemann FC); float64 `0.0 → 1.48`, `0.0 → 5.37`, `0.0 → 1.83`. In float32 the gradient was additionally ~1400x over-scaled the moment the bias left the origin, because the `acosh` floor below was divided by the true radius: the PLFC gradient at bias radius `1e-6` was 1535x its value at `1e-2`, a ratio that is now 1.06x. The Poincaré `HypLinearPoincareBusemann` gyro-bias goes through Möbius addition and was never affected (7.64 before and after). **A model trained with a gyro-bias on any of the three hyperboloid layers was trained with that bias pinned to the origin; retraining under this fix changes results.**
- **Hyperboloid `dist_0`/`logmap_0` recovered the geodesic radius from the time coordinate `x₀`, which cannot resolve a small radius; they now read it off the spatial part.** `dist_0` was `acosh(clip(√c·x₀, 1))/√c`. On the sheet `x₀ = cosh(√c·d)/√c ≈ (1 + c·d²/2)/√c`, so `d` survives only to `sqrt(eps)` resolution (relative error `eps/(2·c·d²)`), and `acosh`'s `1 + 10·eps` domain clamp flattened every float32 radius below `sqrt(20·eps)/√c = 1.5441e-3` onto that floor value (1.5441e-3 was returned for every such point, 1500x too large at radius 1e-6; only the bitwise origin returned 0). The smoothened arm was worse: `smooth_clamp_min` (β = 50) adds `ln2/50` to the `acosh` argument, i.e. a floor of `acosh(1 + ln2/50)/√c = 0.16632/√c` under *every* point that is not bitwise the origin, in **both** dtypes (measured: it returned 0.16632 at every radius ≤ 1e-2 and 0.1815 at radius 0.1, agreeing with the true value only from radius ≈ 1). Slot 0 is now the exact on-sheet identity `arcsinh(√c·‖x_s‖)/√c`, with no domain clamp and a derivative bounded by 1; slot 1 is the same with the spatial radius floored in quadrature at `20·eps`, a floor of `arcsinh(20·eps)/√c` (2.4e-6/√c float32, 4.4e-15/√c float64) that equals the pairwise smoothened floor `2·arcsinh(10·eps)/√c` from 1.1.2 to first order. `logmap_0` is now `[0, arcsinh(u)/u · y_s]` with `u = √c‖y_s‖`: it no longer routes through `dist_0`'s version switch, `‖log_0(y)‖ = dist_0(y)` holds by construction instead of by cancelling a floored numerator against an unfloored denominator, and the trailing `tangent_proj` is dropped (it is the identity at the origin, but it routes through the Minkowski inner product, which turns one `inf` spatial coordinate into an all-NaN vector). `scalar_mul` is now literally `exp_0(r · log_0(x))`; its old normalize-then-rescale detour carried a `sqrt(max(‖v‖², MIN_NORM))` floor, an effective `3.16e-8` floor on `‖log_0 x‖` that was invisible only while the larger `acosh` floor sat in front of it. Median relative error at c = 1, dim 8 (A100, jax 0.9.1; the CPU backend and jax 0.11.0 agree in every floored cell):

    | radius | float32 `dist_0` | float32 `log_0(exp_0(v))` | float64 `dist_0` | float64 round trip |
    | --- | --- | --- | --- | --- |
    | 1e-6 | 1.5e3 → 2.5e-9 | 1.5e3 → 0 | 4.4e-5 → 0 | 4.4e-5 → 1.3e-16 |
    | 1e-3 | 5.4e-1 → 9.8e-8 | 5.4e-1 → 0 | 5.9e-11 → 0 | 1.5e-10 → 1.2e-16 |
    | 1e-2 | 5.0e-4 → 2.7e-8 | 5.2e-4 → 0 | 1.0e-12 → 1.7e-16 | 8.3e-13 → 1.2e-16 |
    | 0.1 | 3.5e-6 → 1.8e-8 | 1.4e-7 → 3.9e-8 | 7.6e-15 → 1.4e-16 | 9.5e-15 → 1.1e-16 |
    | 1 to 40 | ≤7.1e-8 → ≤5.2e-8 | ≤1.0e-7 → ≤8.8e-8 | ≤1.3e-16 → ≤1.9e-16 | ≤1.4e-16 → ≤1.4e-16 |

    Consumers that inherit the fix with no change of their own: gyro addition, `scalar_mul`, `logmap`'s origin fallback, `HyperboloidGyroRMSNorm` (it divides by `dist_0`, so in float32 every sample inside radius 1.5e-3 was mis-normalised), and the three gyro-bias layers above. Pairwise `dist(origin, x)` is a separate code path and still carries an `O(eps/r)` relative error from the `_polar_frame` gap term near the origin (float32: 4.6e-2 at radius 1e-6, ~1e-3 at 1e-4); closing that is a planned separate change, so prefer `dist_0` for distances to the origin. **Supersedes** the claim in the 1.1.2 entry below that origin-chart operations "read the radius straight off the ambient time coordinate and were never affected": reading it off the time coordinate is precisely the defect fixed here. That entry's two-point cancellation claims, and `expmap_0`, are unchanged

### Changed
- **Breaking: `dist_0(..., version_idx=2)` and `version_idx=3` now return the pre-fix `acosh` arms.** Those two slots used to duplicate slots 0 and 1 for the origin distance; they now select `VERSION_LEGACY` / `VERSION_LEGACY_SMOOTHENED`, mirroring what they have meant for the pairwise `dist` since 1.1.2, and exist only to reproduce results computed before this fix. Code that passed 2 or 3 to `dist_0` and expected default behavior must pass `VERSION_DEFAULT` (0) or `VERSION_SMOOTHENED` (1)
- **`dist_0` and `logmap_0` read the radius off `x_s` and ignore `x₀`.** An off-sheet input therefore gets the radius of its `proj` projection, which is the convention `proj` itself follows (it rebuilds `x₀` from `x_s`) and what `ProperVelocity.dist_0` already did. A raw `jax.grad(dist_0)` consequently has its support on `x_s` instead of on `x₀`; the two gradients differ by a multiple of the constraint normal, so after `egrad2rgrad`/`tangent_proj` the Riemannian gradient is identical and the Riemannian optimizers are unaffected
- **`Poincare.expmap_0` clamps the boundary on the scalar instead of re-projecting its output.** It now computes `t = min(tanh(√c·‖v‖), √c·max_norm)` and `res = t/(√c·‖v‖) · v`, so `‖res‖ ≤ max_norm` holds by construction and the trailing `proj` (a second norm reduction over the op's own output) is gone. Under `jax.jit(jax.vmap(...))` at batch 1e7, dim 32 the op compiles to 3 kernels with 1 reduction instead of 5 with 2 (`logs/2026-09-02_submission-numerics/probe_xla_kernels.py`, A100, jax 0.9.1 and 0.11.0). Values move only on rows the projection clamps (‖v‖ ≳ 8/√c in float32): at most 3.4e-7 relative (float32, c ≤ 5) / 6.0e-16 (float64), overshooting `max_norm` by at most 2.2 eps, 50x inside the `eps**0.75` margin; Jacobians agree on unclamped rows. `_gyrovector_core._max_norm` is the factored bound; `proj` itself is unchanged bit-for-bit.
- **Float32 `tanh` and `atanh` in `hyperbolix.utils.math_utils` are closer to correctly rounded** (exact bit-pattern ulps vs a float64 reference, 20k inputs per range, max ulp before → after). `atanh` below 1/8 uses the odd Maclaurin series through x⁹ (float32 only; float64 keeps the single-`log1p` form bit-for-bit): GPU 3 → 1, CPU 2 → 1. `tanh` for |x| ≥ 1/8 (float32; 1/2 in float64) uses `t/(t + 2)` with `t = expm1(2|x|)`: GPU [0.125, 0.9] 4 → 3 and [0.9, 7] 5 → 2; CPU [0.9, 7] 4 → 1; float64 CPU [0.9, 7] 5 → 1. No range regresses on either backend; both wrappers stay bitwise odd with derivative exactly 1 at 0, seams are continuous to ≤ 2 ulp, and the float32 derivative's error vs `1 − tanh²` drops from 1.2e-6 to 2.4e-7. Effect on the float32 Poincaré round trip `logmap_0(expmap_0(v))` (dim 32, c = 1, median relative error): radius 4: 1.69e-5 → 5.4e-6 (GPU) and 5.7e-6 (CPU); radius 1e-3 on CPU 2.4e-7 → 1.2e-7; GPU at radius ≤ 1 unchanged (already at the noise floor). Below 1/8 `tanh` still calls `jnp.tanh` (4 ulp on both backends); the `expm1` form is 6 ulp on CPU there.

### Notes
- **Near-origin float32 accuracy depends on the backend, and the gap is XLA's transcendental kernels, not the formulas.** Exact bit-pattern ulp error against a float64 reference (20k inputs in `[1e-4, 0.9]`, max / mean): `tanh` 4 / 0.85 (XLA GPU), 4 / 0.90 (XLA CPU), 1 / 0.01 (torch CPU), 2 / 0.17 (torch CUDA); `atanh` 3 / 0.45 (XLA GPU), 2 / 0.25 (XLA CPU), 1 / 0.00 (torch CPU), 3 / 0.45 (torch CUDA, bit-identical to XLA GPU); `arcsinh` 2 on all four; `expm1` 1 (GPU) / 5 (CPU). Consequence for float32 Poincaré `logmap_0(expmap_0(v))` at radius 1e-3 (dim 32, median relative error): 2.4e-7 on the CPU backend before the `tanh`/`atanh` change above (1.2e-7 after), exactly 0 on GPU. A torch-based library reaches ~1.5e-8 on CPU because torch's CPU `tanh`/`atanh` are correctly rounded; on CUDA it has no such edge. The closed forms are identical and nothing in hyperbolix is wrong here, but quote the backend alongside any near-origin float32 accuracy number
- **Poincaré round-trip ceiling.** `proj` holds points inside `1/√c` by a margin of `eps**0.75` (`_gyrovector_core._get_max_norm_eps`), which caps the largest representable geodesic radius at `atanh(1 − eps**0.75)/√c`. Past it `expmap_0` saturates and `logmap_0(expmap_0(v))` returns the ceiling instead of `v`. Measured ceilings (median returned radius, in units of `1/√c`): hyperbolix 6.32 (float32) / 13.86 (float64); geoopt and hypLL both 3.11 / 6.10, from fixed margins of 4e-3 and 1e-5; the unguarded closed form 8.66 / 18.72. The wider margin is deliberate, capping the conformal factor at ~3e5 (float32) / ~1e12 (float64)
- **The float64 CPU `log1p` hole that 1.1.1's `atanh` rewrite worked around is fixed upstream** (openxla/xla#46765, shipped in jax ≥ 0.11.1; tracked as jax-ml/jax#39707). The rewrite stays: it costs nothing and still protects jax ≤ 0.11.0, which is what the lockfile pins (0.9.1)

## [1.1.2] - 2026-08-25

### Fixed
- **Hyperboloid `dist`/`logmap`/`sqdist`/`tangent_norm` lost all precision for two points that were each far from the origin and close together — now cancellation-free at any representable radius.** Every one of these routed through the Minkowski inner product `⟨x,y⟩_L = -x₀y₀ + ⟨x_s,y_s⟩`, a subtraction of two positive terms each roughly `e^(√c·(d₀(x)+d₀(y)))` in size; the number of significant digits lost is set by the Gromov-product-like quantity `√c·(d₀(x)+d₀(y)-d(x,y))`, and past `ln(1/eps)` — 15.9 (float32) / 36.0 (float64) — every digit of the result is cancellation noise, not signal. Measured before this fix: float32 `dist` returned `0.0015` for a true distance of `1.0` between two points at radius 10; `logmap` returned `NaN` from radius ~10 (float32) / ~20 (float64); `tangent_norm` returned ~0 (a 100% error, no warning) on exactly-unit tangent vectors past radius 8 (float32) / 20 (float64). Deep metric-learning embeddings routinely sit at radius 30–60, so this was not an edge case. `dist`, `sqdist`, and `logmap` are rewritten to share a `_polar_frame` decomposition built from the hyperbolic law of cosines in half-angle (haversine) form — a sum of non-negative terms with nothing to cancel; `tangent_norm` eliminates `v₀` through the tangency condition, trading one power of `x₀` out of the error (now exact to radius ~15 float32 / ~25 float64, up from ~8/~20). Values inside the previously-trusted regime change within the old float32/float64 noise level; values past the threshold change from silent garbage to correct. `VERSION_LEGACY` (2) and `VERSION_LEGACY_SMOOTHENED` (3) reproduce the pre-fix acosh-based arms bit-for-bit, for reproducing results computed before this fix — `VERSION_DEFAULT` (0) and `VERSION_SMOOTHENED` (1) now point at the stable forms. `expmap` (two-point), `ptransp`, `tangent_proj`, `tangent_inner`, `egrad2rgrad`, and `is_in_manifold` still route through the Minkowski inner product and remain unsafe past the same threshold; origin-chart operations (`dist_0`, `logmap_0`, `expmap_0`) read the radius straight off the ambient time coordinate and were never affected. New `hyperbolix.utils.safe_norm`/`safe_hypot`/`safe_normalize` — overflow- and underflow-free norm primitives the new decomposition depends on — are exported alongside the fix. **Supersedes** the `VERSION_SMOOTHENED` floor value (`≈ 0.166/√c`) quoted in the `smooth_clamp` entry below (1.1.0): that floor belonged to the old acosh-based arm now at `VERSION_LEGACY_SMOOTHENED`. The new `VERSION_SMOOTHENED`'s floor is `2·arcsinh(10·eps)/√c ≈ 2.4e-6/√c` (float32) — see the [numerical-stability guide](user-guide/numerical-stability.md#hyperboloid-distance-versions) for the full version breakdown
- **`lorentz_residual` and `lorentz_midpoint` lost the Minkowski square to catastrophic cancellation at large spatial radius.** Both build a raw ambient vector `h` and normalize it back onto the sheet by `sqrt(|<h,h>_L|)`, and both computed `<h,h>_L = -h_0^2 + ||h_s||^2` naively: for on-sheet inputs `h_0 ≈ ||h_s||`, so each square is `~||s||^2` (spatial radius of the inputs) while their difference is only `O(1/c)` — float32 relative error therefore grows as `eps32 * c * ||s||^2` with `eps32 ≈ 1.19e-7`. Measured against float64 at `c=0.1`, `x ≈ y`, `d=16` (forward / gradient relative error): `9.9e-5` / `3.0e-4` at `||s||=1e2`, `6.1e-3` / `1.8e-2` at `1e3`, `6.4e3` / `1.0` at `1e4` — past `||s|| ~ 1e4` the computed square flips sign, the `abs()` in the normalizer hides the flip, and the `maximum(., 1e-7)` floor silently inflates the output by `2/sqrt(1e-7) ≈ 6325×` (measured 6324.6 at `c=0.1`, `||s||=2e4`, `x == y`). Never a NaN, never a warning, and dormant at norm-capped activations (`||s|| ≲ 20`); the gradient path degrades ~3× earlier in `||s||` than the forward, from a second cancellation in the normalizer's Jacobian. Both functions now evaluate `<h,h>_L` through exact identities — valid for any weights given inputs on the sheet (`<x,x>_L = -1/c`), verified in exact rational arithmetic: for the residual `h = x + w·y`, `<h,h>_L = -(1+w)^2/c - w·<x-y, x-y>_L`; for the midpoint with reference `p` the vertex-nearest input, `delta_m = x_m - p`, `W = Σ_m w_m` and `Delta = Σ_m w_m·delta_m`, `<h,h>_L = -W^2/c - W·Σ_m w_m·<delta_m, delta_m>_L + <Delta, Delta>_L`. Every rounded term is now `O(||x-y||^2)` instead of `O(||s||^2)`: new float32 error at `||s||=1e4` is `2.1e-5` forward / `6.0e-4` gradient (residual) and `1.6e-5` / `4.3e-4` (midpoint, M=8 softmax weights), and no regime where the old form was still usable measured worse (swept `c ∈ {0.01, 0.1, 1}`, `||s||` from 1 to `1e5`, near/far/same-direction/identical points; at `||s|| ≤ 20` old and new agree to ~2e-7 relative, not bit-identical — different rounding order). The one float32 limit that remains is the difference `x - y` itself: points that share a direction at `||s|| ≳ 1e4` lose it to rounding before either form sees it, and neither form is trustworthy there — that regime needs float64. For `w_y >= 0` resp. non-negative weights the identities give `c·|<h,h>_L| >= (1+w)^2` resp. `>= W^2`, so the `abs()` and the `eps` floor are now provably inactive (both kept as insurance). Affected layers: `LorentzResidual`, `HypformerPositionalEncoding`, `HyperbolicFullAttention` (attention aggregation and head averaging), `HyperboloidGyroRMSNorm`, hyperboloid `frechet_mean`, and any direct caller of the two functions. The LResNet reference implementation (He et al. 2025) has the same defect. Both outputs additionally have their time coordinate reconstructed from the spatial part via `spatial_to_hyperboloid` (`x_0 = sqrt(||x_s||^2 + 1/c)`, the same convention as `hrc`/`htc`), so they are on-sheet by construction — as the old self-normalizing `h/sqrt(-c<h,h>_L)` was — even for inputs float storage cannot keep on the sheet (it needs `eps·x_0^2 << 1/c`, violated past `||s|| ~ 3e3` in float32 / `~1e8` in float64, e.g. a Poincaré ball-boundary point lifted to `x_0 ≈ 5.5e11`): the identity assumes exactly on-sheet inputs, so for those it returned a point off the sheet by 4.8e-3 and hyperboloid `frechet_mean`'s Karcher loop diverged to NaN from it; the midpoint's reference `p` is now the vertex-nearest input rather than `points_0`, bounding every `||delta_m|| <= 2||x_m||`. No API change; cost is O(1) extra for the residual and one extra einsum for the midpoint

## [1.1.1] - 2026-08-04

### Added
- **`hyperbolix.utils.capped_exp`** — `exp` with the argument capped at `0.99*log(finfo.max)`, for `exp` of *unconstrained trainable* log-scale parameters. A runaway parameter otherwise overflows to `inf`, which turns into NaN downstream (`inf - inf` in the hyperboloid time-coordinate algebra) and poisons the whole parameter tree within one optimizer step; `capped_exp` saturates finite instead, with a zero (not NaN) gradient past the cap. Below the cap it is a bitwise value- and gradient-identity to `jnp.exp`. Same idiom `LearnableCurvature`'s `"log"` parameterization already used internally, now shared. Applied to the three call sites that had a bare `jnp.exp` on such a parameter: the Busemann FC layer's `alpha = exp(log_scale)`, and both FHCNN/FHNN forward passes' `exp(scale_val) * sigmoid(...)` time-coordinate scaling

### Fixed
- **`atanh`/`sinh`/`cosh` float64 accuracy: XLA's CPU lowerings have large ulp errors that hyperbolix's wrappers previously inherited.** `jnp.atanh` is lowered to `0.5*(log1p(x) - log1p(-x))`, and XLA's CPU float64 `log1p` is up to ~129 ulps off for arguments in `[-0.53, -0.28]` (NumPy's is <1 ulp there) — `atanh` inherited ~129 ulp error for `x` in `~[0.28, 0.53]`, peaking at `sqrt(2)-1`. `jnp.sinh`/`jnp.cosh` are separately inaccurate for large arguments: ~17 ulps for `|x|` in `[16, 512]` and ~496 ulps for `[512, 710]` (float64; the jump sits exactly at the power-of-two boundary 512), and ~24.5 ulps in float32 across the full range. Float32 `atanh` was unaffected. `hyperbolix.utils.math_utils.atanh` is rewritten as the odd-symmetrized single-`log1p` identity `atanh(x) = sign(x)*0.5*log1p(2|x|/(1-|x|))` (computed on `|x|` so the `log1p` argument never enters the bad window, with the sign restored via `where` rather than `sign(x)*...` — a `sign`-multiplied form has zero gradient at `x=0`, not the analytic 1). `sinh` is rewritten as `0.5*(expm1(x) - expm1(-x))` (cancellation-free everywhere: the two `expm1` terms have opposite signs) and `cosh` as `0.5*(exp(x) + exp(-x))` with a `custom_jvp` routing its gradient to the accurate `sinh` form (a bare exp-form `cosh`'s autodiff gradient is the naive cancelling `sinh`, inaccurate near 0). All three rewrites measure at a few ulps across their full domain (vs. the errors above) and are the same speed or faster than the builtins. Raw `jnp.sinh`/`jnp.cosh`/`jnp.acosh` call sites that bypassed the wrappers (the FGG spatial lift, the Busemann FC output map, the wrapped-normal log-density, and the uniform-Poincaré volume/samplers) now route through them, closing the same accuracy gap plus, in the `acosh` case, a singular-gradient exposure at an exact-1.0 domain boundary. No API changes; every downstream distance/logmap/MLR computation using these functions gets more accurate for free

## [1.1.0] - 2026-08-01

A correctness release. A full audit of the test suite replaced shape-only checks and
self-comparisons with independently transcribed NumPy/SciPy oracles, and every library
defect it exposed is fixed below. No new capability beyond `hyp_flatten2d`; several
corrections change behavior or call signatures, each marked **Breaking** and each with the
migration in its entry.

### Added
- **`hyp_flatten2d`** (`hyperbolix.nn_layers`) — radius-preserving flatten of an NHWC hyperboloid feature map into one manifold point per image via `Hyperboloid.log_radius_concat`: `(B, H, W, A) → (B, H·W·(A−1)+1)`. The naive alternative (reshape + `hcat`) inflates the expected spatial radius by `≈ √(H·W)` (a 4×4 map: 4×); LogCat's digamma rescale hands the classifier head the per-pixel radius unchanged. This is the flatten step for `HypConv2DHyperboloidILNN` stacks feeding `HypRegressionHyperboloid`/`FGGLorentzMLR`
- **`Poincare.proj_batch`** — batched sibling of `Hyperboloid.proj_batch` (arbitrary leading axes, bit-identical to `vmap(proj)`); HoroPCA and CO-SNE now use it instead of hand-rolled vmaps
- **`hyperbolix.manifolds._base.default_atol(dtype) = sqrt(finfo(dtype).eps)`** — the library-wide constraint-tolerance convention (float32 3.45e-4, float64 1.49e-8) that `is_in_manifold` / `is_in_tangent_space` resolve `None` through on every manifold
- **`HypformerPositionalEncoding` accepts `param_dtype`** and forwards it to its internal `HTCLinear` — float64 models were previously impossible (weights pinned to float32)
- **New exports**: `hcat_ambient_dim` and `busemann_fc_poincare_output` from `hyperbolix.nn_layers` (both already served as test oracles and answer user-facing sizing questions); `lift_ideals` and `orthonormalize_rows` from `hyperbolix.decomposition`

### Fixed
- **`Hyperboloid.log_radius_concat` scaled the wrong way, amplifying the spatial radius instead of preserving it.** The two digamma arguments were swapped: the shipped `exp(½·(ψ(N·d/2) − ψ(d/2)))` is `≈ √N`, so concatenating `N` blocks *inflated* the expected log spatial radius by `≈ log N` — worse than the plain `hcat` the rescale exists to correct. The correct factor is the shrink `exp(½·(ψ(d/2) − ψ(N·d/2))) ≈ 1/√N`: for Gaussian spatial parts `‖v‖² ~ χ²_k` gives `E[log‖v‖] = ½(ψ(k/2) + log 2)`, so restoring the per-block value after widening `d → N·d` requires dividing by the ratio of the two chi radii. Nothing raised, because the mis-scaled output is still a valid manifold point (Lorentz residual ~1e-13) — only its radius was wrong. The Shi et al. 2026 reference implementation has the same sign error. Sole affected layer: `HypConv2DHyperboloidILNN`, where it inflated the spatial radius ≈9.4× per 3×3 conv, compounding with depth (`hcat`-based layers — FGG, FHNN — never used this path). **Behavior change** — see the coupled default-init change below
- **`hyperbolix.optim.has_manifold_params` always returned `False`.** `nnx.Variable` is itself a registered pytree node whose single child is the raw array, so the unguarded `jax.tree_util.tree_leaves` call unwrapped every `ManifoldParam` into an `ArrayImpl` before the `isinstance` check ran — the function reported "no manifold parameters" for a bare `ManifoldParam`, a dict of parameters, `nnx.state(model, nnx.Param)` (its own documented usage) and a model alike. The traversal now stops at `nnx.Variable` leaves. The Riemannian optimizers were **not** affected: `_riemannian_base` already flattened with that guard everywhere it inspects Variable types, so only callers using `has_manifold_params` as a guard saw the false negative
- **Wrapped-normal `log_prob` crashed on `sample()`'s own output whenever the mean was batched and `sample_shape` non-empty** (both `distributions.wrapped_normal_poincare` and `wrapped_normal_hyperboloid`). Step 2 of the density computation vmapped the sample axis `S` and the mean's batch axis `B` together, so any call with `mu.ndim > 1` plus a non-empty `sample_shape` raised a `ValueError`/`TypeError` on shapes the sampler itself produces; step 1 was also mis-batched (2-D `mu` pushed through the single-point `logmap`, relying on broadcasting). Both steps now share a `_vmap_sample_and_batch` helper that vmaps batch axes innermost and sample axes outermost, matching the `(*S, *B, dim)` layout of the sampling side; previously-working shape combinations are bit-for-bit unchanged. `log_prob` is now pinned from four independent directions (2-D polar quadrature integrating to 1, change-of-variables vs `jacfwd` of the sampling map, the flat `c → 0` limit vs `scipy` MVN, and the Poincaré↔hyperboloid isometry cross-check with no Jacobian — the density is w.r.t. Riemannian volume)
- **Poincaré wrapped-normal `log_prob` gradient at `z == mu` was NaN — in float64 too, not just float32.** The radial norm entered the log-det Jacobian without a floor, so the gradient of the norm at the mean produced NaN; it is now floored at `1e-15` exactly like the hyperboloid sibling (the zero gradient `maximum` contributes on the clamped side is correct — the log-det is stationary at `r = 0`). Separately, the shared `_log_det_jacobian_from_r` helper used a single `jnp.where` across its Taylor gate, so reverse-mode AD propagated NaN from the unselected `log(sinh(x)) − log(x)` branch at exactly `r = 0`; it now uses the double-where idiom. That second defect is unreachable from either `log_prob` once `r` is floored, but is fixed for direct callers of the helper
- **`smooth_clamp` could exceed its own bounds on narrow windows.** All three clamps (`smooth_clamp`, `smooth_clamp_min`, `smooth_clamp_max`) are re-derived as gate-free closed forms: the two-sided clamp is the difference form `min + softplus_β(x−min) − softplus_β(x−max)`, provably inside `(min, max)` for every window and β — composing the two one-sided clamps (the old construction) overshoots a narrow window by up to `ln(1+e^(−βw))/β`, which for `βw ≲ 1` is the size of the window itself; a test now pins that the composition fails so it can't be "simplified" back. Evaluation uses the hinge+remainder split `clip(x) + tail − tail` (tails ≤ `ln 2/β`), avoiding float32 catastrophic cancellation, and the gradient is exactly `sigmoid(β(x−min)) − sigmoid(β(x−max))` with correct one-sided values at the hinge. Downstream consequence: the `VERSION_SMOOTHENED` hyperboloid distance is now monotone with floor `acosh(1+ln2/β)/√c ≈ 0.166/√c` and agrees with `VERSION_DEFAULT` beyond `≈ 0.8/√c` (it was previously non-monotone near coincident points)
- **`Euclidean.is_in_manifold` returned constant `True`, accepting NaN/Inf** — it now checks finiteness. The same defect existed on `is_in_tangent_space` for `Euclidean`, `Poincare`, *and* `Stereographic` (the tangent space of an open subset of R^n is R^n — finiteness is the only constraint); all three fixed
- **Wrapped-normal log-det-Jacobian Taylor gate is now dtype-aware.** The `log(sinh(x)/x)` series switch used a float32-tuned threshold in float64 too; it now sits at `(10³·eps)^(1/6)` per dtype — the crossover where the 2-term series' truncation error (`~x⁶/2835`) meets the direct branch's cancellation error (`~eps·|log x|`) — with the series extended to two terms. Validated against a high-precision oracle on both sides of both gates
- **`LearnableCurvature`'s straight-through clamp amplified gradients in the clamped region.** With `parameterization="log"` at the exponent cap, the pass-through handed the raw parameter a gradient scaled by `dc/draw = c` — up to `e^15` at the cap — so one step could catapult the parameter across its whole range. The pass-through now divides the clamped-region gradient by `max(dc/draw, 1)` (implemented as a stop-gradient carrier swap, so no reciprocal is formed that would flush to zero in float32 at the cap). Interior gradients are bit-identical, the forward value is unchanged, and gradients below the lower clamp are never amplified either
- **`ProperVelocity.dist`/`dist_0` returned NaN gradients at the origin** (`grad dist(0,0)`, `grad dist_0(0)`): bare `jnp.linalg.norm` has a `0/0` VJP at the zero vector. Both now use the module's `_safe_norm`, whose gradient at 0 is exactly 0 — the correct subgradient. Value shifts by at most `1e-15`, below float64 resolution of any distance
- **`uniform_poincare` hardcoded float64 and broke its own shape contract.** `log_prob` now returns `x.dtype` (was always float64) and honours the documented `(..., n) → (...)` signature at any rank (`ndim > 2` raised a `dot_general` `TypeError`); `sample` infers its dtype from `center`, then `manifold_module`, then JAX's default float; `volume` takes a `dtype` argument. The Gauss-Legendre tables are plain Python tuples now, so `import hyperbolix` no longer emits three `UserWarning`s (and silently truncated float32 tables) when `jax_enable_x64` is off
- **The two `hyperbolix.utils.helpers` docstring examples raised `AttributeError`** (they called `proj` on the module, not an instance); both rewritten and verified via doctest

### Changed
- **`HypConv2DHyperboloidILNN` default kernel init is now fan-out and norm-preserving.** `kernel_init_std` accepts `None` (the new default), which resolves to `sqrt(1 / out_spatial)`. Near the origin the PLFC chain linearizes to `y_spatial ≈ W @ u_spatial` (both `asinh` and `sinh` are identity to first order, and the bias starts at 0), so a Gaussian `W` of shape `(O, I)` has RMS gain `std·sqrt(O)`; the corrected LogCat hands over the per-pixel spatial radius unchanged, leaving `std = 1/sqrt(O)` as the unit-gain choice. This is **coupled to the LogCat fix above** and is not optional: the previous fixed `0.02` was calibrated against the pre-fix ~√N amplification (the two errors cancelled at width ~64, `0.02·sqrt(64)·9 ≈ 1.4` for a 3×3 kernel), and under the corrected shrink it contracts the mean spatial norm by 7–15× *per layer* — a depth-3 stack sits on the manifold origin at init, with no clamp or warning to show for it. Probe-measured per-layer spatial-norm ratios at depth 3 (c=0.1, 3×3, 8×8 input, float64): 0.82–0.95 with the new default versus 0.05–0.15 with `0.02`. `kernel_init_std=0.02` still reproduces the Shi et al. 2026 draw bit-for-bit, but it now restores a regime that implicitly assumed the pre-fix LogCat. `HypLinearHyperboloidPLFC` is unchanged — it has no LogCat
- **Breaking: `LorentzConv2D` requires `manifold_module` as its first positional argument**, consistent with every other hyperboloid convolution. It is used as a family-identity check (a Poincaré manifold now fails validation instead of being silently misread as `[time, space]`); no manifold method is called in the forward pass. Update: `LorentzConv2D(in_ch, ...)` → `LorentzConv2D(Hyperboloid(), in_ch, ...)`
- **Breaking: Riemannian optimizer learning-rate schedules are resolved at the pre-increment step count**, matching optax's `scale_by_schedule`: the k-th `update()` reads `schedule(k−1)`, so the first update uses `schedule(0)`. Previously the schedule ran one step ahead of every optax reference (a warmup's step-0 rate was never used). Constant learning rates are unaffected; Adam-style bias correction deliberately keeps the post-increment count, which is also optax's convention — the two timings are different counters by design
- **Breaking: `HypRegressionPoincare` kernel init changed from `normal(0, 1)` to `std = (2·in_dim·out_dim)^{-0.5}`**, matching `HypRegressionPoincarePP` and `HypRegressionHyperboloid` (van Spengler et al. 2023). The unscaled init gave row norms `≈ sqrt(in_dim)` that reappear as the outer `‖a‖` multiplier in the MLR score, scaling the logits linearly: at 128→64 the logit std at init was 74, now 0.58. There is no flag to restore the old init
- **Breaking: one `atol` convention across all manifolds.** `is_in_manifold` / `is_in_tangent_space` take `atol: float | None = None`, resolved via `default_atol` (see Added); an explicit value is never floored, clamped, or dropped. Previously `Hyperboloid.is_in_manifold` silently floored `atol` at `1e-4` (no caller could tighten it), `Poincare` ignored it entirely, and `Hyperboloid.is_in_tangent_space` never forwarded the one it accepted. Ball membership now tests the dimensionless residual `c‖x‖² − 1`, so one tolerance means the same at every curvature. Float64 hyperboloid membership is much stricter: genuinely on-sheet points past hyperbolic distance ~11 need an explicit `atol`
- **Breaking (widening): `Euclidean`, `Stereographic`, and `ProductManifold` `dist`/`dist_0` accept a trailing `version_idx`** (accept-and-ignore, per the `ProperVelocity` precedent), so `compute_pairwise_distances` and `get_delta` now work with all six manifolds instead of raising `TypeError` on three. `ProductManifold` cannot forward it to heterogeneous factors — use `component_dist` for per-factor control. The helpers' dead legacy module-API (`_dist`) fallback is removed
- **Breaking (trivial): `import hyperbolix` exposes all six subpackages** (`decomposition`, `distributions`, `manifolds`, `nn_layers`, `optim`, `utils`) in `__all__`; `hyperbolix.distributions` previously raised `AttributeError` from a fresh interpreter until another module imported it. The lone root layer re-export `hyperbolix.PoincareBatchNorm2D` is removed with no alias — import it from `hyperbolix.nn_layers`
- **Breaking: internal plumbing marked private** — `busemann_core.init_weight_norm_params` → `_init_weight_norm_params`, `busemann_core.busemann_score` → `_busemann_score`, `gyro_normalization.GyroBatchNormBase` → `_GyroBatchNormBase`, `GyroRMSNormBase` → `_GyroRMSNormBase`
- **Internal consolidation**: the `1e-15` gradient-safety floor has one canonical home, `hyperbolix.utils.math_utils.MIN_NORM` (the copies in `_gyrovector_core`, `hyperboloid`, `proper_velocity`, `busemann_core`, `hyperboloid_linear`, and the three distribution modules are gone; `isometry_mappings.MIN_DENOM` renamed away); PV's private `_mobius_gyration` copy is deleted in favour of the shared `_gyrovector_core._gyration`; `_batched_transform` is a thin proven-bit-identical wrapper over `_vmap_sample_and_batch`; `hope()`'s time reconstruction delegates to `spatial_to_hyperboloid` (proven bit-identical); `frechet_variance` is typed manifold-generic (`Manifold`, matching its Hyperboloid/PV callers); `HypLinearPoincare`/`HypRegressionPoincare` drop the never-called `compute_mlr_pp` from their required manifold methods

### Documentation
- Fixed four doc examples that raised on construction (`HypRegressionHyperboloid`, `HyperbolicSoftmaxAttention`, `PoincareBatchNorm2D`, `HypConv2DHyperboloid`), the `riemannian_adam` `b1`/`b2` example, and the `uniform_poincare` example (nonexistent `dim=` argument, sample count passed as the dimension, wrong volume value) — all now execute
- Rewrote the layer initialization-scale table to match actual defaults (split `HypLinear*PP` from the regression heads; added `HypLinearHyperboloidPLFC` and `HypConv2DHyperboloidILNN` rows); fixed ambient-vs-spatial channel-convention contradictions (documented the `HTCLinear.out_features`-is-spatial exception; HCat growth is internal — output width is always `out_channels`); corrected "manifold-valued weights" to "manifold-valued bias" for the legacy Ganea layers and repositioned `mark_manifold_param` as guidance for hand-rolled parameters (built-in layers self-tag)
- Real Hypformer citation (Yang et al. 2025) in 7 docstrings; `LorentzConv2D` attribution unified on LResNet; superseded-by notes on `LorentzConv2D`, `HypLinearPoincare`, `HypRegressionPoincare`; `param_dtype` compute-precision wording corrected across the manifold-free layers; GyroBN shift/weight comment labels un-swapped; `DEVELOPER_GUIDE.md` pointers fixed
- Deleted the three placeholder "coming soon" tutorial notebooks and the dead commented-out `mkdocs-jupyter` configuration; updated stale `docs/index.md` counts
- As a one-off release audit, every `python` fence in `docs/`, `README.md`, and `DEVELOPER_GUIDE.md` was executed by hand to confirm it actually runs (this is not an automated CI check). Four more raised: `smooth_clamp(min_val=, max_val=)` (the arguments are `min_value`/`max_value`), `get_delta(seed=42)` (it takes a `key`), `wrapped_normal_hyperboloid.sample(std=)` (it takes `sigma`), and the `is_in_manifold` example, whose "hyperboloid point" was never on the sheet (`-1.5² + 0.2² + 0.3² + 0.1² = -2.11`, not `-1`) — it now builds the ambient point with `spatial_to_hyperboloid`. The HoroPCA example requested `float64` without enabling x64, so it silently truncated to float32 with three `UserWarning`s
- New sections for two breaking changes that previously appeared only in this changelog: the `atol` convention and `default_atol` ([Numerical Stability](user-guide/numerical-stability.md#the-atol-convention)) and learning-rate-schedule timing ([Optimizers](user-guide/optimizers.md#learning-rate-schedules))
- Corrected the layer and test counts in `README.md` and `docs/index.md`: 20+ → 40+ layers, and "3,500+ tests / 850+ test functions" → 3,551 items across 735 test functions

## [1.0.0] - 2026-07-24

First stable release. The public API is considered complete and stable, reaching functional
parity with the broader hyperbolic deep learning ecosystem; future changes follow semantic
versioning.

### Added
- **CO-SNE** (`hyperbolix.decomposition.CoSNE`) — dimensionality reduction and visualization for hyperbolic data (Guo et al. 2022), the hyperbolic analogue of t-SNE. A thin sklearn-style class (`fit` / `fit_transform`; non-parametric, so no out-of-sample `transform`, matching sklearn's `TSNE`) over a pure, JIT-friendly functional core (`fit_cosne`, `conditional_probabilities`, `joint_probabilities`, `low_dim_probabilities`, `kl_divergence_loss`, `magnitude_loss`). High-dimensional similarities are perplexity-calibrated on **squared** hyperbolic distances (paper Eq. 2); low-dimensional similarities use a heavy-tailed Student-t / Cauchy kernel on Poincaré-ball distances (paper Eq. 9 — both the author's-code plain-distance form, default, and the exact squared-distance form via `exact_cauchy=True`). On top of the t-SNE KL divergence it adds a **magnitude loss** `H = (1/N)·Σᵢ(‖xᵢ‖² − ‖yᵢ‖²)²` (paper Eq. 10) that preserves each point's distance-to-origin (the hierarchy depth). Optimization is two-stage projected gradient descent on the ball (early-exaggeration KL-only exploration, then KL + magnitude), with the KL gradient Riemannian-scaled by `(1 − c‖y‖²)²/4` and the magnitude gradient Euclidean (paper §3.7). Curvature-general (a single `c` for both spaces, passed at fit time); Poincaré input `(N, D)` → `(N, K)` ball coordinates, Hyperboloid input `(N, A)` → `(N, K+1)`. Uses **exact autodiff gradients** of the declared losses — a deliberate deviation from the reference's demonstrably buggy hand gradients — so the reference learning rates do not transfer; the `learning_rate` default (`0.5`) is calibrated on a synthetic-cluster recovery test and documented on the class. Double precision recommended for fitting
- **HoroPCA** (`hyperbolix.decomposition.HoroPCA`) — hyperbolic dimensionality reduction via horospherical projections (Chami et al. 2021). A thin sklearn-style class (`fit` / `transform` / `fit_transform`) over a pure, independently usable and JIT-friendly functional core (`fit_horopca`, `transform_horopca`, `horo_projection`, `horopca_loss`). All computation runs on the hyperboloid (Poincaré input is mapped in via the exact isometry); K ideal points are jointly optimized with Adam + gradient clipping to maximize the pairwise-distance variance of the horospherically projected data, which preserves every Busemann coordinate. Poincaré input `(N, D)` → `(N, K)` ball coordinates; Hyperboloid input `(N, A)` → `(N, K+1)`. `K = 1` is supported via a closed form (the reference's Sherman–Morrison inverse is singular there). Exposes `components_`, `mean_`, `boost_`, `losses_`, and pairwise-variance-based `explained_variance_` / `total_variance_` / `explained_variance_ratio_`. Double precision recommended for fitting
- **`hyperbolix.decomposition.frechet_mean`** — manifold-generic Karcher (Fréchet) mean via fixed-point iteration in a `lax.while_loop` (Lorentz-centroid init on the hyperboloid, projected-mean init on the Poincaré ball). The data-centering primitive HoroPCA builds on; configurable `step_size` / `tol` / `max_iters`
- **`Hyperboloid.lorentz_boost(mu, c)`** — the exact Lorentz boost matrix `B` with `B @ mu = origin` (symmetric, proper, orthochronous), used to Fréchet-mean-center data before fitting. Closes a docs–code gap (the method was already advertised in the manifolds API reference)

### Changed
- **`HTCLinear` default init is now fan-in-aware and norm-preserving.** `init_bound` accepts `None` (the new default), which resolves to `sqrt(3 / in_features)` — per-layer input-Jacobian gain ≈ 1, since `htc` applies no nonlinearity to the matmul output. The previous fixed `U(-0.02, 0.02)` was width-dependent **contractive** (gain `0.02/sqrt(3)·sqrt(in_features)` ≈ 0.09 at fan-in 65): stacks of ≥ 2 layers collapsed input variation below the float32 noise floor, becoming a constant map with ≈0 gradients — training silently frozen from step 0 (observed in goal-conditioned RL: depth-2 HTC head, c=0.5, frozen through 3.1M steps; escape time monotone in the bound). The `0.02` was an in-house convention inherited from the FHNN/FHCNN family, **not** the Hypformer reference — the official Hypformer code initializes with Xavier-uniform gain `sqrt(2)` (`bound = 2·sqrt(3/(in+out))`, ~15× larger than 0.02 at fan-in 65), intended for its ReLU + LayerNorm regime; recover it by passing that bound explicitly. The attention layers (`HyperbolicLinearAttention`, `HyperbolicSoftmaxAttention`, `HyperbolicFullAttention`) and `HypformerPositionalEncoding` now default `init_bound=None` and inherit the fan-in-aware init for their HTC projections (matching the reference, whose Wq/Wk/Wv and positional encoding use the same Xavier init). FHNN/FHCNN/PLFC layers are unchanged — their `0.02` matches their references (verified against the official Chen 2021, Bdeir 2023, and Shi 2026 code). **Behavior change** — existing HTC models pick up the new init unless they pass `init_bound=0.02` explicitly (bit-for-bit reproduction of the old init)

## [0.11.0] - 2026-07-14

### Added
- **κ-Stereographic manifold** (`hyperbolix.manifolds.Stereographic`) — a single constant-curvature manifold spanning hyperbolic, Euclidean, and spherical geometry via a **signed** curvature `c` (Bachmann et al. 2020): `c > 0` is hyperbolic and reproduces `Poincare(c)` exactly (both now build on a shared internal `_gyrovector_core` module, so Möbius addition, gyration, conformal factor, and projection are bit-identical), `c = 0` is the flat limit (carrying the conformal factor-2 metric — see the API reference), and `c < 0` is the stereographic projection of the sphere. Note the **sign flip vs the paper/geoopt** (`κ = -c`), chosen so `c` matches every other hyperbolix manifold. Implements the full `Manifold` protocol plus `conformal_factor`, `gyration`, `geodesic`, `geodesic_unit`, and `antipode`, using curvature-generalized trig functions with a dtype-aware Taylor branch near `c = 0` — gated on both `|κ|` and the series argument `|κ|·‖x‖²`, so extreme chart radii (e.g. near-antipodal spherical points) always keep the exact closed form — and an exact closed-form antipode `x/(c‖x‖²)`. Verified compatible with `ProductManifold` factors and the Riemannian optimizers (`ManifoldParam` + `riemannian_adam`/`riemannian_sgd`). The κ-GCN neural-network layers (`mobius_matvec`, weighted gyromidpoint, `dist2plane`, `sproj`) are not yet included. Double precision recommended when training a signed curvature that may cross zero, per the reference
- **`hyperbolix.utils.tanh`** — clamped hyperbolic tangent whose output stays strictly inside `(-1, 1)`: clips the input at the analytic saturation bound `±0.5·log(2/eps)` *and* the output at `±(1 − 10·eps)`, because XLA's float32 `tanh` saturates to exactly `1.0` slightly before the input bound bites. Matches `atanh`'s own domain guard, so `atanh(tanh(x))` can never reach the pole. Used by the κ-stereographic maps in the hyperbolic regime
- **Signed `identity` reparameterization for `LearnableCurvature`** (`parameterization="identity"`, `c = raw`) — **signed**, crossing zero smoothly with gradient 1, for learning the *sign* of curvature with the `Stereographic` manifold; `init_c` may be zero or negative, and the default clamp resolves to a symmetric `[-10.0, 10.0]` (deliberately including 0) instead of the positive `[0.1, 10.0]` used by `softplus`/`log`. The `log` scheme's exponent is now capped at `0.99·log(finfo.max)`, so a drifting raw parameter yields a large finite curvature instead of `inf`; the cap is gradient-transparent under `straight_through_clamp=True`

## [0.10.2] - 2026-07-02

### Changed
- **Legacy Ganea layers (`HypLinearPoincare`, `HypRegressionPoincare`) accept a `curvature` constructor argument** (float or callable, default 1.0) for their manifold-valued `bias` tag instead of hardcoding 1.0. The Riemannian optimizer uses this tag for the bias update, so it must match the `c` passed at call time — previously any `c ≠ 1.0` silently applied the wrong Riemannian correction to the bias

### Fixed
- Kernel-init corrections across hyperbolic layers: `HypRegressionHyperboloid` now uses the fan-scaled kernel init matching `HypRegressionPoincarePP`, and `HypConv2DHyperboloid` gains an opt-in `reset_params="fan_in"` for BatchNorm-free encoders
- `GyroBatchNormBase` gains a `min_var` floor that prevents collapsed-variance saturation
- `FGGConv2D` docstring stated the default weight init is `"kaiming"`; the actual default is `"lorentz_kaiming"`
- `PoincareBatchNorm2D` now documents that the learned variance is an unconstrained scalar under a sqrt (NaN if driven negative) — kept as-is to match the van Spengler reference, with a workaround noted

## [0.10.1] - 2026-06-30

### Added
- **`GyroBatchNorm` and `GyroRMSNorm`** normalization layers for the Hyperboloid and Proper Velocity manifolds

## [0.10.0] - 2026-06-29

### Added
- **LResNet `lorentz_scale` (He et al. 2025, Eq. 10)** — Klein-geodesic output rescaling that slides a hyperboloid point along the geodesic ray from the origin to control output norm while staying on the manifold — and a constrained `LorentzResidual` module built on it
- **Busemann MLR and BFC layers** (Chen et al. 2026)

### Changed
- **FGG layer init defaults changed to norm-preserving.** `FGGLinear` and `FGGConv2D` now default to `reset_params="fan_out"` (Gaussian, `std=sqrt(1/out_spatial)`) with `init_bias=0.0` and a new `gain=1.0` multiplier, so the output spatial norm tracks the input (`‖z‖ ≈ gain·‖x_spatial‖`) instead of growing as `sqrt(out_channels)` from step 0. This is a **deliberate deviation from the Klis et al. 2026 classification reference** (which normalizes magnitude away with BatchNorm/`FGGMeanOnlyBatchNorm` after each layer): it is motivated by *unnormalized* stacks — e.g. an RL backbone feeding a bounded Poincaré-ball projection, where the fan-in default saturated the projection. The new `gain` knob sets the effective column std to `gain/sqrt(out_spatial)`; it is a no-op for `reset_params="eye"` and is renormalized away under `use_weight_norm=True`. Restore the previous reference-style init with `reset_params="eye", init_bias=0.5` (`FGGLinear`) or `reset_params="lorentz_kaiming", init_bias=0.5` (`FGGConv2D`). `FGGLorentzMLR` (a terminal Euclidean-logits head) is intentionally unchanged. **Behavior change** — existing FGG models pick up the new init unless they pass the old `reset_params`/`init_bias` explicitly

## [0.9.0] - 2026-06-23

### Changed
- Performance improvements across the hyperboloid layers
- Unified the patch-extraction path into a shared `hyperboloid_core._extract_patches` helper (deduplication, no behavior change)

## [0.8.2] - 2026-06-12

### Changed
- **Renamed `HypConv2DHyperboloidPP` → `HypConv2DHyperboloidILNN`** — the Shi et al. 2026 Lorentz convolution (LogCat via `log_radius_concat` + PLFC), with origin padding (`pad_mode="origin"`) and ILNN improvements

## [0.8.1] - 2026-06-12

### Changed
- **Renamed `HypLinearHyperboloidPP` → `HypLinearHyperboloidPLFC`** with PLFC (parametrized Lorentz fully-connected) improvements (Shi et al. 2026)

## [0.8.0] - 2026-06-11

### Added
- **Poincaré vector quantization layers** (`hyperbolix.nn_layers`):
    - `HypVQEmbeddingPoincare`: HVQ-VAE quantizer (Chen et al. 2025) — explicit on-ball codebook held as a non-parameter `nnx.Variable` buffer, geodesic nearest-neighbour selection, copy-gradient straight-through estimator, and a hyperbolic-EMA codebook update (GGBall, Bu et al. 2026, Eqs. 41-43) with optional dead-code revival via the `ema_update` method (called after `optimizer.update`; the optimizer never touches the codebook, only the commitment loss trains the encoder)
    - `HypVQMLRPoincare`: HyperVQ quantizer (Goswami et al. 2025) — implicit codebook = the rows of an internal `HypRegressionPoincarePP`, Gumbel-Softmax straight-through selection on the categorical weights (so plain `optax.adam` trains it), with a `deterministic` flag (toggled by `model.eval()` / `model.train()`) for a deterministic argmax MAP estimate at inference
    - `PoincareVQOutput`: shared pytree-friendly `NamedTuple` return type `(quantized, indices, loss, perplexity, z)`; `quantized` is cast to float32 at the manifold→decoder boundary
    - `poincare_weighted_midpoint`: reusable Poincaré weighted gyromidpoint (GGBall Eq. 41), the Poincaré analog of `lorentz_midpoint` and a weighted generalization of `poincare_midpoint`
    - **`squared_commitment` flag on `HypVQEmbeddingPoincare`** (default `False`). The cited references genuinely disagree on the commitment penalty: HVQ-VAE uses the plain geodesic distance `d(z, sg(q))`, GGBall's L_HVQVAE uses `d²` (the Euclidean VQ-VAE convention). The flag selects between them without touching library code
- **`Hyperboloid.log_radius_concat`** — log-radius-preserving concatenation of N Lorentz points (Shi et al. 2026, Sec. 4.3), the hyperboloid analog of Poincaré β-concatenation. Rescales each block's spatial part by `exp(½·(ψ(N·d/2) − ψ(d/2)))` (digamma `ψ`) so the expected log spatial radius stays invariant to the post-concat dimension, then recomputes the time coordinate to keep the result on the manifold. Reduces to `hcat` when `N = 1`; convolutional layers keep using the unscaled `hcat` by default
- **Isometry mappings now span all three models** — Poincaré ↔ Hyperboloid, Poincaré ↔ Proper Velocity, and Hyperboloid ↔ Proper Velocity. New functions: `pv_to_poincare`, `poincare_to_pv`, `pv_to_hyperboloid`, `hyperboloid_to_pv` (in `hyperbolix.manifolds.isometry_mappings`). All six maps are exact, curvature-correct Riemannian isometries (PVNN Eq. 4; Chen et al. 2026). The direct `pv_to_hyperboloid` (PV coords are the hyperboloid's spatial part; time reconstructed from `⟨z,z⟩_L = -1/c`) avoids the near-boundary blow-up of composing through the ball
- Self-hosted MathJax with a CI external-host allowlist guard

### Changed
- **All NN layers now take a `param_dtype` constructor argument (default `jnp.float32`)** controlling the *storage* dtype of every trainable parameter and persistent state (batch-norm statistics, VQ codebooks, scalar scales/temperatures). The manifold's `dtype` controls *compute* precision only. Under global `jax_enable_x64`, parameters previously materialized as float64 (2× parameter/optimizer memory, float64 checkpoints); they now stay float32 unless `param_dtype=jnp.float64` is passed explicitly. The Riemannian optimizers complete the contract: `egrad2rgrad`/`expmap`/`ptransp` run in the manifold dtype, but returned updates and momentum buffers are cast back to the parameter's storage dtype (previously a float32 `ManifoldParam` next to a float64 manifold silently promoted its optimizer state to float64 after one step). See [Numerical Stability — Storage vs. Compute Dtype](user-guide/numerical-stability.md#storage-vs-compute-dtype). **Breaking changes:**
    - `FGGMeanOnlyBatchNorm(dtype=...)` renamed to `param_dtype=...`
    - `PoincareBatchNorm2D` and `HypVQEmbeddingPoincare` state is no longer pinned to `manifold.dtype` — it defaults to float32; pass `param_dtype=manifold.dtype` to restore the old behavior
    - Checkpoints saved under global x64 before this change carry float64 leaves and will dtype-mismatch on restore into float32-storage modules — re-initialize or cast the state tree on load
- **`Hyperboloid.addition` now implements the Lorentz gyrovector addition** `x ⊕ y = Exp_x(PT_{0→x}(Log_0(y)))` (Chen et al. 2025b; Shi et al. 2026, Eq. 1), and `ProductManifold.addition` delegates it per factor. It forms a gyrocommutative gyrogroup (identity = origin, inverse `⊖x = (-1) ⊙ x = [x₀, -x_s]`) and coincides with Poincaré Möbius addition under the stereographic isometry. This replaces the earlier coordinate-wise formula, which was incorrect at *every* curvature (e.g. `origin ⊕ y ≠ y`) and failed silently via a trailing projection. `Hyperboloid.scalar_mul` (geodesic scaling, = Eq. 2) is unchanged. **Breaking change** — `addition` no longer raises `NotImplementedError`
- **Breaking: `VERSION_LORENTZIAN_PROXY` removed from the Poincaré manifold.** The formula treated Poincaré ball coordinate 0 as a Lorentz time component, so it was only meaningful for hyperboloid inputs and returned `dist(x, x) = -2/c ≠ 0` for ball points; the docs additionally mis-advertised it as "best near boundary". The reference implementation never had it. `dist`/`dist_0` now select between 3 versions (`VERSION_MOBIUS_DIRECT`, `VERSION_MOBIUS`, `VERSION_METRIC_TENSOR`). For a true Lorentzian (squared Minkowski) proxy, convert to the hyperboloid via `isometry_mappings.poincare_to_hyperboloid` first

### Fixed
- **Poincaré ↔ Hyperboloid isometry maps were only correct at `c=1`.** `poincare_to_hyperboloid` / `hyperboloid_to_poincare` used unit-ball stereographic formulas missing their `√c` factors, so round-trips and distance preservation silently failed at any other curvature (the recommended default is `c=0.1`). Both are now curvature-correct at all `c`, verified by round-trip, isometry, commutative-diagram, and extreme-curvature tests across dtypes
- **Test fixture `uniform_points` generated hyperboloid points at the wrong geodesic radius for `c ≠ 1`** (an extra `/√c` on the spatial coordinate, masked by a trailing projection). Corrected so generated points are the exact Poincaré images at any curvature
- **Silent float64 promotion under x64 in hyperboloid layers.** Runtime buffers created without an explicit dtype (`FGGConv2D` SAME/origin padding buffer, causal linear-attention scan carries, full-attention uniform head-averaging weights) now derive their dtype from the input, and the scalar attention parameters (`temperature`, `scale`, `attn_bias`) are cast to the compute dtype at use. Previously, with `jax_enable_x64` active, these defaulted to float64 and promoted float32 activations through the rest of the layer

## [0.7.2] - 2026-05-22

### Changed
- **`ProductManifold` curvature API redesigned** — every geometry method (`dist`, `expmap`, `logmap`, `proj`, `origin`, `tangent_inner`, etc.) now takes a positional `c` argument that must be a sequence of length `n_factors` (one curvature per factor) instead of the silently-ignored scalar `c`. There is no default and no scalar broadcast: pass `product.curvatures` for static curvatures, or build the sequence from `LearnableCurvature` calls for trainable ones. The protocol-level `Curvature` type was widened to `ScalarCurvature | Sequence[ScalarCurvature]` so `ProductManifold` satisfies the `Manifold` protocol — `isinstance(product, Manifold)` is `True` and generic code typed against `Manifold` accepts product instances. The product still has no `c` attribute (use `product.curvatures`). **Breaking change** — call sites that passed `c=0.0` (or any scalar) must now pass a per-factor sequence `(c_0, c_1, …)`

### Fixed
- Corrected the `ProductManifold` API examples and the protocol-conformance claim in the documentation

## [0.7.1] - 2026-05-21

### Added
- **`LearnableCurvature(nnx.Module)`** — canonical module for trainable curvature in `hyperbolix.utils.curvature`, also exported at top level (`from hyperbolix import LearnableCurvature`). Introduces the `softplus` (default, `c = softplus(raw)`, bounded gradient via sigmoid, van Spengler 2023 convention) and `log` (`c = exp(raw)`, scale-invariant gradient `dc/draw = c`, MERU convention, preferred for compiled RL loops) reparameterizations. The default clamp is applied to the recovered `c` (not the raw parameter) as a hard stability guard for long compiled training loops — `[0.1, 10.0]` for both schemes; pass `c_min=None, c_max=None` to disable. Updated by any standard Euclidean `nnx.Optimizer` — no Riemannian optimizer needed

### Changed
- **Manifolds are now plain Python classes** (not `nnx.Module`). This structurally prevents shared-manifold bugs in `nnx.scan`/`nnx.fori_loop` — manifolds become static graphdef attributes with no state pytree entries
- **`learnable=True` removed** from manifold constructors. Use the `LearnableCurvature` module instead — assign one instance per distinct curvature on your model. See the [Manifolds User Guide](user-guide/manifolds.md#static-vs-learnable)
- **`ProductManifold`** is now a plain class; `from_signature` accepts 3- and 4-tuple specs only (5-tuple `learnable` override removed)
- **`learnable_curvature()` / `get_curvature()` functional helpers replaced** by the `LearnableCurvature(nnx.Module)` class. The class bundles the raw parameter, reparameterization scheme, and clamp bounds in one object, making accidental init/recovery mismatches structurally impossible. Call sites change from `c = get_curvature(self.c_raw)` to `c = self.curvature()`. **Breaking change** — no deprecation shim

## [0.7.0] - 2026-05-18

### Added
- **Product manifold** (`hyperbolix.manifolds.ProductManifold`) — heterogeneous-curvature composition $M_1 \times M_2 \times \dots \times M_n$ where each factor may be any base manifold with its own curvature (Gu et al. 2019). Points are flat concatenated arrays of shape `(total_dim,)`; geometry methods take a per-factor curvature sequence as positional `c` at call time. Provides Pythagorean L2 geodesic distance plus auxiliary `dist_l1` / `dist_min` / `component_dist` reductions, full per-factor decomposition of `expmap`/`logmap`/`ptransp`/`proj`/`egrad2rgrad`/`tangent_inner`, an `origin(c)` helper, and a `from_signature` factory accepting 3- and 4-tuple specs
- **Learnable curvature via an `nnx.Module`** — first-class trainable curvature assigned per distinct curvature on the model (later canonicalized into `LearnableCurvature` in v0.7.1)

## [0.6.0] - 2026-04-20

### Added
- **Proper Velocity (PV) manifold** (`hyperbolix.manifolds.ProperVelocity`) — unconstrained $\mathbb{R}^n$ model of hyperbolic geometry from Chen et al. (2026), with complete geometric operations: `addition`, `scalar_mul`, `dist`, `expmap`/`logmap` (at origin and arbitrary base points), `ptransp`/`ptransp_0`, `egrad2rgrad`, and Riemannian inner product
- **Proper Velocity neural-network layers**:
    - `HypLinearPV`: PV fully-connected layer (Thm 5.3 / Eq. 22)
    - `HypConv2DPV`: PV 2D convolution with raw Euclidean patch concatenation (Sec 5.3) — no beta-scaling, dimension-preserving
    - `HypRegressionPV`: PV multinomial-logistic-regression head (Thm 5.2 / Eq. 19)

## [0.5.3] - 2026-04-14

### Added
- Hyperbolic average-pooling layer

## [0.5.2] - 2026-04-02

### Added
- `HypLinearHyperboloidFHNN` and `HypConv2DHyperboloidFHNN` layers (fully-hyperbolic neural network, Bdeir et al. 2023)

## [0.5.1] - 2026-03-27

### Added
- **`PoincareBatchNorm2D`** — Poincaré ball BatchNorm between conv layers (van Spengler et al. 2023)

## [0.5.0] - 2026-03-26

### Removed
- `HypRegressionPoincareHDRL` layer and its tests

## [0.4.2] - 2026-03-24

### Changed
- Removed `HypConv3DHyperboloid`; `beta_scale` is now wrapped in `float`

## [0.4.1] - 2026-03-19

### Changed
- Flattened conv-layer parameter paths via forward-function extraction (cleaner param trees, no behavior change)

## [0.4.0] - 2026-03-17

### Added
- MkDocs Material documentation system
- Complete API reference documentation
- Getting Started guide
- CI/CD workflow for documentation builds

### Changed
- Adopted Flax NNX layer naming convention (`kernel`/`bias`) across all layers

## [0.3.0] - 2026-03-10

### Added
- **FGG-LNN layers** (`FGGLinear`, `FGGConv2D`, `FGGLorentzMLR`) — fully-generalized Lorentz layers (Klis et al. 2026)

## [0.2.1] - 2026-03-06

### Changed
- Migrated benchmarks to Grain data loading

## [0.2.0] - 2026-03-05

### Added
- Class-based manifold API with automatic dtype casting (`Poincare`, `Hyperboloid`, `Euclidean`)
- `Manifold` structural protocol for type-safe manifold dispatch
- Poincaré convolution layers
- Positional encoding layers for hyperbolic Transformers:
    - `lorentz_residual`: Lorentzian midpoint-based residual connection
    - `hope`: Hyperbolic Rotary Positional Encoding (functional)
    - `HyperbolicRoPE`: NNX module wrapper for HOPE
    - `HypformerPositionalEncoding`: Learnable positional encoding with HTCLinear
- **Causal attention masking** (`causal=True`) for all three hyperbolic attention variants:
    - `HyperbolicSoftmaxAttention`: lower-triangular `-inf` mask before softmax
    - `HyperbolicFullAttention`: lower-triangular `-inf` mask on Lorentzian similarity scores
    - `HyperbolicLinearAttention`: O(N) cumulative-sum recurrence via `jax.lax.scan` (Katharopoulos et al. 2020), keeping O(N) complexity in causal mode
- Tiny Shakespeare character-level benchmark (`benchmarks/bench_shakespeare_attention.py`) comparing all four model variants (Euclidean + 3 hyperbolic) with causal attention

### Changed
- **Breaking**: Manifold public functions renamed to private (`dist()` → `_dist()`); use class methods instead
- Replaced `with_precision()` wrapper with the `Poincare(dtype=jnp.float64)` pattern
- **Breaking: `HypformerPositionalEncoding.epsilon` is no longer a learnable `nnx.Param`** — it is a fixed, non-negative constructor float (default 1.0), matching the Hypformer reference (which keeps it a plain non-trainable tensor). A learnable epsilon could be driven below -1 by gradient descent, where `x + epsilon * p` leaves the upper hyperboloid sheet and the `abs()` in the `lorentz_residual` normalizer silently masks the violation. `lorentz_residual` now documents the `w_y >= 0` requirement

## [0.1.4] - 2026-02-10

### Added
- Pure JAX implementation of hyperbolic manifolds (Euclidean, Poincaré, Hyperboloid)
- 13+ neural network layers (linear, convolutional, regression)
- Hypformer components: HTC/HRC with curvature-change support
- 4 hyperbolic activation functions (ReLU, Leaky ReLU, Tanh, Swish)
- Riemannian optimizers (RSGD, RAdam) with automatic manifold detection
- Isometry mappings between the Poincaré ball and hyperboloid models
- Wrapped normal distributions for VAEs
- Comprehensive test suite (1,400+ tests)
- CI/CD pipeline with benchmarking
- vmap-native API design

### Changed
- Migrated from PyTorch to pure JAX/Flax NNX
- Unified package structure: `hyperbolix_jax` → `hyperbolix`

### References
- Based on research by Ganea et al. (2018), Bécigneul & Ganea (2019), Bdeir et al. (2023)

[Unreleased]: https://github.com/timoklein/hyperbolix/compare/v1.1.2...HEAD
[1.1.2]: https://github.com/timoklein/hyperbolix/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/timoklein/hyperbolix/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/timoklein/hyperbolix/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/timoklein/hyperbolix/compare/v0.11.1...v1.0.0
[0.11.1]: https://github.com/timoklein/hyperbolix/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/timoklein/hyperbolix/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/timoklein/hyperbolix/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/timoklein/hyperbolix/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/timoklein/hyperbolix/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/timoklein/hyperbolix/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/timoklein/hyperbolix/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/timoklein/hyperbolix/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/timoklein/hyperbolix/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/timoklein/hyperbolix/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/timoklein/hyperbolix/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/timoklein/hyperbolix/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/timoklein/hyperbolix/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/timoklein/hyperbolix/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/timoklein/hyperbolix/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/timoklein/hyperbolix/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/timoklein/hyperbolix/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/timoklein/hyperbolix/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/timoklein/hyperbolix/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/timoklein/hyperbolix/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/timoklein/hyperbolix/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/timoklein/hyperbolix/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/timoklein/hyperbolix/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/timoklein/hyperbolix/releases/tag/v0.1.4
