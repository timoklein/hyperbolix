# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference Commands

```bash
# Install
uv sync --locked --dev

# Run all tests (4,067 items across 962 test functions) on all cores (pytest-xdist; ~11 min on 12 workers,
# hours single-process: the suite is JAX-compile-heavy)
uv run pytest -n auto

# Run a single test file
uv run pytest tests/test_manifolds.py -v

# Run a single test function
uv run pytest tests/test_manifolds.py::test_dist -v

# Run the dim-2 float32 slice of the dim-parametrized suites (78/388 in test_manifolds.py)
# The parametrization ids spell the dimension as a bare number, e.g. [PoincareBall-c1-2-float32-10]
uv run pytest -k "2-float32"

# Lint and format
uv run ruff check hyperbolix tests
uv run ruff format hyperbolix tests

# Type check
uv run pyright hyperbolix

# All pre-commit hooks
uv run pre-commit run --all-files

# Docs
uv run mkdocs serve
uv run mkdocs build --strict
```

## Verification

After making changes, run the test files that cover the affected code:
```bash
uv run pytest tests/<relevant_test_file>.py -x -v
```
Anything larger than one or two files gets `-n auto` (pytest-xdist); never run the full suite single-process.
For example: manifold changes → `test_manifolds.py`, optimizer changes → `test_optimizers.py`, FGG layer changes → `nn_layers/test_hyperboloid_fgg.py`. Use `-k "2-float32"` to speed up dim-parametrized tests during iteration (78 of the 388 tests in `test_manifolds.py`); it selects only ids whose dimension slot is `2` and whose dtype slot is `float32`.

## Architecture

**hyperbolix** is a pure JAX library for hyperbolic deep learning built on Flax NNX.

### Core design: vmap-native, single-point manifold operations

Manifold methods (`dist`, `expmap`, `logmap`, `proj`, `ptransp`) operate on **single points** `(dim,) -> scalar` or `(dim,) -> (dim,)`. Batching is always explicit via `jax.vmap`. NN layers handle batching internally in their `__call__`.

### Module layout

- **`manifolds/`** — Plain Python classes (not `nnx.Module`): `Poincare`, `Hyperboloid`, `Euclidean`, `ProperVelocity`, `ProductManifold`. Each is instantiated with a dtype (`Poincare(dtype=jnp.float64)`). `Poincare`, `Hyperboloid`, `Euclidean`, `ProperVelocity` conform to the scalar-`c` `Manifold` protocol in `protocol.py`; `ProductManifold` intentionally does **not** (it takes a per-factor `cs: Sequence[Curvature]` sequence instead of a scalar `c`, and has no `c` attribute). `_base.py` has shared `ManifoldBase`.
- **`nn_layers/`** — Flax NNX layers (`nnx.Module`). Two families:
  - *Poincare*: `HypLinearPoincare`, `HypLinearPoincarePP`, `HypConv2DPoincare`, `PoincareBatchNorm2D`, `HypRegressionPoincare` (Ganea et al. 2018, Shimizu et al. 2020, van Spengler et al. 2023)
  - *Hyperboloid*: `FGGLinear`, `FGGConv2D`, `FGGLorentzMLR`, `HTCLinear`, `HypLinearHyperboloidPLFC`, `LorentzConv2D` (Klis et al. 2026, Shimizu et al. 2020, Shi et al. 2026), attention layers, positional encodings, normalization
  - *Hybrid*: `HyperPPFeatureScaling` — Euclidean-space feature scaling applied before `expmap_0` in hybrid networks (RMSNorm + activation + dim scaling + optional learned rescaling)
  - `hyperboloid_core.py` has foundational ops: `hrc()`, `htc()`, `build_spacelike_V()`, `lorentz_midpoint()`
- **`optim/`** — Riemannian optimizers (`riemannian_sgd`, `riemannian_adam`). Uses `ManifoldParam` (subclass of `nnx.Param`) to tag hyperbolic parameters. `_riemannian_base.py` has shared `make_riemannian_optimizer()`.
- **`distributions/`** — Wrapped normal (Poincare/Hyperboloid) and uniform Poincare sampling.
- **`utils/`** — Numerically stable math (`atanh`, `acosh`, `smooth_clamp`), the `LearnableCurvature` module for trainable curvature, and other utilities.

### Key patterns

- **Curvature `c`** is passed dynamically at call time (not stored on layers), enabling learnable curvature via the `LearnableCurvature` module from `hyperbolix.utils.curvature`. Manifolds are plain Python classes with static `c`; `LearnableCurvature` lives on the user's `nnx.Module` and is called at runtime to produce the (optionally clamped) curvature value.
- **`version_idx`** selects distance/operation variants and should be **static** for JIT (compile-size; `lax.switch` also accepts a traced index; use `functools.partial` or `static_argnums`).
- **`ManifoldParam`** tags params for Riemannian optimization. The optimizer auto-detects these and applies Riemannian gradients + projection; all other `nnx.Param`s get standard Euclidean updates.
- **Layers accept `manifold_module`** (a manifold class instance) — never raw functions.
- **NN layer parameter naming** follows Flax NNX conventions: `kernel` (not `weight`), `bias`.

### Weight initialization

- FGG layers (`FGGLinear`, `FGGConv2D`): default `reset_params="fan_out"` (Gaussian `std = sqrt(1/out_spatial)`) + `init_bias=0.0` + `gain=1.0` — **norm-preserving** (`‖z‖ ≈ gain·‖x_spatial‖`), a deliberate deviation from the Klis et al. 2026 BatchNorm-regime reference to keep unnormalized stacks off a bounded projection's ceiling. `gain` is a no-op for `"eye"` and renormalized away under `use_weight_norm=True`. Restore the reference init via `reset_params="eye"` (linear) / `"lorentz_kaiming"` (conv) + `init_bias=0.5`. The other `reset_params` schemes (`xavier`, `kaiming`, `lorentz_kaiming`, `mlr`) use fan-in `std` from **ambient** dims. `FGGLorentzMLR` is unchanged (`reset_params="mlr"`, bias `0.5`)
- FHCNN layers: small uniform `U(-0.02, 0.02)` (Chen 2021 / Bdeir 2023 reference). `HTCLinear` (and the attention/positional wrappers forwarding `init_bound=None`): fan-in-aware uniform `U(-√(3/in), √(3/in))` — norm-preserving (per-layer Jacobian gain ≈ 1); the old fixed `0.02` contracted depth-≥2 stacks below the float32 noise floor (frozen training). `init_bound=0.02` restores the old init bit-for-bit
- HypLinearHyperboloidPLFC: small normal `std = 0.02`, gyro-bias zeros (Shi et al. 2026 PLFC reference init); `kernel_init_std=1.0` recovers the old HNN++-style init (Shimizu et al. 2020). It has no LogCat, so it keeps the reference value
- `HypConv2DHyperboloidILNN` (formerly `HypConv2DHyperboloidPP`; LogCat via `log_radius_concat` + PLFC, origin padding): fan-out normal `std = sqrt(1/out_spatial)` via `kernel_init_std=None` — **norm-preserving** (the PLFC chain linearizes to `y_spatial ≈ W @ u_spatial` at the origin and the fixed LogCat hands over the per-pixel spatial radius, so gain ≈ 1; probe-measured per-layer ratio 0.82–0.95 at depth 3). Coupled to the 2026-07-31 LogCat digamma sign fix: the old fixed `0.02` was tuned against the pre-fix ~√N amplification and is strongly contractive (ratio 0.05–0.15/layer → origin collapse) under the corrected shrink. `kernel_init_std=0.02` restores the Shi et al. 2026 regime bit-for-bit, which implicitly assumed the pre-fix amplification
- Poincare linear layers (`HypLinearPoincare`, `HypLinearPoincarePP`): fan-in normal `std = 1/sqrt(in_dim)`; Poincare regression heads (`HypRegressionPoincare`, `HypRegressionPoincarePP`): scaled normal `std = (2 * in_dim * out_dim)^{-0.5}` (van Spengler et al. 2023)
- Standard inits (He, Xavier) are too large for hyperbolic layers

### Float precision

- Float32 reliable for hyperbolic distances < 7; float64 needed for distances > 10
- Conformal factor lambda grows exponentially near Poincare ball boundary
- Tests parametrize both dtypes with tolerances: `atol=4e-3` (f32), `atol=1e-7` (f64)
- **Loud divergence over silent saturation.** A guard that maps an already non-finite input (an `inf` time coordinate, a point past the float32 manifold) onto a finite, plausible output hides the divergence; a NaN loss is the intended signal. Do not add clamps or saturations whose only remaining job is finiteness on inputs that are already non-finite, and propose removing such guards when found (the MLR `asinh` clamp inherited from the PyTorch code was removed for this reason). Guards that fix real float32 rounding on finite inputs (`floor_at` on divisors, `safe_sqrt` at zero) are a different matter and stay

### Test structure

- `tests/conftest.py`: global fixtures — `seed_jax` (enables float64), `rng`, `dtype`, `tolerance`, `manifold_and_c`, `uniform_points`
- Tests parametrized across seeds (10-12), dtypes (f32/f64), dims (2,5,10,15), manifolds with random curvatures
- CI runs 15 test suites in parallel via matrix strategy

## Dimension naming convention

All tensor variables use capital letter suffixes: `logits_BLV`, `hidden_BD`, `x_BI`. Define a dimension key at the top of each file.

## Flax NNX API (v0.12.0+)

- `nnx.Optimizer` requires `wrt=nnx.Param`
- `optimizer.update()` takes `(model, grads)` not just `(grads)`
- Lists of modules must use `nnx.List([...])` not plain Python `list`

## Pre-commit hooks

Ruff auto-formats on commit (e.g., `x ** 2` -> `x**2`). When editing files after creation, match the reformatted content. Unused imports are auto-removed — always add imports and their usage in the same edit.
