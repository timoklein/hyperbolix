# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference Commands

```bash
# Install
uv sync --locked --dev

# Run all tests (~1,660 tests)
uv run pytest

# Run a single test file
uv run pytest tests/test_manifolds.py -v

# Run a single test function
uv run pytest tests/test_manifolds.py::test_dist -v

# Run a fast subset (dim2 only)
uv run pytest -k "dim2"

# Lint and format
uv run ruff check hyperbolix tests benchmarks
uv run ruff format hyperbolix tests benchmarks

# Type check
uv run pyright hyperbolix

# All pre-commit hooks
uv run pre-commit run --all-files

# Benchmarks
uv run pytest benchmarks/ --benchmark-only

# Docs
uv run mkdocs serve
uv run mkdocs build --strict
```

## Verification

After making changes, run the test files that cover the affected code:
```bash
uv run pytest tests/<relevant_test_file>.py -x -v
```
For example: manifold changes → `test_manifolds.py`, optimizer changes → `test_optimizers.py`, FGG layer changes → `nn_layers/test_hyperboloid_fgg.py`. Use `-k "dim2"` to speed up parametrized tests during iteration.

## Architecture

**hyperbolix** is a pure JAX library for hyperbolic deep learning built on Flax NNX.

### Core design: vmap-native, single-point manifold operations

Manifold methods (`dist`, `expmap`, `logmap`, `proj`, `ptransp`) operate on **single points** `(dim,) -> scalar` or `(dim,) -> (dim,)`. Batching is always explicit via `jax.vmap`. NN layers handle batching internally in their `__call__`.

### Module layout

- **`manifolds/`** — Class-based manifold API: `Poincare`, `Hyperboloid`, `Euclidean`. Each is instantiated with a dtype (`Poincare(dtype=jnp.float64)`). Conforms to `Manifold` protocol in `protocol.py`. `_base.py` has shared `ManifoldBase`.
- **`nn_layers/`** — Flax NNX layers (`nnx.Module`). Two families:
  - *Poincare*: `HypLinearPoincare`, `HypLinearPoincarePP`, `HypConv2DPoincare`, `PoincareBatchNorm2D`, `HypRegressionPoincare` (Ganea et al. 2018, Shimizu et al. 2020, van Spengler et al. 2023)
  - *Hyperboloid*: `FGGLinear`, `FGGConv2D`, `FGGLorentzMLR`, `HTCLinear`, `HypLinearHyperboloidPP`, `LorentzConv2D` (Klis et al. 2026, Shimizu et al. 2020), attention layers, positional encodings, normalization
  - *Hybrid*: `HyperPPFeatureScaling` — Euclidean-space feature scaling applied before `expmap_0` in hybrid networks (RMSNorm + activation + dim scaling + optional learned rescaling)
  - `hyperboloid_core.py` has foundational ops: `hrc()`, `htc()`, `build_spacelike_V()`, `lorentz_midpoint()`
- **`optim/`** — Riemannian optimizers (`riemannian_sgd`, `riemannian_adam`). Uses `ManifoldParam` (subclass of `nnx.Param`) to tag hyperbolic parameters. `_riemannian_base.py` has shared `make_riemannian_optimizer()`.
- **`distributions/`** — Wrapped normal (Poincare/Hyperboloid) and uniform Poincare sampling.
- **`utils/`** — Numerically stable math (`atanh`, `acosh`, `smooth_clamp`) and helpers.

### Key patterns

- **Curvature `c`** is passed dynamically at call time (not stored on layers), enabling learnable curvature via `softplus(param)`.
- **`version_idx`** selects distance/operation variants and must be **static** for JIT (use `functools.partial` or `static_argnums`).
- **`ManifoldParam`** tags params for Riemannian optimization. The optimizer auto-detects these and applies Riemannian gradients + projection; all other `nnx.Param`s get standard Euclidean updates.
- **Layers accept `manifold_module`** (a manifold class instance) — never raw functions.
- **NN layer parameter naming** follows Flax NNX conventions: `kernel` (not `weight`), `bias`.

### Weight initialization

- FGG layers: default `lorentz_kaiming` init with `std = sqrt(1/in_features)` using **ambient** dimensions (not spatial). FGGLinear default `0.5*eye` with bias `0.5`
- FHCNN/HTC layers: small uniform `U(-0.02, 0.02)`
- HyperboloidPP layer: standard normal `std = 1.0` (Shimizu et al. 2020 reference init)
- Poincare layers: scaled normal `std = (2 * in_dim * out_dim)^{-0.5}` (van Spengler et al. 2023)
- Standard inits (He, Xavier) are too large for hyperbolic layers

### Float precision

- Float32 reliable for hyperbolic distances < 7; float64 needed for distances > 10
- Conformal factor lambda grows exponentially near Poincare ball boundary
- Tests parametrize both dtypes with tolerances: `atol=4e-3` (f32), `atol=1e-7` (f64)

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
