# Phase 1 – Architecture & Package Layout

## Decisions
- Backend strategy: prefer a separate JAX namespace (e.g., `hyperbolix_jax`) to avoid import ambiguity, keeping Torch modules intact until cleanup.
- API surface: keep function names and signatures for manifolds/optimizers; provide layer equivalents under the new namespace without guaranteeing drop‑in import paths initially.
- Configuration: add an env or config flag (`HYPERBOLIX_BACKEND`) only if a dispatcher is required; otherwise use explicit imports to keep behaviour obvious.

## TODOs
- [ ] Choose and scaffold package layout: `src_jax/{manifolds,nn_layers,optim,utils}` or `src/hyperbolix_jax/...` with `pyproject.toml` package discovery.
- [ ] Define shared numeric/config module: dtype policy (`jax_enable_x64`), default devices, tolerances.
- [ ] Specify manifold interfaces (pure functions + dataclasses) and optim interfaces (Optax transforms) to target in Phase 3/4.
- [ ] Add import shims or factories for discovery (e.g., `from hyperbolix_jax import manifolds as jax_manifolds`).

## Progress
- 2025-09-26: Architecture scaffolding groundwork
  - Package layout: `pyproject.toml` currently discovers packages under `src` and only includes names matching `hyperbolic_test*`, `manifolds*`, and `utils*`. Creating a sibling `src/hyperbolix_jax` tree keeps Torch code untouched but will require widening the `include` list (or switching to an explicit `packages = [...]`) so the new namespace is installable without restructuring Torch modules.
  - Shared numerics: plan to centralise JAX runtime knobs in `hyperbolix_jax/config.py` (or `.../config/numerics.py`) to synchronise `jax.config.update("jax_enable_x64", True/False)`, default device selection, and common tolerances. Existing Torch code hard-codes dtype via class attributes, so the JAX path needs a single source of truth to avoid scattering `jax.numpy` dtype conversions everywhere.
  - Interfaces: existing Torch manifolds expose stateful classes inherited from `torch.nn.Module`, while tests (`tests/test_manifolds.py`) expect methods like `addition`, `scalar_mul`, and `dist` bound to manifold instances. For JAX we will mirror the surface with dataclass-backed containers that wrap pure functions (manifold ops in `jax.numpy`/`jax.scipy`) and produce Optax-friendly transforms for optimizers, keeping function signatures stable for later dispatcher work.
  - Import shims: rather than a global `HYPERBOLIX_BACKEND` toggle, short-term plan is to offer explicit entry points (e.g., `from hyperbolix_jax import manifolds as jax_manifolds`) and optionally a thin `src/hyperbolix/backends.py` helper that exposes both Torch and JAX registries while avoiding accidental cross-imports during the migration window.

## Acceptance Criteria
- Package builds and imports cleanly side-by-side with the existing Torch code.
- Minimal example demonstrates JAX manifold import and a round-trip exp/log on CPU.
