# JAX & Flax Migration Plan (Adjusted)

## Objectives
- Replace PyTorch with JAX-based equivalents (JAX, Flax, Optax) across manifolds and optimizers first, then selectively port layers.
- Preserve public API semantics for manifolds and optimizers to ease downstream upgrades; layers can ship as a new namespace.
- Maintain or improve numerical stability and performance with JIT/VMAP while avoiding in‑place mutation.

## Critique Summary
- Tooling step was redundant: dependencies are already in `pyproject.toml`; shift focus to verification and workflows.
- Layer porting was too broad for an initial milestone. Prioritize manifolds and optimizers (those covered by tests) before converting all layers.
- Missing an explicit backend architecture decision (dual‑backend vs. new namespace) and acceptance criteria per phase.
- Visualization/HoroPCA are not required for core functionality; move them to an optional, later phase.

## Adjusted Roadmap
- [Phase 0 – Discovery & Baselines](./jax_migration/phase-0-discovery.md): inventory Torch usage, capture fixtures, define tolerances and acceptance criteria.
- [Phase 1 – Architecture & Package Layout](./jax_migration/phase-1-architecture.md): decide backend strategy (dual backend vs. new package), module layout, config flags, and API surface guarantees.
- [Phase 2 – Tooling & Workflows](./jax_migration/phase-1-toolchain.md): verify deps, uv workflows, formatting/lint integration, platform notes.
- [Phase 3 – Core Geometry Port](./jax_migration/phase-2-geometry.md): port `manifolds` and `utils/math_utils.py` to JAX with parity tests.
- [Phase 4 – Optimizers & Training Loops](./jax_migration/phase-4-optim.md): implement Optax equivalents and minimal training harness.
- [Phase 5 – Testing & Parity](./jax_migration/phase-5-testing.md): dual‑backend tests, gradient checks, performance tracking; gate promotion to default backend.
- [Phase 6 – Flax Layers (Scoped)](./jax_migration/phase-3-layers.md): port high‑value layers only (Poincaré HNN/HNN++ and Hyperboloid linear); defer RL and FHNN variants.
- [Phase 7 – Visualization & HoroPCA (Optional)](./jax_migration/phase-7-visualization.md): revisit plotting and dimensionality reduction after core parity.
- [Phase 8 – Cleanup & Documentation](./jax_migration/phase-6-cleanup.md): remove Torch paths once JAX passes all gates; finalize docs.

## Module Migration Notes
- Manifolds: convert to pure `jax.numpy` with `@jax.jit`/`jax.vmap`; replace `torch.jit.script` utils with JAX‑safe numerics; prefer dataclasses over mutable modules.
- Optimizers: express momentum/Adam state as pytrees; provide both expmap and retraction update modes; expose a small adapter that mirrors the current API names.
- Layers: implement as Flax `nn.Module` under a new namespace (e.g., `hyperbolix_jax.nn_layers`) to avoid import breakage; ship a thin factory for easier discovery.
- Tests: parametrise on backend; reuse saved Torch fixtures and compare with relaxed tolerances where needed.

## Realism & Risks
- Realistic first milestone: Phases 0–5 (manifolds + optimizers + tests) within 1–2 sprints, given focused scope.
- Layers and visualisation add significant effort; scope them after core parity to de‑risk.
- Watch for dtype discrepancies (JAX default float32) and in‑place mutation patterns in the current code (must be rewritten functionally).
