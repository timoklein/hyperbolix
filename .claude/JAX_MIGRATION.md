# JAX & Flax Migration Plan (Adjusted)

## Objectives
- Replace PyTorch with JAX-based equivalents (JAX, Flax NNX, Optax) across manifolds and optimizers first, then selectively port layers.
- Preserve public API semantics for manifolds and optimizers to ease downstream upgrades; layers can ship as a new namespace.
- Maintain or improve numerical stability and performance with JIT/VMAP while avoiding in‑place mutation.

## Architecture Decision: Flax NNX
- **Using Flax NNX** (next-generation Flax) instead of Flax Linen for neural network layers
- NNX provides more Pythonic, stateful module design that's closer to PyTorch's nn.Module
- Better integration with JAX transformations and more flexible state management
- Manifolds use Flax struct.dataclass for pure functional operations

## Critique Summary
- **Fixed Critical Issues**: Added Phase 1 for immediate blockers (package discovery configuration in pyproject.toml)
- **Enhanced Structure**: Better phase ordering and numbering consistency; added performance validation and migration utilities phases
- **Missing JAX Configuration**: Added explicit JAX configuration management (jax_enable_x64, device selection)
- **Complexity Assessment**: The ~4,500 LOC codebase requires more systematic analysis of PyTorch-specific features
- **Performance Strategy**: Added dedicated performance validation phase with benchmarking framework
- **Migration Tooling**: Added phase for weight conversion and automated migration utilities
- **Layer porting**: Correctly scoped to high-value layers only after core parity established

## Adjusted Roadmap
- [Phase 0 – Discovery & Baselines](./jax_migration/phase-0-discovery.md): inventory Torch usage, capture fixtures, define tolerances and acceptance criteria.
- [Phase 1 – Critical Configuration Fixes](./jax_migration/phase-1-critical-fixes.md): fix package discovery, JAX configuration management, and immediate blockers.
- [Phase 2 – Architecture & Package Layout](./jax_migration/phase-2-architecture.md): finalize backend strategy, module layout, and API surface guarantees.
- [Phase 3 – Core Geometry Port](./jax_migration/phase-3-geometry.md): port `manifolds` and `utils/math_utils.py` to JAX with parity tests.
- [Phase 4 – Optimizers & Training Loops](./jax_migration/phase-4-optim.md): implement Optax equivalents and minimal training harness.
- [Phase 5 – Testing & Parity](./jax_migration/phase-5-testing.md): dual‑backend tests, gradient checks, performance tracking; gate promotion to default backend.
- [Phase 6 – Performance Validation](./jax_migration/phase-6-performance.md): systematic benchmarking, memory profiling, and optimization.
- [Phase 7 – Flax Layers (Scoped)](./jax_migration/phase-7-layers.md): port high‑value layers only (Poincaré HNN/HNN++ and Hyperboloid linear); defer RL and FHNN variants.
- [Phase 8 – Migration Utilities](./jax_migration/phase-8-migration-utils.md): weight conversion tools and automated migration scripts.
- [Phase 9 – Visualization & HoroPCA (Optional)](./jax_migration/phase-9-visualization.md): revisit plotting and dimensionality reduction after core parity.
- [Phase 10 – Cleanup & Documentation](./jax_migration/phase-10-cleanup.md): remove Torch paths once JAX passes all gates; finalize docs.

## Module Migration Notes
- Manifolds: convert to pure `jax.numpy` with `@jax.jit`/`jax.vmap`; replace `torch.jit.script` utils with JAX‑safe numerics; prefer dataclasses over mutable modules.
- Optimizers: express momentum/Adam state as pytrees; provide both expmap and retraction update modes; expose a small adapter that mirrors the current API names.
- Layers: implement as Flax NNX `nnx.Module` under a new namespace (e.g., `hyperbolix_jax.nn_layers`) to avoid import breakage; ship a thin factory for easier discovery.
- Tests: parametrise on backend; reuse saved Torch fixtures and compare with relaxed tolerances where needed.

## Realism & Risks
- **Realistic first milestone**: Phases 0–6 (manifolds + optimizers + performance validation) within 2–3 sprints, given the 4,500 LOC scope
- **Critical dependencies**: Phase 1 must complete before any other work can proceed (package discovery blocking imports)
- **Complexity factors**: PyTorch-specific features (torch.jit.script, custom autograd) may require significant JAX equivalents
- **Performance risks**: JAX compilation overhead may initially degrade performance; requires systematic optimization
- **Numerical stability**: Watch for dtype discrepancies (JAX default float32), epsilon handling differences, and in‑place mutation patterns that must be rewritten functionally
- **Integration complexity**: Dual-backend testing and weight migration tools add significant overhead but are essential for production deployment
