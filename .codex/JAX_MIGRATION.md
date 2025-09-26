# JAX & Flax Migration Plan

## Objectives
- Replace PyTorch with JAX-based equivalents (JAX, Flax, Optax) across manifolds, neural layers, and optimizers.
- Preserve the public API and hyperbolic math semantics so downstream users can upgrade incrementally.
- Maintain or improve numerical stability and performance while introducing JIT-compiled workflows.

## Roadmap Overview
- [Phase 0 – Discovery & Baselines](./jax_migration/phase-0-discovery.md): inventory Torch dependencies and capture regression fixtures.
- [Phase 1 – Toolchain & Dependencies](./jax_migration/phase-1-toolchain.md): introduce JAX/Flax/Optax deps and uv workflows.
- [Phase 2 – Core Geometry Port](./jax_migration/phase-2-geometry.md): convert manifold primitives to `jax.numpy`.
- [Phase 3 – Neural Layers](./jax_migration/phase-3-layers.md): rebuild layers with Flax `nn.Module`s and transitional factories.
- [Phase 4 – Optimization & Training Loops](./jax_migration/phase-4-optim.md): port optimizers to Optax and refactor training scripts.
- [Phase 5 – Testing & Validation](./jax_migration/phase-5-testing.md): ensure dual-backend parity and strengthen regression coverage.
- [Phase 6 – Cleanup & Documentation](./jax_migration/phase-6-cleanup.md): remove Torch, finalize docs, and prepare releases.

## Cross-Cutting Practices
- Enable double precision in JAX (`jax.config.update("jax_enable_x64", True)`) to match current Torch defaults unless profiling shows an alternative is viable.
- Favor stateless, functional design to leverage JAX JIT/vmap effectively; avoid in-place mutations and global state.
- Document temporary compatibility shims and deprecation timelines so contributors understand when Torch support will be removed.

## Open Questions & Risks
- GPU availability: confirm target environments support required `jaxlib` wheels (CUDA version, driver constraints).
- Performance: benchmark hyperbolic operations under JIT to catch regressions; profile with `jax.profiler` or alternative tooling.
- Custom kernels: identify any future need for specialized ops that may require XLA custom calls or third-party libraries.
