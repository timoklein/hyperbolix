# Phase 2 – Core Geometry Port

## Goals
- Migrate manifold primitives from Torch tensors to pure `jax.numpy` functions while preserving numerical stability.
- Maintain API compatibility for downstream modules that rely on manifold operations.

## Actions
- Convert classes in `src/manifolds/` to operate on `jnp.ndarray`. Where state is minimal, prefer `@dataclass` containers with immutable fields over mutable Torch modules.
- Replace Torch ops (`torch.cat`, `torch.linalg`, broadcasting) with their `jnp` counterparts. Validate corner cases such as Minkowski dot products, projections, and exp/log maps.
- Introduce `jax.jit` wrappers for hot paths and use `jax.vmap` to preserve batch semantics that previously relied on implicit broadcasting.
- Implement utility functions for curvature handling, dtype casting, and safe clipping (e.g., `jnp.clip`) to prevent NaNs near the boundary.
- Mirror Torch-specific assertions with `jax.debug.print` or explicit checks guarded under debug flags.

## Deliverables
- JAX-native manifold modules with parity tests comparing against saved Torch baselines.
- Benchmarks (micro-scale) demonstrating no numerical regressions or quantifying acceptable drift.
