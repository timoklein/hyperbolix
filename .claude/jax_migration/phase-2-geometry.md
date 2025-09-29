# Phase 2 – Core Geometry Port

## TODOs
- [ ] Recreate `src/manifolds/manifold.py` as a JAX‑friendly abstraction:
  - [ ] Use dataclasses + pure functions; register pytrees where state is necessary.
  - [ ] Define a minimal, test‑driven interface (expmap, logmap, proj, egrad2rgrad, tangent_inner, dist, dist_0).
- [ ] Port `Euclidean`, `Hyperboloid`, and `PoincareBall` implementations:
  - [ ] Replace `torch` ops with `jax.numpy` (`jnp.sum`, `jnp.concatenate`), avoid mutation.
  - [ ] Convert Minkowski/conformal helpers into `jit`/`vmap`‑friendly functions.
  - [ ] Ensure numerical guards (smooth clamps, eps) use `jnp.finfo` and are dtype‑aware.
- [ ] Translate `src/utils/math_utils.py` to JAX primitives with stable hyperbolic functions.
- [ ] Implement dtype policy (float64 toggle) so tests pass under both precisions.
- [ ] Add parity tests vs Torch fixtures for all core ops; document acceptable tolerances.

## Progress
- 2025-09-26: Test scaffolding and backend stubs
  - Introduced `src/hyperbolix_jax/` with placeholder `Manifold`, `Euclidean`, `Hyperboloid`, and `PoincareBall` classes that mirror the Torch method surface but currently raise `NotImplementedError`. This gives the JAX test suite concrete symbols to target while we port real kernels.
  - Ported an initial slice of Torch manifold tests into `tests/test_manifolds_jax.py`, swapping `torch` utilities for `jax.numpy`/`numpy` equivalents and parameterising dtype coverage (`float32`, `float64`). Hyperboloid-specific fixtures remain skipped until we have JAX conversions for ball↔hyperboloid transfer.
  - Verified the new suite with `uv run pytest tests/test_manifolds_jax.py -q` (passes with expected skips). Noted repeated warnings about truncated `float64` dtypes because `jax_enable_x64` is unset; capturing this feeds into the Phase 1 config work for a shared dtype policy.

## Notes
- Introduce a `manifolds/common.py` for shared JAX math to avoid duplication.
- Profile `jit` compilation times; batch with `vmap` where broadcasting was relied upon in Torch code.
 
## Acceptance Criteria
- `tests/test_manifolds.py` passes for the JAX backend with backend‑specific tolerances.
- Parity checks against saved Torch outputs succeed for representative inputs.
