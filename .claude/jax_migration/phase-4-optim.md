# Phase 4 – Optimization & Training Loops

## TODOs
- [ ] Recreate `RiemannianSGD` and `RiemannianAdam` as Optax gradient transformations:
  - [ ] Implement manifold-aware gradient projections using Phase 2 functions (`egrad2rgrad`, parallel transport).
  - [ ] Translate momentum/Adam state into pytree structures updated via `optax.apply_updates`.
- [ ] Build end-to-end training loops that:
  - [ ] Initialise Flax parameters, maintain PRNG keys, and compute losses with `jax.value_and_grad`.
  - [ ] Demonstrate both retraction-based and expmap-based update modes controlled by configuration.
- [ ] Provide API shims so callers can request `optim.get_optimizer("riemannian_adam", backend="jax")` during transition.
- [ ] Add regression tests mirroring `tests/test_optimizers.py`, verifying convergence under both expmap/retraction settings.
- [ ] Benchmark optimisers on representative tasks (small synthetic datasets) to confirm learning curves match Torch baselines.

## Notes
- Investigate whether Optax’s `scale_by_adam` and `scale_by_schedule` primitives can be reused; extend with custom curvature-aware logic as needed.
- Capture best practices for checkpointing Optax state and PRNG keys to ensure reproducible training runs.

## Acceptance Criteria
- JAX versions of the optimizers converge on the same targets within test tolerances for both update modes.
- Optimizer APIs are callable via a thin adapter without changing downstream training scripts materially.
