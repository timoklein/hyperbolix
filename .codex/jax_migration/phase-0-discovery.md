# Phase 0 – Discovery & Baselines

## Goals
- Map every PyTorch dependency, including tensor ops, autograd usage, and module inheritance.
- Capture behavioral baselines to use as regression targets during the migration.

## Actions
- Audit `src/manifolds`, `src/nn_layers`, `src/optim`, and `src/utils` for `torch` imports and in-place mutations. Document findings in a shared tracker (e.g., `.codex/jax_migration/todo.csv`).
- Record representative tensors and model weights that stress hyperbolic operations (e.g., large curvature values, boundary points) for later parity tests.
- Freeze the current test behavior by running `uv run pytest -ra -q` and persisting logs/artifacts for comparison.
- Note any global dtype configuration (`torch.set_default_dtype`) to inform `jax.config.update("jax_enable_x64", True)` choices.

## Deliverables
- Inventory doc of Torch touchpoints.
- Saved fixtures/checkpoints for regression testing.
- Baseline test log establishing expected outputs and tolerances.
