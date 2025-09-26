# Phase 5 – Testing & Validation

## Goals
- Ensure migrated JAX components match Torch behavior within acceptable tolerances.
- Maintain comprehensive automated tests across both backends during the transition.

## Actions
- Parametrize existing pytest suites with a `backend` fixture to run Torch and JAX implementations side by side while both exist.
- Create numerical parity tests using saved fixtures from Phase 0. Compare outputs with `jnp.allclose` or `numpy.testing.assert_allclose`, tuning tolerances for mixed precision.
- Add gradient checking tests leveraging `jax.test_util.check_grads` or finite-difference comparisons where analytical gradients are complex.
- Integrate tests into CI, ensuring `uv run pytest` exercises both backends. Capture and review performance metrics (runtime, memory) for regressions.
- Build property-based tests (via `hypothesis` or custom sampling) for manifold invariants to catch subtle numerical issues introduced by the port.

## Deliverables
- Dual-backend test suite with high confidence in functional parity.
- Regression reports/documents summarizing discrepancies and their resolutions.
