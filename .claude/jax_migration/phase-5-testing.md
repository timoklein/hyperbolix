# Phase 5 – Testing & Validation

## TODOs
- [ ] Extend pytest fixtures to emit backend identifiers and JAX PRNG keys (e.g. `@pytest.mark.parametrize("backend", ["torch", "jax"])`).
- [ ] Duplicate manifold tests to assert Torch vs JAX parity using stored fixtures; utilise `numpy.testing.assert_allclose` with backend-specific tolerances.
- [ ] Create gradient checks leveraging `jax.test_util.check_grads` for exp/log maps and regression layers.
- [ ] Integrate XLA compilation warnings into test logs to catch non-jittable code early.
- [ ] Update CI pipelines to run both Torch and JAX suites in parallel until cleanup.
- [ ] Track performance metrics (execution time, peak memory) in test reports to monitor regressions.

## Notes
- Consider adding property-based tests (Hypothesis) for manifold invariants once JAX ports stabilise.
- Provide scripts to regenerate Torch baseline artefacts if APIs change mid-migration.
