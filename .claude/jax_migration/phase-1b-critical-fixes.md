# Phase 1 – Critical Configuration Fixes

## TODOs
- [ ] **URGENT**: Fix `pyproject.toml` package discovery to include `hyperbolix_jax*`
  - [ ] Update line 71: `include = ["hyperbolic_test*", "manifolds*", "utils*", "hyperbolix_jax*"]`
  - [ ] Or switch to explicit packages list: `packages = ["manifolds", "utils", "nn_layers", "optim", "hyperbolix_jax"]`
  - [ ] Verify with `uv run python -c "import hyperbolix_jax; print('OK')"`
- [ ] **JAX Configuration Management**:
  - [ ] Create `hyperbolix_jax/config.py` with `jax_enable_x64` policy
  - [ ] Add device selection utilities (CPU/GPU/TPU detection)
  - [ ] Define shared numerical tolerances and epsilon handling
  - [ ] Add configuration for JIT compilation settings
- [ ] **Dependency Verification**:
  - [ ] Ensure JAX dependencies (flax, optax) are correctly installed
  - [ ] Verify JAX version compatibility with PyTorch (no conflicts)
  - [ ] Test import paths work correctly in development and installed modes

## Progress
- 2025-09-29: **BLOCKER IDENTIFIED**: Current `pyproject.toml` package discovery prevents `hyperbolix_jax` imports
  - The existing JAX scaffolding in `src/hyperbolix_jax/` cannot be imported due to package discovery config
  - This blocks all subsequent development phases
  - Must be fixed immediately before any other work proceeds

## Acceptance Criteria
- `import hyperbolix_jax.manifolds` works without errors
- JAX configuration can be set globally via `hyperbolix_jax.config`
- All dependencies install and import cleanly
- Basic JAX compilation works (test with simple jit function)

## Estimated Effort
- **0.5 days** - Critical path blocker that must be resolved first