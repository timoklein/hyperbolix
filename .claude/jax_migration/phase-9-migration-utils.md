# Phase 8 – Migration Utilities

## TODOs
- [ ] **Weight Conversion Tools**:
  - [ ] Create utilities to convert PyTorch model weights to JAX/Flax format
  - [ ] Handle PyTorch `state_dict` to Flax parameter tree conversion
  - [ ] Support for optimizer state migration (SGD/Adam momentum, etc.)
  - [ ] Validation tools to ensure converted weights produce equivalent outputs
- [ ] **Automated Migration Scripts**:
  - [ ] Script to identify PyTorch-specific code patterns that need JAX equivalents
  - [ ] Automated replacement of common PyTorch operations with JAX equivalents
  - [ ] Analysis tool for torch.jit.script usage and suggested JAX replacements
  - [ ] Code pattern scanner for in-place operations requiring functional rewrites
- [ ] **API Compatibility Helpers**:
  - [ ] Shim layer for gradual migration from PyTorch to JAX APIs
  - [ ] Backend detection and switching utilities
  - [ ] Configuration migration assistant for hyperparameter tuning
  - [ ] Documentation generator for API differences and migration steps

## Deliverables
- `scripts/convert_pytorch_weights.py` - Weight conversion utility
- `scripts/analyze_torch_usage.py` - Code analysis tool
- `hyperbolix_jax/migration/` - Migration helper module
- Migration documentation with step-by-step guides

## Acceptance Criteria
- Existing PyTorch models can be converted to JAX with preserved functionality
- Automated tools reduce manual migration effort by >70%
- Clear migration path documented for downstream users
- Validation tools ensure conversion accuracy

## Estimated Effort
- **2-3 days** - Critical for smooth user migration experience