# Phase 6 – Flax NNX Layers (Scoped)

## Architecture Decision: Flax NNX
- **Using Flax NNX** instead of Flax Linen for more Pythonic, stateful modules
- NNX modules are closer to PyTorch's `nn.Module` design philosophy
- Better state management and integration with JAX transformations
- Explicit parameter handling with `nnx.Param` and state variables

## TODOs (MVP first)
- [ ] Implement Flax NNX equivalents for high‑value layers only:
  - [ ] `HyperbolicLinearPoincare` and `HyperbolicLinearPoincarePP`.
  - [ ] `HyperbolicLinearHyperboloid` (baseline variant).
- [ ] Port `src/nn_layers/helpers.py` to JAX, rewriting multinomial regression kernels with `jnp`.
- [ ] Implement NNX modules with `__init__`/`__call__` logic performing exp/log maps via manifold utilities.
- [ ] Add forward‑pass equivalence tests for the MVP layers.

## Stretch Goals (defer if not needed)
- [ ] FHNN/FHCNN variants and RL‑specific layers.
- [ ] Transitional factories and higher‑level builders.

## Notes
- Use `flax.nnx.Module` as base class instead of `flax.linen.Module`
- Leverage `nnx.Param` for parameters and `nnx.Variable` for state management
- Dropout/activation behaviors should use `jax.nn` functions directly in NNX context
- Parameters stored as `nnx.Param` automatically participate in gradient updates

## Acceptance Criteria
- MVP layer set compiles and runs with Flax NNX; unit tests demonstrate parity with Torch within tolerances.
