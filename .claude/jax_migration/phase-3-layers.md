# Phase 6 – Flax Layers (Scoped)

## TODOs (MVP first)
- [ ] Implement Flax equivalents for high‑value layers only:
  - [ ] `HyperbolicLinearPoincare` and `HyperbolicLinearPoincarePP`.
  - [ ] `HyperbolicLinearHyperboloid` (baseline variant).
- [ ] Port `src/nn_layers/helpers.py` to JAX, rewriting multinomial regression kernels with `jnp`.
- [ ] Provide `setup`/`__call__` logic performing exp/log maps via Phase 2 utilities.
- [ ] Add forward‑pass equivalence tests for the MVP layers.

## Stretch Goals (defer if not needed)
- [ ] FHNN/FHCNN variants and RL‑specific layers.
- [ ] Transitional factories and higher‑level builders.

## Notes
- Evaluate whether dropout/activation behaviours should leverage `flax.linen.Dropout` and `jax.nn` to stay idiomatic.
- For modules relying on `torch.nn.Parameter` (e.g. manifold biases), migrate to explicit parameter trees and ensure they participate in gradient updates.

## Acceptance Criteria
- MVP layer set compiles and runs with Flax; unit tests demonstrate parity with Torch within tolerances.
