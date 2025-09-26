# Phase 3 – Neural Layers

## Goals
- Rebuild neural network layers on top of Flax while maintaining functionality equivalent to the Torch implementations.
- Provide a transitional API enabling side-by-side evaluation of Torch and Flax layers.

## Actions
- Translate classes in `src/nn_layers/` into Flax `nn.Module`s; parameter trees become `FrozenDict` structures managed by Flax.
- Encapsulate manifold-aware math by reusing Phase 2 helpers. Ensure activations and projections run inside `@nn.compact` methods without side effects.
- Offer optional factory functions (`build_layer(backend="jax"|"torch")`) so callers can choose implementations during the migration window.
- Re-implement helper utilities (dtype selection, initializer logic) to use JAX random keys and pure functions. Avoid global state by threading PRNG keys.
- Write example initialization and forward-pass snippets demonstrating how to call the new layers via `flax.linen` APIs.

## Deliverables
- Flax versions of all hyperbolic layers with unit tests referencing both backends.
- Interim compatibility wrappers scheduled for removal in Phase 6.
