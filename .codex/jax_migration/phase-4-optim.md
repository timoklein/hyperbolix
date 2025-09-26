# Phase 4 – Optimization & Training Loops

## Goals
- Replace Torch optimizers and autograd logic with Optax and native JAX differentiation.
- Demonstrate end-to-end training compatibility using the new manifold and layer stacks.

## Actions
- Port `src/optim/` modules to pure functions returning Optax-style gradient transformations. Where custom Riemannian adjustments exist, implement them as Optax wrappers or custom `optax.GradientTransformation` objects.
- Substitute `torch.autograd` calls with `jax.grad`, `jax.jacfwd`, or `jax.value_and_grad` depending on need. Ensure gradient computations remain side-effect free.
- Refactor training scripts to follow the pattern: initialize parameters via Flax, compute loss with `jax.jit`'d functions, update optimizer state functionally.
- Audit for in-place parameter updates and replace with pytree manipulation (`optax.apply_updates`, `jax.tree_util.tree_map`).
- Provide utilities for mixed precision or custom dtype handling if required by hyperbolic math.

## Deliverables
- Optax-backed optimizers mirroring previous Torch behavior.
- Example training loop notebook/script showcasing initialization, forward pass, loss, backward, and parameter update using JAX primitives.
