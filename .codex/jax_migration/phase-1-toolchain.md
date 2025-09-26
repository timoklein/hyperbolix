# Phase 1 – Toolchain & Dependencies

## Goals
- Introduce JAX/Flax/Optax dependencies and supporting tooling without breaking existing PyTorch flows.
- Establish uv-based workflows for installing the new stack.

## Actions
- Update `pyproject.toml` optional dependency groups to include `jax`, `jaxlib`, `flax`, `optax`, and `einops` (if needed for tensor reshaping). Keep Torch until the migration concludes.
- Extend uv commands: `uv pip install .[dev,jax]` for local development and document GPU wheel install steps (e.g., `pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`).
- Add `uv run ruff`, `uv run black`, and `uv run isort` tasks to ensure formatting still works on the enriched code base.
- Create helper modules (`utils/jax_helpers.py`) centralizing `jax.numpy as jnp`, PRNG key management, and pytree utilities used across manifolds/layers.

## Deliverables
- Updated dependency metadata and lockfiles.
- Documented installation instructions for CPU and GPU environments.
- New helper scaffolding committed and referenced by future phases.
