# Phase 0 – Discovery & Baselines

## TODOs
- [x] Catalogue every `torch` import and `torch.nn.Module` subclass across
  `src/manifolds`, `src/nn_layers`, `src/optim`, and `src/utils`; capture the findings in `.codex/codebase.md` or a new tracker so JAX work items stay traceable.
- [x] Enumerate tensor dtype/default settings (e.g. any `torch.set_default_dtype` usage) and note where mixed precision is assumed.
- [x] Snapshot current behaviour:
  - [x] Run `uv run pytest -ra -q` and archive the log for regression comparison.
  - [x] Export representative parameters/activations from `tests/test_manifolds.py` and `tests/test_optimizers.py` (e.g. via `torch.save`) for parity tests.
- [x] Record runtime characteristics for key kernels (exp/log maps, projections, optim steps) using small benchmark scripts to gauge post-JAX performance.

## Progress
- 2025-09-26: Initial Torch surface mapping
  - `src/manifolds`: `euclidean.py`, `manifold.py`, `hyperboloid.py`, and `poincare.py` all import top-level `torch`. `Manifold` and both curved manifolds subclass `torch.nn.Module`, and each manifold constructor homogenises inputs via a per-instance `dtype` attribute instead of global defaults. The curved manifolds also rely on `torch.finfo` checks against curvature tensors when emitting precision warnings.
  - `src/nn_layers`: Every layer module starts with `import torch`. Identified subclasses include `HyperbolicLinear{Poincare,Hyperboloid}[PP/FHNN/FHCNN]`, regression heads, the Riemannian RL layer, and the suite of helper layers (`Expmap`, `Logmap`, projections, activations). Supporting utilities in `nn_layers/helpers.py` expose `get_torch_dtype` to convert strings into concrete `torch.dtype` objects and centralise precision guardrails (e.g. `torch.finfo`, `smooth_clamp`).
  - `src/optim`: Both `riemannian_sgd.py` and `riemannian_adam.py` import `torch` for tensor math but do not define `torch.nn.Module` subclasses.
  - `src/utils`: `helpers.py`, `math_utils.py`, `vis_utils.py`, and `horo_pca.py` depend on `torch` (including `torch.linalg`, `torch.nn`) for numerical routines; no module subclasses appear here.
- Dtype sweep: No instances of `torch.set_default_dtype` or related global default overrides. Precision is governed by manifold-specific `dtype` arguments and by layer parameters (`params_dtype`) that resolve through `get_torch_dtype`, with warnings triggered when manifold and parameter precisions diverge.

## Notes
- Use a shared spreadsheet or markdown table in `.codex/jax_migration/torch-inventory.md` to log discoveries (file, API usage, migration complexity).
- Capture any third-party Torch dependencies (e.g. GeoOpt patterns referenced in docstrings) that might require JAX analogues.
