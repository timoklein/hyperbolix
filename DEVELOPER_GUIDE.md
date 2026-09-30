# Developer Guide

## Setup

```bash
git clone https://github.com/timoklein/hyperbolix.git
cd hyperbolix
uv sync --locked --dev
uv run pre-commit install
```

Source is in `hyperbolix/` (`manifolds/`, `nn_layers/`, `optim/`,
`decomposition/`, `distributions/`, `utils/`), tests in `tests/`, docs in `docs/`.
Update dependencies with `uv lock --upgrade` (or `--upgrade-package jax`), then
`uv sync --locked --dev`.

`main` is the released code; release tags follow `vMAJOR.MINOR.PATCH`. Branch
as `feat/<short-name>` (e.g. `feat/klein-manifold`) or `fix/<short-name>` (e.g.
`fix/float32-stability`). Contributions are welcome as pull requests with tests.

## Checks & tests

Pre-commit hooks (`.pre-commit-config.yaml`) run on staged files:

- isort, Ruff lint (`--fix`) and Ruff format
- Pyright on `hyperbolix/`
- trailing whitespace, end-of-file, YAML/TOML syntax, merge-conflict markers, large files (>500 KB), debug statements

If a hook fixes files, re-stage and commit again. Don't use `--no-verify`; fix
the underlying problem.

```bash
uv run pre-commit run --all-files   # all hooks on all files (--verbose to debug)
uv run ruff check hyperbolix tests
uv run ruff format hyperbolix tests
uv run pyright hyperbolix           # or one file, e.g. hyperbolix/manifolds/poincare.py; --watch to re-check
```

Pyright is configured in `pyproject.toml` (`[tool.pyright]`, `typeCheckingMode = "basic"`).

```bash
# All tests, on all cores (~9 min on 48 workers; hours single-process)
uv run pytest -n auto

# A directory, a file, one test
uv run pytest tests/nn_layers/
uv run pytest tests/nn_layers/test_hyperboloid_fgg.py
uv run pytest tests/test_manifolds.py::test_dist_properties -v

# Fast slice: dimension 2, float32 only (ids spell the dimension as a bare
# number, e.g. [PoincareBall-c1-2-float32-10]; 134 of 650 tests in test_manifolds.py)
uv run pytest -k "2-float32"
```

Out of memory: use fewer xdist workers (`-n 4`) or the `-k "2-float32"` slice.
Passes locally but fails in CI: check `uv lock --check` and `.python-version`
(the Python version CI uses).

## CI

CI (`.github/workflows/ci.yaml`) runs on every push: Lint (Ruff lint and format
check), Type Check (Pyright) and Test (pytest, parallelized across test suites).
All must pass before merging. The docs build
(`.github/workflows/docs.yml`) runs on pushes to `main` and on pull requests to
`main`, and deploys the site from `main`.

Caching is off: `uv sync --locked --dev` takes about 7 s with or without a cache
(see the comment in `ci.yaml`).

## Conventions

Tensor/array local variables carry **shape suffixes**, one capital letter per
dimension. The dimension key:

| Letter | Dimension |
|--------|-----------|
| `B` | batch size |
| `D` | spatial / manifold dimension (`dim`) |
| `A` | ambient dimension (`dim+1`, hyperboloid time+space) |
| `H` | output height |
| `W` | output width |
| `C` | channels |
| `K` | kernel elements (`kh×kw`) |
| `N` | number of points |
| `P` | number of hyperplanes / output classes (MLR `out_dim`) |
| `S` | sequence length |
| `F` | frequency dim (`d//2` in RoPE) |

```python
# poincare_regression.py — MLR forward pass
sub_PBD = addition_fn(p_neg_PD, x, c)            # (P, B, D) from broadcasting
sub_BPD = jnp.transpose(sub_PBD, (1, 0, 2))      # reorder to (B, P, D)

# poincare.py — MLR logits
res_BP = 2 * z_norm_P1.T * signed_dist2hyp_BP  # z_norm.T broadcasts (1, P) over (B, P)
```

- Flattened dims get compound suffixes (`x_flat_NC`) and a comment explaining the merge
- `_B1` for `keepdims=True` results; `_P1.T` or `_1P` for transposed broadcasts
- Each file that uses shape suffixes starts with a `Dimension key:` docstring listing its letters

The numerics and weight-initialization rules are in `CLAUDE.md` at the
repository root.

## Extending Hyperbolix

This section covers the three most common contributor tasks: adding a manifold,
adding a neural network layer, and contributing documentation.

### Adding a New Manifold

1. **Implement the `Manifold` protocol** (`hyperbolix/manifolds/protocol.py`)
   on a new class. Required methods, all single-point `(dim,) → scalar` or
   `(dim,) → (dim,)`:
    - `proj`, `dist`, `dist_0`
    - `expmap`, `expmap_0`, `logmap`, `logmap_0`
    - `addition`, `scalar_mul`, `retraction`
    - `ptransp`, `ptransp_0`
    - `tangent_inner`, `tangent_norm`, `tangent_proj`
    - `egrad2rgrad`, `is_in_manifold`, `is_in_tangent_space`
    - a `dtype` attribute and `_cast` (inherit from `ManifoldBase` to get
      these, the `c` property and the dtype casting)
2. **Add to exports**: `hyperbolix/manifolds/__init__.py`.
3. **Wire into tests**: extend `manifold_and_c` in `tests/conftest.py` so
   your manifold gets exercised by the shared test suite (parametrized over
   seeds, dtypes, dims, and random curvatures).
4. **Add the manifold-specific test file** under `tests/` covering operations
   not covered by the shared suite (e.g. `tests/test_my_manifold.py`).
5. **Document it**:
    - API reference: add a `:::` autoreference block in
      `docs/api-reference/manifolds.md`.
    - User guide: add a row to the "Choosing a Manifold" decision table in
      `docs/user-guide/manifolds.md` and to the convention cheat-sheet.

Manifolds themselves stay plain, immutable Python classes with a static `c`
— don't add a `learnable=True` constructor flag or an `_c_raw` param to the
manifold. Learnable curvature is a separate concern, handled by wrapping the
manifold's `c` with a `hyperbolix.utils.curvature.LearnableCurvature`
instance (an `nnx.Module`) on the *caller's* model and passing its output as
`c` at call time; no changes to the manifold class are needed to support it.

### Adding a New NN Layer

1. **Pick the channel convention** matching the layer family (Hyperboloid
   layers take ambient `d+1`; Poincaré / PV / normalization layers take
   spatial `d`). See [Manifolds Guide §
   Convention Cheat-Sheet](https://timoklein.github.io/hyperbolix/latest/user-guide/manifolds/#convention-cheat-sheet).
2. **Use the family's init scale.** Standard He/Xavier is too large for
   hyperbolic layers; reuse the convention of the closest family (fan-in-aware
   uniform for HTC, norm-preserving fan-out normal for FGG — `reset_params=
   "eye"`/`"lorentz_kaiming"` restores the older BatchNorm-regime reference —
   fan-in-scaled normal for Poincaré PP, etc.). See
   `docs/user-guide/nn-layers.md#initialization-scales`.
3. **Layer constructor conventions**:
    - Accept `manifold_module` (a `Manifold`-protocol instance), not raw
      functions
    - Accept `rngs: nnx.Rngs` keyword-only
    - Name trainable params `kernel` and `bias`
    - Accept `c` (or `c_in` / `c_out`) at **call time**, not in `__init__`
4. **Add tests** under `tests/nn_layers/test_<your_layer>.py`, parametrized
   over the standard fixtures (`seed_jax`, `dtype`, `manifold_and_c`).
   Include at minimum: forward shape, JIT compatibility, gradient finite-ness,
   and an init-distribution sanity check.
5. **Export** from `hyperbolix/nn_layers/__init__.py` and add to its `__all__`.
6. **Document it**:
    - API reference: add a `:::` autoreference to the relevant page under
      `docs/api-reference/nn-layers/` (e.g. `linear.md`, `convolutional.md`,
      `regression.md`) — that's a directory of per-family pages, not a
      single file.
    - User guide: add a row to the relevant decision table in
      `docs/user-guide/nn-layers.md`.

## Docs workflow

The docs site is built with MkDocs + Material + mkdocstrings.

```bash
# Once per checkout: vendor MathJax (gitignored)
uv run python scripts/vendor_mathjax.py

# Live-reload local preview
uv run mkdocs serve

# Strict build (matches CI; fails on broken cross-links and warnings)
uv run mkdocs build --strict
```

Where content goes:

- **`docs/user-guide/`**: decision tables, conventions, composition patterns,
  pitfalls; not per-symbol reference.
- **`docs/api-reference/`**: `:::` autoreference blocks, plus a one-paragraph
  intro per new module.
- **`docs/getting-started.md` / `docs/index.md`**: only user-facing features
  (a new manifold counts; an internal refactor doesn't).
- **`docs/changelog.md`**: every feature, breaking change or notable bug fix
  gets an `[Unreleased]` entry under `### Added` / `### Changed` / `### Fixed`.

Run the strict build before pushing docs changes: CI uses the same flag.
