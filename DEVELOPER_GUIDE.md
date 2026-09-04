# Developer Guide

Quick reference for development workflows and tooling.

## Initial Setup

```bash
# Clone and install
git clone <repo>
cd hyperbolix
uv sync --locked --dev

# Install pre-commit hooks
uv run pre-commit install
```

## Development Workflow

### Before Committing

Pre-commit hooks will automatically run on staged files:

- Ruff linting and formatting
- Trailing whitespace removal
- YAML/TOML validation
- Large file checks

To run manually on all files:

```bash
uv run pre-commit run --all-files
```

### Code Quality Checks

```bash
# Lint with Ruff
uv run ruff check hyperbolix tests

# Format with Ruff
uv run ruff format hyperbolix tests

# Type check with Pyright
uv run pyright hyperbolix
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific test suite
uv run pytest tests/test_manifolds.py
uv run pytest tests/nn_layers/                          # all NN layer tests
uv run pytest tests/nn_layers/test_hyperboloid_fgg.py   # one file

# Fast slice of the dim-parametrized suites: dimension 2, float32 only
# (the ids spell the dimension as a bare number, e.g. [PoincareBall-c1-2-float32-10],
#  so "2-float32" is the selector; 78 of the 386 tests in tests/test_manifolds.py)
uv run pytest -k "2-float32"

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x
```

## CI/CD Pipeline

The CI pipeline runs automatically on push and pull requests:

### Jobs

1. **Lint** - Ruff linting and formatting checks
2. **Type Check** - Pyright static type analysis
3. **Test** - Pytest tests (parallelized across test suites)

### Viewing Results

- **All checks**: Must pass before merging

### CI Caching

The pipeline caches `uv` dependencies (speeds up installation). Cache keys are
based on:

- `uv.lock` file hash
- OS and Python version

## Common Tasks

### Update Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package jax

# Sync environment with updated lock
uv sync --locked --dev
```

### Add New Dependency

```bash
# Runtime dependency
uv add <package>

# Dev dependency
uv add --dev <package>

# Sync environment
uv sync --locked --dev
```

### Fix Pre-commit Hook Issues

```bash
# Pre-commit failed? Hooks auto-fixed files. Re-stage and re-commit:
git add <modified-files>
git commit -m "your message"
```

Do not use `--no-verify` to bypass hook failures. Investigate the failure and
fix the underlying issue — the hooks catch real problems (lint errors,
type-check failures, large files).

### Debug Test Failures

```bash
# Run with debugger
uv run pytest tests/test_manifolds.py --pdb

# Show print statements
uv run pytest tests/test_manifolds.py -s

# Show full diff for assertion failures
uv run pytest tests/test_manifolds.py -vv

# Run specific test
uv run pytest tests/test_manifolds.py::test_dist -v
```

### Profile Performance

```bash
# Run with py-spy profiler
uv run py-spy record -o profile.svg -- python your_script.py

# View profile
open profile.svg
```

## Type Checking

### Running Pyright

```bash
# Check all code
uv run pyright

# Check specific file
uv run pyright hyperbolix/manifolds/poincare.py

# Watch mode (re-check on file changes)
uv run pyright --watch
```

### Type Checking Levels

Current setting: `typeCheckingMode = "basic"`

- `"off"` - No type checking
- `"basic"` - Standard type checking (current)
- `"strict"` - Strict type checking (optional upgrade)

### Common Type Issues

```python
# Missing type annotation
def foo(x):  # ❌ Pyright error
    return x * 2

def foo(x: float) -> float:  # ✅ OK
    return x * 2

# Using jaxtyping for array shapes
from jaxtyping import Float, Array

def dist(x: Float[Array, "dim"], y: Float[Array, "dim"]) -> float:
    return jnp.linalg.norm(x - y)
```

## Performance Tips

### JIT Compilation Best Practices

```python
import jax

# ✅ Good: JIT at top level, vmap for batching
dist_fn = jax.jit(jax.vmap(manifold.dist, in_axes=(0, 0, None)))
distances = dist_fn(x_batch, y_batch, c)

# ❌ Bad: JIT inside loop (recompiles every time)
for x, y in zip(x_batch, y_batch):
    dist = jax.jit(manifold.dist)(x, y, c)  # Don't do this!
```

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
    - `egrad2rgrad`, `is_in_manifold`
    - `c` property and `_cast` (inherit from `ManifoldBase` to get these
      plus dtype-casting machinery for free)
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
   Convention Cheat-Sheet](docs/user-guide/manifolds.md#convention-cheat-sheet).
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
    - Name trainable params `kernel` and `bias` (Flax NNX convention; the
      `simplify-conv-layers` work standardized this across conv layers too)
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

### Documentation Workflow

The docs site is built with MkDocs + Material + mkdocstrings.

```bash
# Live-reload local preview
uv run mkdocs serve

# Strict build (matches CI; fails on broken cross-links and warnings)
uv run mkdocs build --strict
```

Where content goes:

- **`docs/user-guide/`** — synthesis content. Decision tables, conventions,
  composition patterns, pitfalls. *Not* per-symbol reference docs.
- **`docs/api-reference/`** — mostly `:::` autoreference blocks. Add a
  one-paragraph intro for each new module; mkdocstrings handles signatures
  and docstrings automatically.
- **`docs/getting-started.md` / `docs/index.md`** — only update for
  user-facing feature launches (a new manifold counts; an internal refactor
  doesn't).
- **`docs/changelog.md`** — every feature, breaking change, or notable bug
  fix gets an `[Unreleased]` entry under `### Added` / `### Changed` /
  `### Fixed`.

Always run `uv run mkdocs build --strict` before pushing docs changes — the
CI docs build uses the same flag, and dead cross-links between guide pages
fail the build.

## Troubleshooting

### Pre-commit Hook Fails

```bash
# See what failed
git commit -m "message"  # Shows failing hooks

# Run manually to debug
uv run pre-commit run --all-files --verbose

# Update hook versions
uv run pre-commit autoupdate
```

### Tests Pass Locally, Fail in CI

```bash
# Check if it's a caching issue - clear caches in GitHub Actions
# Check if it's a dependency issue - uv.lock might be out of sync
uv lock --check

# Check if it's a Python version issue - CI uses .python-version
cat .python-version
```

### Out of Memory During Tests

```bash
# Run tests sequentially (no parallel)
uv run pytest --maxprocesses=1

# Run smaller test subset (dimension 2, float32)
uv run pytest -k "2-float32"

# Reduce batch sizes in conftest.py
```

## Git Workflow

### Recommended Commit Flow

```bash
# Make changes
git add <files>

# Pre-commit hooks run automatically
git commit -m "descriptive message"

# If hooks modify files, stage and commit again
git add <auto-fixed-files>
git commit -m "descriptive message"

# Push
git push
```

### Branch Strategy

- `main` — release-tracked production code; release tags follow `vMAJOR.MINOR.PATCH`
- Feature branches — `feature/<short-name>` (e.g. `feature/product-manifolds`)
- Fix branches — `fix/<short-name>` (e.g. `fix/float32-stability`)

## Resources

- **CI Pipeline**: `.github/workflows/ci.yaml`
- **Pyright Config**: `pyproject.toml` → `[tool.pyright]`
- **Pre-commit Config**: `.pre-commit-config.yaml`
- **User Documentation**: `docs/` (built with `uv run mkdocs serve`)
