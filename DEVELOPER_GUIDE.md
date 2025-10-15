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
uv run ruff check src tests benchmarks

# Format with Ruff
uv run ruff format src tests benchmarks

# Type check with Pyright
uv run pyright src/hyperbolix_jax
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific test suite
uv run pytest tests/jax/test_manifolds.py
uv run pytest tests/jax/test_nn_layers.py

# Fast tests only (skip slow parametrizations)
uv run pytest -k "dim2"

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x
```

### Running Benchmarks

```bash
# All benchmarks (recommended: use --benchmark-only)
uv run pytest benchmarks/ --benchmark-only

# Run benchmarks without --benchmark-only (slower, includes test execution)
uv run pytest benchmarks/

# Specific benchmark file
uv run pytest benchmarks/bench_manifolds.py --benchmark-only

# Specific test
uv run pytest benchmarks/ -k "test_poincare_dist_with_jit" --benchmark-only

# Save baseline
uv run pytest benchmarks/ --benchmark-only --benchmark-save=my-baseline

# Compare to baseline
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=my-baseline

# Generate histogram
uv run pytest benchmarks/ --benchmark-only --benchmark-histogram

# Quick run (subset of parameters)
uv run pytest benchmarks/ --benchmark-only -k "dim10-batch_size100"
```

**Note**: Benchmark files (`bench_*.py`) are automatically discovered by pytest. The `--benchmark-only` flag is recommended to skip test execution and only measure performance.

See `benchmarks/README.md` for detailed benchmarking guide.

## CI/CD Pipeline

The CI pipeline runs automatically on push and pull requests:

### Jobs

1. **Lint** - Ruff linting and formatting checks
2. **Type Check** - Pyright static type analysis
3. **Test** - Pytest tests (parallelized across test suites)
4. **Benchmark** - Performance regression detection

### Viewing Results

- **Benchmarks**: Artifacts available in GitHub Actions run
- **All checks**: Must pass before merging

### CI Caching

The pipeline caches:

- `uv` dependencies (speeds up installation)
- Benchmark baselines (for performance comparison)

Cache keys are based on:

- `uv.lock` file hash
- OS and Python version
- Branch name (for benchmarks)

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
# Pre-commit failed? Hooks auto-fixed files:
git add <modified-files>
git commit -m "your message"  # Will succeed now

# Skip hooks (emergency only!)
git commit --no-verify -m "your message"
```

### Debug Test Failures

```bash
# Run with debugger
uv run pytest tests/jax/test_manifolds.py --pdb

# Show print statements
uv run pytest tests/jax/test_manifolds.py -s

# Show full diff for assertion failures
uv run pytest tests/jax/test_manifolds.py -vv

# Run specific test
uv run pytest tests/jax/test_manifolds.py::test_dist -v
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
uv run pyright src/hyperbolix_jax/manifolds/poincare.py

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

### Benchmark Before Optimizing

```bash
# Save current performance
uv run pytest benchmarks/ --benchmark-only --benchmark-save=before

# Make changes...

# Compare
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=before
```

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

### Benchmark Results Are Noisy

```bash
# Increase warmup rounds
uv run pytest benchmarks/ --benchmark-only --benchmark-warmup=on

# Increase number of rounds
uv run pytest benchmarks/ --benchmark-only --benchmark-min-rounds=10

# Disable garbage collection during benchmarks
uv run pytest benchmarks/ --benchmark-only --benchmark-disable-gc
```

### Out of Memory During Tests

```bash
# Run tests sequentially (no parallel)
uv run pytest --maxprocesses=1

# Run smaller test subset
uv run pytest -k "dim2"

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

- `main` - Production-ready code
- `nn-layers` - Neural network layer development (current)
- Feature branches - Use descriptive names: `feature/optimizer-port`, `fix/float32-stability`

## Resources

- **CI Pipeline**: `.github/workflows/ci.yaml`
- **Pyright Config**: `pyproject.toml` → `[tool.pyright]`
- **Pre-commit Config**: `.pre-commit-config.yaml`
- **Benchmark Guide**: `benchmarks/README.md`
- **Migration Plan**: `jax_migration.md`
- **Project Handoff**: `handoff.md`
