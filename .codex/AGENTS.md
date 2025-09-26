# Repository Guidelines

## Project Structure & Module Organization
Core source lives in `src/`, split across `manifolds` (geometry primitives), `nn_layers` (PyTorch layers), `optim` (Riemannian optimizers), and `utils` (shared math and visualization helpers). Tests sit in `tests/`, mirroring the package layout and using `tests/conftest.py` for shared fixtures. Project metadata, dependency groups, and tooling defaults live in `pyproject.toml`, while reference material is stored under `docs/`.

## Build, Test, and Development Commands
Provision tooling with uv: `uv venv` to create a virtual environment and `source .venv/bin/activate` to enter it. Install runtime deps via `uv pip install .` and include developer extras with `uv pip install .[dev]` or `uv pip install --group dev`. Run modules ad hoc with `uv run python -m manifolds.hyperboloid` and execute individual scripts using `uv run`. The primary verification flow is `uv run pytest`, with targeted runs like `uv run pytest tests/test_manifolds.py -k poincare`. Use `uv run pytest -ra -q` to mirror the default options baked into the project configuration.

## Coding Style & Naming Conventions
Adhere to the formatting stack configured in `pyproject.toml`: Black and Ruff enforce a 127-character line limit, and isort follows Black's import ordering. Run `uv run ruff check src tests` for linting, `uv run black src tests` for formatting, and `uv run isort src tests` when adjusting imports. Use 4-space indentation, snake_case for functions and variables, PascalCase for classes, and UPPER_SNAKE_CASE for constants. Prefer explicit imports (`from utils.math_utils import mobius_add`) and keep module-level docstrings focused on curvature assumptions or tensor shapes.

## Testing Guidelines
Pytest drives the suite. Add tests alongside code in `tests/`, naming files `test_<component>.py` and test functions `test_<behavior>`. Cover both Euclidean and hyperbolic edge cases, and assert gradient stability when modifying optimizers. Before opening a PR, run `uv run pytest --maxfail=1 --disable-warnings` to catch regressions quickly and confirm the baseline suite passes.

## Commit & Pull Request Guidelines
With no shared history yet, use Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) plus an imperative summary and include relevant tests in each commit. Pull requests should describe the problem, outline the solution, reference issues, and attach logs or screenshots when behavior changes. Confirm `uv run pytest` passes and note any follow-ups before requesting review.
