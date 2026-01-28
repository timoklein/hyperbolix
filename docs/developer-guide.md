# Developer Guide

Contributing to Hyperbolix development.

For detailed development instructions, see [DEVELOPER_GUIDE.md](https://github.com/hyperbolix/hyperbolix/blob/main/DEVELOPER_GUIDE.md) in the repository.

## Quick Links

- **Setup**: Environment configuration with `uv`
- **Testing**: Running test suites and benchmarks
- **Linting**: Pre-commit hooks and code quality
- **CI/CD**: GitHub Actions pipeline
- **Contributing**: Pull request guidelines

## Key Commands

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run benchmarks
uv run pytest benchmarks/ -v

# Linting and formatting
uv run pre-commit run --all-files

# Type checking
uv run pyright hyperbolix/
```

## Project Structure

```
hyperbolix/
├── hyperbolix/           # Source code
│   ├── manifolds/        # Core geometry
│   ├── nn_layers/        # Neural network layers
│   ├── optim/            # Riemannian optimizers
│   ├── distributions/    # Probability distributions
│   └── utils/            # Utilities
├── tests/                # Test suite
├── benchmarks/           # Performance benchmarks
└── docs/                 # Documentation source
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run pre-commit checks
5. Submit a pull request

See the full [DEVELOPER_GUIDE.md](https://github.com/hyperbolix/hyperbolix/blob/main/DEVELOPER_GUIDE.md) for details.
