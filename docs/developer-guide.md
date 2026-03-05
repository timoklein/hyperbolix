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

## Coding Conventions

### Shape Suffixes

All tensor/array local variables in function bodies use **shape suffixes** — single capital letters appended to the variable name encoding each dimension. This makes shapes self-documenting and shape bugs immediately visible.

**Dimension key** (used across the codebase):

| Letter | Dimension |
|--------|-----------|
| `B` | batch size |
| `D` | spatial / manifold dimension (`dim`) |
| `A` | ambient dimension (`dim+1`, hyperboloid time+space) |
| `H` | output height |
| `W` | output width |
| `Z` | output depth (3D conv) |
| `C` | channels |
| `K` | kernel elements (`kh×kw`) |
| `N` | number of points |
| `P` | number of hyperplanes / output classes (MLR `out_dim`) |
| `S` | sequence length |
| `F` | frequency dim (`d//2` in RoPE) |

**Examples:**

```python
# poincare_regression.py — MLR forward pass
sub_PBD = addition_fn(p_neg_PD, x_BD, c)        # (P, B, D) from broadcasting
sub_BPD = jnp.transpose(sub_PBD, (1, 0, 2))      # reorder to (B, P, D)
res_BP  = lambda_p_P1.T * a_norm_P1.T * signed_dist2hyp_BP

# hyperboloid_conv.py — patch extraction
patches_BHWCkhkw = patches_flat_BHW_CKhKw.reshape(B, H, W, C, kh, kw)
patches_BHWkhkwC = patches_BHWCkhkw.transpose(0, 1, 2, 4, 5, 3)

# poincare.py — conformal factor broadcast
res_BP = 2 * z_norm_1P * signed_dist2hyp_BP  # z_norm.T broadcasts (1,P) over (B,P)
```

**Rules:**
- Use compound suffixes for flattened dims: `x_flat_NC` with a comment explaining the merge
- `_B1` for `keepdims=True` results, `_1P` for transposed broadcast tensors
- Add a dimension key docstring at the top of each file that uses shape suffixes

Each annotated file begins with a docstring like:

```python
"""
Dimension key:
  B: batch size     D: manifold dimension
  P: output classes (hyperplanes)
"""
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run pre-commit checks
5. Submit a pull request

See the full [DEVELOPER_GUIDE.md](https://github.com/hyperbolix/hyperbolix/blob/main/DEVELOPER_GUIDE.md) for details.
