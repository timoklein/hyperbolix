# Hyperbolix Benchmarks

Performance benchmarks for the JAX implementation of hyperbolic operations and neural network layers.

## Running Benchmarks

### Run all benchmarks
```bash
# Recommended: Run benchmarks only (skip test execution)
uv run pytest benchmarks/ --benchmark-only -v

# Alternative: Run with test execution (slower, includes warmup)
uv run pytest benchmarks/ -v
```

> **Note**: The `--benchmark-only` flag skips the actual test execution and only measures performance. Without this flag, pytest will run the benchmark code multiple times (once for testing, once for timing), which is slower but more thorough.

### Run specific benchmark file
```bash
uv run pytest benchmarks/bench_manifolds.py --benchmark-only
uv run pytest benchmarks/bench_nn_layers.py --benchmark-only
```

### Run with specific parameters
```bash
# Run only for dim=128, batch_size=1000
uv run pytest benchmarks/ --benchmark-only -k "128-1000"
```

## Saving and Comparing Results

### Save baseline results
```bash
uv run pytest benchmarks/ --benchmark-only --benchmark-save=baseline
```

### Compare against baseline
```bash
# After making changes to the code
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=baseline
```

### Compare two saved runs
```bash
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=0001
```

## Generating Reports

### Generate histogram
```bash
uv run pytest benchmarks/ --benchmark-only --benchmark-histogram
```

### Generate JSON report
```bash
uv run pytest benchmarks/ --benchmark-only --benchmark-json=output.json
```

### View saved benchmarks
```bash
uv run pytest-benchmark list
uv run pytest-benchmark compare
```

## Benchmark Categories

### Manifold Operations (`bench_manifolds.py`)
- Distance computation (with/without JIT)
- Exponential and logarithmic maps
- Comparison of different distance implementations
- Math utility functions (acosh, atanh, etc.)

**Key Metrics:**
- JIT speedup factor (should be 10-100x for large batches)
- Version comparison (which distance formula is fastest)

### Neural Network Layers (`bench_nn_layers.py`)
- Forward pass performance
- Forward + backward (gradient computation)
- Multi-layer composition
- JIT compilation benefits

**Key Metrics:**
- Forward pass time
- Gradient computation time
- JIT overhead (first call) vs runtime (subsequent calls)

## Performance Expectations

### JIT Compilation
- **First call:** Slow (compilation overhead, 100ms-1s)
- **Subsequent calls:** Fast (10-100x speedup over non-JIT)

### Batch Size Scaling
- Small batches (100): Less JIT benefit
- Large batches (1000+): Maximum JIT benefit

### Dimension Scaling
- Low dim (2-10): Fast, memory-bound
- High dim (128+): Compute-bound, benefits from JIT

## Continuous Performance Tracking

The CI pipeline runs a subset of benchmarks and compares against saved baselines:

```bash
# Run in CI
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

This fails the build if performance regresses by more than 10%.

## Troubleshooting

### Benchmarks are inconsistent
- Run with `--benchmark-warmup=on` to stabilize results
- Increase `--benchmark-min-rounds=10` for more samples

### Out of memory
- Reduce batch sizes in `conftest.py`
- Run specific dimension parameters: `-k "dim2"`

### Slow benchmark runs
- Skip warmup: `--benchmark-disable-gc`
- Run subset: `-k "jit"` (only JIT tests)

---

## MNIST Layer Comparison Benchmark

The `bench_mnist_layer_comparison.py` script provides end-to-end training benchmarks comparing different hyperbolic neural network architectures on MNIST.

### Available Models

- **FHCNN-Hybrid**: Fully Hyperbolic CNN with Euclidean embedding
- **FHCNN-Direct**: Fully Hyperbolic CNN with direct projection
- **HTC-Hybrid**: Hyperbolic Transformation Component with Euclidean embedding
- **HTC-Direct**: Hyperbolic Transformation Component with direct projection
- **FHCNN-CNN-Hybrid**: FHCNN-based convolutional network with HRCBatchNorm

### Usage

#### Run all models (default)

```bash
uv run python benchmarks/bench_mnist_layer_comparison.py
```

#### Run specific models

```bash
# Run only the CNN model
uv run python benchmarks/bench_mnist_layer_comparison.py --fhcnn-cnn

# Run FHCNN variants
uv run python benchmarks/bench_mnist_layer_comparison.py --fhcnn-hybrid --fhcnn-direct

# Run HTC variants
uv run python benchmarks/bench_mnist_layer_comparison.py --htc-hybrid --htc-direct
```

#### Custom random seed

```bash
uv run python benchmarks/bench_mnist_layer_comparison.py --fhcnn-cnn --seed 123
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--fhcnn-hybrid` | Run FHCNN with Euclidean embedding |
| `--fhcnn-direct` | Run FHCNN with direct projection |
| `--htc-hybrid` | Run HTC with Euclidean embedding |
| `--htc-direct` | Run HTC with direct projection |
| `--fhcnn-cnn` | Run FHCNN-based CNN with BatchNorm |
| `--all` | Run all models (default if no flags specified) |
| `--seed SEED` | Random seed for reproducibility (default: 42) |
| `-h, --help` | Show help message |

### Output

The benchmark generates:

1. **JSON results** (`results/mnist_benchmark_results.json`):
   - Training losses and validation accuracies per epoch
   - Parameter counts and memory usage
   - Compilation and training times

2. **Comparison plots** (`results/mnist_comparison.png`):
   - Training loss and accuracy curves
   - Training speed comparison
   - Model capacity comparison

3. **Summary table** (printed to console)

### Examples

```bash
# Quick CNN test during development
uv run python benchmarks/bench_mnist_layer_comparison.py --fhcnn-cnn

# Compare FHCNN vs HTC (hybrid variants)
uv run python benchmarks/bench_mnist_layer_comparison.py --fhcnn-hybrid --htc-hybrid

# Full benchmark with custom seed
uv run python benchmarks/bench_mnist_layer_comparison.py --all --seed 999
```
