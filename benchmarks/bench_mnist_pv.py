"""MNIST benchmark for Proper Velocity (PV) neural network layers.

Compares two fully-hyperbolic PV variants — no Euclidean embedding layers:

- PVDirect:  expmap_0(784) + HypLinearPV stack + HypRegressionPV
- PV-CNN:    HypConv2DPV + relu + HypConv2DPV + relu + GAP + HypLinearPV + HypRegressionPV

Both models classify entirely on the PV manifold (HypRegressionPV head).

Metrics: memory footprint, wallclock time, accuracy.

Run with:
    uv run python benchmarks/bench_mnist_pv.py [OPTIONS]

Examples:
    # Run both models
    uv run python benchmarks/bench_mnist_pv.py

    # Run only the direct FC model
    uv run python benchmarks/bench_mnist_pv.py --pv-direct

    # Run only the PV CNN
    uv run python benchmarks/bench_mnist_pv.py --pv-cnn
"""

import argparse
import json

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from benchmarks.utils import (
    benchmark_model,
    load_mnist_data,
    plot_comparison,
    print_summary_table,
)
from hyperbolix.manifolds import ProperVelocity
from hyperbolix.nn_layers import (
    HypConv2DPV,
    HypLinearPV,
    HypRegressionPV,
)

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)

# Class-based manifold instance for NN layers
pv = ProperVelocity(dtype=jnp.float64)

# Default curvature. Chen et al. 2026 use K = -0.5 for PV on CIFAR with a
# ResNet-18 + PVManifoldMLR head, paired with lr=1e-1, weight decay, and 200
# epochs. Our MNIST benchmark runs only 5 epochs at lr=1e-2 (hardcoded in
# benchmarks/utils.py), so the paper's c=0.5 undertrains in our budget: we
# use c=0.1 which converges cleanly for PV-Direct in 5 epochs. The paper also
# references a learnable-curvature config (`learn_k = True`); we deliberately
# keep curvature fixed here to keep the MNIST benchmark simple.
DEFAULT_C = 0.1


# ==============================================================================
# Model Definitions
# ==============================================================================


class PVDirect(nnx.Module):
    """Fully-hyperbolic PV FC with direct pixel-to-manifold projection.

    No Euclidean preprocessing: raw pixels are mapped onto the PV manifold via
    ``expmap_0``, then passed through a PV-only stack. PV advertises stability
    at large radii, so we do not pre-scale the input (unlike the Poincaré
    analogue, where a 0.01 pre-scale is needed to avoid ball-boundary saturation).

    Architecture:
        Input (784) → expmap_0 to PV (784-dim)
                   → HypLinearPV(784→64) + ReLU (on PV manifold)
                   → HypLinearPV(64→64)
                   → HypRegressionPV(64→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.hyp1 = HypLinearPV(manifold_module=pv, in_dim=784, out_dim=64, rngs=rngs)
        self.hyp2 = HypLinearPV(manifold_module=pv, in_dim=64, out_dim=64, rngs=rngs)
        self.output = HypRegressionPV(manifold_module=pv, in_dim=64, out_dim=10, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        del use_running_average  # no BatchNorm in PV layers (yet)
        # Project raw pixels directly to PV — no scaling. PV's unbounded
        # geometry handles large radii without boundary saturation.
        x = jax.vmap(pv.expmap_0, in_axes=(0, None))(x, c)  # (batch, 784)

        # PV layers. ReLU works directly on PV points (paper Sec 5.3).
        x = self.hyp1(x, c)  # (batch, 64)
        x = jax.nn.relu(x)
        x = self.hyp2(x, c)  # (batch, 64)

        # PV MLR classification
        return self.output(x, c)  # (batch, 10)


class FullyHyperbolicCNN_PV(nnx.Module):
    """Fully-hyperbolic CNN built entirely on the PV manifold.

    Unlike the Poincaré conv (which round-trips through the tangent space), the
    PV conv consumes and produces points on the PV manifold directly — ReLU can
    be applied in PV space between layers (paper Sec 5.3). The first conv lifts
    raw pixels via ``expmap_0`` (``input_space="tangent"``); subsequent layers
    stay on the manifold.

    Architecture:
        Input (batch, 784)
        → reshape (batch, 28, 28, 1)
        → HypConv2DPV(1→32, k=3, stride=2, SAME, input_space="tangent") + ReLU
        → HypConv2DPV(32→64, k=3, stride=2, SAME, input_space="manifold") + ReLU
        → GAP (mean over spatial dims — still a PV point, since PV is R^n)
        → HypLinearPV(64→64) + ReLU
        → HypRegressionPV(64→10)

    Init note
    ---------
    Relies on ``HypConv2DPV`` / ``HypLinearPV`` defaulting to He(fan_in) kernel
    init — required for a deep fully-hyperbolic stack to preserve variance
    under ReLU. The paper's ``std=1e-2`` default (kept on ``HypRegressionPV``)
    is tuned for a single MLR head receiving ``O(1)``-variance features from a
    well-initialized backbone and would otherwise cause ~50,000x signal
    attenuation here, pinning logits at O(1e-4) and softmax at uniform.
    """

    def __init__(self, rngs: nnx.Rngs):
        self.hyp_conv1 = HypConv2DPV(
            manifold_module=pv,
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            rngs=rngs,
            stride=2,
            padding="SAME",
            input_space="tangent",  # raw pixels → expmap_0 inside the layer
        )
        self.hyp_conv2 = HypConv2DPV(
            manifold_module=pv,
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            rngs=rngs,
            stride=2,
            padding="SAME",
            input_space="manifold",  # conv1 output already on PV
        )
        self.hyp_linear = HypLinearPV(
            manifold_module=pv,
            in_dim=64,
            out_dim=64,
            rngs=rngs,
            input_space="manifold",
        )
        self.output = HypRegressionPV(
            manifold_module=pv,
            in_dim=64,
            out_dim=10,
            rngs=rngs,
            input_space="manifold",
        )

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        del use_running_average
        # (batch, 784) → (batch, 28, 28, 1)
        x = x.reshape(-1, 28, 28, 1)

        # First PV conv block (tangent in → manifold out)
        x = self.hyp_conv1(x, c)  # (batch, 14, 14, 32) on PV
        x = jax.nn.relu(x)

        # Second PV conv block (manifold in → manifold out)
        x = self.hyp_conv2(x, c)  # (batch, 7, 7, 64) on PV
        x = jax.nn.relu(x)

        # Global average pooling. PV is R^n, so spatial mean is still a PV point.
        x = jnp.mean(x, axis=(1, 2))  # (batch, 64) on PV

        # PV linear + MLR head
        x = self.hyp_linear(x, c)  # (batch, 64)
        x = jax.nn.relu(x)
        return self.output(x, c)  # (batch, 10)


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MNIST benchmark for fully-hyperbolic Proper Velocity models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both models (default)
  python benchmarks/bench_mnist_pv.py

  # Run only the direct FC model
  python benchmarks/bench_mnist_pv.py --pv-direct

  # Run only the PV CNN
  python benchmarks/bench_mnist_pv.py --pv-cnn
        """,
    )

    parser.add_argument("--pv-direct", action="store_true", help="Run PV FC with direct pixel projection")
    parser.add_argument("--pv-cnn", action="store_true", help="Run fully-hyperbolic PV CNN")
    parser.add_argument("--all", action="store_true", help="Run all models (default if no flags specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help=f"Manifold curvature (default: {DEFAULT_C})")

    return parser.parse_args()


def main():
    """Run benchmarks based on command line arguments."""
    args = parse_args()

    run_all = args.all or not (args.pv_direct or args.pv_cnn)

    available_models = [
        (PVDirect, "PV-Direct", args.pv_direct or run_all),
        (FullyHyperbolicCNN_PV, "PV-CNN", args.pv_cnn or run_all),
    ]

    models = [(cls, name) for cls, name, should_run in available_models if should_run]

    if not models:
        print("No models selected. Use --help to see available options.")
        return

    print("=" * 60)
    print("MNIST Proper Velocity Layer Benchmark")
    print("=" * 60)
    print(f"\nRunning {len(models)} model(s): {', '.join(name for _, name in models)}")
    print(f"Random seed: {args.seed}")
    print(f"Curvature c: {args.c}")
    print("\nLoading MNIST data...")
    train_data, test_data = load_mnist_data()

    results = {}
    for model_class, name in models:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {name}")
        print("=" * 60)
        results[name] = benchmark_model(model_class, name, train_data, test_data, seed=args.seed, c=args.c, batch_size=128)

    # Save results
    print("\nSaving results...")
    with open("results/mnist_pv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/mnist_pv_results.json")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(results, "results/mnist_pv_comparison.png")

    # Print summary table
    print_summary_table(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
