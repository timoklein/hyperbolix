"""MNIST benchmark for Poincaré ball neural network layers.

Compares multiple Poincaré variants:
- PoincarePPHybrid: HNN++ with Euclidean embedding (analogous to FHCNNHybrid)
- PoincarePPDirect: HNN++ with direct projection (analogous to FHCNNDirect)
- PoincarePPFC: HNN++ fully connected with wider embedding
- FullyHyperbolicCNN_Poincare: Conv-based, tangent-space activations

All FC models use HypLinearPoincarePP + poincare_relu + HypRegressionPoincarePP.

Metrics: memory footprint, wallclock time, accuracy

Run with:
    uv run python benchmarks/bench_mnist_poincare.py [OPTIONS]

Examples:
    # Run all models
    uv run python benchmarks/bench_mnist_poincare.py

    # Run only the hybrid model
    uv run python benchmarks/bench_mnist_poincare.py --pp-hybrid

    # Run only the direct projection model
    uv run python benchmarks/bench_mnist_poincare.py --pp-direct

    # Run only the FC model
    uv run python benchmarks/bench_mnist_poincare.py --pp-fc

    # Run only the Poincaré CNN
    uv run python benchmarks/bench_mnist_poincare.py --poincare-cnn
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
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import (
    HypConv2DPoincare,
    HypLinearPoincarePP,
    HypRegressionPoincarePP,
    poincare_relu,
)

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)

# Class-based manifold instance for NN layers
poincare = Poincare(dtype=jnp.float64)

# Default curvature — reference (van Spengler et al. 2023) uses init_c=0.1.
# At c=1.0 the ball radius is 1, causing boundary saturation in the HNN++ MLR
# (lam = 2(1-c||x||²) → 0 near boundary, killing input signal).
# At c=0.1 the ball radius is √10 ≈ 3.16, giving much more room.
DEFAULT_C = 0.1


# ==============================================================================
# Model Definitions
# ==============================================================================


class PoincarePPHybrid(nnx.Module):
    """HNN++ with Euclidean embedding (analogous to FHCNNHybrid).

    Architecture:
        Input (784) → Euclidean Linear(784→32) + ReLU
                   → expmap_0 to Poincaré ball (32-dim)
                   → HypLinearPoincarePP(32→64) + poincare_relu
                   → HypLinearPoincarePP(64→64)
                   → HypRegressionPoincarePP(64→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Linear(784, 32, rngs=rngs)
        self.hyp1 = HypLinearPoincarePP(
            manifold_module=poincare,
            in_dim=32,
            out_dim=64,
            rngs=rngs,
        )
        self.hyp2 = HypLinearPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=64,
            rngs=rngs,
        )
        self.output = HypRegressionPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=10,
            rngs=rngs,
        )

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Euclidean embedding
        x = jax.nn.relu(self.embed(x))  # (batch, 32)

        # Project to Poincaré ball (no +1 time dim, unlike hyperboloid)
        x = jax.vmap(poincare.expmap_0, in_axes=(0, None))(x, c)  # (batch, 32)

        # Hyperbolic layers
        x = self.hyp1(x, c)  # (batch, 64)
        x = poincare_relu(x, c)
        x = self.hyp2(x, c)  # (batch, 64)

        # Poincaré MLR classification
        return self.output(x, c)  # (batch, 10)


class PoincarePPDirect(nnx.Module):
    """HNN++ with direct projection (analogous to FHCNNDirect).

    Scales input by 0.01 before expmap_0 to avoid boundary saturation
    (large Euclidean norms would map near the Poincaré ball boundary).

    Architecture:
        Input (784) → scale*0.01
                   → expmap_0 to Poincaré ball (784-dim)
                   → HypLinearPoincarePP(784→64) + poincare_relu
                   → HypLinearPoincarePP(64→64)
                   → HypRegressionPoincarePP(64→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.hyp1 = HypLinearPoincarePP(
            manifold_module=poincare,
            in_dim=784,
            out_dim=64,
            rngs=rngs,
        )
        self.hyp2 = HypLinearPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=64,
            rngs=rngs,
        )
        self.output = HypRegressionPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=10,
            rngs=rngs,
        )

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Scale down to avoid boundary saturation (pixel values [0,1] sum to large norms)
        x = x * 0.01  # (batch, 784)

        # Project to Poincaré ball
        x = jax.vmap(poincare.expmap_0, in_axes=(0, None))(x, c)  # (batch, 784)

        # Hyperbolic layers
        x = self.hyp1(x, c)  # (batch, 64)
        x = poincare_relu(x, c)
        x = self.hyp2(x, c)  # (batch, 64)

        # Poincaré MLR classification
        return self.output(x, c)  # (batch, 10)


class PoincarePPFC(nnx.Module):
    """HNN++ fully connected with wider embedding.

    Architecture:
        Input (784) → Euclidean Linear(784→64) + ReLU
                   → expmap_0 to Poincaré ball (64-dim)
                   → HypLinearPoincarePP(64→64) + poincare_relu
                   → HypLinearPoincarePP(64→64)
                   → HypRegressionPoincarePP(64→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Linear(784, 64, rngs=rngs)
        self.hyp1 = HypLinearPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=64,
            rngs=rngs,
        )
        self.hyp2 = HypLinearPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=64,
            rngs=rngs,
        )
        self.output = HypRegressionPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=10,
            rngs=rngs,
        )

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Wider Euclidean embedding
        x = jax.nn.relu(self.embed(x))  # (batch, 64)

        # Project to Poincaré ball
        x = jax.vmap(poincare.expmap_0, in_axes=(0, None))(x, c)  # (batch, 64)

        # Hyperbolic layers
        x = self.hyp1(x, c)  # (batch, 64)
        x = poincare_relu(x, c)
        x = self.hyp2(x, c)  # (batch, 64)

        # Poincaré MLR classification
        return self.output(x, c)  # (batch, 10)


class FullyHyperbolicCNN_Poincare(nnx.Module):
    """Fully hyperbolic CNN using Poincaré ball (beta-concat + HNN++ approach).

    Matches the reference Poincaré ResNet (van Spengler et al. 2023) computation
    flow: conv layers operate in tangent space internally and return tangent-space
    output. Standard activations (relu) are applied between layers in tangent space.
    Only maps to manifold before the final classification head.

    Hyperboloid uses 2→33→65 (ambient), Poincaré equivalent is 1→32→64 (manifold).

    Architecture:
        Input (batch, 784)
        → reshape (batch, 28, 28, 1) [tangent space: pixel values]
        → HypConv2DPoincare(1→32, k=3, stride=2, SAME)
        → ReLU (in tangent space)
        → (batch, 14, 14, 32) tangent space
        → HypConv2DPoincare(32→64, k=3, stride=2, SAME)
        → ReLU (in tangent space)
        → (batch, 7, 7, 64) tangent space
        → GAP: mean over spatial dims (Euclidean mean in tangent space)
        → (batch, 64) tangent space
        → expmap_0 → manifold
        → HypLinearPoincarePP(64→64) + Poincaré ReLU
        → HypRegressionPoincarePP(64→10)

    References
    ----------
    van Spengler et al. "Poincaré ResNet." ICML 2023.
    Shimizu et al. "Hyperbolic neural networks++." arXiv:2006.08210 (2020).
    """

    def __init__(self, rngs: nnx.Rngs):
        # Conv layers: input/output in tangent space (matching reference)
        self.hyp_conv1 = HypConv2DPoincare(
            manifold_module=poincare,
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            rngs=rngs,
            stride=2,
            padding="SAME",
            input_space="tangent",  # pixels treated as 1D tangent vectors
        )
        self.hyp_conv2 = HypConv2DPoincare(
            manifold_module=poincare,
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            rngs=rngs,
            stride=2,
            padding="SAME",
            input_space="tangent",  # conv1 output is tangent space
        )
        self.hyp_linear = HypLinearPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=64,
            rngs=rngs,
        )
        self.output = HypRegressionPoincarePP(
            manifold_module=poincare,
            in_dim=64,
            out_dim=10,
            rngs=rngs,
        )

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # (batch, 784) → (batch, 28, 28, 1)
        x = x.reshape(-1, 28, 28, 1)

        # First Poincaré conv block (tangent in → tangent out)
        x = self.hyp_conv1(x, c)  # (batch, 14, 14, 32) tangent space
        x = jax.nn.relu(x)  # Standard ReLU in tangent space (no logmap/expmap)

        # Second Poincaré conv block (tangent in → tangent out)
        x = self.hyp_conv2(x, c)  # (batch, 7, 7, 64) tangent space
        x = jax.nn.relu(x)

        # GAP in tangent space: simple Euclidean mean (approximate Fréchet mean at origin)
        x = jnp.mean(x, axis=(1, 2))  # (batch, 64) tangent space

        # Map to manifold for final classification layers
        x = jax.vmap(poincare.expmap_0, in_axes=(0, None))(x, c)  # (batch, 64) on ball

        # HNN++ linear + Poincaré ReLU + regression head
        x = self.hyp_linear(x, c)  # (batch, 64)
        x = poincare_relu(x, c)
        return self.output(x, c)  # (batch, 10)


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MNIST benchmark comparing Poincaré ball neural network layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models (default)
  python benchmarks/bench_mnist_poincare.py

  # Run only the hybrid model
  python benchmarks/bench_mnist_poincare.py --pp-hybrid

  # Run only the direct projection model
  python benchmarks/bench_mnist_poincare.py --pp-direct

  # Run only the FC model
  python benchmarks/bench_mnist_poincare.py --pp-fc

  # Run only the Poincaré CNN
  python benchmarks/bench_mnist_poincare.py --poincare-cnn
        """,
    )

    parser.add_argument("--pp-hybrid", action="store_true", help="Run HNN++ with Euclidean embedding")
    parser.add_argument("--pp-direct", action="store_true", help="Run HNN++ with direct projection")
    parser.add_argument("--pp-fc", action="store_true", help="Run HNN++ fully connected with wider embedding")
    parser.add_argument("--poincare-cnn", action="store_true", help="Run Poincaré CNN (beta-concat + HNN++)")
    parser.add_argument("--all", action="store_true", help="Run all models (default if no flags specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    return parser.parse_args()


def main():
    """Run benchmarks based on command line arguments."""
    args = parse_args()

    # Determine which models to run
    run_all = args.all or not (args.pp_hybrid or args.pp_direct or args.pp_fc or args.poincare_cnn)

    # Build list of models to benchmark
    available_models = [
        (PoincarePPHybrid, "PP-Hybrid", args.pp_hybrid or run_all),
        (PoincarePPDirect, "PP-Direct", args.pp_direct or run_all),
        (PoincarePPFC, "PP-FC", args.pp_fc or run_all),
        (FullyHyperbolicCNN_Poincare, "Poincare-CNN", args.poincare_cnn or run_all),
    ]

    models = [(cls, name) for cls, name, should_run in available_models if should_run]

    if not models:
        print("No models selected. Use --help to see available options.")
        return

    print("=" * 60)
    print("MNIST Poincaré Layer Benchmark")
    print("=" * 60)
    print(f"\nRunning {len(models)} model(s): {', '.join(name for _, name in models)}")
    print(f"Random seed: {args.seed}")
    print("\nLoading MNIST data...")
    train_data, test_data = load_mnist_data()

    results = {}
    for model_class, name in models:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {name}")
        print("=" * 60)
        results[name] = benchmark_model(model_class, name, train_data, test_data, seed=args.seed, c=DEFAULT_C)

    # Save results
    print("\nSaving results...")
    with open("results/mnist_poincare_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/mnist_poincare_results.json")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(results, "results/mnist_poincare_comparison.png")

    # Print summary table
    print_summary_table(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
