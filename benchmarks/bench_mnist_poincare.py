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
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets import load_dataset
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import (
    HypConv2DPoincare,
    HypLinearPoincarePP,
    HypRegressionPoincarePP,
    poincare_relu,
)
from hyperbolix.optim import riemannian_sgd

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
# Data Loading
# ==============================================================================


def load_mnist_data(batch_size: int = 128):
    """Load MNIST using HuggingFace datasets.

    Parameters
    ----------
    batch_size : int
        Batch size for training and evaluation

    Returns
    -------
    train_data : Dataset
        Training dataset with flattened and normalized images
    test_data : Dataset
        Test dataset with flattened and normalized images
    """
    print("  Loading MNIST dataset from HuggingFace...")
    dataset = load_dataset("mnist")

    def prepare_batch(batch):
        # Flatten (28x28 → 784) and normalize [0, 255] → [0, 1]
        images = np.array(batch["image"])  # (batch, 28, 28)
        x = images.reshape(-1, 784).astype(np.float32) / 255.0  # (batch, 784)
        y = np.array(batch["label"]).astype(np.int32)
        return {"image": x, "label": y}

    # Prepare datasets without .with_format() to avoid extra dimensions
    train_data = dataset["train"].map(prepare_batch, batched=True)
    test_data = dataset["test"].map(prepare_batch, batched=True)

    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    return train_data, test_data


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
        Input (784) → scale×0.01
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
# Training Utilities
# ==============================================================================


def loss_fn(model: nnx.Module, x: Array, y: Array, c: float = 1.0, use_running_average: bool = False) -> Array:
    """Cross-entropy loss."""
    logits = model(x, c, use_running_average=use_running_average)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x: Array, y: Array, c: float = 1.0) -> Array:
    """Single training step."""
    loss, grads = nnx.value_and_grad(lambda m, x, y, c: loss_fn(m, x, y, c, use_running_average=False))(model, x, y, c)
    optimizer.update(model, grads)
    return loss


def train_epoch(model: nnx.Module, optimizer: nnx.Optimizer, data_loader: Any, c: float = 1.0) -> dict[str, float]:
    """Train for one epoch.

    Parameters
    ----------
    model : nnx.Module
        Model to train
    optimizer : nnx.Optimizer
        Optimizer for parameter updates
    data_loader : Dataset
        Training dataset
    c : float
        Curvature parameter

    Returns
    -------
    metrics : dict
        Dictionary with 'loss' and 'time' keys
    """
    epoch_start = time.perf_counter()

    losses = []
    for batch in data_loader.iter(batch_size=128):
        x = jnp.array(batch["image"])
        y = jnp.array(batch["label"])
        # Squeeze extra dimension if present
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        loss = train_step(model, optimizer, x, y, c)
        losses.append(float(loss))

    epoch_time = time.perf_counter() - epoch_start
    return {"loss": np.mean(losses), "time": epoch_time}


def evaluate(model: nnx.Module, data_loader: Any, c: float = 1.0) -> float:
    """Compute accuracy on dataset.

    Parameters
    ----------
    model : nnx.Module
        Model to evaluate
    data_loader : Dataset
        Dataset to evaluate on
    c : float
        Curvature parameter

    Returns
    -------
    accuracy : float
        Classification accuracy
    """
    correct = 0
    total = 0

    for batch in data_loader.iter(batch_size=128):
        x = jnp.array(batch["image"])
        y = jnp.array(batch["label"])
        # Squeeze extra dimension if present
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        # Evaluation mode: use_running_average=True for BatchNorm
        logits = model(x, c, use_running_average=True)
        preds = jnp.argmax(logits, axis=-1)
        correct += int(jnp.sum(preds == y))
        total += len(y)

    return correct / total


# ==============================================================================
# Memory Profiling
# ==============================================================================


def count_parameters(model: nnx.Module) -> int:
    """Count trainable parameters."""
    params = nnx.state(model, nnx.Param)
    return sum(x.size for x in jax.tree.leaves(params))


def estimate_memory_mb(model: nnx.Module) -> float:
    """Estimate model memory in MB."""
    params = nnx.state(model, nnx.Param)
    total_bytes = sum(x.nbytes for x in jax.tree.leaves(params))
    return total_bytes / (1024**2)


# ==============================================================================
# Benchmarking
# ==============================================================================


def benchmark_model(
    model_class: type[nnx.Module], model_name: str, train_data: Any, test_data: Any, seed: int = 42
) -> dict[str, Any]:
    """Run full benchmark for one model variant.

    Parameters
    ----------
    model_class : type[nnx.Module]
        Model class to benchmark
    model_name : str
        Name for logging
    train_data : Dataset
        Training dataset
    test_data : Dataset
        Test dataset
    seed : int
        Random seed for reproducibility

    Returns
    -------
    metrics : dict
        Dictionary with all benchmark metrics
    """
    # Initialize model and optimizer
    rngs = nnx.Rngs(params=seed, dropout=seed + 1)
    model = model_class(rngs)

    tx = riemannian_sgd(learning_rate=0.01, momentum=0.9)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Memory metrics
    metrics = {
        "model": model_name,
        "parameters": count_parameters(model),
        "memory_mb": estimate_memory_mb(model),
        "train_losses": [],
        "train_times": [],
        "val_accuracies": [],
    }

    # JIT compilation timing (first call)
    x_dummy = jnp.ones((128, 784))
    y_dummy = jnp.zeros(128, dtype=jnp.int32)

    compile_start = time.perf_counter()
    _ = train_step(model, optimizer, x_dummy, y_dummy, DEFAULT_C)
    metrics["compile_time"] = time.perf_counter() - compile_start

    print(f"  Compilation time: {metrics['compile_time']:.3f}s")

    # Training loop (5 epochs)
    for epoch in range(5):
        epoch_metrics = train_epoch(model, optimizer, train_data, c=DEFAULT_C)
        val_acc = evaluate(model, test_data, c=DEFAULT_C)

        metrics["train_losses"].append(epoch_metrics["loss"])
        metrics["train_times"].append(epoch_metrics["time"])
        metrics["val_accuracies"].append(val_acc)

        print(f"  Epoch {epoch + 1}/5: loss={epoch_metrics['loss']:.4f}, acc={val_acc:.4f}, time={epoch_metrics['time']:.2f}s")

    metrics["final_accuracy"] = metrics["val_accuracies"][-1]
    metrics["total_time"] = sum(metrics["train_times"])

    return metrics


# ==============================================================================
# Visualization
# ==============================================================================


def plot_comparison(results: dict[str, dict[str, Any]], output_path: str):
    """Generate comparison plots.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to their metrics
    output_path : str
        Path to save the plot
    """
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training loss curves
    ax = axes[0, 0]
    for name, metrics in results.items():
        ax.plot(range(1, 6), metrics["train_losses"], marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation accuracy curves
    ax = axes[0, 1]
    for name, metrics in results.items():
        ax.plot(range(1, 6), metrics["val_accuracies"], marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Time per epoch
    ax = axes[1, 0]
    names = list(results.keys())
    times = [results[n]["total_time"] / 5 for n in names]  # Average per epoch
    ax.bar(names, times)
    ax.set_ylabel("Average Time per Epoch (s)")
    ax.set_title("Training Speed Comparison")
    ax.tick_params(axis="x", rotation=45)

    # Plot 4: Memory and parameters
    ax = axes[1, 1]
    x_pos = np.arange(len(names))
    params = [results[n]["parameters"] / 1000 for n in names]  # In thousands
    memory = [results[n]["memory_mb"] for n in names]

    ax2 = ax.twinx()
    ax.bar(x_pos - 0.2, params, 0.4, label="Parameters (k)", color="C0")
    ax2.bar(x_pos + 0.2, memory, 0.4, label="Memory (MB)", color="C1")

    ax.set_ylabel("Parameters (thousands)", color="C0")
    ax2.set_ylabel("Memory (MB)", color="C1")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45)
    ax.set_title("Model Capacity Comparison")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to {output_path}")


def print_summary_table(results: dict[str, dict[str, Any]]):
    """Print summary comparison table.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to their metrics
    """
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Model':<20} {'Params':<12} {'Memory(MB)':<12} {'Time(s)':<10} {'Final Acc':<12} {'Compile(s)':<12}")
    print("-" * 90)

    for name, m in results.items():
        print(
            f"{name:<20} {m['parameters']:<12,} {m['memory_mb']:<12.2f} "
            f"{m['total_time']:<10.2f} {m['final_accuracy']:<12.4f} "
            f"{m['compile_time']:<12.3f}"
        )

    print("=" * 90)


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
        results[name] = benchmark_model(model_class, name, train_data, test_data, seed=args.seed)

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
