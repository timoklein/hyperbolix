"""MNIST benchmark comparing HypLinearHyperboloidFHCNN and HTCLinear.

Compares 4 variants:
- FHCNNHybrid: FHCNN with Euclidean embedding
- FHCNNDirect: FHCNN with direct projection
- HTCHybrid: HTC with Euclidean embedding
- HTCDirect: HTC with direct projection

All models use HypRegressionHyperboloid for fully hyperbolic classification.

Metrics: memory footprint, wallclock time, accuracy

Run with:
    uv run python benchmarks/bench_mnist_layer_comparison.py
"""

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

from hyperbolix.manifolds import hyperboloid
from hyperbolix.nn_layers import HTCLinear, HypLinearHyperboloidFHCNN, HypRegressionHyperboloid
from hyperbolix.optim import riemannian_sgd

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)


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


class FHCNNHybrid(nnx.Module):
    """FHCNN with Euclidean embedding.

    Architecture:
        Input (784) → Euclidean Linear(784→32) + ReLU
                   → Project to Hyperboloid (32→33)
                   → FHCNN Layer 1 (33→65)
                   → FHCNN Layer 2 (65→65)
                   → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Linear(784, 32, rngs=rngs)  # Euclidean embedding
        self.hyp1 = HypLinearHyperboloidFHCNN(
            hyperboloid,
            33,
            65,
            rngs=rngs,
            input_space="manifold",
            learnable_scale=False,
            activation=None,
            normalize=False,
        )
        self.hyp2 = HypLinearHyperboloidFHCNN(
            hyperboloid, 65, 65, rngs=rngs, input_space="manifold", learnable_scale=False, normalize=False
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(self, x: Float[Array, "batch 784"], c: float = 1.0) -> Float[Array, "batch 10"]:
        # Embed in Euclidean space
        x = jax.nn.relu(self.embed(x))  # (batch, 32)

        # Project to hyperboloid
        x = jax.vmap(lambda v: hyperboloid.expmap_0(jnp.concatenate([jnp.zeros(1), v]), c))(x)  # (batch, 33)

        # Hyperbolic layers
        x = self.hyp1(x, c)  # (batch, 65)
        x = self.hyp2(x, c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class FHCNNDirect(nnx.Module):
    """FHCNN with direct projection.

    Architecture:
        Input (784) → Constraint-based projection to Hyperboloid (784→785)
                   → FHCNN Layer 1 (785→65)
                   → FHCNN Layer 2 (65→65)
                   → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        # No Euclidean embedding
        self.hyp1 = HypLinearHyperboloidFHCNN(
            hyperboloid,
            785,
            65,
            rngs=rngs,
            input_space="manifold",
            learnable_scale=False,
            normalize=False,
        )
        self.hyp2 = HypLinearHyperboloidFHCNN(
            hyperboloid, 65, 65, rngs=rngs, input_space="manifold", learnable_scale=False, normalize=False
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(self, x: Float[Array, "batch 784"], c: float = 1.0) -> Float[Array, "batch 10"]:
        # FHCNN-style: treat input as spatial coords, compute time from constraint
        # time = sqrt(||space||^2 + 1/c)
        time_coord = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x], axis=-1)  # (batch, 785)

        # Hyperbolic layers
        x = self.hyp1(x, c)  # (batch, 65)
        x = self.hyp2(x, c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class HTCHybrid(nnx.Module):
    """HTC with Euclidean embedding.

    Architecture:
        Input (784) → Euclidean Linear(784→32) + ReLU
                   → Project to Hyperboloid (32→33)
                   → HTC Layer 1 (33→65)
                   → HTC Layer 2 (65→65)
                   → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Linear(784, 32, rngs=rngs)  # Euclidean embedding
        self.hyp1 = HTCLinear(
            in_features=33,
            out_features=64,  # Spatial dimension (output will be 65 with time)
            rngs=rngs,
            use_bias=True,
        )
        self.hyp2 = HTCLinear(
            in_features=65,
            out_features=64,  # Spatial dimension (output will be 65 with time)
            rngs=rngs,
            use_bias=True,
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(self, x: Float[Array, "batch 784"], c: float = 1.0) -> Float[Array, "batch 10"]:
        # Embed in Euclidean space
        x = jax.nn.relu(self.embed(x))  # (batch, 32)

        # Project to hyperboloid
        x = jax.vmap(lambda v: hyperboloid.expmap_0(jnp.concatenate([jnp.zeros(1), v]), c))(x)  # (batch, 33)

        # Hyperbolic layers (HTC uses c_in and c_out)
        x = self.hyp1(x, c_in=c, c_out=c)  # (batch, 65)
        x = self.hyp2(x, c_in=c, c_out=c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class HTCDirect(nnx.Module):
    """HTC with direct projection.

    Architecture:
        Input (784) → Constraint-based projection to Hyperboloid (784→785)
                   → HTC Layer 1 (785→65)
                   → HTC Layer 2 (65→65)
                   → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        # No Euclidean embedding
        self.hyp1 = HTCLinear(
            in_features=785,
            out_features=64,  # Spatial dimension (output will be 65 with time)
            rngs=rngs,
            use_bias=True,
        )
        self.hyp2 = HTCLinear(
            in_features=65,
            out_features=64,  # Spatial dimension (output will be 65 with time)
            rngs=rngs,
            use_bias=True,
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(self, x: Float[Array, "batch 784"], c: float = 1.0) -> Float[Array, "batch 10"]:
        # FHCNN-style: treat input as spatial coords, compute time from constraint
        # time = sqrt(||space||^2 + 1/c)
        time_coord = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x], axis=-1)  # (batch, 785)

        # Hyperbolic layers
        x = self.hyp1(x, c_in=c, c_out=c)  # (batch, 65)
        x = self.hyp2(x, c_in=c, c_out=c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


# ==============================================================================
# Training Utilities
# ==============================================================================


def loss_fn(model: nnx.Module, x: Array, y: Array, c: float = 1.0) -> Array:
    """Cross-entropy loss."""
    logits = model(x, c)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x: Array, y: Array, c: float = 1.0) -> Array:
    """Single training step."""
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y, c)
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
        logits = model(x, c)
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
    _ = train_step(model, optimizer, x_dummy, y_dummy, 1.0)
    metrics["compile_time"] = time.perf_counter() - compile_start

    print(f"  Compilation time: {metrics['compile_time']:.3f}s")

    # Training loop (5 epochs)
    for epoch in range(5):
        epoch_metrics = train_epoch(model, optimizer, train_data, c=1.0)
        val_acc = evaluate(model, test_data, c=1.0)

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


def plot_comparison(results: dict[str, dict[str, Any]]):
    """Generate comparison plots.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to their metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
    plt.savefig("results/mnist_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlots saved to results/mnist_comparison.png")


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
    print(f"{'Model':<15} {'Params':<12} {'Memory(MB)':<12} {'Time(s)':<10} {'Final Acc':<12} {'Compile(s)':<12}")
    print("-" * 90)

    for name, m in results.items():
        print(
            f"{name:<15} {m['parameters']:<12,} {m['memory_mb']:<12.2f} "
            f"{m['total_time']:<10.2f} {m['final_accuracy']:<12.4f} "
            f"{m['compile_time']:<12.3f}"
        )

    print("=" * 90)


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Run all benchmarks and generate comparison."""
    print("=" * 60)
    print("MNIST Layer Comparison Benchmark")
    print("=" * 60)
    print("\nLoading MNIST data...")
    train_data, test_data = load_mnist_data()

    results = {}
    models = [
        (FHCNNHybrid, "FHCNN-Hybrid"),
        (FHCNNDirect, "FHCNN-Direct"),
        (HTCHybrid, "HTC-Hybrid"),
        (HTCDirect, "HTC-Direct"),
    ]

    for model_class, name in models:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {name}")
        print("=" * 60)
        results[name] = benchmark_model(model_class, name, train_data, test_data, seed=42)

    # Save results
    print("\nSaving results...")
    with open("results/mnist_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/mnist_benchmark_results.json")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(results)

    # Print summary table
    print_summary_table(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
