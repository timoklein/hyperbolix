"""Shared utilities for MNIST benchmark scripts.

Common training, evaluation, profiling, and visualization functions
used by both hyperboloid and Poincaré MNIST benchmarks.
"""

import time
from typing import Any

import grain.python as grain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets import load_dataset
from flax import nnx
from jaxtyping import Array

from hyperbolix.optim import riemannian_sgd

# ==============================================================================
# Data Loading
# ==============================================================================


class MNISTPreprocess(grain.MapTransform):
    """Flatten 28x28 PIL image to 784 float32 and normalize; cast label to int32."""

    def map(self, sample):
        image = np.array(sample["image"]).flatten().astype(np.float32) / 255.0
        label = np.int32(sample["label"])
        return {"image": image, "label": label}


def load_mnist_data():
    """Load MNIST using HuggingFace datasets and return grain MapDatasets.

    Returns
    -------
    train_data : grain.MapDataset
        Training dataset (preprocessed, ready for .shuffle/.batch/.to_iter_dataset)
    test_data : grain.MapDataset
        Test dataset (preprocessed, ready for .batch/.to_iter_dataset)
    """
    print("  Loading MNIST dataset from HuggingFace...")
    dataset = load_dataset("mnist")

    train_data = grain.MapDataset.source(dataset["train"]).map(MNISTPreprocess())
    test_data = grain.MapDataset.source(dataset["test"]).map(MNISTPreprocess())

    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    return train_data, test_data


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


def train_epoch(model: nnx.Module, optimizer: nnx.Optimizer, train_iter, c: float = 1.0) -> dict[str, float]:
    """Train for one epoch.

    Parameters
    ----------
    model : nnx.Module
        Model to train
    optimizer : nnx.Optimizer
        Optimizer for parameter updates
    train_iter : iterable
        Iterable of batched dicts with 'image' and 'label' keys
    c : float
        Curvature parameter

    Returns
    -------
    metrics : dict
        Dictionary with 'loss' and 'time' keys
    """
    epoch_start = time.perf_counter()

    losses = []
    for batch in train_iter:
        x = jnp.array(batch["image"])
        y = jnp.array(batch["label"])
        loss = train_step(model, optimizer, x, y, c)
        losses.append(float(loss))

    epoch_time = time.perf_counter() - epoch_start
    return {"loss": np.mean(losses), "time": epoch_time}


def evaluate(model: nnx.Module, eval_iter, c: float = 1.0) -> float:
    """Compute accuracy on dataset.

    Parameters
    ----------
    model : nnx.Module
        Model to evaluate
    eval_iter : iterable
        Iterable of batched dicts with 'image' and 'label' keys
    c : float
        Curvature parameter

    Returns
    -------
    accuracy : float
        Classification accuracy
    """
    correct = 0
    total = 0

    for batch in eval_iter:
        x = jnp.array(batch["image"])
        y = jnp.array(batch["label"])
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
    model_class: type[nnx.Module],
    model_name: str,
    train_data,
    test_data,
    seed: int = 42,
    c: float = 1.0,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Run full benchmark for one model variant.

    Parameters
    ----------
    model_class : type[nnx.Module]
        Model class to benchmark
    model_name : str
        Name for logging
    train_data : grain.MapDataset
        Training dataset (preprocessed)
    test_data : grain.MapDataset
        Test dataset (preprocessed)
    seed : int
        Random seed for reproducibility
    c : float
        Curvature parameter
    batch_size : int
        Batch size for training and evaluation

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
    x_dummy = jnp.ones((batch_size, 784))
    y_dummy = jnp.zeros(batch_size, dtype=jnp.int32)

    compile_start = time.perf_counter()
    _ = train_step(model, optimizer, x_dummy, y_dummy, c)
    metrics["compile_time"] = time.perf_counter() - compile_start

    print(f"  Compilation time: {metrics['compile_time']:.3f}s")

    # Training loop (5 epochs)
    for epoch in range(5):
        train_iter = train_data.shuffle(seed=seed + epoch).batch(batch_size, drop_remainder=True).to_iter_dataset()
        epoch_metrics = train_epoch(model, optimizer, train_iter, c=c)

        eval_iter = test_data.batch(batch_size, drop_remainder=False).to_iter_dataset()
        val_acc = evaluate(model, eval_iter, c=c)

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
# Shakespeare Data Loading
# ==============================================================================


class SequencePairSource(grain.RandomAccessDataSource):
    """Wraps paired numpy arrays for grain random access."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self._inputs = inputs
        self._targets = targets

    def __getitem__(self, idx):
        return {"input": self._inputs[idx], "target": self._targets[idx]}

    def __len__(self):
        return len(self._inputs)


def load_shakespeare_data(seq_len: int = 128):
    """Load Tiny Shakespeare as character-level sequences using grain.

    Parameters
    ----------
    seq_len : int
        Sequence length for input chunks (targets are shifted by 1).

    Returns
    -------
    train_ds : grain.MapDataset
        Training dataset (ready for .shuffle/.batch/.to_iter_dataset)
    val_ds : grain.MapDataset
        Validation dataset (ready for .batch/.to_iter_dataset)
    vocab_size : int
        Number of unique characters.
    char2idx : dict
        Character to index mapping.
    """
    print("  Loading Tiny Shakespeare dataset...")
    dataset = load_dataset("Trelis/tiny-shakespeare")

    # Concatenate all text rows
    train_text = "\n".join(dataset["train"]["Text"])
    val_text = "\n".join(dataset["test"]["Text"])

    # Character-level tokenization
    chars = sorted(set(train_text + val_text))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}

    print(f"  Vocab size: {vocab_size}, Train chars: {len(train_text)}, Val chars: {len(val_text)}")

    def tokenize_and_chunk(text, sl):
        tokens = np.array([char2idx[ch] for ch in text], dtype=np.int32)
        # Non-overlapping chunks of length seq_len + 1
        n_chunks = len(tokens) // (sl + 1)
        tokens = tokens[: n_chunks * (sl + 1)]
        chunks = tokens.reshape(n_chunks, sl + 1)
        return chunks[:, :-1], chunks[:, 1:]

    train_inputs, train_targets = tokenize_and_chunk(train_text, seq_len)
    val_inputs, val_targets = tokenize_and_chunk(val_text, seq_len)

    print(f"  Train sequences: {train_inputs.shape[0]}, Val sequences: {val_inputs.shape[0]}")

    train_ds = grain.MapDataset.source(SequencePairSource(train_inputs, train_targets))
    val_ds = grain.MapDataset.source(SequencePairSource(val_inputs, val_targets))

    return train_ds, val_ds, vocab_size, char2idx
