"""Tiny Shakespeare benchmark for causal hyperbolic attention.

Compares four autoregressive character-level language models:
- EuclideanTransformer: Standard Euclidean causal transformer (baseline)
- HypLinearTransformer: Hyperbolic transformer with linear attention
- HypSoftmaxTransformer: Hyperbolic transformer with softmax attention
- HypFullTransformer: Hyperbolic transformer with full Lorentzian attention

All models use causal masking so position n only attends to positions m <= n.

Metrics: train_loss, val_loss, val_perplexity, val_accuracy, time, params, memory

Run with:
    uv run python benchmarks/bench_shakespeare_attention.py [OPTIONS]

Examples:
    uv run python benchmarks/bench_shakespeare_attention.py --all
    uv run python benchmarks/bench_shakespeare_attention.py --euclidean --hyp-softmax
    uv run python benchmarks/bench_shakespeare_attention.py --all --epochs 5 --seq-len 64

Dimension key:
    B: batch size     N: sequence length   D: spatial/model dim
    A: ambient dim (D+1)   V: vocab size   F: FFN hidden dim
    H: num heads
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
from flax import nnx

from benchmarks.utils import (
    count_parameters,
    estimate_memory_mb,
    load_shakespeare_data,
)
from hyperbolix.nn_layers import (
    HRCLayerNorm,
    HTCLinear,
    HyperbolicFullAttention,
    HyperbolicLinearAttention,
    HyperbolicRoPE,
    HyperbolicSoftmaxAttention,
    hrc_gelu,
    lorentz_residual,
    spatial_to_hyperboloid,
)

# Enable float64 for numerical stability in hyperbolic ops
jax.config.update("jax_enable_x64", True)


# ==============================================================================
# Euclidean Transformer (baseline)
# ==============================================================================


class EuclideanCausalAttention(nnx.Module):
    """Multi-head causal self-attention in Euclidean space."""

    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.w_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_o = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, x_BND):
        B, N, D = x_BND.shape
        H, K = self.num_heads, self.head_dim

        q_BNHK = self.w_q(x_BND).reshape(B, N, H, K)  # (B, N, H, K)
        k_BNHK = self.w_k(x_BND).reshape(B, N, H, K)
        v_BNHK = self.w_v(x_BND).reshape(B, N, H, K)

        scores_BHNM = jnp.einsum("bnhk,bmhk->bhnm", q_BNHK, k_BNHK) / jnp.sqrt(float(K))
        mask_NM = jnp.tril(jnp.ones((N, N), dtype=jnp.bool_))
        scores_BHNM = jnp.where(mask_NM[None, None, :, :], scores_BHNM, -1e9)
        attn_BHNM = jax.nn.softmax(scores_BHNM, axis=-1)

        out_BNHK = jnp.einsum("bhnm,bmhk->bnhk", attn_BHNM, v_BNHK)
        out_BND = out_BNHK.reshape(B, N, D)
        return self.w_o(out_BND)


class EuclideanTransformerBlock(nnx.Module):
    """Pre-norm transformer block: LN → Attention → residual → LN → FFN → residual."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = EuclideanCausalAttention(d_model, num_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.ff2 = nnx.Linear(d_ff, d_model, rngs=rngs)

    def __call__(self, x_BND):
        x_BND = x_BND + self.attn(self.ln1(x_BND))
        x_BND = x_BND + self.ff2(jax.nn.gelu(self.ff1(self.ln2(x_BND))))
        return x_BND


class EuclideanTransformer(nnx.Module):
    """Euclidean causal transformer for character-level language modeling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        seq_len: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.token_embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos_embed = nnx.Embed(seq_len, d_model, rngs=rngs)
        self.blocks = nnx.List([EuclideanTransformerBlock(d_model, num_heads, d_ff, rngs=rngs) for _ in range(num_layers)])
        self.ln_f = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)
        self.seq_len = seq_len

    def __call__(self, x_BN):
        _B, N = x_BN.shape
        positions_N = jnp.arange(N)
        h_BND = self.token_embed(x_BN) + self.pos_embed(positions_N)  # (B, N, D)
        for block in self.blocks:
            h_BND = block(h_BND)
        h_BND = self.ln_f(h_BND)
        return self.head(h_BND)  # (B, N, V)


# ==============================================================================
# Hyperbolic Transformer (shared architecture, parameterized by attention class)
# ==============================================================================


class HypTransformerBlock(nnx.Module):
    """Hyperbolic transformer block with causal attention on the hyperboloid.

    Architecture: HRCLayerNorm → Attention(causal) → lorentz_residual →
                  HRCLayerNorm → HTCLinear(FFN up) → hrc_gelu → HTCLinear(FFN down) → lorentz_residual
    """

    def __init__(
        self,
        attn_cls: type,
        ambient_dim: int,
        spatial_dim: int,
        num_heads: int,
        d_ff: int,
        *,
        rngs: nnx.Rngs,
    ):
        # A = ambient_dim = spatial_dim + 1
        self.ln1 = HRCLayerNorm(spatial_dim, rngs=rngs)
        if attn_cls is HyperbolicLinearAttention:
            self.attn = attn_cls(ambient_dim, spatial_dim, num_heads=num_heads, power=2.0, rngs=rngs)
        else:
            self.attn = attn_cls(ambient_dim, spatial_dim, num_heads=num_heads, rngs=rngs)
        self.ln2 = HRCLayerNorm(spatial_dim, rngs=rngs)
        # FFN: ambient_dim → d_ff (spatial) → d_ff+1 (ambient after spatial_to_hyperboloid)
        # Then d_ff+1 → spatial_dim (spatial) → spatial_dim+1 (ambient)
        self.ff_up = HTCLinear(ambient_dim, d_ff, rngs=rngs)
        self.ff_down = HTCLinear(d_ff + 1, spatial_dim, rngs=rngs)

    def __call__(self, x_BNA, c=1.0):
        # Attention sub-block
        normed_BNA = self.ln1(x_BNA, c_in=c, c_out=c)
        attn_out_BNA = self.attn(normed_BNA, c_in=c, c_attn=c, c_out=c, causal=True)
        # Lorentz residual skip connection (per-token)
        x_BNA = jax.vmap(jax.vmap(lorentz_residual, in_axes=(0, 0, None, None)), in_axes=(0, 0, None, None))(
            x_BNA, attn_out_BNA, 1.0, c
        )

        # FFN sub-block
        normed_BNA = self.ln2(x_BNA, c_in=c, c_out=c)
        ff_BNA = self.ff_up(normed_BNA, c_in=c, c_out=c)  # (B, N, d_ff+1)
        ff_BNA = hrc_gelu(ff_BNA, c_in=c, c_out=c)
        ff_BNA = self.ff_down(ff_BNA, c_in=c, c_out=c)  # (B, N, A)
        x_BNA = jax.vmap(jax.vmap(lorentz_residual, in_axes=(0, 0, None, None)), in_axes=(0, 0, None, None))(
            x_BNA, ff_BNA, 1.0, c
        )
        return x_BNA


class HyperbolicTransformer(nnx.Module):
    """Hyperbolic causal transformer for character-level language modeling.

    Embeds tokens into Euclidean space, projects onto hyperboloid, applies
    HyperbolicRoPE positional encoding, then N transformer blocks with
    causal attention. Final output is projected back to Euclidean vocab logits.
    """

    def __init__(
        self,
        attn_cls: type,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        seq_len: int,
        c: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.c = c
        self.d_model = d_model
        ambient_dim = d_model + 1  # A = D + 1

        self.token_embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.rope = HyperbolicRoPE(dim=d_model, max_seq_len=seq_len)
        self.blocks = nnx.List(
            [HypTransformerBlock(attn_cls, ambient_dim, d_model, num_heads, d_ff, rngs=rngs) for _ in range(num_layers)]
        )
        self.ln_f = HRCLayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)
        self.seq_len = seq_len

    def __call__(self, x_BN):
        c = self.c
        _B, N = x_BN.shape

        # Embed tokens in Euclidean space, then project to hyperboloid
        emb_BND = self.token_embed(x_BN)  # (B, N, D)
        h_BNA = spatial_to_hyperboloid(emb_BND, c, c)  # (B, N, A)

        # Apply HyperbolicRoPE positional encoding
        positions_N = jnp.arange(N, dtype=jnp.float32)
        h_BNA = self.rope(h_BNA, positions_N, c)  # (B, N, A)

        # Transformer blocks
        for block in self.blocks:
            h_BNA = block(h_BNA, c)

        # Final layer norm
        h_BNA = self.ln_f(h_BNA, c_in=c, c_out=c)

        # Extract spatial components → Euclidean linear head
        h_BND = h_BNA[..., 1:]  # (B, N, D)
        return self.head(h_BND)  # (B, N, V)


# ==============================================================================
# Training Utilities
# ==============================================================================


def shakespeare_loss_fn(model, x_BN, y_BN):
    """Next-token cross-entropy loss over all positions."""
    logits_BNV = model(x_BN)  # (B, N, V)
    return optax.softmax_cross_entropy_with_integer_labels(logits_BNV, y_BN).mean()


@nnx.jit
def shakespeare_train_step(model, optimizer, x_BN, y_BN):
    """Single training step with gradient update."""
    loss, grads = nnx.value_and_grad(shakespeare_loss_fn)(model, x_BN, y_BN)
    optimizer.update(model, grads)
    return loss


def shakespeare_evaluate(model, val_iter):
    """Evaluate model on validation set.

    Parameters
    ----------
    model : nnx.Module
        Model to evaluate
    val_iter : iterable
        Iterable of batched dicts with 'input' and 'target' keys

    Returns
    -------
    dict with keys: val_loss, val_accuracy, val_perplexity
    """
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch in val_iter:
        x_BN = jnp.array(batch["input"])
        y_BN = jnp.array(batch["target"])
        logits_BNV = model(x_BN)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_BNV, y_BN).mean()
        preds_BN = jnp.argmax(logits_BNV, axis=-1)
        total_loss += float(loss)
        total_correct += int(jnp.sum(preds_BN == y_BN))
        total_tokens += y_BN.size
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = float(jnp.exp(jnp.minimum(avg_loss, 20.0)))  # cap to avoid overflow

    return {"val_loss": avg_loss, "val_accuracy": accuracy, "val_perplexity": perplexity}


# ==============================================================================
# Benchmark Runner
# ==============================================================================


def run_shakespeare_benchmark(
    model_name: str,
    model: nnx.Module,
    train_ds,
    val_ds,
    *,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 64,
    seq_len: int = 128,
    learning_rate: float = 3e-4,
) -> dict[str, Any]:
    """Run full Shakespeare benchmark for one model."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    n_params = count_parameters(model)
    mem_mb = estimate_memory_mb(model)
    print(f"  Parameters: {n_params:,}  |  Memory: {mem_mb:.2f} MB")

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # JIT compilation timing with dummy data
    x_dummy = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    y_dummy = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    compile_start = time.perf_counter()
    _ = shakespeare_train_step(model, optimizer, x_dummy, y_dummy)
    compile_time = time.perf_counter() - compile_start
    print(f"  Compile time: {compile_time:.2f}s")

    metrics = {
        "model": model_name,
        "parameters": n_params,
        "memory_mb": mem_mb,
        "compile_time": compile_time,
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "val_perplexities": [],
        "epoch_times": [],
    }

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        train_iter = train_ds.shuffle(seed=seed + epoch).batch(batch_size, drop_remainder=True).to_iter_dataset()
        epoch_losses = []
        for batch in train_iter:
            x_BN = jnp.array(batch["input"])
            y_BN = jnp.array(batch["target"])
            loss = shakespeare_train_step(model, optimizer, x_BN, y_BN)
            epoch_losses.append(float(loss))

        epoch_time = time.perf_counter() - epoch_start
        train_loss = float(np.mean(epoch_losses))

        val_iter = val_ds.batch(batch_size, drop_remainder=False).to_iter_dataset()
        val_metrics = shakespeare_evaluate(model, val_iter)

        metrics["train_losses"].append(train_loss)
        metrics["val_losses"].append(val_metrics["val_loss"])
        metrics["val_accuracies"].append(val_metrics["val_accuracy"])
        metrics["val_perplexities"].append(val_metrics["val_perplexity"])
        metrics["epoch_times"].append(epoch_time)

        print(
            f"  Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={val_metrics['val_accuracy']:.4f}  val_ppl={val_metrics['val_perplexity']:.1f}  "
            f"time={epoch_time:.1f}s"
        )

    metrics["total_time"] = sum(metrics["epoch_times"])
    metrics["final_val_loss"] = metrics["val_losses"][-1]
    metrics["final_val_accuracy"] = metrics["val_accuracies"][-1]
    metrics["final_val_perplexity"] = metrics["val_perplexities"][-1]

    return metrics


# ==============================================================================
# Visualization
# ==============================================================================


def plot_shakespeare_comparison(results: dict[str, dict[str, Any]], output_path: str):
    """Generate comparison plots for Shakespeare benchmark."""
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    names = list(results.keys())
    num_epochs = len(next(iter(results.values()))["train_losses"])
    x_epochs = list(range(1, num_epochs + 1))

    # Plot 1: Training loss
    ax = axes[0, 0]
    for name in names:
        ax.plot(x_epochs, results[name]["train_losses"], marker="o", label=name, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation loss
    ax = axes[0, 1]
    for name in names:
        ax.plot(x_epochs, results[name]["val_losses"], marker="o", label=name, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation perplexity
    ax = axes[1, 0]
    for name in names:
        ax.plot(x_epochs, results[name]["val_perplexities"], marker="o", label=name, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Time per epoch + parameters
    ax = axes[1, 1]
    x_pos = np.arange(len(names))
    times = [results[n]["total_time"] / num_epochs for n in names]
    ax.bar(x_pos, times, color="C0")
    ax.set_ylabel("Avg Time / Epoch (s)")
    ax.set_title("Training Speed")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to {output_path}")


def print_shakespeare_summary(results: dict[str, dict[str, Any]]):
    """Print summary table for Shakespeare benchmark."""
    print("\n" + "=" * 100)
    print("SHAKESPEARE BENCHMARK SUMMARY")
    print("=" * 100)
    cols = f"{'Model':<22} {'Params':<10} {'Mem(MB)':<10} {'ValLoss':<10}"
    cols += f" {'ValAcc':<10} {'ValPPL':<10} {'Time(s)':<10} {'Compile':<10}"
    header = cols
    print(header)
    print("-" * 100)

    for name, m in results.items():
        print(
            f"{name:<22} {m['parameters']:<10,} {m['memory_mb']:<10.2f} "
            f"{m['final_val_loss']:<10.4f} {m['final_val_accuracy']:<10.4f} "
            f"{m['final_val_perplexity']:<10.1f} {m['total_time']:<10.1f} "
            f"{m['compile_time']:<10.2f}"
        )

    print("=" * 100)


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny Shakespeare benchmark comparing causal attention variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--euclidean", action="store_true", help="Run Euclidean transformer baseline")
    parser.add_argument("--hyp-linear", action="store_true", help="Run hyperbolic linear attention transformer")
    parser.add_argument("--hyp-softmax", action="store_true", help="Run hyperbolic softmax attention transformer")
    parser.add_argument("--hyp-full", action="store_true", help="Run hyperbolic full Lorentzian attention transformer")
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension (default: 64)")
    parser.add_argument("--num-heads", type=int, default=2, help="Number of attention heads (default: 2)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers (default: 2)")
    parser.add_argument("--d-ff", type=int, default=128, help="FFN hidden dimension (default: 128)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--curvature", type=float, default=1.0, help="Hyperbolic curvature (default: 1.0)")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    run_all = args.all or not (args.euclidean or args.hyp_linear or args.hyp_softmax or args.hyp_full)

    print("=" * 60)
    print("Tiny Shakespeare Causal Attention Benchmark")
    print("=" * 60)
    print(f"  d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}, d_ff={args.d_ff}")
    print(f"  seq_len={args.seq_len}, batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print(f"  curvature={args.curvature}, seed={args.seed}")

    # Load data
    train_ds, val_ds, vocab_size, _char2idx = load_shakespeare_data(args.seq_len)

    results = {}
    rngs = nnx.Rngs(params=args.seed, dropout=args.seed + 1)

    # Euclidean baseline
    if args.euclidean or run_all:
        model = EuclideanTransformer(
            vocab_size, args.d_model, args.num_heads, args.num_layers, args.d_ff, args.seq_len, rngs=rngs
        )
        results["Euclidean"] = run_shakespeare_benchmark(
            "Euclidean",
            model,
            train_ds,
            val_ds,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
        )

    # Hyperbolic variants
    hyp_models = [
        (args.hyp_linear or run_all, "Hyp-Linear", HyperbolicLinearAttention),
        (args.hyp_softmax or run_all, "Hyp-Softmax", HyperbolicSoftmaxAttention),
        (args.hyp_full or run_all, "Hyp-Full", HyperbolicFullAttention),
    ]

    for should_run, name, attn_cls in hyp_models:
        if not should_run:
            continue
        rngs = nnx.Rngs(params=args.seed, dropout=args.seed + 1)
        model = HyperbolicTransformer(
            attn_cls,
            vocab_size,
            args.d_model,
            args.num_heads,
            args.num_layers,
            args.d_ff,
            args.seq_len,
            c=args.curvature,
            rngs=rngs,
        )
        results[name] = run_shakespeare_benchmark(
            name,
            model,
            train_ds,
            val_ds,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
        )

    # Save results
    output_json = "benchmarks/shakespeare_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Plots
    if len(results) > 1:
        plot_shakespeare_comparison(results, "benchmarks/shakespeare_comparison.png")

    # Summary
    print_shakespeare_summary(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
