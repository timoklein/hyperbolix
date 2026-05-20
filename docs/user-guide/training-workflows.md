# Training Workflows

A hyperbolic training loop is just a normal Flax NNX + Optax loop. The
specifics that differ from Euclidean training are small: pass curvature at
call time, optionally project, and pick reasonable hyperparameters per
manifold. This page shows a minimal working loop, points to the benchmark
files for advanced patterns, and lists the failures you'll actually hit.

## Minimal Training Loop

The following classifier trains on synthetic data and is verified to drop the
loss from ~2.5 to <0.05 in 200 steps. Copy, run, modify.

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HTCLinear, HRCLayerNorm, HypRegressionHyperboloid


class HTCClassifier(nnx.Module):
    def __init__(self, ambient: int, num_classes: int, *, rngs: nnx.Rngs):
        spatial = ambient - 1                                 # HTC out_features is spatial
        self.manifold = Hyperboloid(c=1.0)
        self.fc1 = HTCLinear(in_features=ambient, out_features=spatial, rngs=rngs)
        self.norm = HRCLayerNorm(num_features=spatial, rngs=rngs)
        self.fc2 = HTCLinear(in_features=ambient, out_features=spatial, rngs=rngs)
        self.head = HypRegressionHyperboloid(
            manifold_module=self.manifold,
            in_dim=ambient, out_dim=num_classes, rngs=rngs,
        )

    def __call__(self, x):
        h = self.fc1(x, c_in=1.0, c_out=1.0)
        h = self.norm(h, c_in=1.0, c_out=1.0)
        h = self.fc2(h, c_in=1.0, c_out=1.0)
        return self.head(h, c=1.0)


def make_synthetic(key, n, spatial, num_classes, c=1.0):
    k1, k2 = jax.random.split(key)
    feats = jax.random.normal(k1, (n, spatial)) * 0.3        # small-norm Euclidean features
    time = jnp.sqrt(jnp.sum(feats**2, axis=-1, keepdims=True) + 1.0 / c)
    x = jnp.concatenate([time, feats], axis=-1)              # constraint projection -> ambient
    w = jax.random.normal(k2, (spatial, num_classes))
    y = jnp.argmax(feats @ w, axis=-1)
    return x, y


# Setup
spatial, ambient, num_classes = 32, 33, 10
rngs = nnx.Rngs(0)
x, y = make_synthetic(jax.random.PRNGKey(42), n=256, spatial=spatial, num_classes=num_classes)

model = HTCClassifier(ambient=ambient, num_classes=num_classes, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


def loss_fn(model, x, y):
    return optax.softmax_cross_entropy_with_integer_labels(model(x), y).mean()


@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


for step in range(200):
    loss = train_step(model, optimizer, x, y)
    if step % 40 == 0:
        print(f"step {step:3d}: loss = {loss:.4f}")
```

Two things worth pointing out:

- **Curvature is passed at call time**, not stored on the layer. That's why
  the `__call__` signatures take `c_in` / `c_out` (or `c`).
- **Inputs are constructed as ambient points** via the constraint projection
  (`x_0 = sqrt(||spatial||^2 + 1/c)`). This is Pattern B from the
  [Manifolds guide](manifolds.md#going-euclidean-manifold) — appropriate for
  inputs that aren't guaranteed small-norm. Use `expmap_0` instead when your
  features come from a small-norm Euclidean embedding.

## Benchmark Map

For end-to-end examples on real data, the benchmarks in `benchmarks/` are the
authoritative reference. Each file is a self-contained training script.

| File | What it demonstrates |
|---|---|
| `bench_mnist_hyperboloid.py` | Hyperboloid CNN — both `FHCNNHybrid` (Euclidean stem → `expmap_0` → hyperbolic head) and `FullyHyperbolicCNN_*` (constraint projection at input, hyperbolic throughout) |
| `bench_mnist_poincare.py` | Poincaré CNN with per-layer learnable curvature at `init_c=0.1` (van Spengler 2023 recipe) |
| `bench_mnist_pv.py` | PV CNN — unconstrained $\mathbb{R}^n$ training with no projection step |
| `bench_shakespeare_attention.py` | Hyperbolic transformer (causal attention) with all four variants: Euclidean baseline + softmax / full / linear hyperbolic attention |

The micro-benchmarks (`bench_manifolds.py`, `bench_nn_layers.py`) test
per-operation performance, not training workflows.

## Hyperparameters That Actually Matter

Most defaults from Euclidean training transfer. The exceptions:

| Hyperparameter | Recommendation |
|---|---|
| **Learning rate** | Adam `1e-3` works as a starting point for HTC / HCat / PP / PV. For deep Poincaré nets at `c=0.1`, try `5e-4`–`1e-3` |
| **Gradient clipping** | Norm-clip at `1.0` is a safe default; not strictly required for most setups |
| **Curvature init** | Hyperboloid: `c=1.0`. PV: `c=1.0`. Poincaré (deep nets): `c=0.1` with `learnable_curvature(0.1)` per layer |
| **Float precision** | `float32` is fine for Hyperboloid and PV at modest depths. Use `Poincare(dtype=jnp.float64)` for high curvature, deep nets, or boundary-near training. See [Numerical Stability](numerical-stability.md) |
| **Layer init** | Keep each family's default (HTC uses small uniform, PP uses scaled normal, etc.). Standard He/Xavier is too large for hyperbolic layers — see the [NN Layers guide](nn-layers.md#initialization-scales) |

## Common Training Failures

| Symptom | Likely cause | Fix |
|---|---|---|
| NaN loss in the first step | Init too large, or `version_idx` left dynamic under JIT | Verify per-family init defaults; bind `version_idx` with `functools.partial` before JIT |
| Stuck at random-chance accuracy (Poincaré) | Curvature too high — features hit the boundary where the conformal factor collapses | Use `Poincare(c=0.1)` with `learnable_curvature(0.1)` per layer |
| Loss explodes after some warmup (Hyperboloid float32) | Lorentz-constraint drift accumulated past the layer's tolerance | Periodically call `manifold.proj(x, c)` between blocks, or switch to float64 |
| Loss is decreasing on training data but eval is wild | Float precision mismatch between train and eval | Make sure both code paths use the same dtype on inputs and manifold |
| Using `riemannian_adam` for `HTC` / `FGG` / `PP` / `PV` layers | Wrong optimizer for Euclidean-weighted layers | Switch to `optax.adam` — see [Optimizers Guide](optimizers.md) |

## See Also

- **[NN Layers Guide](nn-layers.md)** — choosing layers, composition patterns, channel conventions.
- **[Optimizers Guide](optimizers.md)** — when (rarely) you need Riemannian
  optimization.
- **[Numerical Stability Guide](numerical-stability.md)** — float32 vs.
  float64, clamping strategies.
- **[Manifolds Guide](manifolds.md)** — manifold operations, curvature
  workflows, common pitfalls.
- **`benchmarks/`** — full training scripts for hyperboloid CNN, Poincaré CNN,
  PV CNN, and hyperbolic transformer on MNIST / Shakespeare.
