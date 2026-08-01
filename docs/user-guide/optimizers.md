# Riemannian Optimizers User Guide

Almost no Hyperbolix user needs a Riemannian optimizer. This page exists to tell
you when you're in the rare case that does.

For per-optimizer signatures, see the
[Optimizers API reference](../api-reference/optimizers.md).

## TL;DR

**Use `optax.adam` (or any Euclidean Optax optimizer).** That includes:

- All modern NN layers — `HTCLinear`, `FGGLinear`, `HypConv2DHyperboloid*`,
  `HypLinearHyperboloidPLFC`, `HypLinearPoincarePP`, `HypLinearPV`, all
  attention / normalization / regression heads.
- Learnable curvature (via the `LearnableCurvature` module).

**The single exception**: the legacy Ganea-style Poincaré layers
(`HypLinearPoincare`, `HypRegressionPoincare`) parameterize their **bias**
*directly on the Poincaré ball* (the kernel stays Euclidean), so they need
`riemannian_adam` (or `riemannian_sgd`). You can usually just migrate to the
`PP` equivalents and stay with `optax.adam`.

## When to Use What

| Setup | Optimizer |
|---|---|
| Any modern hyperbolic NN (HTC, FGG, HCat, PP, HRC, PV) | `optax.adam` |
| Learnable curvature (any manifold) | `optax.adam` |
| Mixed Euclidean / hyperbolic networks | `optax.adam` |
| Legacy `HypLinearPoincare` / `HypRegressionPoincare` | `riemannian_adam` |

That's the whole decision.

## Why Modern Layers Don't Need Riemannian Optimization

Modern hyperbolic layers (HTC, FGG, HNN++, HCat, PV) store their weights as
**Euclidean tensors** and apply the hyperbolic transformation inside the
forward pass. The weights themselves never live on a manifold — they're just
flat parameter tensors with Euclidean gradients. Standard Adam handles them
correctly.

```python
import optax
from flax import nnx

# Standard setup — works for every modern layer family
model = HTCClassifier(...)
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
```

## Learnable Curvature Is Also Euclidean

A common confusion: surely learnable curvature needs Riemannian optimization?
No. The `LearnableCurvature` module stores a plain Euclidean `nnx.Param`
that is reparameterized at call time — via `"softplus"` (default,
`c = softplus(raw)`), `"log"` (`c = exp(raw)`, scale-invariant gradient), or
`"identity"` (`c = raw`, an unconstrained **signed** curvature for the
`Stereographic` manifold) — to produce the curvature value. It gets a normal
Euclidean gradient and a normal Adam update — there's nothing to project,
nothing on a manifold.

```python
from hyperbolix import LearnableCurvature

class Model(nnx.Module):
    def __init__(self, ...):
        self.curvature = LearnableCurvature(init_c=1.0)  # raw is a plain nnx.Param
        ...

optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
# Curvature is updated by Adam alongside all other parameters.
```

## The Legacy Ganea Exception

`HypLinearPoincare` and `HypRegressionPoincare` (Ganea et al. 2018) store
their **bias** as a point on the Poincaré ball (the kernel stays a plain
Euclidean `nnx.Param`). Each gradient step on that bias needs to respect the
ball constraint, which is exactly what `riemannian_adam` provides. The
constructors already tag the bias with `ManifoldParam` internally, so using
them is just:

```python
from hyperbolix.optim import riemannian_adam
from hyperbolix.nn_layers import HypLinearPoincare

model = HypLinearPoincare(manifold_module=poincare, in_dim=32, out_dim=16, rngs=rngs)
optimizer = nnx.Optimizer(model, riemannian_adam(1e-3), wrt=nnx.Param)
```

No manual tagging step is required for these built-in layers. If you're
writing your **own** manifold-valued parameter instead — e.g. a hyperbolic
embedding table — that's when you reach for `ManifoldParam` (or the
`mark_manifold_param` convenience wrapper around an existing `nnx.Param`)
yourself:

```python
from hyperbolix.optim import ManifoldParam

# Tag your own manifold-valued parameter during model init
self.embedding = ManifoldParam(
    init_value, manifold=poincare, curvature=0.1,
)
```

The optimizer walks the model state, applies Riemannian updates wherever it
finds a `ManifoldParam` tag, and falls back to standard Adam everywhere else.
You don't need separate optimizers for the Euclidean and manifold parts.

!!! note "Storage vs. compute dtype"
    The Riemannian update math (`egrad2rgrad`, `expmap`, `ptransp`) runs in
    the **manifold's dtype**, but the returned parameter updates and the
    momentum buffers are cast back to the **parameter's storage dtype**. A
    float32 `ManifoldParam` paired with a `Poincare(dtype=jnp.float64)`
    manifold therefore gets float64-precision geometry while its weights and
    optimizer state stay float32. See
    [Numerical Stability](numerical-stability.md#storage-vs-compute-dtype).

!!! tip "Prefer the `PP` migration"
    If you're picking the legacy Ganea layer for new work, almost always swap
    it for `HypLinearPoincarePP` / `HypRegressionPoincarePP` instead. The
    `PP` variants compute the same operation with Euclidean weights, removing
    the need for Riemannian optimization entirely. The numerics are typically
    cleaner and the optimizer setup is simpler.

## Common Pitfalls

1. **Using `riemannian_adam` for HTC / FGG / PP / PV layers.** Adds compute
   for no benefit (and occasionally slightly worse numerics). Use
   `optax.adam`.
2. **Forgetting to tag a hand-rolled manifold-valued parameter.** The
   built-in legacy layers (`HypLinearPoincare`, `HypRegressionPoincare`)
   already self-tag their bias with `ManifoldParam` in `__init__` — there's
   nothing to do for them. The tag matters only when *you* add your own
   manifold-valued parameter (e.g. a hyperbolic embedding table stored as a
   plain `nnx.Param`): without `ManifoldParam` / `mark_manifold_param`,
   `riemannian_adam` falls back to Euclidean Adam on it and the values drift
   off the ball within a few steps.
3. **Expecting `riemannian_adam` to need a separate optimizer for the
   Euclidean parts.** It doesn't. One optimizer with `wrt=nnx.Param` handles
   everything; dispatch is via the `ManifoldParam` tag.

## See Also

- **[API Reference: Optimizers](../api-reference/optimizers.md)** — full
  signatures for `riemannian_adam`, `riemannian_sgd`, `ManifoldParam`,
  `mark_manifold_param`.
- **[NN Layers Guide](nn-layers.md)** — layer-family overview, including
  which families use Euclidean weights.
- **[Manifolds Guide](manifolds.md#working-with-curvature)** — learnable
  curvature mechanics.
