# Riemannian Optimizers User Guide

Almost no Hyperbolix user needs a Riemannian optimizer. This page exists to tell
you when you're in the rare case that does.

For per-optimizer signatures, see the
[Optimizers API reference](../api-reference/optimizers.md).

## TL;DR

**Use `optax.adam` (or any Euclidean Optax optimizer).** That includes:

- All modern NN layers — `HTCLinear`, `FGGLinear`, `HypConv2DHyperboloid*`,
  `HypLinearHyperboloidPP`, `HypLinearPoincarePP`, `HypLinearPV`, all
  attention / normalization / regression heads.
- Learnable curvature (`Hyperboloid(c=1.0, learnable=True)` and the same on
  Poincaré / PV).

**The single exception**: the legacy Ganea-style Poincaré layers
(`HypLinearPoincare`, `HypRegressionPoincare`) parameterize weights *directly
on the Poincaré ball*, so they need `riemannian_adam` (or `riemannian_sgd`).
You can usually just migrate to the `PP` equivalents and stay with
`optax.adam`.

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
No. Curvature is stored as a single Euclidean `nnx.Param` (`_c_raw`) and
reparameterized through `softplus` to stay positive. The `_c_raw` parameter
gets a normal Euclidean gradient and a normal Adam update — there's nothing
to project, nothing on a manifold.

```python
manifold = Hyperboloid(c=1.0, learnable=True)   # _c_raw is a plain nnx.Param
# ... build model ...
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
# Curvature is updated by Adam alongside all other parameters.
```

## The Legacy Ganea Exception

`HypLinearPoincare` and `HypRegressionPoincare` (Ganea et al. 2018) store
their weights as **points on the Poincaré ball**. Each gradient step needs to
respect the ball constraint, which is exactly what `riemannian_adam`
provides. To use them:

```python
from hyperbolix.optim import riemannian_adam, mark_manifold_param

# Tag manifold-valued parameters during model init
self.weight = mark_manifold_param(
    nnx.Param(init_value), manifold=poincare, curvature=0.1,
)

# Riemannian optimizer auto-dispatches: tagged params get Riemannian updates,
# everything else gets Euclidean Adam.
optimizer = nnx.Optimizer(model, riemannian_adam(1e-3), wrt=nnx.Param)
```

The optimizer walks the model state, applies Riemannian updates wherever it
finds a `ManifoldParam` tag, and falls back to standard Adam everywhere else.
You don't need separate optimizers for the Euclidean and manifold parts.

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
2. **Forgetting `mark_manifold_param` on legacy layer weights.** Without the
   tag, `riemannian_adam` falls back to Euclidean Adam on a manifold-valued
   tensor — the weights drift off the ball within a few steps.
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
