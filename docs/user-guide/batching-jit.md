# Batching & JIT Guide

How to batch and compile hyperbolix code.

## Overview

Every manifold method operates on a **single point**, of shape `(dim,)` (or
`(dim+1,)` ambient for the hyperboloid). There are no `axis` or `keepdim`
arguments: batch with `jax.vmap`, compile with `jax.jit`, and keep `version_idx`
static. NN layers are the exception: they batch internally.

## The vmap-Native API

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Single points
x = jnp.array([0.1, 0.2])
y = jnp.array([0.3, 0.4])
distance = poincare.dist(x, y, c=1.0)  # scalar

# Batches: map over the points, broadcast c with None
x_batch = jnp.array([[0.1, 0.2], [0.15, 0.25], [0.05, 0.1]])  # (3, 2)
y_batch = jnp.array([[0.3, 0.4], [0.35, 0.45], [0.2, 0.3]])   # (3, 2)
distances = jax.vmap(poincare.dist, in_axes=(0, 0, None))(x_batch, y_batch, 1.0)
print(distances.shape)  # (3,)

# One fixed point against a batch: close over it
origin = jnp.zeros(2)
distances = jax.vmap(lambda p: poincare.dist(origin, p, c=1.0))(x_batch)

# Batch of tangent vectors at one base point
v_batch = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100, 2))
points = jax.vmap(poincare.expmap, in_axes=(0, None, None))(v_batch, x, 1.0)
print(points.shape)  # (100, 2)
```

`in_axes` has one entry per positional argument: `0` maps over the leading axis,
`None` passes the same value to every element.

## JIT Compilation

### Static vs Dynamic Arguments

`version_idx` selects an op variant. Keep it **static**, so only the selected
variant compiles; a traced index compiles every variant into one `lax.switch`.
The curvature `c` is **dynamic**: a new value reuses the compiled function. A new
input shape or a new static `version_idx` triggers a recompile.

```python
dist_jit = jax.jit(poincare.dist, static_argnames=["version_idx"])

d1 = dist_jit(x, y, c=1.0, version_idx=0)
d2 = dist_jit(x, y, c=2.5, version_idx=0)  # new c: no recompilation
d3 = dist_jit(x, y, c=1.0, version_idx=1)  # new version_idx: recompiles
```

!!! warning "Learnable Curvature"
    Don't mark `c` static: it recompiles for every value and blocks gradients. Learnable curvature comes from a `LearnableCurvature` module called inside the model (see [Manifolds](manifolds.md#working-with-curvature)).

### Combining vmap and jit

Put `jax.jit` outermost: vmap inside, jit around it (or around the whole train step).

```python
from functools import partial

# version_idx is bound before jit, so it stays static
dist_batched = jax.jit(
    jax.vmap(
        partial(poincare.dist, version_idx=poincare.VERSION_MOBIUS_DIRECT),
        in_axes=(0, 0, None),
    )
)
distances = dist_batched(x_batch, y_batch, 1.0)
print(distances.shape)  # (3,)
```

!!! tip "Why jit goes outermost"
    `jax.vmap(jax.jit(f))` gives the same numbers but traces the vmapped
    function again on every call. On CPU (Poincaré `dist`, dim 16) it was
    about 20x slower at batch 32 and 2.3x slower at batch 4096 than
    `jax.jit(jax.vmap(f))`.

## Neural Network Layers

Layers take a batch directly, with no `vmap`. The hyperboloid activations
(`hyp_relu` and the others) take ambient points `(..., dim+1)` with any leading
batch axes; they are for hyperboloid points only, not Poincaré ball points.

```python
from flax import nnx
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HypLinearPoincare, hyp_relu

layer = HypLinearPoincare(manifold_module=poincare, in_dim=128, out_dim=64, rngs=nnx.Rngs(0))
x_in = jax.vmap(poincare.expmap_0, in_axes=(0, None))(
    0.1 * jax.random.normal(jax.random.PRNGKey(1), (32, 128)), 1.0
)
print(layer(x_in, c=1.0).shape)  # (32, 64)

h_batch = Hyperboloid().proj_batch(jax.random.normal(jax.random.PRNGKey(0), (32, 4)), 1.0)
print(hyp_relu(h_batch, c=1.0).shape)  # (32, 4)
```

## Training Step

Use `nnx.jit`, not `jax.jit`, for a train step: a `jax.jit` step drops the
in-place parameter and optimizer updates, so the model never trains.

```python
import optax
from hyperbolix.nn_layers import HypLinearPoincarePP

model = HypLinearPoincarePP(manifold_module=poincare, in_dim=16, out_dim=4, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, x_batch, y_batch, c):
    def loss_fn(model):
        preds = model(x_batch, c)
        return jnp.mean((preds - y_batch) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


x_train = jax.vmap(poincare.expmap_0, in_axes=(0, None))(
    0.1 * jax.random.normal(jax.random.PRNGKey(0), (32, 16)), 1.0
)
y_train = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (32, 4))
for step in range(5):
    loss = train_step(model, optimizer, x_train, y_train, 1.0)
    print(f"step {step}: loss = {loss:.4f}")  # decreases step by step
```

For a complete loop, see [Training Workflows](training-workflows.md).

## Common Pitfalls

- **Shape mismatch in `in_axes`.** Every argument mapped with `0` must have the
  same leading size. A per-sample `c` needs one value per point; a shared `c` is
  passed with `None`.
- **`jax.jit` on an NNX train step.** It drops the parameter and optimizer
  updates; use `nnx.jit` (see above).

## See Also

- [Manifolds API](../api-reference/manifolds.md): Manifold function signatures
- [NN Layers API](../api-reference/nn-layers/index.md): Layer implementations
- [Training Workflows](training-workflows.md): Complete training examples
- [Numerical Stability](numerical-stability.md): Float precision considerations
