# Decomposition API

Hyperbolic dimensionality reduction. HoroPCA (Chami et al. 2021) reduces the dimension of
data on the hyperboloid or Poincaré ball by jointly optimizing a set of ideal points and
projecting the data horospherically, preserving Busemann coordinates. The Fréchet mean is the
data-centering primitive it builds on.

!!! tip "Use float64 for fitting"
    HoroPCA fits a non-convex objective with Adam; the horospherical projection and Fréchet
    iteration are most accurate in `dtype=jnp.float64`. Enable x64 (`JAX_ENABLE_X64=1`) and
    build the manifold with `Hyperboloid(dtype=jnp.float64)` for reliable fits.

## HoroPCA

A thin sklearn-style class over the functional core: `fit` the components at a chosen
curvature, then `transform` points to their low-dimensional embedding. Poincaré input
`(N, D)` maps to `(N, K)` ball coordinates; Hyperboloid input `(N, A)` maps to `(N, K+1)`
hyperboloid points.

::: hyperbolix.decomposition.horopca.HoroPCA
    options:
      show_source: true
      heading_level: 3
      merge_init_into_class: true

### Usage Example

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.decomposition import HoroPCA

manifold = Hyperboloid(dtype=jnp.float64)
key = jax.random.PRNGKey(0)

# x: (N, A) points on the hyperboloid (A = spatial dim + 1)
x = manifold.proj_batch(jax.random.normal(key, (256, 11), dtype=jnp.float64), c=1.0)

model = HoroPCA(manifold, n_components=2, max_steps=500, lr=1e-2)
model.fit(x, c=1.0, key=jax.random.PRNGKey(1))

z = model.transform(x)                       # (256, 3) — K=2 components + time
print(model.explained_variance_ratio_)       # fraction of pairwise variance retained
```

## Functional core

The pure functions underneath `HoroPCA` — independently usable and JIT-friendly.

::: hyperbolix.decomposition.horopca
    options:
      show_source: true
      heading_level: 3
      members:
        - horo_projection
        - horopca_loss
        - fit_horopca
        - transform_horopca

## Fréchet mean

The Karcher fixed-point Fréchet (Riemannian center of mass) mean, used to center data before
fitting. Manifold-generic — works for `Hyperboloid`, `Poincare`, `ProperVelocity`, and
`Euclidean`.

::: hyperbolix.decomposition.frechet.frechet_mean
    options:
      show_source: true
      heading_level: 3

### Usage Example

```python
import functools
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.decomposition import frechet_mean

manifold = Hyperboloid(dtype=jnp.float64)
x = manifold.proj_batch(jax.random.normal(jax.random.PRNGKey(0), (64, 6), dtype=jnp.float64), c=1.0)

mean = frechet_mean(x, manifold, c=1.0)

# To JIT, close over the (unhashable) manifold argument:
fmean = jax.jit(functools.partial(frechet_mean, manifold=manifold, c=1.0))
mean_jit = fmean(x)
```

## References

- Chami, Gu, Nguyen, Ré. "HoroPCA: Hyperbolic Dimensionality Reduction via Horospherical
  Projections." ICML 2021.
- Karcher, H. "Riemannian center of mass and mollifier smoothing." Comm. Pure Appl. Math. 1977.

See also:

- [Manifolds API](manifolds.md): the underlying geometry (`Hyperboloid.lorentz_boost`,
  `Hyperboloid.busemann`).
- [Utilities API](utils.md): `compute_pairwise_distances` and delta-hyperbolicity metrics.
