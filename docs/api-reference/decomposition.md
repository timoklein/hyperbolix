# Decomposition API

Hyperbolic dimensionality reduction. HoroPCA (Chami et al. 2021) reduces the dimension of
data on the hyperboloid or Poincaré ball by jointly optimizing a set of ideal points and
projecting the data horospherically, preserving Busemann coordinates. CO-SNE (Guo et al. 2022)
is the hyperbolic analogue of t-SNE, adding a magnitude loss that preserves each point's
distance-to-origin (the hierarchy depth). The Fréchet mean is the data-centering primitive
HoroPCA builds on.

!!! tip "Use float64 for fitting"
    HoroPCA fits a non-convex objective with Adam; the horospherical projection and Fréchet
    iteration are most accurate in `dtype=jnp.float64`. CO-SNE likewise fits most reliably in
    float64 — its projected gradient descent operates near the ball boundary, where float32
    resolution runs out. Enable x64 (`JAX_ENABLE_X64=1`) and build the manifold with
    `Hyperboloid(dtype=jnp.float64)` / `Poincare(dtype=jnp.float64)` for reliable fits.

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

## CO-SNE

Hyperbolic t-SNE (Guo et al. 2022): a low-dimensional Poincaré ball embedding that preserves
both pairwise similarities (the t-SNE KL term) and each point's distance-to-origin — the
hierarchy depth — via an added magnitude loss `H = (1/N)·Σᵢ(‖xᵢ‖² − ‖yᵢ‖²)²` (paper Eq. 10).
A thin sklearn-style class over the functional core; being non-parametric it has `fit` /
`fit_transform` but no out-of-sample `transform` (matching sklearn's `TSNE`). Poincaré input
`(N, D)` embeds to `(N, K)` ball coordinates; Hyperboloid input `(N, A)` embeds to `(N, K+1)`.

!!! warning "Calibrated learning rate"
    The exact-autodiff gradients differ in scale from the reference's hand-derived gradients,
    so the reference learning rates do not transfer. The `learning_rate` default (`0.5`) is
    calibrated on a synthetic-cluster recovery test — see the class docstring.

::: hyperbolix.decomposition.cosne.CoSNE
    options:
      show_source: true
      heading_level: 3
      merge_init_into_class: true

### Usage Example

```python
import jax
import jax.numpy as jnp
from hyperbolix.manifolds import Poincare
from hyperbolix.decomposition import CoSNE

manifold = Poincare(dtype=jnp.float64)
key = jax.random.PRNGKey(0)

# x: (N, D) points on the Poincaré ball (single-point proj, batched via vmap)
x = jax.vmap(manifold.proj, in_axes=(0, None))(0.3 * jax.random.normal(key, (200, 10), dtype=jnp.float64), 1.0)

model = CoSNE(manifold, n_components=2)
z = model.fit_transform(x, c=1.0, key=jax.random.PRNGKey(1))  # (200, 2) ball coordinates
print(model.kl_divergence_)                                   # final KL on the unexaggerated P
```

### Functional core

The pure functions underneath `CoSNE` — independently usable and JIT-friendly. For a
precomputed-distance workflow (e.g. HycoCLIP), call `fit_cosne` directly with the input
distance matrix and squared ball norms.

::: hyperbolix.decomposition.cosne
    options:
      show_source: true
      heading_level: 4
      members:
        - conditional_probabilities
        - joint_probabilities
        - low_dim_probabilities
        - kl_divergence_loss
        - magnitude_loss
        - fit_cosne

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
- Guo, Guo, Yu. "CO-SNE: Dimensionality Reduction and Visualization for Hyperbolic Data."
  CVPR 2022.
- van der Maaten, Hinton. "Visualizing Data using t-SNE." JMLR 2008.
- Karcher, H. "Riemannian center of mass and mollifier smoothing." Comm. Pure Appl. Math. 1977.

See also:

- [Manifolds API](manifolds.md): the underlying geometry (`Hyperboloid.lorentz_boost`,
  `Hyperboloid.busemann`).
- [Utilities API](utils.md): `compute_pairwise_distances` and delta-hyperbolicity metrics.
