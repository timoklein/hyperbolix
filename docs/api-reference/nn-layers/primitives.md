# Primitives & Helpers

Cross-cutting functions that back the layers on the other pages — the Hyperboloid
transformation/regularization components, point-assembly and concatenation helpers,
midpoints, residuals, and the statistical reductions used by normalization. They are
collected here (rather than scattered across consumer pages) so a given helper has one
documented home.

## HTC / HRC components

The Hyperbolic Transformation Component (HTC) applies a Euclidean function to the
**full** point; the Hyperbolic Regularization Component (HRC) applies it to the
**space** components only, then reconstructs time. Both support curvature changes
(`c_in → c_out`) and underpin `HTCLinear`, the `HRC*` norms, and `LorentzConv2D`.

::: hyperbolix.nn_layers.htc
    options:
      heading_level: 3

::: hyperbolix.nn_layers.hrc
    options:
      heading_level: 3

## Point assembly & concatenation

Helpers that build valid hyperboloid points from spatial/tangent data and extract
convolution patches.

::: hyperbolix.nn_layers.spatial_to_hyperboloid
    options:
      heading_level: 3

::: hyperbolix.nn_layers.sinh_lift_to_hyperboloid
    options:
      heading_level: 3

::: hyperbolix.nn_layers.build_spacelike_V
    options:
      heading_level: 3

::: hyperbolix.nn_layers.extract_patches
    options:
      heading_level: 3

## Midpoints & residuals

::: hyperbolix.nn_layers.lorentz_midpoint
    options:
      heading_level: 3

::: hyperbolix.nn_layers.lorentz_residual
    options:
      heading_level: 3

::: hyperbolix.nn_layers.lorentz_scale
    options:
      heading_level: 3

## Attention focus transform

::: hyperbolix.nn_layers.focus_transform
    options:
      heading_level: 3

## Poincaré statistics & midpoints

Reductions used by `PoincareBatchNorm2D` and the Poincaré VQ codebook. `frechet_variance`
is manifold-agnostic (mean squared geodesic distance); `poincare_weighted_midpoint`
is the GGBall weighted gyromidpoint (the Poincaré analog of `lorentz_midpoint`).

::: hyperbolix.nn_layers.poincare_midpoint
    options:
      heading_level: 3

::: hyperbolix.nn_layers.poincare_weighted_midpoint
    options:
      heading_level: 3

::: hyperbolix.nn_layers.frechet_variance
    options:
      heading_level: 3

## Example

```python
import jax, jax.numpy as jnp
from hyperbolix.nn_layers import lorentz_residual
from hyperbolix.manifolds import Hyperboloid

hyperboloid, c, d = Hyperboloid(), 1.0, 6
x = jax.random.normal(jax.random.PRNGKey(0), (8, d + 1))
y = jax.random.normal(jax.random.PRNGKey(1), (8, d + 1))
x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)
y = jax.vmap(hyperboloid.proj, in_axes=(0, None))(y, c)
result = lorentz_residual(x, y, w_y=0.3, c=c)  # weighted Lorentzian midpoint (skip connection)
print(result.shape)  # (8, 7)
```
