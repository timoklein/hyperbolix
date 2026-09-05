# Utilities API

Utility functions for hyperbolic deep learning.

## Math Utilities

Numerically stable implementations of hyperbolic functions.

Note which norm primitive goes where. On the **hot path** — every per-sample norm inside `proj`,
`expmap`, `logmap`, the distances and the layer forwards — the library takes a single reduction:
`safe_sqrt(sum(v**2))` when the input can be exactly zero, a plain `sqrt(const + sum(v**2))` when a
strictly positive constant is added, and `floor_at(..., MIN_NORM)` around either when the norm is a
divisor. `safe_norm`, `safe_hypot_norm` and `safe_normalize` are max-scaled and read the input
twice; they are used for the **weight** norms, computed once per forward over a kernel rather than
once per sample, and they are the right choice in user code that needs the full float32 exponent
range. See
[Norms: one reduction, gradient-safe at zero](../user-guide/numerical-stability.md#safe-norms).

::: hyperbolix.utils.math_utils
    options:
      show_source: true
      heading_level: 3
      members:
        - cosh
        - sinh
        - tanh
        - acosh
        - atanh
        - safe_norm
        - safe_normalize
        - safe_sqrt
        - safe_hypot
        - safe_hypot_norm
        - smooth_clamp
        - smooth_clamp_min
        - smooth_clamp_max
        - capped_exp

### Usage Example

```python
from hyperbolix.utils.math_utils import acosh, atanh, smooth_clamp
import jax.numpy as jnp

# Numerically stable hyperbolic functions
x = jnp.array([1.5, 2.0, 10.0])
y = acosh(x)  # Handles edge cases near 1.0

# Smooth clamping for stability
z = jnp.array([0.99, 1.0, 1.01])
z_clamped = smooth_clamp(z, min_value=0.0, max_value=1.0)
```

Use `capped_exp` instead of `jnp.exp` whenever you `exp()` an unconstrained trainable parameter
(a log-scale reparameterization, for example) — a runaway parameter saturates to a large finite
value instead of overflowing to `inf` and NaN-ing the rest of the model on the next optimizer step:

```python
from hyperbolix.utils.math_utils import capped_exp

log_scale = jnp.array(1e6)  # a runaway trainable parameter
jnp.exp(log_scale)  # inf -- would NaN downstream
capped_exp(log_scale)  # finite, saturates at exp(0.99*log(finfo.max))
```

## Matmul Precision

`MATMUL_PRECISION` pins the **geometry** to `jax.lax.Precision.HIGHEST`, avoiding XLA:GPU's default TF32 rounding on the cancellations hyperbolic geometry relies on. It is not user-configurable. Pinned: the vector-vector reductions inside `manifolds/` and the `decomposition/` (HoroPCA) contractions; `lorentz_midpoint` and `poincare_weighted_midpoint`; the conv patch extraction (a 0/1-filter convolution, so a pure data copy); the attention score and aggregation einsums and the spatial residual projection added to that aggregate; and the MLR heads (`HypRegressionHyperboloid` and the three `_compute_mlr` einsums in `manifolds/`), which are decision quantities.

The **layer weight GEMMs** pass no `precision` keyword at all — `HTCLinear` (including the attention Q/K/V projections), `FGGLinear`/`FGGConv2D`, the FHCNN/FHNN and PLFC linears, the Poincaré and proper-velocity linear and convolution layers, `HypVQMLRPoincare`'s codebook matmul, `hybrid_regularization`'s `nnx.Linear`. These follow JAX's own `jax_default_matmul_precision`, which on Ampere/Hopper means TF32. That is where `HIGHEST` actually costs throughput (one TF32 pass becomes a three-pass float32 emulation; measured +5.5 % on a jitted hyperbolic attention forward), so the trade is yours to make:

```python
# JAX-wide (jit-cache aware — changing it re-traces)
jax.config.update("jax_default_matmul_precision", "highest")

# or scoped to a block
with jax.default_matmul_precision("highest"):
    logits = model(x)
```

A deep, fully hyperbolic stack is a good reason to set it: a TF32 error introduced in an early layer's weight GEMM is carried by every layer after it. Measured on an A100 (jax 0.9.1, float32 against a float64 reference), `FGGLinear`'s relative error is 2.6e-4 at the TF32 default against 6.8e-8 under `HIGHEST` — roughly ~1e-4 against ~1e-7 on the layers. `HIGHEST` is a no-op on CPU and for float64 anywhere.

::: hyperbolix.utils.precision
    options:
      show_source: true
      heading_level: 3

## Learnable Curvature

`LearnableCurvature` is an `nnx.Module` that bundles a Euclidean raw parameter, a curvature reparameterization (positive `softplus`/`log`, or the **signed** `identity` — `c = raw` — for the `Stereographic` manifold), and an optional `[c_min, c_max]` clamp into one object. Assign one instance per distinct curvature on your model and call it in the forward pass to obtain the (optionally clamped) curvature.

::: hyperbolix.utils.curvature.LearnableCurvature
    options:
      show_source: true
      heading_level: 3

### Usage Example

```python
from flax import nnx
import optax
from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HypLinearHyperboloidPLFC


class Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.manifold = Hyperboloid(c=1.0)               # static, shared
        self.curvature = LearnableCurvature(             # one per distinct c
            init_c=1.0,
            parameterization="softplus",                 # or "log" (MERU)
            c_min=0.1, c_max=10.0,                       # default clamp
        )
        self.fc = HypLinearHyperboloidPLFC(self.manifold, 33, 65, rngs=rngs)

    def __call__(self, x):
        c = self.curvature()                              # positive, clamped
        return self.fc(x, c=c)


# Updated by any standard Euclidean optimizer — no Riemannian optimizer needed.
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
```

See the [Manifolds User Guide — Working with Curvature](../user-guide/manifolds.md#working-with-curvature) for the full discussion of parameterizations, clamping, the `nnx.scan` sharing rule, and per-factor `ProductManifold` usage.

## Helper Functions

Helper utilities for distance computation and delta-hyperbolicity analysis.

::: hyperbolix.utils.helpers
    options:
      show_source: true
      heading_level: 3
      members:
        - compute_pairwise_distances
        - compute_hyperbolic_delta
        - get_delta

### Usage Examples

#### Pairwise Distances

```python
import jax
import jax.numpy as jnp
from hyperbolix.utils.helpers import compute_pairwise_distances
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Set of points on Poincaré ball
points = jnp.array([
    [0.1, 0.2],
    [0.3, -0.1],
    [-0.2, 0.4],
    [0.0, 0.0]
])

# Compute all pairwise distances
dist_matrix = compute_pairwise_distances(
    points,
    manifold_module=poincare,
    c=1.0,
    version_idx=0
)

# Result: (4, 4) matrix of distances
print(dist_matrix.shape)  # (4, 4)
```

#### Delta-Hyperbolicity

Measure how "hyperbolic" a dataset is using the Gromov delta metric:

```python
import jax
import jax.numpy as jnp
from hyperbolix.utils.helpers import get_delta
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Generate random points
key = jax.random.PRNGKey(0)
points = jax.random.normal(key, (100, 2)) * 0.3

# Project to Poincaré ball
points_proj = jax.vmap(poincare.proj, in_axes=(0, None))(points, 1.0)

# Compute delta-hyperbolicity
delta, diameter, rel_delta = get_delta(
    points_proj,
    manifold_module=poincare,
    c=1.0,
    sample_size=500,  # Points subsampled before the pairwise distance matrix
    key=jax.random.PRNGKey(42),  # Only used when len(points) > sample_size
)

print(f"Delta: {delta:.4f}")
print(f"Diameter: {diameter:.4f}")
print(f"Relative delta: {rel_delta:.4f}")
```

The Gromov delta quantifies tree-likeness:

- δ ≈ 0: Perfect tree structure (hyperbolic)
- δ > 0: Non-tree structure (less hyperbolic)
- δ/diameter: Normalized measure (relative delta)

## Performance Tips

!!! tip "JIT Compilation"
    All utility functions support JIT compilation:

    ```python
    from hyperbolix.manifolds import Poincare

    poincare = Poincare()

    @jax.jit
    def compute_all_distances(points, c):
        return compute_pairwise_distances(
            points,
            manifold_module=poincare,
            c=c,
            version_idx=0
        )
    ```

!!! note "Batching"
    For large datasets, consider batching delta-hyperbolicity computation:

    ```python
    # Use smaller sample_size for faster computation
    delta, diameter, rel_delta = get_delta(
        points,
        manifold_module=poincare,
        c=1.0,
        sample_size=100,  # Reduce from 500 for speed
        seed=42
    )
    ```

## References

- **Gromov Delta**: Gromov, M. (1987). "Hyperbolic groups."

See also:

- [Manifolds API](manifolds.md): Core geometric operations
- [Numerical Stability Guide](../user-guide/numerical-stability.md): Best practices
