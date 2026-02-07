# Utilities API

Utility functions for hyperbolic deep learning.

## Math Utilities

Numerically stable implementations of hyperbolic functions.

::: hyperbolix.utils.math_utils
    options:
      show_source: true
      heading_level: 3
      members:
        - cosh
        - sinh
        - acosh
        - atanh
        - smooth_clamp

### Usage Example

```python
from hyperbolix.utils.math_utils import acosh, atanh
import jax.numpy as jnp

# Numerically stable hyperbolic functions
x = jnp.array([1.5, 2.0, 10.0])
y = acosh(x)  # Handles edge cases near 1.0

# Smooth clamping for stability
z = jnp.array([0.99, 1.0, 1.01])
z_clamped = smooth_clamp(z, min_val=0.0, max_val=1.0)
```

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
from hyperbolix.utils.helpers import compute_pairwise_distances
from hyperbolix.manifolds import poincare
import jax.numpy as jnp

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
from hyperbolix.utils.helpers import get_delta
from hyperbolix.manifolds import poincare
import jax.numpy as jnp

# Generate random points
key = jax.random.PRNGKey(0)
points = jax.random.normal(key, (100, 2)) * 0.3

# Project to Poincaré ball
points_proj = jax.vmap(poincare.proj, in_axes=(0, None))(
    points, 1.0
)

# Compute delta-hyperbolicity
delta, diameter, rel_delta = get_delta(
    points_proj,
    manifold_module=poincare,
    c=1.0,
    sample_size=500,  # Number of 4-point samples
    seed=42
)

print(f"Delta: {delta:.4f}")
print(f"Diameter: {diameter:.4f}")
print(f"Relative delta: {rel_delta:.4f}")
```

The Gromov delta quantifies tree-likeness:

- δ ≈ 0: Perfect tree structure (hyperbolic)
- δ > 0: Non-tree structure (less hyperbolic)
- δ/diameter: Normalized measure (relative delta)

## HoroPCA

Horospherical Principal Component Analysis for dimensionality reduction on hyperbolic manifolds.

::: hyperbolix.utils.horo_pca
    options:
      show_source: true
      heading_level: 3

### Usage Example

```python
from flax import nnx
from hyperbolix.utils.horo_pca import HoroPCA
from hyperbolix.manifolds import poincare, hyperboloid
import jax.numpy as jnp
import jax

# High-dimensional hyperbolic data
key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (100, 10)) * 0.3

# Project to Poincaré ball
data_proj = jax.vmap(poincare.proj, in_axes=(0, None, None))(
    data, 1.0, None
)

# Initialize HoroPCA
horo_pca = HoroPCA(
    manifold_module=poincare,
    n_components=3,  # Reduce to 3 dimensions
    rngs=nnx.Rngs(0)
)

# Fit on data
horo_pca.fit(data_proj, c=1.0)

# Transform new data
data_reduced = horo_pca.transform(data_proj, c=1.0)
print(data_reduced.shape)  # (100, 3)

# Inverse transform (approximate reconstruction)
data_reconstructed = horo_pca.inverse_transform(data_reduced, c=1.0)
```

### Manifold Support

HoroPCA supports both Poincaré and Hyperboloid manifolds:

```python
# Poincaré ball (conformal model)
horo_pca_poincare = HoroPCA(
    manifold_module=poincare,
    n_components=5,
    rngs=rngs
)

# Hyperboloid (Lorentz model)
horo_pca_hyperboloid = HoroPCA(
    manifold_module=hyperboloid,
    n_components=5,
    rngs=rngs
)
```

### How It Works

HoroPCA performs dimensionality reduction via:

1. **Fréchet Mean**: Compute geometric center via gradient descent
2. **Centering**: Apply Lorentz boost to move data to origin
3. **Horospherical Projection**: Project onto lower-dimensional horosphere
4. **Pseudoinverse**: Use Moore-Penrose pseudoinverse for numerical stability

This is more stable than standard PCA on manifolds and preserves hyperbolic structure better.

### Parameters

- `manifold_module`: Manifold module (poincare or hyperboloid)
- `n_components`: Target dimensionality
- `max_iter`: Maximum Fréchet mean iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)

## Performance Tips

!!! tip "JIT Compilation"
    All utility functions support JIT compilation:

    ```python
    @jax.jit
    def compute_all_distances(points, c):
        return compute_pairwise_distances(
            points,
            manifold_module=poincare,
            c=c,
            version=0
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
- **HoroPCA**: Chami, I., et al. (2021). "Low-Distortion Embeddings of Hyperbolic Spaces."

See also:

- [Manifolds API](manifolds.md): Core geometric operations
- [Numerical Stability Guide](../user-guide/numerical-stability.md): Best practices
