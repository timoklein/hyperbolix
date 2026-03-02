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
from hyperbolix.utils.math_utils import acosh, atanh, smooth_clamp
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
