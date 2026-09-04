"""Helper utilities for hyperbolic geometry computations.

This module provides utilities for computing pairwise distances, delta-hyperbolicity
metrics, and other geometric measures on hyperbolic manifolds.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ..manifolds.protocol import Manifold, ScalarCurvature


def compute_pairwise_distances(
    points: Float[Array, "n_points dim"],
    manifold_module: Manifold,
    c: ScalarCurvature,
    version_idx: int = 0,
) -> Float[Array, "n_points n_points"]:
    """Compute pairwise geodesic distances between points on a manifold.

    This function computes the full distance matrix efficiently by leveraging
    JAX's vmap for vectorization. The computation is NOT chunked - the entire
    distance matrix is computed in a single pass using nested vmap operations.

    Memory Considerations:
        For n points, this computes an n-by-n distance matrix in memory. For very
        large point sets (>5000-10000 points depending on available memory),
        consider subsampling or implementing a chunked version. The current
        implementation prioritizes simplicity and leverages XLA's automatic
        memory optimizations.

    Args:
        points: Points on the manifold, shape (n_points, dim)
            For Hyperboloid: dim is ambient dimension (dim+1)
            For PoincareBall: dim is intrinsic dimension
        manifold_module: Manifold instance (anything satisfying the ``Manifold`` protocol)
        c: Curvature parameter (positive scalar; a per-factor sequence for ``ProductManifold``)
        version_idx: Distance version index (manifold-specific, default: 0)
            For Hyperboloid:
                0 = VERSION_DEFAULT (standard acosh with hard clipping)
                1 = VERSION_SMOOTHENED (smoothened distance)
            For PoincareBall:
                0 = VERSION_MOBIUS_DIRECT (direct Möbius formula)
                1 = VERSION_MOBIUS (via addition)
                2 = VERSION_METRIC_TENSOR (metric tensor induced)
            Euclidean, Stereographic and ProductManifold have a single distance
            implementation and accept-and-ignore this argument.

    Returns:
        Symmetric distance matrix of shape (n_points, n_points)

    Examples:
        >>> import jax
        >>> from hyperbolix.manifolds import Hyperboloid
        >>> from hyperbolix.utils.helpers import compute_pairwise_distances
        >>>
        >>> # Generate random hyperboloid points
        >>> manifold = Hyperboloid()
        >>> key = jax.random.PRNGKey(0)
        >>> points = jax.random.normal(key, (100, 11))
        >>> points = jax.vmap(manifold.proj, in_axes=(0, None))(points, 1.0)
        >>>
        >>> # Compute pairwise distances
        >>> distmat = compute_pairwise_distances(
        ...     points, manifold, c=1.0, version_idx=manifold.VERSION_DEFAULT
        ... )
        >>> distmat.shape
        (100, 100)

    Notes:
        - The PyTorch reference implementation used explicit chunking for memory
          management. This JAX version uses vmap and relies on XLA optimization.
        - The distance matrix is symmetric: distmat[i, j] == distmat[j, i]
        - Diagonal elements are zero: distmat[i, i] == 0
        - For large datasets, consider subsampling before calling this function
    """

    # We need to compute dist for all pairs (i, j)
    def dist_fn(x, y):
        return manifold_module.dist(x, y, c, version_idx)

    # Use vmap to vectorize over both dimensions
    # First vmap over y (columns), then over x (rows)
    dist_col = jax.vmap(dist_fn, in_axes=(None, 0))  # Broadcasts x over all y
    dist_matrix_fn = jax.vmap(dist_col, in_axes=(0, None))  # Broadcasts over all x

    # Compute full distance matrix
    distmat = dist_matrix_fn(points, points)

    return distmat


def compute_hyperbolic_delta(distmat: Float[Array, "n_points n_points"], version: str = "average") -> Float[Array, ""]:
    """Compute the delta-hyperbolicity value from a distance matrix.

    Delta-hyperbolicity is a metric space property that quantifies how "tree-like"
    or "hyperbolic" a metric space is. It is based on the Gromov 4-point condition.

    For any four points w, x, y, z in a metric space, define:
        S1 = d(w,x) + d(y,z)
        S2 = d(w,y) + d(x,z)
        S3 = d(w,z) + d(x,y)

    The 4-point condition requires that the two largest of these sums differ by at
    most 2δ. A space is δ-hyperbolic if this holds for all quadruples.

    This implementation uses a reference point (the first point) to compute
    Gromov products efficiently:
        (x|y)_w = [d(w,x) + d(w,y) - d(x,y)] / 2

    Args:
        distmat: Symmetric distance matrix, shape (n_points, n_points)
        version: Which delta statistic to compute (default: "average")
            - "average": Mean of delta values over all point quadruples
            - "smallest": Maximum delta (worst-case over all quadruples)

    Returns:
        Delta-hyperbolicity value (scalar)

    References:
        Gromov, M. (1987). "Hyperbolic groups." Essays in group theory.
        Chami, I., et al. (2021). "HoroPCA: Hyperbolic dimensionality reduction
            via horospherical projections." ICML 2021.

    Examples:
        >>> import jax.numpy as jnp
        >>> from hyperbolix.utils.helpers import compute_hyperbolic_delta
        >>>
        >>> # Create a distance matrix (should be symmetric)
        >>> distmat = jnp.array([
        ...     [0.0, 1.0, 2.0, 3.0],
        ...     [1.0, 0.0, 1.5, 2.5],
        ...     [2.0, 1.5, 0.0, 1.0],
        ...     [3.0, 2.5, 1.0, 0.0]
        ... ])
        >>>
        >>> delta_avg = compute_hyperbolic_delta(distmat, version="average")
        >>> delta_max = compute_hyperbolic_delta(distmat, version="smallest")

    Notes:
        - The result is scaled by 2 because we fix a reference point
        - Lower delta values indicate more hyperbolic (tree-like) structure
        - Euclidean spaces have unbounded delta; hyperbolic spaces have bounded delta
    """
    # Set the first point as reference and compute pairwise Gromov products
    # Gromov product: (x|y)_w = [d(w,x) + d(w,y) - d(x,y)] / 2
    # We use point 0 as reference: (i|j)_0 = [d(0,i) + d(0,j) - d(i,j)] / 2

    # distmat_i0[i, j] = d(i, 0) for all j (broadcast column 0 across columns)
    distmat_i0 = jnp.tile(distmat[:, 0:1], (1, distmat.shape[0]))  # shape: (n, n)
    # distmat_0j[i, j] = d(0, j) for all i (broadcast row 0 across rows)
    distmat_0j = jnp.tile(distmat[0:1, :], (distmat.shape[0], 1))  # shape: (n, n)

    # Compute Gromov product matrix: (i|j)_0 for all pairs (i, j)
    gromov_prod_mat = (distmat_i0 + distmat_0j - distmat) / 2.0  # shape: (n, n)

    # (max, min)-matrix product of the Gromov product matrix with itself:
    #     (G ⊠ G)[i, j] = max_k min( (i|k)_0 , (k|j)_0 )
    # The two operands must be indexed on DIFFERENT rows — the first on i, the second on the
    # summation index k. Indexing both on row i (the historical bug) gives
    # max_k min(G[i,k], G[i,j]) == G[i,j] whenever k = j is admissible, so `delta_matrix` was
    # identically 0 for every input and `get_delta` always reported 0.

    # Computed as a scan over k with a running elementwise maximum, NOT as one broadcast
    # jnp.minimum((n, 1, n), (1, n, n)): that materializes an (n, n, n) buffer — 27 GB in
    # float64 at get_delta's default sample_size=1500 — while the scan needs O(n^2) memory
    # (one (n, n) candidate per step, executed as a compiled loop even when called eagerly).
    def _step(running_max_NN, gromov_k):
        col_N, row_N = gromov_k  # G[:, k] = (i|k)_0 and G[k, :] = (k|j)_0
        cand_NN = jnp.minimum(col_N[:, None], row_N[None, :])  # (i, j) -> min((i|k)_0, (k|j)_0)
        return jnp.maximum(running_max_NN, cand_NN), None

    init_NN = jnp.full_like(gromov_prod_mat, -jnp.inf)
    # xs leading axis is k: (gromov_prod_mat.T)[k] = G[:, k], (gromov_prod_mat)[k] = G[k, :]
    max_min_prod, _ = jax.lax.scan(_step, init_NN, (gromov_prod_mat.T, gromov_prod_mat))

    # Compute delta for each pair: max_k{min((i|k)_0, (k|j)_0)} - (i|j)_0
    delta_matrix = max_min_prod - gromov_prod_mat  # shape: (n, n)

    # Compute the requested statistic
    if version == "average":
        delta = jnp.mean(delta_matrix)
    else:  # "smallest" (which is actually the maximum delta value)
        delta = jnp.max(delta_matrix)

    # Rescale delta since a reference point was fixed
    # The factor of 2 accounts for the fact that we fixed one point
    res = 2.0 * delta

    return res


def get_delta(
    points: Float[Array, "n_points dim"],
    manifold_module: Manifold,
    c: ScalarCurvature,
    version_idx: int = 0,
    sample_size: int = 1500,
    version: str = "average",
    key: Key[Array, ""] | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute delta-hyperbolicity and related metrics for a point set.

    This function subsamples points (if needed), computes the pairwise distance
    matrix, and then calculates the delta-hyperbolicity value along with the
    diameter and relative delta (delta normalized by diameter).

    Args:
        points: Points on the manifold, shape (n_points, dim)
        manifold_module: Manifold instance (anything satisfying the ``Manifold`` protocol)
        c: Curvature parameter (positive scalar; a per-factor sequence for ``ProductManifold``)
        version_idx: Distance version index (manifold-specific, default: 0)
        sample_size: Maximum number of points to use for delta computation
            (default: 1500). If n_points > sample_size, randomly subsample.
        version: Which delta statistic to compute (default: "average")
            - "average": Mean of delta values
            - "smallest": Maximum delta (worst-case)
        key: JAX random key for subsampling (required if n_points > sample_size)

    Returns:
        Tuple of (delta, diameter, relative_delta):
            - delta: Delta-hyperbolicity value
            - diameter: Maximum pairwise distance in the point set
            - relative_delta: delta / diameter (scale-invariant measure)

    Examples:
        >>> import jax
        >>> from hyperbolix.manifolds import Hyperboloid
        >>> from hyperbolix.utils.helpers import get_delta
        >>>
        >>> # Generate random hyperboloid points
        >>> manifold = Hyperboloid()
        >>> key = jax.random.PRNGKey(42)
        >>> points = jax.random.normal(key, (400, 11))
        >>> points = jax.vmap(manifold.proj, in_axes=(0, None))(points, 1.0)
        >>>
        >>> # Compute delta metrics (subsampling 200 of the 400 points)
        >>> key, subkey = jax.random.split(key)
        >>> delta, diam, rel_delta = get_delta(
        ...     points, manifold, c=1.0, sample_size=200, key=subkey
        ... )
        >>> bool(delta > 0.0) and bool(0.0 < rel_delta < 1.0)
        True

    Notes:
        - Subsampling is done randomly without replacement
        - For reproducibility, always provide the same random key
        - The PyTorch version used torch.randperm; we use jax.random.permutation
    """
    n_points = points.shape[0]

    # Subsample points if necessary
    if n_points > sample_size:
        if key is None:
            raise ValueError(f"Random key required for subsampling (n_points={n_points} > sample_size={sample_size})")
        # Random permutation of indices
        indices = jax.random.permutation(key, n_points)[:sample_size]
        sub_points = points[indices]
    else:
        sub_points = points

    # Compute pairwise distances
    distmat = compute_pairwise_distances(sub_points, manifold_module, c, version_idx)

    # Compute delta-hyperbolicity
    delta = compute_hyperbolic_delta(distmat, version)

    # Compute diameter (maximum distance)
    diam = jnp.max(distmat)

    # Compute relative delta (scale-invariant)
    rel_delta = delta / diam

    return delta, diam, rel_delta
