"""Horospherical PCA for hyperbolic dimensionality reduction.

This module provides HoroPCA, a method for dimensionality reduction in hyperbolic space
using horospherical projections. It is a JAX/Flax port of the PyTorch implementation.

References:
    Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
        International Conference on Machine Learning (2021).
    Weize Chen, et al. "Fully hyperbolic neural networks."
        arXiv preprint arXiv:2105.14686 (2021).
"""

from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float

from ..manifolds import hyperboloid
from .helpers import compute_pairwise_distances


def _batched_minkowski_inner(
    points: Float[Array, "batch dim_plus_1"],
    vectors: Float[Array, "n_components dim_plus_1"],
) -> Float[Array, "batch n_components"]:
    """Compute batched Minkowski inner products ⟨points_i, vectors_j⟩_L.

    Efficiently computes all pairwise Minkowski inner products between
    two sets of hyperboloid points using matrix operations.

    Args:
        points: Hyperboloid points, shape (batch, dim+1)
        vectors: Hyperboloid vectors, shape (n_components, dim+1)

    Returns:
        Inner products of shape (batch, n_components) where
        result[i, j] = ⟨points[i], vectors[j]⟩_L = -t_i*t_j + x_i·x_j

    Notes:
        More efficient than nested vmap for large batches as it leverages
        optimized BLAS operations for matrix multiplication.
    """
    time_term = -points[:, :1] @ vectors[:, :1].T
    spatial_term = points[:, 1:] @ vectors[:, 1:].T
    return time_term + spatial_term


def _horo_projection_rank1(
    x: Float[Array, "n_points dim_plus_1"],
    Q: Float[Array, "1 dim_plus_1"],
    c: Float[Array, ""] | float,
) -> Float[Array, "n_points dim_plus_1"]:
    """Special-case horo projection when only one ideal is provided.

    When n_components=1, the target submanifold is a geodesic plane through
    the hyperboloid origin and the ideal direction Q[0]. The horospherical
    projection simplifies to:
    1. Preserve temporal coordinate (distance from origin along geodesic)
    2. Project spatial coordinates onto Q's spatial direction
    3. Re-project onto hyperboloid to satisfy manifold constraint

    This avoids the Sherman-Morrison singularity that occurs when computing
    (Q B Q^T)^{-1} with n_components=1, where the denominator becomes zero.

    Args:
        x: Hyperboloid points of shape (n_points, dim+1)
        Q: Single ideal point of shape (1, dim+1) with Q[0,0]=1
        c: Curvature parameter (positive scalar)

    Returns:
        Projected points of shape (n_points, dim+1)

    Notes:
        - Handles degenerate case where Q's spatial norm is near zero
        - Geometrically equivalent to projecting onto the 1D geodesic subspace
    """
    q_space = Q[0, 1:]
    q_norm = jnp.linalg.norm(q_space)

    # Guard against degenerate case where ideal is at temporal axis
    # In this case, the geodesic plane collapses to the origin
    def project_degenerate():
        # Return points projected to origin (all spatial components = 0)
        origin = hyperboloid._create_origin(c, x.shape[1] - 1, dtype=x.dtype)
        return jnp.broadcast_to(origin[jnp.newaxis, :], x.shape)

    def project_normal():
        q_unit = q_space / q_norm

        # Project spatial components onto ideal direction
        x_time = x[:, :1]
        x_space = x[:, 1:]
        coeffs = jnp.dot(x_space, q_unit)[:, None]
        projected_space = coeffs * q_unit[jnp.newaxis, :]

        # Combine and project back to hyperboloid
        candidate = jnp.concatenate([x_time, projected_space], axis=1)
        return jax.vmap(hyperboloid.proj, in_axes=(0, None))(candidate, c)

    # Use lax.cond for JIT compatibility
    return jax.lax.cond(
        q_norm < hyperboloid.MIN_NORM,
        lambda _: project_degenerate(),
        lambda _: project_normal(),
        operand=None,
    )


def _horo_projection_points(
    x: Float[Array, "n_points dim_plus_1"],
    Q: Float[Array, "n_components dim_plus_1"],
    c: Float[Array, ""] | float,
) -> Float[Array, "n_points dim_plus_1"]:
    """Compute horospherical projections for a batch of points.

    Projects hyperboloid points onto the geodesic submanifold spanned by
    ideal points Q, using the orthogonal geodesic projection formula from
    the HoroPCA paper (Chami et al., 2021).

    Args:
        x: Hyperboloid points of shape (n_points, dim+1)
        Q: Ideal points of shape (n_components, dim+1) with orthonormalized
           spatial coordinates
        c: Curvature parameter (positive scalar)

    Returns:
        Projected points of shape (n_points, dim+1) lying on the
        horospherical submanifold

    Notes:
        - Uses rank-1 fallback when n_components=1 to avoid singularity
        - Uses pseudoinverse for Gram matrix for numerical stability
        - Complexity: O(n_points * n_components * dim + n_components^3)
    """
    n_components = Q.shape[0]
    if n_components == 1:
        return _horo_projection_rank1(x, Q, c)

    # Compute Gram matrix Q B Q^T (Minkowski metric)
    gram = _batched_minkowski_inner(Q, Q)

    # Use pseudoinverse for numerical stability
    # This handles near-singular cases where ideals become nearly linearly dependent
    gram_inv = jnp.linalg.pinv(gram, rtol=1e-6)

    xBQt = _batched_minkowski_inner(x, Q)
    x_coeffs = xBQt @ gram_inv
    mink_proj = x_coeffs @ Q

    def normalize_projection(proj):
        mink_inner = hyperboloid._minkowski_inner(proj, proj)
        norm = jnp.sqrt(jnp.maximum(-c * mink_inner, hyperboloid.MIN_NORM))
        return proj / norm

    spine_proj = jax.vmap(normalize_projection)(mink_proj)

    origin = hyperboloid._create_origin(c, Q.shape[1] - 1, dtype=x.dtype)
    hyperboloid_origin = jnp.broadcast_to(origin, x.shape)

    originBQt = _batched_minkowski_inner(hyperboloid_origin, Q)
    origin_coeffs = originBQt @ gram_inv
    tangents = hyperboloid_origin - (origin_coeffs @ Q)

    def normalize_tangent(tangent):
        tangent_inner = hyperboloid._minkowski_inner(tangent, tangent)
        norm = jnp.sqrt(jnp.maximum(tangent_inner, hyperboloid.MIN_NORM))
        return tangent / norm

    unit_tangents = jax.vmap(normalize_tangent)(tangents)

    def scale_tangent(x_point, spine_point, unit_tangent):
        spine_dist = hyperboloid.dist(x_point, spine_point, c, hyperboloid.VERSION_DEFAULT)
        return spine_dist * unit_tangent

    scaled_tangents = jax.vmap(scale_tangent)(x, spine_proj, unit_tangents)
    res = jax.vmap(hyperboloid.expmap, in_axes=(0, 0, None))(scaled_tangents, spine_proj, c)
    return res


def _projected_variance_loss(
    Q_param: Float[Array, "n_components dim_or_dim_minus_1"],
    x_data: Float[Array, "n_points dim_plus_1"],
    c_val: Float[Array, ""] | float,
) -> Float[Array, ""]:
    """Compute negative generalized variance of horospherical projections.

    This is the core loss function for HoroPCA optimization. It measures the
    spread (variance) of projected points in the target submanifold, computed
    as the mean of squared pairwise geodesic distances. Maximizing variance
    (minimizing negative variance) encourages principal components that preserve
    as much geometric structure as possible.

    Args:
        Q_param: Principal component parameters of shape (n_components, dim)
                 or (n_components, dim-1) depending on manifold type.
                 These are orthonormalized via QR and mapped to hyperboloid ideals.
        x_data: Centered hyperboloid points of shape (n_points, dim+1)
        c_val: Curvature parameter (positive scalar)

    Returns:
        Negative variance (scalar). More negative = higher variance = better fit.

    Notes:
        - Shared by both compute_loss() and fit() to ensure consistent numerics
        - Uses smoothened distance version for gradient stability
        - JIT-compiled when called from fit()
    """
    Q_ortho, _ = jnp.linalg.qr(Q_param.T, mode="reduced")
    Q_ortho = Q_ortho.T
    ones = jnp.ones((Q_ortho.shape[0], 1), dtype=Q_ortho.dtype)
    hyperboloid_ideals = jnp.concatenate([ones, Q_ortho], axis=-1)

    x_proj = _horo_projection_points(x_data, hyperboloid_ideals, c_val)
    distances = compute_pairwise_distances(
        x_proj,
        hyperboloid,
        c_val,
        version_idx=hyperboloid.VERSION_SMOOTHENED,
    )
    var = jnp.mean(distances**2)
    return -var


def compute_frechet_mean(
    x: Float[Array, "n_points dim_plus_1"],
    c: Float[Array, ""] | float,
    max_iters: int = 5_000,
    tol: float = 5e-6,
    lr_candidates: tuple[float, ...] = (1e-2, 2e-2, 5e-3, 4e-2, 2.5e-3),
) -> Float[Array, "1 dim_plus_1"]:
    """Compute the Fréchet mean of hyperboloid points using gradient descent.

    The Fréchet mean minimizes the sum of squared geodesic distances to all points.
    We iterate through multiple learning rates until convergence.

    Args:
        x: Hyperboloid points of shape (n_points, dim+1)
        c: Curvature parameter (positive scalar)
        max_iters: Maximum optimization iterations per learning rate
        tol: Convergence tolerance for gradient norm
        lr_candidates: Learning rates to try sequentially

    Returns:
        Fréchet mean of shape (1, dim+1)

    Notes:
        - Initializes with the centroid of squared Lorentzian distance
        - Tries multiple learning rates sequentially until convergence
        - Uses gradient descent with logmap/expmap for manifold optimization
    """
    n_points = x.shape[0]

    # Initialize mean as centroid of squared Lorentzian distance
    # Sum all points
    x_sum = jnp.sum(x, axis=0, keepdims=True)  # shape: (1, dim+1)

    # Compute normalization: sqrt(c * |<x_sum, x_sum>_L|)
    # _minkowski_inner expects 1D arrays, so squeeze and then add back dimension
    mink_inner = hyperboloid._minkowski_inner(x_sum[0], x_sum[0])
    denom = jnp.sqrt(c * jnp.abs(mink_inner))
    mean_init = x_sum / denom

    # Project onto hyperboloid
    mean_init = jax.vmap(hyperboloid.proj, in_axes=(0, None))(mean_init, c)

    has_converged = False
    mean = mean_init

    # Try multiple learning rates
    for lr in lr_candidates:
        mean = mean_init
        for _ in range(max_iters):
            # Compute logarithmic map of all points with respect to current mean
            # logmap expects single points, so we vmap over the batch
            log_x_fn = jax.vmap(hyperboloid.logmap, in_axes=(0, None, None))
            log_x = log_x_fn(x, mean[0], c)  # shape: (n_points, dim+1)

            # Sum tangent vectors and average
            log_x_sum = jnp.sum(log_x, axis=0, keepdims=True)  # shape: (1, dim+1)
            update = lr * log_x_sum / n_points

            # Compute update norm for convergence check
            update_norm = jnp.linalg.norm(update, axis=-1)

            # Update mean using exponential map
            mean = hyperboloid.expmap(update[0], mean[0], c)
            mean = mean[jnp.newaxis, :]  # shape: (1, dim+1)

            # Check convergence
            if update_norm < tol:
                has_converged = True
                break

        if has_converged:
            break

    # If no learning rate converged, print warning (in JAX we just use the last candidate)
    if not has_converged:
        print(
            "compute_frechet_mean: No convergence with any learning rate. Using the best candidate mean.",
            flush=True,
        )

    return mean


def center_data(
    x: Float[Array, "n_points dim_plus_1"],
    mean: Float[Array, "1 dim_plus_1"],
    c: Float[Array, ""] | float,
) -> Float[Array, "n_points dim_plus_1"]:
    """Center hyperboloid points around their Fréchet mean using Lorentz transformation.

    This function computes the Lorentz boost that maps the mean to the hyperboloid's
    origin and applies it to all data points.

    Args:
        x: Hyperboloid points of shape (n_points, dim+1)
        mean: Fréchet mean of shape (1, dim+1)
        c: Curvature parameter (positive scalar)

    Returns:
        Centered hyperboloid points of shape (n_points, dim+1)

    Notes:
        The Lorentz transformation is constructed from the velocity and gamma factor
        of the mean point. This is more numerically stable than using logmap/expmap.
    """
    # Extract mean components (squeeze from (1, dim+1) to (dim+1,))
    mean_squeezed = mean[0]  # shape: (dim+1,)
    mean_t = mean_squeezed[0:1]  # temporal component, shape: (1,)
    mean_space = mean_squeezed[1:]  # spatial components, shape: (dim,)

    # Compute Lorentz boost parameters
    sqrt_c = jnp.sqrt(c)
    gamma = mean_t * sqrt_c  # shape: (1,)
    velocity = mean_space / mean_t  # shape: (dim,)

    # Build Lorentz boost matrix (transposed version for right multiplication)
    # Top-left block: gamma (scalar)
    block_tl = gamma  # shape: (1,)

    # Top-right block: -gamma * v
    block_tr = -gamma * velocity  # shape: (dim,)

    # Top row: [gamma, -gamma * v]
    top_row = jnp.concatenate([block_tl, block_tr], axis=0)  # shape: (dim+1,)
    top_row = top_row[jnp.newaxis, :]  # shape: (1, dim+1)

    # Bottom-left block: -gamma * v^T
    block_bl = block_tr[:, jnp.newaxis]  # shape: (dim, 1)

    # Bottom-right block: I + (gamma^2 / (1 + gamma)) * v^T v
    dim = mean_space.shape[0]
    identity_n = jnp.eye(dim, dtype=x.dtype)
    vTv = velocity[:, jnp.newaxis] @ velocity[jnp.newaxis, :]  # shape: (dim, dim)
    coefficient = (gamma**2) / (1 + gamma)
    block_br = identity_n + coefficient * vTv  # shape: (dim, dim)

    # Bottom row: [-gamma * v^T, I + (gamma^2 / (1 + gamma)) * v^T v]
    bottom_row = jnp.concatenate([block_bl, block_br], axis=1)  # shape: (dim, dim+1)

    # Full Lorentz boost matrix
    lorentz_boost = jnp.concatenate([top_row, bottom_row], axis=0)  # shape: (dim+1, dim+1)

    # Apply Lorentz transformation: x @ L^T (or equivalently, L @ x^T)
    res = x @ lorentz_boost  # shape: (n_points, dim+1)

    # Project back onto hyperboloid
    res = jax.vmap(hyperboloid.proj, in_axes=(0, None))(res, c)

    return res


class HoroPCA(nnx.Module):
    """Horospherical PCA for hyperbolic dimensionality reduction.

    This class implements HoroPCA using JAX and Flax. It supports both Poincaré ball
    and hyperboloid manifolds. The principal components are represented as ideal points
    in the hyperboloid model's null cone.

    Attributes:
        n_components: Target dimensionality
        n_in_features: Input dimensionality (for hyperboloid: dim+1, for Poincaré: dim)
        manifold_name: Either "poincare" or "hyperboloid"
        c: Curvature parameter (nnx.Variable)
        lr: Learning rate for optimization
        max_steps: Maximum optimization steps
        Q: Principal component parameters (nnx.Param)
        data_mean: Stored Fréchet mean (nnx.Variable, optional)

    References:
        Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
            International Conference on Machine Learning (2021).
    """

    def __init__(
        self,
        n_components: int,
        n_in_features: int,
        manifold_name: str,
        c: float,
        lr: float = 1e-3,
        max_steps: int = 100,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize HoroPCA module.

        Args:
            n_components: Target dimensionality after reduction
            n_in_features: Input dimensionality
                - For Poincaré ball: dim (intrinsic dimension)
                - For Hyperboloid: dim+1 (ambient dimension)
            manifold_name: Either "poincare" or "hyperboloid"
            c: Initial curvature parameter (positive)
            lr: Learning rate for gradient descent
            max_steps: Maximum optimization iterations
            rngs: Flax random number generator state
        """
        self.n_components = n_components
        self.n_in_features = n_in_features
        self.manifold_name = manifold_name
        self.lr = lr
        self.max_steps = max_steps

        # Store curvature as a variable (can be made trainable if needed)
        self.c = nnx.Variable(jnp.array(c, dtype=jnp.float32))

        # Initialize data_mean as None (will be set during fit)
        self.data_mean: nnx.Variable[Float[Array, "1 dim_plus_1"] | None] = nnx.Variable(None)
        self.loss_history = nnx.Variable(jnp.zeros((0,), dtype=jnp.float32))

        # Initialize principal components Q based on manifold type
        if manifold_name == "poincare":
            # For Poincaré: Q has shape (n_components, n_in_features)
            # These represent directions in the Poincaré ball that will be mapped to ideals
            self.Q = nnx.Param(
                jax.random.normal(
                    rngs.params(),
                    shape=(n_components, n_in_features),
                    dtype=jnp.float32,
                )
            )
        elif manifold_name == "hyperboloid":
            # For Hyperboloid: Q has shape (n_components, n_in_features - 1)
            # These represent spatial components that will be augmented to ideals
            self.Q = nnx.Param(
                jax.random.normal(
                    rngs.params(),
                    shape=(n_components, n_in_features - 1),
                    dtype=jnp.float32,
                )
            )
        else:
            raise ValueError(f"Unsupported manifold: {manifold_name}. Use 'poincare' or 'hyperboloid'.")

    def _to_hyperboloid_ideals(self, ideals: Float[Array, "n_components dim"]) -> Float[Array, "n_components dim_plus_1"]:
        """Convert orthonormalized ideal points to hyperboloid null cone directions.

        Ideal points in the hyperboloid are represented by directions of 1-dimensional
        null cones. The input ideals should be orthonormalized in their spatial components.

        Args:
            ideals: Orthonormalized spatial components of shape (n_components, dim)

        Returns:
            Hyperboloid ideal points of shape (n_components, dim+1)
                with temporal component = 1 and spatial components = ideals
        """
        # Prepend ones to temporal component: [1, ideals[0], ideals[1], ...]
        ones = jnp.ones((ideals.shape[0], 1), dtype=ideals.dtype)
        res = jnp.concatenate([ones, ideals], axis=-1)
        return res

    def _horo_projection(
        self,
        x: Float[Array, "n_points dim_plus_1"],
        Q: Float[Array, "n_components dim_plus_1"],
        c: Float[Array, ""] | float,
    ) -> Float[Array, "n_points dim_plus_1"]:
        """Compute horospherical projections onto the geodesic submanifold.

        This projects hyperboloid points onto a lower-dimensional geodesic submanifold
        spanned by ideal points, preserving distances from the "spine" (the projection
        onto the spine itself).

        Args:
            x: Hyperboloid points of shape (n_points, dim+1)
            Q: Hyperboloid ideal points of shape (n_components, dim_plus_1)
            c: Curvature parameter

        Returns:
            Projected points of shape (n_points, dim+1)
        """

        return _horo_projection_points(x, Q, c)

    def compute_loss(self, x: Float[Array, "n_points dim_plus_1"]) -> Float[Array, ""]:
        """Compute negative generalized variance as loss function.

        The loss is the negative mean of squared pairwise distances in the projected
        space. Maximizing variance encourages spreading out the projected points.

        Args:
            x: Hyperboloid points of shape (n_points, dim+1)

        Returns:
            Negative generalized variance (scalar)
        """
        return _projected_variance_loss(self.Q[...], x, self.c[...])

    def fit(self, x: Float[Array, "n_points n_in_features"]) -> None:
        """Fit HoroPCA to the data by finding optimal principal components.

        This method:
        1. Converts data to hyperboloid if needed
        2. Computes Fréchet mean
        3. Centers data around the mean
        4. Optimizes principal components to maximize projected variance

        Args:
            x: Input points of shape (n_points, n_in_features)
                - For Poincaré: shape is (n_points, dim)
                - For Hyperboloid: shape is (n_points, dim+1)
        """
        c = self.c[...]

        # Convert to hyperboloid if needed
        if self.manifold_name == "poincare":
            # Map each Poincaré point to hyperboloid
            def poincare_to_hyperboloid(p):
                # First, convert single Poincaré point to hyperboloid
                # We need to use the poincare module's conversion logic
                # For now, we'll compute it manually using the conformal model
                p_sqnorm = jnp.dot(p, p)
                denom = 1.0 - c * p_sqnorm
                x0 = (1.0 + c * p_sqnorm) / denom
                x_rest = 2 * jnp.sqrt(c) * p / denom
                return jnp.concatenate([x0[jnp.newaxis], x_rest])

            x = jax.vmap(poincare_to_hyperboloid)(x)

        # Compute Fréchet mean
        self.data_mean.set_value(compute_frechet_mean(x, c))

        # Center data around Fréchet mean
        x_centered = center_data(x, self.data_mean[...], c)

        # Set up optimizer
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(self.Q)

        loss_and_grad = jax.jit(jax.value_and_grad(_projected_variance_loss))
        loss_history: list[float] = []

        for _ in range(self.max_steps):
            loss, grads = loss_and_grad(self.Q[...], x_centered, c)
            loss_history.append(float(loss))

            grads = jnp.clip(grads, -1e5, 1e5)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            updated_Q = cast(Float[Array, "n_components dim_or_dim_minus_1"], optax.apply_updates(self.Q[...], updates))
            self.Q[...] = updated_Q

        if loss_history:
            self.loss_history.set_value(jnp.asarray(loss_history, dtype=jnp.float32))
        else:
            self.loss_history.set_value(jnp.zeros((0,), dtype=jnp.float32))

    def transform(
        self,
        x: Float[Array, "n_points n_in_features"],
        recompute_mean: bool = False,
    ) -> Float[Array, "n_points n_components"]:
        """Project points onto the learned lower-dimensional submanifold.

        Args:
            x: Input points of shape (n_points, n_in_features)
            recompute_mean: If True, recompute Fréchet mean (default: False)

        Returns:
            Projected points of shape (n_points, n_components) in Poincaré ball

        Notes:
            The output is in the lower-dimensional Poincaré ball, represented
            by coordinates in the orthonormalized principal component basis.
        """
        c = self.c[...]

        # Convert to hyperboloid if needed
        if self.manifold_name == "poincare":

            def poincare_to_hyperboloid(p):
                p_sqnorm = jnp.dot(p, p)
                denom = 1.0 - c * p_sqnorm
                x0 = (1.0 + c * p_sqnorm) / denom
                x_rest = 2 * jnp.sqrt(c) * p / denom
                return jnp.concatenate([x0[jnp.newaxis], x_rest])

            x = jax.vmap(poincare_to_hyperboloid)(x)

        # Compute or recompute mean
        if recompute_mean or self.data_mean[...] is None:
            self.data_mean.set_value(compute_frechet_mean(x, c))

        # Center data
        x_centered = center_data(x, self.data_mean[...], c)

        # Orthonormalize principal components
        Q_ortho, _ = jnp.linalg.qr(self.Q[...].T, mode="reduced")
        Q_ortho = Q_ortho.T  # shape: (n_components, dim)

        # Map to hyperboloid ideals
        hyperboloid_ideals = self._to_hyperboloid_ideals(Q_ortho)

        # Project onto submanifold
        x_proj = self._horo_projection(x_centered, hyperboloid_ideals, c)

        # Convert back to Poincaré ball
        def hyperboloid_to_poincare(h):
            # h = [h0, h1, h2, ...]
            # Poincaré point = h_rest / (sqrt(c) * (h0 + 1/sqrt(c)))
            sqrt_c = jnp.sqrt(c)
            h_rest = h[1:]
            denom = sqrt_c * (h[0] + 1.0 / sqrt_c)
            return h_rest / denom

        x_poincare = jax.vmap(hyperboloid_to_poincare)(x_proj)  # shape: (n_points, dim)

        # Compute coordinates in lower-dimensional Poincaré ball
        res = x_poincare @ Q_ortho.T  # shape: (n_points, n_components)

        return res
