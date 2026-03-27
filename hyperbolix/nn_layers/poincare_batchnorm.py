"""Poincaré batch normalization for JAX/Flax NNX.

Implements PoincareBatchNorm2D following the Poincaré ResNet
(van Spengler et al. 2023). The layer operates in tangent space
(matching conv layer I/O), mapping to the manifold internally
for geometric operations.

Algorithm:
    1. expmap_0(x) — tangent input → manifold
    2. expmap_0(self.mean) — learned mean → manifold
    3. Compute Einstein midpoint and Fréchet variance on manifold
    4. logmap(x, batch_midpoint) — log-map all points at midpoint
    5. ptransp(v, batch_midpoint, learned_mean) — parallel transport
    6. Rescale by sqrt(learned_var / (batch_var + eps))
    7. expmap(v, learned_mean) — map back to manifold at learned mean
    8. logmap_0(output) — manifold → tangent output

Dimension key:
  B: batch size
  H: height          W: width
  C: channels        N: flattened batch+spatial (B*H*W)
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds.poincare import Poincare

from ._helpers import validate_poincare_manifold


def poincare_midpoint(
    x_NC: Float[Array, "N C"],
    manifold: Poincare,
    c: float,
    eps: float = 1e-6,
) -> Float[Array, "C"]:
    """Compute Einstein midpoint of Poincaré ball points.

    Uses conformal factor weighting: midpoint = Σ(λ²·x) / Σ(λ²),
    then projects onto the ball.

    Parameters
    ----------
    x_NC : Array, shape (N, C)
        Points on the Poincaré ball.
    manifold : Poincare
        Poincaré manifold instance.
    c : float
        Curvature (positive).
    eps : float
        Numerical stability floor (default: 1e-6).

    Returns
    -------
    Array, shape (C,)
        Einstein midpoint on the Poincaré ball.
    """
    # lambda_N1: (N, 1) — conformal factor for each point
    lambda_N1 = manifold.conformal_factor(x_NC, c)  # (N, 1)
    lambda_sq_N1 = lambda_N1**2  # (N, 1)

    # Weighted sum: Σ(λ²·x) / Σ(λ²)
    numerator_C = jnp.sum(lambda_sq_N1 * x_NC, axis=0)  # (C,)
    denominator = jnp.sum(lambda_sq_N1, axis=0) + eps  # (1,)

    midpoint_C = numerator_C / denominator  # (C,)
    return manifold.proj(midpoint_C, c)


def frechet_variance(
    x_NC: Float[Array, "N C"],
    mean_C: Float[Array, "C"],
    manifold: Poincare,
    c: float,
) -> Float[Array, ""]:
    """Compute Fréchet variance: mean squared geodesic distance to mean.

    Parameters
    ----------
    x_NC : Array, shape (N, C)
        Points on the Poincaré ball.
    mean_C : Array, shape (C,)
        Mean point on the Poincaré ball.
    manifold : Poincare
        Poincaré manifold instance.
    c : float
        Curvature (positive).

    Returns
    -------
    Array, scalar
        Mean of squared geodesic distances.
    """
    # dist per point: (N,)
    dists_N = jax.vmap(manifold.dist, in_axes=(0, None, None))(x_NC, mean_C, c)
    return jnp.mean(dists_N**2)


class PoincareBatchNorm2D(nnx.Module):
    """Poincaré batch normalization for 2D feature maps.

    Operates in tangent space (matching conv layer I/O). Internally maps
    to the manifold for geometric operations (midpoint, parallel transport,
    variance rescaling), then maps back to tangent space.

    Follows ``nnx.BatchNorm`` interface: ``use_running_average`` is a
    constructor parameter overridable at call time.

    Parameters
    ----------
    manifold_module : Poincare
        Poincaré manifold instance.
    num_features : int
        Number of channels (Poincaré ball dimension per pixel).
    use_running_average : bool
        If True, use running statistics instead of batch statistics
        (default: False). Overridable at call time.
    momentum : float
        EMA momentum for running statistics (default: 0.9).
    eps : float
        Numerical stability floor (default: 1e-6).

    References
    ----------
    van Spengler et al. "Poincaré ResNet." ICML 2023.
    """

    def __init__(
        self,
        manifold_module: Poincare,
        num_features: int,
        *,
        use_running_average: bool = False,
        momentum: float = 0.9,
        eps: float = 1e-6,
    ):
        validate_poincare_manifold(
            manifold_module,
            required_methods=("expmap_0", "logmap_0", "logmap", "expmap", "ptransp", "proj", "dist", "conformal_factor"),
        )
        self.manifold = manifold_module
        self.num_features = num_features
        self.use_running_average = use_running_average
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters (tangent space)
        self.mean = nnx.Param(jnp.zeros((num_features,)))  # learned mean
        self.var = nnx.Param(jnp.ones(()))  # learned variance (scalar)

        # Running statistics (tangent space)
        self.running_mean = nnx.BatchStat(jnp.zeros((num_features,)))
        self.running_var = nnx.BatchStat(jnp.ones(()))

    def __call__(
        self,
        x_BHWC: Float[Array, "B H W C"],
        c: float = 1.0,
        use_running_average: bool | None = None,
    ) -> Float[Array, "B H W C"]:
        """Apply Poincaré batch normalization.

        Parameters
        ----------
        x_BHWC : Array, shape (B, H, W, C)
            Tangent-space input features.
        c : float
            Curvature (positive, default: 1.0).
        use_running_average : bool or None
            Override constructor setting. None uses constructor value.

        Returns
        -------
        Array, shape (B, H, W, C)
            Normalized tangent-space output.
        """
        # Resolve use_running_average: call-time > constructor
        if use_running_average is None:
            use_running_average = self.use_running_average

        B, H, W, C = x_BHWC.shape

        # Flatten (B, H, W, C) → (N, C)
        x_NC = x_BHWC.reshape(-1, C)  # (N, C)

        # --- Map to manifold ---
        # x_NC are tangent vectors; map to Poincaré ball
        x_manifold_NC = jax.vmap(self.manifold.expmap_0, in_axes=(0, None))(x_NC, c)  # (N, C)

        if use_running_average:
            # Use running stats (eval mode)
            batch_mean_tangent_C = self.running_mean[...]  # (C,)
            batch_var = self.running_var[...]  # scalar
        else:
            # Compute batch stats on manifold
            batch_midpoint_C = poincare_midpoint(x_manifold_NC, self.manifold, c, self.eps)  # (C,)
            batch_var = frechet_variance(x_manifold_NC, batch_midpoint_C, self.manifold, c)  # scalar

            # Map midpoint back to tangent space for running stat storage
            batch_mean_tangent_C = self.manifold.logmap_0(batch_midpoint_C, c)  # (C,)

            # Update running statistics (EMA, no gradient flow)
            self.running_mean[...] = jax.lax.stop_gradient(
                self.momentum * self.running_mean[...] + (1.0 - self.momentum) * batch_mean_tangent_C
            )
            self.running_var[...] = jax.lax.stop_gradient(
                self.momentum * self.running_var[...] + (1.0 - self.momentum) * batch_var
            )

        # Get the manifold-space mean to use for geometric operations
        # (either from batch or from running stats)
        if use_running_average:
            input_mean_C = self.manifold.expmap_0(batch_mean_tangent_C, c)  # (C,)
            current_var = batch_var
        else:
            input_mean_C = batch_midpoint_C  # already computed above  # (C,)
            current_var = batch_var

        # Learned mean on manifold
        learned_mean_C = self.manifold.expmap_0(self.mean[...], c)  # (C,)

        # --- Step 4: logmap all points at input mean ---
        v_NC = jax.vmap(self.manifold.logmap, in_axes=(0, None, None))(x_manifold_NC, input_mean_C, c)  # (N, C)

        # --- Step 5: parallel transport from input mean to learned mean ---
        v_NC = jax.vmap(self.manifold.ptransp, in_axes=(0, None, None, None))(v_NC, input_mean_C, learned_mean_C, c)  # (N, C)

        # --- Step 6: rescale by sqrt(learned_var / (batch_var + eps)) ---
        scale = jnp.sqrt(self.var[...] / (current_var + self.eps))  # scalar
        v_NC = v_NC * scale  # (N, C)

        # --- Step 7: expmap at learned mean ---
        out_NC = jax.vmap(self.manifold.expmap, in_axes=(0, None, None))(v_NC, learned_mean_C, c)  # (N, C)

        # --- Step 8: logmap_0 back to tangent space ---
        out_NC = jax.vmap(self.manifold.logmap_0, in_axes=(0, None))(out_NC, c)  # (N, C)

        # Reshape back to (B, H, W, C)
        return out_NC.reshape(B, H, W, C)
