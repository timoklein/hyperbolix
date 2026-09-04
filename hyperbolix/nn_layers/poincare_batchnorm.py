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
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.manifolds.protocol import Manifold
from hyperbolix.utils.precision import gemm_precision

from ._helpers import validate_poincare_manifold


def poincare_midpoint(
    x_NC: Float[Array, "N C"],
    manifold: Poincare,
    c: float,
    eps: float = 1e-6,
) -> Float[Array, "C"]:
    """Compute the (Einstein) gyromidpoint of Poincaré ball points.

    Delegates to :func:`poincare_weighted_midpoint` (GGBall Eq. 41) with uniform
    weights::

        mu = (1/2) ⊗_c [ Σ λ(x_n)·x_n / Σ (λ(x_n) - 1) ]

    NOTE: an earlier version computed ``Σ(λ²·x) / Σ(λ²)``, which is *not* a
    midpoint (it is not equidistant for two points and is biased toward points
    near the ball boundary, where λ explodes).

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
        Gyromidpoint on the Poincaré ball.
    """
    weights_1N = jnp.ones((1, x_NC.shape[0]), dtype=x_NC.dtype)  # (1, N) uniform
    return poincare_weighted_midpoint(x_NC, weights_1N, manifold, c, eps)[0]


def poincare_weighted_midpoint(
    points_MC: Float[Array, "M C"],
    weights_NM: Float[Array, "N M"],
    manifold: Poincare,
    c: float,
    eps: float = 1e-6,
) -> Float[Array, "N C"]:
    """Weighted Poincaré gyromidpoint over M points (GGBall Eq. 41).

    Generalises :func:`poincare_midpoint` (the unweighted Einstein midpoint of N
    points) to N *weighted* midpoints, mirroring the hyperboloid
    :func:`hyperbolix.nn_layers.lorentz_midpoint`. For weight-vector ``w_n`` over
    the M points::

        mu_n = (1/2) ⊗_c [ Σ_m w_nm · λ(x_m) · x_m  /  Σ_m w_nm · (λ(x_m) - 1) ]

    where ``λ(x) = 2 / (1 - c‖x‖²)`` is the conformal factor and ``⊗_c`` is
    Möbius scalar multiplication. The ``½ ⊗_c`` half-scaling is exactly what turns
    the conformal-weighted Euclidean average into the gyro-midpoint *on* the ball
    (for N copies of a single point it returns that point — see the algebra in
    the tests).

    This is the codebook-centroid operator of the hyperbolic-EMA VQ update
    (GGBall, Bu et al. 2026): pass the assigned encoder points as ``points`` and
    the transposed one-hot assignment matrix ``(num_codes, N)`` as ``weights`` to
    get one midpoint per code.

    Parameters
    ----------
    points_MC : Array, shape (M, C)
        Points on the Poincaré ball with curvature ``c``.
    weights_NM : Array, shape (N, M)
        Combination weights, one row per output midpoint. Typically non-negative
        (assignment / attention weights). Because ``λ(x) ≥ 2`` the denominator
        ``Σ_m w_nm (λ-1)`` is strictly positive whenever a row carries any mass;
        an all-zero row maps to the origin.
    manifold : Poincare
        Poincaré manifold instance (supplies ``conformal_factor``, ``scalar_mul``,
        ``proj``).
    c : float
        Curvature (positive).
    eps : float
        Denominator floor guarding the empty-row 0/0 (default: 1e-6).

    Returns
    -------
    Array, shape (N, C)
        Weighted gyromidpoints on the Poincaré ball.
    """
    # lambda per point — conformal_factor is batched and returns (M, 1).
    lambda_M = manifold.conformal_factor(points_MC, c)[:, 0]  # (M,)

    # Eq. 41 numerator / denominator, contracted over the M axis:
    #   numerator_n = Σ_m w_nm · λ(x_m) · x_m   (N, C)
    #   denom_n     = Σ_m w_nm · (λ(x_m) - 1)   (N,)
    # Both contractions are training-path GEMMs and follow the user's JAX matmul precision
    # (GEMM_PRECISION overrides; see hyperbolix.utils.precision).
    numerator_NC = jnp.einsum("nm,mc->nc", weights_NM, lambda_M[:, None] * points_MC, precision=gemm_precision())  # (N, C)
    denom_N = jnp.einsum("nm,m->n", weights_NM, lambda_M - 1.0, precision=gemm_precision())  # (N,)
    # Empty row (all-zero weights) -> denom 0 and numerator 0; map to origin
    # rather than 0/0. λ ≥ 2 keeps denom > 0 for any row with mass.
    denom_safe_N = jnp.where(jnp.abs(denom_N) < eps, 1.0, denom_N)  # (N,)
    inner_NC = numerator_NC / denom_safe_N[:, None]  # (N, C) — argument of ½ ⊗_c

    # Boundary guard before the Möbius half-scaling (keeps scalar_mul's atanh in domain).
    inner_NC = jax.vmap(manifold.proj, in_axes=(0, None))(inner_NC, c)

    # mu_n = ½ ⊗_c inner_n  (single-point scalar_mul, vmapped over N).
    mu_NC = jax.vmap(lambda v: manifold.scalar_mul(0.5, v, c))(inner_NC)  # (N, C)
    return jax.vmap(manifold.proj, in_axes=(0, None))(mu_NC, c)


def frechet_variance(
    x_NC: Float[Array, "N C"],
    mean_C: Float[Array, "C"],
    manifold: Manifold,
    c: float,
) -> Float[Array, ""]:
    """Compute Fréchet variance: mean squared geodesic distance to mean.

    Manifold-generic: only ``dist`` is used, so any ``Manifold`` works — the gyro
    normalization layers call this with Hyperboloid and ProperVelocity instances too.

    Parameters
    ----------
    x_NC : Array, shape (N, C)
        Points on the manifold.
    mean_C : Array, shape (C,)
        Mean point on the manifold.
    manifold : Manifold
        Manifold instance (anything satisfying the ``Manifold`` protocol).
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
    param_dtype : DTypeLike
        Storage dtype of the learnable parameters and running statistics
        (default: jnp.float32). Compute precision of manifold operations is
        set by ``manifold.dtype``.

    Notes
    -----
    The learned variance ``self.var`` is an unconstrained scalar used as
    ``sqrt(var / (batch_var + eps))`` — if gradient descent drives it
    negative, the sqrt produces NaN. This matches the reference (van
    Spengler's PoincareBatchNorm uses an unconstrained ``nn.Parameter``
    under the same sqrt), so we keep the parameterization; in practice the
    init at 1.0 and typical learning rates keep it positive. If you observe
    NaNs originating here, reparameterize via softplus in your model or
    clamp ``self.var`` after optimizer steps.

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
        param_dtype: DTypeLike = jnp.float32,
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

        # Learnable parameters and running stats (tangent space). Storage is
        # pinned to param_dtype; manifold operations re-promote to the
        # manifold's compute dtype at their boundaries, and the EMA updates in
        # __call__ cast back to the stat dtype at store time.
        self.mean = nnx.Param(jnp.zeros((num_features,), dtype=param_dtype))  # learned mean
        self.var = nnx.Param(jnp.ones((), dtype=param_dtype))  # learned variance (scalar)

        # Running statistics (tangent space)
        self.running_mean = nnx.BatchStat(jnp.zeros((num_features,), dtype=param_dtype))
        self.running_var = nnx.BatchStat(jnp.ones((), dtype=param_dtype))

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

            # Update running statistics (EMA, no gradient flow). The batch
            # stats come from manifold ops (manifold.dtype), so cast back to
            # the stat storage dtype at store time.
            new_running_mean_C = self.momentum * self.running_mean[...] + (1.0 - self.momentum) * batch_mean_tangent_C
            self.running_mean[...] = jax.lax.stop_gradient(new_running_mean_C).astype(self.running_mean[...].dtype)
            new_running_var = self.momentum * self.running_var[...] + (1.0 - self.momentum) * batch_var
            self.running_var[...] = jax.lax.stop_gradient(new_running_var).astype(self.running_var[...].dtype)

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
