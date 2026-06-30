"""Gyrogroup normalization layers for the Hyperboloid and Proper Velocity manifolds.

This module implements two *intrinsic* normalization families that operate directly
on manifold points via gyrovector operations (``addition``, ``scalar_mul``) — no
tangent-space round trips for the affine part — for both the Lorentz/Hyperboloid and
the Proper Velocity (PV) models.

1. **Gyrogroup Batch Normalization** (``HyperboloidGyroBatchNorm``,
   ``ProperVelocityGyroBatchNorm``). Port of GyroBN (Chen et al., "Gyrogroup Batch
   Normalization", ICLR 2024; "Riemannian Batch Normalization: A Gyro Approach",
   2025). For a batch of points it computes a mean point ``mu`` and a scalar Fréchet
   variance, then::

       y = w ⊕ ( (gamma / sqrt(var + eps)) ⊗ ( (⊖mu) ⊕ x ) )

   i.e. *center* by gyro-translating with the inverse batch mean, *scale* by a
   gyro-scalar-multiplication with the inverse Fréchet standard deviation, and *bias*
   by gyro-translating with a learned manifold point ``w = expmap_0(weight)``. Running
   statistics are kept for evaluation, mirroring :class:`PoincareBatchNorm2D`.

2. **Gyro radial RMSNorm** (``HyperboloidGyroRMSNorm``, ``ProperVelocityGyroRMSNorm``).
   A *per-sample*, batch-independent normalizer (no running statistics, identical in
   train and eval, valid at batch size 1 — the properties RL workloads want). Each
   point's geodesic radius is rescaled to a learned target ``gamma`` via a single
   gyro-scalar-multiplication::

       y = (gamma / (dist_0(x) + eps)) ⊗ x      (optionally then  w ⊕ y)

   Because ``scalar_mul`` maps the geodesic radius ``dist_0(x) -> |r| * dist_0(x)``,
   choosing ``r = gamma / (dist_0(x) + eps)`` sends every sample to radius ≈ ``gamma``.
   This is the manifold analog of (Euclidean) RMSNorm: it normalizes magnitude (the
   hierarchy *depth*) while preserving direction; the optional gyro-bias reintroduces
   a learned offset.

Design notes
------------
- A feature-axis LayerNorm/RMSNorm does **not** reuse the GyroBN machinery: GyroBN
  reduces over a *batch of manifold points* to obtain a single mean *point*, whereas
  RMS/LN reduces *per-sample*. The two families therefore share only small helpers
  (manifold storage, the spatial⇄ambient adapter, output projection), not a base.
- The single per-manifold asymmetry is the time coordinate: a Hyperboloid point is
  ambient ``(D+1,)`` while a PV point is spatial ``(D,)``. This collapses to one
  integer ``_time_dims`` (1 for Hyperboloid, 0 for PV) plus the manifold's own
  ``embed_spatial_0`` lift, so the concrete subclasses are tiny.
- ``num_features`` is the **spatial** dimension ``D`` (matching ``FGGMeanOnlyBatchNorm``
  and the HRC ``*Norm`` family). Inputs are on-manifold points whose last axis is
  ambient ``D+1`` (Hyperboloid) or ``D`` (PV).

Dimension key
-------------
  B: batch size (and any extra leading dims; flattened together)
  N: flattened number of points (product of all leading dims)
  F: feature dim of an input point  (ambient: D+1 for Hyperboloid, D for PV)
  D: spatial feature dim (== ``num_features``)

References
----------
Chen et al. "Gyrogroup Batch Normalization." ICLR 2024.
Chen et al. "Riemannian Batch Normalization: A Gyro Approach." 2025.
Shi et al. "Intrinsic Lorentz Neural Network." 2026 (Lorentz gyro addition / scaling).
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds import Manifold
from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.proper_velocity import ProperVelocity

from ._helpers import validate_hyperboloid_manifold, validate_pv_manifold
from .hyperboloid_core import lorentz_midpoint
from .poincare_batchnorm import frechet_variance

# Manifold methods every gyro-normalization layer relies on.
_GYRO_BN_METHODS = ("addition", "scalar_mul", "expmap_0", "logmap_0", "dist", "proj", "embed_spatial_0")
_GYRO_RMS_METHODS = ("addition", "scalar_mul", "expmap_0", "dist_0", "proj", "embed_spatial_0")


# ======================================================================================
# Family 1 — Gyrogroup Batch Normalization
# ======================================================================================


class GyroBatchNormBase(nnx.Module):
    """Shared GyroBN logic. Not instantiated directly — use the manifold subclasses.

    Subclasses set ``_time_dims`` (1 for Hyperboloid, 0 for PV) and override
    :meth:`_batch_mean` with the manifold's batch-mean estimator.

    Parameters
    ----------
    manifold_module : Manifold
        Hyperboloid or ProperVelocity instance (supplies the gyro operations).
    num_features : int
        Spatial feature dimension ``D``.
    use_running_average : bool
        If True, use running statistics instead of batch statistics (default: False).
        Overridable at call time.
    momentum : float
        EMA retention factor for running statistics (``new = momentum * old +
        (1 - momentum) * batch``; default: 0.9, matching :class:`PoincareBatchNorm2D`).
    eps : float
        Numerical stability floor on the variance denominator (default: 1e-6).
    param_dtype : DTypeLike
        Storage dtype of learnable parameters and running statistics (default:
        ``jnp.float32``). Compute precision is set by ``manifold.dtype``.
    """

    # Number of leading (time) coordinates that are NOT spatial features.
    _time_dims: int = 0

    def __init__(
        self,
        manifold_module: Manifold,
        num_features: int,
        *,
        use_running_average: bool = False,
        momentum: float = 0.9,
        eps: float = 1e-6,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.manifold = manifold_module
        self.num_features = num_features
        self.use_running_average = use_running_average
        self.momentum = momentum
        self.eps = eps

        # Learnable affine parameters (spatial tangent / scalar). The bias is a
        # spatial tangent-at-origin vector; lifting it keeps its time coordinate
        # pinned to 0 (so expmap_0 is a clean origin-rooted translation).
        self.bias = nnx.Param(jnp.zeros((num_features,), dtype=param_dtype))  # GyroBN "weight"
        self.gamma = nnx.Param(jnp.ones((), dtype=param_dtype))  # GyroBN "shift" (scale gain)

        # Running statistics (spatial tangent-at-origin mean + scalar variance).
        self.running_mean = nnx.BatchStat(jnp.zeros((num_features,), dtype=param_dtype))
        self.running_var = nnx.BatchStat(jnp.ones((), dtype=param_dtype))

    # -- per-manifold hooks --------------------------------------------------------

    def _batch_mean(self, x_NF: Float[Array, "N F"], c: float) -> Float[Array, "F"]:
        """Estimate the batch mean point. Overridden per manifold."""
        raise NotImplementedError

    def _lift(self, v_D: Float[Array, "D"]) -> Float[Array, "F"]:
        """Spatial vector -> ambient tangent-at-origin (prepends time 0 on Hyperboloid)."""
        return self.manifold.embed_spatial_0(v_D)

    def _lower(self, v_F: Float[Array, "F"]) -> Float[Array, "D"]:
        """Ambient tangent-at-origin -> spatial vector (drops the time coordinate)."""
        return v_F[..., self._time_dims :]

    # -- forward -------------------------------------------------------------------

    def __call__(
        self,
        x: Float[Array, "... F"],
        c: float = 1.0,
        use_running_average: bool | None = None,
    ) -> Float[Array, "... F"]:
        """Apply gyrogroup batch normalization to on-manifold points.

        Parameters
        ----------
        x : Array, shape (..., F)
            On-manifold points. Last axis is ambient: ``D+1`` (Hyperboloid) or ``D`` (PV).
        c : float
            Curvature (positive, default: 1.0).
        use_running_average : bool or None
            Override the constructor setting. None uses the constructor value.
        """
        if use_running_average is None:
            use_running_average = self.use_running_average

        orig_shape = x.shape
        x_NF = x.reshape(-1, orig_shape[-1])  # flatten all leading dims

        if use_running_average:
            mu_F = self.manifold.expmap_0(self._lift(self.running_mean[...]), c)
            var = self.running_var[...]
        else:
            mu_F = self._batch_mean(x_NF, c)
            var = frechet_variance(x_NF, mu_F, self.manifold, c)

            # EMA update (no gradient flow; cast back to the stat storage dtype).
            rm_D = self._lower(self.manifold.logmap_0(mu_F, c))
            new_mean_D = self.momentum * self.running_mean[...] + (1.0 - self.momentum) * rm_D
            self.running_mean[...] = jax.lax.stop_gradient(new_mean_D).astype(self.running_mean[...].dtype)
            new_var = self.momentum * self.running_var[...] + (1.0 - self.momentum) * var
            self.running_var[...] = jax.lax.stop_gradient(new_var).astype(self.running_var[...].dtype)

        # Center: (⊖mu) ⊕ x  (gyro-inverse via reflection through the origin).
        inv_mu_F = self.manifold.scalar_mul(-1.0, mu_F, c)
        x_cent_NF = jax.vmap(self.manifold.addition, in_axes=(None, 0, None))(inv_mu_F, x_NF, c)

        # Scale: (gamma / sqrt(var + eps)) ⊗ x_centered.
        factor = self.gamma[...] / jnp.sqrt(var + self.eps)
        x_scaled_NF = jax.vmap(self.manifold.scalar_mul, in_axes=(None, 0, None))(factor, x_cent_NF, c)

        # Bias: w ⊕ x_scaled  with  w = expmap_0(lift(bias)).
        bias_pt_F = self.manifold.expmap_0(self._lift(self.bias[...]), c)
        x_out_NF = jax.vmap(self.manifold.addition, in_axes=(None, 0, None))(bias_pt_F, x_scaled_NF, c)

        x_out_NF = jax.vmap(self.manifold.proj, in_axes=(0, None))(x_out_NF, c)
        return x_out_NF.reshape(orig_shape)


class HyperboloidGyroBatchNorm(GyroBatchNormBase):
    """GyroBN for the Hyperboloid (Lorentz) model.

    Inputs are ambient ``(..., D+1)`` hyperboloid points; ``num_features`` is the
    spatial dimension ``D``. The batch mean is the closed-form Lorentz centroid
    (HELM, Chen et al. 2024) via :func:`lorentz_midpoint` — exact and JIT-friendly,
    matching the estimator the ILNN GyroBN reference uses in practice.
    """

    _time_dims = 1

    def __init__(self, manifold_module: Hyperboloid, num_features: int, **kwargs):
        validate_hyperboloid_manifold(manifold_module, required_methods=_GYRO_BN_METHODS)
        super().__init__(manifold_module, num_features, **kwargs)

    def _batch_mean(self, x_NF: Float[Array, "N F"], c: float) -> Float[Array, "F"]:
        n = x_NF.shape[0]
        weights_1N = jnp.full((1, n), 1.0 / n, dtype=x_NF.dtype)  # uniform centroid
        return lorentz_midpoint(x_NF, weights_1N, c)[0]


class ProperVelocityGyroBatchNorm(GyroBatchNormBase):
    """GyroBN for the Proper Velocity (PV) model.

    Inputs are ``(..., D)`` PV points; ``num_features = D``. PV has no closed-form
    centroid, so the batch mean is the closed-form **log-Euclidean** mean
    ``expmap_0(mean_i logmap_0(x_i))`` (the GyroBN reference's ``use_euclid_stats``
    mode): no iteration, fully vmap/JIT-clean.
    """

    _time_dims = 0

    def __init__(self, manifold_module: ProperVelocity, num_features: int, **kwargs):
        validate_pv_manifold(manifold_module, required_methods=_GYRO_BN_METHODS)
        super().__init__(manifold_module, num_features, **kwargs)

    def _batch_mean(self, x_NF: Float[Array, "N F"], c: float) -> Float[Array, "F"]:
        v_NF = jax.vmap(self.manifold.logmap_0, in_axes=(0, None))(x_NF, c)
        v_mean_F = jnp.mean(v_NF, axis=0)
        return self.manifold.expmap_0(v_mean_F, c)


# ======================================================================================
# Family 2 — Gyro radial RMSNorm
# ======================================================================================


class GyroRMSNormBase(nnx.Module):
    """Shared gyro radial RMSNorm logic. Not instantiated directly.

    Per-sample, batch-independent: no running statistics, no train/eval distinction.

    Parameters
    ----------
    manifold_module : Manifold
        Hyperboloid or ProperVelocity instance.
    num_features : int
        Spatial feature dimension ``D`` (only used when ``use_bias=True``).
    use_bias : bool
        If True, apply a learned gyro-bias after the radial rescale (default: False,
        matching Euclidean RMSNorm which has a gain but no bias).
    eps : float
        Numerical stability floor on the radius denominator (default: 1e-6).
    param_dtype : DTypeLike
        Storage dtype of learnable parameters (default: ``jnp.float32``).
    """

    def __init__(
        self,
        manifold_module: Manifold,
        num_features: int,
        *,
        use_bias: bool = False,
        eps: float = 1e-6,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.manifold = manifold_module
        self.num_features = num_features
        self.use_bias = use_bias
        self.eps = eps

        # Learned target geodesic radius (gain). Bias created only when requested.
        self.gamma = nnx.Param(jnp.ones((), dtype=param_dtype))
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((num_features,), dtype=param_dtype))

    def _lift(self, v_D: Float[Array, "D"]) -> Float[Array, "F"]:
        """Spatial vector -> ambient tangent-at-origin (prepends time 0 on Hyperboloid)."""
        return self.manifold.embed_spatial_0(v_D)

    def __call__(self, x: Float[Array, "... F"], c: float = 1.0) -> Float[Array, "... F"]:
        """Apply gyro radial RMS normalization to on-manifold points.

        Parameters
        ----------
        x : Array, shape (..., F)
            On-manifold points. Last axis is ambient: ``D+1`` (Hyperboloid) or ``D`` (PV).
        c : float
            Curvature (positive, default: 1.0).
        """
        orig_shape = x.shape
        x_NF = x.reshape(-1, orig_shape[-1])

        # Per-sample radius normalization: dist_0 -> ~gamma via a single gyro scaling.
        r_N = jax.vmap(self.manifold.dist_0, in_axes=(0, None))(x_NF, c)
        factor_N = self.gamma[...] / (r_N + self.eps)
        out_NF = jax.vmap(self.manifold.scalar_mul, in_axes=(0, 0, None))(factor_N, x_NF, c)

        if self.use_bias:
            bias_pt_F = self.manifold.expmap_0(self._lift(self.bias[...]), c)
            out_NF = jax.vmap(self.manifold.addition, in_axes=(None, 0, None))(bias_pt_F, out_NF, c)

        out_NF = jax.vmap(self.manifold.proj, in_axes=(0, None))(out_NF, c)
        return out_NF.reshape(orig_shape)


class HyperboloidGyroRMSNorm(GyroRMSNormBase):
    """Gyro radial RMSNorm for the Hyperboloid (Lorentz) model.

    Inputs are ambient ``(..., D+1)`` hyperboloid points; ``num_features = D``.
    """

    def __init__(self, manifold_module: Hyperboloid, num_features: int, **kwargs):
        validate_hyperboloid_manifold(manifold_module, required_methods=_GYRO_RMS_METHODS)
        super().__init__(manifold_module, num_features, **kwargs)


class ProperVelocityGyroRMSNorm(GyroRMSNormBase):
    """Gyro radial RMSNorm for the Proper Velocity (PV) model.

    Inputs are ``(..., D)`` PV points; ``num_features = D``.
    """

    def __init__(self, manifold_module: ProperVelocity, num_features: int, **kwargs):
        validate_pv_manifold(manifold_module, required_methods=_GYRO_RMS_METHODS)
        super().__init__(manifold_module, num_features, **kwargs)
