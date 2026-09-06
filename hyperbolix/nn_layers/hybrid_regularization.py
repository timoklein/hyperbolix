"""Hyper++ feature scaling for hybrid Euclidean-hyperbolic networks.

Applied after the last Euclidean layer and before expmap_0 to either
the Poincare ball or Hyperboloid.  The layer operates entirely in
Euclidean space but uses hyperbolic geometry to bound the scaling:
rho_max = atanh(alpha) / sqrt(c), which controls both the Poincare
norm and the Hyperboloid time component.

Pipeline (4 sequential steps):
1. RMSNorm (parameter-free, always applied): x / sqrt(mean(x^2) + eps)
2. Lipschitz activation: default tanh, configurable, None to skip
3. Dimension scaling: x * (1 / sqrt(d))
4. Learned rescaling (when alpha is not None):
   rho_max = atanh(alpha) / sqrt(c)
   x_rescale = rho_max * sigmoid(xi_theta(x)) * x

References
----------
van Spengler et al. "Poincare ResNet" (2023), Section 3.2.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

"""
Dimension key:
  B: batch size
  D: embedding dimension
"""


class HyperPPFeatureScaling(nnx.Module):
    """Hyper++ feature scaling for hybrid Euclidean-hyperbolic networks.

    Parameters
    ----------
    dim : int
        Embedding dimension (needed for nnx.Linear and 1/sqrt(d) scaling).
    alpha : float or None, optional
        Value in (0, 1) enabling learned rescaling. None disables it
        and the layer is entirely parameter-free. Default: None.
    activation : Callable or None, optional
        Lipschitz activation function. Default: jax.nn.tanh. None to skip.
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
    rngs : nnx.Rngs
        Random number generators for sub-layers.

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import HyperPPFeatureScaling
    >>>
    >>> # Parameter-free mode
    >>> layer = HyperPPFeatureScaling(dim=64, rngs=nnx.Rngs(0))
    >>> y = layer(x, c=1.0)
    >>>
    >>> # Learned rescaling mode
    >>> layer = HyperPPFeatureScaling(dim=64, alpha=0.9, rngs=nnx.Rngs(0))
    >>> y = layer(x, c=0.1)
    """

    def __init__(
        self,
        dim: int,
        *,
        alpha: float | None = None,
        activation: Callable | None = jax.nn.tanh,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if alpha is not None:
            if not (0.0 < alpha < 1.0):
                msg = f"alpha must be in (0, 1), got {alpha}"
                raise ValueError(msg)
            self._atanh_alpha = math.atanh(alpha)
            # Layer weight GEMM: no `precision` kwarg, so it follows JAX's own
            # `jax_default_matmul_precision` (TF32 on Ampere/Hopper). See hyperbolix.utils.precision.
            self.xi_theta = nnx.Linear(dim, 1, param_dtype=param_dtype, rngs=rngs)
        else:
            self._atanh_alpha = None
            self.xi_theta = None

        self.rms_norm = nnx.RMSNorm(dim, use_scale=False, param_dtype=param_dtype, rngs=rngs)
        self._inv_sqrt_dim = 1.0 / math.sqrt(dim)
        self.activation = activation

    def __call__(
        self,
        x_BD: Float[Array, "B D"],
        c: float = 1.0,
    ) -> Float[Array, "B D"]:
        """Apply Hyper++ feature scaling pipeline.

        Parameters
        ----------
        x_BD : Array of shape (B, D)
            Euclidean features from the last Euclidean layer.
        c : float
            Curvature parameter (positive). Passed at call time per
            library convention to support learnable curvature.

        Returns
        -------
        Array of shape (B, D)
            Scaled features ready for expmap_0.
        """
        # Step 1: RMSNorm (parameter-free, no learned scale)
        x_BD = self.rms_norm(x_BD)

        # Step 2: Lipschitz activation
        if self.activation is not None:
            x_BD = self.activation(x_BD)

        # Step 3: Dimension scaling
        x_BD = x_BD * self._inv_sqrt_dim

        # Step 4: Learned rescaling
        if self._atanh_alpha is not None:
            assert self.xi_theta is not None
            rho_max = self._atanh_alpha / jnp.sqrt(c)  # scalar
            scale_B1 = rho_max * jax.nn.sigmoid(self.xi_theta(x_BD))  # (B, 1)
            x_BD = scale_B1 * x_BD

        return x_BD
