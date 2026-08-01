"""Busemann multinomial logistic regression (BMLR) heads for JAX/Flax NNX.

Dimension key:
  B: batch size
  I: in_spatial (direction dim)        K: num_classes
  Ai: in_ambient (Hyperboloid in_features = I + 1)

Both heads return Euclidean logits via the Busemann point-to-horosphere distance
(Chen et al. 2026, Eq. 8): ``u_k(x) = -alpha_k·B^{v_k}(x) + b_k``. This is the
point-to-*horosphere* lineage (Chen/Atigh/Fan), distinct from the point-to-*hyperplane*
MLRs ``HypRegressionHyperboloid`` (Bdeir et al. 2023) and ``HypRegressionPoincarePP``
(Shimizu et al. 2020). The Lorentz BMLR is the fastest hyperbolic MLR in the paper.
"""

import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare

from ._helpers import validate_hyperboloid_manifold, validate_poincare_manifold
from .busemann_core import _busemann_score, _init_weight_norm_params


class HypRegressionHyperboloidBusemann(nnx.Module):
    """Busemann MLR classification head (Hyperboloid / Lorentz model).

    Computes per-class logits ``u_k(x) = -alpha_k·B^{v_k}(x) + b_k`` from the Lorentz Busemann
    function (point-to-horosphere distance), returning Euclidean logits — not manifold points.

    Parameters
    ----------
    manifold_module : Hyperboloid
        Class-based Hyperboloid manifold instance.
    in_dim : int
        Input ambient dimension (``d + 1``, time included).
    out_dim : int
        Number of classes.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    input_space : str
        ``"manifold"`` (default) or ``"tangent"`` (lift via ``expmap_0`` first). Static for JIT.
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32). Compute precision
        of manifold operations is set by ``manifold.dtype``.

    Notes
    -----
    Parameters use the weight-normalization split (``kernel`` directions, ``log_scale``
    log-magnitudes ``alpha = exp(log_scale)``, ``bias``); see :func:`busemann_core._busemann_score`.

    References
    ----------
    Chen, Schölkopf, and Sebe. "Hyperbolic Busemann Neural Networks." 2026, Sec. 3.
    """

    def __init__(
        self,
        manifold_module: Hyperboloid,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        validate_hyperboloid_manifold(manifold_module, required_methods=("expmap_0", "busemann"))
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space

        # Direction dim is spatial (ambient - 1) for the Lorentz model.
        in_spatial = in_dim - 1
        self.kernel, self.log_scale, self.bias = _init_weight_norm_params(
            rngs, out_dim, in_spatial, std=in_spatial**-0.5, param_dtype=param_dtype
        )

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """Forward pass returning Euclidean Busemann logits, shape (batch, out_dim)."""
        return _busemann_score(self.manifold, x, self.kernel[...], self.log_scale[...], self.bias[...], c, self.input_space)


class HypRegressionPoincareBusemann(nnx.Module):
    """Busemann MLR classification head (Poincaré ball model).

    Computes per-class logits ``u_k(x) = -alpha_k·B^{v_k}(x) + b_k`` from the Poincaré Busemann
    function (point-to-horosphere distance), returning Euclidean logits — not manifold points.

    Parameters
    ----------
    manifold_module : Poincare
        Class-based Poincaré manifold instance.
    in_dim : int
        Input spatial dimension (the ball has no time component).
    out_dim : int
        Number of classes.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    input_space : str
        ``"manifold"`` (default) or ``"tangent"`` (lift via ``expmap_0`` first). Static for JIT.
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32). Compute precision
        of manifold operations is set by ``manifold.dtype``.

    References
    ----------
    Chen, Schölkopf, and Sebe. "Hyperbolic Busemann Neural Networks." 2026, Sec. 3.
    """

    def __init__(
        self,
        manifold_module: Poincare,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        validate_poincare_manifold(manifold_module, required_methods=("expmap_0", "busemann"))
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space

        # Direction dim equals the spatial dim (no time component on the ball).
        in_spatial = in_dim
        self.kernel, self.log_scale, self.bias = _init_weight_norm_params(
            rngs, out_dim, in_spatial, std=in_spatial**-0.5, param_dtype=param_dtype
        )

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """Forward pass returning Euclidean Busemann logits, shape (batch, out_dim)."""
        return _busemann_score(self.manifold, x, self.kernel[...], self.log_scale[...], self.bias[...], c, self.input_space)
