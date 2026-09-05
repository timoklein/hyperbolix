"""Proper Velocity linear layers for JAX/Flax NNX.

Implements the PV fully-connected layer from Chen et al. 2026
(Thm 5.3 / Eq. 22):

    y_k = (1/√c) · sinh(√c · v_k(x))

where ``v_k(x)`` is the PV multinomial-logistic-regression score from Eq. 19
(implemented as ``ProperVelocity.compute_mlr``). An optional inner activation
``sigma`` inside the sinh mirrors the paper's Eq. 23 ablation.

Dimension key:
  B: batch size       I: input dimension       O: output dimension
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds.proper_velocity import ProperVelocity

from ..utils.math_utils import sinh
from ._helpers import validate_pv_manifold


def _pv_fc_forward(
    x_BI: Float[Array, "batch in_dim"],
    kernel_OI: Array,
    bias_O1: Array,
    manifold: ProperVelocity,
    c: float,
    input_space: str,
    inner_activation: Callable[[Array], Array] | None = None,
) -> Float[Array, "batch out_dim"]:
    """Pure-function PV FC forward pass (paper Thm 5.3 / Eq. 22).

    Used by both ``HypLinearPV`` and ``HypConv2DPV``.

    Steps:
      1) Optional lift Euclidean input via ``expmap_0`` if ``input_space == "tangent"``.
      2) Compute PV MLR scores ``v`` via ``manifold.compute_mlr``.
      3) Optionally apply ``inner_activation`` to ``v`` (paper Eq. 23).
      4) Return ``(1/√c) · sinh(√c · v)`` — a PV-manifold point.
    """
    # Map to manifold if needed (static branch — JIT friendly).
    if input_space == "tangent":
        x_BI = jax.vmap(manifold.expmap_0, in_axes=(0, None), out_axes=0)(x_BI, c)

    v_BO = manifold.compute_mlr(x_BI, kernel_OI, bias_O1, c)

    if inner_activation is not None:
        v_BO = inner_activation(v_BO)

    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=v_BO.dtype))
    return sinh(sqrt_c * v_BO) / sqrt_c


class HypLinearPV(nnx.Module):
    """
    Hyperbolic Neural Networks fully connected layer (Proper Velocity model).

    Implements Chen et al. 2026, Thm 5.3 / Eq. 22:

        y_k = (1/√c) · sinh(√c · v_k(x))

    where ``v_k(x)`` is the PV multinomial-logistic-regression signed margin
    from Eq. 19 (implemented as ``ProperVelocity.compute_mlr``). The optional
    ``inner_activation`` realizes the Eq. 23 ablation from the paper.

    Parameters
    ----------
    manifold_module : ProperVelocity
        Class-based Proper Velocity manifold instance.
    in_dim : int
        Dimension of the input space.
    out_dim : int
        Dimension of the output space.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration — changing it after initialization
        requires recompilation.
    inner_activation : Callable[[Array], Array] | None
        Optional activation applied to ``v`` before the outer sinh
        (paper Eq. 23 ablation). Default None.
    kernel_init_std : float | None
        Standard deviation for the Gaussian kernel init. If ``None`` (default),
        uses He scaling ``sqrt(2 / in_dim)`` so stacked layers preserve variance
        under ReLU activations (for small MLR arguments the PV output reduces to
        an Euclidean linear map, so standard He analysis applies). Pass a
        specific value (e.g. ``1e-2``) when using this layer as the final
        classification/regression layer directly before a softmax/MSE loss —
        that matches the paper's ``PVManifoldMLR.reset_parameters`` recipe,
        which is tuned for receiving ``O(1)``-variance features.
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.

    Notes
    -----
    JIT Compatibility:
        Configuration parameters (input_space, inner_activation) are treated
        as static and are baked into the compiled function.

    References
    ----------
    Chen et al. "Proper Velocity Neural Networks." ICLR 2026.
    """

    def __init__(
        self,
        manifold_module: ProperVelocity,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        inner_activation: Callable[[Array], Array] | None = None,
        kernel_init_std: float | None = None,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        validate_pv_manifold(
            manifold_module,
            required_methods=("expmap_0", "compute_mlr"),
        )
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.inner_activation = inner_activation

        # Kernel init: He(in_dim) by default so deep stacks preserve variance
        # under ReLU. Override via ``kernel_init_std`` when using as a final
        # classification layer — see the ``kernel_init_std`` docstring above.
        # Bias init is uniform U(-1e-3, 1e-3), matching the paper reference.
        std = (2.0 / in_dim) ** 0.5 if kernel_init_std is None else kernel_init_std
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim), dtype=param_dtype) * std)
        self.bias = nnx.Param(jax.random.uniform(rngs.params(), (out_dim, 1), dtype=param_dtype, minval=-1e-3, maxval=1e-3))

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the PV linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input — PV manifold points or tangent vectors at origin, depending on
            ``input_space``.
        c : float
            Manifold curvature (default: 1.0).

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output points on the PV manifold.
        """
        return _pv_fc_forward(
            x,
            self.kernel[...],
            self.bias[...],
            self.manifold,
            c,
            self.input_space,
            self.inner_activation,
        )
