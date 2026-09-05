"""Poincaré ball linear layers for JAX/Flax NNX.

Dimension key:
  B: batch size       I: input dimension
  O: output dimension
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds import Manifold
from hyperbolix.manifolds.poincare import Poincare

from ..optim import ManifoldParam
from ..utils.math_utils import sinh
from ._helpers import validate_poincare_manifold


class HypLinearPoincare(nnx.Module):
    """
    Hyperbolic Neural Networks fully connected layer (Poincaré ball model).

    Superseded by :class:`HypLinearPoincarePP` (Shimizu et al. 2020); kept for
    reproduction of Ganea et al. 2018. The Möbius matrix-vector product used here
    routes through ``logmap_0``/``expmap_0``, which is both slower and less stable
    near the ball boundary than the HNN++ formulation.

    Computation steps:
        0) Project the input tensor to the tangent space (optional)
        1) Perform matrix vector multiplication in the tangent space at the origin.
        2) Map the result to the manifold.
        3) Add the manifold bias to the result.

    Parameters
    ----------
    manifold_module : object
        Class-based Poincare manifold instance
    in_dim : int
        Dimension of the input space
    out_dim : int
        Dimension of the output space
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    curvature : float or callable
        Curvature tag for the manifold-valued ``bias`` parameter (default: 1.0).
        The Riemannian optimizer uses this value for the bias update
        (egrad2rgrad, expmap, parallel transport), so it MUST match the ``c``
        passed at ``__call__`` time — a mismatch silently applies the wrong
        Riemannian correction. For learnable curvature, pass a callable
        returning the current value (e.g. ``lambda: model.curvature()``).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.
    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space)
        are treated as static and will be baked into the compiled function. Changing these values after
        JIT compilation will trigger automatic recompilation.

    References
    ----------
    Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
        Advances in neural information processing systems 31 (2018).
    """

    def __init__(
        self,
        manifold_module: Manifold,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        curvature: float | Callable[[], Any] = 1.0,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        validate_poincare_manifold(
            manifold_module,
            required_methods=("proj", "addition", "expmap_0", "logmap_0"),
        )
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space

        # Trainable parameters
        # Tangent space weight (Euclidean) - initialized with std = 1/sqrt(fan_in)
        # to prevent outputs from saturating at the Poincaré ball boundary
        std = 1.0 / jnp.sqrt(in_dim)
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim), dtype=param_dtype) * std)
        # Manifold bias (initialized to small random values to avoid gradient issues at origin)
        self.bias = ManifoldParam(
            jax.random.normal(rngs.params(), (out_dim,), dtype=param_dtype) * 0.01,
            manifold=self.manifold,
            curvature=curvature,
        )

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the hyperbolic linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor where the hyperbolic_axis is last
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Poincaré ball manifold
        """
        # Project bias to manifold
        bias_O = self.manifold.proj(self.bias[...], c)

        # Map to tangent space if needed (static branch - JIT friendly)
        if self.input_space == "manifold":
            x_BI = jax.vmap(self.manifold.logmap_0, in_axes=(0, None), out_axes=0)(x, c)
        else:
            x_BI = x

        # Matrix-vector multiplication in tangent space at origin. Cast the
        # kernel to the input dtype so float64 weights (from global
        # jax_enable_x64) don't promote a float32 computation to float64.
        kernel_OI = self.kernel[...].astype(x_BI.dtype)
        # Layer weight GEMM: no `precision` kwarg, so it follows JAX's own
        # `jax_default_matmul_precision` (TF32 on Ampere/Hopper). See hyperbolix.utils.precision.
        x_BO = jnp.einsum("bi,oi->bo", x_BI, kernel_OI)  # (B, I) @ (I, O) -> (B, O)

        # Map back to manifold
        x_BO = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x_BO, c)

        # Manifold bias addition (Möbius addition for Poincaré)
        res_BO = jax.vmap(self.manifold.addition, in_axes=(0, None, None), out_axes=0)(x_BO, bias_O, c)
        return res_BO


def _poincare_pp_forward(
    x_BI: Float[Array, "batch in_dim"],
    kernel_OI: Array,
    bias_O1: Array,
    manifold: Poincare,
    c: float,
    input_space: str,
    clamping_factor: float,
    smoothing_factor: float,
) -> Float[Array, "batch out_dim"]:
    """Pure-function HNN++ forward pass.

    Used by both HypLinearPoincarePP and HypConv2DPoincare.
    """
    # Map to manifold if needed (static branch - JIT friendly)
    if input_space == "tangent":
        x_BI = jax.vmap(manifold.expmap_0, in_axes=(0, None), out_axes=0)(x_BI, c)

    # Compute multinomial linear regression
    v = manifold.compute_mlr_pp(x_BI, kernel_OI, bias_O1, c, clamping_factor, smoothing_factor)

    # Generalized linear transformation
    sqrt_c = jnp.sqrt(c)
    w_BO = sinh(sqrt_c * v) / sqrt_c
    w2_B1 = jnp.sum(w_BO**2, axis=-1, keepdims=True)
    denom_B1 = 1 + jnp.sqrt(1 + c * w2_B1)
    res_BO = w_BO / denom_B1  # (B, 1) broadcasts over (B, O)

    # Project results to the manifold
    res_BO = jax.vmap(manifold.proj, in_axes=(0, None), out_axes=0)(res_BO, c)

    return res_BO


class HypLinearPoincarePP(nnx.Module):
    """
    Hyperbolic Neural Networks ++ fully connected layer (Poincaré ball model).

    Computation steps:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)
        2) Calculate the generalized linear transformation from the regression score(s)

    Parameters
    ----------
    manifold_module : object
        Class-based Poincare manifold instance
    in_dim : int
        Dimension of the input space
    out_dim : int
        Dimension of the output space
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    clamping_factor : float
        Clamping factor for the multinomial linear regression output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the multinomial linear regression output (default: 50.0)
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.
    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space,
        clamping_factor, smoothing_factor) are treated as static and will be baked into the compiled function.

    References
    ----------
    Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
        arXiv preprint arXiv:2006.08210 (2020).
    """

    def __init__(
        self,
        manifold_module: Poincare,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        validate_poincare_manifold(
            manifold_module,
            required_methods=("proj", "addition", "expmap_0", "logmap_0", "compute_mlr_pp"),
        )
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

        # Trainable parameters
        # Tangent space weight - initialized with std = 1/sqrt(fan_in)
        # to prevent outputs from saturating at the Poincaré ball boundary
        std = 1.0 / jnp.sqrt(in_dim)
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim), dtype=param_dtype) * std)
        # Scalar bias
        self.bias = nnx.Param(jnp.zeros((out_dim, 1), dtype=param_dtype))

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the HNN++ hyperbolic linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor where the hyperbolic_axis is last
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Poincaré ball manifold
        """
        return _poincare_pp_forward(
            x,
            self.kernel[...],
            self.bias[...],
            self.manifold,
            c,
            self.input_space,
            self.clamping_factor,
            self.smoothing_factor,
        )
