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
from hyperbolix.manifolds.hyperboloid import _asinhc
from hyperbolix.manifolds.poincare import Poincare

from ..optim import ManifoldParam
from ..utils.math_utils import _pow2_divisor, floor_at, safe_hypot_norm, safe_sqrt, sinh
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


def _poincare_sinh_lift(s_BO: Float[Array, "batch out_dim"], c: float) -> Float[Array, "batch out_dim"]:
    """Lift sinh values ``s`` to the ball: ``y = w / (1 + sqrt(1 + c‖w‖²))`` with ``w = s/√c``.

    Shared by :func:`_poincare_pp_forward` (``s = sinh(√c·v)``) and the Poincaré Busemann output
    map :func:`~hyperbolix.nn_layers.busemann_core.busemann_fc_poincare_output`. Nothing large is
    squared, so the output stays finite and off the origin for every finite ``s``.
    """
    # Written in s = √c·w with each row divided by D, the power of two at or below max(max|s|, 1):
    #     t = s·(1/D),   y = t / (√c·(1/D + sqrt(‖t‖² + 1/D²))).
    # Squaring w directly overflowed float32 once the sinh argument passed ~44 and returned exactly
    # the origin with a zero gradient. Here |t| < 2, so nothing overflows and the denominator stays
    # small (JAX's div JVP squares it). Multiply by 1/D rather than divide by D: XLA folds (s/D)/X into
    # s/(D·X), which overflows again. D carries no gradient; at s = 0 the Jacobian dy/ds is exactly
    # I/(2√c). Near the sinh clip the backward's cotangent·(1/D) can be subnormal, and XLA:CPU flushes
    # it to zero.
    sqrt_c = jnp.sqrt(c)
    one = jnp.asarray(1.0, dtype=s_BO.dtype)
    scale_B1 = jax.lax.stop_gradient(floor_at(jnp.max(jnp.abs(s_BO), axis=-1, keepdims=True), one))  # (B, 1)
    inv_divisor_B1 = one / _pow2_divisor(scale_B1)  # (B, 1), exact power of two in (0, 1]
    t_BO = s_BO * inv_divisor_B1  # exact unless subnormal (see above); max|t| in [1, 2) when D > 1
    denom_B1 = inv_divisor_B1 + safe_hypot_norm(t_BO, inv_divisor_B1[:, 0])[:, None]  # (B, 1)
    return t_BO / (sqrt_c * denom_B1)  # (B, 1) broadcasts over (B, O)


def _poincare_sinh_logmap_0(s_BO: Float[Array, "batch out_dim"], c: float) -> Float[Array, "batch out_dim"]:
    """``logmap_0(_poincare_sinh_lift(s))`` without the ball point: ``asinh(‖s‖)/(2√c‖s‖)·s``.

    With ``‖s‖ = sinh(u)`` the lift lands at ``√c‖y‖ = tanh(u/2)`` along ``ŝ``, and ``logmap_0``
    reads that radius back as ``u/(2√c)``. Going through the ball, float32's ``proj`` margin capped
    the result at ``√c‖out‖ ≈ 6.33`` (c = 1) with a zero radial gradient. Used by
    :class:`~hyperbolix.nn_layers.poincare_conv.HypConv2DPoincare`; the Jacobian at ``s = 0`` is
    exactly ``I/(2√c)``. With the tangent-input scores, that layer's float32-vs-float64 error
    (relative to the largest entry, patch radius t ≤ 8, c ∈ {0.3, 1}) is ≤ 1.9e-6 on outputs and
    ≤ 3.6e-6 on input gradients; through the ball it was 1.7e-1 … 6.9e-1 / 2.0e-1 … 8.6e-1 at t = 8.
    """
    # The power-of-two rescale of _poincare_sinh_lift, t = s·(1/D): past the sinh clip one entry can
    # be ≈ 7e37, so ‖s‖² -- and ‖s‖ itself for a wide row -- overflows for a finite s. |t| < 2, so the
    # plain sum of squares cannot, and that one reduction gives both ‖t‖ and hypot(‖t‖, 1/D). The
    # route through the ball paid five reductions over (B, O) here (three in the lift, then proj and
    # logmap_0); this pays two.
    sqrt_c = jnp.sqrt(c)
    one = jnp.asarray(1.0, dtype=s_BO.dtype)
    scale_B1 = jax.lax.stop_gradient(floor_at(jnp.max(jnp.abs(s_BO), axis=-1, keepdims=True), one))  # (B, 1)
    divisor_B1 = _pow2_divisor(scale_B1)  # (B, 1), exact power of two >= 1
    inv_divisor_B1 = one / divisor_B1
    t_BO = s_BO * inv_divisor_B1  # exact unless subnormal (see _poincare_sinh_lift)
    t_sq_B1 = jnp.sum(t_BO**2, axis=-1, keepdims=True)  # (B, 1), < 4·O
    t_norm_B1 = safe_sqrt(t_sq_B1)  # exact 0 with a zero VJP at s = 0
    # asinh(‖s‖)/‖t‖ in two branches on the per-row divisor, both exact:
    #  * D > 1: ‖t‖ >= 1 and asinh(‖s‖) = log D + log(‖t‖ + sqrt(‖t‖² + 1/D²)), a sum of non-negative
    #    terms. Where D = 1, ‖t‖ can be 0, so it is replaced by 1 there *before* the division: the
    #    untaken branch's 0/0 never reaches the VJP (double `where`).
    #  * D = 1: t = s, and asinh(‖s‖)/‖s‖ is `_asinhc`, analytic at 0 (value 1, slope 0).
    rescaled_B1 = divisor_B1 > one
    t_norm_safe_B1 = jnp.where(rescaled_B1, t_norm_B1, one)
    asinh_s_B1 = jnp.log(divisor_B1) + jnp.log(t_norm_safe_B1 + jnp.sqrt(t_sq_B1 + inv_divisor_B1**2))
    ratio_B1 = jnp.where(rescaled_B1, asinh_s_B1 / t_norm_safe_B1, _asinhc(t_norm_B1))  # asinh(‖s‖)/‖t‖
    return t_BO * (ratio_B1 / (2 * sqrt_c))  # (B, 1) broadcasts over (B, O)


def _poincare_pp_forward(
    x_BI: Float[Array, "batch in_dim"],
    kernel_OI: Array,
    bias_O1: Array,
    manifold: Poincare,
    c: float,
    input_space: str,
    *,
    tangent_output: bool = False,
) -> Float[Array, "batch out_dim"]:
    """Pure-function HNN++ forward pass.

    Used by both HypLinearPoincarePP and HypConv2DPoincare. A tangent input is scored as
    ``expmap_0`` would place it, without forming the ball point
    (:meth:`~hyperbolix.manifolds.poincare.Poincare._compute_mlr_pp_tangent`). The final lift
    ``y = w / (1 + sqrt(1 + c‖w‖²))`` with ``w = sinh(√c·v)/√c`` is evaluated without squaring
    anything large, so the output stays finite and off the origin for every finite score ``v``
    (see :func:`_poincare_sinh_lift`). ``tangent_output=True`` returns ``logmap_0`` of that lift
    instead, again without the ball point (:func:`_poincare_sinh_logmap_0`).
    """
    # Static branch - JIT friendly
    if input_space == "tangent":
        v_BO = manifold._compute_mlr_pp_tangent(x_BI, kernel_OI, bias_O1, c)
    else:
        v_BO = manifold.compute_mlr_pp(x_BI, kernel_OI, bias_O1, c)

    # Generalized linear transformation y = w / (1 + sqrt(1 + c‖w‖²)), w = sinh(√c·v)/√c.
    s_BO = sinh(jnp.sqrt(c) * v_BO)  # (B, O)
    if tangent_output:
        return _poincare_sinh_logmap_0(s_BO, c)
    res_BO = _poincare_sinh_lift(s_BO, c)

    # Project results to the manifold
    res_BO = jax.vmap(manifold.proj, in_axes=(0, None), out_axes=0)(res_BO, c)

    return res_BO


class HypLinearPoincarePP(nnx.Module):
    """
    Hyperbolic Neural Networks ++ fully connected layer (Poincaré ball model).

    Computation steps:
        1) Compute the multinomial linear regression score(s) — for a tangent input, of the point
           ``expmap_0`` would place on the ball, evaluated from the tangent vector directly
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
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.
    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. The configuration parameter ``input_space``
        is treated as static and will be baked into the compiled function.

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
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        validate_poincare_manifold(
            manifold_module,
            required_methods=("proj", "compute_mlr_pp", "_compute_mlr_pp_tangent"),
        )
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space

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
        )
