"""Hyperboloid linear layers for JAX/Flax NNX.

This module contains linear transformation layers for the hyperboloid manifold,
including the Hyperbolic Transformation Component (HTC) module from the Hypformer paper
and the FGG linear layer from Klis et al. 2026.

For the core HTC/HRC functions, see hyperboloid_core module.

Dimension key:
  B: batch size
  I: in_spatial (in_features - 1)    O: out_spatial (out_features - 1)
  Ai: in_ambient (in_features)       Ao: out_ambient (out_features)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds import Manifold

# MIN_NORM is the library-wide gradient-safety floor for norms: sqrt(sum + MIN_NORM²)
# has a finite VJP at zero input, unlike linalg.norm, whose 0/0 NaN survives any
# post-hoc jnp.where masking. Imported (not redefined) so there is one value.
from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.utils.math_utils import MIN_NORM, capped_exp

from ._helpers import validate_hyperboloid_manifold
from .hyperboloid_core import build_spacelike_V, htc, sinh_lift_to_hyperboloid


def _fhcnn_forward(
    x_BI: Float[Array, "batch in_dim"],
    kernel_OI: Array,
    bias_1O: Array,
    manifold: Manifold,
    c: float,
    input_space: str,
    activation: Callable[[Array], Array] | None,
    normalize: bool,
    scale_val: float | Array,
    eps: float,
) -> Float[Array, "batch out_dim"]:
    """Pure-function FHCNN forward pass.

    Used by both HypLinearHyperboloidFHCNN and HypConv2DHyperboloid.
    """
    # Map to manifold if needed (static branch - JIT friendly)
    if input_space == "tangent":
        x_BI = jax.vmap(manifold.expmap_0, in_axes=(0, None), out_axes=0)(x_BI, c)

    # Apply activation if provided (static branch - JIT friendly)
    if activation is not None:
        x_BI = activation(x_BI)

    # Linear transformation: (B, in_dim) -> (B, out_dim). Cast params to the
    # input dtype so float64 weights (created when jax_enable_x64 is enabled
    # globally) don't silently promote a float32 manifold computation. The
    # scale needs the same guard: a learnable scale param is weak-typed at
    # init (adopts float32) but becomes strong float64 after the first
    # optimizer step. asarray also covers the static Python-float scale.
    kernel_OI = kernel_OI.astype(x_BI.dtype)
    bias_1O = bias_1O.astype(x_BI.dtype)
    scale_val = jnp.asarray(scale_val, dtype=x_BI.dtype)
    x_BO = jnp.einsum("bi,oi->bo", x_BI, kernel_OI) + bias_1O

    # Split into time and space: x0 is first coord, x_rem is spatial
    x0_B1 = x_BO[:, 0:1]  # (B, 1) -- time coordinate
    x_rem_BD = x_BO[:, 1:]  # (B, D) where D = out_dim - 1

    # Static branch - JIT friendly
    if normalize:
        # Safe norm: finite gradient at zero spatial input; the origin mask
        # below still fires (safe norm of a zero vector is MIN_NORM <= 1e-5).
        x_rem_norm_B1 = jnp.sqrt(jnp.sum(x_rem_BD**2, axis=-1, keepdims=True) + MIN_NORM**2)  # (B, 1)

        # Learnable sigmoid scaling. capped_exp: scale_val is unconstrained — a runaway param
        # must saturate finite, not overflow to inf and NaN the time coordinate.
        scale_B1 = capped_exp(scale_val) * jax.nn.sigmoid(x0_B1)  # (B, 1)

        res0_B1 = jnp.sqrt(scale_B1**2 + 1 / c + eps)  # (B, 1)
        res_rem_BD = scale_B1 * x_rem_BD / x_rem_norm_B1  # (B, D)

        res_BA = jnp.concatenate([res0_B1, res_rem_BD], axis=-1)  # (B, A)

        # Cast near-zero-norm vectors to origin
        origin_time_B1 = jnp.sqrt(1 / c) * jnp.ones_like(res0_B1)
        origin_BA = jnp.concatenate([origin_time_B1, jnp.zeros_like(res_rem_BD)], axis=-1)

        mask_B1 = x_rem_norm_B1 <= 1e-5
        res_BA = jnp.where(mask_B1, origin_BA, res_BA)
    else:
        # Reconstruct time from space: x0 = sqrt(||x_rem||^2 + 1/c)
        res0_B1 = jnp.sqrt(jnp.sum(x_rem_BD**2, axis=-1, keepdims=True) + 1 / c)  # (B, 1)
        res_BA = jnp.concatenate([res0_B1, x_rem_BD], axis=-1)  # (B, A)

    return res_BA


def _fhnn_forward(
    x_BI: Float[Array, "batch in_dim"],
    kernel_OI: Array,
    bias_1O: Array,
    manifold: Manifold,
    c: float,
    input_space: str,
    activation: Callable[[Array], Array] | None,
    dropout_fn: Callable[[Array], Array] | None,
    scale_val: Array,
    eps: float,
) -> Float[Array, "batch out_dim"]:
    """Pure-function FHNN forward pass (Chen et al. 2021).

    Time-primary parameterization: the time coordinate is computed via scaled
    sigmoid with an additive floor at 1/sqrt(c), and spatial coordinates are
    rescaled so the output lies on the hyperboloid.
    """
    # Map to manifold if needed (static branch - JIT friendly)
    if input_space == "tangent":
        x_BI = jax.vmap(manifold.expmap_0, in_axes=(0, None), out_axes=0)(x_BI, c)

    # Apply activation if provided (static branch - JIT friendly)
    if activation is not None:
        x_BI = activation(x_BI)

    # Apply dropout if provided (static branch - JIT friendly)
    if dropout_fn is not None:
        x_BI = dropout_fn(x_BI)

    # Linear transformation: (B, in_dim) -> (B, out_dim). Cast params to the
    # input dtype so float64 weights (created when jax_enable_x64 is enabled
    # globally) don't silently promote a float32 manifold computation. The
    # scale param needs the same guard: it is weak-typed at init (adopts
    # float32) but becomes strong float64 after the first optimizer step.
    kernel_OI = kernel_OI.astype(x_BI.dtype)
    bias_1O = bias_1O.astype(x_BI.dtype)
    scale_val = jnp.asarray(scale_val, dtype=x_BI.dtype)
    z_BO = jnp.einsum("bi,oi->bo", x_BI, kernel_OI) + bias_1O  # (B, O)

    # Split into time logit and spatial components
    z0_B1 = z_BO[:, 0:1]  # (B, 1)
    z_rem_BD = z_BO[:, 1:]  # (B, D) where D = out_dim - 1

    # Time coordinate via scaled sigmoid with floor at 1/sqrt(c)
    # y0 > 1/sqrt(c) guaranteed by construction (additive floor, not max)
    # capped_exp: scale_val is unconstrained — see _fhcnn_forward above.
    y0_B1 = capped_exp(scale_val) * jax.nn.sigmoid(z0_B1) + 1.0 / jnp.sqrt(c) + eps  # (B, 1)

    # Target spatial norm from hyperboloid constraint: ||y_s||^2 = y0^2 - 1/c
    # Always real since y0 > 1/sqrt(c) + eps => y0^2 > 1/c
    target_norm_B1 = jnp.sqrt(y0_B1**2 - 1.0 / c)  # (B, 1)

    # Rescale spatial to satisfy hyperboloid constraint.
    # Safe norm: finite gradient at zero spatial input (linalg.norm's VJP at 0
    # is NaN and survives both the maximum() and the jnp.where below).
    z_rem_norm_B1 = jnp.sqrt(jnp.sum(z_rem_BD**2, axis=-1, keepdims=True) + MIN_NORM**2)  # (B, 1)
    z_rem_norm_safe_B1 = jnp.maximum(z_rem_norm_B1, eps)  # avoid division by zero
    y_rem_BD = target_norm_B1 / z_rem_norm_safe_B1 * z_rem_BD  # (B, D)

    # Concatenate time and spatial
    res_BA = jnp.concatenate([y0_B1, y_rem_BD], axis=-1)  # (B, out_dim)

    # Origin fallback: near-zero spatial norm -> hyperboloid origin
    origin_time_B1 = jnp.full_like(y0_B1, 1.0 / jnp.sqrt(c))
    origin_BA = jnp.concatenate([origin_time_B1, jnp.zeros_like(y_rem_BD)], axis=-1)
    mask_B1 = z_rem_norm_B1 <= eps
    res_BA = jnp.where(mask_B1, origin_BA, res_BA)

    return res_BA


def _hyperboloid_plfc_forward(
    x_BAi: Float[Array, "batch in_dim"],
    kernel_OI: Array,
    bias_O1: Array,
    gyro_bias_O: Array | None,
    manifold: Hyperboloid,
    c: float,
    input_space: str,
    clamping_factor: float,
    smoothing_factor: float,
    v_max: float,
) -> Float[Array, "batch out_dim"]:
    """Pure-function PLFC forward pass for the hyperboloid model.

    Used by HypLinearHyperboloidPLFC and HypConv2DHyperboloidILNN. Computes MLR
    scores, guards them, maps back to the hyperboloid via element-wise sinh
    diffeomorphism, and optionally applies an intrinsic gyro-bias.
    """
    # Map to manifold if needed (static branch - JIT friendly)
    if input_space == "tangent":
        x_BAi = jax.vmap(manifold.expmap_0, in_axes=(0, None), out_axes=0)(x_BAi, c)

    # Compute multinomial logistic regression scores: (B, O) where O = out_dim-1
    v_BO = manifold.compute_mlr(x_BAi, kernel_OI, bias_O1, c, clamping_factor, smoothing_factor)

    # Output-side guard + sinh diffeomorphism + time reconstruction (shared with the Busemann FC
    # layer). compute_mlr only bounds its asinh argument, leaving v ∝ ‖kernel_row‖ unbounded; the
    # sinh exponentiates it, so sinh_lift_to_hyperboloid hard-clips sqrt(c)*v to ±v_max before the
    # element-wise sinh (NOT expmap_0, which applies sinh to the norm). The constructor's
    # _assert_v_max_safe keeps the bare jnp.sinh inside the helper safe from float32 overflow.
    res_BAo = sinh_lift_to_hyperboloid(v_BO, c, v_max)  # (B, Ao)

    # Intrinsic gyro-bias: y ← y ⊕ exp_0([0, b]) (Shi et al. 2026, Sec. 4.1)
    if gyro_bias_O is not None:
        bias_tangent_Ao = manifold.embed_spatial_0(gyro_bias_O)
        bias_point_Ao = manifold.expmap_0(bias_tangent_Ao, c)
        res_BAo = jax.vmap(manifold.addition, in_axes=(0, None, None))(res_BAo, bias_point_Ao, c)

    return res_BAo


def _assert_v_max_safe(v_max: float) -> None:
    """Guard the bare-``jnp.sinh`` PLFC output path against ``v_max``-driven float32 overflow.

    The output spatial coordinate is ``sinh(v_max)/sqrt(c)`` and the time component reconstructs from
    ``sum(.**2)``; float32 is the binding case because layer inputs may be float32 regardless of
    ``param_dtype``. Require ``sinh(v_max) < sqrt(finfo(float32).max)`` (≈1.84e19, i.e. ``v_max`` ≲ 45)
    so the squared spatial norm cannot overflow. The Shi et al. 2026 default ``v_max=10`` is far
    inside this bound; a user who needs a larger ``v_max`` should restore the wrapped/smooth sinh
    guard instead.
    """
    f32_safe_sinh = float(jnp.finfo(jnp.float32).max) ** 0.5  # ~1.84e19
    if float(jnp.sinh(v_max)) >= f32_safe_sinh:
        raise ValueError(
            f"v_max={v_max} is too large for the PLFC sinh diffeomorphism: sinh(v_max) would overflow "
            f"the float32 squared spatial norm in the time reconstruction. Use v_max < ~45 "
            f"(default 10), or restore the smooth/wrapped sinh guard for larger values."
        )


def _fgg_weight_init(
    key: Array,
    in_spatial: int,
    out_spatial: int,
    in_ambient: int,
    out_ambient: int,
    reset_params: str,
    param_dtype: DTypeLike,
    gain: float = 1.0,
) -> Float[Array, "in_spatial out_spatial"]:
    """Euclidean weight ``U`` of shape ``(I, O)`` for FGG layers (Klis et al. 2026).

    Shared by ``FGGLinear`` and ``FGGConv2D``. The fan-in std variants compute their
    std from the *ambient* dimensions (``in_ambient``/``out_ambient`` = spatial + 1);
    ``"fan_out"`` uses the *output spatial* dim. The result is pinned to ``param_dtype``
    so a global ``jax_enable_x64`` does not silently promote the bare ``jnp.eye`` /
    ``jax.random.normal`` to float64.

    Parameters
    ----------
    key : Array
        PRNG key for the random init branches.
    in_spatial, out_spatial : int
        Spatial weight dims ``I = in_ambient - 1``, ``O = out_ambient - 1``.
    in_ambient, out_ambient : int
        Ambient dims (including the time coordinate) used for the fan-in std formulas.
    reset_params : str
        One of ``"eye"``, ``"xavier"``, ``"kaiming"``, ``"lorentz_kaiming"``,
        ``"fan_out"``, ``"mlr"``. ``"fan_out"`` draws a Gaussian with
        std ``sqrt(1 / out_spatial)`` (fan-out), which combined with ``gain`` gives an
        effective column std of ``gain / sqrt(out_spatial)`` — norm-preserving output
        (``||z|| ~= gain * ||x_spatial||``) suitable for unnormalized FGG stacks.
    param_dtype : DTypeLike
        Storage dtype the returned weight is pinned to.
    gain : float, optional
        Multiplier applied to the random (non-``"eye"``) branches (default: 1.0). For
        ``"fan_out"`` this scales the norm-preservation factor. No-op for ``"eye"``
        (its ``0.5 * eye`` identity scale must not be rescaled).

    Returns
    -------
    Array, shape (I, O)
        Euclidean weight matrix in ``param_dtype``.
    """
    if reset_params == "eye":
        U_IO = 0.5 * jnp.eye(in_spatial, out_spatial)
    elif reset_params == "xavier":
        U_IO = jax.random.normal(key, (in_spatial, out_spatial)) * jnp.sqrt(1.0 / (in_ambient + out_ambient))
    elif reset_params == "kaiming":
        U_IO = jax.random.normal(key, (in_spatial, out_spatial)) * jnp.sqrt(2.0 / in_ambient)
    elif reset_params == "lorentz_kaiming":
        U_IO = jax.random.normal(key, (in_spatial, out_spatial)) * jnp.sqrt(1.0 / in_ambient)
    elif reset_params == "fan_out":
        # Fan-out std sqrt(1/out_spatial): output spatial norm tracks the input
        # (norm-preserving) instead of growing as sqrt(out_channels) like fan-in.
        # max(.,1) guards the degenerate out_spatial == 0 case (out_features == 1).
        U_IO = jax.random.normal(key, (in_spatial, out_spatial)) * jnp.sqrt(1.0 / max(out_spatial, 1))
    elif reset_params == "mlr":
        U_IO = jax.random.normal(key, (in_spatial, out_spatial)) * jnp.sqrt(5.0 / in_ambient)
    else:
        raise ValueError(
            f"reset_params must be 'eye', 'xavier', 'kaiming', 'lorentz_kaiming', 'fan_out', or 'mlr', got '{reset_params}'"
        )

    # Apply gain to the random branches only -- the 0.5*eye identity init is not rescaled.
    if reset_params != "eye":
        U_IO = U_IO * gain

    # Pin storage dtype (under global x64 the bare jnp.eye / jax.random.normal would be float64).
    return U_IO.astype(param_dtype)


def _get_effective_kernel(
    kernel: Array | None,
    kernel_dir: Array | None,
    kernel_scale: Array | None,
    use_weight_norm: bool,
    eps: float,
) -> Array:
    """Compute effective weight matrix, handling weight normalization."""
    if use_weight_norm:
        assert kernel_scale is not None and kernel_dir is not None
        g_pos_O = jax.nn.softplus(kernel_scale)  # (O,) force positive magnitudes
        v_norm_O = jnp.sqrt(jnp.sum(kernel_dir**2, axis=0) + eps)  # (O,)
        return g_pos_O[None, :] * kernel_dir / v_norm_O[None, :]  # (I, O)
    assert kernel is not None
    return kernel


def _fgg_linear_forward(
    x_BAi: Float[Array, "batch in_features"],
    U_IO: Array,
    bias_O: Array,
    c: float,
    activation: Callable[[jax.Array], jax.Array] | None,
    eps: float,
) -> Float[Array, "batch out_features"]:
    """Pure-function FGG forward: build_spacelike_V -> matmul -> activation -> time reconstruct."""
    # Build V_mink from (kernel, bias) -- Minkowski metric absorbed
    V_AiO = build_spacelike_V(U_IO, bias_O, c, eps)  # (Ai, O)
    # Cast V to match input dtype (avoids float32/float64 scatter warnings)
    V_AiO = V_AiO.astype(x_BAi.dtype)

    # Minkowski inner products via matmul (metric in V)
    z_BO = x_BAi @ V_AiO  # (B, O)

    # Apply Euclidean activation (Lorentzian wrapping implicit via cancellation)
    if activation is not None:
        z_BO = activation(z_BO)

    # Reconstruct hyperboloid point: spatial = z, time from constraint
    y_0_B1 = jnp.sqrt(jnp.sum(z_BO**2, axis=-1, keepdims=True) + 1.0 / c)  # (B, 1)

    return jnp.concatenate([y_0_B1, z_BO], axis=-1)  # (B, Ao)


class HypLinearHyperboloidFHCNN(nnx.Module):
    """
    Fully Hyperbolic Convolutional Neural Networks fully connected layer (Hyperboloid model).

    Computation steps:
        0) Project the input tensor to the manifold (optional)
        1) Apply activation (optional)
        2) a) If normalize is True, compute the time and space coordinates of the output by applying a scaled sigmoid
              of the weight and biases transformed coordinates of the input or the result of the previous step.
           b) If normalize is False, compute the weight and biases transformed space coordinates of the input or the
              result of the previous step and set the time coordinate such that the result lies on the manifold.

    Parameters
    ----------
    manifold_module : object
        Class-based Hyperboloid manifold instance
    in_dim : int
        Dimension of the input space
    out_dim : int
        Dimension of the output space
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    init_scale : float
        Initial value for the sigmoid scale parameter (default: 2.3)
    learnable_scale : bool
        Whether the scale parameter should be learnable (default: False)
    eps : float
        Small value to ensure that the time coordinate is bigger than 1/sqrt(c) (default: 1e-5)
    activation : callable or None
        Activation function to apply before the linear transformation (default: None).
        Note: This is a static configuration - changing it after initialization requires recompilation.
    normalize : bool
        Whether to normalize the space coordinates before rescaling (default: False).
        Note: This is a static configuration - changing it after initialization requires recompilation.
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.
    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space, activation,
        normalize) are treated as static and will be baked into the compiled function.

    Relationship to HTC/HRC:
        When ``normalize=False`` and ``c_in = c_out``, this layer uses the same time reconstruction
        pattern as ``htc``: ``time = sqrt(||space||^2 + 1/c)``. The key difference is that FHCNN
        applies a linear transform to the full input and discards the computed time, while ``htc``
        uses the linear output directly as spatial components. When ``normalize=True``, FHCNN uses
        a learned sigmoid scaling which differs from both htc and hrc.

    See Also
    --------
    htc : Hyperbolic Transformation Component with curvature change support.
        Similar time reconstruction pattern when normalize=False.
    HTCLinear : Module wrapper for htc with learnable linear transformation.

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic convolutional neural networks for computer vision."
        arXiv preprint arXiv:2303.15919 (2023).
    """

    def __init__(
        self,
        manifold_module: Manifold,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        init_scale: float = 2.3,
        learnable_scale: bool = False,
        eps: float = 1e-5,
        activation: Callable[[Array], Array] | None = None,
        normalize: bool = False,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        validate_hyperboloid_manifold(manifold_module, required_methods=("expmap_0",))
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.eps = eps
        self.activation = activation
        self.normalize = normalize

        # Trainable parameters
        bound = 0.02
        weight_init = jax.random.uniform(rngs.params(), (out_dim, in_dim), dtype=param_dtype, minval=-bound, maxval=bound)
        self.kernel = nnx.Param(weight_init)
        self.bias = nnx.Param(jnp.zeros((1, out_dim), dtype=param_dtype))

        # Scale parameter for sigmoid
        if learnable_scale:
            self.scale = nnx.Param(jnp.array(init_scale, dtype=param_dtype))
        else:
            # For non-learnable scale, store as regular Python float (static)
            self.scale = init_scale

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the FHCNN hyperbolic linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor where the hyperbolic_axis is last. x.shape[-1] must equal self.in_dim.
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Hyperboloid manifold
        """
        scale_val = self.scale[...] if isinstance(self.scale, nnx.Param) else self.scale
        return _fhcnn_forward(
            x,
            self.kernel[...],
            self.bias[...],
            self.manifold,
            c,
            self.input_space,
            self.activation,
            self.normalize,
            scale_val,
            self.eps,
        )


class HypLinearHyperboloidFHNN(nnx.Module):
    """Fully Hyperbolic Neural Networks linear layer (Chen et al. 2021).

    Time-primary parameterization: the time coordinate is computed via scaled
    sigmoid with an additive floor at 1/sqrt(c), and spatial coordinates are
    rescaled so the output lies on the hyperboloid.

    Computation steps:
        0) Project the input tensor to the manifold (optional)
        1) Apply activation (optional)
        2) Apply dropout (optional)
        3) Linear transform: z = W @ x + b
        4) Time: y0 = exp(scale) * sigmoid(z0) + 1/sqrt(c) + eps
        5) Spatial: y_rem = sqrt(y0^2 - 1/c) / ||z_rem|| * z_rem
        6) Output: [y0, y_rem] on the hyperboloid

    Parameters
    ----------
    manifold_module : object
        Class-based Hyperboloid manifold instance
    in_dim : int
        Ambient input dimension (d+1, including time)
    out_dim : int
        Ambient output dimension (d+1, including time)
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    init_scale : float
        Initial value for the learnable sigmoid scale (default: 2.3)
    eps : float
        Numerical stability epsilon (default: 1e-5)
    activation : callable or None
        Activation function to apply before the linear transformation (default: None).
        Note: This is a static configuration - changing it after initialization requires recompilation.
    dropout_rate : float or None
        Dropout rate applied before the linear transformation (default: None).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space,
        activation) are treated as static and will be baked into the compiled function.

    Weight Initialization:
        Weights are initialized as tangent vectors at the hyperboloid origin: U(-0.02, 0.02)
        with the time column (column 0) zeroed. This matches the Chen et al. 2021 init.

    Relationship to FHCNN:
        Both FHNN and FHCNN ensure outputs lie on the hyperboloid, but differ in which
        coordinate is primary. FHNN controls the time coordinate via sigmoid with an additive
        floor at 1/sqrt(c), then derives the spatial norm from the hyperboloid constraint.
        FHCNN (normalize=True) controls the spatial norm via sigmoid, then derives time.

    See Also
    --------
    HypLinearHyperboloidFHCNN : FHCNN layer with spatial-primary parameterization.
    HypLinearHyperboloidPLFC : PLFC layer using MLR + sinh diffeomorphism.

    References
    ----------
    Weize Chen, et al. "Fully hyperbolic neural networks."
        arXiv preprint arXiv:2105.14686 (2021).
    """

    def __init__(
        self,
        manifold_module: Manifold,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        init_scale: float = 2.3,
        eps: float = 1e-5,
        activation: Callable[[Array], Array] | None = None,
        dropout_rate: float | None = None,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        validate_hyperboloid_manifold(manifold_module, required_methods=("expmap_0",))
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.eps = eps
        self.activation = activation

        # FHNN weight init: U(-0.02, 0.02) with time column zeroed (tangent vectors at origin)
        bound = 0.02
        weight_init = jax.random.uniform(rngs.params(), (out_dim, in_dim), dtype=param_dtype, minval=-bound, maxval=bound)
        weight_init = weight_init.at[:, 0].set(0.0)
        self.kernel = nnx.Param(weight_init)
        self.bias = nnx.Param(jnp.zeros((1, out_dim), dtype=param_dtype))

        # Learnable scale for the sigmoid (always learnable in FHNN)
        self.scale = nnx.Param(jnp.array(init_scale, dtype=param_dtype))

        # Optional dropout
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
        deterministic: bool = True,
    ) -> Float[Array, "batch out_dim"]:
        """Forward pass through the FHNN hyperbolic linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor. x.shape[-1] must equal self.in_dim.
        c : float
            Manifold curvature (default: 1.0)
        deterministic : bool
            If True, dropout is disabled (default: True).

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Hyperboloid manifold
        """
        # Build dropout closure for the pure function
        dropout_module = self.dropout
        if dropout_module is not None:
            dropout_fn = lambda z: dropout_module(z, deterministic=deterministic)  # noqa: E731
        else:
            dropout_fn = None

        return _fhnn_forward(
            x,
            self.kernel[...],
            self.bias[...],
            self.manifold,
            c,
            self.input_space,
            self.activation,
            dropout_fn,
            self.scale[...],
            self.eps,
        )


class HypLinearHyperboloidPLFC(nnx.Module):
    """
    Point-to-hyperplane Lorentz fully connected (PLFC) layer (Hyperboloid model).

    Each output coordinate is the signed Lorentz distance from the input point to a
    learned Lorentz hyperplane (the Lorentz MLR score). The output point is recovered
    in closed form via the element-wise sinh diffeomorphism and the hyperboloid time
    constraint, so the signed distance from the output to the k-th coordinate
    hyperplane equals the k-th score (margin preservation, Shi et al. 2026, Thm. 1-2).
    This is the Lorentz analog of the HNN++ point-to-hyperplane FC formulation
    (Shimizu et al. 2020). Formerly named ``HypLinearHyperboloidPP``.

    Computation steps:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s) via ``compute_mlr``
        2) Hard-clamp the scaled scores ``sqrt(c)*v`` to ``±v_max`` (output-side guard)
        3) Apply element-wise sinh diffeomorphism to obtain spatial coordinates
        4) Reconstruct time coordinate from the hyperboloid constraint
        5) Optionally add an intrinsic gyro-bias ``y ← y ⊕ exp_0([0, b])``

    Parameters
    ----------
    manifold_module : object
        Class-based Hyperboloid manifold instance
    in_dim : int
        Full input dimension (ambient, d+1)
    out_dim : int
        Full output dimension (ambient, d+1)
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    clamping_factor : float
        Clamping factor for the multinomial linear regression output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the multinomial linear regression output (default: 50.0)
    v_max : float
        Output-side guard: the sinh argument ``sqrt(c)*v`` is hard-clipped to
        ``±v_max``, bounding the output spatial norm by ``sinh(v_max)/sqrt(c)``
        (default: 10.0, matching the Shi et al. 2026 reference implementation).
    use_gyro_bias : bool
        If True, add a learnable intrinsic bias via Lorentz gyroaddition,
        ``y ← y ⊕ exp_0([0, b])`` with ``b`` initialized to zero — the gyrogroup
        identity, so the bias is a no-op at initialization (default: False).
    kernel_init_std : float
        Standard deviation of the normal kernel init (default: 0.02, the Shi et al.
        2026 PLFC reference value). Use 1.0 to recover the previous HNN++-style
        init (Shimizu et al. 2020); note that large kernels push the pre-guard
        scores toward the ``v_max`` saturation regime.
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space,
        clamping_factor, smoothing_factor, v_max) are treated as static and will be baked
        into the compiled function.

    References
    ----------
    Xianglong Shi, Ziheng Chen, Yunhan Jiang, and Nicu Sebe. "Intrinsic Lorentz Neural Network."
        ICLR 2026, Sec. 4.1 (PLFC layer, Theorem 1).
    Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
        arXiv preprint arXiv:2006.08210 (2020).
    """

    def __init__(
        self,
        manifold_module: Hyperboloid,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        v_max: float = 10.0,
        use_gyro_bias: bool = False,
        kernel_init_std: float = 0.02,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        required_methods = ("expmap_0", "compute_mlr")
        if use_gyro_bias:
            required_methods = ("expmap_0", "compute_mlr", "embed_spatial_0", "addition")
        validate_hyperboloid_manifold(manifold_module, required_methods=required_methods)
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        _assert_v_max_safe(v_max)
        self.v_max = v_max

        # Trainable parameters — small normal init (Shi et al. 2026 PLFC reference)
        in_spatial = in_dim - 1
        out_spatial = out_dim - 1
        kernel_init = kernel_init_std * jax.random.normal(rngs.params(), (out_spatial, in_spatial), dtype=param_dtype)
        self.kernel = nnx.Param(kernel_init)
        self.bias = nnx.Param(jnp.zeros((out_spatial, 1), dtype=param_dtype))

        # Gyro-bias: spatial tangent vector at the origin, zero-initialized
        if use_gyro_bias:
            self.gyro_bias = nnx.Param(jnp.zeros((out_spatial,), dtype=param_dtype))
        else:
            self.gyro_bias = None

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the PLFC hyperboloid linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor. x.shape[-1] must equal self.in_dim.
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Hyperboloid manifold
        """
        return _hyperboloid_plfc_forward(
            x,
            self.kernel[...],
            self.bias[...],
            self.gyro_bias[...] if self.gyro_bias is not None else None,
            self.manifold,
            c,
            self.input_space,
            self.clamping_factor,
            self.smoothing_factor,
            self.v_max,
        )


class HTCLinear(nnx.Module):
    """Hyperbolic Transformation Component with learnable linear transformation.

    This module wraps a Euclidean linear layer with the HTC operation, enabling
    learnable transformations between hyperboloid manifolds with different curvatures.

    Parameters
    ----------
    in_features : int
        Input feature dimension (full hyperboloid dimension, including time component).
    out_features : int
        Output spatial dimension (time component is reconstructed automatically).
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    use_bias : bool, optional
        Whether to include a bias term (default: True).
    init_bound : float or None, optional
        Bound for uniform weight initialization: weights are drawn from
        Uniform(-init_bound, init_bound). The default ``None`` resolves to the
        fan-in-aware, norm-preserving bound ``sqrt(3 / in_features)`` (per-layer
        input-Jacobian gain ~= 1; see Notes). Pass ``init_bound=0.02`` to restore
        the previous fixed default bit-for-bit (default: None).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        This layer takes no ``manifold_module``, so compute precision follows the
        input array's dtype (the parameters are cast to it).

    Attributes
    ----------
    kernel : nnx.Param
        Weight matrix of shape (in_features, out_features).
    bias : nnx.Param or None
        Bias vector of shape (out_features,) if use_bias=True, else None.
    eps : float
        Numerical stability parameter.

    Notes
    -----
    Weight Initialization:
        The default bound ``sqrt(3 / in_features)`` is norm-preserving: ``U(-a, a)``
        has entry std ``a / sqrt(3)``, and ``htc`` applies no nonlinearity to the
        matmul output, so the per-layer input-Jacobian RMS gain is
        ``(a / sqrt(3)) * sqrt(in_features) ~= 1``. The previous fixed default
        ``U(-0.02, 0.02)`` (an in-house convention inherited from the FHNN/FHCNN
        family, not the Hypformer reference) is width-dependent *contractive*:
        its gain ``0.02 / sqrt(3) * sqrt(in_features)`` is ~0.09 at
        ``in_features=65``, so stacks of two or more layers collapse
        input-dependent variation below the float32 noise floor -- the stack
        becomes a constant map with near-zero gradients and training freezes from
        step 0 (observed in goal-conditioned RL: depth-2 head, c=0.5, frozen
        through 3.1M steps).

        The official Hypformer reference initializes with Xavier-uniform and gain
        ``sqrt(2)``, i.e. ``bound = 2 * sqrt(3 / (in_features + out_features))``
        (per-layer gain ~= sqrt(2) at equal widths), intended for its
        ReLU + LayerNorm regime where normalization absorbs the growth. Recover it
        by passing that bound explicitly; for bare ReLU stacks without
        normalization, the He-style ``bound = sqrt(6 / in_features)`` compensates
        the ReLU variance halving.

        Caution: the norm-preserving default is validated for shallow stacks
        (depth <= 2) behind a Euclidean trunk at moderate curvature (c=0.5). In
        deep fully-hyperbolic stacks or at large ``c``, any gain above 1 compounds
        as ``gain**depth`` -- monitor output norms.

    See Also
    --------
    hyperbolix.nn_layers.hyperboloid_core.htc : Core HTC function for functional transformations.
    HypLinearHyperboloidFHCNN : Alternative hyperbolic linear layer with sigmoid scaling.

    References
    ----------
    Yang et al., "Hypformer: Exploring Efficient Transformer Fully in
    Hyperbolic Space", 2025.

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import HTCLinear
    >>> from hyperbolix.manifolds import Hyperboloid
    >>>
    >>> # Create layer
    >>> layer = HTCLinear(in_features=5, out_features=8, rngs=nnx.Rngs(0))
    >>>
    >>> # Forward pass
    >>> manifold = Hyperboloid()
    >>> x = jnp.ones((32, 5))  # batch of 32 points
    >>> x = jax.vmap(manifold.proj, in_axes=(0, None))(x, 1.0)
    >>> y = layer(x, c_in=1.0, c_out=2.0)
    >>> y.shape
    (32, 9)  # 8 spatial + 1 time
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        use_bias: bool = True,
        init_bound: float | None = None,
        eps: float = 1e-7,
        param_dtype: DTypeLike = jnp.float32,
    ):
        # Norm-preserving fan-in bound: U(-a, a) has entry std a/sqrt(3), and htc applies
        # no nonlinearity to the matmul output, so the per-layer input-Jacobian RMS gain
        # is (a/sqrt(3)) * sqrt(in_features) ~= 1 for a = sqrt(3/in_features). Any fixed
        # width-independent bound cannot hold gain ~= 1: the old 0.02 default gave gain
        # ~0.09 at in_features=65, so depth>=2 stacks contracted input variation below
        # the float32 noise floor (constant map, zero gradients, frozen training).
        # in_features counts the near-constant time column -- slightly conservative.
        if init_bound is None:
            init_bound = (3.0 / in_features) ** 0.5
        self.kernel = nnx.Param(
            jax.random.uniform(
                rngs.params(), (in_features, out_features), dtype=param_dtype, minval=-init_bound, maxval=init_bound
            )
        )
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_features,), dtype=param_dtype))
        else:
            self.bias = None
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "batch in_features"],
        c_in: float = 1.0,
        c_out: float = 1.0,
    ) -> Float[Array, "batch out_features_plus_1"]:
        """Apply HTC linear transformation.

        Parameters
        ----------
        x : Array of shape (batch, in_features)
            Input points on hyperboloid with curvature c_in.
        c_in : float, optional
            Input curvature (default: 1.0).
        c_out : float, optional
            Output curvature (default: 1.0).

        Returns
        -------
        y : Array of shape (batch, out_features+1)
            Output points on hyperboloid with curvature c_out.
        """

        def linear_fn(z):
            # Cast params to the working dtype so float64 weights (from global
            # jax_enable_x64) don't promote a float32 computation to float64.
            out = z @ self.kernel[...].astype(z.dtype)
            if self.bias is not None:
                out = out + self.bias[...].astype(z.dtype)
            return out

        return htc(x, linear_fn, c_in, c_out, self.eps)


class FGGLinear(nnx.Module):
    """Fast and Geometrically Grounded Lorentz linear layer.

    Implements the FGG linear layer from Klis et al. 2026. The key insight is that
    the sinh/arcsinh cancellation in the Lorentzian activation chain simplifies the
    forward pass to: matmul with spacelike V matrix -> Euclidean activation ->
    time reconstruction. This achieves linear growth of hyperbolic distance (vs
    logarithmic for Chen et al. 2022) and ~3x faster training/inference.

    Forward pass:
        1. Build spacelike V matrix from (U, b) with Minkowski metric absorbed
        2. z = x @ V   (Minkowski inner products via a single matmul)
        3. z = h(z)     (Euclidean activation, e.g. ReLU)
        4. y_0 = sqrt(||z||^2 + 1/c)   (time reconstruction)
        5. y = [y_0, z]   (on hyperboloid)

    Parameters
    ----------
    in_features : int
        Input ambient dimension (D_in + 1), including time component.
    out_features : int
        Output ambient dimension (D_out + 1), including time component.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    activation : Callable or None, optional
        Euclidean activation function applied after matmul (default: None).
    reset_params : str, optional
        Weight initialization scheme: ``"eye"``, ``"xavier"``, ``"kaiming"``,
        ``"lorentz_kaiming"``, ``"fan_out"``, or ``"mlr"`` (default: ``"fan_out"``).
        The ``"fan_out"`` default (std ``sqrt(1/out_spatial)``) is norm-preserving
        (``||z|| ~= gain * ||x_spatial||``), a deliberate deviation from the Klis et al.
        2026 classification reference: it suits unnormalized stacks (e.g. an RL backbone
        feeding a bounded Poincare-ball projection), where the reference's BatchNorm
        regime does not apply. Pass ``reset_params="eye", init_bias=0.5`` to recover the
        previous reference-style init.
    use_weight_norm : bool, optional
        If True, reparameterize U as ``g * v / ||v||`` for weight normalization
        (default: False). Note that ``gain`` and the ``"fan_out"`` scale are
        renormalized away in this mode (the magnitude is set by ``kernel_scale``).
    init_bias : float, optional
        Initial value for bias entries (default: 0.0). A zero bias removes the
        ``~sqrt(out) * init_bias`` quadrature term that ``build_spacelike_V`` injects
        into the time row (``-||w|| * sinh(-sqrt(c) * b / ||w||)``); pair with
        ``"fan_out"`` for norm preservation.
    gain : float, optional
        Multiplier on the random init (default: 1.0). With ``reset_params="fan_out"``
        it sets the effective column std to ``gain / sqrt(out_spatial)``. No-op for
        ``"eye"``; renormalized away under ``use_weight_norm=True``.
    eps : float, optional
        Numerical stability floor (default: 1e-7).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        This layer takes no ``manifold_module``, so compute precision follows the
        input array's dtype (the parameters are cast to it).

    References
    ----------
    Klis et al. "Fast and Geometrically Grounded Lorentz Neural Networks" (2026).

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import FGGLinear
    >>> import jax.numpy as jnp
    >>>
    >>> layer = FGGLinear(33, 65, rngs=nnx.Rngs(0), activation=jax.nn.relu)
    >>> x = jnp.ones((8, 33))
    >>> # project to hyperboloid
    >>> x = x.at[:, 0].set(jnp.sqrt(jnp.sum(x[:, 1:]**2, axis=-1) + 1.0))
    >>> y = layer(x, c=1.0)
    >>> y.shape
    (8, 65)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        activation: Callable[[jax.Array], jax.Array] | None = None,
        reset_params: str = "fan_out",
        use_weight_norm: bool = False,
        init_bias: float = 0.0,
        gain: float = 1.0,
        eps: float = 1e-7,
        param_dtype: DTypeLike = jnp.float32,
    ):
        in_spatial = in_features - 1  # I
        out_spatial = out_features - 1  # O

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        self.eps = eps

        # Euclidean weight U (I, O). Default "fan_out" std sqrt(1/out_spatial) is
        # norm-preserving; gain scales the random init (no-op for "eye").
        U_init = _fgg_weight_init(
            rngs.params(), in_spatial, out_spatial, in_features, out_features, reset_params, param_dtype, gain=gain
        )

        # Weight normalization: decompose kernel = softplus(kernel_scale) * kernel_dir / ||kernel_dir||
        if use_weight_norm:
            # Reference: kernel_dir from reset_params (normalized in forward), kernel_scale fixed magnitude
            self.kernel_dir = nnx.Param(U_init)  # (I, O) direction
            g_init_val = jnp.sqrt(1.0 / (in_features + out_features))
            self.kernel_scale = nnx.Param(jnp.full((out_spatial,), g_init_val, dtype=param_dtype))  # (O,)
        else:
            self.kernel = nnx.Param(U_init)  # (I, O)

        # Bias: init to init_bias
        self.bias = nnx.Param(jnp.full((out_spatial,), init_bias, dtype=param_dtype))  # (O,)

    def _get_kernel(self) -> jax.Array:
        """Return the effective weight matrix, handling weight normalization."""
        return _get_effective_kernel(
            self.kernel[...] if not self.use_weight_norm else None,
            self.kernel_dir[...] if self.use_weight_norm else None,
            self.kernel_scale[...] if self.use_weight_norm else None,
            self.use_weight_norm,
            self.eps,
        )

    def __call__(
        self,
        x_BAi: Float[Array, "batch in_features"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_features"]:
        """Forward pass through the FGG linear layer.

        Parameters
        ----------
        x_BAi : Array, shape (B, Ai)
            Input points on the hyperboloid with curvature ``c``.
            Ai = in_features (ambient dimension).
        c : float, optional
            Curvature parameter (default: 1.0).

        Returns
        -------
        y_BAo : Array, shape (B, Ao)
            Output points on the hyperboloid with curvature ``c``.
            Ao = out_features (ambient dimension).
        """
        U_IO = self._get_kernel()  # (I, O)
        return _fgg_linear_forward(x_BAi, U_IO, self.bias[...], c, self.activation, self.eps)
