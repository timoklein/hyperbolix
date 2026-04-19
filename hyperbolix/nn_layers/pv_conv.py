"""Proper Velocity convolutional layers for JAX/Flax NNX.

Implements the PV convolution from Chen et al. 2026, Sec 5.3. PV's unconstrained
``ℝⁿ`` nature makes this simpler than the Poincaré analogue: patch concatenation
coincides with Euclidean concatenation, so no beta-scaling step is required, and
outputs can stay on the PV manifold without a tangent-space round-trip.

Computation flow:

    pv_input (optional expmap_0) → conv_general_dilated_patches (zero-padded)
                                 → raw concat → HypLinearPV FC → pv_output

Dimension key:
  B: batch size
  H: output height        W: output width
  C: channels (in/out)    K: kernel elements (kh*kw)
  N: flattened batch+spatial (B*H*W)
"""

from collections.abc import Callable

import jax
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds.proper_velocity import ProperVelocity

from ._helpers import validate_pv_manifold
from .pv_linear import _pv_fc_forward


class HypConv2DPV(nnx.Module):
    """
    Hyperbolic 2D Convolutional Layer for the Proper Velocity model.

    Implements Chen et al. 2026, Sec 5.3. Because PV's geometry is unconstrained
    (``ℝⁿ``), patch concatenation coincides with Euclidean concatenation — no
    beta-scaling step is required (unlike ``HypConv2DPoincare``). The layer
    returns points on the PV manifold; between conv layers, activations can be
    applied directly in PV space (paper Sec 5.3 "Activation").

    Computation steps:
        1) Optionally lift tangent-space input to the PV manifold via ``expmap_0``.
        2) Extract patches with zero-padding via ``jax.lax.conv_general_dilated_patches``.
        3) Flatten batch x spatial dimensions.
        4) Apply the PV FC forward (``_pv_fc_forward``, shared with ``HypLinearPV``).
        5) Reshape back to spatial output.

    Parameters
    ----------
    manifold_module : ProperVelocity
        Class-based Proper Velocity manifold instance.
    in_channels : int
        Number of input channels (PV dimension per pixel).
    out_channels : int
        Number of output channels (PV dimension per pixel).
    kernel_size : int or tuple[int, int]
        Size of the convolutional kernel.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    stride : int or tuple[int, int]
        Stride of the convolution (default: 1).
    padding : str
        Padding mode, either 'SAME' or 'VALID' (default: 'SAME').
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration — changing it after initialization
        requires recompilation.
    inner_activation : Callable[[Array], Array] | None
        Optional activation applied inside the outer sinh (paper Eq. 23). Default None.
    clamping_factor : float
        Clamping factor for the PV MLR output (default: 1.0).
    smoothing_factor : float
        Smoothing factor for the PV MLR output (default: 50.0).
    kernel_init_std : float | None
        Standard deviation for the Gaussian kernel init. If ``None`` (default),
        uses He scaling ``sqrt(2 / (kernel_h * kernel_w * in_channels))`` so
        stacked conv blocks preserve variance under ReLU (for small MLR
        arguments the PV output reduces to an Euclidean linear map per patch,
        so standard He analysis applies).

    Notes
    -----
    Output Space:
        Unlike ``HypConv2DPoincare`` (which returns tangent space), this layer
        returns points on the PV manifold. Between conv layers, apply activations
        directly on the PV features — no expmap_0/logmap_0 round-trips needed.

    JIT Compatibility:
        Configuration parameters (padding, input_space, inner_activation,
        clamping_factor, smoothing_factor) are treated as static and are baked
        into the compiled function.

    Dimension math:
        - patch extraction: (H, W, C_in) → (oh, ow, K²·C_in)
        - PV FC: in_dim = K²·C_in, out_dim = C_out

    References
    ----------
    Chen et al. "Proper Velocity Neural Networks." ICLR 2026.
    """

    def __init__(
        self,
        manifold_module: ProperVelocity,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        input_space: str = "manifold",
        inner_activation: Callable[[Array], Array] | None = None,
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        kernel_init_std: float | None = None,
    ):
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be either 'SAME' or 'VALID', got '{padding}'")
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        validate_pv_manifold(
            manifold_module,
            required_methods=("expmap_0", "compute_mlr"),
        )
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_space = input_space
        self.padding = padding
        self.inner_activation = inner_activation
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

        # Handle kernel_size / stride as int or tuple.
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        kernel_h, kernel_w = self.kernel_size
        concat_dim = kernel_h * kernel_w * in_channels

        # Kernel init: He(fan_in) by default so deep conv stacks preserve
        # variance under ReLU. Bias init is uniform U(-1e-3, 1e-3), matching
        # the paper reference.
        std = (2.0 / concat_dim) ** 0.5 if kernel_init_std is None else kernel_init_std
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (out_channels, concat_dim)) * std)
        self.bias = nnx.Param(jax.random.uniform(rngs.params(), (out_channels, 1), minval=-1e-3, maxval=1e-3))

    def __call__(
        self,
        x: Float[Array, "batch height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_height out_width out_channels"]:
        """
        Forward pass through the PV convolutional layer.

        Parameters
        ----------
        x : Array of shape (batch, height, width, in_channels)
            Input feature map. PV manifold points or tangent-space vectors at the
            origin, depending on ``input_space``.
        c : float
            Manifold curvature (default: 1.0).

        Returns
        -------
        out : Array of shape (batch, out_height, out_width, out_channels)
            Output feature map on the PV manifold.
        """
        # Step 1: Optionally lift tangent-space input to the PV manifold.
        if self.input_space == "tangent":
            orig_shape = x.shape
            x_flat_NC = x.reshape(-1, x.shape[-1])  # (B*H*W, C_in)
            x_flat_NC = jax.vmap(self.manifold.expmap_0, in_axes=(0, None))(x_flat_NC, c)
            x = x_flat_NC.reshape(orig_shape)

        # Step 2: Extract patches with zero-padding (raw concat — no beta scaling).
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        patches_BHWKC = jax.lax.conv_general_dilated_patches(
            lhs=x,
            filter_shape=(kernel_h, kernel_w),
            window_strides=(stride_h, stride_w),
            padding=self.padding,
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )  # (B, H, W, K²·C_in)

        batch, out_h, out_w, concat_dim = patches_BHWKC.shape

        # Step 3: Flatten batch x spatial dims for the FC.
        patches_flat_NKC = patches_BHWKC.reshape(-1, concat_dim)  # (N, K²·C_in)

        # Step 4: PV FC — input is already on the PV manifold (no extra lift).
        fc_out_NC = _pv_fc_forward(
            patches_flat_NKC,
            self.kernel[...],
            self.bias[...],
            self.manifold,
            c,
            "manifold",
            self.clamping_factor,
            self.smoothing_factor,
            self.inner_activation,
        )  # (N, C_out) on PV manifold

        # Step 5: Reshape to spatial output.
        return fc_out_NC.reshape(batch, out_h, out_w, self.out_channels)
