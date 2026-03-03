"""Poincaré ball convolutional layers for JAX/Flax NNX.

Implements Poincaré convolution following the reference Poincaré ResNet
(van Spengler et al. 2023). The key design principle is to operate in
tangent space between layers for numerical stability, matching the
reference computation flow:

    tangent input → beta-scale → unfold patches → expmap_0 → HNN++ FC → logmap_0 → tangent output

This avoids the numerically unstable logmap_0 round-trips in beta_concat
that cause NaN when points approach the Poincaré ball boundary.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds.poincare import Poincare

from ._helpers import validate_poincare_manifold
from .poincare_linear import HypLinearPoincarePP


class HypConv2DPoincare(nnx.Module):
    """
    Hyperbolic 2D Convolutional Layer for Poincaré ball model.

    This layer implements hyperbolic convolution following the Poincaré ResNet
    (van Spengler et al. 2023) approach: beta-scaling in tangent space,
    patch extraction, expmap to manifold, HNN++ fully-connected, logmap back
    to tangent space.

    The layer operates in tangent space internally and returns tangent-space
    output, matching the reference implementation. This design avoids the
    numerically unstable logmap_0 round-trips that cause NaN gradients
    when points approach the Poincaré ball boundary.

    Computation steps:
        1) Map to tangent space if input is on manifold
        2) Scale tangent vectors by beta function ratio (beta-concatenation scaling)
        3) Extract patches via im2col (zero-padding in tangent space)
        4) Map concatenated patch vectors to manifold via expmap_0
        5) Apply HNN++ fully-connected layer
        6) Map back to tangent space via logmap_0

    Parameters
    ----------
    manifold_module : Poincare
        Class-based Poincaré manifold instance
    in_channels : int
        Number of input channels (Poincaré ball dimension per pixel)
    out_channels : int
        Number of output channels (Poincaré ball dimension per pixel)
    kernel_size : int or tuple[int, int]
        Size of the convolutional kernel
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    stride : int or tuple[int, int]
        Stride of the convolution (default: 1)
    padding : str
        Padding mode, either 'SAME' or 'VALID' (default: 'SAME')
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'tangent').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    id_init : bool
        If True, use identity initialization (1/2 * I) for the linear sublayer weights,
        matching the reference Poincaré ResNet implementation (default: True).
        The 1/2 factor compensates for the factor of 2 inside the HNN++ distance formula.
    clamping_factor : float
        Clamping factor for the HNN++ linear layer output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the HNN++ linear layer output (default: 50.0)

    Notes
    -----
    Output Space:
        This layer always returns tangent-space output (matching the reference).
        Between conv layers, use standard activations (e.g., jax.nn.relu) directly
        on the tangent-space features. Use expmap_0 to map back to the manifold
        only when needed (e.g., before the classification head).

    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (padding, input_space)
        are treated as static and will be baked into the compiled function.

    Dimension math:
        - beta-scaling + patch extraction: (H, W, C_in) → (oh, ow, K^2 * C_in)
        - HNN++ linear: in_dim = K^2 * C_in, out_dim = C_out

    References
    ----------
    Shimizu et al. "Hyperbolic neural networks++." arXiv:2006.08210 (2020).
    van Spengler et al. "Poincaré ResNet." ICML 2023.
    """

    def __init__(
        self,
        manifold_module: Poincare,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        input_space: str = "tangent",
        id_init: bool = True,
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
    ):
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be either 'SAME' or 'VALID', got '{padding}'")
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration
        validate_poincare_manifold(
            manifold_module,
            required_methods=("expmap_0", "logmap_0", "compute_mlr_pp"),
        )
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_space = input_space
        self.padding = padding

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Handle stride as int or tuple
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Precompute beta function ratio for tangent-space scaling
        # B(n/2, 1/2) / B(n_i/2, 1/2) where n = K^2 * C_in, n_i = C_in
        kernel_h, kernel_w = self.kernel_size
        K2 = kernel_h * kernel_w
        concat_dim = K2 * in_channels
        beta_n = jax.scipy.special.beta(concat_dim / 2.0, 0.5)
        beta_ni = jax.scipy.special.beta(in_channels / 2.0, 0.5)
        self.beta_scale = beta_n / beta_ni

        # HNN++ linear: in_dim = K^2 * C_in, out_dim = C_out
        self.linear = HypLinearPoincarePP(
            manifold_module=self.manifold,
            in_dim=concat_dim,
            out_dim=out_channels,
            rngs=rngs,
            input_space="manifold",  # input is on manifold after expmap_0
            clamping_factor=clamping_factor,
            smoothing_factor=smoothing_factor,
        )

        # Override weight init with identity initialization from reference
        # (van Spengler et al. 2023): W = 1/2 * I(C_out, K^2*C_in)
        # The 1/2 factor compensates for the factor of 2 in the HNN++ distance formula.
        if id_init:
            eye = jnp.eye(out_channels, concat_dim)  # (C_out, K^2*C_in)
            self.linear.weight[...] = 0.5 * eye

    def __call__(
        self,
        x: Float[Array, "batch height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_height out_width out_channels"]:
        """
        Forward pass through the Poincaré convolutional layer.

        Follows the reference computation flow: tangent-space beta-scaling,
        patch extraction, expmap_0, HNN++ FC, logmap_0.

        Parameters
        ----------
        x : Array of shape (batch, height, width, in_channels)
            Input feature map. Can be tangent-space or manifold points
            depending on input_space setting.
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        out : Array of shape (batch, out_height, out_width, out_channels)
            Output feature map in tangent space at the origin.
            Use standard activations (e.g., jax.nn.relu) between layers.
        """
        # Step 1: Map to tangent space if input is on manifold
        if self.input_space == "manifold":
            orig_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1])  # (B*H*W, C_in)
            x_flat = jax.vmap(self.manifold.logmap_0, in_axes=(0, None))(x_flat, c)
            x = x_flat.reshape(orig_shape)

        # Now x is in tangent space: (B, H, W, C_in)

        # Step 2: Scale tangent vectors by beta ratio (matching reference)
        # This replaces the per-point logmap_0→scale→concat→expmap_0 in old beta_concat
        x = x * self.beta_scale

        # Step 3: Extract patches in tangent space using zero-padding
        # Zero-padding is natural in tangent space (zero = origin)
        # Output: (B, oh, ow, K^2 * C_in) - channels already concatenated
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        patches = jax.lax.conv_general_dilated_patches(
            lhs=x,
            filter_shape=(kernel_h, kernel_w),
            window_strides=(stride_h, stride_w),
            padding=self.padding,  # JAX handles SAME/VALID zero-padding
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        # patches shape: (B, oh, ow, C_in * kh * kw)

        batch, out_h, out_w, concat_dim = patches.shape

        # Step 4: Map concatenated patch vectors to manifold via expmap_0
        patches_flat = patches.reshape(-1, concat_dim)  # (B*oh*ow, K^2*C_in)
        manifold_pts = jax.vmap(self.manifold.expmap_0, in_axes=(0, None))(
            patches_flat, c
        )  # (B*oh*ow, K^2*C_in) on Poincaré ball

        # Step 5: Apply HNN++ fully-connected layer
        # Input: (B*oh*ow, K^2*C_in) on manifold → Output: (B*oh*ow, C_out) on manifold
        fc_out = self.linear(manifold_pts, c)

        # Step 6: Map back to tangent space (matching reference: logmap0 at end)
        tangent_out = jax.vmap(self.manifold.logmap_0, in_axes=(0, None))(fc_out, c)  # (B*oh*ow, C_out) in tangent space

        # Reshape to spatial output
        output = tangent_out.reshape(batch, out_h, out_w, self.out_channels)

        return output
