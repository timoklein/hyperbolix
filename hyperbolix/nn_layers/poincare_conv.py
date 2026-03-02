"""Poincaré ball convolutional layers for JAX/Flax NNX."""

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

    This layer implements hyperbolic convolution using beta-concatenation
    (from HNN++, Shimizu et al. 2020) and the HNN++ linear layer, following
    the Poincaré ResNet approach (van Spengler et al. 2023).

    Computation steps:
        1) Extract receptive field (kernel_size x kernel_size) of hyperbolic points
        2) Apply beta-concatenation to combine receptive field points
        3) Pass through HNN++ hyperbolic linear layer

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
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    clamping_factor : float
        Clamping factor for the HNN++ linear layer output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the HNN++ linear layer output (default: 50.0)

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (padding, input_space)
        are treated as static and will be baked into the compiled function.

    Dimension math:
        - beta-concat: K^2 points of dim C_in -> single point of dim K^2 * C_in
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
        input_space: str = "manifold",
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
            required_methods=("expmap_0", "logmap_0", "beta_concat", "compute_mlr_pp"),
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

        # Compute dimensions for the linear layer
        # beta-concat: K^2 points of dim C_in -> single point of dim K^2 * C_in
        kernel_h, kernel_w = self.kernel_size
        K2 = kernel_h * kernel_w
        beta_concat_out_dim = K2 * in_channels

        # HNN++ linear: in_dim = K^2 * C_in, out_dim = C_out
        self.linear = HypLinearPoincarePP(
            manifold_module=self.manifold,
            in_dim=beta_concat_out_dim,
            out_dim=out_channels,
            rngs=rngs,
            input_space="manifold",  # beta_concat output is always on manifold
            clamping_factor=clamping_factor,
            smoothing_factor=smoothing_factor,
        )

    def _extract_patches(
        self,
        x: Float[Array, "batch height width in_channels"],
    ) -> Float[Array, "batch out_height out_width kernel_h kernel_w in_channels"]:
        """Extract patches (receptive fields) from the input using optimized JAX primitives."""
        batch, height, width, in_channels = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # 1. Handle Padding (Manually, to ensure manifold validity)
        if self.padding == "SAME":
            out_height = (height + stride_h - 1) // stride_h
            out_width = (width + stride_w - 1) // stride_w
            pad_h = max((out_height - 1) * stride_h + kernel_h - height, 0)
            pad_w = max((out_width - 1) * stride_w + kernel_w - width, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # Pad with edge values to stay on manifold
            x = jnp.pad(
                x,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )

        # 2. Extract Patches using LAX primitive (High Performance)
        # Output shape: (batch, out_height, out_width, in_channels * kernel_h * kernel_w)
        patches_flat = jax.lax.conv_general_dilated_patches(
            lhs=x,
            filter_shape=(kernel_h, kernel_w),
            window_strides=(stride_h, stride_w),
            padding="VALID",  # We already padded manually
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )

        # 3. Reshape to separate kernel dimensions and channels
        out_h, out_w = patches_flat.shape[1], patches_flat.shape[2]

        # JAX conv_general_dilated_patches output order is "c" + spatial dims from rhs_spec
        # With rhs_spec="OIHW", the flattened dim order is (in_channels, kernel_h, kernel_w)
        # We need (kernel_h, kernel_w, in_channels) for downstream processing
        patches = patches_flat.reshape(batch, out_h, out_w, in_channels, kernel_h, kernel_w)
        patches = patches.transpose(0, 1, 2, 4, 5, 3)  # (batch, out_h, out_w, kh, kw, in_c)

        return patches

    def __call__(
        self,
        x: Float[Array, "batch height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_height out_width out_channels"]:
        """
        Forward pass through the Poincaré convolutional layer.

        Parameters
        ----------
        x : Array of shape (batch, height, width, in_channels)
            Input feature map where each pixel is a point on the Poincaré ball
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        out : Array of shape (batch, out_height, out_width, out_channels)
            Output feature map on the Poincaré ball
        """
        # Map to manifold if needed (static branch - JIT friendly)
        if self.input_space == "tangent":
            # 1. Flatten batch and spatial dims: (B*H*W, C)
            x_flat = x.reshape(-1, x.shape[-1])

            # 2. Apply single vmap over the list of vectors
            x_mapped = jax.vmap(self.manifold.expmap_0, in_axes=(0, None))(x_flat, c)

            # 3. Reshape back to (B, H, W, C)
            x = x_mapped.reshape(x.shape)

        # Extract patches (receptive fields)
        # Shape: (batch, out_h, out_w, kernel_h, kernel_w, in_channels)
        patches = self._extract_patches(x)

        batch, out_h, out_w, kh, kw, in_c = patches.shape

        # Flatten batch and spatial dims for parallel processing
        # Shape: (batch * out_h * out_w, kh * kw, in_channels)
        patches_flat = patches.reshape(-1, kh * kw, in_c)

        # 1. Apply beta-concatenation
        # vmap over the flattened spatial/batch dimension
        # Input: (K^2, C_in) -> Output: (K^2 * C_in,)
        beta_out = jax.vmap(self.manifold.beta_concat, in_axes=(0, None))(patches_flat, c)

        # 2. Apply HNN++ Linear
        # Input: (batch*out_h*out_w, K^2 * C_in) -> Output: (batch*out_h*out_w, C_out)
        linear_out = self.linear(beta_out, c)

        # 3. Reshape back
        output = linear_out.reshape(batch, out_h, out_w, self.out_channels)

        return output
