"""Hyperboloid convolutional layers for JAX/Flax NNX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from .hyperboloid_linear import HypLinearHyperboloidFHCNN


class HypConv2DHyperboloid(nnx.Module):
    """
    Hyperbolic 2D Convolutional Layer for Hyperboloid model.

    This layer implements fully hyperbolic convolution as described in
    "Fully Hyperbolic Convolutional Neural Networks for Computer Vision".

    Computation steps:
        1) Extract receptive field (kernel_size x kernel_size) of hyperbolic points
        2) Apply HCat (Lorentz direct concatenation) to combine receptive field points
        3) Pass through hyperbolic linear layer (LFC)

    Parameters
    ----------
    manifold_module : module
        The Hyperboloid manifold module
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
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

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (padding, input_space)
        are treated as static and will be baked into the compiled function.

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic convolutional neural networks for computer vision."
        arXiv preprint arXiv:2303.15919 (2023).
    """

    def __init__(
        self,
        manifold_module: Any,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        input_space: str = "manifold",
    ):
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be either 'SAME' or 'VALID', got '{padding}'")
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration
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
        # Receptive field: kernel_h x kernel_w pixels, each is a point in in_channels-dim ambient space
        # HCat input: N = kernel_h x kernel_w points, each in in_channels ambient dimensions
        # HCat output: ambient dim = (in_channels - 1) x N + 1
        #            = (in_channels - 1) x (kernel_h x kernel_w) + 1
        kernel_h, kernel_w = self.kernel_size
        N = kernel_h * kernel_w  # Number of points in receptive field
        d = in_channels - 1  # Input manifold dimension
        hcat_out_ambient_dim = d * N + 1  # HCat output ambient dimension

        # Create the linear transformation layer
        # Input: hcat_out_ambient_dim, Output: out_channels
        self.linear = HypLinearHyperboloidFHCNN(
            manifold_module=manifold_module,
            in_dim=hcat_out_ambient_dim,
            out_dim=out_channels,
            rngs=rngs,
            input_space="manifold",  # HCat output is always on manifold
            learnable_scale=False,
            normalize=False,
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
            dimension_numbers=("NHWC", "OIHW", "NHWC"),  # Standard TF/JAX layout
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
        Forward pass through the hyperbolic convolutional layer.

        Parameters
        ----------
        x : Array of shape (batch, height, width, in_channels)
            Input feature map where each pixel is a point on the Hyperboloid manifold
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        out : Array of shape (batch, out_height, out_width, out_channels)
            Output feature map on the Hyperboloid manifold
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

        # 1. Apply HCat
        # vmap over the flattened spatial/batch dimension
        # Input to hcat: (N_points, in_channels) -> Output: (hcat_dim,)
        hcat_out = jax.vmap(self.manifold.hcat, in_axes=(0, None))(patches_flat, c)  # (batch*out_h*out_w, hcat_dim)

        # 2. Apply Linear
        # Input: (hcat_dim,) -> Output: (out_channels,)
        linear_out = self.linear(hcat_out, c)

        # 3. Reshape back
        output = linear_out.reshape(batch, out_h, out_w, self.out_channels)

        return output


class HypConv3DHyperboloid(nnx.Module):
    """
    Hyperbolic 3D Convolutional Layer for Hyperboloid model.

    This layer implements fully hyperbolic 3D convolution, extending the 2D approach
    from "Fully Hyperbolic Convolutional Neural Networks for Computer Vision" to
    volumetric data.

    Computation steps:
        1) Extract receptive field (kernel_size x kernel_size x kernel_size) of hyperbolic points
        2) Apply HCat (Lorentz direct concatenation) to combine receptive field points
        3) Pass through hyperbolic linear layer (LFC)

    Parameters
    ----------
    manifold_module : module
        The Hyperboloid manifold module
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple[int, int, int]
        Size of the convolutional kernel (depth, height, width)
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    stride : int or tuple[int, int, int]
        Stride of the convolution (default: 1)
    padding : str
        Padding mode, either 'SAME' or 'VALID' (default: 'SAME')
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (padding, input_space)
        are treated as static and will be baked into the compiled function.

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic convolutional neural networks for computer vision."
        arXiv preprint arXiv:2303.15919 (2023).
    """

    def __init__(
        self,
        manifold_module: Any,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int, int] = 1,
        padding: str = "SAME",
        input_space: str = "manifold",
    ):
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be either 'SAME' or 'VALID', got '{padding}'")
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_space = input_space
        self.padding = padding

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Handle stride as int or tuple
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride

        # Compute dimensions for the linear layer
        # Receptive field: kernel_d x kernel_h x kernel_w voxels, each is a point in in_channels-dim ambient space
        # HCat input: N = kernel_d x kernel_h x kernel_w points, each in in_channels ambient dimensions
        # HCat output: ambient dim = (in_channels - 1) x N + 1
        #            = (in_channels - 1) x (kernel_d x kernel_h x kernel_w) + 1
        kernel_d, kernel_h, kernel_w = self.kernel_size
        N = kernel_d * kernel_h * kernel_w  # Number of points in receptive field
        d = in_channels - 1  # Input manifold dimension
        hcat_out_ambient_dim = d * N + 1  # HCat output ambient dimension

        # Create the linear transformation layer
        # Input: hcat_out_ambient_dim, Output: out_channels
        self.linear = HypLinearHyperboloidFHCNN(
            manifold_module=manifold_module,
            in_dim=hcat_out_ambient_dim,
            out_dim=out_channels,
            rngs=rngs,
            input_space="manifold",  # HCat output is always on manifold
            learnable_scale=False,
            normalize=False,
        )

    def _extract_patches(
        self,
        x: Float[Array, "batch depth height width in_channels"],
    ) -> Float[Array, "batch out_depth out_height out_width kernel_d kernel_h kernel_w in_channels"]:
        """Extract patches (receptive fields) from the 3D input using optimized JAX primitives."""
        batch, depth, height, width, in_channels = x.shape
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride

        # 1. Handle Padding (Manually, to ensure manifold validity)
        if self.padding == "SAME":
            out_depth = (depth + stride_d - 1) // stride_d
            out_height = (height + stride_h - 1) // stride_h
            out_width = (width + stride_w - 1) // stride_w

            pad_d = max((out_depth - 1) * stride_d + kernel_d - depth, 0)
            pad_h = max((out_height - 1) * stride_h + kernel_h - height, 0)
            pad_w = max((out_width - 1) * stride_w + kernel_w - width, 0)

            pad_front = pad_d // 2
            pad_back = pad_d - pad_front
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # Pad with edge values to stay on manifold
            x = jnp.pad(
                x,
                ((0, 0), (pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )

        # 2. Extract Patches using LAX primitive
        # Output shape: (batch, out_depth, out_height, out_width, in_channels * kernel_d * kernel_h * kernel_w)
        patches_flat = jax.lax.conv_general_dilated_patches(
            lhs=x,
            filter_shape=(kernel_d, kernel_h, kernel_w),
            window_strides=(stride_d, stride_h, stride_w),
            padding="VALID",
            dimension_numbers=("NDHWC", "OIDHW", "NDHWC"),
        )

        # 3. Reshape to separate kernel dimensions and channels
        out_d, out_h, out_w = patches_flat.shape[1], patches_flat.shape[2], patches_flat.shape[3]

        # JAX conv_general_dilated_patches output order is "c" + spatial dims from rhs_spec
        # With rhs_spec="OIDHW", the flattened dim order is (in_channels, kernel_d, kernel_h, kernel_w)
        # We need (kernel_d, kernel_h, kernel_w, in_channels) for downstream processing
        patches = patches_flat.reshape(batch, out_d, out_h, out_w, in_channels, kernel_d, kernel_h, kernel_w)
        patches = patches.transpose(0, 1, 2, 3, 5, 6, 7, 4)  # (batch, out_d, out_h, out_w, kd, kh, kw, in_c)

        return patches

    def __call__(
        self,
        x: Float[Array, "batch depth height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_depth out_height out_width out_channels"]:
        """
        Forward pass through the hyperbolic 3D convolutional layer.

        Parameters
        ----------
        x : Array of shape (batch, depth, height, width, in_channels)
            Input feature map where each voxel is a point on the Hyperboloid manifold
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        out : Array of shape (batch, out_depth, out_height, out_width, out_channels)
            Output feature map on the Hyperboloid manifold
        """
        # Map to manifold if needed (static branch - JIT friendly)
        if self.input_space == "tangent":
            # 1. Flatten batch and spatial dims: (B*D*H*W, C)
            x_flat = x.reshape(-1, x.shape[-1])

            # 2. Apply single vmap over the list of vectors
            x_mapped = jax.vmap(self.manifold.expmap_0, in_axes=(0, None))(x_flat, c)

            # 3. Reshape back to (B, D, H, W, C)
            x = x_mapped.reshape(x.shape)

        # Extract patches (receptive fields)
        # Shape: (batch, out_d, out_h, out_w, kernel_d, kernel_h, kernel_w, in_channels)
        patches = self._extract_patches(x)

        batch, out_d, out_h, out_w, kd, kh, kw, in_c = patches.shape

        # Flatten batch and spatial dims for parallel processing
        # Shape: (batch * out_d * out_h * out_w, kd * kh * kw, in_channels)
        patches_flat = patches.reshape(-1, kd * kh * kw, in_c)

        # 1. Apply HCat
        # vmap over the flattened spatial/batch dimension
        hcat_out = jax.vmap(self.manifold.hcat, in_axes=(0, None))(patches_flat, c)  # (batch*out_d*out_h*out_w, hcat_dim)

        # 2. Apply Linear
        linear_out = self.linear(hcat_out, c)

        # 3. Reshape back
        output = linear_out.reshape(batch, out_d, out_h, out_w, self.out_channels)

        return output
