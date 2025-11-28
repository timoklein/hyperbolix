"""Hyperboloid convolutional layers for JAX/Flax NNX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from .hyperboloid_linear import HypLinearHyperboloid


class HypConvHyperboloid(nnx.Module):
    """
    Hyperbolic Convolutional Layer for Hyperboloid model.

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
        self.linear = HypLinearHyperboloid(
            manifold_module=manifold_module,
            in_dim=hcat_out_ambient_dim,
            out_dim=out_channels,
            rngs=rngs,
            input_space="manifold",  # HCat output is always on manifold
        )

    def _extract_patches(
        self,
        x: Float[Array, "batch height width in_channels"],
    ) -> Float[Array, "batch out_height out_width kernel_h kernel_w in_channels"]:
        """Extract patches (receptive fields) from the input.

        Parameters
        ----------
        x : Array of shape (batch, height, width, in_channels)
            Input feature map

        Returns
        -------
        patches : Array of shape (batch, out_height, out_width, kernel_h, kernel_w, in_channels)
            Extracted patches
        """
        _batch, height, width, _in_channels = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # Calculate output dimensions
        if self.padding == "SAME":
            out_height = (height + stride_h - 1) // stride_h
            out_width = (width + stride_w - 1) // stride_w
            # Calculate padding needed
            pad_height = max((out_height - 1) * stride_h + kernel_h - height, 0)
            pad_width = max((out_width - 1) * stride_w + kernel_w - width, 0)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            # For hyperbolic manifold, we need to pad with valid manifold points
            # Use the origin point: [sqrt(1/c), 0, ..., 0] - but we don't have c here yet
            # Instead, we'll pad with edge values (replicate) which are valid manifold points
            x = jnp.pad(
                x,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",  # Replicate edge values instead of zeros
            )
            # Update height and width after padding
            height = x.shape[1]
            width = x.shape[2]
        else:  # VALID
            out_height = (height - kernel_h) // stride_h + 1
            out_width = (width - kernel_w) // stride_w + 1

        # Extract patches using indexing
        patches_list = []
        for i in range(out_height):
            row_patches = []
            for j in range(out_width):
                h_start = i * stride_h
                w_start = j * stride_w
                patch = x[:, h_start : h_start + kernel_h, w_start : w_start + kernel_w, :]
                row_patches.append(patch)
            patches_list.append(jnp.stack(row_patches, axis=1))
        patches = jnp.stack(patches_list, axis=1)

        return patches  # (batch, out_height, out_width, kernel_h, kernel_w, in_channels)

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
            # vmap over batch, height, width
            x = jax.vmap(
                jax.vmap(jax.vmap(self.manifold.expmap_0, in_axes=(0, None)), in_axes=(0, None)),
                in_axes=(0, None),
            )(x, c)

        # Extract patches (receptive fields)
        patches = self._extract_patches(x)  # (batch, out_h, out_w, kernel_h, kernel_w, in_channels)

        batch, out_height, out_width, kernel_h, kernel_w, in_channels = patches.shape

        # Reshape to apply HCat: (batch, out_h, out_w, kernel_h*kernel_w, in_channels)
        patches_reshaped = patches.reshape(batch, out_height, out_width, kernel_h * kernel_w, in_channels)

        # Apply HCat to each receptive field
        # vmap over batch, out_height, out_width dimensions
        def apply_hcat_single(receptive_field):
            """Apply HCat to a single receptive field of shape (kernel_h*kernel_w, in_channels)."""
            return self.manifold.hcat(receptive_field, c)

        hcat_fn = jax.vmap(jax.vmap(jax.vmap(apply_hcat_single)))
        hcat_output = hcat_fn(patches_reshaped)  # (batch, out_h, out_w, hcat_out_dim)

        # Apply linear transformation to each spatial location
        # vmap over batch, out_height, out_width dimensions
        linear_fn = jax.vmap(jax.vmap(jax.vmap(lambda p: self.linear(p.reshape(1, -1), c).squeeze(0))))
        output = linear_fn(hcat_output)  # (batch, out_h, out_w, out_channels)

        return output
