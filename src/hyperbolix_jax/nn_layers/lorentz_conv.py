"""Lorentz convolutional layers for JAX/Flax NNX.

Implements the Lorentz convolution from "Fully Hyperbolic CNNs" (Eq. 7):
    out = LorentzBoost(DistanceRescaling(RotationConvolution(x)))

This is an alternative to HypConv2DHyperboloid which uses HCat + HypLinear.
The Lorentz convolution uses norm-preserving rotation convolution followed by
distance rescaling and Lorentz boost transformations.

References:
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr.
    "Fully hyperbolic convolutional neural networks for computer vision."
    arXiv preprint arXiv:2303.15919 (2023).
"""

import logging
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

logger = logging.getLogger(__name__)

# Numerical stability constants
MIN_NORM = 1e-15
EPSILON = 1e-6


def _check_norm_preserving_condition(
    in_spatial: int,
    out_spatial: int,
    kernel_size: int,
) -> bool:
    """Check if Algorithm 3 norm-preserving condition is satisfied.

    Per Algorithm 3: The norm-preserving property only holds when
    K_h * K_w * C_in <= C_out (spatial dimensions).

    Args:
        in_spatial: Number of input spatial channels (in_channels - 1)
        out_spatial: Number of output spatial channels (out_channels - 1)
        kernel_size: Total kernel size (kh * kw)

    Returns:
        True if condition is satisfied, False otherwise
    """
    return kernel_size * in_spatial <= out_spatial


def rotation_conv_2d(
    x: Float[Array, "batch H W in_channels"],
    weight: Float[Array, "out_channels_minus_1 in_channels_minus_1 kh kw"],
    c: float,
    stride: tuple[int, int],
    padding: Literal["SAME", "VALID"],
) -> Float[Array, "batch out_H out_W out_channels"]:
    """Norm-rescaling convolution for 2D inputs.

    Applies convolution to spatial components only, then rescales to approximately
    preserve norms, and reconstructs time component from hyperboloid constraint.

    The rescaling formula is:
        z = W·x_s · (avg_pool(‖x_s‖) / ‖W·x_s‖)

    This is an approximation to true rotation - it preserves the average input norm
    over the receptive field rather than implementing orthogonal transforms.

    Args:
        x: Input feature map, shape (batch, H, W, in_channels)
            Each spatial location is a hyperboloid point
        weight: Convolution weights for spatial components only,
            shape (out_channels-1, in_channels-1, kh, kw)
        c: Manifold curvature (positive)
        stride: Convolution stride (stride_h, stride_w)
        padding: Padding mode, either "SAME" or "VALID"

    Returns:
        Output feature map, shape (batch, out_H, out_W, out_channels)

    Notes:
        - The time component is excluded from convolution and reconstructed
          from the hyperboloid constraint.
        - Per Algorithm 3, the norm-preserving property only strictly holds when
          K_h * K_w * (in_channels-1) <= (out_channels-1). When violated, the
          operation still produces valid manifold points but without the
          norm-preserving guarantee.
        - For SAME padding, input is padded with edge values to stay on manifold.
    """
    _, height, width, _ = x.shape
    kh, kw = weight.shape[2], weight.shape[3]

    # Separate spatial components (ignore time - will be reconstructed)
    x_s = x[..., 1:]  # Shape: (batch, H, W, in_channels-1)

    # Handle SAME padding manually with edge values to stay on manifold
    if padding == "SAME":
        out_height = (height + stride[0] - 1) // stride[0]
        out_width = (width + stride[1] - 1) // stride[1]
        pad_h = max((out_height - 1) * stride[0] + kh - height, 0)
        pad_w = max((out_width - 1) * stride[1] + kw - width, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad spatial components with edge values
        x_s = jnp.pad(
            x_s,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        conv_padding = "VALID"  # Use VALID since we padded manually
    else:
        conv_padding = padding

    # Compute spatial norms per pixel: ‖x_s‖
    x_s_norm = jnp.linalg.norm(x_s, axis=-1, keepdims=True)  # Shape: (batch, H', W', 1)

    # Apply standard convolution to spatial components
    # Note: Per Algorithm 3, norm-preserving property only holds when
    # kh * kw * in_spatial <= out_spatial. The rescaling below approximates
    # norm preservation regardless, but mathematical guarantees only hold
    # when the condition is satisfied.
    conv_out: Array = jax.lax.conv_general_dilated(
        lhs=x_s,
        rhs=weight,
        window_strides=stride,
        padding=conv_padding,
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )  # Shape: (batch, out_H, out_W, out_channels-1)

    # Pool input norms over receptive field (average pooling with same kernel/stride)
    # Note: With SAME padding, x_s_norm includes edge-replicated values, so the average
    # at boundaries weights edge values proportionally to their replication. This is
    # consistent with how the convolution operates on the same edge-padded input.
    pooled_norm = jax.lax.reduce_window(
        operand=x_s_norm,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, kh, kw, 1),
        window_strides=(1, stride[0], stride[1], 1),
        padding=conv_padding,
    )
    pooled_norm = pooled_norm / (kh * kw)  # Shape: (batch, out_H, out_W, 1)

    # Compute output norms
    conv_out_norm = jnp.linalg.norm(conv_out, axis=-1, keepdims=True)  # Shape: (batch, out_H, out_W, 1)

    # Rescale: out_s = conv_out · (pooled_input_norm / conv_out_norm)
    # Handle edge case: if conv_out_norm ≈ 0, use pooled_norm directly to avoid explosion
    is_near_zero = conv_out_norm < EPSILON
    scale = jnp.where(
        is_near_zero,
        1.0,  # Don't scale if output is near zero
        pooled_norm / jnp.maximum(conv_out_norm, EPSILON),
    )
    out_s = cast(Array, conv_out * scale)  # Shape: (batch, out_H, out_W, out_channels-1)

    # Reconstruct time component: out_t = √(‖out_s‖² + 1/c)
    out_s_sqnorm = jnp.sum(out_s**2, axis=-1, keepdims=True)  # Shape: (batch, out_H, out_W, 1)
    out_t = jnp.sqrt(jnp.maximum(out_s_sqnorm + 1.0 / c, MIN_NORM))  # Shape: (batch, out_H, out_W, 1)

    # Concatenate time and spatial components
    result = jnp.concatenate([out_t, out_s], axis=-1)  # Shape: (batch, out_H, out_W, out_channels)

    return result


def rotation_conv_3d(
    x: Float[Array, "batch D H W in_channels"],
    weight: Float[Array, "out_channels_minus_1 in_channels_minus_1 kd kh kw"],
    c: float,
    stride: tuple[int, int, int],
    padding: Literal["SAME", "VALID"],
) -> Float[Array, "batch out_D out_H out_W out_channels"]:
    """Norm-rescaling convolution for 3D inputs.

    3D extension of rotation_conv_2d. Applies convolution to spatial components only,
    then rescales to approximately preserve norms, and reconstructs time component
    from hyperboloid constraint.

    Args:
        x: Input feature map, shape (batch, D, H, W, in_channels)
            Each spatial location is a hyperboloid point
        weight: Convolution weights for spatial components only,
            shape (out_channels-1, in_channels-1, kd, kh, kw)
        c: Manifold curvature (positive)
        stride: Convolution stride (stride_d, stride_h, stride_w)
        padding: Padding mode, either "SAME" or "VALID"

    Returns:
        Output feature map, shape (batch, out_D, out_H, out_W, out_channels)

    Notes:
        - The time component is excluded from convolution and reconstructed
          from the hyperboloid constraint.
        - Per Algorithm 3, the norm-preserving property only strictly holds when
          K_d * K_h * K_w * (in_channels-1) <= (out_channels-1). When violated,
          the operation still produces valid manifold points but without the
          norm-preserving guarantee.
        - For SAME padding, input is padded with edge values to stay on manifold.
    """
    _, depth, height, width, _ = x.shape
    kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]

    # Separate spatial components (ignore time - will be reconstructed)
    x_s = x[..., 1:]  # Shape: (batch, D, H, W, in_channels-1)

    # Handle SAME padding manually with edge values to stay on manifold
    if padding == "SAME":
        out_depth = (depth + stride[0] - 1) // stride[0]
        out_height = (height + stride[1] - 1) // stride[1]
        out_width = (width + stride[2] - 1) // stride[2]

        pad_d = max((out_depth - 1) * stride[0] + kd - depth, 0)
        pad_h = max((out_height - 1) * stride[1] + kh - height, 0)
        pad_w = max((out_width - 1) * stride[2] + kw - width, 0)

        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad spatial components with edge values
        x_s = jnp.pad(
            x_s,
            ((0, 0), (pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        conv_padding = "VALID"  # Use VALID since we padded manually
    else:
        conv_padding = padding

    # Compute spatial norms per voxel: ‖x_s‖
    x_s_norm = jnp.linalg.norm(x_s, axis=-1, keepdims=True)  # Shape: (batch, D', H', W', 1)

    # Apply standard 3D convolution to spatial components
    # Note: Per Algorithm 3, norm-preserving property only holds when
    # kd * kh * kw * in_spatial <= out_spatial. The rescaling below approximates
    # norm preservation regardless, but mathematical guarantees only hold
    # when the condition is satisfied.
    conv_out: Array = jax.lax.conv_general_dilated(
        lhs=x_s,
        rhs=weight,
        window_strides=stride,
        padding=conv_padding,
        dimension_numbers=("NDHWC", "OIDHW", "NDHWC"),
    )  # Shape: (batch, out_D, out_H, out_W, out_channels-1)

    # Pool input norms over receptive field (average pooling)
    # Note: With SAME padding, x_s_norm includes edge-replicated values, so the average
    # at boundaries weights edge values proportionally to their replication. This is
    # consistent with how the convolution operates on the same edge-padded input.
    pooled_norm = jax.lax.reduce_window(
        operand=x_s_norm,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, kd, kh, kw, 1),
        window_strides=(1, stride[0], stride[1], stride[2], 1),
        padding=conv_padding,
    )
    pooled_norm = pooled_norm / (kd * kh * kw)  # Shape: (batch, out_D, out_H, out_W, 1)

    # Compute output norms
    conv_out_norm = jnp.linalg.norm(conv_out, axis=-1, keepdims=True)  # Shape: (batch, out_D, out_H, out_W, 1)

    # Rescale: out_s = conv_out · (pooled_input_norm / conv_out_norm)
    # Handle edge case: if conv_out_norm ≈ 0, use pooled_norm directly to avoid explosion
    is_near_zero = conv_out_norm < EPSILON
    scale = jnp.where(
        is_near_zero,
        1.0,  # Don't scale if output is near zero
        pooled_norm / jnp.maximum(conv_out_norm, EPSILON),
    )
    out_s = cast(Array, conv_out * scale)  # Shape: (batch, out_D, out_H, out_W, out_channels-1)

    # Reconstruct time component: out_t = √(‖out_s‖² + 1/c)
    out_s_sqnorm = jnp.sum(out_s**2, axis=-1, keepdims=True)  # Shape: (batch, out_D, out_H, out_W, 1)
    out_t = jnp.sqrt(jnp.maximum(out_s_sqnorm + 1.0 / c, MIN_NORM))  # Shape: (batch, out_D, out_H, out_W, 1)

    # Concatenate time and spatial components
    result = jnp.concatenate([out_t, out_s], axis=-1)  # Shape: (batch, out_D, out_H, out_W, out_channels)

    return result


class LorentzConv2D(nnx.Module):
    """Lorentz Convolution Layer for 2D inputs implementing Eq. 7:

        out = LorentzBoost(DistanceRescaling(RotationConvolution(x)))

    This is an alternative to HypConv2DHyperboloid that uses rotation convolution
    with distance rescaling and Lorentz boost instead of HCat + HypLinear.

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
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold')
    use_distance_rescaling : bool
        Whether to apply distance rescaling (default: True)
    use_boost : bool
        Whether to apply Lorentz boost (default: True)
    dtype : jnp.dtype
        Data type for weights (default: jnp.float32)

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (padding, input_space)
        are treated as static and will be baked into the compiled function.

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr.
        "Fully hyperbolic convolutional neural networks for computer vision."
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
        padding: Literal["SAME", "VALID"] = "SAME",
        input_space: str = "manifold",
        use_distance_rescaling: bool = True,
        use_boost: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_space = input_space
        self.padding: Literal["SAME", "VALID"] = padding
        self.use_distance_rescaling = use_distance_rescaling
        self.use_boost = use_boost
        self.dtype = dtype

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

        # Initialize convolution weights (spatial only: in_channels-1 -> out_channels-1)
        spatial_in = in_channels - 1
        spatial_out = out_channels - 1
        kernel_h, kernel_w = self.kernel_size

        # Check Algorithm 3 norm-preserving condition
        kernel_size_total = kernel_h * kernel_w
        if not _check_norm_preserving_condition(spatial_in, spatial_out, kernel_size_total):
            logger.warning(
                f"LorentzConv2D: Algorithm 3 condition violated "
                f"({kernel_size_total} * {spatial_in} = {kernel_size_total * spatial_in} > {spatial_out}). "
                f"Norm-preserving property does not hold mathematically."
            )

        # Initialize weights with Xavier/Glorot initialization
        # Shape: (out_channels-1, in_channels-1, kh, kw)
        fan_in = spatial_in * kernel_h * kernel_w
        fan_out = spatial_out * kernel_h * kernel_w
        # Use float32 for std computation, then cast to dtype
        std = float(jnp.sqrt(2.0 / (fan_in + fan_out)))

        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (spatial_out, spatial_in, kernel_h, kernel_w), dtype=dtype) * std
        )

        # Initialize boost velocity (if using)
        if use_boost:
            # Small random initialization for velocity
            self.boost_velocity = nnx.Param(jax.random.normal(rngs.params(), (spatial_out,), dtype=dtype) * 0.01)

    def __call__(
        self,
        x: Float[Array, "batch height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_height out_width out_channels"]:
        """Forward pass through the Lorentz convolutional layer.

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
        # 0. Map to manifold if tangent input
        if self.input_space == "tangent":
            # Flatten batch and spatial dims: (B*H*W, C)
            x_flat = x.reshape(-1, x.shape[-1])
            # Apply single vmap over the list of vectors
            x_mapped = jax.vmap(self.manifold.expmap_0, in_axes=(0, None))(x_flat, c)
            # Reshape back to (B, H, W, C)
            x = x_mapped.reshape(x.shape)

        # 1. Rotation Convolution (norm-preserving)
        y = rotation_conv_2d(x, self.weight.value, c, self.stride, self.padding)

        # 2. Distance Rescaling
        if self.use_distance_rescaling:
            # Flatten batch and spatial dims for vmap
            batch, out_h, out_w, out_c = y.shape
            y_flat = y.reshape(-1, out_c)
            y_rescaled = jax.vmap(self.manifold.distance_rescale, in_axes=(0, None, None, None))(y_flat, c, 2000.0, 1.0)
            y = y_rescaled.reshape(batch, out_h, out_w, out_c)

        # 3. Lorentz Boost
        if self.use_boost:
            # Flatten batch and spatial dims for vmap
            batch, out_h, out_w, out_c = y.shape
            y_flat = y.reshape(-1, out_c)
            y_boosted = jax.vmap(self.manifold.lorentz_boost, in_axes=(0, None, None))(y_flat, self.boost_velocity.value, c)
            y = y_boosted.reshape(batch, out_h, out_w, out_c)

        return y


class LorentzConv3D(nnx.Module):
    """Lorentz Convolution Layer for 3D inputs implementing Eq. 7:

        out = LorentzBoost(DistanceRescaling(RotationConvolution(x)))

    3D extension of LorentzConv2D for volumetric data.

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
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold')
    use_distance_rescaling : bool
        Whether to apply distance rescaling (default: True)
    use_boost : bool
        Whether to apply Lorentz boost (default: True)

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (padding, input_space)
        are treated as static and will be baked into the compiled function.

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr.
        "Fully hyperbolic convolutional neural networks for computer vision."
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
        padding: Literal["SAME", "VALID"] = "SAME",
        input_space: str = "manifold",
        use_distance_rescaling: bool = True,
        use_boost: bool = True,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_space = input_space
        self.padding: Literal["SAME", "VALID"] = padding
        self.use_distance_rescaling = use_distance_rescaling
        self.use_boost = use_boost

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

        # Initialize convolution weights (spatial only: in_channels-1 -> out_channels-1)
        spatial_in = in_channels - 1
        spatial_out = out_channels - 1
        kernel_d, kernel_h, kernel_w = self.kernel_size

        # Check Algorithm 3 norm-preserving condition
        kernel_size_total = kernel_d * kernel_h * kernel_w
        if not _check_norm_preserving_condition(spatial_in, spatial_out, kernel_size_total):
            logger.warning(
                f"LorentzConv3D: Algorithm 3 condition violated "
                f"({kernel_size_total} * {spatial_in} = {kernel_size_total * spatial_in} > {spatial_out}). "
                f"Norm-preserving property does not hold mathematically."
            )

        # Initialize weights with Xavier/Glorot initialization
        # Shape: (out_channels-1, in_channels-1, kd, kh, kw)
        fan_in = spatial_in * kernel_d * kernel_h * kernel_w
        fan_out = spatial_out * kernel_d * kernel_h * kernel_w
        # Use float32 for std computation, then cast to dtype
        std = float(jnp.sqrt(2.0 / (fan_in + fan_out)))

        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (spatial_out, spatial_in, kernel_d, kernel_h, kernel_w)) * std
        )

        # Initialize boost velocity (if using)
        if use_boost:
            # Small random initialization for velocity
            self.boost_velocity = nnx.Param(jax.random.normal(rngs.params(), (spatial_out,)) * 0.01)

    def __call__(
        self,
        x: Float[Array, "batch depth height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_depth out_height out_width out_channels"]:
        """Forward pass through the 3D Lorentz convolutional layer.

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
        # 0. Map to manifold if tangent input
        if self.input_space == "tangent":
            # Flatten batch and spatial dims: (B*D*H*W, C)
            x_flat = x.reshape(-1, x.shape[-1])
            # Apply single vmap over the list of vectors
            x_mapped = jax.vmap(self.manifold.expmap_0, in_axes=(0, None))(x_flat, c)
            # Reshape back to (B, D, H, W, C)
            x = x_mapped.reshape(x.shape)

        # 1. Rotation Convolution (norm-preserving)
        y = rotation_conv_3d(x, self.weight.value, c, self.stride, self.padding)

        # 2. Distance Rescaling
        if self.use_distance_rescaling:
            # Flatten batch and spatial dims for vmap
            batch, out_d, out_h, out_w, out_c = y.shape
            y_flat = y.reshape(-1, out_c)
            y_rescaled = jax.vmap(self.manifold.distance_rescale, in_axes=(0, None, None, None))(y_flat, c, 2000.0, 1.0)
            y = y_rescaled.reshape(batch, out_d, out_h, out_w, out_c)

        # 3. Lorentz Boost
        if self.use_boost:
            # Flatten batch and spatial dims for vmap
            batch, out_d, out_h, out_w, out_c = y.shape
            y_flat = y.reshape(-1, out_c)
            y_boosted = jax.vmap(self.manifold.lorentz_boost, in_axes=(0, None, None))(y_flat, self.boost_velocity.value, c)
            y = y_boosted.reshape(batch, out_d, out_h, out_w, out_c)

        return y
