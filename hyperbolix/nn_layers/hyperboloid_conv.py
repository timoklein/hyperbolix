"""Hyperboloid convolutional layers for JAX/Flax NNX.

Dimension key:
  B: batch size
  H: output height        W: output width
  C: channels (in/out)    K: kernel elements (kh*kw)
  A: ambient dimension (in_channels or hcat output dim)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds.hyperboloid import Hyperboloid

from ._helpers import as_pair, validate_hyperboloid_manifold
from .hyperboloid_core import extract_patches, hcat_ambient_dim, hrc
from .hyperboloid_linear import (
    _assert_v_max_safe,
    _fgg_linear_forward,
    _fgg_weight_init,
    _fhcnn_forward,
    _fhnn_forward,
    _get_effective_kernel,
    _hyperboloid_plfc_forward,
)


def _map_input_to_manifold(
    x: Float[Array, "batch height width in_channels"],
    manifold: Hyperboloid,
    input_space: str,
    c: float,
) -> Float[Array, "batch height width in_channels"]:
    """expmap_0 a (B, H, W, C) feature map onto the manifold if ``input_space == 'tangent'``.

    Shared by the conv ``__call__``s; ``input_space`` is a static Python attribute so
    the branch stays JIT-friendly (baked in at trace time).
    """
    if input_space == "tangent":
        x_flat_NC = x.reshape(-1, x.shape[-1])  # (B*H*W, C)
        x_flat_NC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(x_flat_NC, c)
        x = x_flat_NC.reshape(x.shape)  # (B, H, W, C)
    return x


class LorentzConv2D(nnx.Module):
    """
    Lorentz 2D Convolutional Layer using the Hyperbolic Layer (HL) approach.

    This layer applies convolution to the space-like components of Lorentzian
    vectors and reconstructs the time-like component to maintain the manifold
    constraint. This is equivalent to an HRC (Hyperbolic Regularization Component)
    wrapper around a standard Conv2D.

    Computation steps:
        1) Extract space-like components x_s from input x = [x_t, x_s]^T
        2) Apply Euclidean convolution: y_s = Conv2D(x_s)
        3) Reconstruct time component: y_t = sqrt(||y_s||^2 + 1/c)
        4) Return y = [y_t, y_s]^T

    Parameters
    ----------
    in_channels : int
        Number of input channels (ambient dimension, including time component)
    out_channels : int
        Number of output channels (ambient dimension, including time component)
    kernel_size : int or tuple[int, int]
        Size of the convolutional kernel
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    stride : int or tuple[int, int]
        Stride of the convolution (default: 1)
    padding : str or int or tuple
        Padding mode: 'SAME', 'VALID', or explicit padding (default: 'SAME')
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by the manifold's ``dtype``.

    Notes
    -----
    This implementation follows the Hyperbolic Layer (HL) approach from
    "Fully Hyperbolic Convolutional Neural Networks for Computer Vision".

    The layer operates only on space-like components, making it more
    computationally efficient than the HCat-based approach (HypConv2DHyperboloid),
    though it doesn't perform true hyperbolic convolution. Instead, it applies
    Euclidean operations to spatial components and reconstructs the time component.

    See Also
    --------
    hypformer.hrc : Core HRC function this layer is based on
    HypConv2DHyperboloid : Full hyperbolic convolution using HCat concatenation

    References
    ----------
    He, Neil, Menglin Yang, and Rex Ying. "Lorentzian residual neural networks."
    Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2025.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create Euclidean conv layer for space components only
        # in_channels - 1: skip time component at index 0
        # out_channels - 1: time will be reconstructed from constraint
        self.conv = nnx.Conv(
            in_features=in_channels - 1,
            out_features=out_channels - 1,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Float[Array, "batch height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_height out_width out_channels"]:
        """
        Forward pass through the Lorentz convolutional layer.

        This layer is a specific instance of the Hyperbolic Regularization Component (HRC)
        where the regularization function f_r is a 2D convolution. The HRC pattern:
        1. Extracts space components
        2. Applies Euclidean convolution
        3. Reconstructs time component using Lorentz constraint

        Parameters
        ----------
        x : Array of shape (batch, height, width, in_channels)
            Input feature map where x[..., 0] is time component and
            x[..., 1:] are space components on the Lorentz manifold
        c : float
            Manifold curvature parameter (default: 1.0)

        Returns
        -------
        out : Array of shape (batch, out_height, out_width, out_channels)
            Output feature map on the Lorentz manifold

        Notes
        -----
        This implementation uses the HRC function from hypformer.py, demonstrating that
        LorentzConv2D (from LResNet) and HRC (from Hypformer) are mathematically equivalent
        approaches to adapting Euclidean operations for hyperbolic geometry.
        """

        # Define convolution as the HRC regularization function f_r
        def conv_fn(x_space):
            return self.conv(x_space)

        # Apply HRC with curvature-preserving transformation (c_in = c_out = c)
        return hrc(x, conv_fn, c_in=c, c_out=c, eps=1e-8)


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
    manifold_module : object
        Class-based Hyperboloid manifold instance
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
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.

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
        manifold_module: Hyperboloid,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        input_space: str = "manifold",
        param_dtype: DTypeLike = jnp.float32,
    ):
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be either 'SAME' or 'VALID', got '{padding}'")
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration
        validate_hyperboloid_manifold(manifold_module, required_methods=("expmap_0", "hcat"))
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_space = input_space
        self.padding = padding

        self.kernel_size = as_pair(kernel_size)
        self.stride = as_pair(stride)

        # HCat output ambient dim for the linear layer
        hcat_out_ambient_dim = hcat_ambient_dim(in_channels, self.kernel_size)

        # Trainable parameters — owned directly for flat parameter paths
        bound = 0.02
        self.kernel = nnx.Param(
            jax.random.uniform(
                rngs.params(), (out_channels, hcat_out_ambient_dim), dtype=param_dtype, minval=-bound, maxval=bound
            )
        )
        self.bias = nnx.Param(jnp.zeros((1, out_channels), dtype=param_dtype))
        self.scale = 2.3  # not learnable (matches FHCNN default)

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
        x = _map_input_to_manifold(x, self.manifold, self.input_space, c)

        # Extract patches: (B, H, W, kh, kw, C)
        patches_BHWkhkwC = extract_patches(x, self.kernel_size, self.stride, self.padding, pad_mode="edge")
        batch, out_h, out_w, kh, kw, in_c = patches_BHWkhkwC.shape

        # Flatten batch+spatial for parallel processing: (B*H*W, K, C)
        patches_flat_NKC = patches_BHWkhkwC.reshape(-1, kh * kw, in_c)

        # HCat: (K, C) -> (hcat_dim,) per patch
        hcat_out_NA = jax.vmap(self.manifold.hcat, in_axes=(0, None))(patches_flat_NKC, c)  # (B*H*W, hcat_dim)

        # Linear: (hcat_dim,) -> (out_channels,)
        linear_out_NC = _fhcnn_forward(
            hcat_out_NA,
            self.kernel[...],
            self.bias[...],
            self.manifold,
            c,
            "manifold",
            None,
            False,
            self.scale,
            1e-5,
        )  # (B*H*W, out_channels)

        # Reshape back to spatial
        output_BHWC = linear_out_NC.reshape(batch, out_h, out_w, self.out_channels)

        return output_BHWC


class HypConv2DHyperboloidFHNN(nnx.Module):
    """Fully Hyperbolic Neural Networks 2D convolutional layer (Chen et al. 2021).

    Uses HCat (Lorentz direct concatenation) to combine receptive field points,
    then applies the FHNN linear transform (time-primary sigmoid parameterization)
    for channel mixing.

    Computation steps:
        1) Map input to manifold via expmap_0 if input_space="tangent"
        2) Extract receptive field (kernel_size x kernel_size) of hyperbolic points
        3) Apply HCat (Lorentz direct concatenation) to combine receptive field points
        4) Pass through FHNN linear (sigmoid time + spatial rescaling)

    Parameters
    ----------
    manifold_module : Hyperboloid
        Class-based Hyperboloid manifold instance
    in_channels : int
        Number of input channels (ambient dimension, including time component)
    out_channels : int
        Number of output channels (ambient dimension, including time component)
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
    init_scale : float
        Initial value for the learnable sigmoid scale (default: 2.3)
    eps : float
        Numerical stability epsilon (default: 1e-5)
    activation : callable or None
        Activation function to apply before the linear transformation (default: None).
    dropout_rate : float or None
        Dropout rate applied before the linear transformation (default: None).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (padding, input_space,
        activation) are treated as static and baked into the compiled function.

    See Also
    --------
    HypConv2DHyperboloid : Equivalent convolution using FHCNN linear instead of FHNN.
    HypLinearHyperboloidFHNN : The underlying FHNN linear layer.

    References
    ----------
    Weize Chen, et al. "Fully hyperbolic neural networks."
        arXiv preprint arXiv:2105.14686 (2021).
    """

    def __init__(
        self,
        manifold_module: Hyperboloid,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        input_space: str = "manifold",
        init_scale: float = 2.3,
        eps: float = 1e-5,
        activation: Callable[[Array], Array] | None = None,
        dropout_rate: float | None = None,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be either 'SAME' or 'VALID', got '{padding}'")
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration
        validate_hyperboloid_manifold(manifold_module, required_methods=("expmap_0", "hcat"))
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_space = input_space
        self.padding = padding
        self.eps = eps
        self.activation = activation

        self.kernel_size = as_pair(kernel_size)
        self.stride = as_pair(stride)

        # HCat output ambient dim
        hcat_out_ambient_dim = hcat_ambient_dim(in_channels, self.kernel_size)

        # FHNN weight init: U(-0.02, 0.02) with time column zeroed (tangent vectors at origin)
        bound = 0.02
        weight_init = jax.random.uniform(
            rngs.params(), (out_channels, hcat_out_ambient_dim), dtype=param_dtype, minval=-bound, maxval=bound
        )
        weight_init = weight_init.at[:, 0].set(0.0)
        self.kernel = nnx.Param(weight_init)
        self.bias = nnx.Param(jnp.zeros((1, out_channels), dtype=param_dtype))

        # Learnable scale for the sigmoid (always learnable in FHNN)
        self.scale = nnx.Param(jnp.array(init_scale, dtype=param_dtype))

        # Optional dropout
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: Float[Array, "batch height width in_channels"],
        c: float = 1.0,
        deterministic: bool = True,
    ) -> Float[Array, "batch out_height out_width out_channels"]:
        """Forward pass through the FHNN hyperbolic convolutional layer.

        Parameters
        ----------
        x : Array of shape (batch, height, width, in_channels)
            Input feature map where each pixel is a point on the Hyperboloid manifold
        c : float
            Manifold curvature (default: 1.0)
        deterministic : bool
            If True, dropout is disabled (default: True).

        Returns
        -------
        out : Array of shape (batch, out_height, out_width, out_channels)
            Output feature map on the Hyperboloid manifold
        """
        # Map to manifold if needed (static branch - JIT friendly)
        x = _map_input_to_manifold(x, self.manifold, self.input_space, c)

        # Extract patches: (B, H, W, kh, kw, C)
        patches_BHWkhkwC = extract_patches(x, self.kernel_size, self.stride, self.padding, pad_mode="edge")
        batch, out_h, out_w, kh, kw, in_c = patches_BHWkhkwC.shape

        # Flatten batch+spatial for parallel processing: (B*H*W, K, C)
        patches_flat_NKC = patches_BHWkhkwC.reshape(-1, kh * kw, in_c)

        # HCat: (K, C) -> (hcat_dim,) per patch
        hcat_out_NA = jax.vmap(self.manifold.hcat, in_axes=(0, None))(patches_flat_NKC, c)  # (B*H*W, hcat_dim)

        # Build dropout closure for the pure function
        dropout_module = self.dropout
        if dropout_module is not None:
            dropout_fn = lambda z: dropout_module(z, deterministic=deterministic)  # noqa: E731
        else:
            dropout_fn = None

        # FHNN linear: (hcat_dim,) -> (out_channels,)
        linear_out_NC = _fhnn_forward(
            hcat_out_NA,
            self.kernel[...],
            self.bias[...],
            self.manifold,
            c,
            "manifold",  # HCat output is already on manifold
            self.activation,
            dropout_fn,
            self.scale[...],
            self.eps,
        )  # (B*H*W, out_channels)

        # Reshape back to spatial
        return linear_out_NC.reshape(batch, out_h, out_w, self.out_channels)


class FGGConv2D(nnx.Module):
    """Fast and Geometrically Grounded Lorentz 2D convolutional layer.

    Uses HCat (Lorentz direct concatenation) to combine receptive field points,
    then applies FGGLinear for the channel mixing. This matches the reference
    implementation pattern from Klis et al. 2026.

    Computation steps:
        1) Extract receptive field patches, pad with manifold origin if needed
        2) Apply HCat (Lorentz direct concatenation) to combine patch points
        3) Pass through FGGLinear for channel transformation

    Parameters
    ----------
    manifold_module : Hyperboloid
        Class-based Hyperboloid manifold instance.
    in_channels : int
        Input ambient channels (D_in + 1), including time component.
    out_channels : int
        Output ambient channels (D_out + 1), including time component.
    kernel_size : int or tuple[int, int]
        Size of the convolutional kernel.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    stride : int or tuple[int, int], optional
        Stride of the convolution (default: 1).
    padding : str, optional
        Padding mode: ``"SAME"`` or ``"VALID"`` (default: ``"SAME"``).
    pad_mode : str, optional
        How to fill padding pixels: ``"origin"`` fills with the manifold
        origin ``(sqrt(1/c), 0, ..., 0)`` (matching reference), ``"edge"``
        replicates border values (default: ``"origin"``).
    activation : Callable or None, optional
        Euclidean activation for the FGGLinear (default: None).
    reset_params : str, optional
        Weight init for FGGLinear: ``"eye"``, ``"xavier"``, ``"kaiming"``,
        ``"lorentz_kaiming"``, or ``"mlr"`` (default: ``"lorentz_kaiming"``).
    use_weight_norm : bool, optional
        Weight normalization in FGGLinear (default: False).
    init_bias : float, optional
        Initial bias for FGGLinear (default: 0.5).
    eps : float, optional
        Numerical stability floor (default: 1e-7).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.

    References
    ----------
    Klis et al. "Fast and Geometrically Grounded Lorentz Neural Networks" (2026).
    """

    def __init__(
        self,
        manifold_module: Hyperboloid,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        pad_mode: str = "origin",
        activation: Callable | None = None,
        reset_params: str = "lorentz_kaiming",
        use_weight_norm: bool = False,
        init_bias: float = 0.5,
        eps: float = 1e-7,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if padding not in ("SAME", "VALID"):
            raise ValueError(f"padding must be either 'SAME' or 'VALID', got '{padding}'")
        if pad_mode not in ("origin", "edge"):
            raise ValueError(f"pad_mode must be 'origin' or 'edge', got '{pad_mode}'")

        validate_hyperboloid_manifold(manifold_module, required_methods=("hcat",))
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.pad_mode = pad_mode
        self.eps = eps

        self.kernel_size = as_pair(kernel_size)
        self.stride = as_pair(stride)

        # HCat output ambient dim
        hcat_out_ambient = hcat_ambient_dim(in_channels, self.kernel_size)

        # Trainable parameters — owned directly for flat parameter paths
        in_spatial = hcat_out_ambient - 1  # I
        out_spatial = out_channels - 1  # O

        self.activation = activation
        self.use_weight_norm = use_weight_norm

        # Euclidean weight U (I, O); std from ambient dims (hcat_out_ambient, out_channels).
        U_init = _fgg_weight_init(
            rngs.params(), in_spatial, out_spatial, hcat_out_ambient, out_channels, reset_params, param_dtype
        )

        # Weight normalization: decompose kernel = softplus(kernel_scale) * kernel_dir / ||kernel_dir||
        if use_weight_norm:
            self.kernel_dir = nnx.Param(U_init)  # (I, O) direction
            g_init_val = jnp.sqrt(1.0 / (hcat_out_ambient + out_channels))
            self.kernel_scale = nnx.Param(jnp.full((out_spatial,), g_init_val, dtype=param_dtype))  # (O,)
        else:
            self.kernel = nnx.Param(U_init)  # (I, O)

        self.bias = nnx.Param(jnp.full((out_spatial,), init_bias, dtype=param_dtype))  # (O,)

    def __call__(
        self,
        x: Float[Array, "batch height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_height out_width out_channels"]:
        """Forward pass through the FGG convolutional layer.

        Parameters
        ----------
        x : Array, shape (B, H, W, in_channels)
            Input feature map on the hyperboloid.
        c : float, optional
            Curvature parameter (default: 1.0).

        Returns
        -------
        out : Array, shape (B, H', W', out_channels)
            Output feature map on the hyperboloid.
        """
        # Extract patches: (B, H', W', kh, kw, C)
        patches = extract_patches(x, self.kernel_size, self.stride, self.padding, self.pad_mode, c)
        batch, out_h, out_w, kh, kw, in_c = patches.shape

        # Flatten batch+spatial: (B*H'*W', K, C) where K = kh*kw
        patches_flat_NKC = patches.reshape(-1, kh * kw, in_c)

        # HCat: (K, C) -> (hcat_dim,) per patch
        hcat_out_NA = jax.vmap(self.manifold.hcat, in_axes=(0, None))(patches_flat_NKC, c)

        # FGG forward: (hcat_dim,) -> (out_channels,)
        U_IO = _get_effective_kernel(
            getattr(self, "kernel", None),
            getattr(self, "kernel_dir", None),
            getattr(self, "kernel_scale", None),
            self.use_weight_norm,
            self.eps,
        )
        linear_out_NC = _fgg_linear_forward(hcat_out_NA, U_IO, self.bias[...], c, self.activation, self.eps)

        # Reshape back to spatial
        return linear_out_NC.reshape(batch, out_h, out_w, self.out_channels)


class HypConv2DHyperboloidILNN(nnx.Module):
    """
    Intrinsic Lorentz Neural Network (ILNN) 2D convolutional layer (Hyperboloid model).

    Implements the Lorentz convolution of Shi et al. 2026, Eq. (11):
    ``y = PLFC(LogCat({x_patch}))``. Receptive-field points are combined with the
    log-radius-preserving concatenation (LogCat) and mixed across channels by the
    PLFC linear transform (MLR scores -> sinh diffeomorphism -> time
    reconstruction). This extends the HNN++ linearized-kernel convolution
    (Shimizu et al. 2020) by replacing the Euclidean FC and naive concatenation
    with their intrinsic Lorentz counterparts. Formerly named
    ``HypConv2DHyperboloidPP``.

    Computation steps:
        1) Extract receptive field (kernel_size x kernel_size) of hyperbolic points,
           padding with the manifold origin if needed
        2) Apply LogCat (log-radius-preserving concatenation) to combine receptive field points
        3) Pass through PLFC linear (MLR scores -> guard -> sinh -> time reconstruction)
        4) Optionally add an intrinsic gyro-bias ``y <- y (+) exp_0([0, b])``

    Parameters
    ----------
    manifold_module : Hyperboloid
        Class-based Hyperboloid manifold instance
    in_channels : int
        Number of input channels (ambient dimension, including time component)
    out_channels : int
        Number of output channels (ambient dimension, including time component)
    kernel_size : int or tuple[int, int]
        Size of the convolutional kernel
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    stride : int or tuple[int, int]
        Stride of the convolution (default: 1)
    padding : str
        Padding mode, either 'SAME' or 'VALID' (default: 'SAME')
    pad_mode : str
        How to fill padding pixels for 'SAME' padding: ``"origin"`` fills with the
        manifold origin ``(sqrt(1/c), 0, ..., 0)`` (matching the Shi et al. 2026
        reference, which clamps zero-padded times up to ``sqrt(1/c)``), ``"edge"``
        replicates border values (default: ``"origin"``).
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    clamping_factor : float
        Clamping factor for the multinomial linear regression output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the multinomial linear regression output (default: 50.0)
    v_max : float
        Output-side guard: the sinh argument ``sqrt(c)*v`` is smooth-clamped to
        ``±v_max``, bounding the output spatial norm by ``sinh(v_max)/sqrt(c)``
        (default: 10.0, matching the Shi et al. 2026 reference implementation).
    use_gyro_bias : bool
        If True, add a learnable intrinsic bias via Lorentz gyroaddition,
        ``y <- y (+) exp_0([0, b])`` with ``b`` initialized to zero — the gyrogroup
        identity, so the bias is a no-op at initialization (default: False).
    kernel_init_std : float
        Standard deviation of the normal kernel init (default: 0.02, the Shi et al.
        2026 PLFC reference value; the reference conv defers to the PLFC init).
        Use 1.0 to recover the previous HNN++-style init (Shimizu et al. 2020);
        note that large kernels push the pre-guard scores into the ``v_max``
        saturation regime.
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).
        Compute precision of manifold operations is set by ``manifold.dtype``.

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (padding, pad_mode,
        input_space, clamping_factor, smoothing_factor, v_max) are treated as static and baked
        into the compiled function.

    See Also
    --------
    HypLinearHyperboloidPLFC : The underlying PLFC linear layer.
    hyperbolix.manifolds.hyperboloid.Hyperboloid.log_radius_concat : The LogCat operation.

    References
    ----------
    Xianglong Shi, Ziheng Chen, Yunhan Jiang, and Nicu Sebe. "Intrinsic Lorentz Neural Network."
        ICLR 2026, arXiv:2602.23981 (Sec. 4.3: Lorentz convolution, log-radius concatenation).
    Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
        arXiv preprint arXiv:2006.08210 (2020).
    """

    def __init__(
        self,
        manifold_module: Hyperboloid,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        pad_mode: str = "origin",
        input_space: str = "manifold",
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        v_max: float = 10.0,
        use_gyro_bias: bool = False,
        kernel_init_std: float = 0.02,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"padding must be either 'SAME' or 'VALID', got '{padding}'")
        if pad_mode not in ("origin", "edge"):
            raise ValueError(f"pad_mode must be 'origin' or 'edge', got '{pad_mode}'")
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration
        required_methods = ("expmap_0", "compute_mlr", "log_radius_concat")
        if use_gyro_bias:
            required_methods = ("expmap_0", "compute_mlr", "log_radius_concat", "embed_spatial_0", "addition")
        validate_hyperboloid_manifold(manifold_module, required_methods=required_methods)
        self.manifold = manifold_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_space = input_space
        self.padding = padding
        self.pad_mode = pad_mode
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        _assert_v_max_safe(v_max)
        self.v_max = v_max

        self.kernel_size = as_pair(kernel_size)
        self.stride = as_pair(stride)

        # LogCat output ambient dim (same as HCat)
        logcat_out_ambient_dim = hcat_ambient_dim(in_channels, self.kernel_size)

        # Trainable parameters — small normal init (Shi et al. 2026 PLFC reference;
        # the reference conv defers to the PLFC reset_parameters)
        out_spatial = out_channels - 1
        logcat_spatial = logcat_out_ambient_dim - 1
        kernel_init = kernel_init_std * jax.random.normal(rngs.params(), (out_spatial, logcat_spatial), dtype=param_dtype)
        self.kernel = nnx.Param(kernel_init)
        self.bias = nnx.Param(jnp.zeros((out_spatial, 1), dtype=param_dtype))

        # Gyro-bias: spatial tangent vector at the origin, zero-initialized
        if use_gyro_bias:
            self.gyro_bias = nnx.Param(jnp.zeros((out_spatial,), dtype=param_dtype))
        else:
            self.gyro_bias = None

    def __call__(
        self,
        x: Float[Array, "batch height width in_channels"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_height out_width out_channels"]:
        """
        Forward pass through the HNN++ hyperboloid convolutional layer.

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
        x = _map_input_to_manifold(x, self.manifold, self.input_space, c)

        # Extract patches: (B, H', W', kh, kw, C)
        patches_BHWkhkwC = extract_patches(x, self.kernel_size, self.stride, self.padding, self.pad_mode, c)
        batch, out_h, out_w, kh, kw, in_c = patches_BHWkhkwC.shape

        # Flatten batch+spatial for parallel processing: (B*H'*W', K, C)
        patches_flat_NKC = patches_BHWkhkwC.reshape(-1, kh * kw, in_c)

        # LogCat: (K, C) -> (logcat_dim,) per patch — log-radius-preserving concatenation (Shi et al. 2026, Eq. 11)
        logcat_out_NA = jax.vmap(self.manifold.log_radius_concat, in_axes=(0, None))(patches_flat_NKC, c)

        # PLFC linear: (logcat_dim,) -> (out_channels,)
        linear_out_NC = _hyperboloid_plfc_forward(
            logcat_out_NA,
            self.kernel[...],
            self.bias[...],
            self.gyro_bias[...] if self.gyro_bias is not None else None,
            self.manifold,
            c,
            "manifold",  # LogCat output is already on manifold
            self.clamping_factor,
            self.smoothing_factor,
            self.v_max,
        )  # (B*H'*W', out_channels)

        # Reshape back to spatial
        return linear_out_NC.reshape(batch, out_h, out_w, self.out_channels)
