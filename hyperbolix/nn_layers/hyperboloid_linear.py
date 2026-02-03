"""Hyperboloid linear layers for JAX/Flax NNX."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


class HypLinearHyperboloid(nnx.Module):
    """
    Hyperbolic Graph Convolutional Neural Networks fully connected layer (Hyperboloid model).

    Computation steps:
        0) Project the input tensor to the tangent space (optional)
        1) Perform matrix vector multiplication in the tangent space at the origin.
        2) Map the result to the manifold.
        3) Add the manifold bias to the result by means of the parallel transport and exponential map.

    Parameters
    ----------
    manifold_module : module
        The Hyperboloid manifold module
    in_dim : int
        Dimension of the input space
    out_dim : int
        Dimension of the output space
    rngs : nnx.Rngs
        Random number generators for parameter initialization
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space)
        are treated as static and will be baked into the compiled function.

    See Also
    --------
    htc : Hyperbolic Transformation Component that generalizes linear transformations
        with curvature change support. Uses a different mathematical approach (direct
        constraint reconstruction vs. exp/log maps).
    HTCLinear : Module wrapper for htc with learnable linear transformation.

    References
    ----------
    Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
        Advances in neural information processing systems 32 (2019).
    """

    def __init__(
        self,
        manifold_module: Any,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space

        # Trainable parameters
        # Note: We only set the space coordinates since both parameters lie in the tangent space at the Hyperboloid origin
        # The time coordinate (first coordinate) is zero and omitted
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim - 1, in_dim - 1)))
        # Initialize bias to small random values (zeros would be fine but small values are more general)
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (1, out_dim - 1)) * 0.01)

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
            Input tensor where the hyperbolic_axis is last. x.shape[-1] must equal self.in_dim.
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Hyperboloid manifold

        Note
        ----
        The weight lies in the tangent space at the Hyperboloid origin, so its time coordinate is zero and omitted.
        """
        # Map to tangent space if needed (static branch - JIT friendly)
        if self.input_space == "manifold":
            x = jax.vmap(self.manifold.logmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Matrix-Vector multiplication in the tangent space at the Hyperboloid origin
        # Extract space coordinates (all except first time coordinate)
        x_rem = x[:, 1:]  # (batch, in_dim - 1)
        # Matrix multiply: (batch, in_dim-1) @ (in_dim-1, out_dim-1) -> (batch, out_dim-1)
        x = jnp.einsum("bi,oi->bo", x_rem, self.weight)

        # Since the result needs to lie in the tangent space at the origin we must concatenate the time coordinate back
        x = jnp.concatenate([jnp.zeros_like(x[:, :1]), x], axis=-1)  # (batch, out_dim)

        # Map back to manifold
        x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)  # (batch, out_dim)

        # Bias addition via parallel transport and exponential map
        # Concatenate zero time coordinate to bias
        bias = jnp.concatenate([jnp.zeros_like(self.bias[...][:, :1]), self.bias[...]], axis=-1)  # (1, out_dim)
        bias = bias.squeeze(0)  # (out_dim,)

        # Parallel transport bias from origin to each x (vmap over batch)
        pt_bias = jax.vmap(self.manifold.ptransp_0, in_axes=(None, 0, None), out_axes=0)(bias, x, c)  # (batch, out_dim)

        # Add transported bias via exponential map (vmap over batch)
        res = jax.vmap(self.manifold.expmap, in_axes=(0, 0, None), out_axes=0)(pt_bias, x, c)  # (batch, out_dim)

        return res


class HypLinearHyperboloidFHNN(nnx.Module):
    """
    Fully Hyperbolic Neural Networks fully connected layer (Hyperboloid model).

    Computation steps:
        0) Project the input tensor to the manifold (optional)
        1) Apply activation and dropout (optional)
        2) Compute the time coordinate of the output via a scaled sigmoid of the weight and biases transformed
           time coordinate of the input or the result of the previous step.
        3) Compute the space coordinates of the output and rescale it such that the result lies on the manifold.

    Parameters
    ----------
    manifold_module : module
        The Hyperboloid manifold module
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
        Whether the scale parameter should be learnable (default: True)
    eps : float
        Small value to ensure that the time coordinate is bigger than 1/sqrt(c) (default: 1e-5)
    activation : callable or None
        Activation function to apply before the linear transformation (default: None).
        Note: This is a static configuration - changing it after initialization requires recompilation.
    dropout_rate : float or None
        Dropout rate to apply before the activation or linear transformation (default: None)
    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space, activation)
        are treated as static and will be baked into the compiled function. The dropout layer handles train/eval
        mode switching internally in a JIT-compatible way.

    See Also
    --------
    htc : Hyperbolic Transformation Component with curvature change support. Unlike FHNN,
        htc uses constraint-based time reconstruction rather than learned sigmoid scaling.
    HTCLinear : Module wrapper for htc with learnable linear transformation.

    References
    ----------
    Weize Chen, et al. "Fully hyperbolic neural networks."
        arXiv preprint arXiv:2105.14686 (2021).
    """

    def __init__(
        self,
        manifold_module: Any,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        init_scale: float = 2.3,
        learnable_scale: bool = True,
        eps: float = 1e-5,
        activation: Callable[[Array], Array] | None = None,
        dropout_rate: float | None = None,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.eps = eps
        self.activation = activation

        # Trainable parameters
        # FHNN initializes the weights as tangent vectors w.r.t. the Hyperboloid origin
        bound = 0.02
        weight_init = jax.random.uniform(rngs.params(), (out_dim, in_dim), minval=-bound, maxval=bound)
        # Set time coordinate to zero
        weight_init = weight_init.at[:, 0].set(0.0)
        self.weight = nnx.Param(weight_init)
        self.bias = nnx.Param(jnp.zeros((1, out_dim)))

        # Scale parameter for sigmoid
        if learnable_scale:
            self.scale = nnx.Param(jnp.array(init_scale))
        else:
            # For non-learnable scale, store as regular Python float (static)
            self.scale = init_scale

        # Dropout layer (handles train/eval mode internally)
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the FHNN hyperbolic linear layer.

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
        # Map to manifold if needed (static branch - JIT friendly)
        if self.input_space == "tangent":
            x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Apply activation if provided (static branch - JIT friendly)
        if self.activation is not None:
            x = self.activation(x)

        # Apply dropout if provided (static branch - JIT friendly)
        if self.dropout is not None:
            x = self.dropout(x)

        # Linear transformation: (batch, in_dim) @ (in_dim, out_dim)^T + (1, out_dim)
        x = jnp.einsum("bi,oi->bo", x, self.weight) + self.bias  # (batch, out_dim)

        # Extract time and space coordinates
        x0 = x[:, 0:1]  # (batch, 1)
        x_rem = x[:, 1:]  # (batch, out_dim - 1)

        # Compute time coordinate via scaled sigmoid (ensure scale is positive)
        # Handle both learnable (Param) and non-learnable (float) scale
        scale_val = self.scale[...] if isinstance(self.scale, nnx.Param) else self.scale
        res0 = jnp.exp(scale_val) * jax.nn.sigmoid(x0) + 1 / jnp.sqrt(c) + self.eps  # (batch, 1)

        # Compute space coordinates scaling factor
        x_rem_norm = jnp.linalg.norm(x_rem, ord=2, axis=-1, keepdims=True)  # (batch, 1)
        scale = jnp.sqrt(res0**2 - 1 / c) / x_rem_norm  # (batch, 1)
        res_rem = scale * x_rem  # (batch, out_dim - 1)

        # Concatenate time and space coordinates
        res = jnp.concatenate([res0, res_rem], axis=-1)  # (batch, out_dim)

        return res


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
    manifold_module : module
        The Hyperboloid manifold module
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
        manifold_module: Any,
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
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.eps = eps
        self.activation = activation
        self.normalize = normalize

        # Trainable parameters
        bound = 0.02
        weight_init = jax.random.uniform(rngs.params(), (out_dim, in_dim), minval=-bound, maxval=bound)
        self.weight = nnx.Param(weight_init)
        self.bias = nnx.Param(jnp.zeros((1, out_dim)))

        # Scale parameter for sigmoid
        if learnable_scale:
            self.scale = nnx.Param(jnp.array(init_scale))
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
        # Map to manifold if needed (static branch - JIT friendly)
        if self.input_space == "tangent":
            x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Apply activation if provided (static branch - JIT friendly)
        if self.activation is not None:
            x = self.activation(x)

        # Linear transformation
        x = jnp.einsum("bi,oi->bo", x, self.weight) + self.bias  # (batch, out_dim)

        # Extract time and space coordinates
        x0 = x[:, 0:1]  # (batch, 1)
        x_rem = x[:, 1:]  # (batch, out_dim - 1)

        # Static branch - JIT friendly
        if self.normalize:
            # Normalize space coordinates
            x_rem_norm = jnp.linalg.norm(x_rem, ord=2, axis=-1, keepdims=True)  # (batch, 1)

            # Ensure scale is positive (handle both learnable and non-learnable scale)
            scale_val = self.scale[...] if isinstance(self.scale, nnx.Param) else self.scale
            scale = jnp.exp(scale_val) * jax.nn.sigmoid(x0)  # (batch, 1)

            # Compute time coordinate
            res0 = jnp.sqrt(scale**2 + 1 / c + self.eps)  # (batch, 1)

            # Compute normalized space coordinates
            res_rem = scale * x_rem / x_rem_norm  # (batch, out_dim - 1)

            res = jnp.concatenate([res0, res_rem], axis=-1)  # (batch, out_dim)

            # Cast vectors with small space norm to the origin
            # Create origin point
            origin_time = jnp.sqrt(1 / c) * jnp.ones_like(res0)
            origin_space = jnp.zeros_like(res_rem)
            origin = jnp.concatenate([origin_time, origin_space], axis=-1)

            # Apply mask where x_rem_norm is very small (data-dependent control flow - JIT compatible with jnp.where)
            mask = x_rem_norm <= 1e-5
            res = jnp.where(mask, origin, res)
        else:
            # Compute the time component from the space component and concatenate
            res0 = jnp.sqrt(jnp.sum(x_rem**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
            res = jnp.concatenate([res0, x_rem], axis=-1)  # (batch, out_dim)

        return res
