"""Hyperboloid linear layers for JAX/Flax NNX.

This module contains linear transformation layers for the hyperboloid manifold,
including the Hyperbolic Transformation Component (HTC) module from the Hypformer paper.

For the core HTC/HRC functions, see hyperboloid_core module.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from .hyperboloid_core import htc


def _validate_hyperboloid_manifold(manifold_module: Any) -> None:
    required_methods = ("expmap_0",)
    if not all(hasattr(manifold_module, method) for method in required_methods):
        raise TypeError(
            "manifold_module must be a class-based Hyperboloid manifold instance (e.g., hyperbolix.manifolds.Hyperboloid())."
        )


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
        _validate_hyperboloid_manifold(manifold_module)
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
    init_bound : float, optional
        Bound for uniform weight initialization. Weights are initialized from
        Uniform(-init_bound, init_bound). Small values keep initial outputs
        close to the hyperboloid origin for stable training (default: 0.02).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

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
        This layer uses small uniform initialization U(-0.02, 0.02) by default,
        matching the initialization used by FHNN/FHCNN layers. Standard deep learning
        initializations (Xavier, Lecun) produce weights that are too large for
        hyperbolic operations, causing gradient explosion and training instability.

    See Also
    --------
    hyperbolix.nn_layers.hyperboloid_core.htc : Core HTC function for functional transformations.
    HypLinearHyperboloidFHCNN : Alternative hyperbolic linear layer with sigmoid scaling.

    References
    ----------
    Hypformer paper (citation to be added)

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
        init_bound: float = 0.02,
        eps: float = 1e-7,
    ):
        # Small uniform initialization for hyperbolic stability
        # Standard initializations (Lecun, Xavier) are too large and cause gradient explosion
        self.kernel = nnx.Param(
            jax.random.uniform(rngs.params(), (in_features, out_features), minval=-init_bound, maxval=init_bound)
        )
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_features,)))
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
            out = z @ self.kernel[...]
            if self.bias is not None:
                out = out + self.bias[...]
            return out

        return htc(x, linear_fn, c_in, c_out, self.eps)
