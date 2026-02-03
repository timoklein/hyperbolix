"""Hyperboloid linear layers for JAX/Flax NNX."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


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
