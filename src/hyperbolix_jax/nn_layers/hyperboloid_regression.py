"""Hyperboloid regression layers for JAX/Flax NNX."""

from typing import Any

import jax
from flax import nnx
from jaxtyping import Array, Float

from .helpers import compute_mlr_hyperboloid


class HypRegressionHyperboloid(nnx.Module):
    """
    Fully Hyperbolic Convolutional Neural Networks multinomial linear regression layer (Hyperboloid model).

    Computation steps:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

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
    clamping_factor : float
        Clamping factor for the multinomial linear regression output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the multinomial linear regression output (default: 50.0)
    backproject : bool
        Whether to project results back to the manifold (default: True).
        Note: This is a static configuration - changing it after initialization requires recompilation.

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space, backproject,
        clamping_factor, smoothing_factor) are treated as static and will be baked into the compiled function.

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
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        backproject: bool = True,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        self.backproject = backproject

        # Trainable parameters
        # weight lies in the tangent space of the Hyperboloid origin, so the time coordinate along axis is zero
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim - 1)))
        # Scalar bias (initialized to small random values)
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (out_dim, 1)) * 0.01)

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the hyperbolic regression layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor where the hyperbolic_axis is last
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Multinomial linear regression scores
        """
        # Map to manifold if needed (static branch - JIT friendly)
        if self.input_space == "tangent":
            x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None, None), out_axes=0)(x, c, self.backproject)

        # Compute multinomial linear regression
        res = compute_mlr_hyperboloid(
            x,
            self.weight,
            self.bias,
            c,
            self.clamping_factor,
            self.smoothing_factor,
        )

        return res
