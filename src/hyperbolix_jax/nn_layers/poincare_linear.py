"""Poincaré ball linear layers for JAX/Flax NNX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from ..utils.math_utils import sinh
from .helpers import compute_mlr_poincare_pp


class HypLinearPoincare(nnx.Module):
    """
    Hyperbolic Neural Networks fully connected layer (Poincaré ball model).

    Computation steps:
        0) Project the input tensor to the tangent space (optional)
        1) Perform matrix vector multiplication in the tangent space at the origin.
        2) Map the result to the manifold.
        3) Add the manifold bias to the result.

    Parameters
    ----------
    manifold_module : module
        The PoincareBall manifold module
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
        are treated as static and will be baked into the compiled function. Changing these values after
        JIT compilation will trigger automatic recompilation.

    References
    ----------
    Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
        Advances in neural information processing systems 31 (2018).
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
        # Tangent space weight (Euclidean)
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)))
        # Manifold bias (initialized to small random values to avoid gradient issues at origin)
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (1, out_dim)) * 0.01)

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
            Input tensor where the hyperbolic_axis is last
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Poincaré ball manifold
        """
        # Project bias to manifold (bias is (1, out_dim), squeeze to (out_dim,))
        bias = self.manifold.proj(self.bias.squeeze(0), c)

        # Map to tangent space if needed (static branch - JIT friendly)
        if self.input_space == "manifold":
            x = jax.vmap(self.manifold.logmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Matrix-vector multiplication in tangent space at origin
        # (batch, in_dim) @ (in_dim, out_dim) -> (batch, out_dim)
        x = jnp.einsum("bi,oi->bo", x, self.weight)

        # Map back to manifold
        x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Manifold bias addition (Möbius addition for Poincaré)
        # Broadcast bias to match batch dimension
        res = jax.vmap(self.manifold.addition, in_axes=(0, None, None), out_axes=0)(x, bias, c)
        return res


class HypLinearPoincarePP(nnx.Module):
    """
    Hyperbolic Neural Networks ++ fully connected layer (Poincaré ball model).

    Computation steps:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)
        2) Calculate the generalized linear transformation from the regression score(s)

    Parameters
    ----------
    manifold_module : module
        The PoincareBall manifold module
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
    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space,
        clamping_factor, smoothing_factor) are treated as static and will be baked into the compiled function.

    References
    ----------
    Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
        arXiv preprint arXiv:2006.08210 (2020).
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

        # Trainable parameters
        # Tangent space weight
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)))
        # Scalar bias
        self.bias = nnx.Param(jnp.zeros((out_dim, 1)))

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the HNN++ hyperbolic linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor where the hyperbolic_axis is last
        c : float
            Manifold curvature (default: 1.0)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Poincaré ball manifold
        """
        # Map to manifold if needed (static branch - JIT friendly)
        if self.input_space == "tangent":
            x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Compute multinomial linear regression
        v = compute_mlr_poincare_pp(
            x,
            self.weight,
            self.bias,
            c,
            self.clamping_factor,
            self.smoothing_factor,
        )

        # Generalized linear transformation
        sqrt_c = jnp.sqrt(c)
        w = sinh(sqrt_c * v) / sqrt_c  # (batch, out_dim)
        w2 = jnp.sum(w**2, axis=-1, keepdims=True)  # (batch, 1)
        denom = 1 + jnp.sqrt(1 + c * w2)  # (batch, 1)
        res = w / denom  # (batch, out_dim)

        # Project results to the manifold
        res = jax.vmap(self.manifold.proj, in_axes=(0, None), out_axes=0)(res, c)

        return res
