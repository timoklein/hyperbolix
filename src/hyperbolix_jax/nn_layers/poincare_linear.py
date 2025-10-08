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
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold')

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
        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space

        # Tangent space weight (Euclidean)
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)))
        # Manifold bias (initialized to small random values to avoid gradient issues at origin)
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (1, out_dim)) * 0.01)

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
        axis: int = -1,
        backproject: bool = True,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the hyperbolic linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor where the hyperbolic_axis is last
        c : float
            Manifold curvature (default: 1.0)
        axis : int
            Axis along which the tensor is hyperbolic (default: -1)
        backproject : bool
            Whether to project results back to the manifold (default: True)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Poincaré ball manifold
        """
        assert axis == -1, "axis must be -1, reshape your tensor accordingly."

        # Project bias to manifold
        bias = self.manifold.proj(self.bias, c, axis=axis)

        # Map to tangent space if needed
        if self.input_space == "manifold":
            x = self.manifold.logmap_0(x, c, axis=axis)

        # Matrix-vector multiplication in tangent space at origin
        # (batch, in_dim) @ (in_dim, out_dim) -> (batch, out_dim)
        x = jnp.einsum("bi,oi->bo", x, self.weight)

        # Map back to manifold
        x = self.manifold.expmap_0(x, c, axis=axis, backproject=backproject)

        # Manifold bias addition (Möbius addition for Poincaré)
        res = self.manifold.addition(x, bias, c, axis=axis, backproject=backproject)
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
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold')
    clamping_factor : float
        Clamping factor for the multinomial linear regression output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the multinomial linear regression output (default: 50.0)

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
        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

        # Tangent space weight
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)))
        # Scalar bias
        self.bias = nnx.Param(jnp.zeros((out_dim, 1)))

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
        axis: int = -1,
        backproject: bool = True,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the HNN++ hyperbolic linear layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor where the hyperbolic_axis is last
        c : float
            Manifold curvature (default: 1.0)
        axis : int
            Axis along which the tensor is hyperbolic (default: -1)
        backproject : bool
            Whether to project results back to the manifold (default: True)

        Returns
        -------
        res : Array of shape (batch, out_dim)
            Output on the Poincaré ball manifold
        """
        assert axis == -1, "axis must be -1, reshape your tensor accordingly."

        # Map to manifold if needed
        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, c, axis=axis, backproject=backproject)

        # Compute multinomial linear regression
        v = compute_mlr_poincare_pp(
            x,
            self.weight.value,
            self.bias.value,
            c,
            axis,
            self.clamping_factor,
            self.smoothing_factor,
        )

        # Generalized linear transformation
        sqrt_c = jnp.sqrt(c)
        w = sinh(sqrt_c * v) / sqrt_c  # (batch, out_dim)
        w2 = jnp.sum(w**2, axis=axis, keepdims=True)  # (batch, 1)
        denom = 1 + jnp.sqrt(1 + c * w2)  # (batch, 1)
        res = w / denom  # (batch, out_dim)

        if backproject:
            res = self.manifold.proj(res, c, axis=axis)

        return res
