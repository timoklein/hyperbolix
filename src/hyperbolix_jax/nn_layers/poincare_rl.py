"""Poincaré ball reinforcement learning layers for JAX/Flax NNX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from ..utils.math_utils import asinh


class HypRegressionPoincareHDRL(nnx.Module):
    """
    Hyperbolic Deep Reinforcement Learning multinomial linear regression layer (Poincaré ball model).

    Computation steps:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

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
    version : str
        Algorithm version to use: 'standard' or 'rs' (default: 'standard')
        - 'standard': scaled multinomial linear regression score function
        - 'rs': multinomial linear regression with parallel transported a

    References
    ----------
    Edoardo Cetin, Benjamin Chamberlain, Michael Bronstein and Jonathan J Hunt. "Hyperbolic deep reinforcement learning."
        arXiv (2022).
    Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
        arXiv (2020).
    """

    def __init__(
        self,
        manifold_module: Any,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        version: str = "standard",
    ):
        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        assert version in ["standard", "rs"], "version must be either 'standard' or 'rs'"

        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.version = version

        # Tangent space weight
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)))
        # Manifold bias (initialized to small random values)
        # FIXME: Not using ManifoldParameter
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)) * 0.01)

    def _dist2hyperplane(
        self,
        x: Float[Array, "batch 1 out_dim in_dim"],
        a: Float[Array, "out_dim in_dim"],
        p: Float[Array, "out_dim in_dim"],
        c: float,
    ) -> Float[Array, "batch 1 out_dim 1"]:
        """
        Computes the geodesic distance(s) of point(s) x to the hyperplane(s) defined by a and p.
        [Geoopt implementation]

        Parameters
        ----------
        x : Array (batch, 1, out_dim, in_dim)
            PoincareBall point(s)
        a : Array (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at p
        p : Array (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)
        c : float
            Manifold curvature

        Returns
        -------
        res : Array (batch, 1, out_dim, 1)
            The scaled signed geodesic distance(s) of x to the hyperplane(s) defined by a and p.

        References
        ----------
        Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
            arXiv (2020).
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        sqrt_c = jnp.sqrt(c)

        # Möbius subtraction: diff = -p ⊕ x
        # x is (batch, 1, out_dim, in_dim), p is (out_dim, in_dim)
        # We need to broadcast properly
        p_expanded = p[None, None, :, :]  # (1, 1, out_dim, in_dim)
        diff = self.manifold.addition(-p_expanded, x, c, axis=-1, backproject=True)  # (batch, 1, out_dim, in_dim)

        # Compute squared norm of diff
        diff_norm2 = jnp.sum(diff**2, axis=-1, keepdims=True).clip(min=1e-15)  # (batch, 1, out_dim, 1)

        # Compute inner product of diff and a
        a_expanded = a[None, None, :, :]  # (1, 1, out_dim, in_dim)
        sc_diff_a = jnp.sum(diff * a_expanded, axis=-1, keepdims=True)  # (batch, 1, out_dim, 1)

        # Compute norm of a
        a_norm = jnp.linalg.norm(a, ord=2, axis=-1, keepdims=True)  # (out_dim, 1)
        a_norm_expanded = a_norm[None, None, :, :]  # (1, 1, out_dim, 1)

        # Compute numerator and denominator
        num = 2.0 * sc_diff_a  # (batch, 1, out_dim, 1)
        denom = jnp.abs((1 - c * diff_norm2) * a_norm_expanded) + 1e-15  # (batch, 1, out_dim, 1)

        # Compute signed distance
        signed_distance = asinh(sqrt_c * num / denom) / sqrt_c  # (batch, 1, out_dim, 1)

        # Scale by norm of a
        res = signed_distance * a_norm_expanded  # (batch, 1, out_dim, 1)
        return res

    def _compute_mlr(
        self,
        x: Float[Array, "batch in_dim"],
        a: Float[Array, "out_dim in_dim"],
        p: Float[Array, "out_dim in_dim"],
        c: float,
    ) -> Float[Array, "batch out_dim"]:
        """
        HDRL multinomial linear regression implementation with dist2hyperplane from geoopt.

        Parameters
        ----------
        x : Array (batch, in_dim)
            PoincareBall point(s)
        a : Array (out_dim, in_dim)
            Hyperplane tangent normal(s)
        p : Array (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)
        c : float
            Manifold curvature

        Returns
        -------
        res : Array (batch, out_dim)
            The (scaled) multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and p.

        References
        ----------
        Edoardo Cetin, Benjamin Chamberlain, Michael Bronstein and Jonathan J Hunt. "Hyperbolic deep reinforcement learning."
            arXiv (2022)
        Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
            arXiv (2020).
        """
        out_dim, in_dim = a.shape
        input_batch_dims = x.shape[:-1]  # (batch,) if x is (batch, in_dim)

        # Reshape input: (batch, in_dim) -> (batch, 1, in_dim)
        input_reshaped = x.reshape(-1, 1, in_dim)  # (batch, num_spaces=1, dim_per_space=in_dim)

        # Expand to include out_dim: (batch, 1, in_dim) -> (batch, 1, out_dim, in_dim)
        input_expanded = jnp.expand_dims(input_reshaped, axis=-2)  # (batch, 1, 1, in_dim)
        input_expanded = jnp.broadcast_to(input_expanded, (input_reshaped.shape[0], 1, out_dim, in_dim))

        if self.version == "standard":
            # Compute the scaled signed distance to the hyperplane
            # Scale = Euclidean norm instead of the tangent norm of a
            signed_distance = self._dist2hyperplane(input_expanded, a, p, c)  # (batch, 1, out_dim, 1)
        elif self.version == "rs":
            # Parallel transport a to the tangent space at p and return the signed distance to the hyperplane (no scaling)
            p_norm2 = jnp.sum(p**2, axis=-1, keepdims=True)  # (out_dim, 1)
            conformal_factor = 1 - c * p_norm2  # (out_dim, 1) # not actually the conformal factor
            signed_distance = self._dist2hyperplane(input_expanded, a * conformal_factor, p, c)  # (batch, 1, out_dim, 1)
            signed_distance = signed_distance * 2 / conformal_factor[None, None, :, :]  # (batch, 1, out_dim, 1)

        # Sum over last dimension and reshape
        signed_distance = jnp.sum(signed_distance, axis=-1)  # (batch, 1, out_dim)
        res = signed_distance.reshape(*input_batch_dims, out_dim)  # (batch, out_dim)
        return res

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
        axis: int = -1,
        backproject: bool = True,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the HDRL hyperbolic regression layer.

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
            Multinomial linear regression scores
        """
        assert axis == -1, "axis must be -1, reshape your tensor accordingly."

        # Map to manifold if needed
        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, c, axis=axis, backproject=backproject)

        # Project bias to manifold
        bias = self.manifold.proj(self.bias, c, axis=axis)

        # Compute MLR scores
        res = self._compute_mlr(x, self.weight, bias, c)
        return res
