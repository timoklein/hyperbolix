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
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration - changing it after initialization requires recompilation.
    version : str
        Algorithm version to use: 'standard' or 'rs' (default: 'standard').
        - 'standard': scaled multinomial linear regression score function
        - 'rs': multinomial linear regression with parallel transported a
        Note: This is a static configuration - changing it after initialization requires recompilation.
    backproject : bool
        Whether to project results back to the manifold (default: True).
        Note: This is a static configuration - changing it after initialization requires recompilation.

    Notes
    -----
    JIT Compatibility:
        This layer is designed to work with nnx.jit. Configuration parameters (input_space, version, backproject)
        are treated as static and will be baked into the compiled function.

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
        backproject: bool = True,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")
        if version not in ["standard", "rs"]:
            raise ValueError(f"version must be either 'standard' or 'rs', got '{version}'")

        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.version = version
        self.backproject = backproject

        # Trainable parameters
        # Tangent space weight
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)))
        # Manifold bias (initialized to small random values)
        # FIXME: Not using ManifoldParameter
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)) * 0.01)

    def _dist2hyperplane(
        self,
        x: Float[Array, "in_dim"],
        a: Float[Array, "in_dim"],
        p: Float[Array, "in_dim"],
        c: float,
    ) -> Float[Array, ""]:
        """
        Computes the geodesic distance of a single point x to a hyperplane defined by a and p.
        [Geoopt implementation]

        Parameters
        ----------
        x : Array (in_dim,)
            PoincareBall point
        a : Array (in_dim,)
            Hyperplane tangent normal in the tangent space at p
        p : Array (in_dim,)
            Hyperplane PoincareBall translation
        c : float
            Manifold curvature

        Returns
        -------
        res : Scalar
            The scaled signed geodesic distance of x to the hyperplane defined by a and p.

        References
        ----------
        Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
            arXiv (2020).
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        sqrt_c = jnp.sqrt(c)

        # Möbius subtraction: diff = -p ⊕ x
        diff = self.manifold.addition(-p, x, c)

        # Compute squared norm of diff
        diff_norm2 = jnp.maximum(jnp.dot(diff, diff), 1e-15)

        # Compute inner product of diff and a
        sc_diff_a = jnp.dot(diff, a)

        # Compute norm of a
        a_norm = jnp.linalg.norm(a)

        # Compute numerator and denominator
        num = 2.0 * sc_diff_a
        denom = jnp.abs((1 - c * diff_norm2) * a_norm) + 1e-15

        # Compute signed distance
        signed_distance = asinh(sqrt_c * num / denom) / sqrt_c

        # Scale by norm of a
        res = signed_distance * a_norm
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
        # Static branch - JIT friendly
        if self.version == "standard":
            # Compute the scaled signed distance to the hyperplane for each (x_i, a_j, p_j) combination
            # vmap over batch dimension (x), then over out_dim (a, p)
            dist_fn = jax.vmap(
                jax.vmap(self._dist2hyperplane, in_axes=(None, 0, 0, None), out_axes=0),
                in_axes=(0, None, None, None),
                out_axes=0,
            )
            signed_distance = dist_fn(x, a, p, c)  # (batch, out_dim)
        elif self.version == "rs":
            # Parallel transport a to the tangent space at p
            p_norm2 = jnp.sum(p**2, axis=-1, keepdims=True)  # (out_dim, 1)
            conformal_factor = 1 - c * p_norm2  # (out_dim, 1)
            a_scaled = a * conformal_factor  # (out_dim, in_dim)

            # Compute distance with scaled a
            dist_fn = jax.vmap(
                jax.vmap(self._dist2hyperplane, in_axes=(None, 0, 0, None), out_axes=0),
                in_axes=(0, None, None, None),
                out_axes=0,
            )
            signed_distance = dist_fn(x, a_scaled, p, c)  # (batch, out_dim)
            signed_distance = signed_distance * 2 / conformal_factor.T  # (batch, out_dim)

        return signed_distance

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the HDRL hyperbolic regression layer.

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
            x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Project bias to manifold (vmap over out_dim dimension)
        bias = jax.vmap(self.manifold.proj, in_axes=(0, None), out_axes=0)(self.bias, c)

        # Compute MLR scores
        res = self._compute_mlr(x, self.weight, bias, c)
        return res
