"""Poincaré ball regression layers for JAX/Flax NNX."""

import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from ..utils.math_utils import asinh, smooth_clamp
from .helpers import compute_mlr_poincare_pp, safe_conformal_factor


class HypRegressionPoincare(nnx.Module):
    """
    Hyperbolic Neural Networks multinomial linear regression layer (Poincaré ball model).

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
        # Manifold bias (initialized to small random values)
        # FIXME: Not using ManifoldParameter
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)) * 0.01)

    def _compute_mlr(
        self,
        x: Float[Array, "batch in_dim"],
        a: Float[Array, "out_dim in_dim"],
        p: Float[Array, "out_dim in_dim"],
        c: float,
        min_enorm: float = 1e-15,
    ) -> Float[Array, "batch out_dim"]:
        """
        Internal method for computing the multinomial linear regression score(s).

        Parameters
        ----------
        x : Array (batch, in_dim)
            PoincareBall point(s)
        a : Array (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at p
        p : Array (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)
        c : float
            Manifold curvature
        min_enorm : float
            Minimum norm to avoid division by zero

        Returns
        -------
        res : Array (batch, out_dim)
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and p.

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        sqrt_c = jnp.sqrt(c)

        # Compute Möbius subtraction: -p ⊕ x for each (x, p_i) pair
        # p is (out_dim, in_dim), x is (batch, in_dim)
        # We need to compute -p[i] ⊕ x[j] for all i, j
        # Result shape: (batch, out_dim, in_dim)
        p_neg = -p  # (out_dim, in_dim)

        # Vectorize over both batch and out_dim dimensions
        # vmap over batch dimension (axis 0 of x), then over out_dim (axis 0 of p_neg)
        addition_fn = jax.vmap(
            jax.vmap(self.manifold.addition, in_axes=(None, 0, None), out_axes=0),
            in_axes=(0, None, None),
            out_axes=0,
        )
        sub = addition_fn(p_neg, x, c)  # (out_dim, batch, in_dim)
        sub = jnp.transpose(sub, (1, 0, 2))  # (batch, out_dim, in_dim)

        # Compute inner product with a: sum(sub * a, axis=-1)
        # sub is (batch, out_dim, in_dim), a is (out_dim, in_dim)
        suba = jnp.sum(sub * a[None, :, :], axis=-1)  # (batch, out_dim)

        # Compute norm of a
        a_norm = jnp.linalg.norm(a, ord=2, axis=-1, keepdims=True).clip(min=min_enorm)  # (out_dim, 1)

        # Compute conformal factor for sub
        lambda_sub = safe_conformal_factor(sub, c)  # (batch, out_dim, 1)

        # Compute asinh argument
        asinh_arg = sqrt_c * lambda_sub.squeeze(-1) * suba / a_norm.T  # (batch, out_dim)

        # Improve performance by smoothly clamping the input of asinh()
        eps = jnp.finfo(jnp.float32).eps if x.dtype == jnp.float32 else jnp.finfo(jnp.float64).eps
        clamp = self.clamping_factor * float(math.log(2 / eps))
        asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, self.smoothing_factor)

        # Compute signed distance to hyperplane
        signed_dist2hyp = asinh(asinh_arg) / sqrt_c  # (batch, out_dim)

        # Compute conformal factor for p
        lambda_p = safe_conformal_factor(p, c)  # (out_dim, 1)

        # Final result
        res = lambda_p.T * a_norm.T * signed_dist2hyp  # (batch, out_dim)
        return res

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
            x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Project bias to manifold (vmap over out_dim dimension)
        bias = jax.vmap(self.manifold.proj, in_axes=(0, None), out_axes=0)(self.bias, c)

        # Map self.weight from the tangent space at the origin to the tangent space at self.bias
        # vmap over out_dim dimension
        pt_weight = jax.vmap(self.manifold.ptransp_0, in_axes=(0, 0, None), out_axes=0)(self.weight, bias, c)

        # Compute the multinomial linear regression score(s)
        res = self._compute_mlr(x, pt_weight, bias, c)
        return res


class HypRegressionPoincarePP(nnx.Module):
    """
    Hyperbolic Neural Networks ++ multinomial linear regression layer (Poincaré ball model).

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
        # Scalar bias (initialized to small random values)
        self.bias = nnx.Param(jax.random.normal(rngs.params(), (out_dim, 1)) * 0.01)

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the HNN++ hyperbolic regression layer.

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

        # Compute multinomial linear regression
        res = compute_mlr_poincare_pp(
            x,
            self.weight,
            self.bias,
            c,
            self.clamping_factor,
            self.smoothing_factor,
        )

        return res
