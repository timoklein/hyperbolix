"""Poincaré ball regression layers for JAX/Flax NNX.

Dimension key:
  B: batch size
  D: spatial/manifold dimension (in_dim)
  P: number of hyperplanes / output classes (out_dim)
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds.poincare import Poincare

from ..optim import mark_manifold_param
from ..utils.math_utils import asinh, smooth_clamp
from ._helpers import validate_poincare_manifold


class HypRegressionPoincare(nnx.Module):
    """
    Hyperbolic Neural Networks multinomial linear regression layer (Poincaré ball model).

    Computation steps:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

    Parameters
    ----------
    manifold_module : object
        Class-based Poincare manifold instance
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
        manifold_module: Poincare,
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
        validate_poincare_manifold(
            manifold_module,
            required_methods=("proj", "addition", "expmap_0", "ptransp_0", "conformal_factor", "compute_mlr_pp"),
        )
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

        # Trainable parameters
        # Tangent space weight (Euclidean)
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)))
        # Manifold bias (initialized to small random values)
        # Mark as manifold parameter for Riemannian optimization
        self.bias = mark_manifold_param(
            nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)) * 0.01),
            manifold=self.manifold,
            curvature=1.0,  # Default curvature, will be overridden by c parameter in forward pass
        )

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
        p_neg_PD = -p

        # Vectorize over both batch and out_dim dimensions
        addition_fn = jax.vmap(
            jax.vmap(self.manifold.addition, in_axes=(None, 0, None), out_axes=0),
            in_axes=(0, None, None),
            out_axes=0,
        )
        sub_PBD = addition_fn(p_neg_PD, x, c)  # (P, B, D)
        sub_BPD = jnp.transpose(sub_PBD, (1, 0, 2))  # (B, P, D)

        # Inner product with a: sum(sub * a, axis=-1)
        suba_BP = jnp.sum(sub_BPD * a[None, :, :], axis=-1)  # (B, P)

        # Norm of a
        a_norm_P1 = jnp.linalg.norm(a, ord=2, axis=-1, keepdims=True).clip(min=min_enorm)  # (P, 1)

        # Conformal factor for sub
        lambda_sub_BP1 = self.manifold.conformal_factor(sub_BPD, c)  # (B, P, 1)

        # asinh argument
        asinh_arg_BP = sqrt_c * lambda_sub_BP1.squeeze(-1) * suba_BP / a_norm_P1.T  # (B, P)

        # Smoothly clamp the input of asinh()
        eps = jnp.finfo(jnp.float32).eps if x.dtype == jnp.float32 else jnp.finfo(jnp.float64).eps
        clamp = self.clamping_factor * float(math.log(2 / eps))
        asinh_arg_BP = smooth_clamp(asinh_arg_BP, -clamp, clamp, self.smoothing_factor)

        # Signed distance to hyperplane
        signed_dist2hyp_BP = asinh(asinh_arg_BP) / sqrt_c  # (B, P)

        # Conformal factor for p
        lambda_p_P1 = self.manifold.conformal_factor(p, c)  # (P, 1)

        # Final result: .T broadcasts (1, P) over (B, P)
        res_BP = lambda_p_P1.T * a_norm_P1.T * signed_dist2hyp_BP  # (B, P)
        return res_BP

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
        bias_PD = jax.vmap(self.manifold.proj, in_axes=(0, None), out_axes=0)(self.bias[...], c)  # type: ignore[arg-type]

        # Parallel transport weight from tangent space at origin to tangent space at bias
        pt_weight_PD = jax.vmap(self.manifold.ptransp_0, in_axes=(0, 0, None), out_axes=0)(self.weight[...], bias_PD, c)

        # Compute the multinomial linear regression score(s)
        res_BP = self._compute_mlr(x, pt_weight_PD, bias_PD, c)
        return res_BP


class HypRegressionPoincarePP(nnx.Module):
    """
    Hyperbolic Neural Networks ++ multinomial linear regression layer (Poincaré ball model).

    Computation steps:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

    Parameters
    ----------
    manifold_module : object
        Class-based Poincare manifold instance
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
        manifold_module: Poincare,
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
        validate_poincare_manifold(
            manifold_module,
            required_methods=("proj", "addition", "expmap_0", "ptransp_0", "conformal_factor", "compute_mlr_pp"),
        )
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

        # Trainable parameters
        # Tangent space weight — scaled to match reference (van Spengler et al. 2023)
        # Reference uses std = (2 * in_dim * out_dim)^{-0.5}; unscaled normal(0,1) gives
        # row norms ≈ sqrt(in_dim) which overwhelms the MLR output scaling.
        std = 1.0 / jnp.sqrt(2.0 * in_dim * out_dim)
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)) * std)
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
        res = self.manifold.compute_mlr_pp(
            x,
            self.weight[...],
            self.bias[...],
            c,
            self.clamping_factor,
            self.smoothing_factor,
        )

        return res
