"""Hyperboloid regression layers for JAX/Flax NNX.

Dimension key:
  B: batch size
  I: in_spatial (in_features - 1)    K: num_classes
  Ai: in_ambient (in_features)
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.utils.math_utils import asinh as safe_asinh

from ._helpers import validate_hyperboloid_manifold
from .hyperboloid_core import build_spacelike_V


class HypRegressionHyperboloid(nnx.Module):
    """
    Fully Hyperbolic Convolutional Neural Networks multinomial linear regression layer (Hyperboloid model).

    Computation steps:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

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
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic convolutional neural networks for computer vision."
        arXiv preprint arXiv:2303.15919 (2023).
    """

    def __init__(
        self,
        manifold_module: Hyperboloid,
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
        validate_hyperboloid_manifold(manifold_module, required_methods=("expmap_0", "compute_mlr"))
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

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
            x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)

        # Compute multinomial linear regression
        res = self.manifold.compute_mlr(
            x,
            self.weight[...],
            self.bias[...],
            c,
            self.clamping_factor,
            self.smoothing_factor,
        )

        return res


class FGGLorentzMLR(nnx.Module):
    """Fast and Geometrically Grounded Lorentz multinomial logistic regression.

    Outputs Euclidean logits (signed scaled distances to hyperplanes) using the
    FGG spacelike V construction. Unlike ``HypRegressionHyperboloid``, this layer
    uses the distance-to-hyperplane formulation matching the reference fc_mlr
    (``signed_dist2hyperplanes_scaled_angle``).

    Forward pass (matching reference fc_mlr):
        1. Build V_mink from (z, a)
        2. mink = x @ V_mink   (Minkowski inner products)
        3. logits = asinh(sqrt(c) * mink) / sqrt(c)   (signed scaled distances)

    Parameters
    ----------
    in_features : int
        Input ambient dimension (D_in + 1), including time component.
    num_classes : int
        Number of output classes.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    reset_params : str, optional
        Weight initialization scheme for hyperplane normals: ``"mlr"``
        (normal, std=sqrt(5/I)) or ``"default"`` (uniform) (default: ``"mlr"``).
    init_bias : float, optional
        Initial value for bias entries (default: 0.5).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    References
    ----------
    Klis et al. "Fast and Geometrically Grounded Lorentz Neural Networks" (2026), Eq. 23.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        rngs: nnx.Rngs,
        reset_params: str = "mlr",
        init_bias: float = 0.5,
        eps: float = 1e-7,
    ):
        if reset_params not in ("default", "mlr"):
            raise ValueError(f"reset_params must be 'default' or 'mlr', got '{reset_params}'")

        in_spatial = in_features - 1  # I
        self.in_features = in_features
        self.num_classes = num_classes
        self.eps = eps

        # Hyperplane normals (spatial) and bias offsets
        key = rngs.params()
        if reset_params == "mlr":
            # Reference: N(0, sqrt(5/I)) for spatial weights
            std = jnp.sqrt(5.0 / in_spatial)
            self.z = nnx.Param(jax.random.normal(key, (in_spatial, num_classes)) * std)
        else:  # default
            stdv = 1.0 / jnp.sqrt(jnp.array(in_spatial, dtype=jnp.float32))
            self.z = nnx.Param(jax.random.uniform(key, (in_spatial, num_classes), minval=-stdv, maxval=stdv))
        self.a = nnx.Param(jnp.full((num_classes,), init_bias))

    def __call__(
        self,
        x_BAi: Float[Array, "batch in_features"],
        c: float = 1.0,
    ) -> Float[Array, "batch num_classes"]:
        """Forward pass returning Euclidean logits.

        Parameters
        ----------
        x_BAi : Array, shape (B, Ai)
            Input points on the hyperboloid with curvature ``c``.
        c : float, optional
            Curvature parameter (default: 1.0).

        Returns
        -------
        logits_BK : Array, shape (B, K)
            Euclidean logits (signed scaled distances to hyperplanes).
        """
        # 1. Build V_mink from (z, a)
        V_AiK = build_spacelike_V(self.z[...], self.a[...], c, self.eps)  # (Ai, K)
        # Cast V to match input dtype (avoids float32/float64 scatter warnings)
        V_AiK = V_AiK.astype(x_BAi.dtype)

        # 2. Minkowski inner products
        mink_BK = x_BAi @ V_AiK  # (B, K)

        # 3. Signed scaled distances (matching reference fc_mlr: no norm scaling)
        sqrt_c = jnp.sqrt(c)
        logits_BK = safe_asinh(sqrt_c * mink_BK) / sqrt_c  # (B, K)

        return logits_BK
