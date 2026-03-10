"""Hyperboloid linear layers for JAX/Flax NNX.

This module contains linear transformation layers for the hyperboloid manifold,
including the Hyperbolic Transformation Component (HTC) module from the Hypformer paper
and the FGG linear layer from Klis et al. 2026.

For the core HTC/HRC functions, see hyperboloid_core module.

Dimension key:
  B: batch size
  I: in_spatial (in_features - 1)    O: out_spatial (out_features - 1)
  Ai: in_ambient (in_features)       Ao: out_ambient (out_features)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds import Manifold

from ._helpers import validate_hyperboloid_manifold
from .hyperboloid_core import build_spacelike_V, htc


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
        manifold_module: Manifold,
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
        validate_hyperboloid_manifold(manifold_module, required_methods=("expmap_0",))
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

        # Linear transformation: (B, in_dim) → (B, out_dim)
        x_BO = jnp.einsum("bi,oi->bo", x, self.weight) + self.bias

        # Split into time and space: x0 is first coord, x_rem is spatial
        x0_B1 = x_BO[:, 0:1]  # (B, 1) — time coordinate
        x_rem_BD = x_BO[:, 1:]  # (B, D) where D = out_dim - 1

        # Static branch - JIT friendly
        if self.normalize:
            x_rem_norm_B1 = jnp.linalg.norm(x_rem_BD, ord=2, axis=-1, keepdims=True)  # (B, 1)

            # Learnable sigmoid scaling
            scale_val = self.scale[...] if isinstance(self.scale, nnx.Param) else self.scale
            scale_B1 = jnp.exp(scale_val) * jax.nn.sigmoid(x0_B1)  # (B, 1)

            res0_B1 = jnp.sqrt(scale_B1**2 + 1 / c + self.eps)  # (B, 1)
            res_rem_BD = scale_B1 * x_rem_BD / x_rem_norm_B1  # (B, D)

            res_BA = jnp.concatenate([res0_B1, res_rem_BD], axis=-1)  # (B, A)

            # Cast near-zero-norm vectors to origin
            origin_time_B1 = jnp.sqrt(1 / c) * jnp.ones_like(res0_B1)
            origin_BA = jnp.concatenate([origin_time_B1, jnp.zeros_like(res_rem_BD)], axis=-1)

            mask_B1 = x_rem_norm_B1 <= 1e-5
            res_BA = jnp.where(mask_B1, origin_BA, res_BA)
        else:
            # Reconstruct time from space: x₀ = sqrt(||x_rem||² + 1/c)
            res0_B1 = jnp.sqrt(jnp.sum(x_rem_BD**2, axis=-1, keepdims=True) + 1 / c)  # (B, 1)
            res_BA = jnp.concatenate([res0_B1, x_rem_BD], axis=-1)  # (B, A)

        return res_BA


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


class FGGLinear(nnx.Module):
    """Fast and Geometrically Grounded Lorentz linear layer.

    Implements the FGG linear layer from Klis et al. 2026. The key insight is that
    the sinh/arcsinh cancellation in the Lorentzian activation chain simplifies the
    forward pass to: matmul with spacelike V matrix -> Euclidean activation ->
    time reconstruction. This achieves linear growth of hyperbolic distance (vs
    logarithmic for Chen et al. 2022) and ~3x faster training/inference.

    Forward pass:
        1. Build spacelike V matrix from (U, b) with Minkowski metric absorbed
        2. z = x @ V   (Minkowski inner products via a single matmul)
        3. z = h(z)     (Euclidean activation, e.g. ReLU)
        4. y_0 = sqrt(||z||^2 + 1/c)   (time reconstruction)
        5. y = [y_0, z]   (on hyperboloid)

    Parameters
    ----------
    in_features : int
        Input ambient dimension (D_in + 1), including time component.
    out_features : int
        Output ambient dimension (D_out + 1), including time component.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    activation : Callable or None, optional
        Euclidean activation function applied after matmul (default: None).
    reset_params : str, optional
        Weight initialization scheme: ``"eye"``, ``"xavier"``, ``"kaiming"``,
        ``"lorentz_kaiming"``, or ``"mlr"`` (default: ``"eye"``).
    use_weight_norm : bool, optional
        If True, reparameterize U as ``g * v / ||v||`` for weight normalization
        (default: False).
    init_bias : float, optional
        Initial value for bias entries (default: 0.5).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    References
    ----------
    Klis et al. "Fast and Geometrically Grounded Lorentz Neural Networks" (2026).

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import FGGLinear
    >>> import jax.numpy as jnp
    >>>
    >>> layer = FGGLinear(33, 65, rngs=nnx.Rngs(0), activation=jax.nn.relu)
    >>> x = jnp.ones((8, 33))
    >>> # project to hyperboloid
    >>> x = x.at[:, 0].set(jnp.sqrt(jnp.sum(x[:, 1:]**2, axis=-1) + 1.0))
    >>> y = layer(x, c=1.0)
    >>> y.shape
    (8, 65)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        activation: Callable[[jax.Array], jax.Array] | None = None,
        reset_params: str = "eye",
        use_weight_norm: bool = False,
        init_bias: float = 0.5,
        eps: float = 1e-7,
    ):
        if reset_params not in ("eye", "xavier", "kaiming", "lorentz_kaiming", "mlr"):
            raise ValueError(
                f"reset_params must be 'eye', 'xavier', 'kaiming', 'lorentz_kaiming', or 'mlr', got '{reset_params}'"
            )

        in_spatial = in_features - 1  # I
        out_spatial = out_features - 1  # O

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        self.eps = eps

        # Initialize Euclidean weight U: (I, O)
        key = rngs.params()
        if reset_params == "eye":
            U_init = 0.5 * jnp.eye(in_spatial, out_spatial)
        elif reset_params == "xavier":
            std = jnp.sqrt(1.0 / (in_spatial + out_spatial))
            U_init = jax.random.normal(key, (in_spatial, out_spatial)) * std
        elif reset_params == "kaiming":
            std = jnp.sqrt(2.0 / in_spatial)
            U_init = jax.random.normal(key, (in_spatial, out_spatial)) * std
        elif reset_params == "lorentz_kaiming":
            std = jnp.sqrt(1.0 / in_spatial)
            U_init = jax.random.normal(key, (in_spatial, out_spatial)) * std
        else:  # mlr
            std = jnp.sqrt(5.0 / in_spatial)
            U_init = jax.random.normal(key, (in_spatial, out_spatial)) * std

        # Weight normalization: decompose U = softplus(g) * v / ||v||
        if use_weight_norm:
            # Reference: v from reset_params (normalized in forward), g fixed magnitude
            self.v = nnx.Param(U_init)  # (I, O) direction
            g_init_val = jnp.sqrt(1.0 / (in_spatial + out_spatial))
            self.g = nnx.Param(jnp.full((out_spatial,), g_init_val))  # (O,)
        else:
            self.U = nnx.Param(U_init)  # (I, O)

        # Bias: init to init_bias
        self.b = nnx.Param(jnp.full((out_spatial,), init_bias))  # (O,)

    def _get_U(self) -> jax.Array:
        """Return the effective weight matrix, handling weight normalization."""
        if self.use_weight_norm:
            v_IO = self.v[...]  # (I, O)
            g_O = self.g[...]  # (O,)
            g_pos_O = jax.nn.softplus(g_O)  # (O,) force positive magnitudes
            v_norm_O = jnp.sqrt(jnp.sum(v_IO**2, axis=0) + self.eps)  # (O,)
            return g_pos_O[None, :] * v_IO / v_norm_O[None, :]  # (I, O)
        return self.U[...]  # (I, O)

    def __call__(
        self,
        x_BAi: Float[Array, "batch in_features"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_features"]:
        """Forward pass through the FGG linear layer.

        Parameters
        ----------
        x_BAi : Array, shape (B, Ai)
            Input points on the hyperboloid with curvature ``c``.
            Ai = in_features (ambient dimension).
        c : float, optional
            Curvature parameter (default: 1.0).

        Returns
        -------
        y_BAo : Array, shape (B, Ao)
            Output points on the hyperboloid with curvature ``c``.
            Ao = out_features (ambient dimension).
        """
        # 1. Get effective U (handle weight norm)
        U_IO = self._get_U()  # (I, O)

        # 2. Build V_mink from (U, b) — Minkowski metric absorbed
        V_AiO = build_spacelike_V(U_IO, self.b[...], c, self.eps)  # (Ai, O)
        # Cast V to match input dtype (avoids float32/float64 scatter warnings)
        V_AiO = V_AiO.astype(x_BAi.dtype)

        # 3. Minkowski inner products via matmul (metric in V)
        z_BO = x_BAi @ V_AiO  # (B, O)

        # 4. Apply Euclidean activation (Lorentzian wrapping implicit via cancellation)
        if self.activation is not None:
            z_BO = self.activation(z_BO)

        # 5. Reconstruct hyperboloid point: spatial = z, time from constraint
        y_0_B1 = jnp.sqrt(jnp.sum(z_BO**2, axis=-1, keepdims=True) + 1.0 / c)  # (B, 1)

        return jnp.concatenate([y_0_B1, z_BO], axis=-1)  # (B, Ao)
