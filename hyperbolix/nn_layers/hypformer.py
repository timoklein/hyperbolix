"""Hyperbolic Transformation Component (HTC) and Hyperbolic Regularization Component (HRC).

This module implements the HTC and HRC layers from the Hypformer paper, which generalize
hyperbolic neural network operations by supporting curvature changes (c_in → c_out) and
wrapping arbitrary Euclidean functions.

Key features:
- HRC: Wraps Euclidean regularization/activation functions operating on spatial components
- HTC: Wraps Euclidean linear transformations operating on full hyperboloid points
- When c_in = c_out, HRC reduces to existing hyperboloid activations

References
----------
Hypformer paper (citation to be added)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


def hrc(
    x: Float[Array, "... dim_plus_1"],
    f_r: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... out_dim_plus_1"]:
    """Hyperbolic Regularization Component.

    Applies a Euclidean regularization/activation function f_r to the spatial
    components of hyperboloid points, then maps the result to the hyperboloid
    with curvature c_out.

    Mathematical formula:
        space = sqrt(c_in/c_out) * f_r(x[..., 1:])
        time  = sqrt(||space||^2 + 1/c_out)
        output = [time, space]

    When c_in = c_out = c, this reduces to:
        output = [sqrt(||f_r(x_s)||^2 + 1/c), f_r(x_s)]
    which is the pattern used by existing hyperboloid activations.

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
        The first element is the time-like component, remaining are spatial.
    f_r : Callable
        Euclidean function to apply to spatial components. Can be any activation,
        normalization, dropout, etc. Takes spatial components and returns
        transformed spatial components (may change dimension).
    c_in : float
        Input curvature parameter (must be positive, c > 0).
    c_out : float
        Output curvature parameter (must be positive, c > 0).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., out_dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.

    Notes
    -----
    - f_r operates only on spatial components x[..., 1:], not the time component
    - The time component is reconstructed using the hyperboloid constraint:
      -x₀² + ||x_rest||² = -1/c_out
    - This avoids expensive exp/log maps while maintaining mathematical correctness
    - The spatial scaling factor sqrt(c_in/c_out) ensures proper curvature transformation

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers import hrc
    >>> from hyperbolix.manifolds import hyperboloid
    >>>
    >>> # Create a point on the hyperboloid
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> x = hyperboloid.proj(x, c=1.0)
    >>>
    >>> # Apply HRC with ReLU (curvature-preserving)
    >>> y = hrc(x, jax.nn.relu, c_in=1.0, c_out=1.0)
    >>>
    >>> # Apply HRC with curvature change
    >>> y = hrc(x, jax.nn.relu, c_in=1.0, c_out=2.0)
    >>>
    >>> # Custom activation
    >>> def custom_act(z):
    ...     return jax.nn.gelu(z) * 0.5
    >>> y = hrc(x, custom_act, c_in=1.0, c_out=0.5)
    """
    # Extract spatial components
    x_space = x[..., 1:]

    # Apply Euclidean function to spatial components
    out_space = f_r(x_space)

    # Scale spatial components for curvature transformation
    scale = jnp.sqrt(c_in / c_out)
    scaled_space = scale * out_space

    # Compute norm squared of scaled spatial components
    norm_sq = jnp.sum(scaled_space**2, axis=-1)

    # Reconstruct time component using hyperboloid constraint
    # Constraint: -x₀² + ||x_rest||² = -1/c_out
    # => x₀ = sqrt(||x_rest||² + 1/c_out)
    x0_sq = norm_sq + 1.0 / c_out
    x0 = jnp.sqrt(jnp.maximum(x0_sq, eps))

    # Concatenate time and spatial components
    return jnp.concatenate([x0[..., None], scaled_space], axis=-1)


def htc(
    x: Float[Array, "... in_dim_plus_1"],
    f_t: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... out_dim_plus_1"]:
    """Hyperbolic Transformation Component.

    Applies a Euclidean linear transformation f_t to the full hyperboloid point
    (including time component), then maps the result to the hyperboloid with
    curvature c_out.

    Mathematical formula:
        space = sqrt(c_in/c_out) * f_t(x)
        time  = sqrt(||space||^2 + 1/c_out)
        output = [time, space]

    where f_t takes the full (dim+1)-dimensional input and produces the output
    spatial components.

    Parameters
    ----------
    x : Array of shape (..., in_dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
        All components (time and spatial) are passed to f_t.
    f_t : Callable
        Euclidean linear transformation applied to the full input. Takes
        (in_dim+1)-dimensional input and produces out_dim-dimensional output
        (which becomes the spatial components of the output).
    c_in : float
        Input curvature parameter (must be positive, c > 0).
    c_out : float
        Output curvature parameter (must be positive, c > 0).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., out_dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.

    Notes
    -----
    - Unlike HRC, f_t operates on the full point including the time component
    - f_t's output dimension determines the output spatial dimension
    - This is typically used for learnable linear transformations
    - The spatial scaling factor sqrt(c_in/c_out) ensures proper curvature transformation

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers import htc
    >>> from hyperbolix.manifolds import hyperboloid
    >>>
    >>> # Create a point on the hyperboloid
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> x = hyperboloid.proj(x, c=1.0)
    >>>
    >>> # Define a linear transformation
    >>> W = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    >>> def linear(z):
    ...     return z @ W.T
    >>>
    >>> # Apply HTC
    >>> y = htc(x, linear, c_in=1.0, c_out=2.0)
    >>> y.shape
    (4,)  # (3 spatial + 1 time)
    """
    # Apply Euclidean transformation to full input
    # f_t: (in_dim+1,) → (out_dim,)
    out = f_t(x)

    # Scale output for curvature transformation
    scale = jnp.sqrt(c_in / c_out)
    scaled_out = scale * out

    # Compute norm squared of scaled output
    norm_sq = jnp.sum(scaled_out**2, axis=-1)

    # Reconstruct time component using hyperboloid constraint
    x0_sq = norm_sq + 1.0 / c_out
    x0 = jnp.sqrt(jnp.maximum(x0_sq, eps))

    # Concatenate time and spatial components
    return jnp.concatenate([x0[..., None], scaled_out], axis=-1)


# Convenience functions for common activations


def hrc_relu(
    x: Float[Array, "... dim_plus_1"],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... dim_plus_1"]:
    """HRC with ReLU activation.

    Equivalent to hrc(x, jax.nn.relu, c_in, c_out, eps).
    When c_in = c_out = c, this is equivalent to hyp_relu(x, c).

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
    c_in : float
        Input curvature parameter (must be positive).
    c_out : float
        Output curvature parameter (must be positive).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.
    """
    return hrc(x, jax.nn.relu, c_in, c_out, eps)


def hrc_leaky_relu(
    x: Float[Array, "... dim_plus_1"],
    c_in: float,
    c_out: float,
    negative_slope: float = 0.01,
    eps: float = 1e-7,
) -> Float[Array, "... dim_plus_1"]:
    """HRC with LeakyReLU activation.

    When c_in = c_out = c, this is equivalent to hyp_leaky_relu(x, c, negative_slope).

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
    c_in : float
        Input curvature parameter (must be positive).
    c_out : float
        Output curvature parameter (must be positive).
    negative_slope : float, optional
        Negative slope coefficient for LeakyReLU (default: 0.01).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.
    """

    def f_r(z):
        return jax.nn.leaky_relu(z, negative_slope)

    return hrc(x, f_r, c_in, c_out, eps)


def hrc_tanh(
    x: Float[Array, "... dim_plus_1"],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... dim_plus_1"]:
    """HRC with tanh activation.

    When c_in = c_out = c, this is equivalent to hyp_tanh(x, c).

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
    c_in : float
        Input curvature parameter (must be positive).
    c_out : float
        Output curvature parameter (must be positive).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.
    """
    return hrc(x, jnp.tanh, c_in, c_out, eps)


def hrc_swish(
    x: Float[Array, "... dim_plus_1"],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... dim_plus_1"]:
    """HRC with Swish/SiLU activation.

    When c_in = c_out = c, this is equivalent to hyp_swish(x, c).

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
    c_in : float
        Input curvature parameter (must be positive).
    c_out : float
        Output curvature parameter (must be positive).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.
    """
    return hrc(x, jax.nn.swish, c_in, c_out, eps)


def hrc_gelu(
    x: Float[Array, "... dim_plus_1"],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... dim_plus_1"]:
    """HRC with GELU activation.

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
    c_in : float
        Input curvature parameter (must be positive).
    c_out : float
        Output curvature parameter (must be positive).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.
    """
    return hrc(x, jax.nn.gelu, c_in, c_out, eps)


# Flax NNX Modules


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

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import HTCLinear
    >>> from hyperbolix.manifolds import hyperboloid
    >>>
    >>> # Create layer
    >>> layer = HTCLinear(in_features=5, out_features=8, rngs=nnx.Rngs(0))
    >>>
    >>> # Forward pass
    >>> x = jnp.ones((32, 5))  # batch of 32 points
    >>> x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)
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


class HRCDropout(nnx.Module):
    """Hyperbolic Regularization Component with dropout.

    Applies dropout to spatial components of hyperboloid points, then reconstructs
    the time component for the output curvature.

    Parameters
    ----------
    rate : float
        Dropout probability (fraction of units to drop).
    rngs : nnx.Rngs
        Random number generators for dropout.
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Attributes
    ----------
    dropout : nnx.Dropout
        Flax dropout layer.
    eps : float
        Numerical stability parameter.

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import HRCDropout
    >>>
    >>> dropout = HRCDropout(rate=0.1, rngs=nnx.Rngs(dropout=42))
    >>> y = dropout(x, c_in=1.0, c_out=1.0, deterministic=False)
    """

    def __init__(self, rate: float, *, rngs: nnx.Rngs, eps: float = 1e-7):
        self.dropout = nnx.Dropout(rate, rngs=rngs)
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "batch dim_plus_1"],
        c_in: float = 1.0,
        c_out: float = 1.0,
        deterministic: bool = False,
    ) -> Float[Array, "batch dim_plus_1"]:
        """Apply HRC dropout.

        Parameters
        ----------
        x : Array of shape (batch, dim+1)
            Input points on hyperboloid with curvature c_in.
        c_in : float, optional
            Input curvature (default: 1.0).
        c_out : float, optional
            Output curvature (default: 1.0).
        deterministic : bool, optional
            If True, no dropout is applied (for evaluation mode).

        Returns
        -------
        y : Array of shape (batch, dim+1)
            Output points on hyperboloid with curvature c_out.
        """

        def drop_fn(z):
            return self.dropout(z, deterministic=deterministic)

        return hrc(x, drop_fn, c_in, c_out, self.eps)


class HRCLayerNorm(nnx.Module):
    """Hyperbolic Regularization Component with layer normalization.

    Applies layer normalization to spatial components of hyperboloid points,
    then reconstructs the time component for the output curvature.

    Parameters
    ----------
    num_features : int
        Number of spatial features to normalize.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    epsilon : float, optional
        Small value for numerical stability in layer norm (default: 1e-5).
    eps : float, optional
        Small value for numerical stability in HRC (default: 1e-7).

    Attributes
    ----------
    ln : nnx.LayerNorm
        Flax layer normalization.
    eps : float
        Numerical stability parameter for HRC.

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import HRCLayerNorm
    >>>
    >>> ln = HRCLayerNorm(num_features=64, rngs=nnx.Rngs(0))
    >>> y = ln(x, c_in=1.0, c_out=2.0)
    """

    def __init__(
        self,
        num_features: int,
        *,
        rngs: nnx.Rngs,
        epsilon: float = 1e-5,
        eps: float = 1e-7,
    ):
        self.ln = nnx.LayerNorm(num_features, epsilon=epsilon, rngs=rngs)
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "batch dim_plus_1"],
        c_in: float = 1.0,
        c_out: float = 1.0,
    ) -> Float[Array, "batch dim_plus_1"]:
        """Apply HRC layer normalization.

        Parameters
        ----------
        x : Array of shape (batch, dim+1)
            Input points on hyperboloid with curvature c_in.
        c_in : float, optional
            Input curvature (default: 1.0).
        c_out : float, optional
            Output curvature (default: 1.0).

        Returns
        -------
        y : Array of shape (batch, dim+1)
            Output points on hyperboloid with curvature c_out.
        """
        return hrc(x, self.ln, c_in, c_out, self.eps)
