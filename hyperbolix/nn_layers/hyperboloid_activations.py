"""Hyperboloid activation functions for JAX.

This module implements activation functions for the Hyperboloid manifold that apply
activations directly to space components and reconstruct the time component using
the manifold constraint. This approach avoids frequent logarithmic and exponential
maps, providing better numerical stability than the tangent space approach.

The module provides two families of activations:
- hrc_*: Hyperbolic Regularization Component with curvature-change support (c_in → c_out)
- hyp_*: Curvature-preserving wrappers (c_in = c_out = c) for convenience

For the core hrc() function and NNX modules, see hypformer module.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# Import hrc from hypformer (used by regularization NNX modules)
from .hypformer import hrc

# Curvature-Changing Activations (HRC convenience functions)


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


# Curvature-Preserving Activations (convenience wrappers)


def hyp_relu(x: Float[Array, "... dim_plus_1"], c: float) -> Float[Array, "... dim_plus_1"]:
    """Apply ReLU activation to space components of hyperboloid point(s).

    Curvature-preserving wrapper around hrc_relu(x, c_in=c, c_out=c).

    This function applies the ReLU activation function to the spatial components
    of hyperboloid points and reconstructs valid manifold points using the
    hyperboloid constraint.

    Mathematical formula:
        y = [sqrt(||ReLU(x_s)||^2 + 1/c), ReLU(x_s)]

    where x_s are the spatial components x[..., 1:].

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold in ambient space, where
        ... represents arbitrary batch dimensions. The last dimension contains
        the time component (x[..., 0]) and spatial components (x[..., 1:]).
    c : float
        Curvature parameter, must be positive (c > 0).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold, same shape as input.

    Notes
    -----
    - This function applies ReLU only to spatial components, not the time component
    - The time component is reconstructed using the hyperboloid constraint:
      -x₀² + ||x_rest||² = -1/c
    - This approach avoids frequent exp/log maps for better numerical stability
    - Works on arrays of any shape, similar to jax.nn.relu
    - For curvature-changing transformations, use `hrc_relu` which supports
      different input/output curvatures

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic
    convolutional neural networks for computer vision." arXiv preprint
    arXiv:2303.15919 (2023).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers import hyp_relu
    >>>
    >>> # Single point
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> y = hyp_relu(x, c=1.0)
    >>> y.shape
    (4,)
    >>>
    >>> # Batch of points
    >>> x_batch = jnp.ones((8, 5))  # 8 points in 5-dim ambient space
    >>> y_batch = hyp_relu(x_batch, c=1.0)
    >>> y_batch.shape
    (8, 5)
    >>>
    >>> # Multi-dimensional batch (e.g., feature maps)
    >>> x_feature = jnp.ones((4, 16, 16, 10))  # 4 images, 16x16 spatial, 10-dim
    >>> y_feature = hyp_relu(x_feature, c=1.0)
    >>> y_feature.shape
    (4, 16, 16, 10)
    """
    return hrc_relu(x, c_in=c, c_out=c)


def hyp_leaky_relu(
    x: Float[Array, "... dim_plus_1"], c: float, negative_slope: float = 0.01
) -> Float[Array, "... dim_plus_1"]:
    """Apply LeakyReLU activation to space components of hyperboloid point(s).

    Curvature-preserving wrapper around hrc_leaky_relu(x, c_in=c, c_out=c, negative_slope).

    This function applies the LeakyReLU activation function to the spatial
    components of hyperboloid points and reconstructs valid manifold points
    using the hyperboloid constraint.

    Mathematical formula:
        y = [sqrt(||LeakyReLU(x_s)||^2 + 1/c), LeakyReLU(x_s)]

    where x_s are the spatial components x[..., 1:], and
    LeakyReLU(x) = x if x > 0 else negative_slope * x.

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold in ambient space, where
        ... represents arbitrary batch dimensions. The last dimension contains
        the time component (x[..., 0]) and spatial components (x[..., 1:]).
    c : float
        Curvature parameter, must be positive (c > 0).
    negative_slope : float, optional
        Negative slope coefficient for LeakyReLU (default: 0.01).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold, same shape as input.

    Notes
    -----
    - This function applies LeakyReLU only to spatial components
    - The time component is reconstructed using the hyperboloid constraint
    - LeakyReLU allows small negative values (scaled by negative_slope) which
      can help gradient flow compared to standard ReLU
    - Works on arrays of any shape, similar to jax.nn.leaky_relu
    - For curvature-changing transformations, use `hrc_leaky_relu` which
      supports different input/output curvatures

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic
    convolutional neural networks for computer vision." arXiv preprint
    arXiv:2303.15919 (2023).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers import hyp_leaky_relu
    >>>
    >>> # Single point with default negative_slope
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> y = hyp_leaky_relu(x, c=1.0)
    >>> y.shape
    (4,)
    >>>
    >>> # Custom negative_slope
    >>> y = hyp_leaky_relu(x, c=1.0, negative_slope=0.1)
    >>>
    >>> # Batch of points
    >>> x_batch = jnp.ones((8, 5))
    >>> y_batch = hyp_leaky_relu(x_batch, c=1.0, negative_slope=0.01)
    >>> y_batch.shape
    (8, 5)
    """
    return hrc_leaky_relu(x, c_in=c, c_out=c, negative_slope=negative_slope)


def hyp_tanh(x: Float[Array, "... dim_plus_1"], c: float) -> Float[Array, "... dim_plus_1"]:
    """Apply tanh activation to space components of hyperboloid point(s).

    Curvature-preserving wrapper around hrc_tanh(x, c_in=c, c_out=c).

    This function applies the hyperbolic tangent activation function to the
    spatial components of hyperboloid points and reconstructs valid manifold
    points using the hyperboloid constraint.

    Mathematical formula:
        y = [sqrt(||tanh(x_s)||^2 + 1/c), tanh(x_s)]

    where x_s are the spatial components x[..., 1:].

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold in ambient space, where
        ... represents arbitrary batch dimensions. The last dimension contains
        the time component (x[..., 0]) and spatial components (x[..., 1:]).
    c : float
        Curvature parameter, must be positive (c > 0).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold, same shape as input.

    Notes
    -----
    - This function applies tanh only to spatial components
    - The time component is reconstructed using the hyperboloid constraint
    - Tanh naturally bounds outputs in [-1, 1], which can help with stability
    - Works on arrays of any shape, similar to jax.nn.tanh
    - For curvature-changing transformations, use `hrc_tanh` which supports
      different input/output curvatures

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic
    convolutional neural networks for computer vision." arXiv preprint
    arXiv:2303.15919 (2023).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers import hyp_tanh
    >>>
    >>> # Single point
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> y = hyp_tanh(x, c=1.0)
    >>> y.shape
    (4,)
    >>>
    >>> # Batch of points
    >>> x_batch = jnp.ones((8, 5))
    >>> y_batch = hyp_tanh(x_batch, c=1.0)
    >>> y_batch.shape
    (8, 5)
    >>>
    >>> # Verify spatial components are bounded
    >>> import jax
    >>> assert jnp.all(jnp.abs(y_batch[..., 1:]) <= 1.0)
    """
    return hrc_tanh(x, c_in=c, c_out=c)


def hyp_swish(x: Float[Array, "... dim_plus_1"], c: float) -> Float[Array, "... dim_plus_1"]:
    """Apply Swish/SiLU activation to space components of hyperboloid point(s).

    Curvature-preserving wrapper around hrc_swish(x, c_in=c, c_out=c).

    This function applies the Swish (also known as SiLU) activation function
    to the spatial components of hyperboloid points and reconstructs valid
    manifold points using the hyperboloid constraint.

    Swish is defined as: swish(x) = x * sigmoid(x)

    Mathematical formula:
        y = [sqrt(||swish(x_s)||^2 + 1/c), swish(x_s)]

    where x_s are the spatial components x[..., 1:].

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold in ambient space, where
        ... represents arbitrary batch dimensions. The last dimension contains
        the time component (x[..., 0]) and spatial components (x[..., 1:]).
    c : float
        Curvature parameter, must be positive (c > 0).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold, same shape as input.

    Notes
    -----
    - This function applies Swish only to spatial components
    - The time component is reconstructed using the hyperboloid constraint
    - Swish is smooth and non-monotonic, often performing well in deep networks
    - Works on arrays of any shape, similar to jax.nn.swish
    - For curvature-changing transformations, use `hrc_swish` which supports
      different input/output curvatures

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic
    convolutional neural networks for computer vision." arXiv preprint
    arXiv:2303.15919 (2023).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers import hyp_swish
    >>>
    >>> # Single point
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> y = hyp_swish(x, c=1.0)
    >>> y.shape
    (4,)
    >>>
    >>> # Batch of points
    >>> x_batch = jnp.ones((8, 5))
    >>> y_batch = hyp_swish(x_batch, c=1.0)
    >>> y_batch.shape
    (8, 5)
    >>>
    >>> # Multi-dimensional batch
    >>> x_feature = jnp.ones((4, 16, 16, 10))
    >>> y_feature = hyp_swish(x_feature, c=1.0)
    >>> y_feature.shape
    (4, 16, 16, 10)
    """
    return hrc_swish(x, c_in=c, c_out=c)


def hyp_gelu(x: Float[Array, "... dim_plus_1"], c: float) -> Float[Array, "... dim_plus_1"]:
    """Apply GELU activation to space components of hyperboloid point(s).

    Curvature-preserving wrapper around hrc_gelu(x, c_in=c, c_out=c).

    This function applies the Gaussian Error Linear Unit (GELU) activation function
    to the spatial components of hyperboloid points and reconstructs valid manifold
    points using the hyperboloid constraint.

    Mathematical formula:
        y = [sqrt(||GELU(x_s)||^2 + 1/c), GELU(x_s)]

    where x_s are the spatial components x[..., 1:].

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold in ambient space, where
        ... represents arbitrary batch dimensions. The last dimension contains
        the time component (x[..., 0]) and spatial components (x[..., 1:]).
    c : float
        Curvature parameter, must be positive (c > 0).

    Returns
    -------
    y : Array of shape (..., dim+1)
        Output point(s) on the hyperboloid manifold, same shape as input.

    Notes
    -----
    - This function applies GELU only to spatial components
    - The time component is reconstructed using the hyperboloid constraint
    - GELU is smooth and commonly used in transformer architectures
    - Works on arrays of any shape, similar to jax.nn.gelu
    - For curvature-changing transformations, use `hrc_gelu` which supports
      different input/output curvatures

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers import hyp_gelu
    >>>
    >>> # Single point
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> y = hyp_gelu(x, c=1.0)
    >>> y.shape
    (4,)
    >>>
    >>> # Batch of points
    >>> x_batch = jnp.ones((8, 5))
    >>> y_batch = hyp_gelu(x_batch, c=1.0)
    >>> y_batch.shape
    (8, 5)
    >>>
    >>> # Multi-dimensional batch
    >>> x_feature = jnp.ones((4, 16, 16, 10))
    >>> y_feature = hyp_gelu(x_feature, c=1.0)
    >>> y_feature.shape
    (4, 16, 16, 10)
    """
    return hrc_gelu(x, c_in=c, c_out=c)
