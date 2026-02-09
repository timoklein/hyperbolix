"""Core Hypformer operations for hyperboloid manifolds.

This module contains the foundational HRC (Hyperbolic Regularization Component) and
HTC (Hyperbolic Transformation Component) operations from the Hypformer paper. These
are the building blocks used throughout the library for creating hyperbolic neural
network layers with curvature-change support.

Key Components
--------------
- **hrc**: Wraps Euclidean operations on spatial components only
- **htc**: Wraps Euclidean operations on full hyperboloid points

Both functions enable curvature transformations (c_in → c_out) and avoid expensive
exp/log maps by using constraint-based time reconstruction.

References
----------
Hypformer paper (citation to be added)
"""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float


def lorentz_residual(
    x: Float[Array, "... dim_plus_1"],
    y: Float[Array, "... dim_plus_1"],
    w_y: float | Float[Array, ""],
    c: float,
    eps: float = 1e-7,
) -> Float[Array, "... dim_plus_1"]:
    """Lorentzian midpoint-based residual connection (LResNet from HELM).

    Computes the weighted Lorentzian midpoint of x and y, projecting back
    to the hyperboloid:

        ave = x + w_y * y
        result = ave / sqrt(c * |<ave, ave>_L|)

    where <a, a>_L = -a_0^2 + ||a_s||^2 is the Minkowski inner product.

    Parameters
    ----------
    x : Array, shape (..., d+1)
        Points on hyperboloid with curvature c.
    y : Array, shape (..., d+1)
        Points on hyperboloid with curvature c (to be added with weight w_y).
    w_y : float or scalar Array
        Weight for the y contribution.
    c : float
        Curvature parameter (positive, c > 0).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    Returns
    -------
    Array, shape (..., d+1)
        Points on hyperboloid with curvature c.

    References
    ----------
    Chen et al., "Hyperbolic Embeddings for Learning on Manifolds" (HELM), 2024.
    """
    ave = x + w_y * y  # (..., d+1)
    # Minkowski inner: -ave_0^2 + ||ave_s||^2
    mink = -(ave[..., 0:1] ** 2) + jnp.sum(ave[..., 1:] ** 2, axis=-1, keepdims=True)  # (..., 1)
    denom = jnp.sqrt(jnp.maximum(c * jnp.abs(mink), eps))  # (..., 1)
    return ave / denom  # (..., d+1)


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
    which is the pattern used by curvature-preserving hyperboloid activations.

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

    See Also
    --------
    htc : Hyperbolic Transformation Component for full-point operations.

    References
    ----------
    Hypformer paper (citation to be added)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers.hyperboloid_core import hrc
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

    See Also
    --------
    hrc : Hyperbolic Regularization Component for spatial-only operations.
    HTCLinear : Module wrapper for htc with learnable linear transformation.

    References
    ----------
    Hypformer paper (citation to be added)

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers.hyperboloid_core import htc
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
