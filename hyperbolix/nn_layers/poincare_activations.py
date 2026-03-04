"""Poincaré ball activation functions for JAX.

This module implements activation functions for the Poincaré ball model that apply
activations in the tangent space at the origin via logarithmic and exponential maps.

The Poincaré version of a pointwise nonlinearity f is:
    f_P = exp_0^c o f o log_0^c

where exp_0^c and log_0^c are the exponential and logarithmic maps at the origin
of the Poincaré ball with curvature c.

References
----------
van Spengler et al. "Poincaré ResNet." ICML 2023.
Shimizu et al. "Hyperbolic neural networks++." arXiv:2006.08210 (2020).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from hyperbolix.manifolds.poincare import Poincare

# Module-level manifold instance for the activation functions.
# Uses float64 for numerical stability (standard for hyperbolic operations).
_poincare = Poincare(dtype=jnp.float64)


def _apply_in_tangent_space(
    x: Float[Array, "... dim"],
    activation_fn,
    c: float,
) -> Float[Array, "... dim"]:
    """Apply an activation function in the tangent space at the origin.

    Implements f_P = exp_0^c o f o log_0^c for arbitrary batch dimensions.

    Parameters
    ----------
    x : Array of shape (..., dim)
        Input point(s) on the Poincaré ball.
    activation_fn : callable
        Pointwise activation function to apply in tangent space.
    c : float
        Curvature parameter (positive).

    Returns
    -------
    y : Array of shape (..., dim)
        Output point(s) on the Poincaré ball.
    """
    # Flatten all leading dims: (..., dim) -> (N, dim)
    orig_shape = x.shape
    dim = orig_shape[-1]
    x_flat = x.reshape(-1, dim)  # (N, dim)

    # logmap_0: manifold -> tangent space at origin
    t_flat = jax.vmap(_poincare.logmap_0, in_axes=(0, None))(x_flat, c)  # (N, dim)

    # Apply activation in tangent space
    t_act = activation_fn(t_flat)  # (N, dim)

    # expmap_0: tangent space -> manifold
    y_flat = jax.vmap(_poincare.expmap_0, in_axes=(0, None))(t_act, c)  # (N, dim)

    return y_flat.reshape(orig_shape)


def poincare_relu(
    x: Float[Array, "... dim"],
    c: float,
) -> Float[Array, "... dim"]:
    """Poincaré ReLU activation: exp_0^c ∘ ReLU ∘ log_0^c.

    Applies ReLU in the tangent space at the origin of the Poincaré ball,
    then maps back to the manifold. This is the standard nonlinearity for
    Poincaré ball neural networks.

    Parameters
    ----------
    x : Array of shape (..., dim)
        Input point(s) on the Poincaré ball. Supports arbitrary batch
        dimensions (e.g., (batch, H, W, channels) for feature maps).
    c : float
        Curvature parameter (positive).

    Returns
    -------
    y : Array of shape (..., dim)
        Output point(s) on the Poincaré ball.

    References
    ----------
    van Spengler et al. "Poincaré ResNet." ICML 2023.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers import poincare_relu
    >>>
    >>> # Single point
    >>> x = jnp.array([0.1, -0.2, 0.15])
    >>> y = poincare_relu(x, c=1.0)
    >>> y.shape
    (3,)
    >>>
    >>> # Batch of feature maps
    >>> x_batch = jnp.ones((4, 14, 14, 8)) * 0.1
    >>> y_batch = poincare_relu(x_batch, c=1.0)
    >>> y_batch.shape
    (4, 14, 14, 8)
    """
    return _apply_in_tangent_space(x, jax.nn.relu, c)


def poincare_leaky_relu(
    x: Float[Array, "... dim"],
    c: float,
    negative_slope: float = 0.01,
) -> Float[Array, "... dim"]:
    """Poincaré LeakyReLU activation: exp_0^c ∘ LeakyReLU ∘ log_0^c.

    Parameters
    ----------
    x : Array of shape (..., dim)
        Input point(s) on the Poincaré ball.
    c : float
        Curvature parameter (positive).
    negative_slope : float, optional
        Negative slope coefficient (default: 0.01).

    Returns
    -------
    y : Array of shape (..., dim)
        Output point(s) on the Poincaré ball.
    """

    def f(z):
        return jax.nn.leaky_relu(z, negative_slope)

    return _apply_in_tangent_space(x, f, c)


def poincare_tanh(
    x: Float[Array, "... dim"],
    c: float,
) -> Float[Array, "... dim"]:
    """Poincaré tanh activation: exp_0^c ∘ tanh ∘ log_0^c.

    Parameters
    ----------
    x : Array of shape (..., dim)
        Input point(s) on the Poincaré ball.
    c : float
        Curvature parameter (positive).

    Returns
    -------
    y : Array of shape (..., dim)
        Output point(s) on the Poincaré ball.
    """
    return _apply_in_tangent_space(x, jnp.tanh, c)
