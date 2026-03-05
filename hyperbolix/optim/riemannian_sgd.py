"""Riemannian SGD optimizer for JAX/Flax NNX.

This module implements Riemannian Stochastic Gradient Descent (RSGD) as a
standard Optax GradientTransformation. It automatically detects manifold
parameters via metadata and applies appropriate Riemannian operations,
while treating Euclidean parameters with standard SGD.

The optimizer supports:
- Momentum with parallel transport
- Both exponential map (exact) and retraction (first-order approximation)
- Mixed Euclidean/Riemannian parameter optimization

Algorithm (for manifold parameters):
    1. Convert Euclidean gradient to Riemannian gradient: grad = manifold.egrad2rgrad(grad, param, c)
    2. Update momentum: m = momentum * m + grad
    3. Move on manifold: new_param = manifold.expmap(-lr * m, param, c)  # or retraction
    4. Transport momentum: m = manifold.ptransp(m, param, new_param, c)

For Euclidean parameters, standard SGD is applied.

Note: Tensor variables in update_single_leaf match the parameter shape (P),
which varies per leaf. Scalars (lr, count) have no suffix.

References
----------
Bécigneul, Gary, and Octavian-Eugen Ganea. "Riemannian adaptive optimization methods."
    arXiv preprint arXiv:1810.00760 (2018).
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
import optax

from ._riemannian_base import make_riemannian_optimizer


class RSGDState(NamedTuple):
    """State for Riemannian SGD optimizer.

    Attributes
    ----------
    momentum : Any
        Pytree of momentum terms, same structure as parameters
    count : Array
        Step count for schedule handling
    """

    momentum: Any
    count: jnp.ndarray


def riemannian_sgd(
    learning_rate: float | optax.Schedule,
    momentum: float = 0.0,
    use_expmap: bool = True,
) -> optax.GradientTransformation:
    """Create a Riemannian SGD optimizer as an Optax GradientTransformation.

    This optimizer automatically detects manifold parameters via metadata and
    applies Riemannian operations (egrad2rgrad, expmap/retraction, parallel transport),
    while treating Euclidean parameters with standard SGD.

    Parameters
    ----------
    learning_rate : float or optax.Schedule
        Learning rate (static or scheduled)
    momentum : float, default=0.0
        Momentum coefficient (0 for no momentum, typically 0.9)
    use_expmap : bool, default=True
        If True, use exponential map (exact geodesic).
        If False, use retraction (first-order approximation, faster).

    Returns
    -------
    optimizer : optax.GradientTransformation
        An Optax GradientTransformation that can be used with nnx.Optimizer

    Example
    -------
    >>> import jax
    >>> from flax import nnx
    >>> from hyperbolix.optim import riemannian_sgd
    >>> from hyperbolix.nn_layers import HypLinearPoincare
    >>> from hyperbolix.manifolds import poincare
    >>>
    >>> # Create model with manifold parameters
    >>> layer = HypLinearPoincare(poincare, 10, 5, rngs=nnx.Rngs(0))
    >>>
    >>> # Create Riemannian optimizer
    >>> tx = riemannian_sgd(learning_rate=0.01, momentum=0.9, use_expmap=True)
    >>> optimizer = nnx.Optimizer(layer, tx, wrt=nnx.Param)
    >>>
    >>> # Training step
    >>> def loss_fn(model, x):
    ...     y = model(x, c=1.0)
    ...     return jnp.sum(y ** 2)
    >>>
    >>> x = jax.random.normal(jax.random.key(1), (32, 10))
    >>> grads = nnx.grad(loss_fn)(layer, x)
    >>> optimizer.update(grads)  # Automatically handles manifold parameters

    Notes
    -----
    - Compatible with Optax combinators (optax.chain, schedules, etc.)
    - Momentum is parallel transported for manifold parameters
    - Works seamlessly with nnx.Optimizer wrapper
    - Parameters stay on manifold after updates (via expmap/retraction + projection)
    """

    def manifold_leaf_fn(rgrad, moments, param_value, manifold_module, c, lr, count):
        (mom_value,) = moments
        new_mom = momentum * mom_value + rgrad
        lr_cast = lr.astype(new_mom.dtype)
        direction = -lr_cast * new_mom
        # Parallel transport momentum only if momentum > 0
        ptransp_indices = (0,) if momentum > 0.0 else ()
        return direction, (new_mom,), ptransp_indices

    def euclidean_leaf_fn(grad_value, moments, lr, count):
        (mom_value,) = moments
        new_mom = momentum * mom_value + grad_value
        lr_cast = lr.astype(new_mom.dtype)
        param_update = -lr_cast * new_mom
        return param_update, (new_mom,)

    return make_riemannian_optimizer(
        n_moments=1,
        state_cls=RSGDState,
        manifold_leaf_fn=manifold_leaf_fn,
        euclidean_leaf_fn=euclidean_leaf_fn,
        learning_rate=learning_rate,
        use_expmap=use_expmap,
    )
