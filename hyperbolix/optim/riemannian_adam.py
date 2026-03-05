"""Riemannian Adam optimizer for JAX/Flax NNX.

This module implements Riemannian Adam as a standard Optax GradientTransformation.
It automatically detects manifold parameters via metadata and applies appropriate
Riemannian operations (adaptive learning rates on manifolds), while treating
Euclidean parameters with standard Adam.

The optimizer supports:
- Adaptive learning rates with first and second moment estimation
- Parallel transport of first moments (second moments follow PyTorch scalar update)
- Both exponential map (exact) and retraction (first-order approximation)
- Mixed Euclidean/Riemannian parameter optimization

Algorithm (for manifold parameters):
    1. Convert Euclidean gradient to Riemannian gradient: grad = manifold.egrad2rgrad(grad, param, c)
    2. Update first moment: m1 = beta1 * m1 + (1 - beta1) * grad
    3. Update second moment: m2 = beta2 * m2 + (1 - beta2) * <grad, grad>_param
    4. Bias correction: m1_hat = m1 / (1 - beta1^t), m2_hat = m2 / (1 - beta2^t)
    5. Compute direction: direction = m1_hat / (sqrt(m2_hat) + eps)
    6. Move on manifold: new_param = manifold.expmap(-lr * direction, param, c)
    7. Transport moments: m1 = manifold.ptransp(m1, param, new_param, c)
                          # m2 accumulated via tangent inner product, no transport

For Euclidean parameters, standard Adam is applied.

Note: Tensor variables in the update loop match the parameter shape (P),
which varies per leaf. Scalars (lr, count, bias_correction) have no suffix.

References
----------
Bécigneul, Gary, and Octavian-Eugen Ganea. "Riemannian adaptive optimization methods."
    arXiv preprint arXiv:1810.00760 (2018).
Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
import optax

from ._riemannian_base import make_riemannian_optimizer


class RAdamState(NamedTuple):
    """State for Riemannian Adam optimizer.

    Attributes
    ----------
    m1 : Any
        Pytree of first moment estimates (exponential moving average of gradients)
    m2 : Any
        Pytree of second moment estimates (exponential moving average of squared gradients)
    count : Array
        Step count for bias correction
    """

    m1: Any
    m2: Any
    count: jnp.ndarray


def riemannian_adam(
    learning_rate: float | optax.Schedule,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    use_expmap: bool = True,
) -> optax.GradientTransformation:
    """Create a Riemannian Adam optimizer as an Optax GradientTransformation.

    This optimizer automatically detects manifold parameters via metadata and
    applies Riemannian operations with adaptive learning rates, while treating
    Euclidean parameters with standard Adam.

    Parameters
    ----------
    learning_rate : float or optax.Schedule
        Learning rate (static or scheduled)
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates
    eps : float, default=1e-8
        Small constant for numerical stability
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
    >>> from hyperbolix.optim import riemannian_adam
    >>> from hyperbolix.nn_layers import HypLinearPoincare
    >>> from hyperbolix.manifolds import poincare
    >>>
    >>> # Create model with manifold parameters
    >>> layer = HypLinearPoincare(poincare, 10, 5, rngs=nnx.Rngs(0))
    >>>
    >>> # Create Riemannian Adam optimizer
    >>> tx = riemannian_adam(learning_rate=0.001, use_expmap=True)
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
    - First moments are parallel transported for manifold parameters (matching PyTorch behaviour)
    - Second moments follow Geoopt/PyTorch: accumulated as tangent inner products without transport
    - Works seamlessly with nnx.Optimizer wrapper
    - Parameters stay on manifold after updates (via expmap/retraction + projection)
    """

    def manifold_leaf_fn(rgrad, moments, param_value, manifold_module, c, lr, count):
        m1_value, m2_value = moments

        # First moment: EMA of Riemannian gradients
        new_m1 = beta1 * m1_value + (1 - beta1) * rgrad

        # Second moment: EMA of tangent inner product (scalar broadcast to param shape)
        rgrad_sq = manifold_module.tangent_inner(rgrad, rgrad, param_value, c)
        rgrad_sq = jnp.asarray(rgrad_sq, dtype=rgrad.dtype)
        rgrad_sq = jnp.broadcast_to(rgrad_sq, m2_value.shape)
        new_m2 = beta2 * m2_value + (1 - beta2) * rgrad_sq

        # Bias correction
        bias_correction1 = 1 - beta1**count
        bias_correction2 = 1 - beta2**count
        m1_hat = new_m1 / bias_correction1
        m2_hat = new_m2 / bias_correction2

        # Direction (expmap/retraction applied by base)
        lr_cast = lr.astype(m1_hat.dtype)
        direction = -lr_cast * m1_hat / (jnp.sqrt(m2_hat) + eps)

        # Parallel transport m1 only (index 0); m2 stays (no transport)
        return direction, (new_m1, new_m2), (0,)

    def euclidean_leaf_fn(grad_value, moments, lr, count):
        m1_value, m2_value = moments

        new_m1 = beta1 * m1_value + (1 - beta1) * grad_value
        new_m2 = beta2 * m2_value + (1 - beta2) * (grad_value**2)

        bias_correction1 = 1 - beta1**count
        bias_correction2 = 1 - beta2**count
        m1_hat = new_m1 / bias_correction1
        m2_hat = new_m2 / bias_correction2

        lr_cast = lr.astype(m1_hat.dtype)
        param_update = -lr_cast * m1_hat / (jnp.sqrt(m2_hat) + eps)

        return param_update, (new_m1, new_m2)

    return make_riemannian_optimizer(
        n_moments=2,
        state_cls=RAdamState,
        manifold_leaf_fn=manifold_leaf_fn,
        euclidean_leaf_fn=euclidean_leaf_fn,
        learning_rate=learning_rate,
        use_expmap=use_expmap,
    )
