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

References
----------
Bécigneul, Gary, and Octavian-Eugen Ganea. "Riemannian adaptive optimization methods."
    arXiv preprint arXiv:1810.00760 (2018).
Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
"""

from typing import Any, NamedTuple, cast

import jax.numpy as jnp
import optax
from flax import nnx
from jax import tree_util

from .manifold_metadata import get_manifold_info


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

    def init_fn(params: Any) -> RAdamState:
        """Initialize optimizer state.

        Parameters
        ----------
        params : Any
            Pytree of parameters

        Returns
        -------
        state : RAdamState
            Initial optimizer state with zero moments and count
        """
        # Initialize moments as zeros with same structure as params
        m1 = tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        m2 = tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        count = jnp.zeros([], jnp.int32)
        return RAdamState(m1=m1, m2=m2, count=count)

    def update_fn(
        updates: Any,
        state: RAdamState,
        params: Any | None = None,
    ) -> tuple[Any, RAdamState]:
        """Apply Riemannian Adam update.

        Parameters
        ----------
        updates : Any
            Pytree of gradients (typically from nnx.grad)
        state : RAdamState
            Current optimizer state
        params : Any, optional
            Pytree of parameters (required for Riemannian operations)

        Returns
        -------
        updates : Any
            Pytree of parameter updates (new_param - old_param)
        new_state : RAdamState
            Updated optimizer state

        Raises
        ------
        ValueError
            If params is None (required for Riemannian operations)
        """
        if params is None:
            raise ValueError("Riemannian Adam requires params to be provided in update step")

        # Increment step count
        count_inc = state.count + 1

        # Get learning rate (handle both static value and schedule)
        if callable(learning_rate):
            lr_value = learning_rate(count_inc)
        else:
            lr_value = cast(float, learning_rate)
        lr = jnp.asarray(lr_value)

        # Bias correction terms
        bias_correction1 = 1 - beta1**count_inc
        bias_correction2 = 1 - beta2**count_inc

        # Flatten the pytrees so we can process leaves in lock-step while preserving structure.
        def is_variable_leaf(x):
            return isinstance(x, nnx.Variable)

        grad_leaves, treedef = tree_util.tree_flatten(updates, is_leaf=is_variable_leaf)
        m1_leaves, treedef_m1 = tree_util.tree_flatten(state.m1, is_leaf=is_variable_leaf)
        m2_leaves, treedef_m2 = tree_util.tree_flatten(state.m2, is_leaf=is_variable_leaf)
        param_leaves, treedef_params = tree_util.tree_flatten(params, is_leaf=is_variable_leaf)

        if not (treedef == treedef_m1 == treedef_m2 == treedef_params):
            raise ValueError("Gradient, moment, and parameter pytrees must share the same structure.")

        param_update_leaves = []
        new_m1_leaves = []
        new_m2_leaves = []

        for grad_value, m1_value, m2_value, param_variable in zip(
            grad_leaves, m1_leaves, m2_leaves, param_leaves, strict=True
        ):
            # Default to treating parameters as Euclidean tensors
            manifold_info = None
            if hasattr(param_variable, "_var_metadata"):
                manifold_info = get_manifold_info(param_variable)
                param_value = param_variable[...] if isinstance(param_variable, nnx.Variable) else param_variable
            else:
                param_value = param_variable[...] if isinstance(param_variable, nnx.Variable) else param_variable

            if manifold_info is not None:
                manifold_module, c = manifold_info

                rgrad = manifold_module.egrad2rgrad(grad_value, param_value, c)
                new_m1 = beta1 * m1_value + (1 - beta1) * rgrad

                rgrad_sq = manifold_module.tangent_inner(rgrad, rgrad, param_value, c)
                rgrad_sq = jnp.asarray(rgrad_sq, dtype=rgrad.dtype)
                rgrad_sq = jnp.broadcast_to(rgrad_sq, m2_value.shape)
                new_m2 = beta2 * m2_value + (1 - beta2) * rgrad_sq

                m1_hat = new_m1 / bias_correction1
                m2_hat = new_m2 / bias_correction2
                direction = m1_hat / (jnp.sqrt(m2_hat) + eps)

                lr_cast = lr.astype(direction.dtype)
                step = -lr_cast * direction
                if use_expmap:
                    new_param_value = manifold_module.expmap(step, param_value, c)
                else:
                    new_param_value = manifold_module.retraction(step, param_value, c)

                transported_m1 = manifold_module.ptransp(new_m1, param_value, new_param_value, c)
                transported_m2 = new_m2

                param_update = new_param_value - param_value
            else:
                new_m1 = beta1 * m1_value + (1 - beta1) * grad_value
                new_m2 = beta2 * m2_value + (1 - beta2) * (grad_value**2)

                m1_hat = new_m1 / bias_correction1
                m2_hat = new_m2 / bias_correction2

                lr_cast = lr.astype(m1_hat.dtype)
                param_update = -lr_cast * m1_hat / (jnp.sqrt(m2_hat) + eps)
                transported_m1 = new_m1
                transported_m2 = new_m2

            param_update_leaves.append(param_update)
            new_m1_leaves.append(transported_m1)
            new_m2_leaves.append(transported_m2)

        param_updates = tree_util.tree_unflatten(treedef, param_update_leaves)
        new_m1 = tree_util.tree_unflatten(treedef, new_m1_leaves)
        new_m2 = tree_util.tree_unflatten(treedef, new_m2_leaves)

        new_state = RAdamState(m1=new_m1, m2=new_m2, count=count_inc)

        return param_updates, new_state

    return optax.GradientTransformation(init_fn, cast(Any, update_fn))
