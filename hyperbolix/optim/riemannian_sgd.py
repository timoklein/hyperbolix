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

References
----------
Bécigneul, Gary, and Octavian-Eugen Ganea. "Riemannian adaptive optimization methods."
    arXiv preprint arXiv:1810.00760 (2018).
"""

from typing import Any, NamedTuple, cast

import jax.numpy as jnp
import optax
from flax import nnx
from jax import tree_util

from .manifold_metadata import get_manifold_info


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

    def init_fn(params: Any) -> RSGDState:
        """Initialize optimizer state.

        Parameters
        ----------
        params : Any
            Pytree of parameters

        Returns
        -------
        state : RSGDState
            Initial optimizer state with zero momentum
        """
        momentum = tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        count = jnp.zeros([], jnp.int32)
        return RSGDState(momentum=momentum, count=count)

    def update_fn(
        updates: Any,
        state: RSGDState,
        params: Any | None = None,
    ) -> tuple[Any, RSGDState]:
        """Apply Riemannian SGD update.

        Parameters
        ----------
        updates : Any
            Pytree of gradients (typically from nnx.grad)
        state : RSGDState
            Current optimizer state
        params : Any, optional
            Pytree of parameters (required for Riemannian operations)

        Returns
        -------
        updates : Any
            Pytree of parameter updates (new_param - old_param)
        new_state : RSGDState
            Updated optimizer state

        Raises
        ------
        ValueError
            If params is None (required for Riemannian operations)
        """
        if params is None:
            raise ValueError("Riemannian SGD requires params to be provided in update step")

        # Increment step count
        count_inc = state.count + 1

        # Get learning rate (handle both static value and schedule)
        if callable(learning_rate):
            lr_value = learning_rate(count_inc)
        else:
            lr_value = cast(float, learning_rate)
        lr = jnp.asarray(lr_value)

        # We need to traverse the params pytree to access Variable metadata
        # while simultaneously mapping over the gradients and momentum
        def update_single_leaf(grad_value, mom_value, param_variable):
            """Update a single leaf parameter.

            Note: param_variable may be an nnx.Variable or a plain array depending on
            how nnx.Optimizer structures the pytree.
            """
            # Check if this parameter has manifold metadata
            manifold_info = None
            param_value = grad_value  # Default: use grad structure

            if hasattr(param_variable, "_var_metadata"):
                # param_variable is an nnx.Variable with potential metadata
                manifold_info = get_manifold_info(param_variable)
                param_value = param_variable.value if hasattr(param_variable, "value") else param_variable

            if manifold_info is not None:
                # Riemannian parameter update
                manifold_module, c = manifold_info

                # 1. Convert Euclidean gradient to Riemannian gradient
                rgrad = manifold_module.egrad2rgrad(grad_value, param_value, c)

                # 2. Update momentum
                new_momentum = momentum * mom_value + rgrad

                # 3. Move on manifold using exponential map or retraction
                lr_cast = lr.astype(new_momentum.dtype)
                direction = -lr_cast * new_momentum
                if use_expmap:
                    new_param_value = manifold_module.expmap(direction, param_value, c)
                else:
                    new_param_value = manifold_module.retraction(direction, param_value, c)

                # 4. Parallel transport momentum to new location
                if momentum > 0.0:
                    transported_momentum = manifold_module.ptransp(new_momentum, param_value, new_param_value, c)
                else:
                    transported_momentum = new_momentum

                # Return the update (new - old) and transported momentum
                param_update = new_param_value - param_value
                return (param_update, transported_momentum)

            else:
                # Euclidean parameter update (standard SGD with momentum)
                # momentum update: m = momentum * m + grad
                new_momentum = momentum * mom_value + grad_value

                # parameter update: param = param - lr * m
                lr_cast = lr.astype(new_momentum.dtype)
                param_update = -lr_cast * new_momentum

                return (param_update, new_momentum)

        # Apply update to all parameters
        # Use tree_map with is_leaf to handle nnx.Variable nodes correctly
        results = tree_util.tree_map(
            update_single_leaf,
            updates,  # gradients (arrays)
            state.momentum,  # old momentum (arrays)
            params,  # parameters (Variables with metadata)
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )

        # Separate the tuple results
        # Each leaf in results is a tuple (param_update, new_momentum)
        # Use is_leaf to treat tuples as leaves so we can extract their components
        param_updates = tree_util.tree_map(
            lambda r: r[0],
            results,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        new_momentum = tree_util.tree_map(
            lambda r: r[1],
            results,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        new_state = RSGDState(momentum=new_momentum, count=count_inc)

        return param_updates, new_state

    return optax.GradientTransformation(init_fn, cast(Any, update_fn))
