"""Shared scaffolding for Riemannian optimizers.

Provides the common init/update loop for Riemannian optimization:
- Learning rate resolution (static or schedule)
- Pytree traversal with manifold detection
- Euclidean gradient → Riemannian gradient conversion
- Exponential map / retraction for manifold moves
- Parallel transport of first moments

Each optimizer (SGD, Adam) supplies algorithm-specific callbacks
for moment updates and direction computation.
"""

from typing import Any, cast

import jax.numpy as jnp
import optax
from flax import nnx
from jax import tree_util

from .manifold_metadata import get_manifold_info


def _resolve_lr(learning_rate: float | optax.Schedule, count: jnp.ndarray) -> jnp.ndarray:
    """Resolve learning rate from static value or schedule."""
    if callable(learning_rate):
        return jnp.asarray(learning_rate(count))
    return jnp.asarray(cast(float, learning_rate))


def _extract_param_and_manifold(param_variable):
    """Extract parameter value and manifold info from a pytree leaf.

    Returns:
        (param_value, manifold_info) where manifold_info is (manifold_module, c) or None
    """
    manifold_info = None
    if hasattr(param_variable, "_var_metadata"):
        manifold_info = get_manifold_info(param_variable)
    param_value = param_variable[...] if isinstance(param_variable, nnx.Variable) else param_variable
    return param_value, manifold_info


def _apply_manifold_move(direction, param_value, manifold_module, c, use_expmap):
    """Move parameter on manifold via expmap or retraction."""
    if use_expmap:
        return manifold_module.expmap(direction, param_value, c)
    return manifold_module.retraction(direction, param_value, c)


def make_riemannian_optimizer(
    n_moments: int,
    state_cls: Any,
    manifold_leaf_fn: Any,
    euclidean_leaf_fn: Any,
    learning_rate: float | optax.Schedule,
    use_expmap: bool,
) -> optax.GradientTransformation:
    """Build a Riemannian optimizer GradientTransformation.

    Parameters
    ----------
    n_moments : int
        Number of moment terms per parameter (1 for SGD, 2 for Adam).
    state_cls : NamedTuple subclass
        State class constructor. Called as ``state_cls(*moment_trees, count)``.
    manifold_leaf_fn : callable
        ``(rgrad, moments, param_value, manifold_module, c, lr, count)``
        ``-> (direction, new_moments, ptransp_indices)``
        Returns step direction, updated moments, and indices to parallel-transport.
    euclidean_leaf_fn : callable
        ``(grad, moments, lr, count) -> (param_update, new_moments)``
        Returns the parameter update and updated moments for Euclidean params.
    learning_rate : float or optax.Schedule
        Learning rate.
    use_expmap : bool
        Use exponential map (True) or retraction (False).

    Returns
    -------
    optax.GradientTransformation
    """

    def init_fn(params: Any) -> Any:
        moment_trees = tuple(tree_util.tree_map(lambda p: jnp.zeros_like(p), params) for _ in range(n_moments))
        count = jnp.zeros([], jnp.int32)
        return state_cls(*moment_trees, count)

    def update_fn(
        updates: Any,
        state: Any,
        params: Any | None = None,
    ) -> tuple[Any, Any]:
        if params is None:
            raise ValueError("Riemannian optimizer requires params to be provided in update step")

        # Extract moments and count from state (moments are first n_moments fields, count is last)
        moment_states = tuple(state[i] for i in range(n_moments))
        count_inc = state[-1] + 1

        lr = _resolve_lr(learning_rate, count_inc)

        # Flatten all pytrees in lock-step
        def is_variable_leaf(x):
            return isinstance(x, nnx.Variable)

        grad_leaves, treedef = tree_util.tree_flatten(updates, is_leaf=is_variable_leaf)
        moment_leaves_list = [tree_util.tree_flatten(m, is_leaf=is_variable_leaf)[0] for m in moment_states]
        param_leaves = tree_util.tree_flatten(params, is_leaf=is_variable_leaf)[0]

        n_leaves = len(grad_leaves)
        param_update_leaves = []
        new_moment_leaves_list = [[] for _ in range(n_moments)]

        for i in range(n_leaves):
            grad_value = grad_leaves[i]
            moments = tuple(moment_leaves_list[k][i] for k in range(n_moments))
            param_variable = param_leaves[i]

            param_value, manifold_info = _extract_param_and_manifold(param_variable)

            if manifold_info is not None:
                manifold_module, c = manifold_info

                # Convert Euclidean gradient to Riemannian gradient
                rgrad = manifold_module.egrad2rgrad(grad_value, param_value, c)

                # Algorithm-specific moment update and direction computation
                direction, new_moments, ptransp_indices = manifold_leaf_fn(
                    rgrad, moments, param_value, manifold_module, c, lr, count_inc
                )

                # Move on manifold
                new_param_value = _apply_manifold_move(direction, param_value, manifold_module, c, use_expmap)

                # Parallel transport specified moments
                final_moments = list(new_moments)
                for idx in ptransp_indices:
                    final_moments[idx] = manifold_module.ptransp(new_moments[idx], param_value, new_param_value, c)

                param_update = new_param_value - param_value
                for k in range(n_moments):
                    new_moment_leaves_list[k].append(final_moments[k])
            else:
                # Euclidean parameter update
                param_update, new_moments = euclidean_leaf_fn(grad_value, moments, lr, count_inc)
                for k in range(n_moments):
                    new_moment_leaves_list[k].append(new_moments[k])

            param_update_leaves.append(param_update)

        param_updates = tree_util.tree_unflatten(treedef, param_update_leaves)
        new_moment_trees = tuple(tree_util.tree_unflatten(treedef, new_moment_leaves_list[k]) for k in range(n_moments))
        new_state = state_cls(*new_moment_trees, count_inc)

        return param_updates, new_state

    return optax.GradientTransformation(init_fn, cast(Any, update_fn))
