"""Shared scaffolding for Riemannian optimizers.

Provides the common init/update loop for Riemannian optimization:
- Learning rate resolution (static or schedule)
- Pytree traversal with manifold detection
- Euclidean gradient → Riemannian gradient conversion
- Exponential map / retraction for manifold moves
- Parallel transport of first moments
- vmap over leading axes so (N, dim) embedding tables work with the
  single-point manifold API

Each optimizer (SGD, Adam) supplies algorithm-specific callbacks
for moment updates and direction computation.

Manifold detection
------------------
``ManifoldParam`` metadata is captured by ``init_fn``, NOT detected inside
``update_fn``. Flax's ``nnx.Optimizer`` (>= 0.12.5) strips Variables to raw
arrays before calling the optax update, so ``isinstance(leaf, ManifoldParam)``
inside ``update_fn`` can never fire on that path — it would silently apply
Euclidean updates to every parameter. ``init_fn`` still receives the intact
Variables, so the (manifold, curvature) info is recorded there, positionally
aligned with the flattened parameter tree, and resolved at update time.
Live Variables in ``update_fn`` (direct ``tx.update(grad, state, param)``
usage) take precedence over the captured metadata.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import tree_util

from .manifold_metadata import ManifoldParam, _is_variable_leaf

# Dimension key:
#   D: manifold point dim (last axis)    N: number of points in a leaf


def _resolve_lr(learning_rate: float | optax.Schedule, count: jnp.ndarray) -> jnp.ndarray:
    """Resolve learning rate from static value or schedule."""
    if callable(learning_rate):
        return jnp.asarray(learning_rate(count))
    return jnp.asarray(cast(float, learning_rate))


def _unwrap(x: Any) -> Any:
    """Unwrap an nnx.Variable to its raw array value (identity for arrays)."""
    return x[...] if isinstance(x, nnx.Variable) else x


def _manifold_info_spec(leaf: Any) -> tuple[Any, Any] | None:
    """(manifold, curvature_spec) for a ManifoldParam leaf, else None.

    The curvature spec is stored *unevaluated* (it may be a callable for
    learnable curvature) and resolved at update time.
    """
    if isinstance(leaf, ManifoldParam):
        return (leaf.manifold, leaf.curvature)
    return None


def _resolve_info(spec: tuple[Any, Any] | None) -> tuple[Any, Any] | None:
    """Evaluate a stored (manifold, curvature_spec) into (manifold, c)."""
    if spec is None:
        return None
    manifold, curvature = spec
    if callable(curvature):
        curvature = curvature()
    return (manifold, curvature)


def _apply_manifold_move(direction, param_value, manifold_module, c, use_expmap):
    """Move parameter on manifold via expmap or retraction."""
    if use_expmap:
        return manifold_module.expmap(direction, param_value, c)
    return manifold_module.retraction(direction, param_value, c)


def _resolve_leaf_infos(param_leaves: list, captured_infos: tuple | None) -> list[tuple[Any, Any] | None]:
    """Per-leaf (manifold, c) info: live Variables first, else init-captured metadata.

    Live Variables appear in direct ``tx.update(grad, state, param)`` usage; the
    nnx.Optimizer path delivers raw arrays and relies on the metadata captured
    by ``init_fn`` (positionally aligned with the flattened parameter tree).
    """
    if any(_is_variable_leaf(leaf) for leaf in param_leaves):
        return [_resolve_info(_manifold_info_spec(leaf)) for leaf in param_leaves]
    if captured_infos is None:
        raise ValueError(
            "Riemannian optimizer received raw array parameters and no Variable "
            "metadata was captured at init. ManifoldParam detection relies on "
            "init_fn seeing the model's nnx.Variables (Flax's nnx.Optimizer strips "
            "them before the optax update). This usually means tx.init() was called "
            "with raw arrays. If this model has no manifold parameters, use a "
            "standard optax optimizer (e.g. optax.adam) instead."
        )
    if len(captured_infos) != len(param_leaves):
        raise ValueError(
            f"Parameter tree has {len(param_leaves)} leaves but manifold metadata was "
            f"captured for {len(captured_infos)} leaves at init. Did you reuse this "
            "optimizer instance with a different model?"
        )
    return [_resolve_info(spec) for spec in captured_infos]


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
        Operates on a SINGLE manifold point of shape (D,); the base vmaps it
        over any leading axes. Returns step direction, updated moments, and
        indices to parallel-transport. ``count`` here is the POST-increment
        step count (1-indexed), for Adam-style bias correction — distinct
        from (and one step ahead of) the count the schedule in ``lr`` was
        resolved against.
    euclidean_leaf_fn : callable
        ``(grad, moments, lr, count) -> (param_update, new_moments)``
        Returns the parameter update and updated moments for Euclidean params.
        ``count`` is the same post-increment step count described above.
    learning_rate : float or optax.Schedule
        Learning rate, resolved at the PRE-increment count (matching optax's
        ``scale_by_schedule``): the first ``update()`` call reads ``schedule(0)``.
    use_expmap : bool
        Use exponential map (True) or retraction (False).

    Returns
    -------
    optax.GradientTransformation

    Notes
    -----
    Do not reuse one GradientTransformation instance across different models:
    ``init_fn`` captures per-leaf manifold metadata for the parameter tree it
    is initialized with, and a second ``init`` overwrites it.
    """
    # Manifold metadata captured at init time (see module docstring). A plain
    # closure cell: nnx.Optimizer calls tx.init eagerly at construction, while
    # the model's Variables are still intact.
    captured: dict[str, tuple | None] = {"infos": None}

    def init_fn(params: Any) -> Any:
        leaves = tree_util.tree_flatten(params, is_leaf=_is_variable_leaf)[0]
        if any(_is_variable_leaf(leaf) for leaf in leaves):
            captured["infos"] = tuple(_manifold_info_spec(leaf) for leaf in leaves)
        else:
            # Raw arrays at init: metadata never existed. update_fn raises if
            # it also sees raw arrays (it cannot tell manifold from Euclidean).
            captured["infos"] = None
        # Moments are stored as raw arrays (not Variables) so the optimizer
        # state stays a plain JIT-friendly pytree.
        moment_trees = tuple(
            tree_util.tree_map(lambda p: jnp.zeros_like(_unwrap(p)), params, is_leaf=_is_variable_leaf)
            for _ in range(n_moments)
        )
        count = jnp.zeros([], jnp.int32)
        return state_cls(*moment_trees, count)

    def _manifold_update(grad_value, moments, param_value, manifold_module, c, lr, count_inc):
        """Riemannian update for one leaf; vmaps single-point ops over leading axes."""

        def point_update(grad_D, param_D, moments_D):
            rgrad_D = manifold_module.egrad2rgrad(grad_D, param_D, c)
            direction_D, new_moments_D, ptransp_indices = manifold_leaf_fn(
                rgrad_D, moments_D, param_D, manifold_module, c, lr, count_inc
            )
            new_param_D = _apply_manifold_move(direction_D, param_D, manifold_module, c, use_expmap)
            final_moments_D = list(new_moments_D)
            for idx in ptransp_indices:
                final_moments_D[idx] = manifold_module.ptransp(new_moments_D[idx], param_D, new_param_D, c)
            # Storage-dtype contract: manifold methods compute in manifold.dtype
            # (egrad2rgrad/expmap/ptransp _cast their inputs), but the returned
            # update must match the param dtype (optax convention; a float32
            # param next to a float64 manifold must not promote) and moment
            # buffers must keep their init dtype (zeros_like(param)) — without
            # the casts, optimizer state silently turns float64 after one step.
            update_D = (new_param_D - param_D).astype(param_D.dtype)
            final_moments_out_D = tuple(m.astype(m_in.dtype) for m, m_in in zip(final_moments_D, moments_D, strict=True))
            return update_D, final_moments_out_D

        if param_value.ndim == 0:
            raise ValueError("Manifold parameters must have at least one dimension (a point), got a scalar.")
        if param_value.ndim == 1:
            return point_update(grad_value, param_value, moments)

        # (..., D) tables: flatten leading axes to N points and vmap.
        dim = param_value.shape[-1]
        grad_ND = grad_value.reshape(-1, dim)
        param_ND = param_value.reshape(-1, dim)
        moments_ND = tuple(m.reshape(-1, dim) for m in moments)
        update_ND, new_moments_ND = jax.vmap(point_update)(grad_ND, param_ND, moments_ND)
        return (
            update_ND.reshape(param_value.shape),
            tuple(m.reshape(param_value.shape) for m in new_moments_ND),
        )

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

        # Schedules are read at the PRE-increment count, matching optax's
        # scale_by_schedule.update_fn (`step_size_fn(state.count)`, then increment).
        # The post-increment `count_inc` is still used below for Adam-style bias
        # correction (`1 - beta**count`), which optax's scale_by_adam evaluates
        # against the incremented count — schedule timing and bias-correction
        # timing are deliberately different counters in optax and here.
        lr = _resolve_lr(learning_rate, state[-1])

        # Flatten all pytrees in lock-step
        grad_leaves, treedef = tree_util.tree_flatten(updates, is_leaf=_is_variable_leaf)
        moment_leaves_list = [tree_util.tree_flatten(m, is_leaf=_is_variable_leaf)[0] for m in moment_states]
        param_leaves = tree_util.tree_flatten(params, is_leaf=_is_variable_leaf)[0]

        infos = _resolve_leaf_infos(param_leaves, captured["infos"])

        n_leaves = len(grad_leaves)
        param_update_leaves = []
        new_moment_leaves_list = [[] for _ in range(n_moments)]

        for i in range(n_leaves):
            grad_value = _unwrap(grad_leaves[i])
            moments = tuple(_unwrap(moment_leaves_list[k][i]) for k in range(n_moments))
            param_value = _unwrap(param_leaves[i])
            manifold_info = infos[i]

            if manifold_info is not None:
                manifold_module, c = manifold_info
                param_update, new_moments = _manifold_update(
                    grad_value, moments, param_value, manifold_module, c, lr, count_inc
                )
            else:
                # Euclidean parameter update
                param_update, new_moments = euclidean_leaf_fn(grad_value, moments, lr, count_inc)

            param_update_leaves.append(param_update)
            for k in range(n_moments):
                new_moment_leaves_list[k].append(new_moments[k])

        param_updates = tree_util.tree_unflatten(treedef, param_update_leaves)
        new_moment_trees = tuple(tree_util.tree_unflatten(treedef, new_moment_leaves_list[k]) for k in range(n_moments))
        new_state = state_cls(*new_moment_trees, count_inc)

        return param_updates, new_state

    return optax.GradientTransformation(init_fn, cast(Any, update_fn))
