"""Input-model routing shared by :class:`HoroPCA` and :class:`CoSNE`.

One table maps each supported manifold type to the maps both classes route through: hygiene on
the raw input, the direct isometry onto the hyperboloid (HoroPCA's working space), the one onto
the Poincaré ball (CO-SNE's), and the one that returns the K-dimensional Poincaré result in the
input's model. ``None`` means the identity. The isometries do not project, so the hygiene runs
first (none is needed for ProperVelocity, which is all of ℝᵈ).

Dimension key:
  N: number of points
  R: input representation dim (ambient D + 1 for Hyperboloid, spatial D otherwise)
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
from jaxtyping import Array, Float

from ..manifolds import HalfSpace, Hyperboloid, Klein, Manifold, Poincare, ProperVelocity
from ..manifolds import isometry_mappings as iso
from ..manifolds._gyrovector_core import _proj_batch as _proj_batch_ball
from ..manifolds.halfspace import _proj as _proj_halfspace
from ..manifolds.hyperboloid import _proj_batch as _proj_batch_hyperboloid
from ..manifolds.protocol import ScalarCurvature

type _PointMap = Callable[[Float[Array, "R"], ScalarCurvature], Float[Array, "R2"]]
type _BatchMap = Callable[[Float[Array, "N R"], ScalarCurvature], Float[Array, "N R"]]


class _InputModel(NamedTuple):
    hygiene: _BatchMap | None  # batched clean-up onto the input model
    to_hyperboloid: _PointMap | None  # input model → hyperboloid
    to_ball: _PointMap | None  # input model → Poincaré ball
    from_ball: _PointMap | None  # Poincaré ball → input model
    has_time: bool  # representation carries the time coordinate (R = D + 1)


def _halfspace_hygiene(x_NR: Float[Array, "N R"], c: ScalarCurvature) -> Float[Array, "N R"]:
    del c
    return _proj_halfspace(x_NR)


_INPUT_MODELS: tuple[tuple[type, _InputModel], ...] = (
    (
        Hyperboloid,
        _InputModel(_proj_batch_hyperboloid, None, iso.hyperboloid_to_poincare, iso.poincare_to_hyperboloid, True),
    ),
    (Poincare, _InputModel(_proj_batch_ball, iso.poincare_to_hyperboloid, None, None, False)),
    (ProperVelocity, _InputModel(None, iso.pv_to_hyperboloid, iso.pv_to_poincare, iso.poincare_to_pv, False)),
    (Klein, _InputModel(_proj_batch_ball, iso.klein_to_hyperboloid, iso.klein_to_poincare, iso.poincare_to_klein, False)),
    (
        HalfSpace,
        _InputModel(
            _halfspace_hygiene, iso.halfspace_to_hyperboloid, iso.halfspace_to_poincare, iso.poincare_to_halfspace, False
        ),
    ),
)

SUPPORTED_MODELS = "'Poincare', 'Hyperboloid', 'ProperVelocity', 'Klein' or 'HalfSpace'"


def input_model(manifold: Manifold, owner: str) -> _InputModel:
    """Look up the routing for ``manifold``; raise ``ValueError`` for an unsupported model."""
    for cls, model in _INPUT_MODELS:
        if isinstance(manifold, cls):
            return model
    raise ValueError(f"{owner} supports {SUPPORTED_MODELS} manifolds, got {type(manifold).__name__}.")


def map_batch(point_map: _PointMap | None, x_NR: Float[Array, "N R"], c: ScalarCurvature) -> Array:
    """Apply a single-point map over the leading axis (identity for ``None``)."""
    if point_map is None:
        return x_NR
    return jax.vmap(point_map, in_axes=(0, None))(x_NR, c)


def clean(model: _InputModel, x_NR: Float[Array, "N R"], c: ScalarCurvature) -> Float[Array, "N R"]:
    """Run the model's hygiene (identity when it needs none)."""
    return x_NR if model.hygiene is None else model.hygiene(x_NR, c)
