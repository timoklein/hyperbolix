"""Manifold Protocol for structural typing.

Defines the common interface shared by all concrete manifold classes
(``Poincare``, ``Hyperboloid``, ``ProperVelocity``, ``Euclidean``, and
``ProductManifold``). Use ``Manifold`` as a type hint for any parameter
that accepts an arbitrary manifold instance.

This is a ``typing.Protocol`` -- no classes need to explicitly inherit from it.
Structural subtyping ensures that any object with the right methods is accepted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array, Float

# A single curvature value: a Python float (static, fixed curvature) or a scalar
# jax.Array (traced, e.g. the value returned by calling a `LearnableCurvature`
# module). This is what `Poincare`, `Hyperboloid`, `ProperVelocity`, and
# `Euclidean` accept directly.
ScalarCurvature = float | Float[Array, ""]

# Curvature accepted by manifold methods. Single manifolds take a scalar;
# `ProductManifold` takes a per-factor sequence of length ``n_factors``. The
# union lets a single ``Manifold`` protocol cover both shapes — each
# implementation validates the shape it expects at runtime.
Curvature = ScalarCurvature | Sequence[ScalarCurvature]


@runtime_checkable
class Manifold(Protocol):
    """Structural protocol for manifold classes.

    All concrete manifold classes (``Poincare``, ``Hyperboloid``,
    ``ProperVelocity``, ``Euclidean``, ``ProductManifold``) satisfy this
    protocol without modification. For single manifolds, ``c`` is a scalar
    (``ScalarCurvature``); for ``ProductManifold``, ``c`` is a sequence of
    length ``n_factors`` (one curvature per factor).

    The method signatures use the *minimal common interface* so that
    manifold-specific optional parameters (e.g. ``version_idx``, ``atol``)
    do not break compatibility.
    """

    dtype: Any

    def _cast(self, x: Array) -> Array: ...

    # -- Geometry --------------------------------------------------------
    def proj(self, x: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def dist(self, x: Float[Array, ...], y: Float[Array, ...], c: Curvature) -> Float[Array, ""]: ...

    def dist_0(self, x: Float[Array, ...], c: Curvature) -> Float[Array, ""]: ...

    def addition(self, x: Float[Array, ...], y: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def scalar_mul(self, r: float, x: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    # -- Exponential / logarithmic maps ----------------------------------
    def expmap(self, v: Float[Array, ...], x: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def expmap_0(self, v: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def logmap(self, y: Float[Array, ...], x: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def logmap_0(self, y: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def retraction(self, v: Float[Array, ...], x: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    # -- Transport / tangent space ---------------------------------------
    def ptransp(self, v: Float[Array, ...], x: Float[Array, ...], y: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def ptransp_0(self, v: Float[Array, ...], y: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def tangent_inner(
        self, u: Float[Array, ...], v: Float[Array, ...], x: Float[Array, ...], c: Curvature
    ) -> Float[Array, ""]: ...

    def tangent_norm(self, v: Float[Array, ...], x: Float[Array, ...], c: Curvature) -> Float[Array, ""]: ...

    def egrad2rgrad(self, grad: Float[Array, ...], x: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    def tangent_proj(self, v: Float[Array, ...], x: Float[Array, ...], c: Curvature) -> Float[Array, ...]: ...

    # -- Validation ------------------------------------------------------
    def is_in_manifold(self, x: Float[Array, ...], c: Curvature) -> Array: ...

    def is_in_tangent_space(self, v: Float[Array, ...], x: Float[Array, ...], c: Curvature) -> Array: ...
