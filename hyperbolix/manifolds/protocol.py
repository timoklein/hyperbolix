"""Manifold Protocol for structural typing.

Defines the common interface shared by Poincare, Hyperboloid, and Euclidean
manifold classes. Use ``Manifold`` as a type hint for any parameter that
accepts an arbitrary manifold instance.

This is a ``typing.Protocol`` -- no classes need to explicitly inherit from it.
Structural subtyping ensures that any object with the right methods is accepted.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float


@runtime_checkable
class Manifold(Protocol):
    """Structural protocol for manifold classes.

    All three concrete manifold classes (``Poincare``, ``Hyperboloid``,
    ``Euclidean``) satisfy this protocol without modification.

    The method signatures use the *minimal common interface* so that
    manifold-specific optional parameters (e.g. ``version_idx``, ``atol``)
    do not break compatibility.
    """

    dtype: object

    def _cast(self, x: Array) -> Array: ...

    # -- Geometry --------------------------------------------------------
    def proj(self, x: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def dist(self, x: Float[Array, ...], y: Float[Array, ...], c: float) -> Float[Array, ""]: ...

    def dist_0(self, x: Float[Array, ...], c: float) -> Float[Array, ""]: ...

    def addition(self, x: Float[Array, ...], y: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def scalar_mul(self, r: float, x: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    # -- Exponential / logarithmic maps ----------------------------------
    def expmap(self, v: Float[Array, ...], x: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def expmap_0(self, v: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def logmap(self, y: Float[Array, ...], x: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def logmap_0(self, y: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def retraction(self, v: Float[Array, ...], x: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    # -- Transport / tangent space ---------------------------------------
    def ptransp(self, v: Float[Array, ...], x: Float[Array, ...], y: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def ptransp_0(self, v: Float[Array, ...], y: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def tangent_inner(
        self, u: Float[Array, ...], v: Float[Array, ...], x: Float[Array, ...], c: float
    ) -> Float[Array, ""]: ...

    def tangent_norm(self, v: Float[Array, ...], x: Float[Array, ...], c: float) -> Float[Array, ""]: ...

    def egrad2rgrad(self, grad: Float[Array, ...], x: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    def tangent_proj(self, v: Float[Array, ...], x: Float[Array, ...], c: float) -> Float[Array, ...]: ...

    # -- Validation ------------------------------------------------------
    def is_in_manifold(self, x: Float[Array, ...], c: float) -> Array: ...

    def is_in_tangent_space(self, v: Float[Array, ...], x: Float[Array, ...], c: float) -> Array: ...
