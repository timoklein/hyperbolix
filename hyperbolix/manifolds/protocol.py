"""Manifold Protocol for structural typing.

Defines the common interface shared by all concrete manifold classes
(``Poincare``, ``Hyperboloid``, ``ProperVelocity``, ``Euclidean``,
``Stereographic``, and ``ProductManifold``). Use ``Manifold`` as a type hint
for any parameter that accepts an arbitrary manifold instance.

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

# Curvature in either of the two shapes the library accepts: a scalar, or the
# per-factor sequence of length ``n_factors`` that `ProductManifold` takes. Only
# `ProductManifold` needs the union — every other manifold, and the `Manifold`
# protocol itself, is annotated with the scalar arm, which is what its arithmetic
# actually requires. Kept exported because it is the type of `ProductManifold`'s
# `c` parameter and therefore part of the public signature.
Curvature = ScalarCurvature | Sequence[ScalarCurvature]


@runtime_checkable
class Manifold(Protocol):
    """Structural protocol for manifold classes.

    All concrete manifold classes (``Poincare``, ``Hyperboloid``,
    ``ProperVelocity``, ``Euclidean``, ``Stereographic``, ``ProductManifold``)
    satisfy this protocol without modification.

    ``c`` is typed ``ScalarCurvature`` -- this is the *scalar-curvature*
    interface, which is what every single manifold accepts and what the
    arithmetic inside them requires. ``ProductManifold`` declares the wider
    ``Curvature`` union (scalar *or* per-factor sequence) and therefore still
    satisfies the protocol: a parameter may always be wider than the one it
    implements. Code that means to pass a per-factor sequence should annotate
    the manifold as ``ProductManifold``, not as ``Manifold``.

    The method signatures use the *minimal common interface* so that
    manifold-specific optional parameters do not break compatibility. Two such
    parameters are nevertheless uniform across every implementation, and are
    part of the protocol:

    * ``dist`` / ``dist_0`` take a trailing ``version_idx: int``. Manifolds with
      a single distance implementation accept and ignore it (documented on each),
      so a generic caller can always forward one.
    * ``is_in_manifold`` / ``is_in_tangent_space`` take a trailing
      ``atol: float | None = None``, resolved through
      :func:`~hyperbolix.manifolds._base.default_atol` when ``None``. No
      implementation floors, clamps, or silently drops an explicit value;
      unconstrained manifolds (Euclidean, ProperVelocity) document that a
      finiteness test has no tolerance to apply.
    """

    dtype: Any

    def _cast(self, x: Array) -> Array: ...

    # -- Geometry --------------------------------------------------------
    def proj(self, x: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    def dist(
        self, x: Float[Array, ...], y: Float[Array, ...], c: ScalarCurvature, version_idx: int = ...
    ) -> Float[Array, ""]: ...

    def dist_0(self, x: Float[Array, ...], c: ScalarCurvature, version_idx: int = ...) -> Float[Array, ""]: ...

    def addition(self, x: Float[Array, ...], y: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    def scalar_mul(self, r: float | Float[Array, ""], x: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    # -- Exponential / logarithmic maps ----------------------------------
    def expmap(self, v: Float[Array, ...], x: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    def expmap_0(self, v: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    def logmap(self, y: Float[Array, ...], x: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    def logmap_0(self, y: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    def retraction(self, v: Float[Array, ...], x: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    # -- Transport / tangent space ---------------------------------------
    def ptransp(
        self, v: Float[Array, ...], x: Float[Array, ...], y: Float[Array, ...], c: ScalarCurvature
    ) -> Float[Array, ...]: ...

    def ptransp_0(self, v: Float[Array, ...], y: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    def tangent_inner(
        self, u: Float[Array, ...], v: Float[Array, ...], x: Float[Array, ...], c: ScalarCurvature
    ) -> Float[Array, ""]: ...

    def tangent_norm(self, v: Float[Array, ...], x: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ""]: ...

    def egrad2rgrad(self, grad: Float[Array, ...], x: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    def tangent_proj(self, v: Float[Array, ...], x: Float[Array, ...], c: ScalarCurvature) -> Float[Array, ...]: ...

    # -- Validation ------------------------------------------------------
    def is_in_manifold(self, x: Float[Array, ...], c: ScalarCurvature, atol: float | None = ...) -> Array: ...

    def is_in_tangent_space(
        self, v: Float[Array, ...], x: Float[Array, ...], c: ScalarCurvature, atol: float | None = ...
    ) -> Array: ...
