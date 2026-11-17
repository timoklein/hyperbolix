"""Shared helpers for hyperbolic nn layer modules.

Note on layer-side manifold storage: every layer in ``hyperbolix.nn_layers``
stores its manifold via ``self.manifold = manifold_module``. This is safe
even when the same manifold instance is shared across multiple layers (or
reused inside ``nnx.scan`` / ``nnx.fori_loop``) because manifolds are plain
Python classes — not ``nnx.Module`` — so the assignment registers as static
graphdef metadata and contributes nothing to the NNX state pytree. Do not
wrap the assignment in ``nnx.data(...)`` or ``object.__setattr__``; both
would be no-ops here and only obscure intent.
"""

from hyperbolix.manifolds import Hyperboloid, Manifold, Poincare, ProperVelocity, Stereographic


def as_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize an int-or-pair (e.g. kernel_size, stride) to a ``(h, w)`` tuple."""
    return (x, x) if isinstance(x, int) else x


def _validate_manifold_methods(
    manifold_module: Manifold,
    required_methods: tuple[str, ...],
    *,
    manifold_cls: type | tuple[type, ...],
    manifold_name: str,
    example_instance: str,
) -> None:
    # The class is checked, not only the method names: the three models share most method names
    # (`expmap_0`, `logmap_0`, `dist`, ...), so a layer built on another model's instance would
    # run and compute with the wrong geometry.
    if not isinstance(manifold_module, manifold_cls) or not all(
        hasattr(manifold_module, method) for method in required_methods
    ):
        raise TypeError(
            f"manifold_module must be a class-based {manifold_name} manifold instance (e.g., {example_instance}), "
            f"got {type(manifold_module).__name__}."
        )


def validate_hyperboloid_manifold(manifold_module: Manifold, required_methods: tuple[str, ...]) -> None:
    _validate_manifold_methods(
        manifold_module,
        required_methods,
        manifold_cls=Hyperboloid,
        manifold_name="Hyperboloid",
        example_instance="hyperbolix.manifolds.Hyperboloid()",
    )


def validate_poincare_manifold(manifold_module: Manifold, required_methods: tuple[str, ...]) -> None:
    _validate_manifold_methods(
        manifold_module,
        required_methods,
        # `Stereographic` is the same gyrovector space as the ball for c > 0 and has the ball's method
        # names with the ball's meaning, so it stays accepted where it has the required methods.
        manifold_cls=(Poincare, Stereographic),
        manifold_name="Poincare",
        example_instance="hyperbolix.manifolds.Poincare()",
    )


def validate_pv_manifold(manifold_module: Manifold, required_methods: tuple[str, ...]) -> None:
    _validate_manifold_methods(
        manifold_module,
        required_methods,
        manifold_cls=ProperVelocity,
        manifold_name="ProperVelocity",
        example_instance="hyperbolix.manifolds.ProperVelocity()",
    )
