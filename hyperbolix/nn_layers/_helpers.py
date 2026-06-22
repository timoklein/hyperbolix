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

from hyperbolix.manifolds import Manifold


def as_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize an int-or-pair (e.g. kernel_size, stride) to a ``(h, w)`` tuple."""
    return (x, x) if isinstance(x, int) else x


def _validate_manifold_methods(
    manifold_module: Manifold,
    required_methods: tuple[str, ...],
    *,
    manifold_name: str,
    example_instance: str,
) -> None:
    if not all(hasattr(manifold_module, method) for method in required_methods):
        raise TypeError(f"manifold_module must be a class-based {manifold_name} manifold instance (e.g., {example_instance}).")


def validate_hyperboloid_manifold(manifold_module: Manifold, required_methods: tuple[str, ...]) -> None:
    _validate_manifold_methods(
        manifold_module,
        required_methods,
        manifold_name="Hyperboloid",
        example_instance="hyperbolix.manifolds.Hyperboloid()",
    )


def validate_poincare_manifold(manifold_module: Manifold, required_methods: tuple[str, ...]) -> None:
    _validate_manifold_methods(
        manifold_module,
        required_methods,
        manifold_name="Poincare",
        example_instance="hyperbolix.manifolds.Poincare()",
    )


def validate_pv_manifold(manifold_module: Manifold, required_methods: tuple[str, ...]) -> None:
    _validate_manifold_methods(
        manifold_module,
        required_methods,
        manifold_name="ProperVelocity",
        example_instance="hyperbolix.manifolds.ProperVelocity()",
    )
