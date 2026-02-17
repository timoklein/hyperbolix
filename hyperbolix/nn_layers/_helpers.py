"""Shared helpers for hyperbolic nn layer modules."""

from hyperbolix.manifolds import Manifold


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
