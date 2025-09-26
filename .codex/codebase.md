# Codebase Notes

## src/
- `src/__init__.py` – Empty package initializer to make the hyperbolic components importable.
- `src/manifolds/__init__.py` – Re-exports the concrete manifold classes and helpers so callers can import from `src.manifolds` directly.
- `src/manifolds/manifold.py` – Defines the abstract `Manifold` interface and a `ManifoldParameter` wrapper, outlining all geometric operations subclasses must implement.
- `src/manifolds/euclidean.py` – Implements the flat Euclidean manifold with basic arithmetic, distance, and mapping operations consistent with the abstract base.
- `src/manifolds/hyperboloid.py` – Provides the Lorentz model of hyperbolic space, including Minkowski math, exp/log maps, projections, and stability safeguards.
- `src/manifolds/poincare.py` – Implements Möbius gyrovector operations for the Poincaré ball model with conformal factors, gyrations, and stable scalings.
- `src/nn_layers/__init__.py` – Collects manifold-aware neural layer classes for convenient imports across the project.
- `src/nn_layers/helpers.py` – Houses dtype utilities and shared multinomial regression kernels used by the manifold-specific layers.
- `src/nn_layers/hyperbolic_standard_layers.py` – Wraps manifold primitives (exp/log maps, projections, activations) as torch modules for plug-and-play usage.
- `src/nn_layers/hyperboloid_linear_layers.py` – Implements several hyperboloid-compatible linear layers covering GCN, FHNN, and FHCNN variants with manifold-aware bias handling.
- `src/nn_layers/hyperboloid_regression_layers.py` – Provides the hyperboloid multinomial regression head from FH-CNN, including tangent-space weights and smoothing controls.
- `src/nn_layers/poincare_linear_layers.py` – Implements Poincaré-ball linear layers from HNN and HNN++ with options for tangent or manifold inputs and manifold biases.
- `src/nn_layers/poincare_regression_layers.py` – Supplies regression modules for Poincaré models, including bias transport and stabilized distance computations.
- `src/nn_layers/poincare_rl_layers.py` – Adapts the multinomial regression layer for hyperbolic reinforcement learning, offering standard and re-scaled variants.
- `src/optim/__init__.py` – Re-exports the custom Riemannian optimizers for straightforward access via `src.optim`.
- `src/optim/riemannian_sgd.py` – Implements Riemannian SGD with momentum, handling manifold-aware gradients, transport, and expmap/retraction updates.
- `src/optim/riemannian_adam.py` – Extends Adam to Riemannian settings, including tangent inner products, bias corrections, and optional expmap updates.
- `src/utils/__init__.py` – Empty initializer marking the utilities package.
- `src/utils/helpers.py` – Contains distance-matrix utilities and hyperbolicity diagnostics (delta calculation) for manifold datasets.
- `src/utils/math_utils.py` – Provides numerically-stable hyperbolic functions and smooth clamping helpers used throughout the geometry code.
- `src/utils/vis_utils.py` – Builds visualization pipelines for hyperbolic embeddings, including dimensionality reduction and plotting geodesics or hyperplanes.
- `src/utils/horo_pca.py` – Implements the HoroPCA dimensionality-reduction algorithm plus supporting routines like Fréchet means and Lorentz centering.

## tests/
- `tests/__init__.py` – Empty marker ensuring the test suite imports as a package.
- `tests/conftest.py` – Defines shared pytest fixtures for random seeds, dtype control, manifold instances, and synthetic point clouds.
- `tests/test_manifolds.py` – Verifies algebraic properties and numerical stability of manifold operations across Euclidean, hyperboloid, and Poincaré models.
- `tests/test_optimizers.py` – Exercises the custom Riemannian optimizers by confirming convergence toward manifold targets under different update modes.
