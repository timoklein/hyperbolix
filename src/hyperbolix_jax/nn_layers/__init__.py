"""Hyperbolic neural network layers for JAX/Flax NNX."""

from .hyperboloid_linear import (
    HypLinearHyperboloid,
    HypLinearHyperboloidFHCNN,
    HypLinearHyperboloidFHNN,
)
from .hyperboloid_regression import HypRegressionHyperboloid
from .poincare_linear import HypLinearPoincare, HypLinearPoincarePP
from .poincare_regression import HypRegressionPoincare, HypRegressionPoincarePP
from .poincare_rl import HypRegressionPoincareHDRL

__all__ = [
    # Hyperboloid linear layers
    "HypLinearHyperboloid",
    "HypLinearHyperboloidFHCNN",
    "HypLinearHyperboloidFHNN",
    # Poincaré linear layers
    "HypLinearPoincare",
    "HypLinearPoincarePP",
    # Hyperboloid regression layers
    "HypRegressionHyperboloid",
    # Poincaré regression layers
    "HypRegressionPoincare",
    "HypRegressionPoincareHDRL",
    "HypRegressionPoincarePP",
]
