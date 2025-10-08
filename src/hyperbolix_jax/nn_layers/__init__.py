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
from .standard_layers import (
    Expmap,
    Expmap0,
    HyperbolicActivation,
    Logmap,
    Logmap0,
    Proj,
    Retraction,
    TanProj,
)

__all__ = [
    # Standard layers
    "Expmap",
    "Expmap0",
    "Logmap",
    "Logmap0",
    "Proj",
    "TanProj",
    "Retraction",
    "HyperbolicActivation",
    # Poincaré linear layers
    "HypLinearPoincare",
    "HypLinearPoincarePP",
    # Poincaré regression layers
    "HypRegressionPoincare",
    "HypRegressionPoincarePP",
    "HypRegressionPoincareHDRL",
    # Hyperboloid linear layers
    "HypLinearHyperboloid",
    "HypLinearHyperboloidFHNN",
    "HypLinearHyperboloidFHCNN",
    # Hyperboloid regression layers
    "HypRegressionHyperboloid",
]
