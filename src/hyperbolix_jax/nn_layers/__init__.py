"""Hyperbolic neural network layers for JAX/Flax NNX."""

from .hyperboloid_activations import hyp_leaky_relu, hyp_relu, hyp_swish, hyp_tanh
from .hyperboloid_conv import HypConv2DHyperboloid, HypConv3DHyperboloid, HypConvHyperboloid
from .hyperboloid_linear import HypLinearHyperboloid, HypLinearHyperboloidFHCNN, HypLinearHyperboloidFHNN
from .hyperboloid_regression import HypRegressionHyperboloid
from .poincare_linear import HypLinearPoincare, HypLinearPoincarePP
from .poincare_regression import HypRegressionPoincare, HypRegressionPoincarePP
from .poincare_rl import HypRegressionPoincareHDRL

__all__ = [
    # Hyperboloid convolutional layers
    "HypConv2DHyperboloid",
    "HypConv3DHyperboloid",
    "HypConvHyperboloid",  # Backward compatible alias for HypConv2DHyperboloid
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
    # Hyperboloid activation functions
    "hyp_leaky_relu",
    "hyp_relu",
    "hyp_swish",
    "hyp_tanh",
]
