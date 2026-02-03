"""Hyperbolic neural network layers for JAX/Flax NNX."""

from .hyperboloid_activations import (
    hrc_gelu,
    hrc_leaky_relu,
    hrc_relu,
    hrc_swish,
    hrc_tanh,
    hyp_gelu,
    hyp_leaky_relu,
    hyp_relu,
    hyp_swish,
    hyp_tanh,
)
from .hyperboloid_conv import HypConv2DHyperboloid, HypConv3DHyperboloid, HypConvHyperboloid
from .hyperboloid_linear import HypLinearHyperboloidFHCNN
from .hyperboloid_regression import HypRegressionHyperboloid
from .hypformer import HRCDropout, HRCLayerNorm, HTCLinear, hrc, htc
from .lorentz_conv import LorentzConv2D, LorentzConv3D
from .poincare_linear import HypLinearPoincare, HypLinearPoincarePP
from .poincare_regression import HypRegressionPoincare, HypRegressionPoincarePP
from .poincare_rl import HypRegressionPoincareHDRL

__all__ = [
    "HRCDropout",
    "HRCLayerNorm",
    "HTCLinear",
    "HypConv2DHyperboloid",
    "HypConv3DHyperboloid",
    "HypConvHyperboloid",
    "HypLinearHyperboloidFHCNN",
    "HypLinearPoincare",
    "HypLinearPoincarePP",
    "HypRegressionHyperboloid",
    "HypRegressionPoincare",
    "HypRegressionPoincareHDRL",
    "HypRegressionPoincarePP",
    "LorentzConv2D",
    "LorentzConv3D",
    "hrc",
    "hrc_gelu",
    "hrc_leaky_relu",
    "hrc_relu",
    "hrc_swish",
    "hrc_tanh",
    "htc",
    "hyp_gelu",
    "hyp_leaky_relu",
    "hyp_relu",
    "hyp_swish",
    "hyp_tanh",
]
