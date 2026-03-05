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
from .hyperboloid_attention import (
    HyperbolicFullAttention,
    HyperbolicLinearAttention,
    HyperbolicSoftmaxAttention,
    focus_transform,
)
from .hyperboloid_conv import HypConv2DHyperboloid, HypConv3DHyperboloid, LorentzConv2D
from .hyperboloid_core import hrc, htc, lorentz_midpoint, lorentz_residual, spatial_to_hyperboloid
from .hyperboloid_linear import HTCLinear, HypLinearHyperboloidFHCNN
from .hyperboloid_positional import HyperbolicRoPE, HypformerPositionalEncoding, hope
from .hyperboloid_regression import HypRegressionHyperboloid
from .hyperboloid_regularization import HRCBatchNorm, HRCDropout, HRCLayerNorm, HRCRMSNorm
from .poincare_activations import poincare_leaky_relu, poincare_relu, poincare_tanh
from .poincare_conv import HypConv2DPoincare
from .poincare_linear import HypLinearPoincare, HypLinearPoincarePP
from .poincare_regression import HypRegressionPoincare, HypRegressionPoincarePP
from .poincare_rl import HypRegressionPoincareHDRL

__all__ = [
    "HRCBatchNorm",
    "HRCDropout",
    "HRCLayerNorm",
    "HRCRMSNorm",
    "HTCLinear",
    "HypConv2DHyperboloid",
    "HypConv2DPoincare",
    "HypConv3DHyperboloid",
    "HypLinearHyperboloidFHCNN",
    "HypLinearPoincare",
    "HypLinearPoincarePP",
    "HypRegressionHyperboloid",
    "HypRegressionPoincare",
    "HypRegressionPoincareHDRL",
    "HypRegressionPoincarePP",
    "HyperbolicFullAttention",
    "HyperbolicLinearAttention",
    "HyperbolicRoPE",
    "HyperbolicSoftmaxAttention",
    "HypformerPositionalEncoding",
    "LorentzConv2D",
    "focus_transform",
    "hope",
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
    "lorentz_midpoint",
    "lorentz_residual",
    "poincare_leaky_relu",
    "poincare_relu",
    "poincare_tanh",
    "spatial_to_hyperboloid",
]
