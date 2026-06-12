"""Hyperbolic neural network layers for JAX/Flax NNX."""

from .hybrid_regularization import HyperPPFeatureScaling
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
from .hyperboloid_conv import FGGConv2D, HypConv2DHyperboloid, HypConv2DHyperboloidFHNN, HypConv2DHyperboloidPP, LorentzConv2D
from .hyperboloid_core import (
    build_spacelike_V,
    hrc,
    htc,
    hyp_avg_pool2d,
    lorentz_midpoint,
    lorentz_residual,
    spatial_to_hyperboloid,
)
from .hyperboloid_linear import (
    FGGLinear,
    HTCLinear,
    HypLinearHyperboloidFHCNN,
    HypLinearHyperboloidFHNN,
    HypLinearHyperboloidPLFC,
)
from .hyperboloid_positional import HyperbolicRoPE, HypformerPositionalEncoding, hope
from .hyperboloid_regression import FGGLorentzMLR, HypRegressionHyperboloid
from .hyperboloid_regularization import FGGMeanOnlyBatchNorm, HRCBatchNorm, HRCDropout, HRCLayerNorm, HRCRMSNorm
from .poincare_activations import poincare_leaky_relu, poincare_relu, poincare_tanh
from .poincare_batchnorm import PoincareBatchNorm2D, frechet_variance, poincare_midpoint, poincare_weighted_midpoint
from .poincare_conv import HypConv2DPoincare
from .poincare_linear import HypLinearPoincare, HypLinearPoincarePP
from .poincare_regression import HypRegressionPoincare, HypRegressionPoincarePP
from .poincare_vq import HypVQEmbeddingPoincare, HypVQMLRPoincare, PoincareVQOutput
from .pv_conv import HypConv2DPV
from .pv_linear import HypLinearPV
from .pv_regression import HypRegressionPV

__all__ = [
    "FGGConv2D",
    "FGGLinear",
    "FGGLorentzMLR",
    "FGGMeanOnlyBatchNorm",
    "HRCBatchNorm",
    "HRCDropout",
    "HRCLayerNorm",
    "HRCRMSNorm",
    "HTCLinear",
    "HypConv2DHyperboloid",
    "HypConv2DHyperboloidFHNN",
    "HypConv2DHyperboloidPP",
    "HypConv2DPV",
    "HypConv2DPoincare",
    "HypLinearHyperboloidFHCNN",
    "HypLinearHyperboloidFHNN",
    "HypLinearHyperboloidPLFC",
    "HypLinearPV",
    "HypLinearPoincare",
    "HypLinearPoincarePP",
    "HypRegressionHyperboloid",
    "HypRegressionPV",
    "HypRegressionPoincare",
    "HypRegressionPoincarePP",
    "HypVQEmbeddingPoincare",
    "HypVQMLRPoincare",
    "HyperPPFeatureScaling",
    "HyperbolicFullAttention",
    "HyperbolicLinearAttention",
    "HyperbolicRoPE",
    "HyperbolicSoftmaxAttention",
    "HypformerPositionalEncoding",
    "LorentzConv2D",
    "PoincareBatchNorm2D",
    "PoincareVQOutput",
    "build_spacelike_V",
    "focus_transform",
    "frechet_variance",
    "hope",
    "hrc",
    "hrc_gelu",
    "hrc_leaky_relu",
    "hrc_relu",
    "hrc_swish",
    "hrc_tanh",
    "htc",
    "hyp_avg_pool2d",
    "hyp_gelu",
    "hyp_leaky_relu",
    "hyp_relu",
    "hyp_swish",
    "hyp_tanh",
    "lorentz_midpoint",
    "lorentz_residual",
    "poincare_leaky_relu",
    "poincare_midpoint",
    "poincare_relu",
    "poincare_tanh",
    "poincare_weighted_midpoint",
    "spatial_to_hyperboloid",
]
