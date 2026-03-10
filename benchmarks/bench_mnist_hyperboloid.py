"""MNIST benchmark for hyperboloid neural network layers.

Compares multiple hyperboloid variants:
- FHCNNHybrid: FHCNN with Euclidean embedding
- FHCNNDirect: FHCNN with direct projection
- HTCHybrid: HTC with Euclidean embedding
- HTCDirect: HTC with direct projection
- FHCNNCNNHybrid: FHCNN-based CNN with Euclidean embedding and HRCBatchNorm
- FullyHyperbolicCNN_HCat: Fully hyperbolic CNN using HypConv2DHyperboloid (HCat)
- FullyHyperbolicCNN_Lorentz: Fully hyperbolic CNN using LorentzConv2D (HRC)

All models use HypRegressionHyperboloid for fully hyperbolic classification.

Metrics: memory footprint, wallclock time, accuracy

Run with:
    uv run python benchmarks/bench_mnist_hyperboloid.py [OPTIONS]

Examples:
    # Run all models
    uv run python benchmarks/bench_mnist_hyperboloid.py

    # Compare convolutional approaches
    uv run python benchmarks/bench_mnist_hyperboloid.py --fully-hyp-hcat --fully-hyp-lorentz

    # Run only FHCNN variants
    uv run python benchmarks/bench_mnist_hyperboloid.py --fhcnn-hybrid --fhcnn-direct

    # Run only CNN model
    uv run python benchmarks/bench_mnist_hyperboloid.py --fhcnn-cnn

    # Run HTC models
    uv run python benchmarks/bench_mnist_hyperboloid.py --htc-hybrid --htc-direct
"""

import argparse
import json

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from benchmarks.utils import (
    benchmark_model,
    load_mnist_data,
    plot_comparison,
    print_summary_table,
)
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import (
    FGGConv2D,
    FGGLinear,
    FGGLorentzMLR,
    FGGMeanOnlyBatchNorm,
    HRCBatchNorm,
    HTCLinear,
    HypConv2DHyperboloid,
    HypLinearHyperboloidFHCNN,
    HypRegressionHyperboloid,
    LorentzConv2D,
    hrc_relu,
)

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)

# Class-based manifold instance for NN layers
hyperboloid = Hyperboloid(dtype=jnp.float64)


# ==============================================================================
# Model Definitions
# ==============================================================================


class FHCNNHybrid(nnx.Module):
    """FHCNN with Euclidean embedding.

    Architecture:
        Input (784) → Euclidean Linear(784→32) + ReLU
                   → Project to Hyperboloid (32→33)
                   → FHCNN Layer 1 (33→65)
                   → FHCNN Layer 2 (65→65)
                   → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Linear(784, 32, rngs=rngs)  # Euclidean embedding
        self.hyp1 = HypLinearHyperboloidFHCNN(
            hyperboloid,
            33,
            65,
            rngs=rngs,
            input_space="manifold",
            learnable_scale=False,
            activation=None,
            normalize=False,
        )
        self.hyp2 = HypLinearHyperboloidFHCNN(
            hyperboloid, 65, 65, rngs=rngs, input_space="manifold", learnable_scale=False, normalize=False
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Embed in Euclidean space
        x = jax.nn.relu(self.embed(x))  # (batch, 32)

        # Project to hyperboloid
        x = jax.vmap(lambda v: hyperboloid.expmap_0(jnp.concatenate([jnp.zeros(1), v]), c))(x)  # (batch, 33)

        # Hyperbolic layers
        x = self.hyp1(x, c)  # (batch, 65)
        x = self.hyp2(x, c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class FHCNNDirect(nnx.Module):
    """FHCNN with direct projection.

    Architecture:
        Input (784) → Constraint-based projection to Hyperboloid (784→785)
                   → FHCNN Layer 1 (785→65)
                   → FHCNN Layer 2 (65→65)
                   → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        # No Euclidean embedding
        self.hyp1 = HypLinearHyperboloidFHCNN(
            hyperboloid,
            785,
            65,
            rngs=rngs,
            input_space="manifold",
            learnable_scale=False,
            normalize=False,
        )
        self.hyp2 = HypLinearHyperboloidFHCNN(
            hyperboloid, 65, 65, rngs=rngs, input_space="manifold", learnable_scale=False, normalize=False
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # FHCNN-style: treat input as spatial coords, compute time from constraint
        # time = sqrt(||space||^2 + 1/c)
        time_coord = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x], axis=-1)  # (batch, 785)

        # Hyperbolic layers
        x = self.hyp1(x, c)  # (batch, 65)
        x = self.hyp2(x, c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class HTCHybrid(nnx.Module):
    """HTC with Euclidean embedding.

    Architecture:
        Input (784) → Euclidean Linear(784→32) + ReLU
                   → Project to Hyperboloid (32→33)
                   → HTC Layer 1 (33→65)
                   → HTC Layer 2 (65→65)
                   → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Linear(784, 32, rngs=rngs)  # Euclidean embedding
        self.hyp1 = HTCLinear(
            in_features=33,
            out_features=64,  # Spatial dimension (output will be 65 with time)
            rngs=rngs,
            use_bias=True,
        )
        self.hyp2 = HTCLinear(
            in_features=65,
            out_features=64,  # Spatial dimension (output will be 65 with time)
            rngs=rngs,
            use_bias=True,
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Embed in Euclidean space
        x = jax.nn.relu(self.embed(x))  # (batch, 32)

        # Project to hyperboloid
        x = jax.vmap(lambda v: hyperboloid.expmap_0(jnp.concatenate([jnp.zeros(1), v]), c))(x)  # (batch, 33)

        # Hyperbolic layers (HTC uses c_in and c_out)
        x = self.hyp1(x, c_in=c, c_out=c)  # (batch, 65)
        x = self.hyp2(x, c_in=c, c_out=c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class HTCDirect(nnx.Module):
    """HTC with direct projection.

    Architecture:
        Input (784) → Constraint-based projection to Hyperboloid (784→785)
                   → HTC Layer 1 (785→65)
                   → HTC Layer 2 (65→65)
                   → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        # No Euclidean embedding
        self.hyp1 = HTCLinear(
            in_features=785,
            out_features=64,  # Spatial dimension (output will be 65 with time)
            rngs=rngs,
            use_bias=True,
        )
        self.hyp2 = HTCLinear(
            in_features=65,
            out_features=64,  # Spatial dimension (output will be 65 with time)
            rngs=rngs,
            use_bias=True,
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # FHCNN-style: treat input as spatial coords, compute time from constraint
        # time = sqrt(||space||^2 + 1/c)
        time_coord = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x], axis=-1)  # (batch, 785)

        # Hyperbolic layers
        x = self.hyp1(x, c_in=c, c_out=c)  # (batch, 65)
        x = self.hyp2(x, c_in=c, c_out=c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class FHCNNCNNHybrid(nnx.Module):
    """FHCNN-based CNN following the paper more closely.

    Key design decisions:
    - Use Euclidean stem (conv layers) for initial feature extraction
    - Use strided convolutions for downsampling instead of pooling
    - Use global average pooling in Euclidean space to get manageable feature dimensions
    - Project to hyperbolic space with small feature vectors (avoids numerical instability)
    - Use constraint-based projection (not expmap_0) to avoid exponential blowup

    Architecture:
        Input (28x28x1) → Euclidean Conv(1→32, stride=2) + ReLU + BatchNorm
                        → Euclidean Conv(32→64, stride=2) + ReLU + BatchNorm
                        → Global Average Pooling → (batch, 64)
                        → Project to Hyperboloid (64→65) via constraint
                        → FHCNN Linear (65→65)
                        → HRC ReLU
                        → FHCNN Linear (65→65)
                        → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        # Euclidean stem with strided convolutions for downsampling
        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=32,
            kernel_size=(3, 3),
            strides=(2, 2),  # Stride for downsampling: 28→14
            padding="SAME",
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(3, 3),
            strides=(2, 2),  # Stride for downsampling: 14→7
            padding="SAME",
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)

        # Hyperbolic layers
        # After global avg pool: (batch, 64) → project to (batch, 65) on hyperboloid
        self.hyp1 = HypLinearHyperboloidFHCNN(
            hyperboloid,
            65,  # 64 spatial + 1 time
            65,
            rngs=rngs,
            input_space="manifold",
            learnable_scale=False,
            normalize=False,
        )
        self.hyp2 = HypLinearHyperboloidFHCNN(
            hyperboloid,
            65,
            65,
            rngs=rngs,
            input_space="manifold",
            learnable_scale=False,
            normalize=False,
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Reshape flat input to image: (batch, 784) → (batch, 28, 28, 1)
        x = x.reshape(-1, 28, 28, 1)  # (batch, 28, 28, 1)

        # Euclidean feature extraction with strided convolutions
        x = jax.nn.relu(self.conv1(x))  # (batch, 14, 14, 32)
        x = self.bn1(x, use_running_average=use_running_average)

        x = jax.nn.relu(self.conv2(x))  # (batch, 7, 7, 64)
        x = self.bn2(x, use_running_average=use_running_average)

        # Global average pooling in Euclidean space (stable, manageable dimensions)
        x = jnp.mean(x, axis=(1, 2))  # (batch, 64)

        # Project to hyperboloid using constraint-based projection (NOT expmap_0)
        # This avoids exponential blowup for features with large norms
        # time = sqrt(||space||^2 + 1/c)
        time_coord = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x], axis=-1)  # (batch, 65)

        # Hyperbolic layers with activations
        x = self.hyp1(x, c)  # (batch, 65)
        x = hrc_relu(x, c, c)  # HRC ReLU handles arbitrary batch dims

        x = self.hyp2(x, c)  # (batch, 65)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class FullyHyperbolicCNN_HCat(nnx.Module):
    """Fully hyperbolic CNN using HypConv2DHyperboloid (HCat + HypLinear approach).

    This is a truly end-to-end hyperbolic CNN where every operation happens on the
    hyperboloid manifold, using the HCat concatenation approach from the paper.

    Architecture:
        Input (28x28x1) → Project each pixel to Hyperboloid (28x28x2)
                        → HypConv2D (2→33, stride=2) + HRC ReLU + HRC BatchNorm → 14x14x33
                        → HypConv2D (33→65, stride=2) + HRC ReLU + HRC BatchNorm → 7x7x65
                        → Global Average Pooling (extract spatial, pool, reconstruct time)
                        → FHCNN Linear (65→65) + HRC ReLU
                        → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        # Hyperbolic convolutional layers
        # Input: each pixel is a 2D hyperboloid point (1D manifold in 2D ambient space)
        self.hyp_conv1 = HypConv2DHyperboloid(
            manifold_module=hyperboloid,
            in_channels=2,  # 1 spatial + 1 time
            out_channels=33,  # 32 spatial + 1 time
            kernel_size=3,
            rngs=rngs,
            stride=2,  # Downsample 28→14
            padding="SAME",
            input_space="manifold",
        )
        self.hyp_bn1 = HRCBatchNorm(32, rngs=rngs)  # Normalize 32 spatial components

        self.hyp_conv2 = HypConv2DHyperboloid(
            manifold_module=hyperboloid,
            in_channels=33,
            out_channels=65,  # 64 spatial + 1 time
            kernel_size=3,
            rngs=rngs,
            stride=2,  # Downsample 14→7
            padding="SAME",
            input_space="manifold",
        )
        self.hyp_bn2 = HRCBatchNorm(64, rngs=rngs)

        # Hyperbolic linear layers
        self.hyp_linear = HypLinearHyperboloidFHCNN(
            hyperboloid,
            65,
            65,
            rngs=rngs,
            input_space="manifold",
            learnable_scale=False,
            normalize=False,
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Reshape to image
        x = x.reshape(-1, 28, 28, 1)  # (batch, 28, 28, 1)

        # Project each pixel value to hyperboloid
        # For scalar pixel value v, create point [sqrt(v^2 + 1/c), v]
        # This creates a 1D hyperboloid (2D ambient space) at each pixel
        batch_size, h, w, _ = x.shape
        x_flat = x.reshape(batch_size * h * w)  # (batch*28*28,)
        time_coords = jnp.sqrt(x_flat**2 + 1 / c)  # (batch*28*28,)
        x_hyp = jnp.stack([time_coords, x_flat], axis=-1)  # (batch*28*28, 2)
        x = x_hyp.reshape(batch_size, h, w, 2)  # (batch, 28, 28, 2)

        # First hyperbolic conv block
        x = self.hyp_conv1(x, c)  # (batch, 14, 14, 33)
        x = hrc_relu(x, c, c)  # HRC ReLU
        x = self.hyp_bn1(x, c_in=c, c_out=c, use_running_average=use_running_average)  # (batch, 14, 14, 33)

        # Second hyperbolic conv block
        x = self.hyp_conv2(x, c)  # (batch, 7, 7, 65)
        x = hrc_relu(x, c, c)
        x = self.hyp_bn2(x, c_in=c, c_out=c, use_running_average=use_running_average)  # (batch, 7, 7, 65)

        # Global average pooling in hyperbolic space
        # Extract spatial components, pool, reconstruct time
        x_space = x[..., 1:]  # (batch, 7, 7, 64)
        x_space_pooled = jnp.mean(x_space, axis=(1, 2))  # (batch, 64)

        # Reconstruct as hyperboloid point
        time_coord = jnp.sqrt(jnp.sum(x_space_pooled**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x_space_pooled], axis=-1)  # (batch, 65)

        # Hyperbolic linear layer
        x = self.hyp_linear(x, c)  # (batch, 65)
        x = hrc_relu(x, c, c)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class FullyHyperbolicCNN_Lorentz(nnx.Module):
    """Fully hyperbolic CNN using LorentzConv2D (HRC-based approach).

    This CNN uses the simpler LorentzConv2D layer which applies Euclidean
    convolution to spatial components and reconstructs the time component,
    following the Hyperbolic Layer (HL) pattern.

    Architecture:
        Input (28x28x1) → Project each pixel to Hyperboloid (28x28x2)
                        → LorentzConv2D (2→33, stride=2) + HRC BatchNorm + HRC ReLU → 14x14x33
                        → LorentzConv2D (33→65, stride=2) + HRC BatchNorm + HRC ReLU → 7x7x65
                        → Global Average Pooling (extract spatial, pool, reconstruct time)
                        → FHCNN Linear (65→65) + HRC ReLU
                        → HypRegressionHyperboloid MLR (65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        # Hyperbolic convolutional layers using LorentzConv2D (HRC pattern)
        self.hyp_conv1 = LorentzConv2D(
            in_channels=2,  # 1 spatial + 1 time
            out_channels=33,  # 32 spatial + 1 time
            kernel_size=3,
            rngs=rngs,
            stride=2,  # Downsample 28→14
            padding="SAME",
        )
        self.hyp_bn1 = HRCBatchNorm(32, rngs=rngs)  # Normalize 32 spatial components

        self.hyp_conv2 = LorentzConv2D(
            in_channels=33,
            out_channels=65,  # 64 spatial + 1 time
            kernel_size=3,
            rngs=rngs,
            stride=2,  # Downsample 14→7
            padding="SAME",
        )
        self.hyp_bn2 = HRCBatchNorm(64, rngs=rngs)

        # Hyperbolic linear layers
        self.hyp_linear = HypLinearHyperboloidFHCNN(
            hyperboloid,
            65,
            65,
            rngs=rngs,
            input_space="manifold",
            learnable_scale=False,
            normalize=False,
        )
        self.output = HypRegressionHyperboloid(hyperboloid, 65, 10, rngs=rngs, input_space="manifold")

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Reshape to image
        x = x.reshape(-1, 28, 28, 1)  # (batch, 28, 28, 1)

        # Project each pixel value to hyperboloid
        # For scalar pixel value v, create point [sqrt(v^2 + 1/c), v]
        # This creates a 1D hyperboloid (2D ambient space) at each pixel
        batch_size, h, w, _ = x.shape
        x_flat = x.reshape(batch_size * h * w)  # (batch*28*28,)
        time_coords = jnp.sqrt(x_flat**2 + 1 / c)  # (batch*28*28,)
        x_hyp = jnp.stack([time_coords, x_flat], axis=-1)  # (batch*28*28, 2)
        x = x_hyp.reshape(batch_size, h, w, 2)  # (batch, 28, 28, 2)

        # First hyperbolic conv block
        # Conv → BatchNorm → ReLU (matching author's implementation)
        x = self.hyp_conv1(x, c)  # (batch, 14, 14, 33)
        x = self.hyp_bn1(x, c_in=c, c_out=c, use_running_average=use_running_average)  # (batch, 14, 14, 33)
        x = hrc_relu(x, c, c)  # HRC ReLU

        # Second hyperbolic conv block
        # Conv → BatchNorm → ReLU (matching author's implementation)
        x = self.hyp_conv2(x, c)  # (batch, 7, 7, 65)
        x = self.hyp_bn2(x, c_in=c, c_out=c, use_running_average=use_running_average)  # (batch, 7, 7, 65)
        x = hrc_relu(x, c, c)

        # Global average pooling in hyperbolic space
        # Extract spatial components, pool, reconstruct time
        x_space = x[..., 1:]  # (batch, 7, 7, 64)
        x_space_pooled = jnp.mean(x_space, axis=(1, 2))  # (batch, 64)

        # Reconstruct as hyperboloid point
        time_coord = jnp.sqrt(jnp.sum(x_space_pooled**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x_space_pooled], axis=-1)  # (batch, 65)

        # Hyperbolic linear layer
        x = self.hyp_linear(x, c)  # (batch, 65)
        x = hrc_relu(x, c, c)

        # Hyperbolic MLR classification
        return self.output(x, c)  # (batch, 10)


class FGGHybrid(nnx.Module):
    """FGG-LNN with Euclidean embedding.

    Architecture:
        Input (784) → Euclidean Linear(784→32) + ReLU
                   → Project to Hyperboloid (32→33)
                   → FGGLinear(33→65, relu)
                   → FGGLinear(65→65, relu)
                   → FGGLorentzMLR(65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Linear(784, 32, rngs=rngs)  # Euclidean embedding
        self.hyp1 = FGGLinear(33, 65, rngs=rngs, activation=jax.nn.relu, reset_params="eye")
        self.hyp2 = FGGLinear(65, 65, rngs=rngs, activation=jax.nn.relu, reset_params="eye")
        self.output = FGGLorentzMLR(65, 10, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Embed in Euclidean space
        x = jax.nn.relu(self.embed(x))  # (batch, 32)

        # Project to hyperboloid via constraint
        time_coord = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x], axis=-1)  # (batch, 33)

        # FGG layers (activation baked in)
        x = self.hyp1(x, c)  # (batch, 65)
        x = self.hyp2(x, c)  # (batch, 65)

        # FGG MLR classification
        return self.output(x, c)  # (batch, 10)


class FGGDirect(nnx.Module):
    """FGG-LNN with direct projection.

    Architecture:
        Input (784) → Constraint-based projection to Hyperboloid (784→785)
                   → FGGLinear(785→65, relu)
                   → FGGLinear(65→65, relu)
                   → FGGLorentzMLR(65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.hyp1 = FGGLinear(785, 65, rngs=rngs, activation=jax.nn.relu, reset_params="kaiming")
        self.hyp2 = FGGLinear(65, 65, rngs=rngs, activation=jax.nn.relu, reset_params="eye")
        self.output = FGGLorentzMLR(65, 10, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        # Constraint-based projection: time = sqrt(||space||^2 + 1/c)
        time_coord = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x], axis=-1)  # (batch, 785)

        # FGG layers
        x = self.hyp1(x, c)  # (batch, 65)
        x = self.hyp2(x, c)  # (batch, 65)

        # FGG MLR classification
        return self.output(x, c)  # (batch, 10)


class FGGCNNHybrid(nnx.Module):
    """FGG-LNN CNN with Euclidean stem.

    Architecture:
        Input (28x28x1) → Euclidean Conv(1→32, stride=2) + ReLU + BatchNorm → 14x14x32
                        → Euclidean Conv(32→64, stride=2) + ReLU + BatchNorm → 7x7x64
                        → Global Average Pooling → (batch, 64)
                        → Project to Hyperboloid (64→65)
                        → FGGLinear(65→65, relu)
                        → FGGLorentzMLR(65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        # Euclidean stem
        self.conv1 = nnx.Conv(in_features=1, out_features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)

        # Hyperbolic head
        self.hyp1 = FGGLinear(65, 65, rngs=rngs, activation=jax.nn.relu, reset_params="eye")
        self.output = FGGLorentzMLR(65, 10, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        x = x.reshape(-1, 28, 28, 1)  # (batch, 28, 28, 1)

        # Euclidean feature extraction
        x = jax.nn.relu(self.conv1(x))  # (batch, 14, 14, 32)
        x = self.bn1(x, use_running_average=use_running_average)
        x = jax.nn.relu(self.conv2(x))  # (batch, 7, 7, 64)
        x = self.bn2(x, use_running_average=use_running_average)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (batch, 64)

        # Project to hyperboloid
        time_coord = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1 / c)  # (batch, 1)
        x = jnp.concatenate([time_coord, x], axis=-1)  # (batch, 65)

        # FGG layers
        x = self.hyp1(x, c)  # (batch, 65)
        return self.output(x, c)  # (batch, 10)


class FullyHyperbolicCNN_FGG(nnx.Module):
    """Fully hyperbolic CNN using FGGConv2D (HCat + FGGLinear).

    End-to-end hyperbolic CNN where convolutions use FGGConv2D, matching
    the FGG-LNN reference architecture with Weight Norm + Mean-only BatchNorm
    (paper §4.4, Salimans & Kingma 2016).

    Architecture:
        Input (28x28x1) → Project each pixel to Hyperboloid (28x28x2)
                        → FGGConv2D (2→33, stride=2, relu, weight_norm) + MeanOnlyBN → 14x14x33
                        → FGGConv2D (33→65, stride=2, relu, weight_norm) + MeanOnlyBN → 7x7x65
                        → Global Average Pooling (spatial pool + time reconstruct)
                        → FGGLinear(65→65, relu, weight_norm)
                        → FGGLorentzMLR(65→10 classes)
    """

    def __init__(self, rngs: nnx.Rngs):
        self.hyp_conv1 = FGGConv2D(
            hyperboloid,
            in_channels=2,
            out_channels=33,
            kernel_size=3,
            rngs=rngs,
            stride=2,
            padding="SAME",
            activation=jax.nn.relu,
            reset_params="kaiming",
            use_weight_norm=True,
        )
        self.hyp_bn1 = FGGMeanOnlyBatchNorm(32)

        self.hyp_conv2 = FGGConv2D(
            hyperboloid,
            in_channels=33,
            out_channels=65,
            kernel_size=3,
            rngs=rngs,
            stride=2,
            padding="SAME",
            activation=jax.nn.relu,
            reset_params="kaiming",
            use_weight_norm=True,
        )
        self.hyp_bn2 = FGGMeanOnlyBatchNorm(64)

        self.hyp_linear = FGGLinear(65, 65, rngs=rngs, activation=jax.nn.relu, reset_params="eye", use_weight_norm=True)
        self.output = FGGLorentzMLR(65, 10, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch 784"], c: float = 1.0, use_running_average: bool = False
    ) -> Float[Array, "batch 10"]:
        x = x.reshape(-1, 28, 28, 1)  # (batch, 28, 28, 1)

        # Project each pixel to hyperboloid
        batch_size, h, w, _ = x.shape
        x_flat = x.reshape(batch_size * h * w)
        time_coords = jnp.sqrt(x_flat**2 + 1 / c)
        x_hyp = jnp.stack([time_coords, x_flat], axis=-1)
        x = x_hyp.reshape(batch_size, h, w, 2)  # (batch, 28, 28, 2)

        # FGG conv blocks with mean-only BN (paper §4.4)
        x = self.hyp_conv1(x, c)  # (batch, 14, 14, 33)
        x = self.hyp_bn1(x, c_in=c, c_out=c, use_running_average=use_running_average)

        x = self.hyp_conv2(x, c)  # (batch, 7, 7, 65)
        x = self.hyp_bn2(x, c_in=c, c_out=c, use_running_average=use_running_average)

        # Global average pooling (spatial pool + time reconstruct)
        x_space = x[..., 1:]  # (batch, 7, 7, 64)
        x_space_pooled = jnp.mean(x_space, axis=(1, 2))  # (batch, 64)
        time_coord = jnp.sqrt(jnp.sum(x_space_pooled**2, axis=-1, keepdims=True) + 1 / c)
        x = jnp.concatenate([time_coord, x_space_pooled], axis=-1)  # (batch, 65)

        # FGG linear + MLR
        x = self.hyp_linear(x, c)  # (batch, 65)
        return self.output(x, c)  # (batch, 10)


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MNIST benchmark comparing hyperboloid neural network layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models (default)
  python benchmarks/bench_mnist_hyperboloid.py

  # Compare convolutional approaches (HCat vs LorentzConv2D)
  python benchmarks/bench_mnist_hyperboloid.py --fully-hyp-hcat --fully-hyp-lorentz

  # Run only FHCNN variants
  python benchmarks/bench_mnist_hyperboloid.py --fhcnn-hybrid --fhcnn-direct

  # Run only CNN model
  python benchmarks/bench_mnist_hyperboloid.py --fhcnn-cnn

  # Run HTC models
  python benchmarks/bench_mnist_hyperboloid.py --htc-hybrid --htc-direct
        """,
    )

    parser.add_argument("--fhcnn-hybrid", action="store_true", help="Run FHCNN with Euclidean embedding")
    parser.add_argument("--fhcnn-direct", action="store_true", help="Run FHCNN with direct projection")
    parser.add_argument("--htc-hybrid", action="store_true", help="Run HTC with Euclidean embedding")
    parser.add_argument("--htc-direct", action="store_true", help="Run HTC with direct projection")
    parser.add_argument("--fhcnn-cnn", action="store_true", help="Run FHCNN-based CNN with Euclidean embedding and BatchNorm")
    parser.add_argument(
        "--fully-hyp-hcat", action="store_true", help="Run fully hyperbolic CNN using HypConv2D (HCat approach)"
    )
    parser.add_argument(
        "--fully-hyp-lorentz", action="store_true", help="Run fully hyperbolic CNN using LorentzConv2D (HRC approach)"
    )
    parser.add_argument("--fgg-hybrid", action="store_true", help="Run FGG-LNN with Euclidean embedding")
    parser.add_argument("--fgg-direct", action="store_true", help="Run FGG-LNN with direct projection")
    parser.add_argument("--fgg-cnn", action="store_true", help="Run FGG-LNN CNN with Euclidean stem")
    parser.add_argument("--fully-hyp-fgg", action="store_true", help="Run fully hyperbolic CNN using FGGConv2D")
    parser.add_argument("--all", action="store_true", help="Run all models (default if no flags specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    return parser.parse_args()


def main():
    """Run benchmarks based on command line arguments."""
    args = parse_args()

    # Determine which models to run
    # If no specific flags are set (or --all is set), run all models
    run_all = args.all or not (
        args.fhcnn_hybrid
        or args.fhcnn_direct
        or args.htc_hybrid
        or args.htc_direct
        or args.fhcnn_cnn
        or args.fully_hyp_hcat
        or args.fully_hyp_lorentz
        or args.fgg_hybrid
        or args.fgg_direct
        or args.fgg_cnn
        or args.fully_hyp_fgg
    )

    # Build list of models to benchmark
    available_models = [
        (FHCNNHybrid, "FHCNN-Hybrid", args.fhcnn_hybrid or run_all),
        (FHCNNDirect, "FHCNN-Direct", args.fhcnn_direct or run_all),
        (HTCHybrid, "HTC-Hybrid", args.htc_hybrid or run_all),
        (HTCDirect, "HTC-Direct", args.htc_direct or run_all),
        (FHCNNCNNHybrid, "FHCNN-CNN-Hybrid", args.fhcnn_cnn or run_all),
        (FullyHyperbolicCNN_HCat, "FullyHyp-HCat", args.fully_hyp_hcat or run_all),
        (FullyHyperbolicCNN_Lorentz, "FullyHyp-Lorentz", args.fully_hyp_lorentz or run_all),
        (FGGHybrid, "FGG-Hybrid", args.fgg_hybrid or run_all),
        (FGGDirect, "FGG-Direct", args.fgg_direct or run_all),
        (FGGCNNHybrid, "FGG-CNN-Hybrid", args.fgg_cnn or run_all),
        (FullyHyperbolicCNN_FGG, "FullyHyp-FGG", args.fully_hyp_fgg or run_all),
    ]

    models = [(cls, name) for cls, name, should_run in available_models if should_run]

    if not models:
        print("No models selected. Use --help to see available options.")
        return

    print("=" * 60)
    print("MNIST Hyperboloid Layer Benchmark")
    print("=" * 60)
    print(f"\nRunning {len(models)} model(s): {', '.join(name for _, name in models)}")
    print(f"Random seed: {args.seed}")
    print("\nLoading MNIST data...")
    train_data, test_data = load_mnist_data()

    results = {}
    for model_class, name in models:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {name}")
        print("=" * 60)
        results[name] = benchmark_model(model_class, name, train_data, test_data, seed=args.seed, c=1.0, batch_size=128)

    # Save results
    print("\nSaving results...")
    with open("results/mnist_hyperboloid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/mnist_hyperboloid_results.json")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(results, "results/mnist_hyperboloid_comparison.png")

    # Print summary table
    print_summary_table(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
