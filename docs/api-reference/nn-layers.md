# Neural Network Layers API

Hyperbolic neural network layers built with Flax NNX.

## Overview

Hyperbolix provides 15+ neural network layer classes and 5 activation functions for building hyperbolic deep learning models:

- **Linear Layers**: Poincaré and Hyperboloid linear transformations
- **Convolutional Layers**: HCat-based and HRC-based hyperbolic convolutions (2D and 3D)
- **Hypformer Components**: HTC (Hyperbolic Transformation Component) and HRC (Hyperbolic Regularization Component) with curvature-change support
- **Regression Layers**: Single-layer classifiers with Riemannian geometry
- **Activation Functions**: Hyperbolic ReLU, Leaky ReLU, Tanh, Swish, GELU
- **Helper Functions**: Utilities for regression and conformal factor computation

All layers follow Flax NNX conventions and store manifold module references.

## Linear Layers

### Poincaré Linear

::: hyperbolix.nn_layers.HypLinearPoincare
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HypLinearPoincarePP
    options:
      show_source: true
      heading_level: 4

### Hyperboloid Linear

::: hyperbolix.nn_layers.HypLinearHyperboloidFHCNN
    options:
      show_source: true
      heading_level: 4

### Usage Example

```python
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.manifolds import poincare
import jax.numpy as jnp

# Create hyperbolic linear layer
layer = HypLinearPoincare(
    manifold_module=poincare,
    in_dim=32,
    out_dim=16,
    rngs=nnx.Rngs(0)
)

# Forward pass
x = jax.random.normal(nnx.Rngs(1).params(), (10, 32)) * 0.3
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

output = layer(x_proj, c=1.0)
print(output.shape)  # (10, 16)
```

## Convolutional Layers

### Hyperboloid Convolutions

::: hyperbolix.nn_layers.HypConv2DHyperboloid
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HypConv3DHyperboloid
    options:
      show_source: true
      heading_level: 4

### Usage Example

```python
from hyperbolix.nn_layers import HypConv2DHyperboloid
from hyperbolix.manifolds import hyperboloid
from flax import nnx
import jax.numpy as jnp

# Create 2D hyperbolic convolution
conv = HypConv2DHyperboloid(
    manifold_module=hyperboloid,
    out_channels=32,
    kernel_size=(3, 3),
    stride=(1, 1),
    rngs=nnx.Rngs(0)
)

# Input: (batch, height, width, in_channels)
x = jax.random.normal(nnx.Rngs(1).params(), (8, 28, 28, 16))

# Project to hyperboloid
x_ambient = jnp.concatenate([
    jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.0),
    x
], axis=-1)

# Forward pass
output = conv(x_ambient, c=1.0, use_tangent_input=False)
print(output.shape)  # (8, 28, 28, 32×9+1) - dimension grows!
```

!!! warning "Dimensional Growth"
    Hyperboloid convolutions increase dimensionality via HCat operation:

    - Input: `d+1` dimensions
    - Output: `(d×N)+1` dimensions where `N = kernel_height × kernel_width`

    For 3×3 kernel: 3D input → 28D output. Use small kernels or add dimensionality reduction layers.

### LorentzConv2D (HRC-Based)

::: hyperbolix.nn_layers.LorentzConv2D
    options:
      show_source: true
      heading_level: 4

LorentzConv2D provides a simpler, more efficient alternative to HCat-based convolutions by using the Hyperbolic Regularization Component (HRC) pattern from the Hypformer paper.

**Key Differences from HypConv2DHyperboloid:**

| Feature | HypConv2DHyperboloid (HCat) | LorentzConv2D (HRC) |
|---------|---------------------------|-------------------|
| **Method** | HCat concatenation + linear | Euclidean conv on space components |
| **Dimension** | Grows: `(d-1)×N+1` | Preserved |
| **Speed** | Slower (~80s/epoch) | **2.5x faster** (~32s/epoch) |
| **Accuracy** | Higher (~71% on MNIST) | Lower (~46% on MNIST) |
| **Use Case** | Maximum accuracy | Speed/memory efficiency |

**Theoretical Connection:**

LorentzConv2D implements the Hyperbolic Layer (HL) pattern from LResNet, which is mathematically equivalent to the Hyperbolic Regularization Component (HRC) from Hypformer:

```python
# Both approaches:
# 1. Extract space components: x_s = x[..., 1:]
# 2. Apply Euclidean function: y_s = f(x_s)
# 3. Reconstruct time: y_t = sqrt(||y_s||^2 + 1/c)
```

**Usage Example:**

```python
from hyperbolix.nn_layers import LorentzConv2D
from flax import nnx
import jax.numpy as jnp

# Create efficient hyperbolic convolution
conv = LorentzConv2D(
    in_channels=33,    # Including time component
    out_channels=65,   # Including time component
    kernel_size=3,
    stride=2,
    padding="SAME",
    rngs=nnx.Rngs(0)
)

# Input: points on Lorentz manifold (batch, height, width, in_channels)
x = jnp.ones((8, 28, 28, 33))
x_space = x[..., 1:]
x_time = jnp.sqrt(jnp.sum(x_space**2, axis=-1, keepdims=True) + 1.0)
x = jnp.concatenate([x_time, x_space], axis=-1)

# Forward pass
output = conv(x, c=1.0)
print(output.shape)  # (8, 14, 14, 65) - dimensions preserved!
```

!!! tip "When to Use LorentzConv2D"
    Choose LorentzConv2D when:

    - Speed and memory efficiency are priorities
    - Working with resource-constrained environments
    - Acceptable accuracy trade-off for 2.5x speedup

    Choose HypConv2DHyperboloid when:

    - Maximum accuracy is required
    - Willing to accept slower training and dimensional growth

## Hypformer Components

The Hyperbolic Transformation Component (HTC) and Hyperbolic Regularization Component (HRC) from the Hypformer paper provide general-purpose wrappers for adapting Euclidean operations to hyperbolic geometry with curvature-change support.

### Core Functions

::: hyperbolix.nn_layers.hrc
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.htc
    options:
      show_source: true
      heading_level: 4

### HTC/HRC Modules

::: hyperbolix.nn_layers.HTCLinear
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HRCBatchNorm
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HRCLayerNorm
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HRCDropout
    options:
      show_source: true
      heading_level: 4

### Hypformer Example

```python
from hyperbolix.nn_layers import HTCLinear, HRCBatchNorm, hrc_relu
from hyperbolix.manifolds import hyperboloid
from flax import nnx
import jax.numpy as jnp

class HypformerBlock(nnx.Module):
    """Example using HTC/HRC components with curvature change."""

    def __init__(self, in_dim, out_dim, rngs):
        self.linear = HTCLinear(
            in_features=in_dim,
            out_features=out_dim,
            rngs=rngs
        )
        self.bn = HRCBatchNorm(num_features=out_dim, rngs=rngs)

    def __call__(self, x, c_in=1.0, c_out=2.0, use_running_average=False):
        # Linear transformation with curvature change
        x = self.linear(x, c_in=c_in, c_out=c_out)

        # Batch normalization (curvature-preserving)
        x = self.bn(x, c_in=c_out, c_out=c_out,
                    use_running_average=use_running_average)

        # Activation (curvature-preserving)
        x = hrc_relu(x, c_in=c_out, c_out=c_out)

        return x

# Create and use block
block = HypformerBlock(in_dim=33, out_dim=64, rngs=nnx.Rngs(0))

# Input on hyperboloid with curvature 1.0
x = jax.random.normal(nnx.Rngs(1).params(), (32, 33))
x_proj = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

# Transform to curvature 2.0
output = block(x_proj, c_in=1.0, c_out=2.0)
print(output.shape)  # (32, 65) - 64 spatial + 1 time
```

!!! info "HTC vs HRC"
    **HRC (Hyperbolic Regularization Component)**:

    - Applies Euclidean function `f_r` to **space components only**
    - Use for: activations, normalization, dropout, **convolutions**
    - Formula: `space = f_r(x_s)`, `time = sqrt(||space||^2 + 1/c_out)`

    **HTC (Hyperbolic Transformation Component)**:

    - Applies Euclidean function `f_t` to **full point** (time + space)
    - Use for: learnable linear transformations
    - Formula: `space = f_t(x)`, `time = sqrt(||space||^2 + 1/c_out)`

    Both support curvature changes (`c_in → c_out`) for flexible network design.

## Regression Layers

Single-layer classifiers with Riemannian geometry.

### Poincaré Regression

::: hyperbolix.nn_layers.HypRegressionPoincare
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HypRegressionPoincarePP
    options:
      show_source: true
      heading_level: 4

### Hyperboloid Regression

::: hyperbolix.nn_layers.HypRegressionHyperboloid
    options:
      show_source: true
      heading_level: 4

### Reinforcement Learning

::: hyperbolix.nn_layers.HypRegressionPoincareHDRL
    options:
      show_source: true
      heading_level: 4

### Regression Example

```python
from hyperbolix.nn_layers import HypRegressionPoincare
from hyperbolix.manifolds import poincare
from flax import nnx

# Multi-class classification (10 classes)
regressor = HypRegressionPoincare(
    manifold_module=poincare,
    in_dim=32,
    out_dim=10,
    rngs=nnx.Rngs(0)
)

# Input: hyperbolic embeddings
x = jax.random.normal(nnx.Rngs(1).params(), (64, 32)) * 0.3
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

# Forward pass returns logits
logits = regressor(x_proj, c=1.0)
print(logits.shape)  # (64, 10)

# Use with softmax for classification
probs = jax.nn.softmax(logits, axis=-1)
```

## Activation Functions

Hyperbolic activation functions that preserve manifold constraints. All activations follow the HRC pattern: apply function to space components, then reconstruct time.

### Curvature-Preserving Activations

::: hyperbolix.nn_layers.hyp_relu
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.hyp_leaky_relu
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.hyp_tanh
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.hyp_swish
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.hyp_gelu
    options:
      show_source: true
      heading_level: 4

### Curvature-Changing Activations (HRC-based)

For advanced use cases requiring curvature transformations:

::: hyperbolix.nn_layers.hrc_relu
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.hrc_leaky_relu
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.hrc_tanh
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.hrc_swish
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.hrc_gelu
    options:
      show_source: true
      heading_level: 4

### Activation Examples

**Curvature-Preserving Activation:**

```python
from hyperbolix.nn_layers import hyp_relu, hyp_gelu
from hyperbolix.manifolds import hyperboloid
import jax.numpy as jnp

# Points on hyperboloid (ambient coordinates)
x = jax.random.normal(jax.random.PRNGKey(0), (10, 5))
x_ambient = jnp.concatenate([
    jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.0),
    x
], axis=-1)

# Apply hyperbolic ReLU (curvature preserving)
output = hyp_relu(x_ambient, c=1.0)
print(output.shape)  # (10, 6) - same shape

# Verify manifold constraint
constraint = -output[:, 0]**2 + jnp.sum(output[:, 1:]**2, axis=-1)
print(jnp.allclose(constraint, -1.0, atol=1e-5))  # True

# Use GELU instead
output_gelu = hyp_gelu(x_ambient, c=1.0)
```

**Curvature-Changing Activation:**

```python
from hyperbolix.nn_layers import hrc_relu

# Transform from curvature 1.0 to curvature 2.0
output = hrc_relu(x_ambient, c_in=1.0, c_out=2.0)

# Verify new manifold constraint (c=2.0)
constraint = -output[:, 0]**2 + jnp.sum(output[:, 1:]**2, axis=-1)
print(jnp.allclose(constraint, -1.0/2.0, atol=1e-5))  # True
```

!!! info "How Activations Work"
    Hyperbolic activations follow the HRC pattern:

    1. **Extract** space components `x_s = x[..., 1:]`
    2. **Apply** activation to space: `y_s = activation(x_s)`
    3. **Scale** for curvature change: `y_s = sqrt(c_in/c_out) * y_s`
    4. **Reconstruct** time: `y_t = sqrt(||y_s||^2 + 1/c_out)`

    This avoids expensive exp/log maps while preserving geometry and enabling flexible curvature transformations.

## Helper Functions

Utility functions for manifold operations in neural networks.

::: hyperbolix.nn_layers.helpers
    options:
      show_source: true
      heading_level: 3
      members:
        - compute_mlr_poincare_pp
        - compute_mlr_hyperboloid
        - safe_conformal_factor

## Building Models

Example of a complete hyperbolic neural network:

```python
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincare, hyp_relu
from hyperbolix.manifolds import poincare
import jax
import jax.numpy as jnp

class HyperbolicNN(nnx.Module):
    def __init__(self, rngs):
        self.layer1 = HypLinearPoincare(
            manifold_module=poincare,
            in_dim=784,  # MNIST flattened
            out_dim=256,
            rngs=rngs
        )
        self.layer2 = HypLinearPoincare(
            manifold_module=poincare,
            in_dim=256,
            out_dim=128,
            rngs=rngs
        )
        self.layer3 = HypLinearPoincare(
            manifold_module=poincare,
            in_dim=128,
            out_dim=10,
            rngs=rngs
        )

    def __call__(self, x, c=1.0):
        # x: (batch, 784) on Poincaré ball
        x = self.layer1(x, c)
        x = jax.vmap(lambda xi: hyp_relu(xi, c))(x)

        x = self.layer2(x, c)
        x = jax.vmap(lambda xi: hyp_relu(xi, c))(x)

        x = self.layer3(x, c)
        return x

# Create and use model
model = HyperbolicNN(rngs=nnx.Rngs(0))

# Input data (projected to Poincaré ball)
x = jax.random.normal(nnx.Rngs(1).params(), (32, 784)) * 0.1
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

output = model(x_proj, c=1.0)
print(output.shape)  # (32, 10)
```

## References

The neural network layers implement methods from:

- **Ganea et al. (2018)**: "Hyperbolic Neural Networks" - Poincaré linear layers and activations
- **Shimizu et al. (2020)**: "Hyperbolic Neural Networks++" - Enhanced Poincaré operations
- **Bdeir et al. (2023)**: "Fully Hyperbolic Convolutional Neural Networks for Computer Vision" - HCat-based convolutions (`HypConv2DHyperboloid`)
- **Chen et al. (2022)**: "Fully Hyperbolic Neural Networks" - FHCNN linear layers
- **LResNet (2023)**: "Lorentzian ResNet" - HRC-based convolutions (`LorentzConv2D`)
- **Hypformer**: "Hyperbolic Transformers" - HTC/HRC components with curvature-change support

### Key Theoretical Connections

- **HL (Hyperbolic Layer)** from LResNet ≡ **HRC (Hyperbolic Regularization Component)** from Hypformer
- Both apply Euclidean operations to spatial components and reconstruct time using the Lorentz constraint
- `LorentzConv2D` is a specific instance of `hrc()` where `f_r` is a 2D convolution

See also:

- [Manifolds API](manifolds.md): Underlying geometric operations
- [Optimizers API](optimizers.md): Training with Riemannian optimization
- [Training Workflows](../user-guide/training-workflows.md): Complete training examples
