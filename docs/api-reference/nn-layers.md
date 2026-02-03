# Neural Network Layers API

Hyperbolic neural network layers built with Flax NNX.

## Overview

Hyperbolix provides 13+ neural network layer classes and 4 activation functions for building hyperbolic deep learning models:

- **Linear Layers**: Poincaré and Hyperboloid linear transformations
- **Convolutional Layers**: 2D and 3D hyperbolic convolutions
- **Regression Layers**: Single-layer classifiers with Riemannian geometry
- **Activation Functions**: Hyperbolic ReLU, Leaky ReLU, Tanh, Swish
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
x_proj = jax.vmap(poincare.proj, in_axes=(0, None, None))(x, 1.0, None)

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

::: hyperbolix.nn_layers.HypConvHyperboloid
    options:
      show_source: false
      heading_level: 4

!!! note "Backward Compatibility"
    `HypConvHyperboloid` is an alias for `HypConv2DHyperboloid` for backward compatibility.

### Lorentz Convolutions

Implements "Fully Hyperbolic CNNs" (Bdeir et al., 2023).

::: hyperbolix.nn_layers.LorentzConv2D
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.LorentzConv3D
    options:
      show_source: true
      heading_level: 4

### Convolution Example

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

### Lorentz Convolution Example

```python
from hyperbolix.nn_layers import LorentzConv2D
from hyperbolix.manifolds import hyperboloid
from flax import nnx

# Lorentz convolution with boost and rescaling
conv = LorentzConv2D(
    manifold_module=hyperboloid,
    out_channels=64,
    kernel_size=(3, 3),
    stride=(1, 1),
    input_space='tangent',  # or 'ambient'
    use_lorentz_boost=True,
    use_distance_rescale=True,
    rngs=nnx.Rngs(0)
)

# Input in tangent space
x_tangent = jax.random.normal(nnx.Rngs(1).params(), (8, 28, 28, 32))

output = conv(x_tangent, c=1.0)
print(output.shape)  # (8, 28, 28, 64+1) - ambient dimension
```

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
x_proj = jax.vmap(poincare.proj, in_axes=(0, None, None))(x, 1.0, None)

# Forward pass returns logits
logits = regressor(x_proj, c=1.0)
print(logits.shape)  # (64, 10)

# Use with softmax for classification
probs = jax.nn.softmax(logits, axis=-1)
```

## Activation Functions

Hyperbolic activation functions that preserve manifold constraints.

::: hyperbolix.nn_layers.hyp_relu
    options:
      show_source: true
      heading_level: 3

::: hyperbolix.nn_layers.hyp_leaky_relu
    options:
      show_source: true
      heading_level: 3

::: hyperbolix.nn_layers.hyp_tanh
    options:
      show_source: true
      heading_level: 3

::: hyperbolix.nn_layers.hyp_swish
    options:
      show_source: true
      heading_level: 3

### Activation Example

```python
from hyperbolix.nn_layers import hyp_relu
from hyperbolix.manifolds import hyperboloid
import jax.numpy as jnp

# Points on hyperboloid (ambient coordinates)
x = jax.random.normal(jax.random.PRNGKey(0), (10, 5))
x_ambient = jnp.concatenate([
    jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.0/1.0),
    x
], axis=-1)

# Apply hyperbolic ReLU
output = hyp_relu(x_ambient, c=1.0)
print(output.shape)  # (10, 6) - same shape

# Verify manifold constraint
constraint = output[:, 0]**2 - jnp.sum(output[:, 1:]**2, axis=-1) - 1.0/1.0
print(jnp.allclose(constraint, 0.0, atol=1e-5))  # True
```

!!! info "How Activations Work"
    Hyperbolic activations:

    1. Apply activation to **space components** only
    2. Reconstruct **time component** using manifold constraint
    3. Avoid expensive exp/log maps for better stability

    This is more efficient than naive approaches while preserving geometry.

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
x_proj = jax.vmap(poincare.proj, in_axes=(0, None, None))(x, 1.0, None)

output = model(x_proj, c=1.0)
print(output.shape)  # (32, 10)
```

## References

The neural network layers implement methods from:

- Ganea et al. (2018): "Hyperbolic Neural Networks"
- Shimizu et al. (2020): "Hyperbolic Neural Networks++"
- Bdeir et al. (2023): "Fully Hyperbolic Convolutional Neural Networks"

See also:

- [Manifolds API](manifolds.md): Underlying geometric operations
- [Optimizers API](optimizers.md): Training with Riemannian optimization
- [Training Workflows](../user-guide/training-workflows.md): Complete training examples
