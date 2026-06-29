# Neural Network Layers API

Hyperbolic neural network layers built with Flax NNX.

## Overview

Hyperbolix provides 20+ neural network layer classes and 5 activation functions for building hyperbolic deep learning models:

- **Linear Layers**: Poincaré, Hyperboloid, and Proper Velocity linear transformations, including FGG (Fast and Geometrically Grounded) and Busemann FC (point-to-horosphere) layers
- **Convolutional Layers**: HCat-based, HRC-based, FGG, and Proper Velocity hyperbolic convolutions
- **Normalization**: Poincaré batch normalization (`PoincareBatchNorm2D`), HRC-wrapped norms, and FGG mean-only batch norm
- **Hypformer Components**: HTC (Hyperbolic Transformation Component) and HRC (Hyperbolic Regularization Component) with curvature-change support
- **FGG Components**: `FGGLinear`, `FGGConv2D`, `FGGMeanOnlyBatchNorm` from Klis et al. (2026) — linear-distance growth, ~3× faster than prior work
- **Proper Velocity Components**: `HypLinearPV`, `HypConv2DPV`, `HypRegressionPV` from Chen et al. (2026) — unconstrained $\mathbb{R}^n$ geometry with exact Euclidean retraction
- **Attention Layers**: Three hyperbolic attention variants (linear O(N), softmax O(N²), full Lorentzian O(N²)) from the Hypformer paper
- **Positional Encoding**: HOPE (Hyperbolic Rotary PE) and Hypformer learnable positional encodings for Transformers
- **Regression Layers**: Single-layer classifiers with Riemannian geometry, including `FGGLorentzMLR`, `HypRegressionPV`, and the Busemann MLR heads (point-to-horosphere)
- **Vector Quantization**: Poincaré VQ-VAE bottlenecks — `HypVQEmbeddingPoincare` (EMA codebook, GGBall) and `HypVQMLRPoincare` (Gumbel-Softmax over a Poincaré MLR)
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

::: hyperbolix.nn_layers.HypLinearPoincareBusemann
    options:
      show_source: true
      heading_level: 4

### Hyperboloid Linear

::: hyperbolix.nn_layers.HypLinearHyperboloidFHCNN
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HypLinearHyperboloidFHNN
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HypLinearHyperboloidPLFC
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HypLinearHyperboloidBusemann
    options:
      show_source: true
      heading_level: 4

### FGG Linear (Klis et al. 2026)

::: hyperbolix.nn_layers.FGGLinear
    options:
      show_source: true
      heading_level: 4

### Proper Velocity Linear (Chen et al. 2026)

`HypLinearPV` implements the PV fully-connected layer from Chen et al. 2026 (Thm 5.3 / Eq. 22): $y_k = (1/\sqrt{c}) \cdot \sinh(\sqrt{c} \cdot v_k(x))$ where $v_k(x)$ is the PV multinomial-logistic-regression signed margin. Because PV is an unconstrained $\mathbb{R}^n$ model, inputs and outputs live in Euclidean space with no projection step.

::: hyperbolix.nn_layers.HypLinearPV
    options:
      show_source: true
      heading_level: 4

### Usage Example

```python
import jax
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Create hyperbolic linear layer
layer = HypLinearPoincare(
    manifold_module=poincare,
    in_dim=32,
    out_dim=16,
    rngs=nnx.Rngs(0)
)

# Forward pass
x = jax.random.normal(jax.random.PRNGKey(1), (10, 32)) * 0.3
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

::: hyperbolix.nn_layers.HypConv2DHyperboloidFHNN
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HypConv2DHyperboloidILNN
    options:
      show_source: true
      heading_level: 4

### Usage Example

```python
import jax
import jax.numpy as jnp
from hyperbolix.nn_layers import HypConv2DHyperboloid
from hyperbolix.manifolds import Hyperboloid
from flax import nnx

hyperboloid = Hyperboloid()

# Create 2D hyperbolic convolution
conv = HypConv2DHyperboloid(
    manifold_module=hyperboloid,
    in_channels=16,
    out_channels=32,
    kernel_size=(3, 3),
    stride=(1, 1),
    rngs=nnx.Rngs(0)
)

# Input: (batch, height, width, in_channels) — ambient dim = in_channels+1
x = jax.random.normal(jax.random.PRNGKey(1), (8, 28, 28, 16))

# Project to hyperboloid
x_ambient = jnp.concatenate([
    jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.0),
    x
], axis=-1)  # (8, 28, 28, 17)

# Forward pass (input_space="manifold" by default)
output = conv(x_ambient, c=1.0)
print(output.shape)  # (8, 28, 28, 32×9+1) - dimension grows!
```

!!! warning "Dimensional Growth"
    Hyperboloid convolutions increase dimensionality via HCat operation:

    - Input: `d+1` dimensions
    - Output: `(d×N)+1` dimensions where `N = kernel_height × kernel_width`

    For 3×3 kernel: 3D input → 28D output. Use small kernels or add dimensionality reduction layers.

### Poincaré Convolution

::: hyperbolix.nn_layers.HypConv2DPoincare
    options:
      show_source: true
      heading_level: 4

`HypConv2DPoincare` extracts patches, applies beta-concatenation (HNN++, Shimizu et al. 2020) over the receptive field, then passes through a `HypLinearPoincarePP` layer. Dimension math: `K² × C_in → C_out` where `K` is the kernel size.

**Key Differences from Hyperboloid Convolutions:**

| Feature | HypConv2DPoincare | HypConv2DHyperboloid |
|---------|-------------------|----------------------|
| **Model** | Poincaré ball | Hyperboloid |
| **Aggregation** | Beta-concatenation | HCat (Lorentz concatenation) |
| **Dimension** | Preserved | Grows: `(d-1)×K²+1` |
| **Default input** | Tangent space | Manifold (ambient) |

### Usage Example

```python
import jax
import jax.numpy as jnp
from hyperbolix.nn_layers import HypConv2DPoincare
from hyperbolix.manifolds import Poincare
from flax import nnx

poincare = Poincare()

# Create Poincaré 2D convolution
conv = HypConv2DPoincare(
    manifold_module=poincare,
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    stride=1,
    rngs=nnx.Rngs(0)
)

# Input: (batch, height, width, in_channels) in tangent space (default input_space="tangent")
x = jax.random.normal(jax.random.PRNGKey(1), (8, 28, 28, 16)) * 0.1

# Forward pass — returns tangent-space output
output = conv(x, c=1.0)
print(output.shape)  # (8, 28, 28, 32)
```

### Poincaré Batch Normalization

::: hyperbolix.nn_layers.PoincareBatchNorm2D
    options:
      show_source: true
      heading_level: 4

`PoincareBatchNorm2D` operates in tangent space (matching conv layer I/O), mapping to the manifold internally for geometric operations: Einstein midpoint, Fréchet variance, parallel transport, and variance rescaling. Use between Poincaré convolution layers following the reference ResNet pattern: `conv → bn → relu → conv → bn → skip`.

::: hyperbolix.nn_layers.poincare_midpoint
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.frechet_variance
    options:
      show_source: true
      heading_level: 4

#### Poincaré BatchNorm Example

```python
from hyperbolix.nn_layers import HypConv2DPoincare, PoincareBatchNorm2D
from hyperbolix.manifolds import Poincare
from flax import nnx
import jax

poincare = Poincare()

# ResNet-style block: conv → bn → relu
conv = HypConv2DPoincare(
    manifold_module=poincare,
    in_channels=16, out_channels=16,
    kernel_size=3, rngs=nnx.Rngs(0),
)
bn = PoincareBatchNorm2D(poincare, num_features=16)

# Input: tangent-space features (B, H, W, C)
x = jax.random.normal(jax.random.PRNGKey(0), (4, 8, 8, 16)) * 0.1

# Training: use_running_average=False (default)
h = conv(x, c=0.1)
h = bn(h, c=0.1)
h = jax.nn.relu(h)

# Evaluation: use_running_average=True
h_eval = conv(x, c=0.1)
h_eval = bn(h_eval, c=0.1, use_running_average=True)
h_eval = jax.nn.relu(h_eval)
```

### FGGConv2D (Klis et al. 2026)

::: hyperbolix.nn_layers.FGGConv2D
    options:
      show_source: true
      heading_level: 4

`FGGConv2D` combines HCat patch extraction with `FGGLinear` for channel mixing. Unlike `HypConv2DHyperboloid`, it uses the FGG spacelike-V construction, achieving linear growth of hyperbolic distance rather than logarithmic. Supports manifold-origin padding (`pad_mode="origin"`) matching the reference implementation.

| Feature | FGGConv2D | HypConv2DHyperboloid | HypConv2DHyperboloidILNN |
|---------|-----------|----------------------|------------------------|
| **Linear layer** | FGGLinear (V-matrix) | HypLinearHyperboloidFHCNN | HypLinearHyperboloidPLFC (MLR) |
| **Patch concatenation** | HCat | HCat | LogCat (log-radius-preserving) |
| **Distance growth** | Linear | Logarithmic | Logarithmic |
| **Default padding** | Manifold origin | Edge replication | Manifold origin |
| **Weight norm** | Optional (`use_weight_norm`) | No | No |

### Proper Velocity Convolution (Chen et al. 2026)

`HypConv2DPV` implements the PV 2D convolution from Chen et al. 2026 (Sec 5.3). Because PV's geometry is unconstrained ($\mathbb{R}^n$), patch concatenation coincides with Euclidean concatenation — **no beta-scaling step is required** (unlike `HypConv2DPoincare`). Outputs live on the PV manifold, so activations can be applied directly between conv layers without expmap/logmap round-trips.

::: hyperbolix.nn_layers.HypConv2DPV
    options:
      show_source: true
      heading_level: 4

| Feature | HypConv2DPV | HypConv2DPoincare | HypConv2DHyperboloid |
|---------|-------------|-------------------|----------------------|
| **Model** | Proper Velocity ($\mathbb{R}^n$) | Poincaré ball | Hyperboloid |
| **Aggregation** | Raw Euclidean concat | Beta-concatenation | HCat (Lorentz concat) |
| **Dimension** | Preserved | Preserved | Grows: $(d-1) K^2 + 1$ |
| **Default input** | Manifold | Tangent space | Manifold (ambient) |
| **Output** | PV manifold | Tangent space | Hyperboloid (ambient) |

### LorentzConv2D (HRC-Based)

::: hyperbolix.nn_layers.LorentzConv2D
    options:
      show_source: true
      heading_level: 4

LorentzConv2D provides a simpler, more efficient alternative to HCat-based convolutions by using the Hyperbolic Regularization Component (HRC) pattern from the Hypformer paper.

**Key Differences from HypConv2DHyperboloid:**

| Feature | HypConv2DHyperboloid (HCat+FHCNN) | HypConv2DHyperboloidILNN (LogCat+PLFC) | LorentzConv2D (HRC) |
|---------|-----------------------------------|--------------------------------------|-------------------|
| **Method** | HCat + FHCNN linear | LogCat + PLFC (MLR + sinh diffeomorphism) | Euclidean conv on space components |
| **Dimension** | Grows: `(d-1)×N+1` | Grows: `(d-1)×N+1` | Preserved |
| **Speed** | Slower (~80s/epoch) | Similar to FHCNN | **2.5x faster** (~32s/epoch) |
| **Accuracy** | Higher (~71% on MNIST) | Intrinsic Lorentz conv (Shi et al. 2026) | Lower (~46% on MNIST) |
| **Use Case** | HCat accuracy | Deep intrinsic Lorentz networks | Speed/memory efficiency |

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

    Choose HypConv2DHyperboloid or HypConv2DHyperboloidILNN when:

    - Maximum accuracy is required
    - Willing to accept slower training and dimensional growth
    - Use HypConv2DHyperboloidILNN for deeper networks (intrinsic Lorentz conv: LogCat + PLFC, Shi et al. 2026)

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

::: hyperbolix.nn_layers.HRCRMSNorm
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HRCDropout
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.FGGMeanOnlyBatchNorm
    options:
      show_source: true
      heading_level: 4

### Hypformer Example

```python
from hyperbolix.nn_layers import HTCLinear, HRCBatchNorm, HRCRMSNorm, hrc_relu
from hyperbolix.manifolds import Hyperboloid
from flax import nnx
import jax
import jax.numpy as jnp

hyperboloid = Hyperboloid()

class HypformerBlock(nnx.Module):
    """Example using HTC/HRC components with curvature change."""

    def __init__(self, in_dim, out_dim, rngs):
        self.linear = HTCLinear(
            in_features=in_dim,
            out_features=out_dim,
            rngs=rngs
        )
        # Can use BatchNorm or RMSNorm for normalization
        self.bn = HRCBatchNorm(num_features=out_dim, rngs=rngs)
        # self.rms = HRCRMSNorm(num_features=out_dim, rngs=rngs)  # Alternative: faster, simpler

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
x = jax.random.normal(jax.random.PRNGKey(1), (32, 33))
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

## Attention Layers

Three hyperbolic attention variants from the Hypformer paper (Yang et al. 2025, Section 4.3). All operate on hyperboloid points and support independent curvatures for input (`c_in`), attention computation (`c_attn`), and output (`c_out`). All variants support **causal (autoregressive) masking** via the `causal=True` flag, making them suitable for language models and sequence generation tasks.

### Core Utilities

::: hyperbolix.nn_layers.spatial_to_hyperboloid
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.lorentz_midpoint
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.focus_transform
    options:
      show_source: true
      heading_level: 4

### Attention Modules

::: hyperbolix.nn_layers.HyperbolicLinearAttention
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HyperbolicSoftmaxAttention
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HyperbolicFullAttention
    options:
      show_source: true
      heading_level: 4

### Attention Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import (
    HyperbolicLinearAttention,
    HyperbolicSoftmaxAttention,
    HyperbolicFullAttention,
)

hyperboloid = Hyperboloid()

# Input: (batch, seq_len, ambient_dim) on the hyperboloid
B, N, A_in, D_out = 4, 8, 9, 8  # 8-dim spatial + 1 time
key = jax.random.PRNGKey(0)
spatial = jax.random.normal(key, (B, N, A_in - 1)) * 0.1
time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
x = jnp.concatenate([time, spatial], axis=-1)  # (B, N, A_in)

# O(N) linear attention with focus function — fastest, main Hypformer contribution
linear_attn = HyperbolicLinearAttention(
    in_features=A_in,
    out_features=D_out,
    num_heads=2,
    power=2.0,
    rngs=nnx.Rngs(0),
)
y = linear_attn(x, c_in=1.0, c_attn=1.0, c_out=1.0)
print(y.shape)  # (4, 8, 9) — D_out spatial + 1 time

# O(N²) softmax attention in the spatial domain
softmax_attn = HyperbolicSoftmaxAttention(
    in_features=A_in,
    out_features=D_out,
    num_heads=2,
    rngs=nnx.Rngs(0),
)
y = softmax_attn(x, c_in=1.0, c_attn=1.0, c_out=1.0)

# O(N²) full Lorentzian attention — operates entirely on hyperboloid points
full_attn = HyperbolicFullAttention(
    in_features=A_in,
    out_features=D_out,
    num_heads=2,
    rngs=nnx.Rngs(0),
)
y = full_attn(x, c_in=1.0, c_attn=1.0, c_out=1.0)

# Verify outputs are on the hyperboloid
for b in range(B):
    for n in range(N):
        assert hyperboloid.is_in_manifold(y[b, n], c=1.0, atol=1e-4)
```

#### Causal (Autoregressive) Attention

All three variants support causal masking via `causal=True`. Position `n` can only attend to positions `m ≤ n`, which is required for autoregressive tasks like language modeling.

```python
# Bidirectional (default) — each token attends to all tokens
y_bidir = softmax_attn(x, c_in=1.0, c_attn=1.0, c_out=1.0, causal=False)

# Causal — position n only attends to positions 0..n
y_causal = softmax_attn(x, c_in=1.0, c_attn=1.0, c_out=1.0, causal=True)

# Causal is JIT-compatible
@nnx.jit
def forward(model, inp):
    return model(inp, c_in=1.0, c_attn=1.0, c_out=1.0, causal=True)
```

!!! info "Causal masking implementations"
    The three variants implement causal masking differently:

    - **`HyperbolicSoftmaxAttention`** and **`HyperbolicFullAttention`**: Apply a lower-triangular `-inf` mask to the score matrix before softmax — O(N²) in both causal and non-causal mode.
    - **`HyperbolicLinearAttention`**: Uses a cumulative-sum recurrence (`jax.lax.scan`) following Katharopoulos et al. (2020): `S_i = Σ_{j≤i} φ(K_j) V_j^T` computed in O(1) per step → **O(N) total**, making it especially well-suited for long autoregressive sequences.

!!! info "Choosing an Attention Variant"
    | Variant | Complexity | Causal complexity | Mechanism | Best For |
    |---------|-----------|-------------------|-----------|----------|
    | `HyperbolicLinearAttention` | O(N) | **O(N)** | Kernel trick + focus function φ | Long sequences, autoregressive models |
    | `HyperbolicSoftmaxAttention` | O(N²) | O(N²) | Standard softmax on spatial components | Short sequences, simplicity |
    | `HyperbolicFullAttention` | O(N²) | O(N²) | Lorentzian inner product + midpoint | Maximum geometric fidelity |

    All variants support independent curvatures: `c_in` for input, `c_attn` for Q/K/V projections, `c_out` for output.

## Positional Encoding

Positional encoding layers for hyperbolic Transformers and attention mechanisms. These layers enable position-aware models on the hyperboloid manifold while preserving geometric structure.

### Lorentzian Residual Connection

::: hyperbolix.nn_layers.lorentz_residual
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.lorentz_scale
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.LorentzResidual
    options:
      show_source: true
      heading_level: 4

### HOPE (Hyperbolic Rotary Positional Encoding)

::: hyperbolix.nn_layers.hope
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HyperbolicRoPE
    options:
      show_source: true
      heading_level: 4

### Hypformer Positional Encoding

::: hyperbolix.nn_layers.HypformerPositionalEncoding
    options:
      show_source: true
      heading_level: 4

### HOPE Example

```python
from hyperbolix.nn_layers import hope, HyperbolicRoPE
from hyperbolix.manifolds import Hyperboloid
import jax.numpy as jnp
import jax

hyperboloid = Hyperboloid()

# Create sequence of hyperboloid points (batch, seq_len, d+1)
key = jax.random.PRNGKey(42)
batch, seq_len, d = 4, 16, 8
spatial = jax.random.normal(key, (batch, seq_len, d)) * 0.1
time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
z = jnp.concatenate([time, spatial], axis=-1)  # (4, 16, 9)

# Position indices
positions = jnp.arange(seq_len)

# Apply HOPE (functional interface)
z_encoded = hope(z, positions, c=1.0)
print(z_encoded.shape)  # (4, 16, 9)

# Or use the NNX module wrapper
from flax import nnx
rope = HyperbolicRoPE(dim=d, max_seq_len=64, base=10000.0)
z_encoded = rope(z, positions, c=1.0)

# Verify manifold constraint
for b in range(batch):
    for s in range(seq_len):
        assert hyperboloid.is_in_manifold(z_encoded[b, s], c=1.0, atol=1e-4)

# Verify relative position property: <HOPE(q,i), HOPE(k,j)>_L depends only on i-j
q = z[0, 0]  # Single point
k = z[0, 1]  # Another point

# Same relative offset (=3) at different absolute positions
q_enc_0 = hope(q[None, None, :], jnp.array([0]), c=1.0)[0, 0]
k_enc_3 = hope(k[None, None, :], jnp.array([3]), c=1.0)[0, 0]

q_enc_10 = hope(q[None, None, :], jnp.array([10]), c=1.0)[0, 0]
k_enc_13 = hope(k[None, None, :], jnp.array([13]), c=1.0)[0, 0]

# Minkowski inner products should be equal
def minkowski_inner(x, y):
    return -x[0]*y[0] + jnp.sum(x[1:]*y[1:])

ip1 = minkowski_inner(q_enc_0, k_enc_3)
ip2 = minkowski_inner(q_enc_10, k_enc_13)
print(jnp.allclose(ip1, ip2, atol=1e-5))  # True
```

### Hypformer Positional Encoding Example

```python
from hyperbolix.nn_layers import HypformerPositionalEncoding
from hyperbolix.manifolds import Hyperboloid
from flax import nnx

hyperboloid = Hyperboloid()
import jax.numpy as jnp
import jax

# Create positional encoding layer
d = 8  # spatial dimension
in_features = d + 1  # ambient dimension (including time)
pe = HypformerPositionalEncoding(
    in_features=in_features,
    out_features=d,  # output spatial dimension
    rngs=nnx.Rngs(0),
    init_bound=0.02  # small initialization for stability
)

# Input: batch of hyperboloid points
key = jax.random.PRNGKey(42)
batch_size = 32
spatial = jax.random.normal(key, (batch_size, d)) * 0.1
time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
x = jnp.concatenate([time, spatial], axis=-1)  # (32, 9)

# Apply positional encoding
x_encoded = pe(x, c=1.0)
print(x_encoded.shape)  # (32, 9) - shape preserved

# Verify manifold constraint
is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(x_encoded, 1.0)
print(is_valid.all())  # True

# epsilon is a FIXED scalar (default 1.0, matching the Hypformer reference);
# only the HTCLinear weights train. A custom non-negative value can be set:
pe_weak = HypformerPositionalEncoding(in_features=d + 1, out_features=d, rngs=nnx.Rngs(0), epsilon=0.3)
```

!!! warning "Why `epsilon` is not learnable"
    The Hypformer reference keeps `epsilon` a plain (non-trainable) tensor fixed at 1.0.
    Making it a parameter is unsafe: gradient descent can drive it below -1, where
    `x + epsilon * p` leaves the upper hyperboloid sheet and the `abs()` in the
    Lorentzian residual normalizer silently masks the violation instead of raising.
    For the same reason, `lorentz_residual`'s `w_y` must always be non-negative.

### Lorentzian Residual Example

```python
from hyperbolix.nn_layers import lorentz_residual
from hyperbolix.manifolds import Hyperboloid
import jax.numpy as jnp
import jax

hyperboloid = Hyperboloid()

# Create two hyperboloid points
key1, key2 = jax.random.split(jax.random.PRNGKey(42))
d = 6
c = 1.0

spatial_x = jax.random.normal(key1, (d,)) * 0.1
time_x = jnp.sqrt(jnp.sum(spatial_x**2) + 1/c)
x = jnp.concatenate([time_x[None], spatial_x])

spatial_y = jax.random.normal(key2, (d,)) * 0.1
time_y = jnp.sqrt(jnp.sum(spatial_y**2) + 1/c)
y = jnp.concatenate([time_y[None], spatial_y])

# Combine with Lorentzian residual (weighted midpoint)
result = lorentz_residual(x, y, w_y=0.5, c=c)

# Verify output is on hyperboloid
assert hyperboloid.is_in_manifold(result, c, atol=1e-5)

# Works with batches too
x_batch = jax.random.normal(jax.random.PRNGKey(0), (8, d+1))
y_batch = jax.random.normal(jax.random.PRNGKey(1), (8, d+1))

# Project to manifold
x_batch = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x_batch, c)
y_batch = jax.vmap(hyperboloid.proj, in_axes=(0, None))(y_batch, c)

# Apply residual connection
result_batch = lorentz_residual(x_batch, y_batch, w_y=0.3, c=c)
print(result_batch.shape)  # (8, 7)
```

!!! info "Positional Encoding for Hyperbolic Transformers"
    **HOPE (Hyperbolic Rotary Positional Encoding)**:

    - Deterministic, no learnable parameters
    - Based on RoPE: applies rotations to spatial components
    - Preserves relative position information: `⟨HOPE(q,i), HOPE(k,j)⟩_L` depends only on `i-j`
    - Rotation is an isometry: preserves spatial norms
    - Identity at position 0
    - Suitable for long sequences (no learned embeddings to store)

    **HypformerPositionalEncoding**:

    - Learnable (HTCLinear weights), adapts to task
    - Uses HTCLinear + Lorentzian residual
    - `epsilon` (fixed, non-negative) controls position encoding magnitude
    - More flexible but requires training
    - Suitable when position patterns are task-specific

    **Lorentzian Residual**:

    - Building block for both approaches
    - Computes weighted Lorentzian midpoint
    - Used for skip connections in hyperbolic Transformers/ResNets
    - Formula: `ave = x + w_y*y`, normalized to hyperboloid

## Hybrid Euclidean-Hyperbolic

### Hyper++ Feature Scaling

::: hyperbolix.nn_layers.HyperPPFeatureScaling
    options:
      show_source: true
      heading_level: 4

`HyperPPFeatureScaling` is applied after the last Euclidean layer and before `expmap_0` to either the Poincaré ball or Hyperboloid. It operates entirely in Euclidean space but uses hyperbolic geometry to bound the output norm via `rho_max = atanh(alpha) / sqrt(c)`.

### Usage Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from hyperbolix.nn_layers import HyperPPFeatureScaling
from hyperbolix.manifolds import Poincare

poincare = Poincare()

# Parameter-free mode (RMSNorm + tanh + dim scaling only)
layer = HyperPPFeatureScaling(dim=64, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(0), (32, 64))
scaled = layer(x, c=1.0)

# With learned rescaling (adds rho_max * sigmoid(xi(x)) * x)
layer = HyperPPFeatureScaling(dim=64, alpha=0.9, rngs=nnx.Rngs(0))
scaled = layer(x, c=0.1)

# Map to Poincaré ball
expmap_batch = jax.vmap(poincare.expmap_0, in_axes=(0, None))
points = expmap_batch(scaled, 0.1)
```

!!! info "Pipeline Steps"
    1. **RMSNorm** (parameter-free): normalizes feature magnitudes
    2. **Lipschitz activation** (default `tanh`, configurable): bounds per-component values
    3. **Dimension scaling** (`1/sqrt(d)`): ensures norm doesn't grow with dimension
    4. **Learned rescaling** (when `alpha` is set): `rho_max * sigmoid(xi_theta(x)) * x` where `rho_max = atanh(alpha) / sqrt(c)`

    When `alpha is None`, the layer is entirely parameter-free. When set, only `xi_theta` (a linear projection to scalar) has learnable parameters.

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

::: hyperbolix.nn_layers.HypRegressionPoincareBusemann
    options:
      show_source: true
      heading_level: 4

### Hyperboloid Regression

::: hyperbolix.nn_layers.HypRegressionHyperboloid
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.HypRegressionHyperboloidBusemann
    options:
      show_source: true
      heading_level: 4

### Busemann MLR (Chen et al. 2026)

The Busemann MLR heads (`HypRegressionHyperboloidBusemann`, `HypRegressionPoincareBusemann`) decide
with point-to-*horosphere* distances (the Chen/Atigh/Fan lineage), in contrast to the point-to-*hyperplane*
heads above (Ganea/Shimizu/Bdeir). Each logit is $u_k(x) = -\alpha_k B^{v_k}(x) + b_k$ from the closed-form
Busemann function `Hyperboloid.busemann` / `Poincare.busemann`.

### FGG Lorentz MLR (Klis et al. 2026)

::: hyperbolix.nn_layers.FGGLorentzMLR
    options:
      show_source: true
      heading_level: 4

### Proper Velocity Regression (Chen et al. 2026)

`HypRegressionPV` implements the PV multinomial-logistic-regression layer (Thm 5.2 / Eq. 19). The output is the Euclidean signed margin to each PV hyperplane, intended for a standard softmax-cross-entropy loss.

::: hyperbolix.nn_layers.HypRegressionPV
    options:
      show_source: true
      heading_level: 4

#### PV Usage Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import ProperVelocity
from hyperbolix.nn_layers import HypConv2DPV, HypLinearPV, HypRegressionPV

pv = ProperVelocity()
rngs = nnx.Rngs(0)
c = 1.0

# --- HypConv2DPV: dimension-preserving hyperbolic convolution ---
conv = HypConv2DPV(
    manifold_module=pv,
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    rngs=rngs,
    input_space="tangent",  # lift Euclidean features via expmap_0
)
x = jax.random.normal(jax.random.PRNGKey(1), (8, 28, 28, 16)) * 0.1
h = conv(x, c=c)
print(h.shape)  # (8, 28, 28, 32) — on the PV manifold

# --- HypLinearPV: fully-connected PV layer ---
linear = HypLinearPV(
    manifold_module=pv,
    in_dim=32,
    out_dim=64,
    rngs=rngs,
)
h_flat = h.reshape(-1, 32)
h_fc = linear(h_flat, c=c)
print(h_fc.shape)  # (8*28*28, 64)

# --- HypRegressionPV: classification head (Euclidean logits) ---
mlr = HypRegressionPV(
    manifold_module=pv,
    in_dim=64,
    out_dim=10,
    rngs=rngs,
)
logits = mlr(h_fc, c=c)
print(logits.shape)  # (8*28*28, 10) — feed to softmax cross-entropy
```

!!! tip "When to Use Proper Velocity"
    - **Large-radius features** — the unbounded $\mathbb{R}^n$ coordinates avoid both the Poincaré boundary ($\lambda \to 0$) and hyperboloid constraint drift.
    - **Standard Euclidean optimizers** — the retraction is exact Euclidean addition, so Adam/SGD on PV parameters needs no Riemannian wrapper.
    - **Simple conv stacks** — raw Euclidean concat replaces beta-scaling/HCat, and no dimension growth per layer.

### FGG Usage Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import FGGLinear, FGGConv2D, FGGLorentzMLR, FGGMeanOnlyBatchNorm

hyperboloid = Hyperboloid()
rngs = nnx.Rngs(0)

# --- FGGLinear: FC layer with linear hyperbolic distance growth ---
linear = FGGLinear(
    in_features=33,   # 32 spatial + 1 time
    out_features=65,  # 64 spatial + 1 time
    rngs=rngs,
    activation=jax.nn.relu,
    reset_params="lorentz_kaiming",
)
x_B33 = jnp.ones((8, 33))
x_B33 = x_B33.at[:, 0].set(jnp.sqrt(jnp.sum(x_B33[:, 1:]**2, axis=-1) + 1.0))
y_B65 = linear(x_B33, c=1.0)
print(y_B65.shape)  # (8, 65)

# --- FGGConv2D: 2D conv with manifold-origin padding ---
conv = FGGConv2D(
    manifold_module=hyperboloid,
    in_channels=33,
    out_channels=65,
    kernel_size=3,
    rngs=rngs,
    activation=jax.nn.relu,
    pad_mode="origin",   # manifold origin padding (reference default)
)
x_BHWC = jnp.zeros((4, 14, 14, 33))
x_BHWC = x_BHWC.at[..., 0].set(1.0)  # valid origin point at c=1
y_BHWC = conv(x_BHWC, c=1.0)
print(y_BHWC.shape)  # (4, 14, 14, 65)

# --- FGGLorentzMLR: classification head ---
mlr = FGGLorentzMLR(
    in_features=65,
    num_classes=10,
    rngs=rngs,
    reset_params="mlr",   # N(0, sqrt(5/Ai)) where Ai = in_features (ambient)
    init_bias=0.5,
)
logits_B10 = mlr(y_BHWC.reshape(-1, 65), c=1.0)
print(logits_B10.shape)  # (784, 10)

# --- FGGMeanOnlyBatchNorm: pairs with FGGLinear(use_weight_norm=True) ---
# num_features is the SPATIAL (out) dimension (no rngs: zero-initialized state)
bn = FGGMeanOnlyBatchNorm(num_features=64)
y_normed = bn(y_B65, c_in=1.0, c_out=1.0, use_running_average=False)
print(y_normed.shape)  # (8, 65)
```

!!! info "FGG Layer Family"
    The four FGG components from Klis et al. (2026) form a complete layer stack:

    | Layer | Role | Key params |
    |-------|------|-----------|
    | `FGGLinear` | Fully-connected | `reset_params="fan_out"`, `gain=1.0`, `init_bias=0.0`, `use_weight_norm`, `activation` |
    | `FGGConv2D` | 2D convolution | `reset_params="fan_out"`, `gain=1.0`, `init_bias=0.0`, `pad_mode="origin"`, wraps `FGGLinear` |
    | `FGGLorentzMLR` | Classification head | `reset_params="mlr"`, `init_bias=0.5` |
    | `FGGMeanOnlyBatchNorm` | Batch normalization | pairs with `use_weight_norm=True` |

    The `FGGLinear`/`FGGConv2D` defaults (`fan_out` + zero bias) are norm-preserving for *unnormalized* stacks — a deliberate deviation from the Klis et al. classification reference. Restore the reference init with `reset_params="eye"` / `"lorentz_kaiming"` + `init_bias=0.5`.

    **Core insight**: the sinh/arcsinh cancellation in the Lorentzian activation chain reduces the forward pass to a single matmul with a spacelike V matrix, Euclidean activation, then time reconstruction — achieving linear (not logarithmic) growth of hyperbolic distance.

### Regression Example

```python
import jax
from hyperbolix.nn_layers import HypRegressionPoincare
from hyperbolix.manifolds import Poincare
from flax import nnx

poincare = Poincare()

# Multi-class classification (10 classes)
regressor = HypRegressionPoincare(
    manifold_module=poincare,
    in_dim=32,
    out_dim=10,
    rngs=nnx.Rngs(0)
)

# Input: hyperbolic embeddings
x = jax.random.normal(jax.random.PRNGKey(1), (64, 32)) * 0.3
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

# Forward pass returns logits
logits = regressor(x_proj, c=1.0)
print(logits.shape)  # (64, 10)

# Use with softmax for classification
probs = jax.nn.softmax(logits, axis=-1)
```

## Vector Quantization

Hyperbolic VQ-VAE quantizer bottlenecks for the Poincaré ball. Both take encoder features (Euclidean tangent vectors at the origin) and return a `PoincareVQOutput` with the straight-through quantized vector for the decoder, the discrete code indices, an auxiliary loss, and the codebook perplexity. The encoder/decoder stay in user code — these layers are *only* the quantization step.

| Layer | Codebook | Selection | Codebook trained by | Aux loss |
|---|---|---|---|---|
| `HypVQEmbeddingPoincare` | Explicit on-ball table (a non-param **buffer**) | Geodesic nearest-neighbour | Hyperbolic EMA (`ema_update`) | Commitment |
| `HypVQMLRPoincare` | Implicit (rows of a Poincaré MLR) | Gumbel-Softmax over MLR scores | `optax.adam` (Euclidean) | None (recon-only) |

### Embedding VQ (HVQ-VAE, EMA codebook)

::: hyperbolix.nn_layers.HypVQEmbeddingPoincare
    options:
      show_source: true
      heading_level: 4

The codebook is an `nnx.Variable` buffer, so `nnx.Optimizer(model, tx, wrt=nnx.Param)` ignores it — it moves only through `ema_update(z, indices, c)`, the hyperbolic moving average of GGBall (Bu et al. 2026, Eqs. 41-43). Call `ema_update` in the train step **after** `optimizer.update`. Set `dead_code_revival=True` (and pass a `reset_key`) to replace stale codes with random encoder points.

### MLR VQ (HyperVQ, implicit codebook)

::: hyperbolix.nn_layers.HypVQMLRPoincare
    options:
      show_source: true
      heading_level: 4

The codebook is the rows of an internal `HypRegressionPoincarePP` (`bias · kernel`), trained by plain `optax.adam`. Selection is a Gumbel-Softmax straight-through sample over the MLR scores; putting the STE on the categorical weights — not on `z_q` — is what lets the gradient reach the MLR parameters. The layer owns a `deterministic` flag that `model.eval()` / `model.train()` toggle, switching to a deterministic argmax MAP estimate at inference.

### Output type and weighted gyromidpoint

::: hyperbolix.nn_layers.PoincareVQOutput
    options:
      show_source: true
      heading_level: 4

The embedding layer's EMA centroid is the GGBall weighted gyromidpoint, exposed as a reusable helper (the Poincaré analog of `lorentz_midpoint`, generalizing the unweighted `poincare_midpoint`):

::: hyperbolix.nn_layers.poincare_weighted_midpoint
    options:
      show_source: true
      heading_level: 4

### Usage Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypVQEmbeddingPoincare, HypVQMLRPoincare

# Manifold ops run in float64; the quantized decoder input comes back float32,
# and the codebook/cluster-size buffers are stored float32 (param_dtype default).
manifold = Poincare(dtype=jnp.float64)
h = jax.random.normal(jax.random.PRNGKey(1), (256, 64)) * 0.3  # encoder tangent features

# --- HVQ-VAE: EMA codebook (no Riemannian optimizer) ---
vq = HypVQEmbeddingPoincare(
    manifold, num_codes=128, code_dim=64, rngs=nnx.Rngs(0),
    commitment_weight=0.5, ema_decay=0.99, dead_code_revival=True,
    squared_commitment=False,  # False: HVQ-VAE plain d; True: GGBall d² (VQ-VAE convention)
)
out = vq(h, c=1.0)
print(out.quantized.shape, out.quantized.dtype)  # (256, 64) float32 — decoder input
print(out.indices.shape, float(out.loss))        # (256,)  + commitment loss

# In the train step, AFTER optimizer.update(model, grads):
vq.ema_update(out.z, out.indices, c=1.0, reset_key=jax.random.key(7))

# --- HyperVQ: implicit MLR codebook, Gumbel-Softmax ---
mlr_vq = HypVQMLRPoincare(manifold, num_codes=512, code_dim=64, rngs=nnx.Rngs(0))
out = mlr_vq(h, c=1.0, rngs=nnx.Rngs(2))          # rngs drives the Gumbel sample
mlr_vq.eval()                                     # deterministic argmax for inference
out_eval = mlr_vq(h, c=1.0)                        # rngs unused in eval mode
```

!!! info "Two quantizers, two gradient stories"
    - **`HypVQEmbeddingPoincare`** keeps the codebook *on* the ball as a buffer and moves it with a geometric EMA (`ema_update`), entirely separate from the gradient path. Only the commitment loss reaches the encoder; the optimizer never touches the codebook. A copy-gradient STE bridges `logmap_0(q)` to the decoder.
    - **`HypVQMLRPoincare`** has no explicit codebook — quantization is classification over a Poincaré MLR, the STE sits on the categorical weights, and plain `optax.adam` trains everything. Reconstruction-only loss (`output.loss == 0`).

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

### Poincaré Activations

Thin wrappers that apply standard activations in the Poincaré tangent space via `logmap_0 → activation → expmap_0`.

::: hyperbolix.nn_layers.poincare_relu
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.poincare_leaky_relu
    options:
      show_source: true
      heading_level: 4

::: hyperbolix.nn_layers.poincare_tanh
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
import jax
import jax.numpy as jnp
from hyperbolix.nn_layers import hyp_relu, hyp_gelu
from hyperbolix.manifolds import Hyperboloid

hyperboloid = Hyperboloid()

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

## Building Models

Example of a complete hyperbolic neural network:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from hyperbolix.nn_layers import HypLinearPoincare, hyp_relu
from hyperbolix.manifolds import Poincare

poincare = Poincare()

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
x = jax.random.normal(jax.random.PRNGKey(1), (32, 784)) * 0.1
x_proj = jax.vmap(poincare.proj, in_axes=(0, None))(x, 1.0)

output = model(x_proj, c=1.0)
print(output.shape)  # (32, 10)
```

## References

The neural network layers implement methods from:

- **Ganea et al. (2018)**: "Hyperbolic Neural Networks" - Poincaré linear layers and activations
- **Shimizu et al. (2020)**: "Hyperbolic Neural Networks++" - Enhanced Poincaré operations and the linearized-kernel conv formulation (`HypLinearPoincarePP`; basis of `HypLinearHyperboloidPLFC` and `HypConv2DHyperboloidILNN`)
- **Bdeir et al. (2023)**: "Fully Hyperbolic Convolutional Neural Networks for Computer Vision" - HCat-based convolutions (`HypConv2DHyperboloid`)
- **Chen et al. (2022)**: "Fully Hyperbolic Neural Networks" - FHCNN linear layers
- **LResNet (2023)**: "Lorentzian ResNet" - HRC-based convolutions (`LorentzConv2D`)
- **Hypformer (Yang et al. 2025)**: "Hyperbolic Transformers" - HTC/HRC components with curvature-change support
- **Chen et al. (2024)**: "Hyperbolic Embeddings for Learning on Manifolds (HELM)" - HOPE positional encoding and Lorentzian residual connections
- **Klis et al. (2026)**: "Fast and Geometrically Grounded Lorentz Neural Networks" - `FGGLinear`, `FGGConv2D`, `FGGLorentzMLR`, `FGGMeanOnlyBatchNorm`; sinh/arcsinh cancellation for linear hyperbolic distance growth
- **Chen et al. (2026)**: "Proper Velocity Neural Networks" - `HypLinearPV`, `HypConv2DPV`, `HypRegressionPV`; unconstrained $\mathbb{R}^n$ model of hyperbolic geometry with exact Euclidean retraction
- **Chen, Schölkopf & Sebe (2026)**: "Hyperbolic Busemann Neural Networks" (arXiv:2602.18858) - `HypRegressionHyperboloidBusemann`, `HypRegressionPoincareBusemann` (BMLR heads) and `HypLinearHyperboloidBusemann`, `HypLinearPoincareBusemann` (BFC layers); closed-form point-to-horosphere Busemann function (`Hyperboloid.busemann`, `Poincare.busemann`) backing all four
- **Chen et al. (2025)**: "Hyperbolic VQ-VAE (HVQ-VAE)" - `HypVQEmbeddingPoincare`; Poincaré-ball codebook with geodesic nearest-neighbour selection and copy-gradient STE
- **Goswami et al. (2025)**: "HyperVQ" - `HypVQMLRPoincare`; vector quantization as Poincaré-MLR classification with Gumbel-Softmax straight-through selection
- **Bu et al. (2026)**: "GGBall: Graph Generative Model on Poincaré Ball" - hyperbolic-EMA codebook update and weighted gyromidpoint (`ema_update`, `poincare_weighted_midpoint`)
- **Shi et al. (2026)**: "Intrinsic Lorentz Neural Network" (ICLR 2026, arXiv:2602.23981) - point-to-hyperplane Lorentz FC layer with output-side score guard and intrinsic gyro-bias (`HypLinearHyperboloidPLFC`), log-radius concatenation (`Hyperboloid.log_radius_concat`), Lorentz convolution via LogCat + PLFC with origin padding (`HypConv2DHyperboloidILNN`), Lorentz gyroaddition (`Hyperboloid.addition`)

### Key Theoretical Connections

- **HL (Hyperbolic Layer)** from LResNet ≡ **HRC (Hyperbolic Regularization Component)** from Hypformer
- Both apply Euclidean operations to spatial components and reconstruct time using the Lorentz constraint
- `LorentzConv2D` is a specific instance of `hrc()` where `f_r` is a 2D convolution

See also:

- [Manifolds API](manifolds.md): Underlying geometric operations
- [Optimizers API](optimizers.md): Training with Riemannian optimization
- [Training Workflows](../user-guide/training-workflows.md): Complete training examples
