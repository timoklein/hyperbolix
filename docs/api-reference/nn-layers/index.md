# Neural Network Layers API

Hyperbolic neural network layers built with Flax NNX — 20+ layer classes and the
activation/primitive functions that support them, across the Poincaré, Hyperboloid,
and Proper Velocity models. All layers follow Flax NNX conventions and store a
manifold-module reference.

!!! tip "Looking for *which* layer to use?"
    This API reference documents signatures and call semantics. For **layer
    selection, channel conventions, initialization scales, and composition patterns**,
    see the task-oriented **[NN Layers guide](../../user-guide/nn-layers.md)**.

## Pages

| Page | Contents |
|---|---|
| [Linear](linear.md) | Poincaré / Hyperboloid / PV fully-connected layers, including `HTCLinear`, `FGGLinear`, and Busemann FC |
| [Convolutional](convolutional.md) | HCat, intrinsic-Lorentz (ILNN), FGG, Poincaré, and PV 2D convolutions + `hyp_avg_pool2d` |
| [Normalization](normalization.md) | Poincaré BatchNorm, gyro batch/RMS norm (Hyperboloid & PV), HRC norms + dropout, FGG mean-only BN, Euclidean input scaling |
| [Attention & Transformer](attention.md) | Linear O(N), softmax O(N²), and full Lorentzian O(N²) attention with causal masking |
| [Regression & MLR](regression.md) | Point-to-hyperplane and Busemann (point-to-horosphere) classification heads |
| [Activations](activations.md) | Curvature-preserving `hyp_*`, Poincaré, and curvature-changing `hrc_*` |
| [Positional Encoding](positional-encoding.md) | HOPE rotary PE, Hypformer learnable PE, Lorentzian residual |
| [Vector Quantization](vector-quantization.md) | Poincaré VQ-VAE bottlenecks (EMA codebook, MLR-implicit codebook) |
| [Primitives & Helpers](primitives.md) | HTC/HRC components, point assembly, midpoints, residuals, Fréchet variance |

## References

The neural network layers implement methods from:

- **Ganea et al. (2018)**: "Hyperbolic Neural Networks" — Poincaré linear layers and activations
- **Shimizu et al. (2020)**: "Hyperbolic Neural Networks++" — enhanced Poincaré operations and the linearized-kernel conv formulation (`HypLinearPoincarePP`; basis of `HypLinearHyperboloidPLFC` and `HypConv2DHyperboloidILNN`)
- **Bdeir et al. (2023)**: "Fully Hyperbolic Convolutional Neural Networks for Computer Vision" — HCat-based convolutions (`HypConv2DHyperboloid`)
- **Chen et al. (2022)**: "Fully Hyperbolic Neural Networks" — FHCNN linear layers
- **LResNet (2023)**: "Lorentzian ResNet" — HRC-based convolutions (`LorentzConv2D`)
- **Hypformer (Yang et al. 2025)**: "Hyperbolic Transformers" — HTC/HRC components with curvature-change support
- **Chen et al. (2024)**: "Hyperbolic Embeddings for Learning on Manifolds (HELM)" — HOPE positional encoding and Lorentzian residual connections
- **Klis et al. (2026)**: "Fast and Geometrically Grounded Lorentz Neural Networks" — `FGGLinear`, `FGGConv2D`, `FGGLorentzMLR`, `FGGMeanOnlyBatchNorm`; sinh/arcsinh cancellation for linear hyperbolic distance growth
- **Chen et al. (2026)**: "Proper Velocity Neural Networks" — `HypLinearPV`, `HypConv2DPV`, `HypRegressionPV`; unconstrained $\mathbb{R}^n$ model with exact Euclidean retraction
- **Chen, Schölkopf & Sebe (2026)**: "Hyperbolic Busemann Neural Networks" (arXiv:2602.18858) — Busemann MLR heads and BFC layers; closed-form point-to-horosphere Busemann function (`Hyperboloid.busemann`, `Poincare.busemann`)
- **Chen et al. (2025)**: "Hyperbolic VQ-VAE (HVQ-VAE)" — `HypVQEmbeddingPoincare`; Poincaré-ball codebook with geodesic nearest-neighbour selection and copy-gradient STE
- **Goswami et al. (2025)**: "HyperVQ" — `HypVQMLRPoincare`; vector quantization as Poincaré-MLR classification with Gumbel-Softmax straight-through selection
- **Bu et al. (2026)**: "GGBall: Graph Generative Model on Poincaré Ball" — hyperbolic-EMA codebook update and weighted gyromidpoint (`ema_update`, `poincare_weighted_midpoint`)
- **Shi et al. (2026)**: "Intrinsic Lorentz Neural Network" (ICLR 2026, arXiv:2602.23981) — point-to-hyperplane Lorentz FC (`HypLinearHyperboloidPLFC`), log-radius concatenation, Lorentz convolution via LogCat + PLFC (`HypConv2DHyperboloidILNN`), Lorentz gyroaddition

### Key theoretical connections

- **HL (Hyperbolic Layer)** from LResNet ≡ **HRC (Hyperbolic Regularization Component)** from Hypformer — both apply Euclidean operations to spatial components and reconstruct time via the Lorentz constraint. `LorentzConv2D` is the instance of `hrc()` where `f_r` is a 2D convolution.

## See also

- [NN Layers guide](../../user-guide/nn-layers.md) — choosing layers, composition patterns, pitfalls
- [Manifolds API](../manifolds.md) — underlying geometric operations
- [Optimizers API](../optimizers.md) — training with Riemannian optimization
- [Training Workflows](../../user-guide/training-workflows.md) — complete training examples
