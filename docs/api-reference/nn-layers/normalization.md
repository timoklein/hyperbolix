# Normalization

All hyperbolic normalization layers in one place — previously these were scattered
across the convolution and Hypformer sections. For **which** normalizer to use
between which layers, see the
[NN Layers guide](../../user-guide/nn-layers.md#attention-normalization-positional-encoding).

!!! note "Channel convention"
    HRC and gyro norms take the **spatial** dimension `num_features = d` (not ambient).
    `PoincareBatchNorm2D` also takes spatial `d` (Poincaré has no time component).

## Poincaré Batch Normalization

`PoincareBatchNorm2D` operates in tangent space (matching conv-layer I/O), mapping to
the manifold internally for the geometric operations (Einstein midpoint, Fréchet
variance, parallel transport, variance rescaling). Use between Poincaré conv layers
following the ResNet pattern `conv → bn → relu → conv → bn → skip`. The midpoint /
variance helpers it builds on live in [Primitives](primitives.md).

::: hyperbolix.nn_layers.PoincareBatchNorm2D
    options:
      heading_level: 3

## Gyro Normalization (Hyperboloid, Proper Velocity & Poincaré)

Intrinsic normalization that operates **directly on manifold points via gyrovector
operations** (`addition`, `scalar_mul`) — no tangent-space round trip for the affine
part. Batch normalization is provided for the Lorentz/Hyperboloid and Proper Velocity
models (the Poincaré ball's batch normalizer is the tangent-space `PoincareBatchNorm2D`
above); the per-sample radial RMSNorm covers all three.

- **Gyrogroup Batch Normalization** (`HyperboloidGyroBatchNorm`,
  `ProperVelocityGyroBatchNorm`) — a port of GyroBN (Chen et al., ICLR 2024 / 2025).
  *Centers* by gyro-translating with the inverse batch mean, *scales* by the inverse
  Fréchet standard deviation, *biases* with a learned manifold point, and keeps
  running statistics for evaluation (`use_running_average`). The Hyperboloid batch
  mean is the closed-form Lorentz centroid; the PV mean is the closed-form
  log-Euclidean mean. Use for faithful hyperbolic ResNets.
- **Gyro radial RMSNorm** (`HyperboloidGyroRMSNorm`, `ProperVelocityGyroRMSNorm`,
  `PoincareGyroRMSNorm`) — a *per-sample*, batch-independent normalizer (no running
  statistics, identical in train and eval, valid at batch size 1 — the properties RL
  workloads want). Each point's geodesic radius is rescaled to a learned target `gamma`
  via a single gyro scalar-multiplication, with optional gyro-bias (`use_bias`). The
  manifold analog of RMSNorm: normalizes magnitude (hierarchy *depth*) while preserving
  direction. Möbius `scalar_mul` (Poincaré) and Lorentz/PV `scalar_mul` scale geodesic
  radius identically, so the same layer body serves all three models.

::: hyperbolix.nn_layers.HyperboloidGyroBatchNorm
    options:
      heading_level: 3

::: hyperbolix.nn_layers.ProperVelocityGyroBatchNorm
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HyperboloidGyroRMSNorm
    options:
      heading_level: 3

::: hyperbolix.nn_layers.ProperVelocityGyroRMSNorm
    options:
      heading_level: 3

::: hyperbolix.nn_layers.PoincareGyroRMSNorm
    options:
      heading_level: 3

## HRC Normalization & Dropout (Hyperboloid)

HRC-wrapped normalizers and dropout for the Hyperboloid (Hypformer). They apply a
Euclidean operation to the space components and reconstruct the time component, with
curvature-change support. `FGGMeanOnlyBatchNorm` pairs with
`FGGLinear(use_weight_norm=True)`.

::: hyperbolix.nn_layers.HRCBatchNorm
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HRCLayerNorm
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HRCRMSNorm
    options:
      heading_level: 3

::: hyperbolix.nn_layers.HRCDropout
    options:
      heading_level: 3

::: hyperbolix.nn_layers.FGGMeanOnlyBatchNorm
    options:
      heading_level: 3

## Euclidean input scaling (hybrid networks)

`HyperPPFeatureScaling` is **not** an on-manifold normalizer — it runs entirely in
Euclidean space, applied after the last Euclidean layer and *before* `expmap_0` to
either manifold. It is RMSNorm-based (RMSNorm + Lipschitz activation + dimension
scaling + optional learned rescaling), bounding the output norm via
`rho_max = atanh(alpha) / sqrt(c)`. See the
[boundary-pattern recipe](../../user-guide/nn-layers.md#the-euclidean-hyperbolic-boundary).

::: hyperbolix.nn_layers.HyperPPFeatureScaling
    options:
      heading_level: 3

## Example

```python
import jax
from hyperbolix.manifolds import Hyperboloid, ProperVelocity
from hyperbolix.nn_layers import HyperboloidGyroBatchNorm, HyperboloidGyroRMSNorm, ProperVelocityGyroRMSNorm

c = 1.0
hyp = Hyperboloid()
bn = HyperboloidGyroBatchNorm(hyp, num_features=16)   # num_features = spatial D
rms = HyperboloidGyroRMSNorm(hyp, num_features=16)
spatial = jax.random.normal(jax.random.PRNGKey(0), (8, 16)) * 0.2
x_amb = jax.vmap(hyp.expmap_0, in_axes=(0, None))(jax.vmap(hyp.embed_spatial_0)(spatial), c)  # (8, 17)
h = bn(x_amb, c=c)                            # training (running stats updated)
h = bn(x_amb, c=c, use_running_average=True)  # evaluation (frozen stats)
h = rms(x_amb, c=c)                           # per-sample, batch-free

pv_rms = ProperVelocityGyroRMSNorm(ProperVelocity(), num_features=16, use_bias=True)
h_pv = pv_rms(jax.random.normal(jax.random.PRNGKey(1), (8, 16)) * 0.2, c=c)  # (8, 16)
```
