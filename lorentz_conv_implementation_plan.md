# Implementation Plan: Lorentz Convolution Layer

## Overview

Implement a new `LorentzConv2D` layer following the "Fully Hyperbolic CNNs" paper (Eq. 7):

```
out = LorentzBoost(DistanceRescaling(RotationConvolution(x)))
```

This is an **alternative** to the existing `HypConv2DHyperboloid` (which uses HCat + HypLinear).

## Design Decisions

| Decision | Choice |
|----------|--------|
| Transform type | Norm-preserving rescaling: `z = W^T x · ‖x‖ / ‖W^T x‖` |
| Boost velocity scope | Global shared: single `v ∈ R^(out_channels-1)` per layer |
| Algorithm 3 else case | Use standard conv weights without Transform |
| DistanceRescaling params | Fixed constants from paper (m=D_max, s=1.0) |
| Velocity constraint | Projection: `v / max(‖v‖, 1-ε)` |

## File Structure

```
hyperbolix/nn_layers/
├── lorentz_conv.py         # NEW: LorentzConv2D, LorentzConv3D
├── lorentz_transforms.py   # NEW: lorentz_boost, distance_rescale, rotation_conv helpers
└── __init__.py             # UPDATE: Add exports
```

---

## Component 1: lorentz_transforms.py

### 1.1 lorentz_boost()

```python
def lorentz_boost(
    x: Float[Array, "... dim_plus_1"],
    v_raw: Float[Array, "dim"],
    c: float,
) -> Float[Array, "... dim_plus_1"]:
    """Apply Lorentz boost to hyperboloid points.

    B = [[γ,       -γv^T              ],
         [-γv,     I + (γ²/(1+γ))vv^T ]]

    where γ = 1/√(1 - ‖v‖²), ‖v‖ < 1.
    """
```

**Implementation:**
1. Project velocity: `v = v_raw / max(‖v_raw‖, 1-ε)` with ε=1e-6
2. Compute γ = 1/√(1 - ‖v‖²)
3. Split input: `x_t = x[..., 0:1]`, `x_s = x[..., 1:]`
4. Compute `v·x_s = sum(v * x_s, axis=-1, keepdims=True)`
5. New time: `new_t = γ*x_t - γ*(v·x_s)`
6. New spatial: `new_s = -γ*v*x_t + x_s + (γ²/(1+γ))*v*(v·x_s)`
7. Return `concat([new_t, new_s], axis=-1)`

### 1.2 distance_rescale()

```python
def distance_rescale(
    x: Float[Array, "... dim_plus_1"],
    c: float,
    x_t_max: float = 2000.0,
    slope: float = 1.0,
) -> Float[Array, "... dim_plus_1"]:
    """Apply distance rescaling (Eq. 2-3) to bound hyperbolic distances.

    D_rescaled = m · tanh(D · atanh(0.99) / (s·m))
    x_s_rescaled = x_s · sinh(√c · D_rescaled) / sinh(√c · D)
    """
```

**Implementation:**
1. Compute distance from origin: `D = acosh(√c · x_t) / √c`
2. Compute max distance: `D_max = acosh(√c · x_t_max) / √c`
3. Apply Eq. 2: `D_rescaled = D_max · tanh(D · atanh(0.99) / (slope · D_max))`
4. Rescale spatial (Eq. 3): `scale = sinh(√c · D_rescaled) / sinh(√c · D)`
5. `x_s_rescaled = x_s · scale`
6. Reconstruct time: `x_t_rescaled = √(‖x_s_rescaled‖² + 1/c)`

### 1.3 rotation_conv_2d()

```python
def rotation_conv_2d(
    x: Float[Array, "batch H W in_channels"],
    weight: Float[Array, "out_channels in_channels kh kw"],
    c: float,
    stride: tuple[int, int],
    padding: str,
) -> Float[Array, "batch out_H out_W out_channels"]:
    """Norm-preserving rotation convolution.

    For each spatial location:
    1. Extract receptive field patch
    2. Apply conv: z = W @ patch
    3. Rescale: z_out = z · ‖patch‖ / ‖z‖
    4. Reconstruct time from constraint
    """
```

**Implementation approach:**
1. Separate time/spatial: `x_t = x[..., 0:1]`, `x_s = x[..., 1:]`
2. Compute spatial norms: `x_s_norm = ‖x_s‖` per pixel
3. Apply standard conv to spatial components only (weight acts on spatial dims)
4. Pool input norms over receptive field (avg pool same kernel/stride)
5. Compute output norms
6. Rescale: `out_s = conv_out · (pooled_input_norm / out_norm)`
7. Reconstruct time: `out_t = √(‖out_s‖² + 1/c)`

**Weight adaptation (Algorithm 3):**
- Only if `K_h * K_w * (in_channels-1) ≤ (out_channels-1)`
- This condition determines when the norm-preserving property is mathematically valid
- When false: use standard convolution weights

---

## Component 2: lorentz_conv.py

### LorentzConv2D Class

```python
class LorentzConv2D(nnx.Module):
    """Lorentz Convolution implementing Eq. 7:
    out = LorentzBoost(DistanceRescaling(RotationConvolution(x)))
    """

    def __init__(
        self,
        manifold_module: Any,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        rngs: nnx.Rngs,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        input_space: str = "manifold",
        use_distance_rescaling: bool = True,
        use_boost: bool = True,
    ):
        # Validate inputs
        # Store config

        # Initialize conv weights (spatial only: in_channels-1 -> out_channels-1)
        spatial_in = in_channels - 1
        spatial_out = out_channels - 1
        self.weight = nnx.Param(...)  # shape: (spatial_out, spatial_in, kh, kw)

        # Initialize boost velocity (if using)
        if use_boost:
            self.boost_velocity = nnx.Param(
                jax.random.normal(rngs.params(), (spatial_out,)) * 0.01
            )

    def __call__(self, x, c=1.0):
        # 0. Map to manifold if tangent input
        if self.input_space == "tangent":
            x = vmap_expmap_0(x, c)

        # 1. Rotation Convolution (norm-preserving)
        y = rotation_conv_2d(x, self.weight.value, c, self.stride, self.padding)

        # 2. Distance Rescaling
        if self.use_distance_rescaling:
            y = distance_rescale(y, c)

        # 3. Lorentz Boost
        if self.use_boost:
            y = vmap_lorentz_boost(y, self.boost_velocity.value, c)

        return y
```

### LorentzConv3D Class

Same structure, but:
- kernel_size: `tuple[int, int, int]`
- Input shape: `(batch, D, H, W, channels)`
- Uses `rotation_conv_3d` helper

---

## Component 3: Update __init__.py

Add exports:
```python
from .lorentz_conv import LorentzConv2D, LorentzConv3D
from .lorentz_transforms import lorentz_boost, distance_rescale
```

---

## Key Mathematical Formulas

### Hyperboloid Constraint
```
-x_t² + ‖x_s‖² = -1/c,  x_t > 0
```

### Lorentz Boost (preserves manifold)
```
γ = 1/√(1 - ‖v‖²)
new_t = γ·x_t - γ·(v·x_s)
new_s = -γ·v·x_t + x_s + (γ²/(1+γ))·v·(v·x_s)
```

### Distance Rescaling (Eq. 2)
```
D_rescaled = m · tanh(D · atanh(0.99) / (s·m))
where m = D_max, s = 1.0
```

### Norm-Preserving Transform
```
z = W·x · ‖x‖ / ‖W·x‖
```

---

## Implementation Sequence

1. **lorentz_transforms.py**: Core transform functions
   - `lorentz_boost()` with velocity projection
   - `distance_rescale()` with Eq. 2-3
   - `rotation_conv_2d()` with norm-preserving rescaling

2. **lorentz_conv.py**: Layer classes
   - `LorentzConv2D` wiring transforms together
   - `LorentzConv3D` for volumetric data

3. **__init__.py**: Add exports

4. **Tests**: Verify manifold preservation, gradients, JIT compatibility

---

## Critical Files to Reference

| File | Purpose |
|------|---------|
| `hyperbolix/manifolds/hyperboloid.py` | Reuse: `proj`, `expmap_0`, `acosh`, `sinh` |
| `hyperbolix/nn_layers/hyperboloid_conv.py` | Reference: JIT patterns, padding handling |
| `hyperbolix/utils/math_utils.py` | Reuse: `smooth_clamp_min`, numerical utilities |

---

## Numerical Stability Considerations

1. **Velocity norm**: Clamp to `1-ε` to avoid division by zero in γ
2. **Distance rescaling**: Use `smooth_clamp_min` for sinh/acosh inputs
3. **Norm division**: Add ε to denominator in `‖x‖ / ‖W·x‖`
4. **Time reconstruction**: Ensure argument to sqrt is positive
