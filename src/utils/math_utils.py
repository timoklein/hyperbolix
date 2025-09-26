"""Math utils functions for hyperbolic operations with numerically stable limits."""

import torch


@torch.jit.script
def _get_tensor_eps(
    x: torch.Tensor,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")

@torch.jit.script
def smooth_clamp_min(x: torch.Tensor, min_value: float, smoothing_factor: float=50) -> torch.Tensor:
    """Smoothly clamp tensor values to a minimum."""
    eps = _get_tensor_eps(x)
    shift = min_value + eps
    x_clamped = shift + torch.nn.functional.softplus(x - shift, beta=smoothing_factor)
    return torch.where(x < shift, x_clamped, x)

@torch.jit.script
def smooth_clamp_max(x: torch.Tensor, max_value: float, smoothing_factor: float=50) -> torch.Tensor:
    """Smoothly clamp tensor values to a maximum."""
    eps = _get_tensor_eps(x)
    shift = max_value - eps
    x_clamped = shift - torch.nn.functional.softplus(shift - x, beta=smoothing_factor)
    return torch.where(x > shift, x_clamped, x)

@torch.jit.script
def smooth_clamp(x: torch.Tensor, min_value: float, max_value: float, smoothing_factor: float=50) -> torch.Tensor:
    """Smoothly clamp tensor values to a range [min_value, max_value]."""
    x = smooth_clamp_max(x, max_value, smoothing_factor=smoothing_factor)
    return smooth_clamp_min(x, min_value, smoothing_factor=smoothing_factor)

@torch.jit.script
def cosh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic cosine. Domain=(-inf, inf)."""
    # Safe limits as specified in SLEEF
    clamp = 88.0 if x.dtype == torch.float32 else 709.0
    x = smooth_clamp(x, -clamp, clamp)
    return torch.cosh(x)

@torch.jit.script
def sinh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic sine. Domain=(-inf, inf)."""
    # Safe limits as specified in SLEEF
    clamp = 88.0 if x.dtype == torch.float32 else 709.0
    x = smooth_clamp(x, -clamp, clamp)
    return torch.sinh(x)

@torch.jit.script
def tanh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent. Domain=(-inf, inf)."""
    return torch.tanh(x)

@torch.jit.script
def acosh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic cosine. Domain=[1, inf)."""
    x = x.clamp_min(1.0)
    return torch.acosh(x)

@torch.jit.script
def asinh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic sine. Domain=(-inf, inf)."""
    return torch.asinh(x)

@torch.jit.script
def atanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent. Domain=(-1, 1)."""
    eps = _get_tensor_eps(x)
    x = x.clamp(-1 + eps, 1 - eps)
    return torch.atanh(x)
