"""Runtime configuration for hyperbolix JAX backend."""

from typing import Union
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RuntimeConfig:
    """Immutable runtime configuration for hyperbolic math operations.

    This dataclass contains all numerical constants and tolerances needed
    for hyperbolic geometry calculations. It's designed to be:
    - Immutable (frozen=True) to prevent accidental modification
    - Explicit - passed as parameter to functions that need it
    - Testable - easy to create configs for different scenarios
    - Thread-safe - no shared mutable state
    """

    # Core dtype and precision
    dtype: jnp.dtype = jnp.float32

    # Numerical tolerances for manifold operations
    rtol: float = 1e-5  # Relative tolerance for comparisons
    atol: float = 1e-8  # Absolute tolerance for comparisons

    # Manifold-specific tolerances
    min_enorm: float = 1e-15  # Minimum norm for numerical stability
    max_enorm_eps: float = 5e-06  # Maximum norm epsilon (dtype-dependent)

    # Hyperbolic geometry constants
    clamp_factor: float = 15.0  # Factor for smooth clamping in asinh/atanh
    smoothing_factor: float = 50.0  # Smoothing parameter for soft clamping

    # Optimization tolerances
    grad_clip_norm: float = 1.0  # Gradient clipping norm

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dtype not in (jnp.float32, jnp.float64):
            raise ValueError(f"Unsupported dtype: {self.dtype}. Use float32 or float64.")

        # Auto-adjust max_enorm_eps based on dtype if using default
        if self.max_enorm_eps == 5e-06 and self.dtype == jnp.float64:
            # Use struct.replace since this is a Flax struct
            object.__setattr__(self, 'max_enorm_eps', 1e-08)

    @property
    def eps(self) -> float:
        """Machine epsilon for the current dtype."""
        return jnp.finfo(self.dtype).eps

    @property
    def tiny(self) -> float:
        """Smallest normal number for the current dtype."""
        return jnp.finfo(self.dtype).tiny

    @property
    def max_clamp_value(self) -> float:
        """Maximum value for smooth clamping operations."""
        import math
        return self.clamp_factor * float(math.log(2 / self.eps))

    def with_dtype(self, dtype: Union[str, jnp.dtype]) -> 'RuntimeConfig':
        """Create a new config with different dtype, adjusting tolerances."""
        if isinstance(dtype, str):
            dtype = getattr(jnp, dtype)

        # Adjust max_enorm_eps based on dtype
        new_max_enorm_eps = 1e-08 if dtype == jnp.float64 else 5e-06

        return self.replace(
            dtype=dtype,
            max_enorm_eps=new_max_enorm_eps,
        )

    def with_tolerances(self, rtol: float = None, atol: float = None) -> 'RuntimeConfig':
        """Create a new config with different tolerances."""
        updates = {}
        if rtol is not None:
            updates['rtol'] = rtol
        if atol is not None:
            updates['atol'] = atol
        return self.replace(**updates)

    def with_precision(self, precision: str) -> 'RuntimeConfig':
        """Create a new config with 'high' or 'low' precision preset."""
        if precision == "high":
            return self.with_dtype(jnp.float64).with_tolerances(rtol=1e-10, atol=1e-12)
        elif precision == "low":
            return self.with_dtype(jnp.float32).with_tolerances(rtol=1e-4, atol=1e-6)
        else:
            raise ValueError(f"Unknown precision preset: {precision}")


# Common pre-configured instances
DEFAULT_CONFIG = RuntimeConfig()
FLOAT64_CONFIG = RuntimeConfig().with_dtype(jnp.float64)
HIGH_PRECISION_CONFIG = RuntimeConfig().with_precision("high")
LOW_PRECISION_CONFIG = RuntimeConfig().with_precision("low")


def get_dtype_config(dtype: Union[str, jnp.dtype]) -> RuntimeConfig:
    """Get a default config for the specified dtype."""
    if isinstance(dtype, str):
        dtype = getattr(jnp, dtype)

    if dtype == jnp.float64:
        return FLOAT64_CONFIG
    else:
        return DEFAULT_CONFIG


def create_config(
    dtype: Union[str, jnp.dtype] = jnp.float32,
    precision: str = None,
    **kwargs
) -> RuntimeConfig:
    """Create a runtime config with optional overrides.

    Args:
        dtype: Data type (float32 or float64)
        precision: Preset precision level ('high', 'low', or None)
        **kwargs: Additional config parameters to override

    Returns:
        RuntimeConfig instance
    """
    config = get_dtype_config(dtype)

    if precision is not None:
        config = config.with_precision(precision)

    if kwargs:
        # Use struct.replace for overrides
        config = config.replace(**kwargs)

    return config