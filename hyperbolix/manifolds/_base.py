"""Shared base class and constraint-tolerance convention for manifold implementations."""

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array


def default_atol(dtype: DTypeLike) -> float:
    """Default absolute tolerance for a manifold constraint check, ``sqrt(finfo(dtype).eps)``.

    **This is the library-wide convention.** Every ``is_in_manifold`` / ``is_in_tangent_space``
    takes ``atol: float | None = None`` and resolves ``None`` through this function, using the
    dtype of the array being checked. Passing an explicit ``atol`` always wins — no manifold
    floors, clamps, or ignores it.

    ==========  =============  ===========
    dtype       ``eps``        ``atol``
    ==========  =============  ===========
    float32     1.19e-7        3.45e-4
    float64     2.22e-16       1.49e-8
    ==========  =============  ===========

    ``sqrt(eps)`` is the standard slack for a quantity accumulated by a dot product: the
    constraint residuals these checks look at (``⟨x, x⟩_L + 1/c`` on the hyperboloid,
    ``c‖x‖² - 1`` on the ball) are differences of same-magnitude sums, so they carry roughly
    half the mantissa. Half the mantissa is exactly ``sqrt(eps)``.

    Caveat: the tolerance is *absolute*, so it is calibrated for points within a few units of
    the origin. Far from it the constraint residual grows with ``‖x‖²`` and a genuinely
    on-manifold point stops passing. Measured on ``Hyperboloid.proj`` output at ``c = 1``:
    float64 holds to hyperbolic distance ~10 (residual 6e-11 at 7) and fails past ~11
    (residual 1.2e-7); float32 already fails around 7. Pass an explicit ``atol`` when checking
    points that far out — which is also where the library recommends float64 in the first place.

    Args:
        dtype: dtype of the array being validated (e.g. ``x.dtype``).

    Returns:
        Absolute tolerance as a Python float (static, so it is safe under ``jit``).
    """
    return float(jnp.finfo(dtype).eps) ** 0.5


class ManifoldBase:
    """Base class providing shared __init__, _cast, and static curvature.

    Args:
        dtype: Target JAX dtype for computations (default: jnp.float32)
        c: Curvature value (default: 1.0).
    """

    def __init__(
        self,
        dtype: jnp.dtype = jnp.float32,
        *,
        c: float = 1.0,
    ) -> None:
        self.dtype = dtype
        self._c_val = c

    @property
    def c(self) -> float:
        """Current curvature value."""
        return self._c_val

    def _cast(self, x: Array) -> Array:
        """Cast array to target dtype if it's a floating-point array."""
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact):
            return x.astype(self.dtype)
        return x
