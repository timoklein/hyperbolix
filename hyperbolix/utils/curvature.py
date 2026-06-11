"""Learnable curvature module for hyperbolic models.

Provides ``LearnableCurvature`` as the canonical way to add a trainable
curvature parameter. Instantiate one per distinct curvature in your model;
call the module at runtime to obtain the positive (optionally clamped) value.

Example::

    from hyperbolix import LearnableCurvature
    from hyperbolix.manifolds import Hyperboloid

    class Model(nnx.Module):
        def __init__(self, rngs):
            self.manifold = Hyperboloid(c=1.0)
            self.curvature = LearnableCurvature(init_c=1.0)
            self.fc = FGGLinear(33, 65, rngs=rngs)

        def __call__(self, x):
            return self.fc(x, c=self.curvature())

The raw parameter is Euclidean and updated by any ``nnx.Optimizer`` (no
Riemannian optimizer required).
"""

import math
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

Parameterization = Literal["softplus", "log"]


def _inv_softplus(x: float) -> float:
    """Compute inv_softplus(x) = log(exp(x) - 1) in Python floats."""
    if x <= 0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}")
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class LearnableCurvature(nnx.Module):
    """Reparameterized learnable curvature parameter.

    Stores a single Euclidean ``nnx.Param`` whose value is mapped to a
    positive curvature on every forward call. Two parameterizations are
    supported, with optional clamping applied to the recovered curvature
    (not the raw parameter) for hard stability guarantees in compiled
    training loops.

    Usage::

        self.curvature = LearnableCurvature(init_c=0.1)
        ...
        c = self.curvature()  # positive jax.Array

    Args:
        init_c: Initial curvature value. Must be positive. If clamp bounds
            are set, must also satisfy ``c_min <= init_c <= c_max``.
        parameterization: Reparameterization scheme.

            - ``"softplus"`` (default): ``c = softplus(raw)``. Gradient is
              bounded by ``sigmoid(raw) in (0, 1)``; smooth near zero;
              matches the van Spengler et al. 2023 Poincare ResNet convention.
            - ``"log"``: ``c = exp(raw)``. Scale-invariant gradient
              (``dc/draw = c``); preferred when ``c`` may span orders of
              magnitude or for long compiled RL training loops. Matches the
              MERU convention.

        c_min: Lower clamp applied to the recovered ``c``. Default ``0.1``.
            Pass ``None`` to disable.
        c_max: Upper clamp applied to the recovered ``c``. Default ``10.0``.
            Pass ``None`` to disable.
        param_dtype: Storage dtype of the raw parameter (default:
            ``jnp.float32``), pinned so it does not become float64 under
            global ``jax_enable_x64``.

    Sharing note: Do **not** assign the same ``LearnableCurvature`` instance
    to multiple fields if you want independent learnable curvatures —
    instantiate one per location. Sharing creates a shared-reference
    pattern in the NNX pytree that breaks ``nnx.scan`` / ``nnx.fori_loop``
    (same root cause as the pre-refactor manifold bug).
    """

    def __init__(
        self,
        init_c: float = 1.0,
        *,
        parameterization: Parameterization = "softplus",
        c_min: float | None = 0.1,
        c_max: float | None = 10.0,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if init_c <= 0:
            raise ValueError(f"LearnableCurvature requires init_c > 0, got {init_c}")
        if parameterization not in ("softplus", "log"):
            raise ValueError(f"parameterization must be 'softplus' or 'log', got {parameterization!r}")
        if c_min is not None and c_max is not None and c_min > c_max:
            raise ValueError(f"c_min ({c_min}) must be <= c_max ({c_max})")
        if c_min is not None and init_c < c_min:
            raise ValueError(f"init_c ({init_c}) must be >= c_min ({c_min})")
        if c_max is not None and init_c > c_max:
            raise ValueError(f"init_c ({init_c}) must be <= c_max ({c_max})")

        self._parameterization = parameterization
        self._c_min = c_min
        self._c_max = c_max

        if parameterization == "softplus":
            raw_init = _inv_softplus(init_c)
        else:  # "log"
            raw_init = math.log(init_c)

        self.raw = nnx.Param(jnp.array(raw_init, dtype=param_dtype))

    def __call__(self) -> jax.Array:
        if self._parameterization == "softplus":
            c = jax.nn.softplus(self.raw[...])
        else:  # "log"
            c = jnp.exp(self.raw[...])

        if self._c_min is not None or self._c_max is not None:
            c = jnp.clip(c, self._c_min, self._c_max)

        return c
