"""Lorentzian residual (skip) connection module for hyperboloid manifolds.

Wraps the functional
:func:`~hyperbolix.nn_layers.hyperboloid_core.lorentz_residual` (the LResNet
weighted Lorentzian midpoint) in an ``nnx.Module`` that owns its ``y``-weight
``w_y`` as a *safely-constrained* parameter, plus the optional Klein-geodesic
output scaling of
:func:`~hyperbolix.nn_layers.hyperboloid_core.lorentz_scale` (LResNet Eq. 10).

The constraint is the point of the module: ``lorentz_residual`` warns that
``w_y`` must stay non-negative -- its ``abs()`` normalizer silently turns a
``w_y < 0`` geometry violation into a valid-looking but wrong output instead of
raising. Exposing ``w_y`` as a raw ``nnx.Param`` is therefore unsafe; here it is
reparameterized through ``softplus`` so a *trainable* weight can never leave the
upper hyperboloid sheet. ``gamma`` is likewise softplus-constrained to keep the
"slide toward / away from the origin" semantics (``gamma > 0``).

References
----------
He, Neil, Menglin Yang, and Rex Ying. "Lorentzian residual neural networks."
Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2025.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.utils.curvature import _inv_softplus

from .hyperboloid_core import lorentz_residual, lorentz_scale


class LorentzResidual(nnx.Module):
    """LResNet residual (skip) connection with a safely-constrained weight.

    Computes the weighted Lorentzian midpoint of a skip branch ``x`` and a
    residual branch ``y`` (the latter weighted by ``w_y``), optionally rescaling
    the result along the geodesic ray from the origin (LResNet Eq. 10)::

        out = lorentz_residual(x, y, w_y, c)            # weighted Lorentzian midpoint
        if scale:
            out = lorentz_scale(out, gamma, c)          # optional Eq. 10 norm control

    Both ``x`` and ``y`` are points on the hyperboloid with curvature ``c``; the
    caller produces ``y`` (e.g. via a conv / linear layer). This mirrors how
    :class:`~hyperbolix.nn_layers.HypformerPositionalEncoding` composes a
    transform with ``lorentz_residual`` -- except that layer keeps its weight
    *fixed* (a trainable position weight is unsafe there), whereas this module
    makes a *trainable* ``w_y`` safe via the softplus reparameterization.

    Parameters
    ----------
    init_w_y : float, optional
        Initial value of the residual-branch weight ``w_y`` (default: 1.0). Must
        be ``>= 0`` (``> 0`` when ``learnable_weight=True``, since ``softplus``
        cannot represent exactly 0).
    learnable_weight : bool, optional
        If ``True`` (default), ``w_y`` is a softplus-constrained ``nnx.Param``;
        if ``False`` it is a fixed Python float.
    scale : bool, optional
        If ``True``, apply the Eq. 10 Klein-geodesic output scaling after the
        residual (default: ``False`` -- the paper frames it as optional, for
        low-curvature manifolds).
    init_gamma : float, optional
        Initial value of the scaling constant ``gamma`` (default: 2.0, the
        LResNet CIFAR reference value). Must be ``> 0``. Ignored when
        ``scale=False``.
    learnable_scale : bool, optional
        If ``True``, ``gamma`` is a softplus-constrained ``nnx.Param``; if
        ``False`` (default) it is a fixed Python float. Ignored when
        ``scale=False``.
    param_dtype : DTypeLike, optional
        Storage dtype of any learnable raw parameter (default: ``jnp.float32``),
        pinned so it does not become float64 under global ``jax_enable_x64``.
    eps : float, optional
        Numerical stability floor passed to ``lorentz_residual`` / ``lorentz_scale``
        (default: 1e-7).

    Attributes
    ----------
    w_y_raw : nnx.Param or None
        Raw (pre-softplus) weight when ``learnable_weight=True``; otherwise
        ``None`` (the fixed value is stored statically).
    gamma_raw : nnx.Param or None
        Raw (pre-softplus) scaling constant when ``scale and learnable_scale``;
        otherwise ``None``.
    use_scale : bool
        Whether the Eq. 10 scaling is applied.

    References
    ----------
    He, Neil, Menglin Yang, and Rex Ying. "Lorentzian residual neural networks."
    Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2025.
    Residual: :func:`~hyperbolix.nn_layers.hyperboloid_core.lorentz_residual`;
    Eq. 10 scaling: :func:`~hyperbolix.nn_layers.hyperboloid_core.lorentz_scale`.
    """

    def __init__(
        self,
        *,
        init_w_y: float = 1.0,
        learnable_weight: bool = True,
        scale: bool = False,
        init_gamma: float = 2.0,
        learnable_scale: bool = False,
        param_dtype: DTypeLike = jnp.float32,
        eps: float = 1e-7,
    ):
        if init_w_y < 0:
            raise ValueError(f"LorentzResidual requires init_w_y >= 0, got {init_w_y}")
        if learnable_weight and init_w_y == 0:
            raise ValueError(
                "LorentzResidual with learnable_weight=True requires init_w_y > 0 (softplus cannot represent exactly 0)."
            )
        if init_gamma <= 0:
            raise ValueError(f"LorentzResidual requires init_gamma > 0, got {init_gamma}")

        self.use_scale = scale
        self.eps = eps

        # y-weight: softplus-constrained nnx.Param when learnable, plain float otherwise.
        if learnable_weight:
            self.w_y_raw = nnx.Param(jnp.array(_inv_softplus(init_w_y), dtype=param_dtype))
            self._w_y = None
        else:
            self.w_y_raw = None
            self._w_y = init_w_y

        # Eq. 10 scaling constant gamma (only consulted when use_scale is True).
        if scale and learnable_scale:
            self.gamma_raw = nnx.Param(jnp.array(_inv_softplus(init_gamma), dtype=param_dtype))
            self._gamma = None
        else:
            self.gamma_raw = None
            self._gamma = init_gamma

    def __call__(
        self,
        x: Float[Array, "... dim_plus_1"],
        y: Float[Array, "... dim_plus_1"],
        c: float = 1.0,
    ) -> Float[Array, "... dim_plus_1"]:
        """Apply the (optionally scaled) Lorentzian residual connection.

        Parameters
        ----------
        x : Array, shape (..., d+1)
            Skip / main branch on the hyperboloid with curvature ``c``.
        y : Array, shape (..., d+1)
            Residual branch on the hyperboloid with curvature ``c`` (weighted by
            ``w_y``).
        c : float, optional
            Curvature parameter (default: 1.0).

        Returns
        -------
        Array, shape (..., d+1)
            Points on the hyperboloid with curvature ``c``.
        """
        # Recover w_y > 0 via softplus when learnable; cast to the input dtype so a
        # float32-pinned param does not silently downcast a float64 forward pass.
        w_y = jax.nn.softplus(self.w_y_raw[...]) if self.w_y_raw is not None else self._w_y
        out = lorentz_residual(x, y, w_y=jnp.asarray(w_y, dtype=x.dtype), c=c, eps=self.eps)

        if self.use_scale:
            gamma = jax.nn.softplus(self.gamma_raw[...]) if self.gamma_raw is not None else self._gamma
            out = lorentz_scale(out, gamma=jnp.asarray(gamma, dtype=x.dtype), c=c, eps=self.eps)

        return out
