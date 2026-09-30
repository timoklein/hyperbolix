"""Lorentzian residual (skip) connection module for hyperboloid manifolds.

Wraps the functional
:func:`~hyperbolix.nn_layers.hyperboloid_core.lorentz_residual` (the LResNet
weighted Lorentzian midpoint) in an ``nnx.Module`` that owns its ``y``-weight
``w_y`` as a *safely-constrained* parameter, plus the optional Klein-geodesic
output scaling of
:func:`~hyperbolix.nn_layers.hyperboloid_core.lorentz_scale` (LResNet Eq. 10).

The constraint is the point of the module: ``lorentz_residual`` warns that
``w_y`` must stay non-negative -- a ``w_y < 0`` that makes the combination
spacelike gives NaN, and a ``w_y < -1`` that makes it past-directed gives a
valid-looking but wrong point with no error. Exposing ``w_y`` as a raw
``nnx.Param`` is therefore unsafe; here it is
reparameterized through ``softplus`` so a *trainable* weight can never leave the
upper hyperboloid sheet. ``gamma`` is likewise softplus-constrained to keep the
"slide toward / away from the origin" semantics (``gamma > 0``).

For reproducing HELM (He et al. 2025), which uses LResNet with a raw, unconstrained
``w_y`` and an ``exp``-parameterized learnable scale, the module also offers
``weight_parameterization="identity"`` and ``scale_parameterization="exp"``, plus a
call-time ``weight=`` override for per-token gate weights. These are opt-in; the
defaults are the softplus parameterizations above.

References
----------
He, Neil, Menglin Yang, and Rex Ying. "Lorentzian residual neural networks."
Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2025.
"""

import math
from typing import Literal

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

    HELM configuration (He et al. 2025, ``helm/modules/helm_mice.py`` and
    ``mice.py``; HELM's curvature ``k`` is ``1/c`` here). HELM's LResNet keeps
    ``w_y`` as a raw, unconstrained ``nn.Parameter`` and learns the scale as
    ``exp(s_raw)``::

        # after each attention / FFN block: learnable raw w_y, fixed scale sqrt(dim)
        block_res = LorentzResidual(
            weight_parameterization="identity", scale=True, init_gamma=math.sqrt(dim)
        )
        x = block_res(x, attn_out, c=c)

        # MoE weighted sum of routed experts: per-token gate weight, fixed scale 2.0
        weighted_sum = LorentzResidual(learnable_weight=False, scale=True, init_gamma=2.0)
        y = weighted_sum(y, expert_out, c=c, weight=gate_N1)   # gate_N1: shape (N, 1) or (N,)

        # shared + routed experts combine: learnable raw w_y, learnable exp scale init 2.0
        add_experts = LorentzResidual(
            weight_parameterization="identity",
            scale=True,
            init_gamma=2.0,
            learnable_scale=True,
            scale_parameterization="exp",
        )
        out = add_experts(shared_out, y, c=c)

    With ``weight_parameterization="identity"`` a trainable ``w_y`` can go
    negative. Where that makes the combination spacelike,
    :func:`~hyperbolix.nn_layers.hyperboloid_core.lorentz_residual` returns NaN,
    where the HELM reference's ``.abs()`` returns a valid but wrong point; a
    past-directed combination (``w_y < -1``) still gives a valid but wrong point
    with no error (see its Notes). A negative ``w_y`` is outside the domain of
    LResNet's Lemma 4.1, which bounds the normalizer only for
    ``(w_x, w_y) in R+ x R+ \\ {(0, 0)}``. Keep the default softplus
    parameterization unless you are reproducing HELM; there, call
    :func:`project_residual_weights` (or :meth:`project_w_y`) after each
    ``optimizer.update`` to project ``w_y`` back onto ``w_y >= 0``.

    Parameters
    ----------
    init_w_y : float, optional
        Initial value of the residual-branch weight ``w_y`` (default: 1.0). Must
        be ``>= 0`` (``> 0`` when ``learnable_weight=True`` with the softplus
        parameterization, since ``softplus`` cannot represent exactly 0).
    learnable_weight : bool, optional
        If ``True`` (default), ``w_y`` is a learnable ``nnx.Param`` (see
        ``weight_parameterization``); if ``False`` it is a fixed Python float.
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
    weight_parameterization : {"softplus", "identity"}, optional
        How a learnable ``w_y`` is recovered from its raw parameter:
        ``"softplus"`` (default) gives ``w_y = softplus(raw) > 0``;
        ``"identity"`` gives ``w_y = raw``, unconstrained, initialized at
        ``init_w_y`` (HELM's LResNet; project it with :meth:`project_w_y`).
        Ignored when ``learnable_weight=False``.
    scale_parameterization : {"softplus", "exp"}, optional
        How a learnable ``gamma`` is recovered from its raw parameter:
        ``"softplus"`` (default) gives ``gamma = softplus(raw)``; ``"exp"``
        gives ``gamma = exp(raw)`` with ``raw`` initialized at
        ``log(init_gamma)`` (HELM's ``learn_scale=True``). Ignored unless
        ``scale`` and ``learnable_scale``.

    Attributes
    ----------
    w_y_raw : nnx.Param or None
        Raw weight (pre-softplus, or ``w_y`` itself under ``"identity"``) when
        ``learnable_weight=True``; otherwise ``None`` (the fixed value is stored
        statically).
    gamma_raw : nnx.Param or None
        Raw scaling constant (pre-softplus, or ``log gamma`` under ``"exp"``)
        when ``scale and learnable_scale``; otherwise ``None``.
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
        weight_parameterization: Literal["softplus", "identity"] = "softplus",
        scale_parameterization: Literal["softplus", "exp"] = "softplus",
    ):
        if weight_parameterization not in ("softplus", "identity"):
            raise ValueError(
                f"LorentzResidual: weight_parameterization must be 'softplus' or 'identity', got {weight_parameterization!r}"
            )
        if scale_parameterization not in ("softplus", "exp"):
            raise ValueError(
                f"LorentzResidual: scale_parameterization must be 'softplus' or 'exp', got {scale_parameterization!r}"
            )
        if init_w_y < 0:
            raise ValueError(f"LorentzResidual requires init_w_y >= 0, got {init_w_y}")
        if learnable_weight and weight_parameterization == "softplus" and init_w_y == 0:
            raise ValueError(
                "LorentzResidual with learnable_weight=True requires init_w_y > 0 (softplus cannot represent exactly 0)."
            )
        if init_gamma <= 0:
            raise ValueError(f"LorentzResidual requires init_gamma > 0, got {init_gamma}")

        self.use_scale = scale
        self.eps = eps
        self.weight_parameterization = weight_parameterization
        self.scale_parameterization = scale_parameterization

        # y-weight: learnable nnx.Param (softplus-constrained or raw) when learnable, plain float otherwise.
        if learnable_weight:
            raw_w_y = _inv_softplus(init_w_y) if weight_parameterization == "softplus" else init_w_y
            self.w_y_raw = nnx.Param(jnp.array(raw_w_y, dtype=param_dtype))
            self._w_y = None
        else:
            self.w_y_raw = None
            self._w_y = init_w_y

        # Eq. 10 scaling constant gamma (only consulted when use_scale is True).
        if scale and learnable_scale:
            raw_gamma = _inv_softplus(init_gamma) if scale_parameterization == "softplus" else math.log(init_gamma)
            self.gamma_raw = nnx.Param(jnp.array(raw_gamma, dtype=param_dtype))
            self._gamma = None
        else:
            self.gamma_raw = None
            self._gamma = init_gamma

    def __call__(
        self,
        x: Float[Array, "... dim_plus_1"],
        y: Float[Array, "... dim_plus_1"],
        c: float = 1.0,
        weight: float | Float[Array, "..."] | None = None,
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
        weight : float or Array, optional
            If given, used as ``w_y`` in place of the module's own weight (fixed
            or learnable), as HELM's ``LResNet.forward(x, y, weight=...)``. A
            scalar, or one weight per point with shape ``x.shape[:-1]`` or
            ``x.shape[:-1] + (1,)`` (e.g. per-token gate weights). Not
            constrained: a negative value is passed through as is.

        Returns
        -------
        Array, shape (..., d+1)
            Points on the hyperboloid with curvature ``c``.
        """
        # Recover w_y from its raw param when learnable (softplus > 0, or the raw value itself);
        # cast to the input dtype so a float32-pinned param does not silently downcast a float64
        # forward pass. A call-time `weight` overrides the module's own w_y.
        if weight is not None:
            w_y = weight
        elif self.w_y_raw is None:
            w_y = self._w_y
        elif self.weight_parameterization == "softplus":
            w_y = jax.nn.softplus(self.w_y_raw[...])
        else:
            w_y = self.w_y_raw[...]
        out = lorentz_residual(x, y, w_y=jnp.asarray(w_y, dtype=x.dtype), c=c, eps=self.eps)

        if self.use_scale:
            if self.gamma_raw is None:
                gamma = self._gamma
            elif self.scale_parameterization == "softplus":
                gamma = jax.nn.softplus(self.gamma_raw[...])
            else:
                gamma = jnp.exp(self.gamma_raw[...])
            out = lorentz_scale(out, gamma=jnp.asarray(gamma, dtype=x.dtype), c=c, eps=self.eps)

        return out

    def project_w_y(self) -> None:
        """Project a learnable identity-mode ``w_y`` back onto ``w_y >= 0``, in place.

        Call it after each ``optimizer.update(model, grads)`` (inside the same
        ``nnx.jit`` train step is fine), or call :func:`project_residual_weights`
        on the whole model. Together with the optimizer step this is projected
        gradient descent: LResNet's Lemma 4.1 bounds the normalizer only for
        ``w_y >= 0``, and a negative ``w_y`` can make the combination spacelike
        (NaN). Inside the domain HELM's dynamics are unchanged. ``w_y = 0`` is
        valid (the output is ``x``), and the gradient there is nonzero, so the
        weight can move back up.

        A no-op under the softplus parameterization (already ``> 0``) and for a
        fixed weight (``learnable_weight=False``).
        """
        if self.w_y_raw is not None and self.weight_parameterization == "identity":
            self.w_y_raw[...] = jnp.maximum(self.w_y_raw[...], 0)


def project_residual_weights(model: nnx.Module) -> None:
    """Call :meth:`LorentzResidual.project_w_y` on every ``LorentzResidual`` in ``model``.

    Call it after each ``optimizer.update(model, grads)``, inside the same
    ``nnx.jit`` train step if you like::

        optimizer.update(model, grads)
        project_residual_weights(model)

    It keeps every identity-mode (HELM) ``w_y`` in the domain ``w_y >= 0`` of
    LResNet's Lemma 4.1; softplus-mode and fixed weights are left unchanged.

    Parameters
    ----------
    model : nnx.Module
        Any module; its submodules are searched recursively, ``model`` included.
    """
    for _, module in nnx.iter_modules(model):
        if isinstance(module, LorentzResidual):
            module.project_w_y()
