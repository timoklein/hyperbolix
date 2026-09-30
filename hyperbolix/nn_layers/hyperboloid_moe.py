"""Mixture of curvature experts (MiCE) from HELM.

:class:`LorentzMoE` is the feed-forward block of HELM-MiCE (He et al. 2025): a DeepSeek-style
mixture of experts on the hyperboloid in which every routed expert works on its own hyperboloid
sheet, with its own (learnable) curvature. A token is mapped onto an expert's sheet by scaling,
passed through that expert's Lorentzian SwiGLU (:class:`LorentzSwiGLU`), and mapped back; the
gate-weighted routed outputs and the shared-expert outputs are merged by one weighted Lorentzian
centroid. The Lorentzian residual with the block input is left to the caller (the decoder block),
as in HELM.

:class:`LorentzMoEGate` is the router, with DeepSeek-V3's auxiliary-loss-free load balancing
(a non-gradient routing bias, updated by :meth:`LorentzMoEGate.update_bias`), and
:func:`moe_sequence_balance_loss` is DeepSeek-V3's complementary sequence-wise balance loss.

Curvature convention: the paper writes the curvature as ``K < 0``; hyperbolix uses ``c = -K > 0``
with ``<x, x>_L = -1/c``. The HELM code's manifold parameter ``k`` (``<x, x>_L = -k``) is
``1/c``.

Dimension key
-------------
B : batch size (sequences)
S : sequence length (tokens per sequence)
N : tokens, flattened over all leading axes
A : model ambient dimension (``dim``)
O : model spatial dimension (``dim - 1``)
I : expert hidden spatial dimension (``inter_dim``)
H : expert hidden ambient dimension (``inter_dim + 1``)
E : routed experts
R : shared experts
M : merged expert outputs per token (``E + R``)
K : experts selected per token (``top_k``)
L : leading axes of the input, ``x.shape[:-1]`` (any number, flattened into ``N``)

References
----------
He et al., "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts", 2025
(arXiv:2505.24722).
DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024 (arXiv:2412.19437), Sec. 2.1.2.
Wang et al., "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", 2024
(arXiv:2408.15664).
"""

import math
from collections.abc import Sequence
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int

from hyperbolix.manifolds.protocol import ScalarCurvature
from hyperbolix.utils.curvature import LearnableCurvature, Parameterization
from hyperbolix.utils.math_utils import floor_at

from .hyperboloid_core import lorentz_midpoint, spatial_to_hyperboloid
from .hyperboloid_linear import HTCLinear

ScoreFunc = Literal["softmax", "sigmoid"]


def _helm_xavier_bound(in_features: int, out_features: int) -> float:
    """HELM ``LorentzLinear`` init bound: Xavier-uniform with gain ``sqrt(2)``.

    ``gain * sqrt(6 / (fan_in + fan_out)) = sqrt(12 / (in_features + out_features))``, with
    ``in_features`` the ambient input width and ``out_features`` the spatial output width, the
    fan-in/fan-out of HELM's ``nn.Linear(in_features, out_features)``.
    """
    return math.sqrt(12.0 / (in_features + out_features))


class RoutingBias(nnx.Variable):
    """Per-expert routing bias of :class:`LorentzMoEGate`: state, not a parameter.

    It only shifts the top-k selection, which has no gradient, and it moves only through
    :meth:`LorentzMoEGate.update_bias`. Being an ``nnx.Variable`` and not an ``nnx.Param``,
    ``nnx.grad`` and ``nnx.Optimizer(..., wrt=nnx.Param)`` skip it.
    """


class MoERoutingStats(NamedTuple):
    """Per-token routing statistics, the inputs of the load-balancing updates.

    Attributes
    ----------
    affinity : Array, shape (..., E)
        Gate scores normalised to sum to 1 over the experts of each token (DeepSeek-V3's
        ``s'_{i,t}``), in at least float32. Differentiable with respect to the gate kernel.
    mask : Array, shape (..., E)
        1.0 where the expert is among the token's selected ``top_k`` experts (selection on the
        biased scores), else 0.0. Same dtype as ``affinity``; no gradient.
    """

    affinity: Float[Array, "... E"]
    mask: Float[Array, "... E"]


class LorentzMoEGate(nnx.Module):
    """Top-k router of HELM-MiCE with DeepSeek-V3's auxiliary-loss-free balancing bias.

    For a token ``x`` on the hyperboloid with spatial part ``x_s``, following the HELM paper:

    1. Scores ``s_j = act(<x_s, y_j>)``, with ``y_j`` the column ``j`` of :attr:`kernel` and
       ``act`` a softmax over the experts or an elementwise sigmoid, in at least float32.
    2. Selection: the ``top_k`` experts by ``s_j + b_j``, ``b`` the routing bias.
    3. Weights: the *unbiased* scores of the selected experts, renormalised to sum to 1 over the
       selection, times ``route_scale``: ``g_i = route_scale * s_i / sum_{j in TopK} s_j``.

    The bias only changes which experts are chosen, never their weights, and it is not trained by
    gradient descent: call :meth:`update_bias` once per training step with the step's routing
    mask (DeepSeek-V3's auxiliary-loss-free load balancing).

    Parameters
    ----------
    dim : int
        Model ambient dimension ``A``; the gate reads the ``A - 1`` spatial coordinates.
    num_experts : int
        Number of routed experts ``E``.
    top_k : int
        Experts selected per token, ``1 <= top_k <= num_experts``.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    score_func : {"softmax", "sigmoid"}, optional
        Score activation (default: "softmax", HELM's config).
    route_scale : float, optional
        Positive factor on the routing weights (default: 1.0, HELM's config).
    param_dtype : DTypeLike, optional
        Storage dtype of the kernel and bias (default: jnp.float32).

    Attributes
    ----------
    kernel : nnx.Param
        Gate vectors, shape ``(A - 1, E)`` (HELM's ``weight`` transposed to the Flax layout),
        Xavier-uniform with gain 1 as in HELM.
    bias : RoutingBias
        Routing bias, shape ``(E,)``, init 0.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        *,
        rngs: nnx.Rngs,
        score_func: ScoreFunc = "softmax",
        route_scale: float = 1.0,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if dim < 2:
            raise ValueError(f"dim is the ambient model width and must be >= 2, got {dim}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, num_experts={num_experts}], got {top_k}")
        if score_func not in ("softmax", "sigmoid"):
            raise ValueError(f"score_func must be 'softmax' or 'sigmoid', got {score_func!r}")
        if not route_scale > 0:
            raise ValueError(f"route_scale must be positive (the centroid needs non-negative weights), got {route_scale}")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_scale = route_scale
        # Xavier-uniform, gain 1: fan_in = A - 1, fan_out = E, as torch's xavier_uniform_ on HELM's (E, A - 1) weight.
        self.kernel = nnx.Param(nnx.initializers.xavier_uniform()(rngs.params(), (dim - 1, num_experts), param_dtype))
        self.bias = RoutingBias(jnp.zeros((num_experts,), dtype=param_dtype))

    def __call__(self, x_NA: Float[Array, "N A"]) -> tuple[Float[Array, "N K"], Int[Array, "N K"], MoERoutingStats]:
        """Route a batch of tokens.

        Parameters
        ----------
        x_NA : Array, shape (N, A)
            Tokens on the hyperboloid (any curvature: the gate reads the spatial part only).

        Returns
        -------
        weights_NK : Array, shape (N, K)
            Routing weights of the selected experts (at least float32), renormalised over the
            selection and scaled by ``route_scale``.
        indices_NK : int Array, shape (N, K)
            Selected experts, in descending order of biased score.
        stats : MoERoutingStats
            ``affinity`` and ``mask``, each ``(N, E)``.
        """
        score_dtype = jnp.promote_types(x_NA.dtype, jnp.float32)
        space_NO = x_NA[..., 1:].astype(score_dtype)  # (N, O)
        # Gate GEMM: a layer weight GEMM, following JAX's default matmul precision.
        logits_NE = jnp.matmul(space_NO, self.kernel[...].astype(score_dtype))  # (N, E)
        if self.score_func == "softmax":
            scores_NE = jax.nn.softmax(logits_NE, axis=-1)
        else:
            scores_NE = jax.nn.sigmoid(logits_NE)
        biased_NE = scores_NE + self.bias[...].astype(score_dtype)  # (N, E)
        _, indices_NK = jax.lax.top_k(biased_NE, self.top_k)  # (N, K)
        selected_NK = jnp.take_along_axis(scores_NE, indices_NK, axis=-1)  # (N, K), unbiased
        # Renormalise over the selection (HELM paper, both score functions). The floor only acts if
        # every selected score underflowed to 0 (a softmax far from the biased choice); the weights
        # are then 0 instead of 0/0.
        tiny = jnp.asarray(jnp.finfo(score_dtype).tiny, dtype=score_dtype)
        weights_NK = self.route_scale * selected_NK / floor_at(jnp.sum(selected_NK, axis=-1, keepdims=True), tiny)
        mask_NE = jnp.sum(jax.nn.one_hot(indices_NK, self.num_experts, dtype=score_dtype), axis=-2)  # (N, E)
        # DeepSeek-V3's s'_{i,t}: affinities normalised per token (a no-op for softmax up to rounding).
        affinity_NE = scores_NE / floor_at(jnp.sum(scores_NE, axis=-1, keepdims=True), tiny)
        return weights_NK, indices_NK, MoERoutingStats(affinity=affinity_NE, mask=mask_NE)

    def update_bias(self, mask: Float[Array, "... E"], speed: float) -> None:
        """DeepSeek-V3's auxiliary-loss-free bias update, ``b_i += speed * sign(mean_load - load_i)``.

        ``load_i`` is the number of tokens routed to expert ``i`` in ``mask`` and ``mean_load`` its
        mean over the experts: an overloaded expert's bias drops by ``speed``, an underloaded one's
        rises by ``speed``, and an exactly average one's is unchanged. DeepSeek-V3 applies it at
        the end of each training step, over all tokens of the step (gather ``mask`` across data
        shards first); its ``speed`` (``gamma``) was 0.001, HELM's config sets 0.005.

        Parameters
        ----------
        mask : Array, shape (..., E)
            The routing mask of the step (:attr:`MoERoutingStats.mask`), any leading shape.
        speed : float
            Bias update speed ``gamma``.
        """
        if mask.shape[-1] != self.num_experts:
            raise ValueError(f"mask must have {self.num_experts} experts on its last axis, got shape {mask.shape}")
        bias_E = self.bias[...]  # (E,)
        load_E = jnp.sum(mask.reshape(-1, self.num_experts).astype(bias_E.dtype), axis=0)  # (E,)
        self.bias[...] = bias_E + speed * jnp.sign(jnp.mean(load_E) - load_E)


class LorentzSwiGLU(nnx.Module):
    """HELM's Lorentzian SwiGLU feed-forward (HFFN), run on an expert's own hyperboloid sheet.

    For an input ``x`` on the model sheet (curvature ``c``) and an expert curvature ``c_e``:

    1. Map onto the expert sheet, ``x_e = sqrt(c / c_e) x`` (the paper's ``sqrt(K / K_e) x``),
       computed as the spatial scaling with the time coordinate rebuilt at ``c_e``
       (:func:`spatial_to_hyperboloid`), which equals the scaled ambient vector for on-sheet ``x``.
    2. ``h = silu(w1(x_e)_s) * w3(x_e)_s``, the spatial outputs of two HTC linear maps at ``c_e``.
    3. ``h`` gets its time coordinate at ``c_e``.
    4. ``w2`` maps the hidden point to ``A - 1`` spatial coordinates and :func:`htc` scales them by
       ``sqrt(c_e / c)`` and rebuilds the time at ``c``, which is the paper's map back
       ``sqrt(K_e / K) HFFN_e(x_e)`` (the ambient point on the ``c_e`` sheet, scaled).

    With ``c_e = c`` this is HELM's dense ``LorentzFeedForward``.

    Parameters
    ----------
    dim : int
        Model ambient dimension ``A`` (input and output).
    inter_dim : int
        Spatial width ``I`` of the hidden point (HELM's ``inter_dim`` is ambient, ``I + 1``).
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    init_bound : float or None, optional
        Uniform init bound shared by the three maps. ``None`` (the default) gives each HELM's
        reference init, Xavier-uniform with gain ``sqrt(2)``: ``sqrt(12 / (in + out))`` with
        ambient ``in`` and spatial ``out``. Biases start at 0.
    param_dtype : DTypeLike, optional
        Storage dtype of the parameters (default: jnp.float32). Compute follows the input dtype.
    eps : float, optional
        Numerical floor of the time rebuilds (default: 1e-7).

    Attributes
    ----------
    w1, w3 : HTCLinear
        ``A -> I`` gate and up projections.
    w2 : HTCLinear
        ``I + 1 -> A - 1`` down projection.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        *,
        rngs: nnx.Rngs,
        init_bound: float | None = None,
        param_dtype: DTypeLike = jnp.float32,
        eps: float = 1e-7,
    ):
        if dim < 2:
            raise ValueError(f"dim is the ambient model width and must be >= 2, got {dim}")
        if inter_dim < 1:
            raise ValueError(f"inter_dim (spatial) must be >= 1, got {inter_dim}")
        self.dim = dim
        self.inter_dim = inter_dim
        self.eps = eps

        def projection(in_features: int, out_features: int) -> HTCLinear:
            bound = init_bound if init_bound is not None else _helm_xavier_bound(in_features, out_features)
            return HTCLinear(
                in_features, out_features, rngs=rngs, use_bias=True, init_bound=bound, eps=eps, param_dtype=param_dtype
            )

        self.w1 = projection(dim, inter_dim)
        self.w2 = projection(inter_dim + 1, dim - 1)
        self.w3 = projection(dim, inter_dim)

    def __call__(
        self,
        x_NA: Float[Array, "N A"],
        c_expert: ScalarCurvature,
        c: ScalarCurvature = 1.0,
    ) -> Float[Array, "N A"]:
        """Apply the expert.

        Parameters
        ----------
        x_NA : Array, shape (..., A)
            Points on the model hyperboloid with curvature ``c``.
        c_expert : float or scalar Array
            The expert's curvature ``c_e``.
        c : float or scalar Array, optional
            Model curvature (default: 1.0).

        Returns
        -------
        Array, shape (..., A)
            Points on the model hyperboloid with curvature ``c``.
        """
        # HTCLinear annotates its curvatures as float; it takes traced scalars (learnable curvature) at runtime.
        c_e, c_m = cast(float, c_expert), cast(float, c)
        x_expert_NA = spatial_to_hyperboloid(x_NA[..., 1:], c_m, c_e, self.eps)  # (N, A) on the c_e sheet
        gate_NI = self.w1(x_expert_NA, c_e, c_e, return_space=True)  # (N, I)
        up_NI = self.w3(x_expert_NA, c_e, c_e, return_space=True)  # (N, I)
        hidden_NH = spatial_to_hyperboloid(jax.nn.silu(gate_NI) * up_NI, c_e, c_e, self.eps)  # (N, H)
        return self.w2(hidden_NH, c_e, c_m)  # (N, A), mapped back onto the c sheet


def _stacked_experts(
    num: int,
    dim: int,
    inter_dim: int,
    *,
    rngs: nnx.Rngs,
    init_bound: float | None,
    param_dtype: DTypeLike,
    eps: float,
) -> LorentzSwiGLU:
    """``num`` independent :class:`LorentzSwiGLU` experts stacked on a leading parameter axis."""

    @nnx.split_rngs(splits=num)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def create(expert_rngs: nnx.Rngs) -> LorentzSwiGLU:
        return LorentzSwiGLU(dim, inter_dim, rngs=expert_rngs, init_bound=init_bound, param_dtype=param_dtype, eps=eps)

    return create(rngs)


@nnx.vmap(in_axes=(0, 0, None, None), out_axes=0)
def _run_stacked_experts(
    experts: LorentzSwiGLU,
    c_expert: ScalarCurvature,
    x_NA: Float[Array, "N A"],
    c: ScalarCurvature,
) -> Float[Array, "N A"]:
    """Run each stacked expert, at its own curvature, on every token: ``(E,) x (N, A) -> (E, N, A)``."""
    return experts(x_NA, c_expert, c)


class LorentzMoE(nnx.Module):
    """HELM's mixture of curvature experts (MiCE), in the paper's form.

    For a token ``x`` on the hyperboloid with curvature ``c`` (paper, Sec. 4, with ``K = -c``):

    1. The gate (:class:`LorentzMoEGate`) selects ``top_k`` of the ``E`` routed experts and gives
       them weights ``g_i`` (renormalised over the selection; ``g_i = 0`` for the others).
    2. Routed expert ``i`` has its own curvature ``c_i``: ``y_i = sqrt(c_i / c)
       HFFN_i(sqrt(c / c_i) x)``, a :class:`LorentzSwiGLU` on the ``c_i`` sheet, mapped back.
    3. Each of the ``R`` shared experts gives ``z_j = HFFN_j(x)`` at the model curvature ``c``.
    4. Output: the weighted Lorentzian centroid ::

           MiCE(x) = (sum_i g_i y_i + sum_j z_j) / (sqrt(c) ||sum_i g_i y_i + sum_j z_j||_L)

       computed by :func:`lorentz_midpoint` (its cancellation-free default form) over the
       ``E + R`` outputs with weights ``[g, 1, ..., 1]``.

    The block's Lorentzian residual ``x ⊕_L MiCE(x)`` is not part of this module; HELM applies it
    in the decoder block. With :class:`~hyperbolix.nn_layers.LorentzResidual` in HELM's block
    configuration (raw learnable weight, fixed ``sqrt(dim)`` output scale)::

        self.moe = LorentzMoE(dim, inter_dim, num_routed=8, num_shared=1, top_k=2, rngs=rngs)
        self.ffn_res = LorentzResidual(weight_parameterization="identity", scale=True, init_gamma=math.sqrt(dim))
        ...
        moe_out_BSA, stats = self.moe(h_BSA, c)  # h_BSA: the normalised block input
        x_BSA = self.ffn_res(x_BSA, moe_out_BSA, c=c)

    Load balancing, as the paper takes it from DeepSeek-V3, uses the returned ``stats``: add
    ``moe_sequence_balance_loss(stats, alpha)`` to the loss (HELM's ``alpha = 1e-4``), and after
    the optimizer step call ``moe.update_bias(stats.mask, speed)``.

    Curvatures
    ----------
    The routed experts start at ``c_i = linspace(0.1, 2.0, E)`` (the paper: "curvature initiated
    uniformly from -0.1 to -2.0"), or at ``expert_curvatures``. With ``learnable_curvature=True``
    each routed expert has its own :class:`~hyperbolix.utils.curvature.LearnableCurvature`
    (clamped to ``[c_min, c_max]``). The lowest paper expert starts exactly at the default
    ``c_min = 0.1``; under LearnableCurvature's own default hard clamp, the first optimizer step
    that takes it below ``c_min`` pins it at the floor with zero gradient for good. So
    ``straight_through_clamp`` defaults to True here, unlike LearnableCurvature's default False:
    the forward value stays clamped, the gradient still passes, and ``c_i`` can move back up.
    HELM trains the raw curvatures with its AdamW weight decay (0.01); here decay of
    ``curvatures[i].raw`` is up to the caller's optax chain (under ``log`` it pulls ``c_i`` toward 1).

    Dispatch
    --------
    Dense: every routed expert runs on every token (the experts are stacked on a leading
    parameter axis and applied with ``nnx.vmap``), and unselected outputs enter the centroid with
    weight 0, so they get zero gradient for that token. Cost is ``E / top_k`` times that of a
    sparse dispatch, fine for small ``E``; large ``E`` needs a gather or ragged dispatch. One
    difference from a sparse dispatch: a non-finite output of an *unselected* expert still
    reaches the centroid (``0 * inf``) and turns the token's output NaN.

    Differences from the HELM code
    ------------------------------
    The paper's form is implemented; where the released code differs, the code is not followed:

    - **Weights renormalised for softmax too.** The code renormalises the top-k weights only for
      sigmoid (``if self.score_func == "sigmoid": weights /= weights.sum(dim=-1, keepdim=True)``);
      the paper renormalises in both cases.
    - **One centroid, order-free.** The code merges routed outputs one expert at a time with an
      LResNet that rescales the space part by a fixed 2 (``y[idx] = self.weighted_sum(y[idx],
      expert(x[idx]), weight=weights[idx, top, None])``), then merges the shared output with a
      second LResNet with a learnable log-scale (``out = self.add_experts(z, y)``), so the result
      depends on expert order and carries two extra scales. The two LResNets are also built under
      crossed conditions (``add_experts`` under ``n_activated_experts == 2``, used under
      ``n_shared_experts == 1``). Here the output is the paper's single centroid, with no scale.
    - **Curvature init.** The code builds every expert at ``Lorentz(c=1.0, learnable=args.train_curv)``;
      its ``np.linspace(0.1, 2.0, ...)`` list is unused. Here ``c_i = linspace(0.1, 2.0, E)``.
    - **Shared experts** are ``R`` separate SwiGLU experts, each with weight 1 in the centroid
      (the paper's ``sum_j z_j``); the code has one ``LorentzFeedForward`` of width
      ``n_shared_experts * moe_inter_dim``. They agree for ``R = 1``.
    - **Routing bias.** The code's bias is an ``nn.Parameter`` that no gradient reaches and whose
      update is commented out of the training loop. Here it is a :class:`RoutingBias` with the
      DeepSeek-V3 sign update (:meth:`update_bias`); the code's unused ``update_bias`` was
      proportional (``self.bias += self.bias_update_spd * (mean - util)``).
    - **Balance loss.** The code's ``sequence_balance_loss`` fixes ``k=2``, overwrites the
      expert counts with the index tensor (``freq = indices * (E / (k * N))``, a shape error) and
      averages over all tokens of the batch; :func:`moe_sequence_balance_loss` is DeepSeek-V3's
      per-sequence form.
    - **Library forms** replace the code's ``clamp_min(1e-8)`` time rebuilds and centroid
      normalizer: :func:`spatial_to_hyperboloid` and :func:`lorentz_midpoint`.
    - **Shared-expert curvature.** The paper's equations give each shared expert its own curvature
      ``K_{s,j}`` (``z_j = sqrt(K_{s,j}/K) HFFN_j(sqrt(K/K_{s,j}) x)``) but state no init for it;
      here, as in the code, the shared experts run at the model curvature ``c``.
    - **Not implemented:** expert groups (``n_expert_groups``/``n_limited_groups``; HELM's config
      uses one group, where they are a no-op).

    Parameters
    ----------
    dim : int
        Model ambient dimension ``A``.
    inter_dim : int
        Spatial hidden width ``I`` of every expert (HELM's ``mice_inter_dim`` is ambient: HELM's
        1820 is ``inter_dim = 1819`` here).
    num_routed : int
        Number of routed experts ``E``.
    num_shared : int
        Number of shared experts ``R`` (may be 0).
    top_k : int
        Routed experts selected per token.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    route_scale : float, optional
        Positive factor on the routing weights, i.e. the routed experts' total weight against
        each shared expert's 1 (default: 1.0).
    score_func : {"softmax", "sigmoid"}, optional
        Gate score activation (default: "softmax").
    expert_curvatures : sequence of float or None, optional
        Initial curvatures of the ``E`` routed experts; ``None`` (the default) is the paper's
        ``linspace(0.1, 2.0, E)``.
    learnable_curvature : bool, optional
        Train the routed experts' curvatures (default: True). If False they stay fixed.
    curvature_parameterization : {"log", "softplus", "identity"}, optional
        Passed to each ``LearnableCurvature`` (default: "log").
    c_min, c_max : float or None, optional
        Clamp of each learnable expert curvature, passed to ``LearnableCurvature`` (defaults
        0.1 and 10.0, LearnableCurvature's own defaults for ``log``/``softplus``; the expert
        maps need ``c_i > 0``). ``None`` disables that side.
    straight_through_clamp : bool, optional
        Passed to each ``LearnableCurvature`` (default: True, unlike LearnableCurvature's False;
        see "Curvatures" above).
    init_bound : float or None, optional
        Uniform init bound of every expert projection; ``None`` (the default) is HELM's
        ``sqrt(12 / (in + out))`` per projection (see :class:`LorentzSwiGLU`).
    param_dtype : DTypeLike, optional
        Storage dtype of the layer weights (default: jnp.float32). The raw curvatures stay
        float32 (LearnableCurvature's default). Compute follows the input dtype.
    eps : float, optional
        Numerical floor of the time rebuilds and the centroid (default: 1e-7).

    Attributes
    ----------
    gate : LorentzMoEGate
    experts : LorentzSwiGLU
        The ``E`` routed experts, parameters stacked on a leading axis of size ``E``.
    shared_experts : LorentzSwiGLU or None
        The ``R`` shared experts, stacked likewise; None if ``num_shared = 0``.
    curvatures : nnx.List of LearnableCurvature, or None
        One per routed expert if ``learnable_curvature``.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        num_routed: int,
        num_shared: int,
        top_k: int,
        *,
        rngs: nnx.Rngs,
        route_scale: float = 1.0,
        score_func: ScoreFunc = "softmax",
        expert_curvatures: Sequence[float] | None = None,
        learnable_curvature: bool = True,
        curvature_parameterization: Parameterization = "log",
        c_min: float | None = 0.1,
        c_max: float | None = 10.0,
        straight_through_clamp: bool = True,
        init_bound: float | None = None,
        param_dtype: DTypeLike = jnp.float32,
        eps: float = 1e-7,
    ):
        if num_shared < 0:
            raise ValueError(f"num_shared must be >= 0, got {num_shared}")
        if expert_curvatures is None:
            init_curvatures = tuple(float(ci) for ci in np.linspace(0.1, 2.0, num_routed))
        else:
            init_curvatures = tuple(float(ci) for ci in expert_curvatures)
            if len(init_curvatures) != num_routed:
                raise ValueError(f"expert_curvatures needs num_routed={num_routed} values, got {len(init_curvatures)}")
        if not all(math.isfinite(ci) and ci > 0 for ci in init_curvatures):
            raise ValueError(f"expert curvatures must be finite and positive, got {init_curvatures}")

        self.dim = dim
        self.inter_dim = inter_dim
        self.num_routed = num_routed
        self.num_shared = num_shared
        self.top_k = top_k
        self.eps = eps
        self.gate = LorentzMoEGate(
            dim, num_routed, top_k, rngs=rngs, score_func=score_func, route_scale=route_scale, param_dtype=param_dtype
        )
        expert_kwargs = {"rngs": rngs, "init_bound": init_bound, "param_dtype": param_dtype, "eps": eps}
        self.experts = _stacked_experts(num_routed, dim, inter_dim, **expert_kwargs)
        self.shared_experts = _stacked_experts(num_shared, dim, inter_dim, **expert_kwargs) if num_shared > 0 else None
        self.init_curvatures = init_curvatures  # static Python floats; the fixed values if not learnable
        if learnable_curvature:
            self.curvatures = nnx.List(
                [
                    LearnableCurvature(
                        ci,
                        parameterization=curvature_parameterization,
                        c_min=c_min,
                        c_max=c_max,
                        straight_through_clamp=straight_through_clamp,
                    )
                    for ci in init_curvatures
                ]
            )
        else:
            self.curvatures = None

    def routed_curvatures(self) -> Float[Array, "E"]:
        """Current curvatures ``c_i`` of the routed experts, shape ``(E,)``."""
        if self.curvatures is None:
            return jnp.asarray(self.init_curvatures, dtype=jnp.float32)
        return jnp.stack([curvature() for curvature in self.curvatures])

    def update_bias(self, mask: Float[Array, "... E"], speed: float) -> None:
        """Auxiliary-loss-free bias update of the gate; see :meth:`LorentzMoEGate.update_bias`."""
        self.gate.update_bias(mask, speed)

    def __call__(
        self,
        x: Float[Array, "... A"],
        c: ScalarCurvature = 1.0,
    ) -> tuple[Float[Array, "... A"], MoERoutingStats]:
        """Apply the mixture of curvature experts.

        Parameters
        ----------
        x : Array, shape (..., A)
            Tokens on the hyperboloid with curvature ``c``, e.g. ``(N, A)`` or ``(B, S, A)``.
        c : float or scalar Array, optional
            Model curvature (default: 1.0).

        Returns
        -------
        out : Array, shape (..., A)
            The weighted Lorentzian centroid of the expert outputs, on the hyperboloid with
            curvature ``c`` (no residual; see the class docstring).
        stats : MoERoutingStats
            ``affinity`` and ``mask``, each of shape ``(..., E)``, for
            :func:`moe_sequence_balance_loss` and :meth:`update_bias`.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(f"x must have {self.dim} ambient coordinates on its last axis, got shape {x.shape}")
        lead_shape = x.shape[:-1]
        work_dtype = x.dtype
        x_NA = x.reshape(-1, self.dim)  # (N, A)
        # Curvatures in the working dtype, so a float32 curvature does not promote a bfloat16 forward.
        c_work = jnp.asarray(c, dtype=work_dtype)
        c_expert_E = self.routed_curvatures().astype(work_dtype)  # (E,)

        weights_NK, indices_NK, gate_stats = self.gate(x_NA)
        # Dense (N, E) gate weights: g_i for the selected experts, 0 for the rest.
        gate_NE = jnp.sum(jax.nn.one_hot(indices_NK, self.num_routed, dtype=weights_NK.dtype) * weights_NK[..., None], axis=-2)

        outputs_ENA = _run_stacked_experts(self.experts, c_expert_E, x_NA, c_work)  # (E, N, A)
        weights_NM = gate_NE.astype(work_dtype)  # (N, E)
        if self.shared_experts is not None:
            c_shared_R = jnp.broadcast_to(c_work, (self.num_shared,))  # shared experts sit at the model c
            shared_RNA = _run_stacked_experts(self.shared_experts, c_shared_R, x_NA, c_work)  # (R, N, A)
            outputs_ENA = jnp.concatenate([outputs_ENA, shared_RNA], axis=0)  # (M, N, A)
            weights_NM = jnp.concatenate([weights_NM, jnp.ones((x_NA.shape[0], self.num_shared), dtype=work_dtype)], axis=-1)

        points_NMA = jnp.transpose(outputs_ENA, (1, 0, 2))  # (N, M, A)
        centroid_N1A = lorentz_midpoint(points_NMA, weights_NM[:, None, :], c_work, eps=self.eps)  # (N, 1, A)
        out_LA = centroid_N1A[:, 0, :].reshape(*lead_shape, self.dim)  # (L..., A)
        routing_stats = MoERoutingStats(
            affinity=gate_stats.affinity.reshape(*lead_shape, self.num_routed),  # (L..., E)
            mask=gate_stats.mask.reshape(*lead_shape, self.num_routed),  # (L..., E)
        )
        return out_LA, routing_stats


def moe_sequence_balance_loss(stats: MoERoutingStats, alpha: float) -> Float[Array, ""]:
    """DeepSeek-V3's complementary sequence-wise balance loss, averaged over sequences.

    For one sequence of ``T`` tokens, ``E`` routed experts and ``k`` experts per token
    (DeepSeek-V3, Eqs. 17-20)::

        L_bal = alpha * sum_i f_i P_i
        f_i   = E / (k T) * sum_t 1[expert i in the top-k of token t]
        P_i   = 1 / T * sum_t s'_{i,t},   s'_{i,t} = s_{i,t} / sum_j s_{j,t}

    ``f_i`` is the expert's load relative to a uniform share (it has no gradient) and ``P_i`` its
    mean normalised affinity (differentiable in the gate kernel). ``k`` is read off the mask:
    ``k T = sum_{i,t} mask``. HELM's config sets ``alpha = 1e-4``.

    Values. Balanced routing (every expert in the top-k of ``k T / E`` tokens) gives
    ``f_i = 1`` and ``L = alpha`` for any affinities, and so does any routing when the affinities
    are uniform. All tokens routed to the same ``k`` experts gives ``L >= alpha``, with equality
    only for uniform affinities, since the selected experts' affinities sum to at least ``k / E``
    per token. ``alpha`` is not a lower bound in general, though: a token routed to an expert it
    barely prefers lowers ``sum_i f_i P_i`` below 1 (``E = 2``, ``k = 1``, two tokens at
    ``s' = (0.6, 0.4)`` on expert 0 and one at ``(0, 1)`` on expert 1 give ``14/15 alpha``).

    Selection. DeepSeek-V3 writes the indicator with the top-k of the unbiased scores; here the
    mask is the routing actually used, selected on the biased scores, so ``f_i`` is the load that
    the bias is balancing. The two coincide while the bias is zero.

    Parameters
    ----------
    stats : MoERoutingStats
        Routing statistics with shape ``(..., S, E)``: the second-to-last axis is the sequence,
        every leading axis indexes sequences (``(B, S, E)`` from a ``(B, S, A)`` input, or
        ``(S, E)`` for one sequence).
    alpha : float
        Balance factor.

    Returns
    -------
    Array, shape ()
        The loss, averaged over sequences.
    """
    affinity, mask = stats.affinity, stats.mask
    if affinity.ndim < 2 or affinity.shape != mask.shape:
        raise ValueError(
            f"stats must hold (..., S, E) arrays of equal shape, got affinity {affinity.shape} and mask {mask.shape}"
        )
    num_experts = affinity.shape[-1]
    mask = jax.lax.stop_gradient(mask.astype(affinity.dtype))
    load_BE = jnp.sum(mask, axis=-2)  # (..., E), tokens routed to each expert
    # f_i = E * load_i / (k T), with k T = sum_i load_i (every token selects exactly k experts).
    frequency_BE = num_experts * load_BE / jnp.sum(load_BE, axis=-1, keepdims=True)  # (..., E)
    probability_BE = jnp.mean(affinity, axis=-2)  # (..., E), P_i
    return alpha * jnp.mean(jnp.sum(frequency_BE * probability_BE, axis=-1))
