"""Hyperbolic multi-head latent attention (HMLA) from HELM.

:class:`LorentzMLA` is the attention block of HELM (He et al. 2025): DeepSeek-style multi-head
latent attention moved onto the hyperboloid. Keys and values are decompressed from one low-rank
latent point per token, the queries and keys carry a decoupled HOPE-rotated slice, the score is
the negative squared Lorentzian distance, and each head aggregates its values with the weighted
Lorentzian centroid.

Dimension key
-------------
B : batch size
S : query positions
T : key/value positions (``T = S``; self-attention)
H : number of heads
A : model ambient dimension (``dim``)
O : model spatial dimension (``dim - 1``)
R : key/value latent rank (``kv_lora_rank``, spatial)
N : non-rotated query/key dimension per head (``qk_nope_head_dim``, spatial)
P : rotated query/key spatial dimension per head (``qk_rope_head_dim - 1``)
K : query/key spatial dimension per head (``N + P``)
L : query/key ambient dimension per head (``K + 1 = qk_nope_head_dim + qk_rope_head_dim``)
V : value spatial dimension per head (``v_head_dim - 1``)
W : value ambient dimension per head (``V + 1 = v_head_dim``)
F : flattened head outputs (``H * W``)

References
----------
He et al., "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts", 2025
(arXiv:2505.24722).
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int

from .hyperboloid_core import MATMUL_PRECISION, lorentz_midpoint, spatial_to_hyperboloid
from .hyperboloid_linear import HTCLinear
from .hyperboloid_positional import hope_rotate_space
from .hyperboloid_regularization import HRCRMSNorm

# HELM's masked_fill value. Finite, so a fully masked row softmaxes to a finite uniform row
# instead of NaN; it is added in the (at least float32) softmax dtype, where it is representable.
_MASK_FILL = -1e18


def _helm_xavier_bound(in_features: int, out_features: int) -> float:
    """HELM ``LorentzLinear`` init bound: Xavier-uniform with gain ``sqrt(2)``.

    ``gain * sqrt(6 / (fan_in + fan_out)) = sqrt(12 / (in_features + out_features))``, with
    ``in_features`` the ambient input width and ``out_features`` the spatial output width, the
    fan-in/fan-out of HELM's ``nn.Linear(in_features, out_features)``.
    """
    return math.sqrt(12.0 / (in_features + out_features))


class LorentzMLA(nnx.Module):
    """Hyperbolic multi-head latent attention (HMLA, HELM's ``LorentzMLA``).

    Forward pass, for input points ``x`` on the hyperboloid ``<x, x>_L = -1/c``:

    1. ``q = wq(x)`` (spatial, ``H * K`` wide), split per head into ``q_nope`` (``N``) and
       ``q_pe`` (``P``); ``q_pe`` is HOPE-rotated.
    2. ``wkv_a(x)`` (spatial, ``R + P`` wide) is split into the latent ``(R,)`` and one ``k_pe``
       ``(P,)`` per token, shared by all heads and HOPE-rotated.
    3. The latent is RMS-normalized and given a time coordinate (``HRCRMSNorm``, spatial input),
       then ``wkv_b`` decompresses it into ``k_nope`` (``N``) and the value space (``V``) per head.
    4. ``q = [q_nope, q_pe]``, ``k = [k_nope, k_pe]`` and ``v`` get their time coordinates
       rebuilt from the hyperboloid constraint (HELM's "Lorentz concatenation").
    5. Scores ``(2/c + 2 <q, k>_L) / tau`` -- the negative squared Lorentzian distance divided
       by one learnable temperature ``tau`` shared by all heads; masked, softmaxed in at least
       float32.
    6. Each head takes the weighted Lorentzian centroid of its values (:func:`lorentz_midpoint`,
       the cancellation-free form of HELM's ``lorentzian_centroid``), the per-head ambient points
       (time included) are flattened to ``(H * W,)``, and ``wo`` maps them to the output point.
       The centroid's direct variance holds a ``(B, H, S, T, W - 1)`` intermediate, so memory
       grows as ``S^2 * v_head_dim`` per head (HELM's clamp form is a single GEMM).

    Only HELM's ``q_lora_rank = 0`` path (a direct query projection) is implemented; HELM's
    YaRN ``mscale`` adjustment of ``tau`` for extended contexts and its (commented-out) KV cache
    are not.

    Differences from the HELM code
    ------------------------------
    - **Temperature init.** HELM initializes ``tau = sqrt(H * L)`` (``L`` the ambient query/key
      head width), ``sqrt(H)`` larger than the per-head ``sqrt(L)`` of standard scaled
      dot-product attention, so its initial softmax is ``sqrt(H)`` times flatter. That is the
      default here too; ``init_tau=math.sqrt(qk_nope_head_dim + qk_rope_head_dim)`` gives the
      per-head scaling.
    - **No score bias.** HELM adds one learnable scalar to every score
      (``scores / self.softmax_scale + self.bias``). A constant shared by all keys of a row
      cancels in the softmax, so that parameter changes nothing and its gradient is exactly zero;
      it is omitted.
    - **Library forms** replace HELM's clamps: the centroid is :func:`lorentz_midpoint` (no
      ``clamp_min(eps)`` on ``|<h, h>_L|``, and cancellation-free at large radius), the time
      coordinates come from :func:`spatial_to_hyperboloid`.

    Conventions
    -----------
    - **Head dimensions** follow HELM: ``qk_nope_head_dim`` counts *spatial* coordinates, while
      ``qk_rope_head_dim`` and ``v_head_dim`` count *ambient* coordinates (their spatial widths
      are ``qk_rope_head_dim - 1`` and ``v_head_dim - 1``). A query/key head is therefore a point
      with ``qk_nope_head_dim + qk_rope_head_dim`` ambient coordinates and a value head one with
      ``v_head_dim``. ``dim`` is ambient.
    - **Curvature**: HELM's manifold parameter ``k`` has ``<x, x>_L = -k``; hyperbolix's ``c`` has
      ``<x, x>_L = -1/c``, so ``c = 1/k`` and HELM's score ``2k + 2<q, k>_L`` is
      ``2/c + 2<q, k>_L`` here.

    Float32 score floor
    -------------------
    The score is formed from GEMMs of the ambient coordinates, so it inherits the floor of
    :class:`HyperbolicFullAttention` (docs: ``numerical-stability.md#attention-score-floor``):
    ``<q, k>_L = -q_0 k_0 + <q_s, k_s>`` subtracts two terms of size
    ``cosh(a_q) cosh(a_k) / c`` (``a = sqrt(c) d`` the scaled radius) to reach an O(1)
    difference, and the absolute score error is about ``2 eps cosh(a_q) cosh(a_k) / (c tau)``.
    In float32 (``c = 1``, ``tau = 1``) it reaches O(1) near ``a ~ 8``. A bfloat16 input runs
    these GEMMs in bfloat16, whose ``eps`` is ``2^16`` times float32's; since the error grows as
    ``e^(2a)``, that moves the same error down by ``ln(2^16) / 2 ~ 5.5`` nats of radius. The
    weights stay finite either way, just wrong. Keep the radius down, or run the layer in
    float32/float64.

    Parameters
    ----------
    dim : int
        Model ambient dimension ``A`` (input and output points have ``dim`` coordinates).
    num_heads : int
        Number of attention heads ``H``.
    kv_lora_rank : int
        Spatial width ``R`` of the key/value latent.
    qk_nope_head_dim : int
        Spatial width ``N`` of the non-rotated query/key slice per head (may be 0).
    qk_rope_head_dim : int
        Ambient width of the rotated slice; its spatial width ``qk_rope_head_dim - 1`` must be
        even and positive.
    v_head_dim : int
        Ambient width ``W`` of a value head (``>= 2``).
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    rope_base : float, optional
        HOPE frequency base (default: 10000.0, HELM's).
    rms_epsilon : float, optional
        Epsilon of the latent RMSNorm (default: 1e-8, HELM's ``LorentzRMSNorm``).
    init_tau : float or None, optional
        Initial temperature ``tau``. ``None`` (the default) is HELM's
        ``sqrt(num_heads * (qk_nope_head_dim + qk_rope_head_dim))``; see "Differences from the
        HELM code" for the per-head alternative.
    init_bound : float or None, optional
        Uniform init bound shared by all four projections. ``None`` (the default) gives each
        projection HELM's reference init, Xavier-uniform with gain ``sqrt(2)``:
        ``sqrt(12 / (in_features + out_features))`` with ambient ``in_features`` and spatial
        ``out_features``. All biases start at 0.
    param_dtype : DTypeLike, optional
        Storage dtype of the parameters (default: jnp.float32). Compute follows the input dtype.
    eps : float, optional
        Numerical floor for the time rebuilds and the centroid (default: 1e-7).

    Attributes
    ----------
    wq : HTCLinear
        ``dim -> num_heads * (N + P)`` query projection.
    wkv_a : HTCLinear
        ``dim -> R + P`` latent and shared rotated-key projection.
    kv_norm : HRCRMSNorm
        RMSNorm of the ``R`` latent coordinates.
    wkv_b : HTCLinear
        ``R + 1 -> num_heads * (N + V)`` key/value decompression.
    wo : HTCLinear
        ``num_heads * v_head_dim -> dim - 1`` output projection.
    softmax_scale : nnx.Param
        Scalar temperature ``tau`` (HELM's name), init ``init_tau``.

    References
    ----------
    He et al., "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts", 2025
    (arXiv:2505.24722).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        *,
        rngs: nnx.Rngs,
        rope_base: float = 10000.0,
        rms_epsilon: float = 1e-8,
        init_tau: float | None = None,
        init_bound: float | None = None,
        param_dtype: DTypeLike = jnp.float32,
        eps: float = 1e-7,
    ):
        if dim < 2:
            raise ValueError(f"dim is the ambient model width and must be >= 2, got {dim}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if kv_lora_rank < 1:
            raise ValueError(f"kv_lora_rank must be >= 1, got {kv_lora_rank}")
        if qk_nope_head_dim < 0:
            raise ValueError(f"qk_nope_head_dim (spatial) must be >= 0, got {qk_nope_head_dim}")
        rope_spatial = qk_rope_head_dim - 1
        if rope_spatial < 2 or rope_spatial % 2 != 0:
            raise ValueError(
                "qk_rope_head_dim is ambient: its spatial width qk_rope_head_dim - 1 must be even and >= 2 "
                f"(HOPE rotates coordinate pairs), got qk_rope_head_dim={qk_rope_head_dim}"
            )
        if v_head_dim < 2:
            raise ValueError(f"v_head_dim is ambient and must be >= 2, got {v_head_dim}")

        self.dim = dim
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # ambient L
        self.v_head_dim = v_head_dim
        self.rope_base = rope_base
        self.eps = eps

        def projection(in_features: int, out_features: int) -> HTCLinear:
            bound = init_bound if init_bound is not None else _helm_xavier_bound(in_features, out_features)
            return HTCLinear(
                in_features, out_features, rngs=rngs, use_bias=True, init_bound=bound, eps=eps, param_dtype=param_dtype
            )

        self.wq = projection(dim, num_heads * (self.qk_head_dim - 1))
        self.wkv_a = projection(dim, kv_lora_rank + rope_spatial)
        self.kv_norm = HRCRMSNorm(kv_lora_rank, rngs=rngs, epsilon=rms_epsilon, eps=eps, param_dtype=param_dtype)
        self.wkv_b = projection(kv_lora_rank + 1, num_heads * (qk_nope_head_dim + v_head_dim - 1))
        self.wo = projection(num_heads * v_head_dim, dim - 1)
        tau = init_tau if init_tau is not None else math.sqrt(num_heads * self.qk_head_dim)
        self.softmax_scale = nnx.Param(jnp.asarray(tau, dtype=param_dtype))

    def _valid_mask(
        self,
        batch: int,
        seq_len: int,
        causal: bool,
        segment_ids: Int[Array, "B S"] | None,
        attention_mask: Bool[Array, "B S"] | None,
    ) -> Bool[Array, "B S T"] | None:
        """``True`` where query ``s`` may attend to key ``t``; ``None`` when nothing is masked."""
        valid_BST = None

        def combine(current, new):
            return new if current is None else current & new

        if causal:
            causal_ST = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))  # (S, T)
            valid_BST = combine(valid_BST, causal_ST[None])
        if segment_ids is not None:
            if segment_ids.shape != (batch, seq_len):
                raise ValueError(f"segment_ids must have shape (B, S) = {(batch, seq_len)}, got {segment_ids.shape}")
            same_segment_BST = segment_ids[:, :, None] == segment_ids[:, None, :]  # (B, S, T)
            valid_BST = combine(valid_BST, same_segment_BST)
        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError(f"attention_mask must have shape (B, S) = {(batch, seq_len)}, got {attention_mask.shape}")
            valid_key_B1T = attention_mask.astype(jnp.bool_)[:, None, :]  # (B, 1, T)
            valid_BST = combine(valid_BST, valid_key_B1T)
        if valid_BST is None:
            return None
        return jnp.broadcast_to(valid_BST, (batch, seq_len, seq_len))

    def __call__(
        self,
        x_BSA: Float[Array, "B S A"],
        c: float = 1.0,
        *,
        positions: Int[Array, "S"] | Int[Array, "B S"] | None = None,
        causal: bool = True,
        segment_ids: Int[Array, "B S"] | None = None,
        attention_mask: Bool[Array, "B S"] | None = None,
    ) -> Float[Array, "B S A"]:
        """Attend over a sequence of hyperboloid points.

        Parameters
        ----------
        x_BSA : Array, shape (B, S, A)
            Input points on the hyperboloid with curvature ``c``.
        c : float, optional
            Curvature (``<x, x>_L = -1/c``); input, attention and output all use it
            (default: 1.0).
        positions : int Array of shape (S,) or (B, S), optional
            HOPE positions of the tokens; ``None`` uses ``arange(S)`` (default: None).
        causal : bool, optional
            If True, query ``s`` attends only to keys ``t <= s`` (default: True).
        segment_ids : int Array of shape (B, S), optional
            Document ids of packed sequences; a query attends only to keys with the same id
            (HELM's block-diagonal document mask) (default: None).
        attention_mask : bool Array of shape (B, S), optional
            ``True`` marks a valid key; ``False`` keys (e.g. padding) receive no weight
            (default: None).

        Returns
        -------
        Array, shape (B, S, A)
            Output points on the hyperboloid with curvature ``c``.

        Notes
        -----
        The masks are combined with a logical AND and applied as HELM does, by filling the
        masked scores with ``-1e18``: a query row with no valid key gets a uniform (finite)
        softmax over all keys rather than NaN.
        """
        if x_BSA.ndim != 3 or x_BSA.shape[-1] != self.dim:
            raise ValueError(f"x_BSA must have shape (B, S, {self.dim}), got {x_BSA.shape}")
        B, S, _A = x_BSA.shape
        H = self.num_heads
        N = self.qk_nope_head_dim
        P = self.qk_rope_head_dim - 1
        R = self.kv_lora_rank
        V = self.v_head_dim - 1
        eps = self.eps
        work_dtype = x_BSA.dtype

        if positions is None:
            positions = jnp.arange(S, dtype=jnp.int32)  # (S,)
        if positions.shape not in ((S,), (B, S)):
            raise ValueError(f"positions must have shape (S,) = {(S,)} or (B, S) = {(B, S)}, got {positions.shape}")

        # 1. Queries: one GEMM over all heads, then [nope | rope] per head.
        query_space_BSX = self.wq(x_BSA, c, c, return_space=True)  # (B, S, H*K)
        query_space_BSHK = query_space_BSX.reshape(B, S, H, N + P)  # (B, S, H, K)
        query_nope_BSHN = query_space_BSHK[..., :N]  # (B, S, H, N)
        # Heads sit after the sequence axis: positions (S,) -> (S, 1), (B, S) -> (B, S, 1).
        query_rope_BSHP = hope_rotate_space(query_space_BSHK[..., N:], positions[..., None], self.rope_base)  # (B, S, H, P)

        # 2. Latent and the shared rotated key: one GEMM, split [latent | rope].
        latent_rope_BSY = self.wkv_a(x_BSA, c, c, return_space=True)  # (B, S, R+P)
        latent_BSR = latent_rope_BSY[..., :R]  # (B, S, R)
        key_rope_BSP = hope_rotate_space(latent_rope_BSY[..., R:], positions, self.rope_base)  # (B, S, P), one per token

        # 3. Normalize the latent into a point, then decompress keys and values for all heads.
        # nnx.RMSNorm promotes a low-precision input to its float32 scale; cast back so a
        # bfloat16 forward stays bfloat16 (HELM's F.rms_norm keeps the input dtype).
        latent_point_BSZ = self.kv_norm(latent_BSR, c, c, space_only=True).astype(work_dtype)  # (B, S, R+1)
        key_value_BSX = self.wkv_b(latent_point_BSZ, c, c, return_space=True)  # (B, S, H*(N+V))
        key_value_BSHX = key_value_BSX.reshape(B, S, H, N + V)  # (B, S, H, N+V)
        key_nope_BSHN = key_value_BSHX[..., :N]  # (B, S, H, N)
        value_space_BSHV = key_value_BSHX[..., N:]  # (B, S, H, V)

        # 4. Lorentz concatenation: concatenate the spatial parts, rebuild the time coordinate.
        query_BSHL = spatial_to_hyperboloid(
            jnp.concatenate([query_nope_BSHN, query_rope_BSHP], axis=-1), c, c, eps
        )  # (B, S, H, L)
        key_rope_BSHP = jnp.broadcast_to(key_rope_BSP[:, :, None, :], (B, S, H, P))  # shared by all heads
        key_BTHL = spatial_to_hyperboloid(jnp.concatenate([key_nope_BSHN, key_rope_BSHP], axis=-1), c, c, eps)
        value_BTHW = spatial_to_hyperboloid(value_space_BSHV, c, c, eps)  # (B, T, H, W)

        # 5. Scores: negative squared Lorentzian distance 2/c + 2<q, k>_L.
        # The two GEMMs cancel at large radius (class docstring, "Float32 score floor"), so they are
        # pinned HIGHEST as in HyperbolicFullAttention; the projection GEMMs above follow JAX's default.
        lorentz_inner_BHST = -jnp.einsum(
            "bshl,bthl->bhst", query_BSHL[..., :1], key_BTHL[..., :1], precision=MATMUL_PRECISION
        ) + jnp.einsum("bshk,bthk->bhst", query_BSHL[..., 1:], key_BTHL[..., 1:], precision=MATMUL_PRECISION)  # (B, H, S, T)
        # Scale, bias, mask and softmax run in at least float32 (HELM: softmax(dtype=float32)); a
        # float64 forward keeps float64.
        softmax_dtype = jnp.promote_types(work_dtype, jnp.float32)
        lorentz_inner_BHST = lorentz_inner_BHST.astype(softmax_dtype)
        inv_c = jnp.asarray(1.0, dtype=softmax_dtype) / jnp.asarray(c, dtype=softmax_dtype)
        softmax_scale = self.softmax_scale[...].astype(softmax_dtype)
        # No score bias: HELM's `+ self.bias` is one scalar for every key and cancels in the softmax.
        scores_BHST = (2.0 * inv_c + 2.0 * lorentz_inner_BHST) / softmax_scale  # (B, H, S, T)
        valid_BST = self._valid_mask(B, S, causal, segment_ids, attention_mask)
        if valid_BST is not None:
            scores_BHST = jnp.where(valid_BST[:, None], scores_BHST, jnp.asarray(_MASK_FILL, dtype=softmax_dtype))
        weights_BHST = jax.nn.softmax(scores_BHST, axis=-1).astype(value_BTHW.dtype)  # (B, H, S, T)

        # 6. Per-head weighted Lorentzian centroid of the values.
        value_BHTW = jnp.transpose(value_BTHW, (0, 2, 1, 3))  # (B, H, T, W)
        centroid_BHSW = lorentz_midpoint(value_BHTW, weights_BHST, c, eps)  # (B, H, S, W)

        # 7. Flatten the per-head ambient points (time included) and project to the output point.
        heads_BSF = jnp.transpose(centroid_BHSW, (0, 2, 1, 3)).reshape(B, S, H * self.v_head_dim)  # (B, S, H*W)
        return self.wo(heads_BSF, c, c)  # (B, S, A)
