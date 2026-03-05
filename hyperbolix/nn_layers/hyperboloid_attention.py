"""Hyperbolic attention layers for the hyperboloid manifold.

Implements three attention variants from the Hypformer paper (Yang et al. 2025,
Section 4.3), cross-referenced against the official PyTorch implementation:

- **HyperbolicLinearAttention** — O(N) linear attention with focus function
- **HyperbolicSoftmaxAttention** — O(N²) spatial softmax attention
- **HyperbolicFullAttention** — O(N²) Lorentzian similarity + midpoint aggregation

Dimension key
-------------
B : batch size
N : sequence length
H : number of attention heads
D : spatial dimension per head (``out_features``)
A : ambient dimension (``D + 1``)

References
----------
Yang et al., "Hypformer: Exploring Efficient Transformer Fully in
Hyperbolic Space", 2025.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from .hyperboloid_core import lorentz_midpoint, spatial_to_hyperboloid
from .hyperboloid_linear import HTCLinear

# ---------------------------------------------------------------------------
# Focus transform (Eq 19)
# ---------------------------------------------------------------------------


def focus_transform(
    x_D: Float[Array, "... D"],
    temperature: Float[Array, ""],
    power: float,
    eps: float = 1e-7,
) -> Float[Array, "... D"]:
    """Norm-preserving focus function (Eq 19, Hypformer).

    Applies temperature-scaled ReLU followed by element-wise power sharpening
    while preserving the original norm.

    Parameters
    ----------
    x_D : Array, shape (..., D)
        Input spatial features.
    temperature : scalar Array
        Learnable temperature parameter.
    power : float
        Sharpening exponent (``p > 1`` concentrates mass).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    Returns
    -------
    Array, shape (..., D)
        Focus-transformed features with ``||output|| ≈ ||relu(x)/|t|||``.
    """
    # Temperature-scaled ReLU
    scaled_relu_D = (jax.nn.relu(x_D) + eps) / (jnp.abs(temperature) + eps)  # (..., D)

    # Element-wise power sharpening
    sharpened_D = scaled_relu_D**power  # (..., D)

    # Norm-preserving rescaling
    norm_scaled = jnp.sqrt(jnp.sum(scaled_relu_D**2, axis=-1, keepdims=True) + eps)  # (..., 1)
    norm_sharpened = jnp.sqrt(jnp.sum(sharpened_D**2, axis=-1, keepdims=True) + eps)  # (..., 1)

    return (norm_scaled / norm_sharpened) * sharpened_D  # (..., D)


# ---------------------------------------------------------------------------
# Base class — shared query/key/value projection and multi-head management
# ---------------------------------------------------------------------------


class _HyperbolicAttentionBase(nnx.Module):
    """Shared base for all hyperbolic attention variants.

    Handles per-head query/key/value projection via :class:`HTCLinear` and
    provides the ``__call__`` → ``_attend`` dispatch pattern.

    Parameters
    ----------
    in_features : int
        Ambient input dimension (``d_in + 1``).
    out_features : int
        Spatial output dimension per head (``d_out``).
    num_heads : int
        Number of attention heads (default: 1).
    init_bound : float
        Uniform init bound for HTCLinear weights (default: 0.02).
    eps : float
        Numerical stability floor (default: 1e-7).
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_heads: int = 1,
        init_bound: float = 0.02,
        eps: float = 1e-7,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.out_features = out_features
        self.eps = eps

        self.query_projections = nnx.List(
            [HTCLinear(in_features, out_features, rngs=rngs, init_bound=init_bound, eps=eps) for _ in range(num_heads)]
        )
        self.key_projections = nnx.List(
            [HTCLinear(in_features, out_features, rngs=rngs, init_bound=init_bound, eps=eps) for _ in range(num_heads)]
        )
        self.value_projections = nnx.List(
            [HTCLinear(in_features, out_features, rngs=rngs, init_bound=init_bound, eps=eps) for _ in range(num_heads)]
        )

    def _project_qkv(
        self,
        x_BNA: Float[Array, "B N A_in"],
        c_in: float,
        c_attn: float,
    ) -> tuple[Float[Array, "B N H A"], Float[Array, "B N H A"], Float[Array, "B N H A"]]:
        """Project input to query, key, value and stack across heads."""
        query_per_head = [self.query_projections[h](x_BNA, c_in, c_attn) for h in range(self.num_heads)]
        key_per_head = [self.key_projections[h](x_BNA, c_in, c_attn) for h in range(self.num_heads)]
        value_per_head = [self.value_projections[h](x_BNA, c_in, c_attn) for h in range(self.num_heads)]
        return (
            jnp.stack(query_per_head, axis=2),  # (B, N, H, A)
            jnp.stack(key_per_head, axis=2),
            jnp.stack(value_per_head, axis=2),
        )

    def __call__(
        self,
        x: Float[Array, "B N A_in"],
        c_in: float = 1.0,
        c_attn: float = 1.0,
        c_out: float = 1.0,
    ) -> Float[Array, "B N A_out"]:
        """Forward pass.

        Parameters
        ----------
        x : Array, shape (B, N, A_in)
            Input points on the hyperboloid with curvature ``c_in``.
        c_in : float
            Input curvature.
        c_attn : float
            Intermediate attention curvature.
        c_out : float
            Output curvature.

        Returns
        -------
        Array, shape (B, N, out_features + 1)
            Output on the hyperboloid with curvature ``c_out``.
        """
        query_BNHA, key_BNHA, value_BNHA = self._project_qkv(x, c_in, c_attn)
        return self._attend(query_BNHA, key_BNHA, value_BNHA, c_attn, c_out)

    def _attend(
        self,
        query_BNHA: Float[Array, "B N H A"],
        key_BNHA: Float[Array, "B N H A"],
        value_BNHA: Float[Array, "B N H A"],
        c_attn: float,
        c_out: float,
    ) -> Float[Array, "B N A_out"]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Linear attention (Eq 14-19) — O(N)
# ---------------------------------------------------------------------------


class HyperbolicLinearAttention(_HyperbolicAttentionBase):
    """Hyperbolic linear attention with focus function (Eq 14-19).

    The paper's main contribution: O(N) attention using the kernel trick in the
    spatial domain of the hyperboloid.  Focus function φ sharpens query and key.

    Parameters
    ----------
    in_features : int
        Ambient input dimension (``d_in + 1``).
    out_features : int
        Spatial output dimension per head.
    num_heads : int
        Number of attention heads (default: 1).
    power : float
        Focus function sharpening exponent (default: 2.0).
    init_bound : float
        Uniform init bound for weights (default: 0.02).
    eps : float
        Numerical stability floor (default: 1e-7).
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_heads: int = 1,
        power: float = 2.0,
        init_bound: float = 0.02,
        eps: float = 1e-7,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            in_features,
            out_features,
            num_heads=num_heads,
            init_bound=init_bound,
            eps=eps,
            rngs=rngs,
        )
        self.power = power
        self.temperature = nnx.Param(jnp.array(1.0))
        # Spatial residual projection ψ: D → D (shared across heads)
        self.residual_proj = nnx.Linear(out_features, out_features, rngs=rngs)

    def _attend(self, query_BNHA, key_BNHA, value_BNHA, c_attn, c_out):
        eps = self.eps

        # 1. Strip spatial, apply focus to query and key (not value)
        query_spatial_BNHD = query_BNHA[..., 1:]  # (B, N, H, D)
        key_spatial_BNHD = key_BNHA[..., 1:]
        value_spatial_BNHD = value_BNHA[..., 1:]

        focused_query_BNHD = focus_transform(query_spatial_BNHD, self.temperature[...], self.power, eps)
        focused_key_BNHD = focus_transform(key_spatial_BNHD, self.temperature[...], self.power, eps)

        # 2. Linear attention via kernel trick: φ(Q)(φ(K)^T V) / φ(Q)(φ(K)^T 1)
        key_value_product_BHDE = jnp.einsum("bnhd,bnhe->bhde", focused_key_BNHD, value_spatial_BNHD)  # (B, H, D, D)
        numerator_BNHD = jnp.einsum("bnhd,bhde->bnhe", focused_query_BNHD, key_value_product_BHDE)  # (B, N, H, D)

        key_sum_BHD = focused_key_BNHD.sum(axis=1)  # (B, H, D)
        denominator_BNH1 = jnp.einsum("bnhd,bhd->bnh", focused_query_BNHD, key_sum_BHD)[..., None]  # (B, N, H, 1)

        output_spatial_BNHD = numerator_BNHD / (denominator_BNH1 + eps)  # (B, N, H, D)

        # 3. Spatial residual: Z̃_s = Z_s + ψ(V_s)
        output_spatial_BNHD = output_spatial_BNHD + self.residual_proj(value_spatial_BNHD)

        # 4. Average heads + time calibration (Eq 18)
        output_spatial_BND = output_spatial_BNHD.mean(axis=2)  # (B, N, D)

        return spatial_to_hyperboloid(output_spatial_BND, c_attn, c_out, eps)  # (B, N, D+1)


# ---------------------------------------------------------------------------
# Softmax attention — O(N²), spatial-domain
# ---------------------------------------------------------------------------


class HyperbolicSoftmaxAttention(_HyperbolicAttentionBase):
    """Hyperbolic softmax attention in the spatial domain.

    Standard scaled dot-product attention applied to spatial components of
    query, key, value, followed by the same HRC pipeline (residual + time
    calibration) as the linear variant.

    Parameters
    ----------
    in_features : int
        Ambient input dimension (``d_in + 1``).
    out_features : int
        Spatial output dimension per head.
    num_heads : int
        Number of attention heads (default: 1).
    init_bound : float
        Uniform init bound for weights (default: 0.02).
    eps : float
        Numerical stability floor (default: 1e-7).
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_heads: int = 1,
        init_bound: float = 0.02,
        eps: float = 1e-7,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            in_features,
            out_features,
            num_heads=num_heads,
            init_bound=init_bound,
            eps=eps,
            rngs=rngs,
        )
        # Spatial residual projection ψ: D → D (shared across heads)
        self.residual_proj = nnx.Linear(out_features, out_features, rngs=rngs)

    def _attend(self, query_BNHA, key_BNHA, value_BNHA, c_attn, c_out):
        eps = self.eps

        query_spatial_BNHD = query_BNHA[..., 1:]  # (B, N, H, D)
        key_spatial_BNHD = key_BNHA[..., 1:]
        value_spatial_BNHD = value_BNHA[..., 1:]

        head_dim = query_spatial_BNHD.shape[-1]

        # Scaled dot-product attention: softmax(Q_s K_s^T / √D) V_s
        scores_BNHM = jnp.einsum("bnhd,bmhd->bnhm", query_spatial_BNHD, key_spatial_BNHD) / jnp.sqrt(float(head_dim))
        attn_weights_BNHM = jax.nn.softmax(scores_BNHM, axis=-1)  # (B, N, H, M)
        output_spatial_BNHD = jnp.einsum("bnhm,bmhd->bnhd", attn_weights_BNHM, value_spatial_BNHD)  # (B, N, H, D)

        # Spatial residual
        output_spatial_BNHD = output_spatial_BNHD + self.residual_proj(value_spatial_BNHD)

        # Average heads + time calibration
        output_spatial_BND = output_spatial_BNHD.mean(axis=2)  # (B, N, D)

        return spatial_to_hyperboloid(output_spatial_BND, c_attn, c_out, eps)  # (B, N, D+1)


# ---------------------------------------------------------------------------
# Full Lorentzian attention — O(N²)
# ---------------------------------------------------------------------------


class HyperbolicFullAttention(_HyperbolicAttentionBase):
    """Full Lorentzian attention with midpoint aggregation.

    Uses the Lorentzian inner product for similarity and weighted Lorentzian
    midpoint for aggregation — operating on full hyperboloid points throughout.

    Parameters
    ----------
    in_features : int
        Ambient input dimension (``d_in + 1``).
    out_features : int
        Spatial output dimension per head.
    num_heads : int
        Number of attention heads (default: 1).
    init_bound : float
        Uniform init bound for weights (default: 0.02).
    eps : float
        Numerical stability floor (default: 1e-7).
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_heads: int = 1,
        init_bound: float = 0.02,
        eps: float = 1e-7,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            in_features,
            out_features,
            num_heads=num_heads,
            init_bound=init_bound,
            eps=eps,
            rngs=rngs,
        )
        self.scale = nnx.Param(jnp.array(1.0))
        self.attn_bias = nnx.Param(jnp.array(0.0))

    def _attend(self, query_BNHA, key_BNHA, value_BNHA, c_attn, c_out):
        eps = self.eps
        B, N, H, _A = value_BNHA.shape

        # 1. Pairwise Lorentzian similarity: <Q,K>_L = -Q_0 K_0 + Q_s · K_s
        lorentz_inner_BNHM = -jnp.einsum("bnha,bmha->bnhm", query_BNHA[..., 0:1], key_BNHA[..., 0:1]) + jnp.einsum(
            "bnhd,bmhd->bnhm", query_BNHA[..., 1:], key_BNHA[..., 1:]
        )  # (B, N, H, M)
        scores_BNHM = (2.0 + 2.0 * lorentz_inner_BNHM) / (self.scale[...] + eps) + self.attn_bias[...]
        attn_weights_BNHM = jax.nn.softmax(scores_BNHM, axis=-1)  # (B, N, H, M)

        # 2. Weighted Lorentzian midpoint per head
        #    Transpose to (B, H, ...) layout for lorentz_midpoint which expects (..., M, A) and (..., N, M)
        value_BHMA = jnp.transpose(value_BNHA, (0, 2, 1, 3))  # (B, H, M, A)
        attn_weights_BHNM = jnp.transpose(attn_weights_BNHM, (0, 2, 1, 3))  # (B, H, N, M)
        midpoint_BHNA = lorentz_midpoint(value_BHMA, attn_weights_BHNM, c_attn, eps)  # (B, H, N, A)
        midpoint_BNHA = jnp.transpose(midpoint_BHNA, (0, 2, 1, 3))  # (B, N, H, A)

        # 3. Average heads via Lorentzian midpoint (uniform weights)
        uniform_BN1H = jnp.ones((B, N, 1, H)) / H
        averaged_BN1A = lorentz_midpoint(midpoint_BNHA, uniform_BN1H, c_attn, eps)  # (B, N, 1, A)
        output_BNA = averaged_BN1A.squeeze(axis=2)  # (B, N, A)

        # 4. Curvature change via spatial rescaling + time reconstruction
        output_spatial_BND = output_BNA[..., 1:]  # (B, N, D)
        return spatial_to_hyperboloid(output_spatial_BND, c_attn, c_out, eps)  # (B, N, A)
