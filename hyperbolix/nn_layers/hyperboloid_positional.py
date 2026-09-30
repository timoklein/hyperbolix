"""Positional encoding layers for hyperboloid manifolds.

This module provides positional encoding layers for hyperbolic Transformers:

- **HypformerPositionalEncoding**: Learnable relative positional encoding from
  Hypformer, combining HTCLinear with a Lorentzian residual connection.
- **hope** / **HyperbolicRoPE**: Hyperbolic Rotary Positional Encoding (HOPE)
  from HELM, a deterministic rotation-based encoding that preserves manifold
  structure and relative position information. **hope_rotate_space** applies
  the same rotation to a spatial tensor without rebuilding the time coordinate
  (the decoupled RoPE slice of multi-head latent attention).

References
----------
He et al., "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts", 2025 (arXiv:2505.24722).
Yang et al., "Hypformer: Exploring Efficient Transformer Fully in
Hyperbolic Space", 2025.
"""

import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from .hyperboloid_core import lorentz_residual, spatial_to_hyperboloid
from .hyperboloid_linear import HTCLinear


class HypformerPositionalEncoding(nnx.Module):
    """Relative positional encoding from Hypformer.

    Computes a position vector via HTCLinear, then combines it with the input
    using a Lorentzian residual connection:

        p = HTCLinear(x)
        result = lorentz_residual(x, p, w_y=epsilon, c=c)

    where epsilon is a FIXED scalar weight on the position contribution. The
    Hypformer reference keeps it a plain (non-learnable) tensor fixed at 1.0;
    making it a trainable parameter is unsafe because gradient descent can
    drive it below -1, where ``x + epsilon * p`` leaves the upper hyperboloid
    sheet and the ``abs()`` in the residual normalizer silently masks the
    violation (see :func:`~hyperbolix.nn_layers.hyperboloid_core.lorentz_residual`).

    Parameters
    ----------
    in_features : int
        Input ambient dimension (d+1, including time component).
    out_features : int
        Output spatial dimension (d). The HTCLinear output will have ambient
        dimension d+1 (= out_features + 1), matching the input.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    epsilon : float, optional
        Fixed scalar weight for the position encoding contribution
        (default: 1.0, matching the Hypformer reference). Must be >= 0 so the
        Lorentzian residual stays on the upper hyperboloid sheet.
    init_bound : float or None, optional
        Bound for HTCLinear uniform weight initialization; ``None`` resolves to
        HTCLinear's fan-in-aware default ``sqrt(3 / in_features)`` (default: None).
    eps : float, optional
        Numerical stability floor for lorentz_residual (default: 1e-7).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters, forwarded to the internal
        ``HTCLinear`` (default: jnp.float32). Compute precision follows the input
        array's dtype.

    Attributes
    ----------
    htc_linear : HTCLinear
        Linear transformation producing the position encoding vector.
    epsilon : float
        Fixed scalar weight for the position encoding contribution.
    eps : float
        Numerical stability parameter.

    References
    ----------
    He et al., "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts", 2025 (arXiv:2505.24722).
    Yang et al., "Hypformer: Exploring Efficient Transformer Fully in
    Hyperbolic Space", 2025.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        epsilon: float = 1.0,
        init_bound: float | None = None,
        eps: float = 1e-7,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if epsilon < 0:
            raise ValueError(
                f"epsilon must be >= 0 (got {epsilon}): a negative weight can push the "
                "Lorentzian residual off the upper hyperboloid sheet."
            )
        self.htc_linear = HTCLinear(in_features, out_features, rngs=rngs, init_bound=init_bound, param_dtype=param_dtype)
        self.epsilon = epsilon
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "... dim_plus_1"],
        c: float = 1.0,
    ) -> Float[Array, "... dim_plus_1"]:
        """Apply learnable positional encoding.

        Parameters
        ----------
        x : Array, shape (..., d+1)
            Points on hyperboloid with curvature c.
        c : float, optional
            Curvature parameter (default: 1.0).

        Returns
        -------
        Array, shape (..., d+1)
            Positionally-encoded points on hyperboloid with curvature c.
        """
        p = self.htc_linear(x, c_in=c, c_out=c)  # (..., d+1)
        return lorentz_residual(x, p, w_y=self.epsilon, c=c, eps=self.eps)


def _apply_rotary_interleaved(
    x: Float[Array, "... d"],
    cos_vals: Float[Array, "... half_d"],
    sin_vals: Float[Array, "... half_d"],
) -> Float[Array, "... d"]:
    """Apply 2D rotation to interleaved pairs of dimensions.

    Pairs spatial dimensions as (x_0, x_1), (x_2, x_3), ... and applies a
    2D rotation to each pair using the provided cos/sin values.

    Parameters
    ----------
    x : Array, shape (..., d)
        Spatial components (d must be even).
    cos_vals : Array, shape (..., d//2)
        Cosine of rotation angles for each pair.
    sin_vals : Array, shape (..., d//2)
        Sine of rotation angles for each pair.

    Returns
    -------
    Array, shape (..., d)
        Rotated spatial components.
    """
    x_pairs_F2 = x.reshape(*x.shape[:-1], -1, 2)  # (..., F, 2) where F = d//2
    x1_F = x_pairs_F2[..., 0]  # (..., F)
    x2_F = x_pairs_F2[..., 1]  # (..., F)
    y1_F = x1_F * cos_vals - x2_F * sin_vals  # 2D rotation per pair
    y2_F = x1_F * sin_vals + x2_F * cos_vals
    return jnp.stack([y1_F, y2_F], axis=-1).reshape(x.shape)  # (..., d)


def hope_rotate_space(
    x_space: Float[Array, "... seq d"],
    positions: Float[Array, "..."],
    base: float = 10000.0,
) -> Float[Array, "... seq d"]:
    """Rotate a spatial tensor with the HOPE / RoPE interleaved-pair rotation.

    The spatial-only half of :func:`hope`: it rotates adjacent pairs
    ``(x_0, x_1), (x_2, x_3), ...`` of the last axis by the angles
    ``positions * theta_i`` with ``theta_i = 1 / base^(2i/d)``, and does not
    rebuild a time coordinate. This is HELM's ``apply_rotary_emb`` (a complex
    multiply over adjacent pairs), used by multi-head latent attention's
    decoupled RoPE, which rotates only a slice of the space part of the
    queries and keys and assembles the full point afterwards.

    Parameters
    ----------
    x_space : Array, shape (..., seq_len, d)
        Spatial components (``d`` must be even). Any leading axes.
    positions : Array
        Integer position indices, broadcastable against ``x_space.shape[:-1]``
        after the angle axis is appended (``positions[..., None] * theta``):
        ``(seq_len,)`` applies one position sequence to every leading axis,
        ``(batch, seq_len)`` gives each batch row its own positions for an
        ``x_space`` of shape ``(batch, seq_len, d)``. Axes are aligned from the
        right, so other layouts need singleton axes: heads before the sequence,
        ``(batch, heads, seq_len, d)``, take ``(batch, 1, seq_len)``; heads after
        it, ``(batch, seq_len, heads, d)`` as in HELM, take ``positions[:, None]``
        of shape ``(seq_len, 1)``.
    base : float, optional
        Frequency base for rotation angles (default: 10000.0).

    Returns
    -------
    Array, same shape and dtype as ``x_space``
        Rotated spatial components; every pair keeps its Euclidean norm.

    Notes
    -----
    The rotation is computed in ``promote_types(x_space.dtype, float32)`` and cast
    back to the input dtype: bfloat16/float16 inputs are rotated in float32, as
    HELM does (``x.float()`` ... ``.to(dtype)``), float32 stays float32 and float64
    stays float64.

    References
    ----------
    He et al., "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts", 2025 (arXiv:2505.24722).
    """
    d = x_space.shape[-1]
    in_dtype = x_space.dtype
    dtype = jnp.promote_types(in_dtype, jnp.float32)  # f32 for bf16/f16, unchanged for f32/f64
    x_SD = x_space.astype(dtype)  # (..., S, D); no-op for f32/f64

    # Frequency schedule: theta_i = 1 / base^(2i/d). Build the index grid in the
    # compute dtype: bare jnp.arange is int64 under global jax_enable_x64, and the
    # subsequent division would promote the whole encoding to float64.
    freqs_F = 1.0 / (base ** (jnp.arange(0, d, 2, dtype=dtype) / d))  # (F,) where F = d//2
    angles_SF = positions[..., None].astype(dtype) * freqs_F  # (..., S, F), positions' shape + (F,)
    cos_SF = jnp.cos(angles_SF)
    sin_SF = jnp.sin(angles_SF)

    # Rotate interleaved pairs; cos/sin broadcast against the (..., S, F) pair grid.
    rotated_SD = _apply_rotary_interleaved(x_SD, cos_SF, sin_SF)  # (..., S, D)
    return rotated_SD.astype(in_dtype)


def hope(
    z: Float[Array, "... seq d_plus_1"],
    positions: Float[Array, "..."],
    c: float = 1.0,
    base: float = 10000.0,
    eps: float = 1e-7,
) -> Float[Array, "... seq d_plus_1"]:
    """Hyperbolic Rotary Positional Encoding (HOPE).

    Applies RoPE-style rotation to the spatial components of hyperboloid
    points, then reconstructs the time component to satisfy the manifold
    constraint. Equivalent to ``hrc(z, R_{i,Theta}, c, c)`` where R is a
    block-diagonal rotation matrix.

    Since rotation preserves norms, the Minkowski inner product between
    encoded points depends only on the *relative* position offset, giving
    the standard RoPE relative-position property on the hyperboloid.

    Parameters
    ----------
    z : Array, shape (..., seq_len, d+1)
        Points on hyperboloid (d must be even).
    positions : Array
        Integer position indices, broadcastable against ``z.shape[:-1]``:
        ``(seq_len,)`` for one position sequence shared by all leading axes,
        or e.g. ``(batch, seq_len)`` for per-row positions with ``z`` of shape
        ``(batch, seq_len, d+1)``. Axes are aligned from the right, so axes between
        batch and sequence need singleton axes (``(batch, 1, seq_len)`` for
        ``(batch, heads, seq_len, d+1)``). See :func:`hope_rotate_space`.
    c : float, optional
        Curvature parameter (default: 1.0).
    base : float, optional
        Frequency base for rotation angles (default: 10000.0).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    Returns
    -------
    Array, shape (..., seq_len, d+1)
        Rotated points on hyperboloid with curvature c.

    References
    ----------
    He et al., "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts", 2025 (arXiv:2505.24722).
    """
    spatial_SD = z[..., 1:]  # (..., S, D) where S=seq, D=spatial dim

    # Rotate spatial components (interleaved pairs)
    rotated_SD = hope_rotate_space(spatial_SD, positions, base)  # (..., S, D)

    # Reconstruct time via the hyperboloid constraint (scale = 1 since c_in == c_out),
    # which is exactly the hrc(z, R, c, c) tail this function is documented to equal.
    return spatial_to_hyperboloid(rotated_SD, c_in=c, c_out=c, eps=eps)  # (..., S, A)


class HyperbolicRoPE(nnx.Module):
    """NNX module wrapper for HOPE (Hyperbolic Rotary Positional Encoding).

    This is a stateless module (no learnable parameters) that wraps the
    functional :func:`hope` for convenient use in NNX model definitions.

    Parameters
    ----------
    dim : int
        Spatial dimension d (must be even).
    max_seq_len : int, optional
        Maximum sequence length (for documentation; not enforced, default: 2048).
    base : float, optional
        Frequency base for rotation angles (default: 10000.0).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    References
    ----------
    He et al., "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts", 2025 (arXiv:2505.24722).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        eps: float = 1e-7,
    ):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.eps = eps

    def __call__(
        self,
        z: Float[Array, "... seq d_plus_1"],
        positions: Float[Array, "..."],
        c: float = 1.0,
    ) -> Float[Array, "... seq d_plus_1"]:
        """Apply HOPE positional encoding.

        Parameters
        ----------
        z : Array, shape (..., seq_len, d+1)
            Points on hyperboloid (spatial dim must equal self.dim, must be even).
        positions : Array
            Integer position indices, broadcastable against ``z.shape[:-1]``
            (``(seq_len,)`` or e.g. ``(batch, seq_len)``; see :func:`hope`).
        c : float, optional
            Curvature parameter (default: 1.0).

        Returns
        -------
        Array, shape (..., seq_len, d+1)
            Rotated points on hyperboloid.
        """
        return hope(z, positions, c, self.base, self.eps)
