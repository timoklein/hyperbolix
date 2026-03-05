"""Positional encoding layers for hyperboloid manifolds.

This module provides positional encoding layers for hyperbolic Transformers:

- **HypformerPositionalEncoding**: Learnable relative positional encoding from
  Hypformer, combining HTCLinear with a Lorentzian residual connection.
- **hope** / **HyperbolicRoPE**: Hyperbolic Rotary Positional Encoding (HOPE)
  from HELM, a deterministic rotation-based encoding that preserves manifold
  structure and relative position information.

References
----------
Chen et al., "Hyperbolic Embeddings for Learning on Manifolds" (HELM), 2024.
Hypformer paper (citation to be added).
"""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from .hyperboloid_core import lorentz_residual
from .hyperboloid_linear import HTCLinear


class HypformerPositionalEncoding(nnx.Module):
    """Learnable relative positional encoding from Hypformer.

    Computes a position vector via HTCLinear, then combines it with the input
    using a Lorentzian residual connection:

        p = HTCLinear(x)
        result = lorentz_residual(x, p, w_y=epsilon, c=c)

    where epsilon is a learnable scalar magnitude parameter.

    Parameters
    ----------
    in_features : int
        Input ambient dimension (d+1, including time component).
    out_features : int
        Output spatial dimension (d). The HTCLinear output will have ambient
        dimension d+1 (= out_features + 1), matching the input.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    init_bound : float, optional
        Bound for HTCLinear uniform weight initialization (default: 0.02).
    eps : float, optional
        Numerical stability floor for lorentz_residual (default: 1e-7).

    Attributes
    ----------
    htc_linear : HTCLinear
        Linear transformation producing the position encoding vector.
    epsilon : nnx.Param
        Learnable scalar weight for the position encoding contribution.
    eps : float
        Numerical stability parameter.

    References
    ----------
    Chen et al., "Hyperbolic Embeddings for Learning on Manifolds" (HELM), 2024.
    Hypformer paper (citation to be added).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        init_bound: float = 0.02,
        eps: float = 1e-7,
    ):
        self.htc_linear = HTCLinear(in_features, out_features, rngs=rngs, init_bound=init_bound)
        self.epsilon = nnx.Param(jnp.array(1.0))
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
        return lorentz_residual(x, p, w_y=self.epsilon[...], c=c, eps=self.eps)


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


def hope(
    z: Float[Array, "... seq d_plus_1"],
    positions: Float[Array, "seq"],
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
    positions : Array, shape (seq_len,)
        Integer position indices.
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
    Chen et al., "Hyperbolic Embeddings for Learning on Manifolds" (HELM), 2024.
    """
    spatial_SD = z[..., 1:]  # (..., S, D) where S=seq, D=spatial dim
    d = spatial_SD.shape[-1]

    # Frequency schedule: theta_i = 1 / base^(2i/d)
    freqs_F = 1.0 / (base ** (jnp.arange(0, d, 2) / d))  # (F,) where F = d//2
    angles_SF = positions[:, None] * freqs_F[None, :]  # (S, F)
    cos_SF = jnp.cos(angles_SF)
    sin_SF = jnp.sin(angles_SF)

    # Rotate spatial components (interleaved pairs)
    rotated_SD = _apply_rotary_interleaved(spatial_SD, cos_SF, sin_SF)  # (..., S, D)

    # Reconstruct time: t = sqrt(||rotated||^2 + 1/c)
    norm_sq_S1 = jnp.sum(rotated_SD**2, axis=-1, keepdims=True)  # (..., S, 1)
    time_S1 = jnp.sqrt(jnp.maximum(norm_sq_S1 + 1.0 / c, eps))  # (..., S, 1)

    return jnp.concatenate([time_S1, rotated_SD], axis=-1)  # (..., S, A)


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
    Chen et al., "Hyperbolic Embeddings for Learning on Manifolds" (HELM), 2024.
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
        positions: Float[Array, "seq"],
        c: float = 1.0,
    ) -> Float[Array, "... seq d_plus_1"]:
        """Apply HOPE positional encoding.

        Parameters
        ----------
        z : Array, shape (..., seq_len, d+1)
            Points on hyperboloid (spatial dim must equal self.dim, must be even).
        positions : Array, shape (seq_len,)
            Integer position indices.
        c : float, optional
            Curvature parameter (default: 1.0).

        Returns
        -------
        Array, shape (..., seq_len, d+1)
            Rotated points on hyperboloid.
        """
        return hope(z, positions, c, self.base, self.eps)
