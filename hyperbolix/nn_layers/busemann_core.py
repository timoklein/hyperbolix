"""Shared core for Busemann neural-network layers (Chen et al. 2026).

These helpers are the functional core behind the four Busemann layers
(``HypRegression{Hyperboloid,Poincare}Busemann`` and
``HypLinear{Hyperboloid,Poincare}Busemann``). Keeping the parameterization, the
Busemann-logit computation, and the Poincaré output map here means the geometry
lives in one place and the per-model ``nnx.Module`` classes stay thin shells that
only document their channel convention — mirroring how ``hyperboloid_core`` backs
the hyperboloid layer family.

Dimension key:
  B: batch size      I: in_spatial (direction dim)
  K: n_out (num_classes for BMLR, out_spatial for BFC)
  Ai: in_ambient (Hyperboloid in_features = in_spatial + 1)

The BMLR logit (Chen et al. 2026, Eq. 8) is ``u_k(x) = -alpha_k·B^{v_k}(x) + b_k`` with
``alpha_k = exp(log_scale_k) > 0`` and ``v_k = kernel_k / ‖kernel_k‖`` a unit ideal direction
(the Salimans-Kingma weight-normalization split, matching the reference ``weight_v``/``weight_g``).
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds import Manifold

# Gradient-safe floor for the direction-normalization denominator: the same
# ``sqrt(sumsq + MIN_NORM**2)`` safe-norm idiom the manifolds use. Imported (not
# redefined) so there is one value library-wide.
from hyperbolix.utils.math_utils import MIN_NORM


def _init_weight_norm_params(
    rngs: nnx.Rngs,
    n_out: int,
    in_spatial: int,
    std: float,
    param_dtype: DTypeLike = jnp.float32,
) -> tuple[nnx.Param, nnx.Param, nnx.Param]:
    """Initialize the (direction, log-scale, bias) weight-normalization parameters.

    Shared by all four Busemann layers; they differ only in ``std`` and ``in_spatial``.
    The ``log_scale`` is seeded so the initial magnitude ``alpha = exp(log_scale) = ‖kernel_row‖``,
    exactly reproducing the reference ``weight_g = log(‖weight_v‖)`` init.

    Parameters
    ----------
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    n_out : int
        Number of output units (``num_classes`` for BMLR, ``out_spatial`` for BFC).
    in_spatial : int
        Direction dimension (Hyperboloid: ``in_dim - 1``; Poincaré: ``in_dim``).
    std : float
        Standard deviation of the normal kernel init.
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).

    Returns
    -------
    tuple[nnx.Param, nnx.Param, nnx.Param]
        ``(kernel, log_scale, bias)`` with shapes ``(n_out, in_spatial)``, ``(n_out,)``,
        ``(n_out,)``. Exposed as plain ``nnx.Param`` for research workflows (weight resets,
        param-norm logging).
    """
    kernel_KI = std * jax.random.normal(rngs.params(), (n_out, in_spatial), dtype=param_dtype)
    log_scale_K = jnp.log(jnp.maximum(jnp.linalg.norm(kernel_KI, axis=-1), MIN_NORM))
    bias_K = jnp.zeros((n_out,), dtype=param_dtype)
    return nnx.Param(kernel_KI), nnx.Param(log_scale_K), nnx.Param(bias_K)


def _busemann_score(
    manifold: Manifold,
    x_BI: Float[Array, "batch in_features"],
    kernel_KI: Float[Array, "n_out in_spatial"],
    log_scale_K: Float[Array, "n_out"],
    bias_K: Float[Array, "n_out"],
    c: float,
    input_space: str,
) -> Float[Array, "batch n_out"]:
    """Per-output Busemann logit ``u_k(x) = -alpha_k·B^{v_k}(x) + b_k`` (Chen et al. 2026, Eq. 8).

    Used directly by the BMLR heads (which return it as the classification logit) and by the
    BFC layers (which apply an activation, then a model-specific output map). The direction
    weights are row-normalized to the unit sphere here, so callers store unconstrained
    ``kernel`` rows.

    Parameters
    ----------
    manifold : Manifold
        Class-based Hyperboloid or Poincaré instance exposing ``busemann(x, v, c)``.
    x_BI : Array, shape (B, in_features)
        Input points (Hyperboloid: ambient ``d+1``; Poincaré: spatial ``d``). Assumed to be
        at ``manifold.dtype``; params are cast to ``x_BI.dtype`` for the arithmetic.
    kernel_KI : Array, shape (K, I)
        Unnormalized direction weights.
    log_scale_K : Array, shape (K,)
        Log-magnitudes; ``alpha = exp(log_scale) > 0``.
    bias_K : Array, shape (K,)
        Per-output bias.
    c : float
        Curvature (positive).
    input_space : str
        ``"manifold"`` (default usage) or ``"tangent"`` — when ``"tangent"``, the input is
        lifted to the manifold via ``expmap_0`` first.

    Returns
    -------
    Array, shape (B, K)
        The Busemann logits.
    """
    if input_space == "tangent":
        x_BI = jax.vmap(manifold.expmap_0, in_axes=(0, None), out_axes=0)(x_BI, c)

    work_dtype = x_BI.dtype
    kernel_KI = kernel_KI.astype(work_dtype)
    # Row-normalize directions to the unit sphere (gradient-safe at zero rows).
    norm_K1 = jnp.sqrt(jnp.sum(kernel_KI**2, axis=-1, keepdims=True) + MIN_NORM**2)
    v_unit_KI = kernel_KI / norm_K1
    alpha_K = jnp.exp(log_scale_K.astype(work_dtype))
    bias_K = bias_K.astype(work_dtype)

    # B^{v_k}(x_b): single-point manifold.busemann vmapped over classes (inner) and batch (outer).
    busemann_BK = jax.vmap(
        jax.vmap(manifold.busemann, in_axes=(None, 0, None)),
        in_axes=(0, None, None),
    )(x_BI, v_unit_KI, c)  # (B, K)

    return -alpha_K[None, :] * busemann_BK + bias_K[None, :]


def busemann_fc_poincare_output(
    u_BO: Float[Array, "batch out_spatial"],
    c: float,
    v_max: float,
) -> Float[Array, "batch out_spatial"]:
    """Poincaré BFC output map (Chen et al. 2026, Thm. 4.1).

    Maps the (activated) Busemann logits to a Poincaré ball point::

        ω   = sinh(clip(√c · u, ±v_max)) / √c
        y   = ω / (1 + √(1 + c·‖ω‖²))

    The ``clip`` is the same output-side overflow guard as the hyperboloid path
    (:func:`hyperboloid_core.sinh_lift_to_hyperboloid`); callers must
    ``_assert_v_max_safe(v_max)``. The closed form always lands strictly inside the ball
    (``√c·‖y‖ < 1``), so the caller's ``proj`` is defensive against float rounding only.

    Parameters
    ----------
    u_BO : Array, shape (B, O)
        Activated Busemann logits. ``O = out_spatial = out_dim`` (Poincaré has no time coord).
    c : float
        Curvature (positive).
    v_max : float
        Hard clip bound on ``√c · u`` (output-side overflow guard).

    Returns
    -------
    Array, shape (B, O)
        Points inside the Poincaré ball with curvature ``c`` (pre-projection).
    """
    sqrt_c = jnp.sqrt(c)
    omega_BO = jnp.sinh(jnp.clip(sqrt_c * u_BO, -v_max, v_max)) / sqrt_c
    omega_sqnorm_B1 = jnp.sum(omega_BO**2, axis=-1, keepdims=True)
    denom_B1 = 1.0 + jnp.sqrt(1.0 + c * omega_sqnorm_B1)
    return omega_BO / denom_B1
