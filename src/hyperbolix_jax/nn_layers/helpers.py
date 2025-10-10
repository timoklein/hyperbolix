"""Helper functions for hyperbolic neural network layers."""

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import asinh, cosh, sinh, smooth_clamp

# Numerical stability constants mirrored from Poincaré manifold implementation
MAX_NORM_EPS_F32 = 5e-06
MAX_NORM_EPS_F64 = 1e-08

# Dictionary mapping of dtype strings to JAX dtypes
DTYPE_MAP: dict[str, jnp.dtype] = {
    "float32": jnp.float32,
    "float64": jnp.float64,
}


def _get_max_norm_eps(x: Float[Array, "..."]) -> float:
    """Return dtype-dependent epsilon used to clamp conformal denominators."""

    if x.dtype == jnp.float64:
        return MAX_NORM_EPS_F64
    # Default to float32 tolerance for all other dtypes
    return MAX_NORM_EPS_F32


def safe_conformal_factor(
    x: Float[Array, "..."],
    c: float,
) -> Float[Array, "..."]:
    """Numerically stable conformal factor λ(x) = 2 / (1 - c||x||²).

    Mirrors the manifold implementation to avoid division by values near zero when
    points approach the Poincaré ball boundary.

    Args:
        x: Poincaré ball point(s), shape (..., dim)
        c: Manifold curvature

    Returns:
        Conformal factor, shape (..., 1)
    """
    dtype = x.dtype
    c_arr = jnp.asarray(c, dtype=dtype)
    sqrt_c = jnp.sqrt(c_arr)
    max_norm_eps = jnp.asarray(_get_max_norm_eps(x), dtype=dtype)
    x_norm_sq = jnp.sum(x**2, axis=-1, keepdims=True)
    denom_min = 2 * sqrt_c * max_norm_eps - c_arr * max_norm_eps**2
    denom = jnp.maximum(jnp.asarray(1.0, dtype=dtype) - c_arr * x_norm_sq, denom_min)
    return 2.0 / denom


def get_jax_dtype(dtype_str: str) -> jnp.dtype:
    """Convert string dtype representation to JAX dtype.

    Parameters
    ----------
    dtype_str : str
        String representation of dtype ('float32', or 'float64')

    Returns
    -------
    jnp.dtype
        Corresponding JAX dtype

    Raises
    ------
    ValueError
        If dtype_str is not supported
    """
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported dtypes are: {', '.join(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype_str]


def compute_mlr_hyperboloid(
    x: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim_minus_1"],
    r: Float[Array, "out_dim 1"],
    c: float,
    clamping_factor: float,
    smoothing_factor: float,
    min_enorm: float = 1e-15,
) -> Float[Array, "batch out_dim"]:
    """
    Compute 'Fully Hyperbolic Convolutional Neural Networks' multinomial linear regression.

    Parameters
    ----------
    x : Array (batch, in_dim)
        Hyperboloid point(s)
    z : Array (out_dim, in_dim-1)
        Hyperplane tangent normal(s) in tangent space at origin (time coordinate omitted)
    r : Array (out_dim, 1)
        Hyperplane Hyperboloid translation(s) defined by the scalar r and z
    c : float
        Manifold curvature
    clamping_factor : float
        Clamping value for the output
    smoothing_factor : float
        Smoothing factor for the output
    min_enorm : float
        Minimum norm to avoid division by zero (default: 1e-15)

    Returns
    -------
    res : Array (batch, out_dim)
        The multinomial linear regression score(s) of x with respect to the linear model(s) defined by z and r.

    References
    ----------
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic convolutional neural networks for computer vision."
        arXiv preprint arXiv:2303.15919 (2023).
    """
    sqrt_c = jnp.sqrt(c)
    sqrt_cr = sqrt_c * r.T  # (1, out_dim)
    z_norm = jnp.linalg.norm(z, ord=2, axis=-1, keepdims=True).clip(min=min_enorm).T  # (1, out_dim)
    x0 = x[:, 0:1]  # (batch, 1) - time coordinate
    x_rem = x[:, 1:]  # (batch, in_dim-1) - space coordinates
    zx_rem = jnp.einsum("bi,oi->bo", x_rem, z)  # (batch, out_dim)
    alpha = -x0 * sinh(sqrt_cr) * z_norm + cosh(sqrt_cr) * zx_rem  # (batch, out_dim)
    asinh_arg = sqrt_c * alpha / z_norm  # (batch, out_dim)

    # Improve performance by smoothly clamping the input of asinh() to approximately the range of ...
    # ... [-16*clamping_factor, 16*clamping_factor] for float32
    # ... [-36*clamping_factor, 36*clamping_factor] for float64
    eps = jnp.finfo(jnp.float32).eps if x.dtype == jnp.float32 else jnp.finfo(jnp.float64).eps
    clamp = clamping_factor * float(math.log(2 / eps))
    asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, smoothing_factor)  # (batch, out_dim)
    signed_dist2hyp = asinh(asinh_arg) / sqrt_c  # (batch, out_dim)
    res = z_norm * signed_dist2hyp  # (batch, out_dim)
    return res


def compute_mlr_poincare_pp(
    x: Float[Array, "batch in_dim"],
    z: Float[Array, "out_dim in_dim"],
    r: Float[Array, "out_dim 1"],
    c: float,
    clamping_factor: float,
    smoothing_factor: float,
    min_enorm: float = 1e-15,
) -> Float[Array, "batch out_dim"]:
    """
    Compute 'Hyperbolic Neural Networks ++' multinomial linear regression.

    Parameters
    ----------
    x : Array (batch, in_dim)
        PoincareBall point(s)
    z : Array (out_dim, in_dim)
        Hyperplane tangent normal(s) lying in the tangent space at the origin
    r : Array (out_dim, 1)
        Hyperplane PoincareBall translation(s) defined by the scalar r and z
    c : float
        Manifold curvature
    clamping_factor : float
        Clamping value for the output
    smoothing_factor : float
        Smoothing factor for the output
    min_enorm : float
        Minimum norm to avoid division by zero (default: 1e-15)

    Returns
    -------
    res : Array (batch, out_dim)
        The multinomial linear regression score(s) of x with respect to the linear model(s) defined by z and r.

    References
    ----------
    Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
        arXiv preprint arXiv:2006.08210 (2020).
    """
    sqrt_c = jnp.sqrt(c)
    sqrt_c2r = 2 * sqrt_c * r.T  # (1, out_dim)
    z_norm = jnp.linalg.norm(z, ord=2, axis=-1, keepdims=True).clip(min=min_enorm)  # (out_dim, 1)

    # Compute conformal factor (lambda_x)
    lambda_x = safe_conformal_factor(x, c)  # (batch, 1)

    z_unitx = jnp.einsum("bi,oi->bo", x, z / z_norm)  # (batch, out_dim)
    asinh_arg = (1 - lambda_x) * sinh(sqrt_c2r) + sqrt_c * lambda_x * cosh(sqrt_c2r) * z_unitx  # (batch, out_dim)

    # Improve performance by smoothly clamping the input of asinh() to approximately the range of ...
    # ... [-16*clamping_factor, 16*clamping_factor] for float32
    # ... [-36*clamping_factor, 36*clamping_factor] for float64
    eps = jnp.finfo(jnp.float32).eps if x.dtype == jnp.float32 else jnp.finfo(jnp.float64).eps
    clamp = clamping_factor * float(math.log(2 / eps))
    asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, smoothing_factor)  # (batch, out_dim)
    signed_dist2hyp = asinh(asinh_arg) / sqrt_c  # (batch, out_dim)
    res = 2 * z_norm.T * signed_dist2hyp  # (batch, out_dim)
    return res
