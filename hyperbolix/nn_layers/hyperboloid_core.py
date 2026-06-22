"""Core Hypformer operations for hyperboloid manifolds.

This module contains the foundational HRC (Hyperbolic Regularization Component) and
HTC (Hyperbolic Transformation Component) operations from the Hypformer paper, as well
as the spacelike V-matrix construction from FGG-LNN (Klis et al. 2026). These
are the building blocks used throughout the library for creating hyperbolic neural
network layers with curvature-change support.

Key Components
--------------
- **hrc**: Wraps Euclidean operations on spatial components only
- **htc**: Wraps Euclidean operations on full hyperboloid points
- **build_spacelike_V**: Constructs spacelike weight vectors for FGG layers

Both HRC/HTC functions enable curvature transformations (c_in → c_out) and avoid
expensive exp/log maps by using constraint-based time reconstruction.

References
----------
Hypformer paper (citation to be added)
Klis et al. "Fast and Geometrically Grounded Lorentz Neural Networks" (2026)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from hyperbolix.utils.math_utils import cosh as safe_cosh
from hyperbolix.utils.math_utils import sinh as safe_sinh


def build_spacelike_V(
    U_IO: Float[Array, "I O"],
    b_O: Float[Array, "O"],
    c: float,
    eps: float = 1e-7,
) -> Float[Array, "Ai O"]:
    """Build spacelike V matrix with Minkowski metric absorbed.

    Constructs the spacelike weight vectors from Euclidean weights U and bias b
    via parallel transport (Eq. 12, Klis et al. 2026). The Minkowski metric
    signature diag(-1, +1, ..., +1) is absorbed into the time row so that the
    forward pass reduces to a single ``x @ V`` matmul.

    Parameters
    ----------
    U_IO : Array, shape (I, O)
        Euclidean weight matrix. I = in_spatial, O = out_spatial.
    b_O : Array, shape (O,)
        Bias per output neuron.
    c : float
        Curvature parameter (positive).
    eps : float, optional
        Numerical stability floor for column norms (default: 1e-7).

    Returns
    -------
    V_AiO : Array, shape (I+1, O)
        Spacelike V matrix with negated time row (Minkowski metric absorbed).
        Columns are v^(i) with ``v_time_mink = -||w|| * sinh(arg)``.

    References
    ----------
    Klis et al. "Fast and Geometrically Grounded Lorentz Neural Networks" (2026), Eq. 12.
    """
    # Dimension key: I=in_spatial, O=out_spatial, Ai=in_ambient (I+1)

    # Column norms of U: ||w^(i)||_E
    norm_sq_O = jnp.sum(U_IO**2, axis=0)  # (O,)
    norm_O = jnp.sqrt(norm_sq_O + eps)  # (O,) gradient-safe (never 0)

    # Smooth gate: 0 for zero-norm columns, ~1 for normal columns.
    # Ensures arg→0 smoothly when U column→0 (no weight → no bias transport).
    gate_O = norm_sq_O / (norm_sq_O + eps)  # (O,)

    # Argument for sinh/cosh: -sqrt(c) * b / ||w||, gated for zero columns
    arg_O = jnp.clip(-jnp.sqrt(c) * b_O * gate_O / norm_O, -100.0, 100.0)  # (O,)

    # Time row: negate to absorb Minkowski metric I_{1,D}
    # v_time = ||w|| * sinh(arg), v_time_mink = -v_time
    v_time_mink_O = -norm_O * safe_sinh(arg_O)  # (O,)

    # Spatial rows: w^(i) * cosh(arg)
    v_space_IO = U_IO * safe_cosh(arg_O)[None, :]  # (I, O)

    # Stack: [v_time_mink; v_space] -> (I+1, O) = (Ai, O)
    V_AiO = jnp.concatenate([v_time_mink_O[None, :], v_space_IO], axis=0)  # (Ai, O)

    return V_AiO


def extract_patches(
    x: Float[Array, "batch height width in_channels"],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: str,
    pad_mode: str = "edge",
    c: float | None = None,
) -> Float[Array, "batch out_height out_width kernel_h kernel_w in_channels"]:
    """Extract receptive-field patches for hyperboloid convolutions.

    Shared by all hyperboloid conv layers (``HypConv2DHyperboloid``,
    ``HypConv2DHyperboloidFHNN``, ``FGGConv2D``, ``HypConv2DHyperboloidILNN``). The
    patches are returned in **point-major** ``(kh, kw, C)`` order so that the
    downstream per-point ops (``hcat`` / ``log_radius_concat``) can consume them as
    ``(K, C)`` after flattening; this is why the channel-major output of
    ``jax.lax.conv_general_dilated_patches`` is transposed.

    Padding is applied **manually** because the conv primitive zero-pads, and zero is
    not a point on the hyperboloid:

    - ``pad_mode="edge"``   — replicate the border (default).
    - ``pad_mode="origin"`` — fill with the manifold origin ``(sqrt(1/c), 0, ..., 0)``;
      ``c`` is required in this mode (matching the Shi et al. 2026 / Klis et al. 2026
      reference padding).

    Parameters
    ----------
    x : Array, shape (B, H, W, in_channels)
        Input feature map of hyperboloid points (ambient channels, time first).
    kernel_size : tuple[int, int]
        ``(kernel_h, kernel_w)``.
    stride : tuple[int, int]
        ``(stride_h, stride_w)``.
    padding : str
        ``"SAME"`` or ``"VALID"``. Only ``"SAME"`` triggers manual padding.
    pad_mode : str, optional
        ``"edge"`` or ``"origin"`` (default: ``"edge"``).
    c : float or None, optional
        Curvature, required only for ``pad_mode="origin"`` (default: None).

    Returns
    -------
    Array, shape (B, out_height, out_width, kernel_h, kernel_w, in_channels)
        Receptive-field patches in point-major order.
    """
    batch, height, width, in_channels = x.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride

    # 1. Manual padding (the conv primitive's zero-pad is off-manifold)
    if padding == "SAME":
        out_height = (height + stride_h - 1) // stride_h
        out_width = (width + stride_w - 1) // stride_w
        pad_h = max((out_height - 1) * stride_h + kernel_h - height, 0)
        pad_w = max((out_width - 1) * stride_w + kernel_w - width, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_mode == "origin":
            # Pad with manifold origin: (√(1/c), 0, ..., 0).
            # Buffer dtype follows x to avoid silent float64 promotion under x64.
            padded_h = height + pad_h
            padded_w = width + pad_w
            padded = jnp.zeros((batch, padded_h, padded_w, in_channels), dtype=x.dtype)
            padded = padded.at[..., 0].set(jnp.sqrt(1.0 / c))
            x = padded.at[:, pad_top : pad_top + height, pad_left : pad_left + width, :].set(x)
        else:  # edge
            x = jnp.pad(
                x,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )

    # 2. Extract patches — output: (B, H, W, C*kh*kw), channel-major (C, kh, kw)
    patches_flat_BHW_CKhKw = jax.lax.conv_general_dilated_patches(
        lhs=x,
        filter_shape=(kernel_h, kernel_w),
        window_strides=(stride_h, stride_w),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )

    # 3. Reshape to separate channels and kernel dims, then transpose to point-major.
    # conv_general_dilated_patches always emits channel-major (C, kh, kw) regardless of
    # the rhs spec letters, so the transpose to (kh, kw, C) is required (and is cheap).
    out_h, out_w = patches_flat_BHW_CKhKw.shape[1], patches_flat_BHW_CKhKw.shape[2]
    patches_BHWCkhkw = patches_flat_BHW_CKhKw.reshape(batch, out_h, out_w, in_channels, kernel_h, kernel_w)
    patches_BHWkhkwC = patches_BHWCkhkw.transpose(0, 1, 2, 4, 5, 3)  # move C last

    return patches_BHWkhkwC


def spatial_to_hyperboloid(
    spatial: Float[Array, "... D"],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... D_plus_1"]:
    """Scale spatial components and reconstruct time to produce a hyperboloid point.

    Extracts the common tail of ``hrc``/``htc``: curvature scaling + time
    reconstruction via the hyperboloid constraint.

    Parameters
    ----------
    spatial : Array, shape (..., D)
        Spatial components (no time coordinate).
    c_in : float
        Source curvature (positive).
    c_out : float
        Target curvature (positive).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    Returns
    -------
    Array, shape (..., D+1)
        Points on the hyperboloid with curvature ``c_out``.
    """
    scale = jnp.sqrt(c_in / c_out)
    scaled_D = scale * spatial  # (..., D)

    norm_sq = jnp.sum(scaled_D**2, axis=-1)  # (...)
    x0 = jnp.sqrt(jnp.maximum(norm_sq + 1.0 / c_out, eps))  # (...)

    return jnp.concatenate([x0[..., None], scaled_D], axis=-1)  # (..., D+1)


def lorentz_midpoint(
    points: Float[Array, "... M A"],
    weights: Float[Array, "... N M"],
    c: float,
    eps: float = 1e-7,
) -> Float[Array, "... N A"]:
    """Weighted Lorentzian midpoint over M points.

    Generalises :func:`lorentz_residual` (which handles two points) to an
    arbitrary weighted combination used by full attention aggregation and
    multi-head averaging.

    Formula (HELM, Chen et al. 2024):
        ``h = weights @ points``  (weighted sum)
        ``mu = h / (sqrt(c) * ||h||_L)``

    where ``||h||_L = sqrt(-<h,h>_L)`` and ``<h,h>_L = -h_0^2 + ||h_s||^2``.

    Parameters
    ----------
    points : Array, shape (..., M, A)
        Points on the hyperboloid with curvature ``c``.  ``A = d + 1``.
    weights : Array, shape (..., N, M)
        Combination weights (e.g. attention weights, uniform ``1/M``).
    c : float
        Curvature parameter (positive).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    Returns
    -------
    Array, shape (..., N, A)
        Midpoints on the hyperboloid with curvature ``c``.
    """
    # h = sum_m w_{n,m} * points_m  →  (..., N, A)
    h_NA = jnp.einsum("...nm,...ma->...na", weights, points)

    # Minkowski squared norm: <h,h>_L = -h_0^2 + ||h_s||^2  (should be < 0)
    mink_1 = -(h_NA[..., 0:1] ** 2) + jnp.sum(h_NA[..., 1:] ** 2, axis=-1, keepdims=True)  # (..., N, 1)
    denom_1 = jnp.sqrt(jnp.maximum(c * jnp.abs(mink_1), eps))  # (..., N, 1)

    return h_NA / denom_1  # (..., N, A)


def lorentz_residual(
    x: Float[Array, "... dim_plus_1"],
    y: Float[Array, "... dim_plus_1"],
    w_y: float | Float[Array, ""],
    c: float,
    eps: float = 1e-7,
) -> Float[Array, "... dim_plus_1"]:
    """Lorentzian midpoint-based residual connection (LResNet from HELM).

    Computes the weighted Lorentzian midpoint of x and y, projecting back
    to the hyperboloid:

        ave = x + w_y * y
        result = ave / sqrt(c * |<ave, ave>_L|)

    where <a, a>_L = -a_0^2 + ||a_s||^2 is the Minkowski inner product.

    .. warning::
        ``w_y`` must be **non-negative**. For x, y on the upper sheet, any conic
        combination ``x + w_y * y`` with ``w_y >= 0`` stays future-directed
        timelike, so the normalization returns a valid hyperboloid point. For
        ``w_y < 0`` the combination can turn spacelike (``<ave, ave>_L > 0``,
        roughly ``w_y < -1`` for nearby points) or land on the lower sheet
        (``ave_0 < 0``) — the ``abs()`` in the normalizer then converts the
        geometry violation into a "valid-looking" but wrong output instead of
        raising. This is why callers must not expose ``w_y`` as an
        unconstrained learnable parameter.

    Parameters
    ----------
    x : Array, shape (..., d+1)
        Points on hyperboloid with curvature c.
    y : Array, shape (..., d+1)
        Points on hyperboloid with curvature c (to be added with weight w_y).
    w_y : float or scalar Array
        Weight for the y contribution. Must be >= 0 (see warning above).
    c : float
        Curvature parameter (positive, c > 0).
    eps : float, optional
        Numerical stability floor (default: 1e-7).

    Returns
    -------
    Array, shape (..., d+1)
        Points on hyperboloid with curvature c.

    References
    ----------
    Chen et al., "Hyperbolic Embeddings for Learning on Manifolds" (HELM), 2024.
    """
    ave_A = x + w_y * y  # (..., A) where A = d+1
    # Minkowski inner: -ave_0^2 + ||ave_s||^2
    mink_1 = -(ave_A[..., 0:1] ** 2) + jnp.sum(ave_A[..., 1:] ** 2, axis=-1, keepdims=True)  # (..., 1)
    denom_1 = jnp.sqrt(jnp.maximum(c * jnp.abs(mink_1), eps))  # (..., 1)
    return ave_A / denom_1  # (..., A)


def hyp_avg_pool2d(
    x: Float[Array, "... H W dim_plus_1"],
    c: float,
    eps: float = 1e-7,
) -> Float[Array, "... dim_plus_1"]:
    """Global average pooling over 2D spatial dimensions on the hyperboloid.

    Averages the spatial components over the height and width dimensions,
    then reconstructs the time component via the hyperboloid constraint.
    This is the HRC pattern with ``f_r = mean_over_spatial``:

    .. math::

        \\text{space} = \\text{mean}_{H,W}(x[..., 1:])

        x_0 = \\sqrt{\\|\\text{space}\\|^2 + 1/c}

        \\text{output} = [x_0, \\text{space}]

    The function expects the **NHWC** layout used throughout hyperbolix:
    ``(..., H, W, ambient_dim)`` where ``ambient_dim = spatial_dim + 1``.

    This operation is **hyperboloid-only** because it relies on the time/spatial
    decomposition ``x[..., 0]`` (time) and ``x[..., 1:]`` (spatial). Poincaré
    ball points do not have a time component. To pool Poincaré features, first
    convert to hyperboloid via
    :func:`~hyperbolix.manifolds.isometry_mappings.poincare_to_hyperboloid`,
    pool, then convert back.

    Parameters
    ----------
    x : Array, shape (..., H, W, dim+1)
        Hyperboloid feature map with curvature ``c``.  The last axis is
        the ambient dimension (time + spatial).  The two axes before it
        are the spatial height ``H`` and width ``W``.
    c : float
        Curvature parameter (positive, c > 0).
    eps : float, optional
        Numerical stability floor for the time reconstruction
        (default: 1e-7).

    Returns
    -------
    Array, shape (..., dim+1)
        Pooled hyperboloid points with curvature ``c``.

    See Also
    --------
    spatial_to_hyperboloid : Low-level curvature scaling + time reconstruction.
    hrc : General HRC wrapper for arbitrary Euclidean functions.
    lorentz_midpoint : Weighted Lorentzian midpoint (geometrically exact aggregation).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.manifolds import Hyperboloid
    >>> from hyperbolix.nn_layers.hyperboloid_core import hyp_avg_pool2d
    >>>
    >>> hyperboloid = Hyperboloid(dtype=jnp.float32)
    >>> key = jax.random.PRNGKey(0)
    >>> v = jax.random.normal(key, (4, 7, 7, 64), dtype=jnp.float32) * 0.1
    >>> x = jax.vmap(jax.vmap(jax.vmap(
    ...     hyperboloid.expmap_0, in_axes=(0, None)
    ... ), in_axes=(0, None)), in_axes=(0, None))(v, 1.0)
    >>> x.shape
    (4, 7, 7, 65)
    >>> y = hyp_avg_pool2d(x, c=1.0)
    >>> y.shape
    (4, 65)
    """
    # Dimension key: H=height, W=width, D=spatial_dim, A=ambient_dim (D+1)

    x_space_HWD = x[..., 1:]  # (..., H, W, D) — drop time coordinate
    x_pooled_D = jnp.mean(x_space_HWD, axis=(-3, -2))  # (..., D)
    return spatial_to_hyperboloid(x_pooled_D, c_in=c, c_out=c, eps=eps)  # (..., D+1)


def hrc(
    x: Float[Array, "... dim_plus_1"],
    f_r: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... out_dim_plus_1"]:
    """Hyperbolic Regularization Component.

    Applies a Euclidean regularization/activation function f_r to the spatial
    components of hyperboloid points, then maps the result to the hyperboloid
    with curvature c_out.

    Mathematical formula:
        space = sqrt(c_in/c_out) * f_r(x[..., 1:])
        time  = sqrt(||space||^2 + 1/c_out)
        output = [time, space]

    When c_in = c_out = c, this reduces to:
        output = [sqrt(||f_r(x_s)||^2 + 1/c), f_r(x_s)]
    which is the pattern used by curvature-preserving hyperboloid activations.

    Parameters
    ----------
    x : Array of shape (..., dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
        The first element is the time-like component, remaining are spatial.
    f_r : Callable
        Euclidean function to apply to spatial components. Can be any activation,
        normalization, dropout, etc. Takes spatial components and returns
        transformed spatial components (may change dimension).
    c_in : float
        Input curvature parameter (must be positive, c > 0).
    c_out : float
        Output curvature parameter (must be positive, c > 0).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., out_dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.

    Notes
    -----
    - f_r operates only on spatial components x[..., 1:], not the time component
    - The time component is reconstructed using the hyperboloid constraint:
      -x₀² + ||x_rest||² = -1/c_out
    - This avoids expensive exp/log maps while maintaining mathematical correctness
    - The spatial scaling factor sqrt(c_in/c_out) ensures proper curvature transformation

    See Also
    --------
    htc : Hyperbolic Transformation Component for full-point operations.

    References
    ----------
    Hypformer paper (citation to be added)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers.hyperboloid_core import hrc
    >>> from hyperbolix.manifolds import Hyperboloid
    >>>
    >>> # Create a point on the hyperboloid
    >>> manifold = Hyperboloid()
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> x = manifold.proj(x, c=1.0)
    >>>
    >>> # Apply HRC with ReLU (curvature-preserving)
    >>> y = hrc(x, jax.nn.relu, c_in=1.0, c_out=1.0)
    >>>
    >>> # Apply HRC with curvature change
    >>> y = hrc(x, jax.nn.relu, c_in=1.0, c_out=2.0)
    >>>
    >>> # Custom activation
    >>> def custom_act(z):
    ...     return jax.nn.gelu(z) * 0.5
    >>> y = hrc(x, custom_act, c_in=1.0, c_out=0.5)
    """
    x_space_D = x[..., 1:]  # (..., D) spatial components

    out_space_D = f_r(x_space_D)  # (..., D') — may change dim

    # Scale for curvature transformation: sqrt(c_in / c_out)
    scale = jnp.sqrt(c_in / c_out)
    scaled_D = scale * out_space_D  # (..., D')

    # Reconstruct time via hyperboloid constraint: x₀ = sqrt(||x_rest||² + 1/c_out)
    norm_sq = jnp.sum(scaled_D**2, axis=-1)  # (...)
    x0 = jnp.sqrt(jnp.maximum(norm_sq + 1.0 / c_out, eps))  # (...)

    return jnp.concatenate([x0[..., None], scaled_D], axis=-1)  # (..., D'+1)


def htc(
    x: Float[Array, "... in_dim_plus_1"],
    f_t: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    c_in: float,
    c_out: float,
    eps: float = 1e-7,
) -> Float[Array, "... out_dim_plus_1"]:
    """Hyperbolic Transformation Component.

    Applies a Euclidean linear transformation f_t to the full hyperboloid point
    (including time component), then maps the result to the hyperboloid with
    curvature c_out.

    Mathematical formula:
        space = sqrt(c_in/c_out) * f_t(x)
        time  = sqrt(||space||^2 + 1/c_out)
        output = [time, space]

    where f_t takes the full (dim+1)-dimensional input and produces the output
    spatial components.

    Parameters
    ----------
    x : Array of shape (..., in_dim+1)
        Input point(s) on the hyperboloid manifold with curvature c_in.
        All components (time and spatial) are passed to f_t.
    f_t : Callable
        Euclidean linear transformation applied to the full input. Takes
        (in_dim+1)-dimensional input and produces out_dim-dimensional output
        (which becomes the spatial components of the output).
    c_in : float
        Input curvature parameter (must be positive, c > 0).
    c_out : float
        Output curvature parameter (must be positive, c > 0).
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Returns
    -------
    y : Array of shape (..., out_dim+1)
        Output point(s) on the hyperboloid manifold with curvature c_out.

    Notes
    -----
    - Unlike HRC, f_t operates on the full point including the time component
    - f_t's output dimension determines the output spatial dimension
    - This is typically used for learnable linear transformations
    - The spatial scaling factor sqrt(c_in/c_out) ensures proper curvature transformation

    See Also
    --------
    hrc : Hyperbolic Regularization Component for spatial-only operations.
    HTCLinear : Module wrapper for htc with learnable linear transformation.

    References
    ----------
    Hypformer paper (citation to be added)

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.nn_layers.hyperboloid_core import htc
    >>> from hyperbolix.manifolds import Hyperboloid
    >>>
    >>> # Create a point on the hyperboloid
    >>> manifold = Hyperboloid()
    >>> x = jnp.array([1.05, 0.1, -0.2, 0.15])
    >>> x = manifold.proj(x, c=1.0)
    >>>
    >>> # Define a linear transformation
    >>> W = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    >>> def linear(z):
    ...     return z @ W.T
    >>>
    >>> # Apply HTC
    >>> y = htc(x, linear, c_in=1.0, c_out=2.0)
    >>> y.shape
    (4,)  # (3 spatial + 1 time)
    """
    # f_t: (..., A_in) → (..., D_out) where A_in = in_dim+1
    out_D = f_t(x)

    # Scale for curvature transformation
    scale = jnp.sqrt(c_in / c_out)
    scaled_D = scale * out_D  # (..., D_out)

    # Reconstruct time via hyperboloid constraint: x₀ = sqrt(||space||² + 1/c_out)
    norm_sq = jnp.sum(scaled_D**2, axis=-1)  # (...)
    x0 = jnp.sqrt(jnp.maximum(norm_sq + 1.0 / c_out, eps))  # (...)

    return jnp.concatenate([x0[..., None], scaled_D], axis=-1)  # (..., D_out+1)
