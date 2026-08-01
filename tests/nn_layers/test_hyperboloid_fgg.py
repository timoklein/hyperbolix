"""Tests for FGG-LNN layers (Klis et al. 2026).

Tests cover:
- build_spacelike_V: shape, spacelike property, ||v||_L = ||w||_E identity, zero-bias
- FGGLinear: manifold constraint, shape, activations, weight norm, gradients, JIT
- FGGLorentzMLR: output shape, finite gradients, JIT
- FGGConv2D: output shape, manifold constraint, gradients, JIT
- Cancellation verification: simplified vs full chain numerical equivalence

Dimension key: B=batch, I=in_spatial, O=out_spatial, Ai=in_ambient, Ao=out_ambient, K=classes
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import (
    FGGConv2D,
    FGGLinear,
    FGGLorentzMLR,
    FGGMeanOnlyBatchNorm,
    build_spacelike_V,
    extract_patches,
)

jax.config.update("jax_enable_x64", True)

hyperboloid = Hyperboloid(dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hyperboloid_points(key, batch, ambient_dim, c=1.0):
    """Create valid hyperboloid points: x_0 = sqrt(||x_s||^2 + 1/c)."""
    spatial = jax.random.normal(key, (batch, ambient_dim - 1), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0 / c)
    return jnp.concatenate([time, spatial], axis=-1)


def _check_on_hyperboloid(x, c, atol=1e-5):
    """Check Minkowski constraint: -x0^2 + ||x_s||^2 = -1/c."""
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


def _ref_spacelike_V(U_IO, b_O, c):
    """Independent NumPy transcription of Klis et al. 2026 Eq. 12.

    The bias is transported along the geodesic in the direction of the (unit)
    weight vector, which is where the curvature enters::

        arg   = -sqrt(c) * b / ||w||
        v_t   = ||w|| * sinh(arg)      (returned negated: Minkowski metric absorbed)
        v_s   = w * cosh(arg)

    Written without the library's eps floor / zero-column gate; callers therefore
    pass ``eps`` tiny and keep every column norm well away from zero.
    """
    U = np.asarray(U_IO, dtype=np.float64)
    b = np.asarray(b_O, dtype=np.float64)
    norm_O = np.sqrt((U**2).sum(axis=0))
    arg_O = -np.sqrt(c) * b / norm_O
    v_time_mink_O = -norm_O * np.sinh(arg_O)
    v_space_IO = U * np.cosh(arg_O)[None, :]
    return np.concatenate([v_time_mink_O[None, :], v_space_IO], axis=0)


def _ref_fgg_forward(x_BAi, U_IO, b_O, c, activation=None):
    """Independent NumPy transcription of the whole FGG forward (Eq. 12 + matmul + time)."""
    V_AiO = _ref_spacelike_V(U_IO, b_O, c)
    z_BO = np.asarray(x_BAi, dtype=np.float64) @ V_AiO
    if activation is not None:
        z_BO = activation(z_BO)
    y_0_B1 = np.sqrt((z_BO**2).sum(-1, keepdims=True) + 1.0 / c)
    return np.concatenate([y_0_B1, z_BO], axis=-1)


def _make_image(key, batch, hw, in_channels, c):
    """Image-shaped hyperboloid points: (batch, hw, hw, in_channels)."""
    spatial = jax.random.normal(key, (batch, hw, hw, in_channels - 1), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0 / c)
    return jnp.concatenate([time, spatial], axis=-1)


# ===========================================================================
# build_spacelike_V tests
# ===========================================================================


def test_build_spacelike_V_shape():
    """V matrix has shape (Ai, O) = (I+1, O)."""
    U = jnp.eye(4, 3, dtype=jnp.float64)
    b = jnp.zeros(3, dtype=jnp.float64)
    V = build_spacelike_V(U, b, c=1.0)
    assert V.shape == (5, 3)  # (4+1, 3)


def test_build_spacelike_V_spacelike():
    """Columns of V (before metric absorption) should be spacelike: <v,v>_L > 0."""
    key = jax.random.PRNGKey(0)
    U = jax.random.normal(key, (8, 5), dtype=jnp.float64)
    b = jax.random.normal(jax.random.PRNGKey(1), (5,), dtype=jnp.float64)
    V = build_spacelike_V(U, b, c=1.0)  # (9, 5)

    # V has metric absorbed: time row is negated. Undo for norm check.
    # v_time_mink = -v_time, so v_time = -V[0, :]
    v_time = -V[0, :]  # (5,)
    v_space = V[1:, :]  # (8, 5)

    # Lorentzian norm^2: -v_time^2 + ||v_space||^2 (should be > 0 for spacelike)
    lorentz_norm_sq = -(v_time**2) + jnp.sum(v_space**2, axis=0)
    assert jnp.all(lorentz_norm_sq > 0), f"Some columns are not spacelike: {lorentz_norm_sq}"


def test_build_spacelike_V_lorentz_norm_equals_euclidean_norm():
    """Key identity: ||v||_L = ||w||_E for the FGG construction."""
    key = jax.random.PRNGKey(42)
    U = jax.random.normal(key, (6, 4), dtype=jnp.float64)
    b = jax.random.normal(jax.random.PRNGKey(1), (4,), dtype=jnp.float64)
    V = build_spacelike_V(U, b, c=1.0)

    # Undo metric absorption for time
    v_time = -V[0, :]
    v_space = V[1:, :]

    # ||v||_L = sqrt(-v_time^2 + ||v_space||^2) should equal ||w||_E = ||U column||
    lorentz_norm = jnp.sqrt(-(v_time**2) + jnp.sum(v_space**2, axis=0))
    euclidean_norm = jnp.sqrt(jnp.sum(U**2, axis=0))

    assert jnp.allclose(lorentz_norm, euclidean_norm, atol=1e-10)


def test_build_spacelike_V_zero_bias():
    """With b=0, V should be (0, w) — sinh(0)=0, cosh(0)=1."""
    U = jax.random.normal(jax.random.PRNGKey(0), (5, 3), dtype=jnp.float64)
    b = jnp.zeros(3, dtype=jnp.float64)
    V = build_spacelike_V(U, b, c=1.0)

    # Time row (with metric negation): should be ≈ 0 since sinh(0) = 0
    assert jnp.allclose(V[0, :], 0.0, atol=1e-12)
    # Spatial rows: should be U * cosh(0) = U * 1 = U
    assert jnp.allclose(V[1:, :], U, atol=1e-12)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_build_spacelike_V_matches_eq12(c):
    """V equals the independent Eq. 12 transcription, curvature factor included."""
    U = jax.random.normal(jax.random.PRNGKey(0), (4, 3), dtype=jnp.float64)
    b = jnp.array([0.5, -0.8, 1.3], dtype=jnp.float64)

    V = build_spacelike_V(U, b, c=c, eps=1e-14)

    assert V.shape == (5, 3)
    assert jnp.allclose(V, _ref_spacelike_V(U, b, c), atol=1e-10)


def test_build_spacelike_V_bias_transport_depends_on_curvature():
    """The Eq. 12 bias transport scales with sqrt(c) -- V must vary with curvature.

    Regression guard: dropping the ``sqrt(c)`` factor from ``arg`` makes V (and hence
    every FGG layer's weights) curvature-independent. Shape / finiteness / on-manifold
    checks all still pass, because the time coordinate is reconstructed downstream
    from whatever spatial values come out.
    """
    U = jax.random.normal(jax.random.PRNGKey(0), (4, 3), dtype=jnp.float64)
    b = jnp.array([0.5, -0.8, 1.3], dtype=jnp.float64)

    V_low = build_spacelike_V(U, b, c=0.5, eps=1e-14)
    V_high = build_spacelike_V(U, b, c=2.0, eps=1e-14)

    assert not jnp.allclose(V_low, V_high, atol=1e-6), "V is curvature-independent (sqrt(c) bias transport lost)"

    # The dependence is exactly the sqrt(c) rescaling of arg: arg(4c) == 2 * arg(c),
    # so the ratio of transport arguments recovered from the time row must be 2.
    norm_O = jnp.sqrt(jnp.sum(U**2, axis=0))
    arg_c1 = jnp.arcsinh(-build_spacelike_V(U, b, c=1.0, eps=1e-14)[0, :] / norm_O)
    arg_c4 = jnp.arcsinh(-build_spacelike_V(U, b, c=4.0, eps=1e-14)[0, :] / norm_O)
    assert jnp.allclose(arg_c4, 2.0 * arg_c1, atol=1e-10)


def test_build_spacelike_V_zero_bias_is_curvature_free():
    """With b = 0 the transport vanishes, so V is the same at every curvature."""
    U = jax.random.normal(jax.random.PRNGKey(0), (4, 3), dtype=jnp.float64)
    b = jnp.zeros(3, dtype=jnp.float64)

    assert jnp.allclose(build_spacelike_V(U, b, c=0.5), build_spacelike_V(U, b, c=2.0), atol=1e-12)


# ===========================================================================
# FGGLinear tests
# ===========================================================================


def test_fgg_linear_on_manifold():
    """FGGLinear output satisfies the hyperboloid constraint, eagerly and jitted."""
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 17)
    y = layer(x, c=1.0)
    assert y.shape == (16, 33)
    assert _check_on_hyperboloid(y, c=1.0, atol=1e-8)

    @nnx.jit
    def forward(model, inputs, curvature):
        return model(inputs, c=curvature)

    assert jnp.allclose(forward(layer, x, 1.0), y, atol=1e-12)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_fgg_linear_curvatures(c):
    """FGGLinear output is on hyperboloid for various curvatures."""
    layer = FGGLinear(9, 17, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 8, 9, c=c)
    y = layer(x, c=c)
    assert _check_on_hyperboloid(y, c=c, atol=1e-7)


def test_fgg_linear_with_activation():
    """FGGLinear with ReLU activation still produces valid output."""
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0), activation=jax.nn.relu)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 8, 17)
    y = layer(x, c=1.0)
    assert y.shape == (8, 33)
    assert _check_on_hyperboloid(y, c=1.0, atol=1e-8)


def test_fgg_linear_weight_norm():
    """FGGLinear with weight normalization produces valid output."""
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0), use_weight_norm=True)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 8, 17)
    y = layer(x, c=1.0)
    assert y.shape == (8, 33)
    assert _check_on_hyperboloid(y, c=1.0, atol=1e-8)


@pytest.mark.parametrize("reset_params", ["eye", "xavier", "kaiming", "lorentz_kaiming", "fan_out", "mlr"])
def test_fgg_linear_init_schemes(reset_params):
    """All initialization schemes produce valid outputs."""
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0), reset_params=reset_params)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 17)
    y = layer(x, c=1.0)
    assert y.shape == (4, 33)
    assert jnp.all(jnp.isfinite(y))


def test_fgg_linear_dimension_change():
    """FGGLinear handles in_features != out_features."""
    # Upsample
    layer_up = FGGLinear(5, 33, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 5)
    y = layer_up(x, c=1.0)
    assert y.shape == (4, 33)

    # Downsample
    layer_down = FGGLinear(33, 5, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 33)
    y = layer_down(x, c=1.0)
    assert y.shape == (4, 5)


def test_fgg_linear_gradients():
    """FGGLinear has finite gradients."""
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 17)

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)
    assert jnp.isfinite(loss)
    # Check all parameter gradients are finite
    flat_grads = jax.tree.leaves(grads)
    for g in flat_grads:
        assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient: {g}"


# ===========================================================================
# FGG forward numeric oracles (Eq. 12 + matmul + time reconstruction)
# ===========================================================================


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_fgg_linear_forward_matches_reference(c):
    """FGGLinear reproduces the independent NumPy transcription of the forward pass.

    Regression guard: collapsing the spatial output to the origin (``z * 0``) leaves
    a perfectly valid hyperboloid point at every pixel, so the manifold-constraint,
    shape, gradient and JIT tests all still pass while the layer computes nothing.
    """
    layer = FGGLinear(7, 5, rngs=nnx.Rngs(0), init_bias=0.4, eps=1e-14, param_dtype=jnp.float64)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 7, c=c)

    got = layer(x, c=c)
    expected = _ref_fgg_forward(x, layer.kernel[...], layer.bias[...], c)

    assert jnp.allclose(got, expected, atol=1e-10)
    # The reference itself must be non-trivial (guards against an all-origin oracle).
    assert np.max(np.abs(expected[:, 1:])) > 1e-3


def test_fgg_linear_forward_matches_reference_with_activation():
    """The Euclidean activation is applied to the Minkowski scores, not to the time coordinate."""
    layer = FGGLinear(7, 5, rngs=nnx.Rngs(0), init_bias=0.4, activation=jax.nn.relu, eps=1e-14, param_dtype=jnp.float64)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 7)

    got = layer(x, c=1.0)
    expected = _ref_fgg_forward(x, layer.kernel[...], layer.bias[...], 1.0, activation=lambda z: np.maximum(z, 0.0))

    assert jnp.allclose(got, expected, atol=1e-10)


def test_fgg_linear_is_not_constant():
    """Distinct inputs give distinct outputs, and the origin is not a fixed output."""
    c = 1.0
    layer = FGGLinear(7, 5, rngs=nnx.Rngs(0), init_bias=0.4)
    x_a = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 7, c=c)
    x_b = _make_hyperboloid_points(jax.random.PRNGKey(2), 4, 7, c=c)

    y_a = layer(x_a, c=c)
    origin = jnp.concatenate([jnp.full((4, 1), 1.0 / jnp.sqrt(c)), jnp.zeros((4, 4))], axis=-1)

    assert not jnp.allclose(y_a, layer(x_b, c=c), atol=1e-6)
    assert not jnp.allclose(y_a, origin, atol=1e-6)


# ===========================================================================
# FGGLorentzMLR tests
# ===========================================================================


def test_fgg_mlr_finite_output():
    """FGGLorentzMLR produces finite logits of the right shape, eagerly and jitted."""
    mlr = FGGLorentzMLR(33, 5, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 33)
    logits = mlr(x, c=1.0)
    assert logits.shape == (16, 5)
    assert jnp.all(jnp.isfinite(logits))

    @nnx.jit
    def forward(model, inputs, curvature):
        return model(inputs, c=curvature)

    assert jnp.allclose(forward(mlr, x, 1.0), logits, atol=1e-12)


def test_fgg_mlr_gradients():
    """FGGLorentzMLR has finite gradients."""
    mlr = FGGLorentzMLR(17, 5, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 17)

    def loss_fn(model):
        logits = model(x, c=1.0)
        return jnp.sum(logits**2)

    loss, grads = nnx.value_and_grad(loss_fn)(mlr)
    assert jnp.isfinite(loss)
    flat_grads = jax.tree.leaves(grads)
    for g in flat_grads:
        assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_fgg_mlr_curvatures(c):
    """FGGLorentzMLR works for various curvatures."""
    mlr = FGGLorentzMLR(17, 5, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 4, 17, c=c)
    logits = mlr(x, c=c)
    assert jnp.all(jnp.isfinite(logits))


# ===========================================================================
# FGGConv2D tests
# ===========================================================================


def test_fgg_conv2d_output_shape():
    """FGGConv2D produces correct output shape."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
        padding="SAME",
    )
    # Create image-like hyperboloid input: (B, H, W, C)
    key = jax.random.PRNGKey(1)
    spatial = jax.random.normal(key, (2, 8, 8, 4), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)  # (2, 8, 8, 5)

    y = conv(x, c=1.0)
    assert y.shape == (2, 8, 8, 9)


def test_fgg_conv2d_on_manifold():
    """FGGConv2D output satisfies the hyperboloid constraint at each pixel, eagerly and jitted."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
    )
    x = _make_image(jax.random.PRNGKey(1), 2, 6, 5, c=1.0)

    y = conv(x, c=1.0)
    # Flatten to (N, C) and check each point
    y_flat = y.reshape(-1, 9)
    assert y.shape == (2, 6, 6, 9)
    assert _check_on_hyperboloid(y_flat, c=1.0, atol=1e-7)

    @nnx.jit
    def forward(model, inputs, curvature):
        return model(inputs, c=curvature)

    assert jnp.allclose(forward(conv, x, 1.0), y, atol=1e-12)


def test_fgg_conv2d_stride():
    """FGGConv2D with stride reduces spatial dimensions."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
        stride=2,
        padding="SAME",
    )
    spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 4), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)

    y = conv(x, c=1.0)
    assert y.shape == (2, 4, 4, 9)


def test_fgg_conv2d_valid_padding():
    """FGGConv2D with VALID padding reduces spatial dimensions."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
        padding="VALID",
    )
    spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 4), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)

    y = conv(x, c=1.0)
    assert y.shape == (2, 6, 6, 9)  # 8 - 3 + 1 = 6


def test_fgg_conv2d_gradients():
    """FGGConv2D has finite gradients."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        rngs=nnx.Rngs(0),
    )
    spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 4, 2), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(conv)
    assert jnp.isfinite(loss)
    flat_grads = jax.tree.leaves(grads)
    for g in flat_grads:
        assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("c", [0.5, 2.0])
def test_fgg_conv2d_forward_matches_reference(c):
    """FGGConv2D equals HCat-of-patches followed by the reference FGG linear forward.

    Independent legs: ``extract_patches`` + ``manifold.hcat`` (both separately
    oracle-tested in test_hyperboloid_conv.py) feed the NumPy Eq. 12 transcription.
    Catches an origin-collapsed conv forward, which the on-manifold/shape/JIT tests
    accept because the origin is itself a valid hyperboloid point.
    """
    conv = FGGConv2D(
        hyperboloid,
        in_channels=4,
        out_channels=6,
        kernel_size=2,
        rngs=nnx.Rngs(0),
        padding="VALID",
        init_bias=0.3,
        eps=1e-14,
        param_dtype=jnp.float64,
    )
    x = _make_image(jax.random.PRNGKey(2), 1, 3, 4, c=c)

    got = conv(x, c=c)

    patches = extract_patches(x, conv.kernel_size, conv.stride, conv.padding, conv.pad_mode, c)
    batch, out_h, out_w, kh, kw, in_c = patches.shape
    hcat_NA = jax.vmap(hyperboloid.hcat, in_axes=(0, None))(patches.reshape(-1, kh * kw, in_c), c)
    expected = _ref_fgg_forward(hcat_NA, conv.kernel[...], conv.bias[...], c).reshape(batch, out_h, out_w, conv.out_channels)

    assert got.shape == expected.shape
    assert jnp.allclose(got, expected, atol=1e-10)
    assert np.max(np.abs(expected[..., 1:])) > 1e-3


def test_fgg_conv2d_is_not_constant():
    """The conv responds to its input: an all-origin map and a random map differ."""
    c = 1.0
    conv = FGGConv2D(hyperboloid, in_channels=4, out_channels=6, kernel_size=3, rngs=nnx.Rngs(0))
    origin = jnp.zeros((1, 4, 4, 4), dtype=jnp.float64).at[..., 0].set(1.0 / jnp.sqrt(c))
    x = _make_image(jax.random.PRNGKey(3), 1, 4, 4, c=c)

    assert not jnp.allclose(conv(origin, c=c), conv(x, c=c), atol=1e-6)


# ===========================================================================
# Cancellation verification
# ===========================================================================


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_cancellation_equivalence(c):
    """Verify the sinh/arcsinh cancellation: simplified ≈ full chain.

    Full chain:  x -> arcsinh(sqrt(c) * <x,v>_L) / sqrt(c)  [= Lorentz distance]
                 -> sinh(sqrt(c) * activation(distance)) / sqrt(c)  [= Lorentz act]
                 -> reconstruct time

    Simplified:  x -> <x,v>_L  (matmul)
                 -> activation  (Euclidean)
                 -> reconstruct time  [spatial = z, time = sqrt(||z||^2 + 1/c)]

    When activation = identity, sinh(arcsinh(z)) = z exactly.
    The spatial output is z (no division by sqrt(c)), matching the reference.
    """
    key = jax.random.PRNGKey(0)
    in_features, out_features = 9, 5
    batch = 4

    # Create layer with no activation (identity)
    layer = FGGLinear(in_features, out_features, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(key, batch, in_features, c=c)

    # Simplified path (what FGGLinear computes)
    y_simplified = layer(x, c=c)

    # Full chain path: arcsinh -> identity -> sinh -> reconstruct
    U_IO = layer._get_kernel()
    V_AiO = build_spacelike_V(U_IO, layer.bias[...], c, layer.eps)
    z_BO = x @ V_AiO  # Minkowski inner products

    sqrt_c = jnp.sqrt(c)
    dist_BO = jnp.arcsinh(sqrt_c * z_BO) / sqrt_c  # Lorentz distance
    # identity activation (skip)
    # sinh(arcsinh(√c·z)) / √c = √c·z / √c = z
    spatial_BO = jnp.sinh(sqrt_c * dist_BO) / sqrt_c  # = z
    y_0_full_B1 = jnp.sqrt(jnp.sum(spatial_BO**2, axis=-1, keepdims=True) + 1.0 / c)
    y_full = jnp.concatenate([y_0_full_B1, spatial_BO], axis=-1)

    assert jnp.allclose(y_simplified, y_full, atol=1e-10), (
        f"Cancellation failed at c={c}: max diff = {jnp.max(jnp.abs(y_simplified - y_full))}"
    )


# ===========================================================================
# Weight normalization init tests
# ===========================================================================


def test_fgg_linear_weight_norm_init_magnitude():
    """Weight norm g is initialized to fixed sqrt(1/(I+O)), not column norms of U."""
    in_features, out_features = 33, 65
    layer = FGGLinear(in_features, out_features, rngs=nnx.Rngs(0), use_weight_norm=True)

    g_expected = jnp.sqrt(1.0 / (in_features + out_features))
    assert jnp.allclose(layer.kernel_scale[...], g_expected), (
        f"kernel_scale init should be {g_expected}, got {layer.kernel_scale[...]}"
    )


def test_fgg_linear_weight_norm_softplus_positive():
    """Weight norm effective magnitude (softplus(g)) is always positive."""
    # Use kaiming init so all columns are non-zero (eye has zero cols when O > I)
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0), use_weight_norm=True, reset_params="kaiming")
    U_IO = layer._get_kernel()
    col_norms = jnp.sqrt(jnp.sum(U_IO**2, axis=0))
    assert jnp.all(col_norms > 0), "All effective column magnitudes must be positive"


# ===========================================================================
# fan_out / norm-preserving init tests (the new FGG default)
# ===========================================================================


def test_fgg_linear_default_is_fan_out_zero_bias():
    """FGGLinear now defaults to the norm-preserving fan_out init with zero bias."""
    layer = FGGLinear(33, 257, rngs=nnx.Rngs(0))  # out_spatial = 256
    # bias defaults to 0.0 (was 0.5 in the pre-deviation reference init)
    assert jnp.allclose(layer.bias[...], 0.0)
    # kernel std ~ sqrt(1/out_spatial) (gain=1.0 default)
    expected_std = jnp.sqrt(1.0 / 256)
    kernel_std = jnp.std(layer.kernel[...])
    assert jnp.abs(kernel_std - expected_std) < 0.1 * expected_std, (
        f"default kernel std={kernel_std:.5f} should be near fan_out {expected_std:.5f}"
    )


@pytest.mark.parametrize("gain", [1.0, 0.5])
def test_fgg_linear_fan_out_std(gain):
    """fan_out effective column std is gain / sqrt(out_spatial)."""
    out_features = 257  # out_spatial = 256, large for a tight empirical std
    layer = FGGLinear(33, out_features, rngs=nnx.Rngs(0), reset_params="fan_out", gain=gain)
    kernel_std = jnp.std(layer.kernel[...])
    expected_std = gain * jnp.sqrt(1.0 / (out_features - 1))
    assert jnp.abs(kernel_std - expected_std) < 0.1 * expected_std, (
        f"fan_out (gain={gain}) std={kernel_std:.5f} should be near {expected_std:.5f}"
    )


def test_fgg_fan_out_norm_preservation():
    """A deep fan_out stack preserves the spatial norm; the old fan-in default grows it.

    Regression guard for the downstream IMPALA-ResNet saturation bug: the fan-in
    default init inflates the output spatial norm with width/depth, saturating a
    bounded downstream projection. fan_out + bias=0 keeps ||z|| ~= ||x_spatial||.
    """
    width = 128  # ambient = width + 1
    n_layers = 6
    c = 1.0
    x = _make_hyperboloid_points(jax.random.PRNGKey(0), batch=64, ambient_dim=width + 1, c=c)
    n_in = jnp.mean(jnp.linalg.norm(x[:, 1:], axis=-1))

    # fan_out (new default): norm-preserving, identity activation
    h = x
    for i in range(n_layers):
        layer = FGGLinear(width + 1, width + 1, rngs=nnx.Rngs(i), reset_params="fan_out", gain=1.0, init_bias=0.0)
        h = layer(h, c=c)
        assert _check_on_hyperboloid(h, c=c, atol=1e-7)
    n_out_fan_out = jnp.mean(jnp.linalg.norm(h[:, 1:], axis=-1))

    # Old fan-in reference style: norm grows with width/depth + the 0.5 bias term
    h = x
    for i in range(n_layers):
        layer = FGGLinear(width + 1, width + 1, rngs=nnx.Rngs(i), reset_params="lorentz_kaiming", init_bias=0.5)
        h = layer(h, c=c)
    n_out_fan_in = jnp.mean(jnp.linalg.norm(h[:, 1:], axis=-1))

    # fan_out stays within a constant band of the input norm...
    assert 0.25 * n_in < n_out_fan_out < 4.0 * n_in, f"fan_out norm {n_out_fan_out:.3f} should track input norm {n_in:.3f}"
    # ...while the fan-in default is substantially larger (the saturation failure mode).
    assert n_out_fan_in > 2.0 * n_out_fan_out, f"fan-in norm {n_out_fan_in:.3f} should dwarf fan_out norm {n_out_fan_out:.3f}"


@pytest.mark.parametrize("gain", [0.25, 0.5])
def test_fgg_fan_out_gain_scales_norm(gain):
    """With identity activation and bias=0, output spatial norm scales linearly with gain."""
    c = 1.0
    x = _make_hyperboloid_points(jax.random.PRNGKey(0), batch=64, ambient_dim=65, c=c)

    layer_1 = FGGLinear(65, 65, rngs=nnx.Rngs(0), reset_params="fan_out", gain=1.0, init_bias=0.0)
    layer_g = FGGLinear(65, 65, rngs=nnx.Rngs(0), reset_params="fan_out", gain=gain, init_bias=0.0)

    n_1 = jnp.mean(jnp.linalg.norm(layer_1(x, c=c)[:, 1:], axis=-1))
    n_g = jnp.mean(jnp.linalg.norm(layer_g(x, c=c)[:, 1:], axis=-1))

    # z = x_spatial @ U scales exactly with gain (same seed -> same direction).
    assert jnp.abs(n_g - gain * n_1) < 0.2 * gain * n_1, f"gain={gain}: norm {n_g:.4f} should be ~{gain} x {n_1:.4f}"


def test_fgg_fan_out_weight_norm_gain_noop():
    """Under use_weight_norm, gain (and the fan_out scale) are renormalized away."""
    layer_g1 = FGGLinear(17, 33, rngs=nnx.Rngs(0), reset_params="fan_out", gain=1.0, use_weight_norm=True)
    layer_g8 = FGGLinear(17, 33, rngs=nnx.Rngs(0), reset_params="fan_out", gain=8.0, use_weight_norm=True)
    # softplus(kernel_scale) * dir / ||dir|| is invariant to a global scale on dir.
    assert jnp.allclose(layer_g1._get_kernel(), layer_g8._get_kernel(), atol=1e-10)


def test_fgg_conv2d_default_is_fan_out():
    """FGGConv2D defaults to fan_out + zero bias and stays on-manifold."""
    conv = FGGConv2D(hyperboloid, in_channels=5, out_channels=65, kernel_size=3, rngs=nnx.Rngs(0))
    assert jnp.allclose(conv.bias[...], 0.0)
    spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 6, 4), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)
    y = conv(x, c=1.0)
    assert y.shape == (2, 6, 6, 65)
    assert _check_on_hyperboloid(y.reshape(-1, 65), c=1.0, atol=1e-6)

    # The previous reference-style init is still reachable and valid.
    conv_ref = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
        reset_params="lorentz_kaiming",
        init_bias=0.5,
    )
    assert jnp.allclose(conv_ref.bias[...], 0.5)
    y_ref = conv_ref(x, c=1.0)
    assert _check_on_hyperboloid(y_ref.reshape(-1, 9), c=1.0, atol=1e-6)


def test_fgg_invalid_reset_params_errors():
    """Unknown reset_params raises ValueError listing the valid schemes incl. fan_out."""
    with pytest.raises(ValueError, match="fan_out"):
        FGGLinear(17, 33, rngs=nnx.Rngs(0), reset_params="bogus")


# ===========================================================================
# FGGLorentzMLR init tests
# ===========================================================================


def test_fgg_mlr_mlr_init():
    """FGGLorentzMLR with mlr init uses normal distribution with correct std."""
    mlr = FGGLorentzMLR(65, 10, rngs=nnx.Rngs(0), reset_params="mlr")
    # z should be N(0, sqrt(5/64)), check std is in a reasonable range
    kernel_std = jnp.std(mlr.kernel[...])
    expected_std = jnp.sqrt(5.0 / 64)
    assert jnp.abs(kernel_std - expected_std) < 0.1 * expected_std, (
        f"kernel std={kernel_std:.4f} should be near {expected_std:.4f}"
    )
    # bias should be constant 0.5
    assert jnp.allclose(mlr.bias[...], 0.5)


def test_fgg_mlr_default_init():
    """FGGLorentzMLR with default init uses uniform distribution."""
    mlr = FGGLorentzMLR(65, 10, rngs=nnx.Rngs(0), reset_params="default")
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 8, 65)
    logits = mlr(x, c=1.0)
    assert logits.shape == (8, 10)
    assert jnp.all(jnp.isfinite(logits))


# ===========================================================================
# FGGConv2D origin padding tests
# ===========================================================================


def test_fgg_conv2d_origin_padding():
    """FGGConv2D with origin padding fills borders with manifold origin."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
        padding="SAME",
        pad_mode="origin",
    )
    spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 4), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)

    y = conv(x, c=1.0)
    assert y.shape == (2, 8, 8, 9)
    y_flat = y.reshape(-1, 9)
    assert _check_on_hyperboloid(y_flat, c=1.0, atol=1e-6)


def test_fgg_conv2d_edge_padding():
    """FGGConv2D with edge padding still produces valid output."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
        padding="SAME",
        pad_mode="edge",
    )
    spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 4), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)

    y = conv(x, c=1.0)
    assert y.shape == (2, 8, 8, 9)
    y_flat = y.reshape(-1, 9)
    assert _check_on_hyperboloid(y_flat, c=1.0, atol=1e-6)


# ===========================================================================
# FGGMeanOnlyBatchNorm tests
# ===========================================================================


def test_fgg_mean_only_bn_output_shape():
    """FGGMeanOnlyBatchNorm preserves input shape."""
    bn = FGGMeanOnlyBatchNorm(num_features=32)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 33)
    y = bn(x, c_in=1.0, c_out=1.0)
    assert y.shape == (16, 33)


def test_fgg_mean_only_bn_on_manifold():
    """FGGMeanOnlyBatchNorm output is on the hyperboloid."""
    bn = FGGMeanOnlyBatchNorm(num_features=32)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 33)
    y = bn(x, c_in=1.0, c_out=1.0)
    assert _check_on_hyperboloid(y, c=1.0, atol=1e-8)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_fgg_mean_only_bn_curvatures(c):
    """FGGMeanOnlyBatchNorm works for various curvatures."""
    bn = FGGMeanOnlyBatchNorm(num_features=8)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 9, c=c)
    y = bn(x, c_in=c, c_out=c)
    assert _check_on_hyperboloid(y, c=c, atol=1e-7)


def test_fgg_mean_only_bn_no_variance_division():
    """Mean-only BN subtracts mean but does NOT divide by variance.

    Compare: if we subtract mean and add bias=0, the spatial norms should
    NOT be normalized to unit variance (unlike standard BatchNorm).
    """
    bn = FGGMeanOnlyBatchNorm(num_features=32)
    # Use spatially varied inputs
    key = jax.random.PRNGKey(42)
    spatial = jax.random.normal(key, (64, 32), dtype=jnp.float64) * 3.0
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)  # (64, 33)

    y = bn(x, c_in=1.0, c_out=1.0)
    y_spatial = y[:, 1:]

    # After mean subtraction, spatial variance should be close to input variance
    # (not normalized to ~1 like standard BN)
    input_var = jnp.var(spatial, axis=0)
    output_var = jnp.var(y_spatial, axis=0)
    # Mean-only BN preserves variance (centering doesn't change variance)
    assert jnp.allclose(input_var, output_var, atol=0.1)


def test_fgg_mean_only_bn_running_mean_update():
    """Running mean is updated during training and used during eval."""
    bn = FGGMeanOnlyBatchNorm(num_features=4)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 32, 5)

    # Initial running mean is zeros
    assert jnp.allclose(bn.running_mean[...], 0.0)

    # Forward pass in training mode updates running mean
    _ = bn(x, c_in=1.0, c_out=1.0, use_running_average=False)
    assert not jnp.allclose(bn.running_mean[...], 0.0), "Running mean should be updated"

    # Training vs eval should give different results
    y_train = bn(x, c_in=1.0, c_out=1.0, use_running_average=False)
    y_eval = bn(x, c_in=1.0, c_out=1.0, use_running_average=True)
    assert not jnp.allclose(y_train, y_eval, atol=1e-6)


def test_fgg_mean_only_bn_gradients():
    """FGGMeanOnlyBatchNorm has finite gradients."""
    bn = FGGMeanOnlyBatchNorm(num_features=32)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 33)

    def loss_fn(model):
        y = model(x, c_in=1.0, c_out=1.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(bn)
    assert jnp.isfinite(loss)
    flat_grads = jax.tree.leaves(grads)
    for g in flat_grads:
        assert jnp.all(jnp.isfinite(g))


def test_fgg_mean_only_bn_jit():
    """FGGMeanOnlyBatchNorm is JIT-compatible."""
    bn = FGGMeanOnlyBatchNorm(num_features=32)
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 33)

    @nnx.jit
    def forward(model, inputs, c):
        return model(inputs, c_in=c, c_out=c)

    y = forward(bn, x, 1.0)
    assert y.shape == (16, 33)
    assert _check_on_hyperboloid(y, c=1.0, atol=1e-8)


def test_fgg_mean_only_bn_spatial_dims():
    """FGGMeanOnlyBatchNorm works with spatial dimensions (conv feature maps)."""
    bn = FGGMeanOnlyBatchNorm(num_features=8)
    # Simulate conv output: (B, H, W, C) where C=9 (8 spatial + 1 time)
    key = jax.random.PRNGKey(1)
    spatial = jax.random.normal(key, (2, 4, 4, 8), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)  # (2, 4, 4, 9)

    y = bn(x, c_in=1.0, c_out=1.0)
    assert y.shape == (2, 4, 4, 9)
    y_flat = y.reshape(-1, 9)
    assert _check_on_hyperboloid(y_flat, c=1.0, atol=1e-7)


def test_fgg_conv2d_origin_padding_preserves_float32():
    """SAME/origin padding buffer follows the input dtype.

    Regression: the padding buffer was created with jnp.zeros without dtype,
    which under x64 is float64 and silently promoted a float32 input (and the
    entire downstream convolution) to float64.
    """
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
        padding="SAME",
        pad_mode="origin",
    )
    key = jax.random.PRNGKey(1)
    spatial = jax.random.normal(key, (2, 8, 8, 4), dtype=jnp.float32) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)  # (2, 8, 8, 5) float32

    patches = extract_patches(x, conv.kernel_size, conv.stride, conv.padding, conv.pad_mode, c=1.0)
    assert patches.dtype == jnp.float32
