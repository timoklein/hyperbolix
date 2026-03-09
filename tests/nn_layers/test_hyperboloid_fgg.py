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
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import FGGConv2D, FGGLinear, FGGLorentzMLR, FGGMeanOnlyBatchNorm, build_spacelike_V

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
def test_build_spacelike_V_curvatures(c):
    """V matrix construction works for various curvatures."""
    U = jax.random.normal(jax.random.PRNGKey(0), (4, 3), dtype=jnp.float64)
    b = jnp.ones(3, dtype=jnp.float64) * 0.5
    V = build_spacelike_V(U, b, c=c)
    assert V.shape == (5, 3)
    assert jnp.all(jnp.isfinite(V))


# ===========================================================================
# FGGLinear tests
# ===========================================================================


def test_fgg_linear_output_shape():
    """FGGLinear produces correct output shape."""
    layer = FGGLinear(33, 65, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 8, 33)
    y = layer(x, c=1.0)
    assert y.shape == (8, 65)


def test_fgg_linear_on_manifold():
    """FGGLinear output satisfies hyperboloid constraint."""
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 17)
    y = layer(x, c=1.0)
    assert _check_on_hyperboloid(y, c=1.0, atol=1e-8)


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


@pytest.mark.parametrize("reset_params", ["eye", "xavier", "kaiming", "lorentz_kaiming", "mlr"])
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


def test_fgg_linear_jit():
    """FGGLinear is JIT-compatible."""
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 8, 17)

    @nnx.jit
    def forward(model, inputs, curvature):
        return model(inputs, c=curvature)

    y = forward(layer, x, 1.0)
    assert y.shape == (8, 33)
    assert _check_on_hyperboloid(y, c=1.0, atol=1e-8)


# ===========================================================================
# FGGLorentzMLR tests
# ===========================================================================


def test_fgg_mlr_output_shape():
    """FGGLorentzMLR produces correct output shape."""
    mlr = FGGLorentzMLR(65, 10, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 8, 65)
    logits = mlr(x, c=1.0)
    assert logits.shape == (8, 10)


def test_fgg_mlr_finite_output():
    """FGGLorentzMLR produces finite logits."""
    mlr = FGGLorentzMLR(33, 5, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 16, 33)
    logits = mlr(x, c=1.0)
    assert jnp.all(jnp.isfinite(logits))


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


def test_fgg_mlr_jit():
    """FGGLorentzMLR is JIT-compatible."""
    mlr = FGGLorentzMLR(33, 10, rngs=nnx.Rngs(0))
    x = _make_hyperboloid_points(jax.random.PRNGKey(1), 8, 33)

    @nnx.jit
    def forward(model, inputs, curvature):
        return model(inputs, c=curvature)

    logits = forward(mlr, x, 1.0)
    assert logits.shape == (8, 10)
    assert jnp.all(jnp.isfinite(logits))


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
    """FGGConv2D output satisfies hyperboloid constraint at each pixel."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
    )
    spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 6, 4), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)

    y = conv(x, c=1.0)
    # Flatten to (N, C) and check each point
    y_flat = y.reshape(-1, 9)
    assert _check_on_hyperboloid(y_flat, c=1.0, atol=1e-7)


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


def test_fgg_conv2d_jit():
    """FGGConv2D is JIT-compatible."""
    conv = FGGConv2D(
        hyperboloid,
        in_channels=5,
        out_channels=9,
        kernel_size=3,
        rngs=nnx.Rngs(0),
    )
    spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 6, 4), dtype=jnp.float64) * 0.3
    time = jnp.sqrt(jnp.sum(spatial**2, axis=-1, keepdims=True) + 1.0)
    x = jnp.concatenate([time, spatial], axis=-1)

    @nnx.jit
    def forward(model, inputs, curvature):
        return model(inputs, c=curvature)

    y = forward(conv, x, 1.0)
    assert y.shape == (2, 6, 6, 9)
    y_flat = y.reshape(-1, 9)
    assert _check_on_hyperboloid(y_flat, c=1.0, atol=1e-7)


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
    U_IO = layer._get_U()
    V_AiO = build_spacelike_V(U_IO, layer.b[...], c, layer.eps)
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
    in_spatial, out_spatial = 32, 64
    layer = FGGLinear(in_features, out_features, rngs=nnx.Rngs(0), use_weight_norm=True)

    g_expected = jnp.sqrt(1.0 / (in_spatial + out_spatial))
    assert jnp.allclose(layer.g[...], g_expected), f"g init should be {g_expected}, got {layer.g[...]}"


def test_fgg_linear_weight_norm_softplus_positive():
    """Weight norm effective magnitude (softplus(g)) is always positive."""
    # Use kaiming init so all columns are non-zero (eye has zero cols when O > I)
    layer = FGGLinear(17, 33, rngs=nnx.Rngs(0), use_weight_norm=True, reset_params="kaiming")
    U_IO = layer._get_U()
    col_norms = jnp.sqrt(jnp.sum(U_IO**2, axis=0))
    assert jnp.all(col_norms > 0), "All effective column magnitudes must be positive"


# ===========================================================================
# FGGLorentzMLR init tests
# ===========================================================================


def test_fgg_mlr_mlr_init():
    """FGGLorentzMLR with mlr init uses normal distribution with correct std."""
    mlr = FGGLorentzMLR(65, 10, rngs=nnx.Rngs(0), reset_params="mlr")
    # z should be N(0, sqrt(5/64)), check std is in a reasonable range
    z_std = jnp.std(mlr.z[...])
    expected_std = jnp.sqrt(5.0 / 64)
    assert jnp.abs(z_std - expected_std) < 0.1 * expected_std, f"z std={z_std:.4f} should be near {expected_std:.4f}"
    # bias should be constant 0.5
    assert jnp.allclose(mlr.a[...], 0.5)


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
