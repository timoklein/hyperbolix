"""Tests for Poincaré convolutional layers and beta-concatenation operation.

The ``make_poincare_points`` / ``make_tangent_input`` factories come from
``tests/nn_layers/conftest.py`` (shared with ``test_poincare_batchnorm.py``).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers.poincare_conv import HypConv2DPoincare

# ============================================================================
# Beta-Concatenation Tests
# ============================================================================


@pytest.mark.parametrize("M,n_i,c", [(2, 3, 1.0), (3, 4, 1.0), (4, 5, 0.5), (1, 3, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_beta_concat_output_on_manifold(make_poincare_points, M, n_i, c, dtype):
    """Test that beta-concat output lies on the Poincaré ball."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)

    # Create M points on the Poincaré ball of dimension n_i
    points = make_poincare_points(key, (M, n_i), c, dtype)

    # Apply beta-concat
    result = manifold.beta_concat(points, c)

    # Check output shape
    expected_dim = M * n_i
    assert result.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {result.shape}"

    # Check output is on manifold
    assert manifold.is_in_manifold(result, c), "Beta-concat output not on Poincaré ball"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_beta_concat_single_point(make_poincare_points, dtype):
    """Test beta-concat with a single point (M=1) is identity."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    n_i, c = 5, 1.0

    # Create single point
    point = make_poincare_points(key, (n_i,), c, dtype)
    points = point.reshape(1, n_i)

    # Apply beta-concat
    result = manifold.beta_concat(points, c)

    # For M=1, scale = B(n_i/2, 0.5) / B(n_i/2, 0.5) = 1, so result should equal the input
    assert result.shape == (n_i,)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.allclose(result, point, atol=tolerance), f"Single-point beta-concat should be identity: {result} != {point}"


# (``test_beta_concat_output_dimension`` removed: the ``M * n_i`` shape assertion
# it made is already the first assertion of ``test_beta_concat_output_on_manifold``
# over the same (M, n_i) grid.)


# ============================================================================
# HypConv2DPoincare Layer Tests
# ============================================================================
#
# NOTE: The conv layer now returns tangent-space output (matching the reference
# Poincaré ResNet implementation). Tests check for finite output and correct
# shapes, not manifold membership (tangent vectors are unconstrained).


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_output_shape(make_tangent_input, kernel_size, padding, dtype):
    """Test HypConv2DPoincare output shape with different kernel sizes and padding."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    c = 1.0

    # Create tangent-space input (default input_space="tangent")
    x = make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=Poincare(dtype=dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
        padding=padding,
    )

    # Forward pass
    y = layer(x, c=c)

    # Check output shape
    if padding == "SAME":
        expected_height, expected_width = height, width
    else:  # VALID
        expected_height = height - kernel_size + 1
        expected_width = width - kernel_size + 1

    assert y.shape == (batch_size, expected_height, expected_width, out_channels), (
        f"Expected shape ({batch_size}, {expected_height}, {expected_width}, {out_channels}), got {y.shape}"
    )
    # Output is tangent space, should be finite
    assert jnp.isfinite(y).all(), "Output contains NaN or Inf"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_output_mappable_to_manifold(make_tangent_input, dtype):
    """Test that output can be mapped to manifold via expmap_0."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create tangent-space input
    x = make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    # Forward pass (returns tangent space)
    y = layer(x, c=c)

    # Map to manifold and verify all points are valid
    y_flat = y.reshape(-1, out_channels)
    y_manifold = jax.vmap(manifold.expmap_0, in_axes=(0, None))(y_flat, c)
    is_on_manifold = jax.vmap(lambda p: manifold.is_in_manifold(p, c))(y_manifold)
    assert is_on_manifold.all(), "Output mapped to manifold is not valid"


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_stride(make_tangent_input, stride, dtype):
    """Test HypConv2DPoincare with different stride values."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    kernel_size = 3
    c = 1.0

    # Create tangent-space input
    x = make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=Poincare(dtype=dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        rngs=rngs,
        padding="SAME",
    )

    # Forward pass
    y = layer(x, c=c)

    # Check output shape matches expected stride behavior
    expected_height = (height + stride - 1) // stride
    expected_width = (width + stride - 1) // stride
    assert y.shape == (batch_size, expected_height, expected_width, out_channels)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("input_space", ["manifold", "tangent"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_input_space(input_space, dtype):
    """Test HypConv2DPoincare with different input_space settings."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create input
    x = jax.random.normal(key, (batch_size, height, width, in_channels), dtype=dtype) * 0.1

    if input_space == "manifold":
        # Project to manifold
        proj_fn = partial(manifold.proj, c=c)
        x_input = jax.vmap(jax.vmap(jax.vmap(proj_fn, in_axes=0), in_axes=0), in_axes=0)(x)
    else:
        # Keep in tangent space (small tangent vectors at origin)
        x_input = x

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        input_space=input_space,
    )

    # Forward pass
    y = layer(x_input, c=c)

    # Output is in tangent space, should be finite
    assert jnp.isfinite(y).all(), f"Output contains NaN/Inf for input_space={input_space}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_gradient(make_tangent_input, dtype):
    """Test HypConv2DPoincare has valid gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    c = 1.0

    # Create tangent-space input
    x = make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    # Create layer
    rngs = nnx.Rngs(42)
    layer = HypConv2DPoincare(
        manifold_module=Poincare(dtype=dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    # Define loss function
    def loss_fn(model):
        y = model(x, c=c)
        return jnp.sum(y**2)

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    # Check gradients exist and are finite
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert jnp.isfinite(grads.kernel[...]).all(), "Weight gradients contain NaN/Inf"
    assert jnp.isfinite(grads.bias[...]).all(), "Bias gradients contain NaN/Inf"


def _hnnpp_conv1x1_reference(x_NI, kernel_OI, bias_O1, c):
    """Independent NumPy transcription of a 1x1 HypConv2DPoincare forward pass.

    For a 1x1 kernel the patch extraction is the identity and the beta ratio is
    ``B(C/2, 1/2) / B(C/2, 1/2) = 1``, so the layer reduces exactly to

        tangent -> exp_0 -> HNN++ FC -> log_0

    Transcribed from the equations, not from the library:

    * ``exp_0^c(v)   = tanh(√c‖v‖) · v / (√c‖v‖)``
    * ``λ(x)         = 2 / (1 - c‖x‖²)``                        (HNN++ Eq. 26)
    * ``a            = √c·λ·⟨x, ẑ⟩·cosh(2√c r) - (λ-1)·sinh(2√c r)``
    * ``v            = 2‖z‖·asinh(a)/√c``                        (MLR score)
    * ``w            = sinh(√c v)/√c``,  ``y = w / (1 + √(1 + c‖w‖²))``
    * ``log_0^c(y)   = artanh(√c‖y‖) · y / (√c‖y‖)``

    The ``smooth_clamp`` on ``a`` is inert here (the test asserts ``|a| ≪ clamp``),
    and ``proj`` is inert well inside the ball.
    """
    sqrt_c = np.sqrt(c)

    nv = np.linalg.norm(x_NI, axis=-1, keepdims=True)
    z_NI = np.tanh(sqrt_c * nv) * x_NI / (sqrt_c * nv)  # exp_0

    lam_N1 = 2.0 / (1.0 - c * np.sum(z_NI**2, axis=-1, keepdims=True))
    znorm_O1 = np.linalg.norm(kernel_OI, axis=-1, keepdims=True)
    inner_NO = z_NI @ (kernel_OI / znorm_O1).T
    r_1O = bias_O1.T
    arg_NO = sqrt_c * lam_N1 * inner_NO * np.cosh(2 * sqrt_c * r_1O) - (lam_N1 - 1.0) * np.sinh(2 * sqrt_c * r_1O)
    mlr_NO = 2.0 * znorm_O1.T * np.arcsinh(arg_NO) / sqrt_c

    w_NO = np.sinh(sqrt_c * mlr_NO) / sqrt_c
    y_NO = w_NO / (1.0 + np.sqrt(1.0 + c * np.sum(w_NO**2, axis=-1, keepdims=True)))

    ny_N1 = np.linalg.norm(y_NO, axis=-1, keepdims=True)
    out_NO = np.arctanh(sqrt_c * ny_N1) * y_NO / (sqrt_c * ny_N1)  # log_0
    return out_NO, arg_NO


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0], ids=["c0.5", "c1.0", "c2.0"])
def test_hypconv_1x1_matches_hnnpp_fc_reference(c):
    """A 1x1 conv equals the hand-transcribed HNN++ FC, value for value.

    The only absolute-value oracle for this layer: every other conv test here
    checks shapes, finiteness, or that ``expmap_0(output)`` lands on the ball —
    all of which survive an arbitrary rescale of the output (a 0.7x factor on
    the forward pass passed the whole file). Parametrized over c so a dropped or
    hard-coded curvature also fails.

    The jit leg is folded in (A8-15): the compiled forward must reproduce the
    eager values.
    """
    dtype = jnp.float64
    batch, height, width, in_dim, out_dim = 2, 3, 3, 4, 5
    manifold = Poincare(dtype=dtype)

    layer = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=1,
        rngs=nnx.Rngs(0),
        padding="VALID",
        input_space="tangent",
        param_dtype=dtype,
    )
    assert layer.beta_scale == pytest.approx(1.0)  # 1x1 ⇒ no beta rescale
    key_k, key_b, key_x = jax.random.split(jax.random.PRNGKey(3), 3)
    layer.kernel[...] = jax.random.normal(key_k, (out_dim, in_dim), dtype=dtype) * 0.4
    layer.bias[...] = jax.random.normal(key_b, (out_dim, 1), dtype=dtype) * 0.3

    x_BHWI = jax.random.normal(key_x, (batch, height, width, in_dim), dtype=dtype) * 0.25

    y_BHWO = layer(x_BHWI, c=c)

    expected_NO, arg_NO = _hnnpp_conv1x1_reference(
        np.asarray(x_BHWI.reshape(-1, in_dim), dtype=np.float64),
        np.asarray(layer.kernel[...], dtype=np.float64),
        np.asarray(layer.bias[...], dtype=np.float64),
        float(c),
    )
    assert np.max(np.abs(arg_NO)) < 15.0  # smooth_clamp (clamp ≈ 36 in f64) stays inert

    assert np.allclose(np.asarray(y_BHWO.reshape(-1, out_dim)), expected_NO, atol=1e-10)

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    assert jnp.allclose(forward(layer, x_BHWI, c), y_BHWO, atol=1e-12)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0], ids=["c0.5", "c1.0", "c2.0"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_different_curvatures(make_tangent_input, c, dtype):
    """Test HypConv2DPoincare with different curvature values."""
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3

    x = make_tangent_input(key, (batch_size, height, width, in_channels), dtype)

    layer = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=nnx.Rngs(42),
    )

    y = layer(x, c=c)

    # Output is tangent space, verify finite
    assert jnp.isfinite(y).all(), f"Output contains NaN/Inf for curvature {c}"

    # Verify output can be mapped to manifold at this curvature
    y_flat = y.reshape(-1, out_channels)
    y_manifold = jax.vmap(manifold.expmap_0, in_axes=(0, None))(y_flat, c)
    is_on = jax.vmap(partial(manifold.is_in_manifold, c=c))(y_manifold)
    assert is_on.all(), f"Mapped output not on manifold for curvature {c}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypconv_poincare_chained_tangent_flow(make_tangent_input, dtype):
    """Test chaining two conv layers with standard relu in tangent space.

    This is the primary use case: conv → relu → conv (all in tangent space).
    """
    key = jax.random.PRNGKey(42)
    manifold = Poincare(dtype=dtype)
    batch_size, height, width = 2, 8, 8
    c = 1.0

    # Create tangent-space input
    x = make_tangent_input(key, (batch_size, height, width, 3), dtype)

    # Two conv layers
    rngs = nnx.Rngs(42)
    conv1 = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        rngs=rngs,
        stride=2,
    )
    conv2 = HypConv2DPoincare(
        manifold_module=manifold,
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        rngs=nnx.Rngs(43),
        stride=2,
    )

    # Forward: conv → relu → conv → relu (all tangent space)
    y = conv1(x, c)
    y = jax.nn.relu(y)
    y = conv2(y, c)
    y = jax.nn.relu(y)

    # Check output
    expected_h = (height + 1) // 2  # stride=2 SAME
    expected_h2 = (expected_h + 1) // 2
    assert y.shape == (batch_size, expected_h2, expected_h2, 16)
    assert jnp.isfinite(y).all(), "Chained conv output contains NaN/Inf"

    # Gradients through the chain
    def loss_fn(m1, m2):
        h = m1(x, c)
        h = jax.nn.relu(h)
        h = m2(h, c)
        return jnp.sum(h**2)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=(0, 1))(conv1, conv2)
    grads1, grads2 = grads
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads1.kernel[...]).all()
    assert jnp.isfinite(grads2.kernel[...]).all()
