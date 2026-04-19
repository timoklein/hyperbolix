"""Tests for HypConv2DPV (Proper Velocity conv, Chen et al. 2026, Sec 5.3)."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import ProperVelocity
from hyperbolix.nn_layers import HypConv2DPV


def _manifold(dtype: jnp.dtype) -> ProperVelocity:
    return ProperVelocity(dtype=dtype)


def _manifold_input(dtype: jnp.dtype, shape: tuple[int, ...], seed: int = 0) -> jnp.ndarray:
    """Gaussian samples: valid PV points (PV is unconstrained)."""
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=dtype) * 0.3


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_output_shape(kernel_size, padding, dtype):
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
        padding=padding,
    )

    y = layer(x, c=1.0)

    if padding == "SAME":
        expected_h, expected_w = height, width
    else:
        expected_h = height - kernel_size + 1
        expected_w = width - kernel_size + 1

    assert y.shape == (batch_size, expected_h, expected_w, out_channels)
    assert jnp.isfinite(y).all()
    assert y.dtype == dtype


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_stride(stride, dtype):
    batch_size, height, width, in_channels, out_channels = 2, 8, 8, 3, 4
    kernel_size = 3
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        rngs=rngs,
        padding="SAME",
    )

    y = layer(x, c=1.0)

    expected_h = (height + stride - 1) // stride
    expected_w = (width + stride - 1) // stride
    assert y.shape == (batch_size, expected_h, expected_w, out_channels)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_output_on_manifold(dtype):
    """HypConv2DPV returns PV manifold points — verify all are valid."""
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    manifold = _manifold(dtype)
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    y = layer(x, c=1.0)
    y_flat = y.reshape(-1, out_channels)
    is_on = jax.vmap(manifold.is_in_manifold, in_axes=(0, None))(y_flat, 1.0)
    assert bool(is_on.all())


@pytest.mark.parametrize("input_space", ["manifold", "tangent"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_input_space_modes(input_space, dtype):
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        input_space=input_space,
    )

    y = layer(x, c=1.0)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_tangent_matches_manual_lift(dtype):
    """input_space='tangent' must equal manual expmap_0 + input_space='manifold'."""
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 4
    manifold = _manifold(dtype)
    v = _manifold_input(dtype, (batch_size, height, width, in_channels))

    # Manually lift per-pixel.
    v_flat = v.reshape(-1, in_channels)
    x_flat = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v_flat, 1.0)
    x = x_flat.reshape(v.shape)

    manifold_layer = HypConv2DPV(manifold, in_channels, out_channels, 3, rngs=nnx.Rngs(7), input_space="manifold")
    tangent_layer = HypConv2DPV(manifold, in_channels, out_channels, 3, rngs=nnx.Rngs(7), input_space="tangent")

    y_manual = manifold_layer(x, c=1.0)
    y_tangent = tangent_layer(v, c=1.0)

    atol = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.allclose(y_manual, y_tangent, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_gradient(dtype):
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_jitted(dtype):
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y = forward(layer, x, 1.0)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_init_default_he(dtype):
    """Default kernel init is He(fan_in); bias is uniform U(-1e-3, 1e-3).

    Default changed from the paper's z ~ N(0, 1e-2) to He scaling so deep
    fully-hyperbolic PV conv stacks preserve variance under ReLU. Use the
    ``kernel_init_std`` override to restore the paper recipe.
    """
    in_channels, out_channels, kernel_size = 3, 4, 3
    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
    )

    kernel = layer.kernel[...]
    bias = layer.bias[...]

    concat_dim = kernel_size * kernel_size * in_channels
    assert kernel.shape == (out_channels, concat_dim)
    assert bias.shape == (out_channels, 1)
    # He default: std ≈ sqrt(2/concat_dim). For fan_in=27, target std ≈ 0.272.
    expected_std = (2.0 / concat_dim) ** 0.5
    empirical_std = float(jnp.std(kernel))
    assert 0.5 * expected_std < empirical_std < 1.5 * expected_std, (
        f"kernel std {empirical_std:.3e} not within 0.5x..1.5x He target {expected_std:.3e}"
    )
    # Uniform |bias| <= 1e-3.
    assert jnp.max(jnp.abs(bias)) <= 1e-3


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_init_paper_override(dtype):
    """``kernel_init_std=1e-2`` reproduces the paper's PVManifoldMLR recipe."""
    in_channels, out_channels, kernel_size = 3, 4, 3
    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
        kernel_init_std=1e-2,
    )

    kernel = layer.kernel[...]
    # std ~ 1e-2 → each entry well within |0.1| for 42-seeded RNG.
    assert jnp.max(jnp.abs(kernel)) < 0.1, f"kernel too large for std=1e-2 override: max={jnp.max(jnp.abs(kernel))}"


@pytest.mark.parametrize("c", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_different_curvatures(dtype, c):
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    manifold = _manifold(dtype)
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
    )

    y = layer(x, c=c)
    assert jnp.isfinite(y).all()
    y_flat = y.reshape(-1, out_channels)
    is_on = jax.vmap(partial(manifold.is_in_manifold, c=c))(y_flat)
    assert bool(is_on.all())


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_chained_with_activation(dtype):
    """Chain two PV convs with relu applied directly on PV points (Sec 5.3)."""
    manifold = _manifold(dtype)
    batch_size, height, width = 2, 8, 8
    c = 1.0

    x = _manifold_input(dtype, (batch_size, height, width, 3))

    conv1 = HypConv2DPV(manifold, 3, 8, kernel_size=3, stride=2, rngs=nnx.Rngs(42))
    conv2 = HypConv2DPV(manifold, 8, 16, kernel_size=3, stride=2, rngs=nnx.Rngs(43))

    y = conv1(x, c)
    y = jax.nn.relu(y)
    y = conv2(y, c)
    y = jax.nn.relu(y)

    expected_h = (height + 1) // 2
    expected_h2 = (expected_h + 1) // 2
    assert y.shape == (batch_size, expected_h2, expected_h2, 16)
    assert jnp.isfinite(y).all()

    def loss_fn(m1, m2):
        h = m1(x, c)
        h = jax.nn.relu(h)
        h = m2(h, c)
        return jnp.sum(h**2)

    loss, (grads1, grads2) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(conv1, conv2)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads1.kernel[...]).all()
    assert jnp.isfinite(grads2.kernel[...]).all()


@pytest.mark.parametrize("activation", [jax.nn.relu, jnp.tanh])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_conv_inner_activation(dtype, activation):
    """inner_activation applies inside the outer sinh (paper Eq. 23)."""
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 4
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    rngs = nnx.Rngs(42)
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=rngs,
        inner_activation=activation,
    )

    y = layer(x, c=1.0)
    assert jnp.isfinite(y).all()


def test_pv_conv_rejects_bad_padding():
    with pytest.raises(ValueError, match="padding"):
        HypConv2DPV(_manifold(jnp.float32), 3, 4, 3, rngs=nnx.Rngs(0), padding="REFLECT")


def test_pv_conv_rejects_bad_input_space():
    with pytest.raises(ValueError, match="input_space"):
        HypConv2DPV(_manifold(jnp.float32), 3, 4, 3, rngs=nnx.Rngs(0), input_space="nope")


def test_pv_conv_rejects_non_pv_manifold():
    class Fake:
        pass

    with pytest.raises(TypeError, match="ProperVelocity"):
        HypConv2DPV(Fake(), 3, 4, 3, rngs=nnx.Rngs(0))  # type: ignore[arg-type]
