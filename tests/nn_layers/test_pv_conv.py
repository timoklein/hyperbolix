"""Tests for HypConv2DPV (Proper Velocity conv, Chen et al. 2026, Sec 5.3).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypConv2DPV-specific tests stay here.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds import ProperVelocity
from hyperbolix.nn_layers import HypConv2DPV


def _manifold(dtype: jnp.dtype) -> ProperVelocity:
    return ProperVelocity(dtype=dtype)


def _manifold_input(dtype: jnp.dtype, shape: tuple[int, ...], seed: int = 0) -> jnp.ndarray:
    """Gaussian samples: valid PV points (PV is unconstrained)."""
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=dtype) * 0.3


def _pv_fc_reference(x_BI, kernel_OI, bias_O1, c, inner_activation=None):
    """NumPy transcription of the PV FC forward (Chen et al. 2026, Eq. 19 + Eq. 22).

    Independent of the library:

        beta_inv(x) = sqrt(1 + c*||x||^2)
        v_k(x)      = (||z_k||/sqrt(c)) * asinh( cosh(sqrt(c) r_k)*sqrt(c)/||z_k||*<x, z_k>
                                                 - sinh(sqrt(c) r_k)*beta_inv(x) )   (Eq. 19)
        y_k         = sinh(sqrt(c) * sigma(v_k(x))) / sqrt(c)                        (Eq. 22)

    Duplicated from ``test_pv_linear`` so this file's mutation coverage stands alone.
    """
    x_BI = np.asarray(x_BI, dtype=np.float64)
    z_OI = np.asarray(kernel_OI, dtype=np.float64)
    r_O1 = np.asarray(bias_O1, dtype=np.float64)
    sqrt_c = np.sqrt(c)

    z_norm_1O = np.linalg.norm(z_OI, axis=-1)[None, :]
    sqrt_cr_1O = sqrt_c * r_O1.T
    beta_inv_B1 = np.sqrt(1.0 + c * np.sum(x_BI**2, axis=-1, keepdims=True))

    asinh_arg_BO = np.cosh(sqrt_cr_1O) * (sqrt_c / z_norm_1O) * (x_BI @ z_OI.T) - np.sinh(sqrt_cr_1O) * beta_inv_B1
    v_BO = (z_norm_1O / sqrt_c) * np.arcsinh(asinh_arg_BO)
    if inner_activation is not None:
        v_BO = inner_activation(v_BO)
    return np.sinh(sqrt_c * v_BO) / sqrt_c


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


@pytest.mark.parametrize("c", [0.5, 1.0])
@pytest.mark.parametrize("inner_activation", [None, jnp.tanh], ids=["no_inner_act", "tanh_inner_act"])
def test_pv_conv_1x1_matches_eq22_transcription(c, inner_activation):
    """A 1x1 conv is the PV FC applied per pixel: Chen et al. 2026 Eq. 19 + Eq. 22 in NumPy.

    Value oracle (audit A9-07). Pins the outer ``sinh`` of Eq. 22, which every
    previous test in this file (shape, finiteness, on-manifold -- PV is
    unconstrained, so that check is vacuous) left free, and also pins the
    ``inner_activation`` position: inside the sinh, applied to the Eq. 19 margin.
    """
    dtype = jnp.float64
    batch, height, width, in_channels, out_channels = 2, 3, 3, 4, 5

    x = jax.random.normal(jax.random.PRNGKey(0), (batch, height, width, in_channels), dtype=dtype) * 0.5
    layer = HypConv2DPV(
        manifold_module=_manifold(dtype),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding="VALID",
        rngs=nnx.Rngs(42),
        inner_activation=inner_activation,
        param_dtype=dtype,
    )
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(1), (out_channels, in_channels), dtype=dtype)
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(2), (out_channels, 1), dtype=dtype) * 0.3

    y = layer(x, c=c)

    np_activation = np.tanh if inner_activation is not None else None
    expected = _pv_fc_reference(
        np.asarray(x).reshape(-1, in_channels), layer.kernel[...], layer.bias[...], c, inner_activation=np_activation
    ).reshape(batch, height, width, out_channels)

    assert np.allclose(np.asarray(y), expected, atol=1e-11)
    # The outer sinh must be doing visible work here (it is what a "sinh deleted"
    # mutation removes, and asinh(sqrt(c)*y)/sqrt(c) recovers the bare margin v).
    v_of_y = np.arcsinh(np.sqrt(c) * expected) / np.sqrt(c)
    assert np.max(np.abs(expected - v_of_y)) > 0.02


def test_pv_conv_different_curvatures():
    """Curvature reaches the forward: the three outputs are pairwise different."""
    dtype = jnp.float64
    batch_size, height, width, in_channels, out_channels = 2, 4, 4, 3, 3
    manifold = _manifold(dtype)
    x = _manifold_input(dtype, (batch_size, height, width, in_channels))

    layer = HypConv2DPV(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        rngs=nnx.Rngs(42),
        param_dtype=dtype,
    )
    outputs = [layer(x, c=c) for c in (0.1, 1.0, 2.0)]

    for y in outputs:
        assert jnp.isfinite(y).all()
        assert bool(jax.vmap(partial(manifold.is_in_manifold, c=1.0))(y.reshape(-1, out_channels)).all())
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not jnp.allclose(outputs[i], outputs[j], atol=1e-8)


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
