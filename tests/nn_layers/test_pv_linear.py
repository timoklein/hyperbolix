"""Tests for HypLinearPV (Proper Velocity FC layer, Chen et al. 2026 Thm 5.3).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypLinearPV-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds import ProperVelocity
from hyperbolix.nn_layers import HypLinearPV
from hyperbolix.nn_layers.pv_linear import _pv_fc_forward


def _manifold(dtype: jnp.dtype) -> ProperVelocity:
    return ProperVelocity(dtype=dtype)


def _manifold_points(dtype: jnp.dtype, shape: tuple[int, ...], seed: int = 0) -> jnp.ndarray:
    """Gaussian samples: valid PV points (PV is unconstrained)."""
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=dtype) * 0.3


def pv_fc_reference(x_BI, kernel_OI, bias_O1, c, inner_activation=None):
    """NumPy transcription of the PV FC forward (Chen et al. 2026, Eq. 19 + Eq. 22).

    Independent of the library:

        beta_inv(x) = sqrt(1 + c*||x||^2)
        v_k(x)      = (||z_k||/sqrt(c)) * asinh( cosh(sqrt(c) r_k)*sqrt(c)/||z_k||*<x, z_k>
                                                 - sinh(sqrt(c) r_k)*beta_inv(x) )   (Eq. 19)
        y_k         = sinh(sqrt(c) * sigma(v_k(x))) / sqrt(c)                        (Eq. 22)

    The library's smooth clamp on the asinh argument is a value-identity well inside
    its bound (~36 at float64, clamping_factor=1), which the test inputs are.
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
def test_pv_linear_tangent_matches_manual_lift(dtype):
    """input_space='tangent' must equal manual expmap_0 + input_space='manifold'."""
    batch_size, in_dim, out_dim = 6, 5, 4
    v = _manifold_points(dtype, (batch_size, in_dim))
    manifold = _manifold(dtype)

    manifold_layer = HypLinearPV(manifold, in_dim, out_dim, rngs=nnx.Rngs(7), input_space="manifold")
    tangent_layer = HypLinearPV(manifold, in_dim, out_dim, rngs=nnx.Rngs(7), input_space="tangent")

    x = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, 1.0)
    y_manual = manifold_layer(x, c=1.0)
    y_tangent = tangent_layer(v, c=1.0)

    atol = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.allclose(y_manual, y_tangent, atol=atol)


@pytest.mark.parametrize("c", [0.5, 1.0])
@pytest.mark.parametrize("inner_activation", [None, jnp.tanh], ids=["no_inner_act", "tanh_inner_act"])
def test_pv_linear_matches_eq22_transcription(c, inner_activation):
    """Forward equals the Chen et al. 2026 Eq. 19 + Eq. 22 transcription in NumPy.

    Replaces the old pair of tests (a shape-only ``inner_activation`` smoke test and
    an "oracle" that recomputed the expected value with ``manifold.compute_mlr`` --
    the very function under test). Pins the outer ``sinh``: dropping it leaves the
    Euclidean-limit value ``v`` behind, which the margins used here differ from by
    tens of percent.
    """
    dtype = jnp.float64
    batch_size, in_dim, out_dim = 4, 5, 3
    manifold = _manifold(dtype)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_dim), dtype=dtype) * 0.5

    layer = HypLinearPV(
        manifold,
        in_dim,
        out_dim,
        rngs=nnx.Rngs(9),
        inner_activation=inner_activation,
        param_dtype=dtype,
    )
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(1), (out_dim, in_dim), dtype=dtype)
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(2), (out_dim, 1), dtype=dtype) * 0.3

    y = layer(x, c=c)
    np_activation = np.tanh if inner_activation is not None else None
    expected = pv_fc_reference(x, layer.kernel[...], layer.bias[...], c, inner_activation=np_activation)

    assert np.allclose(np.asarray(y), expected, atol=1e-11)
    # The outer sinh must be doing visible work at these margins (it is what a
    # "sinh deleted" mutation removes, and asinh(sqrt(c)*y)/sqrt(c) recovers v).
    v_of_y = np.arcsinh(np.sqrt(c) * expected) / np.sqrt(c)
    assert np.max(np.abs(expected - v_of_y)) > 0.02


def test_pv_linear_different_curvatures():
    """Curvature reaches the forward: the three outputs are pairwise different."""
    dtype = jnp.float64
    batch_size, in_dim, out_dim = 4, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))

    layer = HypLinearPV(_manifold(dtype), in_dim, out_dim, rngs=nnx.Rngs(42), param_dtype=dtype)
    outputs = [layer(x, c=c) for c in (0.1, 1.0, 2.0)]

    assert all(jnp.isfinite(y).all() and y.shape == (batch_size, out_dim) for y in outputs)
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not jnp.allclose(outputs[i], outputs[j], atol=1e-8)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_linear_chained_with_regression(dtype):
    """HypLinearPV output is a valid PV point usable as input to HypRegressionPV."""
    from hyperbolix.nn_layers import HypRegressionPV

    batch_size, in_dim, hidden_dim, out_dim = 4, 5, 8, 3
    x = _manifold_points(dtype, (batch_size, in_dim))
    manifold = _manifold(dtype)

    linear = HypLinearPV(manifold, in_dim, hidden_dim, rngs=nnx.Rngs(42))
    regression = HypRegressionPV(manifold, hidden_dim, out_dim, rngs=nnx.Rngs(43))

    h = linear(x, c=1.0)
    y = regression(h, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(h).all()
    assert jnp.isfinite(y).all()


def test_pv_linear_rejects_bad_input_space():
    with pytest.raises(ValueError, match="input_space"):
        HypLinearPV(_manifold(jnp.float32), 3, 2, rngs=nnx.Rngs(0), input_space="nope")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_linear_init_default_he(dtype):
    """Default kernel init is He(in_dim); bias is uniform U(-1e-3, 1e-3).

    Default changed from the paper's z ~ N(0, 1e-2) to He scaling so deep
    fully-hyperbolic PV stacks preserve variance under ReLU. Use the
    ``kernel_init_std`` override to restore the paper recipe (appropriate
    when this layer is the final classification layer before a softmax
    cross-entropy loss).
    """
    in_dim, out_dim = 64, 32
    layer = HypLinearPV(_manifold(dtype), in_dim, out_dim, rngs=nnx.Rngs(42))

    kernel = layer.kernel[...]
    bias = layer.bias[...]

    assert kernel.shape == (out_dim, in_dim)
    assert bias.shape == (out_dim, 1)
    expected_std = (2.0 / in_dim) ** 0.5
    empirical_std = float(jnp.std(kernel))
    assert 0.5 * expected_std < empirical_std < 1.5 * expected_std, (
        f"kernel std {empirical_std:.3e} not within 0.5x..1.5x He target {expected_std:.3e}"
    )
    assert jnp.max(jnp.abs(bias)) <= 1e-3


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_linear_init_paper_override(dtype):
    """``kernel_init_std=1e-2`` reproduces the paper's PVManifoldMLR recipe."""
    in_dim, out_dim = 64, 32
    layer = HypLinearPV(_manifold(dtype), in_dim, out_dim, rngs=nnx.Rngs(42), kernel_init_std=1e-2)

    kernel = layer.kernel[...]
    assert jnp.max(jnp.abs(kernel)) < 0.1, f"kernel too large for std=1e-2 override: max={jnp.max(jnp.abs(kernel))}"


def test_pv_linear_rejects_non_pv_manifold():
    class Fake:
        pass

    with pytest.raises(TypeError, match="ProperVelocity"):
        HypLinearPV(Fake(), 3, 2, rngs=nnx.Rngs(0))  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_fc_forward_helper_matches_layer(dtype):
    """The exported _pv_fc_forward helper and HypLinearPV must agree bit-for-bit."""
    batch_size, in_dim, out_dim = 4, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))
    manifold = _manifold(dtype)

    layer = HypLinearPV(manifold, in_dim, out_dim, rngs=nnx.Rngs(11))
    y_layer = layer(x, c=1.0)
    y_helper = _pv_fc_forward(
        x,
        layer.kernel[...],
        layer.bias[...],
        manifold,
        1.0,
        "manifold",
        layer.clamping_factor,
        layer.smoothing_factor,
        None,
    )
    atol = 1e-6 if dtype == jnp.float32 else 1e-12
    assert jnp.allclose(y_layer, y_helper, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_linear_identity_at_zero_bias_and_zero_input(dtype):
    """At x=0 with bias=0, the MLR argument collapses to zero, so y must be zero."""
    batch_size, in_dim, out_dim = 3, 4, 5
    manifold = _manifold(dtype)
    x = jnp.zeros((batch_size, in_dim), dtype=dtype)

    layer = HypLinearPV(manifold, in_dim, out_dim, rngs=nnx.Rngs(3))
    # Force zero bias.
    layer.bias = nnx.Param(jnp.zeros_like(layer.bias[...]))

    y = layer(x, c=1.0)
    atol = 1e-5 if dtype == jnp.float32 else 1e-12
    assert jnp.allclose(y, jnp.zeros_like(y), atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_linear_trains_with_euclidean_adam(dtype):
    """Euclidean Adam drives an MSE regression loss down on HypLinearPV + HypRegressionPV.

    PV layers expose plain ``nnx.Param`` (not ``ManifoldParam``) because the PV
    retraction is exact Euclidean addition (paper Sec 4). A vanilla
    ``optax.adam`` optimizer must therefore be sufficient to train them — no
    Riemannian wrapper needed. This guards that contract.
    """
    from hyperbolix.nn_layers import HypRegressionPV

    batch_size, in_dim, hidden_dim, out_dim = 32, 6, 8, 3
    manifold = _manifold(dtype)
    c = 1.0

    key = jax.random.PRNGKey(0)
    k_x, k_t = jax.random.split(key)
    x = jax.random.normal(k_x, (batch_size, in_dim), dtype=dtype) * 0.3
    target = jax.random.normal(k_t, (batch_size, out_dim), dtype=dtype) * 0.5

    class TinyPVModel(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.linear = HypLinearPV(manifold, in_dim, hidden_dim, rngs=rngs)
            self.head = HypRegressionPV(manifold, hidden_dim, out_dim, rngs=rngs)

        def __call__(self, inputs, curvature):
            h = self.linear(inputs, c=curvature)
            return self.head(h, c=curvature)

    model = TinyPVModel(nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    def loss_fn(m):
        y = m(x, c)
        return jnp.mean((y - target) ** 2)

    @nnx.jit
    def step(m, opt):
        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        return loss

    initial_loss = float(loss_fn(model))
    for _ in range(200):
        step(model, optimizer)
    final_loss = float(loss_fn(model))

    assert jnp.isfinite(final_loss)
    assert final_loss < 0.5 * initial_loss, (
        f"Euclidean Adam on PV layers failed to reduce MSE: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
