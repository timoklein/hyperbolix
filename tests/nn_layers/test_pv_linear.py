"""Tests for HypLinearPV (Proper Velocity FC layer, Chen et al. 2026 Thm 5.3).

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypLinearPV-specific tests stay here.
"""

import jax
import jax.numpy as jnp
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


@pytest.mark.parametrize("activation", [jax.nn.relu, jnp.tanh])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_linear_inner_activation(dtype, activation):
    """inner_activation applies inside the outer sinh (paper Eq. 23)."""
    batch_size, in_dim, out_dim = 4, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))

    rngs = nnx.Rngs(42)
    layer = HypLinearPV(
        _manifold(dtype),
        in_dim,
        out_dim,
        rngs=rngs,
        inner_activation=activation,
    )

    y = layer(x, c=1.0)
    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_linear_inner_activation_matches_external(dtype):
    """Applying inner_activation in-layer must equal manual sigma(v) inside sinh(sqrt_c*sigma(v))/sqrt_c."""
    batch_size, in_dim, out_dim = 4, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))
    manifold = _manifold(dtype)
    c = 1.0

    base = HypLinearPV(manifold, in_dim, out_dim, rngs=nnx.Rngs(9), input_space="manifold")
    activated = HypLinearPV(manifold, in_dim, out_dim, rngs=nnx.Rngs(9), input_space="manifold", inner_activation=jnp.tanh)

    # v = compute_mlr(x, kernel, bias, c)
    v = manifold.compute_mlr(x, base.kernel[...], base.bias[...], c, base.clamping_factor, base.smoothing_factor)
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=v.dtype))
    expected = jnp.sinh(sqrt_c * jnp.tanh(v)) / sqrt_c

    y = activated(x, c=c)
    atol = 2e-6 if dtype == jnp.float32 else 1e-12
    assert jnp.allclose(y, expected, atol=atol)


@pytest.mark.parametrize("c", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_linear_different_curvatures(dtype, c):
    batch_size, in_dim, out_dim = 4, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))

    rngs = nnx.Rngs(42)
    layer = HypLinearPV(_manifold(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=c)
    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


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
