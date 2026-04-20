"""Tests for HypRegressionPV (Proper Velocity MLR layer, Chen et al. 2026)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import ProperVelocity
from hyperbolix.nn_layers import HypRegressionPV


def _manifold(dtype: jnp.dtype) -> ProperVelocity:
    return ProperVelocity(dtype=dtype)


def _manifold_points(dtype: jnp.dtype, shape: tuple[int, ...], seed: int = 0) -> jnp.ndarray:
    """Gaussian samples: valid PV points without any projection (PV is unconstrained)."""
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=dtype) * 0.3


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_regression_forward_shape_and_finite(dtype):
    batch_size, in_dim, out_dim = 8, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))

    rngs = nnx.Rngs(42)
    layer = HypRegressionPV(_manifold(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()
    assert y.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_regression_jitted_forward(dtype):
    batch_size, in_dim, out_dim = 8, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))

    rngs = nnx.Rngs(42)
    layer = HypRegressionPV(_manifold(dtype), in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y = forward(layer, x, 1.0)
    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_regression_gradient(dtype):
    batch_size, in_dim, out_dim = 4, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))

    rngs = nnx.Rngs(42)
    layer = HypRegressionPV(_manifold(dtype), in_dim, out_dim, rngs=rngs)

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()
    # Non-trivial kernel gradient: at init, bias≈0 and kernel≈0 (std=1e-2), so gradients
    # should be small but nonzero. We require at least one element to be nonzero.
    assert jnp.any(jnp.abs(grads.kernel[...]) > 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_regression_jitted_gradient(dtype):
    batch_size, in_dim, out_dim = 4, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))

    rngs = nnx.Rngs(42)
    layer = HypRegressionPV(_manifold(dtype), in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def loss_fn(module, inputs, curvature):
        y = module(inputs, c=curvature)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, x, 1.0))(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_regression_tangent_input(dtype):
    """With input_space='tangent', Euclidean input is lifted via expmap_0 inside the layer."""
    batch_size, in_dim, out_dim = 8, 5, 3
    v = _manifold_points(dtype, (batch_size, in_dim))

    rngs = nnx.Rngs(42)
    layer = HypRegressionPV(_manifold(dtype), in_dim, out_dim, rngs=rngs, input_space="tangent")

    y = layer(v, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_regression_tangent_matches_manual_lift(dtype):
    """input_space='tangent' must equal manual expmap_0 + input_space='manifold'."""
    batch_size, in_dim, out_dim = 6, 5, 4
    v = _manifold_points(dtype, (batch_size, in_dim))
    manifold = _manifold(dtype)

    manifold_layer = HypRegressionPV(manifold, in_dim, out_dim, rngs=nnx.Rngs(7), input_space="manifold")
    tangent_layer = HypRegressionPV(manifold, in_dim, out_dim, rngs=nnx.Rngs(7), input_space="tangent")

    x = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, 1.0)
    y_manual = manifold_layer(x, c=1.0)
    y_tangent = tangent_layer(v, c=1.0)

    atol = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.allclose(y_manual, y_tangent, atol=atol)


@pytest.mark.parametrize("c", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pv_regression_different_curvatures(dtype, c):
    batch_size, in_dim, out_dim = 4, 5, 3
    x = _manifold_points(dtype, (batch_size, in_dim))

    rngs = nnx.Rngs(42)
    layer = HypRegressionPV(_manifold(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=c)
    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


def test_pv_regression_rejects_bad_input_space():
    with pytest.raises(ValueError, match="input_space"):
        HypRegressionPV(_manifold(jnp.float32), 3, 2, rngs=nnx.Rngs(0), input_space="nope")


def test_pv_regression_rejects_non_pv_manifold():
    """Validator must reject a non-PV manifold that lacks compute_mlr."""

    class Fake:
        pass

    with pytest.raises(TypeError, match="ProperVelocity"):
        HypRegressionPV(Fake(), 3, 2, rngs=nnx.Rngs(0))  # type: ignore[arg-type]
