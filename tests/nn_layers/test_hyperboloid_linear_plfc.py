"""Tests for HypLinearHyperboloidPLFC (point-to-hyperplane Lorentz FC layer)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import HypLinearHyperboloidPLFC


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


def _check_on_hyperboloid(x, c, atol=1e-5):
    """Check Minkowski constraint: -x0^2 + ||x_s||^2 = -1/c."""
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_forward_shape(dtype):
    """Test HypLinearHyperboloidPLFC output shape and finiteness."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_on_manifold(dtype):
    """Test HypLinearHyperboloidPLFC output lies on the hyperboloid."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    y = layer(x, c=1.0)

    assert _check_on_hyperboloid(y, c=1.0, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted_forward(dtype):
    """Test HypLinearHyperboloidPLFC under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y = forward(layer, x, 1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradient(dtype):
    """Test HypLinearHyperboloidPLFC has valid gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    def loss_fn(model):
        y = model(x, c=1.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jitted_gradient(dtype):
    """Test HypLinearHyperboloidPLFC gradients under nnx.jit."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 4, 6, 10

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    @nnx.jit
    def loss_fn(module, inputs, curvature):
        y = module(inputs, c=curvature)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, x, 1.0))(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tangent_input(dtype):
    """Test HypLinearHyperboloidPLFC with tangent space input."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    # Create tangent vector at origin (time coordinate is 0)
    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    v = v.at[:, 0].set(0.0)

    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs, input_space="tangent")

    y = layer(v, c=1.0)

    assert y.shape == (batch_size, out_dim)
    assert jnp.isfinite(y).all()
    assert _check_on_hyperboloid(y, c=1.0, atol=atol)


def test_default_init_scale():
    """Default kernel init follows the Shi et al. 2026 PLFC reference (std=0.02)."""
    rngs = nnx.Rngs(42)
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(jnp.float32), 65, 65, rngs=rngs)

    kernel = layer.kernel[...]
    assert jnp.std(kernel) == pytest.approx(0.02, rel=0.2)
    assert jnp.all(layer.bias[...] == 0.0)


@pytest.mark.parametrize("c", [0.1, 1.0])
def test_v_max_guard_bounds_output(c):
    """The output-side guard bounds the spatial norm by sinh(v_max)/sqrt(c).

    Regression test for the float32 blow-up: with large kernels (std=1.0, the
    pre-PLFC init) and inputs far from the origin, the unguarded sinh produces
    spatial coordinates ~1e17 whose squared norm overflows in stacked layers.
    """
    key = jax.random.PRNGKey(1)
    batch_size, in_dim, out_dim = 4, 129, 129
    v_max = 10.0

    # Input with spatial norm 20 (geodesic distance ~3.7 from origin at c=1)
    xs = jax.random.normal(key, (batch_size, in_dim - 1), dtype=jnp.float32)
    xs = 20.0 * xs / jnp.linalg.norm(xs, axis=-1, keepdims=True)
    x0 = jnp.sqrt(jnp.sum(xs**2, axis=-1, keepdims=True) + 1.0 / c)
    x = jnp.concatenate([x0, xs], axis=-1)

    rngs = nnx.Rngs(42)
    # kernel_init_std=1.0 restores the pre-guard worst case
    layer = HypLinearHyperboloidPLFC(get_hyperboloid(jnp.float32), in_dim, out_dim, rngs=rngs, kernel_init_std=1.0)

    y = layer(x, c=c)

    spatial_bound = jnp.sinh(v_max) / jnp.sqrt(c)
    assert jnp.isfinite(y).all()
    assert jnp.max(jnp.abs(y[:, 1:])) <= spatial_bound * 1.01

    # Gradients must stay finite in the saturated regime
    def loss_fn(model):
        return jnp.sum(model(x, c=c) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gyro_bias_zero_is_identity(dtype):
    """At init the gyro-bias is zero -> gyroaddition with the origin is a no-op."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 1e-5 if dtype == jnp.float32 else 1e-12

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    layer_plain = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(42))
    layer_gyro = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(42), use_gyro_bias=True)

    y_plain = layer_plain(x, c=1.0)
    y_gyro = layer_gyro(x, c=1.0)

    assert jnp.allclose(y_plain, y_gyro, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gyro_bias_on_manifold_and_trainable(dtype):
    """Nonzero gyro-bias keeps the output on the manifold and receives gradients."""
    key = jax.random.PRNGKey(42)
    batch_size, in_dim, out_dim = 8, 6, 10
    atol = 4e-3 if dtype == jnp.float32 else 1e-7

    v = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, 1.0)

    layer = HypLinearHyperboloidPLFC(get_hyperboloid(dtype), in_dim, out_dim, rngs=nnx.Rngs(42), use_gyro_bias=True)
    bias_key = jax.random.PRNGKey(7)
    layer.gyro_bias[...] = jax.random.normal(bias_key, (out_dim - 1,), dtype=layer.gyro_bias[...].dtype) * 0.3

    y = layer(x, c=1.0)

    assert jnp.isfinite(y).all()
    assert _check_on_hyperboloid(y, c=1.0, atol=atol)

    def loss_fn(model):
        return jnp.sum(model(x, c=1.0) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.gyro_bias[...]).all()
    assert jnp.any(grads.gyro_bias[...] != 0.0)
