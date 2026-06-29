"""Tests for lorentz_scale (LResNet Eq. 10) and the LorentzResidual module."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import LorentzResidual, lorentz_residual, lorentz_scale
from hyperbolix.nn_layers.hyperboloid_core import spatial_to_hyperboloid


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


def _check_on_hyperboloid(x, c, atol=1e-5):
    """Check Minkowski constraint: -x0^2 + ||x_s||^2 = -1/c."""
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


def _make_points(key, batch, dim, dtype, c):
    """Random points on the hyperboloid of ambient dimension ``dim`` (= d+1)."""
    v = jax.random.normal(key, (batch, dim), dtype=dtype) * 0.3
    return jax.vmap(get_hyperboloid(dtype).expmap_0, in_axes=(0, None), out_axes=0)(v, c)


# --------------------------------------------------------------------------- #
# lorentz_scale (Eq. 10) primitive
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("gamma", [2.0, 0.5, -1.5])
def test_lorentz_scale_on_manifold(dtype, c, gamma):
    """Output stays on the hyperboloid for any real gamma (incl. negative)."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    m = _make_points(jax.random.PRNGKey(0), 16, 8, dtype, c)

    out = lorentz_scale(m, gamma, c)

    assert out.shape == m.shape
    assert jnp.isfinite(out).all()
    assert _check_on_hyperboloid(out, c=c, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.1, 1.0])
def test_lorentz_scale_identity_at_one(dtype, c):
    """gamma = 1 is the identity for an on-manifold point."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    m = _make_points(jax.random.PRNGKey(1), 16, 8, dtype, c)

    out = lorentz_scale(m, 1.0, c)

    assert jnp.allclose(out, m, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_lorentz_scale_direction_preserved(dtype):
    """Positive gamma preserves the spatial direction (Klein ray slide)."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    c = 1.0
    m = _make_points(jax.random.PRNGKey(2), 16, 8, dtype, c)

    out = lorentz_scale(m, 2.0, c)

    m_s = m[..., 1:]
    out_s = out[..., 1:]
    m_dir = m_s / jnp.linalg.norm(m_s, axis=-1, keepdims=True)
    out_dir = out_s / jnp.linalg.norm(out_s, axis=-1, keepdims=True)
    assert jnp.allclose(m_dir, out_dir, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_lorentz_scale_norm_monotonic(dtype):
    """gamma > 1 moves away from the origin, gamma < 1 toward it.

    The time coordinate x0 = sqrt(||x_s||^2 + 1/c) is monotone in the spatial
    norm, so it is a faithful proxy for geodesic distance from the origin.
    """
    c = 1.0
    m = _make_points(jax.random.PRNGKey(3), 16, 8, dtype, c)

    farther = lorentz_scale(m, 2.0, c)
    closer = lorentz_scale(m, 0.5, c)

    assert (farther[..., 0] >= m[..., 0]).all()
    assert (closer[..., 0] <= m[..., 0]).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_lorentz_scale_matches_spatial_to_hyperboloid(dtype):
    """lorentz_scale equals scaling the spatial part then reconstructing time."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    c, gamma = 0.5, 1.7
    m = _make_points(jax.random.PRNGKey(4), 16, 8, dtype, c)

    out = lorentz_scale(m, gamma, c)
    expected = spatial_to_hyperboloid(gamma * m[..., 1:], c_in=c, c_out=c)

    assert jnp.allclose(out, expected, atol=atol)


# --------------------------------------------------------------------------- #
# LorentzResidual module
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [False, True])
def test_residual_forward_shape(dtype, scale):
    """Forward returns matching ambient shape and finite values."""
    c = 1.0
    x = _make_points(jax.random.PRNGKey(10), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(11), 8, 6, dtype, c)

    module = LorentzResidual(scale=scale)
    out = module(x, y, c=c)

    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("learnable_weight", [False, True])
@pytest.mark.parametrize("learnable_scale", [False, True])
def test_residual_on_manifold(dtype, scale, learnable_weight, learnable_scale):
    """Output lies on the hyperboloid across all flag combinations."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    c = 0.5
    x = _make_points(jax.random.PRNGKey(12), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(13), 8, 6, dtype, c)

    module = LorentzResidual(
        learnable_weight=learnable_weight,
        scale=scale,
        learnable_scale=learnable_scale,
    )
    out = module(x, y, c=c)

    assert _check_on_hyperboloid(out, c=c, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_residual_jitted_forward(dtype):
    """Module runs under nnx.jit with scaling enabled."""
    c = 1.0
    x = _make_points(jax.random.PRNGKey(14), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(15), 8, 6, dtype, c)

    module = LorentzResidual(scale=True, learnable_scale=True)

    @nnx.jit
    def forward(mod, a, b, curvature):
        return mod(a, b, c=curvature)

    out = forward(module, x, y, c)

    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_residual_learnable_gradients(dtype):
    """Finite gradients flow to the learnable w_y and gamma raw params."""
    c = 1.0
    x = _make_points(jax.random.PRNGKey(16), 4, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(17), 4, 6, dtype, c)

    module = LorentzResidual(learnable_weight=True, scale=True, learnable_scale=True)

    def loss_fn(mod):
        out = mod(x, y, c=c)
        return jnp.sum(out**2)

    loss, grads = nnx.value_and_grad(loss_fn)(module)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.w_y_raw[...])
    assert jnp.isfinite(grads.gamma_raw[...])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_residual_fixed_matches_functional(dtype):
    """Fixed gamma=2, scale=True reproduces lorentz_scale(lorentz_residual(...))."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    c = 1.0
    x = _make_points(jax.random.PRNGKey(18), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(19), 8, 6, dtype, c)

    module = LorentzResidual(learnable_weight=False, init_w_y=1.0, scale=True, init_gamma=2.0)
    got = module(x, y, c=c)

    expected = lorentz_scale(lorentz_residual(x, y, w_y=1.0, c=c), gamma=2.0, c=c)

    assert jnp.allclose(got, expected, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_residual_softplus_keeps_weight_safe(dtype):
    """Even a strongly negative raw weight stays on-manifold (softplus > 0).

    This is the whole reason for the module: a raw nnx.Param(w_y) could drift
    negative and silently corrupt the geometry via the residual's abs()
    normalizer; the softplus reparameterization makes that impossible.
    """
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    c = 1.0
    x = _make_points(jax.random.PRNGKey(20), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(21), 8, 6, dtype, c)

    module = LorentzResidual(learnable_weight=True, init_w_y=1.0)
    # Simulate a training trajectory that drove the raw param very negative.
    module.w_y_raw = nnx.Param(jnp.asarray(-50.0, dtype=dtype))

    out = module(x, y, c=c)

    assert jnp.isfinite(out).all()
    assert _check_on_hyperboloid(out, c=c, atol=atol)


def test_residual_init_validation():
    """Constructor rejects invalid init values."""
    with pytest.raises(ValueError):
        LorentzResidual(init_w_y=-1.0)
    with pytest.raises(ValueError):
        LorentzResidual(learnable_weight=True, init_w_y=0.0)
    with pytest.raises(ValueError):
        LorentzResidual(init_gamma=0.0)
