"""Tests for lorentz_scale (LResNet Eq. 10) and the LorentzResidual module."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import LorentzResidual, lorentz_scale


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


def _dists(a, b, c, dtype):
    """Batched geodesic distances between two (batch, dim) arrays of hyperboloid points."""
    return jax.vmap(get_hyperboloid(dtype).dist, in_axes=(0, 0, None))(a, b, c)


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


# --------------------------------------------------------------------------- #
# LorentzResidual module
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [False, True])
def test_residual_forward_shape_and_jit(dtype, scale):
    """Forward returns matching ambient shape, finite values, and jits to the eager result."""
    atol = 4e-3 if dtype == jnp.float32 else 1e-10
    c = 1.0
    x = _make_points(jax.random.PRNGKey(10), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(11), 8, 6, dtype, c)

    module = LorentzResidual(scale=scale, learnable_scale=scale)
    out = module(x, y, c=c)

    assert out.shape == x.shape
    assert jnp.isfinite(out).all()

    @nnx.jit
    def forward(mod, a, b, curvature):
        return mod(a, b, c=curvature)

    assert jnp.allclose(forward(module, x, y, c), out, atol=atol)


# scale=False makes learnable_scale inert (gamma is never consulted), so only the
# scale=True rows carry a meaningful learnable_scale axis.
RESIDUAL_FLAGS = [
    (False, False, False),
    (False, True, False),
    (True, False, False),
    (True, True, False),
    (True, True, True),
]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale,learnable_weight,learnable_scale", RESIDUAL_FLAGS)
def test_residual_on_manifold(dtype, scale, learnable_weight, learnable_scale):
    """Output lies on the hyperboloid across the non-degenerate flag combinations."""
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
def test_residual_w_y_slides_monotonically_toward_y(dtype):
    """Raising w_y moves the output monotonically toward y and away from x.

    Geometric oracle for the LResNet weighted midpoint: the output is a point on
    the geodesic-like family interpolating x (w_y -> 0) and y (w_y -> inf), so the
    geodesic distance to y must strictly decrease and the distance to x strictly
    increase as w_y grows. A forward that ignores y (or that collapses to one of
    its inputs) cannot produce this ordering.
    """
    c = 1.0
    x = _make_points(jax.random.PRNGKey(18), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(19), 8, 6, dtype, c)
    slack = 1e-3 if dtype == jnp.float32 else 1e-9

    prev_to_y, prev_to_x = None, None
    for w_y in [0.25, 0.5, 1.0, 2.0, 4.0]:
        out = LorentzResidual(learnable_weight=False, init_w_y=w_y)(x, y, c=c)
        to_y = _dists(out, y, c, dtype)
        to_x = _dists(out, x, c, dtype)
        if prev_to_y is not None:
            assert jnp.all(to_y < prev_to_y - slack), f"dist to y did not decrease at w_y={w_y}"
            assert jnp.all(to_x > prev_to_x + slack), f"dist to x did not increase at w_y={w_y}"
        prev_to_y, prev_to_x = to_y, to_x

    # w_y = 1 is the unweighted midpoint: equidistant from both endpoints.
    mid = LorentzResidual(learnable_weight=False, init_w_y=1.0)(x, y, c=c)
    atol = 4e-3 if dtype == jnp.float32 else 1e-9
    assert jnp.allclose(_dists(mid, x, c, dtype), _dists(mid, y, c, dtype), atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_residual_output_depends_on_y(dtype):
    """The residual branch must actually enter the output.

    Regression guard for a silent no-op: dropping the ``w_y * y`` term from
    ``lorentz_residual`` leaves ``ave = x``, whose normalization returns ``x``
    unchanged for on-manifold inputs -- the module becomes the identity while
    every shape / manifold-constraint / gradient test still passes.
    """
    atol = 4e-3 if dtype == jnp.float32 else 1e-6
    c = 0.7
    x = _make_points(jax.random.PRNGKey(30), 8, 6, dtype, c)
    y_a = _make_points(jax.random.PRNGKey(31), 8, 6, dtype, c)
    y_b = _make_points(jax.random.PRNGKey(32), 8, 6, dtype, c)

    module = LorentzResidual(learnable_weight=False, init_w_y=1.0)
    out_a = module(x, y_a, c=c)
    out_b = module(x, y_b, c=c)

    assert not jnp.allclose(out_a, x, atol=atol), "residual output equals the skip branch (y ignored)"
    assert not jnp.allclose(out_a, out_b, atol=atol), "residual output is invariant to y"
    # The gamma=1 scaled path must stay y-sensitive too (lorentz_scale is a pure spatial rescale).
    scaled = LorentzResidual(learnable_weight=False, init_w_y=1.0, scale=True, init_gamma=1.0)
    assert not jnp.allclose(scaled(x, y_a, c=c), x, atol=atol)


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
