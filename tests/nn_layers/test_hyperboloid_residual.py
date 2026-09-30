"""Tests for lorentz_scale (LResNet Eq. 10) and the LorentzResidual module."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import LorentzResidual, lorentz_scale, project_residual_weights
from hyperbolix.nn_layers.hyperboloid_core import _lorentz_sqdist_polar, lorentz_residual, spatial_to_hyperboloid
from hyperbolix.utils.math_utils import floor_at


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
    negative, where the residual returns NaN (spacelike combination) or a
    valid-looking but wrong point (past-directed combination); the softplus
    reparameterization makes that impossible.
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


# --------------------------------------------------------------------------- #
# Large scaled radius: accuracy against float64, never finiteness
#
# Dimension key: A ambient dim (= D + 1)   D spatial dim
# --------------------------------------------------------------------------- #


def _polar_point(a, u_D, c, dtype=jnp.float64):
    """On-sheet point ``x = (cosh a, sinh a * u)/sqrt(c)``, ``||u|| = 1``, built in float64.

    Exactly on the sheet by construction and ``sqrt(c) d(0, x) = a``, so a test can place a
    point at a named scaled geodesic radius rather than at a spatial norm.
    """
    sqrt_c = jnp.sqrt(jnp.asarray(c, dtype=jnp.float64))
    a = jnp.asarray(a, dtype=jnp.float64)
    return jnp.concatenate([(jnp.cosh(a) / sqrt_c)[None], (jnp.sinh(a) / sqrt_c) * u_D]).astype(dtype)


def _unit(seed, d):
    """Unit float64 direction, shape (D,)."""
    u_D = jax.random.normal(jax.random.PRNGKey(seed), (d,), dtype=jnp.float64)
    return u_D / jnp.linalg.norm(u_D)


def _geodesic(p_A, q_A, c):
    """Geodesic distance between two hyperboloid points, evaluated in float64."""
    return float(get_hyperboloid(jnp.float64).dist(jnp.asarray(p_A, jnp.float64), jnp.asarray(q_A, jnp.float64), c))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lorentz_residual_radial_pair_at_radius_9_matches_float64(seed):
    """Two points on one geodesic ray at ``sqrt(c) d = 9``, 0.3 nats apart: geodesic accuracy.

    A radial pair is where ``<x - y, x - y>_L`` cancels hardest: the two points share a
    direction, so the whole Minkowski square is carried by the radial gap and the literal
    ``-d_0^2 + ||d_s||^2`` reads it off two ``O(e^{2a})`` squares. The failure it produced was
    finite and plausible -- an ordinary hyperboloid point in the wrong place -- so the assertion
    is on the geodesic distance to the float64 result, not on finiteness.

    Measured (c = 0.5, D = 64, seeds 0-3): 2.5e-4 ... 3.1e-4. The pre-fix spelling of the same
    function is 1.6e-2 ... 6.1e-2 on these inputs.

    The ``LorentzResidual`` module is checked on the same inputs: it is a thin wrapper, and the
    point is that the layer a user actually calls inherits the accuracy.
    """
    c, d, a, gap = 0.5, 64, 9.0, 0.3
    u_D = _unit(seed, d)
    x64_A, y64_A = _polar_point(a, u_D, c), _polar_point(a + gap, u_D, c)
    x32_A, y32_A = x64_A.astype(jnp.float32), y64_A.astype(jnp.float32)

    err = _geodesic(lorentz_residual(x32_A, y32_A, 1.0, c), lorentz_residual(x64_A, y64_A, 1.0, c), c)
    assert err < 1e-3, f"float32 residual {err:.2e} geodesic from the float64 one"

    module = LorentzResidual(learnable_weight=False, init_w_y=1.0)
    mod_err = _geodesic(module(x32_A, y32_A, c=c), module(x64_A, y64_A, c=c), c)
    assert mod_err < 1e-3, f"float32 LorentzResidual {mod_err:.2e} geodesic from the float64 one"


@pytest.mark.parametrize("w_y", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("c", [0.5, 1.0])
def test_lorentz_residual_matches_literal_definition_float64(c, w_y):
    """float64 ``lorentz_residual`` equals the literal definition, computed in NumPy.

    Definition pin rather than an accuracy pin: ``ave = x + w_y y``, then
    ``z = ave / sqrt(-c <ave, ave>_L)`` with the time slot rebuilt from the spatial part (what
    ``spatial_to_hyperboloid`` does). It fixes what the function *means* independently of how
    the normalizer is spelled, so a future rewrite of the normalizer has to keep reproducing it.

    Generic directions and radii at or below 3, so the literal reference is itself accurate:
    its cancellation is ``eps64 * cosh^2(a)``, which for a radial pair at ``a = 9`` is 4e-9 and
    would make the *reference* the inaccurate side. Measured: at or below 1.1e-15.
    """
    d = 5
    k_a, k_u = jax.random.split(jax.random.PRNGKey(int(10 * c) + int(10 * w_y)))
    a_2 = jax.random.uniform(k_a, (2,), minval=0.5, maxval=3.0, dtype=jnp.float64)
    u_2D = jax.random.normal(k_u, (2, d), dtype=jnp.float64)
    u_2D = u_2D / jnp.linalg.norm(u_2D, axis=-1, keepdims=True)
    x_A, y_A = _polar_point(a_2[0], u_2D[0], c), _polar_point(a_2[1], u_2D[1], c)

    got_A = lorentz_residual(x_A, y_A, w_y, c)

    ave_A = np.asarray(x_A, np.float64) + w_y * np.asarray(y_A, np.float64)
    mink = -(ave_A[0] ** 2) + np.sum(ave_A[1:] ** 2)
    z_s_D = ave_A[1:] / np.sqrt(-c * mink)
    ref_A = jnp.asarray(np.concatenate([[np.sqrt(np.sum(z_s_D**2) + 1.0 / c)], z_s_D]))

    err = _geodesic(got_A, ref_A, c)
    assert err < 1e-12, f"c={c}, w_y={w_y}: residual {err:.2e} geodesic from the literal definition"


# --------------------------------------------------------------------------- #
# Per-point weights, identity / exp parameterizations, call-time weight (HELM)
#
# Dimension key: B batch  L sequence  A ambient dim (= D + 1)  D spatial dim
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("trailing_axis", [False, True])
def test_lorentz_residual_per_point_weight_matches_scalar_loop(dtype, trailing_axis):
    """A weight array of shape (B, L) or (B, L, 1) equals one scalar-weight call per point."""
    atol = 1e-6 if dtype == jnp.float32 else 1e-12
    c = 0.7
    x_BLA = _make_points(jax.random.PRNGKey(40), 12, 6, dtype, c).reshape(3, 4, 6)
    y_BLA = _make_points(jax.random.PRNGKey(41), 12, 6, dtype, c).reshape(3, 4, 6)
    w_BL = jax.random.uniform(jax.random.PRNGKey(42), (3, 4), minval=0.1, maxval=3.0, dtype=dtype)
    w_arg = w_BL[..., None] if trailing_axis else w_BL

    out_BLA = lorentz_residual(x_BLA, y_BLA, w_arg, c)

    assert out_BLA.shape == x_BLA.shape
    for b in range(3):
        for i in range(4):
            ref_A = lorentz_residual(x_BLA[b, i], y_BLA[b, i], w_BL[b, i], c)
            assert jnp.allclose(out_BLA[b, i], ref_A, atol=atol), f"point ({b}, {i})"


def test_lorentz_residual_weight_shape_validation():
    """A weight array that is neither per-point nor scalar is rejected, not silently broadcast."""
    c = 1.0
    x_BLA = _make_points(jax.random.PRNGKey(43), 12, 6, jnp.float32, c).reshape(3, 4, 6)
    with pytest.raises(ValueError):
        lorentz_residual(x_BLA, x_BLA, jnp.ones((4,), dtype=jnp.float32), c)
    # (S, 1) on (B, S, A) with B == S: a trailing-axis-only check would turn it into (S, 1, 1) and
    # silently align it with the batch axis.
    x_SSA = _make_points(jax.random.PRNGKey(56), 16, 6, jnp.float32, c).reshape(4, 4, 6)
    with pytest.raises(ValueError):
        lorentz_residual(x_SSA, x_SSA, jnp.ones((4, 1), dtype=jnp.float32), c)
    # A single point takes a scalar weight (0-d array or Python float) unchanged.
    x_A = x_BLA[0, 0]
    assert jnp.allclose(lorentz_residual(x_A, x_A, jnp.asarray(0.5), c), lorentz_residual(x_A, x_A, 0.5, c))


PARAMETERIZATIONS = [("identity", "softplus"), ("softplus", "exp"), ("identity", "exp")]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("weight_param,scale_param", PARAMETERIZATIONS)
def test_residual_parameterizations_match_softplus_at_init(dtype, weight_param, scale_param):
    """At the same init values (w_y = 1, gamma = 2) the identity / exp parameterizations give the default output."""
    atol = 1e-5 if dtype == jnp.float32 else 1e-10
    c = 1.0
    x = _make_points(jax.random.PRNGKey(44), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(45), 8, 6, dtype, c)
    kwargs = dict(init_w_y=1.0, scale=True, init_gamma=2.0, learnable_scale=True, param_dtype=dtype)

    default = LorentzResidual(**kwargs)(x, y, c=c)
    other = LorentzResidual(**kwargs, weight_parameterization=weight_param, scale_parameterization=scale_param)(x, y, c=c)

    assert jnp.allclose(other, default, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_residual_identity_exp_gradients(dtype):
    """Gradients reach the raw identity w_y and exp gamma, and agree with the softplus ones by the chain rule.

    Same loss at the same (w_y, gamma): dL/dw_y = grad_identity = grad_softplus / sigmoid(raw_w), and
    dL/dgamma = grad_exp / gamma = grad_softplus / sigmoid(raw_gamma).
    """
    rtol = 1e-4 if dtype == jnp.float32 else 1e-9
    c = 1.0
    x = _make_points(jax.random.PRNGKey(46), 4, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(47), 4, 6, dtype, c)
    kwargs = dict(init_w_y=0.8, scale=True, init_gamma=1.5, learnable_scale=True, param_dtype=dtype)
    softplus_mod = LorentzResidual(**kwargs)
    helm_mod = LorentzResidual(**kwargs, weight_parameterization="identity", scale_parameterization="exp")

    def loss_fn(mod):
        return jnp.sum(mod(x, y, c=c)[..., 1:] ** 2)

    g_sp = nnx.grad(loss_fn)(softplus_mod)
    g_helm = nnx.grad(loss_fn)(helm_mod)

    g_w_helm, g_gamma_helm = g_helm.w_y_raw[...], g_helm.gamma_raw[...]
    assert jnp.isfinite(g_w_helm) and jnp.abs(g_w_helm) > 0
    assert jnp.isfinite(g_gamma_helm) and jnp.abs(g_gamma_helm) > 0
    dl_dw = g_sp.w_y_raw[...] / jax.nn.sigmoid(softplus_mod.w_y_raw[...])
    dl_dgamma = g_sp.gamma_raw[...] / jax.nn.sigmoid(softplus_mod.gamma_raw[...])
    assert jnp.allclose(g_w_helm, dl_dw, rtol=rtol)
    assert jnp.allclose(g_gamma_helm / jnp.exp(helm_mod.gamma_raw[...]), dl_dgamma, rtol=rtol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_residual_identity_allows_negative_weight(dtype):
    """Under the identity parameterization a negative raw w_y is used as is (HELM), not squashed positive.

    Rows where ``x - 0.5 y`` stays timelike give an on-sheet point; rows where it turns spacelike
    give NaN (no ``abs()`` in the normalizer). A softplus-squashed weight would give neither NaN.
    """
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    c = 1.0
    x = _make_points(jax.random.PRNGKey(48), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(49), 8, 6, dtype, c)

    module = LorentzResidual(weight_parameterization="identity", param_dtype=dtype)
    module.w_y_raw = nnx.Param(jnp.asarray(-0.5, dtype=dtype))
    out = module(x, y, c=c)

    ave_BA = np.asarray(x, np.float64) - 0.5 * np.asarray(y, np.float64)
    spacelike_B = np.sum(ave_BA[:, 1:] ** 2, axis=-1) - ave_BA[:, 0] ** 2 > 0.0
    assert jnp.isnan(out[spacelike_B]).all()
    assert jnp.isfinite(out[~spacelike_B]).all()
    assert _check_on_hyperboloid(out[~spacelike_B], c=c, atol=atol)
    tol = 1e-12 if dtype == jnp.float64 else 1e-6
    assert jnp.allclose(out, lorentz_residual(x, y, -0.5, c), atol=tol, equal_nan=True)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_residual_call_time_weight_override(dtype):
    """`weight=` replaces the module's own w_y, as a scalar or one weight per point."""
    atol = 1e-6 if dtype == jnp.float32 else 1e-12
    c = 0.5
    x = _make_points(jax.random.PRNGKey(50), 8, 6, dtype, c)
    y = _make_points(jax.random.PRNGKey(51), 8, 6, dtype, c)
    module = LorentzResidual(init_w_y=1.0, scale=True, init_gamma=2.0)

    fixed = LorentzResidual(learnable_weight=False, init_w_y=0.3, scale=True, init_gamma=2.0)
    assert jnp.allclose(module(x, y, c=c, weight=0.3), fixed(x, y, c=c), atol=atol)

    w_B = jax.random.uniform(jax.random.PRNGKey(52), (8,), minval=0.1, maxval=2.0, dtype=dtype)
    # Independent float64 NumPy oracle from the documented formula: ave = x + w y,
    # out = ave / (sqrt(c) sqrt(-<ave, ave>_L)), space *= gamma, time = sqrt(|space|^2 + 1/c).
    x_BA, y_BA = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ave_BA = x_BA + np.asarray(w_B, np.float64)[:, None] * y_BA
    mink_B1 = -(ave_BA[:, :1] ** 2) + np.sum(ave_BA[:, 1:] ** 2, axis=-1, keepdims=True)
    unit_BA = ave_BA / (np.sqrt(c) * np.sqrt(-mink_B1))
    space_BD = 2.0 * unit_BA[:, 1:]
    time_B1 = np.sqrt(np.sum(space_BD**2, axis=-1, keepdims=True) + 1.0 / c)
    expected_BA = np.concatenate([time_B1, space_BD], axis=-1)
    assert_atol = 1e-5 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(np.asarray(module(x, y, c=c, weight=w_B), np.float64), expected_BA, atol=assert_atol)
    np.testing.assert_allclose(np.asarray(module(x, y, c=c, weight=w_B[:, None]), np.float64), expected_BA, atol=assert_atol)


def test_residual_parameterization_validation():
    """Unknown parameterization strings are rejected; identity accepts a zero learnable init."""
    with pytest.raises(ValueError):
        LorentzResidual(weight_parameterization="exp")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LorentzResidual(scale_parameterization="identity")  # type: ignore[arg-type]
    LorentzResidual(weight_parameterization="identity", init_w_y=0.0)


def _helm_lresnet_numpy(x_NA, y_NA, w_y, k, scale, learned_scale):
    """Transcription of HELM's ``LResNet.forward`` (hypercore/nn/conv/conv_util_layers.py) in float64 NumPy.

    HELM's manifold constant ``k`` is ``1/c``: points satisfy ``<x, x>_L = -k``. ``scale`` is the fixed
    factor, or the raw log-scale when ``learned_scale`` (``x_space = exp(scale) * x_space``).
    """
    x_NA, y_NA = np.asarray(x_NA, np.float64), np.asarray(y_NA, np.float64)
    ave_NA = x_NA + y_NA * w_y
    l_inner_N1 = -(ave_NA[..., :1] ** 2) + np.sum(ave_NA[..., 1:] ** 2, axis=-1, keepdims=True)
    denom_N1 = np.sqrt(np.maximum(np.abs(-l_inner_N1), 1e-4))
    out_NA = np.sqrt(k) * ave_NA / denom_N1
    s = np.exp(scale) if learned_scale else scale
    space_ND = s * out_NA[..., 1:]
    time_N1 = np.sqrt(np.maximum(np.sum(space_ND**2, axis=-1, keepdims=True) + k, 1e-4))
    return np.concatenate([time_N1, space_ND], axis=-1)


# The three HELM uses: block residual (learnable raw w_y, fixed scale sqrt(D)); MoE weighted sum
# (per-token weight, fixed scale 2); shared + routed combine (learnable raw w_y, learnable exp scale).
HELM_CONFIGS = ["block_residual", "moe_weighted_sum", "add_experts"]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("config", HELM_CONFIGS)
def test_residual_matches_helm_lresnet(dtype, c, config):
    """The layer reproduces HELM's LResNet (literal |<ave, ave>_L| clamp form) at normal radii."""
    rtol, atol = (1e-4, 1e-5) if dtype == jnp.float32 else (1e-10, 1e-12)
    d = 8
    x_NA = _make_points(jax.random.PRNGKey(53), 16, d + 1, dtype, c)
    y_NA = _make_points(jax.random.PRNGKey(54), 16, d + 1, dtype, c)
    k = 1.0 / c

    if config == "block_residual":
        module = LorentzResidual(weight_parameterization="identity", scale=True, init_gamma=math.sqrt(d), param_dtype=dtype)
        module.w_y_raw = nnx.Param(jnp.asarray(0.7, dtype=dtype))  # a trained, non-init value
        got = module(x_NA, y_NA, c=c)
        ref = _helm_lresnet_numpy(x_NA, y_NA, 0.7, k, math.sqrt(d), learned_scale=False)
    elif config == "moe_weighted_sum":
        w_N1 = jax.random.uniform(jax.random.PRNGKey(55), (16, 1), minval=0.05, maxval=1.0, dtype=dtype)
        module = LorentzResidual(learnable_weight=False, scale=True, init_gamma=2.0)
        got = module(x_NA, y_NA, c=c, weight=w_N1)
        ref = _helm_lresnet_numpy(x_NA, y_NA, np.asarray(w_N1, np.float64), k, 2.0, learned_scale=False)
    else:
        module = LorentzResidual(
            weight_parameterization="identity",
            scale=True,
            init_gamma=2.0,
            learnable_scale=True,
            scale_parameterization="exp",
            param_dtype=dtype,
        )
        module.w_y_raw = nnx.Param(jnp.asarray(1.3, dtype=dtype))
        module.gamma_raw = nnx.Param(jnp.asarray(0.4, dtype=dtype))  # exp(0.4) ~ 1.49, off init
        got = module(x_NA, y_NA, c=c)
        ref = _helm_lresnet_numpy(x_NA, y_NA, 1.3, k, 0.4, learned_scale=True)

    np.testing.assert_allclose(np.asarray(got, np.float64), ref, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Normalizer without abs() / floor: bit-identical for w_y >= 0, NaN when spacelike
#
# Dimension key: N points  A ambient dim (= D + 1)  D spatial dim
# --------------------------------------------------------------------------- #


def _abs_floor_lorentz_residual(x_NA, y_NA, w_N1, c, eps=1e-7):
    """The earlier ``lorentz_residual`` body, normalizing by ``sqrt(floor_at(c * |mink|, eps))``."""
    ave_NA = x_NA + w_N1 * y_NA
    dd_N1 = _lorentz_sqdist_polar(x_NA, y_NA, c)[..., None]
    mink_N1 = -((1.0 + w_N1) ** 2) / c - w_N1 * dd_N1
    denom_N1 = jnp.sqrt(floor_at(c * jnp.abs(mink_N1), eps))
    return spatial_to_hyperboloid((ave_NA / denom_N1)[..., 1:], c_in=c, c_out=c, eps=eps)


def _radius_pairs(c, dtype, d=6):
    """Pairs at scaled radii 0.05 ... 8 (spatial norm up to sinh(8)/sqrt(0.1) = 4.7e3), shape (N, A)."""
    a_R = [0.05, 1.0, 4.0, 7.6, 8.0]
    x_NA = jnp.stack([_polar_point(a, _unit(100 + i, d), c, dtype) for i, a in enumerate(a_R)])
    y_NA = jnp.stack([_polar_point(a, _unit(200 + i, d), c, dtype) for i, a in enumerate(a_R[::-1])])
    return x_NA, y_NA


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("w_y", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("c", [0.1, 1.0])
def test_lorentz_residual_bit_identical_to_abs_floor_form(dtype, w_y, c):
    """For ``w_y >= 0`` dropping the ``abs()`` and the floor changes no bit, eager or jitted.

    ``-c<ave, ave>_L >= (1 + w_y)^2 >= 1`` there, so both were inactive. Checked with a scalar
    weight and with per-point weights (shape ``(N,)``, compared with the ``(N, 1)`` old form).
    """
    x_NA, y_NA = _radius_pairs(c, dtype)
    w_N = jnp.asarray([w_y, 0.3 * w_y, 5.0 * w_y, 1e-3 * w_y, w_y], dtype=dtype)
    residual_jit = jax.jit(lorentz_residual, static_argnums=(3,))
    old_jit = jax.jit(_abs_floor_lorentz_residual, static_argnums=(3,))
    cases = [
        (lorentz_residual(x_NA, y_NA, w_y, c), _abs_floor_lorentz_residual(x_NA, y_NA, w_y, c)),
        (lorentz_residual(x_NA, y_NA, w_N, c), _abs_floor_lorentz_residual(x_NA, y_NA, w_N[:, None], c)),
        (residual_jit(x_NA, y_NA, jnp.asarray(w_y, dtype), c), old_jit(x_NA, y_NA, jnp.asarray(w_y, dtype), c)),
        (residual_jit(x_NA, y_NA, w_N, c), old_jit(x_NA, y_NA, w_N[:, None], c)),
    ]
    for i, (new_NA, old_NA) in enumerate(cases):
        assert np.asarray(new_NA).tobytes() == np.asarray(old_NA).tobytes(), f"case {i}: {new_NA} != {old_NA}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("w_y,a_y", [(-1.0, 1.1), (-0.5, 1.5)])
def test_lorentz_residual_spacelike_combination_is_nan(dtype, w_y, a_y):
    """A negative ``w_y`` that makes ``x + w_y y`` spacelike gives NaN, not a hyperboloid point.

    ``w_y = -1`` is spacelike for every ``x != y``; ``w_y = -0.5`` once the points are far enough
    apart (here ~2.5 nats). Regression: the normalizer was ``sqrt(floor_at(c |<ave, ave>_L|, eps))``,
    which turned the spacelike ``ave`` into a finite, valid-looking point, as the HELM / LResNet
    reference's ``.abs()`` does. The identity-parameterized ``LorentzResidual`` inherits the NaN.
    """
    c, d = 1.0, 6
    x_A = _polar_point(1.0, _unit(300, d), c, dtype)
    y_A = _polar_point(a_y, _unit(301, d), c, dtype)
    ave_A = np.asarray(x_A, np.float64) + w_y * np.asarray(y_A, np.float64)
    assert np.sum(ave_A[1:] ** 2) - ave_A[0] ** 2 > 0.1  # the combination is clearly spacelike

    assert jnp.isnan(lorentz_residual(x_A, y_A, w_y, c)).all()
    assert jnp.isnan(jax.jit(lorentz_residual, static_argnums=(2, 3))(x_A, y_A, w_y, c)).all()

    module = LorentzResidual(weight_parameterization="identity", param_dtype=dtype)
    module.w_y_raw = nnx.Param(jnp.asarray(w_y, dtype=dtype))
    assert jnp.isnan(module(x_A, y_A, c=c)).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("w_y", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("c", [0.1, 1.0])
def test_lorentz_residual_gradients_match_abs_floor_form(dtype, w_y, c):
    """For ``w_y >= 0`` the gradients with respect to ``x``, ``y`` and ``w_y`` match the old form.

    ``d(c |m|)/dm = -c`` for ``m < 0`` and the floor passes the gradient through above ``eps``,
    so the backward multiplies by the same ``-c`` as ``d(-c m)/dm``: bit-identical in eager mode.
    Under ``jit``, XLA compiles the two graphs differently (at ``c = 1`` it drops the multiply by
    ``c``), and a few entries move by rounding: at most 1.5e-6 (float32) / 1.5e-15 (float64) of
    the pair's largest entry over the probe grid, less than the old form's own eager-vs-jit gap
    (up to 2.5e-5 / 2.1e-14). Under ``jit`` the test therefore bounds the difference instead.
    """
    x_NA, y_NA = _radius_pairs(c, dtype)
    cot_NA = jax.random.normal(jax.random.PRNGKey(400), x_NA.shape, dtype=dtype)
    w_N = jnp.asarray([w_y, 0.3 * w_y, 5.0 * w_y, 1e-3 * w_y, w_y], dtype=dtype)
    rtol = 1e-5 if dtype == jnp.float32 else 1e-13

    def new_loss(x, y, w):
        return jnp.sum(cot_NA * lorentz_residual(x, y, w, c))

    def old_loss(x, y, w):
        return jnp.sum(cot_NA * _abs_floor_lorentz_residual(x, y, w[..., None] if jnp.ndim(w) else w, c))

    for w in (jnp.asarray(w_y, dtype), w_N):
        for jit in (False, True):
            new_grad, old_grad = jax.grad(new_loss, argnums=(0, 1, 2)), jax.grad(old_loss, argnums=(0, 1, 2))
            if jit:
                new_grad, old_grad = jax.jit(new_grad), jax.jit(old_grad)
            for name, g_new, g_old in zip("xyw", new_grad(x_NA, y_NA, w), old_grad(x_NA, y_NA, w), strict=True):
                msg = f"d/d{name}, w.ndim={w.ndim}, jit={jit}"
                if not jit:
                    assert np.asarray(g_new).tobytes() == np.asarray(g_old).tobytes(), msg
                else:
                    scale = float(jnp.max(jnp.abs(g_old)))
                    assert float(jnp.max(jnp.abs(g_new - g_old))) <= rtol * scale, msg


# --------------------------------------------------------------------------- #
# Projection of the identity-mode w_y onto w_y >= 0 (projected gradient descent)
#
# Dimension key: N points  A ambient dim (= D + 1)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_project_w_y_clips_identity_and_leaves_others(dtype):
    """Identity mode: a negative raw w_y becomes exactly 0, a positive one is kept bit for bit.

    Softplus mode (whose raw param may be negative) and a fixed weight are left untouched.
    """
    identity = LorentzResidual(weight_parameterization="identity", param_dtype=dtype)
    identity.w_y_raw[...] = jnp.asarray(-0.3, dtype=dtype)
    identity.project_w_y()
    assert identity.w_y_raw[...].dtype == dtype
    assert float(identity.w_y_raw[...]) == 0.0

    identity.w_y_raw[...] = jnp.asarray(0.7, dtype=dtype)
    before = np.asarray(identity.w_y_raw[...]).tobytes()
    identity.project_w_y()
    assert np.asarray(identity.w_y_raw[...]).tobytes() == before

    softplus = LorentzResidual(init_w_y=0.1, param_dtype=dtype)  # raw = softplus^-1(0.1) < 0
    assert float(softplus.w_y_raw[...]) < 0.0
    before = np.asarray(softplus.w_y_raw[...]).tobytes()
    softplus.project_w_y()
    assert np.asarray(softplus.w_y_raw[...]).tobytes() == before

    fixed = LorentzResidual(learnable_weight=False, init_w_y=0.4, weight_parameterization="identity")
    fixed.project_w_y()
    assert fixed.w_y_raw is None and fixed._w_y == 0.4


class _InnerBlock(nnx.Module):
    def __init__(self):
        self.res = LorentzResidual(weight_parameterization="identity")


class _OuterModel(nnx.Module):
    def __init__(self):
        self.res = LorentzResidual(weight_parameterization="identity")
        self.blocks = nnx.List([_InnerBlock()])
        self.softplus_res = LorentzResidual(init_w_y=0.1)


def test_project_residual_weights_walks_nested_modules():
    """`project_residual_weights` projects every identity-mode LorentzResidual, nested ones included."""
    model = _OuterModel()
    model.res.w_y_raw[...] = jnp.asarray(-0.5, dtype=jnp.float32)
    model.blocks[0].res.w_y_raw[...] = jnp.asarray(-2.0, dtype=jnp.float32)
    softplus_raw = float(model.softplus_res.w_y_raw[...])

    project_residual_weights(model)

    assert float(model.res.w_y_raw[...]) == 0.0
    assert float(model.blocks[0].res.w_y_raw[...]) == 0.0
    assert float(model.softplus_res.w_y_raw[...]) == softplus_raw


class _ResidualModel(nnx.Module):
    def __init__(self, dtype):
        self.res = LorentzResidual(weight_parameterization="identity", param_dtype=dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_project_residual_weights_in_jitted_train_step(dtype):
    """SGD on a loss that pushes w_y down: without projection w_y goes negative and the output NaN.

    With the projection after `optimizer.update` (inside `nnx.jit`), w_y stays >= 0 (it sits at 0),
    the output stays finite and on the sheet, and w_y moves back up once the loss pulls it up.
    """
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    c = 1.0
    x_NA = _make_points(jax.random.PRNGKey(500), 8, 6, dtype, c)
    y_NA = _make_points(jax.random.PRNGKey(501), 8, 6, dtype, c)

    def dist_to_y(model):
        return jnp.mean(_dists(model.res(x_NA, y_NA, c=c), y_NA, c, dtype))

    def make_step(sign, project):
        @nnx.jit
        def step(model, optimizer):
            grads = nnx.grad(lambda m: sign * dist_to_y(m))(model)
            optimizer.update(model, grads)
            if project:
                project_residual_weights(model)

        return step

    for project in (False, True):
        model = _ResidualModel(dtype)
        optimizer = nnx.Optimizer(model, optax.sgd(1.0), wrt=nnx.Param)
        step = make_step(-1.0, project)  # maximize the distance to y: pushes w_y down
        w_y = []
        for _ in range(10):
            step(model, optimizer)
            w_y.append(float(model.res.w_y_raw[...]))
        out_NA = model.res(x_NA, y_NA, c=c)
        if not project:
            assert min(w for w in w_y if np.isfinite(w)) < 0.0, w_y
            assert not jnp.isfinite(out_NA).all()
        else:
            assert min(w_y) >= 0.0 and w_y[-1] == 0.0, w_y
            assert jnp.isfinite(out_NA).all()
            assert _check_on_hyperboloid(out_NA, c=c, atol=atol)

    # From w_y = 0, a loss that pulls toward y moves w_y back up.
    step = make_step(1.0, True)
    step(model, optimizer)
    assert float(model.res.w_y_raw[...]) > 0.0


def test_residual_w_y_gradient_nonzero_at_zero_matches_finite_differences():
    """At w_y = 0 the gradient with respect to w_y is nonzero and matches central finite differences.

    This is what lets a projected w_y leave 0 again. Float64; the oracle evaluates the forward
    at w_y = +-h (the -h side is still timelike for these x != y).
    """
    dtype, c, h = jnp.float64, 1.0, 1e-6
    x_NA = _make_points(jax.random.PRNGKey(502), 8, 6, dtype, c)
    y_NA = _make_points(jax.random.PRNGKey(503), 8, 6, dtype, c)
    cot_NA = jax.random.normal(jax.random.PRNGKey(504), x_NA.shape, dtype=dtype)
    module = LorentzResidual(weight_parameterization="identity", init_w_y=0.0, param_dtype=dtype)

    def loss(mod, weight=None):
        return jnp.sum(cot_NA * mod(x_NA, y_NA, c=c, weight=weight))

    grad_ad = float(nnx.grad(loss)(module).w_y_raw[...])
    grad_fd = float((loss(module, h) - loss(module, -h)) / (2 * h))
    assert abs(grad_fd) > 1e-2
    assert np.isclose(grad_ad, grad_fd, rtol=1e-7)
