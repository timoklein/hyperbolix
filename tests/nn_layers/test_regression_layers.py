"""Tests for hyperbolic regression neural network layers.

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypRegression{Poincare,PoincarePP,Hyperboloid}-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import (
    HypRegressionHyperboloid,
    HypRegressionPoincare,
    HypRegressionPoincarePP,
)


def get_poincare(dtype: jnp.dtype) -> Poincare:
    """Get dtype-specific Poincaré manifold instance."""
    return Poincare(dtype=dtype)


def get_hyperboloid(dtype: jnp.dtype) -> Hyperboloid:
    """Get dtype-specific Hyperboloid manifold instance."""
    return Hyperboloid(dtype=dtype)


# --------------------------------------------------------------------------- #
# MLR decision-boundary oracles (audit A9-06)
#
# A hyperbolic MLR head returns a *signed* distance to a learned hyperplane. The
# three properties below are the definition of that object and are independent of
# how the library computes it, so they pin the logit sign that the shape/finiteness
# tests (and the shared layer contract) leave completely free:
#   1. zero on the hyperplane,
#   2. sign follows the side of the hyperplane the point is on,
#   3. magnitude grows monotonically with the geodesic margin.
# --------------------------------------------------------------------------- #
_MARGINS = jnp.array([-0.6, -0.3, -0.1, 0.1, 0.3, 0.6])


def _assert_signed_distance_semantics(logits_T, margins_T):
    """Logits must be negative/positive on the two sides and increase with the margin."""
    assert bool(jnp.all(logits_T[margins_T < 0] < 0.0)), f"expected negative logits below the hyperplane: {logits_T}"
    assert bool(jnp.all(logits_T[margins_T > 0] > 0.0)), f"expected positive logits above the hyperplane: {logits_T}"
    assert bool(jnp.all(jnp.diff(logits_T) > 0.0)), f"logit not monotone in the margin: {logits_T}"


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_hyp_regression_poincare_pp_decision_boundary(c):
    """HNN++ head (Shimizu et al. 2020): signed-distance semantics of the logits."""
    dtype = jnp.float64
    manifold = get_poincare(dtype)
    in_dim, out_dim = 3, 2

    layer = HypRegressionPoincarePP(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)
    kernel_PD = jnp.array([[0.7, -0.3, 0.2], [-0.4, 0.5, 0.1]], dtype=dtype)
    bias_P1 = jnp.array([[0.4], [-0.25]], dtype=dtype)
    layer.kernel[...] = kernel_PD
    layer.bias[...] = bias_P1

    for k in range(out_dim):
        # The hyperplane of class k passes through q_k = exp_0(r_k * z_hat_k),
        # oriented along z_hat_k; walking along that direction crosses it at r_k.
        z_hat_D = kernel_PD[k] / jnp.linalg.norm(kernel_PD[k])
        offsets_T = bias_P1[k, 0] + _MARGINS

        q_D = manifold.expmap_0(bias_P1[k, 0] * z_hat_D, c)
        assert abs(float(layer(q_D[None, :], c=c)[0, k])) < 1e-10

        pts_TD = jax.vmap(manifold.expmap_0, in_axes=(0, None))(offsets_T[:, None] * z_hat_D[None, :], c)
        _assert_signed_distance_semantics(layer(pts_TD, c=c)[:, k], _MARGINS)


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_hyp_regression_poincare_decision_boundary(c):
    """Ganea et al. 2018 head: signed-distance semantics of the logits.

    Here the hyperplane is stored explicitly as a base point ``p`` plus a tangent
    normal that the layer parallel-transports to ``p``, so the boundary is ``p``
    itself and the margin axis is the transported normal.
    """
    dtype = jnp.float64
    manifold = get_poincare(dtype)
    in_dim, out_dim = 3, 2

    layer = HypRegressionPoincare(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), curvature=c, param_dtype=dtype)
    kernel_PD = jnp.array([[0.7, -0.3, 0.2], [-0.4, 0.5, 0.1]], dtype=dtype)
    bias_PD = jnp.array([[0.15, 0.05, -0.1], [-0.2, 0.1, 0.05]], dtype=dtype)
    layer.kernel[...] = kernel_PD
    layer.bias[...] = bias_PD

    for k in range(out_dim):
        p_D = manifold.proj(bias_PD[k], c)
        assert abs(float(layer(p_D[None, :], c=c)[0, k])) < 1e-10

        a_D = manifold.ptransp_0(kernel_PD[k], p_D, c)
        a_hat_D = a_D / jnp.linalg.norm(a_D)
        pts_TD = jax.vmap(manifold.expmap, in_axes=(0, None, None))(_MARGINS[:, None] * a_hat_D[None, :], p_D, c)
        _assert_signed_distance_semantics(layer(pts_TD, c=c)[:, k], _MARGINS)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_regression_poincare_kernel_init_std(dtype):
    """Kernel init matches its Poincaré siblings: std = (2 * in_dim * out_dim)^{-0.5}.

    The Ganea head used a bare ``normal(0, 1)``. The ``||a||`` factors cancel inside the
    ``asinh`` of ``_compute_mlr`` but reappear as the outer ``||a||`` multiplier, so the
    logits scale linearly with the row norm ``~= sqrt(in_dim)`` — the same failure the
    HNN++ and Hyperboloid heads already guard against.
    """
    in_dim, out_dim = 65, 10
    layer = HypRegressionPoincare(get_poincare(dtype), in_dim, out_dim, rngs=nnx.Rngs(42), param_dtype=dtype)

    kernel_std = jnp.std(layer.kernel[...])
    expected_std = 1.0 / jnp.sqrt(2.0 * in_dim * out_dim)
    assert jnp.abs(kernel_std - expected_std) < 0.2 * expected_std

    # Row norms must be O(1)-small, not O(sqrt(in_dim)) as with the unscaled init.
    row_norms = jnp.linalg.norm(layer.kernel[...], axis=-1)
    assert float(jnp.max(row_norms)) < 1.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyp_regression_hyperboloid_kernel_init_std(dtype):
    """Kernel init is fan-scaled by the spatial fan-in (in_dim - 1); an unscaled N(0,1)
    would give row norms ~= sqrt(in_dim - 1), overwhelming the MLR output scaling."""
    in_dim, out_dim = 65, 10
    rngs = nnx.Rngs(42)
    layer = HypRegressionHyperboloid(get_hyperboloid(dtype), in_dim, out_dim, rngs=rngs)

    kernel_std = jnp.std(layer.kernel[...])
    expected_std = 1.0 / jnp.sqrt(2.0 * (in_dim - 1) * out_dim)
    assert jnp.abs(kernel_std - expected_std) < 0.2 * expected_std


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_linear_then_regression_poincare(dtype):
    """Test linear layer followed by regression layer."""
    from hyperbolix.nn_layers import HypLinearPoincare

    key = jax.random.PRNGKey(42)
    batch_size, in_dim, hidden_dim, out_dim = 4, 5, 8, 3

    # Create input
    x = jax.random.normal(key, (batch_size, in_dim), dtype=dtype) * 0.1
    x = get_poincare(dtype).proj(x, c=1.0)

    # Create layers
    rngs = nnx.Rngs(42)
    linear = HypLinearPoincare(get_poincare(dtype), in_dim, hidden_dim, rngs=rngs)
    regression = HypRegressionPoincarePP(get_poincare(dtype), hidden_dim, out_dim, rngs=rngs)

    # Forward pass
    h = linear(x, c=1.0)
    y = regression(h, c=1.0)

    # Check output shape
    assert y.shape == (batch_size, out_dim)
    # Check output is finite
    assert jnp.isfinite(y).all()


# --------------------------------------------------------------------------- #
# Zero-row hyperplane normal (a dead output channel)
#
# Both MLR helpers divide by the Euclidean norm of the per-class hyperplane
# normal and floor that norm. The floor makes the *forward* value finite, but a
# bare ``jnp.linalg.norm`` has VJP ``z/‖z‖ = 0/0 = NaN`` at the zero vector, and
# the floor cannot remove it: the NaN is created inside the norm's own VJP and
# the floor's zero cotangent meets it as ``0 * NaN = NaN``, which then poisons
# the gradient of *every* row, not just the dead one. ``safe_norm``'s
# double-``where`` gives an exact 0 value and an exactly-zero VJP there.
# --------------------------------------------------------------------------- #
_ZERO_ROW = 1


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hyperboloid_mlr_gradient_finite_with_a_zero_normal_row(dtype, jit):
    """``compute_mlr``: an exactly-zero row of ``z`` gives a finite gradient, zero on that row."""
    c, batch, dim, out_dim = 1.0, 6, 4, 3
    manifold = get_hyperboloid(dtype)

    key_x, key_z, key_r = jax.random.split(jax.random.PRNGKey(0), 3)
    spatial_BD = jax.random.normal(key_x, (batch, dim), dtype=dtype) * 0.5
    x_BA = jax.vmap(manifold.proj, in_axes=(0, None))(jnp.concatenate([jnp.zeros((batch, 1), dtype), spatial_BD], axis=-1), c)
    z_PD = jax.random.normal(key_z, (out_dim, dim), dtype=dtype).at[_ZERO_ROW].set(0.0)
    r_P1 = jax.random.normal(key_r, (out_dim, 1), dtype=dtype) * 0.3

    def loss(z):
        return jnp.sum(manifold.compute_mlr(x_BA, z, r_P1, c, 1.0, 50.0) ** 2)

    grad_fn = jax.jit(jax.grad(loss)) if jit else jax.grad(loss)
    g_PD = grad_fn(z_PD)

    assert jnp.isfinite(g_PD).all(), f"non-finite gradient with a zero normal row: {g_PD}"
    # The norm branch contributes exactly nothing (``safe_norm``'s VJP at 0 is 0). What is left
    # for the dead row is the genuine ``d⟨x_s, z⟩`` path, which is real but scaled by the whole
    # expression's ``min_enorm`` prefactor: measured max |g| 1.8e-15 (f32) / 3.0e-15 (f64) here,
    # against ~10 for the live rows. So this bound is "the dead row is inert", not a tolerance.
    assert jnp.max(jnp.abs(g_PD[_ZERO_ROW])) < 1e-12, f"zero row is not inert: {g_PD[_ZERO_ROW]}"
    # The live rows must still get a real gradient — a finite-but-all-zero result would
    # pass the two assertions above while being just as broken.
    assert jnp.max(jnp.abs(jnp.delete(g_PD, _ZERO_ROW, axis=0))) > 1e-3


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_poincare_regression_gradient_finite_with_a_zero_normal_row(dtype, jit):
    """``HypRegressionPoincare``: an exactly-zero row of ``a`` gives a finite gradient."""
    c, batch, in_dim, out_dim = 1.0, 6, 4, 3
    manifold = get_poincare(dtype)
    layer = HypRegressionPoincare(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), param_dtype=dtype)

    x_BD = jax.vmap(manifold.proj, in_axes=(0, None))(
        jax.random.normal(jax.random.PRNGKey(0), (batch, in_dim), dtype=dtype) * 0.5, c
    )
    a_PD = jnp.asarray(layer.kernel[...]).at[_ZERO_ROW].set(0.0)
    p_PD = jax.vmap(manifold.proj, in_axes=(0, None))(jnp.asarray(layer.bias[...]), c)

    def loss(a):
        return jnp.sum(layer._compute_mlr(x_BD, a, p_PD, c) ** 2)

    grad_fn = jax.jit(jax.grad(loss)) if jit else jax.grad(loss)
    g_PD = grad_fn(a_PD)

    assert jnp.isfinite(g_PD).all(), f"non-finite gradient with a zero normal row: {g_PD}"
    assert jnp.all(g_PD[_ZERO_ROW] == 0.0), f"zero row must receive an exactly-zero gradient: {g_PD[_ZERO_ROW]}"
    assert jnp.any(jnp.abs(jnp.delete(g_PD, _ZERO_ROW, axis=0)) > 0.0)
