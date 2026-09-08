"""Consumer regressions for origin derivatives of hyperbolic primitives.

Dimension key:
  N: batch or sequence length    D: spatial dimension
  A: ambient dimension (D + 1)  K: output dimension

The oracles below are literal geometric definitions in a valid spatial chart.
They deliberately do not call the manifold operation used by the consumer.
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix.decomposition import horo_projection
from hyperbolix.manifolds import Hyperboloid, ProperVelocity
from hyperbolix.nn_layers import (
    HyperbolicFullAttention,
    HyperboloidGyroBatchNorm,
    HypLinearHyperboloidBusemann,
    HypRegressionHyperboloidBusemann,
    LorentzResidual,
    ProperVelocityGyroBatchNorm,
)

DTYPES = [jnp.float32, jnp.float64]
C = 0.7


def _tol(dtype: jnp.dtype, *, tight: bool = False) -> float:
    if dtype == jnp.float32:
        return 2e-4 if tight else 8e-4
    return 2e-11 if tight else 2e-9


def _sheet_lift(spatial_D: jax.Array, c: float = C) -> jax.Array:
    """Spatial chart ``s -> (sqrt(1/c + ||s||^2), s)``."""
    inv_c = jnp.asarray(1.0 / c, dtype=spatial_D.dtype)
    time = jnp.sqrt(inv_c + jnp.sum(spatial_D**2, axis=-1))
    return jnp.concatenate([time[..., None], spatial_D], axis=-1)


def _literal_lorentz_mean(points_NA: jax.Array, weights_N: jax.Array, c: float = C) -> jax.Array:
    """Normalize the literal weighted ambient sum and rebuild its time slot."""
    h_A = jnp.einsum("n,na->a", weights_N, points_NA)
    denom = jnp.sqrt(c * (h_A[0] ** 2 - jnp.sum(h_A[1:] ** 2)))
    return _sheet_lift(h_A[1:] / denom, c)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("origin_argument", ["skip", "residual"])
def test_lorentz_residual_origin_jacobians_match_literal_two_point_mean(dtype, origin_argument):
    """The unchanged module keeps both point derivatives of the literal two-point mean."""
    w_y = 0.6
    other_D = jnp.asarray([0.35, -0.2], dtype=dtype)
    zero_D = jnp.zeros((2,), dtype=dtype)
    module = LorentzResidual(init_w_y=w_y, learnable_weight=False, param_dtype=dtype)

    def consumer(var_D):
        x_A = _sheet_lift(var_D if origin_argument == "skip" else other_D)
        y_A = _sheet_lift(var_D if origin_argument == "residual" else other_D)
        return module(x_A, y_A, c=C)

    def oracle(var_D):
        x_A = _sheet_lift(var_D if origin_argument == "skip" else other_D)
        y_A = _sheet_lift(var_D if origin_argument == "residual" else other_D)
        return _literal_lorentz_mean(jnp.stack([x_A, y_A]), jnp.asarray([1.0, w_y], dtype=dtype))

    expected_AD = jax.jacfwd(oracle)(zero_D)
    got_fwd_AD = jax.jacfwd(consumer)(zero_D)
    got_rev_AD = jax.jacrev(consumer)(zero_D)
    tangent_D = jnp.asarray([0.4, -0.7], dtype=dtype)
    _, got_jvp_A = jax.jit(lambda z, dz: jax.jvp(consumer, (z,), (dz,)))(zero_D, tangent_D)
    _, pullback = jax.vjp(consumer, zero_D)
    cotangent_A = jnp.asarray([0.2, -0.4, 0.7], dtype=dtype)

    atol = _tol(dtype)
    assert jnp.allclose(got_fwd_AD, expected_AD, atol=atol, rtol=atol)
    assert jnp.allclose(got_rev_AD, expected_AD, atol=atol, rtol=atol)
    assert jnp.allclose(got_jvp_A, expected_AD @ tangent_D, atol=atol, rtol=atol)
    assert jnp.allclose(pullback(cotangent_A)[0], cotangent_A @ expected_AD, atol=atol, rtol=atol)


def _radial_ratio_from_sq(sq: jax.Array, kind: str) -> jax.Array:
    """Stable sinhc/asinhc as a function of the squared argument."""
    small = sq < jnp.asarray(1e-6, dtype=sq.dtype)
    safe_sq = jnp.where(small, jnp.ones_like(sq), sq)
    z = jnp.sqrt(safe_sq)
    if kind == "sinh":
        ordinary = jnp.sinh(z) / z
        series = 1.0 + sq / 6.0 + sq**2 / 120.0
    else:
        ordinary = jnp.arcsinh(z) / z
        series = 1.0 - sq / 6.0 + 3.0 * sq**2 / 40.0
    return jnp.where(small, series, ordinary)


def _pv_log0(x_D: jax.Array, c: float) -> jax.Array:
    sq = c * jnp.sum(x_D**2, axis=-1, keepdims=True)
    return _radial_ratio_from_sq(sq, "asinh") * x_D


def _pv_exp0(v_D: jax.Array, c: float) -> jax.Array:
    sq = c * jnp.sum(v_D**2, axis=-1, keepdims=True)
    return _radial_ratio_from_sq(sq, "sinh") * v_D


def _inverse_boost_spatial(x_D: jax.Array, y_D: jax.Array, c: float) -> jax.Array:
    """Spatial part of the inverse Lorentz boost ``Lambda_x^-1 y``."""
    inv_c = jnp.asarray(1.0 / c, dtype=x_D.dtype)
    x0 = jnp.sqrt(inv_c + jnp.sum(x_D**2))
    y0 = jnp.sqrt(inv_c + jnp.sum(y_D**2))
    dot = jnp.dot(x_D, y_D)
    return y_D - jnp.sqrt(c) * y0 * x_D + c * dot * x_D / (1.0 + jnp.sqrt(c) * x0)


def _gyro_scale_spatial(x_D: jax.Array, factor: jax.Array, c: float) -> jax.Array:
    sq = c * jnp.sum(x_D**2)
    small = sq < jnp.asarray(1e-6, dtype=x_D.dtype)
    safe_sq = jnp.where(small, jnp.ones_like(sq), sq)
    z = jnp.sqrt(safe_sq)
    ordinary = jnp.sinh(factor * jnp.arcsinh(z)) / z
    series = factor + factor * (factor**2 - 1.0) * sq / 6.0
    return jnp.where(small, series, ordinary) * x_D


def _radial_distance_sq(x_D: jax.Array, c: float) -> jax.Array:
    """Squared distance from the origin as a smooth function of ``c ||x||^2``."""
    sq = c * jnp.sum(x_D**2, axis=-1)
    small = sq < jnp.asarray(1e-6, dtype=sq.dtype)
    safe_sq = jnp.where(small, jnp.ones_like(sq), sq)
    ordinary = jnp.arcsinh(jnp.sqrt(safe_sq)) ** 2 / c
    series = (sq - sq**2 / 3.0 + 8.0 * sq**3 / 45.0) / c
    return jnp.where(small, series, ordinary)


def _gyro_bn_oracle(spatial_ND: jax.Array, *, manifold_name: str, gamma: float, min_var: float, eps: float) -> jax.Array:
    """Literal train-mode GyroBN pipeline with zero gyro-bias."""
    if manifold_name == "hyperboloid":
        points_NA = _sheet_lift(spatial_ND)
        h_A = jnp.mean(points_NA, axis=0)
        denom = jnp.sqrt(C * (h_A[0] ** 2 - jnp.sum(h_A[1:] ** 2)))
        mean_D = h_A[1:] / denom
        points_ND = spatial_ND
    else:
        mean_D = _pv_exp0(jnp.mean(_pv_log0(spatial_ND, C), axis=0), C)
        points_ND = spatial_ND

    centered_ND = jax.vmap(_inverse_boost_spatial, in_axes=(None, 0, None))(mean_D, points_ND, C)
    var = jnp.maximum(jnp.mean(_radial_distance_sq(centered_ND, C)), jnp.asarray(min_var, dtype=spatial_ND.dtype))
    factor = jnp.asarray(gamma, dtype=spatial_ND.dtype) / jnp.sqrt(var + eps)
    return jax.vmap(_gyro_scale_spatial, in_axes=(0, None, None))(centered_ND, factor, C)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("manifold_name", ["hyperboloid", "pv"])
@pytest.mark.parametrize("batch_kind", ["all_origin", "one_origin"])
def test_gyro_batch_norm_origin_input_gradient_matches_closed_form_and_updates(dtype, manifold_name, batch_kind):
    """Train-mode input gradients agree with the closed form and drive an SGD embedding step."""
    D = 2
    if batch_kind == "all_origin":
        spatial_ND = jnp.zeros((3, D), dtype=dtype)
    else:
        spatial_ND = jnp.asarray([[0.0, 0.0], [0.45, -0.2], [-0.1, 0.3]], dtype=dtype)
    coeff_ND = jnp.asarray([[0.7, -0.2], [-0.4, 0.9], [0.3, -0.6]], dtype=dtype)
    gamma, min_var, eps = 0.8, 0.01, 1e-6

    if manifold_name == "hyperboloid":
        manifold = Hyperboloid(dtype=dtype)
        bn_cls = HyperboloidGyroBatchNorm

        def to_input(z):
            return _sheet_lift(z)

        spatial_slice = slice(1, None)
    else:
        manifold = ProperVelocity(dtype=dtype)
        bn_cls = ProperVelocityGyroBatchNorm

        def to_input(z):
            return z

        spatial_slice = slice(None)

    bn = bn_cls(manifold, num_features=D, min_var=min_var, eps=eps, param_dtype=dtype)
    bn.gamma[...] = jnp.asarray(gamma, dtype=dtype)

    def consumer_loss(model, spatial):
        out = model(to_input(spatial), c=C, use_running_average=False)
        return jnp.sum(out[..., spatial_slice] * coeff_ND)

    def oracle_loss(spatial):
        out_ND = _gyro_bn_oracle(spatial, manifold_name=manifold_name, gamma=gamma, min_var=min_var, eps=eps)
        return jnp.sum(out_ND * coeff_ND)

    _, got_grad_ND = nnx.value_and_grad(consumer_loss, argnums=1)(bn, spatial_ND)
    expected_grad_ND = jax.grad(oracle_loss)(spatial_ND)
    learning_rate = 0.03
    tx = optax.sgd(learning_rate)
    updates, _ = tx.update(got_grad_ND, tx.init(spatial_ND), spatial_ND)
    after_ND = optax.apply_updates(spatial_ND, updates)

    atol = _tol(dtype)
    assert jnp.allclose(got_grad_ND, expected_grad_ND, atol=atol, rtol=atol)
    assert jnp.linalg.norm(expected_grad_ND) > jnp.asarray(0.1, dtype=dtype)
    assert jnp.allclose(after_ND, spatial_ND - learning_rate * expected_grad_ND, atol=atol, rtol=atol)
    assert not jnp.array_equal(after_ND, spatial_ND)


def _set_attention_origin_projection_params(layer: HyperbolicFullAttention, value_kernel_AD: jax.Array) -> None:
    """Make Q/K/V equal the origin at zero input while retaining the chosen V Jacobian."""
    for projection in (layer.query_projections[0], layer.key_projections[0]):
        projection.kernel[...] = jnp.zeros_like(projection.kernel[...])
        projection.bias[...] = jnp.zeros_like(projection.bias[...])
    layer.value_projections[0].kernel[...] = value_kernel_AD
    layer.value_projections[0].bias[...] = jnp.zeros_like(layer.value_projections[0].bias[...])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_full_attention_origin_token_jacobian_matches_literal_masked_means(dtype, causal):
    """An exact-zero token keeps the literal midpoint derivative through the causal mask."""
    N = 3
    value_kernel_AD = jnp.asarray([[0.0, 0.0], [0.7, -0.2], [0.3, 0.6]], dtype=dtype)
    layer = HyperbolicFullAttention(3, 2, num_heads=1, param_dtype=dtype, rngs=nnx.Rngs(0))
    _set_attention_origin_projection_params(layer, value_kernel_AD)
    base_ND = jnp.asarray([[0.0, 0.0], [0.25, -0.15], [-0.2, 0.35]], dtype=dtype)

    def consumer(spatial_ND):
        x_1NA = _sheet_lift(spatial_ND)[None, ...]
        return layer(x_1NA, c_in=C, c_attn=C, c_out=C, causal=causal)[0, :, 1:]

    def oracle(spatial_ND):
        value_ND = jnp.matmul(spatial_ND, value_kernel_AD[1:, :])
        value_NA = _sheet_lift(value_ND)
        means = []
        uniform_N = jnp.full((N,), 1.0 / N, dtype=dtype)
        full_mean_A = _literal_lorentz_mean(value_NA, uniform_N)
        for n in range(N):
            if causal:
                visible = n + 1
                weights_N = jnp.full((visible,), 1.0 / visible, dtype=dtype)
                means.append(_literal_lorentz_mean(value_NA[:visible], weights_N)[1:])
            else:
                means.append(full_mean_A[1:])
        return jnp.stack(means)

    expected_NDMD = jax.jacfwd(oracle)(base_ND)
    expected_primal_ND = oracle(base_ND)
    for n in range(N):
        visible = n + 1 if causal else N
        assert jnp.array_equal(expected_NDMD[n, :, visible:, :], jnp.zeros_like(expected_NDMD[n, :, visible:, :]))

    got_fwd_NDMD = jax.jacfwd(consumer)(base_ND)
    got_rev_NDMD = jax.jacrev(consumer)(base_ND)
    tangent_ND = jnp.asarray([[0.2, -0.3], [0.4, 0.1], [-0.5, 0.6]], dtype=dtype)
    primal_ND, got_jvp_ND = jax.jit(lambda z, dz: jax.jvp(consumer, (z,), (dz,)))(base_ND, tangent_ND)

    atol = _tol(dtype)
    assert jnp.allclose(primal_ND, expected_primal_ND, atol=atol, rtol=atol)
    assert jnp.allclose(got_fwd_NDMD, expected_NDMD, atol=atol, rtol=atol)
    assert jnp.allclose(got_rev_NDMD, expected_NDMD, atol=atol, rtol=atol)
    assert jnp.allclose(got_jvp_ND, jnp.einsum("ndmi,mi->nd", expected_NDMD, tangent_ND), atol=atol, rtol=atol)
    if causal:
        assert jnp.array_equal(got_fwd_NDMD[0, :, 1:, :], jnp.zeros_like(got_fwd_NDMD[0, :, 1:, :]))


BUSEMANN_HEADS = [
    pytest.param("regression", id="regression"),
    pytest.param("linear", id="linear"),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_kind", BUSEMANN_HEADS)
def test_busemann_head_zero_embedding_gradient_matches_normalized_kernel_and_updates(dtype, head_kind):
    """A zero embedding sees the analytic normalized-kernel gradient and takes an SGD step."""
    kernel_KD = jnp.asarray([[3.0, 4.0], [-2.0, 1.0]], dtype=dtype)
    alpha_K = jnp.asarray([1.7, 0.6], dtype=dtype)
    manifold = Hyperboloid(dtype=dtype)
    if head_kind == "regression":
        head = HypRegressionHyperboloidBusemann(manifold, 3, 2, rngs=nnx.Rngs(0), param_dtype=dtype)

        def spatial_output(y):
            return y

    else:
        head = HypLinearHyperboloidBusemann(manifold, 3, 3, rngs=nnx.Rngs(0), param_dtype=dtype)

        def spatial_output(y):
            return y[1:]

    head.kernel[...] = kernel_KD
    head.log_scale[...] = jnp.log(alpha_K)
    head.bias[...] = jnp.zeros((2,), dtype=dtype)

    def consumer(spatial_D):
        return spatial_output(head(_sheet_lift(spatial_D)[None, :], c=C)[0])

    expected_KD = alpha_K[:, None] * kernel_KD / jnp.linalg.norm(kernel_KD, axis=-1, keepdims=True)
    zero_D = jnp.zeros((2,), dtype=dtype)
    got_fwd_KD = jax.jacfwd(consumer)(zero_D)
    got_rev_KD = jax.jacrev(consumer)(zero_D)
    tangent_D = jnp.asarray([0.3, -0.5], dtype=dtype)
    _, got_jvp_K = jax.jit(lambda z, dz: jax.jvp(consumer, (z,), (dz,)))(zero_D, tangent_D)
    cotangent_K = jnp.asarray([0.8, -0.4], dtype=dtype)
    _, pullback = jax.vjp(consumer, zero_D)
    got_grad_D = pullback(cotangent_K)[0]
    batched_zero_ND = jnp.zeros((3, 2), dtype=dtype)
    vmapped_NK = jax.jit(jax.vmap(consumer))(batched_zero_ND)

    learning_rate = 0.05
    tx = optax.sgd(learning_rate)
    updates, _ = tx.update(got_grad_D, tx.init(zero_D), zero_D)
    after_D = optax.apply_updates(zero_D, updates)
    expected_grad_D = cotangent_K @ expected_KD

    atol = _tol(dtype, tight=True)
    assert jnp.allclose(got_fwd_KD, expected_KD, atol=atol, rtol=atol)
    assert jnp.allclose(got_rev_KD, expected_KD, atol=atol, rtol=atol)
    assert jnp.allclose(got_jvp_K, expected_KD @ tangent_D, atol=atol, rtol=atol)
    assert jnp.allclose(got_grad_D, expected_grad_D, atol=atol, rtol=atol)
    assert jnp.allclose(vmapped_NK, 0.0, atol=atol)
    assert jnp.allclose(after_D, -learning_rate * expected_grad_D, atol=atol, rtol=atol)
    assert jnp.linalg.norm(after_D) > jnp.asarray(0.01, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_horopca_origin_busemann_coordinate_derivative_matches_ideal_direction(dtype):
    """The K=1 projection preserves the independent Busemann derivative ``-q`` at the origin."""
    q_D = jnp.asarray([3.0, 4.0], dtype=dtype) / 5.0
    zero_D = jnp.zeros((2,), dtype=dtype)

    def projected_coordinate(spatial_D):
        projected_A = horo_projection(_sheet_lift(spatial_D), q_D[None, :], C)
        arg = jnp.sqrt(C) * (projected_A[0] - jnp.dot(projected_A[1:], q_D))
        return jnp.log(arg) / jnp.sqrt(C)

    got_fwd_D = jax.jacfwd(projected_coordinate)(zero_D)
    got_rev_D = jax.jacrev(projected_coordinate)(zero_D)
    tangent_D = jnp.asarray([0.2, -0.7], dtype=dtype)
    _, got_jvp = jax.jit(lambda z, dz: jax.jvp(projected_coordinate, (z,), (dz,)))(zero_D, tangent_D)
    _, pullback = jax.vjp(projected_coordinate, zero_D)

    atol = _tol(dtype)
    assert jnp.allclose(got_fwd_D, -q_D, atol=atol, rtol=atol)
    assert jnp.allclose(got_rev_D, -q_D, atol=atol, rtol=atol)
    assert jnp.allclose(got_jvp, -jnp.dot(q_D, tangent_D), atol=atol, rtol=atol)
    assert jnp.allclose(pullback(jnp.asarray(1.0, dtype=dtype))[0], -q_D, atol=atol, rtol=atol)
