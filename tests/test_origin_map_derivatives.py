"""Origin and chart-switch derivatives for Hyperboloid maps and their PV lifts.

Dimension key: D spatial dimension; A ambient dimension (D + 1); B batch size; P paired spatial width (2*D).
References use NumPy longdouble geometry or converged float64 finite differences.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.manifolds import Hyperboloid, ProperVelocity
from hyperbolix.manifolds.hyperboloid import _logmap_impl


def _np_lift(s_D: np.ndarray, c: float) -> np.ndarray:
    s_D = np.asarray(s_D, dtype=np.float64)
    return np.concatenate(([np.sqrt(1.0 / c + np.dot(s_D, s_D))], s_D))


def _j_lift(s_D: jax.Array, c: jax.Array | float) -> jax.Array:
    c = jnp.asarray(c, dtype=s_D.dtype)
    return jnp.concatenate((jnp.sqrt(1.0 / c + jnp.dot(s_D, s_D))[None], s_D))


def _np_tangent(s_D: np.ndarray, w_D: np.ndarray, c: float) -> np.ndarray:
    x_A = _np_lift(s_D, c)
    w_D = np.asarray(w_D, dtype=np.float64)
    return np.concatenate(([np.dot(s_D, w_D) / x_A[0]], w_D))


def _j_tangent(s_D: jax.Array, w_D: jax.Array, c: jax.Array | float) -> jax.Array:
    x_A = _j_lift(s_D, c)
    return jnp.concatenate(((jnp.dot(s_D, w_D) / x_A[0])[None], w_D))


def _np_lorentz(a_A: np.ndarray, b_A: np.ndarray) -> float:
    return float(-a_A[0] * b_A[0] + np.dot(a_A[1:], b_A[1:]))


def _np_gyro_difference(x_D: np.ndarray, y_D: np.ndarray, c: float) -> np.ndarray:
    x_A, y_A = _np_lift(x_D, c), _np_lift(y_D, c)
    gamma = np.sqrt(c) * x_A[0]
    xy = np.dot(x_D, y_D)
    s_D = y_D - np.sqrt(c) * y_A[0] * x_D + c * xy * x_D / (1.0 + gamma)
    return _np_lift(s_D, c)


def _np_logmap(x_D: np.ndarray, y_D: np.ndarray, c: float) -> np.ndarray:
    x_A, y_A = _np_lift(x_D, c), _np_lift(y_D, c)
    alpha = -c * _np_lorentz(x_A, y_A)
    theta = np.arccosh(alpha)
    return theta * (y_A - alpha * x_A) / np.sqrt(alpha * alpha - 1.0)


def _np_ptransp(x_D: np.ndarray, y_D: np.ndarray, w_D: np.ndarray, c: float) -> np.ndarray:
    x_A, y_A = _np_lift(x_D, c), _np_lift(y_D, c)
    v_A = _np_tangent(x_D, w_D, c)
    scale = _np_lorentz(v_A, y_A) / (1.0 / c - _np_lorentz(x_A, y_A))
    return v_A + scale * (x_A + y_A)


def _central_jacobian(fn: Callable[[np.ndarray], np.ndarray], x_D: np.ndarray, h: float) -> np.ndarray:
    x_D = np.asarray(x_D, dtype=np.float64)
    y = np.asarray(fn(x_D), dtype=np.float64)
    jac = np.empty(y.shape + x_D.shape, dtype=np.float64)
    for i in range(x_D.size):
        step_D = np.zeros_like(x_D)
        step_D[i] = h
        jac[..., i] = (np.asarray(fn(x_D + step_D)) - np.asarray(fn(x_D - step_D))) / (2.0 * h)
    return jac


def _fd_jacobian(fn: Callable[[np.ndarray], np.ndarray], x_D: np.ndarray) -> np.ndarray:
    j5 = _central_jacobian(fn, x_D, 1e-5)
    j6 = _central_jacobian(fn, x_D, 1e-6)
    np.testing.assert_allclose(j6, j5, rtol=4e-6, atol=4e-7)
    return j6


def _fd_scalar(fn: Callable[[float], np.ndarray], c: float) -> np.ndarray:
    d5 = (np.asarray(fn(c + 1e-5)) - np.asarray(fn(c - 1e-5))) / 2e-5
    d6 = (np.asarray(fn(c + 1e-6)) - np.asarray(fn(c - 1e-6))) / 2e-6
    np.testing.assert_allclose(d6, d5, rtol=8e-6, atol=5e-7)
    return d6


def _ld_busemann(s_D: np.ndarray, v_D: np.ndarray, c: float) -> tuple[float, np.ndarray]:
    ld = np.longdouble
    s_D = np.asarray(s_D, dtype=ld)
    v_D = np.asarray(v_D, dtype=ld)
    # The public contract is a unit direction. Normalize the stored components in higher precision
    # so float32 storage error is measured separately from the geometric direction it represents.
    v_D = v_D / np.sqrt(np.dot(v_D, v_D))
    c_ld = ld(c)
    x0 = np.sqrt(ld(1.0) / c_ld + np.dot(s_D, s_D))
    q = np.dot(s_D, v_D)
    gradient_D = (s_D / x0 - v_D) / (np.sqrt(c_ld) * (x0 - q))
    with localcontext() as context:
        context.prec = 80
        s_decimal = [Decimal.from_float(float(value)) for value in np.asarray(s_D, dtype=np.float64)]
        v_decimal = [Decimal.from_float(float(value)) for value in np.asarray(v_D, dtype=np.float64)]
        c_decimal = Decimal.from_float(float(c))
        v_norm = sum(value * value for value in v_decimal).sqrt()
        v_decimal = [value / v_norm for value in v_decimal]
        x0_decimal = (Decimal(1) / c_decimal + sum(value * value for value in s_decimal)).sqrt()
        q_decimal = sum(s * v for s, v in zip(s_decimal, v_decimal, strict=True))
        sqrt_c = c_decimal.sqrt()
        value = (sqrt_c * (x0_decimal - q_decimal)).ln() / sqrt_c
    return float(value), np.asarray(gradient_D, dtype=np.float64)


def _ld_rationalized_busemann_gradient(s_D: np.ndarray, v_D: np.ndarray, c: float) -> np.ndarray:
    """Analytic sheet gradient of the q/p rationalized expression at stored inputs."""
    ld = np.longdouble
    s_D = np.asarray(s_D, dtype=ld)
    v_D = np.asarray(v_D, dtype=ld)
    c_ld = ld(c)
    x0 = np.sqrt(ld(1.0) / c_ld + np.dot(s_D, s_D))
    q = np.dot(s_D, v_D)
    if q < 0:
        return np.asarray((s_D / x0 - v_D) / (np.sqrt(c_ld) * (x0 - q)), dtype=np.float64)
    p_D = s_D - q * v_D
    numerator = ld(1.0) / c_ld + np.dot(p_D, p_D)
    denominator = x0 + q
    dp_transpose_p_D = p_D - v_D * np.dot(v_D, p_D)
    dlog_numerator_D = ld(2.0) * dp_transpose_p_D / numerator
    dlog_denominator_D = (s_D / x0 + v_D) / denominator
    return np.asarray((dlog_numerator_D - dlog_denominator_D) / np.sqrt(c_ld), dtype=np.float64)


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
@pytest.mark.parametrize("v_D", [np.array([1.0, 0.0]), np.array([0.6, 0.8])])
def test_busemann_origin_sheet_gradient_is_negative_direction(c: float, v_D: np.ndarray) -> None:
    manifold = Hyperboloid(dtype=jnp.float64)
    s_D = jnp.zeros(2, dtype=jnp.float64)
    v_D = v_D / np.linalg.norm(v_D)
    got_D = jax.grad(lambda s: manifold.busemann(_j_lift(s, c), jnp.asarray(v_D), c))(s_D)
    np.testing.assert_allclose(got_D, -v_D, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("radius", [8.0, 10.0, 12.0, 14.0])
@pytest.mark.parametrize("angle", [0.0, 1e-4, 0.4], ids=["aligned", "nearly-aligned", "arbitrary"])
def test_busemann_float64_value_and_gradient_match_longdouble(radius: float, angle: float) -> None:
    c = 1.0
    ray_D = np.array([0.6, 0.8])
    orthogonal_D = np.array([-0.8, 0.6])
    s_D = np.sinh(radius) * ray_D
    v_D = np.cos(angle) * ray_D + np.sin(angle) * orthogonal_D
    v_D /= np.linalg.norm(v_D)
    expected, expected_grad_D = _ld_busemann(s_D, v_D, c)
    manifold = Hyperboloid(dtype=jnp.float64)
    s, v = jnp.asarray(s_D), jnp.asarray(v_D)
    got = manifold.busemann(_j_lift(s, c), v, c)
    got_grad_D = jax.grad(lambda z: manifold.busemann(_j_lift(z, c), v, c))(s)
    np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(got_grad_D, expected_grad_D, rtol=3e-8, atol=3e-10)


@pytest.mark.parametrize("radius", [8.0, 10.0, 12.0, 14.0])
@pytest.mark.parametrize("angle", [0.0, 1e-4, 0.4], ids=["aligned", "nearly-aligned", "arbitrary"])
def test_busemann_float32_compares_against_stored_inputs(radius: float, angle: float) -> None:
    c = 1.0
    ray_D = np.array([0.6, 0.8])
    orthogonal_D = np.array([-0.8, 0.6])
    stored_s_D = np.asarray(np.sinh(radius) * ray_D, dtype=np.float32)
    direction_D = np.cos(angle) * ray_D + np.sin(angle) * orthogonal_D
    stored_v_D = np.asarray(direction_D / np.linalg.norm(direction_D), dtype=np.float32)
    expected, geometric_grad_D = _ld_busemann(stored_s_D.astype(np.float64), stored_v_D.astype(np.float64), c)
    arithmetic_grad_D = _ld_rationalized_busemann_gradient(stored_s_D.astype(np.float64), stored_v_D.astype(np.float64), c)
    # A float32 vector is generally not exactly unit length when promoted. Keep that direction-storage
    # effect out of the q/p derivative reference; the value reference above renormalizes it in Decimal80.
    stored_norm_error = abs(float(np.dot(stored_v_D.astype(np.float64), stored_v_D.astype(np.float64))) - 1.0)
    assert np.isfinite(stored_norm_error)
    manifold = Hyperboloid(dtype=jnp.float32)
    s, v = jnp.asarray(stored_s_D), jnp.asarray(stored_v_D)
    got = manifold.busemann(_j_lift(s, c), v, c)
    got_grad_D = jax.grad(lambda z: manifold.busemann(_j_lift(z, c), v, c))(s)
    np.testing.assert_allclose(got, expected, rtol=4e-4, atol=4e-5)
    # At radius a, an O(eps) stored angular error is amplified by O(exp(a)) in the aligned
    # Busemann derivative. This bound isolates input representation from arithmetic error.
    storage_bound = 2.0 * np.finfo(np.float32).eps * np.exp(radius)
    assert np.max(np.abs(np.asarray(got_grad_D) - geometric_grad_D)) <= storage_bound
    assert np.max(np.abs(np.asarray(got_grad_D) - arithmetic_grad_D)) <= storage_bound

    _, generated_grad_D = _ld_busemann(np.sinh(radius) * ray_D, direction_D, c)
    assert np.max(np.abs(geometric_grad_D - generated_grad_D)) <= storage_bound


def test_busemann_normalized_direction_and_curvature_derivatives_match_fd() -> None:
    manifold = Hyperboloid(dtype=jnp.float64)
    s_D = np.sinh(8.0) * np.array([0.6, 0.8])
    raw_v_D = np.array([0.61, 0.79])
    c = 0.7

    def jax_direction(raw_D):
        unit_D = raw_D / jnp.sqrt(jnp.dot(raw_D, raw_D))
        return manifold.busemann(_j_lift(jnp.asarray(s_D), c), unit_D, c)

    def np_direction(raw_D):
        return np.asarray(_ld_busemann(s_D, raw_D, c)[0])

    direction_D = np.asarray(jax.jacrev(jax_direction)(jnp.asarray(raw_v_D)))
    np.testing.assert_allclose(direction_D, _fd_jacobian(np_direction, raw_v_D), rtol=2e-6, atol=2e-7)

    unit_v_D = raw_v_D / np.linalg.norm(raw_v_D)

    def jax_curvature(cc):
        return manifold.busemann(_j_lift(jnp.asarray(s_D), cc), jnp.asarray(unit_v_D), cc)

    def np_curvature(cc):
        return np.asarray(_ld_busemann(s_D, unit_v_D, float(cc))[0])

    dc = np.asarray(jax.jacfwd(jax_curvature)(jnp.asarray(c)))
    np.testing.assert_allclose(dc, _fd_scalar(np_curvature, c), rtol=5e-6, atol=5e-7)


def _hyperboloid_map(
    operation: str,
    x_D: jax.Array,
    y_D: jax.Array,
    w_D: jax.Array,
    c: jax.Array | float,
) -> jax.Array:
    manifold = Hyperboloid(dtype=x_D.dtype)
    x_A, y_A = _j_lift(x_D, c), _j_lift(y_D, c)
    if operation == "difference":
        return manifold.gyro_difference(x_A, y_A, c)
    if operation == "logmap":
        return manifold.logmap(y_A, x_A, c)
    return manifold.ptransp(_j_tangent(x_D, w_D, c), x_A, y_A, c)


def _numpy_map(operation: str, x_D: np.ndarray, y_D: np.ndarray, w_D: np.ndarray, c: float) -> np.ndarray:
    if operation == "difference":
        return _np_gyro_difference(x_D, y_D, c)
    if operation == "logmap":
        return _np_logmap(x_D, y_D, c)
    return _np_ptransp(x_D, y_D, w_D, c)


@pytest.mark.parametrize("operation", ["difference", "ptransp", "logmap"])
@pytest.mark.parametrize("endpoint", ["base", "target"])
def test_hyperboloid_both_origin_endpoint_jacobians_match_fd(operation: str, endpoint: str) -> None:
    c = 1.0
    fixed_D = np.array([0.7, -0.4])
    origin_D = np.zeros(2)
    w_D = np.array([0.3, 0.2])

    def jax_fn(z):
        if endpoint == "base":
            return _hyperboloid_map(operation, z, jnp.asarray(fixed_D), jnp.asarray(w_D), c)
        return _hyperboloid_map(operation, jnp.asarray(fixed_D), z, jnp.asarray(w_D), c)

    def np_fn(z):
        if endpoint == "base":
            return _numpy_map(operation, z, fixed_D, w_D, c)
        return _numpy_map(operation, fixed_D, z, w_D, c)

    fwd_AD = np.asarray(jax.jacfwd(jax_fn)(jnp.asarray(origin_D)))
    rev_AD = np.asarray(jax.jacrev(jax_fn)(jnp.asarray(origin_D)))
    expected_AD = _fd_jacobian(np_fn, origin_D)
    np.testing.assert_allclose(fwd_AD, expected_AD, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(rev_AD, expected_AD, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("operation", ["difference", "ptransp", "logmap"])
@pytest.mark.parametrize("c", [0.3, 2.5])
def test_hyperboloid_origin_curvature_derivative_uses_valid_lifts(operation: str, c: float) -> None:
    x_D = np.zeros(2)
    y_D = np.array([0.7, -0.4])
    w_D = np.array([0.3, 0.2])

    def jax_fn(cc):
        return _hyperboloid_map(operation, jnp.asarray(x_D), jnp.asarray(y_D), jnp.asarray(w_D), cc)

    def np_fn(cc):
        return _numpy_map(operation, x_D, y_D, w_D, float(cc))

    got_A = np.asarray(jax.jacfwd(jax_fn)(jnp.asarray(c)))
    expected_A = _fd_scalar(np_fn, c)
    np.testing.assert_allclose(got_A, expected_A, rtol=8e-6, atol=8e-7)


@pytest.mark.parametrize("operation", ["difference", "ptransp", "logmap"])
@pytest.mark.parametrize("operand", ["base", "target"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("radius", [0.99, 1.01])
def test_threshold_sides_and_both_operands_match_independent_fd(
    operation: str, operand: str, sign: float, radius: float
) -> None:
    c = 1.0
    variable_D = np.array([sign * radius, 0.0])
    far_D = np.array([1.4, 0.0])
    w_D = np.array([0.2, -0.3])

    def jax_fn(z):
        if operand == "base":
            return _hyperboloid_map(operation, z, jnp.asarray(far_D), jnp.asarray(w_D), c)
        return _hyperboloid_map(operation, jnp.asarray(far_D), z, jnp.asarray(w_D), c)

    def np_fn(z):
        if operand == "base":
            return _numpy_map(operation, z, far_D, w_D, c)
        return _numpy_map(operation, far_D, z, w_D, c)

    expected_AD = _fd_jacobian(np_fn, variable_D)
    np.testing.assert_allclose(jax.jacfwd(jax_fn)(jnp.asarray(variable_D)), expected_AD, rtol=7e-6, atol=7e-7)
    np.testing.assert_allclose(jax.jacrev(jax_fn)(jnp.asarray(variable_D)), expected_AD, rtol=7e-6, atol=7e-7)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("radius", [0.0, 0.99, 1.0])
def test_logmap_spatial_coincidence_has_exact_signed_identity_jacobians(dtype, radius: float) -> None:
    manifold = Hyperboloid(dtype=dtype)
    c = jnp.asarray(1.0, dtype=dtype)
    s_D = jnp.asarray([radius, 0.0], dtype=dtype)
    x_A = _j_lift(s_D, c)
    target_DD = jax.jacfwd(lambda y: manifold.logmap(_j_lift(y, c), x_A, c)[1:])(s_D)
    base_DD = jax.jacrev(lambda x: manifold.logmap(x_A, _j_lift(x, c), c)[1:])(s_D)
    expected_DD = np.eye(2, dtype=np.asarray(s_D).dtype)
    np.testing.assert_array_equal(np.asarray(target_DD), expected_DD)
    np.testing.assert_array_equal(np.asarray(base_DD), -expected_DD)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("radius", [1.01, 2.0])
def test_logmap_far_chart_coincidence_has_signed_identity_jacobians_with_rounding(dtype, radius: float) -> None:
    manifold = Hyperboloid(dtype=dtype)
    c = jnp.asarray(1.0, dtype=dtype)
    s_D = jnp.asarray([radius, 0.0], dtype=dtype)
    x_A = _j_lift(s_D, c)
    target_DD = jax.jacfwd(lambda y: manifold.logmap(_j_lift(y, c), x_A, c)[1:])(s_D)
    base_DD = jax.jacrev(lambda x: manifold.logmap(x_A, _j_lift(x, c), c)[1:])(s_D)
    tolerance = 3.0 * np.finfo(np.asarray(s_D).dtype).eps
    np.testing.assert_allclose(target_DD, np.eye(2), rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(base_DD, -np.eye(2), rtol=0.0, atol=tolerance)


def _combined_maps(x_D: jax.Array, y_D: jax.Array, c: jax.Array) -> jax.Array:
    w_D = jnp.asarray([0.2, -0.3], dtype=x_D.dtype)
    return jnp.concatenate(
        [
            _hyperboloid_map("difference", x_D, y_D, w_D, c),
            _hyperboloid_map("ptransp", x_D, y_D, w_D, c),
            _hyperboloid_map("logmap", x_D, y_D, w_D, c),
        ]
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("radius", [0.99, 1.0, 1.01])
def test_jacfwd_jacrev_jvp_vjp_jit_cover_collinearity_and_threshold(dtype, sign: float, radius: float) -> None:
    c = jnp.asarray(1.0, dtype=dtype)
    x_D = jnp.asarray([radius, 0.0], dtype=dtype)
    y_D = jnp.asarray([sign * 1.4, 0.0], dtype=dtype)
    direction_D = jnp.asarray([0.17, -0.23], dtype=dtype)
    cotangent = jnp.linspace(0.1, 0.9, 9, dtype=dtype)
    fn = jax.jit(lambda z: _combined_maps(z, y_D, c))
    fwd_AD = jax.jacfwd(fn)(x_D)
    rev_AD = jax.jacrev(fn)(x_D)
    tolerance = 2e-4 if dtype == jnp.float32 else 2e-10
    np.testing.assert_allclose(fwd_AD, rev_AD, rtol=tolerance, atol=tolerance)
    jvp_A = jax.jvp(fn, (x_D,), (direction_D,))[1]
    np.testing.assert_allclose(jvp_A, fwd_AD @ direction_D, rtol=tolerance, atol=tolerance)
    _, pullback = jax.vjp(fn, x_D)
    np.testing.assert_allclose(pullback(cotangent)[0], cotangent @ fwd_AD, rtol=tolerance, atol=tolerance)
    assert np.all(np.isfinite(np.asarray(jvp_A)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_jit_vmap_mixed_origin_and_threshold_batch_has_finite_vjp(dtype) -> None:
    c = jnp.asarray(1.0, dtype=dtype)
    xs_BD = jnp.asarray([[0.0, 0.0], [0.99, 0.0], [1.0, 0.0], [1.01, 0.0]], dtype=dtype)
    ys_BD = jnp.asarray([[0.7, -0.4], [1.4, 0.0], [-1.4, 0.0], [1.4, 0.0]], dtype=dtype)
    maps = jax.jit(jax.vmap(lambda x, y: _combined_maps(x, y, c)))
    out_BA = maps(xs_BD, ys_BD)
    jac_BABD = jax.jacrev(lambda points: maps(points, ys_BD))(xs_BD)
    assert np.all(np.isfinite(np.asarray(out_BA)))
    assert np.all(np.isfinite(np.asarray(jac_BABD)))


def _logmap_mixed_low_far_points(dtype) -> tuple[jax.Array, jax.Array]:
    e1_D = np.ones(4, dtype=np.float64) / 2.0
    e2_D = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
    far_direction_D = np.cos(0.3) * e1_D + np.sin(0.3) * e2_D
    xs_BD = jnp.asarray(
        np.stack([np.array([0.2, -0.1, 0.05, -0.07]), np.sinh(45.0) * e1_D]),
        dtype=dtype,
    )
    ys_BD = jnp.asarray(
        np.stack([np.array([0.3, 0.1, -0.08, 0.04]), np.sinh(45.0) * far_direction_D]),
        dtype=dtype,
    )
    return xs_BD, ys_BD


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("operand", ["base", "target"])
def test_logmap_mixed_low_radius45_vmap_preserves_selected_derivatives(dtype, operand: str) -> None:
    manifold = Hyperboloid(dtype=dtype)
    c = jnp.asarray(1.0, dtype=dtype)
    xs_BD, ys_BD = _logmap_mixed_low_far_points(dtype)
    variable_BD = xs_BD if operand == "base" else ys_BD
    direction_BD = jnp.linspace(-0.3, 0.4, variable_BD.size, dtype=dtype).reshape(variable_BD.shape)
    cotangent_BA = jnp.linspace(-0.5, 0.6, 10, dtype=dtype).reshape(2, 5)

    def batched(points_BD, implementation):
        bases_BD, targets_BD = (points_BD, ys_BD) if operand == "base" else (xs_BD, points_BD)
        return jax.vmap(lambda x_D, y_D: implementation(_j_lift(y_D, c), _j_lift(x_D, c), c))(bases_BD, targets_BD)

    def public(points_BD):
        return batched(points_BD, manifold.logmap)

    def native(points_BD):
        return batched(points_BD, _logmap_impl)

    if operand == "base":

        def np_low_fn(z_D):
            return _np_logmap(z_D, np.asarray(ys_BD[0], dtype=np.float64), 1.0)

    else:

        def np_low_fn(z_D):
            return _np_logmap(np.asarray(xs_BD[0], dtype=np.float64), z_D, 1.0)

    expected_low_AD = _fd_jacobian(np_low_fn, np.asarray(variable_BD[0], dtype=np.float64))
    tolerance = 8e-6 if dtype == jnp.float32 else 8e-7
    native_tolerance = 16.0 * np.finfo(np.asarray(variable_BD).dtype).eps

    np.testing.assert_array_equal(public(variable_BD), native(variable_BD))
    np.testing.assert_array_equal(jax.jit(public)(variable_BD), jax.jit(native)(variable_BD))

    def jvp(transform, points_BD):
        return jax.jvp(transform, (points_BD,), (direction_BD,))[1]

    def vjp(transform, points_BD):
        return jax.vjp(transform, points_BD)[1](cotangent_BA)[0]

    transforms = {
        "jvp": (jvp, expected_low_AD @ np.asarray(direction_BD[0])),
        "vjp": (vjp, np.asarray(cotangent_BA[0]) @ expected_low_AD),
        "jacfwd": (lambda transform, points_BD: jax.jacfwd(transform)(points_BD), expected_low_AD),
        "jacrev": (lambda transform, points_BD: jax.jacrev(transform)(points_BD), expected_low_AD),
    }
    for name, (transform, expected_low) in transforms.items():
        for compile_transform in (False, True):

            def evaluate(points_BD, transform=transform):
                return transform(public, points_BD)

            def evaluate_native(points_BD, transform=transform):
                return transform(native, points_BD)

            if compile_transform:
                evaluate = jax.jit(evaluate)
                evaluate_native = jax.jit(evaluate_native)
            got = evaluate(variable_BD)
            native_got = evaluate_native(variable_BD)
            assert np.all(np.isfinite(np.asarray(got))), f"non-finite {name}, jit={compile_transform}"
            if name in ("jvp", "vjp"):
                np.testing.assert_allclose(got[0], expected_low, rtol=tolerance, atol=tolerance)
                np.testing.assert_allclose(got[1], native_got[1], rtol=native_tolerance, atol=native_tolerance)
            else:
                np.testing.assert_allclose(got[0, :, 0, :], expected_low, rtol=tolerance, atol=tolerance)
                np.testing.assert_allclose(
                    got[1, :, 1, :], native_got[1, :, 1, :], rtol=native_tolerance, atol=native_tolerance
                )


def test_mixed_batch_jacobian_pins_each_independent_row() -> None:
    c = 1.0
    xs_BD = np.array([[0.0, 0.0], [0.99, 0.0], [1.01, 0.0]])
    ys_BD = np.array([[0.7, -0.4], [1.4, 0.0], [-1.4, 0.0]])
    w_D = np.array([0.2, -0.3])
    maps = jax.jit(jax.vmap(lambda x, y: _combined_maps(x, y, jnp.asarray(c))))
    got_BABD = np.asarray(jax.jacrev(lambda points: maps(points, jnp.asarray(ys_BD)))(jnp.asarray(xs_BD)))
    for row in range(xs_BD.shape[0]):
        y_D = ys_BD[row]

        def np_maps(x, fixed_y_D=y_D):
            return np.concatenate(
                [_numpy_map(operation, x, fixed_y_D, w_D, c) for operation in ("difference", "ptransp", "logmap")]
            )

        expected_AD = _fd_jacobian(np_maps, xs_BD[row])
        np.testing.assert_allclose(got_BABD[row, :, row, :], expected_AD, rtol=8e-6, atol=8e-7)
        for other in range(xs_BD.shape[0]):
            if other != row:
                np.testing.assert_array_equal(got_BABD[row, :, other, :], 0.0)


def _pv_map(operation: str, x_D: jax.Array, y_D: jax.Array, w_D: jax.Array, c: float) -> jax.Array:
    manifold = ProperVelocity(dtype=x_D.dtype)
    if operation == "difference":
        return manifold.gyro_difference(x_D, y_D, c)
    if operation == "logmap":
        return manifold.logmap(y_D, x_D, c)
    return manifold.ptransp(w_D, x_D, y_D, c)


def _numpy_pv_map(operation: str, x_D: np.ndarray, y_D: np.ndarray, w_D: np.ndarray, c: float) -> np.ndarray:
    return _numpy_map(operation, x_D, y_D, w_D, c)[1:]


@pytest.mark.parametrize("operation", ["difference", "ptransp", "logmap"])
@pytest.mark.parametrize("endpoint", ["base", "target"])
def test_proper_velocity_inherits_both_origin_endpoint_derivatives(operation: str, endpoint: str) -> None:
    c = 1.0
    fixed_D = np.array([0.7, -0.4])
    origin_D = np.zeros(2)
    w_D = np.array([0.3, 0.2])

    def jax_fn(z):
        if endpoint == "base":
            return _pv_map(operation, z, jnp.asarray(fixed_D), jnp.asarray(w_D), c)
        return _pv_map(operation, jnp.asarray(fixed_D), z, jnp.asarray(w_D), c)

    def np_fn(z):
        if endpoint == "base":
            return _numpy_pv_map(operation, z, fixed_D, w_D, c)
        return _numpy_pv_map(operation, fixed_D, z, w_D, c)

    fwd_DD = np.asarray(jax.jacfwd(jax_fn)(jnp.asarray(origin_D)))
    rev_DD = np.asarray(jax.jacrev(jax_fn)(jnp.asarray(origin_D)))
    expected_DD = _fd_jacobian(np_fn, origin_D)
    np.testing.assert_allclose(fwd_DD, expected_DD, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(rev_DD, expected_DD, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_proper_velocity_mixed_origin_batch_is_jit_vmap_and_vjp_clean(dtype) -> None:
    manifold = ProperVelocity(dtype=dtype)
    c = jnp.asarray(1.0, dtype=dtype)
    xs_BD = jnp.asarray([[0.0, 0.0], [0.99, 0.0], [1.01, 0.0]], dtype=dtype)
    ys_BD = jnp.asarray([[0.7, -0.4], [0.0, 0.0], [-1.4, 0.0]], dtype=dtype)
    w_D = jnp.asarray([0.3, 0.2], dtype=dtype)

    def one(x_D, y_D):
        return jnp.concatenate(
            [
                manifold.gyro_difference(x_D, y_D, c),
                manifold.ptransp(w_D, x_D, y_D, c),
                manifold.logmap(y_D, x_D, c),
            ]
        )

    maps = jax.jit(jax.vmap(one))
    values_BD = maps(xs_BD, ys_BD)
    jac_BDBD = jax.jacrev(lambda points: maps(points, ys_BD))(xs_BD)
    assert np.all(np.isfinite(np.asarray(values_BD)))
    assert np.all(np.isfinite(np.asarray(jac_BDBD)))


def _ld_busemann_parameter_derivatives(
    s_D: np.ndarray, raw_v_D: np.ndarray, c: float
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Analytic derivatives through the sheet lift and unit-direction map."""
    ld = np.longdouble
    s_D = np.asarray(s_D, dtype=ld)
    raw_v_D = np.asarray(raw_v_D, dtype=ld)
    c_ld = ld(c)
    raw_norm = np.sqrt(np.dot(raw_v_D, raw_v_D))
    v_D = raw_v_D / raw_norm
    x0 = np.sqrt(ld(1.0) / c_ld + np.dot(s_D, s_D))
    q = np.dot(s_D, v_D)
    argument = x0 - q
    sqrt_c = np.sqrt(c_ld)
    log_argument = np.log(sqrt_c * argument)

    ds_D = (s_D / x0 - v_D) / (sqrt_c * argument)
    tangent_s_D = s_D - v_D * np.dot(v_D, s_D)
    draw_v_D = -tangent_s_D / (raw_norm * sqrt_c * argument)
    dc = (ld(1.0) - log_argument) / (ld(2.0) * c_ld * sqrt_c)
    dc -= ld(1.0) / (ld(2.0) * c_ld * c_ld * sqrt_c * x0 * argument)
    return float(q), np.asarray(ds_D, dtype=np.float64), np.asarray(draw_v_D, dtype=np.float64), float(dc)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize(
    ("q_side", "s_values"),
    [("negative", [-0.7, 0.4]), ("zero", [0.0, 0.4]), ("positive", [0.7, 0.4])],
    ids=["q-negative", "q-zero", "q-positive"],
)
def test_busemann_q_branch_derivatives_match_independent_analytic_reference(dtype, q_side: str, s_values) -> None:
    manifold = Hyperboloid(dtype=dtype)
    s_D = jnp.asarray(s_values, dtype=dtype)
    raw_v_D = jnp.asarray([2.0, 0.0], dtype=dtype)
    c = jnp.asarray(0.7, dtype=dtype)

    def value(s, raw_v, curvature):
        unit_v_D = raw_v / jnp.sqrt(jnp.dot(raw_v, raw_v))
        return manifold.busemann(_j_lift(s, curvature), unit_v_D, curvature)

    got_s_D, got_raw_v_D, got_c = jax.grad(value, argnums=(0, 1, 2))(s_D, raw_v_D, c)
    expected_q, expected_s_D, expected_raw_v_D, expected_c = _ld_busemann_parameter_derivatives(
        np.asarray(s_D), np.asarray(raw_v_D), float(c)
    )
    assert np.sign(expected_q) == {"negative": -1.0, "zero": 0.0, "positive": 1.0}[q_side]
    tolerance = 3e-6 if dtype == jnp.float32 else 3e-12
    np.testing.assert_allclose(got_s_D, expected_s_D, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(got_raw_v_D, expected_raw_v_D, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(got_c, expected_c, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_busemann_mixed_jit_vmap_pins_zero_unused_denominator_derivatives(dtype) -> None:
    manifold = Hyperboloid(dtype=dtype)
    radius = 10.0 if dtype == jnp.float32 else 40.0
    high_radius = np.asarray(np.sinh(radius), dtype=np.asarray(jnp.asarray(0.0, dtype=dtype)).dtype)
    xs_BD = jnp.asarray([[-high_radius, 0.0], [0.0, 0.4], [0.7, 0.4]], dtype=dtype)
    raw_vs_BD = jnp.asarray([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]], dtype=dtype)
    cs_B = jnp.asarray([1.0, 0.7, 1.3], dtype=dtype)

    high_v_D = raw_vs_BD[0] / jnp.sqrt(jnp.dot(raw_vs_BD[0], raw_vs_BD[0]))
    high_x_A = _j_lift(xs_BD[0], cs_B[0])
    high_q = jnp.dot(xs_BD[0], high_v_D)
    np.testing.assert_array_equal(np.asarray(high_x_A[0] + high_q), 0.0)

    def one(s_D, raw_v_D, c):
        unit_v_D = raw_v_D / jnp.sqrt(jnp.dot(raw_v_D, raw_v_D))
        return manifold.busemann(_j_lift(s_D, c), unit_v_D, c)

    batched = jax.jit(jax.vmap(one))
    got_s_BBD, got_raw_v_BBD, got_c_BB = jax.jacrev(batched, argnums=(0, 1, 2))(xs_BD, raw_vs_BD, cs_B)
    tolerance = 8e-6 if dtype == jnp.float32 else 8e-12
    for row in range(xs_BD.shape[0]):
        expected_q, expected_s_D, expected_raw_v_D, expected_c = _ld_busemann_parameter_derivatives(
            np.asarray(xs_BD[row]), np.asarray(raw_vs_BD[row]), float(cs_B[row])
        )
        assert np.sign(expected_q) == (-1.0, 0.0, 1.0)[row]
        np.testing.assert_allclose(got_s_BBD[row, row], expected_s_D, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(got_raw_v_BBD[row, row], expected_raw_v_D, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(got_c_BB[row, row], expected_c, rtol=tolerance, atol=tolerance)
        for other in range(xs_BD.shape[0]):
            if other != row:
                np.testing.assert_array_equal(got_s_BBD[row, other], 0.0)
                np.testing.assert_array_equal(got_raw_v_BBD[row, other], 0.0)
                np.testing.assert_array_equal(got_c_BB[row, other], 0.0)


@pytest.mark.parametrize("operation", ["difference", "ptransp"])
@pytest.mark.parametrize("endpoint", ["base", "target"])
def test_coincidence_endpoint_derivatives_match_independent_fd(operation: str, endpoint: str) -> None:
    """At nonzero y=x, compare frame derivatives with independent inverse-boost derivatives."""
    c = 1.0
    coincident_D = np.array([0.9, -0.1])
    w_D = np.array([0.3, -0.2])

    def jax_fn(z_D):
        if endpoint == "base":
            return _hyperboloid_map(operation, z_D, jnp.asarray(coincident_D), jnp.asarray(w_D), c)
        return _hyperboloid_map(operation, jnp.asarray(coincident_D), z_D, jnp.asarray(w_D), c)

    def numpy_fn(z_D):
        if endpoint == "base":
            return _numpy_map(operation, z_D, coincident_D, w_D, c)
        return _numpy_map(operation, coincident_D, z_D, w_D, c)

    expected_AD = _fd_jacobian(numpy_fn, coincident_D)
    np.testing.assert_allclose(jax.jacfwd(jax_fn)(jnp.asarray(coincident_D)), expected_AD, rtol=8e-6, atol=8e-7)
    np.testing.assert_allclose(jax.jacrev(jax_fn)(jnp.asarray(coincident_D)), expected_AD, rtol=8e-6, atol=8e-7)


def _ld_inverse_boost(stored_x_D: np.ndarray, stored_y_D: np.ndarray, c: float) -> np.ndarray:
    """Long-double inverse boost from the exact stored spatial operands."""
    ld = np.longdouble
    x_D = np.asarray(stored_x_D, dtype=ld)
    y_D = np.asarray(stored_y_D, dtype=ld)
    c_ld = ld(c)
    x0 = np.sqrt(ld(1.0) / c_ld + np.dot(x_D, x_D))
    y0 = np.sqrt(ld(1.0) / c_ld + np.dot(y_D, y_D))
    gamma = np.sqrt(c_ld) * x0
    result_D = y_D - np.sqrt(c_ld) * y0 * x_D + c_ld * np.dot(x_D, y_D) * x_D / (ld(1.0) + gamma)
    return np.concatenate((np.asarray([np.sqrt(ld(1.0) / c_ld + np.dot(result_D, result_D))]), result_D))


@pytest.mark.parametrize("operation", ["difference", "ptransp"])
@pytest.mark.parametrize("c", [0.1, 1.0])
@pytest.mark.parametrize("dim", [2, 64])
def test_small_radius_chart_values_and_gradients_match_independent_fd(operation: str, c: float, dim: int) -> None:
    """Check both endpoints near zero and on both sides of the scaled-spatial-radius switch."""
    rng = np.random.default_rng(42)
    direction_D = rng.normal(size=dim)
    direction_D /= np.linalg.norm(direction_D)
    fixed_D = rng.normal(size=dim)
    fixed_D *= 0.7 / (np.sqrt(c) * np.linalg.norm(fixed_D))
    w_D = rng.normal(size=dim)
    w_D /= np.linalg.norm(w_D)
    cotangent_A = rng.normal(size=dim + 1)
    cotangent_A /= np.linalg.norm(cotangent_A)
    pairs = []
    for radius in (0.0, 1e-8, 1e-6, 1e-4, 0.0099, 0.0101, 0.099, 0.101):
        small_D = radius * direction_D / np.sqrt(c)
        pairs.extend((np.concatenate((small_D, fixed_D)), np.concatenate((fixed_D, small_D))))
    # Round once: float32 and the independent reference see identical spatial coordinates.
    stored_BP = np.asarray(pairs, dtype=np.float32)

    def numpy_fn(z_P: np.ndarray, curvature: float) -> np.ndarray:
        z_P = np.asarray(z_P, dtype=np.float64)
        return _numpy_map(operation, z_P[:dim], z_P[dim:], w_D, curvature)

    expected_BA = np.stack([numpy_fn(z_P, c) for z_P in stored_BP])
    expected_grad_BP = np.stack([_fd_jacobian(lambda z: np.dot(numpy_fn(z, c), cotangent_A), z_P) for z_P in stored_BP])
    expected_dc_B = np.asarray(
        [_fd_scalar(lambda curvature, z_P=z_P: np.dot(numpy_fn(z_P, curvature), cotangent_A), c) for z_P in stored_BP]
    )
    for dtype in (jnp.float32, jnp.float64):
        typed_w_D = jnp.asarray(w_D, dtype=dtype)
        typed_cotangent_A = jnp.asarray(cotangent_A, dtype=dtype)

        def evaluate(
            z_P: jax.Array, curvature: jax.Array, typed_w_D=typed_w_D, typed_cotangent_A=typed_cotangent_A
        ) -> tuple[jax.Array, jax.Array]:
            value_A = _hyperboloid_map(operation, z_P[:dim], z_P[dim:], typed_w_D, curvature)
            return jnp.dot(value_A, typed_cotangent_A), value_A

        evaluate_batch = jax.jit(jax.vmap(jax.value_and_grad(evaluate, argnums=(0, 1), has_aux=True), in_axes=(0, None)))
        (_, values_BA), (grad_BP, dc_B) = evaluate_batch(jnp.asarray(stored_BP, dtype=dtype), jnp.asarray(c, dtype=dtype))
        rtol, atol = (2e-5, 2e-6) if dtype == jnp.float32 else (2e-7, 2e-8)
        np.testing.assert_allclose(values_BA, expected_BA, rtol=rtol, atol=atol)
        np.testing.assert_allclose(grad_BP, expected_grad_BP, rtol=rtol, atol=atol)
        np.testing.assert_allclose(dc_B, expected_dc_B, rtol=rtol, atol=atol)


@pytest.mark.parametrize("step", [1e-4, 1e-6])
@pytest.mark.parametrize("kind", ["radial", "angular"])
def test_float32_near_coincident_gyro_difference_matches_longdouble_stored_inputs(step: float, kind: str) -> None:
    """At radius 0.9, centering must preserve small radial and angular differences."""
    c = 1.0
    stored_x_D = np.asarray([0.9, 0.0], dtype=np.float32)
    if kind == "radial":
        stored_y_D = np.asarray([0.9 + step, 0.0], dtype=np.float32)
    else:
        stored_y_D = np.asarray(0.9 * np.array([np.cos(step), np.sin(step)]), dtype=np.float32)
    manifold = Hyperboloid(dtype=jnp.float32)
    got_A = np.asarray(manifold.gyro_difference(_j_lift(jnp.asarray(stored_x_D), c), _j_lift(jnp.asarray(stored_y_D), c), c))
    expected_A = _ld_inverse_boost(stored_x_D, stored_y_D, c)
    np.testing.assert_allclose(got_A, expected_A, rtol=3e-5, atol=3e-7)
    # GPU reciprocal-multiply normalization can give collinear stored points a false chord.
    # Budget eps*radius absolute error for this arithmetic, plus relative error in the update.
    # This is a bound for the evaluated chart, not an error in the long-double stored-input oracle.
    spatial_error = np.linalg.norm(got_A[1:] - expected_A[1:])
    rounding_bound = np.finfo(np.float32).eps * max(np.linalg.norm(stored_x_D), np.linalg.norm(stored_y_D))
    assert spatial_error <= rounding_bound + 3e-5 * np.linalg.norm(expected_A[1:])


@pytest.mark.parametrize("c", [0.1, 1.0])
@pytest.mark.parametrize("radius", [0.005, 0.0099, 0.0101, 0.099, 0.101])
@pytest.mark.parametrize("step", [1e-4, 1e-6])
@pytest.mark.parametrize("kind", ["radial", "angular"])
def test_small_radius_centering_of_rounded_points(c: float, radius: float, step: float, kind: str) -> None:
    """Independently round points made by float64 radial/angular moves around the chart switch."""
    direction_D = np.array([0.6, 0.8])
    x_D = radius * direction_D / np.sqrt(c)
    if kind == "radial":
        y_D = np.sinh(np.arcsinh(radius) + step) * direction_D / np.sqrt(c)
    else:
        y_D = radius * (np.cos(step) * direction_D + np.sin(step) * np.array([-0.8, 0.6])) / np.sqrt(c)
    stored_x_D, stored_y_D = np.asarray(x_D, dtype=np.float32), np.asarray(y_D, dtype=np.float32)
    manifold = Hyperboloid(dtype=jnp.float32)
    evaluate = jax.jit(lambda x, y: manifold.gyro_difference(_j_lift(x, c), _j_lift(y, c), c))
    got_A = np.asarray(evaluate(jnp.asarray(stored_x_D), jnp.asarray(stored_y_D)))
    expected_A = _ld_inverse_boost(stored_x_D, stored_y_D, c)
    rounding_bound = 2 * np.finfo(np.float32).eps * max(np.linalg.norm(stored_x_D), np.linalg.norm(stored_y_D))
    assert np.linalg.norm(got_A[1:] - expected_A[1:]) <= rounding_bound + 3e-5 * np.linalg.norm(expected_A[1:])
    np.testing.assert_allclose(got_A[0], expected_A[0], rtol=2e-7)


def _np_transport_from_stored_ambient(v_A: np.ndarray, x_A: np.ndarray, y_A: np.ndarray, c: float) -> np.ndarray:
    """Closed-form transport using the caller's stored ambient coordinates and curvature."""
    x_A = np.asarray(x_A, dtype=np.float64)
    y_A = np.asarray(y_A, dtype=np.float64)
    v_A = np.asarray(v_A, dtype=np.float64)
    v0 = np.dot(x_A[1:] / x_A[0], v_A[1:])
    tangent_A = np.concatenate(([v0], v_A[1:]))
    lorentz_vy = -v0 * y_A[0] + np.dot(v_A[1:], y_A[1:])
    denominator = 1.0 / c + x_A[0] * y_A[0] - np.dot(x_A[1:], y_A[1:])
    return tangent_A + lorentz_vy / denominator * (x_A + y_A)


def _np_lift_as_float32(s_D: np.ndarray, c: float) -> np.ndarray:
    """Match a float32 sheet input constructed before a float64-curvature call."""
    stored_s_D = np.asarray(s_D, dtype=np.float32)
    stored_c = np.asarray(c, dtype=np.float32)
    stored_time = np.asarray(np.sqrt(np.float32(1.0) / stored_c + np.dot(stored_s_D, stored_s_D)), dtype=np.float32)
    return np.concatenate((stored_time[None], stored_s_D))


def test_ptransp_mixed_point_and_curvature_precision_matches_independent_transforms() -> None:
    """A float32 point with float64 curvature keeps the caller's curvature under every transform."""
    c = jnp.asarray(0.7, dtype=jnp.float64)
    x_D = jnp.asarray([0.75, -0.35], dtype=jnp.float32)
    y_D = jnp.asarray([0.9, -0.15], dtype=jnp.float32)
    w_D = jnp.asarray([0.2, -0.3], dtype=jnp.float32)
    direction_D = jnp.asarray([-0.11, 0.07], dtype=jnp.float32)
    manifold = Hyperboloid(dtype=jnp.float32)

    def transport(y_spatial_D):
        x_A = _j_lift(x_D, c)
        y_A = _j_lift(y_spatial_D, c)
        return manifold.ptransp(_j_tangent(x_D, w_D, c), x_A, y_A, c)

    x_A = np.asarray(_j_lift(x_D, c))
    y_A = np.asarray(_j_lift(y_D, c))
    v_A = np.asarray(_j_tangent(x_D, w_D, c))
    assert x_A.dtype == y_A.dtype == v_A.dtype == np.float32
    expected_A = _np_transport_from_stored_ambient(v_A, x_A, y_A, float(c))

    def reference_directional_derivative(step: float) -> np.ndarray:
        y_plus_D = np.asarray(y_D, dtype=np.float64) + step * np.asarray(direction_D, dtype=np.float64)
        y_minus_D = np.asarray(y_D, dtype=np.float64) - step * np.asarray(direction_D, dtype=np.float64)
        plus_A = _np_transport_from_stored_ambient(v_A, x_A, _np_lift_as_float32(y_plus_D, float(c)), float(c))
        minus_A = _np_transport_from_stored_ambient(v_A, x_A, _np_lift_as_float32(y_minus_D, float(c)), float(c))
        return (plus_A - minus_A) / (2.0 * step)

    expected_jvp_A = reference_directional_derivative(2e-3)
    np.testing.assert_allclose(expected_jvp_A, reference_directional_derivative(1e-3), rtol=3e-3, atol=3e-4)

    eager_A = transport(y_D)
    compiled_A = jax.jit(transport)(y_D)
    batched_BA = jax.jit(jax.vmap(transport))(jnp.stack([y_D, y_D]))
    jvp_A = jax.jvp(transport, (y_D,), (direction_D,))[1]
    tolerance = 3e-6
    np.testing.assert_allclose(eager_A, expected_A, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(compiled_A, expected_A, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(batched_BA[0], expected_A, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(batched_BA[1], expected_A, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(jvp_A, expected_jvp_A, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize("radius", [0.9, 1.2])
def test_ptransp_ignores_caller_tangent_time_at_small_and_large_radius(radius: float) -> None:
    """Transport defines the tangent time from x and spatial components, never caller garbage."""
    c = 1.0
    x_D = jnp.asarray([radius, 0.0], dtype=jnp.float64)
    y_D = jnp.asarray([radius + 0.15, -0.2], dtype=jnp.float64)
    w_D = jnp.asarray([0.2, -0.3], dtype=jnp.float64)
    x_A, y_A = _j_lift(x_D, c), _j_lift(y_D, c)
    tangent_A = _j_tangent(x_D, w_D, c)
    altered_A = tangent_A.at[0].set(tangent_A[0] + 17.0)
    manifold = Hyperboloid(dtype=jnp.float64)
    expected_A = _np_transport_from_stored_ambient(np.asarray(tangent_A), np.asarray(x_A), np.asarray(y_A), c)
    np.testing.assert_allclose(manifold.ptransp(tangent_A, x_A, y_A, c), expected_A, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(manifold.ptransp(altered_A, x_A, y_A, c), expected_A, rtol=2e-12, atol=2e-12)


def _np_logmap_squared_spatial_norm(target_D: np.ndarray, base_D: np.ndarray, c: float) -> float:
    """Squared spatial norm of the closed-form logmap, evaluated outside JAX."""
    x_A = _np_lift(base_D, c)
    y_A = _np_lift(target_D, c)
    alpha = c * (x_A[0] * y_A[0] - np.dot(x_A[1:], y_A[1:]))
    theta = np.arccosh(alpha)
    spatial_D = theta * (y_A[1:] - alpha * x_A[1:]) / np.sqrt(alpha * alpha - 1.0)
    return float(np.dot(spatial_D, spatial_D))


def _np_scalar_hessian(fn: Callable[[np.ndarray], float], x_D: np.ndarray, step: float) -> np.ndarray:
    x_D = np.asarray(x_D, dtype=np.float64)
    eye_DD = np.eye(x_D.size)
    hessian_DD = np.empty((x_D.size, x_D.size))
    for i in range(x_D.size):
        for j in range(x_D.size):
            ei_D, ej_D = step * eye_DD[i], step * eye_DD[j]
            hessian_DD[i, j] = (
                fn(x_D + ei_D + ej_D) - fn(x_D + ei_D - ej_D) - fn(x_D - ei_D + ej_D) + fn(x_D - ei_D - ej_D)
            ) / (4.0 * step**2)
    return hessian_DD


@pytest.mark.parametrize("target_D", [np.array([0.0, 0.0]), np.array([0.3, -0.2]), np.array([1.0, 0.5])])
def test_origin_log_squared_spatial_norm_hessian_matches_independent_scalar_reference(target_D: np.ndarray) -> None:
    """The Cartesian origin derivative covers the second-order loss through logmap."""
    c = 1.0
    manifold = Hyperboloid(dtype=jnp.float64)

    def loss(base_D):
        return jnp.sum(manifold.logmap(_j_lift(jnp.asarray(target_D), c), _j_lift(base_D, c), c)[1:] ** 2)

    got_DD = np.asarray(jax.hessian(loss)(jnp.zeros(2, dtype=jnp.float64)))
    if np.all(target_D == 0.0):
        expected_DD = 2.0 * np.eye(2)
    else:
        expected_DD = _np_scalar_hessian(
            lambda base_D: _np_logmap_squared_spatial_norm(target_D, base_D, c), np.zeros(2), 1e-4
        )
    np.testing.assert_allclose(got_DD, expected_DD, rtol=1e-6, atol=1e-6)
