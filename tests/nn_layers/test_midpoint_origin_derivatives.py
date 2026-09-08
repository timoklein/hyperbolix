"""Midpoint derivatives against the differentiated literal Lorentz mean.

Dimension key: M points, D spatial coordinates, A ambient coordinates, N rows.
The NumPy oracle uses the Lorentz square and no production helpers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.nn_layers.hyperboloid_core import lorentz_midpoint


def _lift(spatial_MD, c):
    time_M1 = jnp.sqrt(1 / c + jnp.sum(spatial_MD**2, axis=-1, keepdims=True))
    return jnp.concatenate((time_M1, spatial_MD), axis=-1)


def _mean(spatial_MD, weights_M, c):
    return lorentz_midpoint(_lift(spatial_MD, c), weights_M[None, :], c)[0]


def _literal_jvp(spatial_MD, weights_M, c, ds_MD, dw_M, dc):
    """Differentiate h_s / sqrt(c * (h_0**2 - h_s**2)) analytically."""
    spatial_MD, weights_M, ds_MD, dw_M = (
        np.asarray(value, dtype=np.float64) for value in (spatial_MD, weights_M, ds_MD, dw_M)
    )
    c, dc = float(c), float(dc)
    time_M = np.sqrt(1 / c + np.sum(spatial_MD**2, axis=-1))
    dt_M = (np.sum(spatial_MD * ds_MD, axis=-1) - dc / (2 * c**2)) / time_M
    time = weights_M @ time_M
    h_D = weights_M @ spatial_MD
    dtime = dw_M @ time_M + weights_M @ dt_M
    dh_D = dw_M @ spatial_MD + weights_M @ ds_MD
    square = time**2 - h_D @ h_D
    denom = np.sqrt(c * square)
    ddenom = (dc * square + 2 * c * (time * dtime - h_D @ dh_D)) / (2 * denom)
    out_D = h_D / denom
    dout_D = dh_D / denom - out_D * ddenom / denom
    out_time = np.sqrt(1 / c + out_D @ out_D)
    dout_time = (out_D @ dout_D - dc / (2 * c**2)) / out_time
    return np.concatenate(([dout_time], dout_D))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.1, 0.5, 3.0])
@pytest.mark.parametrize("case", ["mixed_origin", "all_origin", "coincident", "collinear", "masked"])
def test_midpoint_spatial_weight_curvature_derivatives(dtype, c, case):
    spatial_MD = np.array([[0.0, 0.0], [0.7, -0.2], [-0.3, 0.4]])
    weights_M = np.array([0.2, 0.3, 0.5])
    if case == "all_origin":
        spatial_MD.fill(0)
    elif case == "coincident":
        spatial_MD[:] = [0.4, -0.2]
    elif case == "collinear":
        spatial_MD[:] = [[0, 0], [0.4, -0.2], [0.8, -0.4]]
    elif case == "masked":
        weights_M[:] = [0.0, 0.6, 0.4]
    inputs = (jnp.asarray(spatial_MD, dtype), jnp.asarray(weights_M, dtype), jnp.asarray(c, dtype))
    tolerance = 5e-6 if dtype == jnp.float32 else 2e-12
    jacobians = jax.jacfwd(_mean, argnums=(0, 1, 2))(*inputs)
    reverse = jax.jit(jax.jacrev(_mean, argnums=(0, 1, 2)))(*inputs)
    # Every coordinate of all three input classes is pinned independently.
    for argument, value in enumerate(inputs):
        for index in np.ndindex(value.shape):
            tangent = [np.zeros(part.shape) for part in inputs]
            tangent[argument][index] = 1
            expected_A = _literal_jvp(*inputs, *tangent)
            np.testing.assert_allclose(jacobians[argument][(slice(None), *index)], expected_A, atol=tolerance, rtol=tolerance)
        np.testing.assert_allclose(reverse[argument], jacobians[argument], atol=tolerance, rtol=tolerance)
    tangents = tuple(jnp.linspace(-0.2, 0.3, value.size, dtype=dtype).reshape(value.shape) for value in inputs)
    _, actual_A = jax.jit(lambda *xs: jax.jvp(_mean, xs, tangents))(*inputs)
    np.testing.assert_allclose(actual_A, _literal_jvp(*inputs, *tangents), atol=tolerance, rtol=tolerance)
    cotangent_A = jnp.array([-0.2, 0.5, 0.8], dtype)
    _, pullback = jax.vjp(_mean, *inputs)
    inner = sum(jnp.vdot(a, b) for a, b in zip(pullback(cotangent_A), tangents, strict=True))
    np.testing.assert_allclose(inner, jnp.vdot(cotangent_A, actual_A), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_midpoint_broadcast_and_zero_weight_row_convention(dtype):
    c = jnp.asarray(0.5, dtype)
    spatial_MD = jnp.array([[0, 0], [0.7, -0.2], [-0.3, 0.4]], dtype)
    weights_NM = jnp.array([[0.2, 0.3, 0.5], [0, 0, 0], [0.6, 0, 0.4]], dtype)
    points_MA = _lift(spatial_MD, c)
    batched = jax.jit(lorentz_midpoint)(points_MA, weights_NM, c)
    mapped = jax.jit(jax.vmap(_mean, in_axes=(None, 0, None)))(spatial_MD, weights_NM, c)
    tolerance = 5e-6 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(batched, mapped, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(batched[1], [np.sqrt(2), 0, 0], atol=tolerance, rtol=tolerance)
    # The zero row is independent of the points. Its weight derivative is not
    # specified: a normalized mean has no continuous extension to zero weights.
    masked_jacobian = jax.jacrev(lambda s: _mean(s, weights_NM[1], c))(spatial_MD)
    np.testing.assert_array_equal(masked_jacobian, jnp.zeros_like(masked_jacobian))
    repeated = jnp.stack([points_MA, points_MA])
    np.testing.assert_allclose(lorentz_midpoint(repeated, weights_NM, c), jnp.stack([batched, batched]), atol=tolerance)
