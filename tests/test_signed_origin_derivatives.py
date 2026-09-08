"""Regression tests for signed Möbius derivatives at zero curvature.

Dimension key:
    B: batch cases spanning negative, zero, and positive curvature
    D: stereographic manifold dimension
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds import Stereographic
from hyperbolix.manifolds._gyrovector_core import _mobius_denominator
from hyperbolix.utils import LearnableCurvature

_DTYPES = (jnp.float32, jnp.float64)


def _literal_denominator(x_D: jax.Array, y_D: jax.Array, c: jax.Array, sign: int) -> jax.Array:
    """Literal polynomial oracle, independent of the stable factored implementation."""
    x2 = jnp.dot(x_D, x_D)
    y2 = jnp.dot(y_D, y_D)
    xy = jnp.dot(x_D, y_D)
    return 1 + 2 * sign * c * xy + c**2 * x2 * y2


def _literal_addition(x_D: jax.Array, y_D: jax.Array, c: jax.Array) -> jax.Array:
    """Literal Möbius formula for interior points, where projection is inactive."""
    x2 = jnp.dot(x_D, x_D)
    y2 = jnp.dot(y_D, y_D)
    xy = jnp.dot(x_D, y_D)
    numerator_D = (1 + 2 * c * xy + c * y2) * x_D + (1 - c * x2) * y_D
    denominator = 1 + 2 * c * xy + c**2 * x2 * y2
    return numerator_D / denominator


def _addition_curvature_derivative_at_zero(x_D: jax.Array, y_D: jax.Array) -> jax.Array:
    """Analytic ``d(x ⊕_c y)/dc`` at ``c=0`` for interior operands."""
    x2 = jnp.dot(x_D, x_D)
    y2 = jnp.dot(y_D, y_D)
    xy = jnp.dot(x_D, y_D)
    return y2 * x_D - (x2 + 2 * xy) * y_D


def _tolerances(dtype: jnp.dtype) -> tuple[float, float]:
    return (3e-5, 3e-5) if dtype == jnp.float32 else (2e-12, 2e-12)


def _assert_tree_allclose(actual, expected, dtype: jnp.dtype) -> None:
    atol, rtol = _tolerances(dtype)
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        assert jnp.all(jnp.isfinite(actual_leaf))
        assert jnp.allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


def _cases(dtype: jnp.dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    x_D = jnp.asarray([0.25, -0.40, 0.15], dtype=dtype)
    y_D = jnp.asarray([-0.30, 0.10, 0.45], dtype=dtype)
    zero_D = jnp.zeros_like(x_D)
    x_BD = jnp.stack((x_D, x_D, x_D, zero_D, x_D))
    y_BD = jnp.stack((y_D, y_D, y_D, y_D, zero_D))
    c_B = jnp.asarray([-1e-3, 0.0, 1e-3, 0.0, 0.0], dtype=dtype)
    return x_BD, y_BD, c_B


def _check_all_transforms(actual_fn, oracle_fn, dtype: jnp.dtype, *, vector_output: bool) -> None:
    x_BD, y_BD, c_B = _cases(dtype)
    in_axes = (0, 0, 0)

    actual_values = jax.jit(jax.vmap(actual_fn, in_axes=in_axes))(x_BD, y_BD, c_B)
    oracle_values = jax.jit(jax.vmap(oracle_fn, in_axes=in_axes))(x_BD, y_BD, c_B)
    _assert_tree_allclose(actual_values, oracle_values, dtype)

    for transform in (jax.jacfwd, jax.jacrev):
        actual_jacobian = jax.jit(jax.vmap(transform(actual_fn, argnums=(0, 1, 2)), in_axes=in_axes))(x_BD, y_BD, c_B)
        oracle_jacobian = jax.jit(jax.vmap(transform(oracle_fn, argnums=(0, 1, 2)), in_axes=in_axes))(x_BD, y_BD, c_B)
        _assert_tree_allclose(actual_jacobian, oracle_jacobian, dtype)

    tangent_x_BD = jnp.broadcast_to(jnp.asarray([0.3, -0.2, 0.1], dtype=dtype), x_BD.shape)
    tangent_y_BD = jnp.broadcast_to(jnp.asarray([-0.1, 0.4, 0.2], dtype=dtype), y_BD.shape)
    tangent_c_B = jnp.full_like(c_B, 0.7)

    def transformed_jvp(fn, x_D, y_D, c, tangent_x_D, tangent_y_D, tangent_c):
        return jax.jvp(fn, (x_D, y_D, c), (tangent_x_D, tangent_y_D, tangent_c))[1]

    actual_jvp = jax.jit(jax.vmap(lambda *args: transformed_jvp(actual_fn, *args)))(
        x_BD, y_BD, c_B, tangent_x_BD, tangent_y_BD, tangent_c_B
    )
    oracle_jvp = jax.jit(jax.vmap(lambda *args: transformed_jvp(oracle_fn, *args)))(
        x_BD, y_BD, c_B, tangent_x_BD, tangent_y_BD, tangent_c_B
    )
    _assert_tree_allclose(actual_jvp, oracle_jvp, dtype)

    cotangent = jnp.asarray([0.2, -0.7, 0.5], dtype=dtype) if vector_output else jnp.asarray(1.0, dtype=dtype)

    def transformed_vjp(fn, x_D, y_D, c):
        _, pullback = jax.vjp(fn, x_D, y_D, c)
        return pullback(cotangent)

    actual_vjp = jax.jit(jax.vmap(lambda x_D, y_D, c: transformed_vjp(actual_fn, x_D, y_D, c)))(x_BD, y_BD, c_B)
    oracle_vjp = jax.jit(jax.vmap(lambda x_D, y_D, c: transformed_vjp(oracle_fn, x_D, y_D, c)))(x_BD, y_BD, c_B)
    _assert_tree_allclose(actual_vjp, oracle_vjp, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("sign", (-1, 1))
def test_signed_mobius_denominator_matches_literal_derivatives(dtype, sign):
    def actual_fn(x_D, y_D, c):
        return _mobius_denominator(x_D, y_D, c, sign)

    def oracle_fn(x_D, y_D, c):
        return _literal_denominator(x_D, y_D, c, sign)

    _check_all_transforms(actual_fn, oracle_fn, dtype, vector_output=False)

    x_BD, y_BD, _ = _cases(dtype)
    c0 = jnp.asarray(0.0, dtype=dtype)
    expected_slope = 2 * sign * jnp.dot(x_BD[0], y_BD[0])
    assert jnp.allclose(jax.jacfwd(lambda c: actual_fn(x_BD[0], y_BD[0], c))(c0), expected_slope)
    assert jnp.allclose(jax.jacrev(lambda c: actual_fn(x_BD[0], y_BD[0], c))(c0), expected_slope)

    # The retained MIN_NORM radius floors leave only their O(MIN_NORM * radius) residue when either
    # operand is zero. Pin that inherited numerical allowance tightly around the analytic zero slope.
    zero_D = jnp.zeros_like(x_BD[0])

    def transformed_origin_slope(first_D, second_D, transform):
        def denominator_at_c(c):
            return actual_fn(first_D, second_D, c)

        return transform(denominator_at_c)(c0)

    for first_D, second_D in ((zero_D, y_BD[0]), (x_BD[0], zero_D)):
        for transform in (jax.jacfwd, jax.jacrev):
            origin_slope = transformed_origin_slope(first_D, second_D, transform)
            assert jnp.isfinite(origin_slope)
            assert jnp.allclose(origin_slope, jnp.zeros_like(origin_slope), atol=2e-14, rtol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_signed_addition_matches_literal_derivatives(dtype):
    manifold = Stereographic(dtype=dtype)
    actual_fn = manifold.addition
    _check_all_transforms(actual_fn, _literal_addition, dtype, vector_output=True)


class _CurvatureConsumer(nnx.Module):
    def __init__(self, dtype: jnp.dtype) -> None:
        self.curvature = LearnableCurvature(0.0, parameterization="identity", param_dtype=dtype)
        self.manifold = Stereographic(dtype=dtype)

    def __call__(self, x_D: jax.Array, y_D: jax.Array, weights_D: jax.Array) -> jax.Array:
        return jnp.vdot(weights_D, self.manifold.addition(x_D, y_D, self.curvature()))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_identity_curvature_updates_from_zero_in_analytic_direction(dtype):
    x_D = jnp.asarray([0.25, -0.40, 0.15], dtype=dtype)
    y_D = jnp.asarray([-0.30, 0.10, 0.45], dtype=dtype)
    weights_D = jnp.asarray([0.2, -0.7, 0.5], dtype=dtype)
    learning_rate = jnp.asarray(1e-2, dtype=dtype)
    expected_gradient = jnp.vdot(weights_D, _addition_curvature_derivative_at_zero(x_D, y_D))

    model = _CurvatureConsumer(dtype)
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate), wrt=nnx.Param)
    grads = nnx.grad(lambda m: m(x_D, y_D, weights_D))(model)
    actual_gradient = grads.curvature.raw[...]
    atol, rtol = _tolerances(dtype)
    assert expected_gradient != 0
    assert jnp.allclose(actual_gradient, expected_gradient, atol=atol, rtol=rtol)

    before = model.curvature()
    optimizer.update(model, grads)
    after = model.curvature()
    expected_after = before - learning_rate * expected_gradient
    assert after != before
    assert jnp.sign(after - before) == jnp.sign(-expected_gradient)
    assert jnp.allclose(after, expected_after, atol=atol, rtol=rtol)
