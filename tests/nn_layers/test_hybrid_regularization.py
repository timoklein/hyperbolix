"""Tests for HyperPPFeatureScaling (Hyper++ feature scaling layer).

Dimension key: B=batch, D=embedding dimension

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HyperPPFeatureScaling-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.nn_layers import HyperPPFeatureScaling

jax.config.update("jax_enable_x64", True)

B, D = 8, 16


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["f32", "f64"])
def dtype(request):
    return request.param


@pytest.fixture()
def x_BD(dtype):
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, (B, D), dtype=dtype)


# ---------------------------------------------------------------------------
# 1. Forward shape/dtype and jit, in the parameter-free configurations
#    (the alpha=0.9 configuration is covered by test_layer_contract.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", [None, jax.nn.gelu], ids=["no_activation", "gelu"])
def test_forward_shape_dtype_and_jit_parameter_free(x_BD, dtype, activation):
    layer = HyperPPFeatureScaling(dim=D, activation=activation, alpha=None, rngs=nnx.Rngs(0))

    y_BD = layer(x_BD, c=1.0)

    assert y_BD.shape == (B, D)
    assert y_BD.dtype == dtype
    assert jnp.all(jnp.isfinite(y_BD))
    assert jnp.allclose(jax.jit(lambda x: layer(x, c=1.0))(x_BD), y_BD)


# ---------------------------------------------------------------------------
# 2. Curvature enters only through rho_max = atanh(alpha)/sqrt(c)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.1, 0.5, 2.0])
def test_curvature_scales_output_by_inverse_sqrt_c(x_BD, dtype, c):
    """The learned rescaling is the only c-dependence, so y(c) = y(1)/sqrt(c) exactly.

    Steps 1-3 (RMSNorm, activation, 1/sqrt(d)) and the sigmoid gate all see the
    c-independent pre-rescale features; only rho_max = atanh(alpha)/sqrt(c) carries
    the curvature (van Spengler et al. 2023, Sec. 3.2). The old version of this test
    guarded its single assertion with ``if c != 1.0`` and carried a c=1.0 parameter
    that asserted nothing.
    """
    atol = 4e-3 if dtype == jnp.float32 else 1e-12
    layer = HyperPPFeatureScaling(dim=D, alpha=0.9, rngs=nnx.Rngs(0))

    y_c_BD = layer(x_BD, c=c)
    y_1_BD = layer(x_BD, c=1.0)

    assert not jnp.allclose(y_c_BD, y_1_BD), f"c={c} should differ from c=1.0"
    assert jnp.allclose(y_c_BD * jnp.sqrt(jnp.array(c, dtype=dtype)), y_1_BD, atol=atol)

    # Parameter-free mode has no rho_max, so it must ignore c entirely.
    free_layer = HyperPPFeatureScaling(dim=D, alpha=None, rngs=nnx.Rngs(0))
    assert jnp.allclose(free_layer(x_BD, c=c), free_layer(x_BD, c=1.0), atol=atol)


# ---------------------------------------------------------------------------
# 3. RMSNorm correctness
# ---------------------------------------------------------------------------


def test_rmsnorm_correctness(x_BD, dtype):
    layer = HyperPPFeatureScaling(dim=D, activation=None, alpha=None, rngs=nnx.Rngs(0))
    y_BD = layer(x_BD, c=1.0)

    # Compare against standalone nnx.RMSNorm (no scale) + dim scaling
    ref_rms = nnx.RMSNorm(D, use_scale=False, rngs=nnx.Rngs(0))
    expected_BD = ref_rms(x_BD) / jnp.sqrt(jnp.array(D, dtype=dtype))

    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    assert jnp.allclose(y_BD, expected_BD, atol=atol)


# ---------------------------------------------------------------------------
# 4. Gradient computation (parameter-free mode)
# ---------------------------------------------------------------------------


def test_gradients_without_alpha(dtype):
    """Parameter-free mode: grad w.r.t. input should be finite."""
    key = jax.random.PRNGKey(2)
    x_BD = jax.random.normal(key, (B, D), dtype=dtype)
    layer = HyperPPFeatureScaling(dim=D, rngs=nnx.Rngs(0))

    def loss_fn(x):
        return jnp.sum(layer(x, c=1.0) ** 2)

    grad_BD = jax.grad(loss_fn)(x_BD)
    assert jnp.all(jnp.isfinite(grad_BD))
    assert grad_BD.shape == (B, D)


# ---------------------------------------------------------------------------
# 5. Alpha validation errors
# ---------------------------------------------------------------------------


def test_alpha_out_of_range():
    with pytest.raises(ValueError, match="alpha must be in"):
        HyperPPFeatureScaling(dim=D, alpha=0.0, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="alpha must be in"):
        HyperPPFeatureScaling(dim=D, alpha=1.0, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="alpha must be in"):
        HyperPPFeatureScaling(dim=D, alpha=-0.5, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="alpha must be in"):
        HyperPPFeatureScaling(dim=D, alpha=1.5, rngs=nnx.Rngs(0))


# ---------------------------------------------------------------------------
# 6. Output norm bounded by rho_max
# ---------------------------------------------------------------------------


def test_output_norm_bounded(dtype):
    import math

    alpha = 0.9
    c = 1.0
    rho_max = math.atanh(alpha) / math.sqrt(c)

    key = jax.random.PRNGKey(3)
    # Use larger values to stress-test the bound
    x_BD = jax.random.normal(key, (B, D), dtype=dtype) * 10.0
    layer = HyperPPFeatureScaling(dim=D, alpha=alpha, rngs=nnx.Rngs(0))
    y_BD = layer(x_BD, c=c)

    norms_B = jnp.linalg.norm(y_BD, axis=-1)
    # After RMSNorm + tanh + dim_scaling, each component is bounded by 1/sqrt(D).
    # Then rescaling multiplies by rho_max * sigmoid(...) where sigmoid < 1.
    # So ||y|| < rho_max * sqrt(D) * (1/sqrt(D)) = rho_max
    assert jnp.all(norms_B < rho_max + 1e-5), f"Norms {norms_B} exceed rho_max={rho_max}"


# ---------------------------------------------------------------------------
# 7. Integration: HyperPP -> expmap_0 -> is_in_manifold
# ---------------------------------------------------------------------------


def test_integration_poincare(dtype):
    dim = D
    c = 0.5
    poincare = Poincare(dtype=dtype)

    key = jax.random.PRNGKey(4)
    x_BD = jax.random.normal(key, (B, dim), dtype=dtype)
    layer = HyperPPFeatureScaling(dim=dim, alpha=0.9, rngs=nnx.Rngs(0))
    scaled_BD = layer(x_BD, c=c)

    # expmap_0 on Poincare: tangent vector -> ball point
    expmap_batch = jax.vmap(poincare.expmap_0, in_axes=(0, None))
    points_BD = expmap_batch(scaled_BD, c)

    # All points should be in the Poincare ball
    check_batch = jax.vmap(poincare.is_in_manifold, in_axes=(0, None))
    on_manifold = check_batch(points_BD, c)
    assert jnp.all(on_manifold), "Some points not in Poincare ball"


def test_integration_hyperboloid(dtype):
    dim = D
    c = 0.5
    hyperboloid = Hyperboloid(dtype=dtype)

    key = jax.random.PRNGKey(5)
    x_BD = jax.random.normal(key, (B, dim), dtype=dtype)
    layer = HyperPPFeatureScaling(dim=dim, alpha=0.9, rngs=nnx.Rngs(0))
    scaled_BD = layer(x_BD, c=c)

    # Hyperboloid expmap_0 expects ambient dim (d+1) with first component = 0
    tangent_BDp1 = jnp.concatenate([jnp.zeros((B, 1), dtype=dtype), scaled_BD], axis=-1)
    expmap_batch = jax.vmap(hyperboloid.expmap_0, in_axes=(0, None))
    points_BDp1 = expmap_batch(tangent_BDp1, c)

    # All points should be on the hyperboloid
    check_batch = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))
    on_manifold = check_batch(points_BDp1, c)
    assert jnp.all(on_manifold), "Some points not on hyperboloid"
