"""Regression guards: float32 NN layers must stay float32 under global x64.

``tests/conftest.py`` enables ``jax_enable_x64`` globally. Under that flag any
array built without an explicit ``dtype=`` (weight inits via ``jax.random.*``,
``jnp.zeros/full``, and ``jnp.arange``) is float64. If a layer then combines such
a value with a float32 input in a raw einsum/matmul/concat — i.e. *before* a
``_cast``-ing manifold method — the whole computation silently promotes to
float64, ignoring the manifold dtype contract.

Each test below constructs a float32 manifold + layer, feeds an explicitly
float32 input, and asserts the output (and any persistent state) stays float32.
These would fail on the un-cast code paths the boundary casts now guard.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare
from hyperbolix.nn_layers import (
    FGGMeanOnlyBatchNorm,
    HTCLinear,
    HypLinearHyperboloidFHCNN,
    HypLinearHyperboloidFHNN,
    HypLinearPoincare,
    PoincareBatchNorm2D,
    hope,
)

F32 = jnp.float32


def _hyperboloid_points(key, batch, ambient, c=1.0):
    """A batch of float32 points on the hyperboloid of ambient dim ``ambient``."""
    v = jax.random.normal(key, (batch, ambient), dtype=F32) * 0.1
    return jax.vmap(Hyperboloid(dtype=F32).expmap_0, in_axes=(0, None))(v, c)


def test_x64_is_globally_enabled():
    """Sanity check: the leaks only matter when x64 is on (conftest enables it)."""
    assert jax.config.jax_enable_x64 is True


def test_hope_respects_float32():
    # HOPE rotary encoding: jnp.arange(0, d, 2)/d previously forced float64 freqs.
    key = jax.random.PRNGKey(0)
    batch, seq, d = 2, 6, 8  # spatial d must be even; ambient = d + 1
    tangent = jax.random.normal(key, (batch, seq, d + 1), dtype=F32) * 0.1
    z = jax.vmap(jax.vmap(Hyperboloid(dtype=F32).expmap_0, in_axes=(0, None)), in_axes=(0, None))(tangent, 1.0)
    positions = jnp.arange(seq)  # int — must not drag the output to float64

    out = hope(z, positions, c=1.0)
    assert out.dtype == F32


def test_htclinear_respects_float32():
    # HTCLinear: out = z @ kernel; kernel is float64 under x64 without the cast.
    x = _hyperboloid_points(jax.random.PRNGKey(1), batch=8, ambient=5)
    layer = HTCLinear(in_features=5, out_features=8, rngs=nnx.Rngs(0))
    assert layer(x, c_in=1.0, c_out=1.0).dtype == F32


def test_fhcnn_linear_respects_float32():
    # _fhcnn_forward: einsum(x, kernel) + bias before spatial_to_hyperboloid.
    manifold = Hyperboloid(dtype=F32)
    x = _hyperboloid_points(jax.random.PRNGKey(2), batch=8, ambient=6)
    layer = HypLinearHyperboloidFHCNN(manifold, 6, 10, rngs=nnx.Rngs(0))
    assert layer(x, c=1.0).dtype == F32


def test_fhnn_linear_respects_float32():
    # _fhnn_forward: same einsum-before-reconstruction path as FHCNN.
    manifold = Hyperboloid(dtype=F32)
    x = _hyperboloid_points(jax.random.PRNGKey(3), batch=8, ambient=6)
    layer = HypLinearHyperboloidFHNN(manifold, 6, 10, rngs=nnx.Rngs(0))
    assert layer(x, c=1.0).dtype == F32


def test_poincare_linear_respects_float32():
    # HypLinearPoincare: jnp.einsum(x_BI, kernel) in tangent space.
    manifold = Poincare(dtype=F32)
    x = jax.random.normal(jax.random.PRNGKey(4), (8, 5), dtype=F32) * 0.1  # tangent input
    layer = HypLinearPoincare(manifold, in_dim=5, out_dim=3, input_space="tangent", rngs=nnx.Rngs(0))
    assert layer(x, c=1.0).dtype == F32


def test_poincare_batchnorm_respects_float32():
    # var / running stats were float64 params; the tangent-space scaling promoted.
    manifold = Poincare(dtype=F32)
    bn = PoincareBatchNorm2D(manifold, num_features=8)
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 4, 8), dtype=F32) * 0.1  # tangent input
    out = bn(x, c=1.0)  # train mode (default) — also updates running stats
    assert out.dtype == F32
    # Persistent state must not silently become float64 either.
    assert bn.running_mean[...].dtype == F32
    assert bn.running_var[...].dtype == F32
    assert bn.var[...].dtype == F32


def test_fgg_mean_only_batchnorm_respects_float32():
    # bias / running_mean were float64; z - mean + bias promoted under x64.
    bn = FGGMeanOnlyBatchNorm(num_features=8)
    x = _hyperboloid_points(jax.random.PRNGKey(6), batch=16, ambient=9)
    out = bn(x, c_in=1.0, c_out=1.0)  # train mode — updates running_mean
    assert out.dtype == F32
    assert bn.running_mean[...].dtype == F32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
