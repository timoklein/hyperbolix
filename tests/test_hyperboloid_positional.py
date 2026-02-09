"""Tests for hyperboloid positional encoding layers."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds import hyperboloid
from hyperbolix.nn_layers.hyperboloid_core import lorentz_residual
from hyperbolix.nn_layers.hyperboloid_positional import (
    HyperbolicRoPE,
    HypformerPositionalEncoding,
    hope,
)

# ============================================================================
# Helper: batch-aware Minkowski inner product
# ============================================================================


def _minkowski_inner_batch(x, y):
    """Minkowski inner product, batched: <x,y>_L = -x_0*y_0 + dot(x_s, y_s)."""
    return -x[..., 0] * y[..., 0] + jnp.sum(x[..., 1:] * y[..., 1:], axis=-1)


def _make_hyperboloid_points(key, shape, c=1.0, scale=0.1):
    """Generate valid hyperboloid points: shape = (..., dim) for spatial dims."""
    v = jax.random.normal(key, shape) * scale
    # expmap_0 expects (dim+1,) with v[0]=0, but we use proj for simplicity
    # Build ambient vectors: prepend a 1.0 for time, then project
    time = jnp.ones((*shape[:-1], 1))
    ambient = jnp.concatenate([time, v], axis=-1)  # (..., dim+1)
    return hyperboloid._proj_batch(ambient, c)


def _make_hyperboloid_seq(key, batch, seq_len, d, c=1.0, scale=0.1):
    """Generate (batch, seq_len, d+1) valid hyperboloid points."""
    return _make_hyperboloid_points(key, (batch, seq_len, d), c=c, scale=scale)


# ============================================================================
# Lorentz Residual Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_lorentz_residual_manifold_constraint(dtype, c):
    """Output of lorentz_residual lies on the hyperboloid."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    x = _make_hyperboloid_points(key1, (8, 4), c=c).astype(dtype)
    y = _make_hyperboloid_points(key2, (8, 4), c=c).astype(dtype)

    out = lorentz_residual(x, y, w_y=0.5, c=c)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(out, c)
    assert is_valid.all(), f"Manifold constraint violated for c={c}, dtype={dtype}"


@pytest.mark.parametrize("w_y", [0.0, 0.1, 0.5, 1.0, 2.0])
def test_lorentz_residual_various_weights(w_y):
    """Manifold constraint holds for a range of w_y values."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(99))
    c = 1.0
    x = _make_hyperboloid_points(key1, (8, 6), c=c)
    y = _make_hyperboloid_points(key2, (8, 6), c=c)

    out = lorentz_residual(x, y, w_y=w_y, c=c)

    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(out, c)
    assert is_valid.all(), f"Manifold constraint violated for w_y={w_y}"


def test_lorentz_residual_w_y_zero():
    """With w_y=0, output = x / sqrt(c * |<x,x>_L|) = x / sqrt(c * 1/c) = x."""
    key = jax.random.PRNGKey(7)
    c = 1.0
    x = _make_hyperboloid_points(key, (4, 6), c=c)
    y = jnp.zeros_like(x)  # y doesn't matter when w_y=0

    out = lorentz_residual(x, y, w_y=0.0, c=c)

    # For valid hyperboloid points: <x,x>_L = -1/c, so denom = sqrt(c * 1/c) = 1
    assert jnp.allclose(out, x, atol=1e-5), "w_y=0 should return input (up to normalization)"


def test_lorentz_residual_batch_seq_shape():
    """Lorentz residual preserves (batch, seq, d+1) shape."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(11))
    c = 1.0
    x = _make_hyperboloid_seq(key1, 2, 10, 8, c=c)
    y = _make_hyperboloid_seq(key2, 2, 10, 8, c=c)

    out = lorentz_residual(x, y, w_y=0.3, c=c)
    assert out.shape == x.shape


# ============================================================================
# HOPE / HyperbolicRoPE Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_hope_manifold_constraint_single(dtype, c):
    """HOPE output for a single sequence lies on the hyperboloid."""
    key = jax.random.PRNGKey(42)
    seq_len, d = 16, 8
    z = _make_hyperboloid_seq(key, 1, seq_len, d, c=c).astype(dtype)  # (1, seq, d+1)
    positions = jnp.arange(seq_len)

    out = hope(z, positions, c=c)

    assert out.shape == z.shape
    # Check manifold constraint for each point
    for b in range(1):
        for s in range(seq_len):
            assert hyperboloid.is_in_manifold(out[b, s], c, atol=1e-4), f"Failed at batch={b}, seq={s}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_hope_manifold_constraint_batch(dtype, c):
    """HOPE output for a batch of sequences lies on the hyperboloid."""
    key = jax.random.PRNGKey(55)
    batch, seq_len, d = 4, 12, 6
    z = _make_hyperboloid_seq(key, batch, seq_len, d, c=c).astype(dtype)
    positions = jnp.arange(seq_len)

    out = hope(z, positions, c=c)

    assert out.shape == z.shape
    for b in range(batch):
        for s in range(seq_len):
            assert hyperboloid.is_in_manifold(out[b, s], c, atol=1e-4), f"Failed at batch={b}, seq={s}, c={c}, dtype={dtype}"


def test_hope_spatial_norm_preservation():
    """Rotation is an isometry: ||R * z_s|| == ||z_s|| for each point."""
    key = jax.random.PRNGKey(77)
    batch, seq_len, d = 2, 10, 8
    c = 1.0
    z = _make_hyperboloid_seq(key, batch, seq_len, d, c=c)
    positions = jnp.arange(seq_len)

    out = hope(z, positions, c=c)

    original_norms = jnp.linalg.norm(z[..., 1:], axis=-1)  # (batch, seq)
    rotated_norms = jnp.linalg.norm(out[..., 1:], axis=-1)  # (batch, seq)

    assert jnp.allclose(original_norms, rotated_norms, atol=1e-5), (
        f"Spatial norms not preserved: max diff = {jnp.max(jnp.abs(original_norms - rotated_norms))}"
    )


def test_hope_identity_at_position_zero():
    """HOPE with position=0 should be the identity (rotation by 0 = no-op)."""
    key = jax.random.PRNGKey(88)
    seq_len, d = 1, 8
    c = 1.0
    z = _make_hyperboloid_seq(key, 1, seq_len, d, c=c)  # (1, 1, d+1)
    positions = jnp.array([0])

    out = hope(z, positions, c=c)

    assert jnp.allclose(out, z, atol=1e-6), f"Position 0 should be identity, max diff = {jnp.max(jnp.abs(out - z))}"


def test_hope_relative_position_property():
    """<HOPE(q, i), HOPE(k, j)>_L depends only on (i - j)."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(123))
    d = 8
    c = 1.0

    # Create two distinct hyperboloid points
    q = _make_hyperboloid_points(key1, (d,), c=c)  # (d+1,)
    k = _make_hyperboloid_points(key2, (d,), c=c)  # (d+1,)

    # Test pairs with the same relative offset = 3
    pairs = [(0, 3), (5, 8), (17, 20), (100, 103)]
    inner_products = []

    for i, j in pairs:
        # hope expects (..., seq, d+1) so add batch and seq dims
        q_enc = hope(q[None, None, :], jnp.array([i]), c=c)[0, 0]
        k_enc = hope(k[None, None, :], jnp.array([j]), c=c)[0, 0]
        ip = _minkowski_inner_batch(q_enc, k_enc)
        inner_products.append(ip)

    # All inner products should be equal (same relative offset)
    for idx, ip in enumerate(inner_products[1:], start=1):
        assert jnp.allclose(inner_products[0], ip, atol=1e-5), (
            f"Relative position property violated: pair 0 ip={inner_products[0]:.6f}, pair {idx} ip={ip:.6f}"
        )


def test_hope_different_offsets_differ():
    """<HOPE(q, i), HOPE(k, j)>_L differs for different relative offsets."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(456))
    d = 8
    c = 1.0

    q = _make_hyperboloid_points(key1, (d,), c=c)
    k = _make_hyperboloid_points(key2, (d,), c=c)

    # Offset = 0 vs offset = 5
    q0 = hope(q[None, None, :], jnp.array([0]), c=c)[0, 0]
    k0 = hope(k[None, None, :], jnp.array([0]), c=c)[0, 0]
    ip_offset_0 = _minkowski_inner_batch(q0, k0)

    q5 = hope(q[None, None, :], jnp.array([0]), c=c)[0, 0]
    k5 = hope(k[None, None, :], jnp.array([5]), c=c)[0, 0]
    ip_offset_5 = _minkowski_inner_batch(q5, k5)

    assert not jnp.allclose(ip_offset_0, ip_offset_5, atol=1e-4), "Different offsets should yield different inner products"


def test_hope_shape_preservation():
    """HOPE preserves input shape."""
    key = jax.random.PRNGKey(10)
    batch, seq_len, d = 3, 7, 10
    c = 1.0
    z = _make_hyperboloid_seq(key, batch, seq_len, d, c=c)
    positions = jnp.arange(seq_len)

    out = hope(z, positions, c=c)
    assert out.shape == (batch, seq_len, d + 1)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hope_gradient_flow(dtype):
    """Gradients through HOPE are finite."""
    key = jax.random.PRNGKey(42)
    batch, seq_len, d = 2, 8, 6
    c = 1.0
    z = _make_hyperboloid_seq(key, batch, seq_len, d, c=c).astype(dtype)
    positions = jnp.arange(seq_len)

    def loss_fn(z):
        out = hope(z, positions, c=c)
        return jnp.sum(out**2)

    grad = jax.grad(loss_fn)(z)

    assert jnp.isfinite(grad).all()
    assert grad.shape == z.shape


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hope_jit_compatibility(dtype):
    """HOPE works under JIT compilation."""
    key = jax.random.PRNGKey(42)
    batch, seq_len, d = 2, 8, 6
    c = 1.0
    z = _make_hyperboloid_seq(key, batch, seq_len, d, c=c).astype(dtype)
    positions = jnp.arange(seq_len)

    @jax.jit
    def apply_hope(z):
        return hope(z, positions, c=c)

    out = apply_hope(z)
    out_eager = hope(z, positions, c=c)

    assert jnp.allclose(out, out_eager, atol=1e-6)


def test_hyperbolic_rope_module():
    """HyperbolicRoPE module produces valid output."""
    key = jax.random.PRNGKey(42)
    batch, seq_len, d = 2, 8, 6
    c = 1.0
    z = _make_hyperboloid_seq(key, batch, seq_len, d, c=c)
    positions = jnp.arange(seq_len)

    rope = HyperbolicRoPE(dim=d, max_seq_len=64)
    out = rope(z, positions, c=c)

    assert out.shape == z.shape
    # Check manifold
    for b in range(batch):
        for s in range(seq_len):
            assert hyperboloid.is_in_manifold(out[b, s], c, atol=1e-4)


# ============================================================================
# HypformerPositionalEncoding Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_hypformer_pe_manifold_constraint_single(dtype, c):
    """HypformerPE output for a single point lies on the hyperboloid."""
    key = jax.random.PRNGKey(42)
    d = 6  # spatial dim
    in_features = d + 1  # ambient dim
    x = _make_hyperboloid_points(key, (8, d), c=c).astype(dtype)  # (8, d+1)

    pe = HypformerPositionalEncoding(in_features, d, rngs=nnx.Rngs(0))
    out = pe(x, c=c)

    assert out.shape == x.shape
    is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None))(out, c)
    assert is_valid.all(), f"Manifold constraint violated for c={c}, dtype={dtype}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_hypformer_pe_manifold_constraint_batch_seq(dtype, c):
    """HypformerPE output for (batch, seq, d+1) lies on the hyperboloid."""
    key = jax.random.PRNGKey(55)
    batch, seq_len, d = 2, 10, 6
    in_features = d + 1
    z = _make_hyperboloid_seq(key, batch, seq_len, d, c=c).astype(dtype)

    pe = HypformerPositionalEncoding(in_features, d, rngs=nnx.Rngs(0))
    out = pe(z, c=c)

    assert out.shape == z.shape
    for b in range(batch):
        for s in range(seq_len):
            assert hyperboloid.is_in_manifold(out[b, s], c, atol=1e-4), f"Failed at batch={b}, seq={s}, c={c}, dtype={dtype}"


def test_hypformer_pe_shape_preservation():
    """HypformerPE preserves input shape for various dimensions."""
    key = jax.random.PRNGKey(10)
    for d in [4, 8, 16]:
        in_features = d + 1
        x = _make_hyperboloid_points(key, (5, d), c=1.0)
        pe = HypformerPositionalEncoding(in_features, d, rngs=nnx.Rngs(0))
        out = pe(x, c=1.0)
        assert out.shape == x.shape, f"Shape mismatch for d={d}"


def test_hypformer_pe_learnable_epsilon():
    """epsilon is an nnx.Param and changes during a gradient step."""
    d = 6
    in_features = d + 1
    key = jax.random.PRNGKey(42)
    x = _make_hyperboloid_points(key, (4, d), c=1.0)

    pe = HypformerPositionalEncoding(in_features, d, rngs=nnx.Rngs(0))
    assert isinstance(pe.epsilon, nnx.Param), "epsilon should be nnx.Param"
    epsilon_before = pe.epsilon[...].copy()

    # Take a gradient step
    optimizer = nnx.Optimizer(pe, optax.adam(1e-2), wrt=nnx.Param)

    def loss_fn(model):
        out = model(x, c=1.0)
        return jnp.sum(out**2)

    _loss, grads = nnx.value_and_grad(loss_fn)(pe)
    optimizer.update(pe, grads)

    epsilon_after = pe.epsilon[...]
    assert not jnp.allclose(epsilon_before, epsilon_after, atol=1e-8), "epsilon should change after gradient step"


def test_hypformer_pe_has_htclinear_params():
    """HypformerPE contains HTCLinear kernel and bias parameters."""
    d = 6
    in_features = d + 1
    pe = HypformerPositionalEncoding(in_features, d, rngs=nnx.Rngs(0))

    assert hasattr(pe.htc_linear, "kernel")
    assert isinstance(pe.htc_linear.kernel, nnx.Param)
    assert hasattr(pe.htc_linear, "bias")
    assert isinstance(pe.htc_linear.bias, nnx.Param)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypformer_pe_gradient_flow(dtype):
    """Gradients through HypformerPE are finite."""
    key = jax.random.PRNGKey(42)
    d = 6
    in_features = d + 1
    x = _make_hyperboloid_points(key, (4, d), c=1.0).astype(dtype)

    pe = HypformerPositionalEncoding(in_features, d, rngs=nnx.Rngs(0))

    def loss_fn(model):
        out = model(x, c=1.0)
        return jnp.sum(out**2)

    loss, grads = nnx.value_and_grad(loss_fn)(pe)

    assert jnp.isfinite(loss)
    # Check all parameter gradients are finite
    params = nnx.state(grads, nnx.Param)
    flat_params = jax.tree.leaves(params)
    for p in flat_params:
        assert jnp.isfinite(p).all(), "Non-finite gradient found"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hypformer_pe_jit_compatibility(dtype):
    """HypformerPE works under JIT compilation."""
    key = jax.random.PRNGKey(42)
    d = 6
    in_features = d + 1
    x = _make_hyperboloid_points(key, (4, d), c=1.0).astype(dtype)

    pe = HypformerPositionalEncoding(in_features, d, rngs=nnx.Rngs(0))

    @nnx.jit
    def apply_pe(model, x):
        return model(x, c=1.0)

    out_jit = apply_pe(pe, x)
    out_eager = pe(x, c=1.0)

    assert jnp.allclose(out_jit, out_eager, atol=1e-6)
