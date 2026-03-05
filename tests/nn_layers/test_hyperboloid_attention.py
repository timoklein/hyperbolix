"""Tests for hyperbolic attention layers and supporting utilities."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import (
    HyperbolicFullAttention,
    HyperbolicLinearAttention,
    HyperbolicSoftmaxAttention,
    HypRegressionHyperboloid,
    focus_transform,
    lorentz_midpoint,
    lorentz_residual,
    spatial_to_hyperboloid,
)
from hyperbolix.nn_layers.hyperboloid_core import hrc

hyperboloid = Hyperboloid()


# ---------------------------------------------------------------------------
# Helper: create batch of hyperboloid points with shape (B, N, A)
# ---------------------------------------------------------------------------


def _make_hyp_points(key, batch, seq_len, ambient_dim, c):
    """Create (B, N, A) hyperboloid points."""
    raw = jax.random.normal(key, (batch, seq_len, ambient_dim)) * 0.3
    proj_fn = jax.vmap(jax.vmap(hyperboloid.proj, in_axes=(0, None)), in_axes=(0, None))
    return proj_fn(raw, c)


# ===================================================================
# spatial_to_hyperboloid tests
# ===================================================================


def test_spatial_to_hyperboloid_manifold_constraint():
    """Output satisfies <x,x>_L = -1/c_out."""
    c_out = 2.0
    spatial = jax.random.normal(jax.random.PRNGKey(0), (8, 5))
    out = spatial_to_hyperboloid(spatial, c_in=1.0, c_out=c_out)
    for i in range(8):
        assert hyperboloid.is_in_manifold(out[i], c_out, atol=1e-5)


def test_spatial_to_hyperboloid_equivalence_with_hrc():
    """spatial_to_hyperboloid(f(x[1:]), c_in, c_out) == hrc(x, f, c_in, c_out)."""
    c_in, c_out = 1.0, 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    f = jax.nn.relu
    y_hrc = hrc(x, f, c_in, c_out)
    y_s2h = spatial_to_hyperboloid(f(x[1:]), c_in, c_out)

    assert jnp.allclose(y_hrc, y_s2h, atol=1e-6)


def test_spatial_to_hyperboloid_zero_spatial():
    """Zero spatial → origin [1/sqrt(c_out), 0, …, 0]."""
    c_out = 2.0
    spatial = jnp.zeros(5)
    out = spatial_to_hyperboloid(spatial, c_in=1.0, c_out=c_out)
    expected = jnp.concatenate([jnp.array([1.0 / jnp.sqrt(c_out)]), jnp.zeros(5)])
    assert jnp.allclose(out, expected, atol=1e-6)


def test_spatial_to_hyperboloid_gradients_finite():
    spatial = jax.random.normal(jax.random.PRNGKey(1), (4, 5))

    def loss_fn(s):
        return jnp.sum(spatial_to_hyperboloid(s, 1.0, 2.0) ** 2)

    grads = jax.grad(loss_fn)(spatial)
    assert jnp.all(jnp.isfinite(grads))


# ===================================================================
# lorentz_midpoint tests
# ===================================================================


def test_lorentz_midpoint_on_manifold():
    """Output on manifold for uniform weights."""
    c = 1.0
    key = jax.random.PRNGKey(2)
    M, A = 6, 5
    raw = jax.random.normal(key, (M, A)) * 0.3
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(raw, c)  # (M, A)

    # uniform weights: (1, M)
    weights = jnp.ones((1, M)) / M
    mid = lorentz_midpoint(points, weights, c)  # (1, A)
    assert hyperboloid.is_in_manifold(mid[0], c, atol=1e-5)


def test_lorentz_midpoint_identical_points():
    """Uniform weights over identical points → same point."""
    c = 1.0
    point = hyperboloid.proj(jnp.array([1.5, 0.3, -0.1, 0.2]), c)
    points = jnp.tile(point[None, :], (5, 1))  # (5, A)
    weights = jnp.ones((1, 5)) / 5
    mid = lorentz_midpoint(points, weights, c)  # (1, A)
    assert jnp.allclose(mid[0], point, atol=1e-5)


def test_lorentz_midpoint_matches_lorentz_residual():
    """Matches lorentz_residual for 2-point case with w_y=1."""
    c = 1.0
    key1, key2 = jax.random.split(jax.random.PRNGKey(3))
    x = hyperboloid.proj(jax.random.normal(key1, (4,)) * 0.3, c)
    y = hyperboloid.proj(jax.random.normal(key2, (4,)) * 0.3, c)

    # lorentz_residual: ave = x + w_y * y, normalised
    ref = lorentz_residual(x, y, w_y=1.0, c=c)

    # lorentz_midpoint with weights [1, 1]
    points = jnp.stack([x, y], axis=0)  # (2, A)
    weights = jnp.array([[1.0, 1.0]])  # (1, 2)
    mid = lorentz_midpoint(points, weights, c)  # (1, A)

    assert jnp.allclose(mid[0], ref, atol=1e-5)


def test_lorentz_midpoint_gradients_finite():
    c = 1.0
    key = jax.random.PRNGKey(4)
    raw = jax.random.normal(key, (4, 5)) * 0.3
    points = jax.vmap(hyperboloid.proj, in_axes=(0, None))(raw, c)
    weights = jnp.ones((2, 4)) / 4

    def loss_fn(pts):
        return jnp.sum(lorentz_midpoint(pts, weights, c) ** 2)

    grads = jax.grad(loss_fn)(points)
    assert jnp.all(jnp.isfinite(grads))


# ===================================================================
# focus_transform tests
# ===================================================================


def test_focus_all_non_negative():
    """All outputs non-negative (ReLU ensures this)."""
    x = jax.random.normal(jax.random.PRNGKey(5), (8, 16))
    out = focus_transform(x, jnp.array(1.0), power=2.0)
    assert jnp.all(out >= 0)


def test_focus_norm_preserved():
    """||phi(x)|| ≈ ||relu(x) / |t|||."""
    x = jax.random.normal(jax.random.PRNGKey(6), (8, 16))
    t = jnp.array(1.0)
    eps = 1e-7
    out = focus_transform(x, t, power=2.0, eps=eps)

    e_tilde = (jax.nn.relu(x) + eps) / (jnp.abs(t) + eps)
    expected_norm = jnp.sqrt(jnp.sum(e_tilde**2, axis=-1) + eps)
    actual_norm = jnp.sqrt(jnp.sum(out**2, axis=-1) + eps)

    assert jnp.allclose(actual_norm, expected_norm, atol=1e-5)


def test_focus_higher_power_concentrates():
    """Higher p concentrates mass (entropy decreases)."""
    x = jax.random.normal(jax.random.PRNGKey(7), (16,))
    t = jnp.array(1.0)
    out_low = focus_transform(x, t, power=1.5)
    out_high = focus_transform(x, t, power=4.0)

    # Normalise to distributions
    def entropy(v):
        p = v / (v.sum() + 1e-10)
        p = jnp.clip(p, 1e-10, 1.0)
        return -jnp.sum(p * jnp.log(p))

    assert entropy(out_high) < entropy(out_low)


def test_focus_jit_and_grad():
    x = jax.random.normal(jax.random.PRNGKey(8), (4, 8))
    t = jnp.array(1.0)

    @jax.jit
    def fn(x_in, temp):
        return jnp.sum(focus_transform(x_in, temp, power=2.0) ** 2)

    val = fn(x, t)
    assert jnp.isfinite(val)

    grads = jax.grad(fn, argnums=1)(x, t)
    assert jnp.isfinite(grads)


# ===================================================================
# Attention class tests — shared parametrisation
# ===================================================================

ATTN_CLASSES = [HyperbolicLinearAttention, HyperbolicSoftmaxAttention, HyperbolicFullAttention]
CURVATURE_COMBOS = [(1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (0.5, 1.0, 2.0)]


def _make_attn(cls, in_features, out_features, num_heads, rngs):
    """Instantiate attention module with correct kwargs per class."""
    if cls is HyperbolicLinearAttention:
        return cls(in_features, out_features, num_heads=num_heads, power=2.0, rngs=rngs)
    return cls(in_features, out_features, num_heads=num_heads, rngs=rngs)


@pytest.mark.parametrize("cls", ATTN_CLASSES, ids=lambda c: c.__name__)
def test_attn_output_shape(cls):
    """Output shape: (B, N, out_features + 1)."""
    B, N, A_in, D_out = 2, 4, 6, 8
    rngs = nnx.Rngs(0)
    model = _make_attn(cls, A_in, D_out, num_heads=1, rngs=rngs)
    x = _make_hyp_points(jax.random.PRNGKey(10), B, N, A_in, c=1.0)
    y = model(x, c_in=1.0, c_attn=1.0, c_out=1.0)
    assert y.shape == (B, N, D_out + 1)


@pytest.mark.parametrize("cls", ATTN_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("c_in,c_attn,c_out", CURVATURE_COMBOS)
def test_attn_manifold_constraint(cls, c_in, c_attn, c_out):
    """All outputs on hyperboloid with c_out."""
    B, N, A_in, D_out = 2, 4, 6, 5
    rngs = nnx.Rngs(0)
    model = _make_attn(cls, A_in, D_out, num_heads=1, rngs=rngs)
    x = _make_hyp_points(jax.random.PRNGKey(11), B, N, A_in, c=c_in)
    y = model(x, c_in=c_in, c_attn=c_attn, c_out=c_out)
    for b in range(B):
        for n in range(N):
            assert hyperboloid.is_in_manifold(y[b, n], c_out, atol=1e-4)


@pytest.mark.parametrize("cls", ATTN_CLASSES, ids=lambda c: c.__name__)
def test_attn_gradient_finite(cls):
    """Gradient finite for all parameters."""
    B, N, A_in, D_out = 2, 4, 6, 5
    rngs = nnx.Rngs(0)
    model = _make_attn(cls, A_in, D_out, num_heads=1, rngs=rngs)
    x = _make_hyp_points(jax.random.PRNGKey(12), B, N, A_in, c=1.0)

    def loss_fn(m):
        return jnp.sum(m(x, c_in=1.0, c_attn=1.0, c_out=1.0) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    assert jnp.isfinite(loss)

    leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.parametrize("cls", ATTN_CLASSES, ids=lambda c: c.__name__)
def test_attn_jit(cls):
    """JIT compatible."""
    B, N, A_in, D_out = 2, 4, 6, 5
    rngs = nnx.Rngs(0)
    model = _make_attn(cls, A_in, D_out, num_heads=1, rngs=rngs)
    x = _make_hyp_points(jax.random.PRNGKey(13), B, N, A_in, c=1.0)

    @nnx.jit
    def forward(m, inp):
        return m(inp, c_in=1.0, c_attn=1.0, c_out=1.0)

    y = forward(model, x)
    assert y.shape == (B, N, D_out + 1)
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize("cls", ATTN_CLASSES, ids=lambda c: c.__name__)
def test_attn_single_token(cls):
    """Single token edge case (N=1)."""
    B, N, A_in, D_out = 2, 1, 6, 5
    rngs = nnx.Rngs(0)
    model = _make_attn(cls, A_in, D_out, num_heads=1, rngs=rngs)
    x = _make_hyp_points(jax.random.PRNGKey(14), B, N, A_in, c=1.0)
    y = model(x, c_in=1.0, c_attn=1.0, c_out=1.0)
    assert y.shape == (B, 1, D_out + 1)
    for b in range(B):
        assert hyperboloid.is_in_manifold(y[b, 0], 1.0, atol=1e-4)


@pytest.mark.parametrize("cls", ATTN_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("num_heads", [2, 4])
def test_attn_multi_head(cls, num_heads):
    """Multi-head produces valid output."""
    B, N, A_in, D_out = 2, 4, 6, 5
    rngs = nnx.Rngs(0)
    model = _make_attn(cls, A_in, D_out, num_heads=num_heads, rngs=rngs)
    x = _make_hyp_points(jax.random.PRNGKey(15), B, N, A_in, c=1.0)
    y = model(x, c_in=1.0, c_attn=1.0, c_out=1.0)
    assert y.shape == (B, N, D_out + 1)
    for b in range(B):
        for n in range(N):
            assert hyperboloid.is_in_manifold(y[b, n], 1.0, atol=1e-4)


def test_attn_mechanisms_differ():
    """Linear != softmax != full outputs (different mechanisms)."""
    B, N, A_in, D_out = 2, 4, 6, 5
    rngs = nnx.Rngs(0)
    x = _make_hyp_points(jax.random.PRNGKey(16), B, N, A_in, c=1.0)

    linear = HyperbolicLinearAttention(A_in, D_out, rngs=rngs, power=2.0)
    softmax = HyperbolicSoftmaxAttention(A_in, D_out, rngs=rngs)
    full = HyperbolicFullAttention(A_in, D_out, rngs=rngs)

    y_lin = linear(x)
    y_sm = softmax(x)
    y_full = full(x)

    assert not jnp.allclose(y_lin, y_sm, atol=1e-4)
    assert not jnp.allclose(y_lin, y_full, atol=1e-4)
    assert not jnp.allclose(y_sm, y_full, atol=1e-4)


# ===================================================================
# Overfit / memorisation tests — proves the full pipeline trains
# ===================================================================


class _AttentionClassifier(nnx.Module):
    """Tiny model: attention → pool → classify. For overfit tests only."""

    def __init__(self, attn, ambient_out, num_classes, *, rngs):
        self.attn = attn
        self.head = HypRegressionHyperboloid(
            hyperboloid,
            ambient_out,
            num_classes,
            rngs=rngs,
        )

    def __call__(self, x_BNA, c=1.0):
        attended_BNA = self.attn(x_BNA, c_in=c, c_attn=c, c_out=c)  # (B, N, A)
        # Pool: mean spatial across sequence, reconstruct time
        pooled_spatial_BD = attended_BNA[..., 1:].mean(axis=1)  # (B, D)
        pooled_BA = spatial_to_hyperboloid(pooled_spatial_BD, c, c)  # (B, A)
        return self.head(pooled_BA, c)  # (B, num_classes)


def _run_overfit(attn_cls, num_steps=300, learning_rate=3e-3):
    """Train a tiny attention classifier to memorise 16 random examples.

    Returns (initial_loss, final_loss).
    """
    # Fixed tiny dataset: 16 sequences, 4 tokens, ambient_dim=6, 3 classes
    B, N, A_in, D_out, num_classes = 16, 4, 6, 8, 3
    c = 1.0
    rngs = nnx.Rngs(42)

    x_BNA = _make_hyp_points(jax.random.PRNGKey(99), B, N, A_in, c)
    labels_B = jax.random.randint(jax.random.PRNGKey(100), (B,), 0, num_classes)

    attn = _make_attn(attn_cls, A_in, D_out, num_heads=2, rngs=rngs)
    model = _AttentionClassifier(attn, D_out + 1, num_classes, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(m):
            logits_BC = m(x, c)
            return optax.softmax_cross_entropy_with_integer_labels(logits_BC, y).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    initial_loss = None
    final_loss = None
    for step in range(num_steps):
        loss = train_step(model, optimizer, x_BNA, labels_B)
        if step == 0:
            initial_loss = float(loss)
        if step == num_steps - 1:
            final_loss = float(loss)

    return initial_loss, final_loss


@pytest.mark.parametrize("cls", ATTN_CLASSES, ids=lambda c: c.__name__)
def test_attn_overfit(cls):
    """Each attention variant can memorise a tiny dataset (loss drops >50%)."""
    initial_loss, final_loss = _run_overfit(cls)
    assert final_loss < initial_loss * 0.5, f"Loss did not drop enough: {initial_loss:.4f} → {final_loss:.4f}"
