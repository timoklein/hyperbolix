"""Tests for the Poincaré-ball vector quantization layers.

Implements the verification harness from the porting brief
(``vqvae_nnx/hyperbolix-port.md``):

* **batch-independence** — every per-token op must be vmapped; quantizing a
  1-row slice must match the corresponding row of the full batch (GOTCHA #1);
* **shape / dtype contract** — the decoder input is float32 even under x64,
  indices are int (GOTCHA #2);
* **manifold membership** — lifted points and codebook rows lie on the ball;
* **gradient routing** — the load-bearing checks. For the EMA embedding layer
  the codebook is a *buffer* and receives no gradient (only the commitment loss
  reaches the encoder); for the MLR layer the STE sits on the categorical weights
  so the MLR ``kernel``/``bias`` *do* receive gradient (a regression guard against
  the silent "STE on z_q" bug);
* **train/eval determinism** — two eval forwards with different RNG agree, train
  forwards differ (GOTCHA #3);
* **EMA update** — codes move toward their assigned points, empty codes are kept,
  dead-code revival fires;
* **weighted-gyromidpoint helper** — idempotence, on-ball, empty-row safety.

Dimension key:
  N: tokens     C: code dim     K: num codes
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import (
    HypVQEmbeddingPoincare,
    HypVQMLRPoincare,
    PoincareVQOutput,
    poincare_weighted_midpoint,
)

DTYPES = [jnp.float32, jnp.float64]


def _tol(dtype: jnp.dtype) -> tuple[float, float]:
    """(atol, rtol) matching the project's conftest tolerances."""
    return (4e-3, 4e-3) if dtype == jnp.float32 else (1e-7, 1e-7)


def _inputs(seed: int, n: int, dim: int, dtype: jnp.dtype, scale: float = 0.3) -> jax.Array:
    """Random encoder tangent vectors (mid-ball after expmap_0)."""
    return jax.random.normal(jax.random.key(seed), (n, dim), dtype=dtype) * scale


def _all_on_ball(manifold: Poincare, pts: jax.Array, c: float) -> jax.Array:
    return jax.vmap(lambda p: manifold.is_in_manifold(p, c))(pts).all()


# --------------------------------------------------------------------------- #
# poincare_weighted_midpoint helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_weighted_midpoint_idempotent(dtype, dim, c):
    """Midpoint of N copies of a point returns that point (the ½⊗ gyro identity)."""
    manifold = Poincare(dtype=dtype)
    atol, rtol = _tol(dtype)
    p = manifold.expmap_0(jax.random.normal(jax.random.key(0), (dim,), dtype=dtype) * 0.3, c)
    points_MC = jnp.broadcast_to(p, (5, dim))
    weights_NM = jnp.ones((1, 5), dtype=dtype)
    mid = poincare_weighted_midpoint(points_MC, weights_NM, manifold, c)
    assert mid.shape == (1, dim)
    assert jnp.allclose(mid[0], p, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_weighted_midpoint_on_ball(dtype, dim, c):
    """Distinct points + random non-negative weights → midpoints on the ball."""
    manifold = Poincare(dtype=dtype)
    pts = jax.vmap(manifold.expmap_0, in_axes=(0, None))(_inputs(1, 6, dim, dtype), c)
    weights = jax.random.uniform(jax.random.key(2), (4, 6), dtype=dtype)
    mids = poincare_weighted_midpoint(pts, weights, manifold, c)
    assert mids.shape == (4, dim)
    assert jnp.isfinite(mids).all()
    assert _all_on_ball(manifold, mids, c)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_weighted_midpoint_empty_row_is_origin(dtype, dim, c):
    """An all-zero weight row maps to the origin without 0/0 NaNs."""
    manifold = Poincare(dtype=dtype)
    atol, _ = _tol(dtype)
    pts = jax.vmap(manifold.expmap_0, in_axes=(0, None))(_inputs(3, 6, dim, dtype), c)
    mid = poincare_weighted_midpoint(pts, jnp.zeros((1, 6), dtype=dtype), manifold, c)
    assert jnp.isfinite(mid).all()
    assert jnp.allclose(mid[0], jnp.zeros((dim,), dtype=dtype), atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_weighted_midpoint_jit(dtype, c):
    manifold = Poincare(dtype=dtype)
    pts = jax.vmap(manifold.expmap_0, in_axes=(0, None))(_inputs(4, 6, 5, dtype), c)
    weights = jax.random.uniform(jax.random.key(5), (3, 6), dtype=dtype)
    fn = jax.jit(lambda p, w: poincare_weighted_midpoint(p, w, manifold, c))
    assert jnp.isfinite(fn(pts, weights)).all()


# --------------------------------------------------------------------------- #
# HypVQEmbeddingPoincare (HVQ-VAE, EMA codebook)
# --------------------------------------------------------------------------- #
def _embedding(dtype, dim, num_codes=16, **kw):
    manifold = Poincare(dtype=dtype)
    layer = HypVQEmbeddingPoincare(manifold, num_codes=num_codes, code_dim=dim, rngs=nnx.Rngs(0), **kw)
    return manifold, layer


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_forward_contract(dtype, dim, c):
    manifold, layer = _embedding(dtype, dim)
    x = _inputs(1, 32, dim, dtype)
    out = layer(x, c)
    assert isinstance(out, PoincareVQOutput)
    assert out.quantized.shape == x.shape
    assert out.quantized.dtype == jnp.float32  # manifold→decoder boundary cast
    assert out.indices.shape == (32,)
    assert jnp.issubdtype(out.indices.dtype, jnp.integer)
    assert out.z.dtype == dtype  # manifold island stays in manifold dtype
    assert _all_on_ball(manifold, out.z, c)
    assert jnp.isfinite(out.loss) and jnp.isfinite(out.perplexity)
    assert 1.0 - 1e-4 <= out.perplexity <= layer.num_codes + 1e-4


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_batch_independence(dtype, dim, c):
    """quantizing x[:1] equals the first row of quantizing x (catches un-vmapped ops)."""
    _manifold, layer = _embedding(dtype, dim)
    x = _inputs(2, 16, dim, dtype)
    atol, rtol = _tol(dtype)
    assert jnp.allclose(layer(x[:1], c).quantized[0], layer(x, c).quantized[0], atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_codebook_on_ball(dtype, dim, c):
    manifold, layer = _embedding(dtype, dim)
    assert _all_on_ball(manifold, layer.codebook[...], c)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_ste_forward_identity(dtype, dim, c):
    """Forward value of the STE-bridged output equals logmap_0(selected code)."""
    manifold, layer = _embedding(dtype, dim)
    x = _inputs(3, 16, dim, dtype)
    out = layer(x, c)
    q = layer.codebook[...][out.indices]
    expected = jax.vmap(manifold.logmap_0, in_axes=(0, None))(q, c).astype(jnp.float32)
    atol, rtol = _tol(dtype)
    assert jnp.allclose(out.quantized, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_gradient_routes_to_encoder_not_codebook(dtype, dim, c):
    """Commitment loss reaches the encoder input; the codebook is a non-param buffer."""
    _manifold, layer = _embedding(dtype, dim)
    x = _inputs(4, 16, dim, dtype)
    # (a) the commitment loss flows back to the encoder (its input x).
    g_x = jax.grad(lambda xx: layer(xx, c).loss)(x)
    assert jnp.isfinite(g_x).all()
    assert jnp.any(jnp.abs(g_x) > 0)
    # (b) the codebook is an nnx.Variable buffer, NOT a Param → never differentiated.
    assert isinstance(layer.codebook, nnx.Variable) and not isinstance(layer.codebook, nnx.Param)
    assert len(jax.tree_util.tree_leaves(nnx.state(layer, nnx.Param))) == 0


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_ema_update_moves_assigned_codes(dtype, dim, c):
    manifold, layer = _embedding(dtype, dim, num_codes=16)
    out = layer(_inputs(7, 64, dim, dtype), c)
    before = layer.codebook[...]
    n_dead = layer.ema_update(out.z, out.indices, c)  # revival off
    after = layer.codebook[...]
    assert int(n_dead) == 0
    assert _all_on_ball(manifold, after, c)
    counts = jnp.bincount(out.indices, length=layer.num_codes)
    moved = jnp.linalg.norm(after - before, axis=-1)
    assert jnp.all(moved[counts == 0] == 0)  # empty codes unchanged
    assert jnp.any(moved[counts > 0] > 0)  # assigned codes moved
    # the per-code usage EMA advanced away from its initial credit
    assert jnp.any(layer.cluster_size[...] != 2.0 * layer.dead_code_threshold)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_ema_empty_codes_unchanged(dtype, dim, c):
    """Codes with no assigned points this batch are left exactly as they were."""
    manifold, layer = _embedding(dtype, dim, num_codes=8)
    z = jax.vmap(manifold.expmap_0, in_axes=(0, None))(_inputs(9, 10, dim, dtype), c)
    indices = jnp.zeros((10,), dtype=jnp.int32)  # everything assigned to code 0
    before = layer.codebook[...]
    layer.ema_update(z, indices, c)
    moved = jnp.linalg.norm(layer.codebook[...] - before, axis=-1)
    assert moved[0] > 0  # code 0 moved
    assert jnp.all(moved[1:] == 0)  # codes 1..K-1 untouched


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_ema_dead_code_revival(dtype, dim, c):
    manifold, layer = _embedding(dtype, dim, num_codes=8, dead_code_revival=True, dead_code_threshold=1.0)
    out = layer(_inputs(11, 32, dim, dtype), c)
    layer.cluster_size[...] = jnp.zeros((8,), dtype=dtype)  # force every code stale
    before = layer.codebook[...]
    n_dead = layer.ema_update(out.z, out.indices, c, reset_key=jax.random.key(0))
    assert int(n_dead) > 0
    assert _all_on_ball(manifold, layer.codebook[...], c)
    assert jnp.any(jnp.linalg.norm(layer.codebook[...] - before, axis=-1) > 0)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_embedding_ema_revival_requires_key(dtype):
    _manifold, layer = _embedding(dtype, 5, num_codes=8, dead_code_revival=True)
    out = layer(_inputs(12, 16, 5, dtype), 1.0)
    with pytest.raises(ValueError):
        layer.ema_update(out.z, out.indices, 1.0, reset_key=None)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_embedding_jit(dtype, c):
    _manifold, layer = _embedding(dtype, 8)
    x = _inputs(13, 16, 8, dtype)

    @nnx.jit
    def fwd(m, xx):
        return m(xx, c)

    out = fwd(layer, x)
    assert jnp.isfinite(out.quantized).all() and jnp.isfinite(out.loss)


# --------------------------------------------------------------------------- #
# HypVQMLRPoincare (HyperVQ, MLR codebook)
# --------------------------------------------------------------------------- #
def _mlr(dtype, dim, num_codes=16):
    manifold = Poincare(dtype=dtype)
    layer = HypVQMLRPoincare(manifold, num_codes=num_codes, code_dim=dim, rngs=nnx.Rngs(0))
    return manifold, layer


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_mlr_forward_contract(dtype, dim, c):
    manifold, layer = _mlr(dtype, dim)
    x = _inputs(1, 32, dim, dtype)
    out = layer(x, c, rngs=nnx.Rngs(1))
    assert out.quantized.shape == x.shape
    assert out.quantized.dtype == jnp.float32
    assert out.indices.shape == (32,)
    assert jnp.issubdtype(out.indices.dtype, jnp.integer)
    assert out.z.dtype == dtype
    assert _all_on_ball(manifold, out.z, c)
    assert jnp.isfinite(out.quantized).all()
    assert 1.0 - 1e-4 <= out.perplexity <= layer.num_codes + 1e-4
    assert float(out.loss) == 0.0  # HyperVQ is reconstruction-only


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_mlr_batch_independence_eval(dtype, dim, c):
    """In eval (deterministic) mode the selection is a per-token op (catches un-vmapped lift)."""
    _manifold, layer = _mlr(dtype, dim)
    layer.eval()
    x = _inputs(2, 16, dim, dtype)
    atol, rtol = _tol(dtype)
    assert jnp.allclose(layer(x[:1], c).quantized[0], layer(x, c).quantized[0], atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_mlr_gradient_routes_to_mlr_params(dtype, dim, c):
    """STE on the categorical weights → MLR kernel & bias get non-zero gradient.

    Regression guard: with the STE on z_q instead, these would be exactly zero
    and the implicit codebook would never train.
    """
    _manifold, layer = _mlr(dtype, dim)
    layer.train()
    x = _inputs(4, 16, dim, dtype)
    target = _inputs(5, 16, dim, dtype).astype(jnp.float32)

    def loss_fn(m, rngs):
        out = m(x, c, rngs=rngs)
        return jnp.mean((out.quantized - target) ** 2)

    _loss, grads = nnx.value_and_grad(loss_fn)(layer, nnx.Rngs(7))
    gk = grads.mlr.kernel[...]
    gb = grads.mlr.bias[...]
    assert jnp.isfinite(gk).all() and jnp.isfinite(gb).all()
    assert jnp.any(jnp.abs(gk) > 0)
    assert jnp.any(jnp.abs(gb) > 0)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [2, 8], ids=["dim2", "dim8"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_mlr_train_eval_determinism(dtype, dim, c):
    _manifold, layer = _mlr(dtype, dim)
    x = _inputs(8, 32, dim, dtype)
    # eval: noise is unused → independent RNG streams give identical output.
    layer.eval()
    assert jnp.array_equal(layer(x, c, rngs=nnx.Rngs(1)).quantized, layer(x, c, rngs=nnx.Rngs(2)).quantized)
    # train: Gumbel noise makes independent RNG streams differ.
    layer.train()
    assert not jnp.array_equal(layer(x, c, rngs=nnx.Rngs(1)).quantized, layer(x, c, rngs=nnx.Rngs(2)).quantized)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("c", [0.1, 1.0], ids=["c0.1", "c1.0"])
def test_mlr_jit(dtype, c):
    _manifold, layer = _mlr(dtype, 8)
    x = _inputs(13, 16, 8, dtype)

    @nnx.jit
    def fwd(m, xx, rngs):
        return m(xx, c, rngs=rngs)

    out = fwd(layer, x, nnx.Rngs(3))
    assert jnp.isfinite(out.quantized).all()


# --------------------------------------------------------------------------- #
# Integration smoke: encoder → VQ → decoder, a few steps
# --------------------------------------------------------------------------- #
def test_embedding_integration_smoke():
    """Tiny Euclidean encoder/decoder around the EMA-VQ bottleneck trains a few steps."""
    dtype = jnp.float64
    dim_in, code_dim, num_codes, c = 12, 6, 8, 0.5
    manifold = Poincare(dtype=dtype)

    class Toy(nnx.Module):
        def __init__(self, rngs):
            self.enc = nnx.Linear(dim_in, code_dim, rngs=rngs)
            self.vq = HypVQEmbeddingPoincare(manifold, num_codes=num_codes, code_dim=code_dim, rngs=rngs)
            self.dec = nnx.Linear(code_dim, dim_in, rngs=rngs)

        def __call__(self, x, c):
            out = self.vq(self.enc(x), c)
            return self.dec(out.quantized), out

    model = Toy(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    x = jax.random.normal(jax.random.key(0), (32, dim_in), dtype=jnp.float32)

    def loss_fn(m):
        recon, out = m(x, c)
        return jnp.mean((recon - x) ** 2) + out.loss.astype(jnp.float32), out

    codebook_init = model.vq.codebook[...]
    losses = []
    for _ in range(5):
        (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        model.vq.ema_update(out.z, out.indices, c)  # EMA after the optimizer step
        losses.append(float(loss))

    assert all(jnp.isfinite(jnp.asarray(losses)))
    assert jnp.linalg.norm(model.vq.codebook[...] - codebook_init) > 0  # codebook moved
