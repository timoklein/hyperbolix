"""Tests for LorentzEmbedding, the hyperboloid token embedding of HELM (He et al. 2025).

Dimension key:
  V: vocabulary size   A: ambient dim (S + 1)   S: spatial dim
  B: batch             L: sequence length
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers.hyperboloid_embedding import LorentzEmbedding, lorentz_embedding_init
from hyperbolix.optim import ManifoldParam, riemannian_adam

V, A = 32, 9


def _atol(dtype):
    return 1e-5 if dtype == jnp.float32 else 1e-12


def _on_manifold(x_NA, c, dtype):
    """All rows of an (N, A) array pass Hyperboloid.is_in_manifold at an explicit atol."""
    manifold = Hyperboloid(dtype=dtype)
    return bool(jnp.all(jax.vmap(manifold.is_in_manifold, in_axes=(0, None, None))(x_NA, c, _atol(dtype))))


def _dist_0(x_NA, c, dtype):
    return jax.vmap(Hyperboloid(dtype=dtype).dist_0, in_axes=(0, None))(x_NA, c)


def _helm_reference_init(u_VA, k):
    """float64 NumPy transcription of HELM's init (Lorentz.random_normal -> geoopt expmap0 + project).

    helm/hypercore/manifolds/lorentzian.py::random_normal (normalize over all A coords) and
    geoopt/manifolds/lorentz/math.py::_expmap0, _norm (Minkowski norm clamped at 1e-8), _project.
    HELM curvature k: <x, x>_L = -k.
    """
    u_VA = u_VA / np.linalg.norm(u_VA, axis=-1, keepdims=True)
    minkowski_V1 = -(u_VA[:, :1] ** 2) + np.sum(u_VA[:, 1:] ** 2, axis=-1, keepdims=True)
    nomin_V1 = np.sqrt(np.maximum(minkowski_V1, 1e-8))
    sqrt_k = np.sqrt(k)
    l_v = np.cosh(nomin_V1 / sqrt_k) * sqrt_k
    r_v = sqrt_k * np.sinh(nomin_V1 / sqrt_k) * u_VA / nomin_V1
    p_VA = np.concatenate([l_v + r_v[:, :1], r_v[:, 1:]], axis=-1)
    space_VS = p_VA[:, 1:]
    time_V1 = np.sqrt(k + np.sum(space_VS**2, axis=-1, keepdims=True))
    return np.concatenate([time_V1, space_VS], axis=-1), u_VA[:, 0]


# --------------------------------------------------------------------------- #
# Init
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_manifold_table_on_manifold_at_init(dtype, c):
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(0), c=c, param_dtype=dtype)

    assert isinstance(emb.embedding, ManifoldParam)
    assert emb.embedding.curvature == c
    assert isinstance(emb.embedding.manifold, Hyperboloid)
    table_VA = emb.embedding[...]
    assert table_VA.shape == (V, A)
    assert table_VA.dtype == dtype
    assert _on_manifold(table_VA, c, dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("radius", [0.3, 1.0, 3.0])
def test_init_rows_at_intended_geodesic_radius(dtype, c, radius):
    table_VA = lorentz_embedding_init(jax.random.key(1), V, A, c, radius=radius, dtype=dtype)

    assert _on_manifold(table_VA, c, dtype)
    rtol = 1e-5 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(np.asarray(_dist_0(table_VA, c, dtype)), radius, rtol=rtol)


def test_default_init_radius_is_one():
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(2), c=0.5, param_dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(_dist_0(emb.embedding[...], 0.5, jnp.float64)), 1.0, rtol=1e-12)


def test_init_directions_are_isotropic():
    """Directions u = x_s/‖x_s‖: mean near 0 and second moment near I/S (loose, 4096 rows, S = 8)."""
    n_rows, spatial = 4096, A - 1
    table_VA = lorentz_embedding_init(jax.random.key(3), n_rows, A, 1.0, dtype=jnp.float64)
    space_VS = np.asarray(table_VA[:, 1:])
    u_VS = space_VS / np.linalg.norm(space_VS, axis=-1, keepdims=True)

    # E‖mean‖² = 1/V for i.i.d. unit vectors with zero mean.
    assert np.linalg.norm(u_VS.mean(axis=0)) < 3.0 / np.sqrt(n_rows)
    second_moment_SS = u_VS.T @ u_VS / n_rows
    assert np.abs(second_moment_SS - np.eye(spatial) / spatial).max() < 0.02


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_reference_init_radius_agrees_within_its_u0_perturbation(c):
    """HELM's init lands at radius 1 - O(u_0²); the clean init at exactly 1 (float64).

    The reference normalizes over all A coords and feeds that non-tangent vector to geoopt's
    expmap0, so its Minkowski norm is sqrt(1 - 2·u_0²) and the radius deficit 1 - d lies between
    u_0²/2 and u_0² (logs/2026-09-30_helm-embedding/probe_deficit_bound.out). The last row is
    forced past the 1e-8 clamp (u_0² > 1/2), where the reference radius collapses instead.
    """
    k = 1.0 / c
    n_rows = 512
    u_VA = np.random.default_rng(4).normal(size=(n_rows, A))
    u_VA[-1] = np.concatenate([[5.0], 0.1 * np.ones(A - 1)])
    ref_VA, u0_V = _helm_reference_init(u_VA, k)

    assert _on_manifold(jnp.asarray(ref_VA), c, jnp.float64)
    d_ref_V = np.asarray(_dist_0(jnp.asarray(ref_VA), c, jnp.float64))
    # The two conventions describe the same point set: HELM's own radius formula at k agrees.
    np.testing.assert_allclose(
        d_ref_V, np.sqrt(k) * np.arcsinh(np.linalg.norm(ref_VA[:, 1:], axis=-1) / np.sqrt(k)), atol=1e-12
    )

    clean_VA = lorentz_embedding_init(jax.random.key(4), n_rows, A, c, dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(_dist_0(clean_VA, c, jnp.float64)), 1.0, rtol=1e-12)

    regular = u0_V**2 < 0.5
    # At A = 9 about 3 % of random rows also reach the clamp (probe_reference_radius.out).
    assert not regular[-1] and regular.mean() > 0.9
    deficit_V = 1.0 - d_ref_V[regular]
    u0_sq_V = u0_V[regular] ** 2
    assert np.all(deficit_V >= 0.5 * u0_sq_V - 1e-12)
    assert np.all(deficit_V <= u0_sq_V + 1e-12)
    assert d_ref_V[-1] < 0.5


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("parameterization", ["manifold", "spatial"])
def test_lookup_shape(parameterization):
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(0), parameterization=parameterization)
    ids_BL = jax.random.randint(jax.random.key(5), (3, 7), 0, V)

    assert emb(ids_BL).shape == (3, 7, A)
    assert emb(ids_BL, c_out=0.5).shape == (3, 7, A)
    assert emb(jnp.array(4)).shape == (A,)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_manifold_lookup_returns_table_rows(dtype):
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(0), c=0.5, param_dtype=dtype)
    ids_BL = jax.random.randint(jax.random.key(6), (4, 5), 0, V)

    out_BLA = emb(ids_BL)

    assert out_BLA.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out_BLA), np.asarray(emb.embedding[...])[np.asarray(ids_BL)])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(("c", "c_out"), [(1.0, 0.5), (0.5, 2.0), (2.0, 1.0)])
def test_cross_curvature_scales_onto_the_c_out_sheet(dtype, c, c_out):
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(0), c=c, param_dtype=dtype)
    ids_N = jnp.arange(V)

    same_NA = emb(ids_N)
    out_NA = emb(ids_N, c_out=c_out)

    rtol = 1e-6 if dtype == jnp.float32 else 1e-14
    np.testing.assert_allclose(np.asarray(out_NA), np.asarray(same_NA) * np.sqrt(c / c_out), rtol=rtol)
    assert _on_manifold(out_NA, c_out, dtype)
    # The map keeps the scaled radius sqrt(c)·d.
    np.testing.assert_allclose(
        np.sqrt(c_out) * np.asarray(_dist_0(out_NA, c_out, dtype)),
        np.sqrt(c) * np.asarray(_dist_0(same_NA, c, dtype)),
        rtol=1e-5 if dtype == jnp.float32 else 1e-12,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(("c", "c_out"), [(1.0, None), (0.5, None), (1.0, 0.5), (0.5, 2.0)])
def test_spatial_parameterization(dtype, c, c_out):
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(0), c=c, parameterization="spatial", param_dtype=dtype)
    ids_BL = jax.random.randint(jax.random.key(7), (4, 6), 0, V)

    assert type(emb.embedding) is nnx.Param
    table_VS = emb.embedding[...]
    assert table_VS.shape == (V, A - 1)
    assert table_VS.dtype == dtype

    out_BLA = emb(ids_BL, c_out=c_out)
    target_c = c if c_out is None else c_out

    assert out_BLA.dtype == dtype
    assert _on_manifold(out_BLA.reshape(-1, A), target_c, dtype)
    # Independent float64 NumPy oracle: lift the row onto the sheet of curvature c, then scale the whole
    # ambient vector by sqrt(c / c_out), which lands on the sheet of curvature target_c.
    rows_BLS = np.asarray(table_VS, np.float64)[np.asarray(ids_BL)]
    time_BL1 = np.sqrt(np.sum(rows_BLS**2, axis=-1, keepdims=True) + 1.0 / c)
    lifted_BLA = np.concatenate([time_BL1, rows_BLS], axis=-1)
    expected_BLA = np.sqrt(c / target_c) * lifted_BLA
    np.testing.assert_allclose(np.asarray(out_BLA), expected_BLA, rtol=1e-6 if dtype == jnp.float32 else 1e-12)


def test_spatial_init_is_standard_normal():
    """HELM project_emb=1 uses torch.nn.Embedding's N(0, 1) init."""
    emb = LorentzEmbedding(2048, A, rngs=nnx.Rngs(8), parameterization="spatial", param_dtype=jnp.float64)
    table_VS = np.asarray(emb.embedding[...])
    assert abs(table_VS.mean()) < 0.02
    assert abs(table_VS.std() - 1.0) < 0.02


def test_out_of_range_ids_give_nan():
    """Loud over silent: an out-of-range id is not clamped onto a valid row."""
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(0))
    assert bool(jnp.all(jnp.isnan(emb(jnp.array([V])))))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="parameterization"):
        LorentzEmbedding(V, A, rngs=nnx.Rngs(0), parameterization="euclidean")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="features"):
        LorentzEmbedding(V, 1, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="c must be positive"):
        LorentzEmbedding(V, A, rngs=nnx.Rngs(0), c=0.0)
    with pytest.raises(ValueError, match="integers"):
        LorentzEmbedding(V, A, rngs=nnx.Rngs(0))(jnp.array([1.0]))


# --------------------------------------------------------------------------- #
# Gradients and training
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("parameterization", ["manifold", "spatial"])
def test_gradients_reach_only_looked_up_rows(parameterization):
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(0), c=0.5, parameterization=parameterization, param_dtype=jnp.float64)
    ids_BL = jnp.array([[1, 4, 4], [9, 1, 17]])
    looked_up = np.zeros(V, dtype=bool)
    looked_up[np.unique(np.asarray(ids_BL))] = True

    def loss_fn(model):
        return jnp.sum(model(ids_BL, c_out=1.0)[..., 1:] ** 2)

    # (V, A) for the manifold table, (V, S) for the spatial one.
    grad_table = np.asarray(nnx.grad(loss_fn)(emb).embedding[...])

    assert np.all(grad_table[~looked_up] == 0.0)
    assert np.all(np.linalg.norm(grad_table[looked_up], axis=-1) > 0.0)


class _EmbedAndHead(nnx.Module):
    """A ManifoldParam table next to a Euclidean head, as in HELM's mixed optimizer setup."""

    def __init__(self, c, dtype, rngs):
        self.embed = LorentzEmbedding(V, A, rngs=rngs, c=c, param_dtype=dtype)
        self.head = nnx.Linear(A, 3, rngs=rngs, param_dtype=dtype, dtype=dtype)

    def __call__(self, ids):
        return self.head(self.embed(ids))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("c", [0.5, 2.0])
def test_riemannian_adam_step_keeps_table_on_manifold(dtype, c):
    lr, eps = 1e-2, 1e-8
    model = _EmbedAndHead(c, dtype, nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, riemannian_adam(learning_rate=lr, eps=eps), wrt=nnx.Param)
    ids_BL = jnp.array([[0, 3, 5], [3, 7, 11]])
    target_BL3 = jax.random.normal(jax.random.key(9), (2, 3, 3), dtype=dtype)
    looked_up = np.zeros(V, dtype=bool)
    looked_up[np.unique(np.asarray(ids_BL))] = True

    def loss_fn(m):
        return jnp.mean((m(ids_BL) - target_BL3) ** 2)

    table0_VA = model.embed.embedding[...]
    kernel0 = model.head.kernel[...]
    grads = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads)
    table1_VA = model.embed.embedding[...]

    assert _on_manifold(table1_VA, c, dtype)
    assert not np.allclose(np.asarray(model.head.kernel[...]), np.asarray(kernel0))
    np.testing.assert_allclose(np.asarray(table1_VA[~looked_up]), np.asarray(table0_VA[~looked_up]), atol=_atol(dtype))

    # First RAdam step on each looked-up row: expmap(-lr·rgrad/(‖rgrad‖_x + eps)), a Riemannian move.
    manifold = Hyperboloid(dtype=dtype)
    egrad_VA = grads["embed"]["embedding"][...]

    def expected_row(g_A, x_A):
        rgrad_A = manifold.egrad2rgrad(g_A, x_A, c)
        return manifold.expmap(-lr * rgrad_A / (manifold.tangent_norm(rgrad_A, x_A, c) + eps), x_A, c)

    expected_VA = jax.vmap(expected_row)(egrad_VA, table0_VA)
    np.testing.assert_allclose(
        np.asarray(table1_VA[looked_up]), np.asarray(expected_VA[looked_up]), atol=1e-5 if dtype == jnp.float32 else 1e-12
    )
    # Each looked-up row moved one geodesic step of length lr.
    step_N = jax.vmap(manifold.dist, in_axes=(0, 0, None))(table0_VA[looked_up], table1_VA[looked_up], c)
    np.testing.assert_allclose(np.asarray(step_N), lr, rtol=1e-3 if dtype == jnp.float32 else 1e-6)


@pytest.mark.parametrize("parameterization", ["manifold", "spatial"])
def test_jit_parity(parameterization):
    emb = LorentzEmbedding(V, A, rngs=nnx.Rngs(0), c=0.5, parameterization=parameterization)
    ids_BL = jax.random.randint(jax.random.key(10), (3, 4), 0, V)

    @nnx.jit
    def forward(model, ids, c_out):
        return model(ids, c_out=c_out)

    c_out = jnp.asarray(2.0, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(forward(emb, ids_BL, c_out)), np.asarray(emb(ids_BL, c_out=2.0)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(nnx.jit(lambda m, ids: m(ids))(emb, ids_BL)), np.asarray(emb(ids_BL)), rtol=1e-6)
