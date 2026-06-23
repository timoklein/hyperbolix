"""Numerical-stability gate for replacing the ``smooth_clamp`` overflow guard in ``sinh``/``cosh``
with a hard ``jnp.clip`` (Option C), and for dropping PLFC's redundant outer guard (Option A).

These tests establish the *property* that justifies the change, independent of which clamp the
installed library currently uses, so they remain valid before and after the source edit:

  * In the **valid regime** (|x| < overflow clamp), ``sinh(clip(x))`` and ``sinh(smooth_clamp(x))``
    are bit-/ulp-identical in forward AND gradient (both pass x through untouched).
  * ``grad(sinh∘clip)`` is finite for ALL x, including the saturated tail (clip's VJP is 0 there,
    times the finite ``cosh(clamp)`` → 0, no NaN).
  * The only behavioral difference is in the saturated tail under a *squaring* downstream op, where
    ``sinh²`` overflows to ``inf`` and hard clip yields a ``0·inf = nan`` cotangent while smooth
    yields ``inf`` — BOTH degenerate. This regime is unreachable by the in-scope layers: PLFC clamps
    its sinh input to ±v_max=±10 ≪ 87.83 (the guard we keep), so manifold validity and gradient
    finiteness hold under extreme inputs (tested below and in test_hyperboloid_linear_plfc.py).

Run: ``uv run pytest tests/test_smooth_clamp_vs_clip.py -v`` (conftest enables float64).
"""

import contextlib
import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import hyperbolix.manifolds.hyperboloid as hyp_mod
from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import HypConv2DHyperboloidILNN, HypLinearHyperboloidPLFC
from hyperbolix.utils.math_utils import smooth_clamp

DTYPES = [jnp.float32, jnp.float64]


def _clamp(dtype) -> float:
    """The hyperbolix sinh/cosh overflow bound: 0.99 * log(finfo.max). ~87.83 (f32), ~709.0 (f64)."""
    return math.log(float(jnp.finfo(dtype).max)) * 0.99


# Explicit hard-clip and smooth-clamp wrapped variants (the two implementations under comparison).
def hard_sinh(x):
    c = _clamp(x.dtype)
    return jnp.sinh(jnp.clip(x, -c, c))


def smooth_sinh(x):
    c = _clamp(x.dtype)
    return jnp.sinh(smooth_clamp(x, -c, c))


def hard_cosh(x):
    c = _clamp(x.dtype)
    return jnp.cosh(jnp.clip(x, -c, c))


def smooth_cosh(x):
    c = _clamp(x.dtype)
    return jnp.cosh(smooth_clamp(x, -c, c))


def _grid(dtype, lo, hi, n=4001):
    return jnp.linspace(lo, hi, n, dtype=dtype)


# ==================================================================================================
# 1. Forward equivalence
# ==================================================================================================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn_hard,fn_smooth", [(hard_sinh, smooth_sinh), (hard_cosh, smooth_cosh)])
def test_forward_equivalence_valid_regime(dtype, fn_hard, fn_smooth):
    """Hard clip == smooth clamp (exactly) for |x| < clamp; both finite & saturating beyond."""
    clamp = _clamp(dtype)
    # Valid regime: safely inside the band. Both leave x untouched, so results must be identical.
    x_valid = _grid(dtype, -(clamp - 1.0), clamp - 1.0)
    h, s = fn_hard(x_valid), fn_smooth(x_valid)
    assert jnp.array_equal(h, s)  # bit-identical (both are the no-op branch -> same jnp.sinh/cosh)
    assert jnp.isfinite(h).all()

    # Full range including the saturated tail: both stay finite (no inf/nan), bounded by the dtype.
    x_full = _grid(dtype, -1.5 * clamp, 1.5 * clamp)
    h_full, s_full = fn_hard(x_full), fn_smooth(x_full)
    assert jnp.isfinite(h_full).all(), "hard-clip forward overflowed"
    assert jnp.isfinite(s_full).all(), "smooth-clamp forward overflowed"


# ==================================================================================================
# 2. Gradient equivalence
# ==================================================================================================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn_hard,fn_smooth", [(hard_sinh, smooth_sinh), (hard_cosh, smooth_cosh)])
def test_gradient_equivalence_valid_regime(dtype, fn_hard, fn_smooth):
    """In the interior the gradient is identical (cosh/sinh); everywhere the hard-clip grad is finite.

    The crux test: hard clip's zero gradient in the tail times the finite cosh(clamp) is 0 — NOT a
    NaN (the 0*inf concern materializes only if a downstream op squares the saturated value first;
    grad of sinh∘clip alone is well-defined).
    """
    clamp = _clamp(dtype)
    g_hard = jax.vmap(jax.grad(fn_hard))
    g_smooth = jax.vmap(jax.grad(fn_smooth))

    x_valid = _grid(dtype, -(clamp - 1.0), clamp - 1.0)
    gh, gs = g_hard(x_valid), g_smooth(x_valid)
    atol = 4e-3 if dtype == jnp.float32 else 1e-7
    # Relative comparison: cosh grows huge near the band edge, so compare with generous rtol.
    assert jnp.allclose(gh, gs, atol=atol, rtol=1e-3)
    assert jnp.isfinite(gh).all()

    # Full range incl. tail: hard-clip gradient must be finite EVERYWHERE (0 in the saturated tail).
    x_full = _grid(dtype, -1.5 * clamp, 1.5 * clamp)
    gh_full = g_hard(x_full)
    assert jnp.isfinite(gh_full).all(), "hard-clip sinh/cosh produced a non-finite gradient"
    # And the tail gradient is exactly 0 (clip saturates).
    tail = jnp.abs(x_full) > clamp + 1.0
    assert jnp.all(gh_full[tail] == 0.0)


# ==================================================================================================
# 3. Full VJP through a square+sum (mimics PLFC time reconstruction sum(w**2))
# ==================================================================================================
@pytest.mark.parametrize("dtype", DTYPES)
def test_sinh_clip_vjp_through_square_valid_regime(dtype):
    """VJP of sum(sinh(clip(x))**2) is finite and matches the smooth path in the VALID regime.

    This is the operating regime of every in-scope layer (PLFC bounds the sinh input to ±v_max≪clamp).
    In the saturated tail sinh**2 overflows to inf for BOTH variants — characterized in
    test_saturated_tail_is_degenerate_for_both, not asserted finite here.
    """
    clamp = _clamp(dtype)
    # Bound inputs well inside the band; sinh(clamp-1) is still enormous, so use a modest range that
    # nonetheless exercises large values (sinh(20)~2.4e8) — representative of real saturated MLR scores.
    x = _grid(dtype, -20.0, 20.0)

    def f_hard(a):
        return jnp.sum(jnp.sinh(jnp.clip(a, -clamp, clamp)) ** 2)

    def f_smooth(a):
        return jnp.sum(jnp.sinh(smooth_clamp(a, -clamp, clamp)) ** 2)

    gh = jax.grad(f_hard)(x)
    gs = jax.grad(f_smooth)(x)
    assert jnp.isfinite(gh).all(), "hard-clip sinh∘square VJP not finite in valid regime"
    assert jnp.isfinite(gs).all()
    rtol = 1e-4 if dtype == jnp.float64 else 1e-2
    assert jnp.allclose(gh, gs, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_saturated_tail_is_degenerate_for_both(dtype):
    """Characterizes the saturated tail (|x| >= clamp) for BOTH dtypes — the only regime where hard
    clip and smooth clamp differ, and a regime unreachable by in-scope layers (sinh inputs are
    pre-bounded to +-v_max << clamp; clamp is ~87.8 for f32, ~702.7 for f64).

    Established facts (identical structure for f32 and f64):
      * forward sinh(clamp) is finite and hard ~= smooth (the only forward gap is the tiny smoothing
        width at the knee);
      * sinh(clamp)**2 overflows to inf (f32: 7e37**2; f64: 7e304**2) — the manifold time
        reconstruction would too, for EITHER clamp;
      * grad(sinh o clip) ALONE is finite (0) in the tail — the actual gradient the layers propagate
        is always safe, dtype-independently;
      * grad((sinh o clip)**2) in the tail is non-finite for BOTH clamps, so hard clip is no worse
        than smooth: f32 -> both NaN (smooth's tiny tail gradient underflows to 0*inf=nan); f64 ->
        hard NaN, smooth inf. Either way unusable, and only reachable if sinh**2 already overflowed.
    """
    clamp = _clamp(dtype)
    x_tail = jnp.array([clamp + 5.0, clamp + 50.0], dtype=dtype)

    # Forward: finite, and hard ~= smooth (both saturate to ~sinh(clamp)).
    assert jnp.isfinite(hard_sinh(x_tail)).all()
    assert jnp.allclose(hard_sinh(x_tail), smooth_sinh(x_tail), rtol=1e-3)
    # The square overflows to inf for both dtypes (degenerate regardless of clamp type).
    assert jnp.all(jnp.isinf(hard_sinh(x_tail) ** 2))

    # The gradient that layers actually propagate (sinh alone) is finite (0) in the tail.
    g_alone = jax.vmap(jax.grad(hard_sinh))(x_tail)
    assert jnp.isfinite(g_alone).all()
    assert jnp.all(g_alone == 0.0)

    # Under a squaring downstream (overflow), hard clip is no worse than smooth: both non-finite.
    g_hard_sq = jax.vmap(jax.grad(lambda a: hard_sinh(a) ** 2))(x_tail)
    g_smooth_sq = jax.vmap(jax.grad(lambda a: smooth_sinh(a) ** 2))(x_tail)
    assert not jnp.isfinite(g_hard_sq).any()
    assert not jnp.isfinite(g_smooth_sq).any()


# ==================================================================================================
# 4. Layer equivalence + manifold validity (smooth vs hard clip, library-level)
# ==================================================================================================
@contextlib.contextmanager
def _patched(sinh_fn, cosh_fn):
    """Swap the hyperboloid module-global sinh/cosh that expmap/compute_mlr resolve at call time.

    Lets us A/B smooth vs hard clip without depending on the installed implementation. (The PLFC
    output-path sinh is bare ``jnp.sinh`` after Option A — not a module global — but its input is
    pre-bounded to ±v_max, so smooth/hard/bare coincide there regardless.)
    """
    saved = (hyp_mod.sinh, hyp_mod.cosh)
    hyp_mod.sinh, hyp_mod.cosh = sinh_fn, cosh_fn
    try:
        yield
    finally:
        hyp_mod.sinh, hyp_mod.cosh = saved


def _on_hyperboloid(x, c, atol):
    mink = -(x[..., 0:1] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1, keepdims=True)
    return jnp.allclose(mink, -1.0 / c, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("c", [0.1, 1.0, 10.0])
def test_plfc_smooth_vs_hard_equivalence_and_manifold(dtype, c):
    """PLFC forward under hard-clip sinh/cosh == under smooth clamp, and stays on the hyperboloid."""
    manifold = Hyperboloid(dtype=dtype)
    in_dim, out_dim, batch = 17, 21, 32
    v = jax.random.normal(jax.random.PRNGKey(0), (batch, in_dim), dtype=dtype) * 0.3
    x = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, c)
    layer = HypLinearHyperboloidPLFC(manifold, in_dim, out_dim, rngs=nnx.Rngs(0))

    with _patched(smooth_sinh, smooth_cosh):
        y_smooth = layer(x, c=c)
    with _patched(hard_sinh, hard_cosh):
        y_hard = layer(x, c=c)

    atol = 4e-3 if dtype == jnp.float32 else 1e-9
    assert jnp.isfinite(y_hard).all()
    assert jnp.allclose(y_hard, y_smooth, atol=atol, rtol=1e-4)
    assert _on_hyperboloid(y_hard, c, atol=4e-3 if dtype == jnp.float32 else 1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ilnn_conv_smooth_vs_hard_equivalence_and_manifold(dtype):
    """ILNN conv forward under hard-clip == under smooth clamp, and stays on the hyperboloid."""
    c = 1.0
    manifold = Hyperboloid(dtype=dtype)
    in_c, out_c, batch, hw = 9, 13, 4, 8
    v = jax.random.normal(jax.random.PRNGKey(1), (batch, hw, hw, in_c), dtype=dtype) * 0.3
    x = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v.reshape(-1, in_c), c).reshape(batch, hw, hw, in_c)
    layer = HypConv2DHyperboloidILNN(manifold, in_c, out_c, kernel_size=3, rngs=nnx.Rngs(0))

    with _patched(smooth_sinh, smooth_cosh):
        y_smooth = layer(x, c=c)
    with _patched(hard_sinh, hard_cosh):
        y_hard = layer(x, c=c)

    atol = 4e-3 if dtype == jnp.float32 else 1e-9
    assert jnp.isfinite(y_hard).all()
    assert jnp.allclose(y_hard, y_smooth, atol=atol, rtol=1e-4)
    assert _on_hyperboloid(y_hard, c, atol=4e-3 if dtype == jnp.float32 else 1e-6)


# ==================================================================================================
# 5. Option A: PLFC outer guard makes the wrapped inner sinh redundant -> bare jnp.sinh bit-identical
# ==================================================================================================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("v_max", [10.0, 20.0])
def test_plfc_outer_guard_makes_inner_sinh_redundant(dtype, v_max):
    """smooth_clamp(., +-v_max) bounds the input to (-v_max, v_max) << overflow clamp, so the wrapped
    sinh's internal guard is a no-op: bare jnp.sinh is bit-identical (Option A is provably exact)."""
    sf = 50.0
    x = jnp.linspace(-3 * v_max, 3 * v_max, 5000, dtype=dtype)
    arg = smooth_clamp(x, -v_max, v_max, sf)
    # The guard output never leaves the band, so it can never reach the overflow clamp.
    assert jnp.max(jnp.abs(arg)) <= v_max
    bare = jnp.sinh(arg)
    wrapped_hard = hard_sinh(arg)
    wrapped_smooth = smooth_sinh(arg)
    assert jnp.array_equal(bare, wrapped_hard)  # hard clip is a no-op on |arg| < clamp
    assert jnp.array_equal(bare, wrapped_smooth)  # smooth clamp is a no-op too


# ==================================================================================================
# 6. Extreme inputs: hard clip keeps expmap / layers finite and on-manifold
# ==================================================================================================
def _rel_on_hyperboloid(x, c, rtol):
    """Scale-relative Minkowski check: |(-x0^2 + ||xs||^2) + 1/c| / max(x0^2, 1) <= rtol.

    The absolute constraint residual grows with the coordinate magnitude (cancellation of two large
    squares), so at large geodesic distances only a relative check is meaningful.
    """
    mink = -(x[..., 0] ** 2) + jnp.sum(x[..., 1:] ** 2, axis=-1)
    scale = jnp.maximum(x[..., 0] ** 2, 1.0)
    return jnp.all(jnp.abs(mink + 1.0 / c) / scale <= rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("c", [0.1, 1.0, 100.0])
def test_expmap_extreme_tangent_hard_equals_smooth(dtype, c):
    """expmap_0 of large-norm tangents stays finite under hard clip AND matches smooth clamp.

    Norms are scaled so sqrt(c)*||v|| sweeps from small up to ~1.2x the overflow clamp, engaging the
    guard. The gate question is "is hard clip no worse than smooth?" — answered by finiteness +
    elementwise equivalence (both are identical below the clamp and both saturate to it above).
    On-manifold validity is checked relatively (absolute residual is precision-bound at these scales).
    """
    manifold = Hyperboloid(dtype=dtype)
    dim = 8
    clamp = _clamp(dtype)
    key = jax.random.PRNGKey(7)
    dirs = jax.random.normal(key, (64, dim + 1), dtype=dtype)
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    # sqrt(c) * ||v|| ranges over [0.5, 1.2*clamp]; ||v|| = arg / sqrt(c).
    args = jnp.linspace(0.5, 1.2 * clamp, 64, dtype=dtype)[:, None]
    v = dirs * (args / math.sqrt(c))

    with _patched(hard_sinh, hard_cosh):
        y_hard = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, c)
    with _patched(smooth_sinh, smooth_cosh):
        y_smooth = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v, c)

    # Gate question: "is hard clip no worse than smooth?" -> hard is finite wherever smooth is, and
    # agrees with it there. (At sqrt(c)*||v|| near the clamp the expmap RECONSTRUCTION can overflow
    # via the 1/sqrt(c) amplification for small c — identically for both clamps; not an Option C
    # regression. sinh/cosh themselves stay finite, which tests 1-2 verify directly.)
    finite = jnp.isfinite(y_smooth)
    assert jnp.isfinite(y_hard[finite]).all(), "hard clip non-finite where smooth clamp is finite"
    assert jnp.allclose(y_hard[finite], y_smooth[finite], rtol=1e-2, atol=0.0)
    # Moderate sub-band (arg < 18): coordinates small enough for a meaningful manifold check.
    moderate = args[:, 0] < 18.0
    rtol = 1e-3 if dtype == jnp.float32 else 1e-9
    assert _rel_on_hyperboloid(y_hard[moderate], c, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_plfc_extreme_input_finite_gradient_hard_clip(dtype):
    """PLFC with large inputs + large kernel stays finite (fwd+grad) under hard clip — the ±v_max
    guard (kept) bounds the sinh input so the saturated tail is never reached."""
    c = 1.0
    manifold = Hyperboloid(dtype=dtype)
    in_dim, out_dim, batch = 17, 21, 16
    # Input far from origin (spatial norm 20) + large kernel (pre-PLFC init std=1.0): the worst case.
    xs = jax.random.normal(jax.random.PRNGKey(3), (batch, in_dim - 1), dtype=dtype)
    xs = 20.0 * xs / jnp.linalg.norm(xs, axis=-1, keepdims=True)
    x0 = jnp.sqrt(jnp.sum(xs**2, axis=-1, keepdims=True) + 1.0 / c)
    x = jnp.concatenate([x0, xs], axis=-1)
    layer = HypLinearHyperboloidPLFC(manifold, in_dim, out_dim, rngs=nnx.Rngs(0), kernel_init_std=1.0)

    def loss_fn(m):
        return jnp.sum(m(x, c=c) ** 2)

    with _patched(hard_sinh, hard_cosh):
        loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


# ==================================================================================================
# 7. v_max overflow assertion (introduced with Option A's bare jnp.sinh)
# ==================================================================================================
def test_v_max_overflow_assertion():
    """Constructing PLFC/ILNN with a v_max that would overflow the float32 squared norm raises.

    sinh(v_max) must stay below sqrt(finfo(float32).max) (~1.84e19, v_max ≲ 45) so the bare-jnp.sinh
    output path cannot overflow the time reconstruction.
    """
    manifold = Hyperboloid(dtype=jnp.float32)
    # Safe defaults construct fine.
    HypLinearHyperboloidPLFC(manifold, 8, 8, rngs=nnx.Rngs(0))  # v_max=10 default
    HypLinearHyperboloidPLFC(manifold, 8, 8, rngs=nnx.Rngs(0), v_max=44.0)
    HypConv2DHyperboloidILNN(manifold, 5, 7, kernel_size=3, rngs=nnx.Rngs(0))

    # Too-large v_max is rejected loudly (sinh(50) ≈ 2.6e21 > 1.84e19).
    with pytest.raises(ValueError, match="v_max"):
        HypLinearHyperboloidPLFC(manifold, 8, 8, rngs=nnx.Rngs(0), v_max=50.0)
    with pytest.raises(ValueError, match="v_max"):
        HypConv2DHyperboloidILNN(manifold, 5, 7, kernel_size=3, rngs=nnx.Rngs(0), v_max=88.0)
