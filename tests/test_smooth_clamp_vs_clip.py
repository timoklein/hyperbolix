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

Section 8 then pins ``smooth_clamp`` itself. Every clamp above uses a very wide window (±87.8 for
the sinh/cosh guard, ±v_max for PLFC), which is exactly the regime where the old implementation
happened to be correct; the properties that hold for *narrow* windows are asserted there.

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

# The hard-clip side of every comparison below is the SHIPPED implementation, imported rather than
# re-implemented. The file previously defined its own `jnp.sinh(jnp.clip(x, -c, c))` copies, which
# made it a 33-item gate over code that is not in the library: a `math_utils.sinh -> 0.7*jnp.sinh`
# mutation left all 33 items passing while `test_math_utils.py::test_sinh` failed (audit A3-02).
from hyperbolix.utils.math_utils import cosh as hard_cosh
from hyperbolix.utils.math_utils import sinh as hard_sinh
from hyperbolix.utils.math_utils import smooth_clamp, smooth_clamp_max, smooth_clamp_min

DTYPES = [jnp.float32, jnp.float64]


def _clamp(dtype) -> float:
    """The hyperbolix sinh/cosh overflow bound: 0.99 * log(finfo.max). ~87.83 (f32), ~709.0 (f64)."""
    return math.log(float(jnp.finfo(dtype).max)) * 0.99


# The smooth-clamp alternative the shipped hard clip is being compared against.
def smooth_sinh(x):
    c = _clamp(x.dtype)
    return jnp.sinh(smooth_clamp(x, -c, c))


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
    test_saturated_tail_gradient_is_degenerate_for_both_clamps, not asserted finite here.
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
def test_saturated_tail_gradient_is_degenerate_for_both_clamps(dtype):
    """Characterizes the saturated tail (|x| >= clamp) for BOTH dtypes — the only regime where hard
    clip and smooth clamp differ, and a regime unreachable by in-scope layers (sinh inputs are
    pre-bounded to +-v_max << clamp; clamp is ~87.8 for f32, ~702.7 for f64).

    NOTE FOR WHOEVER SEES THIS FAIL: the last two assertions pin a NEGATIVE property — that
    ``grad((sinh o clamp)**2)`` is non-finite for both clamps. A failure here most likely means the
    behaviour IMPROVED (someone made the saturated-tail gradient finite), not that something
    regressed. Re-baseline the assertion rather than reverting the improvement.

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


# ==================================================================================================
# 8. smooth_clamp itself: bounded for EVERY window width (narrow-window overshoot regression)
# ==================================================================================================
# Every clamp in sections 1-7 is two-sided with a very wide window, which is the regime where the
# old implementation was correct. The old ``smooth_clamp`` composed two *gated* one-sided clamps,
# ``jnp.where(x < min + eps, min + eps + sp(x - min - eps), x)``; since ``sp(0) = log(2)/beta`` and
# not 0, the softplus branch did not glue onto the identity branch at the switch — it jumped by
# ``log(2)/beta`` (0.0139 at the default beta=50). Composed, that displacement escaped the window
# whenever ``max_value - min_value < log(2)/beta``.
#
# ``smooth_clamp`` is now the *difference* ``min + sp(x - min) - sp(x - max)`` (evaluated in a
# cancellation-free hinge+remainder form), which is inside the window for every width and every
# beta: ``sp(u) - sp(u - beta*w) < w`` reduces to ``1 < exp(beta*w)``.

CLAMP_BETAS = [1.0, 10.0, 50.0, 500.0]
# (0.0, 0.001) and (0.0, 0.01) are narrower than log(2)/beta at beta=50 (0.01386) — the regime that
# used to overshoot. (-1.5, 1.5) and (-35.0, 35.0) are the wide windows the library actually uses.
CLAMP_WINDOWS = [(0.0, 0.001), (0.0, 0.01), (-1.5, 1.5), (-35.0, 35.0)]


def _softplus_beta_f64(u, beta: float):
    """``log(1 + exp(beta*u))/beta`` via logaddexp — an oracle spelled differently from the source.

    ``math_utils`` evaluates the clamp as ``clip(x) + tail(x - min) - tail(x - max)``; this is the
    literal scaled softplus, so agreement between the two is a real check of the algebra rather than
    a restatement of it. It is float64-only: the literal form is exactly what loses precision for
    large ``|x|`` (see the docstring of ``_softplus_tail``), so it is only an oracle near the window.
    """
    return jnp.logaddexp(beta * jnp.asarray(u, dtype=jnp.float64), 0.0) / beta


def _mono_atol(dtype, lo: float, hi: float, beta: float) -> float:
    """A few ulp of the largest quantity the output is built from.

    The clamp is strictly increasing in real arithmetic (its derivative is a difference of sigmoids,
    positive everywhere), but consecutive float outputs can still wobble down by an ulp or two on a
    fine grid. This allowance is ~7 orders of magnitude below the ``log(2)/beta`` backward jump the
    old gated implementation had, so it does not blunt the property being tested.
    """
    scale = max(abs(lo), abs(hi), math.log(2.0) / beta)
    return 8.0 * float(jnp.spacing(jnp.asarray(scale, dtype=dtype)))


def test_smooth_clamp_narrow_window_stays_in_range():
    """The reported repro: window [0, 0.01] at beta=50 (width < log(2)/beta = 0.013863).

    Historical values from the composed implementation, float32: x=0.0 -> 0.013863 and x=-0.001 ->
    0.013369, i.e. 39% and 34% above ``max_value`` — for x=0.0, an input that needs no clamping at
    all. Both inputs now land inside the window.
    """
    lo, hi, beta = 0.0, 0.01, 50.0
    assert hi - lo < math.log(2.0) / beta, "repro requires a window narrower than log(2)/beta"

    for x0 in [0.0, -0.001, 0.001, 0.005]:
        y = float(smooth_clamp(jnp.array(x0, dtype=jnp.float32), lo, hi, smoothing_factor=beta))
        assert lo <= y <= hi, f"x={x0} left the window: {y}"

    # x = min_value exactly is the worst case (the old gate fired there and displaced by log(2)/beta).
    y0 = float(smooth_clamp(jnp.array(0.0, dtype=jnp.float32), lo, hi, smoothing_factor=beta))
    assert y0 == pytest.approx(0.0043814, abs=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("beta", CLAMP_BETAS)
@pytest.mark.parametrize("lo,hi", CLAMP_WINDOWS)
def test_smooth_clamp_stays_in_window_for_every_width_and_beta(dtype, beta, lo, hi):
    """Bounded + monotone + finite over the whole real line, for narrow and wide windows alike."""
    # Compare against the bounds *as represented in ``dtype``*: float32's nearest value to 0.001 is
    # 0.0010000000475, so upcasting the output to a Python float and comparing to the literal would
    # report a spurious 4.8e-11 overshoot. jnp comparisons against weak-typed Python floats stay in
    # the array's own dtype, which is the meaningful statement.
    span = max(hi - lo, 1.0)
    x = jnp.linspace(lo - 3 * span, hi + 3 * span, 2001, dtype=dtype)
    y = smooth_clamp(x, lo, hi, smoothing_factor=beta)

    assert jnp.isfinite(y).all()
    assert jnp.all(y >= lo), f"below min_value by {float(jnp.min(y) - jnp.asarray(lo, dtype)):.3e}"
    assert jnp.all(y <= hi), f"above max_value by {float(jnp.max(y) - jnp.asarray(hi, dtype)):.3e}"
    assert jnp.all(jnp.diff(y) >= -_mono_atol(dtype, lo, hi, beta)), "not monotone increasing"

    # Extremes, including the ones a hyperbolic layer can actually produce after an overflow.
    extreme = jnp.array([-1e30, -1e6, lo, (lo + hi) / 2, hi, 1e6, 1e30, -jnp.inf, jnp.inf], dtype=dtype)
    y_ext = smooth_clamp(extreme, lo, hi, smoothing_factor=beta)
    assert jnp.isfinite(y_ext).all(), "non-finite output for extreme (incl. infinite) input"
    assert jnp.all((y_ext >= lo) & (y_ext <= hi))

    # Strictly inside within 4/beta of the window, where the softplus remainder is >= exp(-4)/beta
    # and so is still resolvable next to the bound. Further out it underflows and the output
    # saturates to the bound exactly — inclusion holds, strictness is a float32/64 resolution limit.
    near = jnp.linspace(lo - 4.0 / beta, hi + 4.0 / beta, 1001, dtype=dtype)
    y_near = smooth_clamp(near, lo, hi, smoothing_factor=beta)
    assert jnp.all((y_near > lo) & (y_near < hi)), "not strictly inside the open window near the bounds"


@pytest.mark.parametrize("beta", CLAMP_BETAS)
@pytest.mark.parametrize("lo,hi", CLAMP_WINDOWS)
def test_smooth_clamp_is_not_a_composition_of_the_one_sided_clamps(beta, lo, hi):
    """``smooth_clamp != smooth_clamp_min o smooth_clamp_max`` — and must not be turned back into one.

    Making the one-sided clamps gate-free fixes *them* but does NOT fix the composition: chaining
    them overshoots ``max_value`` by ``log(1 + exp(-beta*w))/beta`` in the limit of large x, for
    EVERY window — it is merely invisible when ``beta*w`` is large. At window [0, 0.001] and
    beta=50 the composition reaches 0.00828, 8x ``max_value``, where the shipped difference returns
    0.00049. This test exists so that "simplify smooth_clamp to reuse the one-sided helpers" fails
    loudly.
    """
    x = jnp.linspace(lo - 2.0, hi + 2.0, 1001, dtype=jnp.float32)
    y = smooth_clamp(x, lo, hi, beta)
    assert jnp.all((y >= lo) & (y <= hi))

    # sup_x composition(x) = lo + sp(hi - lo) = hi + log(1 + exp(-beta*w))/beta, approached as
    # x -> +inf (both stages are increasing), so evaluate far enough out to sit on it.
    overshoot = math.log1p(math.exp(-beta * (hi - lo))) / beta
    x_far = jnp.array(hi + 20.0 / beta + 10.0, dtype=jnp.float32)
    composed_far = float(smooth_clamp_min(smooth_clamp_max(x_far, hi, beta), lo, beta))
    assert composed_far == pytest.approx(hi + overshoot, rel=1e-5, abs=4e-7)

    # Whenever that displacement is resolvable in float32 it is an actual out-of-window value.
    if overshoot > 8.0 * float(jnp.spacing(jnp.asarray(max(abs(hi), overshoot), dtype=jnp.float32))):
        assert composed_far > float(jnp.asarray(hi, dtype=jnp.float32)), "composition should overshoot here"


@pytest.mark.parametrize("beta", CLAMP_BETAS)
@pytest.mark.parametrize("lo,hi", CLAMP_WINDOWS)
def test_smooth_clamp_matches_the_softplus_difference_oracle(beta, lo, hi):
    """Forward value == ``min + sp(x - min) - sp(x - max)`` computed independently in float64."""
    span = max(hi - lo, 1.0)
    x = jnp.linspace(lo - 2 * span, hi + 2 * span, 501, dtype=jnp.float64)
    expected = lo + _softplus_beta_f64(x - lo, beta) - _softplus_beta_f64(x - hi, beta)
    got = smooth_clamp(x, lo, hi, smoothing_factor=beta)
    assert jnp.allclose(got, expected, rtol=1e-12, atol=1e-12 * max(1.0, abs(lo), abs(hi)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("beta", CLAMP_BETAS)
def test_smooth_clamp_derivative_is_the_sigmoid_difference(dtype, beta):
    """d/dx = sigmoid(beta(x - min)) - sigmoid(beta(x - max)) in (0, 1), exact AT the bounds too.

    The bounds are the interesting points: the implementation splits the clamp into a hard hinge
    plus a smooth remainder, and the two kinks have to cancel there. At ``x == min_value`` the hinge
    contributes 1/2 and the remainder 0, which is ``sigmoid(0) = 1/2``.
    """
    lo, hi = -1.5, 1.5
    x = jnp.array([lo - 1.0, lo, lo + 0.05, 0.0, hi - 0.05, hi, hi + 1.0, -1e6, 1e6], dtype=dtype)
    grads = jax.vmap(jax.grad(lambda v: smooth_clamp(v, lo, hi, smoothing_factor=beta)))(x)
    expected = jax.nn.sigmoid(beta * (x - lo)) - jax.nn.sigmoid(beta * (x - hi))

    atol = 4e-3 if dtype == jnp.float32 else 1e-9
    assert jnp.allclose(grads, expected, atol=atol, rtol=1e-5)
    assert jnp.isfinite(grads).all()
    # A slope of exactly 1 would mean no clamping at all; a negative slope would break monotonicity.
    # (Far outside the window the sigmoid difference underflows to exactly 0, as for a hard clip.)
    assert jnp.all((grads >= 0.0) & (grads <= 1.0))
    # Spelled out at the lower bound, the point where the hinge and the remainder each contribute a
    # one-sided kink that has to cancel: sigmoid(0) - sigmoid(-beta*w) = 1/2 - sigmoid(-beta*w).
    at_lo = float(grads[jnp.argmin(jnp.abs(x - lo))])
    assert at_lo == pytest.approx(0.5 - float(jax.nn.sigmoid(-beta * (hi - lo))), abs=1e-6)

    # One-sided clamps: same statement with a single sigmoid.
    g_min = jax.vmap(jax.grad(lambda v: smooth_clamp_min(v, lo, smoothing_factor=beta)))(x)
    g_max = jax.vmap(jax.grad(lambda v: smooth_clamp_max(v, hi, smoothing_factor=beta)))(x)
    assert jnp.allclose(g_min, jax.nn.sigmoid(beta * (x - lo)), atol=atol, rtol=1e-5)
    assert jnp.allclose(g_max, jax.nn.sigmoid(beta * (hi - x)), atol=atol, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_smooth_clamp_gradients_are_finite_everywhere(dtype):
    """No NaN from any of the three clamps at extreme inputs, on the bounds, or through a downstream."""
    lo, hi, beta = -1.5, 1.5, 50.0
    x = jnp.array([-1e30, -1e6, -1e3, lo, 0.0, hi, 1e3, 1e6, 1e30], dtype=dtype)

    for fn in (
        lambda v: smooth_clamp(v, lo, hi, smoothing_factor=beta),
        lambda v: smooth_clamp_min(v, lo, smoothing_factor=beta),
        lambda v: smooth_clamp_max(v, hi, smoothing_factor=beta),
    ):
        assert jnp.isfinite(jax.vmap(jax.grad(fn))(x)).all()

    # Through the guarded sinh the layers actually use (v_max-style window, squared loss).
    def loss(v):
        return jnp.sum(hard_sinh(smooth_clamp(v, -10.0, 10.0)) ** 2)

    assert jnp.isfinite(jax.grad(loss)(x)).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_smooth_clamp_is_the_identity_in_the_interior(dtype):
    """Deviation from x decays like exp(-beta*d)/beta with d = distance to the nearer bound.

    This is the one behavioral give-back of dropping the gate: the old implementation returned x
    *exactly* in the interior. At beta=50 the deviation is under the float32 noise floor from ~0.3
    away from both bounds, which every in-library caller satisfies by a wide margin (the MLR clamps
    sit at ±16.6 (f32) / ±35.4 (f64) around arguments that are O(1)).
    """
    lo, hi, beta = -1.5, 1.5, 50.0
    for d in [0.05, 0.1, 0.2, 0.3, 0.5]:
        x = jnp.array(lo + d, dtype=jnp.float64)
        deviation = float(smooth_clamp(x, lo, hi, smoothing_factor=beta) - x)
        # The +ulp slack is the rounding of ``x + deviation`` back onto the float64 grid: at d=0.5
        # the deviation is 2.8e-13 on an O(1) value whose ulp is 2.2e-16.
        assert 0.0 < deviation <= math.exp(-beta * d) / beta + 2.0 * float(jnp.spacing(jnp.abs(x)))

    # Bit-for-bit identity once the deviation drops below one ulp of the values involved. That is
    # ~0.3 from each bound in float32 (deviation 6.1e-9 vs ulp ~1.2e-7) and ~0.8 in float64, which
    # resolves far more of the tail (deviation 8.5e-20 vs ulp ~1.1e-16).
    d_identity = 0.3 if dtype == jnp.float32 else 0.8
    x = jnp.linspace(lo + d_identity, hi - d_identity, 4001, dtype=dtype)
    assert jnp.array_equal(smooth_clamp(x, lo, hi, smoothing_factor=beta), x)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("beta", CLAMP_BETAS)
def test_one_sided_clamps_are_the_two_sided_tails(dtype, beta):
    """``smooth_clamp_min/max`` == ``smooth_clamp`` with the opposite bound pushed away, and bounded."""
    lo, hi = -1.5, 1.5
    x = jnp.linspace(-6.0, 6.0, 2001, dtype=dtype)

    y_min = smooth_clamp_min(x, lo, smoothing_factor=beta)
    y_max = smooth_clamp_max(x, hi, smoothing_factor=beta)
    assert jnp.all(y_min >= lo) and jnp.all(y_max <= hi)
    assert jnp.isfinite(y_min).all() and jnp.isfinite(y_max).all()

    # 1e4 is far enough that the far-bound softplus remainder underflows for every beta tested.
    assert jnp.allclose(y_min, smooth_clamp(x, lo, 1e4, smoothing_factor=beta), atol=0.0, rtol=0.0)
    assert jnp.allclose(y_max, smooth_clamp(x, -1e4, hi, smoothing_factor=beta), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("beta", CLAMP_BETAS)
def test_clamps_are_1_lipschitz_across_the_bound(dtype, beta):
    """No jump at the bound: |Δy| <= |Δx| on a grid straddling it.

    This is the property the old gate broke and the direct cause of the overshoot. On this grid the
    old ``smooth_clamp_min`` produced |Δy|/|Δx| ≈ 1.4e4 at the switch point (a ``log(2)/beta``
    discontinuity over one grid step); the derivative of the current form never exceeds 1.
    """
    lo, hi = -1.5, 1.5
    x = jnp.linspace(lo - 1e-3, lo + 1e-3, 2001, dtype=dtype)
    dx = float(x[1] - x[0])
    for y in (
        smooth_clamp(x, lo, hi, smoothing_factor=beta),
        smooth_clamp_min(x, lo, smoothing_factor=beta),
        smooth_clamp_max(x, lo, smoothing_factor=beta),
    ):
        assert float(jnp.max(jnp.abs(jnp.diff(y)))) / dx <= 1.01
