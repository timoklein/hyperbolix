"""Tail-saturation properties of the shipped ``sinh``/``cosh`` overflow guard, plus the
``smooth_clamp`` family's own property pins.

``hyperbolix.utils.math_utils.sinh``/``cosh`` guard against overflow with a hard clip
(``clamp_to``) at ``0.99 * log(finfo.max)`` — ~87.83 (f32), ~702.7 (f64). The first section pins
what that guard does on its own, with no comparator:

  * forward stays finite over the whole real line, saturating past the clamp;
  * ``grad(sinh o clip)`` is finite EVERYWHERE and exactly 0 in the saturated tail — the gradient
    the layers actually propagate is always safe;
  * even under a squaring downstream (where the forward value itself overflows to ``inf``), the
    gradient is exactly 0, because the ``where``-based clamp's VJP is a ``select`` and never
    multiplies the overflowed cotangent by a zero factor;
  * the layers that feed the guard never reach that tail: PLFC/ILNN bound their ``sinh`` input to
    +-``v_max`` (default 10, hard-capped at construction) << the clamp, so their forward and
    gradient stay finite under extreme inputs.

The second section pins ``smooth_clamp`` itself, which is still used by the MLR ``asinh`` clamp in
``poincare.py`` / ``hyperboloid.py`` / ``proper_velocity.py`` / ``poincare_regression.py``. Those
call sites use wide windows; the properties asserted there are the ones that hold for *narrow*
windows too, which is where the old gated implementation was wrong.

Run: ``uv run pytest tests/test_smooth_clamp.py -v`` (conftest enables float64).
"""

import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.nn_layers import HypConv2DHyperboloidILNN, HypLinearHyperboloidPLFC

# The hard-clip guard under test is the SHIPPED implementation, imported rather than re-implemented.
# This file previously defined its own `jnp.sinh(jnp.clip(x, -c, c))` copies, which made it a gate
# over code that is not in the library: a `math_utils.sinh -> 0.7*jnp.sinh` mutation left every item
# passing while `test_math_utils.py::test_sinh` failed (audit A3-02).
from hyperbolix.utils.math_utils import cosh as hard_cosh
from hyperbolix.utils.math_utils import sinh as hard_sinh
from hyperbolix.utils.math_utils import smooth_clamp, smooth_clamp_max, smooth_clamp_min

DTYPES = [jnp.float32, jnp.float64]


def _clamp(dtype) -> float:
    """The hyperbolix sinh/cosh overflow bound: 0.99 * log(finfo.max). ~87.83 (f32), ~702.7 (f64)."""
    return math.log(float(jnp.finfo(dtype).max)) * 0.99


def _grid(dtype, lo, hi, n=4001):
    return jnp.linspace(lo, hi, n, dtype=dtype)


# ==================================================================================================
# 1. The shipped hard-clip guard: finite forward and gradient over the whole real line
# ==================================================================================================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", [hard_sinh, hard_cosh], ids=["sinh", "cosh"])
def test_hard_clip_forward_and_gradient_are_finite_everywhere(dtype, fn):
    """Forward and gradient are finite across the clamp, and the tail gradient is exactly 0."""
    clamp = _clamp(dtype)
    x_full = _grid(dtype, -1.5 * clamp, 1.5 * clamp)

    assert jnp.isfinite(fn(x_full)).all(), "hard-clip forward overflowed"

    g_full = jax.vmap(jax.grad(fn))(x_full)
    assert jnp.isfinite(g_full).all(), "hard-clip sinh/cosh produced a non-finite gradient"
    tail = jnp.abs(x_full) > clamp + 1.0
    assert jnp.all(g_full[tail] == 0.0), "clip must kill the gradient in the saturated tail"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", [hard_sinh, hard_cosh], ids=["sinh", "cosh"])
def test_saturated_tail_gradient_is_zero_for_hard_clip(dtype, fn):
    """The saturated tail (|x| >= clamp), including under a squaring downstream op.

    Squaring the saturated value overflows to ``inf`` (f32: 7e37**2; f64: 7e304**2), which is what
    the hyperboloid time reconstruction ``sum(w**2)`` would do. The gradient survives anyway: the
    ``where``-based clamp's VJP is a ``select``, so the overflowed ``2*sinh`` cotangent is discarded
    rather than multiplied by a ``0`` factor (which would give ``0 * inf = NaN``).

    This regime is unreachable by the in-scope layers — PLFC/ILNN bound their sinh input to
    +-v_max << clamp — so this pins the guard's worst case, not an operating point.
    """
    clamp = _clamp(dtype)
    x_tail = jnp.array([clamp + 5.0, clamp + 50.0], dtype=dtype)

    assert jnp.isfinite(fn(x_tail)).all()
    assert jnp.all(jnp.isinf(fn(x_tail) ** 2)), "the square is expected to overflow at the clamp"

    g_alone = jax.vmap(jax.grad(fn))(x_tail)
    assert jnp.isfinite(g_alone).all()
    assert jnp.all(g_alone == 0.0)

    g_sq = jax.vmap(jax.grad(lambda a: fn(a) ** 2))(x_tail)
    assert jnp.isfinite(g_sq).all()
    assert jnp.all(g_sq == 0.0)


# ==================================================================================================
# 2. Extreme inputs: the layers that feed the guard stay finite
# ==================================================================================================
@pytest.mark.parametrize("dtype", DTYPES)
def test_plfc_extreme_input_finite_gradient_hard_clip(dtype):
    """PLFC with large inputs + large kernel stays finite (fwd+grad) — the +-v_max guard bounds the
    sinh input so the saturated tail is never reached."""
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

    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(grads.kernel[...]).all()
    assert jnp.isfinite(grads.bias[...]).all()


# ==================================================================================================
# 3. v_max overflow assertion (PLFC/ILNN construction guard)
# ==================================================================================================
def test_v_max_overflow_assertion():
    """Constructing PLFC/ILNN with a v_max that would overflow the float32 squared norm raises.

    sinh(v_max) must stay below sqrt(finfo(float32).max) (~1.84e19, v_max <~ 45) so the sinh
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
# 4. smooth_clamp itself: bounded for EVERY window width (narrow-window overshoot regression)
# ==================================================================================================
# Every clamp above is two-sided with a very wide window, which is the regime where the
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
