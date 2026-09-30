"""Tests for curvature helpers and manifold-as-plain-class behavior."""

import math

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import Euclidean, Hyperboloid, Poincare, ProperVelocity

jax.config.update("jax_enable_x64", True)


HYPERBOLIC_MANIFOLDS = [
    lambda **kw: Poincare(**kw),
    lambda **kw: Hyperboloid(**kw),
    lambda **kw: ProperVelocity(**kw),
]


# ===========================================================================
# 1. Manifolds are plain classes, not nnx.Module
# ===========================================================================


class TestManifoldIsPlainClass:
    @pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
    def test_not_nnx_module(self, make_manifold):
        m = make_manifold()
        assert not isinstance(m, nnx.Module)

    def test_euclidean_not_nnx_module(self):
        m = Euclidean()
        assert not isinstance(m, nnx.Module)

    @pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
    def test_no_learnable_attribute(self, make_manifold):
        m = make_manifold()
        assert not hasattr(m, "_learnable")
        assert not hasattr(m, "_c_raw")


# ===========================================================================
# 2. Static curvature on manifolds
# ===========================================================================


@pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
class TestStaticCurvature:
    def test_default_c_value(self, make_manifold):
        m = make_manifold()
        assert m.c == 1.0

    def test_custom_c_value(self, make_manifold):
        m = make_manifold(c=0.5)
        assert m.c == 0.5

    def test_c_is_float(self, make_manifold):
        m = make_manifold(c=2.0)
        assert isinstance(m.c, float)


class TestEuclidean:
    def test_fixed_c_zero(self):
        m = Euclidean()
        assert m.c == 0.0

    # ``test_not_nnx_module`` used to live here too, byte-identical to
    # ``TestManifoldIsPlainClass::test_euclidean_not_nnx_module`` above; kept only once.


# ===========================================================================
# 3. LearnableCurvature module
# ===========================================================================

# init_c on a clamp bound: (parameterization, c_min, c_max, init_c), with explicit bounds so the cases do not depend
# on the default bounds. Rounding the inverse of init_c to the storage dtype puts the recovered c one float outside
# the bound for five of these 22 dtype cases on XLA:CPU (float32 log at 0.1 and 0.2, float32 softplus at 0.2 and
# float64 softplus at 0.3 below c_min; float64 log at 10 above c_max), one float inside it for two (float64 log and
# softplus at 0.1, above c_min), and exactly on it for the other 15, every identity case among them.
INIT_ON_BOUND_CASES = [
    ("log", 0.1, 10.0, 0.1),
    ("log", 0.1, 10.0, 10.0),
    ("log", 0.2, 5.0, 0.2),
    ("log", 0.2, 5.0, 5.0),
    ("softplus", 0.1, 10.0, 0.1),
    ("softplus", 0.1, 10.0, 10.0),
    ("softplus", 0.2, 5.0, 0.2),
    ("softplus", 0.3, 3.0, 0.3),
    ("identity", -10.0, 10.0, -10.0),
    ("identity", -10.0, 10.0, 10.0),
    ("identity", -3.0, 0.3, 0.3),
]
UNCLAMPED = {"log": jnp.exp, "softplus": jax.nn.softplus, "identity": lambda raw: raw}
INVERSE = {"log": math.log, "softplus": lambda c: math.log(math.expm1(c)), "identity": lambda c: c}  # for c <= 20


def _dc_draw(parameterization: str, raw: float) -> float:
    """The unclamped map's derivative at ``raw``, in Python float64: the oracle for every gradient below."""
    return {"log": math.exp(raw), "softplus": 1.0 / (1.0 + math.exp(-raw)), "identity": 1.0}[parameterization]


class TestLearnableCurvatureInit:
    @pytest.mark.parametrize("init_c", [0.1, 0.5, 1.0, 5.0, 10.0])
    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_init_recovery_within_default_bounds(self, init_c, parameterization):
        c = LearnableCurvature(init_c, parameterization=parameterization)
        assert jnp.allclose(c(), init_c, atol=1e-5)

    @pytest.mark.parametrize("init_c", [0.01, 50.0, 100.0])
    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_init_recovery_with_disabled_clamp(self, init_c, parameterization):
        c = LearnableCurvature(init_c, parameterization=parameterization, c_min=None, c_max=None)
        assert jnp.allclose(c(), init_c, atol=1e-4)

    @pytest.mark.parametrize("init_c", [25.0, 50.0])
    def test_softplus_init_recovery_above_20_is_exact_in_float64(self, init_c):
        """Above 20 the inverse is ``x + log1p(-exp(-x))``, not ``x``: float64 softplus recovers ``init_c`` to a few ulps.

        With ``return x`` the forward gave ``25 + e^-25 = 25.000000000013888`` for ``init_c = 25`` (3909 ulps off). At
        ``init_c = 50``, ``e^-50`` is below half an ulp, so both inverses pass there.
        """
        curvature = LearnableCurvature(init_c, parameterization="softplus", c_max=1000.0, param_dtype=jnp.float64)
        c = curvature()
        assert c.dtype == jnp.float64
        assert abs(float(c) - init_c) <= 4 * math.ulp(init_c)

    def test_raw_is_nnx_param(self):
        c = LearnableCurvature(0.1)
        assert isinstance(c.raw, nnx.Param)

    def test_is_nnx_module(self):
        c = LearnableCurvature(1.0)
        assert isinstance(c, nnx.Module)

    @pytest.mark.parametrize("bad_c", [-1.0, 0.0])
    def test_nonpositive_init_raises(self, bad_c):
        with pytest.raises(ValueError, match="init_c > 0"):
            LearnableCurvature(bad_c)

    def test_init_below_c_min_raises(self):
        with pytest.raises(ValueError, match=r"init_c.*c_min"):
            LearnableCurvature(0.05, c_min=0.1, c_max=10.0)

    def test_init_above_c_max_raises(self):
        with pytest.raises(ValueError, match=r"init_c.*c_max"):
            LearnableCurvature(20.0, c_min=0.1, c_max=10.0)

    def test_c_min_greater_than_c_max_raises(self):
        with pytest.raises(ValueError, match=r"c_min.*c_max"):
            LearnableCurvature(1.0, c_min=10.0, c_max=1.0)

    def test_unknown_parameterization_raises(self):
        with pytest.raises(ValueError, match="parameterization"):
            LearnableCurvature(1.0, parameterization="quadratic")  # type: ignore[arg-type]

    @pytest.mark.parametrize("parameterization", ["softplus", "log", "identity"])
    def test_nan_init_c_raises(self, parameterization):
        # NaN slips through every range check (all NaN comparisons are False); reject it up front rather
        # than silently store a NaN raw param. `identity` accepts negatives/zero, so it needs this too.
        with pytest.raises(ValueError, match="finite"):
            LearnableCurvature(float("nan"), parameterization=parameterization)

    @pytest.mark.parametrize("parameterization", ["softplus", "log", "identity"])
    def test_raw_param_dtype_pinned_to_float32_under_x64(self, parameterization):
        # This module enables jax_enable_x64 globally. The raw param must still default to float32 (its
        # param_dtype default) so a learnable curvature cannot silently promote the whole manifold graph
        # to float64 — the dtype-leak class the hyperbolic-dl guidance warns about. Opt-in float64 storage
        # via param_dtype is still honored.
        assert LearnableCurvature(1.0, parameterization=parameterization).raw[...].dtype == jnp.float32
        assert (
            LearnableCurvature(1.0, parameterization=parameterization, param_dtype=jnp.float64).raw[...].dtype == jnp.float64
        )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])
    @pytest.mark.parametrize(("parameterization", "c_min", "c_max", "init_c"), INIT_ON_BOUND_CASES)
    def test_init_on_a_bound_passes_the_inward_gradient_and_blocks_the_outward_one(
        self, parameterization, c_min, c_max, init_c, dtype
    ):
        """At step 0 a loss that pulls ``c`` inside gets the analytic ``dL/draw``; one that pushes it out gets exactly 0.

        ``raw`` is the plain inverse of ``init_c`` rounded to the storage dtype, so the recovered ``c`` can land one
        float outside the bound: on XLA:CPU float32 ``exp(float32(log 0.1)) = 0.099999994 < 0.1``, which froze
        ``LearnableCurvature(init_c=0.1)`` under the old hard clamp. The clamp maps it onto the bound, passes the
        inward gradient and blocks the outward one. A ``c`` that rounds exactly onto the bound gets the same
        projected-gradient treatment, so the outward gradient passes only where ``c`` rounds inside the interval.
        The oracle is ``±dc/draw`` in Python float64 at the stored ``raw``.
        """
        curvature = LearnableCurvature(init_c, parameterization=parameterization, c_min=c_min, c_max=c_max, param_dtype=dtype)
        plain_inverse = jnp.array(INVERSE[parameterization](init_c), dtype=dtype)
        assert curvature.raw[...] == plain_inverse  # no construction-time nudge
        on_floor = init_c == c_min
        bound = jnp.asarray(c_min if on_floor else c_max, dtype=dtype)
        unclamped = UNCLAMPED[parameterization](curvature.raw[...])
        outside = bool(unclamped < bound) if on_floor else bool(unclamped > bound)
        on_or_outside = bool(unclamped <= bound) if on_floor else bool(unclamped >= bound)
        assert curvature() == (bound if outside else unclamped)

        inward = -1.0 if on_floor else 1.0  # dL/dc under which a descent step moves c into the interval
        grad_in = nnx.grad(lambda m: inward * m())(curvature).raw[...]
        grad_out = nnx.grad(lambda m: -inward * m())(curvature).raw[...]
        dc_draw = _dc_draw(parameterization, float(curvature.raw[...]))
        rel = 1e-6 if dtype == jnp.float32 else 1e-12
        assert grad_in.dtype == dtype
        assert float(grad_in) == pytest.approx(inward * dc_draw, rel=rel)
        if on_or_outside:
            assert float(grad_out) == 0.0
        else:
            assert float(grad_out) == pytest.approx(-inward * dc_draw, rel=rel)

    def test_builds_under_jit_vmap_and_eval_shape_and_its_clamp_gradient_batches(self):
        """Construction under ``nnx.jit``, ``nnx.vmap`` and ``nnx.eval_shape``; the clamp's backward under ``jit``/``vmap``.

        The batched gradient covers ``c`` below ``c_min``, inside twice and above ``c_max``, for both loss directions,
        and must match the unbatched gradient element by element.
        """
        bounds = {"c_min": 0.1, "c_max": 10.0}
        eager = LearnableCurvature(0.1, **bounds)
        assert nnx.jit(lambda: LearnableCurvature(0.1, **bounds))().raw[...] == eager.raw[...]
        abstract = nnx.eval_shape(lambda: LearnableCurvature(0.1, **bounds))
        assert jax.tree.leaves(nnx.state(abstract))[0].dtype == jnp.float32

        raws = jnp.array([-3.5, 0.0, 1.0, 3.5], dtype=jnp.float32)  # c = 0.03, 1, 2.7, 33

        @nnx.vmap(in_axes=0, out_axes=0)
        def build(raw):
            curvature = LearnableCurvature(0.1, **bounds)
            curvature.raw[...] = raw
            return curvature

        @nnx.jit
        @nnx.vmap(in_axes=(0, None), out_axes=0)
        def batched_grad(curvature, sign):
            return nnx.grad(lambda m: sign * m())(curvature).raw[...]

        ensemble = build(raws)
        assert ensemble.raw[...].shape == (4,)
        for sign, blocked in ((1.0, [True, False, False, False]), (-1.0, [False, False, False, True])):
            batched = batched_grad(ensemble, sign)
            assert [bool(g == 0.0) for g in batched] == blocked
            for raw, grad in zip(raws, batched, strict=True):
                single = LearnableCurvature(0.1, **bounds)
                single.raw[...] = raw
                expected = nnx.grad(lambda m, s=sign: s * m())(single).raw[...]
                assert float(grad) == pytest.approx(float(expected), rel=1e-6)


class TestLearnableCurvatureClamping:
    """Clamping applies to the recovered c, not the raw param."""

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_default_clamp_upper_bound(self, parameterization):
        c = LearnableCurvature(1.0, parameterization=parameterization)
        # Push raw to a huge value; default c_max=10.0 must hold.
        c.raw[...] = jnp.array(100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(10.0)

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_default_clamp_lower_bound(self, parameterization):
        c = LearnableCurvature(1.0, parameterization=parameterization)
        # Push raw to a very negative value; default c_min=0.1 must hold.
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(0.1)

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_disabled_clamp_allows_extremes(self, parameterization):
        c = LearnableCurvature(1.0, parameterization=parameterization, c_min=None, c_max=None)
        c.raw[...] = jnp.array(20.0, dtype=jnp.float32)
        # No clamp: softplus(20) ≈ 20, exp(20) ≈ 4.85e8 — both well above 10.
        assert float(c()) > 10.0

    def test_custom_clamp_bounds(self):
        c = LearnableCurvature(0.5, parameterization="log", c_min=0.2, c_max=2.0)
        c.raw[...] = jnp.array(10.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(2.0)
        c.raw[...] = jnp.array(-10.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(0.2)

    @pytest.mark.parametrize("init_c", [0.1, 1.0, 2.5, 50.0])
    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_default_bounds_sit_a_decade_either_side_of_init_c(self, parameterization, init_c):
        """Without explicit bounds, softplus/log clamp to ``[init_c / 10, init_c * 10]``: ``[0.1, 10]`` at ``init_c = 1``."""
        c = LearnableCurvature(init_c, parameterization=parameterization)
        c.raw[...] = jnp.array(-1e4, dtype=jnp.float32)  # c = 0 for both parameterizations
        assert float(c()) == float(jnp.float32(init_c / 10))
        c.raw[...] = jnp.array(1e4, dtype=jnp.float32)  # c = 1e4 (softplus), ~1.4e38 (log, exponent cap)
        assert float(c()) == float(jnp.float32(init_c * 10))

    def test_explicit_bounds_win_over_the_init_relative_default(self):
        """A given bound is used verbatim and ``None`` disables it; the other bound keeps its default. ``identity``
        keeps its symmetric ``[-10, 10]`` whatever the init."""
        cases = [
            (LearnableCurvature(0.1, c_min=0.05, c_max=2.0), 0.05, 2.0),
            (LearnableCurvature(0.1, parameterization="softplus", c_min=0.02), 0.02, 1.0),
            (LearnableCurvature(0.1, c_min=None), 0.0, 1.0),  # exp(-1e4) = 0, no floor
            (LearnableCurvature(2.5, parameterization="identity"), -10.0, 10.0),
        ]
        for c, lo, hi in cases:
            c.raw[...] = jnp.array(-1e4, dtype=jnp.float32)
            assert float(c()) == float(jnp.float32(lo))
            c.raw[...] = jnp.array(1e4, dtype=jnp.float32)
            assert float(c()) == float(jnp.float32(hi))


class TestLearnableCurvatureGradients:
    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_gradient_flow(self, parameterization):
        class Holder(nnx.Module):
            def __init__(self):
                self.curvature = LearnableCurvature(0.5, parameterization=parameterization)

        holder = Holder()

        def loss_fn(h):
            return h.curvature() ** 2

        _loss, grads = nnx.value_and_grad(loss_fn)(holder)
        grad_val = grads.curvature.raw[...]
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0

    def test_log_parameterization_scale_invariance(self):
        """For c = exp(raw), dc/draw = c — scale-invariant gradient."""
        c = LearnableCurvature(2.0, parameterization="log", c_min=None, c_max=None)

        def fn(m):
            return m()

        _val, grads = nnx.value_and_grad(fn)(c)
        # d(exp(raw))/draw = exp(raw) = c
        assert jnp.allclose(grads.raw[...], 2.0, atol=1e-5)

    def test_softplus_parameterization_sigmoid_gradient(self):
        """For c = softplus(raw), dc/draw = sigmoid(raw) ∈ (0, 1)."""
        c = LearnableCurvature(1.0, parameterization="softplus", c_min=None, c_max=None)
        raw_val = float(c.raw[...])

        def fn(m):
            return m()

        _val, grads = nnx.value_and_grad(fn)(c)
        expected = float(jax.nn.sigmoid(jnp.array(raw_val)))
        assert jnp.allclose(grads.raw[...], expected, atol=1e-5)

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_clamp_blocks_an_outward_gradient_past_c_max(self, parameterization):
        """Past ``c_max`` a loss that would raise ``c`` further gets exactly 0, so ``raw`` does not drift outward.

        The old hard clamp zeroed the gradient in both directions here, which pinned ``c`` for good; the new clamp
        keeps only this outward half (the inward half is the next test).
        """
        c = LearnableCurvature(1.0, parameterization=parameterization)
        # raw=15.0 pushes c well past c_max=10.0 for both parameterizations (softplus(15)~=15,
        # exp(15)~=3.3e6). (The former exp() overflow at large raw is now guarded — see
        # test_log_parameterization_no_nan_gradient_on_exp_overflow.)
        c.raw[...] = jnp.array(15.0, dtype=jnp.float32)

        val, grads = nnx.value_and_grad(lambda m: -m())(c)
        assert float(val) == pytest.approx(-10.0)  # forward value clamped
        assert float(grads.raw[...]) == 0.0

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_clamp_passes_an_inward_gradient_past_c_max_undamped(self, parameterization):
        """Past ``c_max`` a loss that lowers ``c`` gets the unclamped map's own ``dc/draw``, written out as a value.

        A ``!= 0`` check would pass any rescaled gradient. The removed ``straight_through_clamp`` divided this one by
        ``max(dc/draw, 1)``, to 1 for ``log`` at ``raw = 15``, because its ``raw`` kept drifting outward while the
        loss pushed ``c`` out. The new clamp blocks that push, so in training ``raw`` stays within the crossing step
        (and the optimizer's momentum) of the bound and needs no damping; ``raw = 15`` is set by hand here.
        """
        c = LearnableCurvature(1.0, parameterization=parameterization)
        c.raw[...] = jnp.array(15.0, dtype=jnp.float32)  # softplus(15)~=15, exp(15)~=3.3e6

        val, grads = nnx.value_and_grad(lambda m: m())(c)
        assert float(val) == pytest.approx(10.0)  # forward value clamped
        assert float(grads.raw[...]) == pytest.approx(_dc_draw(parameterization, 15.0), rel=1e-6)

    @pytest.mark.parametrize("side", ["below_c_min", "above_c_max"])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])
    @pytest.mark.parametrize("parameterization", ["softplus", "log", "identity"])
    def test_clamp_outside_passes_the_inward_gradient_and_blocks_the_outward_one(self, parameterization, dtype, side):
        """``c`` pushed outside by a large step (``raw`` one unit past the bound's inverse), in both dtypes.

        A loss that pulls ``c`` back inside gets the analytic ``dL/draw`` of the unclamped map at the stored ``raw``,
        one that pushes it further out gets exactly 0. The old hard clamp gave 0 to both.
        """
        c_min, c_max = (-10.0, 10.0) if parameterization == "identity" else (0.1, 10.0)
        curvature = LearnableCurvature(1.0, parameterization=parameterization, c_min=c_min, c_max=c_max, param_dtype=dtype)
        inverse = INVERSE[parameterization]
        below = side == "below_c_min"
        curvature.raw[...] = jnp.array(inverse(c_min) - 1.0 if below else inverse(c_max) + 1.0, dtype=dtype)
        assert curvature() == jnp.asarray(c_min if below else c_max, dtype=dtype)

        inward = -1.0 if below else 1.0  # dL/dc under which a descent step moves c back into the interval
        grad_in = nnx.grad(lambda m: inward * m())(curvature).raw[...]
        grad_out = nnx.grad(lambda m: -inward * m())(curvature).raw[...]
        dc_draw = _dc_draw(parameterization, float(curvature.raw[...]))
        assert grad_in.dtype == dtype
        assert float(grad_in) == pytest.approx(inward * dc_draw, rel=1e-6 if dtype == jnp.float32 else 1e-12)
        assert float(grad_out) == 0.0

    def test_log_parameterization_no_nan_gradient_on_exp_overflow(self):
        # Regression: exp(raw) overflowed float32 to +inf for large raw, making the clamp's cotangent times dc/draw
        # 0*inf = NaN. The exponent cap must keep the forward pinned at c_max and the gradient finite even where
        # exp() would have overflowed, for a loss that lowers c and one that raises it. Both gradients are 0 here,
        # from the cap's own zero derivative above it; raw only gets this far when set by hand.
        c = LearnableCurvature(1.0, parameterization="log")
        c.raw[...] = jnp.array(100.0, dtype=jnp.float32)  # exp(100) overflows float32

        for sign in (1.0, -1.0):
            val, grads = nnx.value_and_grad(lambda m, s=sign: s * m())(c)
            assert jnp.isfinite(val) and float(val) == pytest.approx(10.0 * sign)  # forward pinned at c_max, not NaN/0
            assert jnp.isfinite(grads.raw[...])

    @pytest.mark.parametrize(("parameterization", "lr"), [("softplus", 5e-2), ("log", 2e-3)])
    def test_clamp_lets_curvature_re_enter_the_interval_over_many_steps(self, parameterization, lr):
        """The ratchet was a *multi-step* failure: once ``c`` left the interval under the old hard clamp it never came back.

        Plain SGD on ``(c - target)**2`` with bounds ``[0.1, 10]``: 60 steps with the target at 12, past ``c_max``,
        then 200 steps with it at 5. ``c`` reaches the ceiling and rests there, and while the loss pushes outward the
        gradient is exactly 0, so ``raw`` stays where the crossing step left it instead of drifting on (the removed
        ``straight_through_clamp`` let it drift, and needed a re-entry damping for that). Once the target moves
        inside, the first step brings ``c`` off the ceiling, undamped and without overshooting the interval, and
        ``c`` converges to 5. The learning rates differ because ``log``'s ``dc/draw = c`` is ~10 near the ceiling.
        """
        curvature = LearnableCurvature(5.0, parameterization=parameterization, c_min=0.1, c_max=10.0)
        optimizer = nnx.Optimizer(curvature, optax.sgd(lr), wrt=nnx.Param)

        def run(target: float, n_steps: int) -> tuple[list[float], list[float]]:
            cs, raws = [], []
            for _ in range(n_steps):
                grads = nnx.grad(lambda m: (m() - target) ** 2)(curvature)
                optimizer.update(curvature, grads)
                cs.append(float(curvature()))
                raws.append(float(curvature.raw[...]))
            return cs, raws

        cs, raws = run(12.0, 60)
        first_on_ceiling = cs.index(10.0)  # raises if c never reached c_max
        assert first_on_ceiling < 20
        assert set(cs[first_on_ceiling:]) == {10.0}
        assert set(raws[first_on_ceiling:]) == {raws[first_on_ceiling]}  # no drift past the bound
        cs, _ = run(5.0, 200)
        assert 0.1 < cs[0] < 10.0  # one step brings c back inside
        assert cs[-1] == pytest.approx(5.0, abs=0.05)

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_curvature_started_on_the_floor_leaves_it_as_soon_as_the_target_moves_inside(self, parameterization):
        """Toy fit: Adam(1e-2) on ``(c - target)**2``, float32, ``c`` started on ``c_min = 0.1``.

        20 steps with the target at 0.05, below the floor, then 30 with it at 0.3. ``c`` rests on the floor while
        the loss pushes it out and leaves it on the first step after the target moves inside, for both: on XLA:CPU
        ``log``'s init lands one float below the floor and ``softplus``'s exactly on it, the clamp blocks every
        outward step in both cases, and ``raw`` never moves. While a tie still passed the outward gradient,
        ``softplus`` took 7 steps: the first outward step passed and Adam's momentum carried ``raw`` 0.05 on. Under
        the old hard clamp ``c`` stayed at exactly 0.1 for good; under the removed ``straight_through_clamp`` it left
        only after ``raw`` had walked back from 0.85 nats below (55 steps, after 100 steps below the floor).
        """
        curvature = LearnableCurvature(0.1, parameterization=parameterization, c_min=0.1, c_max=10.0)
        optimizer = nnx.Optimizer(curvature, optax.adam(1e-2), wrt=nnx.Param)
        floor = float(jnp.float32(0.1))

        def step(target: float) -> float:
            grads = nnx.grad(lambda m: (m() - target) ** 2)(curvature)
            optimizer.update(curvature, grads)
            return float(curvature())

        assert [step(0.05) for _ in range(20)] == [floor] * 20
        cs = [step(0.3) for _ in range(30)]
        assert cs[9] > floor  # left the floor within 10 steps
        assert all(cs[i + 1] > cs[i] for i in range(9, len(cs) - 1))
        assert cs[-1] > 0.12

    @pytest.mark.parametrize("raw_value", [-1.5, -0.5, 0.0, 0.7, 1.5])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])
    @pytest.mark.parametrize("parameterization", ["softplus", "log", "identity"])
    def test_clamp_leaves_interior_gradients_untouched(self, parameterization, dtype, raw_value):
        """Strictly inside ``[c_min, c_max]`` the gradient is the full chain rule, whichever way the loss pulls ``c``.

        ``log``'s scale-invariant ``dc/draw = c`` is the reason to pick that parameterization (the MERU convention),
        so the custom backward must hand it through untouched in both directions and both dtypes.
        """
        c = LearnableCurvature(1.0, parameterization=parameterization, param_dtype=dtype)
        c.raw[...] = jnp.array(raw_value, dtype=dtype)
        lower_bound = -10.0 if parameterization == "identity" else 0.1  # identity's clamp is symmetric
        dc_draw = _dc_draw(parameterization, float(c.raw[...]))
        rel = 1e-6 if dtype == jnp.float32 else 1e-12
        for sign in (1.0, -1.0):
            val, g = nnx.value_and_grad(lambda m, s=sign: s * m())(c)
            assert lower_bound < sign * float(val) < 10.0, "this raw_value must leave c strictly inside the clamp interval"
            assert float(g.raw[...]) == pytest.approx(sign * dc_draw, rel=rel)

    @pytest.mark.parametrize("raw_value", [-100.0, -12.0, -3.0, 0.0, 0.5, 2.0, 15.0, 100.0])
    @pytest.mark.parametrize("parameterization", ["softplus", "log", "identity"])
    def test_clamp_forward_value_is_the_plain_clip(self, parameterization, raw_value):
        """The custom backward is a gradient-only contract: ``c()`` equals ``jnp.clip`` of the unclamped map bit for bit.

        At every ``raw``: inside the interval, past either bound, and past the ``log`` exponent cap, where the
        reference's ``exp(100)`` overflows to ``inf`` and clips to the same ``c_max``.
        """
        c = LearnableCurvature(1.0, parameterization=parameterization)
        c.raw[...] = jnp.array(raw_value, dtype=jnp.float32)
        lower_bound = -10.0 if parameterization == "identity" else 0.1
        assert float(c()) == float(jnp.clip(UNCLAMPED[parameterization](c.raw[...]), lower_bound, 10.0))

    @pytest.mark.parametrize("parameterization", ["softplus", "log"])
    def test_clamp_passes_the_inward_gradient_below_the_lower_bound_unscaled(self, parameterization):
        """Far below ``c_min`` the inward gradient is the unclamped ``dc/draw`` itself, neither rescaled nor amplified.

        Every parameterization's gain collapses toward 0 there (``exp(-20) = 2.06e-9``, ``sigmoid(-20) =
        2.06e-9``); the gradient is that gain, and the outward one is 0.
        """
        c = LearnableCurvature(1.0, parameterization=parameterization)
        c.raw[...] = jnp.array(-20.0, dtype=jnp.float32)  # c ≈ 2.06e-9, far under c_min = 0.1

        val, grads_in = nnx.value_and_grad(lambda m: -m())(c)
        grads_out = nnx.grad(lambda m: m())(c)
        assert float(val) == pytest.approx(-0.1)  # forward pinned at c_min
        assert float(grads_in.raw[...]) == pytest.approx(-_dc_draw(parameterization, -20.0), rel=1e-4)
        assert float(grads_out.raw[...]) == 0.0


# ===========================================================================
# 4. Vmap compatibility (manifolds as plain classes)
# ===========================================================================


@pytest.mark.parametrize("make_manifold", HYPERBOLIC_MANIFOLDS, ids=["Poincare", "Hyperboloid", "PV"])
class TestVmapCompatibility:
    def test_vmap_with_fixed_c(self, make_manifold):
        m = make_manifold(c=0.5)
        x = jnp.array([[0.1, 0.2], [0.05, 0.15]], dtype=jnp.float32)
        y = jnp.array([[0.3, 0.1], [0.2, 0.05]], dtype=jnp.float32)

        if isinstance(m, Hyperboloid):
            x = jax.vmap(m.proj, in_axes=(0, None))(jnp.concatenate([jnp.ones((2, 1)), x], axis=-1), 0.5)
            y = jax.vmap(m.proj, in_axes=(0, None))(jnp.concatenate([jnp.ones((2, 1)), y], axis=-1), 0.5)

        dists = jax.vmap(m.dist, in_axes=(0, 0, None))(x, y, m.c)
        assert dists.shape == (2,)
        assert jnp.all(jnp.isfinite(dists))

    def test_vmap_expmap_0(self, make_manifold):
        m = make_manifold(c=0.5)
        if isinstance(m, Hyperboloid):
            v = jnp.array([[0.0, 0.1, 0.2], [0.0, 0.05, 0.15]], dtype=jnp.float32)
        else:
            v = jnp.array([[0.1, 0.2], [0.05, 0.15]], dtype=jnp.float32)

        result = jax.vmap(m.expmap_0, in_axes=(0, None))(v, m.c)
        assert result.shape == v.shape
        assert jnp.all(jnp.isfinite(result))


# ===========================================================================
# 5. Training integration with LearnableCurvature
# ===========================================================================


PARAMETERIZATIONS = ["softplus", "log"]


class TestTrainingIntegration:
    @pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
    def test_poincare_curvature_trains(self, parameterization):
        from hyperbolix.nn_layers import HypLinearPoincarePP, HypRegressionPoincarePP

        manifold = Poincare(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.curvature = LearnableCurvature(init_c=1.0, parameterization=parameterization)
                self.fc = HypLinearPoincarePP(m, 4, 3, rngs=rngs)
                self.head = HypRegressionPoincarePP(m, 3, 2, rngs=rngs)

            def __call__(self, x):
                c = self.curvature()
                h = self.fc(x, c)
                return self.head(h, c)

        model = Model(manifold, nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (16, 4), dtype=jnp.float32) * 0.1
        target = jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float32)

        def loss_fn(m):
            logits = m(x)
            return jnp.mean((logits - target) ** 2)

        c_before = float(model.curvature())
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(model.curvature())
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    @pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
    def test_hyperboloid_curvature_trains(self, parameterization):
        from hyperbolix.nn_layers import FGGLinear, FGGLorentzMLR

        manifold = Hyperboloid(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.curvature = LearnableCurvature(init_c=1.0, parameterization=parameterization)
                self.fc = FGGLinear(5, 4, rngs=rngs, activation=jax.nn.relu)
                self.head = FGGLorentzMLR(4, 3, rngs=rngs)

            def __call__(self, x):
                c = self.curvature()
                h = self.fc(x, c)
                return self.head(h, c)

        model = Model(manifold, nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

        key = jax.random.PRNGKey(0)
        x_spatial = jax.random.normal(key, (16, 4), dtype=jnp.float32) * 0.1
        x = manifold.proj_batch(
            jnp.concatenate([jnp.ones((16, 1), dtype=jnp.float32), x_spatial], axis=-1),
            1.0,
        )
        target = jax.random.randint(jax.random.PRNGKey(1), (16,), 0, 3)

        def loss_fn(m):
            logits = m(x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()

        c_before = float(model.curvature())
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(model.curvature())
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0

    @pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
    def test_pv_curvature_trains(self, parameterization):
        from hyperbolix.nn_layers import HypLinearPV, HypRegressionPV

        manifold = ProperVelocity(c=1.0)

        class Model(nnx.Module):
            def __init__(self, m, rngs: nnx.Rngs):
                self.manifold = m
                self.curvature = LearnableCurvature(init_c=1.0, parameterization=parameterization)
                self.fc = HypLinearPV(m, 4, 3, rngs=rngs)
                self.head = HypRegressionPV(m, 3, 2, rngs=rngs)

            def __call__(self, x):
                c = self.curvature()
                h = self.fc(x, c=c)
                return self.head(h, c=c)

        model = Model(manifold, nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (16, 4), dtype=jnp.float32) * 0.3
        target = jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float32)

        def loss_fn(m):
            y = m(x)
            return jnp.mean((y - target) ** 2)

        c_before = float(model.curvature())
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        c_after = float(model.curvature())
        assert c_before != c_after, f"Curvature did not change: {c_before}"
        assert jnp.isfinite(jnp.array(c_after))
        assert c_after > 0


# ===========================================================================
# 6. Scan compatibility (regression test for the original bug)
# ===========================================================================


class TestScanCompatibility:
    @pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
    def test_fori_loop_with_shared_manifold_and_learnable_c(self, parameterization):
        """Regression test: shared manifold + LearnableCurvature in nnx.fori_loop.

        This was the original bug: when ManifoldBase was an nnx.Module with a
        learnable _c_raw param, sharing the manifold across layers caused
        NNX graph deduplication to fail inside fori_loop with
        'ValueError: Dict key mismatch'. Now that manifolds are plain classes,
        they're static graphdef attributes — no deduplication issues. The
        LearnableCurvature instance lives once on the model so there is no
        shared-reference aliasing.
        """
        from hyperbolix.nn_layers import HypLinearPoincarePP

        manifold = Poincare(c=0.1)

        class Model(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.manifold = manifold
                self.curvature = LearnableCurvature(init_c=0.1, parameterization=parameterization)
                self.l1 = HypLinearPoincarePP(manifold, 4, 4, rngs=nnx.Rngs(0))
                self.l2 = HypLinearPoincarePP(manifold, 4, 4, rngs=nnx.Rngs(1))

            def __call__(self, x):
                c = self.curvature()
                h = self.l1(x, c)
                return jnp.sum(self.l2(h, c))

        model = Model(nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

        x = jnp.ones((4, 4), dtype=jnp.float32) * 0.1

        def train_step(i, carry):
            model, optimizer = carry
            _, grads = nnx.value_and_grad(lambda m: m(x))(model)
            optimizer.update(model, grads)
            return model, optimizer

        model, optimizer = nnx.fori_loop(0, 3, train_step, (model, optimizer))
        loss = model(x)
        assert jnp.isfinite(loss)
        assert float(model.curvature()) > 0


# ===========================================================================
# 7. Signed `identity` parameterization (Stereographic manifold)
# ===========================================================================


class TestLearnableCurvatureIdentity:
    """The signed ``identity`` parameterization (``c = raw``) for the Stereographic manifold.

    Unlike softplus/log (strictly positive), identity spans hyperbolic (``c>0``), Euclidean
    (``c=0``), and spherical (``c<0``); its default clamp is the symmetric magnitude cap
    ``[-10, 10]`` (which *includes* 0), not the positive ``[0.1, 10]`` window.
    """

    @pytest.mark.parametrize("init_c", [-2.0, -0.5, 0.0, 1.5, 9.0])
    def test_identity_recovers_signed_init(self, init_c):
        c = LearnableCurvature(init_c, parameterization="identity")
        assert jnp.allclose(c(), init_c, atol=1e-6)

    @pytest.mark.parametrize("init_c", [-1.0, 0.0, -0.05])
    def test_identity_accepts_nonpositive_init(self, init_c):
        # softplus/log would raise here (they cannot represent c <= 0); identity must not.
        c = LearnableCurvature(init_c, parameterization="identity")
        assert jnp.isfinite(c())

    def test_identity_signed_default_clamp(self):
        c = LearnableCurvature(0.0, parameterization="identity")
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(-10.0)  # symmetric lower bound, NOT the softplus +0.1
        c.raw[...] = jnp.array(100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(10.0)

    def test_identity_default_clamp_includes_zero(self):
        # The Euclidean point c=0 must be reachable; a naive c_min=0.1 carryover would forbid it.
        c = LearnableCurvature(0.0, parameterization="identity")
        c.raw[...] = jnp.array(0.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(0.0)

    def test_identity_disabled_clamp(self):
        c = LearnableCurvature(-2.0, parameterization="identity", c_min=None, c_max=None)
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(-100.0)

    def test_identity_custom_signed_bounds(self):
        c = LearnableCurvature(0.0, parameterization="identity", c_min=-3.0, c_max=3.0)
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(-3.0)
        c.raw[...] = jnp.array(100.0, dtype=jnp.float32)
        assert float(c()) == pytest.approx(3.0)

    def test_identity_below_signed_c_min_raises(self):
        # init below the symmetric default lower bound (-10) is still range-checked.
        with pytest.raises(ValueError, match=r"init_c.*c_min"):
            LearnableCurvature(-20.0, parameterization="identity")

    def test_identity_gradient_is_one(self):
        """For c = raw, dc/draw = 1 everywhere — no zero-crossing obstruction (the whole point)."""
        c = LearnableCurvature(1.5, parameterization="identity", c_min=None, c_max=None)

        def fn(m):
            return m()

        _val, grads = nnx.value_and_grad(fn)(c)
        assert jnp.allclose(grads.raw[...], 1.0, atol=1e-6)

    def test_identity_gradient_finite_at_zero(self):
        """The zero-crossing point c=0 has a finite, well-defined gradient (== 1)."""
        c = LearnableCurvature(0.0, parameterization="identity", c_min=None, c_max=None)

        def fn(m):
            return m()

        val, grads = nnx.value_and_grad(fn)(c)
        assert float(val) == pytest.approx(0.0)
        assert jnp.isfinite(grads.raw[...])
        assert jnp.allclose(grads.raw[...], 1.0, atol=1e-6)

    def test_identity_grad_through_stereographic_at_zero(self):
        """Grad w.r.t. curvature is finite through Stereographic.dist at c=0 — guards the
        κ-trig Taylor seam from the learnable-curvature side."""
        from hyperbolix.manifolds import Stereographic

        manifold = Stereographic(dtype=jnp.float32)
        x = jnp.array([0.1, 0.2, -0.05], dtype=jnp.float32)
        y = jnp.array([-0.2, 0.15, 0.1], dtype=jnp.float32)
        c = LearnableCurvature(0.0, parameterization="identity", c_min=None, c_max=None)

        def fn(m):
            return manifold.dist(x, y, m())

        val, grads = nnx.value_and_grad(fn)(c)
        assert jnp.isfinite(val)
        assert jnp.isfinite(grads.raw[...])

    def test_identity_clamp_keeps_the_inward_gradient(self):
        c = LearnableCurvature(0.0, parameterization="identity")
        c.raw[...] = jnp.array(-100.0, dtype=jnp.float32)  # past the -10 lower bound

        val, grads_in = nnx.value_and_grad(lambda m: -m())(c)
        grads_out = nnx.grad(lambda m: m())(c)
        assert float(val) == pytest.approx(10.0)  # forward value clamped to -10
        # For the identity parameterization the unclamped derivative is exactly 1.0; a rescaled gradient (any
        # nonzero multiple) would pass a bare `!= 0.0` check. The loss that lowers c further gets exactly 0.
        assert float(grads_in.raw[...]) == -1.0
        assert float(grads_out.raw[...]) == 0.0


class TestStereographicCurvatureTraining:
    def test_signed_curvature_trains(self):
        """A learnable signed curvature drives a Stereographic distance objective: the curvature
        updates, stays finite across steps, and is free to be negative (spherical)."""
        from hyperbolix.manifolds import Stereographic

        manifold = Stereographic(dtype=jnp.float32)

        class Model(nnx.Module):
            def __init__(self):
                self.manifold = manifold
                self.curvature = LearnableCurvature(init_c=-1.0, parameterization="identity")
                self.point = nnx.Param(jnp.array([0.1, 0.2, -0.05], dtype=jnp.float32))

            def __call__(self, target):
                c = self.curvature()
                return self.manifold.dist(self.point[...], target, c) ** 2

        model = Model()
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
        target = jnp.array([-0.15, 0.1, 0.2], dtype=jnp.float32)

        def loss_fn(m):
            return m(target)

        c_before = float(model.curvature())
        losses = []
        for _ in range(25):
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            losses.append(float(loss))

        c_after = float(model.curvature())
        assert all(jnp.isfinite(jnp.array(loss_val)) for loss_val in losses)
        assert c_before != c_after, "signed curvature did not update"
        assert jnp.isfinite(jnp.array(c_after))
