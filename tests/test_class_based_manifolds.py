"""Contract tests for the class-based manifold API: jit, vmap, grad and dtype.

One parametrized matrix (4 manifolds x 4 contracts = 16 items) replaces the 44 near-duplicate
per-manifold tests this file used to hold. The rewrite is driven by the audit finding that the
old file passed unchanged with a 2x-wrong Poincare distance: every assertion in it was a
``dtype ==``, ``shape ==``, ``isfinite`` or ``> 0`` check, so nothing pinned a value.

What each contract now checks:

* jit   — the jitted call equals the eager call (exactly for ``dist``, to within a ULP for the
          composite maps — see ULP_TOL), the distance equals an independent closed form, and a
          second call with the same signature does not retrace (the trace counter of
          ``_counted``); the old tests only called a jitted function twice with different values
          and asserted the two results differed.
* vmap  — the batched call equals a Python loop over the eager call, element for element.
* grad  — gradients are finite, non-zero, and identical to ``value_and_grad``'s.
* dtype — explicitly-``float32`` inputs are cast up by a float64 manifold and explicitly-
          ``float64`` inputs are cast down by a float32 manifold (``tests/conftest.py`` enables
          x64 globally, so without the explicit ``.astype`` the cast path is never exercised),
          and both dtypes agree numerically.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperbolix.manifolds.euclidean import Euclidean
from hyperbolix.manifolds.hyperboloid import VERSION_DEFAULT, VERSION_LEGACY, Hyperboloid
from hyperbolix.manifolds.poincare import VERSION_MOBIUS_DIRECT, Poincare
from hyperbolix.manifolds.proper_velocity import ProperVelocity

# ---------------------------------------------------------------------------
# Independent distance oracles (NumPy transcriptions of the textbook closed forms).
# They must not call back into hyperbolix: their job is to pin the metric normalization that
# the library's own cross-checks cannot (a factor error shared by dist/logmap is invisible to
# every consistency test in the suite).


def _oracle_euclidean(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """d(x, y) = ‖x - y‖."""
    return float(np.linalg.norm(x - y))


def _oracle_poincare(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """d(x, y) = (1/√c)·arcosh(1 + 2c‖x-y‖² / ((1-c‖x‖²)(1-c‖y‖²)))."""
    num = 2.0 * c * np.sum((x - y) ** 2)
    den = (1.0 - c * np.sum(x**2)) * (1.0 - c * np.sum(y**2))
    return float(np.arccosh(1.0 + num / den) / np.sqrt(c))


def _oracle_hyperboloid(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """d(x, y) = (1/√c)·arcosh(-c·⟨x, y⟩_L), with ⟨x, y⟩_L = -x₀y₀ + x_s·y_s."""
    lorentz = -x[0] * y[0] + float(np.dot(x[1:], y[1:]))
    return float(np.arccosh(-c * lorentz) / np.sqrt(c))


def _oracle_pv(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """PV points are the spatial parts of hyperboloid points: lift and use the Lorentz form.

    x ↦ (√(1/c + ‖x‖²), x) satisfies ⟨X, X⟩_L = -1/c, so d_PV(x, y) = d_hyperboloid(X, Y).
    """
    x0 = np.sqrt(1.0 / c + np.sum(x**2))
    y0 = np.sqrt(1.0 / c + np.sum(y**2))
    lorentz = -x0 * y0 + float(np.dot(x, y))
    return float(np.arccosh(-c * lorentz) / np.sqrt(c))


# ---------------------------------------------------------------------------
# Case table


class ManifoldCase(NamedTuple):
    """One manifold under test, with raw (pre-projection) points and a distance oracle."""

    name: str
    make: Callable[[jnp.dtype], object]
    c: float
    version_idx: int | None
    x_raw: list[float]
    y_raw: list[float]
    x2_raw: list[float]
    y2_raw: list[float]
    v_raw: list[float]
    oracle: Callable[[np.ndarray, np.ndarray, float], float]


CASES = [
    ManifoldCase(
        name="Euclidean",
        make=lambda dtype: Euclidean(dtype=dtype),
        c=0.0,
        version_idx=None,
        # d = √2 — the old Euclidean "no recompilation" test compared this to the equally
        # translation-invariant d([1,1],[2,2]) and so would have passed for any constant.
        x_raw=[0.0, 0.0],
        y_raw=[1.0, 1.0],
        x2_raw=[1.0, 2.0],
        y2_raw=[3.0, 4.0],
        v_raw=[0.5, 0.5],
        oracle=_oracle_euclidean,
    ),
    ManifoldCase(
        name="PoincareBall",
        make=lambda dtype: Poincare(dtype=dtype),
        c=1.0,
        version_idx=VERSION_MOBIUS_DIRECT,
        x_raw=[0.1, 0.2],
        y_raw=[0.3, 0.4],
        x2_raw=[0.15, 0.25],
        y2_raw=[0.35, 0.45],
        v_raw=[0.05, 0.05],
        oracle=_oracle_poincare,
    ),
    ManifoldCase(
        name="Hyperboloid",
        make=lambda dtype: Hyperboloid(dtype=dtype),
        c=1.0,
        version_idx=VERSION_DEFAULT,
        x_raw=[1.0, 0.1, 0.2],
        y_raw=[1.0, 0.3, 0.4],
        x2_raw=[1.0, 0.15, 0.25],
        y2_raw=[1.0, 0.35, 0.45],
        v_raw=[0.0, 0.05, 0.05],
        oracle=_oracle_hyperboloid,
    ),
    ManifoldCase(
        # Same points as the "Hyperboloid" case above, bound to VERSION_LEGACY instead of
        # VERSION_DEFAULT: the acosh-based oracle is exact at these small radii for both arms,
        # so this pins the legacy switch slot against the same independent oracle.
        name="HyperboloidLegacy",
        make=lambda dtype: Hyperboloid(dtype=dtype),
        c=1.0,
        version_idx=VERSION_LEGACY,
        x_raw=[1.0, 0.1, 0.2],
        y_raw=[1.0, 0.3, 0.4],
        x2_raw=[1.0, 0.15, 0.25],
        y2_raw=[1.0, 0.35, 0.45],
        v_raw=[0.0, 0.05, 0.05],
        oracle=_oracle_hyperboloid,
    ),
    ManifoldCase(
        name="ProperVelocity",
        make=lambda dtype: ProperVelocity(dtype=dtype),
        c=1.0,
        version_idx=None,
        x_raw=[0.1, 0.2],
        y_raw=[0.3, 0.4],
        x2_raw=[0.15, 0.25],
        y2_raw=[0.35, 0.45],
        v_raw=[0.05, 0.05],
        oracle=_oracle_pv,
    ),
]

CASE_IDS = [case.name for case in CASES]

# jit-vs-eager is asserted at *zero* tolerance wherever that holds (it does for ``dist`` on all
# four manifolds, and for every vmap-vs-loop comparison below). XLA is free to fuse and reorder
# float operations inside a jitted trace, and for the composite exp/log maps that costs a single
# ULP, so those two use ULP_TOL. Both bounds are many orders of magnitude tighter than any
# semantic difference (a dropped factor, a wrong ``version_idx`` branch) could hide in.
ULP_TOL = {"rtol": 1e-13, "atol": 1e-15}

# Tolerance for "the float64 kernel on float32-rounded inputs matches it on exact inputs" in
# ``test_dtype_policy_casts_inputs_to_manifold_dtype``. Derived, not guessed -- see the comment
# at the assertion: 4x the 2.11e-6 input-rounding bound of the worst case (HyperboloidLegacy).
DTYPE_ROUNDING_RTOL = 1e-5


def _prepare(case: ManifoldCase, dtype: jnp.dtype):
    """Instantiate the manifold and project the raw points/vectors onto it."""
    manifold = case.make(dtype)
    proj = functools.partial(manifold.proj, c=case.c)
    x, y, x2, y2 = (proj(jnp.asarray(raw, dtype=dtype)) for raw in (case.x_raw, case.y_raw, case.x2_raw, case.y2_raw))
    v = manifold.tangent_proj(jnp.asarray(case.v_raw, dtype=dtype), x, case.c)
    return manifold, x, y, x2, y2, v


def _dist_fn(case: ManifoldCase, manifold):
    """``dist`` with the case's (static) ``version_idx`` bound, leaving ``(x, y, c)``."""
    if case.version_idx is None:
        return manifold.dist
    return functools.partial(manifold.dist, version_idx=case.version_idx)


def _counted(fn: Callable) -> tuple[Callable, list[int]]:
    """Return ``(wrapper, traces)``; ``traces`` grows by one entry each time the body is traced.

    This is how "the second call did not recompile" is asserted. ``jitted._cache_size()`` cannot
    express it: jax keeps a single process-global ``PjitFunctionCache(capacity=8192)`` shared by
    every jitted callable, including the ones it builds internally for eager dispatch. Earlier
    tests in the same process can fill it, the freshly inserted entry is then evicted between two
    calls of this function, and ``_cache_size()`` reads 0 for a function that never recompiled —
    so that assertion measured cache residency and test order, not retracing. Counting how often
    the wrapped body runs under tracing is order-independent, and it still fails (counter 2) when
    a genuine retrace happens.
    """
    traces: list[int] = []

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        traces.append(1)
        return fn(*args, **kwargs)

    return wrapper, traces


# ---------------------------------------------------------------------------
# Contracts


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_jit_matches_eager_and_reuses_cache(case: ManifoldCase) -> None:
    """Jitted calls must equal eager calls, and must not recompile on the second call."""
    manifold, x, y, x2, y2, v = _prepare(case, jnp.float64)
    dist = _dist_fn(case, manifold)

    # Absolute anchor: the library distance equals an independently transcribed closed form.
    assert float(dist(x, y, case.c)) == pytest.approx(case.oracle(np.asarray(x), np.asarray(y), case.c or 1.0), rel=1e-9)

    counted_dist, dist_traces = _counted(dist)
    dist_jit = jax.jit(counted_dist)
    assert jnp.allclose(dist_jit(x, y, case.c), dist(x, y, case.c), rtol=0, atol=0)
    assert len(dist_traces) == 1

    # Second call, same shapes/dtypes/static arguments: must reuse the existing trace.
    assert jnp.allclose(dist_jit(x2, y2, case.c), dist(x2, y2, case.c), rtol=0, atol=0)
    assert len(dist_traces) == 1

    expmap_jit = jax.jit(manifold.expmap)
    assert jnp.allclose(expmap_jit(v, x, case.c), manifold.expmap(v, x, case.c), **ULP_TOL)

    logmap_jit = jax.jit(manifold.logmap)
    assert jnp.allclose(logmap_jit(y, x, case.c), manifold.logmap(y, x, case.c), **ULP_TOL)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_vmap_matches_per_point_loop(case: ManifoldCase) -> None:
    """Batched calls must equal a Python loop over the single-point calls, element for element."""
    manifold, x, y, x2, y2, v = _prepare(case, jnp.float64)
    dist = _dist_fn(case, manifold)

    x_batch = jnp.stack([x, x2])
    y_batch = jnp.stack([y, y2])
    v_batch = jnp.stack([v, v])

    dist_batched = jax.vmap(dist, in_axes=(0, 0, None))(x_batch, y_batch, case.c)
    assert dist_batched.shape == (2,)
    assert jnp.allclose(dist_batched, jnp.stack([dist(x, y, case.c), dist(x2, y2, case.c)]), rtol=0, atol=0)

    expmap_batched = jax.vmap(manifold.expmap, in_axes=(0, 0, None))(v_batch, x_batch, case.c)
    expected = jnp.stack([manifold.expmap(v, x, case.c), manifold.expmap(v, x2, case.c)])
    assert jnp.allclose(expmap_batched, expected, rtol=0, atol=0)

    proj_batched = jax.vmap(manifold.proj, in_axes=(0, None))(x_batch, case.c)
    assert jnp.allclose(proj_batched, jnp.stack([manifold.proj(x, case.c), manifold.proj(x2, case.c)]), rtol=0, atol=0)

    # vmap and jit compose without changing the result.
    composed = jax.jit(lambda a, b: jax.vmap(dist, in_axes=(0, 0, None))(a, b, case.c))(x_batch, y_batch)
    assert jnp.allclose(composed, dist_batched, rtol=0, atol=0)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_grad_is_finite_nonzero_and_matches_value_and_grad(case: ManifoldCase) -> None:
    """Gradients must flow through the class methods and agree with ``value_and_grad``."""
    manifold, x, y, _, _, v = _prepare(case, jnp.float64)
    dist = _dist_fn(case, manifold)

    def dist_loss(x_raw):
        return dist(manifold.proj(x_raw, case.c), y, case.c)

    grad = jax.grad(dist_loss)(x)
    assert grad.shape == x.shape
    assert grad.dtype == jnp.float64
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.linalg.norm(grad) > 0

    value, grad_pair = jax.value_and_grad(dist_loss)(x)
    assert value.dtype == jnp.float64
    assert jnp.allclose(value, dist_loss(x), rtol=0, atol=0)
    assert jnp.allclose(grad_pair, grad, rtol=0, atol=0)

    def expmap_loss(v_in):
        return jnp.sum(manifold.expmap(v_in, x, case.c) ** 2)

    expmap_grad = jax.grad(expmap_loss)(v)
    assert jnp.all(jnp.isfinite(expmap_grad))
    assert jnp.linalg.norm(expmap_grad) > 0


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_dtype_policy_casts_inputs_to_manifold_dtype(case: ManifoldCase) -> None:
    """The manifold's own dtype wins over the input dtype, in both directions.

    Inputs are built with an explicit ``.astype`` because ``tests/conftest.py`` enables x64
    globally: an unannotated ``jnp.array([0.1, 0.2])`` is float64 here, so a test written that
    way never exercises ``ManifoldBase._cast`` at all (audit A3-03).
    """
    manifold_f64, x64, y64, _, _, _ = _prepare(case, jnp.float64)
    manifold_f32, x32, y32, _, _, _ = _prepare(case, jnp.float32)
    dist_f64 = _dist_fn(case, manifold_f64)
    dist_f32 = _dist_fn(case, manifold_f32)

    assert x32.dtype == jnp.float32 and x64.dtype == jnp.float64

    # float32 inputs into a float64 manifold are promoted.
    d_up = dist_f64(x32, y32, case.c)
    assert d_up.dtype == jnp.float64

    # float64 inputs into a float32 manifold are demoted.
    d_down = dist_f32(x64, y64, case.c)
    assert d_down.dtype == jnp.float32

    # Bound methods keep their own instance's dtype, and the two agree numerically.
    assert jnp.allclose(d_up, d_down, rtol=1e-5)
    # ``d_up`` runs the float64 kernel on float32-rounded inputs, so the gap is input-rounding
    # amplified by the distance formula. Legacy hyperboloid arm, d = acosh(u)/√c with
    # u = -c⟨x,y⟩_L: dd/du = 1/(√c·√(u²-1)) and u - 1 ≈ c·d²/2 for small d, so
    # Δd/d ≈ Δu/(c·d²); with Δu ≤ eps32·(|x₀y₀| + |x_s·y_s|) = 1.50e-7 and d = 0.2662 that is
    # 2.11e-6. Times a safety factor of 4 (rounded up) = 1e-5. CPU and GPU differ because the
    # Lorentz inner product's reduction order differs: the measured gap is 4.1e-7 on CPU and
    # 2.11e-6 on an A100, i.e. the A100 sits exactly at the bound and the old rtol=1e-6 failed
    # there. Every other case in CASES measures ≤ 4.6e-8.
    # (Derivation and a 20-pair margin check: logs/2026-09-03_test_infra/probe_dtype_policy_bound.py)
    assert jnp.allclose(d_up, dist_f64(x64, y64, case.c), rtol=DTYPE_ROUNDING_RTOL)

    # The same holds under jit, with one compilation per dtype.
    counted_f64, traces_f64 = _counted(dist_f64)
    counted_f32, traces_f32 = _counted(dist_f32)
    dist_f64_jit = jax.jit(counted_f64)
    dist_f32_jit = jax.jit(counted_f32)
    assert dist_f64_jit(x64, y64, case.c).dtype == jnp.float64
    assert dist_f32_jit(x32, y32, case.c).dtype == jnp.float32
    assert len(traces_f64) == 1
    assert len(traces_f32) == 1

    # Re-calling each with its own dtype reuses that trace: still one compilation per dtype.
    assert dist_f64_jit(x64, y64, case.c).dtype == jnp.float64
    assert dist_f32_jit(x32, y32, case.c).dtype == jnp.float32
    assert len(traces_f64) == 1
    assert len(traces_f32) == 1
