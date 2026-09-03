"""Closed-form value oracles for manifold internals that the property suites cannot pin.

Companion to ``tests/test_manifolds.py`` (generic identities over the fixture point clouds) and
``tests/test_class_based_manifolds.py`` (jit/vmap/grad/dtype contracts). Everything here compares a
library call against an **independent** expression — a hand-derived closed form, a NumPy/SciPy
transcription, or a finite-difference estimate — rather than against another library call.

Audit specs covered (F6-M1): M1-01 MLR value oracles, M1-02 exact Riemannian-gradient leg,
M1-04 projection boundary clamp, M1-08 sinh lift, M1-09 beta-concatenation, M1-10 conformal factor
at signed curvature, M1-11 protocol conformance, M1-12 isometry gradients, M1-14 Lorentz boost.

Dimension key:
    B: batch          P: number of MLR classes / hyperplanes
    D: spatial dim    A: ambient dim (= D + 1 on the hyperboloid, time coordinate first)
"""

from __future__ import annotations

import decimal
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

from hyperbolix.manifolds import (
    Euclidean,
    Hyperboloid,
    Manifold,
    Poincare,
    ProductManifold,
    ProperVelocity,
    Stereographic,
)
from hyperbolix.manifolds import isometry_mappings as iso
from hyperbolix.manifolds._gyrovector_core import _conformal_factor, _conformal_factor_batch, _proj
from hyperbolix.nn_layers.hyperboloid_core import sinh_lift_to_hyperboloid
from hyperbolix.utils.math_utils import MIN_NORM

jax.config.update("jax_enable_x64", True)

# Float64 instances throughout: the oracles below are exact algebraic identities, so the f32
# parametrization would only measure rounding. Precision behavior is owned by test_precision.py.
F64 = jnp.float64

# Curvatures far from 1.0 in both directions — a c-independent bug (a dropped √c, a wrong power)
# is invisible at c = 1.0, where every √c factor equals one.
CURVATURES = [0.3, 1.0, 2.5]


# =============================================================================================
# M1-01 — compute_mlr value oracles (Hyperboloid + ProperVelocity)
#
# Ported from the Poincaré ``compute_mlr_pp`` oracles in tests/test_precision.py, which had no
# hyperboloid/PV counterpart: both siblings were covered only by shape + isfinite + dtype checks,
# so a sign flip or a dropped cosh/sinh term in either passed the whole suite.
#
# Both models put the hyperplane's closest point to the origin at the SAME place along the unit
# normal ẑ, at PV/spatial coordinate ``sinh(√c·r)/√c``:
#
#   Hyperboloid (Bdeir et al. 2023): logit = ‖z‖·asinh(√c·alpha/‖z‖)/√c,
#       alpha = cosh(√c·r)·⟨x_s, z⟩ - x_t·sinh(√c·r)·‖z‖
#   ProperVelocity (Chen et al. 2026 Thm 5.2): logit = (‖z‖/√c)·asinh(beta),
#       beta = cosh(√c·r)·(√c/‖z‖)·⟨x, z⟩ - sinh(√c·r)·√(1 + c‖x‖²)
#
# Substituting x = (sinh(√c·r)/√c)·ẑ (with the hyperboloid time coordinate cosh(√c·r)/√c fixed by
# the constraint) makes alpha and beta vanish identically — that is the zero-on-hyperplane oracle.
# =============================================================================================


def _hyperplane_offset(c: float, r: float) -> float:
    """Spatial coordinate along ẑ of the hyperplane point closest to the origin: sinh(√c·r)/√c."""
    sqrt_c = np.sqrt(c)
    return float(np.sinh(sqrt_c * r) / sqrt_c)


def _hyperboloid_point_on_hyperplane(z_D: jnp.ndarray, c: float, r: float) -> jnp.ndarray:
    """Hyperboloid point ``(cosh(√c·r)/√c, sinh(√c·r)/√c · ẑ)`` — on the decision boundary."""
    sqrt_c = np.sqrt(c)
    z_hat_D = z_D / jnp.linalg.norm(z_D)
    x_t = float(np.cosh(sqrt_c * r) / sqrt_c)
    x_s_D = _hyperplane_offset(c, r) * z_hat_D
    return jnp.concatenate([jnp.array([x_t], dtype=F64), x_s_D])


@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("r_val", [-0.6, 0.0, 0.4])
def test_hyperboloid_mlr_vanishes_on_its_own_hyperplane(c: float, r_val: float):
    """The logit is zero exactly on the decision boundary, at every curvature and offset.

    The ``r != 0`` cases are the discriminating ones: at ``r = 0`` the ``x_t·sinh(√c·r)`` term
    drops out entirely, so a sign flip on it (or on ``x_t``) survives. Running all three offsets
    together also pins the relative sign of the two terms in ``alpha``.
    """
    manifold = Hyperboloid(dtype=F64)
    z_PD = jnp.array([[0.7, -0.3, 0.2]], dtype=F64)
    r_P1 = jnp.array([[r_val]], dtype=F64)
    x_A = _hyperboloid_point_on_hyperplane(z_PD[0], c, r_val)

    # The constructed point must genuinely be on the manifold, else "logit == 0" proves nothing.
    assert bool(manifold.is_in_manifold(x_A, c, atol=1e-10))

    logit = manifold.compute_mlr(x_A[None], z_PD, r_P1, c, clamping_factor=10.0, smoothing_factor=50.0)

    assert float(logit[0, 0]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_mlr_matches_closed_form_at_zero_offset(c: float):
    """At ``r = 0``: logit = ‖z‖·asinh(√c·⟨x_s, z⟩/‖z‖)/√c, transcribed in NumPy.

    Pins the two factors the zero-on-hyperplane test cannot see: the outer ``‖z‖`` multiplier
    (dropping it leaves the boundary at zero) and the ``1/√c`` scaling of the asinh.
    """
    manifold = Hyperboloid(dtype=F64)
    z_PD = jnp.array([[0.7, -0.3, 0.2], [-0.4, 0.9, 0.1]], dtype=F64)
    r_P1 = jnp.zeros((2, 1), dtype=F64)
    x_s_D = jnp.array([0.2, -0.1, 0.15], dtype=F64)
    x_A = jnp.concatenate([jnp.array([np.sqrt(1.0 / c + float(jnp.dot(x_s_D, x_s_D)))], dtype=F64), x_s_D])

    logits_BP = manifold.compute_mlr(x_A[None], z_PD, r_P1, c, clamping_factor=10.0, smoothing_factor=50.0)

    sqrt_c = np.sqrt(c)
    z_np = np.asarray(z_PD, dtype=np.float64)
    z_norm_P = np.linalg.norm(z_np, axis=-1)
    expected_P = z_norm_P * np.arcsinh(sqrt_c * (z_np @ np.asarray(x_s_D)) / z_norm_P) / sqrt_c

    assert np.allclose(np.asarray(logits_BP[0]), expected_P, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_mlr_sign_follows_side_and_margin_is_monotone(c: float):
    """Sliding the point along ẑ through the boundary flips the logit sign and grows |logit|.

    Distinguishes a genuine signed margin from ``|margin|`` or from a constant: an implementation
    that returned ``asinh(|alpha|)`` would still vanish on the hyperplane.
    """
    manifold = Hyperboloid(dtype=F64)
    z_PD = jnp.array([[0.7, -0.3, 0.2]], dtype=F64)
    r_val = 0.4
    r_P1 = jnp.array([[r_val]], dtype=F64)
    z_hat_D = z_PD[0] / jnp.linalg.norm(z_PD[0])
    base = _hyperplane_offset(c, r_val)

    offsets = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    logits = []
    for delta in offsets:
        x_s_D = (base + float(delta)) * z_hat_D
        x_t = np.sqrt(1.0 / c + float(jnp.dot(x_s_D, x_s_D)))
        x_A = jnp.concatenate([jnp.array([x_t], dtype=F64), x_s_D])
        logits.append(float(manifold.compute_mlr(x_A[None], z_PD, r_P1, c, 10.0, 50.0)[0, 0]))

    assert logits[0] < logits[1] < logits[2] < logits[3] < logits[4], "margin is not monotone along ẑ"
    assert logits[0] < 0.0 and logits[1] < 0.0, "points on the origin side must score negative"
    assert logits[3] > 0.0 and logits[4] > 0.0, "points on the far side must score positive"
    assert logits[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("r_val", [-0.6, 0.0, 0.4])
def test_pv_mlr_vanishes_on_its_own_hyperplane(c: float, r_val: float):
    """PV sibling of the hyperboloid boundary oracle (Chen et al. 2026, Thm 5.2 / Eq. 19).

    Same construction, same closed-form offset ``sinh(√c·r)/√c`` — PV has no time coordinate, so
    the boundary point is just that multiple of ẑ. The ``beta`` terms cancel only if the ``cosh``,
    ``sinh`` and ``√(1 + c‖x‖²)`` factors are all in place with the right relative sign.
    """
    manifold = ProperVelocity(dtype=F64)
    z_PD = jnp.array([[0.7, -0.3, 0.2]], dtype=F64)
    r_P1 = jnp.array([[r_val]], dtype=F64)
    z_hat_D = z_PD[0] / jnp.linalg.norm(z_PD[0])
    x_D = _hyperplane_offset(c, r_val) * z_hat_D

    logit = manifold.compute_mlr(x_D[None], z_PD, r_P1, c, clamping_factor=10.0, smoothing_factor=50.0)

    assert float(logit[0, 0]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("c", CURVATURES)
def test_pv_mlr_matches_closed_form_at_zero_offset(c: float):
    """At ``r = 0``: logit = (‖z‖/√c)·asinh(√c·⟨x, z⟩/‖z‖), transcribed in NumPy."""
    manifold = ProperVelocity(dtype=F64)
    z_PD = jnp.array([[0.7, -0.3, 0.2], [-0.4, 0.9, 0.1]], dtype=F64)
    r_P1 = jnp.zeros((2, 1), dtype=F64)
    x_D = jnp.array([0.2, -0.1, 0.15], dtype=F64)

    logits_BP = manifold.compute_mlr(x_D[None], z_PD, r_P1, c, clamping_factor=10.0, smoothing_factor=50.0)

    sqrt_c = np.sqrt(c)
    z_np = np.asarray(z_PD, dtype=np.float64)
    z_norm_P = np.linalg.norm(z_np, axis=-1)
    expected_P = (z_norm_P / sqrt_c) * np.arcsinh(sqrt_c * (z_np @ np.asarray(x_D)) / z_norm_P)

    assert np.allclose(np.asarray(logits_BP[0]), expected_P, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("c", CURVATURES)
def test_pv_mlr_sign_follows_side_and_margin_is_monotone(c: float):
    """PV sibling of the signed-margin oracle."""
    manifold = ProperVelocity(dtype=F64)
    z_PD = jnp.array([[0.7, -0.3, 0.2]], dtype=F64)
    r_val = 0.4
    r_P1 = jnp.array([[r_val]], dtype=F64)
    z_hat_D = z_PD[0] / jnp.linalg.norm(z_PD[0])
    base = _hyperplane_offset(c, r_val)

    offsets = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    x_BD = jnp.stack([(base + float(d)) * z_hat_D for d in offsets])
    logits = np.asarray(manifold.compute_mlr(x_BD, z_PD, r_P1, c, 10.0, 50.0)[:, 0])

    assert np.all(np.diff(logits) > 0.0), "margin is not monotone along ẑ"
    assert logits[0] < 0.0 and logits[1] < 0.0
    assert logits[3] > 0.0 and logits[4] > 0.0
    assert logits[2] == pytest.approx(0.0, abs=1e-12)


# =============================================================================================
# M1-02 — exact Riemannian gradient (extends the metric-duality leg landed in F2)
#
# ``tests/test_manifolds.py::test_egrad2rgrad_is_metric_dual`` pins the defining property
# ⟨g, v⟩_euclid = g_x(rgrad, v) against tangent-projected probes v. That is a *relational* check:
# it constrains rgrad only on the tangent space, so on the Poincaré ball (where the tangent space
# is everything) it is tight but on the Hyperboloid it says nothing about the normal component
# convention. The two tests below pin the VALUES outright from each model's own formula.
# =============================================================================================


@pytest.mark.parametrize("c", CURVATURES)
def test_poincare_egrad2rgrad_equals_grad_over_lambda_squared(c: float):
    """Poincaré: rgrad = g/λ_x² = g·(1 - c‖x‖²)²/4, transcribed in NumPy.

    The conformal metric is g_x = λ_x²·I, so the dual of a Euclidean gradient divides by λ_x².
    A wrong power (λ instead of λ², the historical ``2(1 - c‖x‖²)`` transcription) changes the
    value at every non-origin point but still yields a "tangent" vector.
    """
    manifold = Poincare(dtype=F64)
    x_D = jnp.array([0.2, -0.15, 0.3], dtype=F64) / np.sqrt(c)
    g_D = jnp.array([1.0, -2.0, 0.5], dtype=F64)

    rgrad_D = manifold.egrad2rgrad(g_D, x_D, c)

    x_np, g_np = np.asarray(x_D), np.asarray(g_D)
    lam = 2.0 / (1.0 - c * float(x_np @ x_np))
    expected_D = g_np / lam**2

    assert np.allclose(np.asarray(rgrad_D), expected_D, rtol=1e-12, atol=1e-14)
    # The λ² denominator must actually bite: at these points it is far from 1.
    assert not np.allclose(np.asarray(rgrad_D), g_np, rtol=1e-3)


@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_egrad2rgrad_equals_minkowski_projection(c: float):
    """Hyperboloid: rgrad = g_L + c·⟨x, g_L⟩_L·x with g_L the time-sign-flipped Euclidean gradient.

    Independent NumPy transcription using the explicit Minkowski form J = diag(-1, 1, …, 1). The
    metric-duality test is blind to the normal component here because it probes only with tangent
    vectors, for which ⟨·, x⟩_L = 0.
    """
    manifold = Hyperboloid(dtype=F64)
    x_A = manifold.proj(jnp.array([1.5, 0.3, -0.4, 0.2], dtype=F64), c)
    g_A = jnp.array([0.7, 1.0, -2.0, 0.5], dtype=F64)

    rgrad_A = manifold.egrad2rgrad(g_A, x_A, c)

    x_np = np.asarray(x_A)
    g_lorentz = np.asarray(g_A).copy()
    g_lorentz[0] = -g_lorentz[0]
    minkowski = np.diag([-1.0] + [1.0] * (x_np.shape[0] - 1))
    expected_A = g_lorentz + c * float(x_np @ minkowski @ g_lorentz) * x_np

    assert np.allclose(np.asarray(rgrad_A), expected_A, rtol=1e-10, atol=1e-12)
    # Must be Minkowski-orthogonal to x, i.e. genuinely tangent.
    assert float(np.asarray(rgrad_A) @ minkowski @ x_np) == pytest.approx(0.0, abs=1e-10)


# =============================================================================================
# M1-04 — the boundary-clamp branch of the shared gyrovector projection
#
# ``_proj`` is the only guard keeping downstream ``1/(1 - c‖x‖²)`` divisions finite. Its clamp
# branch (``norm > max_norm``) is never exercised by the fixture point clouds, which are sampled
# strictly inside the ball, so nothing pinned the dtype-scaled safety margin.
# =============================================================================================


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("c", [1.0, 4.0])
def test_proj_clamps_outside_points_strictly_inside_the_ball(dtype, c: float):
    """A point far outside lands at radius ``1/√c - eps**0.75``, strictly inside, direction kept.

    ``c‖proj(x)‖² < 1`` *strictly* is the property that matters: a clamp to exactly ``1/√c`` would
    make the conformal factor's denominator zero and every downstream division singular.
    """
    outside_D = jnp.array([10.0, 0.0, 0.0], dtype=dtype)

    projected_D = _proj(outside_D, c)

    margin = float(jnp.finfo(dtype).eps ** 0.75)
    radius = 1.0 / np.sqrt(c)
    norm = float(jnp.linalg.norm(projected_D))
    assert norm == pytest.approx(radius - margin, rel=1e-6)
    assert float(c * jnp.sum(projected_D.astype(jnp.float64) ** 2)) < 1.0
    # Direction preserved: the clamp is a pure radial rescale.
    assert np.allclose(np.asarray(projected_D)[1:], 0.0, atol=1e-12)
    assert float(projected_D[0]) > 0.0


@pytest.mark.parametrize("c", [1.0, 4.0])
def test_proj_is_identity_strictly_inside_the_ball(c: float):
    """Interior points pass through bit-for-bit — the clamp must not shrink everything."""
    inside_D = jnp.array([0.1, -0.2, 0.05], dtype=F64) / np.sqrt(c)

    assert np.array_equal(np.asarray(_proj(inside_D, c)), np.asarray(inside_D))


def test_proj_boundary_margin_scales_with_dtype_precision():
    """The safety margin is ``eps**0.75``, so float32's is orders of magnitude wider than float64's.

    A hard-coded constant margin (or a margin dropped entirely) would make these two equal.
    """
    outside_f32 = jnp.array([10.0, 0.0, 0.0], dtype=jnp.float32)
    outside_f64 = jnp.array([10.0, 0.0, 0.0], dtype=F64)

    margin_f32 = 1.0 - float(jnp.linalg.norm(_proj(outside_f32, 1.0)))
    margin_f64 = 1.0 - float(jnp.linalg.norm(_proj(outside_f64, 1.0)))

    assert margin_f32 > 0.0 and margin_f64 > 0.0
    assert margin_f32 > 1000.0 * margin_f64


@pytest.mark.parametrize("c", [0.0, -0.5, -2.0])
def test_proj_has_no_boundary_for_nonpositive_curvature(c: float):
    """For ``c <= 0`` (Euclidean / spherical κ-stereographic) the space is all of R^d: identity."""
    far_D = jnp.array([10.0, -7.0, 3.0], dtype=F64)

    assert np.array_equal(np.asarray(_proj(far_D, c)), np.asarray(far_D))


# =============================================================================================
# M1-10 — conformal factor across the sign of the curvature, single point vs batch
# =============================================================================================


@pytest.mark.parametrize("c", [2.0, 0.5, 0.0, -0.5, -2.0])
def test_conformal_factor_matches_closed_form_at_signed_curvature(c: float):
    """λ_x = 2/(1 - c‖x‖²) at hyperbolic, Euclidean and spherical curvature.

    The ``c <= 0`` half is the point of the sweep: with ``c < 0`` the denominator is ``1 + |c|‖x‖²``
    and λ < 2, so any ``abs(c)`` "generalization" of the formula (a natural-looking edit, since the
    boundary floor beside it genuinely uses ``|c|``) inverts the correction. At ``c = 0`` the factor
    is exactly 2 — the gyrovector Euclidean-limit convention, not 1.
    """
    x_D = jnp.array([0.2, -0.1], dtype=F64)

    lam_single = float(_conformal_factor(x_D, c))
    lam_batch = float(_conformal_factor_batch(x_D, c)[0])

    expected = 2.0 / (1.0 - c * float(jnp.dot(x_D, x_D)))
    assert lam_single == pytest.approx(expected, rel=1e-14)
    assert lam_batch == pytest.approx(expected, rel=1e-14)
    if c == 0.0:
        assert lam_single == pytest.approx(2.0, rel=1e-14)


@pytest.mark.parametrize("c", [2.0, 0.0, -0.5])
def test_conformal_factor_accessor_broadcasts_and_matches_the_single_point_helper(c: float):
    """``Stereographic.conformal_factor`` keeps a trailing 1-axis and agrees row-by-row.

    Guards the accessor wiring (batch helper vs single-point helper) and the ``(..., 1)`` shape the
    NN layers rely on for broadcasting against ``(..., dim)`` features.
    """
    manifold = Stereographic(dtype=F64)
    x_BD = jnp.array([[0.2, -0.1], [0.0, 0.0], [0.35, 0.15]], dtype=F64)

    lam_B1 = manifold.conformal_factor(x_BD, c)

    assert lam_B1.shape == (3, 1)
    per_point = np.array([float(_conformal_factor(x_BD[i], c)) for i in range(3)])
    assert np.allclose(np.asarray(lam_B1)[:, 0], per_point, rtol=1e-14)
    # Origin row: λ_0 = 2 at every curvature.
    assert float(lam_B1[1, 0]) == pytest.approx(2.0, rel=1e-14)


# =============================================================================================
# M1-08 — sinh lift to the hyperboloid (shared output map of the PLFC and Busemann linear layers)
# =============================================================================================


@pytest.mark.parametrize("c", CURVATURES)
def test_sinh_lift_is_elementwise_sinh_over_sqrt_c_with_lorentz_time(c: float):
    """y_s = sinh(√c·s)/√c elementwise, y_t fixed by ⟨y, y⟩_L = -1/c.

    The ``/√c`` and the ``√c`` inside the ``sinh`` are separate factors that both vanish at c = 1;
    the two off-unit curvatures are what make this a real check.
    """
    v_max = 5.0
    spatial_BO = jnp.array([[0.3, -0.5, 0.1], [0.0, 0.2, -0.05]], dtype=F64)

    lifted_BA = sinh_lift_to_hyperboloid(spatial_BO, c, v_max)

    sqrt_c = np.sqrt(c)
    expected_spatial = np.sinh(sqrt_c * np.asarray(spatial_BO)) / sqrt_c
    assert np.allclose(np.asarray(lifted_BA)[:, 1:], expected_spatial, rtol=1e-12, atol=1e-14)

    time_B = np.asarray(lifted_BA)[:, 0]
    space_sq_B = np.sum(np.asarray(lifted_BA)[:, 1:] ** 2, axis=-1)
    assert np.allclose(time_B**2 - space_sq_B, 1.0 / c, rtol=1e-12)
    assert np.all(time_B > 0.0), "must land on the upper sheet"


@pytest.mark.parametrize("c", [0.3, 2.5])
def test_sinh_lift_saturates_at_v_max(c: float):
    """Past the clip bound the output stops moving: the overflow guard is a hard clip on √c·s.

    Without it ``sinh`` exponentiates an unbounded score. Doubling an already-saturated input must
    change nothing, and the saturated spatial value must equal ``sinh(v_max)/√c``.
    """
    v_max = 5.0
    big_BO = jnp.array([[100.0, -200.0]], dtype=F64)

    lifted_once = sinh_lift_to_hyperboloid(big_BO, c, v_max)
    lifted_twice = sinh_lift_to_hyperboloid(2.0 * big_BO, c, v_max)

    assert np.allclose(np.asarray(lifted_once), np.asarray(lifted_twice), rtol=1e-14)
    saturated = np.sinh(v_max) / np.sqrt(c)
    assert float(lifted_once[0, 1]) == pytest.approx(saturated, rel=1e-12)
    assert float(lifted_once[0, 2]) == pytest.approx(-saturated, rel=1e-12)


# =============================================================================================
# M1-09 — beta-concatenation (HNN++, Shimizu et al. 2020)
# =============================================================================================


@pytest.mark.parametrize("c", CURVATURES)
def test_beta_concat_applies_the_scipy_beta_ratio_in_the_tangent_space(c: float):
    """expmap_0( B(n/2, ½)/B(n_i/2, ½) · concat(logmap_0(p_i)) ), with the ratio from SciPy.

    The beta ratio is the entire content of the operation — drop it (scale = 1) and the result is
    a plain tangent-space concatenation, which still lands on the ball and still has the right
    shape. Only a value oracle separates the two.
    """
    manifold = Poincare(dtype=F64)
    points_MD = jnp.array([[0.1, 0.2], [0.3, -0.1], [0.05, 0.05]], dtype=F64) / np.sqrt(c)

    result_N = manifold.beta_concat(points_MD, c)

    n_factors, dim_i = points_MD.shape
    scale = scipy.special.beta(n_factors * dim_i / 2.0, 0.5) / scipy.special.beta(dim_i / 2.0, 0.5)
    tangent_N = jnp.concatenate([scale * manifold.logmap_0(p, c) for p in points_MD])
    expected_N = manifold.expmap_0(tangent_N, c)

    assert result_N.shape == (n_factors * dim_i,)
    assert np.allclose(np.asarray(result_N), np.asarray(expected_N), rtol=1e-12, atol=1e-14)
    assert scale == pytest.approx(0.5333333333333333, rel=1e-12), "M=3, n_i=2 -> B(3,.5)/B(1,.5)"
    assert bool(manifold.is_in_manifold(result_N, c, atol=1e-10))


def test_beta_concat_does_not_leak_float64_from_the_scipy_beta_call():
    """A float32 input stays float32 under global x64.

    ``jax.scipy.special.beta`` returns a *strongly*-typed float64 scalar (unlike most scalar math,
    which stays weakly typed), so without an explicit cast the ratio silently promotes the whole
    computation. Covered for the layer path in test_dtype_respected_x64.py; pinned here on the
    manifold method itself, next to the value oracle that explains why the cast exists.
    """
    manifold = Poincare(dtype=jnp.float32)
    points_MD = jnp.array([[0.1, 0.2], [0.3, -0.1]], dtype=jnp.float32)

    assert manifold.beta_concat(points_MD, 1.0).dtype == jnp.float32


# =============================================================================================
# M1-14 — Lorentz boost (HoroPCA Fréchet-mean centering)
#
# tests/test_horopca.py checks that the boost sends the mean to the origin and preserves distances
# *after* ``proj_batch``, which re-normalizes onto the constraint surface and so hides an
# almost-Lorentz matrix. The algebraic invariants below are checked on the raw matrix.
# =============================================================================================


@pytest.mark.parametrize("c", CURVATURES)
def test_lorentz_boost_is_a_symmetric_lorentz_transformation(c: float):
    """BᵀJB = J with J = diag(-1, 1, …, 1), and B = Bᵀ (a *pure* boost, no rotation part).

    No ``proj`` anywhere: this is the defining identity of the Lorentz group, so a wrong
    ``gamma²/(1 + gamma)`` coefficient in the spatial block fails here even though the boosted
    points would still be re-projected onto the hyperboloid downstream.
    """
    manifold = Hyperboloid(dtype=F64)
    mu_A = manifold.proj(jnp.array([2.0, 0.3, -0.4, 0.5], dtype=F64), c)

    boost_AA = manifold.lorentz_boost(mu_A, c)

    boost_np = np.asarray(boost_AA)
    minkowski = np.diag([-1.0, 1.0, 1.0, 1.0])
    assert np.allclose(boost_np, boost_np.T, rtol=1e-12, atol=1e-14), "boost must be symmetric"
    assert np.allclose(boost_np.T @ minkowski @ boost_np, minkowski, rtol=1e-10, atol=1e-12)
    assert float(np.linalg.det(boost_np)) == pytest.approx(1.0, rel=1e-10), "proper (det = +1)"
    assert boost_np[0, 0] > 0.0, "orthochronous (preserves the time direction)"


@pytest.mark.parametrize("c", CURVATURES)
def test_lorentz_boost_sends_mu_to_the_origin_without_projection(c: float):
    """B @ mu == origin exactly, straight out of the matrix product."""
    manifold = Hyperboloid(dtype=F64)
    mu_A = manifold.proj(jnp.array([2.0, 0.3, -0.4, 0.5], dtype=F64), c)

    boosted_A = manifold.lorentz_boost(mu_A, c) @ mu_A

    assert np.allclose(np.asarray(boosted_A), np.asarray(manifold.create_origin(c, 3)), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", CURVATURES)
def test_lorentz_boost_inverse_is_the_boost_of_the_time_reflected_point(c: float):
    """The inverse boost is the boost of ``[mu_t, -mu_s]``: B(-v) @ B(v) == I, no ``proj``.

    This is the documented way to undo a HoroPCA centering; asserting it on the bare product keeps
    a projection from papering over an inverse that is only approximately right.
    """
    manifold = Hyperboloid(dtype=F64)
    mu_A = manifold.proj(jnp.array([2.0, 0.3, -0.4, 0.5], dtype=F64), c)
    mu_reflected_A = jnp.concatenate([mu_A[:1], -mu_A[1:]])

    boost_AA = manifold.lorentz_boost(mu_A, c)
    inverse_AA = manifold.lorentz_boost(mu_reflected_A, c)

    assert np.allclose(np.asarray(inverse_AA @ boost_AA), np.eye(4), rtol=1e-10, atol=1e-12)
    assert np.allclose(np.asarray(boost_AA @ inverse_AA), np.eye(4), rtol=1e-10, atol=1e-12)


def test_lorentz_boost_of_the_origin_is_the_identity():
    """The ``gamma -> 1`` limit is benign: boosting by the origin does nothing."""
    manifold = Hyperboloid(dtype=F64)
    c = 1.3

    boost_AA = manifold.lorentz_boost(manifold.create_origin(c, 3), c)

    assert np.allclose(np.asarray(boost_AA), np.eye(4), rtol=1e-12, atol=1e-14)


# =============================================================================================
# M1-11 — protocol conformance sweep
# =============================================================================================


@pytest.mark.parametrize(
    "manifold",
    [
        Euclidean(),
        Poincare(),
        Hyperboloid(),
        ProperVelocity(),
        Stereographic(),
        ProductManifold((Poincare(), 2), (Hyperboloid(), 3)),
    ],
    ids=["Euclidean", "Poincare", "Hyperboloid", "ProperVelocity", "Stereographic", "Product"],
)
def test_every_manifold_satisfies_the_manifold_protocol(manifold):
    """Every shipped manifold is a structural ``Manifold``.

    ``Manifold`` is the public type hint for "any manifold" across the layer and optimizer APIs, so
    a method renamed or dropped on one model silently narrows what those APIs accept.
    """
    assert isinstance(manifold, Manifold)


def test_manifold_protocol_rejects_a_non_manifold():
    """The protocol is not vacuous: an arbitrary object is not a ``Manifold``.

    ``runtime_checkable`` protocols only check method *presence*, so without this the sweep above
    would pass even if ``Manifold`` had been emptied of its members.
    """

    class NotAManifold:
        dtype = jnp.float32

    assert not isinstance(NotAManifold(), Manifold)
    assert not isinstance(object(), Manifold)


# =============================================================================================
# M1-12 — isometry-mapping gradients at degenerate inputs
#
# The model-conversion maps sit between every hybrid layer and its manifold. Each one divides or
# takes a square root at the origin or at the ball boundary, i.e. exactly where a naive
# implementation produces a NaN cotangent that no forward-value test can see.
# =============================================================================================

_DEGENERATE_CASES = [
    ("pv_to_poincare", iso.pv_to_poincare, jnp.zeros(3, dtype=F64)),
    ("poincare_to_pv", iso.poincare_to_pv, jnp.zeros(3, dtype=F64)),
    ("pv_to_hyperboloid", iso.pv_to_hyperboloid, jnp.zeros(3, dtype=F64)),
    ("hyperboloid_to_pv", iso.hyperboloid_to_pv, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=F64)),
    ("poincare_to_hyperboloid", iso.poincare_to_hyperboloid, jnp.zeros(3, dtype=F64)),
    ("hyperboloid_to_poincare", iso.hyperboloid_to_poincare, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=F64)),
]


@pytest.mark.parametrize(("name", "fn", "x_at_origin"), _DEGENERATE_CASES, ids=[c[0] for c in _DEGENERATE_CASES])
def test_isometry_gradients_are_finite_at_the_origin(name: str, fn, x_at_origin):
    """Every model conversion has a finite, non-zero gradient at its own origin."""
    grad = jax.grad(lambda v: jnp.sum(fn(v, 1.0)))(x_at_origin)

    assert bool(jnp.all(jnp.isfinite(grad))), f"{name} has a non-finite gradient at the origin"
    assert float(jnp.sum(jnp.abs(grad))) > 0.0, f"{name} gradient collapsed to zero at the origin"


def test_isometry_gradients_are_finite_at_the_poincare_boundary():
    """``poincare_to_pv`` blows up in value at ‖y‖ -> 1/√c but must keep a finite gradient.

    The ``MIN_NORM`` floor is what makes this hold; removing it turns the boundary cotangent into
    a NaN that propagates through any hybrid model touching the ball edge.
    """
    near_boundary_D = jnp.array([1.0 - 1e-12, 0.0], dtype=F64)

    grad = jax.grad(lambda v: jnp.sum(iso.poincare_to_pv(v, 1.0)))(near_boundary_D)

    assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.parametrize("c", CURVATURES)
def test_pv_to_poincare_jacobian_matches_central_differences(c: float):
    """Autodiff Jacobian of ``pv_to_poincare`` vs a central-difference estimate.

    Anchors the derivative itself, not just its finiteness: a mapping whose forward pass is right
    but whose ``jnp.sqrt`` guard silently changes the local slope would pass every round-trip test
    in test_isometry_mappings.py and fail here.
    """
    x_D = jnp.array([0.4, -0.7, 0.25], dtype=F64)

    jacobian = np.asarray(jax.jacfwd(lambda v: iso.pv_to_poincare(v, c))(x_D))

    eps = 1e-6
    numeric = np.zeros_like(jacobian)
    for j in range(x_D.shape[0]):
        step = np.zeros(x_D.shape[0])
        step[j] = eps
        plus = np.asarray(iso.pv_to_poincare(x_D + jnp.asarray(step), c))
        minus = np.asarray(iso.pv_to_poincare(x_D - jnp.asarray(step), c))
        numeric[:, j] = (plus - minus) / (2.0 * eps)

    assert np.allclose(jacobian, numeric, rtol=1e-6, atol=1e-9)
    # Non-trivial map: the Jacobian is not a scaled identity at a generic off-axis point.
    off_diagonal = jacobian - np.diag(np.diag(jacobian))
    assert np.max(np.abs(off_diagonal)) > 1e-3


# =============================================================================================
# M1-15 — Hyperboloid ambient-chart stability: 250-digit Decimal oracles for dist / logmap
#
# Every ambient formula built on ``⟨x, y⟩_L = -x₀y₀ + ⟨x_s, y_s⟩`` subtracts two positive numbers
# of size ``e^(a+b)/(4c)`` to obtain ``cosh(θ)/c``, where ``a = √c·d₀(x)``, ``b = √c·d₀(y)`` and
# ``θ = √c·d(x, y)``. The surviving significand shrinks by ``e^(a + b - θ)`` — twice the Gromov
# product — so all precision is gone past ``ln(1/eps)``: 15.9 in float32, 36.0 in float64.
# The pre-fix ``dist`` returned 0.0015 for a true distance of 1.0 at radius 10 in float32, and
# ``logmap`` returned NaN from radius ~10 (f32) / ~20 (f64).
#
# Unlike the rest of this file these oracles run in BOTH dtypes: the whole point is how the error
# grows with the radius, and that is dtype-specific.
#
# The reference is a pure-stdlib ``decimal`` computation of ``d = arcosh(-c⟨x, y⟩_L)/√c`` and
# ``log_x(y) = (θ/sinh θ)·(y - cosh(θ)·x)`` at a precision chosen so the cancellation cannot
# reach the result. Three construction rules matter, and violating any of them measures input
# rounding instead of algorithm error:
#
#   1. the SPATIAL parts are float arrays (in the dtype under test) and the time coordinate is
#      derived from them IN Decimal, so the oracle's point is exactly on the manifold;
#   2. the shared direction ``e1`` has all components equal, so two points on the same ray
#      normalize to the same unit vector and the angle between them is exactly 0;
#   3. tolerances carry an explicit input-resolution term (see ``_hyperboloid_input_resolution``)
#      — one ulp of an ambient coordinate is worth a finite amount of geodesic length, and no
#      implementation reading only the ambient coordinates can beat it.
# =============================================================================================

_HYP_RADII = (0.01, 3.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0)
"""``√c·d₀`` of the first point. 80 is the last radius whose ``x₀ = cosh(80)`` fits float32."""

_HYP_CONFIGS = (
    # ("radial", Δ): second point further out along the SAME ray, √c·d = Δ.
    ("radial", 1e-6),
    ("radial", 1e-3),
    ("radial", 1.0),
    ("radial", 10.0),
    # ("angular", ψ): second point at the SAME radius, angle ψ at the origin.
    ("angular", 1e-3),
    ("angular", 0.1),
    ("angular", 1.0),
    ("angular", np.pi),
    # both at once
    ("generic", 0.5),
)


class _HyperboloidCase(NamedTuple):
    """One (x, y) pair plus its Decimal-oracle truth. See ``_hyperboloid_case``."""

    x_A: jnp.ndarray
    y_A: jnp.ndarray
    d_ref: float
    u_ref_A: np.ndarray  # log_x(y), Decimal oracle cast to float64
    resolution: float  # absolute geodesic uncertainty implied by one ulp of the inputs
    label: str


def _hyperboloid_basis(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal spatial directions; ``e1`` has all components equal.

    That equality is load-bearing for the ``radial`` configurations: ``s1·e1`` and ``s2·e1`` then
    normalize to the *same* float unit vector for any positive scalars ``s1``, ``s2``, so the angle between the two points is
    exactly 0 and the pair is genuinely collinear in the dtype under test. With a generic direction
    the two normalizations differ by ~eps, which the angular term amplifies by
    ``sqrt(r_x r_y) ~ e^((a+b)/2)/(2 sqrt(c))``; measured on CPU, that overtakes the true radial
    separation from radius **≈16 in float32** (a generic ray already fails at ``a = 20``), not
    at the radius 80 this docstring used to claim. The crossover sits at ``ln(1/eps)``, so float64
    keeps the same construction usable to radius ~36. Either way it is an artifact of the
    construction, not of the library: the float32 inputs do not contain the angular information
    the assertion needs.
    """
    e1 = np.ones(dim) / np.sqrt(dim)
    e2 = np.zeros(dim)
    e2[0], e2[1] = 1.0, -1.0
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _hyperboloid_decimal_oracle(x_s: np.ndarray, y_s: np.ndarray, c: float, prec: int):
    """``(d, log_x(y))`` in ``prec``-digit decimal arithmetic, from the spatial parts alone.

    The time coordinates are derived here as ``x₀ = sqrt(1/c + Σx_s²)``, so the oracle's points are
    exactly on the manifold. ``arg = c·(x₀y₀ - ⟨x_s, y_s⟩) = -c⟨x, y⟩_L = cosh θ`` (note the sign),
    ``θ = ln(arg + sqrt(arg² - 1))``, ``d = θ/√c``, and ``log_x(y) = (θ/sinh θ)·(y - arg·x)``.
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec
        c_d = decimal.Decimal(float(c))
        xs = [decimal.Decimal(float(v)) for v in x_s]
        ys = [decimal.Decimal(float(v)) for v in y_s]
        x0 = (1 / c_d + sum(v * v for v in xs)).sqrt()
        y0 = (1 / c_d + sum(v * v for v in ys)).sqrt()
        arg = c_d * (x0 * y0 - sum(a * b for a, b in zip(xs, ys, strict=True)))
        theta = (arg + (arg * arg - 1).sqrt()).ln() if arg > 1 else decimal.Decimal(0)
        d = theta / c_d.sqrt()
        if theta == 0:
            u = [decimal.Decimal(0)] * (len(xs) + 1)
        else:
            factor = theta / ((theta.exp() - (-theta).exp()) / 2)
            u = [factor * (y0 - arg * x0)] + [factor * (b - arg * a) for a, b in zip(xs, ys, strict=True)]
        return d, u, x0, y0


def _hyperboloid_input_resolution(a: float, b: float, c: float, d_ref: float, eps: float) -> float:
    """Absolute geodesic uncertainty of ``d(x, y)`` implied by one ulp of the ambient inputs.

    Two independent legs, both first-order perturbations of the haversine decomposition
    ``sinh(θ/2) = hypot(P, q)``:

    * **angular** — the unit directions ``x̂_s``, ``ŷ_s`` are only defined to ~eps per component, so
      the chord ``‖x̂_s - ŷ_s‖`` carries an absolute error ~eps and ``q = ½√c·√(r_x·r_y)·chord``
      carries ``q_eps = ½√c·√(r_x·r_y)·eps``. Propagating through ``S² = P² + q²`` gives
      ``(2·q·q_eps + q_eps²)/(√c·C·S)``. The quadratic term is what dominates for nearly collinear
      pairs at large radius, where the *fake* eps-angle can exceed the true one.
    * **radial** — one ulp of the time coordinate ``x₀`` perturbs ``P = sinh((a - b)/2)`` by ~eps/2,
      worth ``eps/(√c·C)`` of geodesic length.

    This is a property of the ambient chart, not of any implementation: at radius ``a`` an ambient
    coordinate has an ulp of ``eps·e^a``, and no function of ``(x, y)`` alone can resolve two points
    closer together than that. Cases where the bound exceeds 10% of the true distance are simply not
    representable and are skipped.
    """
    sqrt_c = np.sqrt(c)
    half = sqrt_c * d_ref / 2.0
    if half > 350.0 or d_ref == 0.0:
        return np.inf
    sinh_half, cosh_half = np.sinh(half), np.cosh(half)
    p = np.sinh((a - b) / 2.0)
    q = np.sqrt(max(sinh_half**2 - p * p, 0.0))
    r_x, r_y = np.sinh(a) / sqrt_c, np.sinh(b) / sqrt_c
    q_eps = 0.5 * sqrt_c * np.sqrt(r_x * r_y) * eps
    angular = (2.0 * q * q_eps + q_eps**2) / (sqrt_c * cosh_half * sinh_half)
    radial = eps / (sqrt_c * cosh_half)
    return float(angular + radial)


def _hyperboloid_case(a: float, kind: str, param: float, c: float, dim: int, dtype) -> _HyperboloidCase | None:
    """Build one oracle case, or ``None`` if it is not representable in ``dtype``."""
    if kind == "radial":
        b, psi = a + param, 0.0
    elif kind == "angular":
        b, psi = a, param
    else:
        b, psi = a + 1.0, param
    if max(a, b) > 88.0:  # cosh(88) overflows float32
        return None

    e1, e2 = _hyperboloid_basis(dim)
    sqrt_c = np.sqrt(c)
    x_s = np.asarray((np.sinh(a) / sqrt_c) * e1, dtype=dtype)
    direction = e1 if psi == 0.0 else np.cos(psi) * e1 + np.sin(psi) * e2
    y_s = np.asarray((np.sinh(b) / sqrt_c) * direction, dtype=dtype)
    if not (np.all(np.isfinite(x_s)) and np.all(np.isfinite(y_s))):
        return None

    prec = int(4.0 * max(a, b) / np.log(10.0)) + 40
    d_dec, u_dec, x0, y0 = _hyperboloid_decimal_oracle(x_s, y_s, c, prec)
    d_ref = float(d_dec)
    if d_ref == 0.0:
        return None
    x_A = np.concatenate([[float(x0)], x_s]).astype(dtype)
    y_A = np.concatenate([[float(y0)], y_s]).astype(dtype)
    u_ref_A = np.array([float(v) for v in u_dec])
    if not (np.all(np.isfinite(x_A)) and np.all(np.isfinite(y_A)) and np.all(np.isfinite(u_ref_A))):
        return None

    eps = float(np.finfo(dtype).eps)
    resolution = _hyperboloid_input_resolution(a, b, c, d_ref, eps)
    label = f"c={c} dim={dim} a={a} {kind}({param:g}) d={d_ref:.6g}"
    return _HyperboloidCase(jnp.asarray(x_A), jnp.asarray(y_A), d_ref, u_ref_A, resolution, label)


def _hyperboloid_cases(c: float, dtype):
    """Every representable, resolvable case of the (radius x configuration x dim) grid."""
    for dim in (2, 10):
        for a in _HYP_RADII:
            for kind, param in _HYP_CONFIGS:
                case = _hyperboloid_case(a, kind, param, c, dim, dtype)
                if case is None or case.resolution > 0.1 * case.d_ref:
                    continue
                yield case


_HYP_DTYPES = [(jnp.float32, 2e-6), (jnp.float64, 1e-13)]
_HYP_DTYPE_IDS = ["f32", "f64"]


@pytest.mark.parametrize(("dtype", "rtol"), _HYP_DTYPES, ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_dist_matches_the_decimal_oracle(dtype, rtol: float, c: float):
    """``dist`` vs a 250-digit reference across radii 0.01 .. 80 and separations 1e-6 .. 2·radius.

    Measured worst case over the grid: 19% of the tolerance below (float64) and 14% (float32), i.e.
    a factor 5-7 of margin (was 30%/46% before the radial gap term was rewritten to subtract only
    the spatial radii). The pre-fix ``acosh(-c⟨x, y⟩_L)`` arm is wrong by 100% on most of this
    grid past radius 10 — it returns 0 wherever the clipped argument rounds to exactly 1.
    """
    manifold = Hyperboloid(dtype=dtype)
    n_checked = 0
    for case in _hyperboloid_cases(c, dtype):
        n_checked += 1
        d = float(manifold.dist(case.x_A, case.y_A, c))
        tol = rtol * case.d_ref + 4.0 * case.resolution
        assert abs(d - case.d_ref) <= tol, f"{case.label}: {d} vs {case.d_ref} (tol {tol})"
    assert n_checked >= 40, f"grid collapsed to {n_checked} cases"


@pytest.mark.parametrize(("dtype", "rtol"), _HYP_DTYPES, ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_logmap_matches_the_decimal_oracle(dtype, rtol: float, c: float):
    """``logmap`` vs the same reference, scaled by ``‖u_ref‖∞``.

    The input-resolution term is weighted 8x more heavily than for ``dist``: the tangent vector is
    ``d·(cos φ·e_rad + sin φ·e_ang)``, and near a purely radial geodesic ``cos φ ≈ ±1``, so a
    *relative* perturbation of ``sinh(θ/2)`` lands on ``cos φ`` at full size while it reaches ``d``
    only after division by ``√c·C·d``. Measured worst case: 37% (float64) / 19% (float32) of tol.
    """
    manifold = Hyperboloid(dtype=dtype)
    for case in _hyperboloid_cases(c, dtype):
        u = np.asarray(manifold.logmap(case.y_A, case.x_A, c), dtype=np.float64)
        scale = float(np.max(np.abs(case.u_ref_A)))
        tol = (rtol + 32.0 * case.resolution / case.d_ref) * scale
        assert np.max(np.abs(u - case.u_ref_A)) <= tol, f"{case.label}: max|u - u_ref| too large"


@pytest.mark.parametrize(("dtype", "rtol"), _HYP_DTYPES, ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_logmap_is_tangent_relative_to_its_own_scale(dtype, rtol: float, c: float):
    """``|⟨u, x⟩_L| / (‖u‖∞·‖x‖∞)`` stays at the rounding floor — a RELATIVE tangency test.

    An absolute ``atol`` on ``⟨u, x⟩_L`` is meaningless here: at radius 30 both ``u`` and ``x`` have
    ambient components of order 1e13, so a perfectly tangent vector still shows an inner product of
    order 1e10. No correct implementation passes an absolute test.

    This also pins that :func:`_logmap` does *not* call ``_tangent_proj``: that helper routes
    through the cancelling ``_minkowski_inner`` and returns NaN here from radius ~10 (float32).
    """
    manifold = Hyperboloid(dtype=dtype)
    tang_rtol = 3e-7 if dtype is jnp.float32 else 5e-15
    for case in _hyperboloid_cases(c, dtype):
        u = np.asarray(manifold.logmap(case.y_A, case.x_A, c), dtype=np.float64)
        x = np.asarray(case.x_A, dtype=np.float64)
        assert np.all(np.isfinite(u)), f"{case.label}: logmap is not finite"
        inner = -u[0] * x[0] + float(np.dot(u[1:], x[1:]))
        relative = abs(inner) / (np.max(np.abs(u)) * np.max(np.abs(x)))
        tol = tang_rtol + 8.0 * case.resolution / case.d_ref
        assert relative <= tol, f"{case.label}: relative tangency {relative:.3e} > {tol:.3e}"


@pytest.mark.parametrize(("dtype", "rtol"), _HYP_DTYPES, ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_tangent_norm_of_logmap_equals_dist(dtype, rtol: float, c: float):
    """``‖log_x(y)‖_x == d(x, y)`` wherever the ambient chart can still represent it.

    ``tangent_norm``'s own resolution limit is ``eps·√c·x₀`` *relative* — one power of ``x₀`` worse
    than ``dist``'s, because a tangent vector's ambient components are ``e^a`` times its Riemannian
    length, so the radial/angular split it needs is destroyed by input rounding first. Cases past
    ``eps·√c·x₀ > 0.05`` (radius ~28 in float64, ~13 in float32) are therefore skipped: they are
    not representable, not wrong. Within the representable range the measured worst case is 12%
    (float64) / 8.9% (float32) of the tolerance (was 30%/45% before the radial gap term was
    rewritten to subtract only the spatial radii).
    """
    manifold = Hyperboloid(dtype=dtype)
    eps = float(np.finfo(dtype).eps)
    n_checked = 0
    for case in _hyperboloid_cases(c, dtype):
        chart_limit = eps * np.sqrt(c) * float(case.x_A[0])
        if chart_limit > 0.05:
            continue
        n_checked += 1
        u = manifold.logmap(case.y_A, case.x_A, c)
        norm = float(manifold.tangent_norm(u, case.x_A, c))
        tol = rtol * case.d_ref + 4.0 * case.resolution + 4.0 * chart_limit * case.d_ref
        assert abs(norm - case.d_ref) <= tol, f"{case.label}: ‖log‖ = {norm} vs d = {case.d_ref}"
    assert n_checked >= 20, f"grid collapsed to {n_checked} cases"


@pytest.mark.parametrize(("dtype", "rtol"), _HYP_DTYPES, ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_sqdist_matches_the_oracle_closed_form(dtype, rtol: float, c: float):
    """``d_L² = (2/c)·(cosh(√c·d) - 1)`` with ``d`` from the Decimal oracle (Law et al. 2019).

    Restricted to ``√c·d < 30`` so the closed form itself does not overflow; the point is the
    *identity*, and the cancellation being tested lives at every scale.
    """
    manifold = Hyperboloid(dtype=dtype)
    for case in _hyperboloid_cases(c, dtype):
        theta = np.sqrt(c) * case.d_ref
        if theta > 30.0:
            continue
        # 4·sinh²(θ/2)/c, not (2/c)(cosh θ - 1): the latter cancels for small θ (at θ = 1e-6 it
        # loses 12 digits) and would measure the reference's rounding, not the library's.
        expected = 4.0 * np.sinh(theta / 2.0) ** 2 / c
        got = float(manifold.sqdist(case.x_A, case.y_A, c))
        # d_L² = 4·S²/c, so its relative error is twice that of S = sinh(θ/2). Converting the
        # distance tolerance into an S tolerance costs a factor C/S = coth(θ/2): the same input
        # resolution buys a much tighter distance than it does a squared Lorentzian distance.
        sinh_half, cosh_half = np.sinh(theta / 2.0), np.cosh(theta / 2.0)
        rel = (rtol * case.d_ref + 4.0 * case.resolution) * np.sqrt(c) * cosh_half / sinh_half
        rel_floor = 16.0 * float(np.finfo(dtype).eps)  # plain rounding of the squaring itself
        assert got == pytest.approx(expected, rel=4.0 * rel + rel_floor, abs=1e-300)


@pytest.mark.parametrize(("dtype", "rtol"), _HYP_DTYPES, ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_dist_is_bitwise_symmetric(dtype, rtol: float, c: float):
    """``d(x, y) == d(y, x)`` to the bit, on every grid case.

    The haversine form is symmetric by construction only if the two arguments enter symmetrically:
    ``P`` flips sign (and ``S = |hypot(P, q)|`` does not), ``chord`` and ``√r_x·√r_y`` are
    symmetric. A one-sided floor or a ``x₀/r_x`` factor leaking into the distance would break it.
    """
    manifold = Hyperboloid(dtype=dtype)
    for case in _hyperboloid_cases(c, dtype):
        forward = np.asarray(manifold.dist(case.x_A, case.y_A, c))
        backward = np.asarray(manifold.dist(case.y_A, case.x_A, c))
        assert np.array_equal(forward, backward), case.label


# ---------------------------------------------------------------------------------------------
# Configurations with an analytic answer — no oracle needed, so nothing can be circular.
# ---------------------------------------------------------------------------------------------


def _hyperboloid_point(a: float, c: float, direction: np.ndarray, dtype) -> jnp.ndarray:
    """Point at geodesic radius ``a/√c`` along the unit spatial ``direction``."""
    sqrt_c = np.sqrt(c)
    return jnp.asarray(np.concatenate([[np.cosh(a) / sqrt_c], (np.sinh(a) / sqrt_c) * direction]).astype(dtype))


@pytest.mark.parametrize(("dtype", "rtol"), _HYP_DTYPES, ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("a", [0.5, 5.0, 20.0, 45.0])
def test_hyperboloid_dist_on_a_shared_ray_is_the_radius_difference(dtype, rtol: float, c: float, a: float):
    """ψ = 0: ``d = |a - b|/√c`` exactly, at every radius. ψ = π: ``d = (a + b)/√c``.

    Both are one-dimensional facts about the geodesic through the origin, so they need no oracle at
    all — and they are precisely the configurations the cancelling ``acosh`` form gets worst, since
    the Gromov product ``a + b - θ`` equals ``2·min(a, b)`` on a shared ray.

    The float32 legs stop at ``a = 5``: past radius ~16 float32 cannot resolve collinearity in the
    ambient chart on *any* backend (see :func:`_hyperboloid_basis`), and on XLA:GPU the float32
    divide is only faithfully rounded (<=2 ulp, not correctly rounded), so ``_polar_frame``'s two
    unit directions come back 1 ulp apart and the angular term multiplies that by
    ``sqrt(r_x r_y) ~ e^((a+b)/2)/(2 sqrt(c))``. The large-radius legs stay in float64, where the
    divide is exact.
    """
    if dtype == jnp.float32 and a > 5.0:
        pytest.skip("float32 cannot resolve collinearity past radius ~16 in the ambient chart")
    e1, _ = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=dtype)
    b = a + 1.5
    same_ray = float(manifold.dist(_hyperboloid_point(a, c, e1, dtype), _hyperboloid_point(b, c, e1, dtype), c))
    assert same_ray == pytest.approx(1.5 / np.sqrt(c), rel=max(rtol, 1e-6))

    opposite = float(manifold.dist(_hyperboloid_point(a, c, e1, dtype), _hyperboloid_point(b, c, -e1, dtype), c))
    assert opposite == pytest.approx((a + b) / np.sqrt(c), rel=max(rtol, 1e-6))


@pytest.mark.parametrize(("dtype", "rtol"), _HYP_DTYPES, ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("a", [1e-4, 1e-3, 1e-2, 0.5, 5.0, 20.0, 45.0])
def test_hyperboloid_dist_and_logmap_to_the_origin_match_the_origin_variants(dtype, rtol: float, c: float, a: float):
    """``dist(x, origin) == dist_0(x)`` and ``logmap(origin, x) == d·e_rad``.

    ``dist_0``/``logmap_0`` read the geodesic radius off the spatial part, which resolves it at
    every radius, so they are an independent reference for the general two-point code path. The log
    map to the origin must be exactly the inward unit radial direction scaled by the distance.

    The ``20·eps/a`` slack on the pairwise comparison (a no-op for ``a >~ 1``; ``dist_0`` is always
    held to the full tolerance against the analytic ``a/√c``) dates from when ``_polar_frame``
    formed ``u_x - u_y = (x₀ + r_x) - (y₀ + r_y)`` directly, losing ``eps·x₀`` and leaving an
    O(eps/a) relative error — 2.5e-4 at ``a = 1e-4`` in float32 and 5.4e-12 in float64, against a
    ``dist_0`` exact to 4.1e-8 / 0 there. With the gap read off the spatial radii the same cells
    measure 7.4e-8 (float32) and 7.9e-12 (float64), and the worst cell on the whole ``a`` grid is
    2.1e-7 (float32). What is left in float64 is no longer the gap but the ``MIN_NORM`` floor on
    ``r_y`` at the origin, worth ``½·MIN_NORM/a`` relative — 5e-12 at ``a = 1e-4``, which is why
    the float64 column does not fall to zero. The slack is kept as it is; it is now several orders
    wider than what either arm needs.
    """
    e1, _ = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=dtype)
    x_A = _hyperboloid_point(a, c, e1, dtype)
    origin_A = manifold.create_origin(c, 4)

    tight = max(rtol, 1e-6)
    pairwise_rel = max(tight, 20.0 * float(jnp.finfo(dtype).eps) / a)

    d = float(manifold.dist(x_A, origin_A, c))
    d_0 = float(manifold.dist_0(x_A, c))
    assert d_0 == pytest.approx(a / np.sqrt(c), rel=tight)
    assert d == pytest.approx(d_0, rel=pairwise_rel)
    assert d == pytest.approx(a / np.sqrt(c), rel=pairwise_rel)

    sqrt_c = np.sqrt(c)
    e_rad_A = -sqrt_c * np.concatenate([[np.sinh(a) / sqrt_c], (np.cosh(a) / sqrt_c) * e1])
    u = np.asarray(manifold.logmap(origin_A, x_A, c), dtype=np.float64)
    assert np.allclose(u, d * e_rad_A, rtol=max(rtol, 1e-6) * 10.0, atol=0.0)

    # ...and the reverse direction falls back to logmap_0, which is exact at the origin.
    u0 = np.asarray(manifold.logmap(x_A, origin_A, c), dtype=np.float64)
    assert np.allclose(u0, np.asarray(manifold.logmap_0(x_A, c), dtype=np.float64), rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------------------------
# dist_0 / logmap_0 read the radius off the spatial part (WS-A follow-up to the pairwise fix).
#
# The ``acosh(√c·x₀)`` arm these replaced could not resolve a small radius at all: ``acosh``'s
# ``1 + 10·eps`` domain clamp flattened every float32 radius below ``sqrt(20·eps) = 1.5e-3`` onto
# that single value (54% error at 1e-3, 1500x at 1e-6), and above it ``x₀ = cosh(√c·d)/√c`` still
# only stores ``d`` to ``sqrt(eps)`` resolution.
# ---------------------------------------------------------------------------------------------

# Spatial radii spanning the formerly-unrepresentable regime (1e-8 … 1e-3) and the working range.
# The geodesic radius of such a point is ``arcsinh(√c·r)/√c ≤ 4.9/√c`` even at r = 40, so nothing
# here approaches a float32 overflow for the curvatures under test.
_DIST_0_RADII = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0, 20.0, 40.0]
# Tighter than ``_HYP_DTYPES``: the arcsinh arm is accurate to ~1e-7 (float32) / ~3e-16 (float64)
# at every radius above, so there is no reason to allow the two-point arm's slack here.
_DIST_0_RTOL = {jnp.float32: 1e-6, jnp.float64: 1e-14}


def _hyperboloid_point_from_spatial(x_s: np.ndarray, c: float, dtype) -> jnp.ndarray:
    """Ambient point with the given spatial part; ``x₀`` completed from the constraint."""
    x0 = np.sqrt(1.0 / c + float(np.dot(x_s, x_s)))
    return jnp.asarray(np.concatenate([[x0], x_s]).astype(dtype))


def _hyperboloid_point_at_spatial_radius(r: float, c: float, direction: np.ndarray, dtype) -> jnp.ndarray:
    """Point with spatial part ``r·direction``; ``x₀`` completed from the constraint.

    Its geodesic radius is ``arcsinh(√c·r)/√c`` — the value ``dist_0`` must return. Built from the
    spatial part rather than from ``cosh(a)``/``sinh(a)`` so the small radius survives the cast: at
    ``r = 1e-6`` the ``cosh`` time coordinate is 1.0 to the last float32 bit, which is exactly the
    information loss under test.
    """
    return _hyperboloid_point_from_spatial(r * direction, c, dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_dist_0_matches_the_arcsinh_closed_form_at_every_radius(dtype, c: float):
    """``dist_0(x) == arcsinh(√c‖x_s‖)/√c`` (slot 0) and its floored twin (slot 1).

    The identity is exact on the sheet, so this is an oracle rather than a cross-check: slot 1's
    own closed form is ``arcsinh(hypot(√c‖x_s‖, 20·eps))/√c``, which is the same number to well
    inside the tolerance once the radius clears the floor.
    """
    manifold = Hyperboloid(dtype=dtype)
    sqrt_c = np.sqrt(c)
    eps = float(jnp.finfo(dtype).eps)
    rtol = _DIST_0_RTOL[dtype]
    rng = np.random.default_rng(11)

    for r in _DIST_0_RADII:
        direction = rng.normal(size=8)
        direction /= np.linalg.norm(direction)
        x_A = _hyperboloid_point_at_spatial_radius(r, c, direction, dtype)

        expected = np.arcsinh(sqrt_c * r) / sqrt_c
        assert float(manifold.dist_0(x_A, c, version_idx=0)) == pytest.approx(expected, rel=rtol), f"r={r}"

        expected_floored = np.arcsinh(np.hypot(sqrt_c * r, 20.0 * eps)) / sqrt_c
        assert float(manifold.dist_0(x_A, c, version_idx=1)) == pytest.approx(expected_floored, rel=rtol), f"r={r}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_logmap_0_is_the_radius_along_the_input_direction(dtype, c: float):
    """``log_0(x) = [0, d₀(x)·x̂_s]``: time slot exactly 0, direction preserved, norm ``= dist_0``."""
    manifold = Hyperboloid(dtype=dtype)
    sqrt_c = np.sqrt(c)
    rtol = _DIST_0_RTOL[dtype]
    rng = np.random.default_rng(12)

    for r in _DIST_0_RADII:
        direction = rng.normal(size=8)
        direction /= np.linalg.norm(direction)
        x_A = _hyperboloid_point_at_spatial_radius(r, c, direction, dtype)

        v_A = np.asarray(manifold.logmap_0(x_A, c), dtype=np.float64)
        assert v_A[0] == 0.0, f"r={r}: log_0 must be tangent at the origin, so its time slot is 0"

        norm = float(np.linalg.norm(v_A[1:]))
        assert norm == pytest.approx(np.arcsinh(sqrt_c * r) / sqrt_c, rel=rtol), f"r={r}"
        assert np.allclose(v_A[1:] / norm, direction, rtol=rtol, atol=rtol), f"r={r}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
def test_hyperboloid_logmap_0_inverts_expmap_0_across_the_paper_radius_grid(dtype):
    """``log_0(exp_0(v)) == v`` at every radius on the paper grid, dim 8.

    Pre-fix float32 medians were 1.5e3 at ``r = 1e-6`` and 5.4e-1 at ``r = 1e-3`` — the round trip
    did not merely lose precision, it returned a different point. The float64 tolerance is pinned
    two orders below the measured median (1.4e-16) rather than at the plan's 1e-13.
    """
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    rtol = 1e-6 if dtype == jnp.float32 else 1e-14
    rng = np.random.default_rng(13)

    for r in (1e-3, 1e-2, 0.1, 1.0, 5.0, 10.0, 20.0):
        directions_ND = rng.normal(size=(64, 8))
        directions_ND /= np.linalg.norm(directions_ND, axis=-1, keepdims=True)
        v_NA = jnp.asarray(np.concatenate([np.zeros((64, 1)), r * directions_ND], axis=-1).astype(dtype))

        points_NA = jax.vmap(manifold.expmap_0, in_axes=(0, None))(v_NA, c)
        back_NA = jax.vmap(manifold.logmap_0, in_axes=(0, None))(points_NA, c)

        residual_N = np.linalg.norm(np.asarray(back_NA - v_NA, dtype=np.float64), axis=-1) / r
        assert float(np.median(residual_N)) < rtol, f"r={r}: median relative round-trip error"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("a", [5.0, 30.0])
def test_hyperboloid_dist_is_exactly_zero_with_an_exactly_zero_gradient_at_coincidence(dtype, a: float):
    """``d(x, x) == 0`` and ``∇d(x, x) == 0`` bit for bit, at any radius.

    The haversine form gets this for free — ``P`` and ``chord`` are both exactly 0 — so the
    ``where(x == y, 0.0, ...)`` guard the ``acosh`` arm needs is gone. That guard only ever fixed
    the *forward* value; the gradient still had to survive ``acosh'`` near its pole.
    """
    e1, _ = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=dtype)
    c = 1.3
    x_A = _hyperboloid_point(a, c, e1, dtype)

    assert float(manifold.dist(x_A, x_A, c)) == 0.0
    assert float(manifold.sqdist(x_A, x_A, c)) == 0.0
    grad = np.asarray(jax.grad(lambda p: manifold.dist(p, x_A, c))(x_A))
    assert np.all(grad == 0.0), grad


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_dist_gradient_at_the_origin_is_the_inward_unit_direction(dtype, c: float):
    """At ``x`` = origin, ``∇_{x_s} d(x, y) = -ŷ_s`` with ``|∇| = 1``.

    This is the test that pins the ``MIN_NORM`` *floor* on the spatial radius (rather than an
    exact-zero ``where``): at the origin ``q ∝ √MIN_NORM`` while ``∂chord/∂x_s ∝ 1/MIN_NORM``, and
    ``∂S/∂q = q/S`` restores the last ``√MIN_NORM``. The floors cancel exactly and the gradient is
    floor-independent. An exact-zero guard would return 0 here and freeze every parameter
    initialized at the origin.
    """
    e1, _ = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=dtype)
    origin_A = manifold.create_origin(c, 4)
    y_A = _hyperboloid_point(2.0, c, e1, dtype)

    grad = np.asarray(jax.grad(lambda p: manifold.dist(p, y_A, c))(origin_A), dtype=np.float64)

    assert np.all(np.isfinite(grad))
    assert np.allclose(grad[1:], -e1, rtol=1e-5, atol=1e-6)
    assert float(np.linalg.norm(grad[1:])) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
def test_hyperboloid_gyro_bias_at_the_origin_has_a_nonzero_jacobian(dtype):
    """A zero-initialized gyro-bias must receive gradient. On 1.1.2 it received exactly none.

    ``x ⊕ exp_0([0, b])`` goes through ``_logmap_0``, whose ``dist_0`` carried a bitwise
    ``where(x == origin, 0, ...)`` guard. At ``b = 0`` the guard selected the constant branch, so
    ``∂(x ⊕ ·)/∂b`` was the **exactly zero matrix** in both dtypes (measured Frobenius norm 0.0,
    float32 and float64), freezing every origin-initialized hyperboloid gyro-bias at its init
    value — ``hyperboloid_linear.py``, ``hyperboloid_conv.py`` and both Busemann FC layers.
    With the radius read off the spatial part the guard is gone; the same Jacobian now has
    Frobenius norm 3.2806 for the point below.
    """
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    dim = 8
    rng = np.random.default_rng(0)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    x_A = _hyperboloid_point(1.0, c, direction, dtype)  # on-sheet, geodesic radius 1

    def biased(b_D):
        bias_pt_A = manifold.expmap_0(jnp.concatenate([jnp.zeros((1,), b_D.dtype), b_D]), c)
        return manifold.addition(x_A, bias_pt_A, c)

    jac_AD = np.asarray(jax.jacfwd(biased)(jnp.zeros((dim,), dtype=dtype)), dtype=np.float64)
    assert np.all(np.isfinite(jac_AD))
    assert float(np.linalg.norm(jac_AD)) > 0.0, "zero-initialized gyro-bias receives no gradient"

    # The companion guarantee: dist_0's own gradient at the origin must stay finite for both
    # non-legacy arms (safe_norm / safe_hypot give an exactly-zero, hence finite, VJP there).
    origin_A = manifold.create_origin(c, dim)
    for version_idx in (0, 1):
        g_A = jax.grad(lambda p, v=version_idx: manifold.dist_0(p, c, version_idx=v))(origin_A)
        assert bool(jnp.all(jnp.isfinite(g_A))), f"non-finite dist_0 gradient at the origin, slot {version_idx}"


# ---------------------------------------------------------------------------------------------
# The pairwise radial gap is built from the spatial radii too (A6 follow-up to the origin chart).
#
# ``_polar_frame`` needs ``u_x - u_y`` with ``u = x₀ + ‖x_s‖``. Forming that difference directly
# throws away ``eps·x₀``, which near the origin is the whole answer; the identity
# ``x₀ - y₀ = (r_x² - r_y²)/(x₀ + y₀)`` moves the subtraction onto the spatial radii, where it is
# Sterbenz-exact. The two tests below pin the near-origin regime the fix targets.
# ---------------------------------------------------------------------------------------------

_PAIRWISE_ORIGIN_RTOL = {jnp.float32: 1e-6, jnp.float64: 1e-13}


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("r", [1e-6, 1e-4, 1e-2])
def test_hyperboloid_pairwise_dist_to_the_origin_matches_dist_0_near_the_origin(dtype, c: float, r: float):
    """``dist(x, origin) == dist_0(x)`` inside the radius where the ambient chart stops resolving.

    ``dist_0`` reads ``arcsinh(√c‖x_s‖)/√c`` off the spatial part and is exact here, so it is an
    independent oracle for the two-point path. With ``u_x - u_y`` formed directly this fails in
    float32 by 4.6e-2 relative at ``r = 1e-6``, 1.4e-4 at 1e-4 and 9.8e-6 at 1e-2 — the ``eps·x₀``
    that the subtraction discards, divided by a gap of size ``r``. Reading the gap off the spatial
    radii brings all three to ≤1.9e-7 relative (measured over the three curvatures here; ≤3.1e-7
    over the wider probe grid).

    The point is built from its spatial part, never from ``cosh(a)``: at ``r = 1e-6`` the time
    coordinate is 1.0 to the last float32 bit, which is precisely the information the pairwise arm
    used to depend on.

    The tolerance carries an absolute ``MIN_NORM`` term, and it is the **float32** arm that this
    test discriminates. At ``y`` exactly at the origin ``r_y`` hits the ``MIN_NORM`` floor that
    :func:`_polar_frame` needs for its origin gradient, and the resulting angular term
    ``q = ½√c·√(r_x·MIN_NORM)`` adds ``½·MIN_NORM/r_x`` *relative* — a flat ``MIN_NORM/(2√c)`` ≈
    5e-16 of absolute distance, independent of the gap term and present before and after this fix.
    That floor sits above the float64 rtol at these radii but three to seven orders below the
    float32 one, so only the float32 assertions can fail on the gap term.
    """
    manifold = Hyperboloid(dtype=dtype)
    rng = np.random.default_rng(21)
    direction = rng.normal(size=8)
    direction /= np.linalg.norm(direction)
    x_A = _hyperboloid_point_at_spatial_radius(r, c, direction, dtype)
    origin_A = manifold.create_origin(c, 8)

    d = float(manifold.dist(x_A, origin_A, c))
    d_0 = float(manifold.dist_0(x_A, c))
    analytic = np.arcsinh(np.sqrt(c) * r) / np.sqrt(c)
    tol = _PAIRWISE_ORIGIN_RTOL[dtype] * d_0 + MIN_NORM
    assert abs(d - d_0) <= tol, f"dist(x, origin) = {d} vs dist_0 = {d_0} (tol {tol})"
    # ...and against the analytic value too, so this cannot pass by two matching wrong answers.
    assert abs(d - analytic) <= tol, f"dist(x, origin) = {d} vs arcsinh form {analytic} (tol {tol})"


@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_pairwise_dist_between_two_nearby_near_origin_points_is_float32_exact(c: float):
    """Two points at spatial radius 1e-3, a separation of ~1e-3 apart: float32 within 1e-6 of float64.

    Neither point is at the origin, so the ``MIN_NORM`` floor of the previous test is absent and the
    gap term is the only thing under test. The float64 evaluation of the *same* float32 inputs is
    the reference — it isolates the algorithm from the input rounding, which at this radius is
    ~1e-4 relative. Directly subtracting ``u_x - u_y`` puts the float32 answer 3.3e-5 off here;
    from the spatial radii it is 1.5e-7 (worst over the three curvatures).

    The float64 leg builds its own float64 points (so the float32 ``x₀`` cannot leak in) and goes
    against the 60-digit ``Decimal`` oracle, which is what makes this test non-vacuous in that
    dtype: the direct subtraction is off by up to 9.4e-14 relative there, against ≤4.4e-16 from the
    spatial radii.
    """
    rng = np.random.default_rng(22)
    d1 = rng.normal(size=8)
    d1 /= np.linalg.norm(d1)
    d2 = rng.normal(size=8)
    d2 /= np.linalg.norm(d2)

    r, sep = 1e-3, 1e-3
    x_s = r * d1
    y_s = x_s + sep * d2

    x_32 = _hyperboloid_point_from_spatial(x_s, c, jnp.float32)
    y_32 = _hyperboloid_point_from_spatial(y_s, c, jnp.float32)
    # Same numbers, evaluated in float64: any difference is the float32 arithmetic, not the input.
    x_64 = jnp.asarray(np.asarray(x_32, dtype=np.float64))
    y_64 = jnp.asarray(np.asarray(y_32, dtype=np.float64))

    got = float(Hyperboloid(dtype=jnp.float32).dist(x_32, y_32, c))
    reference = float(Hyperboloid(dtype=jnp.float64).dist(x_64, y_64, c))
    assert reference > 0.0
    assert abs(got - reference) / reference < 1e-6

    # And the float64 arm itself, on float64 points, against a reference that shares no arithmetic
    # with it: the Decimal oracle derives x₀ from the spatial part in 60-digit precision.
    x_f64 = _hyperboloid_point_from_spatial(x_s, c, jnp.float64)
    y_f64 = _hyperboloid_point_from_spatial(y_s, c, jnp.float64)
    got_64 = float(Hyperboloid(dtype=jnp.float64).dist(x_f64, y_f64, c))
    exact = float(_hyperboloid_decimal_oracle(np.asarray(x_f64[1:]), np.asarray(y_f64[1:]), c, 60)[0])
    assert abs(got_64 - exact) / exact < 1e-14


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("a", [2.0, 8.0, 10.0, 30.0, 45.0])
def test_hyperboloid_dist_and_logmap_derivatives_stay_finite_and_bounded(dtype, a: float):
    """``|∇_x d|∞ ≤ 1`` and a finite ``logmap`` Jacobian, out to radius 45 in both dtypes.

    The distance is 1-Lipschitz in the Riemannian metric; in the ambient chart the metric only
    contracts, so the ambient gradient can never exceed 1 in magnitude. A NaN or an exploding
    entry here is the signature of the old ``acosh``/``_minkowski_inner`` path, which returns NaN
    gradients from radius ~10 in float32.
    """
    e1, e2 = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    x_A = _hyperboloid_point(a, c, e1, dtype)
    y_A = _hyperboloid_point(a, c, np.cos(0.3) * e1 + np.sin(0.3) * e2, dtype)

    grad = np.asarray(jax.grad(lambda p: manifold.dist(p, y_A, c))(x_A), dtype=np.float64)
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad)) <= 1.0 + 1e-5

    jac = jax.jacobian(lambda p: manifold.logmap(y_A, p, c))(x_A)
    assert bool(jnp.all(jnp.isfinite(jac))), f"logmap Jacobian is not finite at radius {a}"
    jac_y = jax.jacobian(lambda p: manifold.logmap(p, x_A, c))(y_A)
    assert bool(jnp.all(jnp.isfinite(jac_y)))


@pytest.mark.parametrize("a", [1.0, 4.0, 8.0])
def test_hyperboloid_dist_gradient_agrees_between_float32_and_float64(a: float):
    """The float32 gradient tracks the float64 one where float32 can still represent the geometry."""
    e1, e2 = _hyperboloid_basis(4)
    c = 1.0
    y_dir = np.cos(0.7) * e1 + np.sin(0.7) * e2

    grads = {}
    for dtype in (jnp.float32, jnp.float64):
        manifold = Hyperboloid(dtype=dtype)
        x_A = _hyperboloid_point(a, c, e1, dtype)
        y_A = _hyperboloid_point(a + 0.5, c, y_dir, dtype)
        grads[dtype] = np.asarray(jax.grad(lambda p, m=manifold, y=y_A: m.dist(p, y, c))(x_A), dtype=np.float64)

    scale = np.max(np.abs(grads[jnp.float64]))
    assert np.max(np.abs(grads[jnp.float32] - grads[jnp.float64])) <= 2e-5 * scale


@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_stable_and_legacy_arms_agree_in_the_legacy_regime(c: float):
    """version_idx 0/1 (stable) == 2/3 (legacy acosh) while ``√c·d₀ < 2.5``.

    The cap is essential and one-sided: below it the cancellation in ``⟨x, y⟩_L`` has not yet eaten
    the significand, so the two arms must agree to near machine precision. Above it the *legacy*
    arm is the one drifting, which is the whole reason for the rewrite — comparing there would pin
    the bug rather than the fix.
    """
    e1, e2 = _hyperboloid_basis(5)
    manifold = Hyperboloid(dtype=jnp.float64)
    for a in (0.05, 0.8, 1.7, 2.4):
        for psi in (0.0, 0.3, 1.2, np.pi):
            for b in (a, min(a + 0.6, 2.5)):
                x_A = _hyperboloid_point(a, c, e1, jnp.float64)
                y_A = _hyperboloid_point(b, c, np.cos(psi) * e1 + np.sin(psi) * e2, jnp.float64)
                if a == b and psi == 0.0:
                    continue  # coincident: both arms return exactly 0
                stable = float(manifold.dist(x_A, y_A, c, version_idx=0))
                legacy = float(manifold.dist(x_A, y_A, c, version_idx=2))
                assert stable == pytest.approx(legacy, rel=1e-11), f"a={a} b={b} psi={psi}"
                # The smoothened pair agrees too, above their (different) positive floors.
                stable_s = float(manifold.dist(x_A, y_A, c, version_idx=1))
                assert stable_s == pytest.approx(stable, rel=1e-11, abs=1e-13)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
def test_hyperboloid_smoothened_arm_has_a_strictly_positive_floor(dtype):
    """version_idx 1 never returns 0, and equals the plain arm everywhere above its floor.

    The floor is ``2·arcsinh(10·eps)/√c``: 2.4e-6 in float32, 4.4e-15 in float64. It is applied in
    quadrature (``hypot(S, ε)``), not through ``smooth_clamp_min``, whose ``log(2)/β`` softplus
    remainder would shift *every* distance in the working range by ~0.028.
    """
    e1, e2 = _hyperboloid_basis(3)
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    x_A = _hyperboloid_point(2.0, c, e1, dtype)

    floor = float(manifold.dist(x_A, x_A, c, version_idx=1))
    expected_floor = 2.0 * np.arcsinh(10.0 * float(jnp.finfo(dtype).eps))
    assert floor > 0.0
    assert floor == pytest.approx(expected_floor, rel=1e-5)

    y_A = _hyperboloid_point(2.0, c, np.cos(0.4) * e1 + np.sin(0.4) * e2, dtype)
    assert float(manifold.dist(x_A, y_A, c, version_idx=1)) == pytest.approx(
        float(manifold.dist(x_A, y_A, c, version_idx=0)), rel=1e-6
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
def test_hyperboloid_dist_0_smoothened_arm_has_a_strictly_positive_floor(dtype, c: float):
    """``dist_0(origin, version_idx=1)`` equals ``arcsinh(20·eps)/√c``, and slot 0 is exactly 0.

    2.4e-6 (float32) / 4.4e-15 (float64) at ``c = 1`` — first-order the pairwise smoothened arm's
    ``2·arcsinh(10·eps)/√c``, so the two agree on what "never exactly zero" means. The legacy
    smoothened arm's ``smooth_clamp_min`` put the floor at 0.16632/√c instead, i.e. 80% error at a
    true radius of 0.1, in **both** dtypes.
    """
    manifold = Hyperboloid(dtype=dtype)
    origin_A = manifold.create_origin(c, 8)

    floor = float(manifold.dist_0(origin_A, c, version_idx=1))
    expected_floor = float(np.arcsinh(20.0 * float(jnp.finfo(dtype).eps)) / np.sqrt(c))
    assert floor > 0.0
    assert floor == pytest.approx(expected_floor, rel=1e-6)

    # The plain arm has a hard zero at the origin (safe_norm returns exactly 0 there).
    assert float(manifold.dist_0(origin_A, c, version_idx=0)) == 0.0


# Recorded on 1.1.2 (commit 370e9db) before the rewrite, at the on-sheet point with spatial radius
# 1e-3, c = 1.0, dim 8, direction e1 — i.e. a true geodesic radius of 9.99999833e-4. Slot 2 shows
# acosh's 1 + 10·eps domain clamp (float32 floor sqrt(20·eps) = 1.5441e-3, a 54% overstatement);
# slot 3 shows smooth_clamp_min's log(2)/50 remainder (floor acosh(1 + log(2)/50) = 0.16632).
_DIST_0_LEGACY_GOLDEN = {
    (jnp.float32, 2): 0.0015440807910636067,
    (jnp.float64, 2): 0.0009999998333921221,
    (jnp.float64, 3): 0.16632065504083338,
}
# Slot 3 in float32 is the one entry that is not bit-stable across XLA backends (the softplus in
# smooth_clamp_min differs in the last ulp: 0.16632071137 on CPU, 0.16632072628 on an A100), so it
# is pinned to the recorded value at one float32 ulp instead of bitwise.
_DIST_0_LEGACY_GOLDEN_F32_SMOOTHENED = 0.16632071137428284


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
def test_hyperboloid_dist_0_legacy_slots_reproduce_the_pre_fix_values(dtype):
    """Slots 2/3 must return the pre-fix numbers bit for bit — that is the whole point of keeping them.

    Slots 2/3 used to duplicate 0/1 for ``dist_0`` (unlike ``dist``, where they were already the
    legacy acosh arms). They now carry the acosh implementation, so a result computed before this
    change stays reproducible by naming ``VERSION_LEGACY``.
    """
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    e1 = np.zeros(8)
    e1[0] = 1.0
    x_A = _hyperboloid_point_at_spatial_radius(1e-3, c, e1, dtype)

    assert float(manifold.dist_0(x_A, c, version_idx=2)) == _DIST_0_LEGACY_GOLDEN[(dtype, 2)]
    if dtype == jnp.float64:
        assert float(manifold.dist_0(x_A, c, version_idx=3)) == _DIST_0_LEGACY_GOLDEN[(dtype, 3)]
    else:
        assert float(manifold.dist_0(x_A, c, version_idx=3)) == pytest.approx(_DIST_0_LEGACY_GOLDEN_F32_SMOOTHENED, rel=1e-6)

    # Well above both legacy floors the four slots describe the same geometry again.
    far_A = _hyperboloid_point_at_spatial_radius(5.0, c, e1, dtype)
    stable = float(manifold.dist_0(far_A, c, version_idx=0))
    assert stable == pytest.approx(np.arcsinh(5.0), rel=1e-6)
    for version_idx in (2, 3):
        assert float(manifold.dist_0(far_A, c, version_idx=version_idx)) == pytest.approx(stable, rel=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
def test_hyperboloid_dist_obeys_the_triangle_inequality_at_radius_30(dtype):
    """Random triples deep in the formerly-broken regime still satisfy ``d(x,z) ≤ d(x,y) + d(y,z)``.

    At radius 30 the pre-fix float32 ``dist`` returned 0 for most pairs, which passes the triangle
    inequality vacuously — so this is paired with a check that the distances are not degenerate.
    """
    rng = np.random.default_rng(0)
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    for _ in range(20):
        pts = []
        for _ in range(3):
            direction = rng.normal(size=4)
            direction /= np.linalg.norm(direction)
            pts.append(_hyperboloid_point(30.0, c, direction, dtype))
        d_xy = float(manifold.dist(pts[0], pts[1], c))
        d_yz = float(manifold.dist(pts[1], pts[2], c))
        d_xz = float(manifold.dist(pts[0], pts[2], c))
        assert d_xz <= d_xy + d_yz + 1e-4 * max(1.0, d_xz)
        assert d_xy > 1.0, "distances collapsed — the regime is not being exercised"


def test_hyperboloid_dist_matches_the_oracle_at_tiny_curvature():
    """c = 1e-6: the ``√c`` factors are 1e-3 apart from 1, which a c-independent bug hides.

    ``√c·d₀ = 10`` here means a *geodesic* radius of 1e4 and a spatial radius of ~1.1e7 — the one
    corner of the parameter space where the ``1/√c`` scaling and the exponential growth pull in
    opposite directions.
    """
    c = 1e-6
    case = _hyperboloid_case(10.0, "angular", 0.3, c, 4, np.float64)
    assert case is not None
    manifold = Hyperboloid(dtype=jnp.float64)
    d = float(manifold.dist(case.x_A, case.y_A, c))
    assert abs(d - case.d_ref) <= 1e-13 * case.d_ref + 4.0 * case.resolution


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
def test_hyperboloid_dist_and_logmap_are_jit_and_vmap_clean(dtype):
    """Jitted == eager for the new arms, and a vmapped call equals the Python loop.

    ``version_idx`` is static, so the switch must be resolved at trace time; the frame is a
    ``NamedTuple`` (a pytree), which is what keeps it traceable at all.
    """
    e1, e2 = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    xs = jnp.stack([_hyperboloid_point(a, c, e1, dtype) for a in (0.5, 5.0, 12.0)])
    ys = jnp.stack([_hyperboloid_point(a + 0.3, c, np.cos(0.4) * e1 + np.sin(0.4) * e2, dtype) for a in (0.5, 5.0, 12.0)])

    for version_idx in (0, 1):
        eager = np.array([float(manifold.dist(x, y, c, version_idx=version_idx)) for x, y in zip(xs, ys, strict=True)])
        jitted = jax.jit(manifold.dist, static_argnames=["version_idx"])
        assert np.array_equal(
            np.array([float(jitted(x, y, c, version_idx=version_idx)) for x, y in zip(xs, ys, strict=True)]), eager
        )
        batched = jax.vmap(manifold.dist, in_axes=(0, 0, None, None))(xs, ys, c, version_idx)
        assert np.array_equal(np.asarray(batched, dtype=np.float64), eager)

    logs_eager = np.stack([np.asarray(manifold.logmap(y, x, c)) for x, y in zip(xs, ys, strict=True)])
    logs_vmap = np.asarray(jax.vmap(manifold.logmap, in_axes=(0, 0, None))(ys, xs, c))
    assert np.allclose(logs_vmap, logs_eager, rtol=1e-6, atol=0.0)
    assert np.allclose(np.asarray(jax.jit(manifold.logmap)(ys[0], xs[0], c)), logs_eager[0], rtol=1e-6, atol=0.0)


def test_hyperboloid_dist_survives_the_last_representable_float32_radius():
    """Radius 85 (``x₀ = 4.1e36``, one step below float32's 3.4e38 ceiling) computes cleanly.

    The intermediate ``√u_x·√u_y`` is 8e36 here, while the "simplification" ``√(u_x·u_y)`` would
    need 6.4e73 and overflow. Beyond representability (``x₀`` already inf) the result must be inf,
    never NaN — an inf stays visible, a NaN poisons every parameter it reaches.
    """
    e1, e2 = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=jnp.float32)
    c = 1.0
    x_A = _hyperboloid_point(85.0, c, e1, jnp.float32)
    y_A = _hyperboloid_point(84.0, c, np.cos(0.5) * e1 + np.sin(0.5) * e2, jnp.float32)
    assert np.isfinite(float(x_A[0])) and float(x_A[0]) > 1e36

    d = float(manifold.dist(x_A, y_A, c))
    assert np.isfinite(d)
    assert d == pytest.approx(85.0 + 84.0 - 2.0 * np.log(1.0 / np.sin(0.25)), rel=1e-3)

    overflowed_A = x_A.at[0].set(jnp.inf)
    d_inf = float(manifold.dist(overflowed_A, y_A, c))
    assert np.isinf(d_inf), f"expected inf for an unrepresentable point, got {d_inf}"


# ---------------------------------------------------------------------------------------------
# tangent_norm: the ambient Minkowski norm is wrong by 100% on unit tangent vectors at radius 10.
# ---------------------------------------------------------------------------------------------


def _ambient_lorentz_norm(v: np.ndarray) -> float:
    """The pre-fix body: ``sqrt(clip(-v₀² + ‖v_s‖², 0) + MIN_NORM²)``, which ignores ``x`` entirely."""
    return float(np.sqrt(max(-(v[0] ** 2) + float(np.dot(v[1:], v[1:])), 0.0) + 1e-15**2))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("a", [0.5, 4.0, 8.0, 12.0])
def test_hyperboloid_tangent_norm_is_one_on_exactly_unit_tangent_vectors(dtype, c: float, a: float):
    """``‖e_rad‖_x = ‖e_ang‖_x = ‖(e_rad + e_ang)/√2‖_x = 1``, built analytically.

    The frame vectors are unit by construction (``⟨e_rad, e_rad⟩_L = c·(x₀² - r_x²) = 1``), so this
    needs no oracle. Eliminating ``v₀`` through the tangency condition costs one power of ``x₀`` in
    the error instead of two, which is the entire fix.
    """
    e1, e2 = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=dtype)
    sqrt_c = np.sqrt(c)
    x_A = _hyperboloid_point(a, c, e1, dtype)
    e_rad = -sqrt_c * np.concatenate([[np.sinh(a) / sqrt_c], (np.cosh(a) / sqrt_c) * e1])
    e_ang = np.concatenate([[0.0], e2])

    tol = 1e-4 if dtype is jnp.float32 else 1e-9
    for v_np in (e_rad, e_ang, (e_rad + e_ang) / np.sqrt(2.0)):
        v = jnp.asarray(v_np.astype(dtype))
        assert float(manifold.tangent_norm(v, x_A, c)) == pytest.approx(1.0, abs=tol)


def test_hyperboloid_tangent_norm_beats_the_ambient_minkowski_form_at_large_radius():
    """Regression pin: at radius 20 the ambient form collapses to the ``MIN_NORM`` floor.

    ``-v₀² + ‖v_s‖²`` subtracts two numbers of size ``(√c·x₀)² = 2.4e16`` to get 1, so in float64
    the difference is pure rounding — it clips to 0 and the old ``+ MIN_NORM²`` floor returns 1e-15
    for a vector whose true norm is exactly 1. The base-point form returns 1.
    """
    e1, _ = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=jnp.float64)
    c, a = 1.0, 20.0
    x_A = _hyperboloid_point(a, c, e1, jnp.float64)
    e_rad = -np.concatenate([[np.sinh(a)], np.cosh(a) * e1])

    assert float(manifold.tangent_norm(jnp.asarray(e_rad), x_A, c)) == pytest.approx(1.0, abs=1e-9)
    assert _ambient_lorentz_norm(e_rad) < 1e-3, "the ambient form no longer fails — pin is stale"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=_HYP_DTYPE_IDS)
def test_hyperboloid_tangent_norm_is_zero_with_a_finite_gradient_at_the_zero_vector(dtype):
    """``‖0‖_x == 0`` exactly, with a finite (zero) gradient — no ``MIN_NORM²`` floor needed."""
    e1, _ = _hyperboloid_basis(4)
    manifold = Hyperboloid(dtype=dtype)
    c = 1.0
    x_A = _hyperboloid_point(3.0, c, e1, dtype)
    zero = jnp.zeros_like(x_A)

    assert float(manifold.tangent_norm(zero, x_A, c)) == 0.0
    grad = np.asarray(jax.grad(lambda v: manifold.tangent_norm(v, x_A, c))(zero))
    assert np.all(np.isfinite(grad))


# =============================================================================================
# Hyperboloid parallel transport — a VALUE oracle for the general x -> y transport
#
# Not one of the F6-M1 audit specs: it closes a gap this file left. ``ptransp`` was covered only by
# *property* tests in tests/test_manifolds.py (it is an isometry, it round-trips, it agrees with
# ``ptransp_0`` at the origin). Those constrain the map up to a rotation of the target tangent
# space: a transport that is *an* isometry T_xM -> T_yM but not the Levi-Civita one — say the
# correct formula composed with a spatial rotation, or one built from the wrong pair of points —
# satisfies every one of them. Only a value oracle separates them.
#
# The closed form (Lou et al. 2020, "Differentiating through the Fréchet mean", the same reference
# the implementation cites; standard for the Lorentz model since Nickel & Kiela 2018):
#
#     PT_{x->y}(v) = v + ⟨y, v⟩_L / (1/c - ⟨x, y⟩_L) · (x + y),   ⟨a, b⟩_L = -a₀b₀ + ⟨a_s, b_s⟩
#
# Independence, on the same terms as the rest of this file: the expression is transcribed into
# NumPy float64 here, and *every operand it is fed is also built in NumPy* — the base points come
# from the NumPy ``exp_0`` closed form ``cosh(√c‖v_s‖)/√c · e₀ + sinh(√c‖v_s‖)/(√c‖v_s‖) · v``
# rather than from ``manifold.expmap_0``, and the tangent vectors are Minkowski-projected in NumPy
# (``v = w + c⟨x, w⟩_L·x``, which annihilates ⟨x, ·⟩_L because ⟨x, x⟩_L = -1/c). No hyperbolix call
# appears anywhere upstream of ``expected_BA``, so this is library-vs-reference, not
# library-vs-itself.
#
# This is the oracle used for the ``Hyperboloid``/``ptransp`` row of the reproduction repo's
# Table 2, where its own tangency invariant ⟨y, PT v⟩_L = 0 was checked to 1e-14..1e-15 over the
# whole (curvature x dimension) grid; ``test_..._oracle_is_tangent_at_the_target`` re-runs that
# guard here so the oracle cannot rot silently underneath the comparison.
# =============================================================================================

_PTRANSP_DIMS = [2, 10]
"""Spatial dimensions: the minimal case plus one generic value, as in tests/conftest.py."""

_PTRANSP_N = 64
"""Sampled (x, y, v) triples per cell."""


def _np_lorentz(x_BA: np.ndarray, y_BA: np.ndarray) -> np.ndarray:
    """⟨x, y⟩_L = -x₀y₀ + ⟨x_s, y_s⟩, time coordinate first. Pure NumPy, float64."""
    return -x_BA[..., 0] * y_BA[..., 0] + np.sum(x_BA[..., 1:] * y_BA[..., 1:], axis=-1)


def _np_hyperboloid_expmap_0(v_BA: np.ndarray, c: float) -> np.ndarray:
    """NumPy ``exp_0``: builds the sample points without calling the library's own map."""
    sqrt_c = np.sqrt(c)
    arg_B1 = sqrt_c * np.linalg.norm(v_BA[..., 1:], axis=-1, keepdims=True)
    time_B1 = np.cosh(arg_B1) / sqrt_c
    space_BD = np.sinh(arg_B1) / arg_B1 * v_BA[..., 1:]
    return np.concatenate([time_B1, space_BD], axis=-1)


def _np_hyperboloid_ptransp(v_BA: np.ndarray, x_BA: np.ndarray, y_BA: np.ndarray, c: float) -> np.ndarray:
    """``PT_{x->y}(v) = v + ⟨y, v⟩_L/(1/c - ⟨x, y⟩_L)·(x + y)``, transcribed in NumPy float64."""
    coef_B1 = (_np_lorentz(y_BA, v_BA) / (1.0 / c - _np_lorentz(x_BA, y_BA)))[..., None]
    return v_BA + coef_B1 * (x_BA + y_BA)


def _np_sample_hyperboloid_points(rng: np.random.Generator, n: int, dim: int, c: float) -> np.ndarray:
    """``n`` points at geodesic radius in ``[0.05, 2]/√c``, via the NumPy ``exp_0`` oracle.

    Radii are capped at ``2/√c`` on purpose: past that the ambient chart's own cancellation starts
    to dominate (that regime is what the Decimal-oracle section above is for), and this test is
    about the transport formula, not about the conditioning of ⟨x, y⟩_L.
    """
    direction_BD = rng.normal(size=(n, dim))
    direction_BD /= np.linalg.norm(direction_BD, axis=-1, keepdims=True)
    radius_B1 = (0.05 + 1.95 * rng.random((n, 1))) / np.sqrt(c)
    v_BA = np.concatenate([np.zeros((n, 1)), direction_BD * radius_B1], axis=-1)
    return _np_hyperboloid_expmap_0(v_BA, c)


def _np_sample_tangent_at(rng: np.random.Generator, x_BA: np.ndarray, c: float) -> np.ndarray:
    """Generic tangent vectors at ``x``, Minkowski-projected and rescaled to Riemannian norm ~1.

    ``v = w + c⟨x, w⟩_L·x`` is exactly tangent because ⟨x, x⟩_L = -1/c, so the correction cancels
    ⟨x, w⟩_L. The ambient ``w`` has a non-zero time component, so ``v`` is a *generic* tangent
    vector rather than one of the purely spatial ones ``ptransp_0`` is usually fed — a transport
    that mishandles the time coordinate has nowhere to hide.
    """
    w_BA = rng.normal(size=x_BA.shape)
    v_BA = w_BA + c * _np_lorentz(x_BA, w_BA)[..., None] * x_BA
    norm_B1 = np.sqrt(_np_lorentz(v_BA, v_BA))[..., None]  # positive: the metric is Riemannian on T_xM
    return v_BA / norm_B1


def _ptransp_case(c: float, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, v, PT_{x->y}v)`` for one cell — everything NumPy float64, seeded per cell."""
    # Positional integer stream ids, not hash(): str/float hashing is salted per process, so a
    # hash-derived seed would silently change the sampled grid from run to run.
    rng = np.random.default_rng([0, dim, round(1000 * c)])
    x_BA = _np_sample_hyperboloid_points(rng, _PTRANSP_N, dim, c)
    y_BA = _np_sample_hyperboloid_points(rng, _PTRANSP_N, dim, c)
    v_BA = _np_sample_tangent_at(rng, x_BA, c)
    return x_BA, y_BA, v_BA, _np_hyperboloid_ptransp(v_BA, x_BA, y_BA, c)


@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", _PTRANSP_DIMS)
def test_hyperboloid_ptransp_oracle_is_tangent_at_the_target(c: float, dim: int):
    """Guards the oracle itself: ⟨y, PT v⟩_L == 0 and the transport is a Lorentz isometry.

    Both hold identically for the closed form given ⟨x, v⟩_L = 0, so a mistranscription (a sign, a
    ``1/c`` turned into ``c``, ``x + y`` written as ``x - y``) breaks them — and the value
    comparison below would otherwise be measuring a broken reference.
    """
    x_BA, y_BA, v_BA, expected_BA = _ptransp_case(c, dim)

    assert np.max(np.abs(_np_lorentz(x_BA, v_BA))) < 1e-12, "the sampled v is not tangent at x"
    assert np.max(np.abs(_np_lorentz(y_BA, expected_BA))) < 1e-12, "PT v is not tangent at y"
    # An isometry: the Riemannian norm is carried across unchanged.
    assert np.allclose(_np_lorentz(expected_BA, expected_BA), _np_lorentz(v_BA, v_BA), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("dim", _PTRANSP_DIMS)
def test_hyperboloid_ptransp_matches_the_numpy_oracle(c: float, dim: int):
    """``Hyperboloid.ptransp(v, x, y, c)`` equals the NumPy closed form to 1e-12 absolute.

    Argument order is the library's own: ``ptransp(v, x, y, c)`` transports ``v`` *from* ``x`` *to*
    ``y``, and the ambient layout keeps the time coordinate at index 0. Measured worst case over the
    grid: 1.9e-14, i.e. ~2% of the tolerance — in line with the 5.5e-14 the reproduction repo
    records for this operation over a comparable float64 grid.
    """
    manifold = Hyperboloid(dtype=F64)
    x_BA, y_BA, v_BA, expected_BA = _ptransp_case(c, dim)

    ptransp_batch = jax.vmap(manifold.ptransp, in_axes=(0, 0, 0, None))
    got_BA = np.asarray(ptransp_batch(jnp.asarray(v_BA), jnp.asarray(x_BA), jnp.asarray(y_BA), c), dtype=np.float64)

    assert np.max(np.abs(got_BA - expected_BA)) <= 1e-12
    # Non-degenerate: the transport genuinely moves the vector, so "matches" is not "is the
    # identity". At these radii the correction term is of order the vector itself.
    assert np.max(np.abs(expected_BA - v_BA)) > 0.1
