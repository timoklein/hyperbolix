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

    The ``MIN_DENOM`` floor is what makes this hold; removing it turns the boundary cotangent into
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
