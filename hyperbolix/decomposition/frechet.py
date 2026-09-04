"""Fréchet (Karcher) mean on Riemannian manifolds via fixed-point iteration.

Manifold-generic weighted-uniform Fréchet mean: the point minimizing the sum of squared
geodesic distances to a set of points. Computed by the Karcher fixed-point iteration —
repeatedly average the log-map tangents at the current estimate and step along the
exponential map — inside a ``jax.lax.while_loop``.

This is the data-centering primitive for HoroPCA (Chami et al. 2021), which assumes
Fréchet-mean-zero data before fitting horospherical components.

Dimension key:
  N: number of points
  R: manifold representation dim (ambient d+1 for Hyperboloid, spatial d for
     Poincare / ProperVelocity / Euclidean)

References:
    Karcher, H. "Riemannian center of mass and mollifier smoothing." Comm. Pure Appl.
        Math. 1977.
    Chami et al. "HoroPCA: Hyperbolic dimensionality reduction via horospherical
        projections." ICML 2021.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..manifolds import Hyperboloid, Manifold, Poincare
from ..manifolds.protocol import ScalarCurvature
from ..nn_layers.hyperboloid_core import lorentz_midpoint


def frechet_mean(
    x_NR: Float[Array, "N R"],
    manifold: Manifold,
    c: ScalarCurvature,
    *,
    step_size: float = 1.0,
    tol: float = 1e-8,
    max_iters: int = 100,
    init_D: Float[Array, "R"] | None = None,
) -> Float[Array, "R"]:
    """Compute the weighted-uniform Fréchet mean of ``x_NR`` via Karcher iteration.

    Minimizes ``Σ_i d(μ, x_i)²`` over the manifold. Each iteration averages the log-map
    tangents at the current estimate and steps along the exponential map::

        v   = step_size · mean_i logmap(x_i, μ)      # tangent at μ
        μ   ← proj(expmap(v, μ))
        δ   = ‖v‖_μ                                   # Riemannian tangent norm

    stopping when ``δ ≤ tol`` or ``max_iters`` is reached. With ``step_size = 1`` this is
    the standard Karcher fixed point (Newton-like on the negative-curvature squared-distance
    functional). The loop is not differentiated (``while_loop`` is not reverse-mode
    differentiable) — used purely as a numerical solver.

    The initial estimate is dispatched by manifold type: Hyperboloid uses the closed-form
    Lorentz centroid (:func:`lorentz_midpoint` with uniform weights); Poincaré uses the
    projected Euclidean mean; every other manifold uses the first point. Pass ``init_D`` to
    override (a point in the manifold's representation space).

    Not JIT-decorated here (the ``manifold`` argument is unhashable). To compile, wrap a
    closure::

        fmean = jax.jit(functools.partial(frechet_mean, manifold=H, c=1.0))

    Args:
        x_NR: Points on the manifold, shape (N, R). ``R`` is the ambient dim (d+1) for
            Hyperboloid, the spatial dim (d) for Poincaré / ProperVelocity / Euclidean.
        manifold: Manifold instance (satisfies the ``Manifold`` protocol).
        c: Curvature (positive scalar for single manifolds).
        step_size: Karcher step size (default 1.0 — the standard fixed point).
        tol: Convergence tolerance on the Riemannian tangent-update norm (default 1e-8).
        max_iters: Maximum iterations (default 100).
        init_D: Optional explicit initial estimate, shape (R,), in representation space.

    Returns:
        Fréchet mean, shape (R,), on the manifold.

    Notes:
        In float32 the update norm stalls around ``≈1e-6`` (the boundary of f32
        resolution), so a ``tol`` below that just runs the loop to ``max_iters`` — harmless,
        the estimate is already converged to f32 precision.
    """
    x_NR = manifold._cast(x_NR)
    dtype = manifold.dtype
    num_points = x_NR.shape[0]

    # -- Initial estimate (Python-level dispatch on manifold type) -----------------------
    if init_D is not None:
        mean_R = jnp.asarray(init_D, dtype=dtype)
    elif isinstance(manifold, Hyperboloid):
        # Closed-form Lorentz centroid with uniform weights (M = N points → 1 midpoint).
        weights_1N = jnp.full((1, num_points), 1.0 / num_points, dtype=dtype)
        mean_R = lorentz_midpoint(x_NR, weights_1N, c)[0]
    elif isinstance(manifold, Poincare):
        # Projected Euclidean mean (the ball is convex, so the mean stays inside it).
        mean_R = manifold.proj(jnp.mean(x_NR, axis=0), c)
    else:
        # Euclidean / ProperVelocity and any other manifold: first point is a valid start.
        mean_R = x_NR[0]

    # -- Karcher fixed-point iteration (lax.while_loop) ---------------------------------
    logmap_batched = jax.vmap(manifold.logmap, in_axes=(0, None, None))  # log_μ(x_i) for all i

    def cond_fn(carry: tuple[Array, Array, Array]) -> Array:
        _, delta, it = carry
        return (delta > tol) & (it < max_iters)

    def body_fn(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        mean, _, it = carry
        logs_NR = logmap_batched(x_NR, mean, c)  # (N, R) tangents at mean
        v_R = step_size * jnp.mean(logs_NR, axis=0)  # (R,) averaged tangent
        new_mean = manifold.proj(manifold.expmap(v_R, mean, c), c)
        new_delta = manifold.tangent_norm(v_R, mean, c)
        return new_mean, new_delta, it + 1

    delta0 = jnp.asarray(jnp.inf, dtype=dtype)  # force at least one iteration
    it0 = jnp.asarray(0, dtype=jnp.int32)
    mean_R, _, _ = lax.while_loop(cond_fn, body_fn, (mean_R, delta0, it0))
    return mean_R
