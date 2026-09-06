"""JAX test fixtures mirroring the PyTorch conftest.py.

This file contains global fixtures for JAX-based tests, providing compatible
interfaces with the PyTorch test fixtures but using JAX/NumPy random operations.
"""

from __future__ import annotations

import os

# Each xdist worker builds its own JAX GPU client, and each one preallocates 75% of the memory
# still free at that moment. Past the second or third worker the leftovers are too small to run
# on: the suite fails with `RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN` /
# `INTERNAL: Autotuning failed for HLO ... No valid config found!` / `CUDA_ERROR_OUT_OF_MEMORY`
# (measured: 3/6 at -n 4, 26/47 at -n 12, 38/47 at -n auto on a free 40 GB A100; a
# `RuntimeError: Bad StatusOr access` during client bring-up, before any test runs, has also been
# seen). Allocating on demand instead lets all workers share one GPU: 76/76 pass at -n 12.
# `setdefault`, not assignment: an explicit setting from the caller's environment
# wins. Must be set before the first `import jax` in the process, hence its place among the
# imports rather than further down.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hyperbolix as hj


def _assert_pythonpath_checkout_is_the_imported_one() -> None:
    """Fail loudly when ``PYTHONPATH`` names a checkout that is *not* the one under test.

    Why this exists: comparing a branch against its parent commit is usually done by pointing
    ``PYTHONPATH`` at a second worktree. Several things silently outrank ``PYTHONPATH`` on
    ``sys.path`` -- ``python -m pytest`` prepends the current directory (CPython's ``-m``, not
    pytest), and under ``--import-mode=prepend`` pytest prepends the rootdir as well. When the
    current directory is itself a checkout, ``import hyperbolix`` then resolves to the local
    tree and the "this fails on the parent commit" gate quietly tests the branch's own code.

    The check is a no-op when ``PYTHONPATH`` is unset or holds no directory with a
    ``hyperbolix`` package, so ordinary ``uv run pytest`` runs are unaffected.
    """
    raw = os.environ.get("PYTHONPATH", "")
    if not raw:
        return

    imported = Path(hj.__file__).resolve().parent
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        expected = Path(entry).resolve() / "hyperbolix"
        if not (expected / "__init__.py").is_file():
            continue
        # First PYTHONPATH entry carrying a hyperbolix package: that is the checkout the caller
        # asked for. Anything else winning means something was prepended ahead of it.
        if expected == imported:
            return
        raise RuntimeError(
            "PYTHONPATH names a hyperbolix checkout that pytest did NOT import.\n"
            f"  PYTHONPATH asks for: {expected}\n"
            f"  actually imported:   {imported}\n"
            "Something is prepended ahead of PYTHONPATH on sys.path. Common causes and fixes:\n"
            "  * `python -m pytest` inserts the current directory at sys.path[0]; run the\n"
            "    `pytest` console script instead, or cd somewhere that is not a checkout.\n"
            "  * `--import-mode=prepend` (pytest's default) inserts the rootdir; this repo sets\n"
            "    `--import-mode=importlib` in pyproject.toml -- do not override it.\n"
            "Refusing to run: the results would describe the wrong checkout."
        )


_assert_pythonpath_checkout_is_the_imported_one()

# Enable float64 support in JAX
jax.config.update("jax_enable_x64", True)

# Independent RNG stream ids for the package-scoped data fixtures. Each fixture derives its
# generator from (seed_jax, stream_id, ...) instead of drawing from a single shared generator,
# so the values a test sees no longer depend on which other tests ran first. Without this a
# failure found in a full run may not reproduce under ``-k`` (and vice versa).
_CURVATURE_STREAM = {"euclidean": 0, "poincare": 1, "hyperboloid": 2, "pv": 3}
_POINTS_STREAM = {"euclidean": 10, "poincare": 11, "hyperboloid": 12, "pv": 13}


@pytest.fixture(scope="package", params=[10])
def seed_jax(request: pytest.FixtureRequest) -> int:
    """Global seed for JAX reproducibility.

    Single seed by design: every property checked by the fixture-driven manifold tests is a
    pointwise identity over an i.i.d. sample, so a second seed only redraws the same
    distribution. The ``params`` list is kept so re-widening the axis stays a one-line change.
    """
    return request.param


@pytest.fixture
def rng(seed_jax: int) -> np.random.Generator:
    """Fresh per-test NumPy RNG, seeded from ``seed_jax``.

    Function-scoped on purpose: several test bodies draw from this generator, and a shared
    package-scoped generator made those draws depend on test execution order.
    """
    return np.random.default_rng(seed=seed_jax)


@pytest.fixture(scope="package", params=[jnp.float32, jnp.float64])
def dtype(request: pytest.FixtureRequest) -> jnp.dtype:
    """Test both float32 and float64 precision (mirroring PyTorch)."""
    return jnp.dtype(request.param)


@pytest.fixture(scope="package")
def tolerance(dtype: jnp.dtype) -> tuple[float, float]:
    """Numerical tolerances for floating point comparisons (matches PyTorch)."""
    if dtype == jnp.float32:
        return 4e-3, 4e-3  # atol, rtol
    return 1e-7, 1e-7  # float64


def _make_manifold_and_c(manifold_name: str, dtype: jnp.dtype, seed: int, draw: int = 0):
    """Build a (manifold_instance, curvature) pair with a sampled curvature.

    Curvatures are sampled the same way as PyTorch (exponential, rate=0.5), from a generator
    derived from (seed, manifold, draw) so the value is independent of test execution order.
    ``draw`` selects an independent curvature sample; the generic property tests run two draws
    per hyperbolic manifold because scale-dependent bugs have historically been caught only at
    curvatures far from 1 (see the sampling note on the hyperboloid generator below).
    """
    if manifold_name == "euclidean":
        # Euclidean always has c=0
        return hj.manifolds.Euclidean(dtype=dtype), 0.0

    gen = np.random.default_rng([seed, _CURVATURE_STREAM[manifold_name], draw])
    # Random positive curvature (exponential distribution, rate=0.5).
    # Matches PyTorch: torch.empty(1).exponential_(0.5)
    c = float(gen.exponential(scale=2.0))  # scale = 1/rate

    if manifold_name == "poincare":
        return hj.manifolds.Poincare(dtype=dtype), c
    elif manifold_name == "hyperboloid":
        return hj.manifolds.Hyperboloid(dtype=dtype), c
    elif manifold_name == "pv":
        return hj.manifolds.ProperVelocity(dtype=dtype), c
    raise ValueError(f"Unknown manifold: {manifold_name}")


@pytest.fixture(
    scope="package",
    params=[("euclidean", 0), ("poincare", 0), ("poincare", 1), ("hyperboloid", 0), ("hyperboloid", 1)],
    ids=["Euclidean", "PoincareBall-c0", "PoincareBall-c1", "Hyperboloid-c0", "Hyperboloid-c1"],
)
def manifold_and_c(request: pytest.FixtureRequest, dtype: jnp.dtype, seed_jax: int):
    """Fixture providing (manifold_instance, curvature) tuples.

    Two independent curvature draws per hyperbolic manifold: a single sampled curvature would
    lose the fault-detection power of testing widely separated scales.

    ProperVelocity is intentionally absent: ``tests/test_pv_manifold.py`` covers every property
    this fixture's consumers check for PV, with strictly stronger oracles (metric duality,
    paper closed forms, private-helper cross-checks).
    """
    manifold_name, draw = request.param
    return _make_manifold_and_c(manifold_name, dtype, seed_jax, draw)


@pytest.fixture(scope="package")
def poincare_and_c(dtype: jnp.dtype, seed_jax: int):
    """(Poincare, c) for Poincaré-specific tests (Möbius gyration, Apollonian metric)."""
    return _make_manifold_and_c("poincare", dtype, seed_jax)


@pytest.fixture(scope="package")
def hyperboloid_and_c(dtype: jnp.dtype, seed_jax: int):
    """(Hyperboloid, c) for Hyperboloid-specific tests (Lorentz gyroaddition, sqdist)."""
    return _make_manifold_and_c("hyperboloid", dtype, seed_jax)


def _sample_uniform_points(manifold, c: float, dim: int, dtype: jnp.dtype, seed: int) -> jnp.ndarray:
    """Generate uniformly distributed points on the manifold.

    Mirrors the PyTorch uniform_points fixture, generating the same number
    and distribution of points but using NumPy arrays converted to JAX.
    """
    num_pts = 2_500 * 6  # Same as PyTorch
    np_dtype = np.dtype(dtype.name)

    if isinstance(manifold, hj.manifolds.Euclidean):
        rng = np.random.default_rng([seed, _POINTS_STREAM["euclidean"], dim])
        # Euclidean: uniform in box [-100, 100]^d
        lower, upper = -100.0, 100.0
        data = rng.uniform(lower, upper, size=(num_pts, dim)).astype(np_dtype)
        return jnp.asarray(data)

    elif isinstance(manifold, hj.manifolds.Poincare):
        rng = np.random.default_rng([seed, _POINTS_STREAM["poincare"], dim])
        # Poincaré ball: uniform sampling using spherical coordinates
        # Matches PyTorch approach
        random_dirs = rng.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
        random_dirs /= np.linalg.norm(random_dirs, axis=-1, keepdims=True)
        random_radii = rng.random((num_pts, 1)).astype(np_dtype) ** (1.0 / dim)
        # Scale to ball of radius 1/√c
        points = (random_dirs * random_radii) / np.sqrt(c)
        points = jnp.asarray(points, dtype=dtype)
        proj_batch = jax.vmap(manifold.proj, in_axes=(0, None))
        return proj_batch(points, c)

    elif isinstance(manifold, hj.manifolds.Hyperboloid):
        rng = np.random.default_rng([seed, _POINTS_STREAM["hyperboloid"], dim])
        # Hyperboloid: generate points on upper sheet
        # Mirrors PyTorch approach: generate in Poincaré, scale, convert
        random_dirs = rng.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
        random_dirs /= np.linalg.norm(random_dirs, axis=-1, keepdims=True)
        random_radii = rng.random((num_pts, 1)).astype(np_dtype) ** (1.0 / dim)
        poincare_points = (random_dirs * random_radii) / np.sqrt(c)

        # Scale by 0.5 to account for representational limitations (matches PyTorch)
        poincare_points = poincare_points * 0.5

        # Convert Poincaré to Hyperboloid
        # Curvature-correct inverse stereographic projection (radius-1/√c ball).
        # Given a Poincaré point p with ||p||² < 1/c, its hyperboloid image is
        # [x₀, x_rest] where
        #   x₀     = (1 + c||p||²) / (√c · (1 - c||p||²))   (only the time part carries 1/√c)
        #   x_rest = 2p / (1 - c||p||²)                     (NO extra /√c — matches
        #            isometry_mappings.poincare_to_hyperboloid)
        # The earlier code carried an extra /√c on x_rest, which was correct only at
        # c=1 and was silently masked by the proj() below (same bug class as the
        # historical poincare_to_hyperboloid defect).
        p = poincare_points
        p_sqnorm = np.sum(p**2, axis=-1, keepdims=True)
        denom = 1.0 - c * p_sqnorm
        denom = np.maximum(denom, 1e-15)  # Avoid division by zero

        sqrt_c = np.sqrt(c)
        x0 = (1.0 + c * p_sqnorm) / (sqrt_c * denom)
        x_rest = 2.0 * p / denom

        points = np.concatenate([x0, x_rest], axis=-1).astype(np_dtype)
        points = jnp.asarray(points, dtype=dtype)

        # Project to ensure they're on the manifold
        proj_batch = jax.vmap(manifold.proj, in_axes=(0, None))
        return proj_batch(points, c)

    elif isinstance(manifold, hj.manifolds.ProperVelocity):
        rng = np.random.default_rng([seed, _POINTS_STREAM["pv"], dim])
        # Proper Velocity: unconstrained ℝⁿ. Gaussian samples scaled to 1/√c keep
        # typical geodesic distance to origin of order asinh(1) ~ 0.88, mirroring
        # the hyperbolic manifolds' "moderate distance" regime.
        data = rng.normal(0.0, 1.0, size=(num_pts, dim)).astype(np_dtype)
        data = data / np.sqrt(c)
        return jnp.asarray(data, dtype=dtype)

    else:
        raise ValueError("Unknown manifold module")


@pytest.fixture(scope="package", params=[2, 10])
def uniform_points(manifold_and_c, dtype: jnp.dtype, request: pytest.FixtureRequest, seed_jax: int) -> jnp.ndarray:
    """Uniform points on the ``manifold_and_c`` manifold.

    ``dim`` axis is {2, 10}: no manifold has a dimension-dependent code path, so ``dim`` is
    pure vectorization width — 2 (disk / ambient-3 hyperboloid edge case) plus one generic value.
    """
    manifold, c = manifold_and_c
    return _sample_uniform_points(manifold, c, request.param, dtype, seed_jax)


@pytest.fixture(scope="package", params=[2, 10])
def poincare_points(poincare_and_c, dtype: jnp.dtype, request: pytest.FixtureRequest, seed_jax: int) -> jnp.ndarray:
    """Uniform points on the Poincaré ball (companion to ``poincare_and_c``)."""
    manifold, c = poincare_and_c
    return _sample_uniform_points(manifold, c, request.param, dtype, seed_jax)


@pytest.fixture(scope="package", params=[2, 10])
def hyperboloid_points(hyperboloid_and_c, dtype: jnp.dtype, request: pytest.FixtureRequest, seed_jax: int) -> jnp.ndarray:
    """Uniform points on the hyperboloid (companion to ``hyperboloid_and_c``)."""
    manifold, c = hyperboloid_and_c
    return _sample_uniform_points(manifold, c, request.param, dtype, seed_jax)
