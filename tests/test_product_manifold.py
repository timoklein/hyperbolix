"""Tests for ProductManifold.

Covers construction, decomposition correctness, distance properties,
exp/log round-trips, origin, JIT/vmap, gradient flow, and edge cases
(single-factor, all-Euclidean).

API note: ProductManifold methods take a positional ``c`` argument that
must be a sequence of per-factor curvatures (length ``n_factors``). Tests
pass ``product.curvatures`` (factor-stored values) for static cases and
an explicit ``(c_h, c_p)`` tuple for the learnable-curvature test. The
fixture is named ``cs`` locally to reflect the sequence shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from hyperbolix import LearnableCurvature
from hyperbolix.manifolds import (
    Euclidean,
    Hyperboloid,
    Manifold,
    Poincare,
    ProductManifold,
    ProperVelocity,
)

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_product_point(product: ProductManifold, rng: np.random.Generator) -> jnp.ndarray:
    """Generate a valid point on the product manifold."""
    parts = []
    for m, dim in zip(product.factors, product.dims, strict=True):
        if isinstance(m, Euclidean):
            p = rng.normal(0, 1.0, size=(dim,)).astype(np.float64)
        elif isinstance(m, Poincare):
            p = rng.normal(0, 0.3, size=(dim,)).astype(np.float64)
            p = np.array(m.proj(jnp.asarray(p, dtype=m.dtype), m.c))
        elif isinstance(m, Hyperboloid):
            spatial_dim = dim - 1
            spatial = rng.normal(0, 0.3, size=(spatial_dim,)).astype(np.float64)
            ambient = np.concatenate([[1.5], spatial])
            p = np.array(m.proj(jnp.asarray(ambient, dtype=m.dtype), m.c))
        elif isinstance(m, ProperVelocity):
            p = rng.normal(0, 1.0, size=(dim,)).astype(np.float64)
        else:
            raise ValueError(f"Unknown manifold type: {type(m)}")
        parts.append(jnp.asarray(p, dtype=product.dtype))
    return jnp.concatenate(parts)


def _make_small_tangent(
    product: ProductManifold,
    x: jnp.ndarray,
    rng: np.random.Generator,
    cs,
    scale: float = 0.1,
) -> jnp.ndarray:
    """Generate a small tangent vector at x on the product manifold."""
    raw = jnp.asarray(rng.normal(0, scale, size=(product.total_dim,)), dtype=product.dtype)
    return product.tangent_proj(raw, x, cs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRODUCT_CONFIGS = [
    pytest.param(
        [(Poincare, 3, 1.0), (Hyperboloid, 3, 1.0)],
        id="P3xH2",
    ),
    pytest.param(
        [(Poincare, 2, 0.5), (Euclidean, 3, 0.0), (Hyperboloid, 4, 1.5)],
        id="P2xE3xH3",
    ),
    pytest.param(
        [(ProperVelocity, 3, 0.8), (Poincare, 2, 0.3)],
        id="PV3xP2",
    ),
    pytest.param(
        [(Hyperboloid, 5, 1.0)],
        id="H4_single",
    ),
    pytest.param(
        [(Poincare, 2, 0.5), (Poincare, 3, 1.0)],
        id="P2xP3",
    ),
]


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["f32", "f64"])
def dtype(request):
    return jnp.dtype(request.param)


@pytest.fixture
def tolerance(dtype):
    if dtype == jnp.float32:
        return 5e-3, 5e-3
    return 1e-6, 1e-6


@pytest.fixture(params=PRODUCT_CONFIGS)
def product(request, dtype):
    """Create a ProductManifold from the config."""
    config = request.param
    factors = []
    for entry in config:
        manifold_cls, dim, c = entry
        if manifold_cls is Euclidean:
            factors.append((Euclidean(dtype=dtype), dim))
        else:
            factors.append((manifold_cls(dtype=dtype, c=c), dim))
    return ProductManifold(*factors, dtype=dtype)


@pytest.fixture
def cs(product):
    """Per-factor curvatures for ``product`` — used as the static ``cs`` arg."""
    return product.curvatures


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def two_points(product, rng):
    """Two valid points on the product manifold."""
    x = _make_product_point(product, rng)
    y = _make_product_point(product, rng)
    return x, y


@pytest.fixture
def point_and_tangent(product, rng, cs):
    """A valid point and a small tangent vector at that point."""
    x = _make_product_point(product, rng)
    v = _make_small_tangent(product, x, rng, cs)
    return x, v


# ===========================================================================
# 1. Construction tests
# ===========================================================================


class TestConstruction:
    def test_basic_creation(self):
        p = ProductManifold((Hyperboloid(), 5), (Poincare(), 3))
        assert p.total_dim == 8
        assert p.n_factors == 2
        assert p.dims == (5, 3)

    def test_single_factor(self):
        p = ProductManifold((Euclidean(), 4))
        assert p.total_dim == 4
        assert p.n_factors == 1

    def test_many_factors(self):
        factors = [(Hyperboloid(c=float(i + 1)), 3) for i in range(16)]
        p = ProductManifold(*factors)
        assert p.total_dim == 48
        assert p.n_factors == 16

    def test_no_factors_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ProductManifold()

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="dim must be >= 1"):
            ProductManifold((Poincare(), 0))

    def test_invalid_manifold_type_raises(self):
        with pytest.raises(TypeError, match="ManifoldBase"):
            ProductManifold(("not_a_manifold", 3))

    def test_curvatures_property(self):
        p = ProductManifold(
            (Hyperboloid(c=2.0), 5),
            (Poincare(c=0.5), 3),
            (Euclidean(), 4),
        )
        curvs = p.curvatures
        assert len(curvs) == 3
        assert curvs[0] == 2.0
        assert curvs[1] == 0.5
        assert curvs[2] == 0.0

    def test_no_c_attribute(self):
        """ProductManifold deliberately has no scalar ``c``."""
        p = ProductManifold((Poincare(), 3), (Hyperboloid(), 5))
        assert not hasattr(p, "c")

    def test_factors_property(self):
        h = Hyperboloid(c=1.0)
        p = Poincare(c=0.1)
        product = ProductManifold((h, 5), (p, 3))
        assert product.factors[0] is h
        assert product.factors[1] is p

    def test_from_signature(self):
        pm = ProductManifold.from_signature(
            (Hyperboloid, 3, 4, 1.0),
            (Poincare, 2, 3, 0.1),
            (Euclidean, 5, 1),
        )
        assert pm.n_factors == 8
        assert pm.total_dim == 4 * 3 + 3 * 2 + 5
        assert all(isinstance(f, Hyperboloid) for f in pm.factors[:4])
        assert all(isinstance(f, Poincare) for f in pm.factors[4:7])
        assert isinstance(pm.factors[7], Euclidean)

    def test_from_signature_invalid_tuple_length_raises(self):
        with pytest.raises(ValueError, match="3-tuple"):
            ProductManifold.from_signature((Hyperboloid, 3, 2, 1.0, "extra"))

    def test_not_nnx_module(self):
        pm = ProductManifold((Hyperboloid(), 5), (Poincare(), 3))
        assert not isinstance(pm, nnx.Module)

    def test_repr(self):
        p = ProductManifold((Hyperboloid(c=1.0), 5), (Poincare(c=0.1), 3))
        r = repr(p)
        assert "Hyperboloid" in r
        assert "Poincare" in r
        assert "total_dim=8" in r


# ===========================================================================
# 2. cs validation
# ===========================================================================


class TestCurvatureSequenceValidation:
    """ProductManifold methods require an explicit per-factor curvature sequence."""

    def test_scalar_c_raises(self, product, rng):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        with pytest.raises(TypeError, match="sequence"):
            product.dist(x, y, 0.0)  # type: ignore[arg-type]

    def test_wrong_length_raises(self, product, rng):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        too_short = (1.0,) * (product.n_factors + 1)
        with pytest.raises(ValueError, match="curvatures"):
            product.dist(x, y, too_short)

    def test_satisfies_manifold_protocol(self, product):
        """ProductManifold satisfies the structural ``Manifold`` protocol.

        The protocol-level ``Curvature`` type was widened to include
        sequences, so the product's per-factor sequence shape unifies with
        the scalar shape used by ``Poincare``/``Hyperboloid``/etc. under a
        single protocol.
        """
        assert isinstance(product, Manifold)


# ===========================================================================
# 3. Split / Combine
# ===========================================================================


class TestSplitCombine:
    def test_split_shapes(self, product, rng):
        x = _make_product_point(product, rng)
        parts = product.split(x)
        assert len(parts) == product.n_factors
        for part, dim in zip(parts, product.dims, strict=True):
            assert part.shape == (dim,)

    def test_roundtrip(self, product, rng):
        x = _make_product_point(product, rng)
        parts = product.split(x)
        reconstructed = product.combine(*parts)
        assert jnp.allclose(x, reconstructed)

    def test_combine_wrong_count_raises(self, product, rng):
        x = _make_product_point(product, rng)
        parts = product.split(x)
        with pytest.raises(ValueError, match="Expected"):
            product.combine(*parts, jnp.zeros(3))


# ===========================================================================
# 4. Decomposition correctness
# ===========================================================================


class TestDecomposition:
    """Verify that product ops equal per-factor ops concatenated."""

    def test_proj_decomposes(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        result = product.proj(x, cs)

        parts = product.split(x)
        expected = jnp.concatenate([m.proj(p, c) for m, p, c in zip(product.factors, parts, cs, strict=True)])
        assert jnp.allclose(result, expected, atol=atol)

    def test_expmap_0_decomposes(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        v = jnp.asarray(rng.normal(0, 0.1, (product.total_dim,)), dtype=product.dtype)
        result = product.expmap_0(v, cs)

        parts = product.split(v)
        expected = jnp.concatenate([m.expmap_0(p, c) for m, p, c in zip(product.factors, parts, cs, strict=True)])
        assert jnp.allclose(result, expected, atol=atol)

    def test_logmap_0_decomposes(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        result = product.logmap_0(x, cs)

        parts = product.split(x)
        expected = jnp.concatenate([m.logmap_0(p, c) for m, p, c in zip(product.factors, parts, cs, strict=True)])
        assert jnp.allclose(result, expected, atol=atol)

    def test_expmap_decomposes(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng, cs, scale=0.05)
        result = product.expmap(v, x, cs)

        v_parts, x_parts = product.split(v), product.split(x)
        expected = jnp.concatenate(
            [m.expmap(vp, xp, c) for m, vp, xp, c in zip(product.factors, v_parts, x_parts, cs, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_logmap_decomposes(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        result = product.logmap(y, x, cs)

        x_parts, y_parts = product.split(x), product.split(y)
        expected = jnp.concatenate(
            [m.logmap(yp, xp, c) for m, yp, xp, c in zip(product.factors, y_parts, x_parts, cs, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_ptransp_decomposes(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng, cs)
        result = product.ptransp(v, x, y, cs)

        v_parts, x_parts, y_parts = product.split(v), product.split(x), product.split(y)
        expected = jnp.concatenate(
            [m.ptransp(vp, xp, yp, c) for m, vp, xp, yp, c in zip(product.factors, v_parts, x_parts, y_parts, cs, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_egrad2rgrad_decomposes(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        grad = jnp.asarray(rng.normal(0, 0.1, (product.total_dim,)), dtype=product.dtype)
        result = product.egrad2rgrad(grad, x, cs)

        g_parts, x_parts = product.split(grad), product.split(x)
        expected = jnp.concatenate(
            [m.egrad2rgrad(gp, xp, c) for m, gp, xp, c in zip(product.factors, g_parts, x_parts, cs, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_addition_decomposes(self, product, rng, cs, tolerance):
        # ProductManifold.addition delegates the gyrovector addition to each factor. Every
        # factor — Hyperboloid included (Lorentz gyroaddition, Shi et al. 2026) — now supports
        # it, so this exercises the per-factor decomposition across all configs.
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        result = product.addition(x, y, cs)

        x_parts, y_parts = product.split(x), product.split(y)
        expected = jnp.concatenate(
            [m.addition(xp, yp, c) for m, xp, yp, c in zip(product.factors, x_parts, y_parts, cs, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_tangent_inner_decomposes(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        u = _make_small_tangent(product, x, rng, cs)
        v = _make_small_tangent(product, x, rng, cs)
        result = product.tangent_inner(u, v, x, cs)

        u_parts, v_parts, x_parts = product.split(u), product.split(v), product.split(x)
        expected = sum(
            m.tangent_inner(up, vp, xp, c)
            for m, up, vp, xp, c in zip(product.factors, u_parts, v_parts, x_parts, cs, strict=True)
        )
        assert jnp.allclose(result, expected, atol=atol)


# ===========================================================================
# 5. Distance properties
# ===========================================================================


class TestDistance:
    def test_self_distance_zero(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        assert jnp.allclose(product.dist(x, x, cs), 0.0, atol=atol)

    def test_positive_definite(self, product, rng, cs):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d = product.dist(x, y, cs)
        assert d > 0 or jnp.allclose(x, y)

    def test_symmetric(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        assert jnp.allclose(product.dist(x, y, cs), product.dist(y, x, cs), atol=atol)

    def test_triangle_inequality(self, product, rng, cs):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        z = _make_product_point(product, rng)
        dxy = product.dist(x, y, cs)
        dyz = product.dist(y, z, cs)
        dxz = product.dist(x, z, cs)
        assert dxz <= dxy + dyz + 1e-5

    def test_pythagorean_decomposition(self, product, rng, cs, tolerance):
        """d_P^2 == sum(d_i^2)"""
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d_product = product.dist(x, y, cs)
        d_components = product.component_dist(x, y, cs)
        expected = jnp.sqrt(jnp.sum(d_components**2))
        assert jnp.allclose(d_product, expected, atol=atol)

    def test_dist_0(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        o = product.origin(cs)
        d_from_origin = product.dist_0(x, cs)
        d_to_origin = product.dist(x, o, cs)
        assert jnp.allclose(d_from_origin, d_to_origin, atol=atol)

    def test_dist_l1(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d_l1 = product.dist_l1(x, y, cs)
        d_components = product.component_dist(x, y, cs)
        assert jnp.allclose(d_l1, jnp.sum(d_components), atol=atol)

    def test_dist_min(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d_min = product.dist_min(x, y, cs)
        d_components = product.component_dist(x, y, cs)
        assert jnp.allclose(d_min, jnp.min(d_components), atol=atol)

    def test_l2_geq_component_dists(self, product, rng, cs):
        """L2 product distance >= every component distance."""
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d_l2 = product.dist(x, y, cs)
        d_components = product.component_dist(x, y, cs)
        assert jnp.all(d_l2 >= d_components - 1e-6)


# ===========================================================================
# 6. Exp/log round-trips
# ===========================================================================


class TestRoundTrips:
    def test_expmap0_logmap0_roundtrip(self, product, rng, cs, tolerance):
        """logmap_0(expmap_0(v)) ≈ v for small v in tangent space at origin."""
        atol, _ = tolerance
        raw = jnp.asarray(rng.normal(0, 0.05, (product.total_dim,)), dtype=product.dtype)
        o = product.origin(cs)
        v = product.tangent_proj(raw, o, cs)
        x = product.expmap_0(v, cs)
        v_recovered = product.logmap_0(x, cs)
        assert jnp.allclose(v, v_recovered, atol=atol)

    def test_logmap0_expmap0_roundtrip(self, product, rng, cs, tolerance):
        """expmap_0(logmap_0(x)) ≈ x."""
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        v = product.logmap_0(x, cs)
        x_recovered = product.expmap_0(v, cs)
        assert jnp.allclose(x, x_recovered, atol=atol)

    def test_expmap_logmap_roundtrip(self, product, rng, cs, tolerance):
        """logmap(expmap(v, x), x) ≈ v for small v."""
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng, cs, scale=0.05)
        y = product.expmap(v, x, cs)
        v_recovered = product.logmap(y, x, cs)
        assert jnp.allclose(v, v_recovered, atol=atol)


# ===========================================================================
# 7. Origin
# ===========================================================================


class TestOrigin:
    def test_origin_on_manifold(self, product, cs):
        o = product.origin(cs)
        assert o.shape == (product.total_dim,)
        assert bool(product.is_in_manifold(o, cs))

    def test_origin_dist_0_is_zero(self, product, cs, tolerance):
        atol, _ = tolerance
        o = product.origin(cs)
        assert jnp.allclose(product.dist_0(o, cs), 0.0, atol=atol)

    def test_origin_matches_expmap0_zeros(self, product, cs, tolerance):
        atol, _ = tolerance
        o = product.origin(cs)
        z = jnp.zeros(product.total_dim, dtype=product.dtype)
        o2 = product.expmap_0(z, cs)
        assert jnp.allclose(o, o2, atol=atol)


# ===========================================================================
# 8. Tangent space
# ===========================================================================


class TestTangentSpace:
    def test_tangent_norm_consistent_with_inner(self, product, rng, cs, tolerance):
        """tangent_norm(v, x) == sqrt(tangent_inner(v, v, x))"""
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng, cs)
        norm = product.tangent_norm(v, x, cs)
        inner = product.tangent_inner(v, v, x, cs)
        assert jnp.allclose(norm, jnp.sqrt(jnp.maximum(inner, 0.0)), atol=atol)

    def test_projection_idempotent(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        x_proj = product.proj(x, cs)
        x_proj2 = product.proj(x_proj, cs)
        assert jnp.allclose(x_proj, x_proj2, atol=atol)

    def test_is_in_manifold(self, product, rng, cs):
        x = _make_product_point(product, rng)
        x = product.proj(x, cs)
        assert bool(product.is_in_manifold(x, cs))


# ===========================================================================
# 9. JIT / vmap
# ===========================================================================


class TestJITVmap:
    def test_dist_jit(self, product, rng, cs):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        dist_jit = jax.jit(product.dist)
        result = dist_jit(x, y, cs)
        expected = product.dist(x, y, cs)
        assert jnp.allclose(result, expected)

    def test_expmap_0_jit(self, product, rng, cs):
        v = jnp.asarray(rng.normal(0, 0.1, (product.total_dim,)), dtype=product.dtype)
        expmap_jit = jax.jit(product.expmap_0)
        result = expmap_jit(v, cs)
        expected = product.expmap_0(v, cs)
        assert jnp.allclose(result, expected)

    def test_proj_jit(self, product, rng, cs):
        x = _make_product_point(product, rng)
        proj_jit = jax.jit(product.proj)
        result = proj_jit(x, cs)
        expected = product.proj(x, cs)
        assert jnp.allclose(result, expected)

    def test_dist_vmap(self, product, rng, cs):
        """vmap over a batch of point pairs; broadcast cs with None."""
        batch_size = 8
        xs = jnp.stack([_make_product_point(product, rng) for _ in range(batch_size)])
        ys = jnp.stack([_make_product_point(product, rng) for _ in range(batch_size)])
        dist_batch = jax.vmap(product.dist, in_axes=(0, 0, None))
        results = dist_batch(xs, ys, cs)
        assert results.shape == (batch_size,)
        for i in range(batch_size):
            expected = product.dist(xs[i], ys[i], cs)
            assert jnp.allclose(results[i], expected, atol=1e-5)

    def test_expmap_0_vmap(self, product, rng, cs):
        batch_size = 8
        vs = jnp.asarray(rng.normal(0, 0.1, (batch_size, product.total_dim)), dtype=product.dtype)
        expmap_batch = jax.vmap(product.expmap_0, in_axes=(0, None))
        results = expmap_batch(vs, cs)
        assert results.shape == (batch_size, product.total_dim)


# ===========================================================================
# 10. Gradient flow
# ===========================================================================


class TestGradients:
    def test_dist_grad_finite(self, product, rng, cs):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        grad_fn = jax.grad(lambda x_: product.dist(x_, y, cs))
        g = grad_fn(x)
        assert jnp.all(jnp.isfinite(g))

    def test_expmap_0_grad_finite(self, product, rng, cs):
        raw = jnp.asarray(rng.normal(0, 0.1, (product.total_dim,)), dtype=product.dtype)
        o = product.origin(cs)
        v = product.tangent_proj(raw, o, cs)
        y = _make_product_point(product, rng)
        grad_fn = jax.grad(lambda v_: product.dist(product.expmap_0(v_, cs), y, cs))
        g = grad_fn(v)
        assert jnp.all(jnp.isfinite(g))

    def test_learnable_curvature_gradient(self):
        """Gradients flow to per-factor LearnableCurvature instances via the
        ProductManifold's ``cs`` argument (no factor-wise workaround).

        Verifies:
          1) Grad leaves are finite and non-trivial.
          2) An SGD step actually moves the curvatures.
        """
        pm = ProductManifold(
            (Hyperboloid(c=1.0, dtype=jnp.float64), 3),
            (Poincare(c=0.5, dtype=jnp.float64), 2),
            dtype=jnp.float64,
        )
        rng_np = np.random.default_rng(123)
        x = _make_product_point(pm, rng_np)
        y = _make_product_point(pm, rng_np)

        class CurvModel(nnx.Module):
            def __init__(self):
                self.pm = pm
                self.curv_h = LearnableCurvature(init_c=1.0)
                self.curv_p = LearnableCurvature(init_c=0.5)

            def __call__(self, x_, y_):
                cs = (self.curv_h(), self.curv_p())
                return self.pm.dist(x_, y_, cs)

        model = CurvModel()
        c_before = jnp.array([float(model.curv_h()), float(model.curv_p())])

        def loss_fn(m):
            return m(x, y)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        grad_state = nnx.state(grads, nnx.Param)
        grad_leaves = jax.tree.leaves(grad_state)
        assert len(grad_leaves) == 2, f"expected 2 param gradients, got {len(grad_leaves)}"
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
        assert any(jnp.any(jnp.abs(g) > 1e-10) for g in grad_leaves)

        optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=1.0), wrt=nnx.Param)
        optimizer.update(model, grads)
        c_after = jnp.array([float(model.curv_h()), float(model.curv_p())])
        assert jnp.all(jnp.isfinite(c_after))
        assert jnp.any(jnp.abs(c_after - c_before) > 1e-6)


# ===========================================================================
# 11. Edge cases / Equivalence
# ===========================================================================


class TestEdgeCases:
    def test_single_factor_matches_base(self):
        """Single-factor product should behave like the factor alone."""
        h = Hyperboloid(c=1.0, dtype=jnp.float64)
        pm = ProductManifold((h, 5), dtype=jnp.float64)
        cs = (1.0,)
        rng = np.random.default_rng(99)

        spatial = rng.normal(0, 0.3, (4,)).astype(np.float64)
        ambient = np.concatenate([[1.5], spatial])
        x = jnp.asarray(h.proj(jnp.array(ambient), 1.0))
        spatial2 = rng.normal(0, 0.3, (4,)).astype(np.float64)
        ambient2 = np.concatenate([[1.5], spatial2])
        y = jnp.asarray(h.proj(jnp.array(ambient2), 1.0))

        d_base = h.dist(x, y, 1.0)
        d_product = pm.dist(x, y, cs)
        assert jnp.allclose(d_base, d_product, atol=1e-7)

        v = jnp.asarray(rng.normal(0, 0.05, (5,)), dtype=jnp.float64)
        v = h.tangent_proj(v, x, 1.0)
        exp_base = h.expmap(v, x, 1.0)
        exp_product = pm.expmap(v, x, cs)
        assert jnp.allclose(exp_base, exp_product, atol=1e-7)

    def test_all_euclidean_matches_flat(self):
        """Product of Euclidean factors acts like higher-dim Euclidean."""
        e1 = Euclidean(dtype=jnp.float64)
        e2 = Euclidean(dtype=jnp.float64)
        pm = ProductManifold((e1, 3), (e2, 4), dtype=jnp.float64)
        cs = pm.curvatures  # (0.0, 0.0)
        e_flat = Euclidean(dtype=jnp.float64)

        rng = np.random.default_rng(42)
        x = jnp.asarray(rng.normal(0, 1, (7,)), dtype=jnp.float64)
        y = jnp.asarray(rng.normal(0, 1, (7,)), dtype=jnp.float64)

        d_product = pm.dist(x, y, cs)
        d_flat = e_flat.dist(x, y, 0.0)
        assert jnp.allclose(d_product, d_flat, atol=1e-10)

    def test_parallel_transport_preserves_norm(self, product, rng, cs, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng, cs, scale=0.05)

        norm_before = product.tangent_norm(v, x, cs)
        v_transported = product.ptransp(v, x, y, cs)
        norm_after = product.tangent_norm(v_transported, y, cs)
        assert jnp.allclose(norm_before, norm_after, atol=atol)
