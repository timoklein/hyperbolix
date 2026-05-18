"""Tests for ProductManifold.

Covers construction, protocol compliance, decomposition correctness,
distance properties, exp/log round-trips, origin, JIT/vmap, gradient flow,
and edge cases (single-factor, all-Euclidean).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _make_small_tangent(product: ProductManifold, x: jnp.ndarray, rng: np.random.Generator, scale: float = 0.1) -> jnp.ndarray:
    """Generate a small tangent vector at x on the product manifold."""
    raw = jnp.asarray(rng.normal(0, scale, size=(product.total_dim,)), dtype=product.dtype)
    return product.tangent_proj(raw, x, 0.0)


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
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def two_points(product, rng):
    """Two valid points on the product manifold."""
    x = _make_product_point(product, rng)
    y = _make_product_point(product, rng)
    return x, y


@pytest.fixture
def point_and_tangent(product, rng):
    """A valid point and a small tangent vector at that point."""
    x = _make_product_point(product, rng)
    v = _make_small_tangent(product, x, rng)
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

    def test_c_property_raises(self):
        p = ProductManifold((Poincare(), 3), (Hyperboloid(), 5))
        with pytest.raises(TypeError, match="per-factor curvatures"):
            _ = p.c

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

    def test_from_signature_learnable(self):
        pm = ProductManifold.from_signature(
            (Hyperboloid, 3, 2, 1.0),
            learnable=True,
        )
        assert all(hasattr(f, "_c_raw") for f in pm.factors)

    def test_repr(self):
        p = ProductManifold((Hyperboloid(c=1.0), 5), (Poincare(c=0.1), 3))
        r = repr(p)
        assert "Hyperboloid" in r
        assert "Poincare" in r
        assert "total_dim=8" in r


# ===========================================================================
# 2. Protocol compliance
# ===========================================================================


class TestProtocol:
    def test_isinstance_manifold(self, product):
        assert isinstance(product, Manifold)

    def test_has_all_protocol_methods(self, product):
        required = [
            "proj",
            "dist",
            "dist_0",
            "addition",
            "scalar_mul",
            "expmap",
            "expmap_0",
            "logmap",
            "logmap_0",
            "retraction",
            "ptransp",
            "ptransp_0",
            "tangent_inner",
            "tangent_norm",
            "egrad2rgrad",
            "tangent_proj",
            "is_in_manifold",
            "is_in_tangent_space",
        ]
        for name in required:
            assert hasattr(product, name), f"Missing protocol method: {name}"


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

    def test_proj_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        result = product.proj(x, 0.0)

        parts = product.split(x)
        expected = jnp.concatenate([m.proj(p, m.c) for m, p in zip(product.factors, parts, strict=True)])
        assert jnp.allclose(result, expected, atol=atol)

    def test_expmap_0_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        v = jnp.asarray(rng.normal(0, 0.1, (product.total_dim,)), dtype=product.dtype)
        result = product.expmap_0(v, 0.0)

        parts = product.split(v)
        expected = jnp.concatenate([m.expmap_0(p, m.c) for m, p in zip(product.factors, parts, strict=True)])
        assert jnp.allclose(result, expected, atol=atol)

    def test_logmap_0_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        result = product.logmap_0(x, 0.0)

        parts = product.split(x)
        expected = jnp.concatenate([m.logmap_0(p, m.c) for m, p in zip(product.factors, parts, strict=True)])
        assert jnp.allclose(result, expected, atol=atol)

    def test_expmap_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng, scale=0.05)
        result = product.expmap(v, x, 0.0)

        v_parts, x_parts = product.split(v), product.split(x)
        expected = jnp.concatenate(
            [m.expmap(vp, xp, m.c) for m, vp, xp in zip(product.factors, v_parts, x_parts, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_logmap_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        result = product.logmap(y, x, 0.0)

        x_parts, y_parts = product.split(x), product.split(y)
        expected = jnp.concatenate(
            [m.logmap(yp, xp, m.c) for m, yp, xp in zip(product.factors, y_parts, x_parts, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_ptransp_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng)
        result = product.ptransp(v, x, y, 0.0)

        v_parts, x_parts, y_parts = product.split(v), product.split(x), product.split(y)
        expected = jnp.concatenate(
            [m.ptransp(vp, xp, yp, m.c) for m, vp, xp, yp in zip(product.factors, v_parts, x_parts, y_parts, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_egrad2rgrad_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        grad = jnp.asarray(rng.normal(0, 0.1, (product.total_dim,)), dtype=product.dtype)
        result = product.egrad2rgrad(grad, x, 0.0)

        g_parts, x_parts = product.split(grad), product.split(x)
        expected = jnp.concatenate(
            [m.egrad2rgrad(gp, xp, m.c) for m, gp, xp in zip(product.factors, g_parts, x_parts, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_addition_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        result = product.addition(x, y, 0.0)

        x_parts, y_parts = product.split(x), product.split(y)
        expected = jnp.concatenate(
            [m.addition(xp, yp, m.c) for m, xp, yp in zip(product.factors, x_parts, y_parts, strict=True)]
        )
        assert jnp.allclose(result, expected, atol=atol)

    def test_tangent_inner_decomposes(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        u = _make_small_tangent(product, x, rng)
        v = _make_small_tangent(product, x, rng)
        result = product.tangent_inner(u, v, x, 0.0)

        u_parts, v_parts, x_parts = product.split(u), product.split(v), product.split(x)
        expected = sum(
            m.tangent_inner(up, vp, xp, m.c) for m, up, vp, xp in zip(product.factors, u_parts, v_parts, x_parts, strict=True)
        )
        assert jnp.allclose(result, expected, atol=atol)


# ===========================================================================
# 5. Distance properties
# ===========================================================================


class TestDistance:
    def test_self_distance_zero(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        assert jnp.allclose(product.dist(x, x, 0.0), 0.0, atol=atol)

    def test_positive_definite(self, product, rng):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d = product.dist(x, y, 0.0)
        assert d > 0 or jnp.allclose(x, y)

    def test_symmetric(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        assert jnp.allclose(product.dist(x, y, 0.0), product.dist(y, x, 0.0), atol=atol)

    def test_triangle_inequality(self, product, rng):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        z = _make_product_point(product, rng)
        dxy = product.dist(x, y, 0.0)
        dyz = product.dist(y, z, 0.0)
        dxz = product.dist(x, z, 0.0)
        assert dxz <= dxy + dyz + 1e-5

    def test_pythagorean_decomposition(self, product, rng, tolerance):
        """d_P^2 == sum(d_i^2)"""
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d_product = product.dist(x, y, 0.0)
        d_components = product.component_dist(x, y)
        expected = jnp.sqrt(jnp.sum(d_components**2))
        assert jnp.allclose(d_product, expected, atol=atol)

    def test_dist_0(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        o = product.origin()
        d_from_origin = product.dist_0(x, 0.0)
        d_to_origin = product.dist(x, o, 0.0)
        assert jnp.allclose(d_from_origin, d_to_origin, atol=atol)

    def test_dist_l1(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d_l1 = product.dist_l1(x, y)
        d_components = product.component_dist(x, y)
        assert jnp.allclose(d_l1, jnp.sum(d_components), atol=atol)

    def test_dist_min(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d_min = product.dist_min(x, y)
        d_components = product.component_dist(x, y)
        assert jnp.allclose(d_min, jnp.min(d_components), atol=atol)

    def test_l2_geq_component_dists(self, product, rng):
        """L2 product distance >= every component distance."""
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        d_l2 = product.dist(x, y, 0.0)
        d_components = product.component_dist(x, y)
        assert jnp.all(d_l2 >= d_components - 1e-6)


# ===========================================================================
# 6. Exp/log round-trips
# ===========================================================================


class TestRoundTrips:
    def test_expmap0_logmap0_roundtrip(self, product, rng, tolerance):
        """logmap_0(expmap_0(v)) ≈ v for small v in tangent space at origin."""
        atol, _ = tolerance
        raw = jnp.asarray(rng.normal(0, 0.05, (product.total_dim,)), dtype=product.dtype)
        o = product.origin()
        v = product.tangent_proj(raw, o, 0.0)
        x = product.expmap_0(v, 0.0)
        v_recovered = product.logmap_0(x, 0.0)
        assert jnp.allclose(v, v_recovered, atol=atol)

    def test_logmap0_expmap0_roundtrip(self, product, rng, tolerance):
        """expmap_0(logmap_0(x)) ≈ x."""
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        v = product.logmap_0(x, 0.0)
        x_recovered = product.expmap_0(v, 0.0)
        assert jnp.allclose(x, x_recovered, atol=atol)

    def test_expmap_logmap_roundtrip(self, product, rng, tolerance):
        """logmap(expmap(v, x), x) ≈ v for small v."""
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng, scale=0.05)
        y = product.expmap(v, x, 0.0)
        v_recovered = product.logmap(y, x, 0.0)
        assert jnp.allclose(v, v_recovered, atol=atol)


# ===========================================================================
# 7. Origin
# ===========================================================================


class TestOrigin:
    def test_origin_on_manifold(self, product):
        o = product.origin()
        assert o.shape == (product.total_dim,)
        assert bool(product.is_in_manifold(o, 0.0))

    def test_origin_dist_0_is_zero(self, product, tolerance):
        atol, _ = tolerance
        o = product.origin()
        assert jnp.allclose(product.dist_0(o, 0.0), 0.0, atol=atol)

    def test_origin_matches_expmap0_zeros(self, product, tolerance):
        atol, _ = tolerance
        o = product.origin()
        z = jnp.zeros(product.total_dim, dtype=product.dtype)
        o2 = product.expmap_0(z, 0.0)
        assert jnp.allclose(o, o2, atol=atol)


# ===========================================================================
# 8. Tangent space
# ===========================================================================


class TestTangentSpace:
    def test_tangent_norm_consistent_with_inner(self, product, rng, tolerance):
        """tangent_norm(v, x) == sqrt(tangent_inner(v, v, x))"""
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng)
        norm = product.tangent_norm(v, x, 0.0)
        inner = product.tangent_inner(v, v, x, 0.0)
        assert jnp.allclose(norm, jnp.sqrt(jnp.maximum(inner, 0.0)), atol=atol)

    def test_projection_idempotent(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        x_proj = product.proj(x, 0.0)
        x_proj2 = product.proj(x_proj, 0.0)
        assert jnp.allclose(x_proj, x_proj2, atol=atol)

    def test_is_in_manifold(self, product, rng):
        x = _make_product_point(product, rng)
        x = product.proj(x, 0.0)
        assert bool(product.is_in_manifold(x, 0.0))


# ===========================================================================
# 9. JIT / vmap
# ===========================================================================


class TestJITVmap:
    def test_dist_jit(self, product, rng):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        dist_jit = jax.jit(product.dist, static_argnames=[])
        result = dist_jit(x, y, 0.0)
        expected = product.dist(x, y, 0.0)
        assert jnp.allclose(result, expected)

    def test_expmap_0_jit(self, product, rng):
        v = jnp.asarray(rng.normal(0, 0.1, (product.total_dim,)), dtype=product.dtype)
        expmap_jit = jax.jit(product.expmap_0)
        result = expmap_jit(v, 0.0)
        expected = product.expmap_0(v, 0.0)
        assert jnp.allclose(result, expected)

    def test_proj_jit(self, product, rng):
        x = _make_product_point(product, rng)
        proj_jit = jax.jit(product.proj)
        result = proj_jit(x, 0.0)
        expected = product.proj(x, 0.0)
        assert jnp.allclose(result, expected)

    def test_dist_vmap(self, product, rng):
        """vmap over a batch of point pairs."""
        batch_size = 8
        xs = jnp.stack([_make_product_point(product, rng) for _ in range(batch_size)])
        ys = jnp.stack([_make_product_point(product, rng) for _ in range(batch_size)])
        dist_batch = jax.vmap(product.dist, in_axes=(0, 0, None))
        results = dist_batch(xs, ys, 0.0)
        assert results.shape == (batch_size,)
        for i in range(batch_size):
            expected = product.dist(xs[i], ys[i], 0.0)
            assert jnp.allclose(results[i], expected, atol=1e-5)

    def test_expmap_0_vmap(self, product, rng):
        batch_size = 8
        vs = jnp.asarray(rng.normal(0, 0.1, (batch_size, product.total_dim)), dtype=product.dtype)
        expmap_batch = jax.vmap(product.expmap_0, in_axes=(0, None))
        results = expmap_batch(vs, 0.0)
        assert results.shape == (batch_size, product.total_dim)


# ===========================================================================
# 10. Gradient flow
# ===========================================================================


class TestGradients:
    def test_dist_grad_finite(self, product, rng):
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        grad_fn = jax.grad(lambda x_: product.dist(x_, y, 0.0))
        g = grad_fn(x)
        assert jnp.all(jnp.isfinite(g))

    def test_expmap_0_grad_finite(self, product, rng):
        raw = jnp.asarray(rng.normal(0, 0.1, (product.total_dim,)), dtype=product.dtype)
        o = product.origin()
        v = product.tangent_proj(raw, o, 0.0)
        y = _make_product_point(product, rng)
        grad_fn = jax.grad(lambda v_: product.dist(product.expmap_0(v_, 0.0), y, 0.0))
        g = grad_fn(v)
        assert jnp.all(jnp.isfinite(g))

    def test_learnable_curvature_gradient(self):
        """Gradients flow to learnable curvature parameters."""
        pm = ProductManifold(
            (Hyperboloid(c=1.0, learnable=True, dtype=jnp.float64), 3),
            (Poincare(c=0.5, learnable=True, dtype=jnp.float64), 2),
            dtype=jnp.float64,
        )
        rng = np.random.default_rng(123)
        x = _make_product_point(pm, rng)
        y = _make_product_point(pm, rng)

        def loss_fn(model):
            return model.dist(x, y, 0.0)

        grads = jax.grad(loss_fn)(pm)
        c_raw_grads = [grads._factors[i]._c_raw[...] for i in range(2)]
        assert all(jnp.isfinite(g) for g in c_raw_grads)
        assert any(jnp.abs(g) > 1e-10 for g in c_raw_grads)


# ===========================================================================
# 11. Edge cases / Equivalence
# ===========================================================================


class TestEdgeCases:
    def test_single_factor_matches_base(self):
        """Single-factor product should behave like the factor alone."""
        h = Hyperboloid(c=1.0, dtype=jnp.float64)
        pm = ProductManifold((h, 5), dtype=jnp.float64)
        rng = np.random.default_rng(99)

        spatial = rng.normal(0, 0.3, (4,)).astype(np.float64)
        ambient = np.concatenate([[1.5], spatial])
        x = jnp.asarray(h.proj(jnp.array(ambient), 1.0))
        spatial2 = rng.normal(0, 0.3, (4,)).astype(np.float64)
        ambient2 = np.concatenate([[1.5], spatial2])
        y = jnp.asarray(h.proj(jnp.array(ambient2), 1.0))

        d_base = h.dist(x, y, 1.0)
        d_product = pm.dist(x, y, 0.0)
        assert jnp.allclose(d_base, d_product, atol=1e-7)

        v = jnp.asarray(rng.normal(0, 0.05, (5,)), dtype=jnp.float64)
        v = h.tangent_proj(v, x, 1.0)
        exp_base = h.expmap(v, x, 1.0)
        exp_product = pm.expmap(v, x, 0.0)
        assert jnp.allclose(exp_base, exp_product, atol=1e-7)

    def test_all_euclidean_matches_flat(self):
        """Product of Euclidean factors acts like higher-dim Euclidean."""
        e1 = Euclidean(dtype=jnp.float64)
        e2 = Euclidean(dtype=jnp.float64)
        pm = ProductManifold((e1, 3), (e2, 4), dtype=jnp.float64)
        e_flat = Euclidean(dtype=jnp.float64)

        rng = np.random.default_rng(42)
        x = jnp.asarray(rng.normal(0, 1, (7,)), dtype=jnp.float64)
        y = jnp.asarray(rng.normal(0, 1, (7,)), dtype=jnp.float64)

        d_product = pm.dist(x, y, 0.0)
        d_flat = e_flat.dist(x, y, 0.0)
        assert jnp.allclose(d_product, d_flat, atol=1e-10)

    def test_parallel_transport_preserves_norm(self, product, rng, tolerance):
        atol, _ = tolerance
        x = _make_product_point(product, rng)
        y = _make_product_point(product, rng)
        v = _make_small_tangent(product, x, rng, scale=0.05)

        norm_before = product.tangent_norm(v, x, 0.0)
        v_transported = product.ptransp(v, x, y, 0.0)
        norm_after = product.tangent_norm(v_transported, y, 0.0)
        assert jnp.allclose(norm_before, norm_after, atol=atol)
