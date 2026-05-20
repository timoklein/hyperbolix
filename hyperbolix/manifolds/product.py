"""Product manifold - combines multiple sub-manifolds into a single space.

A product manifold P = M1 x M2 x ... x Mn where each Mi can be any manifold
(Poincaré, Hyperboloid, Euclidean, ProperVelocity) with its own curvature.

Points are represented as flat concatenated arrays of shape (total_dim,).
All single-point operations follow the vmap-native pattern: use jax.vmap
for batching.

JIT Compilation & Batching
---------------------------
Python for-loops over factors unroll at JIT trace time since the product
structure is static. Each iteration traces a different code path (different
manifold type), which is correct and efficient.

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.manifolds import ProductManifold, Hyperboloid, Poincare
    >>>
    >>> product = ProductManifold((Hyperboloid(c=1.0), 5), (Poincare(c=0.1), 3))
    >>> x = product.origin()  # shape (8,)
    >>> parts = product.split(x)  # (array(5,), array(3,))
    >>>
    >>> # Batch distance with vmap
    >>> dist_batch = jax.vmap(product.dist, in_axes=(0, 0, None))
    >>> distances = dist_batch(x_batch, y_batch, 0.0)  # c is ignored

References:
    Gu et al. "Learning Mixed-Curvature Representations in Product Spaces."
    ICLR 2019.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._base import ManifoldBase


class ProductManifold:
    """Product manifold P = M1 x M2 x ... x Mn.

    Each factor is specified as ``(manifold_instance, dim)`` where ``dim`` is
    the point dimension (array slice width): ambient for Hyperboloid (d+1),
    spatial for Poincaré/Euclidean/ProperVelocity (d).

    Curvature is managed per-factor via each sub-manifold's ``.c`` property.
    The ``c`` parameter on protocol methods is accepted for compatibility but
    ignored — each factor uses its own curvature.

    Args:
        *factors: Tuples of ``(manifold_instance, dim)``. At least one required.
        dtype: Target JAX dtype for computations (default: jnp.float32).

    Examples:
        >>> product = ProductManifold((Hyperboloid(c=1.0), 5), (Poincare(c=0.1), 3))
        >>> x = product.origin()           # shape (8,)
        >>> parts = product.split(x)       # (array(5,), array(3,))
        >>> d = product.dist(x, y, c=0.0)  # c ignored, uses per-factor curvatures

        From a signature (repeated factors):

        >>> product = ProductManifold.from_signature(
        ...     (Hyperboloid, 3, 8, 1.0),  # 8 copies of 3D-ambient Hyperboloid
        ...     (Poincare, 2, 4, 0.1),      # 4 copies of 2D Poincaré
        ... )
    """

    def __init__(
        self,
        *factors: tuple[ManifoldBase, int],
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if len(factors) < 1:
            raise ValueError("ProductManifold requires at least one factor")

        self.dtype = dtype

        manifolds = []
        dims = []
        slices = []
        pos = 0
        for i, (manifold, dim) in enumerate(factors):
            if not isinstance(manifold, ManifoldBase):
                raise TypeError(f"Factor {i}: expected ManifoldBase instance, got {type(manifold).__name__}")
            if dim < 1:
                raise ValueError(f"Factor {i}: dim must be >= 1, got {dim}")
            manifolds.append(manifold)
            dims.append(dim)
            slices.append(slice(pos, pos + dim))
            pos += dim

        self._factors = tuple(manifolds)
        self._dims = tuple(dims)
        self._slices = tuple(slices)
        self._total_dim = pos

    def _cast(self, x: Array) -> Array:
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact):
            return x.astype(self.dtype)
        return x

    # -- Properties --------------------------------------------------------

    @property
    def c(self) -> None:
        raise TypeError(
            "ProductManifold has per-factor curvatures. Use .curvatures to get all or .factors[i].c for a specific factor."
        )

    @property
    def total_dim(self) -> int:
        return self._total_dim

    @property
    def n_factors(self) -> int:
        return len(self._factors)

    @property
    def dims(self) -> tuple[int, ...]:
        return self._dims

    @property
    def factors(self) -> tuple[ManifoldBase, ...]:
        return self._factors

    @property
    def curvatures(self) -> tuple[float | jax.Array, ...]:
        return tuple(m.c for m in self._factors)

    # -- Split / Combine ---------------------------------------------------

    def split(self, x: Float[Array, "total_dim"]) -> tuple[Array, ...]:
        """Split a product point into per-factor components."""
        return tuple(x[s] for s in self._slices)

    def combine(self, *parts: Array) -> Float[Array, "total_dim"]:
        """Combine per-factor components into a product point."""
        if len(parts) != len(self._factors):
            raise ValueError(f"Expected {len(self._factors)} parts, got {len(parts)}")
        return jnp.concatenate(list(parts))

    # -- Origin ------------------------------------------------------------

    def origin(self) -> Float[Array, "total_dim"]:
        """Construct the product origin point.

        Uses expmap_0(zeros) per factor, which returns the manifold origin
        for all manifold types without isinstance checks.
        """
        parts = []
        for m, dim in zip(self._factors, self._dims, strict=True):
            zero_tangent = jnp.zeros(dim, dtype=self.dtype)
            parts.append(m.expmap_0(zero_tangent, m.c))
        return jnp.concatenate(parts)

    # -- Geometry (per-factor decomposable) --------------------------------

    def proj(self, x: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, "total_dim"]:
        """Project point onto product manifold (per-factor projection)."""
        del c
        x = self._cast(x)
        parts = self.split(x)
        return jnp.concatenate([m.proj(p, m.c) for m, p in zip(self._factors, parts, strict=True)])

    def addition(
        self, x: Float[Array, "total_dim"], y: Float[Array, "total_dim"], c: float = 0.0
    ) -> Float[Array, "total_dim"]:
        """Manifold addition (per-factor)."""
        del c
        x, y = self._cast(x), self._cast(y)
        x_parts, y_parts = self.split(x), self.split(y)
        return jnp.concatenate([m.addition(xp, yp, m.c) for m, xp, yp in zip(self._factors, x_parts, y_parts, strict=True)])

    def scalar_mul(self, r: float, x: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, "total_dim"]:
        """Scalar multiplication (same scalar r applied to all factors)."""
        del c
        x = self._cast(x)
        parts = self.split(x)
        return jnp.concatenate([m.scalar_mul(r, p, m.c) for m, p in zip(self._factors, parts, strict=True)])

    # -- Exponential / logarithmic maps ------------------------------------

    def expmap(self, v: Float[Array, "total_dim"], x: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, "total_dim"]:
        """Exponential map (per-factor)."""
        del c
        v, x = self._cast(v), self._cast(x)
        v_parts, x_parts = self.split(v), self.split(x)
        return jnp.concatenate([m.expmap(vp, xp, m.c) for m, vp, xp in zip(self._factors, v_parts, x_parts, strict=True)])

    def expmap_0(self, v: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, "total_dim"]:
        """Exponential map from origin (per-factor)."""
        del c
        v = self._cast(v)
        parts = self.split(v)
        return jnp.concatenate([m.expmap_0(p, m.c) for m, p in zip(self._factors, parts, strict=True)])

    def logmap(self, y: Float[Array, "total_dim"], x: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, "total_dim"]:
        """Logarithmic map (per-factor)."""
        del c
        y, x = self._cast(y), self._cast(x)
        y_parts, x_parts = self.split(y), self.split(x)
        return jnp.concatenate([m.logmap(yp, xp, m.c) for m, yp, xp in zip(self._factors, y_parts, x_parts, strict=True)])

    def logmap_0(self, y: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, "total_dim"]:
        """Logarithmic map to origin (per-factor)."""
        del c
        y = self._cast(y)
        parts = self.split(y)
        return jnp.concatenate([m.logmap_0(p, m.c) for m, p in zip(self._factors, parts, strict=True)])

    def retraction(
        self, v: Float[Array, "total_dim"], x: Float[Array, "total_dim"], c: float = 0.0
    ) -> Float[Array, "total_dim"]:
        """Retraction (per-factor)."""
        del c
        v, x = self._cast(v), self._cast(x)
        v_parts, x_parts = self.split(v), self.split(x)
        return jnp.concatenate([m.retraction(vp, xp, m.c) for m, vp, xp in zip(self._factors, v_parts, x_parts, strict=True)])

    # -- Transport / tangent space -----------------------------------------

    def ptransp(
        self,
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
        c: float = 0.0,
    ) -> Float[Array, "total_dim"]:
        """Parallel transport of v from x to y (per-factor)."""
        del c
        v, x, y = self._cast(v), self._cast(x), self._cast(y)
        v_parts, x_parts, y_parts = self.split(v), self.split(x), self.split(y)
        return jnp.concatenate(
            [m.ptransp(vp, xp, yp, m.c) for m, vp, xp, yp in zip(self._factors, v_parts, x_parts, y_parts, strict=True)]
        )

    def ptransp_0(
        self, v: Float[Array, "total_dim"], y: Float[Array, "total_dim"], c: float = 0.0
    ) -> Float[Array, "total_dim"]:
        """Parallel transport from origin to y (per-factor)."""
        del c
        v, y = self._cast(v), self._cast(y)
        v_parts, y_parts = self.split(v), self.split(y)
        return jnp.concatenate([m.ptransp_0(vp, yp, m.c) for m, vp, yp in zip(self._factors, v_parts, y_parts, strict=True)])

    def tangent_inner(
        self,
        u: Float[Array, "total_dim"],
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: float = 0.0,
    ) -> Float[Array, ""]:
        """Riemannian inner product (sum of per-factor inner products)."""
        del c
        u, v, x = self._cast(u), self._cast(v), self._cast(x)
        u_parts, v_parts, x_parts = self.split(u), self.split(v), self.split(x)
        inners = jnp.stack(
            [m.tangent_inner(up, vp, xp, m.c) for m, up, vp, xp in zip(self._factors, u_parts, v_parts, x_parts, strict=True)]
        )
        return jnp.sum(inners)

    def tangent_norm(self, v: Float[Array, "total_dim"], x: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, ""]:
        """Riemannian norm (sqrt of tangent inner product with itself)."""
        return jnp.sqrt(jnp.maximum(self.tangent_inner(v, v, x, c), 0.0))

    def egrad2rgrad(
        self, grad: Float[Array, "total_dim"], x: Float[Array, "total_dim"], c: float = 0.0
    ) -> Float[Array, "total_dim"]:
        """Convert Euclidean gradient to Riemannian gradient (per-factor)."""
        del c
        grad, x = self._cast(grad), self._cast(x)
        g_parts, x_parts = self.split(grad), self.split(x)
        return jnp.concatenate([m.egrad2rgrad(gp, xp, m.c) for m, gp, xp in zip(self._factors, g_parts, x_parts, strict=True)])

    def tangent_proj(
        self, v: Float[Array, "total_dim"], x: Float[Array, "total_dim"], c: float = 0.0
    ) -> Float[Array, "total_dim"]:
        """Project vector onto tangent space (per-factor)."""
        del c
        v, x = self._cast(v), self._cast(x)
        v_parts, x_parts = self.split(v), self.split(x)
        return jnp.concatenate(
            [m.tangent_proj(vp, xp, m.c) for m, vp, xp in zip(self._factors, v_parts, x_parts, strict=True)]
        )

    # -- Distance ----------------------------------------------------------

    def component_dist(
        self,
        x: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
    ) -> Float[Array, "n_factors"]:
        """Per-factor geodesic distances as a vector."""
        x, y = self._cast(x), self._cast(y)
        x_parts, y_parts = self.split(x), self.split(y)
        return jnp.stack([m.dist(xp, yp, m.c) for m, xp, yp in zip(self._factors, x_parts, y_parts, strict=True)])

    def dist(self, x: Float[Array, "total_dim"], y: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, ""]:
        """Product distance (L2/Riemannian): d = sqrt(sum d_i^2)."""
        del c
        d_per_factor = self.component_dist(x, y)
        return jnp.sqrt(jnp.sum(d_per_factor**2))

    def dist_0(self, x: Float[Array, "total_dim"], c: float = 0.0) -> Float[Array, ""]:
        """Distance from origin (L2/Riemannian)."""
        del c
        x = self._cast(x)
        parts = self.split(x)
        d_sq = jnp.stack([m.dist_0(p, m.c) ** 2 for m, p in zip(self._factors, parts, strict=True)])
        return jnp.sqrt(jnp.sum(d_sq))

    def dist_l1(self, x: Float[Array, "total_dim"], y: Float[Array, "total_dim"]) -> Float[Array, ""]:
        """L1 product distance: d = sum d_i."""
        return jnp.sum(self.component_dist(x, y))

    def dist_min(self, x: Float[Array, "total_dim"], y: Float[Array, "total_dim"]) -> Float[Array, ""]:
        """Min product distance: d = min d_i.

        Warning:
            This is NOT a true metric: positive-definiteness fails. Two
            distinct points ``x != y`` that agree on any single factor
            (``x_i == y_i``) yield ``d = 0``. Use only when the
            "closest-factor" interpretation is semantically intended;
            for an actual distance use ``dist`` (L2) or ``dist_l1``.
        """
        return jnp.min(self.component_dist(x, y))

    # -- Validation --------------------------------------------------------

    def is_in_manifold(self, x: Float[Array, "total_dim"], c: float = 0.0) -> Array:
        """Check if point is on product manifold (all factors valid)."""
        del c
        x = self._cast(x)
        parts = self.split(x)
        checks = [m.is_in_manifold(p, m.c) for m, p in zip(self._factors, parts, strict=True)]
        return jnp.all(jnp.stack(checks))

    def is_in_tangent_space(self, v: Float[Array, "total_dim"], x: Float[Array, "total_dim"], c: float = 0.0) -> Array:
        """Check if vector is in tangent space at x (all factors valid)."""
        del c
        v, x = self._cast(v), self._cast(x)
        v_parts, x_parts = self.split(v), self.split(x)
        checks = [m.is_in_tangent_space(vp, xp, m.c) for m, vp, xp in zip(self._factors, v_parts, x_parts, strict=True)]
        return jnp.all(jnp.stack(checks))

    # -- Factory methods ---------------------------------------------------

    @classmethod
    def from_signature(
        cls,
        *specs: tuple,
        dtype: jnp.dtype = jnp.float32,
    ) -> "ProductManifold":
        """Create product manifold from a signature specification.

        Each spec is one of:
          - ``(ManifoldClass, dim, count)`` — uses ``c=1.0``.
          - ``(ManifoldClass, dim, count, curvature)`` — sets ``c``.

        Euclidean factors silently ignore the ``curvature`` field
        (Euclidean has fixed ``c=0``; see ``Euclidean.__init__``).

        Args:
            *specs: Signature tuples (3- or 4-tuple as described above).
            dtype: Target JAX dtype.

        Returns:
            ProductManifold instance.

        Examples:
            >>> pm = ProductManifold.from_signature(
            ...     (Hyperboloid, 3, 8, 1.0),
            ...     (Poincare, 2, 4, 0.1),
            ...     (Euclidean, 8, 1),
            ... )
        """
        from .euclidean import Euclidean

        factors: list[tuple[ManifoldBase, int]] = []
        for spec in specs:
            if len(spec) == 3:
                manifold_cls, dim, count = spec
                curvature = 1.0
            elif len(spec) == 4:
                manifold_cls, dim, count, curvature = spec
            else:
                raise ValueError(
                    "Expected 3-tuple (ManifoldClass, dim, count) or 4-tuple "
                    f"(ManifoldClass, dim, count, curvature); got {len(spec)}-tuple"
                )

            for _ in range(count):
                if issubclass(manifold_cls, Euclidean):
                    instance = manifold_cls(dtype=dtype)
                else:
                    instance = manifold_cls(dtype=dtype, c=curvature)
                factors.append((instance, dim))

        return cls(*factors, dtype=dtype)

    # -- Repr --------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for m, d in zip(self._factors, self._dims, strict=True):
            name = type(m).__name__
            parts.append(f"{name}(dim={d}, c={m.c})")
        signature = " x ".join(parts)
        return f"ProductManifold({signature}, total_dim={self._total_dim})"
