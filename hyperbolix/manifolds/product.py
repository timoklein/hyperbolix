"""Product manifold - combines multiple sub-manifolds into a single space.

A product manifold P = M1 x M2 x ... x Mn where each Mi can be any manifold
(Poincaré, Hyperboloid, Euclidean, ProperVelocity). Each factor has its own
curvature, supplied **at call time** as a per-factor sequence ``c`` — there
is no scalar broadcast.

Points are represented as flat concatenated arrays of shape (total_dim,).
All single-point operations follow the vmap-native pattern: use jax.vmap
for batching.

Per-factor curvature contract
-----------------------------
Every geometry method takes a positional ``c: Curvature`` argument that must
be a sequence of length ``n_factors``. ``c[i]`` is routed into the i-th
factor. There is no default and no scalar broadcast — callers must pass the
per-factor sequence explicitly. Use ``product.curvatures`` to read the
factor-stored values as a static default, or pass the outputs of
``LearnableCurvature`` modules for trainable curvatures.

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from hyperbolix.manifolds import ProductManifold, Hyperboloid, Poincare
    >>>
    >>> product = ProductManifold((Hyperboloid(c=1.0), 5), (Poincare(c=0.1), 3))
    >>> c = product.curvatures                   # (1.0, 0.1)
    >>> x = product.origin(c)                    # shape (8,)
    >>> parts = product.split(x)                 # (array(5,), array(3,))
    >>>
    >>> # Batch distance with vmap (broadcast c across the batch with None)
    >>> dist_batch = jax.vmap(product.dist, in_axes=(0, 0, None))
    >>> distances = dist_batch(x_batch, y_batch, c)

JIT Compilation & Batching
---------------------------
Python for-loops over factors unroll at JIT trace time since the product
structure is static. Each iteration traces a different code path (different
manifold type), which is correct and efficient.

References:
    Gu et al. "Learning Mixed-Curvature Representations in Product Spaces."
    ICLR 2019.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.math_utils import floor_at
from ._base import ManifoldBase
from .protocol import Curvature, ScalarCurvature

# Version selection constant, present so ``dist`` / ``dist_0`` can accept the ``version_idx``
# that manifold-generic callers always forward. See ``ProductManifold.dist`` for why it cannot
# be propagated to the factors.
VERSION_DEFAULT = 0


class ProductManifold:
    """Product manifold P = M1 x M2 x ... x Mn.

    Each factor is specified as ``(manifold_instance, dim)`` where ``dim`` is
    the point dimension (array slice width): ambient for Hyperboloid (d+1),
    spatial for Poincaré/Euclidean/ProperVelocity (d).

    Curvature is supplied at call time as a per-factor sequence:

        product.dist(x, y, (c0, c1, ...))         # explicit tuple
        product.dist(x, y, product.curvatures)    # use factor-stored values

    Factor instances may carry an initial curvature value (``Hyperboloid(c=1.0)``),
    but that value is **not** consulted by ProductManifold's methods — it is
    only useful as a static default to read off via ``product.curvatures``.

    ProductManifold satisfies the ``Manifold`` protocol with ``c`` typed as
    ``Curvature`` (a union of scalar and sequence-of-scalars), so generic code
    written against ``Manifold`` accepts product instances too. The product
    itself has **no ``c`` attribute** — there is no single scalar curvature.

    Args:
        *factors: Tuples of ``(manifold_instance, dim)``. At least one required.
        dtype: Target JAX dtype for computations (default: jnp.float32).

    Examples:
        >>> product = ProductManifold((Hyperboloid(c=1.0), 5), (Poincare(c=0.1), 3))
        >>> c = product.curvatures                   # (1.0, 0.1)
        >>> x = product.origin(c)
        >>> parts = product.split(x)
        >>> d = product.dist(x, y, c)

        From a signature (repeated factors):

        >>> product = ProductManifold.from_signature(
        ...     (Hyperboloid, 3, 8, 1.0),  # 8 copies of 3D-ambient Hyperboloid
        ...     (Poincare, 2, 4, 0.1),      # 4 copies of 2D Poincaré
        ... )
    """

    VERSION_DEFAULT = VERSION_DEFAULT

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

    def _validate_c(self, c: Curvature) -> Sequence[ScalarCurvature]:
        """Validate a per-factor curvature sequence.

        Raises ``TypeError`` if ``c`` has no length (e.g. a scalar passed by
        mistake) and ``ValueError`` if its length does not match
        ``n_factors``. Returns ``c`` unchanged on success so the validation
        can be inlined at the top of each method.
        """
        try:
            n = len(c)  # type: ignore[arg-type]
        except TypeError as e:
            raise TypeError(
                f"ProductManifold expects a sequence of {len(self._factors)} per-factor curvatures; "
                f"got {type(c).__name__}. Pass a tuple, e.g. product.curvatures or (c0, c1, ...)."
            ) from e
        if n != len(self._factors):
            raise ValueError(f"Expected {len(self._factors)} curvatures (one per factor), got {n}.")
        return c  # type: ignore[return-value]

    # -- Properties --------------------------------------------------------

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
    def curvatures(self) -> tuple[ScalarCurvature, ...]:
        """Per-factor curvatures stored on the factor instances.

        Useful as the ``c`` argument when curvature is static, e.g.
        ``product.dist(x, y, product.curvatures)``. For learnable curvature,
        build the tuple from ``LearnableCurvature`` calls instead.
        """
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

    def origin(self, c: Curvature) -> Float[Array, "total_dim"]:
        """Construct the product origin point under per-factor curvatures.

        Uses ``expmap_0(zeros, c[i])`` per factor, which returns the manifold
        origin for all manifold types without isinstance checks.
        """
        cs = self._validate_c(c)
        parts = []
        for m, dim, c_i in zip(self._factors, self._dims, cs, strict=True):
            zero_tangent = jnp.zeros(dim, dtype=self.dtype)
            parts.append(m.expmap_0(zero_tangent, c_i))
        return jnp.concatenate(parts)

    # -- Geometry (per-factor decomposable) --------------------------------

    def proj(
        self,
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Project point onto product manifold (per-factor projection)."""
        cs = self._validate_c(c)
        x = self._cast(x)
        parts = self.split(x)
        return jnp.concatenate([m.proj(p, c_i) for m, p, c_i in zip(self._factors, parts, cs, strict=True)])

    def addition(
        self,
        x: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Manifold addition (per-factor)."""
        cs = self._validate_c(c)
        x, y = self._cast(x), self._cast(y)
        x_parts, y_parts = self.split(x), self.split(y)
        return jnp.concatenate(
            [m.addition(xp, yp, c_i) for m, xp, yp, c_i in zip(self._factors, x_parts, y_parts, cs, strict=True)]
        )

    def scalar_mul(
        self,
        r: float,
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Scalar multiplication (same scalar r applied to all factors)."""
        cs = self._validate_c(c)
        x = self._cast(x)
        parts = self.split(x)
        return jnp.concatenate([m.scalar_mul(r, p, c_i) for m, p, c_i in zip(self._factors, parts, cs, strict=True)])

    # -- Exponential / logarithmic maps ------------------------------------

    def expmap(
        self,
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Exponential map (per-factor)."""
        cs = self._validate_c(c)
        v, x = self._cast(v), self._cast(x)
        v_parts, x_parts = self.split(v), self.split(x)
        return jnp.concatenate(
            [m.expmap(vp, xp, c_i) for m, vp, xp, c_i in zip(self._factors, v_parts, x_parts, cs, strict=True)]
        )

    def expmap_0(
        self,
        v: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Exponential map from origin (per-factor)."""
        cs = self._validate_c(c)
        v = self._cast(v)
        parts = self.split(v)
        return jnp.concatenate([m.expmap_0(p, c_i) for m, p, c_i in zip(self._factors, parts, cs, strict=True)])

    def logmap(
        self,
        y: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Logarithmic map (per-factor)."""
        cs = self._validate_c(c)
        y, x = self._cast(y), self._cast(x)
        y_parts, x_parts = self.split(y), self.split(x)
        return jnp.concatenate(
            [m.logmap(yp, xp, c_i) for m, yp, xp, c_i in zip(self._factors, y_parts, x_parts, cs, strict=True)]
        )

    def logmap_0(
        self,
        y: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Logarithmic map to origin (per-factor)."""
        cs = self._validate_c(c)
        y = self._cast(y)
        parts = self.split(y)
        return jnp.concatenate([m.logmap_0(p, c_i) for m, p, c_i in zip(self._factors, parts, cs, strict=True)])

    def retraction(
        self,
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Retraction (per-factor)."""
        cs = self._validate_c(c)
        v, x = self._cast(v), self._cast(x)
        v_parts, x_parts = self.split(v), self.split(x)
        return jnp.concatenate(
            [m.retraction(vp, xp, c_i) for m, vp, xp, c_i in zip(self._factors, v_parts, x_parts, cs, strict=True)]
        )

    # -- Transport / tangent space -----------------------------------------

    def ptransp(
        self,
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Parallel transport of v from x to y (per-factor)."""
        cs = self._validate_c(c)
        v, x, y = self._cast(v), self._cast(x), self._cast(y)
        v_parts, x_parts, y_parts = self.split(v), self.split(x), self.split(y)
        return jnp.concatenate(
            [
                m.ptransp(vp, xp, yp, c_i)
                for m, vp, xp, yp, c_i in zip(self._factors, v_parts, x_parts, y_parts, cs, strict=True)
            ]
        )

    def ptransp_0(
        self,
        v: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Parallel transport from origin to y (per-factor)."""
        cs = self._validate_c(c)
        v, y = self._cast(v), self._cast(y)
        v_parts, y_parts = self.split(v), self.split(y)
        return jnp.concatenate(
            [m.ptransp_0(vp, yp, c_i) for m, vp, yp, c_i in zip(self._factors, v_parts, y_parts, cs, strict=True)]
        )

    def tangent_inner(
        self,
        u: Float[Array, "total_dim"],
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, ""]:
        """Riemannian inner product (sum of per-factor inner products)."""
        cs = self._validate_c(c)
        u, v, x = self._cast(u), self._cast(v), self._cast(x)
        u_parts, v_parts, x_parts = self.split(u), self.split(v), self.split(x)
        inners = jnp.stack(
            [
                m.tangent_inner(up, vp, xp, c_i)
                for m, up, vp, xp, c_i in zip(self._factors, u_parts, v_parts, x_parts, cs, strict=True)
            ]
        )
        return jnp.sum(inners)

    def tangent_norm(
        self,
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, ""]:
        """Riemannian norm (sqrt of tangent inner product with itself)."""
        return jnp.sqrt(floor_at(self.tangent_inner(v, v, x, c), 0.0))

    def egrad2rgrad(
        self,
        grad: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Convert Euclidean gradient to Riemannian gradient (per-factor)."""
        cs = self._validate_c(c)
        grad, x = self._cast(grad), self._cast(x)
        g_parts, x_parts = self.split(grad), self.split(x)
        return jnp.concatenate(
            [m.egrad2rgrad(gp, xp, c_i) for m, gp, xp, c_i in zip(self._factors, g_parts, x_parts, cs, strict=True)]
        )

    def tangent_proj(
        self,
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "total_dim"]:
        """Project vector onto tangent space (per-factor)."""
        cs = self._validate_c(c)
        v, x = self._cast(v), self._cast(x)
        v_parts, x_parts = self.split(v), self.split(x)
        return jnp.concatenate(
            [m.tangent_proj(vp, xp, c_i) for m, vp, xp, c_i in zip(self._factors, v_parts, x_parts, cs, strict=True)]
        )

    # -- Distance ----------------------------------------------------------

    def component_dist(
        self,
        x: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, "n_factors"]:
        """Per-factor geodesic distances as a vector."""
        cs = self._validate_c(c)
        x, y = self._cast(x), self._cast(y)
        x_parts, y_parts = self.split(x), self.split(y)
        return jnp.stack([m.dist(xp, yp, c_i) for m, xp, yp, c_i in zip(self._factors, x_parts, y_parts, cs, strict=True)])

    def dist(
        self,
        x: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
        c: Curvature,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Product distance (L2/Riemannian): d = sqrt(sum d_i^2).

        ``version_idx`` is accepted and **ignored**: it is a per-manifold index (Poincaré's 2
        means the metric-tensor form, the Hyperboloid has no such variant), so one value cannot
        be forwarded to a heterogeneous product. Each factor uses its own default. The argument
        exists so manifold-generic callers that always forward one (e.g.
        ``hyperbolix.utils.helpers.compute_pairwise_distances``) work with product manifolds;
        call ``component_dist`` directly if you need per-factor control.
        """
        del version_idx
        d_per_factor = self.component_dist(x, y, c)
        return jnp.sqrt(jnp.sum(d_per_factor**2))

    def dist_0(
        self,
        x: Float[Array, "total_dim"],
        c: Curvature,
        version_idx: int = VERSION_DEFAULT,
    ) -> Float[Array, ""]:
        """Distance from origin (L2/Riemannian). ``version_idx`` accepted and ignored (see ``dist``)."""
        del version_idx
        cs = self._validate_c(c)
        x = self._cast(x)
        parts = self.split(x)
        d_sq = jnp.stack([m.dist_0(p, c_i) ** 2 for m, p, c_i in zip(self._factors, parts, cs, strict=True)])
        return jnp.sqrt(jnp.sum(d_sq))

    def dist_l1(
        self,
        x: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, ""]:
        """L1 product distance: d = sum d_i."""
        return jnp.sum(self.component_dist(x, y, c))

    def dist_min(
        self,
        x: Float[Array, "total_dim"],
        y: Float[Array, "total_dim"],
        c: Curvature,
    ) -> Float[Array, ""]:
        """Min product distance: d = min d_i.

        Warning:
            This is NOT a true metric: positive-definiteness fails. Two
            distinct points ``x != y`` that agree on any single factor
            (``x_i == y_i``) yield ``d = 0``. Use only when the
            "closest-factor" interpretation is semantically intended;
            for an actual distance use ``dist`` (L2) or ``dist_l1``.
        """
        return jnp.min(self.component_dist(x, y, c))

    # -- Validation --------------------------------------------------------

    def is_in_manifold(
        self,
        x: Float[Array, "total_dim"],
        c: Curvature,
        atol: float | None = None,
    ) -> Array:
        """Check if point is on product manifold (all factors valid).

        ``atol`` is forwarded unchanged to every factor, so ``None`` lets each factor resolve
        its own dtype-aware default (:func:`~hyperbolix.manifolds._base.default_atol`).
        """
        cs = self._validate_c(c)
        x = self._cast(x)
        parts = self.split(x)
        checks = [m.is_in_manifold(p, c_i, atol) for m, p, c_i in zip(self._factors, parts, cs, strict=True)]
        return jnp.all(jnp.stack(checks))

    def is_in_tangent_space(
        self,
        v: Float[Array, "total_dim"],
        x: Float[Array, "total_dim"],
        c: Curvature,
        atol: float | None = None,
    ) -> Array:
        """Check if vector is in tangent space at x (all factors valid); ``atol`` forwarded per factor."""
        cs = self._validate_c(c)
        v, x = self._cast(v), self._cast(x)
        v_parts, x_parts = self.split(v), self.split(x)
        checks = [
            m.is_in_tangent_space(vp, xp, c_i, atol)
            for m, vp, xp, c_i in zip(self._factors, v_parts, x_parts, cs, strict=True)
        ]
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
          - ``(ManifoldClass, dim, count)`` — uses ``c=1.0`` as factor init.
          - ``(ManifoldClass, dim, count, curvature)`` — sets factor init ``c``.

        The ``curvature`` here is an **initial / default** value stored on the
        factor; it is only consulted via ``product.curvatures``. Geometry
        methods take ``c`` at call time.

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
            parts.append(f"{name}(dim={d}, c_init={m.c})")
        signature = " x ".join(parts)
        return f"ProductManifold({signature}, total_dim={self._total_dim})"
