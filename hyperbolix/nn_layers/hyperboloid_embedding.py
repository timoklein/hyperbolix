"""Hyperboloid token embedding (the ``LorentzEmbeddings`` table of HELM, He et al. 2025).

Dimension key:
  V: vocabulary size (num_embeddings)
  A: ambient dim (features = S + 1, time coordinate first)
  S: spatial dim
  ...: the id batch shape, e.g. (B, L)

Curvature convention: HELM's ``Lorentz(k)`` stores points with ``<x, x>_L = -k``; hyperbolix's
``Hyperboloid`` at curvature ``c`` stores them with ``<x, x>_L = -1/c``, so ``c = 1/k``. Geodesic
distances agree between the two (``d(o, x) = asinh(sqrt(c)·‖x_s‖)/sqrt(c)`` in both).

References
----------
He, Neil, et al. "HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts."
arXiv:2505.24722 (2025). Reference code: ``helm/hypercore/nn/attention/lorentz_word_emb.py``
(``LorentzEmbeddings``) and ``helm/modules/helm_mice.py`` (``project_emb``).
"""

from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int

from hyperbolix.manifolds.hyperboloid import Hyperboloid

from ..optim import ManifoldParam
from .hyperboloid_core import spatial_to_hyperboloid


def lorentz_embedding_init(
    key: jax.Array,
    num_embeddings: int,
    features: int,
    c: float,
    *,
    radius: float = 1.0,
    dtype: DTypeLike = jnp.float32,
) -> Float[Array, "V A"]:
    """Hyperboloid table with isotropic directions at a fixed geodesic radius.

    Each row is ``Hyperboloid.expmap_0((0, radius·u))`` at curvature ``c``, with ``u`` drawn
    uniformly from the spatial unit sphere (a normalized standard-normal draw over the ``S = A - 1``
    spatial coordinates). Every row therefore sits at geodesic distance ``radius`` from the origin,
    i.e. scaled radius ``sqrt(c)·radius``.

    This is the clean form of HELM's init. HELM (``Lorentz.random_normal`` → geoopt ``expmap0`` +
    ``project``) normalizes a standard-normal draw over all ``A`` coordinates, time included, and
    passes that non-tangent vector to ``expmap0``, whose Minkowski norm ``n = sqrt(‖u_s‖² - u_0²)``
    it clamps at ``1e-8`` before the projection rebuilds the time slot. With ``u_0² ~ 1/A`` the
    resulting radius is ``1 - O(u_0²)`` at every ``k`` (measured deficit ``1 - d`` between
    ``u_0²/2`` and ``u_0²``; all rows within 0.012 of 1 at ``A = 768``, down to 0.31 at ``A = 8``,
    where 3 % of rows hit the clamp; ``logs/2026-09-30_helm-embedding/probe_reference_radius.out``).
    The intended radius is 1 in geodesic distance, independent of the curvature, which is the
    default here.

    Args:
        key: PRNG key for the direction draw.
        num_embeddings: Vocabulary size ``V``.
        features: Ambient dimension ``A`` (``>= 2``).
        c: Curvature the table is stored on (positive).
        radius: Geodesic distance of every row from the origin (default 1.0, HELM's scale).
        dtype: Storage dtype of the table.

    Returns:
        Table of shape ``(V, A)`` on the hyperboloid of curvature ``c``.
    """
    manifold = Hyperboloid(dtype=jnp.dtype(dtype))
    direction_VS = jax.random.normal(key, (num_embeddings, features - 1), dtype=dtype)
    direction_VS = direction_VS / jnp.linalg.norm(direction_VS, axis=-1, keepdims=True)
    tangent_VA = jnp.concatenate(
        [jnp.zeros((num_embeddings, 1), dtype=dtype), jnp.asarray(radius, dtype=dtype) * direction_VS], axis=-1
    )
    return jax.vmap(manifold.expmap_0, in_axes=(0, None))(tangent_VA, c)


class LorentzEmbedding(nnx.Module):
    """Token embedding on the hyperboloid (HELM's ``LorentzEmbeddings`` without positional encoding).

    Looks up rows of a ``(V, A)`` table of hyperboloid points at curvature ``c`` and, when
    ``c_out`` is given, maps them onto the ``c_out`` sheet by scaling the whole ambient vector by
    ``sqrt(c/c_out)`` (HELM: ``emb * sqrt(k_out/k_in)``). That map keeps the scaled radius
    ``sqrt(c)·d`` of every row.

    Two parameterizations, matching HELM's two configurations:

    - ``"manifold"`` (HELM ``project_emb=0``, the 120M model): the table is a
      :class:`~hyperbolix.optim.ManifoldParam` of shape ``(V, A)`` on ``Hyperboloid`` at ``c``,
      initialized by :func:`lorentz_embedding_init` (isotropic directions at geodesic radius
      ``init_radius``, the clean form of HELM's ``random_normal`` init). Train it with
      :func:`~hyperbolix.optim.riemannian_adam`, which ``nnx.Optimizer(model, riemannian_adam(lr),
      wrt=nnx.Param)`` applies to this leaf while every other ``nnx.Param`` takes a plain Adam
      step. HELM uses the model learning rate and weight decay 0 for the table (and geoopt's
      ``stabilize=1``, which ``riemannian_adam`` already does at every step; see its docstring),
      and AdamW for the Euclidean parameters.
    - ``"spatial"`` (HELM ``project_emb=1``, the 1B model): a Euclidean ``nnx.Param`` of shape
      ``(V, A - 1)`` with ``N(0, 1)`` entries (``torch.nn.Embedding``'s init) holding the spatial
      coordinates at ``c``; the forward pass rebuilds the time slot with
      :func:`~hyperbolix.nn_layers.hyperboloid_core.spatial_to_hyperboloid`. Train it with a
      Euclidean optimizer. Its rows start at spatial norm ``≈ sqrt(A - 1)``, i.e. geodesic radius
      ``≈ asinh(sqrt(c·(A - 1)))/sqrt(c)`` (4.0 at ``A = 768``, ``c = 1``). HELM's projection
      rebuilds time at ``k_in`` and applies no ``c_out`` map; here ``c_out`` acts as in the
      manifold path.

    The table's curvature ``c`` is fixed at construction: a ``ManifoldParam`` lives on one sheet,
    and the optimizer reads its curvature from the parameter's metadata.

    Args:
        num_embeddings: Vocabulary size ``V``.
        features: Ambient dimension ``A`` of the output points (spatial dim + 1, ``>= 2``).
        rngs: NNX RNG streams; the init draws from ``rngs.params()``.
        c: Curvature the table is stored on (positive, default 1.0).
        parameterization: ``"manifold"`` (default) or ``"spatial"``, see above.
        init_radius: Geodesic radius of every row at init for ``"manifold"`` (default 1.0, HELM's
            scale); ignored for ``"spatial"``.
        param_dtype: Storage dtype of the table (default ``jnp.float32``, pinned so it does not
            become float64 under ``jax_enable_x64``). The output has this dtype.

    Attributes:
        embedding: The table, a ``ManifoldParam`` ``(V, A)`` or an ``nnx.Param`` ``(V, A - 1)``.
        manifold: The ``Hyperboloid`` instance the ``"manifold"`` table is tagged with.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        rngs: nnx.Rngs,
        c: float = 1.0,
        parameterization: Literal["manifold", "spatial"] = "manifold",
        init_radius: float = 1.0,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if parameterization not in ("manifold", "spatial"):
            raise ValueError(f"LorentzEmbedding: parameterization must be 'manifold' or 'spatial', got {parameterization!r}")
        if features < 2:
            raise ValueError(f"LorentzEmbedding: features is the ambient dim (spatial + 1) and must be >= 2, got {features}")
        if num_embeddings < 1:
            raise ValueError(f"LorentzEmbedding: num_embeddings must be >= 1, got {num_embeddings}")
        if not c > 0:
            raise ValueError(f"LorentzEmbedding: c must be positive, got {c}")
        if init_radius < 0:
            raise ValueError(f"LorentzEmbedding: init_radius must be >= 0, got {init_radius}")

        self.num_embeddings = num_embeddings
        self.features = features
        self.c = c
        self.parameterization = parameterization
        self.manifold = Hyperboloid(dtype=jnp.dtype(param_dtype))

        key = rngs.params()
        if parameterization == "manifold":
            table_VA = lorentz_embedding_init(key, num_embeddings, features, c, radius=init_radius, dtype=param_dtype)
            self.embedding = ManifoldParam(table_VA, manifold=self.manifold, curvature=c)
        else:
            self.embedding = nnx.Param(jax.random.normal(key, (num_embeddings, features - 1), dtype=param_dtype))

    def __call__(
        self,
        ids: Int[Array, "..."],
        c_out: float | Float[Array, ""] | None = None,
    ) -> Float[Array, "... A"]:
        """Look up token ids.

        Args:
            ids: Integer token ids of any shape ``(...)``. Out-of-range ids give NaN rows
                (``jnp.take``'s fill mode) rather than a silently clamped valid row.
            c_out: Curvature of the output sheet; ``None`` (default) keeps the table's ``c``. May be
                a traced value (e.g. from ``LearnableCurvature``).

        Returns:
            Points of shape ``(..., A)`` on the hyperboloid of curvature ``c_out`` (or ``c``).
        """
        if not jnp.issubdtype(jnp.asarray(ids).dtype, jnp.integer):
            raise ValueError(f"LorentzEmbedding: ids must be integers, got dtype {jnp.asarray(ids).dtype}")
        dtype = self.embedding[...].dtype
        if self.parameterization == "spatial":
            table_VS = self.embedding[...]  # (V, S)
            spatial_S = jnp.take(table_VS, ids, axis=0)  # (..., S)
            target_c = self.c if c_out is None else c_out
            return spatial_to_hyperboloid(
                spatial_S, jnp.asarray(self.c, dtype=dtype), jnp.asarray(target_c, dtype=dtype)
            )  # (..., A)

        table_VA = self.embedding[...]  # (V, A)
        rows_A = jnp.take(table_VA, ids, axis=0)  # (..., A)
        if c_out is None:
            return rows_A
        # <s·x, s·x>_L = -s²/c = -1/c_out for s = sqrt(c/c_out): the whole ambient vector scales.
        sheet_scale = jnp.sqrt(jnp.asarray(self.c, dtype=dtype) / jnp.asarray(c_out, dtype=dtype))  # scalar
        return sheet_scale * rows_A
