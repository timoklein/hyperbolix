"""Poincaré-ball vector quantization layers for JAX/Flax NNX.

Two hyperbolic VQ-VAE quantizer bottlenecks, ported from the reference
implementations in the ``vqvae_nnx`` project. Both take encoder features
(Euclidean tangent vectors at the origin) and return a quantized tangent vector
for the decoder plus the discrete code indices and metrics. The encoder/decoder
themselves stay in user code — these layers are *only* the quantization step.

``HypVQEmbeddingPoincare`` (HVQ-VAE, Chen et al. 2025)
    Explicit Poincaré-ball codebook held as a **non-parameter buffer**, geodesic
    nearest-neighbour selection, copy-gradient straight-through estimator (STE),
    and a hyperbolic exponential-moving-average codebook update (GGBall, Bu et
    al. 2026, Eqs. 41-43) with optional dead-code revival. The codebook receives
    no gradient — it moves only via :meth:`HypVQEmbeddingPoincare.ema_update`,
    called in the train step after ``optimizer.update``.

``HypVQMLRPoincare`` (HyperVQ, Goswami et al. 2025)
    Implicit codebook = the rows of a Poincaré MLR (``HypRegressionPoincarePP``).
    Selection is Gumbel-Softmax "quantization-as-classification" with the STE on
    the categorical weights, so plain Euclidean Adam trains it. A ``deterministic``
    flag (toggled by ``model.eval()`` / ``model.train()``) switches to a
    deterministic argmax at inference.

Dimension key:
  N: flattened batch * spatial tokens     C: code / latent dim (= D)
  K: num_codes                            M: points to average
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds.poincare import Poincare

from ._helpers import validate_poincare_manifold
from .poincare_batchnorm import poincare_weighted_midpoint
from .poincare_regression import HypRegressionPoincarePP


class PoincareVQOutput(NamedTuple):
    """Result of a Poincaré VQ forward pass (a JAX-pytree friendly tuple).

    Attributes
    ----------
    quantized : Array, shape (N, C)
        Decoder input — a tangent vector at the origin, cast to ``out_dtype``
        (default float32) so it does not silently promote a float32 decoder to
        float64 under ``jax_enable_x64``. Carries the straight-through gradient.
    indices : Array, shape (N,), int32
        Selected code index per token. Feeds perplexity and the EMA update.
    loss : Array, scalar
        Auxiliary VQ loss to add to the caller's reconstruction loss — the
        commitment term for the embedding layer, ``0`` for the MLR layer
        (HyperVQ is reconstruction-only). In the manifold dtype.
    perplexity : Array, scalar float32
        Codebook usage perplexity ``exp(-Σ p_k log p_k)`` in ``[1, num_codes]``.
    z : Array, shape (N, C)
        Encoder points lifted onto the ball (``expmap_0`` of the input). In the
        manifold dtype; pass straight to :meth:`HypVQEmbeddingPoincare.ema_update`.
    """

    quantized: Array
    indices: Array
    loss: Array
    perplexity: Array
    z: Array


def _codebook_perplexity(indices_N: Array, num_codes: int) -> Array:
    """Codebook perplexity ``exp(-Σ_k p_k log p_k)`` (a float32 logging metric).

    Range ``[1, num_codes]``: 1 means a single code is ever picked (collapse),
    ``num_codes`` means uniform usage. ``+1e-10`` guards ``log(0)`` for unused
    codes (their ``p_k log p_k → 0`` in the limit).
    """
    one_hot_NK = jax.nn.one_hot(indices_N, num_codes, dtype=jnp.float32)  # (N, K)
    p_K = jnp.mean(one_hot_NK, axis=0)  # (K,) batch usage frequency per code
    entropy = -jnp.sum(p_K * jnp.log(p_K + 1e-10))  # nats
    return jnp.exp(entropy).astype(jnp.float32)


def _weighted_midpoint_op(x_C: Array, w: Array, manifold: Poincare, c: float) -> Array:
    """The single-point weighted-midpoint operator ``[x, w]_c`` (GGBall Eq. 43).

    Building block of the two-point EMA blend (Eq. 42): it maps a ball point ``x``
    and a scalar weight ``w`` into the representation that Möbius addition then
    combines into the gyromidpoint. For unit weight it is the identity
    (``[x, 1]_c = x``)::

        [x, w]_c = ( w · λ(x) · x ) / ( 1 + sqrt(1 + c · w² · λ(x)² · ‖x‖²) )

    where ``λ(x) = 2 / (1 - c‖x‖²)``. Distinct from
    :func:`poincare_weighted_midpoint` (Eq. 41, the M-point centroid) — this is
    the *two-point* blend's per-operand term and stays private to the VQ update.
    """
    lam = manifold.conformal_factor(x_C[None, :], c)[0, 0]  # scalar λ(x)
    x_sqnorm = jnp.dot(x_C, x_C)  # ‖x‖²
    numerator_C = w * lam * x_C
    denom = 1.0 + jnp.sqrt(1.0 + c * w**2 * lam**2 * x_sqnorm)
    return numerator_C / denom


class HypVQEmbeddingPoincare(nnx.Module):
    """Poincaré-ball vector quantizer with an EMA codebook (HVQ-VAE).

    Forward (the lookup of Algorithm 1, copy-gradient STE of §3.2):
        x  --expmap_0-->  z          (lift encoder tangent vector onto the ball)
        z  --argmin d_D--> k         (geodesic nearest code; non-differentiable)
        q = codebook[k]
        quantized = x + sg(logmap_0(q) - x)   (STE: forward = logmap_0(q),
                                               backward flows to the encoder via x)

    The codebook is an ``nnx.Variable`` *buffer*, not a parameter: with
    ``nnx.Optimizer(model, tx, wrt=nnx.Param)`` the optimizer ignores it, and it
    receives no gradient (the commitment loss stop-gradients ``q``, the STE
    stop-gradients ``logmap_0(q) - x``). It moves only through :meth:`ema_update`,
    which is the hyperbolic EMA of GGBall Eqs. 41-42.

    Parameters
    ----------
    manifold_module : Poincare
        Class-based Poincaré manifold instance (carries the dtype).
    num_codes : int
        Codebook size ``K``.
    code_dim : int
        Code / latent dimension ``C``.
    rngs : nnx.Rngs
        RNGs for codebook initialisation.
    init_scale : float
        Std of the zero-centred Gaussian codebook init (default: 1e-3). Tiny so
        the codes start well inside the ball.
    commitment_weight : float
        Weight on the commitment loss ``E[d_D(z, sg(q))]`` returned in
        ``output.loss`` (default: 0.5, matching the reference ``ω₃``).
    ema_decay : float
        ``β`` in Eq. 42 — weight on the OLD code (slow update). Also the
        cluster-size EMA decay (default: 0.99).
    dead_code_threshold : float
        Per-code usage below which a code is "stale" and revived (default: 1.0).
    dead_code_revival : bool
        Enable GGBall dead-code revival in :meth:`ema_update` (default: False).
    out_dtype : jnp.dtype
        Dtype of ``output.quantized`` (default: float32) — the manifold→decoder
        boundary cast that prevents silent float64 promotion of the decoder.
    """

    def __init__(
        self,
        manifold_module: Poincare,
        num_codes: int,
        code_dim: int,
        *,
        rngs: nnx.Rngs,
        init_scale: float = 1e-3,
        commitment_weight: float = 0.5,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        dead_code_revival: bool = False,
        out_dtype: jnp.dtype = jnp.float32,  # type: ignore[assignment]
    ) -> None:
        validate_poincare_manifold(
            manifold_module,
            required_methods=("expmap_0", "logmap_0", "dist", "proj", "addition", "scalar_mul", "conformal_factor"),
        )
        self.manifold = manifold_module
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_revival = dead_code_revival
        self.out_dtype = out_dtype

        # Codebook BUFFER (nnx.Variable, not nnx.Param): a small zero-centred
        # Gaussian projected onto the ball. The init is tiny so the proj is a
        # no-op for any reasonable curvature (c=1.0 nominal here; every forward /
        # ema_update applies the true call-time c). Float dtype from the manifold.
        init_KC = jax.random.normal(rngs.params(), (num_codes, code_dim), dtype=manifold_module.dtype) * init_scale
        init_KC = jax.vmap(manifold_module.proj, in_axes=(0, None))(init_KC, 1.0)
        self.codebook = nnx.Variable(init_KC)

        # Per-code EMA usage count for dead-code revival. Init to 2·threshold so a
        # never-used code survives a grace window before its first revival.
        self.cluster_size = nnx.Variable(jnp.full((num_codes,), 2.0 * dead_code_threshold, dtype=manifold_module.dtype))

    def __call__(self, x_NC: Float[Array, "N C"], c: float = 1.0) -> PoincareVQOutput:
        """Quantize encoder tangent vectors ``x_NC`` at curvature ``c``.

        ``x_NC`` is the encoder output in the tangent space at the origin (plain
        Euclidean features); ``c`` is supplied at call time (learnable-curvature
        friendly).
        """
        manifold = self.manifold

        # Lift to the ball. expmap_0 is single-vector — MUST be vmapped, else it
        # reduces over the whole batch and the geometry becomes batch-size dependent.
        z_NC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(x_NC, c)  # (N, C) on ball

        # Geodesic nearest-neighbour (non-differentiable). Double-vmap dist over
        # the N tokens and K codes.
        codebook_KC = self.codebook[...]
        dists_NK = jax.vmap(jax.vmap(manifold.dist, in_axes=(None, 0, None)), in_axes=(0, None, None))(
            z_NC, codebook_KC, c
        )  # (N, K)
        indices_N = jnp.argmin(dists_NK, axis=-1).astype(jnp.int32)  # (N,)
        q_NC = codebook_KC[indices_N]  # (N, C) selected codes on the ball

        # Copy-gradient STE in the tangent space at the origin (§3.2): forward
        # value is logmap_0(q); backward, the gradient bypasses the argmin and
        # flows to the encoder via x_NC. Both operands are tangent vectors.
        q_tangent_NC = jax.vmap(manifold.logmap_0, in_axes=(0, None))(q_NC, c)  # (N, C)
        quantized_NC = x_NC + jax.lax.stop_gradient(q_tangent_NC - x_NC)

        # Commitment loss E[d_D(z, sg(q))] — pulls the encoder toward the codes.
        # (The codebook term E[d_D(sg(z), q)] is dropped: EMA owns the codebook.)
        commit_N = jax.vmap(manifold.dist, in_axes=(0, 0, None))(z_NC, jax.lax.stop_gradient(q_NC), c)
        loss = self.commitment_weight * jnp.mean(commit_N)

        perplexity = _codebook_perplexity(indices_N, self.num_codes)
        return PoincareVQOutput(
            quantized=quantized_NC.astype(self.out_dtype),
            indices=indices_N,
            loss=loss,
            perplexity=perplexity,
            z=z_NC,
        )

    def ema_update(
        self,
        z_NC: Float[Array, "N C"],
        indices_N: Array,
        c: float = 1.0,
        *,
        reset_key: Array | None = None,
    ) -> Array:
        """In-place hyperbolic EMA codebook update — GGBall Eqs. 41-42.

        Call this in the train step *after* ``optimizer.update`` (the codebook is
        a buffer the optimizer never touches). Each code is pulled toward the
        weighted gyromidpoint ``μ_j`` (Eq. 41) of the encoder points assigned to
        it this batch, then blended with its old position (Eq. 42, weight
        ``ema_decay`` = β on the old code). Codes with no assigned points are left
        unchanged. When ``dead_code_revival`` is set, a per-code usage estimate is
        advanced and any code below ``dead_code_threshold`` is replaced by a random
        encoder point from this batch (needs ``reset_key``).

        Pass ``output.z`` (lifted encoder points) and ``output.indices`` from the
        forward pass, and the same ``c``. Mutates ``self.codebook`` and
        ``self.cluster_size``; returns ``n_dead`` (codes revived this step, 0 if
        revival is off). All math runs in the manifold dtype.
        """
        manifold = self.manifold
        K = self.num_codes
        dtype = manifold.dtype
        decay = self.ema_decay

        z_NC = z_NC.astype(dtype)
        old_KC = self.codebook[...].astype(dtype)

        # Assignment matrix A_NK = one-hot(argmin) — the indicator w_ij of Eq. 41.
        A_NK = jax.nn.one_hot(indices_N, K, dtype=dtype)  # (N, K)
        counts_K = jnp.sum(A_NK, axis=0)  # (K,) points assigned to each code
        has_pts_K = counts_K > 0  # (K,) codes touched this batch

        # Eq. 41: per-code gyromidpoint of the assigned points. weights = Aᵀ maps
        # the N points to one midpoint per code → (K, C).
        mu_KC = poincare_weighted_midpoint(z_NC, A_NK.T, manifold, c)  # (K, C)

        # Eq. 42: c_j^{t+1} = proj( [c_j^t, β]_c ⊕_c [μ_j, 1-β]_c ).
        beta = jnp.asarray(decay, dtype=dtype)
        one_minus_beta = jnp.asarray(1.0, dtype=dtype) - beta

        def blend(old_C: Array, mu_C: Array) -> Array:
            term_old = _weighted_midpoint_op(old_C, beta, manifold, c)  # [c^t, β]_c
            term_mu = _weighted_midpoint_op(mu_C, one_minus_beta, manifold, c)  # [μ, 1-β]_c
            return manifold.proj(manifold.addition(term_old, term_mu, c), c)

        new_KC = jax.vmap(blend)(old_KC, mu_KC)  # (K, C)

        # Empty codes keep their old position.
        updated_KC = jnp.where(has_pts_K[:, None], new_KC, old_KC)  # (K, C)

        # Advance the per-code usage estimate (cheap; the no-reset path is
        # numerically identical since the revival subgraph below is masked out).
        new_cluster_size_K = decay * self.cluster_size[...] + (1.0 - decay) * counts_K  # (K,)

        n_dead = jnp.asarray(0, dtype=jnp.int32)
        if self.dead_code_revival:  # static bool → subgraph traced out when off
            if reset_key is None:
                raise ValueError("ema_update requires reset_key when dead_code_revival=True")
            dead_K = new_cluster_size_K < self.dead_code_threshold  # (K,)
            reset_rows_K = jax.random.randint(reset_key, (K,), 0, z_NC.shape[0])  # (K,)
            revival_KC = jax.vmap(manifold.proj, in_axes=(0, None))(z_NC[reset_rows_K], c)  # (K, C)
            updated_KC = jnp.where(dead_K[:, None], revival_KC, updated_KC)
            # Give a revived code a fair trial: reset its usage to the initial credit.
            revived_credit = jnp.asarray(2.0 * self.dead_code_threshold, dtype=dtype)
            new_cluster_size_K = jnp.where(dead_K, revived_credit, new_cluster_size_K)
            n_dead = jnp.sum(dead_K).astype(jnp.int32)

        self.codebook[...] = updated_KC
        self.cluster_size[...] = new_cluster_size_K
        return n_dead


class HypVQMLRPoincare(nnx.Module):
    """Poincaré-ball vector quantizer with an implicit MLR codebook (HyperVQ).

    Quantization-as-classification: a Poincaré MLR (``HypRegressionPoincarePP``)
    scores each lifted token against ``num_codes`` hyperplanes, a code is sampled
    via Gumbel-Softmax with a straight-through estimator on the **categorical
    weights**, and the quantized vector is ``z_q = st_weights @ codes`` where the
    implicit codebook is ``codes = bias · kernel`` (the MLR's scalar offsets times
    its tangent normals). Putting the STE on the weights — not on ``z_q`` — is what
    lets the gradient reach the MLR parameters; plain ``optax.adam`` then trains
    them. HyperVQ is reconstruction-only, so ``output.loss`` is 0.

    The ``deterministic`` flag (set by ``model.eval()`` / ``model.train()`` via
    ``nnx.set_attributes``) switches the categorical selection from a noisy
    Gumbel-Softmax sample (training) to a deterministic argmax MAP estimate
    (inference) — without it, every eval forward pass would sample a different
    code and reconstructions would ghost.

    Parameters
    ----------
    manifold_module : Poincare
        Class-based Poincaré manifold instance (carries the dtype).
    num_codes : int
        Codebook size ``K`` (= MLR ``out_dim``).
    code_dim : int
        Code / latent dimension ``C`` (= MLR ``in_dim``).
    rngs : nnx.Rngs
        RNGs for the MLR parameter init.
    clamping_factor, smoothing_factor : float
        Passed through to ``HypRegressionPoincarePP`` (asinh clamp; defaults
        1.0 / 50.0).
    out_dtype : jnp.dtype
        Dtype of ``output.quantized`` (default: float32) — the manifold→decoder
        boundary cast.
    """

    def __init__(
        self,
        manifold_module: Poincare,
        num_codes: int,
        code_dim: int,
        *,
        rngs: nnx.Rngs,
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        out_dtype: jnp.dtype = jnp.float32,  # type: ignore[assignment]
    ) -> None:
        validate_poincare_manifold(manifold_module, required_methods=("expmap_0",))
        self.manifold = manifold_module
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.out_dtype = out_dtype
        # Toggled by model.eval()/model.train(): noisy Gumbel sample while training,
        # deterministic argmax MAP at eval.
        self.deterministic = False
        # Input already lifted onto the ball by this layer → keep MLR default
        # input_space="manifold" so it does not double-lift.
        self.mlr = HypRegressionPoincarePP(
            manifold_module,
            in_dim=code_dim,
            out_dim=num_codes,
            rngs=rngs,
            clamping_factor=clamping_factor,
            smoothing_factor=smoothing_factor,
        )

    def __call__(
        self,
        x_NC: Float[Array, "N C"],
        c: float = 1.0,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> PoincareVQOutput:
        """Quantize encoder tangent vectors ``x_NC`` at curvature ``c``.

        ``rngs`` supplies the Gumbel noise and is required while training; it is
        unused (and may be ``None``) at eval (``deterministic=True``).
        """
        manifold = self.manifold

        # Lift to the ball (single-vector op — MUST vmap), then score with the MLR.
        z_NC = jax.vmap(manifold.expmap_0, in_axes=(0, None))(x_NC, c)  # (N, C) on ball
        scores_NK = self.mlr(z_NC, c)  # (N, K)

        if self.deterministic:
            # Eval: deterministic MAP — argmax over clean scores, no Gumbel noise.
            k_idx_N = jnp.argmax(scores_NK, axis=-1)
            st_weights_NK = jax.nn.one_hot(k_idx_N, self.num_codes, dtype=scores_NK.dtype)
        else:
            # Train: Gumbel-Softmax with a straight-through estimator on the weights.
            if rngs is None:
                raise ValueError("rngs is required for the training (Gumbel) path; got None")
            noised_NK = scores_NK + jax.random.gumbel(rngs(), scores_NK.shape, dtype=scores_NK.dtype)
            soft_NK = jax.nn.softmax(noised_NK, axis=-1)  # differentiable
            k_idx_N = jnp.argmax(soft_NK, axis=-1)  # hard selection (non-differentiable)
            hard_NK = jax.nn.one_hot(k_idx_N, self.num_codes, dtype=soft_NK.dtype)
            # STE: forward uses hard (discrete), backward uses soft (differentiable).
            st_weights_NK = soft_NK + jax.lax.stop_gradient(hard_NK - soft_NK)

        indices_N = k_idx_N.astype(jnp.int32)

        # Implicit codebook = bias · kernel; (K,1)·(K,C) → (K,C). The matmul keeps
        # the op differentiable w.r.t. the MLR parameters.
        codes_KC = self.mlr.bias[...] * self.mlr.kernel[...]  # (K, C)
        z_q_NC = st_weights_NK @ codes_KC  # (N, C) — gradient flows to the MLR

        perplexity = _codebook_perplexity(indices_N, self.num_codes)
        return PoincareVQOutput(
            quantized=z_q_NC.astype(self.out_dtype),
            indices=indices_N,
            loss=jnp.zeros((), dtype=z_NC.dtype),  # HyperVQ is reconstruction-only
            perplexity=perplexity,
            z=z_NC,
        )
