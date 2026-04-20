"""Proper Velocity regression layers for JAX/Flax NNX.

Implements the PV multinomial-logistic-regression layer from Chen et al. 2026
(Thm 5.2 / Eq. 19). Delegates to ``ProperVelocity.compute_mlr`` which returns
Euclidean logits suitable as input to a softmax-cross-entropy loss.

Dimension key:
  B: batch size       I: input dimension       P: number of classes (out_dim)
"""

import jax
from flax import nnx
from jaxtyping import Array, Float

from hyperbolix.manifolds.proper_velocity import ProperVelocity

from ._helpers import validate_pv_manifold


class HypRegressionPV(nnx.Module):
    """
    Hyperbolic multinomial logistic regression layer (Proper Velocity model).

    Implements Chen et al. 2026, Thm 5.2 / Eq. 19, delegating to
    ``ProperVelocity.compute_mlr``. The output is the Euclidean signed margin
    to each PV hyperplane and is intended to be consumed by a standard
    softmax-cross-entropy classification loss.

    Computation steps:
        0) Optionally lift tangent-space input via ``expmap_0``.
        1) Compute the PV MLR scores via ``manifold.compute_mlr``.

    Parameters
    ----------
    manifold_module : ProperVelocity
        Class-based Proper Velocity manifold instance.
    in_dim : int
        Dimension of the input space.
    out_dim : int
        Dimension of the output space (number of classes).
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold').
        Note: This is a static configuration — changing it after initialization
        requires recompilation.
    clamping_factor : float
        Clamping factor for the MLR output (default: 1.0).
    smoothing_factor : float
        Smoothing factor for the MLR output (default: 50.0).

    Notes
    -----
    JIT Compatibility:
        Configuration parameters (input_space, clamping_factor, smoothing_factor)
        are treated as static and are baked into the compiled function.

    References
    ----------
    Chen et al. "Proper Velocity Neural Networks." ICLR 2026.
    """

    def __init__(
        self,
        manifold_module: ProperVelocity,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        validate_pv_manifold(
            manifold_module,
            required_methods=("expmap_0", "compute_mlr"),
        )
        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

        # Kernel init: small normal with std = 1e-2 (matches paper reference
        # PVManifoldMLR.reset_parameters in the Chen et al. repo).
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (out_dim, in_dim)) * 1e-2)
        # Bias init: uniform U(-1e-3, 1e-3) (matches paper reference).
        self.bias = nnx.Param(jax.random.uniform(rngs.params(), (out_dim, 1), minval=-1e-3, maxval=1e-3))

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """
        Forward pass through the PV regression layer.

        Parameters
        ----------
        x : Array of shape (batch, in_dim)
            Input tensor — PV manifold points or tangent-space vectors.
        c : float
            Manifold curvature (default: 1.0).

        Returns
        -------
        res : Array of shape (batch, out_dim)
            MLR logits (Euclidean, suitable for cross-entropy).
        """
        if self.input_space == "tangent":
            x = jax.vmap(self.manifold.expmap_0, in_axes=(0, None), out_axes=0)(x, c)

        return self.manifold.compute_mlr(
            x,
            self.kernel[...],
            self.bias[...],
            c,
            self.clamping_factor,
            self.smoothing_factor,
        )
