"""Busemann fully connected (BFC) layers for JAX/Flax NNX.

Dimension key:
  B: batch size
  I: in_spatial (direction dim)        O: out_spatial
  Ai: in_ambient (Hyperboloid in_features = I + 1)
  Ao: out_ambient (Hyperboloid out_features = O + 1)

Each output coordinate is a Busemann logit ``u_k(x) = -alpha_k·B^{v_k}(x) + b_k`` (optionally
passed through an activation ``φ``), then mapped back to the manifold in closed form
(Chen et al. 2026, Thms. 4.1/4.2):

  - Lorentz (Thm. 4.2):  ``y_s = sinh(√c·φ(u))/√c``, ``y_t = √(1/c + ‖y_s‖²)`` — the same
    ``sinh_lift_to_hyperboloid`` output map as the PLFC layer.
  - Poincaré (Thm. 4.1): ``ω = sinh(√c·φ(u))/√c``, ``y = ω / (1 + √(1 + c‖ω‖²))``.

An optional intrinsic gyro-bias ``y ← y ⊕ exp_0(b)`` is available (default off), matching the
paper's Sec. 4.2 generalization and ``HypLinearHyperboloidPLFC``'s gyro-bias convention.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from hyperbolix.manifolds.hyperboloid import Hyperboloid
from hyperbolix.manifolds.poincare import Poincare

from ._helpers import validate_hyperboloid_manifold, validate_poincare_manifold
from .busemann_core import busemann_fc_poincare_output, busemann_score, init_weight_norm_params
from .hyperboloid_core import sinh_lift_to_hyperboloid
from .hyperboloid_linear import _assert_v_max_safe


class HypLinearHyperboloidBusemann(nnx.Module):
    """Busemann fully connected (BFC) layer (Hyperboloid / Lorentz model).

    Maps a hyperboloid point to a hyperboloid point (Chen et al. 2026, Thm. 4.2). Each output
    spatial coordinate is the activated Busemann logit ``φ(-alpha_k·B^{v_k}(x) + b_k)``; the point
    is recovered via the shared ``sinh_lift_to_hyperboloid`` output map (the same closed form
    as ``HypLinearHyperboloidPLFC``, but with a point-to-horosphere score instead of
    point-to-hyperplane).

    Parameters
    ----------
    manifold_module : Hyperboloid
        Class-based Hyperboloid manifold instance.
    in_dim : int
        Input ambient dimension (``d + 1``, time included).
    out_dim : int
        Output ambient dimension (``d_out + 1``, time included).
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    input_space : str
        ``"manifold"`` (default) or ``"tangent"`` (lift via ``expmap_0`` first). Static for JIT.
    activation : Callable or None
        Optional Euclidean activation ``φ`` applied to the Busemann logits (default: identity).
        Avoid ``relu`` when stacking several of these layers on high-dimensional input: at a
        random weight-normalized direction init, the Busemann score is positive for all but a
        narrow cone of directions, so ``relu`` can zero every output unit at once and collapse the
        whole layer to the manifold origin — a fixed point that self-reinforces through later
        layers. ``tanh`` avoids this by never fully zeroing the logit.
    v_max : float
        Output-side guard: ``√c·φ(u)`` is hard-clipped to ``±v_max`` before the sinh
        diffeomorphism (default: 10.0, the Shi et al. 2026 reference value).
    use_gyro_bias : bool
        If True, add a learnable intrinsic bias ``y ← y ⊕ exp_0([0, b])`` via Lorentz
        gyroaddition, ``b`` zero-initialized (gyrogroup identity ⇒ no-op at init) (default: False).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).

    References
    ----------
    Chen, Schölkopf, and Sebe. "Hyperbolic Busemann Neural Networks." 2026, Thm. 4.2.
    """

    def __init__(
        self,
        manifold_module: Hyperboloid,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        activation: Callable[[Array], Array] | None = None,
        v_max: float = 10.0,
        use_gyro_bias: bool = False,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        required_methods = ("expmap_0", "busemann")
        if use_gyro_bias:
            required_methods = ("expmap_0", "busemann", "embed_spatial_0", "addition")
        validate_hyperboloid_manifold(manifold_module, required_methods=required_methods)
        _assert_v_max_safe(v_max)

        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.activation = activation
        self.v_max = v_max

        in_spatial = in_dim - 1
        out_spatial = out_dim - 1
        # van Spengler / reference BFC init, computed from the spatial dims.
        self.kernel, self.log_scale, self.bias = init_weight_norm_params(
            rngs, out_spatial, in_spatial, std=(2 * in_spatial * out_spatial) ** -0.5, param_dtype=param_dtype
        )

        if use_gyro_bias:
            self.gyro_bias = nnx.Param(jnp.zeros((out_spatial,), dtype=param_dtype))
        else:
            self.gyro_bias = None

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """Forward pass returning a hyperboloid point, shape (batch, out_dim)."""
        u_BO = busemann_score(self.manifold, x, self.kernel[...], self.log_scale[...], self.bias[...], c, self.input_space)
        if self.activation is not None:
            u_BO = self.activation(u_BO)
        y_BAo = sinh_lift_to_hyperboloid(u_BO, c, self.v_max)

        if self.gyro_bias is not None:
            # Paper notation F(x) ⊕ b (Sec. 4.2), matching HypLinearHyperboloidPLFC's gyro-bias.
            # (The reference code applies b ⊕ F(x); the two coincide at the zero-init identity.)
            bias_tangent_Ao = self.manifold.embed_spatial_0(self.gyro_bias[...].astype(y_BAo.dtype))
            bias_point_Ao = self.manifold.expmap_0(bias_tangent_Ao, c)
            y_BAo = jax.vmap(self.manifold.addition, in_axes=(0, None, None))(y_BAo, bias_point_Ao, c)

        return y_BAo


class HypLinearPoincareBusemann(nnx.Module):
    """Busemann fully connected (BFC) layer (Poincaré ball model).

    Maps a Poincaré ball point to a Poincaré ball point (Chen et al. 2026, Thm. 4.1). Each
    output coordinate is the activated Busemann logit ``φ(-alpha_k·B^{v_k}(x) + b_k)``; the point is
    recovered via ``ω = sinh(√c·φ(u))/√c``, ``y = ω / (1 + √(1 + c‖ω‖²))``, then projected.

    Parameters
    ----------
    manifold_module : Poincare
        Class-based Poincaré manifold instance.
    in_dim : int
        Input spatial dimension (the ball has no time component).
    out_dim : int
        Output spatial dimension.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    input_space : str
        ``"manifold"`` (default) or ``"tangent"`` (lift via ``expmap_0`` first). Static for JIT.
    activation : Callable or None
        Optional Euclidean activation ``φ`` applied to the Busemann logits (default: identity).
        Avoid ``relu`` when stacking several of these layers on high-dimensional input — same
        origin-collapse risk as :class:`HypLinearHyperboloidBusemann` (see its ``activation`` doc);
        ``tanh`` avoids it.
    v_max : float
        Output-side guard: ``√c·φ(u)`` is hard-clipped to ``±v_max`` before the sinh map
        (default: 10.0).
    use_gyro_bias : bool
        If True, add a learnable intrinsic bias ``y ← y ⊕ exp_0(b)`` via Möbius addition,
        ``b`` zero-initialized (gyrogroup identity ⇒ no-op at init) (default: False).
    param_dtype : DTypeLike
        Storage dtype of the trainable parameters (default: jnp.float32).

    References
    ----------
    Chen, Schölkopf, and Sebe. "Hyperbolic Busemann Neural Networks." 2026, Thm. 4.1.
    """

    def __init__(
        self,
        manifold_module: Poincare,
        in_dim: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
        input_space: str = "manifold",
        activation: Callable[[Array], Array] | None = None,
        v_max: float = 10.0,
        use_gyro_bias: bool = False,
        param_dtype: DTypeLike = jnp.float32,
    ):
        if input_space not in ["tangent", "manifold"]:
            raise ValueError(f"input_space must be either 'tangent' or 'manifold', got '{input_space}'")

        required_methods = ("expmap_0", "busemann")
        if use_gyro_bias:
            required_methods = ("expmap_0", "busemann", "addition")
        validate_poincare_manifold(manifold_module, required_methods=required_methods)
        _assert_v_max_safe(v_max)

        self.manifold = manifold_module
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_space = input_space
        self.activation = activation
        self.v_max = v_max

        # No time component: spatial dim equals the model dim.
        in_spatial = in_dim
        out_spatial = out_dim
        self.kernel, self.log_scale, self.bias = init_weight_norm_params(
            rngs, out_spatial, in_spatial, std=(2 * in_spatial * out_spatial) ** -0.5, param_dtype=param_dtype
        )

        if use_gyro_bias:
            self.gyro_bias = nnx.Param(jnp.zeros((out_spatial,), dtype=param_dtype))
        else:
            self.gyro_bias = None

    def __call__(
        self,
        x: Float[Array, "batch in_dim"],
        c: float = 1.0,
    ) -> Float[Array, "batch out_dim"]:
        """Forward pass returning a Poincaré ball point, shape (batch, out_dim)."""
        u_BO = busemann_score(self.manifold, x, self.kernel[...], self.log_scale[...], self.bias[...], c, self.input_space)
        if self.activation is not None:
            u_BO = self.activation(u_BO)
        y_BO = busemann_fc_poincare_output(u_BO, c, self.v_max)
        # Closed form already lands strictly inside the ball; proj guards against float rounding.
        y_BO = jax.vmap(self.manifold.proj, in_axes=(0, None))(y_BO, c)

        if self.gyro_bias is not None:
            # Paper notation F(x) ⊕ b (Sec. 4.2): Möbius addition with the bias point on the right.
            bias_point_O = self.manifold.expmap_0(self.gyro_bias[...].astype(y_BO.dtype), c)
            y_BO = jax.vmap(self.manifold.addition, in_axes=(0, None, None))(y_BO, bias_point_O, c)

        return y_BO
