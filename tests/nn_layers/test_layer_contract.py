"""Shared forward/gradient/JIT contract for every hyperbolic NN layer.

This module replaces the six-slot boilerplate template
(``forward_shape`` / ``on_manifold`` / ``jitted_forward`` / ``gradient`` /
``jitted_gradient`` / ``tangent_input`` [+ ``stride`` / ``different_curvatures``
for convs]) that used to be copy-pasted, once per layer class, across twelve
test files.  Every layer is described by one :class:`LayerSpec` row and the
contract tests below are parametrized over the table.

The contract is *stronger* than the clones it replaces:

* ``test_forward_contract`` asserts ``jit(layer)(x) == layer(x)``; the old
  ``test_jitted_forward`` clones only re-asserted shape/finiteness and never
  compared against the eager result.
* ``test_gradient_contract`` asserts at least one kernel gradient entry is
  nonzero (previously only ``test_pv_regression.py`` did this) and that the
  jitted gradient equals the eager gradient (previously never compared).
* ``test_tangent_input_contract`` asserts the two ``input_space`` branches
  *agree* — ``layer(v, input_space="tangent") == layer(expmap_0(v),
  input_space="manifold")`` — instead of merely checking that both branches run.
* ``test_output_on_manifold_across_curvatures`` sweeps curvature for every
  manifold-valued layer, not only for the three conv files that happened to have
  a curvature ``for``-loop (whose first failure also hid the later curvatures).

Dtype parametrization follows the audit's B1 rule: the ``dtype`` axis is kept
only where float32-vs-float64 can change the outcome (the gradient tests, where
float32 is the NaN-prone path).  Shape/structural assertions run at float32
only — output shape and the jit/eager equality are dtype-independent code paths
— and the tangent-branch equality runs at float64, where the two branches must
agree to ~1e-10 rather than ~1e-5.

Per-layer deviations that are *not* cosmetic live in the spec table as data:
the extra parameters some layers own (``scale`` on the FHNN family,
``log_scale``/``gyro_bias`` on the Busemann family, the nested
``xi_theta.kernel``/``xi_theta.bias`` of ``HyperPPFeatureScaling``) are listed
in ``grad_paths`` and asserted finite by ``test_gradient_contract``; whether a
layer returns a manifold point or Euclidean logits is ``output_is_manifold_point``;
whether it has an ``input_space`` option is ``has_input_space``.

Dimension key:
  B: batch size        H, W: feature-map height/width
  I: input dim         O: output dim
  C: channels          N: flattened points (B*H*W)
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, reduce

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid, Poincare, ProperVelocity
from hyperbolix.nn_layers import (
    HypConv2DHyperboloid,
    HypConv2DHyperboloidFHNN,
    HypConv2DHyperboloidILNN,
    HypConv2DPV,
    HyperPPFeatureScaling,
    HypLinearHyperboloidBusemann,
    HypLinearHyperboloidFHNN,
    HypLinearHyperboloidPLFC,
    HypLinearPoincare,
    HypLinearPoincareBusemann,
    HypLinearPoincarePP,
    HypLinearPV,
    HypRegressionHyperboloid,
    HypRegressionHyperboloidBusemann,
    HypRegressionPoincare,
    HypRegressionPoincareBusemann,
    HypRegressionPoincarePP,
    HypRegressionPV,
)

# Busemann layers were tested at c=0.7 upstream; keep that curvature so the
# consolidated contract still exercises a non-unit curvature for them.
C_BUSEMANN = 0.7

# Conv geometry shared by every conv spec (small on purpose: 18 specs x jit).
CONV_B, CONV_HW = 2, 4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _hyperboloid(dtype):
    return Hyperboloid(dtype=dtype)


def _poincare(dtype):
    return Poincare(dtype=dtype)


def _proper_velocity(dtype):
    return ProperVelocity(dtype=dtype)


def _tangent(shape, dtype, *, zero_time: bool = False, scale: float = 0.1, seed: int = 0):
    """Tangent vector(s) at the origin, last axis = ambient/embedding dim.

    ``zero_time`` zeroes coordinate 0, which the hyperboloid model requires of a
    tangent vector at the origin (``<v, e_0>_L = 0``).
    """
    v = jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=dtype) * scale
    if zero_time:
        v = v.at[..., 0].set(0.0)
    return v


def _lift(v, manifold, c):
    """``expmap_0`` every vector along the last axis (works for (B, I) and (B, H, W, C))."""
    flat_ND = v.reshape(-1, v.shape[-1])
    lifted_ND = jax.vmap(manifold.expmap_0, in_axes=(0, None))(flat_ND, c)
    return lifted_ND.reshape(v.shape)


def _resolve(obj, path: str):
    """Follow a dotted attribute path, e.g. ``"xi_theta.kernel"``."""
    return reduce(getattr, path.split("."), obj)


def _tol(dtype, f32, f64):
    return f32 if dtype == jnp.float32 else f64


# --------------------------------------------------------------------------- #
# Spec table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LayerSpec:
    """One row per layer class: how to build it, feed it, and what it promises."""

    id: str
    make_layer: Callable  # (dtype, input_space) -> nnx.Module
    make_tangent: Callable  # (dtype) -> tangent-space input array
    out_shape: tuple[int, ...]
    grad_paths: tuple[str, ...]
    manifold_fn: Callable | None  # (dtype) -> manifold instance; None => no manifold
    output_is_manifold_point: bool
    has_input_space: bool = True
    c: float = 1.0
    # Conv-only: (dtype, kernel_size, padding, stride) -> nnx.Module
    make_conv: Callable | None = None
    conv_in_channels: int = 3
    conv_out_channels: int = 3


def _linear_spec(
    layer_cls,
    *,
    id_,
    manifold_fn,
    in_dim,
    out_dim,
    output_is_manifold_point,
    grad_paths=("kernel", "bias"),
    zero_time=False,
    c=1.0,
    batch=8,
    ctor_kwargs=None,
):
    """Spec for a layer with the ``(manifold, in_dim, out_dim)`` positional signature."""
    kwargs = dict(ctor_kwargs or {})
    return LayerSpec(
        id=id_,
        make_layer=lambda dtype, input_space: layer_cls(
            manifold_fn(dtype), in_dim, out_dim, rngs=nnx.Rngs(0), input_space=input_space, **kwargs
        ),
        make_tangent=lambda dtype: _tangent((batch, in_dim), dtype, zero_time=zero_time, scale=0.2),
        out_shape=(batch, out_dim),
        grad_paths=grad_paths,
        manifold_fn=manifold_fn,
        output_is_manifold_point=output_is_manifold_point,
        c=c,
    )


def _conv_spec(
    layer_cls,
    *,
    id_,
    manifold_fn,
    in_channels=3,
    out_channels=3,
    grad_paths=("kernel", "bias"),
    zero_time=False,
):
    """Spec for a conv layer with the keyword ``(manifold_module, in_channels, ...)`` signature."""

    def make_conv(dtype, kernel_size, padding, stride, input_space="manifold"):
        return layer_cls(
            manifold_module=manifold_fn(dtype),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_space=input_space,
            rngs=nnx.Rngs(0),
        )

    return LayerSpec(
        id=id_,
        make_layer=lambda dtype, input_space: make_conv(dtype, 2, "SAME", 1, input_space),
        make_tangent=lambda dtype: _tangent((CONV_B, CONV_HW, CONV_HW, in_channels), dtype, zero_time=zero_time, scale=0.2),
        out_shape=(CONV_B, CONV_HW, CONV_HW, out_channels),
        grad_paths=grad_paths,
        manifold_fn=manifold_fn,
        output_is_manifold_point=True,
        make_conv=make_conv,
        conv_in_channels=in_channels,
        conv_out_channels=out_channels,
    )


SPECS: tuple[LayerSpec, ...] = (
    # --- Proper Velocity (unconstrained model: is_in_manifold reduces to isfinite,
    #     so the on-manifold check here is structural, not a numeric oracle) ---
    _linear_spec(
        HypLinearPV,
        id_="pv_linear",
        manifold_fn=_proper_velocity,
        in_dim=5,
        out_dim=3,
        output_is_manifold_point=True,
    ),
    _linear_spec(
        HypRegressionPV,
        id_="pv_regression",
        manifold_fn=_proper_velocity,
        in_dim=5,
        out_dim=3,
        output_is_manifold_point=False,
    ),
    _conv_spec(HypConv2DPV, id_="pv_conv", manifold_fn=_proper_velocity),
    # --- Poincare regression heads (Euclidean logits) ---
    _linear_spec(
        HypRegressionPoincare,
        id_="regression_poincare",
        manifold_fn=_poincare,
        in_dim=5,
        out_dim=3,
        output_is_manifold_point=False,
    ),
    _linear_spec(
        HypRegressionPoincarePP,
        id_="regression_poincare_pp",
        manifold_fn=_poincare,
        in_dim=5,
        out_dim=3,
        output_is_manifold_point=False,
    ),
    _linear_spec(
        HypRegressionHyperboloid,
        id_="regression_hyperboloid",
        manifold_fn=_hyperboloid,
        in_dim=5,
        out_dim=3,
        output_is_manifold_point=False,
        zero_time=True,
    ),
    # --- Poincare linear layers (manifold points) ---
    _linear_spec(
        HypLinearPoincare,
        id_="linear_poincare",
        manifold_fn=_poincare,
        in_dim=5,
        out_dim=3,
        output_is_manifold_point=True,
    ),
    _linear_spec(
        HypLinearPoincarePP,
        id_="linear_poincare_pp",
        manifold_fn=_poincare,
        in_dim=5,
        out_dim=3,
        output_is_manifold_point=True,
    ),
    # --- Busemann heads (logits) and BFC layers (manifold points) ---
    _linear_spec(
        HypRegressionHyperboloidBusemann,
        id_="bmlr_hyperboloid",
        manifold_fn=_hyperboloid,
        in_dim=9,
        out_dim=5,
        output_is_manifold_point=False,
        grad_paths=("kernel", "log_scale", "bias"),
        zero_time=True,
        c=C_BUSEMANN,
    ),
    _linear_spec(
        HypRegressionPoincareBusemann,
        id_="bmlr_poincare",
        manifold_fn=_poincare,
        in_dim=8,
        out_dim=5,
        output_is_manifold_point=False,
        grad_paths=("kernel", "log_scale", "bias"),
        c=C_BUSEMANN,
    ),
    _linear_spec(
        HypLinearHyperboloidBusemann,
        id_="bfc_hyperboloid",
        manifold_fn=_hyperboloid,
        in_dim=9,
        out_dim=7,
        output_is_manifold_point=True,
        grad_paths=("kernel", "log_scale", "bias", "gyro_bias"),
        zero_time=True,
        c=C_BUSEMANN,
        ctor_kwargs={"use_gyro_bias": True},
    ),
    _linear_spec(
        HypLinearPoincareBusemann,
        id_="bfc_poincare",
        manifold_fn=_poincare,
        in_dim=8,
        out_dim=6,
        output_is_manifold_point=True,
        grad_paths=("kernel", "log_scale", "bias", "gyro_bias"),
        c=C_BUSEMANN,
        ctor_kwargs={"use_gyro_bias": True},
    ),
    # --- Hyperboloid linear layers ---
    _linear_spec(
        HypLinearHyperboloidFHNN,
        id_="linear_fhnn",
        manifold_fn=_hyperboloid,
        in_dim=6,
        out_dim=10,
        output_is_manifold_point=True,
        grad_paths=("kernel", "bias", "scale"),
        zero_time=True,
    ),
    _linear_spec(
        HypLinearHyperboloidPLFC,
        id_="linear_plfc",
        manifold_fn=_hyperboloid,
        in_dim=6,
        out_dim=10,
        output_is_manifold_point=True,
        zero_time=True,
    ),
    # --- Hyperboloid conv layers ---
    _conv_spec(
        HypConv2DHyperboloidFHNN,
        id_="conv_fhnn",
        manifold_fn=_hyperboloid,
        grad_paths=("kernel", "bias", "scale"),
        zero_time=True,
    ),
    _conv_spec(HypConv2DHyperboloidILNN, id_="conv_ilnn", manifold_fn=_hyperboloid, zero_time=True),
    _conv_spec(HypConv2DHyperboloid, id_="conv_hyperboloid", manifold_fn=_hyperboloid, zero_time=True),
    # --- Hybrid: Euclidean feature scaling, no manifold and no input_space ---
    LayerSpec(
        id="hyperpp_feature_scaling",
        make_layer=lambda dtype, input_space: HyperPPFeatureScaling(dim=16, alpha=0.9, rngs=nnx.Rngs(0)),
        make_tangent=lambda dtype: _tangent((8, 16), dtype, scale=1.0),
        out_shape=(8, 16),
        grad_paths=("xi_theta.kernel", "xi_theta.bias"),
        manifold_fn=None,
        output_is_manifold_point=False,
        has_input_space=False,
    ),
)

CONV_SPECS = tuple(s for s in SPECS if s.make_conv is not None)
MANIFOLD_SPECS = tuple(s for s in SPECS if s.output_is_manifold_point)
TANGENT_SPECS = tuple(s for s in SPECS if s.has_input_space)
# ProperVelocity is unconstrained (``is_in_manifold`` == ``isfinite``), so a
# curvature sweep of its membership check carries no information.
CURVATURE_SPECS = tuple(s for s in MANIFOLD_SPECS if s.manifold_fn is not _proper_velocity)


def _ids(specs):
    return [s.id for s in specs]


def _make_input(spec, dtype, c):
    """The layer's ``input_space="manifold"`` input: a lifted tangent draw."""
    v = spec.make_tangent(dtype)
    if spec.manifold_fn is None:
        return v
    return _lift(v, spec.manifold_fn(dtype), c)


def _assert_on_manifold(spec, y, dtype, c):
    manifold = spec.manifold_fn(dtype)
    y_ND = y.reshape(-1, y.shape[-1])
    on = jax.vmap(partial(manifold.is_in_manifold, c=c))(y_ND)
    assert bool(on.all()), f"{spec.id}: output left the manifold at c={c}"


# --------------------------------------------------------------------------- #
# Contract 1 — forward: shape, dtype, finiteness, manifold membership, jit == eager
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", SPECS, ids=_ids(SPECS))
def test_forward_contract(spec):
    """Shape/dtype/finiteness of the forward, and that ``nnx.jit`` reproduces it exactly.

    Structural assertions, so float32 only (B1): output shape and the jit/eager
    code path do not branch on dtype, and float32 is the stricter dtype check
    (it is the one that a silent x64 promotion would break).
    """
    dtype = jnp.float32
    layer = spec.make_layer(dtype, "manifold")
    x = _make_input(spec, dtype, spec.c)

    y = layer(x, c=spec.c)

    assert y.shape == spec.out_shape
    assert y.dtype == dtype
    assert jnp.isfinite(y).all()
    if spec.output_is_manifold_point:
        _assert_on_manifold(spec, y, dtype, spec.c)

    @nnx.jit
    def forward(module, inputs, curvature):
        return module(inputs, c=curvature)

    y_jit = forward(layer, x, spec.c)
    assert y_jit.shape == y.shape
    assert jnp.allclose(y_jit, y, atol=1e-5, rtol=1e-5), f"{spec.id}: jitted forward differs from eager"


# --------------------------------------------------------------------------- #
# Contract 2 — manifold membership across curvatures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", [0.5, 2.0])
@pytest.mark.parametrize("spec", CURVATURE_SPECS, ids=_ids(CURVATURE_SPECS))
def test_output_on_manifold_across_curvatures(spec, c):
    """A manifold-valued layer must return manifold points at non-unit curvature too."""
    dtype = jnp.float32
    layer = spec.make_layer(dtype, "manifold")
    x = _make_input(spec, dtype, c)

    y = layer(x, c=c)

    assert jnp.isfinite(y).all()
    _assert_on_manifold(spec, y, dtype, c)


# --------------------------------------------------------------------------- #
# Contract 3 — gradients: finite, nonzero, and jit == eager
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("spec", SPECS, ids=_ids(SPECS))
def test_gradient_contract(spec, dtype):
    """Every declared parameter gets a finite gradient, the first one is nonzero,
    and ``nnx.jit`` produces the same gradient as the eager computation.

    The dtype axis is load-bearing here: float32 is where an unguarded ``norm``
    or ``atanh`` at a singular point turns a gradient into NaN.
    """
    layer = spec.make_layer(dtype, "manifold")
    x = _make_input(spec, dtype, spec.c)

    def loss_fn(model):
        return jnp.sum(model(x, c=spec.c) ** 2)

    loss, grads_eager = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    for path in spec.grad_paths:
        g = _resolve(grads_eager, path)[...]
        assert jnp.isfinite(g).all(), f"{spec.id}: non-finite gradient for {path}"

    # Non-trivial signal: at least one entry of the primary parameter moves.
    primary = _resolve(grads_eager, spec.grad_paths[0])[...]
    assert jnp.any(jnp.abs(primary) > 0), f"{spec.id}: gradient for {spec.grad_paths[0]} is identically zero"

    @nnx.jit
    def jitted_grad(module, inputs, curvature):
        return nnx.grad(lambda m: jnp.sum(m(inputs, c=curvature) ** 2))(module)

    grads_jit = jitted_grad(layer, x, spec.c)

    # Gradient entries within one kernel span several orders of magnitude, so the
    # jit/eager tolerance is scaled by the largest entry rather than absolute.
    rel = _tol(dtype, 1e-5, 1e-11)
    for path in spec.grad_paths:
        g_e = _resolve(grads_eager, path)[...]
        g_j = _resolve(grads_jit, path)[...]
        atol = rel * float(jnp.maximum(jnp.max(jnp.abs(g_e)), 1.0))
        assert jnp.allclose(g_j, g_e, atol=atol, rtol=1e-4), f"{spec.id}: jitted gradient differs for {path}"


# --------------------------------------------------------------------------- #
# Contract 4 — the two input_space branches must agree
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", TANGENT_SPECS, ids=_ids(TANGENT_SPECS))
def test_tangent_input_contract(spec):
    """``input_space="tangent"`` is exactly an internal ``expmap_0`` then the manifold branch.

    The old ``test_tangent_input`` / ``test_input_space`` clones only checked
    that each branch produced a finite on-manifold output; they never checked
    that the branches agree, so a dropped or wrongly-curved internal lift would
    have survived.  float64 only: this is an exact-agreement assertion, and at
    float32 the tolerance would have to be loosened by five orders of magnitude.
    """
    dtype = jnp.float64
    manifold = spec.manifold_fn(dtype)
    v = spec.make_tangent(dtype)
    x = _lift(v, manifold, spec.c)

    layer_tangent = spec.make_layer(dtype, "tangent")
    layer_manifold = spec.make_layer(dtype, "manifold")

    y_tangent = layer_tangent(v, c=spec.c)
    y_manifold = layer_manifold(x, c=spec.c)

    assert y_tangent.shape == spec.out_shape
    assert jnp.allclose(y_tangent, y_manifold, atol=1e-9), (
        f"{spec.id}: input_space='tangent' disagrees with an explicit expmap_0 + 'manifold'"
    )


# --------------------------------------------------------------------------- #
# Contract 5 — conv spatial arithmetic (kernel size, padding, stride)
# --------------------------------------------------------------------------- #
_CONV_SHAPE_CASES = [
    # (kernel_size, padding, stride)
    (1, "SAME", 1),
    (3, "VALID", 1),
    (3, "SAME", 2),
]


@pytest.mark.parametrize("kernel_size, padding, stride", _CONV_SHAPE_CASES)
@pytest.mark.parametrize("spec", CONV_SPECS, ids=_ids(CONV_SPECS))
def test_conv_spatial_shape(spec, kernel_size, padding, stride):
    """Output spatial size follows the padding/stride arithmetic (dtype-independent)."""
    dtype = jnp.float32
    height = width = 8
    v = _tangent(
        (CONV_B, height, width, spec.conv_in_channels),
        dtype,
        zero_time=spec.manifold_fn is _hyperboloid,
        scale=0.2,
    )
    x = _lift(v, spec.manifold_fn(dtype), 1.0)

    layer = spec.make_conv(dtype, kernel_size, padding, stride)
    y = layer(x, c=1.0)

    if padding == "SAME":
        expected_h = (height + stride - 1) // stride
        expected_w = (width + stride - 1) // stride
    else:
        expected_h = (height - kernel_size) // stride + 1
        expected_w = (width - kernel_size) // stride + 1

    assert y.shape == (CONV_B, expected_h, expected_w, spec.conv_out_channels)
    assert jnp.isfinite(y).all()
