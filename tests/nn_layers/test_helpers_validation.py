"""Tests for the shared ``nn_layers`` helpers and the Busemann Poincaré output map.

Three things that no per-layer file covers:

* ``as_pair`` — every conv layer normalizes ``kernel_size`` / ``stride`` through it, but the
  whole suite only ever passes square kernels and isotropic strides, so the tuple branch was
  never distinguished from ``(x[0], x[0])``.
* ``validate_{hyperboloid,poincare,pv}_manifold`` — the error path is what tells a user which
  manifold family a layer belongs to; nothing asserted that the message names the right one.
* ``busemann_fc_poincare_output`` — Chen et al. 2026 gives the Poincaré BFC output map
  (Thm. 4.1) and the hyperboloid one (Thm. 4.2) as two faces of the same construction; the
  two code paths are written independently and must agree through the model isometry.

Dimension key:
  B: batch size      O: out_spatial      H: image height     W: image width
  C: channels
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.manifolds.isometry_mappings import hyperboloid_to_poincare
from hyperbolix.nn_layers import HypConv2DHyperboloid, HypConv2DPoincare, sinh_lift_to_hyperboloid
from hyperbolix.nn_layers._helpers import (
    as_pair,
    validate_hyperboloid_manifold,
    validate_poincare_manifold,
    validate_pv_manifold,
)
from hyperbolix.nn_layers.busemann_core import busemann_fc_poincare_output

poincare_f64 = Poincare(dtype=jnp.float64)
hyperboloid_f64 = Hyperboloid(dtype=jnp.float64)


# ===================================================================
# as_pair — the tuple branch
# ===================================================================


def test_as_pair_normalizes_int_and_preserves_tuple_order():
    """``as_pair`` duplicates an int and passes a pair through *unswapped*."""
    assert as_pair(3) == (3, 3)
    assert as_pair((3, 2)) == (3, 2)
    assert as_pair((2, 3)) == (2, 3)


@pytest.mark.parametrize(
    "padding,expected_hw",
    [
        # VALID: floor((in - k)/s) + 1 per axis, computed here from the input geometry
        # rather than read back off the layer.
        ("VALID", ((9 - 3) // 2 + 1, (7 - 2) // 1 + 1)),
        # SAME: ceil(in / s) per axis.
        ("SAME", (-(-9 // 2), -(-7 // 1))),
    ],
)
def test_conv_honours_non_square_kernel_and_anisotropic_stride(padding, expected_hw):
    """A (3, 2) kernel with a (2, 1) stride produces the two *different* output extents.

    The suite's conv tests all use square kernels and equal strides, under which
    ``as_pair((h, w)) -> (h, h)`` (or a reversed pair) is indistinguishable from the correct
    normalization. Non-square input (9 x 7) is used as well, so a height/width transposition
    anywhere in the patch extraction shows up as a shape error rather than silently matching.
    """
    B, C_in, C_out = 2, 2, 4
    layer = HypConv2DPoincare(
        poincare_f64,
        C_in,
        C_out,
        (3, 2),
        stride=(2, 1),
        padding=padding,
        rngs=nnx.Rngs(0),
    )
    assert layer.kernel_size == (3, 2)
    assert layer.stride == (2, 1)

    x_BHWC = jax.random.normal(jax.random.PRNGKey(0), (B, 9, 7, C_in), dtype=jnp.float64) * 0.1
    y = layer(x_BHWC, c=1.0)

    assert y.shape == (B, expected_hw[0], expected_hw[1], C_out)
    assert jnp.all(jnp.isfinite(y))


def test_conv_int_and_equivalent_tuple_are_bitwise_identical():
    """``kernel_size=3, stride=2`` and ``(3, 3), (2, 2)`` are the same layer, value for value.

    Both layers are built from the same seed, so any difference would have to come from the
    normalization itself.
    """
    B, C_in, C_out = 2, 2, 4
    kwargs = {"padding": "SAME", "rngs": nnx.Rngs(0)}
    layer_int = HypConv2DPoincare(poincare_f64, C_in, C_out, 3, stride=2, **kwargs)
    layer_tuple = HypConv2DPoincare(poincare_f64, C_in, C_out, (3, 3), stride=(2, 2), **kwargs)

    x_BHWC = jax.random.normal(jax.random.PRNGKey(1), (B, 9, 7, C_in), dtype=jnp.float64) * 0.1
    assert jnp.array_equal(layer_int(x_BHWC, c=1.0), layer_tuple(x_BHWC, c=1.0))


# ===================================================================
# Manifold-family validation
# ===================================================================


class _NotAManifold:
    """An object with none of the manifold methods — stands in for a wrong-family instance."""


@pytest.mark.parametrize(
    "validator,expected_name,forbidden_names",
    [
        (validate_hyperboloid_manifold, "Hyperboloid", ("Poincare", "ProperVelocity")),
        (validate_poincare_manifold, "Poincare", ("Hyperboloid", "ProperVelocity")),
        (validate_pv_manifold, "ProperVelocity", ("Hyperboloid",)),
    ],
    ids=["hyperboloid", "poincare", "pv"],
)
def test_validator_names_its_own_manifold_family(validator, expected_name, forbidden_names):
    """The TypeError must name the family the layer needs, and no other one.

    A copy-paste slip in ``_validate_manifold_methods``'s keyword arguments would send a user
    of a hyperboloid layer off to instantiate a Poincaré ball; the message text is the only
    thing that distinguishes the three wrappers, since they share one implementation.
    """
    with pytest.raises(TypeError) as excinfo:
        validator(_NotAManifold(), required_methods=("expmap_0", "logmap_0"))  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert expected_name in message
    for other in forbidden_names:
        assert other not in message
    # The message must also point at something constructible, not just a family name.
    assert f"hyperbolix.manifolds.{expected_name}()" in message


def test_validator_accepts_a_manifold_with_the_required_methods():
    """No exception when every required method is present (the guard is not unconditional)."""
    validate_poincare_manifold(poincare_f64, required_methods=("expmap_0", "logmap_0", "compute_mlr_pp"))
    validate_hyperboloid_manifold(hyperboloid_f64, required_methods=("expmap_0", "minkowski_inner"))


def test_layers_reject_the_wrong_manifold_family():
    """The wiring, not just the helper: each layer forwards its own family name.

    ``Hyperboloid`` has no ``compute_mlr_pp`` and ``Poincare`` has no ``hcat``, so each layer
    below sees a genuinely unusable instance of the other family.
    """
    with pytest.raises(TypeError, match="Poincare"):
        HypConv2DPoincare(hyperboloid_f64, 2, 4, 3, rngs=nnx.Rngs(0))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Hyperboloid"):
        HypConv2DHyperboloid(poincare_f64, 3, 5, 3, rngs=nnx.Rngs(0))  # type: ignore[arg-type]


# ===================================================================
# Busemann Poincaré output map: Thm. 4.1 vs Thm. 4.2 (Chen et al. 2026)
# ===================================================================


def _ref_busemann_fc_poincare(u_BO, c, v_max):
    """NumPy transcription of Chen et al. 2026, Thm. 4.1.

    ``omega = sinh(clip(sqrt(c)*u, +-v_max)) / sqrt(c)``, ``y = omega / (1 + sqrt(1 + c*||omega||^2))``.
    """
    u = np.asarray(u_BO, np.float64)
    sqrt_c = np.sqrt(c)
    omega = np.sinh(np.clip(sqrt_c * u, -v_max, v_max)) / sqrt_c
    denom = 1.0 + np.sqrt(1.0 + c * (omega**2).sum(-1, keepdims=True))
    return omega / denom


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_busemann_fc_poincare_output_matches_thm_4_1_transcription(c):
    """The Poincaré output map reproduces a hand-written Thm. 4.1 reference."""
    v_max = 5.0
    u_BO = jax.random.normal(jax.random.PRNGKey(2), (4, 5), dtype=jnp.float64)

    got = busemann_fc_poincare_output(u_BO, c, v_max)
    assert jnp.allclose(got, _ref_busemann_fc_poincare(u_BO, c, v_max), rtol=0, atol=1e-14)

    # The map always lands strictly inside the radius-1/sqrt(c) ball (the layer's ``proj`` is
    # only a rounding guard), and the radius genuinely depends on c.
    assert jnp.all(jnp.sqrt(c) * jnp.linalg.norm(got, axis=-1) < 1.0)


@pytest.mark.parametrize("c", [0.3, 1.0, 2.5])
def test_busemann_poincare_and_hyperboloid_output_maps_agree_through_the_isometry(c):
    """Thm. 4.1 == stereographic projection of Thm. 4.2, for the same activated logits.

    ``busemann_fc_poincare_output`` (Poincaré BFC) and ``sinh_lift_to_hyperboloid`` (the
    hyperboloid BFC / PLFC output map) are written independently in different modules; the
    papers present them as the same construction in the two models, so composing the
    hyperboloid one with ``hyperboloid_to_poincare`` must return the Poincaré one.  Dropping
    the curvature from either denominator, or clipping in one path but not the other, breaks
    the agreement at c != 1 while leaving each path internally plausible.
    """
    v_max = 5.0
    u_BO = jax.random.normal(jax.random.PRNGKey(3), (4, 5), dtype=jnp.float64)

    direct = busemann_fc_poincare_output(u_BO, c, v_max)
    lifted_BA = sinh_lift_to_hyperboloid(u_BO, c, v_max)
    via_hyperboloid = jax.vmap(hyperboloid_to_poincare, in_axes=(0, None))(lifted_BA, c)

    assert jnp.allclose(direct, via_hyperboloid, rtol=0, atol=1e-14)
    # Non-vacuous: the two agree on a non-degenerate point set, not on a collapsed origin.
    assert float(jnp.max(jnp.abs(direct))) > 0.05


def test_busemann_output_maps_clip_identically_in_the_saturated_tail():
    """Beyond ``v_max`` both paths freeze at the same clipped point.

    ``sinh`` exponentiates its argument, so the shared ``clip`` is what keeps the layer finite;
    a clip present in only one of the two output maps would make the models disagree exactly
    where the guard matters.
    """
    c, v_max = 1.0, 2.0
    u_BO = jnp.array([[10.0, -10.0, 0.5], [3.0, 3.0, 3.0]], dtype=jnp.float64)
    saturated = jnp.array([[v_max, -v_max, 0.5], [v_max, v_max, v_max]], dtype=jnp.float64)

    direct = busemann_fc_poincare_output(u_BO, c, v_max)
    via_hyperboloid = jax.vmap(hyperboloid_to_poincare, in_axes=(0, None))(sinh_lift_to_hyperboloid(u_BO, c, v_max), c)
    assert jnp.allclose(direct, via_hyperboloid, rtol=0, atol=1e-14)

    # The clip is active: the far-out inputs give the same answer as the bound itself.
    assert jnp.allclose(direct, busemann_fc_poincare_output(saturated, c, v_max), rtol=0, atol=1e-14)
