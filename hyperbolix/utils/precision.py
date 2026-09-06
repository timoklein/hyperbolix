r"""Matmul precision for the float32 dot products that touch manifold data.

:data:`MATMUL_PRECISION` (``jax.lax.Precision.HIGHEST``, not user-configurable) is pinned on
the sites where the *geometry* is computed, and nowhere else. Most layer weight GEMMs pass no
``precision`` keyword at all and follow JAX's own ``jax_default_matmul_precision``, which on
Ampere/Hopper means TF32. The exception is the point-to-hyperplane layers, whose kernel dot *is*
one of the pinned geometry einsums — see the two lists below.

Pinned to ``HIGHEST``
---------------------
- the vector-vector reductions inside ``manifolds/`` (behind ``dist``, ``addition``,
  ``gyration``, ``expmap``, ``logmap``, ``ptransp``, ``tangent_inner``, ``egrad2rgrad``, the
  Lorentz inner product, the Busemann coordinate and the boundary factors) and the
  ``decomposition/`` (HoroPCA) contractions. XLA lowers a vector-vector dot to a reduction
  rather than a tensor-core GEMM, so ``HIGHEST`` costs essentially nothing here;
- ``lorentz_midpoint`` and ``poincare_weighted_midpoint`` — the two Fréchet-mean normalizers,
  whose whole design is a cancellation-free identity that only holds to ``O(eps)`` if its
  inputs are float32-accurate;
- the patch extraction in the hyperboloid, Poincaré and proper-velocity convolutions. XLA
  implements ``conv_general_dilated_patches`` as a convolution against a 0/1 filter, i.e. a
  pure data copy, so TF32 there would round every input to a 10-bit mantissa for nothing;
- the attention score and aggregation einsums in ``hyperboloid_attention.py`` (the Lorentzian
  similarity, the softmax scores and weighted values, the linear-attention kernel sums) and the
  spatial residual projection that is added to that aggregate;
- the point-to-hyperplane einsums — ``Hyperboloid._compute_mlr``, ``Poincare._compute_mlr_pp``,
  ``ProperVelocity._compute_mlr``, and the ``HypRegressionHyperboloid`` / ``FGGLorentzMLR``
  heads — because the ``asinh`` argument they form is a difference of a radial and an angular
  term that cancels. The hidden PLFC, Poincaré++ and proper-velocity linear and convolution
  layers *are* those einsums applied to their own kernel, so their weight GEMM runs at
  ``HIGHEST`` too: ``HypLinearHyperboloidPLFC``, ``HypLinearPoincarePP``, ``HypLinearPV``,
  ``HypConv2DHyperboloidILNN``, ``HypConv2DPoincare`` and ``HypConv2DPV``.

Following JAX's default
-----------------------
The remaining layer weight GEMMs — ``HTCLinear`` (including the attention Q/K/V projections),
``FGGLinear``/``FGGConv2D``, the FHCNN and FHNN linears, ``HypLinearPoincare``,
``HypVQMLRPoincare``'s implicit-codebook matmul and ``hybrid_regularization``'s ``nnx.Linear``
— carry no ``precision`` keyword. These are the sites where ``HIGHEST`` actually costs
throughput (it replaces one TF32 pass with a three-pass float32 emulation; measured +5.5 % on a
jitted hyperbolic attention forward), and whether to pay that is the user's call, not the
library's.

One of them does carry a cancellation and is still left on the default: the FGG hidden dot
``x @ V`` absorbs the Minkowski metric into ``V``, so the ``-x0*V0 + <x_s, V_s>`` cancellation
happens *inside* a single accumulation — the same product ``FGGLorentzMLR`` pins. The comment on
the ``jnp.matmul`` in ``nn_layers/hyperboloid_linear._fgg_linear_forward`` states the reasoning.
The depth grid below is why it is left alone: **at depth 16** an FGG stack's float32 gradient
error under the default is smaller than an ``HTCLinear`` stack's, which has no cancellation at
all, so the cancellation is not what dominates there.

To run **everything** in full float32, set JAX's own knob — one line, and jit-cache aware, so
changing it re-traces::

    jax.config.update("jax_default_matmul_precision", "highest")

or scoped to a block::

    with jax.default_matmul_precision("highest"):
        logits = model(x)

A deep, fully hyperbolic stack is a good reason to do so: the pinned sites keep the geometry
honest, but a TF32 error introduced in an early layer's weight GEMM is carried forward by
every layer after it. Measured on an A100 (jax 0.9.1, ambient dim 33, batch 64,
``c in {0.1, 1.0}`` x spatial radius ``in {0.5, 5, 20}``, float32 against a float64 reference):
``FGGLinear`` 2.6e-4 under TF32 against 6.8e-8 under ``HIGHEST``, ``FGGLorentzMLR``
1.3e-4 … 3.6e-4 against 3.6e-8 … 9.6e-8 — roughly ~1e-4 against ~1e-7 on the layers, three of
float32's seven significant digits.

Depth decides whether that matters, and at depth 2 it does not. An independent teacher-student
comparison (5 seeds per arm, ``D = 128``, ``B = 256``, ``c = 1``, 2 000 Adam updates,
independent A100 measurement, jax 0.9.1) found no mean heldout-loss difference between the
default and global ``HIGHEST``: 0.27736 against 0.27738 for an ``HTCLinear`` stack, 0.28610
against 0.28614 for an ``FGGLinear`` one, against a per-arm seed spread of ~0.015-0.017. The
float32-vs-float64 gradient error does grow with depth. At depth 16, ``c = 1``, input geodesic
radius 5 (same measurement; the grid is in
``logs/2026-09-06_numerics_independent_review/training/gpu/depth.json``): the ``HTCLinear``
stack's parameter gradient is 1.539 % relative L2 off a float64 reference under the default
against 2.5e-6 under global ``HIGHEST``, and its input gradient 2.013 % against 1.2e-5; the
``FGGLinear`` stack at the same cell is 0.199 % against 8.0e-7 and 0.274 % against 1.0e-6. That
cell is the **worst** of a 24-cell grid (depths 2/8/16, ``c in {0.1, 1}``, input radius
``{0.5, 5}``), with one initialisation and one input draw per cell and no seeds — so these are
maxima over the grid, not means, and n = 1 per cell. For a deep or gradient-sensitive stack, set
the global knob (the snippet above).

Why the geometry is pinned rather than left to the same knob
------------------------------------------------------------
On Ampere/Hopper, XLA:GPU defaults float32 matmuls to TF32, whose 10-bit mantissa carries
~1e-3 relative error — far coarser than float32's ~1e-7. Hyperbolic geometry spends that
budget on cancellations whose entire design assumes float32 accuracy:

- the Lorentz inner product ``<x,y>_L = -x_0 y_0 + <x_s, x_s>``, formed as a difference of two
  dots (hyperboloid ``dist``/``expmap``/``ptransp``, the attention scores, the Busemann
  coordinate, the HoroPCA ideal-span projections);
- the ``lorentz_midpoint`` normalizer, a cancellation-free identity that exists precisely to
  keep the float32 error at ``O(eps)`` at any radius;
- the Poincaré/stereographic boundary factors ``1 - c*||x||^2`` and
  ``1 - 2c<x,y> + c^2||x||^2||y||^2``, which cancel to ``O(eps)`` for a near-boundary point;
- the MLR ``asinh`` arguments on both models, a difference of a radial and an angular term.

Measured on an A100 (jax 0.9.1): the f32-vs-f64 relative error of ``lorentz_midpoint`` is
4.6e-5 … 2.6e-4 at the TF32 default and 2.6e-8 … 1.6e-7 with ``HIGHEST`` — a ~2000x accuracy
loss that ``HIGHEST`` removes. The eager float32 gradient of the hyperboloid MLR improves from
1.5e-4 to 2.7e-7 relative over the same switch. These are not throughput-critical sites, so
the library pays for the accuracy rather than exposing the choice.

This module lives in ``utils/`` — a neutral home both ``manifolds/`` and ``nn_layers/`` can
import from, since ``manifolds/`` must not depend on ``nn_layers/``. ``MATMUL_PRECISION`` was
originally defined in ``hyperbolix.nn_layers.hyperboloid_core``, which re-exports it, so
``hyperboloid_core.MATMUL_PRECISION`` keeps working.

``HIGHEST`` is a no-op on CPU (no TF32 path) and for float64 anywhere, so the pinning costs
nothing outside float32 GPU matmuls.
"""

import jax

MATMUL_PRECISION = jax.lax.Precision.HIGHEST
"""Precision for every dot product that computes geometry rather than a layer's weights.

Fixed at ``HIGHEST`` and not user-configurable: accuracy here is a property of the geometry,
not a knob. The module docstring lists the pinned sites and the measurements behind them. The
layer weight GEMMs pass no ``precision`` keyword and follow ``jax_default_matmul_precision``
instead — all but the point-to-hyperplane layers, whose kernel dot is one of the pinned
einsums.
"""
