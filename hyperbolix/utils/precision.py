r"""Matmul precision for the float32 dot products that touch manifold data.

:data:`MATMUL_PRECISION` (``jax.lax.Precision.HIGHEST``, not user-configurable) is pinned on
the sites where the *geometry* is computed, and nowhere else. A layer's weight GEMM passes no
``precision`` keyword at all and follows JAX's own ``jax_default_matmul_precision``, which on
Ampere/Hopper means TF32, with two exceptions: the point-to-hyperplane layers, whose kernel dot
*is* one of the pinned geometry einsums, and the FGG layers, whose ``x @ V`` is a Minkowski inner
product — see the two lists below.

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
  ``HypConv2DHyperboloidILNN``, ``HypConv2DPoincare`` and ``HypConv2DPV``;
- the FGG product ``x @ V`` in ``nn_layers/hyperboloid_linear._fgg_linear_forward``, the one
  weight GEMM of ``FGGLinear`` and ``FGGConv2D``. ``V`` absorbs the Minkowski metric, so the
  ``-x0*V0 + <x_s, V_s>`` cancellation happens *inside* a single accumulation — the same
  product ``FGGLorentzMLR`` pins. It followed the default until it was measured on a trained
  network: see "Why the FGG product is pinned" below.

Following JAX's default
-----------------------
The remaining layer weight GEMMs — ``HTCLinear`` (including the attention Q/K/V projections),
the FHCNN and FHNN linears, ``HypLinearPoincare``, ``LorentzConv2D``'s Euclidean convolution,
``HypVQMLRPoincare``'s implicit-codebook matmul and ``hybrid_regularization``'s ``nnx.Linear``
— carry no ``precision`` keyword. These are sites where ``HIGHEST`` actually costs
throughput (it replaces one TF32 pass with a three-pass float32 emulation; measured +5.5 % on a
jitted hyperbolic attention forward), and whether to pay that is the user's call, not the
library's.

To run **everything** in full float32, set JAX's own knob — one line, and jit-cache aware, so
changing it re-traces::

    jax.config.update("jax_default_matmul_precision", "highest")

or scoped to a block::

    with jax.default_matmul_precision("highest"):
        logits = model(x)

A deep, fully hyperbolic stack of these layers is a good reason to do so: the pinned sites keep
the geometry honest, but a TF32 error introduced in an early layer's weight GEMM is carried
forward by every layer after it. The size of that error on one layer was measured on the FGG
product before it was pinned (A100, jax 0.9.1, ambient dim 33, batch 64,
``c in {0.1, 1.0}`` x spatial radius ``in {0.5, 5, 20}``, float32 against a float64 reference):
``FGGLinear`` 2.6e-4 under TF32 against 6.8e-8 under ``HIGHEST`` — roughly ~1e-4 against
~1e-7 on a layer, three of float32's seven significant digits. The ``FGGLinear`` arms of the
depth comparisons in the next paragraph predate the pin in the same way: their "default" is the
FGG product at TF32, which the library no longer runs, and they stay as the comparison for
``HTCLinear``.

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
maxima over the grid, not means, and n = 1 per cell. What that grid does record is a radius climb
(follow-up probes and their re-derivation in
``logs/2026-09-06_numerics_review_followup/depth_reversal/`` and its ``audit/``): ``HTCLinear``
applies its kernel to the full ambient point ``(x_0, x_s)`` with zero bias at init, so
``||z||^2 ~ x_0^2 + ||x_s||^2 = (1 + 2 sinh(rho_in)^2)/c`` and the curvature-scaled radius
``rho = sqrt(c) * r`` maps forward as ``rho_out = asinh(sqrt(1 + 2 sinh(rho_in)^2))``, an increment
of exactly ``log(sqrt(2)) = 0.347`` at large ``rho`` — though from a small input radius the first
layer jumps to ~0.9 rather than adding 0.35 — and against the recorded per-layer radii that closed
form holds within +-0.10 nats (54 % of layers within +-0.04; the residuals are layer-index-dependent,
i.e. the gain of that one weight draw); excluding the input->layer-1 step, HTC's mean per-layer
increment is 0.36-0.38 nats at depths 8 and 16 and 0.37-0.45 at depth 2 (individual layers
0.29-0.45), against -0.011 … +0.002 for ``FGGLinear`` (individual -0.041 … +0.022). At the one pair
where it can be checked ``c`` enters only through ``sqrt(c) * r`` — depth-16 HTC at ``rho = 6.67``
(``c = 0.1``) and 6.77 (``c = 1``) give 1.639e-3 and 1.643e-3, 0.24 % apart — but that is n = 1, the
matching depth-8 and depth-2 pairs are 10 % and 12 % apart, and the cells do not collapse onto one
curve (sorted by ``rho`` the HTC cells invert twice, by 9.0 % and 5.9 %, and ``FGGLinear`` at depth
16, ``rho = 4.88`` is 2.2x worse than HTC at depth 2, ``rho = 5.77``). That the climb is *what makes*
the TF32 error grow is a **hypothesis**, with evidence in one direction only: rescaling the HTC kernel
by ``1/sqrt(2)`` at ``c = 1``, radius 5, depth 16 holds the radius flat (5.00 -> 4.55) and cuts the
parameter-gradient error from 3.23e-3 to 8.31e-4 (3.9x, disjoint ranges, Welch t = 4.6), but the
opposite rescaling by ``sqrt(2)`` runs the radius up to 15.6 without raising the error (3.39e-3,
t = 0.20), and in that same emulation HTC and FGG are indistinguishable at the cell where the GPU grid
puts them 7.75x apart (3.23e-3 against 3.11e-3, t = 0.23) — CPU TF32 emulation, 8 rounding
realizations, one weight draw, and a different HTC draw from the grid's. At depth 2 the HTC/FGG
parameter-gradient ratios are 0.87/0.93/0.97/0.97, which is not an ordering: the single-GEMM TF32
floor with two perturbed operands is ``sqrt(2)*u/sqrt(3) ~ 4.0e-4`` for TF32's unit roundoff
``u = 2^-11`` (3.97e-4 measured on these operands), the ``r = 0.5`` cells sit at that floor
(2.9-3.4e-4) while the ``c = 1``, ``r = 5`` cells are 2.3x above it (9.2e-4, 9.5e-4), and on cell
means the emulation flips the depth-2 sign in exactly one of the four. Single HTC cells are not tight
either — the emulation's realization sd is 6-61 % of the cell mean for HTC against 2-11 % for FGG, and
on its weight draw HTC depth 8 is worse than depth 16 in one cell (1.15e-2 against 3.23e-3). Finally,
the grid's error is one relative L2 over the whole flattened gradient tree, and at depth 16, ``c = 1``,
radius 5 the head kernel carries 86 % of HTC's squared error and 87 % of its squared reference norm
(``FGGLinear`` 49 % / 34 %); the per-leaf errors are individually elevated (HTC layers 5-15 ~1.7e-2
against FGG's ~2.5e-3), so this is not a layer-counting artefact, but the reported number is in
practice the head-kernel gradient's relative error, on a gradient whose norm the radius inflates
(60.8 against 3.53). For a deep or gradient-sensitive stack of these layers, set the global knob
(the snippet above).

Why the FGG product is pinned
-----------------------------
Until this pin ``FGGLinear`` and ``FGGConv2D`` followed the default as well, although their
``x @ V`` carries the cancellation described in the first list. The depth grid above was the
reason: at depth 16 an FGG stack's float32 gradient error under TF32 is smaller than an
``HTCLinear`` stack's, which has no cancellation at all. That grid measures gradients at one
initialisation per cell. A float64 replay of a trained network overrules that reading.

The network is a Procgen PPO agent (BigFish) with an IMPALA-style encoder of 15 ``FGGConv2D``
and one ``FGGLinear``, ``c = 0.1``, trained on an A100 under the JAX default. 72 checkpoints,
2 048 states each, were replayed in float32 at the JAX default, in float32 with
``jax_default_matmul_precision="highest"`` and in float64 (audit scripts:
``logs/2026-10-07_rl-f64-replay-audit/``). At the JAX default the greedy action changes 120
times over the 72 checkpoints. In the runs trained without a cap on the feature radius, the
relative gap of the head input to float64 is 6e-4 to 7.5e-3 (maxima over the replayed states)
and the actor logits are up to 0.14 off (the largest of the 6 144 states of one checkpoint age,
three seeds; the median is 2.8e-4 to 1.0e-3). With ``"highest"`` no greedy action changes, the
head input is within about 5e-6 and the actor logits within 1.03e-4, and the forward gaps of
those runs drop by a factor of 740 to
2 570. The 16 FGG products are the only products in that network that the setting changes, and
they are the products this pin sets to ``HIGHEST``. The relative error enters at the first conv
stage (4e-4) and stays near 1e-3: in this network it does not grow with depth. Not measured: any
effect on the return.

Why the geometry is pinned rather than left to the same knob
------------------------------------------------------------
On Ampere/Hopper, XLA:GPU defaults float32 matmuls to TF32, whose 10-bit mantissa carries
~1e-3 relative error — far coarser than float32's ~1e-7. Hyperbolic geometry spends that
budget on cancellations whose entire design assumes float32 accuracy:

- the Lorentz inner product ``<x,y>_L = -x_0 y_0 + <x_s, y_s>``, formed as a difference of two
  dots (hyperboloid ``dist``/``expmap``/``ptransp``, the attention scores, the Busemann
  coordinate, the HoroPCA ideal-span projections) or inside one accumulation, with the metric
  absorbed into the weight (the FGG product and the ``FGGLorentzMLR`` logits);
- the ``lorentz_midpoint`` normalizer, a cancellation-free identity that exists precisely to
  keep the float32 error at ``O(eps)`` at any radius;
- the Poincaré/stereographic boundary factors ``1 - c*||x||^2`` and
  ``1 - 2c<x,y> + c^2||x||^2||y||^2``, which cancel to ``O(eps)`` for a near-boundary point;
- the MLR ``asinh`` arguments on both models, a difference of a radial and an angular term.

Measured on an A100 (jax 0.9.1): the f32-vs-f64 relative error of ``lorentz_midpoint`` is
4.6e-5 … 2.6e-4 at the TF32 default and 2.6e-8 … 1.6e-7 with ``HIGHEST`` — a ~2000x accuracy
loss that ``HIGHEST`` removes. The eager float32 gradient of the hyperboloid MLR improves from
1.5e-4 to 2.7e-7 relative over the same switch. These are not throughput-critical sites, so
the library pays for the accuracy rather than exposing the choice. The FGG product is different
in kind, the weight GEMM of every FGG layer, and is pinned on the measurement in the section
above. Its cost on one H100: 2 % on the forward pass and 9 % on forward plus backward for an
IMPALA-ResNet of FGG layers at batch 1,024, and 5 to 16 % on forward plus backward for a single layer.

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
einsums, and the FGG layers, whose ``x @ V`` is a Minkowski inner product.
"""
