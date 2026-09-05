r"""Matmul precision for the float32 dot products that touch manifold data.

:data:`MATMUL_PRECISION` (``jax.lax.Precision.HIGHEST``, not user-configurable) is pinned on
the sites where the *geometry* is computed, and nowhere else. Everything else — the layer
weight GEMMs — passes no ``precision`` keyword at all and follows JAX's own
``jax_default_matmul_precision``, which on Ampere/Hopper means TF32.

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
- the MLR heads, which are decision quantities rather than hidden activations:
  ``HypRegressionHyperboloid``, ``Hyperboloid._compute_mlr``, ``Poincare._compute_mlr_pp`` and
  ``ProperVelocity._compute_mlr``.

Following JAX's default
-----------------------
The layer weight GEMMs — ``HTCLinear`` (including the attention Q/K/V projections),
``FGGLinear``/``FGGConv2D``, the FHCNN and FHNN linears, ``HypLinearHyperboloidPLFC``, the
Poincaré and proper-velocity linear and convolution layers, ``HypVQMLRPoincare``'s
implicit-codebook matmul and ``hybrid_regularization``'s ``nnx.Linear`` — carry no
``precision`` keyword. These are the
sites where ``HIGHEST`` actually costs throughput (it replaces one TF32 pass with a
three-pass float32 emulation; measured +5.5 % on a jitted hyperbolic attention forward), and
whether to pay that is the user's call, not the library's.

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
layer weight GEMMs pass no ``precision`` keyword and follow
``jax_default_matmul_precision`` instead.
"""
