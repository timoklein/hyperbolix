"""Matmul precision for the float32 dot products that touch manifold data.

The library pins one setting, :data:`MATMUL_PRECISION`, on every float32 dot-product
primitive (``jnp.dot`` / ``jnp.einsum`` / ``jnp.matmul`` / ``lax.conv_general_dilated*`` /
``nnx.Linear``) that consumes or produces points, tangent vectors or hyperplane normals.
It lives in ``utils/`` — a neutral home both ``manifolds/`` and ``nn_layers/`` can import
from, since ``manifolds/`` must not depend on ``nn_layers/``. It was originally defined in
``hyperbolix.nn_layers.hyperboloid_core``, which now re-exports it, so
``hyperboloid_core.MATMUL_PRECISION`` keeps working.

Why
---
On Ampere/Hopper, XLA:GPU defaults float32 matmuls to TF32, whose 10-bit mantissa carries
~1e-3 relative error — far coarser than float32's ~1e-7. Hyperbolic geometry spends that
budget on cancellations whose entire design assumes float32 accuracy:

- the Lorentz inner product ``<x,y>_L = -x_0 y_0 + <x_s, y_s>``, formed as a difference of
  two dots (hyperboloid ``dist``/``expmap``/``ptransp``, the attention scores, the Busemann
  coordinate, the HoroPCA ideal-span projections) or, with the Minkowski metric absorbed
  into the weight matrix, *inside a single accumulation* (``FGGLinear``,
  ``FGGLorentzMLR``, ``HypRegressionHyperboloid``);
- the ``lorentz_midpoint`` normalizer, a cancellation-free identity that exists precisely to
  keep the float32 error at ``O(eps)`` at any radius;
- the Poincaré/stereographic boundary factors ``1 - c*||x||^2`` and
  ``1 - 2c<x,y> + c^2||x||^2||y||^2``, which cancel to ``O(eps)`` for a near-boundary point;
- the MLR ``asinh`` arguments on both models, a difference of a radial and an angular term.

Measured on an A100 (jax 0.9.1): the f32-vs-f64 relative error of ``lorentz_midpoint`` is
4.6e-5 … 2.6e-4 at the TF32 default and 2.6e-8 … 1.6e-7 with ``HIGHEST`` — a ~2000x accuracy
loss that ``HIGHEST`` removes. The eager float32 gradient of the hyperboloid MLR improves
from 1.5e-4 to 2.7e-7 relative over the same switch.

``HIGHEST`` is a no-op on CPU (no TF32 path) and for float64 anywhere, so this costs nothing
outside float32 GPU matmuls; on tensor-core GPUs it trades GEMM throughput for accuracy
(``HIGHEST`` replaces one TF32 pass with a three-pass float32 emulation — measured +5.5 % on
a jitted hyperbolic attention forward). The setting is deliberately *not* a per-call keyword
on the manifold or layer signatures: accuracy here is a property of the geometry, not a knob.
"""

import jax

MATMUL_PRECISION = jax.lax.Precision.HIGHEST
