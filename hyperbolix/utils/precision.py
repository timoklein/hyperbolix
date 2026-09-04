"""Matmul precision for the float32 dot products that touch manifold data.

Two settings live here, because the two kinds of float32 matmul in this library have
different accuracy needs.

:data:`MATMUL_PRECISION` (``jax.lax.Precision.HIGHEST``, not user-configurable) is pinned on
the *geometry* dots: the vector-vector reductions inside the manifolds
(``jnp.dot(x, y)`` behind ``dist``/``addition``/``expmap``/``logmap``/``ptransp`` and the
boundary factors) and the ``decomposition/`` (HoroPCA) contractions. These are reductions,
not tensor-core GEMMs — XLA lowers a vector-vector dot to a reduction — so ``HIGHEST`` costs
essentially nothing there, and the accuracy it buys is large.

:func:`gemm_precision` supplies the precision for the *training-path* GEMMs: the layer
matmuls, einsums, convolutions and ``nnx.Linear``\\ s in ``nn_layers/``, plus the three
batched MLR einsums in ``manifolds/``. It returns :data:`GEMM_PRECISION`, which defaults to
``None`` — meaning JAX's own precision setting governs, so on Ampere/Hopper a float32 GEMM
runs in TF32 unless the user says otherwise. That is a deliberate default: these are the
sites where ``HIGHEST`` actually costs throughput (it replaces one TF32 pass with a
three-pass float32 emulation; measured +5.5 % on a jitted hyperbolic attention forward), and
whether that trade is worth making is the user's call, not the library's.

This module lives in ``utils/`` — a neutral home both ``manifolds/`` and ``nn_layers/`` can
import from, since ``manifolds/`` must not depend on ``nn_layers/``. ``MATMUL_PRECISION`` was
originally defined in ``hyperbolix.nn_layers.hyperboloid_core``, which now re-exports both
names, so ``hyperboloid_core.MATMUL_PRECISION`` keeps working.

Choosing the GEMM precision
---------------------------
Either knob works; pick one.

- JAX-wide (recommended, and jit-cache aware — changing it re-traces)::

      jax.config.update("jax_default_matmul_precision", "highest")
      # or, scoped:
      with jax.default_matmul_precision("highest"):
          ...

- hyperbolix-wide, which restores 1.2.0's forced-``HIGHEST`` GEMMs exactly::

      import jax
      import hyperbolix.utils.precision

      hyperbolix.utils.precision.GEMM_PRECISION = jax.lax.Precision.HIGHEST

  :func:`gemm_precision` reads the module attribute at call time, so rebinding it on *this*
  module is enough — no need to patch every importing module. **Set it before you construct
  the model and before the first trace**: ``nnx.Linear`` captures its ``precision`` at
  construction, and every other site captures it when the enclosing function is traced.

Why the geometry dots stay HIGHEST
----------------------------------
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
from 1.5e-4 to 2.7e-7 relative over the same switch. That is the scale of the effect the
default-``None`` GEMMs are exposed to as well, which is why the override exists.

``HIGHEST`` is a no-op on CPU (no TF32 path) and for float64 anywhere, so neither setting
costs anything outside float32 GPU matmuls.
"""

import jax

MATMUL_PRECISION = jax.lax.Precision.HIGHEST
"""Precision for the manifold vector dots and the ``decomposition/`` contractions.

Fixed at ``HIGHEST``: accuracy here is a property of the geometry, not a knob, and these
sites lower to reductions rather than tensor-core GEMMs, so the setting is close to free.
"""

GEMM_PRECISION: jax.lax.Precision | None = None
"""Precision for the training-path GEMMs, or ``None`` to follow JAX's own setting.

``None`` (the default) means a float32 GEMM uses whatever
``jax_default_matmul_precision`` / :func:`jax.default_matmul_precision` says — TF32 on
Ampere+ unless the user changed it. Rebind this attribute to
``jax.lax.Precision.HIGHEST`` *before model construction and before the first trace* to
restore hyperbolix 1.2.0's forced-``HIGHEST`` GEMMs.
"""


def gemm_precision() -> jax.lax.Precision | None:
    """Return the precision to attach to a training-path GEMM.

    Reads :data:`GEMM_PRECISION` at call time, so rebinding that attribute on this module
    takes effect for every call site without patching the importing modules. The value is
    captured when the enclosing function is traced (and at construction for
    ``nnx.Linear``), so set it before building the model.

    Returns:
        The configured ``jax.lax.Precision``, or ``None`` to defer to JAX's own default
        matmul precision.
    """
    return GEMM_PRECISION
