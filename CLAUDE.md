# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --locked --dev

# Full suite on all cores: ~9 min on 48 CPU workers, hours single-process (the suite is JAX-compile-heavy)
uv run pytest -n auto

# One file, and its dim-2 float32 slice while iterating
# (the ids spell the dimension as a bare number, e.g. [PoincareBall-c1-2-float32-10])
uv run pytest tests/test_manifolds.py -x -v
uv run pytest tests/test_manifolds.py -k "2-float32"

uv run ruff check hyperbolix tests
uv run ruff format hyperbolix tests
uv run pyright hyperbolix
uv run pre-commit run --all-files
uv run mkdocs build --strict
```

## Verification

- Run the test files that cover the change (manifolds → `test_manifolds.py`, optimizers → `test_optimizers.py`, FGG layers → `nn_layers/test_hyperboloid_fgg.py`). More than one or two files get `-n auto`; never run the full suite single-process.
- When someone else's job holds the GPU, set `JAX_PLATFORMS=cpu`. Nothing else is needed on a GPU box: `tests/conftest.py` already stops the xdist workers from preallocating.

## Architecture

Pure JAX on Flax NNX.

- **Manifold ops take single points**, `(dim,) -> scalar` or `(dim,) -> (dim,)`; batch with an explicit `jax.vmap`. NN layers batch internally in `__call__`.
- **Manifolds are plain Python classes** (not `nnx.Module`), instantiated with a dtype (`Poincare(dtype=jnp.float64)`). All but `ProductManifold` follow the scalar-`c` `Manifold` protocol in `manifolds/protocol.py`; `ProductManifold` intentionally doesn't (a per-factor `cs` sequence, no `c` attribute).
- **Curvature `c` is passed at call time**, never stored on a layer. Trainable curvature is `LearnableCurvature` (`utils/curvature.py`): it lives on the user's `nnx.Module` and returns the (optionally clamped) `c`.
- **`version_idx`** selects an op variant. Keep it static under JIT (`functools.partial` or `static_argnums`): a traced index compiles every variant into one `lax.switch`.
- **Layers take `manifold_module`** (a manifold instance), never raw functions, and name their parameters `kernel`/`bias`.
- **`ManifoldParam`** (an `nnx.Param` subclass) tags manifold-valued parameters. The Riemannian optimizers in `optim/` detect it and give every other parameter a Euclidean update.
- New hyperboloid layers reuse the shared ops in `nn_layers/hyperboloid_core.py` (`hrc`, `htc`, `lorentz_midpoint`, …).

### Weight initialization

Standard inits (He, Xavier) are too large for hyperbolic layers, and a too-small init freezes a stack without any NaN (the per-layer gain compounds below float32 resolution). Several defaults therefore deviate from their paper's init on purpose, to stay norm-preserving, and each keeps a switch that restores the reference bit-for-bit. Don't revert them to the reference:

- `FGGLinear`/`FGGConv2D`: `reset_params="fan_out"`, `init_bias=0.0`. The Klis et al. 2026 reference (BatchNorm regime) is `reset_params="eye"` (linear) / `"lorentz_kaiming"` (conv) with `init_bias=0.5`.
- `HTCLinear`: fan-in uniform `U(-√(3/in), √(3/in))`. `init_bound=0.02` restores the old init, which froze depth-≥2 stacks.
- `HypConv2DHyperboloidILNN`: fan-out normal (`kernel_init_std=None`). `kernel_init_std=0.02` restores Shi et al. 2026, which was tuned against the LogCat digamma sign bug and collapses to the origin now that the sign is fixed.
- `HypLinearHyperboloidPLFC` keeps the reference `std=0.02`: it has no LogCat.

Full table: `docs/user-guide/nn-layers.md#initialization-scales`.

## Numerics

### Rules

- **Pairwise ops are cancellation-free**, down to the stored point's own rounding or the chart's ceiling. Build new ones from the existing forms, not the textbook ones:
  - the Poincaré `asinh` distance `2·asinh(√c‖x−y‖/√(B_x·B_y))/√c`, `B_x = 1 − c‖x‖²`;
  - the gyration regrouped in `s = x + y`;
  - the hyperboloid primitives (`dist`, `logmap`, `ptransp`, `gyro_difference`, `busemann`), never the literal `minkowski_inner`;
  - `floor_at` on divisors, `safe_sqrt` at zero.

  A layer with tangent input and output computes in closed form from the tangent vector, not through a round trip via a stored ball point.
- **Loud divergence over silent saturation, without extra checks.** Don't map an already non-finite input (an `inf` time coordinate, a point past the float32 manifold) onto a finite, plausible output: a NaN loss is the intended signal. Remove guards whose only job is finiteness on already non-finite input; the MLR `asinh` clamp went for this reason. Loud means *don't hide*, not *detect everything*:
  - add no `isfinite`/`where` conditions just to make failures loud, except in critical training code (the PV layers, the hyperboloid);
  - never add them in the transcendentals: the `math_utils.sinh`/`cosh` clips have too many downstream uses;
  - arithmetic that overflows on its own stays, and is documented.

  Guards that fix real float32 rounding on finite inputs (`floor_at`, `safe_sqrt`) stay.
- **No custom JVPs unless really necessary, and then only on critical training paths** (the hyperboloid, PV). The same holds for benchmark campaigns and ulp-level audits. A model included for completeness (e.g. HalfSpace) gets the cancellation-free formulas that come cheap, plain autodiff, the usual tests and a short docs note.
- **Derivative tests need independent oracles.** A spelling that normalizes a direction or switches branches (`safe_normalize`, `where(r > 0, …)`) can be value-correct yet have a zero or wrong-sign derivative at the origin, at coincident points or at `c = 0`. Float32-vs-float64 tests don't catch it. Test such a site against central finite differences at those inputs. They are reachable: zero-init residual branches, padding, `LearnableCurvature(init_c=0.0, parameterization="identity")`.
- **Geometry dots stay `precision=MATMUL_PRECISION`** (`HIGHEST`): MLR logits, Lorentz midpoints, attention scores, point-to-hyperplane kernels. Their results cancel, and TF32 on Ampere/Hopper costs ~2000× accuracy there. Weight GEMMs on tangent vectors follow the JAX default.

### Speed against accuracy

- Trade-offs between speed and precision or stability are the maintainer's call, made on measured timings; kernel counts are only a proxy. Put both costs in one table, with a recommendation.
- A fix that reaches the training regime and costs ≤ ~20 % at the layer level is worth it; one at ~2× is not. A fix for a regime training never reaches becomes a docs paragraph instead.
- Per-PR timing gates (new/old ≤ 1.06 on CPU, ≤ 1.10 on an A100) are soft. A row a few percent over the gate with an accuracy gain is kept and reported, not reworked to pass the number.
- A100 A/A runs have reached 1.11 at B = 4096. Interleave old and new in alternating processes, and read the ratio against an A/A run.
- Keep code that measured no slower, and don't convert a site to a "faster" spelling without a measurement. Never unpin a geometry dot to pass a gate.

### Where each model stops

Pairwise ops hold to the scaled radius `a = √c·d` below (float32 / float64, `c = 1`). Past it the chart cannot represent the point, which is not a bug. Mechanisms and numbers: `docs/user-guide/numerical-stability.md`.

| Model | Good to `a` | Limit |
|---|---|---|
| Hyperboloid | 16.6 / much further | the storage floor `eps·sinh(a)/√c` |
| Poincaré | 12.6 / 27.7 (13.8 / 28.9 at `c = 0.1`) | the ball chart's ceiling. `dist` slot 1 (`VERSION_MOBIUS`) saturates there: 12.637 for a true 14.4, gradient relative error 1.0 |
| Klein | 6.32 / 13.86 | half the Poincaré radius, and the floor `eps·cosh²(a)`. The maps into Klein don't project, so call `Klein.proj` after mapping far points in |
| HalfSpace | `inf`/NaN past 88.7 / 709.8 | the floor `≈ 0.4·(eps/2)·cosh(√c·δ)/√c`, where `δ` is the distance to the vertical axis through `e_n/√c` |

Two things still cancel at large radius: `HyperbolicFullAttention`'s GEMM-formed scores, and a `ptransp` step below the storage floor.

## Tests

- Fixtures (`tests/conftest.py`): `seed_jax` (enables float64), `rng`, `dtype`, `tolerance`, `manifold_and_c`, `uniform_points`. Tests cover both dtypes, with `atol=4e-3` (f32) and `1e-7` (f64).
- A numerics regression test compares float32 with float64 at the input that failed, and fails on the code before the fix.
- CI runs only the test files listed in the `test-suite` matrix of `.github/workflows/ci.yaml`: add each new test file to an entry.

## Conventions

- Shape suffixes on tensor names, one capital letter per dimension (`logits_BLV`, `hidden_BD`), with a dimension key at the top of each file.
- Flax NNX (v0.12+): `nnx.Optimizer(..., wrt=nnx.Param)`; `optimizer.update(model, grads)`; lists of modules are `nnx.List([...])`.
- Pre-commit runs ruff. It reformats on commit (`x ** 2` → `x**2`), so match the reformatted text in later edits, and it strips unused imports, so add an import together with its first use.
