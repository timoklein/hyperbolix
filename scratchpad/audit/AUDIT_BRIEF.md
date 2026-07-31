# Audit brief (reconstructed after /tmp wipe) — remaining Stage-2 batches

## ORCHESTRATION STATE (updated 2026-07-31, pre-compaction checkpoint)

Committed on test-overhaul: ac62f2d (F0 ci/config), 5a3f82d (F2 conftest+manifold core),
1a3db93 (F1 LayerSpec suite), 5d6fba2 (F4 distributions/optim), 774262f (F3 stereo/precision/
math). Suite: 3,161 items (baseline 9,404), all green, ruff clean.
RUNNING: F5a (attention/fgg/residual oracles) + F5b (poincare/gyro/vq/activations oracles),
both Opus, editing disjoint tests/nn_layers files, will notify on completion.
QUEUE after F5a/F5b land (orchestrator reviews diff + commits per batch, max 2 agents,
Opus for fixes / Sonnet for mechanical verification):
  1. F5c (per-layer residual oracles — section F5c below)
  2. F6-M1 + F6-M2 (gap tests + lib fixes #1-#3 — sections below)
  3. F6-M3
  4. F7 sweep (ruff/pyright/naming) + FINAL GATES: full suite green (run per-file or per-dir,
     NEVER one big multi-file process — OOM at 30GB), item-count accounting vs 9,404,
     coverage diff (baseline from branch point fb14e0d in a worktree vs final; >0.5% drop on
     any hyperbolix/ file blocks), CI matrix dry-run, mutation spot-check (~15 audit mutations
     re-run, expect all caught — the journal number), CLAUDE.md test-count update (~1,660 → new).
Deferred/open: lib bugs #4-#6 → GitHub issues at the end; Stage-0 task cleanup (coverage
baseline was never recorded — compute from fb14e0d worktree during final gates).
User prefs this session: max 2 concurrent test-running agents (laptop, 16 threads/30GB —
12 crashed it, saved to memory); ONE file per pytest invocation (multi-file processes OOM);
Opus/Sonnet by difficulty; orchestrator (Fable) reviews + commits, agents never commit.

Source: the full consolidated audit (12 agents + independent mutation verification, 19/19
delete verdicts CONFIRMED). Original per-agent reports were lost in a reboot; this brief
preserves everything the remaining fix batches need. Repo: /home/timo/1Projects/hyperbolix,
branch test-overhaul. Baseline 9,404 collected items.

Already landed: F0 (ci/config, ac62f2d), F2 (conftest+manifold core, 5a3f82d),
F1 (LayerSpec contract suite, 1a3db93), F3+F4 (working-tree, being committed now).

## Live library bugs (approved: fix #1-#3 on this branch, in batch F6; #4-#6 deferred to issues)

1. `hyperbolix/utils/helpers.py:171-185` compute_hyperbolic_delta ≡ 0 for every input:
   the max-min product indexes both operands on row i (`gromov_prod_mat[:, None, :]` and
   `gromov_prod_mat[:, :, None]`), so max_k min(G[i,k],G[i,j]) ≡ G[i,j]. Fix: second operand
   must be indexed [k,j] (use `gromov_prod_mat.T[None,:,:]`). Oracle for the new test:
   star metric (delta 0 — non-discriminating alone), path metric, 4-cycle C4 metric
   (delta avg 0.25 / max 2.0), plus scaling sentinel delta(2*D) == 2*delta(D), plus an
   O(n^4) brute-force cross-check on a random 4-point metric.
2. `wrapped_normal_{poincare,hyperboloid}.py` log_prob step 2 (mu.ndim>1 branch): crashes
   (ValueError/TypeError) on sample's own output when mu is batched AND sample_shape
   non-empty — sample axis S and batch axis B get vmapped together. Fix: mirror step 1's
   n_sample_dims logic in step 2.
3. `wrapped_normal_poincare.py:218`: no floor on r → log_prob gradient NaN in float32 at
   z=mu (f64 fine). Hyperboloid sibling floors r at 1e-15 (line ~211). Also
   `_wrapped_normal_base.py:47` `_log_det_jacobian_from_r` single jnp.where → NaN grad at
   r=0 even in f64. Fix: floor r like hyperboloid + double-where idiom (r_safe = where(small,
   1.0, r)). No custom_jvp anywhere in the library.
4. (deferred) smooth_clamp can exceed max_value when window < ln2/beta (~0.0139 at beta=50).
5. (deferred) Euclidean.is_in_manifold returns True unconditionally incl. NaN (test pins
   current behavior with TODO, landed in F2).
6. (deferred) f32 Taylor-gate dtype comparison in _log_det_jacobian_from_r (bounded impact).

## Uncaught mutations still open (each needs an oracle; scope = what passed under mutation)

- FHNN spatial output sign flip (hyperboloid_linear.py:159) — both FHNN files blind. → F5c
- FHCNN sign flip (hyperboloid_linear.py:84) — test_hyperboloid_conv.py blind. → F5c
- PLFC/ILNN forward collapsed to origin (v_BO*0.0) — 54/56 blind. → F5c
- lorentz_residual/positional y-term dropped (layer = identity) — 113 items blind. → F5a
- build_spacelike_V curvature dropped from bias transport (Klis Eq. 12) — FGG file blind. → F5a
- FGGConv2D/FGGLinear forward → constant origin — FGGConv 100% blind. → F5a
- Attention q/k/v roles swapped — all 61 attention items blind incl. overfit tests. → F5a
- VQ argmin→argmax (farthest code) — 85/85 blind. → F5b (A8-02 fix: recompute dists_NK
  independently, assert argmin equality)
- VQ STE deleted — 149/149 blind. → F5b (A8-01: jax.grad(sum(quantized))(x) == ones)
- HypConv2DPoincare ×0.7 — 48/48 blind. → F5b (A8-03: 1x1-conv-equals-HNN++-FC oracle)
- PoincareBatchNorm2D no-op — 32/34 blind. → F5b (A8-04/06 + Fréchet-variance oracle)
- frechet_variance ≡ 0 — 211/218 blind. → F5b (two symmetric points at geodesic distance d
  → var == d²; also M3-G12)
- GyroBatchNorm centering dropped — 180/184 blind. → F5b (A8-07: output batch Fréchet mean at
  bias point, Fréchet std ≈ γ; RMSNorm sibling already has a real closed-form oracle to copy)
- hyp_leaky_relu / hyp_swish → identity — zero coverage. → F5b (A8-09: per-activation
  semantics test; relu/tanh already have real oracles)
- Poincaré MLR logit sign flip — 47 items blind. → F5c (A9-06: zero-on-hyperplane,
  sign-follows-side, monotone margin; port from Poincaré compute_mlr_pp oracles)
- PV outer sinh deleted (Chen 2026 Eq. 22) — test_pv_conv 47/47 blind. → F5c (A9-07)
- Busemann logit sign flip (busemann_core.py:130) — 151 items blind. → F5c + M3-G16
- HypLinearPoincare Möbius matvec zeroed (constant layer) — 67 items blind. → F5c (A9 generic
  constant-collapse guard: two different inputs → different outputs, ~3 lines, apply per layer)
- Poincaré/Hyperboloid dist: no absolute closed-form anchor (only relative cross-checks) —
  partially closed by F2's c=1 NumPy oracles in test_class_based_manifolds.py.

## F5a — files: tests/nn_layers/test_hyperboloid_{attention,residual,pool,positional,fgg}.py, test_hypformer.py

Deletes (mutation-confirmed): A7.1 the five hrc_equals_hyp_* tests in test_hypformer.py:112-161
(hyp_relu IS hrc_relu(x,c,c) — same function); A7.2 test_pool_matches_hrc (pool:139); A7.3
test_lorentz_scale_matches_spatial_to_hyperboloid (residual:96); A7.4
test_spatial_to_hyperboloid_equivalence_with_hrc (attention:51); A7.7 two FGG shape-only tests
(fgg:126,230); A7.14 two hrc norm shape tests (hypformer:582,689).
Strengthens: A7.5 residual:189 test_residual_fixed_matches_functional → replace shared-path
assert with monotone-distance-toward-y check; A7.6 fold 12 jit tests into eager siblings with
allclose(jitted,eager) (attention 186,261,482; residual:150; hypformer 297,314,332,617,722;
fgg 211,262,390,769 — pool:217 is the model, already real).
New oracles (close the 4 uncaught mutations above): residual/positional must-depend-on-y;
attention vs naive einsum reference at f64 small seq (also distinguishes q/k/v roles);
FGG forward numeric oracle (non-constant + value check vs hand-computed small case);
build_spacelike_V curvature-dependence check (outputs at c=0.5 vs c=2 differ as Eq. 12 says).
B1 trims: A7.9/A7.10 inert learnable_scale axis when scale=False (18 items); A7.11/12 pool
axes (11); A7.13 positional rank-3-vs-batch duplicate pairs (12); A7.15 keep A8's activation
oracles, drop hypformer duplicates. Overfit tests: keep but add the einsum-reference check
(they proved non-discriminating alone).
Do-not-touch: fgg:732 mean-only-BN variance test (V6 lead REJECTED — it's tight);
test_hrc_dropout_training_vs_eval (flake window negligible).

## F5b — files: tests/nn_layers/test_poincare_{activations,batchnorm,conv,linear_gradients,vq}.py, test_gyro_normalization.py, test_hyperboloid_activations.py

Deletes: A8-05 batchnorm:119 nonneg-variance (V7); A8-08 activations:169,186,201 shape matrix
(keep 1 multi-dim item); A8-10 activations:222 formula-test (V7); A8-17 gyro:277 rms shape
(subsumed by radius-normalization oracle).
Strengthens (see uncaught-mutations section for the oracles): A8-01, A8-02, A8-03, A8-04,
A8-06, A8-07, A8-09, A8-11 (leaky_relu negative-slope actual scaling), A8-12 (swish smooth-
at-zero: compare the two points or delete), A8-13/14/15 jit folds (activations 426-510,
gyro 205,358, batchnorm 214, conv 283), A8-23 (poincare_relu: parametrize c, cover
negative_slope), A8-24 (linear_gradients: ADD f32-manifold parametrization, +10 items — file
guards an f32 overflow bug but ran f64-only; do NOT touch existing seeds/assertions).
B1 trims: A8-16 gyro RMS dim axis {2,10} (36); A8-18/19 vq c-axis on codebook tests (12);
A8-20 beta_concat merge (8); A8-21 for-loop→parametrize; A8-22 hoist duplicated
_make_poincare_points/_make_tangent_input helpers to tests/nn_layers/conftest.py (create it).
Do-not-touch: test_poincare_linear_gradients.py existing seeds 0-9 tests;
test_poincare_midpoint_equidistant_two_points (real closed form); VQ batch-independence test
(valid metamorphic test); VQ EMA fixed-point + dead-code-revival tests (already pinned).

## F5c — files: tests/nn_layers/test_hyperboloid_linear_{fhnn,plfc}.py, test_hyperboloid_conv{,_fhnn,_ilnn}.py, test_regression_layers.py, test_pv_conv.py, test_busemann_layers.py, test_hybrid_regularization.py, test_nn_layers.py, test_pv_linear.py, test_pv_regression.py

(Residual per-layer work AFTER F1's consolidation; clones already deleted.)
- A6-01 conv_ilnn:271 (now moved/renamed?) test_forward_is_plfc_of_log_radius_concat →
  rename test_single_patch_uses_row_major_logcat_ordering; replace the
  manifold.log_radius_concat call with an independent digamma-based transcription.
- A6-02/03 numeric oracles: FHNN/FHCNN forward value check (hand-computed small case or
  independent NumPy transcription of W·x spatial + time reconstruction — must catch sign
  flip); PLFC/ILNN non-constant + value check (must catch origin-collapse).
- A6 B1: dtype axis off shape-only tests that remain (A6-15/16/19); A6-18/21 subsumption
  deletes if F1 left them.
- A9-02 hybrid_regularization:60 test_curvature_varies_output: drop the `if c != 1.0` guard
  → params {0.1,0.5,2.0} all assert not-allclose; drop c=1.0 param.
- A9-06 MLR decision-boundary oracles in test_regression_layers.py + test_busemann_layers.py
  (+ M3-G16 busemann sign pin): zero-on-hyperplane, sign-follows-side, monotone margin.
- A9-07 PV forward oracle in test_pv_conv.py (independent transcription of Chen Eq. 22 for a
  1x1 conv / single point).
- A9-10 *_different_curvatures: keep 3 curvatures, drop dtype axis, assert outputs differ
  across c. A9-14 hybrid shape merge if F1 left them. Constant-collapse guard (2 inputs →
  different outputs) for HypLinearPoincare in test_nn_layers.py (A9 UM-4).
Do-not-touch: test_nn_layers.py:164 PP gradient (only coverage); test_pv_regression.py:54,72;
hcat/log-radius oracle block in test_hyperboloid_conv.py:20-254.

## F6 — missing-test batches (all 42 specs approved; each new test must fail under its target mutation in a scratch worktree and pass at HEAD)

F6-M1 (manifolds/utils; files: tests/test_manifolds.py or new tests/test_manifold_oracles.py,
tests/test_manifold_curvature.py, tests/test_helpers.py, tests/test_math_utils.py,
tests/test_product_manifold.py, + LIB FIX #1):
- M1-01 Hyperboloid/PV compute_mlr value oracles (zero-on-hyperplane + r=0 closed form,
  ported from the Poincaré compute_mlr_pp tests; top-ranked gap). 45 LOC.
- M1-02 already partially landed in F2 (metric duality); add the Hyperboloid tangent-projected
  probe leg if missing + NumPy rgrad = g/λ² exact leg.
- M1-03 = lib fix #1 + delta oracles (see bugs above). 35 LOC + 5-line fix.
- M1-04 _gyrovector_core._proj boundary clamp branch: points outside ball project strictly
  inside with dtype-scaled margin. 25 LOC.
- M1-05 ProductManifold.dist_min non-metric counterexample (dist_min=0 for distinct points;
  triangle violation). 35 LOC.
- M1-06 LearnableCurvature straight-through multi-step re-entry vs plain-clamp control
  (200 SGD steps re-enter [c_min,c_max] under straight_through=True, ratchet under False). 40.
- M1-07 smooth_clamp_{min,max} gradient values = sigmoid(beta*(x-shift)); dropped-
  smoothing_factor mutation must fail. 30.
- M1-08 sinh_lift_to_hyperboloid: elementwise sinh(√c·s)/√c value + Lorentz constraint +
  saturation invariant. 25.
- M1-09 beta_concat vs scipy.special.beta ratio + dtype-leak guard. 25.
- M1-10 _conformal_factor_batch c≤0 branch + accessor: λ = 2/(1-c‖x‖²) at c∈{2,.5,0,-.5,-2},
  single-point vs batch. 20.
- M1-11 protocol conformance isinstance sweep. 8.
- M1-12 isometry gradients finite at degenerate inputs + one central-difference check. 30.
- M1-14 _lorentz_boost: B.T@J@B == J, B symmetric, inverse-boost without proj cover. 20.
F6-M2 (distributions/decomposition; files: tests/test_wrapped_normal.py,
tests/test_uniform_poincare.py, tests/test_frechet_mean.py, tests/test_horopca.py + LIB
FIXES #2,#3):
- M2-01 log_prob 4 oracles — density is w.r.t. RIEMANNIAN VOLUME (verified: polar quadrature
  with sinh^{n-1} area element ≈ 1.000004; jacfwd change-of-variables with λ(z)^n correction
  agrees 1.7e-16 Poincaré / 4.6e-15 hyperboloid Gram form; Poincaré↔hyperboloid log_prob
  agree 1.2e-14 with NO Jacobian correction): (a) 2D quadrature ≤1e-3; (b) CoV ≤1e-10;
  (c) c=1e-4 flat limit vs scipy MVN; (d) isometry cross-check ≤1e-10. ~110 LOC.
- M2-02 may already be landed by F4 (sample statistics σ-sensitivity) — verify, don't duplicate.
- M2-03 = lib fix #2 + batched-mu/sample_shape agreement test.
- M2-08 = lib fix #3 + grad-at-mean test (finite + ≈0 both models, f32+f64).
- M2-04/05 may already be landed by F4 (KS radial CDF n≥3, volume n=3 closed form) — verify.
- M2-06 transform_horopca: pairwise distances hyperboloid vs K-ball match. 18.
- M2-07 frechet_mean Euclidean exact mean + PV stationarity. 20.
- M2-09 _common.gaussian_log_prob/sigma_to_cov vs scipy + both ValueError branches. 22.
- M2-10 frechet iteration knobs change result; raises on reverse-mode grad. 30.
- M2-11 uniform sample non-origin center KS. 10. M2-12 orthonormalize_rows rank-deficient. 12.
- M2-13 HoroPCA validation raises ×4. 16. M2-14 direction sampler vs Beta((n-1)/2,(n-1)/2)
  marginal of (u1+1)/2 — moment oracle is provably vacuous, use the Beta marginal. 12.
- M2-15 rejection-sampler termination n≤50, R≤8. 12.
F6-M3 (optim + nn_layers helpers; files: tests/test_optimizers.py,
tests/nn_layers/test_poincare_vq.py, test_gyro_normalization.py, test_poincare_batchnorm.py,
test_hyperboloid_attention.py, new tests/nn_layers/test_helpers_validation.py):
- M3-G1 one RSGD step == expmap(-lr·egrad2rgrad(g,x,c),x,c) and != expmap(-lr·g,...); c→0
  optax agreement. (F4 may have landed the λ² exact-step — verify, extend to the != leg.)
- M3-G2 pure-Euclidean model: riemannian_adam/sgd bit-match optax.adam/sgd. 35.
- M3-G3/G4/G7 lr schedule varies step; callable curvature re-evaluated per update; scalar-
  param + leaf-count-mismatch guards raise. ~60.
- M3-G5/G6 transported momentum Minkowski-tangent on Hyperboloid; Adam m2 is scalar broadcast
  of ⟨g,g⟩_x not elementwise g². 50.
- M3-G8 use_expmap=False retraction: ≠ expmap at large lr, first-order agree at small lr. 20.
- M3-G9/G10/G11/G12/G16 — if F5b/F5c landed these oracles, verify and skip; else implement
  per uncaught-mutations section.
- M3-G13 linear-attention scan: fix QUADRATIC_ATTN_CLASSES exclusion so the existing einsum
  oracle covers the linear class; f64 long-seq drift bound. 15.
- M3-G14 as_pair tuple branch: non-square kernel (3,2)/stride (2,1) shape correctness; int vs
  tuple bitwise identical. 30.
- M3-G15 validate_{hyperboloid,poincare}_manifold TypeError naming right model. 12.
- M3-G17 busemann_fc_poincare_output == hyperboloid_to_poincare(sinh_lift(...)); Thm 4.1 vs
  4.2 consistency. 40.

## Do-not-touch (global)

test_dtype_respected_x64.py (beyond the one landed hardening edit); poincare_linear_gradients
seeds 0-9; conftest sampled-curvature scheme + its comment; TestSingularPointGradients
(test_precision.py); test_nnx_optimizer_boundary_point_stays_on_manifold; test_cosne.py
(house-style model); hcat/log-radius block (hyperboloid_conv.py:20-254); fgg:732;
test_busemann_along_ideal_ray assertions; test_apollonian_matches_boundary_supremum dims.

## Rules for every fix agent

Work directly on branch test-overhaul, never commit (orchestrator commits). ONE pytest at a
time, targeted selections; user's laptop, 16 threads/30GB. Strengthen = the recorded mutation
must now FAIL (verify headline ones in a scratch worktree under scratchpad/worktrees/, own
`uv sync` inside the worktree — the main venv's editable install points at main sources).
New tests must fail under their target mutation and pass at HEAD. ruff check + format clean.


## STATE UPDATE (2026-07-31, post-F5a/F5b) — brief recovered from transcript after /tmp wipe #2
- This file now lives IN THE REPO (untracked) at scratchpad/audit/AUDIT_BRIEF.md because /tmp keeps getting wiped. Delete after F7.
- F5a committed as d6bf8ab (326->295 items). F5b committed as 01ed348 (540->486 items, + tests/nn_layers/conftest.py).
- All 13 F5a/F5b mutation validations passed protocol. Suite ~3,106 items. Branch head: 01ed348.
- Remaining queue: F5c + F6-M1 (dispatched in parallel, 2-agent cap), then F6-M2, F6-M3, F7 + final gates.

## OOM ROOT CAUSE (2026-07-31 investigation — supersedes the multi-file-pytest theory)
All four kernel OOM kills were compute_hyperbolic_delta materializing an eager (n,n,n) buffer
(64GB at n=2000 f64) via tests/test_helpers.py::test_subsampling's un-subsampled diam_full leg.
JAX async dispatch + the test discarding `delta` swallowed the error; overcommit made death a
race against process exit ("multi-file OOM" was a misdiagnosis of that race). get_delta's
DEFAULT sample_size=1500 (27GB) is equally lethal — real library bug, new #7.
MITIGATION FOR ALL FUTURE pytest RUNS (agents included): wrap as
  bash -c 'ulimit -v 12000000; uv run pytest <file> -q'
so a runaway allocation fails as a clean JaxRuntimeError instead of an OOM kill.

## STATE UPDATE (2026-07-31 late, pre-compaction #2)
- b662eb5 committed: compute_hyperbolic_delta correctness fix (lib bug #1) + lax.scan O(n^2)
  memory rewrite (new bug #7) + M1-03 value oracles + hardened test_subsampling.
  hyperbolix/utils/helpers.py and tests/test_helpers.py are FROZEN (M1-03 complete).
- Branch head: b662eb5 (9 commits). Committed so far: F0, F2, F1, F4, F3, F5a (d6bf8ab),
  F5b (01ed348), delta fix (b662eb5).
- RUNNING: F5c (per-layer residual oracles) + F6-M1 (remaining specs M1-01..M1-14 minus 03),
  both Opus, resumed with the MANDATORY pytest wrapper:
  bash -c 'ulimit -v 12000000; cd <abs dir> && timeout 600 uv run pytest <ONE file> -q'
  and explicit-cd rule (shell cwd persists; caused worktree/main mixups).
- User approvals this stretch: full delta fix on branch (bug #7); keep 2 agents + ulimit guard.
- On each agent completion: orchestrator reviews diff, spot-checks mutation transcripts,
  runs capped single-file verifications, commits. Then dispatch F6-M2 (wrapped-normal
  log_prob oracles + lib fixes #2,#3) into the freed slot, then F6-M3, then F7 + final gates
  (spec in ORCHESTRATION STATE above; add: mutation spot-check count now includes delta).

## LIB BUG #8 (2026-07-31, user report, VERIFIED) — LogCat inverted digamma scale → batch F6-L
`_log_radius_concat` (hyperbolix/manifolds/hyperboloid.py:761) has the digamma arguments
swapped: shipped `digamma(n_total/2) - digamma(d/2)` gives scale > 1 (amplifies ~sqrt(N));
correct is `digamma(d/2) - digamma(n_total/2)` (shrink ~1/sqrt(N)), matching the HNN++
beta-concat analog it cites. Orchestrator verified 2026-07-31: E[log||v_spatial||] invariant
(the docstring's own contract) is preserved ONLY by the swapped form at every (N,d) probed
((9,16),(9,32),(9,64),(4,32),(2,32)); shipped overshoots WORSE than plain hcat. Manifold
residual ~1e-13 either way → nothing raises; N=1 gives scale=1 under both forms (that test
does not discriminate). Upstream ILNN reference (Longchentong/ILNN LConv.py) has the same
bug — faithful port of an upstream error. Blast radius: only HypConv2DHyperboloidILNN
(hyperboloid_conv.py:858); FGG/other convs use hcat. Net effect ~9.4x spatial-radius
inflation per 3x3 ILNN conv layer, compounding with depth; user's Atari sweep saw 19/130
ILNN cells with absorbing origin-collapsed deep blocks (0/130 for FGG) — consistent,
causality untested.

USER DECISION (AskUserQuestion 2026-07-31): "Fix + norm-preserving init". The sign fix is
NOT drop-in — user's forward probe shows fixing the sign alone collapses a depth-3+ ILNN
stack to the origin AT INIT under the current default kernel init (std 0.02 was tuned
against the buggy amplification; the two errors cancel at t=0, so the bug is invisible at
init and nothing is clamped). Fix must ship with a re-derived default init (HTC/FGG
precedent: RL-norm-preserving default, reference init recoverable via arg).

### F6-L spec (Opus; DO NOT DISPATCH until F5c completes — file overlap on conv tests)
Owns: hyperbolix/manifolds/hyperboloid.py (_log_radius_concat + docstring formula line),
hyperbolix/nn_layers/hyperboloid_conv.py (ILNN default init), tests/nn_layers/
test_hyperboloid_conv.py, tests/nn_layers/test_hyperboloid_conv_ilnn.py. Disjoint from
F6-M2 (distributions) → may run in parallel with it under the 2-agent cap.
- L-01 lib fix: swap digamma args at hyperboloid.py:761; correct the docstring formula
  (psi(n_i/2) - psi(n/2)) accordingly.
- L-02 oracle rewrite: tests/nn_layers/test_hyperboloid_conv.py:186-254 block — the
  `_expected_scale` helper transcribes the SHIPPED formula (self-confirming, V1). Replace
  with the INVARIANT oracle: Gaussian spatial parts across several (N,d) incl. asymmetric
  ones, assert E[log||v_spatial||] preserved to tolerance (invariant is the contract, the
  formula is implementation). Keep on-manifold, N=1->hcat-identity, and time-formula tests
  (scale-direction invariant). Do not touch the hcat sub-block.
- L-03 init re-derivation: norm-preserving default kernel init for HypConv2DHyperboloidILNN
  under the FIXED LogCat — per-layer gain ~1 at init (user probe: compensation ratio ~ N =
  K^2, i.e. ~9.25x for 3x3; derive properly, don't hardcode 9.25). Reference/old init must
  stay recoverable via the existing kernel_init_std-style arg (bit-for-bit if feasible, as
  HTC's init_bound=0.02 does). Update CLAUDE.md init section + docstrings.
- L-04 init probe test: deterministic f64 forward at init through a depth-3 ILNN conv stack
  (small spatial input, c=0.1, fixed seed); assert per-layer mean spatial norm neither
  collapses (>0.1x per layer) nor blows up (<10x per layer). Must FAIL under (a) fixed sign
  + old 0.02 init [collapse], and pass at new HEAD.
- L-05 mutation gates (worktree protocol as usual): (m1) revert digamma swap -> L-02
  invariant test FAILS; (m2) drop the scale entirely (hcat behavior) -> L-02 FAILS (hcat
  also misses the target, verified); (m3) revert init to fixed 0.02 -> L-04 FAILS.
- Coordination: F5c was messaged 2026-07-31 not to pin the shipped scale direction and to
  leave conv.py:186-254 untouched; check F5c's final report for compliance before dispatch.
  The old do-not-touch entry "hcat/log-radius block conv:20-254" is SUPERSEDED for the
  logcat sub-block (186-254): F6-L owns rewriting it. hcat tests remain do-not-touch.
- GitHub-issues list addendum (file with #4-#6 at the end): LogCat at the conv->FC flatten
  (flatten = concat of H'*W' manifold points; fixed-sign LogCat is a drop-in rescale there;
  user measured FGG head cap occupancy 100%->0% at init). Docs/feature suggestion, not a
  bug; include the user's report tables.

## STATE UPDATE (2026-07-31, post-F6-M1)
- F6-M1 committed as b0e1999 (+866 lines: new tests/test_manifold_oracles.py 90 items, curvature
  re-entry, smooth_clamp gradient oracles, dist_min non-metric counterexamples). All 11 mutation
  validations verified in agent report; orchestrator re-ran all 4 files capped (90/98/17/312 pass)
  + ruff. M1-13 does not exist (numbering skip). test_manifolds.py unmodified (297).
- F6-M2 DISPATCHED (Opus): log_prob 4 oracles + lib fixes #2/#3 + M2-02..M2-15, worktree
  fix-F6M2. F5c still RUNNING (nn_layers residual oracles; warned re LogCat bug #8).
- Deferred-issues addendum (file with #4-#6 + LogCat-flatten at the end): F6-M1 finding on
  LearnableCurvature straight_through_clamp — with the `log` parameterization dc/draw = c
  (~3.3e6 past the boundary), so plain SGD's first step overshoots the whole clamp interval
  and c ends pinned at c_min instead of converging (softplus converges fine; Adam does not
  rescue log — second moment lags the decaying gradient). Pinned as-is in
  test_manifold_curvature.py::test_straight_through_clamp_lets_curvature_re_enter...; consider
  a library-side note or a gradient-rescaled straight-through variant.

## STATE UPDATE (2026-07-31, post-F5c)
- F5c committed as c871e25 (12 nn_layers files, 165->166 items: +27 NumPy-transcription
  oracles, -26 vacuous; all 8 mutation gates verified; orchestrator re-ran all 12 files
  capped, all pass; ruff clean). F5c's ILNN tests are LogCat-scale-agnostic
  (_implied_logcat_scale) — they must survive the F6-L fix unweakened.
- RUNNING: F6-M2 (distributions oracles + lib fixes #2/#3, worktree fix-F6M2,
  _wrapped_normal_base.py already modified in main tree — do NOT commit it with other
  batches) + F6-L (LogCat digamma fix + norm-preserving ILNN init, worktree fix-F6L,
  spec §F6-L above). 2-agent cap holds.
- Remaining queue after these: F6-M3, then F7 + final gates (+ GitHub issues #4-#6,
  LogCat-flatten note, straight-through/log-parameterization SGD overshoot).

## STATE UPDATE (2026-07-31, post-F6-L)
- F6-L committed as 213021e: LogCat digamma swap (lib bug #8) + ILNN fan-out default init
  (kernel_init_std=None -> sqrt(1/out_spatial); 0.02 restores the reference draw bit-for-bit)
  + chi-closed-form invariant oracle rewrite of the conv LogCat block + depth-3 init probe
  + CLAUDE.md/changelog entries. All 3 mutation gates verified; orchestrator re-ran repro
  (invariant now holds at every (N,d)) + 3 test files capped, green. Task #6 completed.
  Note: released 1.0.0 changelog entry still prints the old (wrong) formula as if correct —
  left intact per Keep-a-Changelog; flag to user for a possible annotation.
- Agent self-reported rule slip: one `git checkout -- .` inside its own throwaway mutation
  worktree right before force-removal (ask-tier construct; harmless here, but surfaced).
- RUNNING: F6-M2 (distributions, worktree fix-F6M2) + F6-M3 (optim/attention/VQ/BN gap tests
  M3-G1..G17, worktree fix-F6M3, NEW file tests/nn_layers/test_helpers_validation.py,
  test-side only — no lib changes). 2-agent cap holds.
- Remaining queue: review+commit F6-M2 and F6-M3, then F7 sweep + final gates + GitHub
  issues (#4-#6, LogCat-at-flatten, straight-through log-parameterization overshoot).

## STATE UPDATE (2026-07-31, post-F6-M2)
- F6-M2 committed as 0bd8b1a (7 files, +729/-49): lib fixes #2 (log_prob S/B batching —
  BOTH steps, shared _vmap_sample_and_batch helper) and #3 (Poincare r floor + double-where;
  NaN was f64 too; base-helper defect unreachable from log_prob -> direct-call test) + the
  four log_prob value oracles + M2-04..M2-15. 16 mutation validations incl. fix reversions.
  Orchestrator re-ran the 4 owned files + test_cosne capped: all green. Items 230->291.
- ALL approved lib bugs now fixed: #1 (b662eb5), #7 (b662eb5), #8 (213021e), #2+#3 (0bd8b1a).
- F7 TODO addendum: add changelog entries for fixes #2/#3 (F6-M2 did not touch
  docs/changelog.md; F6-L's entries are the template). Agent-reported caveats to carry into
  the journal notes: frechet_mean tol=1e-8 leaves ~7e-8 geodesic gap; orthonormalize_rows
  does not preserve row span on rank-deficient input (documented caveat, not bug).
- RUNNING: F6-M3 only (optim/attention/VQ/BN gaps, worktree fix-F6M3, test-side only;
  test_optimizers.py already modified in main tree — exclude from any other staging).
- After F6-M3: F7 sweep (Sonnet) + FINAL GATES: full suite green per-file capped, item
  accounting vs 9,404 baseline, coverage diff vs fb14e0d (worktree baseline), CI matrix
  dry-run, ~15-mutation re-run spot-check (now incl. delta + logcat + log_prob mutations),
  CLAUDE.md "~1,660 tests" count update, GitHub issues (#4-#6, LogCat-flatten,
  straight-through log overshoot), delete scratchpad/audit + scratchpad/f6l.

## STATE UPDATE (2026-07-31, F7 prep)
- User instruction "dispatch F6-L once F5c lands" was already satisfied (c871e25, 213021e).
- Dispatched F7-prep baseline agent (Sonnet): coverage baseline at branch point fb14e0d in
  worktree scratchpad/worktrees/baseline-fb14e0d, per-file capped runs, results to
  scratchpad/audit/baseline_fb14e0d/ (coverage_baseline.json, runs.tsv, SUMMARY.md).
  Expected: tests/test_helpers.py may error under the VA cap at that ref (pre-fix delta bug).
- Concurrency: F6-M3 + baseline agent = 2 test runners (at cap).

## STATE UPDATE (2026-07-31, post-F6-M3) — ALL FIX BATCHES LANDED
- F6-M3 committed as 46d6011 (+631 lines, +29 items): optimizer optax-parity suite,
  causal linear-attention naive-einsum oracle (QUADRATIC_ATTN_CLASSES exclusion removed),
  NEW tests/nn_layers/test_helpers_validation.py (as_pair, validators, busemann Thm 4.1/4.2).
  18 mutation validations verified; G9-G12/G16 confirmed already-landed, not duplicated.
  Orchestrator re-ran the 3 files capped + ruff: green.
- Branch head 46d6011; 14 overhaul commits; every planned batch F0-F6 committed.
- Deferred-issues list for F7 (final): lib #4 (smooth_clamp window < ln2/beta), #5
  (Euclidean.is_in_manifold NaN), #6 (f32 Taylor-gate dtype), LogCat-at-flatten usage note,
  LearnableCurvature log-parameterization SGD overshoot, LR schedules evaluated at
  state.count+1 (off-by-one vs optax scale_by_schedule; TODO pinned in
  test_lr_schedule_changes_the_step_size_each_update), LorentzConv2D has no
  manifold-family validation (design note, takes no manifold_module).
- WAITING on baseline coverage run (background script in worktree baseline-fb14e0d,
  sequential per-file, ulimit inherited; agent resumes automatically to export
  coverage_baseline.json + SUMMARY.md and remove the worktree).
- THEN F7 (Sonnet) + FINAL GATES per the F7 TODO addendum above (incl. changelog entries
  for fixes #2/#3, CLAUDE.md test-count update, GitHub issues, scratchpad cleanup).
