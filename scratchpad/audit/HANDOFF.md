# HANDOFF — test-overhaul final gates (F7)

**Written 2026-07-31 on Timo's laptop; intended executor: a fresh Claude session on a
compute server.** Companion file: `AUDIT_BRIEF.md` in this directory (full specs, bug log,
per-batch state updates). Read its "F6-M3" / "F7 TODO addendum" / final state-update
sections before starting. Everything in `scratchpad/audit/` is working material — it gets
deleted in the very last cleanup commit.

## 1. Where things stand

Branch `test-overhaul`, head `46d6011`, 14 overhaul commits on top of branch point
`fb14e0d`. **Every fix batch (F0–F6) is landed and committed.** Do not re-run or re-review
them. The five library bugs found during the overhaul are all fixed:

| Bug | Fix commit | What |
|---|---|---|
| #1 delta ≡ 0 indexing | `b662eb5` | compute_hyperbolic_delta max-min product indexed both operands on row i |
| #7 delta O(n³) memory | `b662eb5` | eager (n,n,n) broadcast → lax.scan O(n²); 64GB → 553MB |
| #8 LogCat inverted digamma | `213021e` | amplifies √N → shrinks 1/√N + coupled fan-out ILNN init (`kernel_init_std=None` → `sqrt(1/out_spatial)`) |
| #2 log_prob S/B batching | `0bd8b1a` | both steps rebuilt on `_vmap_sample_and_batch` |
| #3 log_prob NaN grad at mean | `0bd8b1a` | Poincaré r floor + double-where in `_log_det_jacobian_from_r` |

Suite: **9,243 collected items at `fb14e0d` (tests/ only; 9,404 with the since-removed
benchmark collection) → 3,285 now.** Per-file counts for the baseline are saved in
`baseline_fb14e0d/collect_per_file.tsv`.

## 2. Remaining work, in order

### 2a. Coverage baseline at fb14e0d (redo — the laptop run was stopped)

Procedure that was in flight (script template: `baseline_fb14e0d/run_coverage.sh`):

1. `git worktree add scratchpad/worktrees/baseline-fb14e0d fb14e0d --detach`
2. `uv sync --locked --dev` **inside the worktree**, then verify `import hyperbolix`
   resolves to the worktree path (TRAP: the main venv's editable install points at the
   main checkout — skipping this check silently measures the wrong code).
3. **TRAP: `pytest-cov` is NOT in the lockfile at `fb14e0d`** — every `--cov` run exits 4
   (usage error) without it. `uv pip install pytest-cov` into the worktree venv first
   (throwaway venv, off-lockfile is fine).
4. Per-file `pytest <file> --cov=hyperbolix --cov-append --cov-report= -q`, then
   `coverage json -o coverage_baseline.json` + `coverage report`.
5. **TRAP: `tests/test_helpers.py` at `fb14e0d` contains the pre-fix delta test that
   eagerly allocates ~64GB** (f64, n=2000; the async-dispatch swallow means it may even
   "pass" while poisoning the machine — this OOM-killed the laptop four times). Options:
   run that one file under `bash -c 'ulimit -v <cap>; ...'` so it fails as a clean
   JaxRuntimeError (record + move on; costs a sliver of helpers.py baseline coverage), or
   if the server has ≥ 80GB free RAM, let it run and note the choice. Either way, decide
   deliberately — do not let it fail as a kernel OOM kill.
6. Keep `coverage_baseline.json`, `runs.tsv` (file, exit, seconds), and a summary;
   remove the worktree.

### 2b. Current-branch coverage + full-suite green (can run concurrently with 2a)

Same per-file `--cov-append` pattern at HEAD in the main checkout (pytest-cov IS in the
current lockfile — no extra install). This doubles as the "full suite green" gate: every
file must exit 0. Current expected: 46 test files (45 at fb14e0d + new
`tests/nn_layers/test_helpers_validation.py`), 3,285 items.

### 2c. Gates (all must pass; a failure blocks and needs triage, not silence)

1. **Full suite green** — from 2b, all files exit 0.
2. **Item accounting** — `--collect-only -q` total vs the table above; explain any drift
   from 3,285 (agents' per-file before/after tables are in the AUDIT_BRIEF state updates
   and commit messages).
3. **Coverage no-drop** — per-file line coverage diff (2b vs 2a): any `hyperbolix/` file
   dropping **> 0.5%** blocks. Expected direction is up (the overhaul added oracles).
   Known acceptable artifact: if 2a capped `test_helpers.py`, baseline `utils/helpers.py`
   coverage is understated — note it, don't "fix" it.
4. **CI matrix dry-run** — every entry in `.github/workflows/ci.yaml`'s test matrix (F0
   expanded it) collects and passes locally; no test file missing from the matrix
   (compare `find tests -name "test_*.py"` against the matrix entries).
5. **Mutation spot-check re-run** (the journal number: "N/N seeded faults caught").
   Fresh worktree at HEAD + worktree-local `uv sync`. Re-apply ~15 mutations sampled from
   the batch reports and verify at least one test fails for each. Suggested sample
   (details in AUDIT_BRIEF "Uncaught mutations" + the F5a/F5b/F5c/F6-* commit messages):
   delta indexing revert (helpers.py); LogCat digamma revert AND scale=1; ILNN init back
   to 0.02 (sign fix kept); log_prob fix #2 revert; fix #3 floor revert; FHNN spatial
   sign flip (hyperboloid_linear.py:159); PLFC origin collapse (v*0); PV outer sinh
   deleted; Poincaré HNN++ MLR sign flip; Busemann logit sign flip (busemann_core.py:130);
   VQ argmin→argmax; VQ STE deleted; GyroBN centering dropped; attention q/k/v swap;
   causal-scan outer product transposed; RAdam m2 → elementwise g².
6. **Hygiene sweep** (Sonnet-grade): `ruff check` + `ruff format` + `pyright hyperbolix`
   clean; naming/docstring consistency pass over the new test files.

### 2d. Docs & bookkeeping edits (commit on the branch)

- `docs/changelog.md`: add **Unreleased entries for lib fixes #2 and #3** (F6-M2 shipped
  code without changelog entries; the `213021e` LogCat entries are the style template).
- `CLAUDE.md`: update "~1,660 tests" in Quick Reference to the final count.
- Optional, user's call was left open: annotate the released 0.11.0 changelog line that
  still prints the pre-fix LogCat formula as if correct (`docs/changelog.md:85`).

### 2e. GitHub issues to file (gh CLI; one issue each, cite commits/tests)

1. Lib bug #4: `smooth_clamp` can exceed `max_value` when window < ln2/beta (~0.0139 at
   beta=50).
2. Lib bug #5: `Euclidean.is_in_manifold` returns True unconditionally, incl. NaN
   (pinned with TODO in F2).
3. Lib bug #6: f32 Taylor-gate dtype comparison in `_log_det_jacobian_from_r` (bounded
   impact).
4. LR schedules evaluated at `state.count + 1` — off by one vs optax `scale_by_schedule`
   (TODO pinned in `test_lr_schedule_changes_the_step_size_each_update`).
5. LogCat at the conv→FC flatten: usage note / possible helper (measurements in the
   user's report, reproduced in AUDIT_BRIEF "LIB BUG #8" section).
6. LearnableCurvature `log` parameterization: straight-through clamp + plain SGD
   overshoots the entire clamp interval (dc/draw = c); pinned in
   `test_manifold_curvature.py` multi-step test.
7. Design note: `LorentzConv2D` takes no `manifold_module` → no manifold-family
   validation (only conv without it).

### 2f. Final cleanup + report

- Delete `scratchpad/` from the repo (`scratchpad/audit/`, `scratchpad/f6l/` probe
  scripts, `scratchpad/worktrees/` leftovers) in a final commit.
- Close out task-list items #1 (Stage 0 baseline) and #5 (Stage 2).
- Final report for Timo: per-gate results; the two journal numbers (9,243 → final item
  count with strictly stronger assertions; N/N mutations caught); coverage delta summary;
  links/numbers of the filed issues.

## 3. Server adaptations (the laptop rules that do NOT carry over)

The brief's rules "max 2 test-running agents", "ONE file per pytest invocation",
"ulimit -v 12000000 on every run" were sizing for a 16-thread/30GB laptop that this
session OOM-killed four times before bug #7 was fixed. On the server: **check
`nproc`/`free -g` first, then scale** (that check itself is a standing user preference).
Coverage sweeps parallelize trivially by sharding files across processes with separate
`COVERAGE_FILE`s and `coverage combine` at the end. At HEAD the suite has no known lethal
test (bug #7 is fixed; the 2000-point leg now peaks ~870MB capped). The one place a memory
guard is still genuinely needed is 2a's `test_helpers.py` at `fb14e0d` (see the trap).

Rules that DO carry over regardless of hardware:
- Worktree mutation runs need a worktree-local `uv sync` + import-resolution check.
- Explicit `cd` in the same command as every run (tool shell cwd persists between calls).
- Orchestrator reviews diffs and makes commits; sub-agents never commit.
- Frozen/do-not-touch: see AUDIT_BRIEF "Do-not-touch (global)"; additionally
  `hyperbolix/utils/helpers.py` + `tests/test_helpers.py` (M1-03 final), and F5c's
  scale-agnostic ILNN tests (`_implied_logcat_scale`) must not be re-pinned to any scale
  formula.
- Commit messages end with the Claude co-author line (see `git log`).

## 4. If running unattended

Launch with `claude --permission-mode dontAsk` so ask-tier calls fail fast instead of
stalling the session; surface any denials in the final report. Never SIGKILL; verify PIDs
before any kill; no `&` backgrounding in tool commands (use the detach helpers).
