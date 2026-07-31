#!/bin/bash
# Sequential per-file coverage accumulation for baseline fb14e0d.
# ulimit set once here; inherited by every child pytest invocation below.
ulimit -v 12000000
WORKTREE=/home/timo/1Projects/hyperbolix/scratchpad/worktrees/baseline-fb14e0d
OUTDIR=/home/timo/1Projects/hyperbolix/scratchpad/audit/baseline_fb14e0d
cd "$WORKTREE" || exit 1

rm -f .coverage .coverage.*
RUNS="$OUTDIR/runs.tsv"
printf "file\texit_code\tseconds\tcollected\n" > "$RUNS"

find tests -name "test_*.py" | sort | while read -r f; do
  collected=$(awk -F': ' -v file="$f" '$1==file {print $2}' "$OUTDIR/collect_per_file.tsv" | tail -1)
  logfile="$OUTDIR/logs/$(echo "$f" | tr '/' '_').log"
  mkdir -p "$OUTDIR/logs"
  start=$(date +%s)
  timeout 900 uv run pytest "$f" --cov=hyperbolix --cov-append --cov-report= -q > "$logfile" 2>&1
  code=$?
  end=$(date +%s)
  secs=$((end - start))
  printf "%s\t%s\t%s\t%s\n" "$f" "$code" "$secs" "$collected" >> "$RUNS"
  echo "DONE $f exit=$code secs=$secs"
done

echo "ALL_DONE"
