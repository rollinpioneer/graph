#!/usr/bin/env bash
# Run from a NEW output root. Complete training before any final test.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${REPO:?set REPO to pinned descendant worktree}"
: "${V6_TOOLS:?set byte-locked V6 tools path}"
: "${DATA:?set a new experiment data root}"
PY=${PY:-python}
cd "$ROOT"
test ! -e "$DATA"
mkdir -p "$(dirname "$DATA")"
mkdir "$DATA"
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B tools/verify_package.py --root "$ROOT"
"$PY" -B tools/history_scan.py --repo "$REPO" --out "$DATA/family_history_scan.json"
"$PY" -B -m p2b plan --out "$DATA/plan"
"$PY" -B -m p2b preflight --repo "$REPO" --v6-tools "$V6_TOOLS" --out "$DATA/preflight"
# Required server smoke belongs in its own non-research directory.
for M in TASK_ONLY GRAPH_FULL_PBRS; do
 "$PY" -B -m p2b smoke --repo "$REPO" --v6-tools "$V6_TOOLS" --method "$M" --seed 211 --out "$DATA/smoke/$M"
done
METHODS=(TASK_ONLY COUNT_PBRS COUNT_EVENTS_PBRS GEOM_COUNT_EVENTS_PBRS GRAPH_COST_PBRS GRAPH_FULL_PBRS)
SEEDS=(211 223 227 229 233 239 241 251)
failed=0
for S in "${SEEDS[@]}"; do
 for M in "${METHODS[@]}"; do
  if ! "$PY" -B -m p2b train-job --repo "$REPO" --v6-tools "$V6_TOOLS" --method "$M" --seed "$S" --out "$DATA/training/$M/seed_$S"; then
   printf '%s,%s\n' "$M" "$S" >>"$DATA/failed_jobs.csv"
   failed=1
  fi
 done
done
if [[ "$failed" == 1 ]]; then
 echo 'TRAINING_MATRIX_INCOMPLETE. Failures retained; no hidden retry or final comparative claim.'
 exit 2
fi
"$PY" -B -m p2b seal-test --root "$DATA" --out "$DATA/seal"
for S in "${SEEDS[@]}"; do
 for M in "${METHODS[@]}"; do
  "$PY" -B -m p2b test-job --repo "$REPO" --v6-tools "$V6_TOOLS" --method "$M" --seed "$S" --seal "$DATA/seal" --out "$DATA/test/$M/seed_$S"
 done
done
"$PY" -B -m p2b summarize --root "$DATA" --out "$DATA/summary"
