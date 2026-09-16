#!/usr/bin/env bash
set -Eeuo pipefail
# Run from the frozen package, not from the development Python working directory.
if [[ $# -ne 4 ]]; then echo 'usage: run_formal.sh REPO V6_TOOLS EXTRA_METHODS NEW_OUTPUT_ROOT' >&2; exit 2; fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-python}
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REPO=$1; V6_TOOLS=$2; EXTRA=$3; OUT=$4
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
"$PY" -B tools/preflight.py --repo "$REPO" --v6-tools "$V6_TOOLS" --extra-methods "$EXTRA" --out "$OUT/preflight.json"
"$PY" -B -m unittest discover -s tests -v > "$OUT/tests.log" 2>&1
"$PY" -B -m p2a prepare --out "$OUT/data"
"$PY" -B -m p2a weights --data "$OUT/data" --v6-tools "$V6_TOOLS" --extra-methods "$EXTRA" --out "$OUT/weights"
METHODS=(BC_UNIFORM SPARSE_TERMINAL LINEAR_A_FIRST_R1 LINEAR_B_FIRST_R1 UNORDERED_VALID_COUNT VALID_COUNT_PLUS_MATCHED_EVENTS_V1 V6_CAP_COST_ONLY V6_CAP_POTENTIAL V6_WEIGHT_SHUFFLED)
failed=0
for SEED in 17 29 43 71 101; do
 for M in "${METHODS[@]}"; do
  MODEL="$OUT/models/$M/$SEED"
  if "$PY" -B -m p2a train --weights "$OUT/weights" --method "$M" --seed "$SEED" --device "${P2A_DEVICE:-cpu}" --out "$MODEL"; then
    ok=1
    for SPLIT in validation test; do
      if ! "$PY" -B -m p2a evaluate --checkpoint "$MODEL/policy.pt" --data "$OUT/data" --split "$SPLIT" --out "$OUT/eval/$SPLIT/$M/$SEED"; then ok=0; failed=1; fi
    done
    printf '{"method":"%s","seed":%s,"trained":true,"evaluation_complete":%s}\n' "$M" "$SEED" "$([[ $ok -eq 1 ]] && echo true || echo false)" >> "$OUT/job_status.jsonl"
  else
    failed=1
    printf '{"method":"%s","seed":%s,"trained":false}\n' "$M" "$SEED" >> "$OUT/job_status.jsonl"
  fi
 done
done
if [[ $failed -ne 0 ]]; then
 printf '{"execution_status":"PARTIAL_TRAINING_MATRIX","policy_utility_evidence":"NOT_COMPLETE","confirmation_passed":false}\n' > "$OUT/partial_decision.json"
 exit 1
fi
"$PY" -B tools/audit_training.py --models "$OUT/models" --out "$OUT/initialization_batch_parity.json"
"$PY" -B -m p2a analyze --results "$OUT/eval/test" --out "$OUT/analysis"
