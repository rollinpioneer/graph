#!/usr/bin/env bash
set -euo pipefail
usage() { cat <<'EOF'
Usage: reproduce_from_cached_predictions.sh --ensemble-test PATH --ensemble-diagnostic PATH --output DIR [--max-content-groups N] [--dry-run]

Validates cached ensemble predictions without requiring packaged checkpoints. A complete reward/ablation reconstruction additionally needs the frozen oracle-trace path listed in manifests/external_artifact_manifest.tsv.
EOF
}
ensemble_test=""; ensemble_diagnostic=""; output=""; groups=""; dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ensemble-test) ensemble_test="$2"; shift 2;;
    --ensemble-diagnostic) ensemble_diagnostic="$2"; shift 2;;
    --output) output="$2"; shift 2;;
    --max-content-groups) groups="$2"; shift 2;;
    --dry-run) dry_run=true; shift;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$ensemble_test" && -n "$ensemble_diagnostic" && -n "$output" ]] || { usage >&2; exit 2; }
gzip -t "$ensemble_test"; gzip -t "$ensemble_diagnostic"
mkdir -p "$output"
python3 - "$ensemble_test" "$ensemble_diagnostic" "$groups" "$output" <<'PY'
import gzip,json,sys
test,diag,limit,out=sys.argv[1:]
def groups(path):
 with gzip.open(path,'rt') as h:
  return {json.loads(line)['content_group_id'] for line in h if line.strip()}
t,d=groups(test),groups(diag)
if not t or not d: raise SystemExit('empty cached prediction input')
if limit and len(t)>int(limit):
 print(f'cached test contains {len(t)} groups; dry-run uses validation only')
open(out+'/dry_run_manifest.json','w').write(json.dumps({'test_content_groups':len(t),'diagnostic_content_groups':len(d),'max_content_groups_requested':limit or None},indent=2)+'\n')
PY
if [[ "$dry_run" == true ]]; then
  echo CACHED_PREDICTION_DRY_RUN_PASS > "$output/dry_run_status.txt"
  echo CACHED_PREDICTION_DRY_RUN_PASS
  exit 0
fi
echo "Cached inputs are valid. Use the copied tools plus externally listed frozen traces/seed predictions for the full reward, ablation, and statistics reconstruction." >&2
exit 0
