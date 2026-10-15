#!/usr/bin/env bash

set -euo pipefail

CUPID_ROOT=/home/xushijie/CUPID
source "$CUPID_ROOT/stage3_transport_env.sh"
LOG_DIR="$CUPID_ROOT/logs/stage3_transport"
STATUS_DIR="$CUPID_ROOT/status/stage3_transport"
PID_DIR="$CUPID_ROOT/pids/stage3_transport"
RESULT_DIR="$CUPID_ROOT/results/stage3_transport"
MANIFEST_DIR="$CUPID_ROOT/manifests/stage3_transport"
OUTPUT_DIR="$CUPID_ROOT/frozen/stage3_transport/filter_selection"
PIPELINE_LOG="$LOG_DIR/filter_id_freezer.log"
LOCK_FILE="$PID_DIR/filter_id_freezer.lock"
PID_FILE="$PID_DIR/filter_id_freezer.pid"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$PID_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 2
printf '%s\n' "$$" > "$PID_FILE"
rm -f "$STATUS_DIR/filter_id_freezer.fail"

cleanup() {
    rm -f "$PID_FILE"
}
fail() {
    rc=$?
    {
        printf 'FAIL\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'return_code=%d\n' "$rc"
        printf 'line=%s\n' "${BASH_LINENO[0]:-unknown}"
        printf 'log=%s\n' "$PIPELINE_LOG"
    } > "$STATUS_DIR/filter_id_freezer.fail"
    rm -f "$STATUS_DIR/filter_id_freezer.running"
    cleanup
    exit "$rc"
}
trap cleanup EXIT
trap fail ERR

{
    printf 'WAITING_FOR_STABILITY_DECISION\n'
    printf 'pid=%s\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'filtered_training_auto_start=false\n'
} > "$STATUS_DIR/filter_id_freezer.running"

while [[ ! -f "$STATUS_DIR/stability_diagnosis.pass" ]]; do
    if [[ -f "$STATUS_DIR/analysis_pipeline.fail" ]]; then
        printf 'analysis pipeline failed before filter-ID decision\n' >&2
        false
    fi
    sleep 30
done

if [[ -f "$STATUS_DIR/stability_diagnosis.strong_pass" ]]; then
    test ! -e "$OUTPUT_DIR"
    python "$CUPID_ROOT/tools/freeze_stage3_transport_filter_ids.py" \
        --diagnosis-dir "$RESULT_DIR/stability_diagnosis" \
        --split-json "$MANIFEST_DIR/training_split/dataset_split.json" \
        --strong-pass-status "$STATUS_DIR/stability_diagnosis.strong_pass" \
        --output-dir "$OUTPUT_DIR" \
        --threshold 0.90 --proportion 0.20 \
        > "$PIPELINE_LOG" 2>&1
    test -f "$OUTPUT_DIR/filter_ids.json"
    test -f "$OUTPUT_DIR/output_files_sha256.txt"
    {
        printf 'PASS\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'output_dir=%s\n' "$OUTPUT_DIR"
        printf 'filtered_training_auto_started=false\n'
    } > "$STATUS_DIR/filter_id_freezer.pass"
    printf 'event=filter_ids_frozen time=%s output=%s\n' \
        "$(date --iso-8601=seconds)" "$OUTPUT_DIR" >> "$PIPELINE_LOG"
elif [[ -f "$STATUS_DIR/stability_diagnosis.stop" ]]; then
    {
        printf 'SKIP\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'reason=stability_gate_did_not_strong_pass\n'
        printf 'filtered_training_authorized=false\n'
    } > "$STATUS_DIR/filter_id_freezer.skip"
    printf 'event=filter_id_freezer_skip time=%s\n' \
        "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
else
    printf 'stability diagnosis PASS lacks strong-pass or stop decision\n' >&2
    false
fi

rm -f "$STATUS_DIR/filter_id_freezer.running" "$STATUS_DIR/filter_id_freezer.fail"
