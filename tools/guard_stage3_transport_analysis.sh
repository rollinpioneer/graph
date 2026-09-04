#!/usr/bin/env bash

set -u

CUPID_ROOT=/home/xushijie/CUPID
STATUS_DIR="$CUPID_ROOT/status/stage3_transport"
LOG_DIR="$CUPID_ROOT/logs/stage3_transport"
PID_DIR="$CUPID_ROOT/pids/stage3_transport"
LOCK_FILE="$PID_DIR/analysis_guard.lock"
PID_FILE="$PID_DIR/analysis_guard.pid"
CHILD_PID_FILE="$PID_DIR/analysis_pipeline.pid"
EVENT_LOG="$LOG_DIR/analysis_pipeline.log"

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$PID_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 2
printf '%s\n' "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

while [[ ! -f "$STATUS_DIR/analysis_pipeline.pass" \
          && ! -f "$STATUS_DIR/analysis_pipeline.fail" ]]; do
    bash "$CUPID_ROOT/tools/run_stage3_transport_analysis_pipeline.sh" &
    child=$!
    printf '%s\n' "$child" > "$CHILD_PID_FILE"
    wait "$child"
    rc=$?
    rm -f "$CHILD_PID_FILE"
    if [[ -f "$STATUS_DIR/analysis_pipeline.pass" \
          || -f "$STATUS_DIR/analysis_pipeline.fail" ]]; then
        break
    fi
    printf 'event=analysis_child_exit time=%s return_code=%d restart=true\n' \
        "$(date --iso-8601=seconds)" "$rc" >> "$EVENT_LOG"
    sleep 15
done
