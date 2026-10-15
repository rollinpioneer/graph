#!/usr/bin/env bash

set -u

CUPID_ROOT=/home/xushijie/CUPID
ENV_FILE=/home/__compress_data/xushijie/article/original_research/private/smtp_163.env
PID_DIR="$CUPID_ROOT/pids/stage3_transport"
STATUS_DIR="$CUPID_ROOT/status/stage3_transport"
LOG_DIR="$CUPID_ROOT/logs/stage3_transport"
LOCK_FILE="$PID_DIR/pipeline_email_guard.lock"
PID_FILE="$PID_DIR/pipeline_email_guard.pid"
DONE_FILE="$STATUS_DIR/pipeline_email_monitor.done"
LOG_FILE="$LOG_DIR/pipeline_email_guard.log"

mkdir -p "$PID_DIR" "$STATUS_DIR" "$LOG_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 2
printf '%s\n' "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

while [[ ! -f "$DONE_FILE" ]]; do
    printf 'event=monitor_launch time=%s\n' "$(date --iso-8601=seconds)" >> "$LOG_FILE"
    /home/xushijie/.conda/envs/cupid/bin/python \
        "$CUPID_ROOT/tools/monitor_stage3_transport_pipeline_email.py" \
        --root "$CUPID_ROOT" --env-file "$ENV_FILE" --poll-sec 30 \
        >> "$LOG_FILE" 2>&1
    rc=$?
    printf 'event=monitor_exit time=%s rc=%d\n' \
        "$(date --iso-8601=seconds)" "$rc" >> "$LOG_FILE"
    [[ -f "$DONE_FILE" ]] || sleep 15
done
