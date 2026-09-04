#!/usr/bin/env bash

set -u
CUPID_ROOT=/home/xushijie/CUPID
LOG_DIR="$CUPID_ROOT/logs/stage3_transport"
PID_DIR="$CUPID_ROOT/pids/stage3_transport"
SUPERVISOR_PID_FILE="$PID_DIR/train_supervisor.pid"
TRAIN_PID_FILE="$PID_DIR/train_child.pid"
MONITOR_PID_FILE="$PID_DIR/gpu_monitor.pid"
PEAK_FILE="$LOG_DIR/train_peak_gpu_memory_mib.txt"
EVENT_LOG="$LOG_DIR/gpu_monitor.log"
LOCK_FILE="$PID_DIR/gpu_monitor.lock"

mkdir -p "$LOG_DIR" "$PID_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 2
printf '%s\n' "$$" > "$MONITOR_PID_FILE"
trap 'rm -f "$MONITOR_PID_FILE"' EXIT
peak=0
[[ -s "$PEAK_FILE" ]] && read -r peak < "$PEAK_FILE"

while true; do
    supervisor_pid=""
    [[ -s "$SUPERVISOR_PID_FILE" ]] && read -r supervisor_pid < "$SUPERVISOR_PID_FILE"
    [[ -n "$supervisor_pid" ]] && kill -0 "$supervisor_pid" 2>/dev/null || break
    child_pid=""
    [[ -s "$TRAIN_PID_FILE" ]] && read -r child_pid < "$TRAIN_PID_FILE"
    used=0
    if [[ -n "$child_pid" ]]; then
        sample=$(nvidia-smi -i 1 --query-compute-apps=pid,used_memory \
            --format=csv,noheader,nounits 2>/dev/null || true)
        matched=$(awk -F, -v pid="$child_pid" '
            { gsub(/ /, "", $1); gsub(/ /, "", $2) }
            $1 == pid { print $2; exit }
        ' <<< "$sample")
        [[ -n "$matched" ]] && used=$matched
    fi
    if (( used > peak )); then
        peak=$used
        printf '%s\n' "$peak" > "$PEAK_FILE.tmp"
        mv "$PEAK_FILE.tmp" "$PEAK_FILE"
        printf 'time=%s child_pid=%s peak_gpu_memory_mib=%d\n' \
            "$(date --iso-8601=seconds)" "$child_pid" "$peak" >> "$EVENT_LOG"
    fi
    sleep 10
done

