#!/usr/bin/env bash

set -uo pipefail

CUPID_ROOT=/home/xushijie/CUPID
EXPERIMENT_ID=transport_filter50_pilot_20260725
ARMS=(baseline cupid50 quality50 random50)
STATUS_ROOT="$CUPID_ROOT/status/$EXPERIMENT_ID"
PID_ROOT="$CUPID_ROOT/pids/$EXPERIMENT_ID"
LOG_ROOT="$CUPID_ROOT/logs/$EXPERIMENT_ID"
RESULT_ROOT="$CUPID_ROOT/results/$EXPERIMENT_ID"
MANIFEST_ROOT="$CUPID_ROOT/manifests/$EXPERIMENT_ID"
PID_FILE="$PID_ROOT/finalize.pid"
LOCK_FILE="$PID_ROOT/finalize.lock"
EVENT_LOG="$LOG_ROOT/finalize.log"

mkdir -p "$STATUS_ROOT" "$PID_ROOT" "$LOG_ROOT" "$RESULT_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 2
printf '%s\n' "$$" > "$PID_FILE"
cleanup() { rm -f "$PID_FILE"; }
terminate() {
    {
        printf 'TERMINATED\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'signal=%s\n' "$1"
    } > "$STATUS_ROOT/overall.terminated"
    rm -f "$STATUS_ROOT/overall.running"
    cleanup
    trap - EXIT
    exit 143
}
trap cleanup EXIT
trap 'terminate INT' INT
trap 'terminate TERM' TERM

{
    printf 'WAITING_FOR_ARMS\n'
    printf 'pid=%s\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
} > "$STATUS_ROOT/overall.running"

while true; do
    all_pass=true
    for arm in "${ARMS[@]}"; do
        if [[ -f "$STATUS_ROOT/$arm/pipeline.unrecoverable" ||
              -f "$STATUS_ROOT/$arm/pipeline.terminated" ]]; then
            {
                printf 'FAIL\n'
                printf 'time=%s\n' "$(date --iso-8601=seconds)"
                printf 'failed_arm=%s\n' "$arm"
            } > "$STATUS_ROOT/overall.fail"
            rm -f "$STATUS_ROOT/overall.running"
            printf 'event=overall_fail time=%s arm=%s\n' \
                "$(date --iso-8601=seconds)" "$arm" >> "$EVENT_LOG"
            exit 1
        fi
        if [[ ! -f "$STATUS_ROOT/$arm/pipeline.pass" ]]; then
            all_pass=false
            pid_file="$PID_ROOT/$arm/supervisor.pid"
            if [[ -s "$pid_file" ]]; then
                read -r arm_pid < "$pid_file"
                if ! kill -0 "$arm_pid" 2>/dev/null; then
                    {
                        printf 'FAIL\n'
                        printf 'time=%s\n' "$(date --iso-8601=seconds)"
                        printf 'failed_arm=%s\n' "$arm"
                        printf 'reason=supervisor_lost\n'
                        printf 'pid=%s\n' "$arm_pid"
                    } > "$STATUS_ROOT/overall.fail"
                    rm -f "$STATUS_ROOT/overall.running"
                    printf 'event=supervisor_lost time=%s arm=%s pid=%s\n' \
                        "$(date --iso-8601=seconds)" "$arm" "$arm_pid" >> "$EVENT_LOG"
                    exit 1
                fi
            fi
        fi
    done
    [[ "$all_pass" == true ]] && break
    sleep 30
done

source "$CUPID_ROOT/env.sh"
summary_log="$LOG_ROOT/summary.log"
if ! python "$CUPID_ROOT/tools/summarize_transport_filter50_pilot.py" \
    --manifest-root "$MANIFEST_ROOT" \
    --output-dir "$RESULT_ROOT" > "$summary_log" 2>&1; then
    {
        printf 'FAIL\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'reason=summary_failed\n'
        printf 'log=%s\n' "$summary_log"
    } > "$STATUS_ROOT/overall.fail"
    rm -f "$STATUS_ROOT/overall.running"
    exit 1
fi

decision=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
    "$RESULT_ROOT/pilot_summary.json")
{
    printf 'PASS\n'
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'decision=%s\n' "$decision"
    printf 'result=%s\n' "$RESULT_ROOT/pilot_summary.json"
} > "$STATUS_ROOT/overall.pass"
if [[ "$decision" == "ADVANCE_TO_MULTI_SEED_CONFIRMATION" ]]; then
    cp "$STATUS_ROOT/overall.pass" "$STATUS_ROOT/overall.advance"
else
    cp "$STATUS_ROOT/overall.pass" "$STATUS_ROOT/overall.stop"
fi
rm -f "$STATUS_ROOT/overall.running" "$STATUS_ROOT/overall.fail"
printf 'event=overall_pass time=%s decision=%s\n' \
    "$(date --iso-8601=seconds)" "$decision" >> "$EVENT_LOG"
