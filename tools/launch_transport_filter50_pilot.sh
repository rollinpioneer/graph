#!/usr/bin/env bash

set -euo pipefail

CUPID_ROOT=/home/xushijie/CUPID
EXPERIMENT_ID=transport_filter50_pilot_20260725
MANIFEST="$CUPID_ROOT/manifests/$EXPERIMENT_ID/experiment_manifest.json"
SMTP_ENV=/home/__compress_data/xushijie/article/original_research/private/smtp_163.env
LOG_ROOT="$CUPID_ROOT/logs/$EXPERIMENT_ID"
PID_ROOT="$CUPID_ROOT/pids/$EXPERIMENT_ID"
FROZEN_ROOT="$CUPID_ROOT/frozen/$EXPERIMENT_ID"

test -f "$MANIFEST"
test -f "$SMTP_ENV"
mkdir -p "$LOG_ROOT" "$PID_ROOT" "$FROZEN_ROOT"

launch_arm() {
    arm=$1
    gpu=$2
    mode=$3
    filter_file=$4
    mkdir -p "$LOG_ROOT/$arm" "$PID_ROOT/$arm"
    cmd=(
        bash "$CUPID_ROOT/tools/run_transport_filter50_arm.sh"
        --manifest "$MANIFEST"
        --arm "$arm"
        --gpu "$gpu"
        --mode "$mode"
    )
    if [[ -n "$filter_file" ]]; then
        cmd+=(--filter-ids-file "$filter_file")
    fi
    {
        printf 'arm=%s gpu=%s mode=%s command=' "$arm" "$gpu" "$mode"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } >> "$FROZEN_ROOT/launch_commands.txt"
    setsid "${cmd[@]}" </dev/null >> "$LOG_ROOT/$arm/launcher.log" 2>&1 &
    printf '%s\n' "$!" > "$PID_ROOT/$arm/launcher.pid"
}

if [[ ! -f "$FROZEN_ROOT/launch_commands.txt" ]]; then
    {
        printf 'experiment=%s\n' "$EXPERIMENT_ID"
        printf 'launch_time=%s\n' "$(date --iso-8601=seconds)"
        printf 'manifest=%s\n' "$MANIFEST"
        printf 'evaluation_seeds=200000..200099\n'
        printf 'filtered_epochs=2301\n'
        printf 'checkpoint_resume=true\n'
    } > "$FROZEN_ROOT/launch_commands.txt"
fi

launch_arm baseline 2 eval_only ""
launch_arm cupid50 1 train_then_eval \
    "$CUPID_ROOT/manifests/$EXPERIMENT_ID/filter_ids/cupid50.txt"
launch_arm quality50 4 train_then_eval \
    "$CUPID_ROOT/manifests/$EXPERIMENT_ID/filter_ids/quality50.txt"
launch_arm random50 5 train_then_eval \
    "$CUPID_ROOT/manifests/$EXPERIMENT_ID/filter_ids/random50.txt"

setsid bash "$CUPID_ROOT/tools/run_transport_filter50_finalize.sh" \
    </dev/null >> "$LOG_ROOT/finalize_launcher.log" 2>&1 &
printf '%s\n' "$!" > "$PID_ROOT/finalize_launcher.pid"

setsid python "$CUPID_ROOT/tools/monitor_transport_filter50_pilot_email.py" \
    --root "$CUPID_ROOT" --env-file "$SMTP_ENV" --poll-sec 30 \
    </dev/null >> "$LOG_ROOT/email_monitor_launcher.log" 2>&1 &
printf '%s\n' "$!" > "$PID_ROOT/email_monitor_launcher.pid"

date --iso-8601=seconds > "$FROZEN_ROOT/launched_at.txt"
sha256sum \
    "$MANIFEST" \
    "$CUPID_ROOT/tools/run_transport_filter50_arm.sh" \
    "$CUPID_ROOT/tools/run_transport_filter50_finalize.sh" \
    "$CUPID_ROOT/tools/summarize_transport_filter50_pilot.py" \
    "$CUPID_ROOT/tools/monitor_transport_filter50_pilot_email.py" \
    "$CUPID_ROOT/tools/launch_transport_filter50_pilot.sh" \
    > "$FROZEN_ROOT/execution_sha256.txt"

printf 'TRANSPORT FILTER50 PILOT LAUNCHED\n'
