#!/usr/bin/env bash

set -euo pipefail
CUPID_ROOT=/home/xushijie/CUPID
source "$CUPID_ROOT/stage3_transport_env.sh"
LOG_DIR="$CUPID_ROOT/logs/stage3_transport"
STATUS_DIR="$CUPID_ROOT/status/stage3_transport"
PID_DIR="$CUPID_ROOT/pids/stage3_transport"
FROZEN_DIR="$CUPID_ROOT/frozen/stage3_transport"
MANIFEST_DIR="$CUPID_ROOT/manifests/stage3_transport"
PIPELINE_LOG="$LOG_DIR/rollout_pipeline.log"
LOCK_FILE="$PID_DIR/rollout_pipeline.lock"
PID_FILE="$PID_DIR/rollout_pipeline.pid"
TRAIN_PASS="$STATUS_DIR/train.pass"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$PID_DIR" "$FROZEN_DIR" "$MANIFEST_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 2
printf '%s\n' "$$" > "$PID_FILE"
cleanup() { rm -f "$PID_FILE"; }
fail() {
    rc=$?
    {
        printf 'FAIL\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'return_code=%d\n' "$rc"
        printf 'line=%s\n' "${BASH_LINENO[0]:-unknown}"
        printf 'log=%s\n' "$PIPELINE_LOG"
    } > "$STATUS_DIR/rollout_pipeline.fail"
    rm -f "$STATUS_DIR/rollout_pipeline.running"
    cleanup
    exit "$rc"
}
trap cleanup EXIT
trap fail ERR
{
    printf 'WAITING_FOR_TRAINING_PASS\n'
    printf 'pid=%s\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
} > "$STATUS_DIR/rollout_pipeline.running"

while [[ ! -f "$TRAIN_PASS" ]]; do
    if [[ -f "$STATUS_DIR/train.unrecoverable" || -f "$STATUS_DIR/train.terminated" ]]; then
        printf 'training reached terminal failure before PASS\n' >&2
        false
    fi
    if [[ -s "$PID_DIR/train_supervisor.pid" ]]; then
        read -r supervisor_pid < "$PID_DIR/train_supervisor.pid"
        if ! kill -0 "$supervisor_pid" 2>/dev/null; then
            printf 'training supervisor PID is stale before PASS\n' >&2
            false
        fi
    fi
    sleep 30
done

ROLLOUT_CKPT="$STAGE3_TRANSPORT_TRAIN_DIR/checkpoints/latest.ckpt"
test -f "$ROLLOUT_CKPT"
test -f "$STAGE3_TRANSPORT_TRAIN_DIR/logs.json.txt"
test -f "$STAGE3_TRANSPORT_DATASET"
python "$CUPID_ROOT/tools/audit_final_training.py" \
    --train-dir "$STAGE3_TRANSPORT_TRAIN_DIR" --expected-last-epoch 1750 \
    > "$LOG_DIR/final_training_audit.log" 2>&1

mkdir -p "$FROZEN_DIR/training"
find "$STAGE3_TRANSPORT_TRAIN_DIR/checkpoints" -maxdepth 1 -type f -print0 \
    | sort -z | xargs -0 sha256sum > "$FROZEN_DIR/training/checkpoint_sha256.txt"
sha256sum "$STAGE3_TRANSPORT_DATASET" > "$FROZEN_DIR/training/dataset_sha256.txt"
cd "$REPO_DIR"
python "$CUPID_ROOT/tools/export_dataset_split.py" \
    --checkpoint "$ROLLOUT_CKPT" --output-dir "$MANIFEST_DIR/training_split" \
    > "$LOG_DIR/export_dataset_split.log" 2>&1
{
    printf 'rollout_checkpoint=%s\n' "$ROLLOUT_CKPT"
    printf 'selection_reason=pre-registered latest checkpoint\n'
    sha256sum "$ROLLOUT_CKPT"
} > "$FROZEN_DIR/rollout_checkpoint.txt"

test ! -e "$STAGE3_TRANSPORT_EVAL_DIR"
rollout_start=$SECONDS
date --iso-8601=seconds > "$LOG_DIR/rollout100_start.txt"
python eval_save_episodes.py \
    --output_dir="$STAGE3_TRANSPORT_EVAL_DIR" \
    --train_dir="$STAGE3_TRANSPORT_TRAIN_DIR" \
    --train_ckpt=latest \
    --num_episodes="$STAGE3_TRANSPORT_NUM_ROLLOUTS" \
    --test_start_seed="$STAGE3_TRANSPORT_TEST_START_SEED" \
    --overwrite=False --device=cuda:0 \
    > "$LOG_DIR/rollout100.log" 2>&1
rollout_wall_seconds=$((SECONDS - rollout_start))
date --iso-8601=seconds > "$LOG_DIR/rollout100_end.txt"
printf '%s\n' "$rollout_wall_seconds" > "$LOG_DIR/rollout100_wall_time_seconds.txt"

python "$CUPID_ROOT/tools/audit_rollout_pool.py" \
    --eval-dir "$STAGE3_TRANSPORT_EVAL_DIR" \
    --test-start-seed "$STAGE3_TRANSPORT_TEST_START_SEED" \
    --expected-episodes "$STAGE3_TRANSPORT_NUM_ROLLOUTS" \
    --output-dir "$MANIFEST_DIR/rollout100" \
    > "$LOG_DIR/audit_rollout100.log" 2>&1
python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["episode_count"] == 100; assert s["success_count"] >= 5; assert s["failure_count"] >= 5; print("TRANSPORT ROLLOUT CLASS-BALANCE GATE PASS")' \
    "$MANIFEST_DIR/rollout100/rollout_summary.json" \
    > "$LOG_DIR/rollout100_class_balance_gate.log" 2>&1

mkdir -p "$FROZEN_DIR/rollout100"
find "$STAGE3_TRANSPORT_EVAL_DIR" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$FROZEN_DIR/rollout100/all_files_sha256.txt"
du -sh "$STAGE3_TRANSPORT_EVAL_DIR" > "$FROZEN_DIR/rollout100/disk_usage.txt"
{
    printf 'eval_dir=%s\n' "$STAGE3_TRANSPORT_EVAL_DIR"
    printf 'test_start_seed=%s\n' "$STAGE3_TRANSPORT_TEST_START_SEED"
    printf 'num_rollouts=%s\n' "$STAGE3_TRANSPORT_NUM_ROLLOUTS"
    printf 'rollout_checkpoint=%s\n' "$ROLLOUT_CKPT"
    sha256sum "$ROLLOUT_CKPT"
} > "$FROZEN_DIR/rollout100/frozen_settings.txt"

touch "$STATUS_DIR/rollout100.pass"
{
    printf 'PASS\n'
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'eval_dir=%s\n' "$STAGE3_TRANSPORT_EVAL_DIR"
    printf 'rollout_wall_time_seconds=%d\n' "$rollout_wall_seconds"
} > "$STATUS_DIR/rollout_pipeline.pass"
rm -f "$STATUS_DIR/rollout_pipeline.running" "$STATUS_DIR/rollout_pipeline.fail"
printf 'event=rollout_pipeline_pass time=%s eval_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$STAGE3_TRANSPORT_EVAL_DIR" >> "$PIPELINE_LOG"
