#!/usr/bin/env bash

set -uo pipefail

CUPID_ROOT=/home/xushijie/CUPID
LOG_DIR="$CUPID_ROOT/logs"
STATUS_DIR="$CUPID_ROOT/status"
FROZEN_DIR="$CUPID_ROOT/frozen"
MANIFEST_DIR="$CUPID_ROOT/manifests"
TOOL_DIR="$CUPID_ROOT/tools"
RUN_ENV="$CUPID_ROOT/current_run.env"
TRAIN_PASS="$STATUS_DIR/formal_train_supervised.pass"
TRAIN_PID_FILE="$LOG_DIR/formal_train_supervisor.pid"
PIPELINE_PID_FILE="$LOG_DIR/post_training_rollout_pipeline.pid"
PIPELINE_LOG="$LOG_DIR/post_training_rollout_pipeline.log"
LOCK_FILE="$LOG_DIR/post_training_rollout_pipeline.lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$FROZEN_DIR" "$MANIFEST_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '%s pipeline already running\n' "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
    exit 2
fi

printf '%s\n' "$$" > "$PIPELINE_PID_FILE"
{
    printf 'WAITING_FOR_TRAINING_PASS\n'
    printf 'pid=%s\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'training_pass_file=%s\n' "$TRAIN_PASS"
} > "$STATUS_DIR/post_training_rollout_pipeline.running"
cleanup() {
    rm -f "$PIPELINE_PID_FILE"
}
fail() {
    rc=$?
    {
        printf 'FAIL\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'return_code=%d\n' "$rc"
        printf 'line=%s\n' "${BASH_LINENO[0]:-unknown}"
        printf 'log=%s\n' "$PIPELINE_LOG"
    } > "$STATUS_DIR/post_training_rollout_pipeline.fail"
    rm -f "$STATUS_DIR/post_training_rollout_pipeline.running"
    printf 'event=pipeline_fail time=%s rc=%d line=%s\n' \
        "$(date --iso-8601=seconds)" "$rc" "${BASH_LINENO[0]:-unknown}" >> "$PIPELINE_LOG"
    cleanup
    exit "$rc"
}
trap cleanup EXIT
trap fail ERR

printf 'event=wait_for_training time=%s\n' "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
while [[ ! -f "$TRAIN_PASS" ]]; do
    if [[ -s "$TRAIN_PID_FILE" ]]; then
        read -r supervisor_pid < "$TRAIN_PID_FILE"
        if ! kill -0 "$supervisor_pid" 2>/dev/null; then
            printf 'training supervisor disappeared before PASS\n' >&2
            false
        fi
    else
        printf 'training supervisor pid file disappeared before PASS\n' >&2
        false
    fi
    sleep 30
done

source "$RUN_ENV"
source "$CUPID_ROOT/env.sh"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_PATH="$REPO_DIR/data/robomimic/datasets/square/mh/low_dim_abs.hdf5"
export ROLLOUT_CKPT="$TRAIN_DIR/checkpoints/latest.ckpt"
export EVAL_DATE=20260721_cupid_square_rollout100
export TRAIN_NAME
TRAIN_NAME=$(basename "$TRAIN_DIR")
export EVAL_DIR="$REPO_DIR/data/outputs/eval_save_episodes/$EVAL_DATE/$TRAIN_NAME/latest"
export TEST_START_SEED=100000
export NUM_ROLLOUTS=100

printf 'event=training_pass_detected time=%s train_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$TRAIN_DIR" >> "$PIPELINE_LOG"

test -d "$TRAIN_DIR"
test -d "$TRAIN_DIR/checkpoints"
test -f "$ROLLOUT_CKPT"
test -f "$TRAIN_DIR/logs.json.txt"
test -f "$DATASET_PATH"

python "$TOOL_DIR/audit_final_training.py" \
    --train-dir "$TRAIN_DIR" --expected-last-epoch 1750 \
    > "$LOG_DIR/final_training_audit.log" 2>&1

mkdir -p "$FROZEN_DIR/training"
find "$TRAIN_DIR/checkpoints" -maxdepth 1 -type f -print0 \
    | sort -z | xargs -0 sha256sum \
    > "$FROZEN_DIR/training/checkpoint_sha256.txt"
find "$TRAIN_DIR/checkpoints" -maxdepth 1 -type f \
    -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort \
    > "$FROZEN_DIR/training/checkpoint_files.txt"

cd "$REPO_DIR"
{
    echo "HEAD:"
    git rev-parse HEAD || true
    echo
    echo "status:"
    git status --short || true
    echo
    echo "submodules:"
    git submodule status || true
} > "$FROZEN_DIR/training/code_state.txt" 2>&1
git diff > "$FROZEN_DIR/training/local_code_diff.patch" || true
git diff --cached > "$FROZEN_DIR/training/staged_code_diff.patch" || true

conda env export --no-builds > "$FROZEN_DIR/training/conda_environment_frozen.yaml"
pip freeze > "$FROZEN_DIR/training/pip_freeze.txt"
{
    date
    hostname
    cat /etc/os-release
    python --version
    nvidia-smi
} > "$FROZEN_DIR/training/system_frozen.txt" 2>&1
sha256sum "$DATASET_PATH" > "$FROZEN_DIR/training/dataset_sha256.txt"
ls -lh "$DATASET_PATH" > "$FROZEN_DIR/training/dataset_file.txt"

python "$TOOL_DIR/export_dataset_split.py" \
    --checkpoint "$ROLLOUT_CKPT" \
    --output-dir "$MANIFEST_DIR/training_split" \
    > "$LOG_DIR/export_dataset_split.log" 2>&1

{
    echo "rollout_checkpoint=$ROLLOUT_CKPT"
    echo "selection_reason=pre-registered latest checkpoint"
    sha256sum "$ROLLOUT_CKPT"
} > "$FROZEN_DIR/rollout_checkpoint.txt"

{
    printf 'export ROLLOUT_CKPT=%q\n' "$ROLLOUT_CKPT"
    printf 'export EVAL_DATE=%q\n' "$EVAL_DATE"
    printf 'export EVAL_DIR=%q\n' "$EVAL_DIR"
    printf 'export TEST_START_SEED=%q\n' "$TEST_START_SEED"
    printf 'export NUM_ROLLOUTS=%q\n' "$NUM_ROLLOUTS"
} >> "$RUN_ENV"

test ! -e "$EVAL_DIR"
date --iso-8601=seconds > "$LOG_DIR/rollout100_start.txt"
rollout_start=$SECONDS
cd "$REPO_DIR"
python eval_save_episodes.py \
    --output_dir="$EVAL_DIR" \
    --train_dir="$TRAIN_DIR" \
    --train_ckpt=latest \
    --num_episodes="$NUM_ROLLOUTS" \
    --test_start_seed="$TEST_START_SEED" \
    --overwrite=False \
    --device=cuda:0 \
    > "$LOG_DIR/rollout100.log" 2>&1
rollout_wall_seconds=$((SECONDS - rollout_start))
date --iso-8601=seconds > "$LOG_DIR/rollout100_end.txt"
printf '%s\n' "$rollout_wall_seconds" > "$LOG_DIR/rollout100_wall_time_seconds.txt"

python "$TOOL_DIR/audit_rollout_pool.py" \
    --eval-dir "$EVAL_DIR" \
    --test-start-seed "$TEST_START_SEED" \
    --expected-episodes "$NUM_ROLLOUTS" \
    --output-dir "$MANIFEST_DIR/rollout100" \
    > "$LOG_DIR/audit_rollout100.log" 2>&1

python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["episode_count"] == 100; assert s["success_count"] >= 5; assert s["failure_count"] >= 5; print("ROLLOUT CLASS-BALANCE GATE PASS")' \
    "$MANIFEST_DIR/rollout100/rollout_summary.json" \
    > "$LOG_DIR/rollout100_class_balance_gate.log" 2>&1

mkdir -p "$FROZEN_DIR/rollout100"
find "$EVAL_DIR" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$FROZEN_DIR/rollout100/all_files_sha256.txt"
du -sh "$EVAL_DIR" > "$FROZEN_DIR/rollout100/disk_usage.txt"
{
    echo "eval_dir=$EVAL_DIR"
    echo "test_start_seed=$TEST_START_SEED"
    echo "num_rollouts=$NUM_ROLLOUTS"
    echo "rollout_checkpoint=$ROLLOUT_CKPT"
    sha256sum "$ROLLOUT_CKPT"
} > "$FROZEN_DIR/rollout100/frozen_settings.txt"

touch "$STATUS_DIR/rollout100_frozen.pass"
{
    printf 'PASS\n'
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'train_dir=%s\n' "$TRAIN_DIR"
    printf 'eval_dir=%s\n' "$EVAL_DIR"
    printf 'rollout_wall_time_seconds=%d\n' "$rollout_wall_seconds"
} > "$STATUS_DIR/post_training_rollout_pipeline.pass"
rm -f "$STATUS_DIR/post_training_rollout_pipeline.running"
rm -f "$STATUS_DIR/post_training_rollout_pipeline.fail"
printf 'event=pipeline_pass time=%s eval_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$EVAL_DIR" >> "$PIPELINE_LOG"
