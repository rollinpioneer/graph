#!/usr/bin/env bash

set -o pipefail

CUPID_ROOT=/home/xushijie/CUPID
LOG_DIR="$CUPID_ROOT/logs"
STATUS_DIR="$CUPID_ROOT/status"
SUPERVISOR_PID_FILE="$LOG_DIR/formal_train_supervisor.pid"
TRAIN_PID_FILE="$LOG_DIR/formal_train_child.pid"
RESTART_FILE="$LOG_DIR/formal_train_restart_count.txt"
EVENT_LOG="$LOG_DIR/formal_train_supervisor.log"
LOCK_FILE="$LOG_DIR/formal_train_supervisor.lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '%s supervisor already running\n' "$(date --iso-8601=seconds)" >> "$EVENT_LOG"
    exit 2
fi

printf '%s\n' "$$" > "$SUPERVISOR_PID_FILE"
printf '%s\n' "0" > "$RESTART_FILE"

child_pid=""
cleanup() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    rm -f "$TRAIN_PID_FILE" "$SUPERVISOR_PID_FILE"
}
handle_signal() {
    cleanup
    trap - EXIT
    exit 143
}
trap cleanup EXIT
trap handle_signal INT TERM

attempt=0
while true; do
    attempt=$((attempt + 1))
    restart_count=$((attempt - 1))
    printf '%s\n' "$restart_count" > "$RESTART_FILE"
    attempt_log=$(printf '%s/formal_train1751_supervised_attempt_%03d.log' "$LOG_DIR" "$attempt")
    start_time=$(date --iso-8601=seconds)
    attempt_start_seconds=$SECONDS

    source "$CUPID_ROOT/experiment_env.sh"
    cd "$REPO_DIR" || exit 3

    {
        printf 'event=attempt_start time=%s attempt=%d restart_count=%d log=%s\n' \
            "$start_time" "$attempt" "$restart_count" "$attempt_log"
    } >> "$EVENT_LOG"
    {
        printf 'RUNNING\n'
        printf 'supervisor_pid=%s\n' "$$"
        printf 'attempt=%d\n' "$attempt"
        printf 'restart_count=%d\n' "$restart_count"
        printf 'physical_gpu=1\n'
        printf 'rollout_n_envs=8\n'
        printf 'checkpoint_resume=true\n'
        printf 'attempt_log=%s\n' "$attempt_log"
        printf 'start_time=%s\n' "$start_time"
    } > "$STATUS_DIR/formal_train_supervised.running.tmp"
    mv "$STATUS_DIR/formal_train_supervised.running.tmp" \
        "$STATUS_DIR/formal_train_supervised.running"

    python train.py \
        --config-dir=configs/low_dim/square_mh/diffusion_policy_cnn \
        --config-name=config.yaml \
        name=train_diffusion_unet_lowdim \
        hydra.run.dir="$TRAIN_DIR" \
        training.seed=0 \
        training.num_epochs=1751 \
        training.resume=true \
        training.checkpoint_every=50 \
        training.rollout_every=50 \
        checkpoint.topk.k=3 \
        task.dataset.seed=0 \
        task.dataset.val_ratio=0.04 \
        +task.dataset.dataset_mask_kwargs.train_ratio=0.64 \
        +task.dataset.dataset_mask_kwargs.uniform_quality=true \
        task.env_runner.n_envs=8 \
        dataloader.num_workers=0 \
        val_dataloader.num_workers=0 \
        logging.mode=offline \
        logging.name="$TRAIN_NAME" \
        logging.group="$TRAIN_DATE" \
        logging.project=cupid \
        multi_run.wandb_name_base="$TRAIN_NAME" \
        multi_run.run_dir="$TRAIN_DIR" \
        > "$attempt_log" 2>&1 &
    child_pid=$!
    printf '%s\n' "$child_pid" > "$TRAIN_PID_FILE"
    wait "$child_pid"
    return_code=$?
    child_pid=""
    rm -f "$TRAIN_PID_FILE"
    end_time=$(date --iso-8601=seconds)
    wall_time_seconds=$((SECONDS - attempt_start_seconds))

    printf 'event=attempt_end time=%s attempt=%d return_code=%d wall_time_seconds=%d\n' \
        "$end_time" "$attempt" "$return_code" "$wall_time_seconds" >> "$EVENT_LOG"

    if [[ "$return_code" -eq 0 ]]; then
        {
            printf 'PASS\n'
            printf 'attempt=%d\n' "$attempt"
            printf 'restart_count=%d\n' "$restart_count"
            printf 'end_time=%s\n' "$end_time"
            printf 'wall_time_seconds=%d\n' "$wall_time_seconds"
            printf 'attempt_log=%s\n' "$attempt_log"
        } > "$STATUS_DIR/formal_train_supervised.pass"
        rm -f "$STATUS_DIR/formal_train_supervised.running"
        exit 0
    fi

    {
        printf 'RESTART_PENDING\n'
        printf 'attempt=%d\n' "$attempt"
        printf 'return_code=%d\n' "$return_code"
        printf 'end_time=%s\n' "$end_time"
        printf 'wall_time_seconds=%d\n' "$wall_time_seconds"
        printf 'attempt_log=%s\n' "$attempt_log"
        printf 'next_retry_seconds=30\n'
    } > "$STATUS_DIR/formal_train_supervised.last_failure"
    sleep 30
done
