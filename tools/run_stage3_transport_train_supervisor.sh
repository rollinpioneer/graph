#!/usr/bin/env bash

set -uo pipefail

CUPID_ROOT=/home/xushijie/CUPID
source "$CUPID_ROOT/stage3_transport_env.sh"
LOG_DIR="$CUPID_ROOT/logs/stage3_transport"
STATUS_DIR="$CUPID_ROOT/status/stage3_transport"
PID_DIR="$CUPID_ROOT/pids/stage3_transport"
EVENT_LOG="$LOG_DIR/train_supervisor.log"
LOCK_FILE="$PID_DIR/train_supervisor.lock"
SUPERVISOR_PID_FILE="$PID_DIR/train_supervisor.pid"
TRAIN_PID_FILE="$PID_DIR/train_child.pid"
RESTART_FILE="$LOG_DIR/train_restart_count.txt"
MAX_RAPID_FAILURES=5

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$PID_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'event=duplicate_supervisor time=%s\n' "$(date --iso-8601=seconds)" >> "$EVENT_LOG"
    exit 2
fi

printf '%s\n' "$$" > "$SUPERVISOR_PID_FILE"
child_pid=""
terminal_state=""
attempt=0
cleanup() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    rm -f "$TRAIN_PID_FILE" "$SUPERVISOR_PID_FILE"
}
handle_signal() {
    signal_name=$1
    terminal_state=terminated
    {
        printf 'TERMINATED\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'signal=%s\n' "$signal_name"
        printf 'attempt=%d\n' "$attempt"
    } > "$STATUS_DIR/train.terminated"
    rm -f "$STATUS_DIR/train.running"
    printf 'event=supervisor_terminated time=%s signal=%s attempt=%d\n' \
        "$(date --iso-8601=seconds)" "$signal_name" "$attempt" >> "$EVENT_LOG"
    cleanup
    trap - EXIT
    exit 143
}
trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

mkdir -p "$STAGE3_TRANSPORT_TRAIN_DIR"
cat > "$CUPID_ROOT/stage3_transport_run.env" <<EOF
export TRAIN_DIR='$STAGE3_TRANSPORT_TRAIN_DIR'
export TRAIN_NAME='$STAGE3_TRANSPORT_NAME'
export EVAL_DIR='$STAGE3_TRANSPORT_EVAL_DIR'
export DATASET_PATH='$STAGE3_TRANSPORT_DATASET'
export TEST_START_SEED='$STAGE3_TRANSPORT_TEST_START_SEED'
export NUM_ROLLOUTS='$STAGE3_TRANSPORT_NUM_ROLLOUTS'
export PHYSICAL_GPU='1'
EOF

rapid_failures=0
while true; do
    attempt=$((attempt + 1))
    restart_count=$((attempt - 1))
    printf '%s\n' "$restart_count" > "$RESTART_FILE"
    attempt_log=$(printf '%s/train_attempt_%03d.log' "$LOG_DIR" "$attempt")
    start_time=$(date --iso-8601=seconds)
    attempt_start=$SECONDS
    rm -f "$STATUS_DIR/train.terminated" "$STATUS_DIR/train.unrecoverable"

    {
        printf 'RUNNING\n'
        printf 'supervisor_pid=%s\n' "$$"
        printf 'attempt=%d\n' "$attempt"
        printf 'restart_count=%d\n' "$restart_count"
        printf 'physical_gpu=1\n'
        printf 'visible_gpu=0\n'
        printf 'checkpoint_resume=true\n'
        printf 'train_dir=%s\n' "$STAGE3_TRANSPORT_TRAIN_DIR"
        printf 'attempt_log=%s\n' "$attempt_log"
        printf 'start_time=%s\n' "$start_time"
    } > "$STATUS_DIR/train.running.tmp"
    mv "$STATUS_DIR/train.running.tmp" "$STATUS_DIR/train.running"
    printf 'event=attempt_start time=%s attempt=%d restart_count=%d log=%s\n' \
        "$start_time" "$attempt" "$restart_count" "$attempt_log" >> "$EVENT_LOG"

    cd "$REPO_DIR"
    python train.py \
        --config-dir=configs/low_dim/transport_mh/diffusion_policy_cnn \
        --config-name=config.yaml \
        name=train_diffusion_unet_lowdim \
        hydra.run.dir="$STAGE3_TRANSPORT_TRAIN_DIR" \
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
        logging.name="$STAGE3_TRANSPORT_NAME" \
        logging.group="$STAGE3_TRANSPORT_DATE" \
        logging.project=cupid \
        multi_run.wandb_name_base="$STAGE3_TRANSPORT_NAME" \
        multi_run.run_dir="$STAGE3_TRANSPORT_TRAIN_DIR" \
        > "$attempt_log" 2>&1 &
    child_pid=$!
    printf '%s\n' "$child_pid" > "$TRAIN_PID_FILE"
    set +e
    wait "$child_pid"
    return_code=$?
    set -e
    child_pid=""
    rm -f "$TRAIN_PID_FILE"
    end_time=$(date --iso-8601=seconds)
    wall_time_seconds=$((SECONDS - attempt_start))
    printf 'event=attempt_end time=%s attempt=%d return_code=%d wall_time_seconds=%d\n' \
        "$end_time" "$attempt" "$return_code" "$wall_time_seconds" >> "$EVENT_LOG"

    if [[ "$return_code" -eq 0 ]]; then
        {
            printf 'PASS\n'
            printf 'attempt=%d\n' "$attempt"
            printf 'restart_count=%d\n' "$restart_count"
            printf 'end_time=%s\n' "$end_time"
            printf 'wall_time_seconds=%d\n' "$wall_time_seconds"
            printf 'train_dir=%s\n' "$STAGE3_TRANSPORT_TRAIN_DIR"
            printf 'attempt_log=%s\n' "$attempt_log"
        } > "$STATUS_DIR/train.pass"
        rm -f "$STATUS_DIR/train.running" "$STATUS_DIR/train.last_failure"
        exit 0
    fi

    if (( wall_time_seconds < 300 )); then
        rapid_failures=$((rapid_failures + 1))
    else
        rapid_failures=0
    fi
    {
        printf 'RESTART_PENDING\n'
        printf 'attempt=%d\n' "$attempt"
        printf 'restart_count=%d\n' "$restart_count"
        printf 'return_code=%d\n' "$return_code"
        printf 'wall_time_seconds=%d\n' "$wall_time_seconds"
        printf 'rapid_failures=%d\n' "$rapid_failures"
        printf 'attempt_log=%s\n' "$attempt_log"
        printf 'next_retry_seconds=30\n'
    } > "$STATUS_DIR/train.last_failure"

    if (( rapid_failures >= MAX_RAPID_FAILURES )); then
        cp "$STATUS_DIR/train.last_failure" "$STATUS_DIR/train.unrecoverable"
        sed -i '1s/.*/UNRECOVERABLE/' "$STATUS_DIR/train.unrecoverable"
        rm -f "$STATUS_DIR/train.running"
        printf 'event=unrecoverable time=%s attempt=%d rapid_failures=%d\n' \
            "$end_time" "$attempt" "$rapid_failures" >> "$EVENT_LOG"
        exit "$return_code"
    fi
    sleep 30
done
