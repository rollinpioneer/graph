#!/usr/bin/env bash

set -uo pipefail

CUPID_ROOT=/home/xushijie/CUPID
EXPERIMENT_ID=transport_filter50_pilot_20260725
MANIFEST=""
ARM=""
PHYSICAL_GPU=""
MODE=""
FILTER_IDS_FILE=""

while (($#)); do
    case "$1" in
        --manifest) MANIFEST=$2; shift 2 ;;
        --arm) ARM=$2; shift 2 ;;
        --gpu) PHYSICAL_GPU=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        --filter-ids-file) FILTER_IDS_FILE=$2; shift 2 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if [[ -z "$MANIFEST" || -z "$ARM" || -z "$PHYSICAL_GPU" || -z "$MODE" ]]; then
    printf 'manifest, arm, gpu, and mode are required\n' >&2
    exit 2
fi
if [[ "$MODE" != "eval_only" && "$MODE" != "train_then_eval" ]]; then
    printf 'unsupported mode: %s\n' "$MODE" >&2
    exit 2
fi
if [[ "$MODE" == "train_then_eval" && ! -s "$FILTER_IDS_FILE" ]]; then
    printf 'missing filter IDs: %s\n' "$FILTER_IDS_FILE" >&2
    exit 2
fi

source "$CUPID_ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="$CUPID_ROOT/$EXPERIMENT_ID"
LOG_DIR="$CUPID_ROOT/logs/$EXPERIMENT_ID/$ARM"
STATUS_DIR="$CUPID_ROOT/status/$EXPERIMENT_ID/$ARM"
PID_DIR="$CUPID_ROOT/pids/$EXPERIMENT_ID/$ARM"
MANIFEST_DIR="$CUPID_ROOT/manifests/$EXPERIMENT_ID/$ARM"
FROZEN_DIR="$CUPID_ROOT/frozen/$EXPERIMENT_ID/$ARM"
EVENT_LOG="$LOG_DIR/events.log"
LOCK_FILE="$PID_DIR/supervisor.lock"
SUPERVISOR_PID_FILE="$PID_DIR/supervisor.pid"
CHILD_PID_FILE="$PID_DIR/child.pid"

BASELINE_NAME=20260722_cupid_transport_stage3_train_diffusion_unet_lowdim_transport_mh_0
BASELINE_TRAIN_DIR="$REPO_DIR/data/outputs/train/20260722_cupid_transport_stage3/$BASELINE_NAME"
DATE_TAG=20260725_cupid_transport_filter50_pilot
if [[ "$MODE" == "eval_only" ]]; then
    TRAIN_NAME="$BASELINE_NAME"
    TRAIN_DIR="$BASELINE_TRAIN_DIR"
    EXPECTED_LAST_EPOCH=1750
else
    TRAIN_NAME="${DATE_TAG}_train_diffusion_unet_lowdim_transport_mh_0-${ARM}"
    TRAIN_DIR="$REPO_DIR/data/outputs/train/$DATE_TAG/$TRAIN_NAME"
    EXPECTED_LAST_EPOCH=2300
fi
EVAL_DIR="$REPO_DIR/data/outputs/eval_save_episodes/${DATE_TAG}_eval100/$TRAIN_NAME/latest"

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$STATUS_DIR" "$PID_DIR" "$MANIFEST_DIR" "$FROZEN_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'event=duplicate_supervisor time=%s arm=%s\n' \
        "$(date --iso-8601=seconds)" "$ARM" >> "$EVENT_LOG"
    exit 2
fi
if [[ -f "$STATUS_DIR/pipeline.pass" ]]; then
    exit 0
fi

printf '%s\n' "$$" > "$SUPERVISOR_PID_FILE"
child_pid=""
cleanup() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    rm -f "$CHILD_PID_FILE" "$SUPERVISOR_PID_FILE"
}
terminate() {
    signal_name=$1
    {
        printf 'TERMINATED\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'arm=%s\n' "$ARM"
        printf 'signal=%s\n' "$signal_name"
    } > "$STATUS_DIR/pipeline.terminated"
    rm -f "$STATUS_DIR/pipeline.running"
    printf 'event=supervisor_terminated time=%s arm=%s signal=%s\n' \
        "$(date --iso-8601=seconds)" "$ARM" "$signal_name" >> "$EVENT_LOG"
    cleanup
    trap - EXIT
    exit 143
}
trap cleanup EXIT
trap 'terminate INT' INT
trap 'terminate TERM' TERM

rm -f "$STATUS_DIR/pipeline.terminated" "$STATUS_DIR/pipeline.unrecoverable"
{
    printf 'RUNNING\n'
    printf 'experiment=%s\n' "$EXPERIMENT_ID"
    printf 'arm=%s\n' "$ARM"
    printf 'mode=%s\n' "$MODE"
    printf 'physical_gpu=%s\n' "$PHYSICAL_GPU"
    printf 'visible_gpu=0\n'
    printf 'checkpoint_resume=true\n'
    printf 'train_dir=%s\n' "$TRAIN_DIR"
    printf 'eval_dir=%s\n' "$EVAL_DIR"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
} > "$STATUS_DIR/pipeline.running.tmp"
mv "$STATUS_DIR/pipeline.running.tmp" "$STATUS_DIR/pipeline.running"
printf 'event=pipeline_start time=%s arm=%s mode=%s physical_gpu=%s\n' \
    "$(date --iso-8601=seconds)" "$ARM" "$MODE" "$PHYSICAL_GPU" >> "$EVENT_LOG"

if [[ "$MODE" == "train_then_eval" && ! -f "$STATUS_DIR/training.pass" ]]; then
    filter_ids_csv=$(paste -sd, "$FILTER_IDS_FILE")
    filter_ids_literal="[$filter_ids_csv]"
    attempt=0
    rapid_failures=0
    while true; do
        attempt=$((attempt + 1))
        attempt_log=$(printf '%s/train_attempt_%03d.log' "$LOG_DIR" "$attempt")
        attempt_start=$SECONDS
        {
            printf 'RUNNING\n'
            printf 'arm=%s\n' "$ARM"
            printf 'attempt=%d\n' "$attempt"
            printf 'restart_count=%d\n' "$((attempt - 1))"
            printf 'physical_gpu=%s\n' "$PHYSICAL_GPU"
            printf 'checkpoint_resume=true\n'
            printf 'train_dir=%s\n' "$TRAIN_DIR"
            printf 'attempt_log=%s\n' "$attempt_log"
            printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
        } > "$STATUS_DIR/training.running.tmp"
        mv "$STATUS_DIR/training.running.tmp" "$STATUS_DIR/training.running"

        train_cmd=(
            python train.py
            --config-dir=configs/low_dim/transport_mh/diffusion_policy_cnn
            --config-name=config.yaml
            name=train_diffusion_unet_lowdim
            "hydra.run.dir=$TRAIN_DIR"
            training.seed=0
            training.num_epochs=2301
            training.resume=true
            training.checkpoint_every=50
            training.rollout_every=50
            checkpoint.topk.k=1
            task.dataset.seed=0
            task.dataset.val_ratio=0.04
            +task.dataset.dataset_mask_kwargs.train_ratio=0.64
            +task.dataset.dataset_mask_kwargs.uniform_quality=true
            "+task.dataset.dataset_mask_kwargs.filter_episode_ids=$filter_ids_literal"
            task.env_runner.n_envs=8
            dataloader.num_workers=0
            val_dataloader.num_workers=0
            logging.mode=offline
            "logging.name=$TRAIN_NAME"
            "logging.group=$DATE_TAG"
            logging.project=cupid
            "multi_run.wandb_name_base=$TRAIN_NAME"
            "multi_run.run_dir=$TRAIN_DIR"
        )
        {
            printf 'event=train_attempt_start time=%s arm=%s attempt=%d command=' \
                "$(date --iso-8601=seconds)" "$ARM" "$attempt"
            printf '%q ' "${train_cmd[@]}"
            printf '\n'
        } >> "$EVENT_LOG"

        mkdir -p "$TRAIN_DIR"
        cd "$REPO_DIR"
        "${train_cmd[@]}" > "$attempt_log" 2>&1 &
        child_pid=$!
        printf '%s\n' "$child_pid" > "$CHILD_PID_FILE"
        set +e
        wait "$child_pid"
        rc=$?
        set -e
        child_pid=""
        rm -f "$CHILD_PID_FILE"
        wall_seconds=$((SECONDS - attempt_start))
        printf 'event=train_attempt_end time=%s arm=%s attempt=%d rc=%d wall_seconds=%d\n' \
            "$(date --iso-8601=seconds)" "$ARM" "$attempt" "$rc" "$wall_seconds" >> "$EVENT_LOG"

        if [[ "$rc" -eq 0 ]]; then
            break
        fi
        if ((wall_seconds < 300)); then
            rapid_failures=$((rapid_failures + 1))
        else
            rapid_failures=0
        fi
        {
            printf 'RESTART_PENDING\n'
            printf 'arm=%s\n' "$ARM"
            printf 'attempt=%d\n' "$attempt"
            printf 'restart_count=%d\n' "$((attempt - 1))"
            printf 'return_code=%d\n' "$rc"
            printf 'wall_time_seconds=%d\n' "$wall_seconds"
            printf 'rapid_failures=%d\n' "$rapid_failures"
            printf 'attempt_log=%s\n' "$attempt_log"
            printf 'next_retry_seconds=30\n'
        } > "$STATUS_DIR/training.last_failure"
        if ((rapid_failures >= 5)); then
            cp "$STATUS_DIR/training.last_failure" "$STATUS_DIR/pipeline.unrecoverable"
            sed -i '1s/.*/UNRECOVERABLE/' "$STATUS_DIR/pipeline.unrecoverable"
            rm -f "$STATUS_DIR/training.running" "$STATUS_DIR/pipeline.running"
            printf 'event=unrecoverable time=%s arm=%s phase=training\n' \
                "$(date --iso-8601=seconds)" "$ARM" >> "$EVENT_LOG"
            exit "$rc"
        fi
        sleep 30
    done

    audit_log="$LOG_DIR/training_audit.log"
    cd "$REPO_DIR"
    if ! python "$CUPID_ROOT/tools/audit_final_training.py" \
        --train-dir "$TRAIN_DIR" --expected-last-epoch "$EXPECTED_LAST_EPOCH" \
        > "$audit_log" 2>&1; then
        {
            printf 'UNRECOVERABLE\n'
            printf 'arm=%s\n' "$ARM"
            printf 'phase=training_audit\n'
            printf 'audit_log=%s\n' "$audit_log"
        } > "$STATUS_DIR/pipeline.unrecoverable"
        rm -f "$STATUS_DIR/training.running" "$STATUS_DIR/pipeline.running"
        printf 'event=unrecoverable time=%s arm=%s phase=training_audit\n' \
            "$(date --iso-8601=seconds)" "$ARM" >> "$EVENT_LOG"
        exit 1
    fi
    sha256sum "$TRAIN_DIR/checkpoints/latest.ckpt" > "$FROZEN_DIR/latest_checkpoint_sha256.txt"
    {
        printf 'PASS\n'
        printf 'arm=%s\n' "$ARM"
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'attempt=%d\n' "$attempt"
        printf 'restart_count=%d\n' "$((attempt - 1))"
        printf 'train_dir=%s\n' "$TRAIN_DIR"
    } > "$STATUS_DIR/training.pass"
    rm -f "$STATUS_DIR/training.running" "$STATUS_DIR/training.last_failure"
else
    audit_log="$LOG_DIR/training_audit.log"
    cd "$REPO_DIR"
    if ! python "$CUPID_ROOT/tools/audit_final_training.py" \
        --train-dir "$TRAIN_DIR" --expected-last-epoch "$EXPECTED_LAST_EPOCH" \
        > "$audit_log" 2>&1; then
        {
            printf 'UNRECOVERABLE\n'
            printf 'arm=%s\n' "$ARM"
            printf 'phase=baseline_training_audit\n'
            printf 'audit_log=%s\n' "$audit_log"
        } > "$STATUS_DIR/pipeline.unrecoverable"
        rm -f "$STATUS_DIR/pipeline.running"
        exit 1
    fi
fi

if [[ ! -f "$STATUS_DIR/evaluation.pass" ]]; then
    rollout_attempt=0
    rollout_rapid_failures=0
    while true; do
        rollout_attempt=$((rollout_attempt + 1))
        rollout_log=$(printf '%s/eval_attempt_%03d.log' "$LOG_DIR" "$rollout_attempt")
        rollout_start=$SECONDS
        overwrite=False
        if [[ -e "$EVAL_DIR" ]]; then
            overwrite=True
        fi
        {
            printf 'RUNNING\n'
            printf 'arm=%s\n' "$ARM"
            printf 'attempt=%d\n' "$rollout_attempt"
            printf 'physical_gpu=%s\n' "$PHYSICAL_GPU"
            printf 'eval_dir=%s\n' "$EVAL_DIR"
            printf 'test_start_seed=200000\n'
            printf 'num_episodes=100\n'
        } > "$STATUS_DIR/evaluation.running.tmp"
        mv "$STATUS_DIR/evaluation.running.tmp" "$STATUS_DIR/evaluation.running"

        eval_cmd=(
            python eval_save_episodes.py
            "--output_dir=$EVAL_DIR"
            "--train_dir=$TRAIN_DIR"
            --train_ckpt=latest
            --num_episodes=100
            --test_start_seed=200000
            "--overwrite=$overwrite"
            --device=cuda:0
        )
        {
            printf 'event=eval_attempt_start time=%s arm=%s attempt=%d command=' \
                "$(date --iso-8601=seconds)" "$ARM" "$rollout_attempt"
            printf '%q ' "${eval_cmd[@]}"
            printf '\n'
        } >> "$EVENT_LOG"

        cd "$REPO_DIR"
        "${eval_cmd[@]}" > "$rollout_log" 2>&1 &
        child_pid=$!
        printf '%s\n' "$child_pid" > "$CHILD_PID_FILE"
        set +e
        wait "$child_pid"
        rc=$?
        set -e
        child_pid=""
        rm -f "$CHILD_PID_FILE"
        wall_seconds=$((SECONDS - rollout_start))
        printf 'event=eval_attempt_end time=%s arm=%s attempt=%d rc=%d wall_seconds=%d\n' \
            "$(date --iso-8601=seconds)" "$ARM" "$rollout_attempt" "$rc" "$wall_seconds" >> "$EVENT_LOG"

        if [[ "$rc" -eq 0 ]]; then
            break
        fi
        if ((wall_seconds < 300)); then
            rollout_rapid_failures=$((rollout_rapid_failures + 1))
        else
            rollout_rapid_failures=0
        fi
        if ((rollout_rapid_failures >= 5)); then
            {
                printf 'UNRECOVERABLE\n'
                printf 'arm=%s\n' "$ARM"
                printf 'phase=evaluation\n'
                printf 'return_code=%d\n' "$rc"
                printf 'attempt=%d\n' "$rollout_attempt"
                printf 'log=%s\n' "$rollout_log"
            } > "$STATUS_DIR/pipeline.unrecoverable"
            rm -f "$STATUS_DIR/evaluation.running" "$STATUS_DIR/pipeline.running"
            printf 'event=unrecoverable time=%s arm=%s phase=evaluation\n' \
                "$(date --iso-8601=seconds)" "$ARM" >> "$EVENT_LOG"
            exit "$rc"
        fi
        sleep 30
    done

    audit_log="$LOG_DIR/evaluation_audit.log"
    cd "$REPO_DIR"
    if ! python "$CUPID_ROOT/tools/audit_rollout_pool.py" \
        --eval-dir "$EVAL_DIR" \
        --test-start-seed 200000 \
        --expected-episodes 100 \
        --output-dir "$MANIFEST_DIR/rollout100" \
        > "$audit_log" 2>&1; then
        {
            printf 'UNRECOVERABLE\n'
            printf 'arm=%s\n' "$ARM"
            printf 'phase=evaluation_audit\n'
            printf 'audit_log=%s\n' "$audit_log"
        } > "$STATUS_DIR/pipeline.unrecoverable"
        rm -f "$STATUS_DIR/evaluation.running" "$STATUS_DIR/pipeline.running"
        printf 'event=unrecoverable time=%s arm=%s phase=evaluation_audit\n' \
            "$(date --iso-8601=seconds)" "$ARM" >> "$EVENT_LOG"
        exit 1
    fi
    sha256sum "$TRAIN_DIR/checkpoints/latest.ckpt" > "$FROZEN_DIR/evaluated_checkpoint_sha256.txt"
    {
        printf 'PASS\n'
        printf 'arm=%s\n' "$ARM"
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'attempt=%d\n' "$rollout_attempt"
        printf 'wall_time_seconds=%d\n' "$wall_seconds"
        printf 'eval_dir=%s\n' "$EVAL_DIR"
    } > "$STATUS_DIR/evaluation.pass"
    rm -f "$STATUS_DIR/evaluation.running"
fi

{
    printf 'PASS\n'
    printf 'experiment=%s\n' "$EXPERIMENT_ID"
    printf 'arm=%s\n' "$ARM"
    printf 'mode=%s\n' "$MODE"
    printf 'physical_gpu=%s\n' "$PHYSICAL_GPU"
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'train_dir=%s\n' "$TRAIN_DIR"
    printf 'eval_dir=%s\n' "$EVAL_DIR"
} > "$STATUS_DIR/pipeline.pass"
rm -f "$STATUS_DIR/pipeline.running" "$STATUS_DIR/pipeline.unrecoverable"
printf 'event=pipeline_pass time=%s arm=%s\n' \
    "$(date --iso-8601=seconds)" "$ARM" >> "$EVENT_LOG"
