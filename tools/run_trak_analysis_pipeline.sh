#!/usr/bin/env bash

set -uo pipefail

CUPID_ROOT=/home/xushijie/CUPID
LOG_DIR="$CUPID_ROOT/logs"
STATUS_DIR="$CUPID_ROOT/status"
RESULT_DIR="$CUPID_ROOT/results"
FROZEN_DIR="$CUPID_ROOT/frozen"
TOOL_DIR="$CUPID_ROOT/tools"
PIPELINE_LOG="$LOG_DIR/trak_analysis_pipeline.log"
PID_FILE="$LOG_DIR/trak_analysis_pipeline.pid"
CHILD_PID_FILE="$LOG_DIR/trak_analysis_child.pid"
LOCK_FILE="$LOG_DIR/trak_analysis_pipeline.lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RESULT_DIR" "$FROZEN_DIR/influence"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '%s TRAK analysis pipeline already running\n' "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
    exit 2
fi

printf '%s\n' "$$" > "$PID_FILE"
cleanup() { rm -f "$PID_FILE" "$CHILD_PID_FILE"; }
fail() {
    rc=$?
    {
        printf 'FAIL\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'return_code=%d\n' "$rc"
        printf 'line=%s\n' "${BASH_LINENO[0]:-unknown}"
        printf 'log=%s\n' "$PIPELINE_LOG"
    } > "$STATUS_DIR/trak_analysis_pipeline.fail"
    rm -f "$STATUS_DIR/trak_analysis_pipeline.running"
    printf 'event=pipeline_fail time=%s rc=%d line=%s\n' \
        "$(date --iso-8601=seconds)" "$rc" "${BASH_LINENO[0]:-unknown}" >> "$PIPELINE_LOG"
    cleanup
    exit "$rc"
}
trap cleanup EXIT
trap fail ERR

source "$CUPID_ROOT/current_run.env"
source "$CUPID_ROOT/env.sh"
export CUDA_VISIBLE_DEVICES=1
export TRAK_NAME=default_trak_results-proj_dim=4000-lambda_reg=0.0-num_ckpts=1-seed=0-loss_fn=square-num_timesteps=64
export TRAK_DIR="$EVAL_DIR/$TRAK_NAME"
export INFLUENCE_DIR="$RESULT_DIR/influence_layers"

test -f "$STATUS_DIR/rollout100_frozen.pass"
test -f "$ROLLOUT_CKPT"
test -d "$EVAL_DIR/episodes"
test ! -e "$TRAK_DIR"
test ! -e "$INFLUENCE_DIR"
test ! -e "$RESULT_DIR/rollout_budget_random"
test ! -e "$RESULT_DIR/rollout_budget_stratified"

{
    printf 'TRAK_RUNNING\n'
    printf 'pid=%s\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'physical_gpu=1\n'
    printf 'proj_dim=4000\n'
    printf 'proj_max_batch_size=32\n'
    printf 'num_timesteps=64\n'
    printf 'trak_dir=%s\n' "$TRAK_DIR"
} > "$STATUS_DIR/trak_analysis_pipeline.running"
rm -f "$STATUS_DIR/trak_analysis_pipeline.fail"
printf 'event=trak_start time=%s proj_max_batch_size=32\n' \
    "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
date --iso-8601=seconds > "$LOG_DIR/trak_start.txt"
trak_start=$SECONDS

run_trak() {
    proj_batch=$1
    log_path=$2
    cd "$REPO_DIR"
    python train_trak_diffusion.py \
        --model_id=0 \
        --exp_name="$TRAK_NAME" \
        --eval_dir="$EVAL_DIR" \
        --train_dir="$TRAIN_DIR" \
        --train_ckpt=latest \
        --model_keys="model." \
        --modelout_fn=DiffusionLowdimFunctionalModelOutput \
        --gradient_co=DiffusionLowdimFunctionalGradientComputer \
        --proj_dim=4000 \
        --proj_max_batch_size="$proj_batch" \
        --lambda_reg=0.0 \
        --use_half_precision=0 \
        --loss_fn=square \
        --num_timesteps=64 \
        --batch_size=128 \
        --device=cuda:0 \
        --overwrite=0 \
        --seed=0 \
        --featurize_holdout=1 \
        --finalize_scores=1 \
        > "$log_path" 2>&1 &
    child_pid=$!
    printf '%s\n' "$child_pid" > "$CHILD_PID_FILE"
    wait "$child_pid"
    rc=$?
    rm -f "$CHILD_PID_FILE"
    return "$rc"
}

if run_trak 32 "$LOG_DIR/trak.log"; then
    trak_rc=0
else
    trak_rc=$?
fi

if [[ "$trak_rc" -ne 0 ]]; then
    if rg -qi 'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory' "$LOG_DIR/trak.log"; then
        failed_dir="${TRAK_DIR}_failed_proj_batch32_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$failed_dir" ]]
        [[ ! -e "$TRAK_DIR" ]] || mv "$TRAK_DIR" "$failed_dir"
        printf 'event=trak_oom_retry time=%s archived=%s proj_max_batch_size=16\n' \
            "$(date --iso-8601=seconds)" "$failed_dir" >> "$PIPELINE_LOG"
        sed -i 's/proj_max_batch_size=32/proj_max_batch_size=16/' \
            "$STATUS_DIR/trak_analysis_pipeline.running"
        run_trak 16 "$LOG_DIR/trak_retry_batch16.log"
    else
        printf 'TRAK failed for a non-OOM reason\n' >&2
        false
    fi
fi

trak_wall_seconds=$((SECONDS - trak_start))
date --iso-8601=seconds > "$LOG_DIR/trak_end.txt"
printf '%s\n' "$trak_wall_seconds" > "$LOG_DIR/trak_wall_time_seconds.txt"
test -f "$TRAK_DIR/scores/all_episodes.mmap"
find "$TRAK_DIR" -maxdepth 3 -type f -printf '%s %p\n' | sort \
    > "$LOG_DIR/trak_files.txt"
touch "$STATUS_DIR/trak.pass"
printf 'event=trak_pass time=%s wall_time_seconds=%d\n' \
    "$(date --iso-8601=seconds)" "$trak_wall_seconds" >> "$PIPELINE_LOG"

{
    printf 'INFLUENCE_EXPORT_RUNNING\n'
    printf 'pid=%s\n' "$$"
    printf 'trak_dir=%s\n' "$TRAK_DIR"
} > "$STATUS_DIR/trak_analysis_pipeline.running"
python "$TOOL_DIR/export_influence_layers.py" \
    --train-dir "$TRAIN_DIR" \
    --eval-dir "$EVAL_DIR" \
    --train-ckpt latest \
    --trak-dir "$TRAK_DIR" \
    --split-manifest "$CUPID_ROOT/manifests/training_split/dataset_split.json" \
    --rollout-manifest "$CUPID_ROOT/manifests/rollout100/episode_manifest.csv" \
    --output-dir "$INFLUENCE_DIR" \
    > "$LOG_DIR/export_influence_layers.log" 2>&1
touch "$STATUS_DIR/influence_layers.pass"

{
    printf 'DEMONSTRATION_SCORES_RUNNING\n'
    printf 'pid=%s\n' "$$"
} > "$STATUS_DIR/trak_analysis_pipeline.running"
cd "$REPO_DIR"
python eval_demonstration_scores.py \
    --exp_name=default_demonstration_scores-seed=0 \
    --eval_dir="$EVAL_DIR" \
    --train_dir="$TRAIN_DIR" \
    --train_ckpt=latest \
    --result_date=default \
    --overwrite=0 \
    --device=cpu \
    --seed=0 \
    --use_half_precision=0 \
    --compute_holdout=1 \
    --eval_offline_policy_loss=0 \
    --eval_offline_action_diversity=0 \
    --eval_offline_state_diversity=0 \
    --eval_online_state_similarity=0 \
    --eval_online_demo_score=0 \
    --eval_online_trak_influence=1 \
    > "$LOG_DIR/demonstration_scores.log" 2>&1
test -f "$EVAL_DIR/default_demonstration_scores-seed=0/online_trak_influence.pkl"
touch "$STATUS_DIR/demonstration_scores.pass"

{
    printf 'STABILITY_ANALYSIS_RUNNING\n'
    printf 'pid=%s\n' "$$"
} > "$STATUS_DIR/trak_analysis_pipeline.running"
cd "$REPO_DIR"
python "$TOOL_DIR/analyze_rollout_budget_stability.py" \
    --train-dir="$TRAIN_DIR" --eval-dir="$EVAL_DIR" --train-ckpt=latest \
    --result-date=default --budgets 5 10 25 50 100 --repeats 20 \
    --seed 20260720 --top-fraction 0.20 --sampling random \
    --output-dir="$RESULT_DIR/rollout_budget_random" \
    > "$LOG_DIR/rollout_budget_random.log" 2>&1
touch "$STATUS_DIR/random_analysis.pass"

python "$TOOL_DIR/analyze_rollout_budget_stability.py" \
    --train-dir="$TRAIN_DIR" --eval-dir="$EVAL_DIR" --train-ckpt=latest \
    --result-date=default --budgets 5 10 25 50 100 --repeats 20 \
    --seed 20260720 --top-fraction 0.20 --sampling stratified \
    --output-dir="$RESULT_DIR/rollout_budget_stratified" \
    > "$LOG_DIR/rollout_budget_stratified.log" 2>&1
touch "$STATUS_DIR/stratified_analysis.pass"

find "$TRAK_DIR" "$INFLUENCE_DIR" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$FROZEN_DIR/influence/all_files_sha256.txt"
du -sh "$TRAK_DIR" "$INFLUENCE_DIR" \
    > "$FROZEN_DIR/influence/disk_usage.txt"

{
    printf 'PASS\n'
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'trak_wall_time_seconds=%d\n' "$trak_wall_seconds"
    printf 'trak_dir=%s\n' "$TRAK_DIR"
    printf 'influence_dir=%s\n' "$INFLUENCE_DIR"
} > "$STATUS_DIR/trak_analysis_pipeline.pass"
rm -f "$STATUS_DIR/trak_analysis_pipeline.running" "$STATUS_DIR/trak_analysis_pipeline.fail"
printf 'event=pipeline_pass time=%s\n' "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
