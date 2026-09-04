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
LOCK_FILE="$LOG_DIR/trak_analysis_pipeline.lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RESULT_DIR" "$FROZEN_DIR/influence"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '%s TRAK analysis pipeline already running\n' "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
    exit 2
fi

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
    } > "$STATUS_DIR/trak_analysis_pipeline.fail"
    rm -f "$STATUS_DIR/trak_analysis_pipeline.running"
    printf 'event=postprocess_fail time=%s rc=%d line=%s\n' \
        "$(date --iso-8601=seconds)" "$rc" "${BASH_LINENO[0]:-unknown}" >> "$PIPELINE_LOG"
    cleanup
    exit "$rc"
}
trap cleanup EXIT
trap fail ERR

source "$CUPID_ROOT/current_run.env"
source "$CUPID_ROOT/env.sh"
export TRAK_NAME=default_trak_results-proj_dim=4000-lambda_reg=0.0-num_ckpts=1-seed=0-loss_fn=square-num_timesteps=64
export TRAK_DIR="$EVAL_DIR/$TRAK_NAME"
export INFLUENCE_DIR="$RESULT_DIR/influence_layers"

test -f "$STATUS_DIR/rollout100_frozen.pass"
test -f "$STATUS_DIR/trak.pass"
test -f "$TRAK_DIR/scores/all_episodes.mmap"
rm -f "$STATUS_DIR/trak_analysis_pipeline.fail"
printf 'event=postprocess_resume time=%s source=completed_trak_scores\n' \
    "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"

if [[ ! -f "$STATUS_DIR/influence_layers.pass" ]]; then
    {
        printf 'INFLUENCE_EXPORT_RUNNING\n'
        printf 'pid=%s\n' "$$"
        printf 'trak_dir=%s\n' "$TRAK_DIR"
    } > "$STATUS_DIR/trak_analysis_pipeline.running"
    cd "$REPO_DIR"
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
fi

if [[ ! -f "$STATUS_DIR/demonstration_scores.pass" ]]; then
    {
        printf 'DEMONSTRATION_SCORES_RUNNING\n'
        printf 'pid=%s\n' "$$"
    } > "$STATUS_DIR/trak_analysis_pipeline.running"
    cd "$REPO_DIR"
    python eval_demonstration_scores.py \
        --exp_name=default_demonstration_scores-seed=0 \
        --eval_dir="$EVAL_DIR" --train_dir="$TRAIN_DIR" --train_ckpt=latest \
        --result_date=default --overwrite=0 --device=cpu --seed=0 \
        --use_half_precision=0 --compute_holdout=1 \
        --eval_offline_policy_loss=0 --eval_offline_action_diversity=0 \
        --eval_offline_state_diversity=0 --eval_online_state_similarity=0 \
        --eval_online_demo_score=0 --eval_online_trak_influence=1 \
        > "$LOG_DIR/demonstration_scores.log" 2>&1
    test -f "$EVAL_DIR/default_demonstration_scores-seed=0/online_trak_influence.pkl"
    touch "$STATUS_DIR/demonstration_scores.pass"
fi

{
    printf 'STABILITY_ANALYSIS_RUNNING\n'
    printf 'pid=%s\n' "$$"
} > "$STATUS_DIR/trak_analysis_pipeline.running"
cd "$REPO_DIR"
if [[ ! -f "$STATUS_DIR/random_analysis.pass" ]]; then
    python "$TOOL_DIR/analyze_rollout_budget_stability.py" \
        --train-dir="$TRAIN_DIR" --eval-dir="$EVAL_DIR" --train-ckpt=latest \
        --result-date=default --budgets 5 10 25 50 100 --repeats 20 \
        --seed 20260720 --top-fraction 0.20 --sampling random \
        --output-dir="$RESULT_DIR/rollout_budget_random" \
        > "$LOG_DIR/rollout_budget_random.log" 2>&1
    touch "$STATUS_DIR/random_analysis.pass"
fi

if [[ ! -f "$STATUS_DIR/stratified_analysis.pass" ]]; then
    python "$TOOL_DIR/analyze_rollout_budget_stability.py" \
        --train-dir="$TRAIN_DIR" --eval-dir="$EVAL_DIR" --train-ckpt=latest \
        --result-date=default --budgets 5 10 25 50 100 --repeats 20 \
        --seed 20260720 --top-fraction 0.20 --sampling stratified \
        --output-dir="$RESULT_DIR/rollout_budget_stratified" \
        > "$LOG_DIR/rollout_budget_stratified.log" 2>&1
    touch "$STATUS_DIR/stratified_analysis.pass"
fi

find "$TRAK_DIR" "$INFLUENCE_DIR" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$FROZEN_DIR/influence/all_files_sha256.txt"
du -sh "$TRAK_DIR" "$INFLUENCE_DIR" > "$FROZEN_DIR/influence/disk_usage.txt"
trak_wall_seconds=$(cat "$LOG_DIR/trak_wall_time_seconds.txt")
{
    printf 'PASS\n'
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'trak_wall_time_seconds=%s\n' "$trak_wall_seconds"
    printf 'trak_dir=%s\n' "$TRAK_DIR"
    printf 'influence_dir=%s\n' "$INFLUENCE_DIR"
} > "$STATUS_DIR/trak_analysis_pipeline.pass"
rm -f "$STATUS_DIR/trak_analysis_pipeline.running" "$STATUS_DIR/trak_analysis_pipeline.fail"
printf 'event=pipeline_pass time=%s resume=postprocess\n' "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
