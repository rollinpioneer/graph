#!/usr/bin/env bash

set -uo pipefail

CUPID_ROOT=/home/xushijie/CUPID
source "$CUPID_ROOT/stage3_transport_env.sh"
LOG_DIR="$CUPID_ROOT/logs/stage3_transport"
STATUS_DIR="$CUPID_ROOT/status/stage3_transport"
RESULT_DIR="$CUPID_ROOT/results/stage3_transport"
FROZEN_DIR="$CUPID_ROOT/frozen/stage3_transport"
MANIFEST_DIR="$CUPID_ROOT/manifests/stage3_transport"
PID_DIR="$CUPID_ROOT/pids/stage3_transport"
TOOL_DIR="$CUPID_ROOT/tools"
PIPELINE_LOG="$LOG_DIR/analysis_pipeline.log"
PID_FILE="$PID_DIR/analysis_pipeline.pid"
CHILD_PID_FILE="$PID_DIR/analysis_child.pid"
LOCK_FILE="$PID_DIR/analysis_pipeline.lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RESULT_DIR" "$FROZEN_DIR/influence" "$PID_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '%s TRAK analysis pipeline already running\n' "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
    exit 2
fi

printf '%s\n' "$$" > "$PID_FILE"
child_pid=""
cleanup() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE" "$CHILD_PID_FILE"
}
fail() {
    rc=$?
    {
        printf 'FAIL\n'
        printf 'time=%s\n' "$(date --iso-8601=seconds)"
        printf 'return_code=%d\n' "$rc"
        printf 'line=%s\n' "${BASH_LINENO[0]:-unknown}"
        printf 'log=%s\n' "$PIPELINE_LOG"
    } > "$STATUS_DIR/analysis_pipeline.fail"
    rm -f "$STATUS_DIR/analysis_pipeline.running"
    printf 'event=pipeline_fail time=%s rc=%d line=%s\n' \
        "$(date --iso-8601=seconds)" "$rc" "${BASH_LINENO[0]:-unknown}" >> "$PIPELINE_LOG"
    cleanup
    exit "$rc"
}
trap cleanup EXIT
trap fail ERR

export TRAK_NAME=default_trak_results-proj_dim=4000-lambda_reg=0.0-num_ckpts=1-seed=0-loss_fn=square-num_timesteps=64
export TRAIN_DIR="$STAGE3_TRANSPORT_TRAIN_DIR"
export EVAL_DIR="$STAGE3_TRANSPORT_EVAL_DIR"
export ROLLOUT_CKPT="$TRAIN_DIR/checkpoints/latest.ckpt"
export TRAK_DIR="$EVAL_DIR/$TRAK_NAME"
export INFLUENCE_DIR="$RESULT_DIR/influence_layers"

{
    printf 'WAITING_FOR_ROLLOUT_PASS\n'
    printf 'pid=%s\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
} > "$STATUS_DIR/analysis_pipeline.running"
while [[ ! -f "$STATUS_DIR/rollout100.pass" ]]; do
    if [[ -f "$STATUS_DIR/rollout_pipeline.fail" ]]; then
        printf 'rollout pipeline failed before analysis gate\n' >&2
        false
    fi
    sleep 30
done

test -f "$ROLLOUT_CKPT"
test -d "$EVAL_DIR/episodes"
test ! -e "$TRAK_DIR"
test ! -e "$INFLUENCE_DIR"

{
    printf 'WAITING_FOR_TRAK_RESOURCE_GATE\n'
    printf 'pid=%s\n' "$$"
    printf 'required_gpu_free_mib=28000\n'
} > "$STATUS_DIR/analysis_pipeline.running"
while true; do
    free_mib=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits \
        2>/dev/null | tr -d ' ' || true)
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 28000 )); then
        break
    fi
    printf 'event=resource_wait time=%s gpu_free_mib=%s\n' \
        "$(date --iso-8601=seconds)" "${free_mib:-unavailable}" >> "$PIPELINE_LOG"
    sleep 60
done

{
    printf 'TRAK_SMOKE_RUNNING\n'
    printf 'pid=%s\n' "$$"
    printf 'physical_gpu=1\n'
    printf 'proj_dim=4000\n'
    printf 'proj_max_batch_size=32\n'
    printf 'num_timesteps=64\n'
    printf 'batch_size=128\n'
} > "$STATUS_DIR/analysis_pipeline.running"
smoke_dir="$RESULT_DIR/trak_smoke"
test ! -e "$smoke_dir"
cd "$REPO_DIR"
python "$TOOL_DIR/stage3_transport_trak_smoke.py" \
    --checkpoint "$ROLLOUT_CKPT" --output-dir "$smoke_dir" \
    --batch-size 128 --source-samples 4096 \
    --proj-dim 4000 --proj-max-batch-size 32 \
    --num-timesteps 64 --seed 0 > "$LOG_DIR/trak_smoke.log" 2>&1 &
child_pid=$!
printf '%s\n' "$child_pid" > "$CHILD_PID_FILE"
wait "$child_pid"
child_pid=""
rm -f "$CHILD_PID_FILE"
test -f "$smoke_dir/result.json"
python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "PASS"; assert r["score_shape"] == [4096,128]; print("TRANSPORT TRAK SMOKE PASS")' \
    "$smoke_dir/result.json" > "$LOG_DIR/trak_smoke_gate.log"
python "$TOOL_DIR/stage3_transport_trak_feasibility.py" \
    --smoke "$smoke_dir/result.json" \
    --rollout-summary "$MANIFEST_DIR/rollout100/rollout_summary.json" \
    --disk-root "$CUPID_ROOT" \
    --output "$smoke_dir/feasibility.json" \
    > "$LOG_DIR/trak_feasibility.log" 2>&1
touch "$STATUS_DIR/trak_smoke.pass"

# Recheck immediately before the formal job; another process may have claimed memory.
free_mib=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits \
    | tr -d ' ')
(( free_mib >= 28000 ))

{
    printf 'TRAK_RUNNING\n'
    printf 'pid=%s\n' "$$"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'physical_gpu=1\n'
    printf 'proj_dim=4000\n'
    printf 'proj_max_batch_size=32\n'
    printf 'num_timesteps=64\n'
    printf 'trak_dir=%s\n' "$TRAK_DIR"
} > "$STATUS_DIR/analysis_pipeline.running"
rm -f "$STATUS_DIR/analysis_pipeline.fail"
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
    child_pid=""
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
            "$STATUS_DIR/analysis_pipeline.running"
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
} > "$STATUS_DIR/analysis_pipeline.running"
python "$TOOL_DIR/export_influence_layers.py" \
    --train-dir "$TRAIN_DIR" \
    --eval-dir "$EVAL_DIR" \
    --train-ckpt latest \
    --trak-dir "$TRAK_DIR" \
    --split-manifest "$MANIFEST_DIR/training_split/dataset_split.json" \
    --rollout-manifest "$MANIFEST_DIR/rollout100/episode_manifest.csv" \
    --output-dir "$INFLUENCE_DIR" \
    > "$LOG_DIR/export_influence_layers.log" 2>&1
touch "$STATUS_DIR/influence_layers.pass"

{
    printf 'DEMONSTRATION_SCORES_RUNNING\n'
    printf 'pid=%s\n' "$$"
} > "$STATUS_DIR/analysis_pipeline.running"
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
    printf 'STABILITY_DIAGNOSIS_RUNNING\n'
    printf 'pid=%s\n' "$$"
} > "$STATUS_DIR/analysis_pipeline.running"
diagnosis_dir="$RESULT_DIR/stability_diagnosis"
python "$TOOL_DIR/stage3_transport_stability_diagnosis.py" \
    --matrix "$INFLUENCE_DIR/rollout_demo_influence.npy" \
    --manifest "$MANIFEST_DIR/rollout100/episode_manifest.csv" \
    --split-json "$MANIFEST_DIR/training_split/dataset_split.json" \
    --official-scores "$INFLUENCE_DIR/final_demo_scores.csv" \
    --output-dir "$diagnosis_dir" \
    --budgets 10 25 50 75 \
    --proportions 0.05 0.10 0.20 0.30 \
    --subsample-repeats 1000 --bootstrap-reps 300 \
    --cross-pool-repeats 100 --selection-size 50 --seed 20260723 \
    > "$LOG_DIR/stability_diagnosis.log" 2>&1
decision=$(tr -d '\r\n' < "$diagnosis_dir/stage2b_decision.txt")
if [[ "$decision" == "PASS_STABILITY_WEIGHTED_CORE_CANDIDATE" ]]; then
    {
        printf 'STRONG_PASS\n'
        printf 'decision=%s\n' "$decision"
        printf 'filtered_retraining_authorized_for_planning=true\n'
    } > "$STATUS_DIR/stability_diagnosis.strong_pass"
else
    {
        printf 'STOP\n'
        printf 'decision=%s\n' "$decision"
        printf 'filtered_retraining_authorized=false\n'
    } > "$STATUS_DIR/stability_diagnosis.stop"
fi
touch "$STATUS_DIR/stability_diagnosis.pass"

find "$smoke_dir" "$TRAK_DIR" "$INFLUENCE_DIR" "$diagnosis_dir" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$FROZEN_DIR/influence/all_files_sha256.txt"
du -sh "$TRAK_DIR" "$INFLUENCE_DIR" "$diagnosis_dir" \
    > "$FROZEN_DIR/influence/disk_usage.txt"

{
    printf 'PASS\n'
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'trak_wall_time_seconds=%d\n' "$trak_wall_seconds"
    printf 'trak_dir=%s\n' "$TRAK_DIR"
    printf 'influence_dir=%s\n' "$INFLUENCE_DIR"
    printf 'stability_decision=%s\n' "$decision"
} > "$STATUS_DIR/analysis_pipeline.pass"
rm -f "$STATUS_DIR/analysis_pipeline.running" "$STATUS_DIR/analysis_pipeline.fail"
printf 'event=pipeline_pass time=%s\n' "$(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
