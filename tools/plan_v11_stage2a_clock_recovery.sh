#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$ROOT"
ALLOW="$ROOT/runs/stage_2a/orchestrator/CLOCK_RECOVERY_ALLOW.json"
CLOCK_STOP="$ROOT/runs/stage_2a/orchestrator/STOP_CLOCK_INTEGRITY.json"
HIST_STOP="$ROOT/experiments/part_2_exploration/stage_2a/orchestrator/STOP_SUPERSEDED_BY_PLAN_V1_1.json"
if [[ ! -f "$CLOCK_STOP" ]]; then echo MISSING_STOP_CLOCK_INTEGRITY; exit 75; fi
if [[ ! -f "$HIST_STOP" ]]; then echo MISSING_HISTORICAL_STAGE2A_STOP; exit 75; fi
if [[ ! -f "$ALLOW" ]]; then echo MISSING_CLOCK_RECOVERY_ALLOW; exit 75; fi
export PYTHONPATH="$ROOT/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$ROOT/.venv-stage0a/bin/python"
LOGDIR="$ROOT/runs/stage_2a/logs"
mkdir -p "$LOGDIR"
STAMP=$(python3 -c 'import json; from pathlib import Path; print(json.loads(Path("runs/stage_2a/orchestrator/CLOCK_RECOVERY_ALLOW.json").read_text())["stamp"])')
SHA=$(python3 -c 'import json; from pathlib import Path; print(json.loads(Path("runs/stage_2a/orchestrator/CLOCK_RECOVERY_ALLOW.json").read_text())["configsha8"])')
echo "recovery stamp=$STAMP configsha8=$SHA"
run_cli() {
  local gpu="$1"; shift
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m cp_disr.cli --root "$ROOT" stage-2a-v11-run --gpu 0 --stamp "$STAMP" --configsha "$SHA" "$@"
}
phase="${1:-startup-and-two}"
if [[ "$phase" == "startup" || "$phase" == "startup-and-two" ]]; then
  echo "=== startup T_B ==="
  run_cli 0 --phase startup --task T_B | tee "$LOGDIR/recovery_startup_T_B.log"
  grep -q STARTUP_GATES_PASS "$LOGDIR/recovery_startup_T_B.log"
  echo "=== startup T_C ==="
  run_cli 0 --phase startup --task T_C | tee "$LOGDIR/recovery_startup_T_C.log"
  grep -q STARTUP_GATES_PASS "$LOGDIR/recovery_startup_T_C.log"
fi
if [[ "$phase" == "two" || "$phase" == "startup-and-two" ]]; then
  mapfile -t gpus < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' | sort -nr | awk '{print $2}' | head -n 2)
  echo "using gpus ${gpus[*]}"
  nohup env CUDA_VISIBLE_DEVICES="${gpus[0]}" "$PY" -m cp_disr.cli --root "$ROOT" stage-2a-v11-run --gpu 0 --stamp "$STAMP" --configsha "$SHA" --phase train --task T_B --method B0 > "$LOGDIR/recovery_train_T_B_B0.log" 2>&1 &
  echo $! > "$LOGDIR/recovery_train_T_B_B0.pid"
  nohup env CUDA_VISIBLE_DEVICES="${gpus[1]:-${gpus[0]}}" "$PY" -m cp_disr.cli --root "$ROOT" stage-2a-v11-run --gpu 0 --stamp "$STAMP" --configsha "$SHA" --phase train --task T_B --method B1-K > "$LOGDIR/recovery_train_T_B_B1-K.log" 2>&1 &
  echo $! > "$LOGDIR/recovery_train_T_B_B1-K.pid"
  echo "started T_B B0 pid=$(cat "$LOGDIR/recovery_train_T_B_B0.pid") T_B B1-K pid=$(cat "$LOGDIR/recovery_train_T_B_B1-K.pid")"
fi
