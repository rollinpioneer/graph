#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
export MUJOCO_GL=egl
if [[ -f /home/__compress_data/xushijie/DASHSCOPE_API_KEY ]]; then
  export DASHSCOPE_API_KEY="$(python3 - <<'PY'
from pathlib import Path
lines=[ln.strip() for ln in Path("/home/__compress_data/xushijie/DASHSCOPE_API_KEY").read_text(encoding="utf-8").splitlines()]
print(next((x for x in lines if x and not x.startswith("#")), ""), end="")
PY
)"
fi
echo "key_present=${DASHSCOPE_API_KEY:+yes}"
PY="$ROOT/.venv-stage0a/bin/python"
LOGDIR="$ROOT/runs/stage_1a/final_test_id/logs"
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CP_DISR_GPU=0
"$PY" "$ROOT/tools/plan_v11_stage1a_final_eval.py" --phase freeze
"$PY" "$ROOT/tools/plan_v11_stage1a_final_eval.py" --phase register
"$PY" "$ROOT/tools/plan_v11_stage1a_final_eval.py" --phase materialize --gpu 0
# two-method eval in parallel; each process sees a single GPU as device 0
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/tools/plan_v11_stage1a_final_eval.py" --phase eval --method B2 --gpu 0 >"$LOGDIR/eval_B2.log" 2>&1 &
PID_B2=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/tools/plan_v11_stage1a_final_eval.py" --phase eval --method Full --gpu 0 >"$LOGDIR/eval_Full.log" 2>&1 &
PID_FULL=$!
echo "eval pids B2=$PID_B2 Full=$PID_FULL"
wait "$PID_B2"
ST_B2=$?
wait "$PID_FULL"
ST_FULL=$?
echo "eval exit B2=$ST_B2 Full=$ST_FULL"
if [[ "$ST_B2" -ne 0 || "$ST_FULL" -ne 0 ]]; then
  echo "eval worker failed" >&2
  exit 2
fi
"$PY" "$ROOT/tools/plan_v11_stage1a_final_eval.py" --phase report
