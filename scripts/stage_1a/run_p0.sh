#!/bin/bash
set -euo pipefail
ROOT=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$ROOT"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PYTHONPATH="$ROOT/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
KEYFILE=/home/__compress_data/xushijie/DASHSCOPE_API_KEY
if [ -z "${DASHSCOPE_API_KEY:-}" ] && [ -f "$KEYFILE" ]; then
  DASHSCOPE_API_KEY=$(tr -d '\r\n ' < "$KEYFILE")
  export DASHSCOPE_API_KEY
fi
unset DASHSCOPE_API_KEY_FILE
PY="$ROOT/.venv-stage0a/bin/python"
mkdir -p experiments/part_1_smoke/stage_1a_p0
exec "$PY" -u -m cp_disr --root "$ROOT" stage-1a-preflight --all-p0 --gpu 0
