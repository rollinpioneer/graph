#!/bin/bash
set -euo pipefail
ROOT=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$ROOT"
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
unset DASHSCOPE_API_KEY || true
PY="$ROOT/.venv-stage0a/bin/python"
# pick a GPU with most free memory for the gate env
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPU
echo "gate gpu $GPU"
"$PY" -m cp_disr --root "$ROOT" stage-2a-startup-gate --gpu 0
