#!/usr/bin/env bash
set -euo pipefail
REPO=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$REPO"
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1,$2}' | sort -k2 -nr | head -1 | awk '{print $1}')
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export PYTHONPATH="$REPO/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
PY="$REPO/.venv-stage0a/bin/python"
echo "Using GPU $GPU"
echo "python $PY"
exec "$PY" -m cp_disr stage-1a-run --gpu 0 "$@"
