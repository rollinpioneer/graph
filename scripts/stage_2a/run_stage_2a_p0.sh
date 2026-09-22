#!/usr/bin/env bash
set -euo pipefail
REPO=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$REPO"
export MUJOCO_GL=egl
if [ -z "${DASHSCOPE_API_KEY:-}" ] && [ -f /home/__compress_data/xushijie/DASHSCOPE_API_KEY ]; then
  DASHSCOPE_API_KEY="$(tr -d '\r\n' < /home/__compress_data/xushijie/DASHSCOPE_API_KEY)"
  export DASHSCOPE_API_KEY
fi
GPU_SEL="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$2); print $2+0, $1+0}' | sort -nr | awk '{print $2; exit}')"
export CUDA_VISIBLE_DEVICES="${GPU_SEL}"
export PYTHONPATH="$REPO/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
PYBIN="$REPO/.venv-stage0a/bin/python"
mkdir -p experiments/part_2_exploration/stage_2a_p0
echo "selected_physical_gpu=$GPU_SEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee experiments/part_2_exploration/stage_2a_p0/launch_gpu.txt
exec "$PYBIN" -m cp_disr stage-2a-p0 --gpu 0
