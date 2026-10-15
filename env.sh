#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
export MKL_INTERFACE_LAYER=""
conda activate cupid
export PYTHONNOUSERSITE=1
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib64/nvidia"
export WANDB_MODE=offline
export WANDB_SILENT=true
export CUPID_ROOT=/home/xushijie/CUPID
export REPO_DIR=$CUPID_ROOT/repo
export LOG_DIR=$CUPID_ROOT/logs
export RESULT_DIR=$CUPID_ROOT/results
export TOOL_DIR=$CUPID_ROOT/tools
export STATUS_DIR=$CUPID_ROOT/status
