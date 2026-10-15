#!/bin/bash
# 2k-update timing/smoke for arms G (official GFP) and F (eta_temperature=0), task1 seed 0, eval off, run one after the other.
# Same settings as the L1-L1.6 timing_2k.sh.
R=/home/xushijie3/work/gfp_eta0_ablation_20261015
PY=/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python
GPU=$1
TASK=humanoidmaze-medium-navigate-singletask-task1-v0
cd $R/source/gfp
mkdir -p $R/runs/timing
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv -i $GPU > $R/runs/timing/gpu_before.csv
for ARM in G F; do
  case $ARM in
    G) A="agent=gfp eval_flow_policy=false";;
    F) A="agent=gfp agent.eta_temperature=0 eval_flow_policy=false";;
  esac
  D=$R/runs/timing/$ARM
  mkdir -p $D
  env -u LD_LIBRARY_PATH -u JAX_DEFAULT_MATMUL_PRECISION -u JAX_PLATFORMS HOME=$R/home CUDA_VISIBLE_DEVICES=$GPU \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_MODE=disabled PYTHONUNBUFFERED=1 MPLCONFIGDIR=$R/cache/matplotlib \
    /usr/bin/time -v -o $D/time_v.txt $PY $R/scripts/entry_spawn.py main.py \
    $A env_name=$TASK seed=0 \
    offline_steps=2000 online_steps=0 log_on_wandb=false log_interval=500 eval_interval=0 eval_episodes=10 n_eval_envs=10 \
    save_dir=$D/out hydra.run.dir=$D/hydra > $D/stdout.log 2>&1
  echo "$ARM exit=$?" >> $R/runs/timing/status.txt
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i $GPU >> $R/runs/timing/gpu_after.csv
echo DONE >> $R/runs/timing/status.txt
