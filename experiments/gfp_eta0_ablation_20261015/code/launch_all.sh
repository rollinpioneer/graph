#!/bin/bash
# Launch the 16 formal jobs (arms G and F), staggered 15 s, each pinned to its own 6 cores.
# Placement: G and F of the same task and seed share a GPU where possible; at most 3 jobs per GPU.
R=/home/xushijie3/work/gfp_eta0_ablation_20261015
L=$R/runs/launch_logs; mkdir -p $L
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv > $L/gpu_before.csv
while read ARM TN SEED GPU CPUS; do
  bash $R/scripts/launch_formal.sh $ARM $TN $SEED $GPU $CPUS >> $L/launch.log 2>&1
  sleep 15
done <<J
G 1 0 1 0-5
F 1 0 1 6-11
G 1 1 7 12-17
F 1 1 7 18-23
G 1 2 5 24-29
F 1 2 5 30-35
G 1 3 4 36-41
F 1 3 4 42-47
G 4 0 2 48-53
F 4 0 2 54-59
G 4 1 0 60-65
F 4 1 0 66-71
G 4 2 5 72-77
F 4 2 4 78-83
G 4 3 1 84-89
F 4 3 7 90-95
J
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv > $L/gpu_after.csv
echo ALL_LAUNCHED $(date -u +%FT%TZ) >> $L/launch.log
