#!/bin/bash
# Launch the remaining 6 formal jobs after the first launcher was stopped (task4 seed 1 pair moved from GPU 0 to GPU 6).
R=/home/xushijie3/work/gfp_eta0_ablation_20261015
L=$R/runs/launch_logs
while read ARM TN SEED GPU CPUS; do
  bash $R/scripts/launch_formal.sh $ARM $TN $SEED $GPU $CPUS >> $L/launch.log 2>&1
  sleep 15
done <<J
G 4 1 6 60-65
F 4 1 6 66-71
G 4 2 5 72-77
F 4 2 4 78-83
G 4 3 1 84-89
F 4 3 7 90-95
J
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv > $L/gpu_after.csv
echo ALL_LAUNCHED $(date -u +%FT%TZ) >> $L/launch.log
