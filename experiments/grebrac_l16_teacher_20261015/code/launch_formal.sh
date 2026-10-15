#!/bin/bash
# Launch one formal GREBRAC-L16-TEACHER job. Usage: launch_formal.sh ARM TASKNUM SEED GPU CPUSET
# ARM: T (GFP structure with a deterministic teacher). The ReBRAC control (R) comes from GREBRAC-L1-V1.
# Refuses to overwrite an existing run directory.
set -u
R=/home/xushijie3/work/grebrac_l16_teacher_20261015
PY=/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python
ARM=$1; TN=$2; SEED=$3; GPU=$4; CPUS=$5
TASK=humanoidmaze-medium-navigate-singletask-task${TN}-v0
case $ARM in
  T) A="agent=rebrac agent.guidance=true agent.eta_temperature=0.001 agent.guidance_ref=teacher agent.guidance_normalize=false";;
  *) echo "bad arm"; exit 2;;
esac
D=$R/runs/formal/${ARM}_task${TN}_s${SEED}
if [ -e "$D" ]; then echo "exists: $D"; exit 3; fi
mkdir -p "$D"
CMD="$PY $R/scripts/entry_spawn.py main.py $A env_name=$TASK seed=$SEED offline_steps=1000000 online_steps=0 is_final=true log_on_wandb=false eval_episodes=100 n_eval_envs=10 eval_interval=100000 log_interval=5000 save_interval=100000 save_last_checkpoint=true save_dir=$D/out hydra.run.dir=$D/hydra"
cat > "$D/job.json" <<J
{"arm": "$ARM", "task": "$TASK", "seed": $SEED, "gpu_index": $GPU,
 "gpu_uuid": "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)", "cpus": "$CPUS",
 "launched_utc": "$(date -u +%FT%TZ)", "command": "$CMD"}
J
cd $R/source/gfp
(
  env -u LD_LIBRARY_PATH -u JAX_DEFAULT_MATMUL_PRECISION -u JAX_PLATFORMS HOME=$R/home CUDA_VISIBLE_DEVICES=$GPU \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_MODE=disabled PYTHONUNBUFFERED=1 MPLCONFIGDIR=$R/cache/matplotlib \
    taskset -c $CPUS /usr/bin/time -v -o $D/time_v.txt $CMD > $D/stdout.log 2>&1
  echo "{\"exit\": $?, \"finished_utc\": \"$(date -u +%FT%TZ)\"}" > $D/exit.json
) < /dev/null > /dev/null 2>&1 &
echo "launched $D"
