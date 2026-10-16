#!/bin/bash
# Stage-2 confirmation (CARD_CONFIRM_SCRATCH_20261016.md). Usage: launch_scratch.sh ARM TASKNUM SEED GPU CPUSET
# ARM: I (IQL-inspired GFP, expectile 0.7, from scratch) | G (official GFP, same command as scripts/launch_formal.sh).
# Same official main.py arguments as the formal runs. Refuses to overwrite an existing run directory.
set -u
R=/home/xushijie3/work/gfp_eta0_ablation_20261015
PY=/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python
S=$R/analysis_p0p1/confirm_scratch
ARM=$1; TN=$2; SEED=$3; GPU=$4; CPUS=$5
TASK=humanoidmaze-medium-navigate-singletask-task${TN}-v0
case $ARM in
  I) E="$S/code/entry_iql.py"; A="agent=gfp eval_flow_policy=false +agent.expectile=0.7"; D=$S/runs/I_task${TN}_s${SEED};;
  G) E="$R/scripts/entry_spawn.py"; A="agent=gfp eval_flow_policy=false"; D=$R/runs/formal/G_task${TN}_s${SEED};;
  *) echo "bad arm"; exit 2;;
esac
if [ -e "$D" ]; then echo "exists: $D"; exit 3; fi
mkdir -p "$D"
CMD="$PY $E main.py $A env_name=$TASK seed=$SEED offline_steps=1000000 online_steps=0 is_final=true log_on_wandb=false eval_episodes=100 n_eval_envs=10 eval_interval=100000 log_interval=5000 save_interval=100000 save_last_checkpoint=true save_dir=$D/out hydra.run.dir=$D/hydra"
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
