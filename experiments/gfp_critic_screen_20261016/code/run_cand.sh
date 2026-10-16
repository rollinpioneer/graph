#!/bin/bash
# usage: run_cand.sh RUN FORK ARM PARAM GPU CORES -- 200k candidate fork, then frozen eval at +100k/+150k/+200k
# (100 episodes, eval seed 4242, official sample_actions), as run_x3.sh
R=/home/xushijie3/work/gfp_eta0_ablation_20261015; A=$R/analysis_p0p1; X=$A/critic_screen
PY=/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python
RUN=$1; FORK=$2; ARM=$3; PARAM=$4; GPU=$5; CORES=$6; TAG=${RUN}_${ARM}${PARAM}; O=$X/s2/$TAG
cd $R/source/gfp
E="env -u LD_LIBRARY_PATH HOME=$R/home MUJOCO_GL=egl OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU taskset -c $CORES"
$E $PY $X/code/cand_fork.py train $A/out/probe/states.npz $O $RUN $FORK 200000 $ARM $PARAM > $X/logs/train_$TAG.log 2>&1 || exit 1
grep -q INVALID $X/logs/train_$TAG.log && exit 2
$E $PY $A/code/exp_c_eval.py $X/out/eval_$TAG.json $RUN $FORK 4242 10 $O/params_$((FORK+100000)).pkl $O/params_$((FORK+150000)).pkl $O/params_$((FORK+200000)).pkl > $X/logs/eval_$TAG.log 2>&1
