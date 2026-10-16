#!/bin/bash
# usage: run_x3.sh RUN FORK GPU CORES -- 200k X3 fork, then frozen eval at +100k/+150k/+200k (100 episodes, seed 4242)
R=/home/xushijie3/work/gfp_eta0_ablation_20261015; A=$R/analysis_p0p1; X=$A/x3
PY=/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python
RUN=$1; FORK=$2; GPU=$3; CORES=$4; O=$X/s1/${RUN}_X3
cd $R/source/gfp
E="env -u LD_LIBRARY_PATH HOME=$R/home MUJOCO_GL=egl OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU taskset -c $CORES"
$E $PY $X/code/x3_fork.py $A/out/probe/states.npz $O $RUN $FORK 200000 > $X/logs/train_${RUN}.log 2>&1 || exit 1
$E $PY $A/code/exp_c_eval.py $X/out/eval_x3_${RUN}.json $RUN $FORK 4242 10 $O/params_$((FORK+100000)).pkl $O/params_$((FORK+150000)).pkl $O/params_$((FORK+200000)).pkl > $X/logs/eval_${RUN}.log 2>&1
