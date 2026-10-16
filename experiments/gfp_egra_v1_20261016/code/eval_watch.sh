#!/bin/bash
# usage: eval_watch.sh RUN FORK ARM GPU CORES  -- waits for training to finish, then frozen eval at +100k and +200k
R=/home/xushijie3/work/gfp_eta0_ablation_20261015; A=$R/analysis_p0p1; D=$A/egra
PY=/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python
RUN=$1; FORK=$2; ARM=$3; GPU=$4; CORES=$5; O=$D/s1/${RUN}_$ARM
while [ ! -f $O/train_$ARM.json ]; do sleep 30; done
cd $R/source/gfp
env -u LD_LIBRARY_PATH HOME=$R/home MUJOCO_GL=egl OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU taskset -c $CORES \
  $PY $A/code/exp_c_eval.py $D/out/eval_s1_${RUN}_$ARM.json $RUN $FORK 4242 10 $O/params_$((FORK+100000)).pkl $O/params_$((FORK+200000)).pkl
