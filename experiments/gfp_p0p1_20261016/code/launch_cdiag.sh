#!/bin/bash
R=/home/xushijie3/work/gfp_eta0_ablation_20261015; A=$R/analysis_p0p1; O=$A/out/cdiag; L=$A/logs_cdiag
PY=/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python
cd $R/source/gfp
E="env -u LD_LIBRARY_PATH HOME=$R/home MUJOCO_GL=egl OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false"
ARMS4="A0 A1 A4 A2 A3 A5 F0 F2 F3"
spec4(){ for n in "$@"; do for a in $ARMS4; do echo -n "${n}:1000000:$a "; done; done; }
setsid -f $E CUDA_VISIBLE_DEVICES=0 taskset -c 60-63 $PY $A/code/p0_fixedref.py $A/out/probe/states.npz $O/p0_fixedref.json G_task1_s0:500000:600000 G_task1_s0:600000:700000 G_task1_s0:700000:800000 G_task1_s0:900000:1000000 G_task1_s3:800000:900000 G_task1_s1:600000:700000 F_task1_s0:600000:700000 G_task4_s3:800000:900000 > $L/p0.log 2>&1
setsid -f $E CUDA_VISIBLE_DEVICES=4 taskset -c 64-99 $PY $A/code/c1_rank.py $O/c1_rank.json 10 32 50 G_task4_s0:1000000 G_task4_s1:1000000 G_task4_s2:1000000 G_task4_s3:1000000 F_task4_s0:1000000 F_task4_s1:1000000 F_task4_s2:1000000 F_task4_s3:1000000 > $L/c1.log 2>&1
setsid -f $E CUDA_VISIBLE_DEVICES=0 taskset -c 100-103 $PY $A/code/c2_replay.py $O/c2_p1.json 4242 3 $(spec4 G_task4_s0 F_task4_s0) > $L/c2_p1.log 2>&1
setsid -f $E CUDA_VISIBLE_DEVICES=4 taskset -c 104-107 $PY $A/code/c2_replay.py $O/c2_p2.json 4242 3 $(spec4 G_task4_s1 F_task4_s1) > $L/c2_p2.log 2>&1
setsid -f $E CUDA_VISIBLE_DEVICES=5 taskset -c 108-111 $PY $A/code/c2_replay.py $O/c2_p3.json 4242 3 $(spec4 G_task4_s2 F_task4_s2) > $L/c2_p3.log 2>&1
setsid -f $E CUDA_VISIBLE_DEVICES=7 taskset -c 56-59 $PY $A/code/c2_replay.py $O/c2_p4.json 4242 3 $(spec4 G_task4_s3 F_task4_s3) > $L/c2_p4.log 2>&1
T1=""; for n in G_task1_s1 G_task1_s0 F_task1_s0; do for a in A0 A2 A3 A4; do T1="$T1 ${n}:1000000:$a"; done; done
setsid -f $E CUDA_VISIBLE_DEVICES=7 taskset -c 52-55 $PY $A/code/c2_replay.py $O/c2_t1.json 4242 3 $T1 > $L/c2_t1.log 2>&1
