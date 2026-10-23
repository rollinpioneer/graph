#!/usr/bin/env bash
set -e
source ~/work/grasp/eg_sub25_b_f05_repl_v1_20261019/activate.sh
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
cd ~/work/grasp/eg_causal_mechanism_v2_20261023/r14
for scene in 9 31 33; do
  python code/pulse_train.py source --scene "$scene"
done
