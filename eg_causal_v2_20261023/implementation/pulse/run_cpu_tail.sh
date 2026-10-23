#!/usr/bin/env bash
set -e
source ~/work/grasp/eg_sub25_b_f05_repl_v1_20261019/activate.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/work/grasp/eg_causal_mechanism_v2_20261023/r14
for scene in 82 76 64; do
  python code/evaluate_pulse.py --source "$scene" --workers 24
done
