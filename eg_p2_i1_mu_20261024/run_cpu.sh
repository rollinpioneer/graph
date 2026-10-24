#!/usr/bin/env bash
set -e
source ~/work/grasp/eg_sub25_b_f05_repl_v1_20261019/activate.sh
cd ~/work/grasp/eg_p2_i1_mu_20261024
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# First: bash run_cpu.sh replay
# After replay passes: run shard 0/1/2 on separate hosts, 4 workers each.
# Existing completed outputs are intentionally not overwritten.
python evaluate_p2.py "$@"
