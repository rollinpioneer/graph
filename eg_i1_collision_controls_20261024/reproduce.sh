#!/usr/bin/env bash
# Run one disjoint shard per host. Existing registered datasets/SDFs are required.
set -e
source "$HOME/work/grasp/eg_sub25_b_f05_repl_v1_20261019/activate.sh"
cd "$HOME/work/grasp/eg_i1_collision_controls_20261024"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_controls.py "${1:-0}" 4
python mechanism_check.py "${1:-0}"
