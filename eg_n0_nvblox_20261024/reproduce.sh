#!/usr/bin/env bash
# Run only the two-frame preflight; no calibration or C3 evaluation is authorized by this script.
set -euo pipefail
N0_ROOT="$HOME/work/grasp/eg_n0_nvblox_20261024"
source "$HOME/work/grasp/eg_sub25_b_f05_repl_v1_20261019/activate.sh"
cd "$N0_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python infer_frozen.py pilot
# Existing N0 env: torch 2.9.1+cu126, open3d 0.18.0, numpy 1.26.4,
# official v0.0.10 cu12ubuntu22 wheel; see source_manifest.json.
unset PYTHONPATH LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES="${N0_GPU:-0}" env/bin/python check_nvblox.py
env/bin/python analyze_preflight.py
