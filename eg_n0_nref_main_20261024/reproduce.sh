#!/usr/bin/env bash
# Reproduction on the original asset host. Do not overwrite historical source rounds.
set -euo pipefail
source "$HOME/work/grasp/eg_sub25_b_f05_repl_v1_20261019/activate.sh"
N0_ROOT="$HOME/work/grasp/eg_n0_nref_main_20261024"
N0_PY="$HOME/work/grasp/eg_n0_nvblox_20261024/env/bin/python"
cd "$N0_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [ -f statistics.json ]; then
  # Existing results: verify evidence, without rerunning the formal comparison.
  env -u PYTHONPATH -u LD_LIBRARY_PATH "$N0_PY" finalize.py
  exit 0
fi
# Use an empty independent round directory with the supplied code/protocol and
# original EG/GraspNet assets. Inference targets the documented N0_ROOT.
python infer_frozen.py train
python infer_frozen.py P2
for i in 0 1 2; do python pipeline.py train "$i" 4 & done
wait
# nvblox runtime needs the A100 host's CUDA NPP system library (libnppc.so.12).
env -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="${N0_GPU:-0}" "$N0_PY" nv_worker.py train
if [ ! -f frozen_config.json ]; then env -u PYTHONPATH -u LD_LIBRARY_PATH "$N0_PY" calibrate.py; fi
for split in C3 P2; do
  for i in 0 1 2; do python pipeline.py "$split" "$i" 4 & done
  wait
  env -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="${N0_GPU:-0}" "$N0_PY" nv_worker.py "$split"
done
for i in 0 1 2; do
  python after_scores.py dense "$i" 4 &
  python after_scores.py joint "$i" 4 &
done
wait
env -u PYTHONPATH -u LD_LIBRARY_PATH "$N0_PY" time_reference.py
env -u PYTHONPATH -u LD_LIBRARY_PATH "$N0_PY" analyze.py
env -u PYTHONPATH -u LD_LIBRARY_PATH "$N0_PY" finalize.py
env -u PYTHONPATH -u LD_LIBRARY_PATH "$N0_PY" make_report.py
