#!/usr/bin/env bash
set -euo pipefail
OLD=/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011
ROUND=/home/xushijie3/work/graph_pc_p123_card_20261011/experiments/pc_p123_run_20261011
export PATH="$OLD/env/bin:$PATH"
export LD_LIBRARY_PATH="$OLD/env/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MAX_JOBS=4 MPLBACKEND=Agg WANDB_MODE=disabled
export TORCH_EXTENSIONS_DIR="$ROUND/runtime/torch_extensions"
exec "$OLD/env/bin/python" -u "$ROUND/evaluate_validation.py" "$@"
