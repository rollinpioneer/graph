#!/usr/bin/env bash
set -euo pipefail
OLD=/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011
ROUND="$(cd "$(dirname "$0")" && pwd)"
export PATH="$OLD/env/bin:$PATH"
export LD_LIBRARY_PATH="$OLD/env/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MAX_JOBS=4 MPLBACKEND=Agg WANDB_MODE=disabled
export TORCH_EXTENSIONS_DIR="$ROUND/runtime/torch_extensions"
WORKER="${PC_P123_WORKER:-evaluate.py}"
case "$WORKER" in evaluate.py|evaluate_validation.py) ;; *) exit 2 ;; esac
exec "$OLD/env/bin/python" -u "$ROUND/$WORKER" "$@"
