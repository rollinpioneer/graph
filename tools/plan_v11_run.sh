#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
export MUJOCO_GL=egl
# do not print key
if [[ -f /home/__compress_data/xushijie/DASHSCOPE_API_KEY ]]; then
  export DASHSCOPE_API_KEY="$(python3 - <<'PY'
from pathlib import Path
lines=[ln.strip() for ln in Path("/home/__compress_data/xushijie/DASHSCOPE_API_KEY").read_text(encoding="utf-8").splitlines()]
print(next((x for x in lines if x and not x.startswith("#")), ""), end="")
PY
)"
fi
echo "key_present=${DASHSCOPE_API_KEY:+yes}"
STAGE="${1:?stage}"
case "$STAGE" in
  0a) "$ROOT/.venv-stage0a/bin/python" "$ROOT/tools/plan_v11_finalize_0a.py" ;;
  0b) "$ROOT/.venv-stage0a/bin/python" "$ROOT/tools/plan_v11_stage0b.py" ;;
  0c)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
    export CP_DISR_GPU=0
    "$ROOT/.venv-stage0a/bin/python" "$ROOT/tools/plan_v11_stage0c.py"
    ;;
  0d)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
    export CP_DISR_GPU=0
    "$ROOT/.venv-stage0a/bin/python" "$ROOT/tools/plan_v11_stage0d.py"
    ;;
  docs) "$ROOT/.venv-stage0a/bin/python" "$ROOT/tools/plan_v11_migration_close.py" ;;
  *) echo "unknown $STAGE"; exit 2 ;;
esac
