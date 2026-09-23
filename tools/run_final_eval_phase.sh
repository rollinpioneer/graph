#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -f /home/__compress_data/xushijie/DASHSCOPE_API_KEY ]]; then
  export DASHSCOPE_API_KEY="$(python3 - <<'PY'
from pathlib import Path
lines=[ln.strip() for ln in Path("/home/__compress_data/xushijie/DASHSCOPE_API_KEY").read_text(encoding="utf-8").splitlines()]
print(next((x for x in lines if x and not x.startswith("#")), ""), end="")
PY
)"
fi
echo "key_present=${DASHSCOPE_API_KEY:+yes} cuda=$CUDA_VISIBLE_DEVICES phase=$1"
exec "$ROOT/.venv-stage0a/bin/python" "$ROOT/tools/plan_v11_stage1a_final_eval.py" "$@"
