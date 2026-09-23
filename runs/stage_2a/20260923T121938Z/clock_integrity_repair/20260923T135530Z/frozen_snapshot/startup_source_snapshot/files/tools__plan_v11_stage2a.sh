#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$ROOT"
export PYTHONPATH="$ROOT/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -f /home/__compress_data/xushijie/DASHSCOPE_API_KEY ]]; then
  export DASHSCOPE_API_KEY="$(python3 - <<'PY'
from pathlib import Path
lines=[ln.strip() for ln in Path("/home/__compress_data/xushijie/DASHSCOPE_API_KEY").read_text(encoding="utf-8").splitlines()]
print(next(ln for ln in lines if ln and not ln.startswith("#")))
PY
)"
fi
echo "key_present=${DASHSCOPE_API_KEY:+yes}"
PY="$ROOT/.venv-stage0a/bin/python"
LOGDIR="$ROOT/runs/stage_2a/logs"
mkdir -p "$LOGDIR"

phase="${1:-orchestrate}"
shift || true

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' | sort -nr | awk '{print $2}' | head -n "$1"
}

run_cli() {
  local gpu="$1"; shift
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m cp_disr.cli --root "$ROOT" stage-2a-v11-run --gpu 0 "$@"
}

if [[ "$phase" == "orchestrate" ]]; then
  echo "=== deadline tests ==="
  CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest tests/test_deadline_semantics.py -q --tb=short | tee "$LOGDIR/deadline_tests.log"
  echo "=== sync-1a ==="
  run_cli 0 --phase sync-1a | tee "$LOGDIR/sync_1a.log"
  echo "=== freeze ==="
  run_cli 0 --phase freeze | tee "$LOGDIR/freeze.log"
  STAMP=$(python3 - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1/runs/stage_2a/CURRENT.json").read_text())["stamp"])
PY
)
  SHA=$(python3 - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1/runs/stage_2a/CURRENT.json").read_text())["configsha8"])
PY
)
  echo "stamp=$STAMP configsha8=$SHA"
  echo "=== materialize ==="
  run_cli 0 --phase materialize --stamp "$STAMP" --configsha "$SHA" | tee "$LOGDIR/materialize.log"
  echo "=== startup T_B ==="
  run_cli 0 --phase startup --task T_B --stamp "$STAMP" --configsha "$SHA" | tee "$LOGDIR/startup_T_B.log"
  echo "=== startup T_C ==="
  run_cli 0 --phase startup --task T_C --stamp "$STAMP" --configsha "$SHA" | tee "$LOGDIR/startup_T_C.log"
  jobs=(
    "T_B B0"
    "T_B B1-K"
    "T_B B2"
    "T_B Full"
    "T_C B0"
    "T_C B1-K"
    "T_C B2"
    "T_C Full"
  )
  i=0
  while [[ $i -lt ${#jobs[@]} ]]; do
    mapfile -t gpus < <(pick_gpus 2)
    g0=${gpus[0]:-0}
    g1=${gpus[1]:-0}
    pair1=${jobs[$i]}
    pair2=${jobs[$((i+1))]:-}
    set -- $pair1
    t1=$1; m1=$2
    echo "launch $t1 $m1 gpu=$g0"
    run_cli "$g0" --phase train --task "$t1" --method "$m1" --stamp "$STAMP" --configsha "$SHA" --resume > "$LOGDIR/train_${t1}_${m1}.log" 2>&1 &
    p1=$!
    p2=""
    if [[ -n "$pair2" ]]; then
      set -- $pair2
      t2=$1; m2=$2
      echo "launch $t2 $m2 gpu=$g1"
      run_cli "$g1" --phase train --task "$t2" --method "$m2" --stamp "$STAMP" --configsha "$SHA" --resume > "$LOGDIR/train_${t2}_${m2}.log" 2>&1 &
      p2=$!
    fi
    wait $p1
    ec1=$?
    if [[ -n "$p2" ]]; then
      wait $p2
      ec2=$?
    else
      ec2=0
    fi
    echo "pair done ec1=$ec1 ec2=$ec2"
    if [[ $ec1 -ne 0 || $ec2 -ne 0 ]]; then
      echo "training pair failed; stop launching more jobs"
      exit 2
    fi
    i=$((i+2))
  done
  echo "=== select ==="
  run_cli 0 --phase select --stamp "$STAMP" --configsha "$SHA" | tee "$LOGDIR/select.log"
  echo "=== eval ==="
  run_cli 0 --phase eval --stamp "$STAMP" --configsha "$SHA" | tee "$LOGDIR/eval.log"
  echo "=== report ==="
  run_cli 0 --phase report --stamp "$STAMP" --configsha "$SHA" | tee "$LOGDIR/report.log"
  echo ORCHESTRATE_DONE
  exit 0
fi

run_cli "${GPU:-0}" --phase "$phase" "$@"
