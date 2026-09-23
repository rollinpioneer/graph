#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/__compress_data/xushijie/graph_cp_disr_v2_1
cd "$ROOT"
CLOCK_STOP="$ROOT/runs/stage_2a/orchestrator/STOP_CLOCK_INTEGRITY.json"
if [[ -f "$CLOCK_STOP" ]]; then
  echo "REFUSING_DISPATCH_STOP_CLOCK_INTEGRITY"
  cat "$CLOCK_STOP"
  exit 75
fi
export PYTHONPATH="$ROOT/src:/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -f /home/__compress_data/xushijie/DASHSCOPE_API_KEY ]]; then
  export DASHSCOPE_API_KEY="$(python3 -c 'from pathlib import Path; lines=[ln.strip() for ln in Path("/home/__compress_data/xushijie/DASHSCOPE_API_KEY").read_text(encoding="utf-8").splitlines()]; print(next(ln for ln in lines if ln and not ln.startswith("#")))')"
fi
PY="$ROOT/.venv-stage0a/bin/python"
STAMP=20260923T121938Z
SHA=79c690f9
LOGDIR="$ROOT/runs/stage_2a/logs"
mkdir -p "$LOGDIR"
run_cli() {
  local gpu="$1"; shift
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m cp_disr.cli --root "$ROOT" stage-2a-v11-run --gpu 0 --stamp "$STAMP" --configsha "$SHA" "$@"
}
echo "waiting for materialize pid if any"
for i in $(seq 1 360); do
  if pgrep -f "stage-2a-v11-run .*--phase materialize" >/dev/null; then
    echo "materialize still running t=$i"
    sleep 20
    continue
  fi
  break
done
if ! grep -q 'MATERIALIZED' "$LOGDIR/materialize.log"; then
  echo "materialize did not record MATERIALIZED; last log:"
  tail -n 80 "$LOGDIR/materialize.log"
  exit 2
fi
echo "=== deadline tests ==="
CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest tests/test_deadline_semantics.py -q --tb=short | tee "$LOGDIR/deadline_tests.log"
echo "=== startup ==="
run_cli 0 --phase startup --task T_B | tee "$LOGDIR/startup_T_B.log"
if ! grep -q 'STARTUP_GATES_PASS' "$LOGDIR/startup_T_B.log"; then
  echo "T_B startup hard checks did not pass"
  exit 2
fi
run_cli 0 --phase startup --task T_C | tee "$LOGDIR/startup_T_C.log"
if ! grep -q 'STARTUP_GATES_PASS' "$LOGDIR/startup_T_C.log"; then
  echo "T_C startup hard checks did not pass"
  exit 2
fi
pick_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' | sort -nr | awk '{print $2}' | head -n "$1"
}
jobs=("T_B B0" "T_B B1-K" "T_B B2" "T_B Full" "T_C B0" "T_C B1-K" "T_C B2" "T_C Full")
i=0
while [[ $i -lt ${#jobs[@]} ]]; do
  if [[ -f "$CLOCK_STOP" ]]; then
    echo "REFUSING_DISPATCH_STOP_CLOCK_INTEGRITY"
    cat "$CLOCK_STOP"
    exit 75
  fi
  mapfile -t gpus < <(pick_gpus 2)
  g0=${gpus[0]:-0}
  g1=${gpus[1]:-0}
  set -- ${jobs[$i]}
  t1=$1; m1=$2
  echo "launch $t1 $m1 gpu=$g0"
  run_cli "$g0" --phase train --task "$t1" --method "$m1" --resume > "$LOGDIR/train_${t1}_${m1}.log" 2>&1 &
  p1=$!
  p2=""
  t2=""; m2=""
  if [[ $((i+1)) -lt ${#jobs[@]} ]]; then
    set -- ${jobs[$((i+1))]}
    t2=$1; m2=$2
    echo "launch $t2 $m2 gpu=$g1"
    run_cli "$g1" --phase train --task "$t2" --method "$m2" --resume > "$LOGDIR/train_${t2}_${m2}.log" 2>&1 &
    p2=$!
  fi
  ec1=0
  ec2=0
  wait "$p1" || ec1=$?
  if [[ -n "$p2" ]]; then
    wait "$p2" || ec2=$?
  fi
  echo "pair done $t1 $m1 ec=$ec1 ${t2:-none} ${m2:-none} ec=$ec2"
  if [[ $ec1 -ne 0 || $ec2 -ne 0 ]]; then
    echo TRAIN_PAIR_FAILED
    exit 2
  fi
  for pair in "$t1 $m1" "$t2 $m2"; do
    [[ "$pair" == " " || -z "${pair// }" ]] && continue
    set -- $pair
    tt=$1; mm=$2
    [[ -z "${tt:-}" || -z "${mm:-}" ]] && continue
    summary="$ROOT/runs/stage_2a/$tt/$mm/seed_0/${STAMP}_${SHA}/job_summary.json"
    if [[ ! -f "$summary" ]]; then
      echo "missing job_summary $summary"
      exit 2
    fi
  done
  i=$((i+2))
done
run_cli 0 --phase select | tee "$LOGDIR/select.log"
if ! grep -q 'SELECTION_FROZEN' "$LOGDIR/select.log"; then
  echo "select did not freeze"
  exit 2
fi
run_cli 0 --phase eval | tee "$LOGDIR/eval.log"
run_cli 0 --phase report | tee "$LOGDIR/report.log"
echo CONTINUE_DONE
