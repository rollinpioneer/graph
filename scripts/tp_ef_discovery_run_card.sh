#!/bin/bash
set -e
export WT=$HOME/graph_cp_disr_tp_ef
export PY=$HOME/envs/lerobotpi0-xfs/bin/python
export PYTHONPATH="$WT/src:$WT" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
unset DASHSCOPE_API_KEY DASHSCOPE_API_KEY_FILE
cd $WT
OUT=$WT/$(cat /tmp/xsj2_tpef_out.txt)
mkdir -p $OUT
S="$PY scripts/tp_ef_discovery.py"
echo "== gate0";      CUDA_VISIBLE_DEVICES="" $S gate0 --root $WT --output $OUT
echo "== inventory";  CUDA_VISIBLE_DEVICES="" $S inventory --root $WT --output $OUT
GPUS=$($S gpus --root $WT | $PY -c "import sys,json;f,_=json.load(sys.stdin);print(','.join(map(str,f[:2])))")
echo "== gpus selected: '$GPUS'"
if [ -z "$GPUS" ]; then echo STOPPED_CAPTURE_GPU_UNAVAILABLE; exit 3; fi
FIRST=${GPUS%%,*}
echo "== register (gpu $FIRST)"; CUDA_VISIBLE_DEVICES=$FIRST $S register --root $WT --output $OUT
echo "== wave 1";     $S run-wave --root $WT --output $OUT --wave 1 --gpus $GPUS
echo "== epsilon";    $S freeze-epsilon --root $WT --output $OUT
echo "== wave 2";     $S run-wave --root $WT --output $OUT --wave 2 --gpus $GPUS
echo "== analyze";    CUDA_VISIBLE_DEVICES="" $S analyze --root $WT --output $OUT
echo "== verify";     CUDA_VISIBLE_DEVICES="" $S verify --root $WT --output $OUT
echo "== DONE"
