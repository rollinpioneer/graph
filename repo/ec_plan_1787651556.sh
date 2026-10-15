#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
cd /home/__compress_data/xushijie/CUPID/repo

# ---- substrate/tick ----
mkdir -p /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test
date +%s > /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/.start
echo "[plan] launching substrate/tick at $(date -Is)" >> /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/train.log
/home/xushijie/.conda/envs/cupid/bin/python -u /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/tick.py >> /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/train.log 2>&1 &
CHILD=$!
echo $CHILD > /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/.child.pid
while kill -0 $CHILD 2>/dev/null; do date +%s > /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/.heartbeat; sleep 30; done
wait $CHILD; RC=$?
echo $RC > /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/.rc
date +%s > /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/.end
echo "[plan] substrate/tick exited rc=$RC at $(date -Is)" >> /home/__compress_data/xushijie/CUPID/repo/runs/_substrate_test/train.log
[ "$RC" = "0" ] || { echo "ABORT after substrate/tick"; exit 1; }

echo "ALL_DONE"
