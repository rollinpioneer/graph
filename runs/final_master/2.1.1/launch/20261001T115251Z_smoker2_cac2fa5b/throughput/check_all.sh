#!/bin/bash
cd /home/xushijie2/graph_cp_disr_final_tb_launch
OUT=$(cat /home/xushijie2/tmp/r2/R2_OUT.txt)
date -u +%FT%TZ
echo "pids alive: $(pgrep -af 'final_tb_launch.py train|launch_k1_w3|throughput_monitor' | grep -v bash | awk '{print $1}' | tr '\n' ' ')"
echo "HEAD $(git rev-parse --short HEAD) tracked-dirty: $(git status --porcelain --untracked-files=no | wc -l)"
echo "bad-log files: $(grep -il 'Traceback\|DataIntegrity\|nan' $OUT/logs/train_*.log | tr '\n' ' ')"
python3 /home/xushijie2/tmp/r2/analyze_throughput.py > /tmp/r2_window.txt 2>&1
python3 - <<PY
import json,glob,os
r=json.load(open("$OUT/throughput/window_report.json"))
for p,v in r["ppo_updates"].items(): print("PPO", p, [(x["N"], x["wall_min"]) for x in v][-4:])
s=json.load(open("$OUT/launch_state.json"))
print("ledger", {p:(v["status"],v.get("physical_gpu")) for p,v in s["plans"].items()}, "attempts", s["new_rl_attempts_used"], "/", s["new_rl_attempts_cap"])
for p,v in s["plans"].items():
    rd=v.get("run_dir")
    if rd and os.path.exists(rd+"/job_summary.json"):
        j=json.load(open(rd+"/job_summary.json")); print("DONE", p, {k:j.get(k) for k in ("stop_reason","complete_updates","valid_transitions","interaction_seconds","hard_fail")})
PY
PYTHONPATH=src timeout 30 /home/xushijie2/envs/lerobotpi0-xfs/bin/python scripts/final_tb_launch.py status --root . --out $OUT 2>&1 | tail -1 | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
print('storage free_GB', round(d['storage']['free_bytes']/1e9,2), 'reserve_GB', round(d['storage']['reserve_bytes']/1e9,2), {p:round(v['bytes_written']/1e6) for p,v in d['storage']['running'].items()}, 'MB written')
for p,v in d['plans'].items(): print('status', p, v['status'], 'since_last_write', {k:round(x) for k,x in (v.get('seconds_since_last_write') or {}).items() if k in ('transition_log.jsonl','episode_log.jsonl')})"
tail -n 1 $OUT/throughput/samples.jsonl | python3 -c "
import sys,json
d=json.loads(sys.stdin.read()); print('sample', d['utc'], d['transition_lines'], 'iowait', d['iowait_frac'], 'gpu0-2', {k:d['gpus'][k] for k in ('0','1','2')})"
for p in R-TB-E-0 R-TB-DK-1 R-TB-K-1; do grep -h "PPO complete" $OUT/logs/train_$p.log | tail -n 1 | cut -c1-150; done
