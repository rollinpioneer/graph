#!/usr/bin/env python3
"""Passive throughput monitor (read-only): per-run transition line counts, GPU util/mem, iowait, loadavg, free bytes.
Writes one JSON line per sample to $OUT/throughput/samples.jsonl. Never touches run directories."""
import json, os, sys, time, glob, subprocess
OUT = sys.argv[1]; PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
os.makedirs(os.path.join(OUT, "throughput"), exist_ok=True)
plan = json.load(open(os.path.join(OUT, "launch_plan.json")))["plans"]
dirs = {pid: v["planned_run_dir"] for pid, v in plan.items()}
def cpu():
    f = open("/proc/stat").readline().split()[1:]
    f = list(map(int, f)); return sum(f), f[4]  # total, iowait
def nlines(p):
    try:
        with open(p, "rb") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return None
def gpus():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
        return {int(a.split(",")[0]): [int(x) for x in a.split(",")[1:]] for a in r.strip().splitlines()}
    except Exception as e:
        return {"error": str(e)}
prev = cpu()
while True:
    t = time.time(); cur = cpu(); dt = max(cur[0] - prev[0], 1); iow = (cur[1] - prev[1]) / dt; prev = cur
    st = os.statvfs(OUT)
    row = {"t": t, "utc": time.strftime("%H:%M:%S", time.gmtime(t)), "iowait_frac": round(iow, 4), "loadavg": os.getloadavg(),
           "free_bytes": st.f_bavail * st.f_frsize, "gpus": gpus(),
           "transition_lines": {pid: nlines(os.path.join(d, "transition_log.jsonl")) for pid, d in dirs.items() if os.path.isdir(d)}}
    with open(os.path.join(OUT, "throughput", "samples.jsonl"), "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    time.sleep(PERIOD)
