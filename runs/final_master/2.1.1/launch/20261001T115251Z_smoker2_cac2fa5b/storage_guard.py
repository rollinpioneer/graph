#!/usr/bin/env python3
"""Reversible disk-full guard for the three registered T_B training workers (R2).
Only action: SIGSTOP when /home free bytes < STOP_BELOW; SIGCONT when free bytes >= RESUME_AT or when this guard is terminated.
Never kills, never deletes, never touches run files. Every action is logged to OUT/storage_guard.jsonl."""
import json, os, signal, sys, time
OUT = open("/home/xushijie2/tmp/r2/R2_OUT.txt").read().strip()
STOP_BELOW = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e9
RESUME_AT = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0e9
DRY = len(sys.argv) > 3 and sys.argv[3] == "dry"
LOG = os.path.join(OUT, "storage_guard.jsonl")
def log(**kw):
    kw["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); kw["guard_pid"] = os.getpid()
    with open(LOG, "a") as f: f.write(json.dumps(kw, sort_keys=True) + "\n")
def free():
    st = os.statvfs(OUT); return st.f_bavail * st.f_frsize
def workers():
    try: s = json.load(open(os.path.join(OUT, "launch_state.json")))["plans"]
    except Exception: return {}
    res = {}
    for p, v in s.items():
        pid = v.get("pid")
        if v.get("status") == "RUNNING" and pid:
            try:
                cmd = open("/proc/%d/cmdline" % pid, "rb").read().decode(errors="replace")
                if "final_tb_launch.py" in cmd or "launch_k1_w3.py" in cmd: res[p] = int(pid)
            except Exception: pass
    return res
paused = {}
def cleanup(*_):
    for p, pid in list(paused.items()):
        try: os.kill(pid, signal.SIGCONT)
        except Exception: pass
    log(event="GUARD_EXIT_RESUMED_ALL", paused=list(paused))
    sys.exit(0)
signal.signal(signal.SIGTERM, cleanup); signal.signal(signal.SIGINT, cleanup)
log(event="GUARD_START", stop_below=STOP_BELOW, resume_at=RESUME_AT, dry=DRY, free=free(), workers=workers())
n = 0
while True:
    f = free()
    if not paused and f < STOP_BELOW:
        w = workers(); log(event="WOULD_STOP" if DRY else "SIGSTOP", free=f, workers=w)
        if not DRY:
            for p, pid in w.items():
                try: os.kill(pid, signal.SIGSTOP); paused[p] = pid
                except Exception as e: log(event="STOP_FAILED", plan=p, err=str(e))
        else: log(event="DRY_EXIT"); sys.exit(0)
    elif paused and f >= RESUME_AT:
        for p, pid in list(paused.items()):
            try: os.kill(pid, signal.SIGCONT)
            except Exception as e: log(event="CONT_FAILED", plan=p, err=str(e))
        log(event="SIGCONT", free=f, plans=list(paused)); paused = {}
    n += 1
    if n % 150 == 0: log(event="HEARTBEAT", free=f, paused=list(paused))
    time.sleep(2)
