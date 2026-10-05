"""Per-worker reversible overload guard (v2) for the extra six STRUCT-GEN workers: SIGSTOP / SIGCONT only, never kills, never touches files.

usage: python sg_concurrency_guard2.py <out_dir> plan1,plan2,...
A worker is paused when the host is overloaded (load1>75 or MemAvailable<30 GiB) or when ITS OWN GPU is nearly full (used>34000 MiB, e.g. a foreign job);
it resumes after 10 consecutive good samples (load1<45, MemAvailable>60 GiB, own GPU<30000 MiB). Unaffected workers keep running.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
extra = sys.argv[2].split(",")
LOAD_PAUSE, LOAD_RESUME = 75.0, 45.0
MEM_PAUSE_GIB, MEM_RESUME_GIB = 30.0, 60.0
GPU_PAUSE_MIB, GPU_RESUME_MIB = 38000, 36000
GOOD = 10
log = out / "concurrency_guard.jsonl"


def emit(event, **kw):
    with log.open("a") as f:
        f.write(json.dumps({"event": event, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}) + "\n")


def tree(pid):
    pids = [pid]
    try:
        for k in subprocess.check_output(["pgrep", "-P", str(pid)], text=True).split():
            pids += tree(int(k))
    except subprocess.CalledProcessError:
        pass
    return pids


def state_of(pid):
    try:
        return open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


def plans():
    s = json.loads((out / "launch_state.json").read_text())
    return {p: s["plans"][p] for p in extra if s["plans"][p].get("status") == "RUNNING" and s["plans"][p].get("pid")}


def host():
    load1 = float(open("/proc/loadavg").read().split()[0])
    mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1024 / 1024
    gpu = [int(x) for x in subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()]
    return load1, mem, gpu


def signal_plan(plan, sig):
    n = 0
    for pid in tree(int(plan["pid"])):
        try:
            os.kill(pid, sig)
            n += 1
        except OSError:
            pass
    return n


emit("GUARD3_START", thresholds={"load_pause": LOAD_PAUSE, "load_resume": LOAD_RESUME, "mem_pause_gib": MEM_PAUSE_GIB, "mem_resume_gib": MEM_RESUME_GIB, "own_gpu_pause_mib": GPU_PAUSE_MIB,
                                 "own_gpu_resume_mib": GPU_RESUME_MIB}, group=extra)
paused = {p: state_of(int(v["pid"])) == "T" for p, v in plans().items()}
good = {p: 0 for p in paused}
tick = 0
try:
    while True:
        cur = plans()
        if not cur and tick > 3:
            emit("GUARD3_END", reason="extra workers finished")
            break
        load1, mem, gpu = host()
        host_bad = load1 > LOAD_PAUSE or mem < MEM_PAUSE_GIB
        host_ok = load1 < LOAD_RESUME and mem > MEM_RESUME_GIB
        for p, v in cur.items():
            g = int(v["physical_gpu"])
            paused.setdefault(p, False)
            good.setdefault(p, 0)
            bad = host_bad or gpu[g] > GPU_PAUSE_MIB
            ok = host_ok and gpu[g] < GPU_RESUME_MIB
            if not paused[p] and bad:
                n = signal_plan(v, signal.SIGSTOP)
                paused[p], good[p] = True, 0
                emit("PAUSE_WORKER", plan=p, gpu=g, load1=load1, mem_gib=round(mem, 1), own_gpu_mib=gpu[g], processes=n)
            elif paused[p]:
                good[p] = good[p] + 1 if ok else 0
                if good[p] >= GOOD:
                    n = signal_plan(v, signal.SIGCONT)
                    paused[p] = False
                    emit("RESUME_WORKER", plan=p, gpu=g, load1=load1, mem_gib=round(mem, 1), own_gpu_mib=gpu[g], processes=n)
        if tick % 20 == 0:
            emit("SAMPLE", load1=load1, mem_gib=round(mem, 1), gpu_mib=gpu, paused=sorted(p for p, x in paused.items() if x))
        tick += 1
        time.sleep(30)
finally:
    for p, v in plans().items():
        if paused.get(p):
            signal_plan(v, signal.SIGCONT)
            emit("RESUME_ON_EXIT", plan=p)
