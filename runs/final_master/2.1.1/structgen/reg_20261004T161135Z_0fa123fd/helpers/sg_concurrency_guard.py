"""Reversible overload guard for the extra six STRUCT-GEN workers (SIGSTOP / SIGCONT only; never kills, never touches files).

usage: python sg_concurrency_guard.py <out_dir> plan1,plan2,...
Pauses the extra group when the host looks overloaded and resumes it when it has recovered, which reverts to the 6-wide schedule without losing any run.
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
GPU_PAUSE_MIB, GPU_RESUME_MIB = 36000, 30000
GOOD_SAMPLES_TO_RESUME = 10
log = out / "concurrency_guard.jsonl"
paused, good, last_event = False, 0, None


def emit(event, **kw):
    with log.open("a") as f:
        f.write(json.dumps({"event": event, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}) + "\n")


def pids_of_extra():
    try:
        state = json.loads((out / "launch_state.json").read_text())
    except Exception:
        return [], False
    pids, any_running = [], False
    for p in extra:
        plan = state["plans"].get(p, {})
        if plan.get("status") == "RUNNING" and plan.get("pid"):
            any_running = True
            pids.append(int(plan["pid"]))
            try:
                kids = subprocess.check_output(["pgrep", "-P", str(plan["pid"])], text=True).split()
                for k in kids:
                    pids.append(int(k))
                    try:
                        pids += [int(x) for x in subprocess.check_output(["pgrep", "-P", k], text=True).split()]
                    except subprocess.CalledProcessError:
                        pass
            except subprocess.CalledProcessError:
                pass
    return pids, any_running


def host():
    load1 = float(open("/proc/loadavg").read().split()[0])
    mem = [l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()
    mem_gib = int(mem[1]) / 1024 / 1024
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()
    return load1, mem_gib, max(int(x) for x in gpu)


def send(sig):
    pids, _ = pids_of_extra()
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass
    return len(pids)


emit("GUARD_START", thresholds={"load_pause": LOAD_PAUSE, "load_resume": LOAD_RESUME, "mem_pause_gib": MEM_PAUSE_GIB, "mem_resume_gib": MEM_RESUME_GIB, "gpu_pause_mib": GPU_PAUSE_MIB}, group=extra)
tick = 0
try:
    while True:
        _pids, any_running = pids_of_extra()
        if not any_running and tick > 3:
            emit("GUARD_END", reason="extra workers finished", paused_now=paused)
            break
        load1, mem_gib, gpu_max = host()
        bad = load1 > LOAD_PAUSE or mem_gib < MEM_PAUSE_GIB or gpu_max > GPU_PAUSE_MIB
        ok = load1 < LOAD_RESUME and mem_gib > MEM_RESUME_GIB and gpu_max < GPU_RESUME_MIB
        if not paused and bad:
            n = send(signal.SIGSTOP)
            paused, good = True, 0
            emit("PAUSE_EXTRA_GROUP", load1=load1, mem_gib=mem_gib, gpu_max_mib=gpu_max, processes=n)
        elif paused:
            good = good + 1 if ok else 0
            if good >= GOOD_SAMPLES_TO_RESUME:
                n = send(signal.SIGCONT)
                paused = False
                emit("RESUME_EXTRA_GROUP", load1=load1, mem_gib=mem_gib, gpu_max_mib=gpu_max, processes=n)
        if tick % 20 == 0:
            emit("SAMPLE", load1=load1, mem_gib=round(mem_gib, 1), gpu_max_mib=gpu_max, paused=paused)
        tick += 1
        time.sleep(30)
finally:
    if paused:
        send(signal.SIGCONT)
        emit("RESUME_ON_EXIT")
