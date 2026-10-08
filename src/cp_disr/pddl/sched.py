"""Dependency-respecting parallel task scheduler (one task per free GPU / bounded CPU slots, resumable by receipts, retries for training tasks) for the public-Depots card.

A manifest task is {id, argv, receipt, requires, resource: 'gpu'|'cpu', input_hashes, [training_run_id]}. A task counts as done when its receipt exists, its status is a done-state and its
``input_fingerprint`` equals the manifest fingerprint. Handlers write receipts with ``finish``. Failed tasks block only their dependants; independent tasks continue.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

DONE = ("DONE", "REUSED_EXACT", "SKIPPED_BY_RULE")


def fp_of(task):
    return hashlib.sha256(json.dumps({k: task.get(k) for k in ("id", "argv", "input_hashes")}, sort_keys=True).encode()).hexdigest()


def wj(p, doc):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def finish(rr, tid, status, outputs=(), **extra):
    outs = [{"path": str(Path(o).resolve()), "sha256": sha_file(o)} for o in outputs]
    wj(Path(rr) / "receipts" / ("%s.json" % tid), {"task": tid, "status": status, "input_fingerprint": os.environ.get("TASK_FINGERPRINT"), "outputs": outs,
                                                   "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})


def run_all(rr, cpu_slots=4, poll=10):
    rr = Path(rr)
    man = json.loads((rr / "manifest.json").read_text())
    tasks = {t["id"]: t for t in man["tasks"]}
    gpus = list(man["gpus"])
    env0 = {**os.environ, **man["env"]}
    attempts = Counter()
    failed = set()

    def receipt(tid):
        p = Path(tasks[tid]["receipt"])
        try:
            r = json.loads(p.read_text()) if p.is_file() else None
        except json.JSONDecodeError:
            r = None
        return r if r and r.get("status") in DONE and r.get("input_fingerprint") == fp_of(tasks[tid]) else None
    running = {}
    free = list(gpus)
    pending = [t for t in tasks if not receipt(t)]
    while pending or running:
        for tid, (proc, g) in list(running.items()):
            if proc.poll() is not None:
                if g is not None:
                    free.append(g)
                del running[tid]
                if not receipt(tid):
                    attempts[tid] += 1
                    if attempts[tid] < (3 if tasks[tid].get("training_run_id") else 1):
                        pending.append(tid)
                    else:
                        failed.add(tid)
        cpu_running = sum(1 for tid, (_p, g) in running.items() if g is None)
        for tid in list(pending):
            if tid in failed:
                pending.remove(tid)
                continue
            req = tasks[tid].get("requires", [])
            if any(r in failed for r in req):
                failed.add(tid)
                pending.remove(tid)
                continue
            if not all(receipt(r) for r in req):
                continue
            if not all(receipt(r) or r in failed for r in tasks[tid].get("after", [])):                      # soft ordering: wait until these are settled (done or failed), never blocked by their failure
                continue
            t = tasks[tid]
            if t.get("resource", "gpu") == "gpu":
                if not free:
                    continue
                g = free.pop(0)
            else:
                if cpu_running >= cpu_slots:
                    continue
                g = None
                cpu_running += 1
            pending.remove(tid)
            env = {**env0, "TASK_FINGERPRINT": fp_of(t)}
            env["CUDA_VISIBLE_DEVICES"] = str(g) if g is not None else ""
            log = open(rr / "driver_logs" / ("%s.log" % tid), "ab")
            running[tid] = (subprocess.Popen(t["argv"], cwd=man["repo_root"], env=env, stdout=log, stderr=subprocess.STDOUT), g)
        time.sleep(poll)
    wj(rr / "scheduler_receipt.json", {"done": sorted(t for t in tasks if receipt(t)), "failed": sorted(failed), "attempts": dict(attempts), "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 1 if failed else 0
