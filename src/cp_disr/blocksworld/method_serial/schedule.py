"""Linux serial-driver reference; integrate real stage commands in the experiment repo.

No model/training implementation is included. Every command must write a JSON
receipt with a terminal status and hash-identified outputs. Local method failures
do not block independent stages; global identity/leakage failures stop the suite.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

DONE_STATES = {"DONE","REUSED_EXACT","MATHEMATICALLY_EQUIVALENT_REUSED"}
ALLOWED_SKIPS = {"UNINFORMATIVE_LABELS"}
LOCAL_STATES = {"TECHNICAL_INCOMPLETE","BLOCKED_DEPENDENCY"}
GLOBAL_STATES = {"BLOCKED_PATH_RESOLUTION","GLOBAL_INTEGRITY_FAILURE","DATA_LEAKAGE"}
ALL_TERMINAL = DONE_STATES | ALLOWED_SKIPS | LOCAL_STATES | GLOBAL_STATES

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)

def check_new_root(raw: str) -> Path:
    p=Path(raw).expanduser().resolve(strict=True)
    if not p.is_dir():
        raise ValueError("storage_root must be a directory")
    if "xushijie2" in str(p):
        raise ValueError("new root resolves into legacy xushijie2")
    if "xushijie3" not in str(p):
        raise ValueError("verify an xushijie3 root, not a guessed alternative")
    if not os.access(p,os.W_OK|os.X_OK):
        raise PermissionError("storage_root not writable")
    return p

def within(path: Path, parent: Path) -> Path:
    p=path.expanduser().resolve()
    try:p.relative_to(parent)
    except ValueError as e:raise ValueError(f"output outside new root: {p}") from e
    return p

@contextmanager
def unique_lock(path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError("suite driver already active") from e
        try:
            f.seek(0);f.truncate()
            f.write(json.dumps({"pid":os.getpid(),"time":time.time()}));f.flush()
            yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def receipt_valid(receipt: dict, storage: Path) -> bool:
    if receipt.get("status") not in DONE_STATES:
        return False
    outputs=receipt.get("outputs",[])
    if not outputs:
        return False
    for item in outputs:
        path=within(Path(item["path"]),storage)
        if not path.is_file() or sha256(path) != item["sha256"]:
            return False
    return True

def fingerprint(task: dict) -> str:
    # Must include actual source/data/config hashes supplied by the Agent.
    keep={k:task.get(k) for k in ("id","argv","input_hashes","training_run_id","group")}
    return hashlib.sha256(json.dumps(keep,sort_keys=True).encode()).hexdigest()

def order_pending(tasks: list[dict], priority: str) -> list[dict]:
    """Reorder only the contiguous CAL/REC family block, preserve within-group order."""
    family={"calibration","recurrent"}
    positions=[i for i,t in enumerate(tasks) if t.get("group") in family]
    if not positions:
        return tasks
    first,last=min(positions),max(positions)
    if any(tasks[i].get("group") not in family for i in range(first,last+1)):
        raise ValueError("CAL/REC tasks must form one contiguous family block")
    groups=(["calibration","recurrent"] if priority=="CALIBRATION_THEN_RECURRENT"
            else ["recurrent","calibration"])
    middle=[t for g in groups for t in tasks[first:last+1] if t.get("group")==g]
    return tasks[:first]+middle+tasks[last+1:]

def run_manifest(manifest_path: Path) -> int:
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("execution_authorized") is not True:
        raise PermissionError("record explicit authorization before run-all")
    storage=check_new_root(m["storage_root"])
    run=within(Path(m["run_root"]),storage)
    cwd=within(Path(m["repo_root"]),storage)
    run.mkdir(parents=True,exist_ok=True)
    if not cwd.is_dir():
        raise ValueError("repo_root missing")
    tasks=list(m["tasks"])
    ids=[t["id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids")
    training_ids={t["training_run_id"] for t in tasks if t.get("training_run_id")}
    if len(training_ids)>int(m.get("max_new_training_runs",9)):
        raise ValueError("new training cap exceeded")
    state_path=run/"stage_state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {"stages":{}}
    priority="CALIBRATION_THEN_RECURRENT"
    env=os.environ.copy()
    for key,name in (("TMPDIR","tmp"),("XDG_CACHE_HOME","cache"),("TORCH_HOME","torch")):
        d=run/name;d.mkdir(exist_ok=True);env[key]=str(d)
    env["PYTHONUNBUFFERED"]="1"
    env.update({k:str(v) for k,v in m.get("env",{}).items()})
    blocked=False
    with unique_lock(run/".suite.lock"):
        pending=tasks[:]
        while pending:
            if pending[0].get("kind")=="order_cal_rec":
                t=pending.pop(0)
                pf=within(Path(m["priority_receipt"]),storage)
                if pf.is_file():
                    pr=json.loads(pf.read_text())
                    if pr.get("priority") in {"CALIBRATION_THEN_RECURRENT","RECURRENT_THEN_CALIBRATION"}:
                        priority=pr["priority"]
                pending=order_pending(pending,priority)
                state["stages"][t["id"]]={"status":"DONE","priority":priority,
                                          "fingerprint":fingerprint(t)}
                atomic_json(state_path,state)
                continue
            task=pending.pop(0)
            tid=task["id"];fp=fingerprint(task)
            prior=state["stages"].get(tid)
            if prior:
                if prior.get("fingerprint") != fp:
                    raise ValueError(f"semantic task fingerprint changed: {tid}")
                if prior.get("status") in DONE_STATES:
                    if receipt_valid(prior.get("receipt",{}),storage):
                        continue
                    raise ValueError(f"completed artifacts invalid: {tid}")
                if prior.get("status") in ALLOWED_SKIPS|LOCAL_STATES:
                    continue  # No silent reattempt. Explicit repair needs a new documented manifest.
            missing=[d for d in task.get("requires",[])
                     if state["stages"].get(d,{}).get("status") not in DONE_STATES]
            if missing:
                state["stages"][tid]={"status":"BLOCKED_DEPENDENCY","requires":missing,"fingerprint":fp}
                atomic_json(state_path,state)
                continue
            argv=task.get("argv",[])
            if not argv or not all(isinstance(a,str) for a in argv):
                raise ValueError(f"stage requires argument-list command: {tid}")
            receipt_path=within(Path(task["receipt"]),storage)
            # A valid crash-completed receipt is reusable; an invalid stale one is not.
            if receipt_path.is_file():
                old=json.loads(receipt_path.read_text())
                if old.get("input_fingerprint")==fp and receipt_valid(old,storage):
                    state["stages"][tid]={"status":old["status"],"receipt":old,"fingerprint":fp}
                    atomic_json(state_path,state)
                    continue
            state["stages"][tid]={"status":"RUNNING","fingerprint":fp,"started_at":time.time()}
            atomic_json(state_path,state)
            logfile=run/"driver_logs"/f"{tid}.log";logfile.parent.mkdir(exist_ok=True)
            returncode = 127
            # a training stage resumes from its own epoch snapshot after a crash (at most 3 attempts = 2 resumes); other stages run once
            for attempt in range(3 if task.get("training_run_id") else 1):
                returncode = 127
                with logfile.open("ab") as log:
                    try:
                        proc=subprocess.run(argv,cwd=str(cwd),env={**env,"TASK_FINGERPRINT":fp},
                                            stdout=log,stderr=subprocess.STDOUT,check=False)
                        returncode = proc.returncode
                    except OSError as exc:
                        log.write((f"STAGE_SPAWN_ERROR: {exc}\\n").encode())
                if returncode == 0:
                    break
            receipt={}
            try:
                if receipt_path.is_file():
                    receipt=json.loads(receipt_path.read_text())
            except (OSError,json.JSONDecodeError):
                receipt={}
            status=receipt.get("status","TECHNICAL_INCOMPLETE")
            if status not in ALL_TERMINAL:
                status="TECHNICAL_INCOMPLETE"
            if status in DONE_STATES and (returncode or
                 receipt.get("input_fingerprint")!=fp or not receipt_valid(receipt,storage)):
                status="TECHNICAL_INCOMPLETE"
            if task.get("failure_scope")=="global" and status not in DONE_STATES:
                status="GLOBAL_INTEGRITY_FAILURE"
            state["stages"][tid]={"status":status,"fingerprint":fp,"receipt":receipt,
                                  "returncode":returncode,"finished_at":time.time(),
                                  "log":str(logfile)}
            atomic_json(state_path,state)
            if status in GLOBAL_STATES:
                blocked=True
                break
        statuses=[v["status"] for v in state["stages"].values()]
        final=("C1_METHOD_SERIAL_SUITE_BLOCKED" if blocked else
               "C1_METHOD_SERIAL_SUITE_PARTIAL_TECHNICAL" if any(s in LOCAL_STATES for s in statuses) else
               "C1_METHOD_SERIAL_SUITE_COMPLETE_WITH_SKIPS" if any(s in ALLOWED_SKIPS for s in statuses) else
               "C1_METHOD_SERIAL_SUITE_COMPLETE")
        atomic_json(run/"driver_receipt.json",{
            "state":final,"priority":priority,"stages":state["stages"],
            "planned_training_ids":sorted(training_ids),"finished_at":time.time(),
            "note":"Driver status only. Scientific final_summary/claim_boundary must be produced by report stage."})
    return 2 if blocked else (1 if any(s in LOCAL_STATES for s in statuses) else 0)

def safe_cli_run(manifest: Path) -> int:
    """Write an emergency receipt on uncaught global/configuration errors when safe."""
    try:
        return run_manifest(manifest)
    except (Exception, KeyboardInterrupt) as exc:
        error = {"state":"C1_METHOD_SERIAL_SUITE_BLOCKED",
                 "exception":type(exc).__name__, "message":str(exc),
                 "time":time.time(), "complete":False}
        try:
            data=json.loads(manifest.read_text(encoding="utf-8"))
            storage=check_new_root(data["storage_root"])
            run=within(Path(data["run_root"]),storage)
            atomic_json(run/"driver_emergency_receipt.json",error)
        except Exception:
            # Never write to an unverified/legacy destination merely to save a log.
            pass
        print(json.dumps(error,ensure_ascii=False),file=sys.stderr)
        return 130 if isinstance(exc,KeyboardInterrupt) else 2

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",type=Path)
    args=p.parse_args()
    raise SystemExit(safe_cli_run(args.manifest))
