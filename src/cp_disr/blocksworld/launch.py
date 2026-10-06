"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1 registration, ledger and training launch (runbook 12, 16, 21). Import-light: torch is imported only when a run starts."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import yaml

CARD = "CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1"
OUT_ROOT_REL = Path("runs/final_master/c1_route_b/blocksworld_main_v1")
PREP_REL = OUT_ROOT_REL / "prep"
TRAIN_SPLIT_REL = "configs/splits/c1_bw_train_dev_v1.json"
EVAL_SPLIT_NAMES = ("c1_bw_a0_iso_v1.json", "c1_bw_a1_color_reverse_v1.json", "c1_bw_a2_noniso_v1.json", "c1_bw_b_scale_v1.json")
RUN_IDS = ("R-C1-BW-B2-0", "R-C1-BW-QMARK-0", "R-C1-BW-ASNET-0")
MIN_FREE_BYTES = 20 * 1024 ** 3


class LaunchError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def check_source_identity(root, prep_commit):
    """HEAD must be the registered prep commit and the tracked tree clean (untracked results are fine)."""
    head = git(root, "rev-parse", "HEAD")
    if head != prep_commit:
        raise LaunchError("source drift: HEAD %s != registered prep commit %s" % (head, prep_commit))
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise LaunchError("source drift: tracked tree is dirty")
    return head


def open_training_split(path):
    """The ONLY way a training code path may read a split file: the train/dev file. Any A0 / A1 / A2 / B file is refused."""
    name = Path(path).name
    if name in EVAL_SPLIT_NAMES or name != Path(TRAIN_SPLIT_REL).name:
        raise LaunchError("a training code path may not open %s" % name)
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    lock = path.with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_json_atomic(path, doc):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def register(root, prep_commit, authorization_text, runs, ppo, half_life, stamp=None):
    root = Path(root).resolve()
    check_source_identity(root, prep_commit)
    split = root / TRAIN_SPLIT_REL
    doc = open_training_split(split)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = root / OUT_ROOT_REL / ("%s_%s" % (stamp, prep_commit[:8]))
    if run_root.exists():
        raise LaunchError("already registered: %s" % run_root)
    for sub in ("registration", "runs", "eval", "results", "receipts"):
        (run_root / sub).mkdir(parents=True)
    auth = {"card": CARD, "prep_commit": prep_commit, "amendment": "CP-DISR-C1-BW-HOP-CLOSEOUT-A01", "authorization_text": authorization_text,
            "authorization_text_sha256": hashlib.sha256(authorization_text.encode()).hexdigest(), "authorized_training_runs": 3, "not_authorized": ["extra seeds", "imitation", "new algorithm", "retry of a started run"],
            "registered": utc_now()}
    write_json_atomic(run_root / "registration" / "authorization.json", auth)
    configs, ledger_runs = {}, {}
    for rid in RUN_IDS:
        spec = runs[rid]
        cfg = {"run_id": rid, "method": spec["method"], "seed": spec["seed"], "train_dev_split": str(split), "train_dev_split_sha256": sha256_file(split), "out_dir": str(run_root / "runs" / rid),
               "prep_commit": prep_commit, "half_life": doc["half_life"], "rollout_n": ppo["rollout_n"], "Ncap": ppo["Ncap"], "max_updates": ppo["max_updates"], "Tcap_wall_seconds": ppo["Tcap_wall_seconds"],
               "eval_points": ppo["eval_points"]}
        path = run_root / "registration" / (rid + ".yaml")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        os.chmod(path, 0o444)
        configs[rid] = {"path": str(path), "sha256": sha256_file(path)}
        ledger_runs[rid] = {"method": spec["method"], "seed": spec["seed"], "status": "NOT_STARTED", "config": configs[rid]}
    write_json_atomic(run_root / "registration" / "launch_state.json", {"card": CARD, "attempts_used": 0, "attempts_cap": 3, "runs": ledger_runs, "created": utc_now()})
    write_json_atomic(run_root / "registration" / "launch_plan.json", {"card": CARD, "prep_commit": prep_commit, "configs": configs, "run_root": str(run_root)})
    return run_root


def _state_path(run_root):
    return Path(run_root) / "registration" / "launch_state.json"


def reserve(run_root, run_id, pid, gpu):
    sp = _state_path(run_root)
    with locked(sp):
        st = json.loads(sp.read_text())
        r = st["runs"][run_id]
        if r["status"] != "NOT_STARTED":
            raise LaunchError("duplicate start refused: %s is %s" % (run_id, r["status"]))
        if st["attempts_used"] >= st["attempts_cap"]:
            raise LaunchError("attempt cap reached")
        r.update({"status": "RUNNING", "pid": pid, "gpu": gpu, "started": utc_now()})
        st["attempts_used"] += 1
        write_json_atomic(sp, st)


def finish(run_root, run_id, status, **extra):
    sp = _state_path(run_root)
    with locked(sp):
        st = json.loads(sp.read_text())
        st["runs"][run_id].update({"status": status, "finished": utc_now(), **extra})
        write_json_atomic(sp, st)


def train(root, run_root, run_id, gpu):
    """Run one registered training. CUDA_VISIBLE_DEVICES must already pin the GPU (set by the CLI before torch is imported)."""
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    plan = json.loads((run_root / "registration" / "launch_plan.json").read_text())
    check_source_identity(root, plan["prep_commit"])
    entry = plan["configs"][run_id]
    if sha256_file(entry["path"]) != entry["sha256"]:
        raise LaunchError("run config changed after registration")
    cfg = yaml.safe_load(Path(entry["path"]).read_text())
    if sha256_file(cfg["train_dev_split"]) != cfg["train_dev_split_sha256"]:
        raise LaunchError("train split changed after registration")
    free = os.statvfs(str(run_root)).f_bavail * os.statvfs(str(run_root)).f_frsize
    if free < MIN_FREE_BYTES:
        raise LaunchError("storage start gate: %d bytes free" % free)
    reserve(run_root, run_id, os.getpid(), gpu)
    cfg = dict(cfg, device="cuda:0")
    try:
        from .train import run_training
        acct = run_training(cfg)
        finish(run_root, run_id, "COMPLETE", accounting=acct)
        return acct
    except BaseException as exc:                                    # recorded, never swallowed; no automatic rerun
        finish(run_root, run_id, "STOPPED", error="%s: %s" % (type(exc).__name__, exc), traceback=traceback.format_exc()[-3000:])
        raise


def status(run_root):
    st = json.loads(_state_path(run_root).read_text())
    return {"attempts_used": st["attempts_used"], "runs": {k: {x: v.get(x) for x in ("method", "status", "gpu", "pid", "started", "finished")} for k, v in st["runs"].items()}}
