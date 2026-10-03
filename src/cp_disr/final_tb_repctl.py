"""CP-DISR-TB-REP-CONTROLS-01 launch binding: Absolute Successor and Matched Neural Composition on frozen T_B.

This module adds no trainer and changes no scientific or runtime semantics. It re-uses the frozen production path
unchanged (``final_tb.run_train`` -> ``stage2a_v11.train_job``) and only adds what the six pre-registered attempts need:

* a plan table of exactly six attempts (B2-ABS and B1-K+NC, seeds 0/1/2, one attempt each, no retries),
* a concurrency-3 ledger (the baseline ledger is capped at two workers and three/four attempts),
* a pre-training engineering smoke (<= 2 episodes, no optimizer, no provider, no test split),
* release tokens bound to the prep commit and to the prep receipts, and a restartable supervisor.

Import-light: nothing here imports torch or the simulator at module import time.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import final_tb as ftb
from .common import BindingError, canonical

CARD = "CP-DISR-TB-REP-CONTROLS-01"
BASE_COMMIT = "9422b837cf1dfc43afc056a77bdb59d63512b3b6"
TASK = ftb.TASK
PREP_REL = Path("runs/final_master/2.1.1/repctl/prep")
REGISTRATION_ROOT_REL = Path("runs/final_master/2.1.1/repctl")
RUN_ROOT_REL = Path("runs/final_master/2.1.1/T_B")
CLI = "final_tb_repctl_launch.py"

# Frozen before any RL run. queue_order is the only scheduling rule; no result may change it.
PLAN_TABLE = {
    "R-TB-ABS-0": {"method": "B2-ABS", "seed": 0, "queue_order": 1},
    "R-TB-NC-0": {"method": "B1-K+NC", "seed": 0, "queue_order": 2},
    "R-TB-ABS-1": {"method": "B2-ABS", "seed": 1, "queue_order": 3},
    "R-TB-NC-1": {"method": "B1-K+NC", "seed": 1, "queue_order": 4},
    "R-TB-ABS-2": {"method": "B2-ABS", "seed": 2, "queue_order": 5},
    "R-TB-NC-2": {"method": "B1-K+NC", "seed": 2, "queue_order": 6},
}
METHODS = ("B1-K+NC", "B2-ABS")
MAX_WORKERS = 3
MAX_ATTEMPTS = len(PLAN_TABLE)
GPU_CANDIDATES = (0, 1, 2, 3, 4, 5, 6)
GPU_MAX_USED_MIB = 1500
GPU_MAX_UTIL = 20
START_FREE_MIN_BYTES = 60 * 1024 ** 3
SMOKE_CASE = "T_B_dev_00"
SMOKE_MAX_SKILLS_PER_EPISODE = 3
SMOKE_CAPS = {"episodes": 2, "provider_requests": 0, "optimizer_steps": 0, "test_ids_accessed": 0}
NOT_AUTHORIZED = [
    "provider", "VLM or prior", "T_P", "Family B", "RoboCasa", "planner", "multi-step rollout", "learned world model",
    "test30", "independent holdout", "extra seeds", "structural generalization", "new method designed after seeing results",
    "a retry or replacement of any started attempt", "a change of reward, PPO, Tcap, task, candidate set or evaluator",
]
PREP_RECEIPTS = (
    "representation_control_spec.json", "information_boundary.md", "representation_capacity_audit.json",
    "semantic_test_receipt.json", "mutation_test_receipt.json", "legacy_equivalence_receipt.json", "smoke_receipt.json",
    "baseline_identity.json",
)


def utc_now() -> str:
    return ftb.utc_now()


def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- plan / context
def validate_assignment(plan_id, method, seed) -> dict:
    if plan_id not in PLAN_TABLE:
        raise BindingError("unknown plan_id %r" % (plan_id,))
    row = PLAN_TABLE[plan_id]
    if method != row["method"]:
        raise BindingError("%s is %s, not %s" % (plan_id, row["method"], method))
    if isinstance(seed, bool) or not isinstance(seed, int) or int(seed) != row["seed"]:
        raise BindingError("%s requires training_seed=%s, got %r" % (plan_id, row["seed"], seed))
    return dict(row)


@dataclass(frozen=True)
class RepctlRunContext(ftb.RunContext):
    """`final_tb.RunContext` whose worker assertion uses this card's plan table (the baseline table is untouched)."""

    def assert_worker(self, task_id, method):
        if task_id != self.task_id:
            raise BindingError("run context is bound to %s, not %s" % (self.task_id, task_id))
        if method != self.method:
            raise BindingError("run context is bound to %s, not %s" % (self.method, method))
        validate_assignment(self.plan_id, self.method, self.training_seed)


def context_from_dict(cfg: dict) -> RepctlRunContext:
    missing = [k for k in ftb.REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise BindingError("run config missing keys: %s" % missing)
    for k, v in cfg.items():
        if ftb.is_placeholder(v):
            raise BindingError("unresolved placeholder in run config: %s=%r" % (k, v))
    validate_assignment(cfg["plan_id"], cfg["method"], cfg["training_seed"])
    for k in ("output_directory", "runtime_manifest", "train_split"):
        if not os.path.isabs(str(cfg[k])):
            raise BindingError("%s must be an absolute path" % k)
    if re.fullmatch(r"[0-9a-f]{40}", str(cfg["source_commit"])) is None:
        raise BindingError("source_commit must be a full commit sha")
    if str(cfg["prior_mode"]) != ftb.PRIOR_MODE:
        raise BindingError("prior_mode must be absent")
    if int(cfg["study_envelope_ncap"]) != ftb.N_CAP or float(cfg["study_envelope_tcap"]) != ftb.TCAP_SECONDS:
        raise BindingError("study envelope must equal the frozen Ncap/Tcap")
    if str(cfg["evaluation_rule"]) != ftb.EVALUATION_RULE:
        raise BindingError("evaluation_rule must be %r" % ftb.EVALUATION_RULE)
    if str(cfg.get("task_id", TASK)) != TASK:
        raise BindingError("task_id must be T_B")
    allowed = set(ftb.REQUIRED_CONFIG_KEYS) | {"task_id", "render_gpu_device_id", "physical_gpu_index", "gpu_uuid"}
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise BindingError("unknown run-config keys: %s" % unknown)
    return RepctlRunContext(
        plan_id=cfg["plan_id"], attempt_id=str(cfg["attempt_id"]), method=cfg["method"], training_seed=int(cfg["training_seed"]),
        source_commit=cfg["source_commit"], output_directory=str(cfg["output_directory"]), runtime_manifest=str(cfg["runtime_manifest"]),
        train_split=str(cfg["train_split"]), prior_mode=ftb.PRIOR_MODE, study_envelope_ncap=ftb.N_CAP, study_envelope_tcap=ftb.TCAP_SECONDS,
        evaluation_rule=ftb.EVALUATION_RULE, task_id=TASK, render_gpu_device_id=int(cfg.get("render_gpu_device_id", 0)),
        physical_gpu_index=None if cfg.get("physical_gpu_index") is None else int(cfg["physical_gpu_index"]), gpu_uuid=cfg.get("gpu_uuid"),
    )


def load_run_config(path) -> RepctlRunContext:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise BindingError("run config must be a mapping")
    return context_from_dict(cfg)


# ----------------------------------------------------------------------------- ledger
class RepctlLedger(ftb.Ledger):
    """Six attempts, three concurrent workers, one attempt per plan; a started attempt is never removed or renumbered."""

    def init(self, plans: dict) -> dict:
        with self.locked():
            if self.state_path.exists():
                raise BindingError("ledger already initialised: %s" % self.state_path)
            state = {"created": utc_now(), "card": CARD, "new_rl_attempts_used": 0, "new_rl_attempts_cap": MAX_ATTEMPTS,
                     "max_concurrent_workers": MAX_WORKERS, "plans": {pid: dict(v) for pid, v in plans.items()}}
            ftb.write_json_atomic(self.state_path, state)
            return state

    def reserve(self, plan_id, attempt_id, run_dir, pid, physical_gpu) -> dict:
        with self.locked():
            state = self.read()
            if plan_id not in state["plans"] or plan_id not in PLAN_TABLE:
                raise BindingError("plan %s is not registered" % plan_id)
            plan = state["plans"][plan_id]
            if plan["status"] != "NOT_STARTED":
                raise BindingError("duplicate start refused: %s is %s" % (plan_id, plan["status"]))
            if state["new_rl_attempts_used"] >= state["new_rl_attempts_cap"]:
                raise BindingError("new RL attempt cap reached")
            running = [p for p, v in state["plans"].items() if v["status"] == "RUNNING"]
            if len(running) >= MAX_WORKERS:
                raise BindingError("at most %d concurrent workers" % MAX_WORKERS)
            for other, v in state["plans"].items():
                if other != plan_id and v.get("run_dir") == str(run_dir):
                    raise BindingError("run directory already reserved by %s" % other)
                if other != plan_id and physical_gpu is not None and v["status"] == "RUNNING" and v.get("physical_gpu") == physical_gpu:
                    raise BindingError("GPU %s already holds a running worker (%s)" % (physical_gpu, other))
            self.reservations.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.reservations / (plan_id + ".json")), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"plan_id": plan_id, "attempt_id": attempt_id, "run_dir": str(run_dir), "pid": pid,
                                    "physical_gpu": physical_gpu, "time": utc_now()}, sort_keys=True) + "\n")
            plan.update({"status": "RUNNING", "attempt_id": attempt_id, "run_dir": str(run_dir), "pid": pid,
                         "physical_gpu": physical_gpu, "started": utc_now()})
            state["new_rl_attempts_used"] += 1
            ftb.write_json_atomic(self.state_path, state)
            return plan


# ----------------------------------------------------------------------------- prep receipts
def source_files(root) -> dict:
    root = Path(root)
    rels = [
        "src/cp_disr/neural.py", "src/cp_disr/stage2a_v11.py", "src/cp_disr/repctl_policy.py", "src/cp_disr/tb_repctl_checks.py", "src/cp_disr/final_tb_repctl.py",
        "src/cp_disr/final_tb.py", "src/cp_disr/torch_rl.py", "src/cp_disr/rl.py", "src/cp_disr/collector.py",
        "src/cp_disr/phase_a_v12.py", "src/cp_disr/graph.py", "src/cp_disr/contracts.py", "src/cp_disr/persistence.py",
        "src/cp_disr/platforms/libero/runtime_factory.py", "scripts/final_tb_repctl_launch.py", "scripts/tb_repctl_receipts.py",
    ]
    return {r: (ftb.sha256_file(root / r) if (root / r).is_file() else None) for r in rels}


def verify_prep_receipts(root, prep_dir=None, source_commit=None) -> dict:
    """Every prep receipt exists, passes and is bound to the files that will be trained with."""
    root = Path(root)
    prep_dir = Path(prep_dir) if prep_dir else root / PREP_REL
    missing = [f for f in PREP_RECEIPTS if not (prep_dir / f).is_file()]
    if missing:
        raise BindingError("prep receipts missing: %s" % missing)
    spec = _json(prep_dir / "representation_control_spec.json")
    if spec.get("card") != CARD or spec.get("base_commit") != BASE_COMMIT:
        raise BindingError("spec is not this card's / base commit")
    if spec.get("plan_table") != PLAN_TABLE:
        raise BindingError("spec plan table differs from the frozen plan table")
    now = {k: v for k, v in source_files(root).items() if k not in ("src/cp_disr/final_tb_repctl.py", "scripts/final_tb_repctl_launch.py", "scripts/tb_repctl_receipts.py")}
    frozen = {k: v for k, v in spec["source_files_sha256"].items() if k in now}
    drift = {k: (frozen.get(k), v) for k, v in now.items() if frozen.get(k) != v}
    if drift:
        raise BindingError("source drifted after the prep receipts were produced: %s" % sorted(drift))
    verdicts = {}
    for f in ("semantic_test_receipt.json", "mutation_test_receipt.json", "legacy_equivalence_receipt.json", "smoke_receipt.json", "representation_capacity_audit.json"):
        doc = _json(prep_dir / f)
        verdicts[f] = doc.get("verdict")
        if doc.get("verdict") != "PASS":
            raise BindingError("%s verdict is %r, not PASS" % (f, doc.get("verdict")))
    return {"prep_dir": str(prep_dir), "verdicts": verdicts,
            "receipt_sha256": {f: ftb.sha256_file(prep_dir / f) for f in PREP_RECEIPTS}}


def configure_v11(root, ctx):
    """`final_tb.configure_v11` plus the in-process registration of the two controls.

    neural.py and stage2a_v11.py stay byte-identical to the base commit (protected production files). For this process only:
    the control names join ``neural.METHODS`` (so ``canonical_method`` / ``method_dir_name`` accept them), join
    ``EMPTY_PRIOR_METHODS`` (absent prior, no sampler, exactly like B2 and B1-K+E) and ``make_policy`` builds ``RepctlPolicy``
    for them. Every legacy method still goes through the untouched production factory.
    """
    from . import neural, repctl_policy
    v11 = ftb.configure_v11(root, ctx)
    neural.METHODS = tuple(dict.fromkeys(tuple(neural.METHODS) + METHODS))
    v11.EMPTY_PRIOR_METHODS = set(v11.EMPTY_PRIOR_METHODS) | set(METHODS)
    if not hasattr(v11.make_policy, "__wrapped_production__"):
        v11.make_policy = repctl_policy.make_control_policy(v11, v11.make_policy)
    return v11


# ----------------------------------------------------------------------------- registration / release
def fs_free(path) -> int:
    return ftb.fs_free_bytes(path)


def start_gate(path, free_fn=None) -> dict:
    free = (free_fn or fs_free)(path)
    if free < START_FREE_MIN_BYTES:
        raise BindingError("storage start gate: %d bytes free < %d" % (free, START_FREE_MIN_BYTES))
    return {"free_bytes": int(free), "min_required": START_FREE_MIN_BYTES}


def register(root, out_dir, prep_commit, authorization_text, stamp=None, git=ftb.git_out, prep_dir=None) -> dict:
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    ftb.check_source_identity(root, prep_commit, git=git)
    receipts = verify_prep_receipts(root, prep_dir)
    if (out_dir / "launch_state.json").exists():
        raise BindingError("already registered: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    auth = {"card": CARD, "prep_commit": prep_commit, "base_commit": BASE_COMMIT, "plan_table": PLAN_TABLE, "queue": sorted(PLAN_TABLE, key=lambda p: PLAN_TABLE[p]["queue_order"]),
            "authorization_text": authorization_text, "authorization_text_sha256": ftb.sha256_text(authorization_text),
            "caps": {"rl_attempts": MAX_ATTEMPTS, "max_concurrent_workers": MAX_WORKERS, "ncap": ftb.N_CAP, "tcap": ftb.TCAP_SECONDS, "provider": 0,
                     "tp_training": 0, "formal_test": 0, "test30_read": 0},
            "prep_receipts": receipts, "not_authorized": NOT_AUTHORIZED, "registered": utc_now()}
    ftb.write_json_atomic(out_dir / "authorization.json", auth)
    split = ftb.derive_noprior_split(root, out_dir / "train_split_tb_noprior.json")
    manifest = ftb.derive_runtime_manifest(root, out_dir / "runtime_manifest_tb_resolved.yaml", split["path"])
    profile = ftb.resolve_frozen_profile(root, manifest["path"])
    splits = ftb.split_lists(root)
    ledger_plans, configs = {}, {}
    (out_dir / "run_configs").mkdir(exist_ok=True)
    for plan_id, row in sorted(PLAN_TABLE.items(), key=lambda kv: kv[1]["queue_order"]):
        attempt = "%s-%s-%s" % (plan_id, stamp, prep_commit[:8])
        run_dir = root / RUN_ROOT_REL / row["method"] / ("seed_%d" % row["seed"]) / attempt
        cfg = {"plan_id": plan_id, "attempt_id": attempt, "method": row["method"], "training_seed": row["seed"], "source_commit": prep_commit,
               "output_directory": str(run_dir), "runtime_manifest": manifest["path"], "train_split": split["path"], "prior_mode": ftb.PRIOR_MODE,
               "study_envelope_ncap": ftb.N_CAP, "study_envelope_tcap": ftb.TCAP_SECONDS, "evaluation_rule": ftb.EVALUATION_RULE}
        context_from_dict(cfg)
        cfg_path = out_dir / "run_configs" / (plan_id + ".yaml")
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        os.chmod(cfg_path, 0o444)
        configs[plan_id] = {"path": str(cfg_path), "sha256": ftb.sha256_file(cfg_path)}
        ledger_plans[plan_id] = {"method": row["method"], "seed": row["seed"], "status": "NOT_STARTED", "attempt_id": attempt,
                                 "planned_run_dir": str(run_dir), "queue_order": row["queue_order"]}
    RepctlLedger(out_dir).init(ledger_plans)
    plan = {"card": CARD, "prep_commit": prep_commit, "stamp": stamp, "plans": ledger_plans, "configs": configs,
            "derived": {"runtime_manifest": manifest, "train_split": split}, "frozen_runtime": ftb.FROZEN_RUNTIME, "frozen_profile": profile,
            "envelope": ftb.envelope(), "eval_points": list(ftb.EVAL_POINTS), "train_ids_sha256": ftb.sha256_text(canonical(splits["train"])),
            "dev_ids_sha256": ftb.sha256_text(canonical(splits["dev"])), "split_file_sha256": splits["sha256"],
            "queue": auth["queue"], "init": "FROM_SCRATCH", "warm_start": False}
    ftb.write_json_atomic(out_dir / "launch_plan.json", plan)
    return plan


def release(root, out_dir, prep, free_fn=None, git=ftb.git_out) -> list:
    """One token per plan, issued together: all six seeds are frozen before any run starts."""
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    ftb.check_source_identity(root, prep, git=git)
    receipts = verify_prep_receipts(root)
    auth = _json(out_dir / "authorization.json")
    if auth.get("card") != CARD or auth.get("prep_commit") != prep:
        raise BindingError("registration is not this card's / prep commit")
    state = RepctlLedger(out_dir).read()
    if any(v["status"] != "NOT_STARTED" for v in state["plans"].values()):
        raise BindingError("tokens are issued once, before any run starts")
    gate = start_gate(next(iter(state["plans"].values()))["planned_run_dir"], free_fn)
    evidence = {"prep_receipts": receipts["receipt_sha256"], "storage_start_gate": gate, "tokens_issued_together": True}
    return [ftb.issue_token(out_dir, "train", prep, plan_id, evidence) for plan_id in sorted(PLAN_TABLE, key=lambda p: PLAN_TABLE[p]["queue_order"])]


# ----------------------------------------------------------------------------- train
def run_repctl_train(root, out, plan_id, gpu, token_path, git=ftb.git_out, free_fn=None, runner=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    auth = _json(out / "authorization.json")
    prep = auth["prep_commit"]
    if auth.get("card") != CARD:
        raise BindingError("registration is not this card's")
    ftb.verify_token(token_path, out, "train", prep, plan_id)
    ftb.check_source_identity(root, prep, git=git)
    verify_prep_receipts(root)
    launch = _json(out / "launch_plan.json")
    cfg_entry = launch["configs"][plan_id]
    cfg_path = Path(cfg_entry["path"])
    if ftb.sha256_file(cfg_path) != cfg_entry["sha256"]:
        raise BindingError("run config changed after registration")
    gate = start_gate(launch["plans"][plan_id]["planned_run_dir"], free_fn)
    ftb.bind_worker_gpu(gpu)
    uuid = ftb.query_gpu_uuid(gpu)
    os.chdir(str(root))
    base = load_run_config(cfg_path)
    ctx = dataclasses.replace(base, physical_gpu_index=int(gpu), render_gpu_device_id=int(gpu), gpu_uuid=uuid)
    if ctx.plan_id != plan_id:
        raise BindingError("run config is not %s" % plan_id)
    ftb.write_json_atomic(out / ("launch_evidence_%s.json" % plan_id), {"time": utc_now(), "pid": os.getpid(), "gpu": int(gpu), "gpu_uuid": uuid,
                                                                       "start_gate": gate, "prep_commit": prep, "init": "FROM_SCRATCH", "warm_start": False})
    v11 = configure_v11(root, ctx)
    return (runner or ftb.run_train)(root, out, ctx, gpu, v11_module=v11, ledger=RepctlLedger(out))


# ----------------------------------------------------------------------------- supervisor
def gpu_table(runner=subprocess.check_output) -> dict:
    text = runner(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    table = {}
    for line in text.strip().splitlines():
        idx, used, util = [x.strip() for x in line.split(",")]
        table[int(idx)] = {"memory_used_mib": int(used), "utilization": int(util)}
    return table


def pick_gpu(table: dict, taken) -> int | None:
    for idx in GPU_CANDIDATES:
        row = table.get(idx)
        if idx in taken or row is None:
            continue
        if row["memory_used_mib"] <= GPU_MAX_USED_MIB and row["utilization"] <= GPU_MAX_UTIL:
            return idx
    return None


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    state = Path("/proc/%d/stat" % int(pid))
    try:
        return state.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def supervise(root, out, python=sys.executable, poll_seconds=60.0, max_ticks=None, gpu_fn=gpu_table, spawn=None, sleep=time.sleep, log=print):
    """Start the six pre-registered plans in the frozen queue order, at most three at a time. Never kills, restarts or reruns."""
    root, out = Path(root).resolve(), Path(out).resolve()
    ledger = RepctlLedger(out)
    launch = _json(out / "launch_plan.json")
    queue = launch["queue"]
    state_path = out / "supervisor_state.json"
    sup = _json(state_path) if state_path.is_file() else {"spawned": {}, "halted": None, "events": []}

    def save():
        ftb.write_json_atomic(state_path, sup)

    def event(msg, **kw):
        sup["events"].append({"time": utc_now(), "msg": msg, **kw})
        save()
        log("[repctl-supervisor] %s %s" % (msg, kw if kw else ""))

    spawn = spawn or (lambda cmd, logfile: subprocess.Popen(cmd, stdout=open(logfile, "ab"), stderr=subprocess.STDOUT, start_new_session=True,
                                                            env={**os.environ, "PYTHONPATH": str(root / "src")}, cwd=str(root)))
    procs = {}
    ticks = 0
    while True:
        ticks += 1
        state = ledger.read()
        statuses = {p: v["status"] for p, v in state["plans"].items()}
        ours = {p: sup["spawned"][p] for p in sup["spawned"]}
        alive = [p for p, meta in ours.items() if (procs[p].poll() is None if p in procs else _alive(meta.get("pid")))]
        for p in list(ours):
            if p not in alive and statuses.get(p) == "RUNNING":
                event("worker exited while the ledger still says RUNNING", plan=p)
                sup["halted"] = sup["halted"] or "worker %s vanished" % p
            if p not in alive and statuses.get(p) == "NOT_STARTED" and not sup["halted"]:
                sup["halted"] = "worker %s exited before reserving its plan" % p
                event("halting new workers", reason=sup["halted"])
            if statuses.get(p) == "STOPPED" and not sup["halted"]:
                sup["halted"] = "plan %s STOPPED: %s" % (p, state["plans"][p].get("error") or state["plans"][p].get("reason"))
                event("halting new workers", reason=sup["halted"])
        pending = [p for p in queue if statuses[p] == "NOT_STARTED" and p not in sup["spawned"]]
        if not pending and not alive:
            event("all plans finished", statuses=statuses)
            return {"statuses": statuses, "halted": sup["halted"]}
        if sup["halted"] and not alive:
            event("halted with no live workers", statuses=statuses, reason=sup["halted"])
            return {"statuses": statuses, "halted": sup["halted"]}
        if pending and not sup["halted"] and len(alive) < MAX_WORKERS:
            plan_id = pending[0]
            taken = {state["plans"][p].get("physical_gpu") for p in alive if state["plans"][p].get("physical_gpu") is not None} | {m["gpu"] for p, m in ours.items() if p in alive}
            gpu = pick_gpu(gpu_fn(), taken)
            if gpu is None:
                event("no idle GPU; waiting", plan=plan_id)
            else:
                (out / "logs").mkdir(exist_ok=True)
                cmd = [python, str(root / "scripts" / CLI), "train", "--root", str(root), "--out", str(out), "--plan", plan_id, "--gpu", str(gpu),
                       "--token", str(out / "release_tokens" / ("train_%s.json" % plan_id))]
                proc = spawn(cmd, str(out / "logs" / ("train_%s.log" % plan_id)))
                procs[plan_id] = proc
                sup["spawned"][plan_id] = {"pid": proc.pid, "gpu": gpu, "time": utc_now(), "cmd": cmd}
                event("spawned worker", plan=plan_id, gpu=gpu, pid=proc.pid)
        if max_ticks is not None and ticks >= max_ticks:
            return {"statuses": statuses, "halted": sup["halted"], "ticks": ticks}
        sleep(poll_seconds)


# ----------------------------------------------------------------------------- engineering smoke
def run_smoke(root, out_dir, gpu, git=ftb.git_out):
    """<= 2 episodes (one per control), no optimizer, no provider, no test split. Engineering path only."""
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "smoke_receipt.json"
    if receipt_path.exists():
        raise BindingError("smoke already attempted (no retries): %s" % receipt_path)
    ftb.bind_worker_gpu(gpu)
    import torch
    from .collector import Collector
    os.chdir(str(root))
    split = ftb.derive_noprior_split(root, out_dir / "train_split_tb_noprior.json")
    manifest = ftb.derive_runtime_manifest(root, out_dir / "runtime_manifest_tb_resolved.yaml", split["path"])
    uuid = ftb.query_gpu_uuid(gpu)
    device = torch.device("cuda", 0)
    receipt = {"card": CARD, "gpu": int(gpu), "gpu_uuid": uuid, "case": SMOKE_CASE, "caps": SMOKE_CAPS, "started": utc_now(),
               "head": git(root, "rev-parse", "HEAD"), "provider_requests": 0, "optimizer_steps": 0, "test_ids_accessed": 0,
               "episodes": 0, "methods": {}, "verdict": "FAIL"}
    ftb.write_json_atomic(receipt_path.with_suffix(".started.json"), {"started": receipt["started"], "note": "attempt charged before the first environment call"})
    failures = []
    for method, plan_id in (("B2-ABS", "R-TB-ABS-0"), ("B1-K+NC", "R-TB-NC-0")):
        row = {"method": method, "verdict": "FAIL", "steps": []}
        receipt["methods"][method] = row
        bundle = None
        try:
            ctx = RepctlRunContext(plan_id=plan_id, attempt_id="SMOKE-" + plan_id, method=method, training_seed=PLAN_TABLE[plan_id]["seed"],
                                   source_commit=receipt["head"], output_directory=str(out_dir / "unused_run_dir"), runtime_manifest=manifest["path"],
                                   train_split=split["path"], physical_gpu_index=int(gpu), render_gpu_device_id=int(gpu), gpu_uuid=uuid)
            v11 = configure_v11(root, ctx)
            v11.bind_H(root)
            receipt["episodes"] += 1
            bundle = v11.make_bundle(root, TASK)
            snap = bundle.start_case(SMOKE_CASE)
            row["reset_ok"] = True
            snap, prior, _ = v11.apply_episode_prior(method, bundle, snap, None, eval_original=True)
            bundle.current_snapshot = snap
            row["prior_edges"] = len(snap.prior_edges)
            torch.manual_seed(0)
            policy = v11.make_policy(bundle.template, method, device)
            policy.eval()
            rows_seen = []
            hook = policy.contract.register_forward_pre_hook(lambda _m, args: rows_seen.append(tuple(args[1].shape)))
            with torch.no_grad():
                out = policy(snap, policy.initial_hidden())
            hook.remove()
            legal = [bool(m) for m in out.mask.tolist()]
            row["candidate_ids"] = list(out.candidate_ids)
            row["mask_matches_snapshot"] = legal == [bool(m) for m in snap.mask]
            row["n_legal"] = sum(legal)
            row["logits_finite_on_legal"] = bool(torch.isfinite(out.logits[out.mask]).all())
            row["value_finite"] = bool(torch.isfinite(out.value))
            row["q_finite_on_legal"] = bool(torch.isfinite(out.q[out.mask]).all())
            goals = len(bundle.template.goals)
            if method == "B2-ABS":
                expected = [(1 + goals, 128)] * row["n_legal"]
                row["representation"] = {"kind": "absolute_successor_rows", "readout_row_shapes": rows_seen, "expected": expected,
                                         "nominal_apply_used": bool(out.diagnostics.get("successor_used"))}
                row["representation_shape_ok"] = rows_seen == expected and bool(out.diagnostics.get("successor_used"))
            else:
                counts = out.diagnostics["effect_tokens"]
                expected = [(counts[c], 128) for c, ok in zip(out.candidate_ids, legal) if ok]
                row["representation"] = {"kind": "neural_composition_rows", "readout_row_shapes": rows_seen, "expected": expected,
                                         "nominal_apply_used": bool(out.diagnostics.get("successor_used"))}
                row["representation_shape_ok"] = rows_seen == expected and not out.diagnostics.get("successor_used")
            counters = {"executor_entered": 0, "executor_returned": 0, "transitions": 0}
            real = bundle.executor

            class _Counting:
                def __getattr__(self, name):
                    return getattr(real, name)

                def execute(self, candidate, timeout):
                    counters["executor_entered"] += 1
                    result = real.execute(candidate, timeout)
                    counters["executor_returned"] += 1
                    return result
            bundle.executor = _Counting()
            collector = Collector(bundle, policy)
            collector.reset_episode(snap.env_id, snap.episode_id)
            for i in range(SMOKE_MAX_SKILLS_PER_EPISODE):
                t, res = collector.step(snap, deterministic=True)
                step = {"index": i, "transition": t is not None}
                if t is None:
                    step["no_transition"] = str(res)[:300]
                    row["steps"].append(step)
                    break
                counters["transitions"] += 1
                step.update(selected=t.selected_candidate_id, duration=float(t.duration), old_logp_finite=bool(abs(float(t.old_logp)) < float("inf")),
                            terminated=bool(t.terminated), truncated=bool(t.truncated))
                row["steps"].append(step)
                snap = t.next_snapshot
                bundle.current_snapshot = snap
                if t.terminated or t.truncated:
                    break
            bundle.executor = real
            row["counters"] = counters
            ok = (row["reset_ok"] and row["mask_matches_snapshot"] and row["n_legal"] > 0 and row["logits_finite_on_legal"] and row["value_finite"]
                  and row["q_finite_on_legal"] and row["representation_shape_ok"] and counters["transitions"] >= 1
                  and counters["executor_entered"] == counters["executor_returned"] == counters["transitions"]
                  and all(s.get("old_logp_finite") and s.get("duration", 0) > 0 for s in row["steps"] if s["transition"]))
            row["verdict"] = "PASS" if ok else "FAIL"
        except BaseException as exc:  # recorded, never swallowed
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            row["traceback"] = traceback.format_exc()
            failures.append(method)
        finally:
            if bundle is not None:
                with contextlib.suppress(Exception):
                    bundle.environment.close()
    receipt["finished"] = utc_now()
    receipt["verdict"] = "PASS" if not failures and all(r["verdict"] == "PASS" for r in receipt["methods"].values()) and receipt["episodes"] <= SMOKE_CAPS["episodes"] else "FAIL"
    receipt["note"] = "engineering smoke only; no method conclusion may be drawn from it"
    ftb.write_json_atomic(receipt_path, receipt)
    return receipt


# ----------------------------------------------------------------------------- status / CLI
def status(out) -> dict:
    state = RepctlLedger(out).read()
    return {"card": state.get("card"), "attempts_used": state["new_rl_attempts_used"], "attempts_cap": state["new_rl_attempts_cap"],
            "plans": {p: {k: v.get(k) for k in ("method", "seed", "status", "physical_gpu", "pid", "stop_reason", "complete_updates", "valid_transitions")}
                      for p, v in state["plans"].items()}}


def build_parser():
    ap = argparse.ArgumentParser(prog="final_tb_repctl_launch", description="%s entry. Explicit subcommands only." % CARD)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("register", "release", "train", "supervise", "status", "smoke", "verify-prep"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        if name not in ("verify-prep",):
            p.add_argument("--out", required=True)
        if name == "register":
            p.add_argument("--prep-commit", required=True)
            p.add_argument("--authorization-text-file", required=True)
            p.add_argument("--stamp")
        if name == "train":
            p.add_argument("--plan", required=True)
            p.add_argument("--gpu", type=int, required=True)
            p.add_argument("--token", required=True)
        if name == "smoke":
            p.add_argument("--gpu", type=int, required=True)
        if name == "supervise":
            p.add_argument("--poll-seconds", type=float, default=60.0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    try:
        if args.command == "verify-prep":
            print(canonical(verify_prep_receipts(root)))
        elif args.command == "register":
            text = Path(args.authorization_text_file).read_text(encoding="utf-8")
            print(canonical(register(root, args.out, args.prep_commit, text, stamp=args.stamp)))
        elif args.command == "release":
            prep = _json(Path(args.out) / "authorization.json")["prep_commit"]
            print(canonical([str(p) for p in release(root, args.out, prep)]))
        elif args.command == "train":
            print(canonical(run_repctl_train(root, args.out, args.plan, args.gpu, args.token)))
        elif args.command == "supervise":
            print(canonical(supervise(root, args.out, poll_seconds=args.poll_seconds)))
        elif args.command == "smoke":
            print(canonical(run_smoke(root, args.out, args.gpu)))
        elif args.command == "status":
            print(canonical(status(args.out)))
    except BindingError as exc:
        print(canonical({"status": "BLOCKED", "error": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
