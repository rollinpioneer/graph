"""CP-DISR-TB-STRUCT-GEN-V1 launch binding: twelve attempts (4 representations x 3 seeds) on the structural-generalization suite.

No trainer and no scientific change: the production path (``final_tb.run_train`` -> ``stage2a_v11.train_job``) is reused. In process only:
the two controls are registered (as in the representation-control card), the goal adaptor routes ``make_bundle`` through
``struct_gen_runtime``, and the dev evaluation size is 12 (4 cases per train cell). Import-light: no torch / simulator at import time.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import final_tb as ftb
from . import final_tb_e1 as e1
from . import final_tb_repctl as rep
from .common import BindingError, canonical

CARD = "CP-DISR-TB-STRUCT-GEN-V1"
SEED_BALANCE_RESULT_COMMIT = "b602dfd1bc6f1241ebed1a474d51e6f6875d52a2"
REPCTL_RESULT_COMMIT = "4d0c426f67dfa1ce23247f89879c0b4952367d53"
TASK = ftb.TASK
CLI = "final_tb_structgen_launch.py"
PREP_REL = Path("runs/final_master/2.1.1/structgen/prep")
TRAIN_DEV_REL = Path("configs/splits/struct_gen_v1_train_dev.json")
TEST_REL = Path("configs/splits/struct_gen_v1_test.json")
RUN_ROOT_REL = Path("runs/final_master/2.1.1/T_B")
DEV_EPISODES = 12
MAX_WORKERS = 6
METHOD_OF = {"E": "B1-K+E", "B2": "B2", "ABS": "B2-ABS", "NC": "B1-K+NC"}
FAMILY_ORDER = ("E", "B2", "ABS", "NC")
PLAN_TABLE = {}
for _seed in (0, 1, 2):          # seed-major queue: the order is frozen and never depends on a result
    for _i, _fam in enumerate(FAMILY_ORDER):
        PLAN_TABLE["R-TB-SG-%s-%d" % (_fam, _seed)] = {"method": METHOD_OF[_fam], "family": _fam, "seed": _seed, "queue_order": _seed * 4 + _i + 1}
MAX_ATTEMPTS = len(PLAN_TABLE)
GPU_CANDIDATES = (0, 1, 2, 3, 4, 5, 6)
GPU_MAX_USED_MIB = 20000
RELEASE_FILE = "training_release.json"
NOT_AUTHORIZED = ["seed 3+", "new representation", "VLM / prior", "T_P", "Family B", "RoboCasa", "test split used for any training or selection decision", "change of Tcap/Ncap/PPO/reward",
                  "retry or replacement of a started attempt", "a case, depth level or binding dropped or selected after seeing results"]


def utc_now():
    return ftb.utc_now()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
class StructgenRunContext(ftb.RunContext):
    def assert_worker(self, task_id, method):
        if task_id != self.task_id:
            raise BindingError("run context is bound to %s, not %s" % (self.task_id, task_id))
        if method != self.method:
            raise BindingError("run context is bound to %s, not %s" % (self.method, method))
        validate_assignment(self.plan_id, self.method, self.training_seed)


def context_from_dict(cfg: dict) -> StructgenRunContext:
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
    return StructgenRunContext(
        plan_id=cfg["plan_id"], attempt_id=str(cfg["attempt_id"]), method=cfg["method"], training_seed=int(cfg["training_seed"]), source_commit=cfg["source_commit"],
        output_directory=str(cfg["output_directory"]), runtime_manifest=str(cfg["runtime_manifest"]), train_split=str(cfg["train_split"]), prior_mode=ftb.PRIOR_MODE,
        study_envelope_ncap=ftb.N_CAP, study_envelope_tcap=ftb.TCAP_SECONDS, evaluation_rule=ftb.EVALUATION_RULE, task_id=TASK,
        render_gpu_device_id=int(cfg.get("render_gpu_device_id", 0)), physical_gpu_index=None if cfg.get("physical_gpu_index") is None else int(cfg["physical_gpu_index"]),
        gpu_uuid=cfg.get("gpu_uuid"))


def load_run_config(path) -> StructgenRunContext:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise BindingError("run config must be a mapping")
    return context_from_dict(cfg)


class StructgenLedger(ftb.Ledger):
    """Twelve attempts, at most six concurrent workers, one attempt per plan; a started attempt is never removed or renumbered."""

    def init(self, plans: dict) -> dict:
        with self.locked():
            if self.state_path.exists():
                raise BindingError("ledger already initialised: %s" % self.state_path)
            state = {"created": utc_now(), "card": CARD, "new_rl_attempts_used": 0, "new_rl_attempts_cap": MAX_ATTEMPTS, "max_concurrent_workers": MAX_WORKERS,
                     "plans": {pid: dict(v) for pid, v in plans.items()}}
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
            if len([p for p, v in state["plans"].items() if v["status"] == "RUNNING"]) >= MAX_WORKERS:
                raise BindingError("at most %d concurrent workers" % MAX_WORKERS)
            for other, v in state["plans"].items():
                if other != plan_id and v.get("run_dir") == str(run_dir):
                    raise BindingError("run directory already reserved by %s" % other)
                if other != plan_id and physical_gpu is not None and v["status"] == "RUNNING" and v.get("physical_gpu") == physical_gpu:
                    raise BindingError("GPU %s already holds a running worker (%s)" % (physical_gpu, other))
            self.reservations.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.reservations / (plan_id + ".json")), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"plan_id": plan_id, "attempt_id": attempt_id, "run_dir": str(run_dir), "pid": pid, "physical_gpu": physical_gpu, "time": utc_now()}, sort_keys=True) + "\n")
            plan.update({"status": "RUNNING", "attempt_id": attempt_id, "run_dir": str(run_dir), "pid": pid, "physical_gpu": physical_gpu, "started": utc_now()})
            state["new_rl_attempts_used"] += 1
            ftb.write_json_atomic(self.state_path, state)
            return plan


# ----------------------------------------------------------------------------- v11 configuration (in process only)
def configure_v11(root, ctx):
    from . import phase_a_v12 as v12
    from . import struct_gen_runtime
    v11 = rep.configure_v11(root, ctx)      # ftb.configure_v11 + in-process registration of B2-ABS / B1-K+NC
    v11.DEV_EPISODES = DEV_EPISODES
    v12.DEV10_N = DEV_EPISODES
    struct_gen_runtime.install(v11)
    return v11


# ----------------------------------------------------------------------------- release gate
RELEASE_GATE_FILES = (
    "src/cp_disr/struct_gen.py", "src/cp_disr/struct_gen_runtime.py", "src/cp_disr/final_tb_structgen.py", "src/cp_disr/neural.py", "src/cp_disr/repctl_policy.py",
    "src/cp_disr/final_tb_repctl.py", "src/cp_disr/stage2a_v11.py", "src/cp_disr/torch_rl.py", "src/cp_disr/platforms/libero/task_evaluator.py",
    "src/cp_disr/platforms/libero/skill_executor.py", "src/cp_disr/platforms/libero/runtime_factory.py", "configs/splits/struct_gen_v1_train_dev.json",
    "configs/splits/struct_gen_v1_test.json", "configs/runtime/stage_2a_contract_registry.yaml",
)
REQUIRED_PREP_FILES = (
    "structural_generalization_spec.json", "representation_challenge_manifest.json", "dependency_depth_audit.json", "binding_novelty_audit.json", "shortcut_audit.json",
    "physical_qualification_plan.json", "physical_qualification_results.json", "physical_qualification_amendment.json", "struct_gen_split_manifest.json",
    "representation_identity.json", "capacity_identity.json", "offline_test_receipt.json", "training_release.json", "prep_summary.md", "verify.json",
)


def verify_release(root, prep_dir=None) -> dict:
    root = Path(root)
    prep_dir = Path(prep_dir) if prep_dir else root / PREP_REL
    missing = [f for f in REQUIRED_PREP_FILES if not (prep_dir / f).is_file()]
    if missing:
        raise BindingError("prep files missing: %s" % missing)
    rel = _json(prep_dir / RELEASE_FILE)
    if rel.get("card") != CARD or rel.get("verdict") != "PASS":
        raise BindingError("training_release.json is not a PASS")
    if rel.get("plan_table") != PLAN_TABLE:
        raise BindingError("training_release plan table differs from the frozen plan table")
    drift = {f: (h, ftb.sha256_file(root / f)) for f, h in rel["file_sha256"].items() if (root / f).is_file() and h != ftb.sha256_file(root / f)}
    drift.update({f: (h, None) for f, h in rel["file_sha256"].items() if not (root / f).is_file()})
    if drift:
        raise BindingError("files changed after the release gate: %s" % sorted(drift))
    for name, verdict in rel["gates"].items():
        if verdict not in ("PASS", "PASS_UNDER_DISCLOSED_AMENDMENT"):
            raise BindingError("release gate %s is %s" % (name, verdict))
    return rel


# ----------------------------------------------------------------------------- registration / release tokens / train
def register(root, out_dir, prep_commit, authorization_text, stamp=None, git_fn=ftb.git_out) -> dict:
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    ftb.check_source_identity(root, prep_commit, git=git_fn)
    rel = verify_release(root)
    if (out_dir / "launch_state.json").exists():
        raise BindingError("already registered: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    split_path = (root / TRAIN_DEV_REL).resolve()
    split_doc = _json(split_path)
    if split_doc.get("test") or split_doc.get("test_count"):
        raise BindingError("the training split file must not carry test rows")
    auth = {"card": CARD, "prep_commit": prep_commit, "seed_balance_result_commit": SEED_BALANCE_RESULT_COMMIT, "representation_controls_result_commit": REPCTL_RESULT_COMMIT,
            "plan_table": PLAN_TABLE, "queue": sorted(PLAN_TABLE, key=lambda p: PLAN_TABLE[p]["queue_order"]), "authorization_text": authorization_text,
            "authorization_text_sha256": ftb.sha256_text(authorization_text), "release_sha256": ftb.sha256_file(root / PREP_REL / RELEASE_FILE),
            "caps": {"rl_attempts": MAX_ATTEMPTS, "max_concurrent_workers": MAX_WORKERS, "ncap": ftb.N_CAP, "tcap": ftb.TCAP_SECONDS, "provider": 0, "tp_training": 0},
            "not_authorized": NOT_AUTHORIZED, "registered": utc_now()}
    ftb.write_json_atomic(out_dir / "authorization.json", auth)
    manifest = ftb.derive_runtime_manifest(root, out_dir / "runtime_manifest_tb_resolved.yaml", str(split_path))
    profile = ftb.resolve_frozen_profile(root, manifest["path"])
    split = {"path": str(split_path), "sha256": ftb.sha256_file(split_path)}
    ledger_plans, configs = {}, {}
    (out_dir / "run_configs").mkdir(exist_ok=True)
    for plan_id, row in sorted(PLAN_TABLE.items(), key=lambda kv: kv[1]["queue_order"]):
        attempt = "%s-%s-%s" % (plan_id, stamp, prep_commit[:8])
        run_dir = root / RUN_ROOT_REL / row["method"] / ("seed_%d" % row["seed"]) / attempt
        cfg = {"plan_id": plan_id, "attempt_id": attempt, "method": row["method"], "training_seed": row["seed"], "source_commit": prep_commit, "output_directory": str(run_dir),
               "runtime_manifest": manifest["path"], "train_split": split["path"], "prior_mode": ftb.PRIOR_MODE, "study_envelope_ncap": ftb.N_CAP,
               "study_envelope_tcap": ftb.TCAP_SECONDS, "evaluation_rule": ftb.EVALUATION_RULE}
        context_from_dict(cfg)
        cfg_path = out_dir / "run_configs" / (plan_id + ".yaml")
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        os.chmod(cfg_path, 0o444)
        configs[plan_id] = {"path": str(cfg_path), "sha256": ftb.sha256_file(cfg_path)}
        ledger_plans[plan_id] = {"method": row["method"], "family": row["family"], "seed": row["seed"], "status": "NOT_STARTED", "attempt_id": attempt, "planned_run_dir": str(run_dir),
                                 "queue_order": row["queue_order"]}
    StructgenLedger(out_dir).init(ledger_plans)
    plan = {"card": CARD, "prep_commit": prep_commit, "stamp": stamp, "plans": ledger_plans, "configs": configs, "derived": {"runtime_manifest": manifest, "train_split": split},
            "test_split_sha256": ftb.sha256_file(root / TEST_REL), "frozen_runtime": ftb.FROZEN_RUNTIME, "frozen_profile": profile, "envelope": ftb.envelope(), "eval_points": list(ftb.EVAL_POINTS),
            "dev_episodes": DEV_EPISODES, "queue": auth["queue"], "init": "FROM_SCRATCH", "warm_start": False, "release_gates": rel["gates"]}
    ftb.write_json_atomic(out_dir / "launch_plan.json", plan)
    return plan


def release_tokens(root, out_dir, prep, free_fn=None, git_fn=ftb.git_out) -> list:
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    ftb.check_source_identity(root, prep, git=git_fn)
    verify_release(root)
    auth = _json(out_dir / "authorization.json")
    if auth.get("card") != CARD or auth.get("prep_commit") != prep:
        raise BindingError("registration is not this card's / prep commit")
    state = StructgenLedger(out_dir).read()
    if any(v["status"] != "NOT_STARTED" for v in state["plans"].values()):
        raise BindingError("tokens are issued once, before any run starts")
    gate = rep.start_gate(next(iter(state["plans"].values()))["planned_run_dir"], free_fn)
    evidence = {"release_sha256": auth["release_sha256"], "storage_start_gate": gate, "tokens_issued_together": True}
    return [ftb.issue_token(out_dir, "train", prep, p, evidence) for p in sorted(PLAN_TABLE, key=lambda p: PLAN_TABLE[p]["queue_order"])]


def run_structgen_train(root, out, plan_id, gpu, token_path, git_fn=ftb.git_out, free_fn=None, runner=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    auth = _json(out / "authorization.json")
    prep = auth["prep_commit"]
    if auth.get("card") != CARD:
        raise BindingError("registration is not this card's")
    ftb.verify_token(token_path, out, "train", prep, plan_id)
    ftb.check_source_identity(root, prep, git=git_fn)
    verify_release(root)
    launch = _json(out / "launch_plan.json")
    cfg_entry = launch["configs"][plan_id]
    cfg_path = Path(cfg_entry["path"])
    if ftb.sha256_file(cfg_path) != cfg_entry["sha256"]:
        raise BindingError("run config changed after registration")
    gate = rep.start_gate(launch["plans"][plan_id]["planned_run_dir"], free_fn)
    ftb.bind_worker_gpu(gpu)
    uuid = ftb.query_gpu_uuid(gpu)
    os.chdir(str(root))
    base = load_run_config(cfg_path)
    ctx = dataclasses.replace(base, physical_gpu_index=int(gpu), render_gpu_device_id=int(gpu), gpu_uuid=uuid)
    if ctx.plan_id != plan_id:
        raise BindingError("run config is not %s" % plan_id)
    ftb.write_json_atomic(out / ("launch_evidence_%s.json" % plan_id), {"time": utc_now(), "pid": os.getpid(), "gpu": int(gpu), "gpu_uuid": uuid, "start_gate": gate,
                                                                       "prep_commit": prep, "init": "FROM_SCRATCH", "warm_start": False})
    v11 = configure_v11(root, ctx)
    return (runner or ftb.run_train)(root, out, ctx, gpu, v11_module=v11, ledger=StructgenLedger(out))


# ----------------------------------------------------------------------------- supervisor (queue order, <= MAX_WORKERS)
def pick_gpu(table: dict, taken) -> int | None:
    free = [(row["memory_used_mib"], idx) for idx, row in table.items() if idx in GPU_CANDIDATES and idx not in taken and row["memory_used_mib"] <= GPU_MAX_USED_MIB]
    return min(free)[1] if free else None


def supervise(root, out, python=sys.executable, poll_seconds=60.0, max_ticks=None, gpu_fn=rep.gpu_table, spawn=None, sleep=time.sleep, log=print):
    root, out = Path(root).resolve(), Path(out).resolve()
    ledger = StructgenLedger(out)
    queue = _json(out / "launch_plan.json")["queue"]
    state_path = out / "supervisor_state.json"
    sup = _json(state_path) if state_path.is_file() else {"spawned": {}, "halted": None, "events": []}

    def event(msg, **kw):
        sup["events"].append({"time": utc_now(), "msg": msg, **kw})
        ftb.write_json_atomic(state_path, sup)
        log("[structgen-supervisor] %s %s" % (msg, kw if kw else ""))
    spawn = spawn or (lambda cmd, logfile: subprocess.Popen(cmd, stdout=open(logfile, "ab"), stderr=subprocess.STDOUT, start_new_session=True,
                                                            env={**os.environ, "PYTHONPATH": str(root / "src")}, cwd=str(root)))
    procs, ticks = {}, 0
    while True:
        ticks += 1
        state = ledger.read()
        statuses = {p: v["status"] for p, v in state["plans"].items()}
        alive = [p for p, meta in sup["spawned"].items() if (procs[p].poll() is None if p in procs else rep._alive(meta.get("pid")))]
        for p in list(sup["spawned"]):
            if p not in alive and statuses.get(p) in ("RUNNING", "NOT_STARTED") and not sup["halted"]:
                sup["halted"] = "worker %s ended while its plan is %s" % (p, statuses.get(p))
                event("halting new workers", reason=sup["halted"])
            if statuses.get(p) == "STOPPED" and not sup["halted"]:
                sup["halted"] = "plan %s STOPPED: %s" % (p, state["plans"][p].get("error") or state["plans"][p].get("reason"))
                event("halting new workers", reason=sup["halted"])
        pending = [p for p in queue if statuses[p] == "NOT_STARTED" and p not in sup["spawned"]]
        if (not pending or sup["halted"]) and not alive:
            event("finished", statuses=statuses, halted=sup["halted"])
            return {"statuses": statuses, "halted": sup["halted"]}
        if pending and not sup["halted"] and len(alive) < MAX_WORKERS:
            plan_id = pending[0]
            taken = {m["gpu"] for p, m in sup["spawned"].items() if p in alive}
            gpu = pick_gpu(gpu_fn(), taken)
            if gpu is None:
                event("no usable GPU; waiting", plan=plan_id)
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


# ----------------------------------------------------------------------------- storage guard (all workers)
def find_worker_pids(out_dir, proc_root="/proc") -> list:
    try:
        state = _json(Path(out_dir) / "launch_state.json")
    except (OSError, ValueError):
        return []
    pids = []
    for plan in state["plans"].values():
        pid = plan.get("pid")
        if plan.get("status") != "RUNNING" or not pid:
            continue
        try:
            cmd = (Path(proc_root) / str(int(pid)) / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if CLI in cmd:
            pids += [int(pid)] + e1.descendants(int(pid), proc_root)
    return pids


def run_guard(out: Path, wait_seconds: float) -> dict:
    launch = _json(out / "launch_plan.json")
    watch = next(iter(launch["plans"].values()))["planned_run_dir"]
    guard = e1.StorageGuard(ftb._nearest_existing(watch), out / "storage_guard.jsonl", lambda: find_worker_pids(out))
    guard._log("GUARD_START", thresholds=guard.cfg, mount=e1.mount_of(watch), free_bytes=guard.free_fn(guard.watch_path), workers=find_worker_pids(out))

    def on_signal(signum, _frame):
        guard.release_all("signal_%s" % signum)
        ftb.write_json_atomic(out / "storage_guard_summary.json", {**guard.summary(), "ended": "signal", "utc": utc_now()})
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    t0, seen = time.time(), {"worker": False}

    def stop():
        if find_worker_pids(out):
            seen["worker"] = True
            return False
        return seen["worker"] or (time.time() - t0) > wait_seconds
    summary = guard.run(stop)
    guard._log("GUARD_END", **summary)
    ftb.write_json_atomic(out / "storage_guard_summary.json", {**summary, "ended": "workers_finished" if seen["worker"] else "no_worker_started", "utc": utc_now()})
    return summary


# ----------------------------------------------------------------------------- CLI
def status(out) -> dict:
    state = StructgenLedger(out).read()
    return {"card": state.get("card"), "attempts_used": state["new_rl_attempts_used"], "attempts_cap": state["new_rl_attempts_cap"],
            "plans": {p: {k: v.get(k) for k in ("method", "seed", "status", "physical_gpu", "pid", "stop_reason", "complete_updates", "valid_transitions")} for p, v in state["plans"].items()}}


def build_parser():
    ap = argparse.ArgumentParser(prog="final_tb_structgen_launch", description="%s entry. Explicit subcommands only." % CARD)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("register", "release", "train", "supervise", "guard", "status"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        p.add_argument("--out", required=True)
        if name == "register":
            p.add_argument("--prep-commit", required=True)
            p.add_argument("--authorization-text-file", required=True)
            p.add_argument("--stamp")
        if name == "train":
            p.add_argument("--plan", required=True)
            p.add_argument("--gpu", type=int, required=True)
            p.add_argument("--token", required=True)
        if name == "supervise":
            p.add_argument("--poll-seconds", type=float, default=60.0)
        if name == "guard":
            p.add_argument("--wait-seconds", type=float, default=900.0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root, out = Path(args.root).resolve(), Path(args.out).expanduser().resolve()
    try:
        if args.command == "register":
            plan = register(root, out, args.prep_commit, Path(args.authorization_text_file).read_text(encoding="utf-8"), stamp=args.stamp)
            print(canonical({"registered": True, "plans": sorted(plan["plans"])}))
        elif args.command == "release":
            print(canonical([str(p) for p in release_tokens(root, out, _json(out / "authorization.json")["prep_commit"])]))
        elif args.command == "train":
            print(canonical(run_structgen_train(root, out, args.plan, args.gpu, args.token)))
        elif args.command == "supervise":
            print(canonical(supervise(root, out, poll_seconds=args.poll_seconds)))
        elif args.command == "guard":
            print(canonical(run_guard(out, args.wait_seconds)))
        elif args.command == "status":
            print(canonical(status(out)))
    except BindingError as exc:
        print(canonical({"status": "BLOCKED", "error": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
