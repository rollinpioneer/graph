"""Thin, explicit launch layer for the final T_B candidate-consequence comparison.

CP-DISR-NEXT-ENGINEERING-FIRST-TB-1.  This module adds *no* trainer and changes no scientific
or public-runtime semantics.  It binds explicit run identity (plan/attempt/method/seed/source),
a derived no-prior split and a derived runtime manifest to the existing production runner
(`stage2a_v11.train_job`, Policy, PPO, Collector, immutable generations).

Import-light on purpose: nothing here imports torch or the simulator at module import time, so
`check`, `register` and `--help` can never construct an environment.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .common import BindingError, canonical

TASK = "T_B"
PLAN_TABLE = {
    "R-TB-E-0": {"method": "B1-K+E", "seed": 0, "release_order": 1},
    "R-TB-DK-1": {"method": "B2", "seed": 1, "release_order": 2},
    "R-TB-K-1": {"method": "B1-K", "seed": 1, "release_order": 3},
}
METHODS = tuple(sorted({v["method"] for v in PLAN_TABLE.values()}))
MAX_NEW_RL_ATTEMPTS = 3
MAX_WORKERS = 2

# Frozen T_B runtime values (exact floats; never recomputed or recalibrated).
H_SECONDS = 23.09999999999752
D_REF_SECONDS = 4.199999999997672
TASK_DEADLINE_SECONDS = 60.0
N_CAP = 16384
ROLLOUT_N = 1024
MAX_UPDATES = 16
DEV10_N = 10
EVAL_POINTS = (0, 4096, 8192, 16384)
EVALUATION_RULE = "0/4096/8192/final; frozen dev10"
TCAP_SECONDS = float(N_CAP) * float(D_REF_SECONDS)  # "16384 * frozen_d_ref"
PRIOR_MODE = "absent"

FROZEN_RUNTIME = {
    "task_id": TASK,
    "task_deadline_seconds": TASK_DEADLINE_SECONDS,
    "H_seconds": H_SECONDS,
    "d_ref_seconds": D_REF_SECONDS,
    "study_envelope_ncap": N_CAP,
    "study_envelope_tcap_definition": "16384 * frozen_d_ref",
    "rollout_n": ROLLOUT_N,
    "max_complete_updates": MAX_UPDATES,
    "prior_mode": PRIOR_MODE,
}

SPLIT_REL = Path("configs/splits/T_B_phase_a_v13_r1_dev10.json")
RUNTIME_REL = Path("experiments/manifests/runtime_manifest_v211.yaml")
REFERENCE_REL = Path("runs/stage_0a/reference_execution_manifest.json")
TASK_RESOLVED_REL = Path("configs/tasks/resolved/T_B.yaml")
PROFILE_SOURCES = (TASK_RESOLVED_REL, SPLIT_REL, REFERENCE_REL, RUNTIME_REL)
S0_BPLAN_REL = Path("runs/final_master/S0/60c26621/b_plan_dev/episodes.json")

# Smoke budget (A2).
SMOKE_CAPS = {"episode_attempts": 2, "construction_attempts": 4, "explicit_resets": 4, "skill_calls": 24}
SMOKE_PER_EPISODE_SKILLS = 12
PREFLIGHT_EXPECTED_PER_EPISODE = {"construction_attempts": 2, "explicit_resets": 2}

OLD_ROOT_RE = re.compile(r"/home/(?:xushijie2|__compress_data/xushijie)/graph_cp_disr_[A-Za-z0-9_]+")
PLACEHOLDER_RE = re.compile(r"^(?:[A-Z][A-Z_]{3,}|(?:ABSOLUTE|RESOLVED|FROZEN|NEW_AND|PREP)_.*)$")


# ----------------------------------------------------------------------------- small helpers
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def git_out(root, *args) -> str:
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def is_placeholder(value) -> bool:
    return isinstance(value, str) and PLACEHOLDER_RE.match(value) is not None


# ----------------------------------------------------------------------------- plan / identity rules
def validate_assignment(plan_id, method, seed) -> dict:
    """Plan id, method and training seed must match the base-17 table exactly."""
    if plan_id not in PLAN_TABLE:
        raise BindingError("unknown plan_id %r" % (plan_id,))
    if method not in METHODS:
        raise BindingError("unknown or non-T_B method %r" % (method,))
    row = PLAN_TABLE[plan_id]
    if method != row["method"]:
        raise BindingError("%s is %s, not %s" % (plan_id, row["method"], method))
    if isinstance(seed, bool) or not isinstance(seed, int) or int(seed) != row["seed"]:
        raise BindingError("%s requires training_seed=%s, got %r" % (plan_id, row["seed"], seed))
    return dict(row)


def check_source_identity(root, expected_commit, git=git_out) -> dict:
    """HEAD must be the registered prep commit and the *tracked* tree clean (untracked results are fine)."""
    head = git(root, "rev-parse", "HEAD")
    if head != expected_commit:
        raise BindingError("source drift: HEAD %s != registered prep commit %s" % (head, expected_commit))
    dirty = git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty.strip():
        raise BindingError("source drift: tracked tree is dirty:\n%s" % dirty)
    return {"head": head, "tracked_tree_clean": True}


def envelope(profile=None) -> dict:
    """The frozen study envelope. It does not depend on any pause point."""
    return {"ncap": N_CAP, "tcap": TCAP_SECONDS, "d_ref": D_REF_SECONDS, "H": H_SECONDS}


def envelope_stop(count, seconds):
    """Which cap (if any) has been reached; Tcap is never recomputed from pause points."""
    if count >= N_CAP:
        return "Ncap"
    if seconds >= TCAP_SECONDS:
        return "Tcap"
    return None


# ----------------------------------------------------------------------------- run context
@dataclass(frozen=True)
class RunContext:
    plan_id: str
    attempt_id: str
    method: str
    training_seed: int
    source_commit: str
    output_directory: str
    runtime_manifest: str
    train_split: str
    prior_mode: str = PRIOR_MODE
    study_envelope_ncap: int = N_CAP
    study_envelope_tcap: float = TCAP_SECONDS
    evaluation_rule: str = EVALUATION_RULE
    task_id: str = TASK
    render_gpu_device_id: int = 0
    physical_gpu_index: int | None = None
    gpu_uuid: str | None = None

    def assert_worker(self, task_id, method):
        if task_id != self.task_id:
            raise BindingError("run context is bound to %s, not %s" % (self.task_id, task_id))
        if method != self.method:
            raise BindingError("run context is bound to %s, not %s" % (self.method, method))
        validate_assignment(self.plan_id, self.method, self.training_seed)

    def config_fields(self) -> dict:
        return {
            "plan_id": self.plan_id, "attempt_id": self.attempt_id, "source_commit": self.source_commit,
            "output_directory": self.output_directory, "runtime_manifest_abs": self.runtime_manifest,
            "train_split_abs": self.train_split, "prior_mode": self.prior_mode,
            "study_envelope_ncap": self.study_envelope_ncap, "study_envelope_tcap": self.study_envelope_tcap,
            "evaluation_rule": self.evaluation_rule, "render_gpu_device_id": self.render_gpu_device_id,
            "physical_gpu_index": self.physical_gpu_index, "gpu_uuid": self.gpu_uuid,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sampler": None, "provider": None,
        }

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


REQUIRED_CONFIG_KEYS = (
    "plan_id", "attempt_id", "method", "training_seed", "source_commit", "output_directory",
    "runtime_manifest", "train_split", "prior_mode", "study_envelope_ncap", "study_envelope_tcap",
    "evaluation_rule",
)


def context_from_dict(cfg: dict) -> RunContext:
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise BindingError("run config missing keys: %s" % missing)
    for k, v in cfg.items():
        if is_placeholder(v):
            raise BindingError("unresolved placeholder in run config: %s=%r" % (k, v))
    validate_assignment(cfg["plan_id"], cfg["method"], cfg["training_seed"])
    for k in ("output_directory", "runtime_manifest", "train_split"):
        if not os.path.isabs(str(cfg[k])):
            raise BindingError("%s must be an absolute path" % k)
    if re.fullmatch(r"[0-9a-f]{40}", str(cfg["source_commit"])) is None:
        raise BindingError("source_commit must be a full commit sha")
    if str(cfg["prior_mode"]) != PRIOR_MODE:
        raise BindingError("prior_mode must be absent for all three T_B methods")
    if int(cfg["study_envelope_ncap"]) != N_CAP:
        raise BindingError("study_envelope_ncap must equal the frozen %d" % N_CAP)
    if float(cfg["study_envelope_tcap"]) != TCAP_SECONDS:
        raise BindingError("study_envelope_tcap must equal 16384*frozen_d_ref=%r" % TCAP_SECONDS)
    if str(cfg["evaluation_rule"]) != EVALUATION_RULE:
        raise BindingError("evaluation_rule must be %r" % EVALUATION_RULE)
    if str(cfg.get("task_id", TASK)) != TASK:
        raise BindingError("task_id must be T_B")
    allowed = set(REQUIRED_CONFIG_KEYS) | {"task_id", "render_gpu_device_id", "physical_gpu_index", "gpu_uuid"}
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise BindingError("unknown run-config keys: %s" % unknown)
    return RunContext(
        plan_id=cfg["plan_id"], attempt_id=str(cfg["attempt_id"]), method=cfg["method"],
        training_seed=int(cfg["training_seed"]), source_commit=cfg["source_commit"],
        output_directory=str(cfg["output_directory"]), runtime_manifest=str(cfg["runtime_manifest"]),
        train_split=str(cfg["train_split"]), prior_mode=PRIOR_MODE, study_envelope_ncap=N_CAP,
        study_envelope_tcap=TCAP_SECONDS, evaluation_rule=EVALUATION_RULE, task_id=TASK,
        render_gpu_device_id=int(cfg.get("render_gpu_device_id", 0)),
        physical_gpu_index=None if cfg.get("physical_gpu_index") is None else int(cfg["physical_gpu_index"]),
        gpu_uuid=cfg.get("gpu_uuid"),
    )


def load_run_config(path) -> RunContext:
    text = Path(path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise BindingError("run config must be a mapping")
    return context_from_dict(cfg)


# ----------------------------------------------------------------------------- frozen profile
def resolve_frozen_profile(root, runtime_manifest_path=None) -> dict:
    """Read the four profile sources and confirm the frozen T_B values. No env, no torch."""
    root = Path(root)
    ref = json.loads((root / REFERENCE_REL).read_text(encoding="utf-8"))
    rt_path = Path(runtime_manifest_path) if runtime_manifest_path else root / RUNTIME_REL
    runtime = yaml.safe_load(rt_path.read_text(encoding="utf-8"))["runtime"]
    task_yaml = yaml.safe_load((root / TASK_RESOLVED_REL).read_text(encoding="utf-8"))
    out = {
        "task_id": task_yaml["task_id"],
        "task_deadline_seconds": float(runtime["task_deadlines"][TASK]),
        "H_seconds": float(ref["H"]),
        "runtime_suite_H_seconds": float(runtime["suite_H_seconds"]),
        "d_ref_seconds": float(ref["families"][TASK]["d_ref_median"]),
        "runtime_d_ref_seconds": float(runtime["reference_skill_seconds_by_task"][TASK]),
        "resolved_yaml_reference_skill_seconds": float(task_yaml["reference_skill_seconds"]),
        "resolved_yaml_deadline_seconds": float(task_yaml["deadline_seconds"]),
    }
    conflicts = []
    if out["task_id"] != TASK:
        conflicts.append("resolved task_id %r" % out["task_id"])
    if out["H_seconds"] != H_SECONDS or out["runtime_suite_H_seconds"] != H_SECONDS:
        conflicts.append("H differs from frozen")
    for key in ("d_ref_seconds", "runtime_d_ref_seconds", "resolved_yaml_reference_skill_seconds"):
        if out[key] != D_REF_SECONDS:
            conflicts.append("%s differs from frozen" % key)
    for key in ("task_deadline_seconds", "resolved_yaml_deadline_seconds"):
        if out[key] != TASK_DEADLINE_SECONDS:
            conflicts.append("%s differs from frozen" % key)
    if conflicts:
        raise BindingError("frozen T_B profile conflicts with trusted sources: %s" % conflicts)
    out.update({"study_envelope_ncap": N_CAP, "study_envelope_tcap": TCAP_SECONDS, "rollout_n": ROLLOUT_N,
                "max_complete_updates": MAX_UPDATES, "prior_mode": PRIOR_MODE,
                "tcap_definition": FROZEN_RUNTIME["study_envelope_tcap_definition"]})
    return out


def split_lists(root) -> dict:
    """Real train/dev id lists (not count fields) of the frozen dev10 split."""
    doc = json.loads((Path(root) / SPLIT_REL).read_text(encoding="utf-8"))
    train = [r["case_id"] for r in doc["train"]]
    dev = [r["case_id"] for r in doc["dev"]]
    if len(train) != 64 or len(dev) != DEV10_N or len(set(train)) != 64 or len(set(dev)) != DEV10_N:
        raise BindingError("T_B split must be train64/dev10 with unique ids (got %d/%d)" % (len(train), len(dev)))
    if doc.get("test") or doc.get("test_count"):
        raise BindingError("frozen dev10 split must carry no test rows")
    if any(r.get("second_role") != "second_object" for r in doc["train"] + doc["dev"]):
        raise BindingError("T_B rows must bind the double-target second object")
    return {"train": train, "dev": dev, "sha256": sha256_file(Path(root) / SPLIT_REL)}


# ----------------------------------------------------------------------------- derived inputs
def derive_noprior_split(root, out_path) -> dict:
    """Same cases/coordinates/ids as the frozen dev10 split; cache pointers moved to audit-only fields."""
    root = Path(root)
    src = root / SPLIT_REL
    doc = json.loads(src.read_text(encoding="utf-8"))
    derived = json.loads(json.dumps(doc))
    for part in ("train", "dev"):
        for row in derived[part]:
            ref = {k: row.pop(k) for k in ("cache_dir", "cache_key", "cache_status") if k in row}
            row["source_cache_ref_audit_only"] = ref
    derived["test"] = []
    derived["test_count"] = 0
    derived["final_tb_noprior"] = {
        "prior_mode": PRIOR_MODE, "derived_from": str(SPLIT_REL), "source_sha256": sha256_file(src),
        "semantics": "effective R is empty for B1-K, B1-K+E and B2; cache pointers are audit-only and never read",
        "not_a_missing_cache_policy": True, "test_ids_accessed": False,
    }
    out_path = Path(out_path)
    write_json_atomic(out_path, derived)
    return {"path": str(out_path), "sha256": sha256_file(out_path), "source_sha256": sha256_file(src)}


def derive_runtime_manifest(root, out_path, split_path) -> dict:
    """Original manifest stays read-only. Every workspace path is rebound to `root`."""
    root = Path(root).resolve()
    src = root / RUNTIME_REL
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    replaced = []

    def rebind(value):
        if isinstance(value, str):
            new = OLD_ROOT_RE.sub(str(root), value)
            if new != value:
                replaced.append({"old": value, "new": new})
            return new
        if isinstance(value, dict):
            return {k: rebind(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rebind(v) for v in value]
        return value

    doc = rebind(doc)
    rt = doc["runtime"]
    rt["repository_path"] = str(root)
    rt["experiment_root"] = str(root / "experiments")
    factory_src = root / "src/cp_disr/platforms/libero/runtime_factory.py"
    doc["runtime_factory"]["source_path"] = str(factory_src)
    doc["runtime_factory"]["sha256_original_manifest"] = doc["runtime_factory"].get("sha256")
    doc["runtime_factory"]["sha256"] = sha256_file(factory_src)
    rt["active_task_id"] = TASK
    rt.setdefault("task_splits", {})[TASK] = str(split_path)
    doc["final_tb_derived"] = {
        "derived_from": str(RUNTIME_REL), "source_sha256": sha256_file(src), "bound_root": str(root),
        "task_id": TASK, "prior_mode": PRIOR_MODE, "note": "original manifest not modified",
    }
    text = yaml.safe_dump(doc, sort_keys=False)
    leftovers = sorted(set(re.findall(r"graph_cp_disr_[A-Za-z0-9_]+", text)) - {root.name})
    if leftovers:
        raise BindingError("derived manifest still references other workspaces: %s" % leftovers)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return {"path": str(out_path), "sha256": sha256_file(out_path), "rebound": replaced, "bound_root": str(root)}


# ----------------------------------------------------------------------------- coordinator ledger
class Ledger:
    """Single-writer (flock) launch ledger. Workers write only their own run directory."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.state_path = self.dir / "launch_state.json"
        self.lock_path = self.dir / ".ledger.lock"
        self.reservations = self.dir / "reservations"

    @contextlib.contextmanager
    def locked(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def init(self, plans: dict) -> dict:
        with self.locked():
            if self.state_path.exists():
                raise BindingError("ledger already initialised: %s" % self.state_path)
            state = {"created": utc_now(), "new_rl_attempts_used": 0, "new_rl_attempts_cap": MAX_NEW_RL_ATTEMPTS,
                     "plans": {pid: dict(v) for pid, v in plans.items()}}
            write_json_atomic(self.state_path, state)
            return state

    def reserve(self, plan_id, attempt_id, run_dir, pid, physical_gpu) -> dict:
        with self.locked():
            state = self.read()
            if plan_id not in state["plans"]:
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
                f.write(json.dumps({"plan_id": plan_id, "attempt_id": attempt_id, "run_dir": str(run_dir),
                                    "pid": pid, "physical_gpu": physical_gpu, "time": utc_now()}, sort_keys=True) + "\n")
            plan.update({"status": "RUNNING", "attempt_id": attempt_id, "run_dir": str(run_dir), "pid": pid,
                         "physical_gpu": physical_gpu, "started": utc_now()})
            state["new_rl_attempts_used"] += 1
            write_json_atomic(self.state_path, state)
            return plan

    def finish(self, plan_id, status, **fields) -> dict:
        with self.locked():
            state = self.read()
            plan = state["plans"][plan_id]
            plan.update({"status": status, "finished": utc_now(), **fields})
            write_json_atomic(self.state_path, state)
            return plan


# ----------------------------------------------------------------------------- release tokens
def authorization_digest(out_dir) -> str:
    return sha256_file(Path(out_dir) / "authorization.json")


def issue_token(out_dir, kind, prep_commit, plan_id=None, evidence=None) -> Path:
    out_dir = Path(out_dir)
    token = {"kind": kind, "plan_id": plan_id, "prep_commit": prep_commit, "evidence": evidence or {},
             "authorization_sha256": authorization_digest(out_dir), "issued": utc_now()}
    path = out_dir / "release_tokens" / ("%s%s.json" % (kind, "" if plan_id is None else "_" + plan_id))
    if path.exists():
        raise BindingError("release token already exists: %s" % path)
    write_json_atomic(path, token)
    return path


def verify_token(token_path, out_dir, kind, prep_commit, plan_id=None) -> dict:
    """Refuse unless a matching, authorization-bound token exists. Runs before any env/torch import."""
    if not token_path or not Path(token_path).is_file():
        raise BindingError("no release token: %s requires an explicit valid release token" % kind)
    token = json.loads(Path(token_path).read_text(encoding="utf-8"))
    if token.get("kind") != kind or token.get("plan_id") != plan_id:
        raise BindingError("release token is for %s/%s, not %s/%s" % (token.get("kind"), token.get("plan_id"), kind, plan_id))
    if token.get("prep_commit") != prep_commit:
        raise BindingError("release token prep commit mismatch")
    auth = Path(out_dir) / "authorization.json"
    if not auth.is_file() or token.get("authorization_sha256") != sha256_file(auth):
        raise BindingError("release token is not bound to the registered authorization")
    return token


# ----------------------------------------------------------------------------- v11 configuration
def make_source_hashes(root, ctx):
    root = Path(root)
    files = {
        "final_tb": root / "src/cp_disr/final_tb.py",
        "stage2a_v11": root / "src/cp_disr/stage2a_v11.py",
        "phase_a_v12_persistence": root / "src/cp_disr/phase_a_v12.py",
        "collector": root / "src/cp_disr/collector.py",
        "torch_rl": root / "src/cp_disr/torch_rl.py",
        "neural": root / "src/cp_disr/neural.py",
        "persistence": root / "src/cp_disr/persistence.py",
        "runtime_factory": root / "src/cp_disr/platforms/libero/runtime_factory.py",
        "runtime_manifest_derived": Path(ctx.runtime_manifest),
        "train_split_derived": Path(ctx.train_split),
        "dev10_split_original": root / SPLIT_REL,
    }
    hashes = {k: (sha256_file(p) if p.is_file() else None) for k, p in files.items()}
    hashes["git_commit"] = ctx.source_commit
    hashes["plan_id"] = ctx.plan_id
    hashes["attempt_id"] = ctx.attempt_id
    return hashes


def _fixed_mechanism_scope(root, method, checkpoint, label, device):
    job = Path(checkpoint).parent.parent
    with (job / "fixed_mechanism_scope.jsonl").open("a", encoding="utf-8") as f:
        f.write(canonical({"label": label, "method": method, "checkpoint": str(checkpoint), "effective_prior": 0,
                           "test_id_used": False, "used_for_training": False,
                           "note": "no-prior T_B run; structure audited in production transition/update diagnostics"}) + "\n")


def configure_v11(root, ctx):
    """Bind the explicit run context and the T_B envelope on the production runner (config only)."""
    from . import phase_a_v12 as v12
    from . import stage2a_v11 as v11
    v11.N_CAP = N_CAP
    v11.ROLLOUT_N = ROLLOUT_N
    v11.MAX_UPDATES = MAX_UPDATES
    v11.DEV_EPISODES = DEV10_N
    v11.TASKS = (TASK,)
    v11.METHODS = METHODS
    v11.ENABLED_SPLITS = dict(v11.ENABLED_SPLITS)
    v11.ENABLED_SPLITS[TASK] = Path(ctx.train_split)
    v12.EVAL_POINTS = EVAL_POINTS
    v12.DEV10_N = DEV10_N
    v12._source_hashes = lambda r: make_source_hashes(r, ctx)
    v12._fixed_mechanism_diagnostic = _fixed_mechanism_scope
    # Same immutable-generation persistence / dev10 schedule as the historical T_B R1 runs.
    v11.save_ckpt = v12._save_ckpt_generation
    v11.start_episode = v12._start_episode_with_state
    v11.maybe_eval_at_n = v12._maybe_eval_at_points
    v11.select_checkpoint = v12._select_checkpoint
    v11.bind_run_context(ctx)
    return v11


# ----------------------------------------------------------------------------- GPU binding / evidence
def bind_worker_gpu(physical_index, environ=None) -> dict:
    """Pin one physical GPU for this process BEFORE torch/robosuite import.

    robosuite 1.4's EGL context maps a single-digit CUDA_VISIBLE_DEVICES straight to the EGL device
    index, so the physical index is also the render device id; torch sees it as cuda:0.
    """
    env = os.environ if environ is None else environ
    if isinstance(physical_index, bool) or int(physical_index) < 0:
        raise BindingError("invalid physical GPU index %r" % (physical_index,))
    env["CUDA_VISIBLE_DEVICES"] = str(int(physical_index))
    env["MUJOCO_GL"] = "egl"
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE"):
        env.pop(key, None)
    return {"physical_gpu_index": int(physical_index), "render_gpu_device_id": int(physical_index),
            "torch_device": "cuda:0", "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]}


def _smi(args, runner=subprocess.check_output) -> str:
    return runner(["nvidia-smi", *args], text=True)


def query_gpu_uuid(physical_index, runner=subprocess.check_output) -> str:
    return _smi(["-i", str(int(physical_index)), "--query-gpu=uuid", "--format=csv,noheader"], runner).strip()


def parse_process_table(text, pid) -> list:
    """Rows of `nvidia-smi` Processes table that belong to `pid`: [{'gpu': i, 'type': 'C'|'G'|'C+G'}]."""
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+(C\+G|C|G)\s+", line)
        if m and int(m.group(2)) == int(pid):
            rows.append({"gpu": int(m.group(1)), "type": m.group(3)})
    return rows


def gpu_evidence(pid, expected_physical, runner=subprocess.check_output) -> dict:
    """Physical-device evidence: which GPUs hold this process' CUDA and EGL(graphics) contexts."""
    rows = parse_process_table(_smi([], runner), pid)
    gpus = sorted({r["gpu"] for r in rows})
    types = sorted({r["type"] for r in rows})
    return {"pid": int(pid), "rows": rows, "gpus": gpus, "types": types, "expected_physical": int(expected_physical),
            "on_expected_only": bool(rows) and gpus == [int(expected_physical)],
            "graphics_context_seen": any("G" in t for t in types)}


# ----------------------------------------------------------------------------- smoke
def classify_step(skill, controller_exit, held, task_success, engineering_error=None) -> str:
    """Separate facts: controller exit, Verifier Held, task success, engineering failure."""
    if engineering_error:
        return "ENGINEERING_FAILURE"
    if controller_exit != "NORMAL_TERMINATION":
        return "CONTROLLER_EXIT:%s" % controller_exit
    if skill == "PICK" and held != "TRUE":
        return "HELD_NOT_CONFIRMED"  # NORMAL_TERMINATION + Held FALSE/UNKNOWN is not a grasp confirmation
    return "STEP_OK"


def classify_episode(step_verdicts, final_task_success) -> str:
    if not step_verdicts:
        return "FAIL:NO_STEPS"
    bad = [v for v in step_verdicts if v != "STEP_OK"]
    if bad:
        return "FAIL:%s" % bad[0]
    if final_task_success is not True:
        return "FAIL:TASK_NOT_SUCCESS"
    return "PASS"


def select_smoke_cases(root, n=2, episodes=None) -> list:
    """First n frozen-dev10 cases (case_id order) that have saved S0 b_plan_dev success trajectories."""
    root = Path(root)
    dev = set(split_lists(root)["dev"])
    path = root / S0_BPLAN_REL
    if episodes is None:
        if not path.is_file():
            raise BindingError("no reusable T_B S0 b_plan_dev trajectory records: %s" % S0_BPLAN_REL)
        episodes = json.loads(path.read_text(encoding="utf-8"))
    chosen = []
    for ep in sorted(episodes, key=lambda e: e["case_id"]):
        if ep["case_id"] not in dev or not ep.get("success") or ep.get("status") != "SUCCESS":
            continue
        decisions = ep.get("decisions") or []
        script = [d.get("first_action") for d in decisions]
        if not decisions or any(a is None for a in script) or len(script) > SMOKE_PER_EPISODE_SKILLS:
            continue
        if decisions[-1]["evaluator"].get("success") is not True:
            continue
        chosen.append({
            "case_id": ep["case_id"], "script": script,
            "historic_exits": [d["execution"]["controller_exit"] for d in decisions],
            "historic_sim_durations": [d["execution"]["sim_duration"] for d in decisions],
            "evidence_sha256": sha256_text(canonical(ep)),
        })
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise BindingError("fewer than %d reusable dev10 success trajectories (found %d)" % (n, len(chosen)))
    return chosen


class BudgetExceeded(BindingError):
    pass


class BudgetMeter:
    """Real call counts (attempts, not only successes) against hard caps."""

    def __init__(self, caps, used=None):
        self.caps = dict(caps)
        self.used = {k: 0 for k in caps}
        if used:
            self.used.update(used)

    def take(self, key, n=1):
        if self.used[key] + n > self.caps[key]:
            raise BudgetExceeded("%s cap %d would be exceeded" % (key, self.caps[key]))
        self.used[key] += n


@contextlib.contextmanager
def count_env_calls(meter):
    """Count real environment constructions/resets (attempts, incl. factory bootstrap) without altering them."""
    from .platforms.libero import d0_env
    cls = d0_env.D0ManipulationEnv
    orig_init, orig_reset = cls.__init__, cls.reset

    def counting_init(self, *a, **k):
        meter.take("construction_attempts")
        return orig_init(self, *a, **k)

    def counting_reset(self, *a, **k):
        meter.take("explicit_resets")
        return orig_reset(self, *a, **k)

    cls.__init__, cls.reset = counting_init, counting_reset
    try:
        yield meter
    finally:
        cls.__init__, cls.reset = orig_init, orig_reset


def smoke_budget_path(out_dir) -> Path:
    return Path(out_dir) / "smoke_budget.json"


def load_smoke_budget(out_dir) -> dict:
    p = smoke_budget_path(out_dir)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"caps": dict(SMOKE_CAPS), "used": {k: 0 for k in SMOKE_CAPS}, "episodes": []}


def preflight_smoke_budget(budget, per_episode=None) -> None:
    """Refuse to start a smoke episode that could exceed any cap (worst case, before any construction)."""
    need = {"episode_attempts": 1, "construction_attempts": PREFLIGHT_EXPECTED_PER_EPISODE["construction_attempts"],
            "explicit_resets": PREFLIGHT_EXPECTED_PER_EPISODE["explicit_resets"], "skill_calls": SMOKE_PER_EPISODE_SKILLS}
    need.update(per_episode or {})
    for key, n in need.items():
        if budget["used"][key] + n > budget["caps"][key]:
            raise BudgetExceeded("smoke budget: %s used %d + need %d > cap %d" % (key, budget["used"][key], n, budget["caps"][key]))


def run_smoke(root, out_dir, case_index, physical_gpu, ctx, v11_module=None, evidence_fn=gpu_evidence):
    """One real T_B engineering smoke: replay a saved S0 dev success trajectory through the production
    Collector/Verifier/Evaluator. Heavy imports happen here only, after token/source/budget checks."""
    out_dir = Path(out_dir)
    budget = load_smoke_budget(out_dir)
    if case_index == 2:
        first = [e for e in budget["episodes"] if e["index"] == 1]
        if not first or first[0]["episode_verdict"] != "PASS":
            raise BindingError("second smoke runs only if the first passed")
    if any(e["index"] == case_index for e in budget["episodes"]):
        raise BindingError("smoke case %d already attempted; no retries" % case_index)
    preflight_smoke_budget(budget)
    cases = select_smoke_cases(root, 2)
    case = cases[case_index - 1]
    selection_path = out_dir / "smoke_selection.json"
    if not selection_path.exists():  # frozen before any reset
        write_json_atomic(selection_path, {"frozen_before_any_reset": True, "cases": cases, "source": str(S0_BPLAN_REL),
                                           "split": str(SPLIT_REL), "split_sha256": split_lists(root)["sha256"]})
    import torch
    from .collector import Collector
    from .common import DataIntegrityError, DiagnosticAbort
    v11 = v11_module or configure_v11(root, ctx)
    meter = BudgetMeter(budget["caps"], budget["used"])
    meter.take("episode_attempts")
    result = {"index": case_index, "case_id": case["case_id"], "script": case["script"], "steps": [],
              "physical_gpu": int(physical_gpu), "render_gpu_device_id": ctx.render_gpu_device_id,
              "provider_requests": 0, "optimizer_steps": 0, "test_ids_accessed": 0}
    device = torch.device("cuda", 0)
    verdicts, final_success, engineering_error = [], None, None
    bundle = None
    try:
        with count_env_calls(meter):
            v11.bind_H(root)
            bundle = v11.make_bundle(root, TASK)
            result["gpu_after_factory"] = evidence_fn(os.getpid(), physical_gpu)
            if bundle.render_gpu_device_id != ctx.render_gpu_device_id or bundle.env_factory is None:
                raise BindingError("render device binding did not reach the runtime bundle")
            snap = bundle.start_case(case["case_id"])
            result["gpu_after_start_case"] = evidence_fn(os.getpid(), physical_gpu)
            snap, prior, source_n = v11.apply_episode_prior("B1-K", bundle, snap, None, eval_original=True)
            bundle.current_snapshot = snap
            result["initial"] = {"restore_receipt": bundle.restore_receipt, "snapshot_identity": bundle.snapshot_identity,
                                 "candidate_ids": list(snap.candidate_ids), "mask": [bool(m) for m in snap.mask],
                                 "prior_edges": len(snap.prior_edges), "prior_mode": prior.audit_mode,
                                 "facts": {k: getattr(v, "name", str(v)) for k, v in snap.facts.values.items()}}
            result["forward_probe"] = forward_probe(v11, bundle, snap, device)
            torch.manual_seed(0)
            policy = v11.make_policy(bundle.template, "B1-K", device)
            policy.eval()
            selector = ScriptedSelector(policy)
            collector = Collector(bundle, selector)
            collector.reset_episode(snap.env_id, snap.episode_id)
            clock_prev = float(bundle.clock.now_seconds())
            for i, action in enumerate(case["script"]):
                step = {"index": i, "action": action}
                if action not in snap.candidate_ids:
                    step.update(verdict="ACTION_NOT_IN_CANDIDATES")
                    result["steps"].append(step); verdicts.append(step["verdict"]); break
                if not snap.mask[snap.candidate_ids.index(action)]:
                    step.update(verdict="MASKED_ACTION")  # never forced through
                    result["steps"].append(step); verdicts.append(step["verdict"]); break
                meter.take("skill_calls")
                selector.target = action
                try:
                    t, res = collector.step(snap, deterministic=True)
                except (DiagnosticAbort, DataIntegrityError) as exc:
                    engineering_error = "%s: %s" % (type(exc).__name__, exc)
                    step.update(verdict="ENGINEERING_FAILURE", error=engineering_error)
                    result["steps"].append(step); verdicts.append("ENGINEERING_FAILURE"); break
                raw = dict(getattr(bundle.executor, "last", None) or {})
                if t is None:
                    step.update(verdict="NO_TRANSITION", result=str(res))
                    result["steps"].append(step); verdicts.append("NO_TRANSITION"); break
                skill, args = action.split(":")[1], action.split(":")[2]
                held = None
                if skill == "PICK":
                    value = t.next_snapshot.facts.values.get("p:Held:%s" % args)
                    held = None if value is None else getattr(value, "name", str(value))
                clock_now = float(bundle.clock.now_seconds())
                final_success = bool(res.success if not isinstance(res, dict) else res.get("success"))
                verdict = classify_step(skill, raw.get("controller_exit"), held, final_success)
                step.update(verdict=verdict, controller_exit=raw.get("controller_exit"), held_fact=held,
                            steps=raw.get("steps"), sim_duration=raw.get("sim_duration"), duration=t.duration,
                            clock_start=clock_prev, clock_end=clock_now, clock_monotonic=clock_now > clock_prev,
                            historic_duration_match=(abs(float(t.duration) - float(case["historic_sim_durations"][i])) < 1e-6),
                            task_success=final_success, evaluator_reason=getattr(res, "reason", None),
                            terminated=bool(t.terminated), truncated=bool(t.truncated),
                            mask_hash=sha256_text(canonical([bool(m) for m in snap.mask])))
                clock_prev = clock_now
                result["steps"].append(step); verdicts.append(verdict)
                snap = t.next_snapshot
                bundle.current_snapshot = snap
                if verdict != "STEP_OK" or t.terminated or t.truncated:
                    break
            result["gpu_end"] = evidence_fn(os.getpid(), physical_gpu)
    except BudgetExceeded:
        raise
    except Exception as exc:  # construction / reset / import failures are engineering failures, recorded
        engineering_error = "%s: %s" % (type(exc).__name__, exc)
        verdicts.append("ENGINEERING_FAILURE")
    finally:
        if bundle is not None:
            with contextlib.suppress(Exception):
                bundle.environment.close()
    gpu_ok = all(result.get(k, {}).get("on_expected_only") for k in ("gpu_after_factory", "gpu_after_start_case", "gpu_end")) if "gpu_end" in result else False
    verdict = classify_episode(verdicts, final_success)
    if verdict == "PASS" and not gpu_ok:
        verdict = "FAIL:RENDER_DEVICE_NOT_CONFIRMED"
    if verdict == "PASS" and not all(s.get("clock_monotonic") for s in result["steps"]):
        verdict = "FAIL:CLOCK_NOT_MONOTONIC"
    result.update(episode_verdict=verdict, engineering_error=engineering_error, final_task_success=final_success,
                  budget_used=dict(meter.used))
    budget["used"] = dict(meter.used)
    budget["episodes"].append({"index": case_index, "case_id": case["case_id"], "episode_verdict": verdict})
    write_json_atomic(out_dir / ("smoke_case%d.json" % case_index), result)
    write_json_atomic(smoke_budget_path(out_dir), budget)
    reread = json.loads((out_dir / ("smoke_case%d.json" % case_index)).read_text(encoding="utf-8"))
    result["persisted_sha256"] = sha256_file(out_dir / ("smoke_case%d.json" % case_index))
    result["reread_ok"] = reread["episode_verdict"] == verdict
    return result


class ScriptedSelector:
    """Policy proxy: real forward pass, but all probability mass on the next scripted *legal* candidate.
    Reuses the production Collector path unchanged; a masked candidate is never forced through."""

    def __init__(self, policy):
        self._policy = policy
        self.target = None

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def __call__(self, snapshot, hidden):
        import dataclasses
        import torch
        out = self._policy(snapshot, hidden)
        idx = out.candidate_ids.index(self.target)
        logits = torch.full_like(out.logits, float("-inf"))
        if bool(out.mask[idx]):
            logits[idx] = 0.0
        dist = torch.distributions.Categorical(logits=logits)
        return dataclasses.replace(out, logits=logits, distribution=dist)


def forward_probe(v11, bundle, snap, device) -> dict:
    """One real forward per method on the real initial snapshot (no env step, no optimizer)."""
    import torch
    probes = {}
    for method in METHODS:
        torch.manual_seed(0)
        policy = v11.make_policy(bundle.template, method, device)
        policy.eval()
        with torch.no_grad():
            out = policy(snap, policy.initial_hidden())
        diag = out.diagnostics or {}
        legal = [bool(m) for m in out.mask.tolist()]
        probes[method] = {
            "n_candidates": len(out.candidate_ids), "n_legal": sum(legal),
            "logits_finite_on_legal": bool(torch.isfinite(out.logits[out.mask]).all()),
            "value_finite": bool(torch.isfinite(out.value)),
            "successor_used": bool(diag.get("successor_used")),
            "effect_token_counts": {k: int(v) for k, v in (diag.get("effect_tokens") or {}).items()},
            "prior_inputs_all_zero": all(int(torch.count_nonzero(v)) == 0 for v in (diag.get("prior_inputs") or {}).values()),
            "effective_prior_edges": len(snap.prior_edges),
        }
    return probes


# ----------------------------------------------------------------------------- training (production runner)
def run_train(root, out_dir, ctx, physical_gpu, v11_module=None, ledger=None, git=git_out):
    """Reserve the plan, then call the production `train_job` with the bound context. No new trainer."""
    out_dir = Path(out_dir)
    ledger = ledger or Ledger(out_dir)
    check_source_identity(root, ctx.source_commit, git=git)
    ctx.assert_worker(TASK, ctx.method)
    if Path(ctx.output_directory).exists():
        raise BindingError("attempt directory already exists (no reuse): %s" % ctx.output_directory)
    ledger.reserve(ctx.plan_id, ctx.attempt_id, ctx.output_directory, os.getpid(), int(physical_gpu))
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(int(physical_gpu)) or torch.cuda.device_count() != 1:
        ledger.finish(ctx.plan_id, "STOPPED", reason="GPU binding not isolated")
        raise BindingError("worker must see exactly its one physical GPU")
    os.chdir(str(root))
    v11 = v11_module or configure_v11(root, ctx)
    device = torch.device("cuda", 0)
    prof = v11.bind_H(root)
    if prof["H"] != H_SECONDS or float(prof["d_ref"][TASK]) != D_REF_SECONDS or float(prof["Tcap"][TASK]) != TCAP_SECONDS:
        ledger.finish(ctx.plan_id, "STOPPED", reason="profile differs from frozen")
        raise BindingError("runtime profile differs from the frozen T_B values")
    prof["Ncap"], prof["max_updates"] = N_CAP, MAX_UPDATES
    hashes = make_source_hashes(root, ctx)
    cfg_sha = sha256_text(canonical(ctx.as_dict()))[:8]
    try:
        summary = v11.train_job(root, TASK, ctx.method, device, torch.cuda.get_device_name(0), prof, hashes,
                                ctx.attempt_id, cfg_sha, max_updates=MAX_UPDATES, resume=False)
    except BaseException as exc:
        ledger.finish(ctx.plan_id, "STOPPED", error="%s: %s" % (type(exc).__name__, exc))
        raise
    status = "COMPLETE" if summary.get("complete_updates") == MAX_UPDATES or summary.get("stop_reason") in ("Ncap", "Tcap") else "STOPPED"
    ledger.finish(ctx.plan_id, status, stop_reason=summary.get("stop_reason"),
                  complete_updates=summary.get("complete_updates"), valid_transitions=summary.get("valid_transitions"))
    return summary


def first_update_gate(run_dir) -> dict:
    """Engineering-only first-update gate. Success rate is never consulted."""
    run_dir = Path(run_dir)
    reasons = []
    chk_path = run_dir / "first_update_selfcheck.json"
    if not chk_path.is_file():
        return {"passed": False, "reasons": ["first_update_selfcheck.json missing"]}
    chk = json.loads(chk_path.read_text(encoding="utf-8"))
    if chk.get("n_transitions") != ROLLOUT_N:
        reasons.append("first update did not use %d transitions" % ROLLOUT_N)
    if not chk.get("optimizer_steps_total"):
        reasons.append("no optimizer steps")
    if chk.get("param_changed") is not True or chk.get("init_param_changed") is not True:
        reasons.append("parameters did not change")
    audit = chk.get("clock_audit") or {}
    if audit.get("N_clock_invalid") or audit.get("N_unknown") or audit.get("technical_n") or audit.get("mask_mismatch_n"):
        reasons.append("clock/mask audit not clean")
    if audit.get("N_clock_verified_valid") != ROLLOUT_N:
        reasons.append("not every transition clock-verified")
    fresh = list(run_dir.glob("fresh_load_update_complete_1*.json"))
    if not fresh:
        reasons.append("no fresh-process reload evidence for generation update_complete_1")
    store = run_dir / "persistence"
    try:
        from .persistence import GenerationStore
        GenerationStore(store).verify("update_complete_1")
    except Exception as exc:
        reasons.append("generation update_complete_1 not verifiable: %s" % exc)
    return {"passed": not reasons, "reasons": reasons, "optimizer_steps": chk.get("optimizer_steps_total"),
            "losses": chk.get("losses"), "pre_clip_logged": bool(chk.get("losses"))}


# ----------------------------------------------------------------------------- A0 scans (no env)
def classify_run_dir(run_dir, process_cmdlines=()) -> dict:
    run_dir = Path(run_dir)
    summary = run_dir / "job_summary.json"
    info = {"path": str(run_dir)}
    if summary.is_file():
        doc = json.loads(summary.read_text(encoding="utf-8"))
        full = doc.get("complete_updates") == MAX_UPDATES or doc.get("stop_reason") in ("Ncap", "Tcap")
        info.update(status="COMPLETE" if full and not doc.get("hard_fail") else "STOPPED",
                    complete_updates=doc.get("complete_updates"), fragment_updates=doc.get("fragment_updates"),
                    N=doc.get("valid_transitions"), T=doc.get("interaction_seconds"))
        return info
    if any(str(run_dir) in c for c in process_cmdlines):
        info["status"] = "RUNNING"
        return info
    has_weights = any(run_dir.glob("checkpoints/*.pt")) or any(run_dir.glob("persistence/generations/*/model.pt"))
    info["status"] = "STOPPED" if has_weights else "UNVERIFIED"
    info["note"] = "no job_summary; " + ("weights present" if has_weights else "no weights (sidecar-only or empty)")
    return info


def scan_plan_status(search_roots, process_cmdlines=()) -> dict:
    """Existing plan state from known run layouts only (no whole-disk scan)."""
    out = {}
    for pid, row in PLAN_TABLE.items():
        found = []
        for base in search_roots:
            base = Path(base)
            for layout in ("runs/final_master/2.1.1/T_B", "runs/v13_r1/T_B", "runs/stage_2a/T_B", "runs/v13_r3_recovery/T_B"):
                seed_dir = base / layout / row["method"] / ("seed_%d" % row["seed"])
                if seed_dir.is_dir():
                    for attempt in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                        found.append(classify_run_dir(attempt, process_cmdlines))
        if not found:
            out[pid] = {"method": row["method"], "seed": row["seed"], "status": "NOT_STARTED", "found": [],
                        "searched": [str(r) for r in search_roots]}
        else:
            order = ["RUNNING", "COMPLETE", "STOPPED", "UNVERIFIED"]
            status = sorted((f["status"] for f in found), key=order.index)[0]
            out[pid] = {"method": row["method"], "seed": row["seed"], "status": status, "found": found}
    return out


STATIC_FINDINGS = {
    "F1_cli_seed_and_E_choice": ("src/cp_disr/cli.py", "stage-2a-v11-run lacks --seed and B1-K+E"),
    "F2_train_job_seed_hardcoded": ("src/cp_disr/stage2a_v11.py", "train_job hardcodes seed 0"),
    "F3_plusE_not_empty_R": ("src/cp_disr/stage2a_v11.py", "EMPTY_PRIOR_METHODS lacks B1-K+E"),
    "F4_manifest_old_workspace": (str(RUNTIME_REL), "manifest paths bound to graph_cp_disr_v13_recovery"),
    "F5_r1_adapter_fixed": ("src/cp_disr/phase_a_v13_r1.py", "R1 adapter fixes seed0/old run ids/old workspace"),
    "F6_dry_run_builds_env": ("src/cp_disr/stage2a_v11.py", "local_dry_run/startup_hard_checks build environments"),
    "F7_phase_all_materialize": ("src/cp_disr/stage2a_v11.py", "--phase all touches cache/materialize/later stages"),
    "F8_render_device_not_explicit": ("src/cp_disr/platforms/libero/runtime_factory.py", "make_env render id not propagated"),
}


def _train_job_body(text):
    start = text.index("def train_job(")
    end = text.index("\ndef freeze_selection(", start)
    return text[start:end]


def static_review(read) -> dict:
    """`read(rel_path) -> str`. Detects the known entry problems by source text, before/after."""
    v11 = read("src/cp_disr/stage2a_v11.py")
    cli = read("src/cp_disr/cli.py")
    rt = read(str(RUNTIME_REL))
    r1 = read("src/cp_disr/phase_a_v13_r1.py")
    fac = read("src/cp_disr/platforms/libero/runtime_factory.py")
    body = _train_job_body(v11)
    empty = re.search(r"EMPTY_PRIOR_METHODS\s*=\s*\{([^}]*)\}", v11)
    cli_block = cli[cli.index('if name == "stage-2a-v11-run"'):][:900] if 'if name == "stage-2a-v11-run"' in cli else ""
    present = {
        "F1_cli_seed_and_E_choice": ("--seed" not in cli_block) and ("B1-K+E" not in cli_block),
        "F2_train_job_seed_hardcoded": "s1.seed_all(0)" in body,
        "F3_plusE_not_empty_R": not (empty and "B1-K+E" in empty.group(1)),
        "F4_manifest_old_workspace": "graph_cp_disr_v13_recovery" in rt,
        "F5_r1_adapter_fixed": "graph_cp_disr_v2_1" in r1 and '"seed_0"' in r1,
        "F6_dry_run_builds_env": "def local_dry_run" in v11 and "bundle.start_case" in v11[v11.index("def local_dry_run"):],
        "F7_phase_all_materialize": 'phase="all"' in v11 and "materialize_all" in v11,
        "F8_render_device_not_explicit": "render_gpu_device_id" not in fac.split("def create_task_runtime")[1].split("def create_ta_runtime")[0],
    }
    return {k: {"present": bool(v), "file": STATIC_FINDINGS[k][0], "what": STATIC_FINDINGS[k][1]} for k, v in present.items()}


def git_reader(root, ref):
    return lambda rel: subprocess.check_output(["git", "show", "%s:%s" % (ref, rel)], cwd=str(root), text=True)


def tree_reader(root):
    return lambda rel: (Path(root) / rel).read_text(encoding="utf-8")


FORBIDDEN_CALLS = {"local_dry_run", "startup_hard_checks", "materialize_all", "freeze_enabled_splits", "PriorSampler",
                   "DashScopeProvider", "request_and_process", "cmd_stage_2a_v11_run"}
FORBIDDEN_MODULES = {"phase_a_v13_r1", "phase_a_v13_r3", "stage2a_runner", "vlm_provider", "vlm_cache_pipeline"}


def entry_forbidden_hits(sources: dict) -> dict:
    """Code (not strings/comments) in the new entry must never name provider/materialize/dry-run/old-adapter paths."""
    import ast
    hits = {}
    for name, text in sources.items():
        found = set()
        for node in ast.walk(ast.parse(text)):
            ident = None
            if isinstance(node, ast.Name):
                ident = node.id
            elif isinstance(node, ast.Attribute):
                ident = node.attr
            elif isinstance(node, ast.alias):
                ident = node.name.split(".")[-1]
            elif isinstance(node, ast.ImportFrom) and node.module:
                ident = node.module.split(".")[-1]
            if ident in FORBIDDEN_CALLS or ident in FORBIDDEN_MODULES:
                found.add(ident)
        if found:
            hits[name] = sorted(found)
    return hits


# ----------------------------------------------------------------------------- check / register / summarize
def storage_report(paths) -> dict:
    out = {}
    for p in paths:
        st = os.statvfs(str(p))
        out[str(p)] = {"free_bytes": st.f_bavail * st.f_frsize, "total_bytes": st.f_blocks * st.f_frsize,
                       "free_gib": round(st.f_bavail * st.f_frsize / 2 ** 30, 2)}
    return out


def identity_report(root) -> dict:
    root = Path(root).resolve()
    import cp_disr
    return {
        "user": os.environ.get("USER") or os.popen("id -un").read().strip(),
        "hostname": os.uname().nodename, "python": sys.executable,
        "cp_disr_import": str(Path(cp_disr.__file__).resolve()),
        "import_under_root": str(Path(cp_disr.__file__).resolve()).startswith(str(root)),
        "root": str(root), "head": git_out(root, "rev-parse", "HEAD"), "branch": git_out(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_dirty": git_out(root, "status", "--porcelain", "--untracked-files=no"),
        "tmpdir": os.environ.get("TMPDIR"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run_check(root, out_dir, search_roots=(), baseline_ref=None, process_cmdlines=None) -> dict:
    """A0: identity, frozen profile, real split lists, plan status, static review. Builds nothing."""
    root, out_dir = Path(root).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if process_cmdlines is None:
        ps = subprocess.run(["ps", "-u", os.environ.get("USER", ""), "-o", "pid,cmd"], capture_output=True, text=True).stdout
        process_cmdlines = [l for l in ps.splitlines() if "final_tb" in l or "stage-2a" in l]
    profile = resolve_frozen_profile(root)
    lists = split_lists(root)
    status = scan_plan_status([root, *map(Path, search_roots)], process_cmdlines)
    review = {"after": static_review(tree_reader(root)), "baseline_ref": baseline_ref}
    if baseline_ref:
        review["before"] = static_review(git_reader(root, baseline_ref))
    review["entry_forbidden_hits"] = entry_forbidden_hits({
        "src/cp_disr/final_tb.py": (root / "src/cp_disr/final_tb.py").read_text(encoding="utf-8"),
        "scripts/final_tb_launch.py": (root / "scripts/final_tb_launch.py").read_text(encoding="utf-8")})
    report = {"identity": identity_report(root), "frozen_profile": profile,
              "split": {"train_n": len(lists["train"]), "dev_n": len(lists["dev"]), "dev_ids": lists["dev"],
                        "sha256": lists["sha256"]},
              "plan_status": status, "storage": storage_report([root, out_dir]),
              "processes_checked": process_cmdlines, "environment_constructions": 0, "provider_requests": 0}
    write_json_atomic(out_dir / "run_status_and_budget.json", {"plans": status, "budget": {
        "new_rl_attempts_cap": MAX_NEW_RL_ATTEMPTS, "new_rl_attempts_used": 0, "smoke": dict(SMOKE_CAPS),
        "provider": 0, "tp_training": 0, "formal_test": 0, "elastic": 0}, "storage": report["storage"]})
    write_json_atomic(out_dir / "binding_and_source_review.json", {"review": review, "report": report})
    return report


def run_register(root, out_dir, prep_commit, authorization_text, stamp=None, git=git_out) -> dict:
    """Derive manifest/split, write run configs and launch plan, init ledger, issue the smoke token."""
    root, out_dir = Path(root).resolve(), Path(out_dir)
    check_source_identity(root, prep_commit, git=git)
    if (out_dir / "launch_state.json").exists():
        raise BindingError("already registered: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    auth = {"authorization_text": authorization_text, "authorization_text_sha256": sha256_text(authorization_text),
            "card": "CP-DISR-NEXT-ENGINEERING-FIRST-TB-1", "baseline": "7776f3a5996acd20bcf77b15a09ec772b4d86a2a",
            "prep_commit": prep_commit, "caps": {"smoke": dict(SMOKE_CAPS), "new_rl_attempts": MAX_NEW_RL_ATTEMPTS,
                                                  "workers": MAX_WORKERS, "ncap": N_CAP, "tcap": TCAP_SECONDS},
            "not_authorized": ["provider", "T_P training", "automatic S2/S3", "formal test", "method upgrade", "elastic"],
            "registered": utc_now()}
    write_json_atomic(out_dir / "authorization.json", auth)
    split = derive_noprior_split(root, out_dir / "train_split_tb_noprior.json")
    manifest = derive_runtime_manifest(root, out_dir / "runtime_manifest_tb_resolved.yaml", split["path"])
    plans, configs = {}, {}
    for pid, row in sorted(PLAN_TABLE.items(), key=lambda kv: kv[1]["release_order"]):
        attempt = "%s-%s-%s" % (pid, stamp, prep_commit[:8])
        run_dir = root / "runs/final_master/2.1.1/T_B" / row["method"] / ("seed_%d" % row["seed"]) / attempt
        cfg = {"plan_id": pid, "attempt_id": attempt, "method": row["method"], "training_seed": row["seed"],
               "source_commit": prep_commit, "output_directory": str(run_dir),
               "runtime_manifest": manifest["path"], "train_split": split["path"], "prior_mode": PRIOR_MODE,
               "study_envelope_ncap": N_CAP, "study_envelope_tcap": TCAP_SECONDS, "evaluation_rule": EVALUATION_RULE}
        context_from_dict(cfg)  # validates: no placeholders, frozen values
        cfg_path = out_dir / "run_configs" / ("%s.yaml" % pid)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        os.chmod(cfg_path, 0o444)
        configs[pid] = {"path": str(cfg_path), "sha256": sha256_file(cfg_path)}
        plans[pid] = {"method": row["method"], "seed": row["seed"], "status": "NOT_STARTED", "attempt_id": attempt,
                      "planned_run_dir": str(run_dir), "release_order": row["release_order"]}
    Ledger(out_dir).init(plans)
    launch_plan = {"prep_commit": prep_commit, "stamp": stamp, "plans": plans, "configs": configs,
                   "derived": {"runtime_manifest": manifest, "train_split": split}, "frozen_runtime": FROZEN_RUNTIME,
                   "envelope": envelope(), "eval_points": list(EVAL_POINTS),
                   "release_order": [p for p, _ in sorted(PLAN_TABLE.items(), key=lambda kv: kv[1]["release_order"])]}
    write_json_atomic(out_dir / "launch_plan.json", launch_plan)
    token = issue_token(out_dir, "smoke", prep_commit)
    launch_plan["smoke_token"] = str(token)
    return launch_plan


def summarize_runs(out_dir) -> dict:
    out_dir = Path(out_dir)
    state = Ledger(out_dir).read()
    rows = {}
    for pid, plan in state["plans"].items():
        run_dir = plan.get("run_dir") or plan.get("planned_run_dir")
        entry = {"status": plan["status"], "method": plan["method"], "seed": plan["seed"], "run_dir": run_dir}
        rd = Path(run_dir)
        if rd.is_dir():
            js = rd / "job_summary.json"
            if js.is_file():
                doc = json.loads(js.read_text(encoding="utf-8"))
                entry.update({k: doc.get(k) for k in ("complete_updates", "fragment_updates", "valid_transitions",
                                                      "interaction_seconds", "optimizer_steps", "stop_reason", "hard_fail",
                                                      "train_success_episodes", "step0_dev_success", "final_dev_success")})
            em = rd / "eval_metrics.csv"
            if em.is_file():
                import csv
                with em.open(encoding="utf-8") as f:
                    entry["dev10"] = [{k: r.get(k) for k in ("skill_transitions", "success_n", "success_rate", "checkpoint")} for r in csv.DictReader(f)]
            ckpts = []
            for p in sorted(rd.glob("checkpoints/*.pt")):
                ckpts.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
            entry["checkpoints"] = ckpts
            entry["first_update_gate"] = first_update_gate(rd) if (rd / "first_update_selfcheck.json").is_file() else None
        rows[pid] = entry
    nontrivial = [pid for pid, e in rows.items() if e.get("method") == "B1-K+E" and any(
        int(r.get("skill_transitions") or 0) > 0 and int(r.get("success_n") or 0) >= 1 for r in e.get("dev10", []))]
    return {"plans": rows, "nontrivial_learning_plus_E": bool(nontrivial),
            "new_rl_attempts_used": state["new_rl_attempts_used"], "provider": 0, "tp_training": 0, "formal_test": 0, "elastic": 0}


def run_release(out_dir, stage, prep_commit, plan_id=None, evidence_dir=None, git=git_out, predecessor_blocked_reason=None) -> Path:
    """Issue a release token only when its prerequisite evidence exists."""
    out_dir = Path(out_dir)
    if stage == "train":
        if plan_id not in PLAN_TABLE:
            raise BindingError("unknown plan %r" % (plan_id,))
        receipt = out_dir / "smoke_receipt.json"
        if not receipt.is_file() or json.loads(receipt.read_text(encoding="utf-8")).get("label") != "TB_RUNTIME_SMOKE_PASS":
            raise BindingError("training release requires TB_RUNTIME_SMOKE_PASS smoke_receipt.json")
        order = PLAN_TABLE[plan_id]["release_order"]
        evidence = {"smoke_receipt_sha256": sha256_file(receipt)}
        if order > 1:  # later plans wait for the previous run's first-update engineering gate
            prev = [p for p, r in PLAN_TABLE.items() if r["release_order"] == order - 1][0]
            plan = Ledger(out_dir).read()["plans"][prev]
            if predecessor_blocked_reason:
                # method-specific binding block of the predecessor (never started, never ran): recorded, not silent
                if plan["status"] not in ("NOT_STARTED", "BLOCKED"):
                    raise BindingError("predecessor %s is %s; the block bypass applies only to a plan that never ran" % (prev, plan["status"]))
                evidence["predecessor_blocked"] = {"plan": prev, "status": plan["status"], "reason": str(predecessor_blocked_reason)}
            else:
                gate = first_update_gate(plan.get("run_dir", ""))
                if not gate["passed"]:
                    raise BindingError("previous run %s has not passed the first-update gate: %s" % (prev, gate["reasons"]))
                evidence["previous_first_update_gate"] = {"plan": prev, **gate}
        return issue_token(out_dir, "train", prep_commit, plan_id, evidence)
    raise BindingError("unsupported release stage %r" % stage)


def write_smoke_receipt(out_dir) -> dict:
    out_dir = Path(out_dir)
    budget = load_smoke_budget(out_dir)
    verdicts = {e["index"]: e["episode_verdict"] for e in budget["episodes"]}
    if verdicts.get(1) == "PASS" and verdicts.get(2) == "PASS":
        label = "TB_RUNTIME_SMOKE_PASS"
    elif any(v != "PASS" for v in verdicts.values()):
        label = "TB_LAUNCH_BLOCKED_RUNTIME"
    else:
        return {"final": False, "verdicts": verdicts}
    receipt = {"label": label, "verdicts": verdicts, "budget": budget, "provider_requests": 0, "optimizer_steps": 0,
               "test_ids_accessed": 0, "note": "engineering gate only; a failure blocks training and is not a method failure",
               "issued": utc_now()}
    write_json_atomic(out_dir / "smoke_receipt.json", receipt)
    return receipt


def phase_report(run_dir) -> dict:
    """Passive, phase-aware progress view. Never kills or restarts anything."""
    run_dir = Path(run_dir)
    now = time.time()
    marks = {}
    for name in ("transition_log.jsonl", "train_metrics.csv", "eval_metrics.csv", "persistence/raw_events.jsonl",
                 "first_update_selfcheck.json", "episode_log.jsonl"):
        p = run_dir / name
        if p.exists():
            marks[name] = round(now - p.stat().st_mtime, 1)
    return {"seconds_since_last_write": marks,
            "note": "PPO/publish/eval phases write no transitions; silence alone is not a stall"}


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(prog="final_tb_launch", description=(
        "CP-DISR final T_B launch entry. Explicit subcommands only (no default 'all'). "
        "check/register/status/summarize never build an environment; smoke/train need a release token."))
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("check", "register", "release", "smoke", "train", "summarize", "status"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        p.add_argument("--out", required=True, help="registration/evidence directory")
        if name == "check":
            p.add_argument("--search-root", action="append", default=[])
            p.add_argument("--baseline-ref")
        if name == "register":
            p.add_argument("--prep-commit", required=True)
            p.add_argument("--authorization-text-file", required=True)
            p.add_argument("--stamp")
        if name == "release":
            p.add_argument("--stage", choices=["train"], required=True)
            p.add_argument("--plan", required=True)
            p.add_argument("--predecessor-blocked-reason")
        if name in ("smoke", "train"):
            p.add_argument("--gpu", type=int, required=True)
            p.add_argument("--token")
        if name == "smoke":
            p.add_argument("--case-index", type=int, choices=[1, 2], required=True)
        if name == "train":
            p.add_argument("--plan", required=True)
    return ap


def _registered_prep(out):
    auth = Path(out) / "authorization.json"
    if not auth.is_file():
        raise BindingError("no registered authorization in %s" % out)
    return json.loads(auth.read_text(encoding="utf-8"))["prep_commit"]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out)
    try:
        if args.command == "check":
            report = run_check(root, out, args.search_root, args.baseline_ref)
            print(json.dumps({"plan_status": {k: v["status"] for k, v in report["plan_status"].items()}}))
        elif args.command == "register":
            plan = run_register(root, out, args.prep_commit, Path(args.authorization_text_file).read_text(encoding="utf-8"), args.stamp)
            print(json.dumps({"registered": True, "smoke_token": plan["smoke_token"]}))
        elif args.command == "release":
            print(json.dumps({"token": str(run_release(out, args.stage, _registered_prep(out), args.plan,
                                                       predecessor_blocked_reason=args.predecessor_blocked_reason))}))
        elif args.command == "summarize":
            summary = summarize_runs(out)
            write_json_atomic(out / "run_summary.json", summary)
            print(json.dumps({"plans": {k: v["status"] for k, v in summary["plans"].items()}}))
        elif args.command == "status":
            state = Ledger(out).read()
            print(json.dumps({pid: {"status": p["status"], **phase_report(p["run_dir"])} if p.get("run_dir") else {"status": p["status"]}
                              for pid, p in state["plans"].items()}))
        elif args.command in ("smoke", "train"):
            # 1) authorization first: no token -> refuse before GPU binding, torch or any environment code
            prep = _registered_prep(out)
            kind = "smoke" if args.command == "smoke" else "train"
            verify_token(args.token, out, kind, prep, None if kind == "smoke" else args.plan)
            check_source_identity(root, prep)
            launch = json.loads((out / "launch_plan.json").read_text(encoding="utf-8"))
            if args.command == "smoke":
                preflight_smoke_budget(load_smoke_budget(out))
            else:
                if args.plan not in PLAN_TABLE:
                    raise BindingError("unknown plan %r" % args.plan)
                cfg_path = Path(launch["configs"][args.plan]["path"])
                if sha256_file(cfg_path) != launch["configs"][args.plan]["sha256"]:
                    raise BindingError("run config changed after registration")
            # 2) only now: bind exactly one physical GPU before torch/robosuite import
            bind_worker_gpu(args.gpu)
            uuid = query_gpu_uuid(args.gpu)
            os.chdir(str(root))
            if args.command == "train":
                base = load_run_config(cfg_path)
                import dataclasses
                ctx = dataclasses.replace(base, physical_gpu_index=int(args.gpu), render_gpu_device_id=int(args.gpu), gpu_uuid=uuid)
                summary = run_train(root, out, ctx, args.gpu)
                print(json.dumps({"plan": args.plan, "stop_reason": summary.get("stop_reason"),
                                  "complete_updates": summary.get("complete_updates")}))
            else:
                ctx = RunContext(
                    plan_id="SMOKE-TB", attempt_id="smoke-case%d" % args.case_index, method="B1-K", training_seed=0,
                    source_commit=prep, output_directory=str(out / "smoke_run"),
                    runtime_manifest=launch["derived"]["runtime_manifest"]["path"], train_split=launch["derived"]["train_split"]["path"],
                    render_gpu_device_id=int(args.gpu), physical_gpu_index=int(args.gpu), gpu_uuid=uuid)
                res = run_smoke(root, out, args.case_index, args.gpu, ctx)
                receipt = write_smoke_receipt(out)
                print(json.dumps({"case": args.case_index, "verdict": res["episode_verdict"], "receipt": receipt.get("label")}))
        return 0
    except BindingError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
