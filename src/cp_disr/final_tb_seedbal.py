"""CP-DISR-TB-SEED-BALANCE-01 launch binding: matched current-profile seeds for B2 and B1-K+E (three attempts).

No trainer and no scientific change: the production path (``final_tb.run_train`` -> ``stage2a_v11.train_job``) is reused
unchanged for the legacy methods B2 and B1-K+E. This module only adds the registration, ledger, preflight, release tokens,
three-worker launch and a three-worker storage guard for exactly three new attempts:

    R-TB-B2-0-CURRENT  (B2, seed 0)   R-TB-B2-2-CURRENT  (B2, seed 2)   R-TB-E-2-CURRENT  (B1-K+E, seed 2)

Import-light: nothing here imports torch or the simulator at module import time.
"""
from __future__ import annotations

import argparse
import ast
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

CARD = "CP-DISR-TB-SEED-BALANCE-01"
RESULT_COMMIT = "4d0c426f67dfa1ce23247f89879c0b4952367d53"  # codex/cp-disr-tb-representation-controls result commit (full SHA, resolved)
BASE_COMMIT = rep.BASE_COMMIT
TASK = ftb.TASK
CLI = "final_tb_seedbal_launch.py"
OLD_ROOT = "/home/xushijie2/graph_cp_disr_final_tb_launch"
PREP_REL = Path("runs/final_master/2.1.1/seedbal/prep")
RUN_ROOT_REL = Path("runs/final_master/2.1.1/T_B")
MAX_WORKERS = 3
MAX_ATTEMPTS = 3
PLAN_TABLE = {
    "R-TB-B2-0-CURRENT": {"method": "B2", "seed": 0, "queue_order": 1, "reference_plan": "R-TB-DK-1"},
    "R-TB-B2-2-CURRENT": {"method": "B2", "seed": 2, "queue_order": 2, "reference_plan": "R-TB-DK-1"},
    "R-TB-E-2-CURRENT": {"method": "B1-K+E", "seed": 2, "queue_order": 3, "reference_plan": "R-TB-E-0"},
}
METHODS = ("B1-K+E", "B2")
# existing current-profile evidence (read-only): plan id -> (method, seed, launch-dir run config, launch dir)
EXISTING = {
    "R-TB-DK-1": {"method": "B2", "seed": 1, "config": "runs/final_master/2.1.1/launch/20261001T115251Z_smoker2_cac2fa5b/run_configs/R-TB-DK-1.yaml"},
    "R-TB-E-0": {"method": "B1-K+E", "seed": 0, "config": "runs/final_master/2.1.1/launch/20261001T115251Z_smoker2_cac2fa5b/run_configs/R-TB-E-0.yaml"},
    "R-TB-E-1": {"method": "B1-K+E", "seed": 1, "config": "runs/final_master/2.1.1/launch/20261002T031938Z_e1_elastic01_e4dc34ae/run_configs/R-TB-E-1.yaml"},
}
ALLOWED_CONFIG_DIFFERENCES = {"plan_id", "attempt_id", "training_seed", "source_commit", "output_directory", "runtime_manifest", "train_split"}
EXECUTION_PATH_FILES = (
    "src/cp_disr/rl.py", "src/cp_disr/collector.py", "src/cp_disr/torch_rl.py", "src/cp_disr/adapters.py", "src/cp_disr/evaluation.py",
    "src/cp_disr/execution.py", "src/cp_disr/graph.py", "src/cp_disr/contracts.py", "src/cp_disr/facts.py", "src/cp_disr/phase_a_v12.py",
    "src/cp_disr/persistence.py", "src/cp_disr/stage2a_v11.py", "src/cp_disr/neural.py", "src/cp_disr/stage1a_smoke.py",
    "src/cp_disr/platforms/libero/skill_executor.py", "src/cp_disr/platforms/libero/verifier.py", "src/cp_disr/platforms/libero/task_evaluator.py",
    "src/cp_disr/platforms/libero/runtime_factory.py", "src/cp_disr/platforms/libero/snapshot.py", "src/cp_disr/platforms/libero/safety.py",
    "src/cp_disr/platforms/libero/clock.py", "src/cp_disr/platforms/libero/observations.py", "src/cp_disr/platforms/libero/perception.py",
    "configs/runtime/stage_2a_contract_registry.yaml", "experiments/manifests/runtime_manifest_v211.yaml", "configs/splits/T_B_phase_a_v13_r1_dev10.json",
    "configs/tasks/resolved/T_B.yaml", "runs/stage_0a/reference_execution_manifest.json",
)
FINAL_TB_FUNCTIONS = ("run_train", "configure_v11", "make_source_hashes", "bind_worker_gpu", "check_source_identity", "resolve_frozen_profile", "split_lists",
                      "derive_noprior_split", "derive_runtime_manifest", "envelope", "envelope_stop")
NOT_AUTHORIZED = ["seed 3 or above", "new ABS / NC runs", "B1-K / B0", "T_P", "Family B", "RoboCasa", "provider", "test30", "structural generalization",
                  "a retry or replacement of any started attempt", "any change of representation, task, training budget or evaluator", "more than three workers"]
EARLY_RULES = {
    "EARLY_SUCCESS": "dev10 == 10/10 at N=4096 or N=8192 (first_effective_checkpoint = 4096 if 4096 already 10/10, else 8192)",
    "FINAL_ONLY": "4096 != 10/10 and 8192 != 10/10 and final == 10/10",
    "NEVER": "final != 10/10",
}
CONCLUSION_RULES = (
    "if B2 EARLY seeds <= 1 of 3 -> SEED_SENSITIVITY_HIGH",
    "elif +E EARLY seeds >= 1 of 3 -> MIXED_REPRESENTATION_PATTERN",
    "elif B2 >= 2/3 EARLY and ABS >= 2/3 EARLY and +E 0/3 EARLY and NC 0/3 EARLY -> MECHANISM_PATTERN_REPLICATED",
    "else -> INCONCLUSIVE",
)


def utc_now():
    return ftb.utc_now()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def blob_sha(root, ref, rel):
    try:
        data = subprocess.check_output(["git", "show", "%s:%s" % (ref, rel)], cwd=str(root), stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- plan / context / ledger
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
class SeedbalRunContext(ftb.RunContext):
    def assert_worker(self, task_id, method):
        if task_id != self.task_id:
            raise BindingError("run context is bound to %s, not %s" % (self.task_id, task_id))
        if method != self.method:
            raise BindingError("run context is bound to %s, not %s" % (self.method, method))
        validate_assignment(self.plan_id, self.method, self.training_seed)


def context_from_dict(cfg: dict) -> SeedbalRunContext:
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
    return SeedbalRunContext(
        plan_id=cfg["plan_id"], attempt_id=str(cfg["attempt_id"]), method=cfg["method"], training_seed=int(cfg["training_seed"]),
        source_commit=cfg["source_commit"], output_directory=str(cfg["output_directory"]), runtime_manifest=str(cfg["runtime_manifest"]),
        train_split=str(cfg["train_split"]), prior_mode=ftb.PRIOR_MODE, study_envelope_ncap=ftb.N_CAP, study_envelope_tcap=ftb.TCAP_SECONDS,
        evaluation_rule=ftb.EVALUATION_RULE, task_id=TASK, render_gpu_device_id=int(cfg.get("render_gpu_device_id", 0)),
        physical_gpu_index=None if cfg.get("physical_gpu_index") is None else int(cfg["physical_gpu_index"]), gpu_uuid=cfg.get("gpu_uuid"))


def load_run_config(path) -> SeedbalRunContext:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise BindingError("run config must be a mapping")
    return context_from_dict(cfg)


class SeedbalLedger(ftb.Ledger):
    """Three attempts, three concurrent workers, one attempt per plan; started attempts are never removed or renumbered."""

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


# ----------------------------------------------------------------------------- preflight
def _function_asts(source: str, names) -> dict:
    tree = ast.parse(source)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            out[node.name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
    return out


def _normalize_manifest(text: str, split_path: str) -> str:
    text = text.replace(split_path, "<TRAIN_SPLIT>")
    return re.sub(r"/home/(?:xushijie2|__compress_data/xushijie)/graph_cp_disr_[A-Za-z0-9_]+", "<ROOT>", text)


def build_preflight(root, out_dir, prep_commit, split, manifest, configs, git_fn=git, old_root=OLD_ROOT, ps_text=None, free_fn=None) -> dict:
    """Training-free equivalence evidence: new plans vs the existing current-profile runs, apart from seed / attempt / path."""
    root, out_dir = Path(root), Path(out_dir)
    checks, rows = {}, {}
    # ---- 0. identity (full 40-hex SHAs only)
    head = git_fn(root, "rev-parse", "HEAD")
    refs = {"prep_commit_head": head, "result_commit_expected": RESULT_COMMIT, "base_commit": BASE_COMMIT}
    for name, sha in refs.items():
        if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise BindingError("%s is not a full SHA" % name)
    checks["prep_is_head"] = head == prep_commit
    try:
        git_fn(root, "merge-base", "--is-ancestor", RESULT_COMMIT, head)
        checks["result_commit_is_ancestor"] = True
    except subprocess.CalledProcessError:
        checks["result_commit_is_ancestor"] = False
    checks["tracked_tree_clean"] = git_fn(root, "status", "--porcelain", "--untracked-files=no") == ""
    ps_text = ps_text if ps_text is not None else subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    leftovers = [l.strip()[:160] for l in ps_text.splitlines()[1:] if re.search(r"final_tb_repctl_launch\.py|final_tb_seedbal_launch\.py|final_tb_e1_launch\.py|final_tb_launch\.py\s+train|stage-2a-v11-run", l)]
    checks["no_residual_training_worker"] = not leftovers
    changed_tracked = git_fn(root, "diff", "--name-only", "--diff-filter=MD", RESULT_COMMIT, "HEAD").split()
    checks["no_pre_existing_tracked_file_modified_since_result_commit"] = changed_tracked == []
    rows["modified_tracked_files_since_result_commit"] = changed_tracked
    # ---- 1. existing runs
    existing = {}
    for plan_id, meta in EXISTING.items():
        dirs = sorted(Path(old_root).glob("runs/final_master/2.1.1/T_B/%s/seed_%d/%s-*" % (meta["method"], meta["seed"], plan_id)))
        if len(dirs) != 1:
            raise BindingError("expected exactly one existing run dir for %s" % plan_id)
        cfg = json.loads((dirs[0] / "resolved_config.json").read_text())
        job = json.loads((dirs[0] / "job_summary.json").read_text())
        existing[plan_id] = {"run_dir": str(dirs[0]), "source_commit": cfg["hashes"]["git_commit"], "method": job["method"], "seed": cfg["training_seed"],
                             "H": job["H"], "d_ref": job["d_ref"], "Tcap": job["Tcap"], "Ncap": cfg["Ncap"], "deadline": cfg["task_deadline_seconds"],
                             "train_split_derived_sha256": cfg["hashes"]["train_split_derived"], "dev10_sha256": cfg["hashes"]["dev10_split_original"],
                             "neural_py_sha256": cfg["hashes"]["neural"], "torch_rl_py_sha256": cfg["hashes"]["torch_rl"], "stage2a_v11_sha256": cfg["hashes"]["stage2a_v11"],
                             "n_params_from_final_checkpoint": None}
    rows["existing_runs"] = existing
    commits = sorted({v["source_commit"] for v in existing.values()})
    rows["existing_source_commits_full"] = commits
    # ---- 2. execution-path files and production functions
    file_rows, differing = {}, []
    for rel in EXECUTION_PATH_FILES:
        row = {c: blob_sha(root, c, rel) for c in commits}
        row["HEAD"] = blob_sha(root, "HEAD", rel)
        row["equal"] = len({v for v in row.values()}) == 1
        file_rows[rel] = row
        if not row["equal"]:
            differing.append(rel)
    rows["execution_path_files"] = file_rows
    func_rows = {}
    for c in commits + ["HEAD"]:
        src = subprocess.check_output(["git", "show", "%s:src/cp_disr/final_tb.py" % c], cwd=str(root), text=True)
        func_rows[c] = _function_asts(src, FINAL_TB_FUNCTIONS)
    rows["final_tb_production_function_ast"] = func_rows
    checks["execution_path_files_identical"] = differing == []
    rows["execution_path_files_differing"] = differing
    checks["final_tb_production_functions_identical"] = all(func_rows[c] == func_rows["HEAD"] for c in commits)
    # ---- 3. frozen constants and splits
    ref = next(iter(existing.values()))
    checks["frozen_constants_equal_existing"] = all(
        v["H"] == ftb.H_SECONDS and v["d_ref"] == ftb.D_REF_SECONDS and v["Tcap"] == ftb.TCAP_SECONDS and v["Ncap"] == ftb.N_CAP and v["deadline"] == ftb.TASK_DEADLINE_SECONDS
        for v in existing.values())
    checks["train_split_equal_existing"] = all(split["sha256"] == v["train_split_derived_sha256"] for v in existing.values())
    checks["dev10_split_equal_existing"] = all(split["source_sha256"] == v["dev10_sha256"] for v in existing.values())
    rows["train_split_sha256"] = split["sha256"]
    rows["dev10_source_sha256"] = split["source_sha256"]
    # ---- 4. scientific config equivalence per new plan vs its reference run config
    cfg_rows = {}
    ok = True
    for plan_id, row in PLAN_TABLE.items():
        new_cfg = yaml.safe_load(Path(configs[plan_id]["path"]).read_text())
        refs_ids = ["R-TB-DK-1"] if row["method"] == "B2" else ["R-TB-E-0", "R-TB-E-1"]
        per_ref = {}
        for rid in refs_ids:
            cfg_file = root / EXISTING[rid]["config"]
            cfg_file = cfg_file if cfg_file.is_file() else Path(old_root) / EXISTING[rid]["config"]
            old_cfg = yaml.safe_load(cfg_file.read_text())
            diff = sorted(k for k in set(new_cfg) | set(old_cfg) if new_cfg.get(k) != old_cfg.get(k))
            per_ref[rid] = {"differing_keys": diff, "only_allowed": set(diff) <= ALLOWED_CONFIG_DIFFERENCES, "method_equal": new_cfg["method"] == old_cfg["method"]}
            ok = ok and per_ref[rid]["only_allowed"] and per_ref[rid]["method_equal"]
        cfg_rows[plan_id] = per_ref
    rows["run_config_vs_existing"] = cfg_rows
    checks["run_configs_differ_only_in_seed_attempt_source_path"] = ok
    # runtime manifest (derived) modulo paths
    manifest_rel = "runs/final_master/2.1.1/launch/20261001T115251Z_smoker2_cac2fa5b/runtime_manifest_tb_resolved.yaml"
    manifest_file = root / manifest_rel if (root / manifest_rel).is_file() else Path(old_root) / manifest_rel
    old_manifest = manifest_file.read_text()
    old_text = re.sub(r"/home/[^\s:]*train_split_tb_noprior\.json", "<TRAIN_SPLIT>", old_manifest)
    new_text = Path(manifest["path"]).read_text()
    new_text = new_text.replace(split["path"], "<TRAIN_SPLIT>")
    checks["runtime_manifest_equal_modulo_paths"] = (re.sub(r"/home/(?:xushijie2|__compress_data/xushijie)/graph_cp_disr_[A-Za-z0-9_]+", "<ROOT>", old_text)
                                                    == re.sub(r"/home/(?:xushijie2|__compress_data/xushijie)/graph_cp_disr_[A-Za-z0-9_]+", "<ROOT>", new_text))
    # ---- 5. PPO / optimizer / schedule read from code
    import inspect
    from . import torch_rl
    ppo = {"lr": inspect.signature(torch_rl.PPO.__init__).parameters["lr"].default, "epochs": inspect.signature(torch_rl.PPO.update).parameters["epochs"].default,
           "minibatch": inspect.signature(torch_rl.PPO.update).parameters["minibatch"].default, "sequence_length": inspect.signature(torch_rl.PPO.update).parameters["sequence_length"].default,
           "rollout_n": ftb.ROLLOUT_N, "max_updates": ftb.MAX_UPDATES, "eval_points": list(ftb.EVAL_POINTS), "prior_mode": ftb.PRIOR_MODE}
    rows["ppo_and_schedule"] = ppo
    checks["torch_rl_identical_to_existing_runs"] = all(blob_sha(root, c, "src/cp_disr/torch_rl.py") == blob_sha(root, "HEAD", "src/cp_disr/torch_rl.py") for c in commits)
    # ---- 6. parameter count vs existing final checkpoints
    import torch
    from . import tb_repctl_checks as C
    template = C.tb_template(root)
    counts = {}
    for method in METHODS:
        counts[method] = sum(p.numel() for p in C.make_policy(template, method, 0).parameters())
    rows["fresh_parameter_count"] = counts
    ck_ok = True
    for plan_id, meta in EXISTING.items():
        ck = sorted(Path(existing[plan_id]["run_dir"]).glob("checkpoints/final_n_*.pt"))
        state = torch.load(ck[0], map_location="cpu", weights_only=False)["model"]
        n = sum(v.numel() for v in state.values())
        existing[plan_id]["n_params_from_final_checkpoint"] = n
        ck_ok = ck_ok and n == counts[meta["method"]]
    checks["parameter_count_equal_existing_checkpoints"] = ck_ok
    # ---- 7. seed chain
    chain = seed_chain_evidence(root)
    rows["seed_chain"] = chain
    checks["seed_chain_static_and_runtime"] = chain["ok"]
    # ---- 8. storage estimate
    sizes = []
    for plan_id in existing:
        sizes.append(sum(f.stat().st_size for f in Path(existing[plan_id]["run_dir"]).rglob("*") if f.is_file()))
    free = (free_fn or rep.fs_free)(out_dir)
    peak = int(max(sizes) * 3 * ftb.STORAGE_TEMP_MARGIN)
    rows["storage"] = {"existing_run_bytes": sizes, "estimated_peak_bytes_3_runs_with_margin": peak, "free_bytes_now": int(free),
                       "thresholds_used_by_guard": e1.STORAGE_GATE, "hard_reserved_margin_ok": free - peak >= e1.STORAGE_GATE["hard_reserved_margin_bytes"]}
    checks["storage_headroom"] = rows["storage"]["hard_reserved_margin_ok"] and free >= e1.STORAGE_GATE["start_free_min_bytes"]
    checks["no_smoke_needed_registration_only"] = changed_tracked == [] and differing == [] and checks["final_tb_production_functions_identical"]
    return {"card": CARD, "created": utc_now(), "refs": refs, "plan_table": PLAN_TABLE, "checks": checks, "details": rows, "smoke_episodes": 0,
            "verdict": "PASS" if all(checks.values()) else "FAIL"}


def seed_chain_evidence(root) -> dict:
    """The explicit training seed reaches Python/NumPy/Torch RNG, policy initialisation, manifests and eval metadata; findings that are
    seed-independent by production design are recorded, not changed."""
    root = Path(root)
    v11_src = (root / "src/cp_disr/stage2a_v11.py").read_text()
    tree = ast.parse(v11_src)
    train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_job")
    body_src = ast.get_source_segment(v11_src, train)
    seed_assign = re.search(r"seed = 0 if ctx is None else int\(ctx\.training_seed\)", body_src) is not None
    seed_all_calls = re.findall(r"s1\.seed_all\(([^)]*)\)", body_src)
    smoke = (root / "src/cp_disr/stage1a_smoke.py").read_text()
    seed_all_def = re.search(r"def seed_all\(seed\):\n\s+random\.seed\(seed\)\n\s+np\.random\.seed\(seed\)\n\s+torch\.manual_seed\(seed\)", smoke) is not None
    eval_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "eval_episodes")
    eval_src = ast.get_source_segment(v11_src, eval_fn)
    eval_isolated = "rng_before = s1.capture_rng()" in eval_src and "s1.restore_rng(rng_before)" in eval_src
    rf = (root / "src/cp_disr/platforms/libero/runtime_factory.py").read_text()
    next_case = re.search(r"def next_case\(self, task_cases, seed\):\n\s+return task_cases\[self\._n % len\(task_cases\)\]", rf) is not None
    import torch
    from . import tb_repctl_checks as C
    template = C.tb_template(root)

    def digest(method, seed):
        policy = C.make_policy(template, method, seed)
        h = hashlib.sha256()
        for k, v in policy.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    init = {m: {s: digest(m, s) for s in (0, 1, 2)} for m in METHODS}
    init_ok = all(len(set(init[m].values())) == 3 for m in METHODS) and all(digest(m, 2) == init[m][2] for m in METHODS)
    ok = seed_assign and seed_all_calls == ["seed"] and seed_all_def and eval_isolated and init_ok
    return {"train_job_seed_from_run_context": seed_assign, "train_job_seed_all_calls": seed_all_calls, "seed_all_sets_python_numpy_torch": seed_all_def,
            "eval_rng_isolated_capture_restore": eval_isolated, "policy_init_differs_per_seed_and_repeats": init_ok,
            "policy_init_digest_prefix": {m: {s: d[:16] for s, d in v.items()} for m, v in init.items()},
            "finding_episode_order_is_seed_independent_by_production_design": {"next_case_ignores_seed": next_case,
                "meaning": "the train-case cycle is task_cases[n % 64] for every run, existing and new alike; seeds differ through Python/NumPy/Torch RNG "
                           "(policy init, action sampling) and simulator noise, not through case order. Unchanged: changing it would alter production semantics."},
            "ok": bool(ok and next_case)}


def verify_preflight(out_dir, prep_commit) -> dict:
    doc = _json(Path(out_dir) / "seed_balance_preflight.json")
    if doc.get("card") != CARD or doc.get("verdict") != "PASS" or doc["refs"]["prep_commit_head"] != prep_commit:
        raise BindingError("seed_balance_preflight.json is not a PASS for this prep commit")
    return doc


# ----------------------------------------------------------------------------- registration / release / train
def register(root, out_dir, prep_commit, authorization_text, stamp=None, git_fn=ftb.git_out, preflight_fn=None) -> dict:
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    ftb.check_source_identity(root, prep_commit, git=git_fn)
    if (out_dir / "launch_state.json").exists():
        raise BindingError("already registered: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    auth = {"card": CARD, "prep_commit": prep_commit, "result_commit": RESULT_COMMIT, "plan_table": PLAN_TABLE,
            "queue": sorted(PLAN_TABLE, key=lambda p: PLAN_TABLE[p]["queue_order"]), "authorization_text": authorization_text,
            "authorization_text_sha256": ftb.sha256_text(authorization_text), "early_rules": EARLY_RULES, "conclusion_rules": CONCLUSION_RULES,
            "caps": {"rl_attempts": MAX_ATTEMPTS, "max_concurrent_workers": MAX_WORKERS, "ncap": ftb.N_CAP, "tcap": ftb.TCAP_SECONDS, "provider": 0,
                     "tp_training": 0, "formal_test": 0, "test30_read": 0, "smoke_episodes": 0}, "not_authorized": NOT_AUTHORIZED, "registered": utc_now()}
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
        ledger_plans[plan_id] = {"method": row["method"], "seed": row["seed"], "status": "NOT_STARTED", "attempt_id": attempt, "planned_run_dir": str(run_dir),
                                 "queue_order": row["queue_order"]}
    SeedbalLedger(out_dir).init(ledger_plans)
    pre = (preflight_fn or build_preflight)(root, out_dir, prep_commit, split, manifest, configs)
    ftb.write_json_atomic(out_dir / "seed_balance_preflight.json", pre)
    plan = {"card": CARD, "prep_commit": prep_commit, "stamp": stamp, "plans": ledger_plans, "configs": configs, "derived": {"runtime_manifest": manifest, "train_split": split},
            "frozen_runtime": ftb.FROZEN_RUNTIME, "frozen_profile": profile, "envelope": ftb.envelope(), "eval_points": list(ftb.EVAL_POINTS),
            "train_ids_sha256": ftb.sha256_text(canonical(splits["train"])), "dev_ids_sha256": ftb.sha256_text(canonical(splits["dev"])),
            "queue": auth["queue"], "init": "FROM_SCRATCH", "warm_start": False, "preflight_verdict": pre["verdict"]}
    ftb.write_json_atomic(out_dir / "launch_plan.json", plan)
    return plan


def release(root, out_dir, prep, free_fn=None, git_fn=ftb.git_out) -> list:
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    ftb.check_source_identity(root, prep, git=git_fn)
    verify_preflight(out_dir, prep)
    auth = _json(out_dir / "authorization.json")
    if auth.get("card") != CARD or auth.get("prep_commit") != prep:
        raise BindingError("registration is not this card's / prep commit")
    state = SeedbalLedger(out_dir).read()
    if any(v["status"] != "NOT_STARTED" for v in state["plans"].values()):
        raise BindingError("tokens are issued once, before any run starts")
    gate = rep.start_gate(next(iter(state["plans"].values()))["planned_run_dir"], free_fn)
    evidence = {"preflight_sha256": ftb.sha256_file(out_dir / "seed_balance_preflight.json"), "storage_start_gate": gate, "tokens_issued_together": True}
    return [ftb.issue_token(out_dir, "train", prep, plan_id, evidence) for plan_id in sorted(PLAN_TABLE, key=lambda p: PLAN_TABLE[p]["queue_order"])]


def run_seedbal_train(root, out, plan_id, gpu, token_path, git_fn=ftb.git_out, free_fn=None, runner=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    auth = _json(out / "authorization.json")
    prep = auth["prep_commit"]
    if auth.get("card") != CARD:
        raise BindingError("registration is not this card's")
    ftb.verify_token(token_path, out, "train", prep, plan_id)
    ftb.check_source_identity(root, prep, git=git_fn)
    verify_preflight(out, prep)
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
    return (runner or ftb.run_train)(root, out, ctx, gpu, ledger=SeedbalLedger(out))


def launch_all(root, out, python=sys.executable, gpu_fn=rep.gpu_table, spawn=None, log=print) -> dict:
    """Start the three plans at once on three distinct idle GPUs (one-shot; no supervisor, no retry)."""
    root, out = Path(root).resolve(), Path(out).resolve()
    state = SeedbalLedger(out).read()
    if any(v["status"] != "NOT_STARTED" for v in state["plans"].values()):
        raise BindingError("launch is one-shot: a plan is already started")
    spawn = spawn or (lambda cmd, logfile: subprocess.Popen(cmd, stdout=open(logfile, "ab"), stderr=subprocess.STDOUT, start_new_session=True,
                                                            env={**os.environ, "PYTHONPATH": str(root / "src")}, cwd=str(root)))
    table, taken, started = gpu_fn(), set(), {}
    (out / "logs").mkdir(exist_ok=True)
    plans = sorted(state["plans"], key=lambda p: state["plans"][p]["queue_order"])
    assignment = {}
    for plan_id in plans:
        gpu = rep.pick_gpu(table, taken)
        if gpu is None:
            raise BindingError("fewer than three idle GPUs; nothing was started")
        taken.add(gpu)
        assignment[plan_id] = gpu
    for plan_id, gpu in assignment.items():
        cmd = [python, str(root / "scripts" / CLI), "train", "--root", str(root), "--out", str(out), "--plan", plan_id, "--gpu", str(gpu),
               "--token", str(out / "release_tokens" / ("train_%s.json" % plan_id))]
        proc = spawn(cmd, str(out / "logs" / ("train_%s.log" % plan_id)))
        started[plan_id] = {"gpu": gpu, "pid": proc.pid, "time": utc_now()}
        log("[seedbal] started %s on GPU %s pid %s" % (plan_id, gpu, proc.pid))
    ftb.write_json_atomic(out / "launch_started.json", started)
    return started


# ----------------------------------------------------------------------------- storage guard (three workers)
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
    state = SeedbalLedger(out).read()
    return {"card": state.get("card"), "attempts_used": state["new_rl_attempts_used"], "attempts_cap": state["new_rl_attempts_cap"],
            "plans": {p: {k: v.get(k) for k in ("method", "seed", "status", "physical_gpu", "pid", "stop_reason", "complete_updates", "valid_transitions")} for p, v in state["plans"].items()}}


def build_parser():
    ap = argparse.ArgumentParser(prog="final_tb_seedbal_launch", description="%s entry. Explicit subcommands only." % CARD)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("register", "release", "train", "launch", "guard", "status"):
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
        if name == "guard":
            p.add_argument("--wait-seconds", type=float, default=900.0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root, out = Path(args.root).resolve(), Path(args.out).expanduser().resolve()
    try:
        if args.command == "register":
            plan = register(root, out, args.prep_commit, Path(args.authorization_text_file).read_text(encoding="utf-8"), stamp=args.stamp)
            print(canonical({"registered": True, "preflight_verdict": plan["preflight_verdict"], "plans": sorted(plan["plans"])}))
        elif args.command == "release":
            print(canonical([str(p) for p in release(root, out, _json(out / "authorization.json")["prep_commit"])]))
        elif args.command == "train":
            print(canonical(run_seedbal_train(root, out, args.plan, args.gpu, args.token)))
        elif args.command == "launch":
            print(canonical(launch_all(root, out)))
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
