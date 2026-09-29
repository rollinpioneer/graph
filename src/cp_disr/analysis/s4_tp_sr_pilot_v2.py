"""S4 T_P_SR V2 pilot (CP-DISR-S4-TP-SR-PILOT-V2-1): observation-chain diagnosis (Wave D) and minimal geometry prototype (Wave P).

Reuses the production branch executor (execute_registered_branch), reserve/claim/finish receipts and paired-restore QA unchanged.
No provider, representation, RL, optimizer, elastic, S2/S3 or formal-test code path exists in this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cp_disr.analysis.s1_integration import (IntegrationError, _append_jsonl, _atomic_json, _load_attempts, _transaction_lock, finish_branch_attempt,
                                             reserve_branch_attempt, sha256_file)

CARD_ID = "CP-DISR-S4-TP-SR-PILOT-V2-1"
BASE_COMMIT = "fbddf84f9cc3f0d39ce55c4f1f5e81bbddcc8ff6"
TASK_ID = "T_P_SR"
DIRECT = "a:PICK:target:v1"
RELOC = "a:PICK:interferer:v1"
WAVES = {"D": {"dir": "diagnostics", "key": "wave_d_branch_attempts", "cap": 4}, "P": {"dir": "physical", "key": "wave_p_branch_attempts", "cap": 16}}
VERIFIER_FACT_IDS = ("p:GripperEmpty", "p:Held:target", "p:Held:interferer", "p:OnTable:target", "p:OnTable:interferer", "p:Open:container",
                     "p:Inside:target:container", "p:Inside:interferer:container", "p:AtBuffer:target:buffer", "p:AtBuffer:interferer:buffer")
INTERFERENCE_PHASES = ("HOVER", "DESCEND", "PRESS", "CLOSE", "LIFT_SHOW", "LIFT_HOVER_FALLBACK")
ROOT_LABELS = ("PHYSICAL_GRASP_INTERFERENCE", "CONTROLLER_EXECUTION_FAILURE", "PERCEPTION_VERIFIER_MISMATCH", "PLANNER_FACT_INPUT_ERROR", "UNRESOLVED",
               "FALSE_SYMBOLIC_GOAL_FROM_PUBLIC_FACTS", "EVALUATOR_SYMBOLIC_DISAGREEMENT", "NOT_REPRODUCED_SUCCESS")
RESOLVED_LABELS = tuple(l for l in ROOT_LABELS if l not in ("UNRESOLVED", "NOT_REPRODUCED_SUCCESS"))


class StopRun(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rd(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def write_csv(path, rows, fields=None):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha_json(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------ ledger
def init_ledger(out, cfg):
    led = {k: {"cap": int(v), "used": 0} for k, v in cfg["budgets"].items()}
    _atomic_json(Path(out) / "budget_ledger.json", led)
    (Path(out) / "budget_events.jsonl").touch()
    return led


def charge_many(out, keys, ref=""):
    """Atomic: either every key has room and all are charged by one, or nothing is consumed."""
    out = Path(out)
    with _transaction_lock(out):
        led = rd(out / "budget_ledger.json")
        for k in keys:
            if int(led[k]["used"]) + 1 > int(led[k]["cap"]):
                raise StopRun("STOPPED_BUDGET_EXHAUSTED", f"{k} {led[k]['used']}+1>{led[k]['cap']}")
        for k in keys:
            led[k]["used"] = int(led[k]["used"]) + 1
            _append_jsonl(out / "budget_events.jsonl", {"event": "charge", "key": k, "used": led[k]["used"], "cap": led[k]["cap"], "ref": ref, "ts": now()})
        _atomic_json(out / "budget_ledger.json", led)
        return {k: led[k]["used"] for k in keys}


def event(out, kind, **f):
    _append_jsonl(Path(out) / "scheduling_events.jsonl", {"event": kind, "timestamp_utc": now(), "monotonic": time.monotonic(), **f})


# ------------------------------------------------------------------ protected inventory
def hash_tree(root, rels):
    root = Path(root)
    files = {}
    for rel in rels:
        base = root / rel
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(root))] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    return files


def protected_inventory(root, cfg, out, which):
    root, out = Path(root), Path(out)
    inv = {"baseline_commit": BASE_COMMIT, "created_utc": now(), "roots": cfg["protected_roots"], "files": hash_tree(root, cfg["protected_roots"])}
    inv["file_count"] = len(inv["files"])
    (out / "inventory").mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "inventory" / f"protected_{which}.json", inv)
    return inv


def protected_unchanged(out):
    a, b = rd(Path(out) / "inventory/protected_before.json")["files"], rd(Path(out) / "inventory/protected_after.json")["files"]
    return a == b, sorted(set(a) ^ set(b))[:10] + [k for k in a if k in b and a[k] != b[k]][:10]


# ------------------------------------------------------------------ freeze-spec
def out_dir_for(root, config_path):
    return Path(root) / "runs/final_master/S4/tp_soft_relocation_pilot_v2" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{sha256_file(config_path)[:8]}"


CODE_FILES = ("src/cp_disr/analysis/s4_tp_sr_pilot_v2.py", "scripts/s4_tp_sr_pilot_v2.py", "src/cp_disr/platforms/libero/tp_sr_instrumentation.py",
              "src/cp_disr/platforms/libero/tp_sr_v2_env.py", "src/cp_disr/platforms/libero/tp_sr_v2_runtime.py", "src/cp_disr/platforms/libero/runtime_factory.py",
              "configs/final_master/s4_tp_sr_pilot_v2.yaml", "configs/tasks/resolved/T_P_SR_V2.yaml", "configs/runtime/tp_sr_v2_contract_registry.yaml")


def freeze_spec(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    head = git(root, "rev-parse", "HEAD")
    if subprocess.call(["git", "-C", str(root), "merge-base", "--is-ancestor", BASE_COMMIT, head]) != 0:
        raise StopRun("STOPPED_WORKTREE_PREFLIGHT", "HEAD does not descend from the base commit")
    for sub in ("inventory", "spec", "diagnostics", "prototype", "physical", "captures", "decision", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    inv = protected_inventory(root, cfg, out, "before")
    init_ledger(out, cfg)
    _atomic_json(out / "authorization.json", {"card_id": CARD_ID, **cfg["authorization"], "caps": cfg["budgets"], "recorded_utc": now()})
    _atomic_json(out / "source_identity.json", {"head": head, "base_commit": BASE_COMMIT, "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
                                                "files_sha256": {f: sha256_file(root / f) for f in CODE_FILES if (root / f).is_file()},
                                                "git_status_at_freeze": git(root, "status", "--short"), "config_sha256": sha256_file(config_path)})
    _atomic_json(out / "stage_manifest.json", {"card_id": CARD_ID, "status": "SPEC_FROZEN", "created_utc": now(), "phases_done": ["freeze-spec"], "head": head})
    _atomic_json(out / "decision/decision_state.json", {"status": "IN_PROGRESS", "wave_d_status": "NOT_RUN", "wave_p_released": False, "mechanism_feasibility": "NOT_DETERMINED",
                                                          "provider_authorized": False, "s2_authorized": False})
    return {"status": "SPEC_FROZEN", "out": str(out), "protected_files": inv["file_count"]}


def mark_phase(out, phase, **fields):
    p = Path(out) / "stage_manifest.json"
    doc = rd(p)
    if phase not in doc.setdefault("phases_done", []):
        doc["phases_done"].append(phase)
    doc.update(fields)
    _atomic_json(p, doc)


# ------------------------------------------------------------------ runtime manifest / registration
def build_manifest(root, out, cfg, split_rel, name, contract_rel="configs/runtime/tp_sr_v2_contract_registry.yaml"):
    root, out = Path(root), Path(out)
    base = json.loads(json.dumps(yaml.safe_load((root / cfg["sources"]["s1_manifest_base"]).read_text())))
    rt = base["runtime"]
    rt["active_task_id"] = TASK_ID
    rt["repository_path"] = str(root)
    rt["experiment_root"] = str(root / "experiments")
    rt["stage_2a_contract_path"] = contract_rel
    rt["reference_skill_seconds_by_task"] = {TASK_ID: 3.649999999999709}
    rt["skill_timeouts"] = {TASK_ID: {"PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}}
    rt["task_deadlines"] = {TASK_ID: 60.0}
    rt["task_evaluator_version"] = {TASK_ID: "cp-disr-tpsr-task-evaluator-v1"}
    rt["task_assets"] = {TASK_ID: "src/cp_disr/platforms/libero/d0_env.py", "license": "assets/cp_disr/LICENSE", "owned_by": "cp_disr_project"}
    rt["task_splits"] = {TASK_ID: split_rel}
    rt["decision_cap_by_task"] = {TASK_ID: int(cfg["planner"]["decision_cap"])}
    src = root / "src/cp_disr/platforms/libero/tp_sr_v2_runtime.py"
    base["runtime_factory"] = {"module": cfg["runtime_module"], "factory": "create", "source_path": str(src), "sha256": sha256_file(src)}
    base["manifest_status"] = "S4_TP_SR_V2_PILOT_RUNTIME_BOUND_TRAINING_NOT_STARTED"
    base["note"] = "S4 T_P_SR V2 pilot runtime; no provider, no training, no formal test."
    base["vlm_runtime"] = {"api_account_authorized": "MUST_VERIFY_MODEL_ACCESS", "automatic_fallback_allowed": False,
                           "base_http_api_url": "https://dashscope.aliyuncs.com/api/v1", "region": "cn-beijing"}
    path = out / "spec" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
    return path


def perception_identity():
    return {"variant": "TP_SR_V1_NEAREST_PALETTE_COMPAT_LOCAL", "adapter": "TPSRNearestPalettePerceptionAdapter", "global_monkeypatch": False,
            "base_perception_version": "cp-disr-d0-rgbd-colorseg-v2"}


def init_phys_dir(phys, cap):
    for sub in ("witnesses/branch_receipts", "witnesses/restore_receipts", "witnesses/initial_state_checks", "branch_results", "worker_logs"):
        (phys / sub).mkdir(parents=True, exist_ok=True)
    _atomic_json(phys / "budget_ledger.json", {"physical_witness_episodes": {"cap": cap, "used": 0}})
    (phys / "budget_events.jsonl").touch()
    _atomic_json(phys / "attempt_registry.json", {})


def restore_seed(case, repeat):
    return int(hashlib.sha256(f"{case}|{repeat}".encode()).hexdigest()[:8], 16)


def make_branch(wave, case, cand, route, repeat, manifest_path, extra=None):
    ident = f"s4_tp_sr_v2|{wave}|{case}|{cand}|{repeat}"
    bid = hashlib.sha256(ident.encode()).hexdigest()[:16]
    return {"branch_id": bid, "attempt_id": bid, "case_id": case, "candidate_id": cand, "first_action": cand, "route": route, "repeat": repeat,
            "restore_seed": restore_seed(case, repeat), "selection_hash": hashlib.sha256(ident.encode()).hexdigest(), "manifest_path": str(manifest_path),
            "authorized": True, "execute_now": True, "status": "REGISTERED", "wave": wave, **(extra or {})}


def prepare_diagnostics(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    phys = out / "diagnostics"
    if (phys / "witnesses/e4_branch_registration.json").is_file():
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "diagnostic registration already exists")
    v1 = root / cfg["sources"]["v1_output"]
    split = rd(v1 / "pool/T_P_SR_pool_split.json")
    rows = {r["case_id"]: r for r in split["dev"]}
    sel = [rows[c] for c in cfg["wave_d"]["cases"]]
    split_path = out / "spec/T_P_SR_V2_diag_split.json"
    _atomic_json(split_path, {"task_id": TASK_ID, "task_family_id": cfg["task_family_id"], "train": [], "dev": sel, "test": []})
    manifest_path = build_manifest(root, out, cfg, os.path.relpath(split_path, root) if not str(split_path).startswith(str(root)) else str(split_path.relative_to(root)), "runtime_manifest_D.yaml")
    src_ident = {f: sha256_file(root / f) for f in CODE_FILES if (root / f).is_file()}
    branches = []
    for row in sel:
        for route, cand in cfg["wave_d"]["routes"].items():
            branches.append(make_branch("D", row["case_id"], cand, route, int(cfg["wave_d"]["repeat"]), manifest_path,
                                        {"source_v1_config_sha256": sha_json(row), "runtime_source_sha256": src_ident["src/cp_disr/platforms/libero/tp_sr_v2_runtime.py"],
                                         "instrumentation_source_sha256": src_ident["src/cp_disr/platforms/libero/tp_sr_instrumentation.py"],
                                         "perception_identity": perception_identity()}))
    seeds = {}
    for b in branches:
        seeds.setdefault(b["case_id"], set()).add(b["restore_seed"])
    if len(branches) != 4 or any(len(v) != 1 for v in seeds.values()):
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "wave D must register exactly four branches with paired seeds")
    init_phys_dir(phys, WAVES["D"]["cap"])
    reg = {"status": "REGISTERED", "card_id": CARD_ID, "physical_witness_episodes_cap": WAVES["D"]["cap"], "scenes": cfg["wave_d"]["cases"], "branches": branches}
    _atomic_json(phys / "witnesses/e4_branch_registration.json", reg)
    _atomic_json(phys / "diagnostic_branch_registration.json", reg)
    for p in (phys / "witnesses/e4_branch_registration.json", phys / "diagnostic_branch_registration.json"):
        p.chmod(0o444)
    mark_phase(out, "prepare-diagnostics", diagnostic_branches=len(branches))
    return {"status": "PREPARED", "branches": [(b["case_id"], b["route"], b["branch_id"]) for b in branches]}


# ------------------------------------------------------------------ worker
def worker_dry_run(root, out, wave, branch_id):
    """Zero-cost: hash gate + grounding + registration lookup. Reserves nothing and constructs no environment."""
    import importlib
    from cp_disr.runtime import require_runtime
    root, out = Path(root), Path(out)
    reg = rd(out / WAVES[wave]["dir"] / "witnesses/e4_branch_registration.json")
    branch = next((b for b in reg["branches"] if b["branch_id"] == branch_id), None)
    if branch is None:
        raise StopRun("STOPPED_RUNTIME_BINDING", "branch not registered")
    manifest = yaml.safe_load(Path(branch["manifest_path"]).read_text())
    spec = require_runtime(manifest)
    mod = importlib.import_module(spec["module"])
    if Path(mod.__file__).resolve() != Path(spec["source_path"]).resolve() or not callable(getattr(mod, spec["factory"], None)):
        raise StopRun("STOPPED_RUNTIME_BINDING", "module/source mismatch")
    from cp_disr.platforms.libero.tp_sr_runtime import ground_task_contracts
    rtm = manifest["runtime"]
    ground_task_contracts(Path(rtm["repository_path"]) / rtm["stage_2a_contract_path"], rtm["skill_timeouts"][TASK_ID])
    split = rd(Path(rtm["repository_path"]) / rtm["task_splits"][TASK_ID])
    if branch["case_id"] not in {r["case_id"] for r in split["dev"]}:
        raise StopRun("STOPPED_RUNTIME_BINDING", "case missing from split")
    return {"dry_run": "OK", "branch_id": branch_id}


def physical_worker(root, out, branch_id, wave):
    from cp_disr.analysis.s1_revision_resume import _runner_config, execute_registered_branch
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    from cp_disr.platforms.libero import perception as _perception
    from cp_disr.platforms.libero.tp_sr_instrumentation import RecordingPlanner
    from cp_disr.runtime import load_runtime
    root, out = Path(root), Path(out)
    phys = out / WAVES[wave]["dir"]
    reg = rd(phys / "witnesses/e4_branch_registration.json")
    branch = next(b for b in reg["branches"] if b["branch_id"] == branch_id)
    mask_before = _perception._mask
    holder = {}

    def factory(manifest, br):
        bundle = load_runtime(manifest)
        bundle.configure(out, phys, br)
        holder["bundle"] = bundle
        return bundle

    def planner_factory():
        manifest = yaml.safe_load(Path(branch["manifest_path"]).read_text())
        _rt, _tid, d_ref, _dl, _cap = _runner_config(manifest)
        inner = BPlanPlanner(SearchConfig(depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=d_ref))
        return RecordingPlanner(inner, holder["bundle"].recorder)

    t0 = time.time()
    res = jsonable(execute_registered_branch(root, branch_id, phys, bundle_factory=factory, planner_factory=planner_factory))
    bundle = holder.get("bundle")
    counts = bundle.env_counts() if bundle is not None else {}
    if bundle is not None and bundle.recorder is not None:
        bundle.recorder.close(counts)
    mask_after = _perception._mask
    res.update({"worker_pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "worker_wall_seconds": time.time() - t0,
                "wave": wave, "env_counts": counts, "recorder_errors": len(bundle.recorder.errors) if bundle is not None and bundle.recorder is not None else -1,
                "perception_identity": perception_identity(), "global_perception_mask_unchanged": bool(mask_before is mask_after and mask_after.__module__ == _perception.__name__),
                "provider_module_loaded": "cp_disr.analysis.s4_tp_sr_provider" in sys.modules})
    _atomic_json(phys / "branch_results" / f"{branch_id}.json", res)
    return res


def _spawn_env(root, gpu):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "CP_DISR_PHYSICAL_GPU_INDEX": str(gpu), "PYTHONPATH": f"{root}/src:{root}",
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(k, None)
    return env


def _spawn_worker(root, out, wave, branch_id, gpu):
    phys = Path(out) / WAVES[wave]["dir"]
    log = open(phys / "worker_logs" / f"{branch_id}.log", "ab")
    cmd = [sys.executable, str(Path(root) / "scripts/s4_tp_sr_pilot_v2.py"), "physical-worker", "--root", str(root), "--output", str(out), "--branch-id", branch_id, "--wave", wave]
    return subprocess.Popen(cmd, env=_spawn_env(root, gpu), stdout=log, stderr=subprocess.STDOUT, cwd=str(root), start_new_session=True)


def run_wave(root, config_path, out, wave, gpus, max_workers):
    """Dispatch registered branches with one worker per GPU. Each dispatch charges the wave, total and environment-reset budgets once."""
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    info = WAVES[wave]
    phys = out / info["dir"]
    reg = rd(phys / "witnesses/e4_branch_registration.json")
    attempts = _load_attempts(phys)
    todo = [b["branch_id"] for b in reg["branches"] if attempts.get(b["branch_id"]) is None]
    if not todo:
        return {"status": "NOTHING_TO_RUN"}
    if wave == "P" and rd(out / "diagnostics/wave_d_gate.json").get("wave_d_status") != "PASS":
        raise StopRun("STOPPED_WAVE_P_NOT_RELEASED", "Wave D technical gate has not passed")
    try:
        worker_dry_run(root, out, wave, todo[0])
    except Exception as exc:  # noqa: BLE001
        event(out, "dry_run_failed_before_any_reservation", error=str(exc))
        raise StopRun("STOPPED_RUNTIME_BINDING", str(exc)) from exc
    slots = list(gpus)[:max(1, min(int(max_workers), 4 if wave == "P" else int(cfg["wave_d"]["workers"]), len(gpus)))]
    running, faults, walls, consecutive = {}, [], {}, 0
    t_first = t_last = None
    event(out, "wave_start", wave=wave, workers=len(slots), gpus=slots, branches=len(todo))
    while running or (todo and not faults):
        for gpu in slots:
            if gpu in running or not todo or faults:
                continue
            bid = todo.pop(0)
            try:
                charge_many(out, [info["key"], "total_branch_attempts", "environment_resets"], ref=bid)
                reserve_branch_attempt(phys, bid)
            except (IntegrationError, StopRun) as exc:
                faults.append({"branch_id": bid, "fault": f"RESERVE:{exc}"})
                continue
            proc = _spawn_worker(root, out, wave, bid, gpu)
            t0 = time.monotonic()
            t_first = t0 if t_first is None else t_first
            running[gpu] = (bid, proc, t0)
            event(out, "branch_launch", wave=wave, branch_id=bid, gpu=gpu, pid=proc.pid, slots_in_use=len(running))
        for gpu, (bid, proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            state = _load_attempts(phys).get(bid)
            p = phys / "branch_results" / f"{bid}.json"
            res = rd(p) if p.is_file() else {}
            if rc != 0 or not res:
                faults.append({"branch_id": bid, "fault": f"WORKER_EXIT:{rc}", "attempt_state": state})
                if state in ("RESERVED", "STARTED"):
                    finish_branch_attempt(phys, bid, "UNKNOWN", error_phase="worker_process_exit", exit_code=rc)
            elif res.get("execution_status") == "EXCEPTION":
                consecutive += 1
                if consecutive >= 2:
                    faults.append({"branch_id": bid, "fault": "CONSECUTIVE_EXCEPTIONS:" + str(res.get("termination_reason"))})
            else:
                consecutive = 0
            walls[bid] = time.monotonic() - t0
            t_last = time.monotonic()
            event(out, "branch_finish", wave=wave, branch_id=bid, gpu=gpu, exit_code=rc, wall_seconds=walls[bid], slots_in_use=len(running) - 1)
            del running[gpu]
        time.sleep(1.0)
    span = (t_last - t_first) if t_first is not None and t_last is not None else None
    report = {"wave": wave, "workers": len(slots), "gpus": slots, "n": len(walls), "span_seconds": span, "aggregate_branches_per_second": (len(walls) / span) if span else None,
              "median_wall_seconds": float(np.median(list(walls.values()))) if walls else None, "faults": faults,
              "single_worker_baseline": "NOT_MEASURED", "speedup_claim": "NONE (no matched single-worker baseline; no extra attempts spent)"}
    _atomic_json(phys / "throughput_report.json", report)
    _append_jsonl(phys / "worker_events.jsonl", {"event": "wave_done", **report})
    mark_phase(out, f"run-{info['dir']}")
    return report
