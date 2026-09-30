"""Family B qualification protocol; never trains or runs formal test."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import yaml

from cp_disr.platforms.libero.family_b_runtime import ACTION_IDS, TASK_ID
from cp_disr.runtime import require_runtime

CARD = "CP-DISR-S4-FAMILY-B-STAGING-1"
BASE = "196c46c7b3ce7b3d6f88935d41707a5c1c2662df"
SOURCE_MANIFEST = Path("/home/xushijie2/graph_cp_disr_s4_soft_ordering_repair/runs/final_master/S4/family_a_soft_ordering_repair/20260930T035311Z_2f80581b/spec/runtime_manifest_T_P_SO_MVP.yaml")
CONTEXTS = ("B_PENDING", "C_PENDING", "BOTH_PENDING")
PADS = ("pad_u", "pad_v")
TECHNICAL = {("layout_0", "B_PENDING", 0), ("layout_1", "C_PENDING", 0)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def config(root, path):
    cfg = yaml.safe_load((Path(root) / path).read_text())
    if (cfg["card_id"], cfg["base_commit"], cfg["task_id"]) != (CARD, BASE, TASK_ID):
        raise ValueError("STOPPED_SOURCE_IDENTITY")
    if cfg["budgets"]["branch_attempts"] != 24 or cfg["budgets"]["explicit_resets"] != 24:
        raise ValueError("STOPPED_BUDGET_BINDING")
    if cfg["authorization"]["rl"] or cfg["authorization"]["optimizer"]:
        raise ValueError("STOPPED_UNAUTHORIZED_METHOD")
    return cfg


def inspect_bindings(root, cfg, out):
    """Resolve actual production paths and inherited independent skill cost, no env."""
    root, out = Path(root).resolve(), Path(out)
    if not SOURCE_MANIFEST.is_file():
        raise ValueError("STOPPED_BINDING_OR_SEMANTICS:reference source")
    prior = yaml.safe_load(SOURCE_MANIFEST.read_text())
    ref = float(prior["runtime"]["reference_skill_seconds_by_task"]["T_P_SO_MVP"])
    if not math.isfinite(ref) or ref <= 0:
        raise ValueError("STOPPED_BINDING_OR_SEMANTICS:reference cost")
    from cp_disr.platforms.libero.family_b_runtime import build_task_template, create_family_b_runtime
    template = build_task_template(root / cfg["runtime"]["contract_path"])
    if {c.id for c in template.contracts} != set(ACTION_IDS):
        raise ValueError("STOPPED_BINDING_OR_SEMANTICS:contracts")
    runtime = dict(prior["runtime"])
    runtime.update({
        "repository_path": str(root), "active_task_id": TASK_ID,
        "contract_path": cfg["runtime"]["contract_path"],
        "layouts_path": cfg["runtime"]["layouts_path"],
        "reference_skill_seconds_by_task": {TASK_ID: ref},
        "task_deadlines": {TASK_ID: float(cfg["runtime"]["task_deadlines"][TASK_ID])},
        "skill_timeouts": {TASK_ID: {"PICK": 9., "PLACE": 8., "PLACE_BUFFER": 8.}},
        "task_assets": {TASK_ID: "src/cp_disr/platforms/libero/family_b_env.py",
                        "license": "assets/cp_disr/LICENSE", "owned_by": "cp_disr_project"},
        "task_evaluator_version": {TASK_ID: "family-b-evaluator-v1"},
        "task_splits": {TASK_ID: cfg["runtime"]["layouts_path"]},
        "decision_cap_by_task": {TASK_ID: 6},
    })
    src = root / "src/cp_disr/platforms/libero/family_b_runtime.py"
    manifest = dict(prior)
    manifest["runtime"] = runtime
    manifest["runtime_factory"] = {
        "factory": "create_family_b_runtime",
        "module": "cp_disr.platforms.libero.family_b_runtime",
        "source_path": str(src), "sha256": sha(src),
    }
    require_runtime(manifest)
    bundle = create_family_b_runtime(manifest)
    if bundle.environment.__class__.__name__ != "_NullEnv":
        raise ValueError("STOPPED_BINDING_OR_SEMANTICS:factory constructed env")
    binding = {
        "status": "PASS", "reference_skill_seconds": ref,
        "reference_source": "pinned independent T_P_SO_MVP median, pilot-only inheritance",
        "reference_manifest_sha256": sha(SOURCE_MANIFEST),
        "factory_source_sha256": sha(src),
        "contract_ids": list(ACTION_IDS), "environment_constructions": 0,
    }
    write(out / "bindings/binding_manifest.json", binding)
    (out / "spec").mkdir(parents=True, exist_ok=True)
    (out / "spec/runtime_manifest_T_P_FB.yaml").write_text(yaml.safe_dump(manifest, sort_keys=True))
    (out / "config_resolved.yaml").write_text(yaml.safe_dump({**cfg, "resolved_reference_skill_seconds": ref}, sort_keys=True))
    return binding


def freeze(root, cfg, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    phys = out / "physical"
    if (phys / "witnesses/e4_branch_registration.json").exists():
        raise ValueError("STOPPED_SOURCE_IDENTITY:already frozen")
    binding = inspect_bindings(root, cfg, out)
    layouts_path = root / cfg["runtime"]["layouts_path"]
    layouts = read(layouts_path)
    if [x["layout_id"] for x in layouts["layouts"]] != ["layout_0", "layout_1"]:
        raise ValueError("STOPPED_GEOMETRY_PRECHECK:layout set")
    geometry_rows = geometry_precheck(layouts)
    write(out / "geometry/candidate_selection.json", {"tested_layouts": 2,
         "analytic_candidates_examined": 2, "rows": geometry_rows})
    write(out / "geometry/layouts.json", layouts)
    manifest_path = out / "spec/runtime_manifest_T_P_FB.yaml"
    freeze_id = digest({
        "card": CARD, "base": BASE,
        "config": sha(root / "configs/final_master/s4_family_b_staging.yaml"),
        "layout": sha(layouts_path),
        "contracts": sha(root / cfg["runtime"]["contract_path"]),
        "manifest": sha(manifest_path),
    })[:16]
    branches = []
    for layout in ("layout_0", "layout_1"):
        for context in CONTEXTS:
            for repeat in (0, 1):
                seed = int(hashlib.sha256(f"{freeze_id}:{layout}:{repeat}".encode()).hexdigest()[:8], 16)
                for pad in PADS:
                    key = {"freeze_id": freeze_id, "layout": layout, "context": context,
                           "repeat": repeat, "candidate": pad}
                    branch_id = digest(key)[:16]
                    branches.append({
                        **key, "branch_id": branch_id, "case_id": layout,
                        "restore_seed": seed,
                        "candidate_id": f"a:PLACE_BUFFER:carrier:{pad}:v1",
                        "prefix": cfg["contexts"][context]["prefix"],
                        "wave": "technical" if (layout, context, repeat) in TECHNICAL else "remaining",
                        "manifest_path": str(manifest_path), "authorized": True,
                        "execute_now": True, "plan_id": branch_id, "attempt_id": branch_id,
                    })
    if len(branches) != len({b["branch_id"] for b in branches}) or len(branches) != 24:
        raise ValueError("STOPPED_SOURCE_IDENTITY:branch identities")
    if sum(b["wave"] == "technical" for b in branches) != 4:
        raise ValueError("STOPPED_SOURCE_IDENTITY:technical identities")
    registration = {
        "card_id": CARD, "freeze_id": freeze_id, "branches": branches,
        "manifest_sha256": sha(manifest_path), "layouts_sha256": sha(layouts_path),
        "reference_skill_seconds": binding["reference_skill_seconds"],
    }
    write(phys / "witnesses/e4_branch_registration.json", registration)
    write(phys / "registration.json", registration)
    write(phys / "budget_ledger.json", {"physical_witness_episodes": {"cap": 24, "used": 0}})
    write(out / "budget_ledger.json", {
        "branch_attempts": {"cap": 24, "used": 0},
        "explicit_resets": {"cap": 24, "used": 0},
        "live_skill_calls": {"cap": 144, "used": 0},
        "relation_first_calls": {"cap": 4, "used": 0},
        "action_first_calls": {"cap": 6, "used": 0},
        "provider_retries": {"cap": 10, "used": 0},
        "rl_transitions": {"cap": 0, "used": 0},
        "optimizer_steps": {"cap": 0, "used": 0},
    })
    write(out / "source_identity.json", {
        "base_commit": BASE, "source_manifest_sha256": sha(SOURCE_MANIFEST),
        "config_sha256": sha(root / "configs/final_master/s4_family_b_staging.yaml"),
        "contract_sha256": sha(root / cfg["runtime"]["contract_path"]),
        "layout_sha256": sha(layouts_path), "freeze_id": freeze_id,
        "runtime_sources": freeze_source_hashes(root),
    })
    return {"status": "FROZEN", "freeze_id": freeze_id, "branches": 24, "technical": 4}


def branch(out, branch_id):
    registration = read(Path(out) / "physical/witnesses/e4_branch_registration.json")
    return next(b for b in registration["branches"] if b["branch_id"] == branch_id)

class BranchStop(RuntimeError):
    pass


def _values(snapshot):
    return {k: v.value for k, v in snapshot.facts.values.items()}


def _truth(snapshot, fact_id):
    from cp_disr.facts import Truth
    return snapshot.facts.values.get(fact_id, Truth.UNKNOWN) is Truth.TRUE


def _action(bundle, snapshot, aid, stage, rows, deadline):
    from cp_disr.adapters import EvaluationInput
    from cp_disr.platforms.libero.tp_sr_instrumentation import jsonable
    if aid not in snapshot.candidate_ids or not bool(snapshot.mask[snapshot.candidate_ids.index(aid)]):
        raise BranchStop("SCRIPT_DIVERGENCE_MASK:" + aid)
    contract = next(c for c in snapshot.template.contracts if c.id == aid)
    start = float(bundle.clock.now_seconds())
    execution = bundle.executor.execute(aid, float(contract.timeout_seconds))
    raw = bundle.executor.last
    obs = bundle.observations.observe()
    measurement = bundle.perception.infer(obs)
    records = bundle.verifier.verify(measurement, execution)
    end = float(bundle.clock.now_seconds())
    task = bundle.evaluator.evaluate(EvaluationInput(
        task_id=TASK_ID, env_id=snapshot.env_id, episode_id=snapshot.episode_id,
        evidence_refs=tuple(execution.evidence_ids),
        elapsed_seconds=end-bundle.episode_start_seconds,
        interval_start_seconds=start-bundle.episode_start_seconds,
        interval_end_seconds=end-bundle.episode_start_seconds))
    next_snapshot = bundle.snapshot_builder.build(snapshot, records, obs, execution, end)
    rows.append({
        "stage": stage, "action_id": aid, "start_sim": start, "end_sim": end,
        "duration_sim": end-start, "controller_exit": execution.controller_exit,
        "task_success": bool(task.success), "task_reason": task.reason,
        "facts": _values(next_snapshot),
        "candidate_ids": list(next_snapshot.candidate_ids),
        "candidate_mask": [bool(x) for x in next_snapshot.mask],
        "raw_skill": jsonable(raw),
    })
    if execution.controller_exit not in ("NORMAL_TERMINATION", "SUCCESS"):
        raise BranchStop("SKILL_FAILURE:" + execution.controller_exit)
    if end-bundle.episode_start_seconds >= deadline and not task.success:
        raise BranchStop("DEADLINE")
    return next_snapshot, task


def _boundary(bundle, snapshot, rows, out):
    import numpy as np
    from cp_disr.platforms.libero.perception import _metric_depth
    from cp_disr.platforms.libero.tp_sr_instrumentation import jsonable
    env = bundle.environment
    d = Path(out) / "captures" / bundle.recorder.branch_id / "boundary"
    d.mkdir(parents=True, exist_ok=True)
    obs = env.public_observation()
    bundle.recorder._png(d / "rgb.png", obs["rgb"])
    np.save(d / "depth_metric.npy", np.asarray(_metric_depth(env, obs["depth"]), dtype=np.float32))
    public = {
        "fact_values": _values(snapshot),
        "candidate_ids": list(snapshot.candidate_ids),
        "candidate_mask": [bool(x) for x in snapshot.mask],
        "proprio": jsonable(obs["proprio"]),
        "history": [r["action_id"] for r in rows],
        "sim_time": float(env.sim.data.time),
        "public_object_xyz": jsonable(env.last_perception),
        "public_layout": jsonable(env.public_layout()),
        "goals": [str(g) for g in snapshot.template.goals],
    }
    write(d / "public.json", public)
    write(d / "snapshot_encoding.json", {
        "env_id": snapshot.env_id, "episode_id": snapshot.episode_id,
        "decision_id": snapshot.decision_id,
        "base_input": list(snapshot.base_input),
        "candidate_features": [list(x) for x in snapshot.candidate_features],
        "candidate_ids": list(snapshot.candidate_ids),
        "mask": [bool(x) for x in snapshot.mask],
        "observation_ref": snapshot.observation_ref,
        "clock_seconds": snapshot.clock_seconds,
        "execution_summary": [list(x) for x in snapshot.execution_summary],
        "prior_edges": [list(x) for x in snapshot.prior_edges],
    })
    write(d / "qa_state.json", jsonable({
        "qa_only": True, "qpos": env.sim.data.qpos,
        "qvel": env.sim.data.qvel, "hidden_truth": env.hidden_truth()}))
    return public


def worker(root, out, branch_id):
    import time
    import traceback
    from cp_disr.analysis.s1_integration import claim_branch_attempt, finish_branch_attempt
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    from cp_disr.platforms.libero.tp_sr_instrumentation import RecordingPlanner
    from cp_disr.runtime import load_runtime
    root, out = Path(root), Path(out)
    phys = out / "physical"
    b = branch(out, branch_id)
    manifest_path = Path(b["manifest_path"])
    if sha(manifest_path) != read(phys / "registration.json")["manifest_sha256"]:
        raise RuntimeError("STOPPED_SOURCE_IDENTITY:manifest drift")
    receipt = claim_branch_attempt(phys, branch_id)
    bundle = None
    rows = []
    final = {
        "branch_id": branch_id, "layout": b["case_id"], "context": b["context"],
        "repeat": b["repeat"], "candidate": b["candidate"], "wave": b["wave"],
        "restore_seed": b["restore_seed"], "receipt_id": receipt["reservation_id"],
        "status": "EXCEPTION", "task_success": False, "actions": rows,
        "time_to_task_success": None,
    }
    wall_start = time.monotonic()
    try:
        manifest = yaml.safe_load(manifest_path.read_text())
        bundle = load_runtime(manifest)
        bundle.configure(out, b)
        snapshot = bundle.start_case(b["case_id"], restore_seed=int(b["restore_seed"]))
        if not bundle.restore_verified:
            raise BranchStop("RESTORE_RECEIPT_INVALID")
        deadline = float(manifest["runtime"]["task_deadlines"][TASK_ID])
        final["restore_receipt"] = bundle.restore_receipt
        final["initial_sim_time"] = float(bundle.episode_start_seconds)
        for aid in b["prefix"]:
            snapshot, task = _action(bundle, snapshot, aid, "setup", rows, deadline)
            if task.terminated or task.truncated:
                raise BranchStop("SCRIPT_DIVERGENCE_SETUP_TERMINAL:" + task.reason)
        if not _truth(snapshot, "p:Held:carrier"):
            raise BranchStop("SCRIPT_DIVERGENCE_CARRIER_NOT_HELD")
        required_done = {"B_PENDING": ("obj_c",), "C_PENDING": ("obj_b",),
                         "BOTH_PENDING": ()}[b["context"]]
        if any(not _truth(snapshot, f"p:Inside:{obj}:receiver") for obj in required_done):
            raise BranchStop("SCRIPT_DIVERGENCE_SETUP_GOAL")
        both = [f"a:PLACE_BUFFER:carrier:{pad}:v1" for pad in PADS]
        if any(cid not in snapshot.candidate_ids or
               not snapshot.mask[snapshot.candidate_ids.index(cid)] for cid in both):
            raise BranchStop("SCRIPT_DIVERGENCE_UV_MASK")
        boundary = _boundary(bundle, snapshot, rows, out)
        final["boundary_sha256"] = sha(out / "captures" / branch_id / "boundary/public.json")
        final["boundary_sim_time"] = boundary["sim_time"]
        candidate_start = float(bundle.clock.now_seconds())
        final["candidate_start_sim_time"] = candidate_start
        ref = float(read(out / "bindings/binding_manifest.json")["reference_skill_seconds"])
        planner = RecordingPlanner(BPlanPlanner(SearchConfig(
            depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0,
            reference_skill_seconds=ref)), bundle.recorder)
        aid = b["candidate_id"]
        for decision in range(6-len(b["prefix"])):
            stage = "candidate" if decision == 0 else "continuation"
            snapshot, task = _action(bundle, snapshot, aid, stage, rows, deadline)
            if task.terminated or task.truncated:
                end = float(bundle.clock.now_seconds())
                final.update({
                    "status": task.reason, "task_success": bool(task.success),
                    "terminal_sim_time": end,
                    "time_to_task_success": end-candidate_start if task.success else None,
                    "terminal_planner": "NOT_APPLICABLE_TERMINAL",
                })
                break
            remaining = deadline-(float(bundle.clock.now_seconds())-bundle.episode_start_seconds)
            plan = planner.plan(snapshot.facts, snapshot.template, remaining)
            rows[-1]["planner_status"] = plan.status
            rows[-1]["planner_plan"] = list(plan.plan)
            if plan.status != "PLAN_FOUND" or not plan.plan:
                final["status"] = plan.status
                final["planner_reason"] = plan.reason
                break
            aid = plan.plan[0]
        else:
            final["status"] = "DECISION_CAP"
    except BranchStop as exc:
        final["status"] = str(exc).split(":", 1)[0]
        final["detail"] = str(exc)
    except Exception as exc:
        final["status"] = "EXCEPTION"
        final["detail"] = f"{type(exc).__name__}:{exc}"
        final["traceback"] = traceback.format_exc()[-4000:]
    finally:
        final["worker_wall_seconds"] = time.monotonic()-wall_start
        if bundle is not None:
            final["env_counts"] = bundle.env_counts()
            final["recorder_errors"] = list(bundle.recorder.errors) if bundle.recorder else []
            if bundle.recorder:
                bundle.recorder.close(bundle.env_counts())
            bundle.environment.close()
        write(phys / "branch_results" / f"{branch_id}.json", final)
        finish_branch_attempt(phys, branch_id,
                              "COMPLETED" if final["status"] != "EXCEPTION" else "FAILED",
                              execution_status=final["status"])
    return final


def _result(out, branch_id):
    return read(Path(out) / "physical/branch_results" / f"{branch_id}.json")


def _registered(out, wave=None):
    rows = read(Path(out) / "physical/registration.json")["branches"]
    return [b for b in rows if wave is None or b["wave"] == wave]


def _root_budget_update(out, attempt_increment=0, reset_increment=0):
    path = Path(out) / "budget_ledger.json"
    ledger = read(path)
    for key, delta in (("branch_attempts", attempt_increment), ("explicit_resets", reset_increment)):
        if ledger[key]["used"] + delta > ledger[key]["cap"]:
            raise ValueError("STOPPED_BUDGET_EXHAUSTED:" + key)
        ledger[key]["used"] += delta
    write(path, ledger)


def _worker_env(gpu):
    import os
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
    })
    for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(key, None)
    return env


def run_wave(root, cfg, out, wave, gpus=(1, 3), max_workers=2):
    import subprocess
    import sys
    import time
    from cp_disr.analysis.s1_integration import reserve_branch_attempt
    root, out = Path(root).resolve(), Path(out).resolve()
    if wave not in ("technical", "remaining"):
        raise ValueError("unknown wave")
    if wave == "remaining" and read(out / "physical/technical_gate.json").get("status") != "PASS":
        raise ValueError("STOPPED_TECHNICAL_WAVE:gate not PASS")
    preflight_path = out / "logs/offline_preflight.json"
    if not preflight_path.is_file() or read(preflight_path).get("status") != "PASS":
        raise ValueError("STOPPED_BINDING_OR_SEMANTICS:offline preflight not PASS")
    reg = read(out / "physical/registration.json")
    manifest_path = out / "spec/runtime_manifest_T_P_FB.yaml"
    if sha(manifest_path) != reg["manifest_sha256"]:
        raise ValueError("STOPPED_SOURCE_IDENTITY:manifest drift")
    source = read(out / "source_identity.json")
    if source["runtime_sources"] != freeze_source_hashes(root):
        raise ValueError("STOPPED_SOURCE_IDENTITY:runtime source drift")
    require_runtime(yaml.safe_load(manifest_path.read_text()))
    branches = _registered(out, wave)
    if len(branches) != (4 if wave == "technical" else 20):
        raise ValueError("STOPPED_SOURCE_IDENTITY:wave cardinality")
    if any((out / "physical/branch_results" / f'{b["branch_id"]}.json').exists() for b in branches):
        raise ValueError("STOPPED_DUPLICATE_ATTEMPT")
    slots = list(gpus)[:min(max_workers, len(gpus))]
    if not slots or len(set(slots)) != len(slots):
        raise ValueError("STOPPED_GPU_BINDING")
    active, todo, faults = {}, list(branches), []
    started = time.monotonic()
    logs = out / "physical/worker_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while todo or active:
        for gpu in slots:
            if gpu in active or not todo or len(faults) >= 2:
                continue
            b = todo.pop(0)
            reserve_branch_attempt(out / "physical", b["branch_id"])
            _root_budget_update(out, attempt_increment=1, reset_increment=1)
            log = (logs / f'{b["branch_id"]}.log').open("ab")
            cmd = [
                sys.executable, str(root / "scripts/family_b_pilot.py"), "worker",
                "--root", str(root), "--output", str(out),
                "--branch-id", b["branch_id"],
            ]
            process = subprocess.Popen(cmd, cwd=root, env=_worker_env(gpu),
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            active[gpu] = (process, log, b["branch_id"])
        if not active:
            break
        time.sleep(0.5)
        for gpu, (process, log, bid) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[gpu]
            path = out / "physical/branch_results" / f"{bid}.json"
            if not path.is_file():
                faults.append({"branch_id": bid, "exit_code": code, "reason": "RESULT_MISSING"})
            else:
                row = read(path)
                if row["status"] == "EXCEPTION":
                    faults.append({"branch_id": bid, "exit_code": code, "reason": row.get("detail")})
                else:
                    faults.clear()
    completed = [read(out / "physical/branch_results" / f'{b["branch_id"]}.json')
                 for b in _registered(out)
                 if (out / "physical/branch_results" / f'{b["branch_id"]}.json').is_file()]
    ledger_path = out / "budget_ledger.json"
    ledger = read(ledger_path)
    observed_skills = sum(len(r.get("actions", [])) for r in completed)
    observed_resets = sum(r.get("env_counts", {}).get("reset_calls", 0) for r in completed)
    if observed_skills > ledger["live_skill_calls"]["cap"]:
        raise ValueError("STOPPED_BUDGET_EXHAUSTED:live skills")
    ledger["live_skill_calls"]["used"] = observed_skills
    ledger["observed_explicit_resets"] = observed_resets
    write(ledger_path, ledger)
    elapsed = time.monotonic()-started
    result = {
        "wave": wave, "dispatched": len(branches)-len(todo),
        "undispatched": len(todo), "worker_fault_streak": len(faults),
        "faults": faults, "wall_seconds": elapsed,
        "gpus": slots,
        "executed_skills": observed_skills,
        "observed_explicit_resets": observed_resets,
        "branches_per_wall_second": (len(branches)-len(todo))/elapsed,
        "executed_skills_per_wall_second": observed_skills/elapsed,
        "matched_single_worker_speedup": "NOT_MEASURED",
    }
    write(out / "physical" / f"{wave}_dispatch.json", result)
    return result


def check_technical(root, cfg, out):
    import numpy as np
    out = Path(out)
    branches = _registered(out, "technical")
    problems = []
    results = []
    for b in branches:
        bid = b["branch_id"]
        path = out / "physical/branch_results" / f"{bid}.json"
        if not path.is_file():
            problems.append(f"{bid}:result missing")
            continue
        row = read(path)
        results.append(row)
        actions = row["actions"]
        if row["status"] != "TASK_SUCCESS" or not row["task_success"]:
            problems.append(f"{bid}:not task success:{row['status']}")
        if len(actions) != 6 or [a["action_id"] for a in actions[:3]] != b["prefix"]:
            problems.append(f"{bid}:skill sequence")
        if len(actions) > 3 and actions[3]["action_id"] != b["candidate_id"]:
            problems.append(f"{bid}:candidate mismatch")
        remaining = ("obj_b" if b["context"] == "B_PENDING" else "obj_c")
        expected_suffix = [f"a:PICK:{remaining}:v1",
                           f"a:PLACE:{remaining}:receiver:v1"]
        if len(actions) == 6 and [a["action_id"] for a in actions[4:]] != expected_suffix:
            problems.append(f"{bid}:continuation mismatch")
        if row.get("recorder_errors"):
            problems.append(f"{bid}:recorder errors")
        if row.get("env_counts", {}).get("reset_calls") != 1:
            problems.append(f"{bid}:explicit reset count")
        cap = out / "captures" / bid
        if not (cap / "boundary/public.json").is_file():
            problems.append(f"{bid}:boundary missing")
            continue
        for i in range(len(actions)):
            d = cap / f"action_{i:02d}"
            for f in ("before_rgb.png", "after_rgb.png", "before_depth.npy",
                      "after_depth.npy", "facts.json", "evaluator.json", "perception.json"):
                if not (d / f).is_file():
                    problems.append(f"{bid}:action_{i:02d}/{f} missing")
            if i >= 3 and i < len(actions)-1 and not (d / "planner.json").is_file():
                problems.append(f"{bid}:nonterminal planner missing")
            if i == len(actions)-1 and (d / "planner.json").exists():
                problems.append(f"{bid}:terminal planner unexpectedly present")
    for layout, context, repeat in TECHNICAL:
        pair = [b for b in branches if (b["layout"], b["context"], b["repeat"]) == (layout, context, repeat)]
        if len(pair) != 2:
            problems.append(f"{layout}/{context}:pair missing")
            continue
        paths = [out / "captures" / b["branch_id"] / "boundary" for b in pair]
        if not all((p / "public.json").is_file() and (p / "qa_state.json").is_file() for p in paths):
            continue
        p0, p1 = (read(p / "public.json") for p in paths)
        if p0["fact_values"] != p1["fact_values"] or p0["candidate_ids"] != p1["candidate_ids"] or p0["candidate_mask"] != p1["candidate_mask"]:
            problems.append(f"{layout}/{context}:public boundary mismatch")
        q0, q1 = (read(p / "qa_state.json") for p in paths)
        for name in ("qpos", "qvel"):
            if not np.allclose(q0[name], q1[name], atol=1e-8, rtol=0):
                problems.append(f"{layout}/{context}:{name} mismatch")
    ledger = read(out / "physical/budget_ledger.json")["physical_witness_episodes"]
    if ledger["used"] != 4:
        problems.append("technical budget != 4")
    gate = {"status": "PASS" if not problems else "FAIL", "problems": problems,
            "attempts": ledger["used"], "terminal_results": len(results),
            "remaining_released": not problems}
    write(out / "physical/technical_gate.json", gate)
    return gate


def geometry_precheck(layouts):
    """Pure analytic clearance and path proxy; never policy/truth input."""
    import csv
    import io
    import numpy as np
    show = np.array([0.02, -0.08])
    rows = []
    for layout in layouts["layouts"]:
        lid = layout["layout_id"]
        movable = {k: np.array(layout[k + "_xy"], dtype=float)
                   for k in ("carrier", "obj_b", "obj_c")}
        pads = {"pad_u": np.array(layout["pad_u_xy"], dtype=float),
                "pad_v": np.array(layout["pad_v_xy"], dtype=float)}
        receiver = np.array(layout["receiver_xy"], dtype=float)
        for a, pa in movable.items():
            if not np.all(np.isfinite(pa)) or np.max(np.abs(pa)) > 0.35:
                raise ValueError("STOPPED_GEOMETRY_PRECHECK:workspace:" + a)
            for b, pb in movable.items():
                if a < b and np.linalg.norm(pa-pb) <= 0.07:
                    raise ValueError("STOPPED_GEOMETRY_PRECHECK:object overlap")
            for pad, pp in pads.items():
                if np.linalg.norm(pa-pp) <= 0.08:
                    raise ValueError("STOPPED_GEOMETRY_PRECHECK:object/pad overlap")
            if np.linalg.norm(pa-receiver) <= 0.10:
                raise ValueError("STOPPED_GEOMETRY_PRECHECK:object/receiver overlap")
        if np.linalg.norm(pads["pad_u"]-pads["pad_v"]) <= 0.15:
            raise ValueError("STOPPED_GEOMETRY_PRECHECK:pad alias")
        if any(np.linalg.norm(p-receiver) <= 0.10 for p in pads.values()):
            raise ValueError("STOPPED_GEOMETRY_PRECHECK:pad/receiver overlap")
        for target in ("obj_b", "obj_c"):
            for pad, point in pads.items():
                proxy = float(np.linalg.norm(show-point)+np.linalg.norm(point-movable[target]))
                rows.append({"layout": lid, "next_object": target, "pad": pad,
                             "complete_path_proxy_m": proxy,
                             "proxy_role": "DESIGN_PROXY_ONLY"})
    grouped = {(r["layout"], r["next_object"], r["pad"]): r["complete_path_proxy_m"] for r in rows}
    for lid in ("layout_0", "layout_1"):
        b = grouped[(lid, "obj_b", "pad_u")]-grouped[(lid, "obj_b", "pad_v")]
        c = grouped[(lid, "obj_c", "pad_u")]-grouped[(lid, "obj_c", "pad_v")]
        if b*c >= 0 or min(abs(b), abs(c)) <= 0.05:
            raise ValueError("STOPPED_GEOMETRY_PRECHECK:no predicted reversal")
    if grouped[("layout_0", "obj_b", "pad_u")] == grouped[("layout_1", "obj_b", "pad_u")]:
        raise ValueError("STOPPED_GEOMETRY_PRECHECK:layouts identical")
    return rows


def freeze_source_hashes(root):
    root = Path(root)
    paths = (
        "src/cp_disr/platforms/libero/family_b_env.py",
        "src/cp_disr/platforms/libero/family_b_adapters.py",
        "src/cp_disr/platforms/libero/family_b_runtime.py",
        "src/cp_disr/platforms/libero/skill_executor.py",
        "src/cp_disr/platforms/libero/runtime_factory.py",
        "src/cp_disr/vlm_provider.py",
        "src/cp_disr/neural.py",
        "src/cp_disr/analysis/family_b_pilot.py",
        "src/cp_disr/analysis/family_b_provider.py",
        "src/cp_disr/analysis/family_b_representation.py",
        "src/cp_disr/analysis/family_b_reference_bank.py",
        "src/cp_disr/analysis/family_b_closeout.py",
        "scripts/family_b_pilot.py",
        "experiments/sources/v2.1_interfaces/system_prompt.txt",
    )
    return {p: sha(root / p) for p in paths}


def provider_preflight(root, cfg, out):
    """Fake transport through production payload/parser/cache/verify, zero API calls."""
    import tempfile
    from cp_disr.platforms.libero.family_b_runtime import build_task_template
    from cp_disr.vlm_provider import (
        MODEL, REGION, ENDPOINT, SDK_VERSION, ProviderConfig, validate_payload)
    from cp_disr.vlm_cache_pipeline import (
        request_and_process, write_audit_cache, verify_audit_cache)
    from cp_disr.vlm import CACHE_FIELDS, cache_key
    root, out = Path(root), Path(out)
    provider_config = ProviderConfig()
    template = build_task_template(root / cfg["runtime"]["contract_path"])
    schema_path = root / "schemas/relation_schema.json"
    schema = read(schema_path)
    image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLqWQAAAABJRU5ErkJggg=="
    prompt = "Family B synthetic preflight; public goal and full contract IDs only."
    messages = [{"role": "system", "content": [{"text": prompt}]}]
    for n in range(3):
        messages += [
            {"role": "user", "content": [{"text": f"fewshot {n}"}, {"image": image}]},
            {"role": "assistant", "content": [{"text": '{"schema_version":"m1_soft_relations_v2","relations":[]}'}]},
        ]
    messages.append({"role": "user", "content": [
        {"text": json.dumps({"task_id": TASK_ID, "actions": list(ACTION_IDS),
                              "goals": [g.fact_id for g in template.goals],
                              "objects": ["carrier", "obj_b", "obj_c", "receiver", "pad_u", "pad_v"]})},
        {"image": image}]})
    payload = {
        "model": MODEL, "messages": messages, "temperature": 0,
        "max_tokens": 2048, "response_format": {"type": "json_object"},
        "enable_thinking": False, "enable_search": False,
        "stream": False, "result_format": "message",
    }
    validate_payload(payload)
    relation = {
        "relation_id": "synthetic_preflight_only",
        "type": "SOFT_SUPPORTS",
        "source_ref": "a:PLACE_BUFFER:carrier:pad_u:v1",
        "target_ref": "a:PICK:obj_b:v1",
        "effect_fact_ref": "p:AtBuffer:carrier:pad_u",
    }

    class Fake:
        def send(self, request):
            validate_payload(request)
            return {
                "raw": {"output": {"choices": [{
                    "finish_reason": "stop",
                    "message": {"content": [{"text": json.dumps({
                        "schema_version": "m1_soft_relations_v2",
                        "relations": [relation]})}]},
                }]}},
                "error_type": "OK", "status_code": 200,
                "request_id": "synthetic-preflight", "latency_seconds": 0,
            }

    execution = request_and_process(Fake(), payload, template, schema)
    if execution["status"] != "SUCCESS" or len(execution["processing"]["accepted"]) != 1:
        raise ValueError("STOPPED_BINDING_OR_SEMANTICS:provider admission")
    manifest = {
        "split": "dev", "task_definition_hash": sha(root / "configs/tasks/resolved/T_P_FB.yaml"),
        "initial_RGB_content_hash": digest(image), "preprocessing_hash": digest("synthetic"),
        "object_binding_hash": digest(["carrier", "obj_b", "obj_c", "receiver", "pad_u", "pad_v"]),
        "allowed_ID_hash": digest(list(ACTION_IDS)), "contract_version": sha(root / cfg["runtime"]["contract_path"]),
        "predicate_version": "family-b-v1", "model_snapshot": MODEL,
        "sdk_api_version": SDK_VERSION, "region": REGION, "endpoint": ENDPOINT,
        "prompt_hash": digest(prompt), "fewshot_hash": digest(messages[1:7]),
        "schema_hash": sha(schema_path),
        "decoding_config": {"temperature": 0, "max_tokens": 2048, "response_format": "json_object"},
        "initial_facts_hash": digest("synthetic"), "asset_binding_hash": sha(root / cfg["runtime"]["layouts_path"]),
        "request_payload_hash": digest(payload), "scene_id": "family-b-synthetic-preflight",
        "task_id": TASK_ID, "synthetic_unit_fixture": True,
    }
    if set(CACHE_FIELDS)-set(manifest):
        raise ValueError("STOPPED_BINDING_OR_SEMANTICS:cache identity")
    with tempfile.TemporaryDirectory() as tmp:
        cache = write_audit_cache(Path(tmp), manifest, prompt, {"synthetic_unit_fixture": True}, execution)
        verified, edges = verify_audit_cache(cache)
        if verified["cache_key"] != cache_key(manifest) or len(edges) != 1:
            raise ValueError("STOPPED_BINDING_OR_SEMANTICS:fake audit cache")
    result = {
        "status": "PASS", "task_id": TASK_ID, "provider_model": provider_config.model,
        "provider_region": provider_config.region, "provider_endpoint": provider_config.endpoint,
        "provider_sdk": provider_config.sdk_version, "fake_cache_verified": True,
        "accepted_synthetic_relation": 1, "live_requests": 0,
    }
    write(out / "provider/provider_preflight.json", result)
    return result


def analyze_physical(root, cfg, out):
    """Protocol completeness and scientific mechanism are separate outcomes."""
    import csv
    from collections import defaultdict
    root, out = Path(root), Path(out)
    gate_path = out / "physical/technical_gate.json"
    stopped = gate_path.is_file() and read(gate_path).get("status") == "FAIL"
    registered = _registered(out, "technical" if stopped else None)
    # An engineering stop may leave technical branches undispatched. Audit
    # every reserved attempt, not the unspent entries in the frozen registry.
    attempts = read(out / "physical/attempt_registry.json") if stopped and (
        out / "physical/attempt_registry.json").is_file() else {}
    branches = [b for b in registered if not stopped or b["branch_id"] in attempts]
    rows = []
    missing = []
    for b in branches:
        path = out / "physical/branch_results" / f'{b["branch_id"]}.json'
        if not path.is_file():
            missing.append(b["branch_id"])
            continue
        r = read(path)
        cost = r.get("time_to_task_success")
        if r.get("task_success"):
            if cost is None or not isinstance(cost, (int, float)) or cost <= 0:
                missing.append(b["branch_id"] + ":invalid success time")
        elif cost is not None:
            missing.append(b["branch_id"] + ":failure time must be null")
        cap = out / "captures" / b["branch_id"]
        for i in range(len(r.get("actions", []))):
            d = cap / f"action_{i:02d}"
            for name in ("before_rgb.png", "after_rgb.png", "before_depth.npy",
                         "after_depth.npy", "facts.json", "evaluator.json", "perception.json"):
                if not (d / name).is_file():
                    missing.append(f'{b["branch_id"]}:action_{i:02d}/{name}')
            if i >= len(b["prefix"]) and i < len(r["actions"])-1 and not (d / "planner.json").is_file():
                missing.append(f'{b["branch_id"]}:planner_{i:02d}')
        rows.append({
            "branch_id": b["branch_id"], "layout": b["layout"],
            "context": b["context"], "repeat": b["repeat"],
            "candidate": b["candidate"], "task_success": bool(r.get("task_success")),
            "terminal_status": r.get("status"), "cost_sim_seconds": cost,
            "setup_sim_seconds": (
                sum(a["duration_sim"] for a in r.get("actions", []) if a["stage"] == "setup")
                if r.get("actions") else None),
            "episode_sim_seconds": (
                r.get("terminal_sim_time")-r.get("initial_sim_time")
                if r.get("terminal_sim_time") is not None and
                   r.get("initial_sim_time") is not None else None),
        })
    csv_path = out / "physical/paired_consequences.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "branch_id", "layout", "context", "repeat", "candidate",
            "task_success", "terminal_status", "cost_sim_seconds",
            "setup_sim_seconds", "episode_sim_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["layout"], r["context"], r["repeat"])][r["candidate"]] = r
    pairs = []
    for layout in ("layout_0", "layout_1"):
        for context in CONTEXTS:
            for repeat in (0, 1):
                pair = grouped[(layout, context, repeat)]
                u, v = pair.get("pad_u"), pair.get("pad_v")
                complete = bool(u and v and u["task_success"] and v["task_success"])
                delta = u["cost_sim_seconds"]-v["cost_sim_seconds"] if complete else None
                pairs.append({
                    "layout": layout, "context": context, "repeat": repeat,
                    "u_branch": u["branch_id"] if u else None,
                    "v_branch": v["branch_id"] if v else None,
                    "u_success": u["task_success"] if u else None,
                    "v_success": v["task_success"] if v else None,
                    "delta_u_minus_v_sim_seconds": delta,
                    "complete_success_pair": complete,
                })
    write(out / "physical/paired_costs.json", pairs)
    restore_rows, restore_problems = paired_restore_audit(out, branches)
    missing.extend(restore_problems)
    problems = list(missing)
    ledger = read(out / "physical/budget_ledger.json")["physical_witness_episodes"]
    if ledger["used"] != len(branches) or (not stopped and len(branches) != 24) or len(rows) != len(branches):
        problems.append("attempts/results do not match stopped/full scope")
    if not stopped and any(not p["complete_success_pair"] for p in pairs):
        problems.append("not all 12 paired contexts have two successes")
    tolerance = float(cfg["metrics"]["difference_tolerance_seconds"])
    reversal = True
    direction = {}
    for layout in ("layout_0", "layout_1"):
        for context in ("B_PENDING", "C_PENDING"):
            observed = [p["delta_u_minus_v_sim_seconds"] for p in pairs
                        if p["layout"] == layout and p["context"] == context]
            if len(observed) != 2 or any(v is None or abs(v) <= tolerance for v in observed):
                reversal = False
                continue
            if observed[0]*observed[1] <= 0:
                reversal = False
                continue
            direction[(layout, context)] = 1 if observed[0] > 0 else -1
        if (layout, "B_PENDING") in direction and (layout, "C_PENDING") in direction:
            if direction[(layout, "B_PENDING")] == direction[(layout, "C_PENDING")]:
                reversal = False
    if len(direction) != 4:
        reversal = False
    if len(set(direction.values())) != 2:
        reversal = False
    status = "UNRESOLVED" if stopped else "ESTABLISHED" if not problems and reversal else (
        "UNRESOLVED" if missing else "NOT_ESTABLISHED")
    result = {
        "artifact_integrity_status": "PASS" if not missing else "FAIL",
        "execution_protocol_status": "STOPPED" if stopped and len(rows) == 4 else
                                     "COMPLETE" if len(rows) == 24 else "PARTIAL",
        "physical_mechanism_status": status,
        "complete_success_pairs": sum(p["complete_success_pair"] for p in pairs),
        "total_pairs": 12, "total_branches": 24,
        "cost_reversal_pass": reversal, "problems": problems,
        "provider_released": status == "ESTABLISHED",
        "time_unit": "MuJoCo simulation seconds",
        "scope": "two selected layouts, two repeats per context; no significance claim",
    }
    write(out / "physical/mechanism_gate.json", result)
    return result


def paired_restore_audit(out, branches):
    import numpy as np
    out = Path(out)
    groups = {}
    for b in branches:
        groups.setdefault((b["layout"], b["context"], b["repeat"]), {})[b["candidate"]] = b
    records, problems = [], []
    initial_by_source = {}
    for key, candidates in sorted(groups.items()):
        layout, context, repeat = key
        if set(candidates) != set(PADS):
            problems.append(f"{key}:candidate pair incomplete")
            continue
        paths = {pad: out / "captures" / b["branch_id"] for pad, b in candidates.items()}
        needed = ("boundary/public.json", "boundary/qa_state.json", "initial_rgb.png")
        if any(not (base / name).is_file() for base in paths.values() for name in needed):
            problems.append(f"{key}:restore capture missing")
            continue
        pub = {pad: read(base / "boundary/public.json") for pad, base in paths.items()}
        qa = {pad: read(base / "boundary/qa_state.json") for pad, base in paths.items()}
        public_equal = all(pub["pad_u"].get(k) == pub["pad_v"].get(k)
                           for k in ("fact_values", "candidate_ids", "candidate_mask"))
        qpos_equal = np.allclose(qa["pad_u"]["qpos"], qa["pad_v"]["qpos"], atol=1e-8, rtol=0)
        qvel_equal = np.allclose(qa["pad_u"]["qvel"], qa["pad_v"]["qvel"], atol=1e-8, rtol=0)
        rgb_hashes = {pad: sha(base / "initial_rgb.png") for pad, base in paths.items()}
        initial_equal = rgb_hashes["pad_u"] == rgb_hashes["pad_v"]
        initial_by_source.setdefault((layout, repeat), set()).update(rgb_hashes.values())
        good = public_equal and qpos_equal and qvel_equal and initial_equal
        if not good:
            problems.append(f"{key}:paired restore mismatch")
        records.append({
            "layout": layout, "context": context, "repeat": repeat,
            "public_equal": public_equal, "qpos_equal": bool(qpos_equal),
            "qvel_equal": bool(qvel_equal), "initial_rgb_equal": initial_equal,
            "qa_only": True, "pass": bool(good),
        })
    for key, hashes in initial_by_source.items():
        if len(hashes) != 1:
            problems.append(f"{key}:initial source differs across contexts")
    write(out / "physical/paired_restore.json", {"records": records, "problems": problems})
    return records, problems


def offline_preflight(root, cfg, out):
    """Compile and two independent offline test groups; zero simulator resets."""
    import subprocess
    import sys
    root, out = Path(root).resolve(), Path(out)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    inspect = inspect_bindings(root, cfg, out)
    fake = provider_preflight(root, cfg, out)
    geometry = geometry_precheck(read(root / cfg["runtime"]["layouts_path"]))
    compile_cmd = [sys.executable, "-m", "compileall", "-q",
                   "src/cp_disr", "scripts", "tests"]
    with (logs / "compileall.log").open("w") as handle:
        compiled = subprocess.run(compile_cmd, cwd=root, stdout=handle,
                                  stderr=subprocess.STDOUT, check=False)
    groups = (
        ("semantics_runtime", [
            "tests/test_family_b_semantics.py", "tests/test_family_b_runtime.py"]),
        ("runner_analysis", [
            "tests/test_family_b_runner.py", "tests/test_family_b_analysis.py"]),
    )
    running = []
    for name, files in groups:
        log = (logs / f"tests_{name}.log").open("w")
        process = subprocess.Popen([sys.executable, "-m", "pytest", "-q", *files],
                                   cwd=root, stdout=log, stderr=subprocess.STDOUT)
        running.append((name, process, log))
    codes = {}
    for name, process, log in running:
        codes[name] = process.wait()
        log.close()
    result = {
        "status": "PASS" if compiled.returncode == 0 and all(x == 0 for x in codes.values())
                  and inspect["status"] == fake["status"] == "PASS" and len(geometry) == 8 else "FAIL",
        "compile_exit": compiled.returncode,
        "targeted_test_exits": codes, "geometry_proxy_rows": len(geometry),
        "environment_constructions": 0, "physical_attempts": 0, "provider_calls": 0,
    }
    write(out / "logs/offline_preflight.json", result)
    return result
