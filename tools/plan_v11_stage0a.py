#!/usr/bin/env python3
"""Plan v1.1 Stage 0A: bind D0/T_B/T_C and measure H, d_ref. No PPO."""
from __future__ import annotations
import json, os, time, math, hashlib, statistics, traceback
from pathlib import Path
import numpy as np

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
import sys
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cp_disr.common import BindingError, canonical, digest
from cp_disr.stage2a_p0 import (
    TASK_ROLE, SCRIPTED, spec_from_row, bind_env, build_task_split, write_json, _log
)
from cp_disr.adapters import EvaluationInput
from cp_disr.rl import set_suite_half_life

OUT = ROOT / "runs" / "stage_0a"
REP = ROOT / "reports"
ST = ROOT / "status"
MIG = ROOT / "migration"
for p in (OUT, REP, ST, MIG):
    p.mkdir(parents=True, exist_ok=True)

FAMILIES = ["D0", "T_B", "T_C"]
MAX_ATTEMPTS = 10
SUCCESS_TARGET = 5

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def patch_split_bases():
    p = ROOT / "src/cp_disr/stage2a_p0.py"
    t = p.read_text(encoding="utf-8")
    old = "    train_base = 2100 if task_id == \"T_A\" else 3100\n    dev_base = 6100 if task_id == \"T_A\" else 7100"
    neu = "    train_base = {\"T_A\": 2100, \"T_C\": 3100, \"T_B\": 4100}[task_id]\n    dev_base = {\"T_A\": 6100, \"T_C\": 7100, \"T_B\": 8100}[task_id]"
    if old in t:
        p.write_text(t.replace(old, neu), encoding="utf-8")
        # reload module
        import importlib, cp_disr.stage2a_p0 as m
        importlib.reload(m)

def load_split(task_id):
    mapping = {
        "D0": ROOT / "configs/splits/D0_stage_1a.json",
        "T_C": ROOT / "configs/splits/T_C_stage_2a.json",
        "T_B": ROOT / "configs/splits/T_B_stage_0a.json",
    }
    return json.loads(mapping[task_id].read_text(encoding="utf-8"))

def ensure_tb_split():
    path = ROOT / "configs/splits/T_B_stage_0a.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from cp_disr.stage2a_p0 import build_task_split
    doc = build_task_split("T_B")
    doc["version"] = "plan-v1.1-stage-0a"
    doc["reuse_reason"] = "New T_B family; not a T_A relabel. Goals: Inside(target,container) AND AtBuffer(second_object,buffer)."
    path.write_text(canonical(doc) + "\n", encoding="utf-8")
    return doc

def load_timing():
    import json
    p = ROOT / "experiments/part_2_exploration/stage_2a_p0/timing_calibration.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def timeouts_for(task_id, runtime):
    timing = load_timing()
    if task_id in timing and isinstance(timing[task_id], dict) and "skill_timeouts" in timing[task_id]:
        return dict(timing[task_id]["skill_timeouts"])
    if task_id == "T_B" and "T_A" in timing:
        return dict(timing["T_A"]["skill_timeouts"])
    if task_id == "D0":
        return {"OPEN": 16.0, "PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}
    t = runtime.get("skill_timeouts") if isinstance(runtime.get("skill_timeouts"), dict) else {}
    if task_id in t and isinstance(t[task_id], dict):
        return t[task_id]
    return {"OPEN": 16.0, "PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}

def deadline_for(task_id, runtime, tmap):
    timing = load_timing()
    if task_id in timing and isinstance(timing[task_id].get("task_deadline_seconds"), (int, float)):
        return float(timing[task_id]["task_deadline_seconds"])
    if task_id == "T_B" and "T_A" in timing:
        # T_B script uses PLACE+PLACE_BUFFER like T_C length
        return float(tmap["OPEN"] + 2 * tmap["PICK"] + tmap["PLACE"] + tmap["PLACE_BUFFER"] + 10.0)
    if task_id == "D0":
        return 43.0
    d = runtime.get("task_deadlines")
    if isinstance(d, dict) and isinstance(d.get(task_id), (int, float)):
        return float(d[task_id])
    return 90.0

def scripted_seq(task_id):
    if task_id == "D0":
        return ["a:OPEN:container:v1", "a:PICK:target:v1", "a:PLACE:target:container:v1"]
    return SCRIPTED[task_id]

def run_one(task_id, row, tmap, deadline, gpu):
    spec = spec_from_row(task_id if task_id != "D0" else "T_A", row, deadline=deadline)
    # D0 uses second_object and D0 evaluator; bind_env uses CaseSpec.task_id
    if task_id == "D0":
        spec.task_id = "D0"
        spec.second_role = "second_object"
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(spec, gpu=gpu, deadline=deadline)
    evaluator.deadline = float(deadline)
    evaluator.reset_episode()
    skill_durs = []
    try:
        start_ep = clock.now_seconds()
        last_reason = "NONE"
        success = False
        for sid in scripted_seq(task_id):
            skill = sid.split(":")[1]
            exe = executor.execute(sid, float(tmap[skill]))
            skill_durs.append(float(exe.get("sim_duration") or (exe["end_seconds"] - exe["start_seconds"])))
            elapsed = clock.now_seconds() - start_ep
            ev = evaluator.evaluate(EvaluationInput(task_id, "ref", row["case_id"], exe.get("evidence_ids") or (), elapsed, exe["start_seconds"], exe["end_seconds"]))
            last_reason = ev.reason
            if ev.success:
                success = True
                break
        end_ep = clock.now_seconds()
        if not success:
            ev = evaluator.evaluate(EvaluationInput(task_id, "ref", row["case_id"], (), end_ep - start_ep, start_ep, end_ep))
            last_reason = ev.reason
            success = bool(ev.success or evaluator.goal_true())
        return {
            "task_id": task_id,
            "case_id": row["case_id"],
            "success": bool(success),
            "reason": last_reason,
            "episode_seconds": float(end_ep - start_ep),
            "skill_durations": skill_durs,
            "n_skills": len(skill_durs),
            "goal_true": bool(evaluator.goal_true()),
        }
    finally:
        env.close()

def main():
    gpu = int(os.environ.get("CP_DISR_GPU", "0"))
    patch_split_bases()
    tb = ensure_tb_split()
    # runtime from existing 2A-P0 resolved defaults if present
    rt_path = ROOT / "experiments/part_2_exploration/stage_2a_p0"
    # find runtime yaml
    candidates = list(ROOT.glob("experiments/**/runtime_manifest.yaml"))
    runtime = {}
    for c in candidates:
        try:
            import yaml
            doc = yaml.safe_load(c.read_text(encoding="utf-8")) or {}
            if isinstance(doc, dict) and "runtime" in doc:
                runtime = doc["runtime"]
                runtime_src = str(c)
                break
            if isinstance(doc, dict) and "skill_timeouts" in doc:
                runtime = doc
                runtime_src = str(c)
                break
        except Exception:
            continue
    else:
        runtime_src = None
    # also try resolved_defaults
    rd = ROOT / "experiments/configs/resolved_defaults.yaml"
    if rd.exists():
        import yaml
        resolved = yaml.safe_load(rd.read_text(encoding="utf-8")) or {}
    else:
        resolved = {}
    runtime = (resolved.get("runtime") or runtime) if resolved else runtime

    ledger = []
    family_stats = {}
    for task_id in FAMILIES:
        split = load_split(task_id)
        # independent development/preflight pool: use dev cases, skip train/test
        pool = list(split.get("dev") or [])
        if not pool:
            raise BindingError("no dev cases for " + task_id)
        tmap = timeouts_for(task_id, runtime)
        deadline = deadline_for(task_id, runtime, tmap)
        attempts = []
        successes = []
        for i in range(MAX_ATTEMPTS):
            if len(successes) >= SUCCESS_TARGET:
                break
            row = pool[i % len(pool)]
            _log("reference %s try %s case %s" % (task_id, i, row["case_id"]))
            try:
                rec = run_one(task_id, row, tmap, deadline, gpu)
            except Exception as e:
                rec = {"task_id": task_id, "case_id": row["case_id"], "success": False, "reason": "EXCEPTION", "error": type(e).__name__, "episode_seconds": None, "skill_durations": [], "n_skills": 0}
                traceback.print_exc()
            rec["attempt_index"] = i
            attempts.append(rec)
            ledger.append(rec)
            (OUT / "reference_attempts.jsonl").open("a", encoding="utf-8").write(canonical(rec) + "\n")
            if rec.get("success"):
                successes.append(rec)
            _log("  -> success=%s reason=%s T=%s" % (rec.get("success"), rec.get("reason"), rec.get("episode_seconds")))
        T_med = statistics.median([s["episode_seconds"] for s in successes]) if successes else None
        durs = [d for s in successes for d in (s.get("skill_durations") or [])]
        d_ref = statistics.median(durs) if durs else None
        family_stats[task_id] = {
            "attempts": len(attempts),
            "successes": len(successes),
            "T_ref_median": T_med,
            "d_ref_median": d_ref,
            "success_case_ids": [s["case_id"] for s in successes],
        }
    ok = all(family_stats[t]["successes"] >= SUCCESS_TARGET and family_stats[t]["T_ref_median"] for t in FAMILIES)
    H = max(family_stats[t]["T_ref_median"] for t in FAMILIES) if ok else None
    payload = {
        "stage": "0A",
        "plan_version": "1.1",
        "method_version": "2.1.1",
        "runtime_src": runtime_src,
        "families": family_stats,
        "H": H,
        "ok": ok,
        "new_rl_runs": 0,
    }
    write_json(OUT / "reference_execution_manifest.json", payload)
    write_json(ROOT / "migration" / "reference_execution_manifest.json", payload)
    print(canonical(payload))
    if not ok:
        raise SystemExit("Stage 0A reference incomplete")
    set_suite_half_life(H)
    print("H", H)

if __name__ == "__main__":
    main()
