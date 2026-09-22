"""Stage 2A-P0: T_A/T_C runtime binding, 168 caches, 24-job registry. No PPO."""
from __future__ import annotations

import csv
import shutil
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .common import BindingError, canonical, digest
from .contracts import Registry, from_dict
from .facts import Truth
from .graph import Goal, build_template
from .adapters import EvaluationInput
from .platforms.libero.d0_env import CaseSpec, make_env
from .platforms.libero.skill_executor import SkillExecutor
from .platforms.libero.perception import PerceptionAdapter
from .platforms.libero.verifier import FactVerifier
from .platforms.libero.task_evaluator import TaskEvaluator
from .platforms.libero.safety import SafetyManager
from .platforms.libero.clock import DurationProvider
from .platforms.libero.runtime_factory import PREDICATES, TASK_OBJECTS, TASK_GOALS, create_task_runtime, sha_file
from .stage0c import PREPROCESSING, prepare_scene, file_hash
from .vlm_provider import ProviderConfig, DashScopeProvider, validate_payload
from .vlm_cache_pipeline import request_and_process, write_audit_cache
from .vlm import cache_key
from .stage2a_runner import validate_runner, planned_jobs

P0 = Path("experiments/part_2_exploration/stage_2a_p0")
CACHE_ROOT = Path("experiments/vlm_cache/stage_2a")
QUAL_BUDGET = 90.0
ATTEMPTS = 5
TASK_ROLE = {"D0": "second_object", "T_A": "second_object", "T_B": "second_object", "T_C": "interferer"}
SCRIPTED = {
    "T_A": [
        "a:OPEN:container:v1",
        "a:PICK:target:v1",
        "a:PLACE:target:container:v1",
        "a:PICK:second_object:v1",
        "a:PLACE:second_object:container:v1",
    ],
    "T_C": [
        "a:OPEN:container:v1",
        "a:PICK:interferer:v1",
        "a:PLACE_BUFFER:interferer:buffer:v1",
        "a:PICK:target:v1",
        "a:PLACE:target:container:v1",
    ],
    "T_B": [
        "a:OPEN:container:v1",
        "a:PICK:target:v1",
        "a:PLACE:target:container:v1",
        "a:PICK:second_object:v1",
        "a:PLACE_BUFFER:second_object:buffer:v1",
    ],
}


def _read(path):
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(obj) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fingerprint_tree(path: Path, exclude_parts=()) -> str:
    path = Path(path)
    if not path.exists():
        return "MISSING"
    recs = []
    if path.is_file():
        recs.append((path.as_posix(), sha_bytes(path.read_bytes())))
    else:
        for p in sorted(path.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(path).as_posix()
            if any(part in rel.split("/") for part in exclude_parts):
                continue
            if "__pycache__" in rel or p.name.endswith(".pyc"):
                continue
            recs.append((rel, sha_bytes(p.read_bytes())))
    return digest(recs)


def _log(msg):
    print("[stage2a-p0] " + str(msg), flush=True)


def objects_for(task_id):
    return dict(TASK_OBJECTS[task_id])


def grounded_ids(task_id):
    role = TASK_ROLE[task_id]
    return [
        "a:OPEN:container:v1",
        "a:PICK:target:v1",
        f"a:PICK:{role}:v1",
        "a:PLACE:target:container:v1",
        f"a:PLACE:{role}:container:v1",
        "a:PLACE_BUFFER:target:buffer:v1",
        f"a:PLACE_BUFFER:{role}:buffer:v1",
    ]


def qualify_skills(task_id):
    role = TASK_ROLE[task_id]
    ids = [
        "a:OPEN:container:v1",
        "a:PICK:target:v1",
        f"a:PICK:{role}:v1",
        "a:PLACE:target:container:v1",
        f"a:PLACE_BUFFER:{role}:buffer:v1",
    ]
    if task_id == "T_A":
        ids.append("a:PLACE:second_object:container:v1")
    return ids


def _xy(rng, x_lo, x_hi, y_lo, y_hi):
    return [float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi))]


def build_task_split(task_id: str):
    role = TASK_ROLE[task_id]
    train, dev = [], []
    container = [0.18, 0.12]
    buffer = [-0.18, 0.12]
    train_base = {"T_A": 2100, "T_C": 3100, "T_B": 4100}.get(task_id, 4100)
    dev_base = {"T_A": 6100, "T_C": 7100, "T_B": 8100}.get(task_id, 8100)
    for i in range(64):
        rng = np.random.RandomState(train_base + i)
        t = _xy(rng, -0.22, -0.02, -0.18, -0.02)
        if task_id == "T_C":
            c = np.array(container, dtype=float)
            tt = np.array(t, dtype=float)
            vec = c - tt
            nrm = float(np.linalg.norm(vec)) or 1.0
            s = (tt + 0.038 * vec / nrm).tolist()
        else:
            s = _xy(rng, 0.04, 0.22, -0.18, -0.02)
            while float(np.linalg.norm(np.array(t) - np.array(s))) < 0.09:
                s = _xy(rng, 0.04, 0.22, -0.18, -0.02)
        train.append({
            "case_id": f"{task_id}_train_{i:02d}",
            "split": "train",
            "seed": int(train_base + i),
            "target_xy": t,
            "second_xy": s,
            "container_xy": container,
            "buffer_xy": buffer,
            "lid_closed": True,
            "second_role": role,
        })
    for i in range(20):
        rng = np.random.RandomState(dev_base + i)
        t = _xy(rng, -0.22, -0.02, -0.18, -0.02)
        if task_id == "T_C":
            c = np.array(container, dtype=float)
            tt = np.array(t, dtype=float)
            vec = c - tt
            nrm = float(np.linalg.norm(vec)) or 1.0
            s = (tt + 0.038 * vec / nrm).tolist()
        else:
            s = _xy(rng, 0.04, 0.22, -0.18, -0.02)
            while float(np.linalg.norm(np.array(t) - np.array(s))) < 0.09:
                s = _xy(rng, 0.04, 0.22, -0.18, -0.02)
        dev.append({
            "case_id": f"{task_id}_dev_{i:02d}",
            "split": "dev",
            "seed": int(dev_base + i),
            "target_xy": t,
            "second_xy": s,
            "container_xy": container,
            "buffer_xy": buffer,
            "lid_closed": True,
            "second_role": role,
        })
    ids = [r["case_id"] for r in train + dev]
    if len(set(ids)) != len(ids):
        raise BindingError("duplicate case ids")
    return {
        "task_id": task_id,
        "version": "stage-2a-p0",
        "policy": "frozen_before_vlm_or_training",
        "train_count": 64,
        "dev_count": 20,
        "test_isolated": True,
        "reuse_stage_0c": False,
        "reuse_reason": "Stage 0C T_A/T_C caches are historical: action registry dropped MOVE and signed goals/runtime objects changed. Old caches retained, not reused.",
        "second_role": role,
        "train": train,
        "dev": dev,
    }


def spec_from_row(task_id, row, deadline=90.0):
    return CaseSpec(
        case_id=row["case_id"],
        split=row["split"],
        seed=int(row["seed"]),
        target_xy=tuple(row["target_xy"]),
        second_xy=tuple(row["second_xy"]),
        container_xy=tuple(row["container_xy"]),
        buffer_xy=tuple(row["buffer_xy"]),
        lid_closed=bool(row.get("lid_closed", True)),
        task_id=task_id,
        second_role=TASK_ROLE[task_id],
        deadline=float(deadline),
    )


def bind_env(case, gpu=0, deadline=90.0):
    env = make_env(case, gpu=gpu)
    env.last_perception = {}
    env.reset()
    env._apply_case_poses()
    env._open_gripper_reset()
    clock = DurationProvider(env)
    safety = SafetyManager(env)
    perception = PerceptionAdapter(env)
    env.refresh_perception = lambda e=env, p=perception: p.infer(e.public_observation())
    verifier = FactVerifier(env)
    evaluator = TaskEvaluator(env, deadline, task_id=case.task_id)
    executor = SkillExecutor(env, safety, clock)
    return env, clock, safety, perception, verifier, evaluator, executor


def hidden_qa(env):
    h = env.hidden_truth()
    c = h["container"]
    b = h["buffer"]
    table = float(h["table_top_z"])
    role = getattr(env, "second_role", "second_object")
    out = {}
    for name in ("target", role):
        p = h[name]
        out[f"OnTable:{name}"] = abs(p[2] - (table + 0.02)) < 0.05
        out[f"Inside:{name}:container"] = abs(p[0] - c[0]) <= 0.03 and abs(p[1] - c[1]) <= 0.03 and p[2] <= table + 0.12
        out[f"AtBuffer:{name}:buffer"] = abs(p[0] - b[0]) <= 0.08 and abs(p[1] - b[1]) <= 0.08
        out[f"Held:{name}"] = float(np.linalg.norm(p[:2] - h["eef_pos"][:2])) < 0.07 and p[2] > table + 0.06
    lid = h["lid"]
    out["Open:container"] = ((lid[0] - c[0]) ** 2 + (lid[1] - c[1]) ** 2) ** 0.5 > 0.10
    g = h["gripper_qpos"]
    out["GripperEmpty"] = float(np.mean(np.abs(g))) > 0.02
    return out


def expected_post(skill_id):
    parts = skill_id.split(":")
    skill, args = parts[1], parts[2:-1]
    if skill == "OPEN":
        return "p:Open:container"
    if skill == "PICK":
        return f"p:Held:{args[0]}"
    if skill == "PLACE":
        return f"p:Inside:{args[0]}:{args[1]}"
    if skill == "PLACE_BUFFER":
        return f"p:AtBuffer:{args[0]}:{args[1]}"
    raise BindingError("unknown skill " + skill_id)




def fact_value(records, fid):
    for rec in records:
        if rec.fact_id == fid:
            return rec.value
    return Truth.UNKNOWN


def setup_for_skill(task_id, env, executor, skill_id, timeout):
    parts = skill_id.split(":")
    skill, args = parts[1], parts[2:-1]
    logs = []
    role = TASK_ROLE[task_id]
    if skill in ("PLACE", "PLACE_BUFFER"):
        logs.append(executor.execute("a:OPEN:container:v1", timeout))
        if task_id == "T_C" and args and args[0] == "target":
            logs.append(executor.execute(f"a:PICK:{role}:v1", timeout))
            logs.append(executor.execute(f"a:PLACE_BUFFER:{role}:buffer:v1", timeout))
        logs.append(executor.execute(f"a:PICK:{args[0]}:v1", timeout))
    elif skill == "PICK" and task_id == "T_C" and args and args[0] == "target":
        logs.append(executor.execute("a:OPEN:container:v1", timeout))
        logs.append(executor.execute(f"a:PICK:{role}:v1", timeout))
        logs.append(executor.execute(f"a:PLACE_BUFFER:{role}:buffer:v1", timeout))
    return logs

def qa_key(fact_id):
    return fact_id[2:] if fact_id.startswith("p:") else fact_id


def qualify_controllers(task_id, gpu=0):
    rows = []
    durations = defaultdict(list)
    role = TASK_ROLE[task_id]
    for skill_id in qualify_skills(task_id):
        ok = 0
        for attempt in range(ATTEMPTS):
            seed = (8000 if task_id == "T_A" else 9000) + qualify_skills(task_id).index(skill_id) * 10 + attempt
            txy = [-0.12, -0.10]
            cxy = [0.18, 0.12]
            if task_id == "T_C":
                t = np.array(txy, dtype=float)
                c = np.array(cxy, dtype=float)
                vec = c - t
                nrm = float(np.linalg.norm(vec)) or 1.0
                sxy = (t + 0.038 * vec / nrm).tolist()
            else:
                sxy = [0.10, -0.10]
            case = spec_from_row(task_id, {
                "case_id": f"ctrl_{task_id}_{skill_id}_{attempt}",
                "split": "dev",
                "seed": seed,
                "target_xy": txy,
                "second_xy": sxy,
                "container_xy": cxy,
                "buffer_xy": [-0.18, 0.12],
                "lid_closed": True,
            })
            env, clock, safety, perception, verifier, evaluator, executor = bind_env(case, gpu=gpu)
            try:
                setup_for_skill(task_id, env, executor, skill_id, QUAL_BUDGET)
                obs_before = env.public_observation()
                rec_before = verifier.verify(perception.infer(obs_before), None)
                exe = executor.execute(skill_id, QUAL_BUDGET)
                obs_after = env.public_observation()
                rec_after = verifier.verify(perception.infer(obs_after), exe)
                qa = hidden_qa(env)
                fid = expected_post(skill_id)
                v_val = fact_value(rec_after, fid)
                qa_val = bool(qa.get(qa_key(fid), False))
                verified = exe["controller_exit"] == "NORMAL_TERMINATION" and v_val == Truth.TRUE and qa_val
                if verified:
                    ok += 1
                    durations[skill_id.split(":")[1]].append(float(exe["sim_duration"]))
                wrong_object = False
                if skill_id.startswith("a:PICK:"):
                    other = role if "target" in skill_id else "target"
                    if fact_value(rec_after, f"p:Held:{other}") == Truth.TRUE and qa.get(f"Held:{other}"):
                        wrong_object = True
                rows.append({
                    "task_id": task_id,
                    "skill_id": skill_id,
                    "attempt": attempt,
                    "seed": seed,
                    "controller_exit": exe["controller_exit"],
                    "sim_duration": float(exe["sim_duration"]),
                    "steps": int(exe["steps"]),
                    "verifier_fact": fid,
                    "verifier_value": v_val.name if hasattr(v_val, "name") else str(v_val),
                    "hidden_qa": bool(qa_val),
                    "verified_success": bool(verified),
                    "wrong_object": bool(wrong_object),
                    "timeout": bool(exe["timeout"]),
                    "nan_blocked": bool(exe["nan_blocked"]),
                    "rejected": bool(exe["rejected"]),
                    "interrupted": bool(exe["interrupted"]),
                    "policy_input_keys": sorted(list(obs_after.keys())),
                })
            finally:
                env.close()
        _log("%s %s verified %s/%s" % (task_id, skill_id, ok, ATTEMPTS))
    return rows, durations


def tc_interference_probe(gpu=0):
    """Record geometric blocking; not a qualification gate. Target PICK/PLACE are qualified after relocating interferer."""
    t = np.array([-0.12, -0.10], dtype=float)
    c = np.array([0.18, 0.12], dtype=float)
    vec = c - t
    nrm = float(np.linalg.norm(vec)) or 1.0
    s = t + 0.038 * vec / nrm
    between = float(np.dot(s - t, vec) / (nrm * nrm))
    dist_ts = float(np.linalg.norm(s - t))
    row = {
        "target_xy": t.tolist(),
        "interferer_xy": s.tolist(),
        "container_xy": c.tolist(),
        "target_interferer_distance": dist_ts,
        "interferer_fraction_along_target_to_container": between,
        "geometrically_between": bool(0.0 < between < 1.0),
        "rule": "second_xy = target + 0.038 * unit(container-target)",
    }
    case = spec_from_row("T_C", {
        "case_id": "tc_interference",
        "split": "dev",
        "seed": 42,
        "target_xy": t.tolist(),
        "second_xy": s.tolist(),
        "container_xy": c.tolist(),
        "buffer_xy": [-0.18, 0.12],
        "lid_closed": True,
    })
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(case, gpu=gpu)
    try:
        exe = executor.execute("a:PICK:target:v1", QUAL_BUDGET)
        recs = verifier.verify(perception.infer(env.public_observation()), exe)
        row["pick_target_without_clearing"] = {
            "exit": exe["controller_exit"],
            "held_target": fact_value(recs, "p:Held:target").name,
            "duration": float(exe["sim_duration"]),
        }
    finally:
        env.close()
    return row


def qualify_verifier(task_id, gpu=0):
    role = TASK_ROLE[task_id]
    scenarios = [
        ("empty_gripper", None),
        ("successful_hold", "a:PICK:target:v1"),
        ("container_open", "a:OPEN:container:v1"),
        ("buffer_role", f"a:PLACE_BUFFER:{role}:buffer:v1"),
        ("normal_place_target", "a:PLACE:target:container:v1"),
    ]
    rows = []
    case = spec_from_row(task_id, {
        "case_id": f"ver_{task_id}",
        "split": "dev",
        "seed": 42,
        "target_xy": [-0.12, -0.10],
        "second_xy": [0.10, -0.10] if task_id == "T_A" else [-0.12 + 0.038 * 0.8, -0.10 + 0.038 * 0.6],
        "container_xy": [0.18, 0.12],
        "buffer_xy": [-0.18, 0.12],
        "lid_closed": True,
    })
    if task_id == "T_C":
        t = np.array([-0.12, -0.10]); c = np.array([0.18, 0.12]); vec = c - t; nrm = float(np.linalg.norm(vec)) or 1.0
        case = spec_from_row(task_id, {
            "case_id": f"ver_{task_id}", "split": "dev", "seed": 42,
            "target_xy": t.tolist(), "second_xy": (t + 0.038 * vec / nrm).tolist(),
            "container_xy": c.tolist(), "buffer_xy": [-0.18, 0.12], "lid_closed": True,
        })
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(case, gpu=gpu)
    try:
        for name, skill in scenarios:
            if skill:
                setup_for_skill(task_id, env, executor, skill, QUAL_BUDGET)
                executor.execute(skill, QUAL_BUDGET)
            obs = env.public_observation()
            meas = perception.infer(obs)
            recs = verifier.verify(meas, None)
            payload = json.dumps(meas.measurements, default=str)
            leaked = ("hidden_truth" in payload) or ("body_xpos" in payload) or ('"qpos"' in payload)
            rows.append({
                "task_id": task_id,
                "scenario": name,
                "skill": skill,
                "leaked_hidden_into_verifier": leaked,
                "verifier": {r.fact_id: r.value.name for r in recs},
            })
    finally:
        env.close()
    return rows


def evaluator_unit_checks(task_id, gpu=0):
    results = []
    case = spec_from_row(task_id, {
        "case_id": f"eval_{task_id}",
        "split": "dev",
        "seed": 9,
        "target_xy": [-0.12, -0.10],
        "second_xy": [0.10, -0.10],
        "container_xy": [0.18, 0.12],
        "buffer_xy": [-0.18, 0.12],
        "lid_closed": True,
    })
    if task_id == "T_C":
        t = np.array([-0.12, -0.10]); c = np.array([0.18, 0.12]); vec = c - t; nrm = float(np.linalg.norm(vec)) or 1.0
        case = spec_from_row(task_id, {
            "case_id": f"eval_{task_id}", "split": "dev", "seed": 9,
            "target_xy": t.tolist(), "second_xy": (t + 0.038 * vec / nrm).tolist(),
            "container_xy": c.tolist(), "buffer_xy": [-0.18, 0.12], "lid_closed": True,
        })
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(case, gpu=gpu, deadline=90.0)
    try:
        evaluator.reset_episode()
        r0 = evaluator.evaluate(EvaluationInput(task_id, "e", "ep", (), 1.0, 0.0, 1.0))
        results.append({"task_id": task_id, "name": "incomplete_zero", "pass": (not r0.success) and r0.reward_events == ()})
        for sid in SCRIPTED[task_id]:
            exe = executor.execute(sid, QUAL_BUDGET)
            ev = evaluator.evaluate(EvaluationInput(task_id, "e", "ep", exe["evidence_ids"], clock.now_seconds(), exe["start_seconds"], exe["end_seconds"]))
            results.append({
                "task_id": task_id,
                "name": "after_" + sid,
                "skill": sid,
                "success": ev.success,
                "reward": ev.reward_events,
                "reason": ev.reason,
            })
        rewards = [item for item in results if item.get("reward")]
        terminal = [item for item in results if item.get("success")]
        if task_id == "T_A":
            one_obj = [item for item in results if item.get("name") == "after_a:PLACE:target:container:v1"]
            results.append({"task_id": task_id, "name": "one_object_zero", "pass": bool(one_obj) and not one_obj[0]["success"] and not one_obj[0]["reward"]})
            results.append({"task_id": task_id, "name": "both_inside_reward", "pass": bool(terminal) and any(r.get("reward") for r in terminal)})
        else:
            buf = [item for item in results if item.get("name") == "after_a:PLACE_BUFFER:interferer:buffer:v1"]
            results.append({"task_id": task_id, "name": "interferer_buffer_zero", "pass": bool(buf) and not buf[0]["success"] and not buf[0]["reward"]})
            results.append({"task_id": task_id, "name": "target_inside_reward", "pass": bool(terminal) and any(r.get("reward") for r in terminal)})
        r_last = evaluator.evaluate(EvaluationInput(task_id, "e", "ep", (), clock.now_seconds() + 1.0, 0.0, clock.now_seconds() + 1.0))
        results.append({"task_id": task_id, "name": "no_repeat_reward", "pass": r_last.reward_events == ()})
        env.reset()
        env._apply_case_poses()
        env._open_gripper_reset()
        evaluator_d = TaskEvaluator(env, 0.5, task_id=task_id)
        evaluator_d.reset_episode()
        rd = evaluator_d.evaluate(EvaluationInput(task_id, "e", "ep2", (), 0.6, 0.0, 0.6))
        results.append({"task_id": task_id, "name": "deadline_zero", "pass": (not rd.success) and (not rd.truncated) and rd.reward_events == () and bool(rd.terminated) and rd.reason == "DEADLINE", "reason": rd.reason})
    finally:
        env.close()
    return results


def calibrate_timing(task_id, durations):
    timeouts = {}
    for skill in ("OPEN", "PICK", "PLACE", "PLACE_BUFFER"):
        xs = list(durations.get(skill) or [])
        if not xs:
            raise BindingError(f"No verified-success duration for {task_id} {skill}")
        xs = sorted(float(x) for x in xs)
        p95 = xs[max(0, int(math.ceil(0.95 * len(xs)) - 1))]
        timeouts[skill] = float(max(math.ceil(p95 * 2.0), math.ceil(max(xs) * 1.25) + 2.0, 8.0))
    all_ok = [float(d) for xs in durations.values() for d in xs]
    d_ref = float(statistics.median(all_ok))
    if task_id == "T_A":
        deadline = float(timeouts["OPEN"] + 2 * timeouts["PICK"] + 2 * timeouts["PLACE"] + 10.0)
        rule = "timeout=max(ceil(p95*2), ceil(max*1.25)+2, 8); deadline=OPEN+2*PICK+2*PLACE+10; d_ref=median verified-success"
    else:
        deadline = float(timeouts["OPEN"] + 2 * timeouts["PICK"] + timeouts["PLACE"] + timeouts["PLACE_BUFFER"] + 10.0)
        rule = "timeout=max(ceil(p95*2), ceil(max*1.25)+2, 8); deadline=OPEN+2*PICK+PLACE+PLACE_BUFFER+10; d_ref=median verified-success"
    return {
        "task_id": task_id,
        "rule": rule,
        "qualification_attempt_budget_seconds": QUAL_BUDGET,
        "skill_timeouts": timeouts,
        "task_deadline_seconds": deadline,
        "reference_skill_seconds": d_ref,
        "copied_from_d0": False,
        "d0_deadline_seconds": 43.0,
        "d0_d_ref": 3.55,
        "samples": {k: v for k, v in durations.items()},
    }

def capture_split_images(root: Path, task_id, split_doc, gpu=0):
    mapping = []
    dest_root = root / "experiments/stage_2a_inputs" / task_id
    dest_root.mkdir(parents=True, exist_ok=True)
    for row in split_doc["train"] + split_doc["dev"]:
        dest = dest_root / row["split"] / row["case_id"]
        dest.mkdir(parents=True, exist_ok=True)
        rgb_path = dest / "rgb.png"
        if (dest / "CAPTURE_COMPLETE").exists() and rgb_path.exists():
            mapping.append({**row, "image_ref": str(rgb_path.relative_to(root)), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(root))})
            continue
        spec = spec_from_row(task_id, row)
        env = make_env(spec, gpu=gpu)
        try:
            env.reset()
            env._apply_case_poses()
            env._open_gripper_reset()
            obs = env.public_observation()
            from PIL import Image
            Image.fromarray(np.asarray(obs["rgb"]).astype(np.uint8)).save(rgb_path)
            np.save(dest / "depth.npy", np.asarray(obs["depth"]))
            write_json(dest / "reset_config.json", row)
            hidden = env.hidden_truth()
            write_json(dest / "hidden_truth_qa.json", {k: np.asarray(v).tolist() if hasattr(v, "tolist") else v for k, v in hidden.items()})
            (dest / "CAPTURE_COMPLETE").write_text("complete\n", encoding="utf-8")
        finally:
            env.close()
        mapping.append({**row, "image_ref": str(rgb_path.relative_to(root)), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(root))})
    return mapping


def _grounded_template(root: Path, task_id):
    doc = _read(root / "configs/runtime/stage_2a_contract_registry.yaml")
    if any(c["name"] == "MOVE" for c in doc["contracts"]):
        raise BindingError("MOVE must not appear in Stage 2A runtime registry")
    objects = objects_for(task_id)
    reg = Registry(doc["predicate_types"])
    for c in doc["contracts"]:
        reg.register(from_dict(c))
    contracts = reg.ground(objects)
    goals = tuple(Goal(g, 1) for g in TASK_GOALS[task_id])
    template = build_template(contracts, goals, PREDICATES, objects)
    return doc, contracts, template


def initial_facts(task_id):
    role = TASK_ROLE[task_id]
    return {
        "p:GripperEmpty": "TRUE",
        "p:Held:target": "FALSE",
        f"p:Held:{role}": "FALSE",
        "p:OnTable:target": "TRUE",
        f"p:OnTable:{role}": "TRUE",
        "p:Open:container": "FALSE",
        "p:Inside:target:container": "FALSE",
        f"p:Inside:{role}:container": "FALSE",
        "p:AtBuffer:target:buffer": "FALSE",
        f"p:AtBuffer:{role}:buffer": "FALSE",
    }


def generate_task_caches(root: Path, task_id, mapping, gpu=0):
    vlm = _read(root / "experiments/manifests/vlm_manifest.yaml")
    config = ProviderConfig(model=vlm["model_snapshot"], region=vlm["region"], endpoint=vlm["base_http_api_url"], sdk_version=vlm["sdk_version"])
    provider = DashScopeProvider(config)
    if not provider.key_present():
        raise BindingError("DASHSCOPE_API_KEY missing from process environment")
    few_path = root / "experiments/stage_0c_inputs/fewshots/few_shot_manifest.json"
    few_doc = json.loads(few_path.read_text(encoding="utf-8"))
    few = few_doc.get("examples", [])
    examples = [prepare_scene(root, s, True) for s in few]
    prompt_path = root / "experiments/sources/v2.1_interfaces/system_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    schema_path = root / "schemas/relation_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    messages = [{"role": "system", "content": [{"text": prompt}]}]
    for ex, record in zip(examples, few):
        messages.extend([
            {"role": "user", "content": ex["content"]},
            {"role": "assistant", "content": [{"text": canonical(record["expected_json"])}]},
        ])
    contract_doc, contracts, template = _grounded_template(root, task_id)
    actions = sorted(c.id for c in contracts)
    props = sorted(n.id for n in template.nodes if n.kind == "PROPOSITION")
    contract_sha = file_hash(root / "configs/runtime/stage_2a_contract_registry.yaml")
    task_path = root / f"configs/tasks/resolved/{task_id}.yaml"
    if not task_path.exists():
        raise BindingError("resolved task missing: " + str(task_path))
    task_sha = file_hash(task_path)
    ledger = root / P0 / f"{task_id}_cache_request_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    results = []
    import base64
    from PIL import Image
    role = TASK_ROLE[task_id]
    signed = [{"fact_id": g, "sign": 1} for g in TASK_GOALS[task_id]]
    for row in mapping:
        image = root / row["image_ref"]
        with Image.open(image) as im:
            mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[im.format]
        scene = {
            "scene_id": row["case_id"],
            "task_id": task_id,
            "split": row["split"],
            "image_ref": row["image_ref"],
            "image_sha256": row["image_sha256"],
            "object_table": [
                {"id": "target", "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": role, "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "container", "type": "container", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "buffer", "type": "buffer", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
            ],
            "signed_goals": signed,
            "initial_facts": initial_facts(task_id),
            "allowed_action_ids": actions,
            "allowed_proposition_ids": props,
            "contracts_sha256": contract_sha,
            "task_definition_sha256": task_sha,
            "predicate_version": "CP-DISR-v2.1",
            "asset_binding_sha256": digest({"layout": [row["target_xy"], row["second_xy"], row["container_xy"], row["buffer_xy"]], "runtime": "stage-2a-p0", "task_id": task_id}),
        }
        context = {k: scene[k] for k in ["scene_id", "task_id", "object_table", "signed_goals", "initial_facts", "allowed_action_ids", "allowed_proposition_ids"]}
        context["object_table"] = sorted(context["object_table"], key=lambda x: x["id"])
        context["skill_contracts"] = contract_doc
        context["registered_effect_edges"] = [e for e in template.edges if e[2] in ("ADD", "DEL")]
        content = [
            {"image": "data:image/" + mime + ";base64," + base64.b64encode(image.read_bytes()).decode()},
            {"text": canonical(context)},
        ]
        payload = {
            "model": config.model,
            "messages": messages + [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
            "enable_search": False,
            "stream": False,
            "result_format": "message",
        }
        validate_payload(payload)
        manifest = {
            "split": row["split"],
            "task_definition_hash": scene["task_definition_sha256"],
            "initial_RGB_content_hash": scene["image_sha256"],
            "preprocessing_hash": digest(PREPROCESSING),
            "object_binding_hash": digest(context["object_table"]),
            "allowed_ID_hash": digest({"actions": actions, "propositions": props, "signed_goals": scene["signed_goals"]}),
            "contract_version": contract_sha,
            "predicate_version": scene["predicate_version"],
            "model_snapshot": config.model,
            "sdk_api_version": config.sdk_version,
            "region": config.region,
            "endpoint": config.endpoint,
            "prompt_hash": file_hash(prompt_path),
            "fewshot_hash": digest(few_doc),
            "schema_hash": file_hash(schema_path),
            "decoding_config": {"temperature": 0, "max_tokens": 2048, "max_relations": 8, "thinking": False, "response_format": "json_object"},
            "scene_id": scene["scene_id"],
            "task_id": task_id,
            "synthetic_unit_fixture": False,
            "initial_facts_hash": digest(scene["initial_facts"]),
            "input_context_hash": digest(context),
            "request_payload_hash": digest(payload),
            "asset_binding_hash": scene["asset_binding_sha256"],
        }
        key = cache_key(manifest)
        path = root / CACHE_ROOT / row["split"] / key
        man_path = path / "manifest.json"
        reusable = False
        if path.exists() and (path / "COMPLETE").is_file() and man_path.is_file():
            try:
                man_doc = json.loads(man_path.read_text(encoding="utf-8"))
            except Exception:
                man_doc = {}
            reusable = man_doc.get("processing_status") == "SUCCESS"
        if reusable:
            results.append({"case_id": row["case_id"], "split": row["split"], "task_id": task_id, "cache_key": key, "status": "REUSED", "cache_dir": str(path.relative_to(root))})
            continue
        if path.exists() and not reusable:
            att = path / "failed_attempts"
            n = len(list(att.glob("attempt_*"))) if att.is_dir() else 0
            dest = att / ("attempt_%03d" % n)
            dest.mkdir(parents=True, exist_ok=True)
            budget_path = path / "retry_budget.json"
            used = n + 1
            if budget_path.is_file():
                import json as _json
                try:
                    used = int(_json.loads(budget_path.read_text(encoding="utf-8")).get("used") or 0) + 1
                except Exception:
                    pass
            budget_path.write_text(canonical({"scene_id": row["case_id"], "cache_key": key, "used": used}) + "\n", encoding="utf-8")
            raise BindingError("Refusing to rmtree existing cache path; retry budget persisted")
        with ledger.open("a", encoding="utf-8") as f:
            f.write(canonical({"scene_id": row["case_id"], "state": "REQUEST_STARTED", "time": time.time()}) + "\n")
        execution = request_and_process(provider, payload, template, schema, ())
        written = write_audit_cache(root / CACHE_ROOT, manifest, prompt, scene, execution)
        rec = {
            "case_id": row["case_id"],
            "split": row["split"],
            "task_id": task_id,
            "cache_key": key,
            "status": execution["status"],
            "cache_dir": str(Path(written).relative_to(root)),
            "attempts": len(execution["attempts"]),
            "accepted": 0 if not execution.get("processing") else len(execution["processing"].get("accepted") or []),
            "raw_relations": 0 if not execution.get("processing") else len((execution["processing"].get("parsed") or {}).get("relations") or []),
        }
        results.append(rec)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(canonical(rec) + "\n")
        if rec["status"] != "SUCCESS":
            written_path = Path(written)
            complete_file = written_path / "COMPLETE"
            if complete_file.exists():
                complete_file.unlink()
            (written_path / "FAILED").write_text(str(rec["status"]) + "\n", encoding="utf-8")
        last_err = execution["attempts"][-1]["response"].get("error_type")
        if last_err in ("AUTHORIZATION", "REQUEST_REJECTED", "SDK_ERROR"):
            raise BindingError("Provider rejected frozen configuration; stop without fallback: " + str(last_err))
        _log("cache %s %s %s" % (task_id, row["case_id"], rec["status"]))
    return results

def scripted_dev20(task_id, split_doc, timeout_map, deadline, gpu=0):
    rows = []
    seq = SCRIPTED[task_id]
    for row in split_doc["dev"]:
        spec = spec_from_row(task_id, row, deadline=deadline)
        env, clock, safety, perception, verifier, evaluator, executor = bind_env(spec, gpu=gpu, deadline=deadline)
        evaluator.deadline = float(deadline)
        evaluator.reset_episode()
        try:
            rewards = []
            skill_log = []
            hidden_in_policy = False
            for sid in seq:
                skill = sid.split(":")[1]
                exe = executor.execute(sid, float(timeout_map[skill]))
                elapsed = clock.now_seconds()
                ev = evaluator.evaluate(EvaluationInput(task_id, "env0", row["case_id"], exe["evidence_ids"], elapsed, exe["start_seconds"], exe["end_seconds"]))
                rewards.extend(ev.reward_events)
                obs = env.public_observation()
                payload = json.dumps({k: (np.asarray(v).shape if hasattr(v, "shape") else v) for k, v in obs.items()}, default=str)
                if "hidden_truth" in payload or "body_xpos" in payload:
                    hidden_in_policy = True
                skill_log.append({"skill": sid, "exit": exe["controller_exit"], "duration": float(exe["sim_duration"]), "eval": ev.reason, "success": ev.success, "exception": bool(exe["interrupted"])})
            final = evaluator.evaluate(EvaluationInput(task_id, "env0", row["case_id"], (), clock.now_seconds(), 0.0, clock.now_seconds()))
            rows.append({
                "task_id": task_id,
                "case_id": row["case_id"],
                "seed": row["seed"],
                "goal_true": evaluator.goal_true(),
                "success": final.success,
                "terminated": final.terminated,
                "truncated": final.truncated,
                "reward_count": len([r for r in rewards if r[1] == 1.0]),
                "repeat_reward": len([r for r in rewards if r[1] == 1.0]) > 1,
                "hidden_in_policy": hidden_in_policy,
                "controller_exception": any(s["exception"] for s in skill_log),
                "over_deadline": bool(final.truncated),
                "skills": json.dumps(skill_log),
            })
        finally:
            env.close()
        _log("scripted %s %s success=%s" % (task_id, row["case_id"], rows[-1]["success"]))
    return rows


def hidden_truth_leakage_check(task_id, gpu=0):
    case = spec_from_row(task_id, {
        "case_id": f"leak_{task_id}", "split": "dev", "seed": 3,
        "target_xy": [-0.12, -0.10], "second_xy": [0.10, -0.10],
        "container_xy": [0.18, 0.12], "buffer_xy": [-0.18, 0.12], "lid_closed": True,
    })
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(case, gpu=gpu)
    try:
        from .platforms.libero.observations import ObservationProvider
        obs = ObservationProvider(env, clock).observe()
        meas = perception.infer(env.public_observation())
        recs = verifier.verify(meas, None)
        payload = json.dumps({"obs": obs.__dict__, "meas": meas.measurements, "facts": [r.reason for r in recs]}, default=str)
        hidden = env.hidden_truth()
        leaked = False
        hits = []
        role = TASK_ROLE[task_id]
        for name in ("target", role, "lid"):
            xyz = hidden[name]
            token = f"{float(xyz[0]):.5f}"
            if token in payload:
                leaked = True
                hits.append(name)
        return {"task_id": task_id, "leaked": leaked, "hits": hits, "public_keys": list(env.public_observation().keys()), "hidden_keys": list(hidden.keys())}
    finally:
        env.close()


def _absmax(value):
    if value is None:
        return 0.0
    try:
        import torch
        if hasattr(value, "detach"):
            return float(value.detach().abs().max().cpu())
    except Exception:
        pass
    return 0.0


def model_dry_run(root, task_id, method, case_id, manifest, output: Path):
    import torch
    from .neural import Policy
    from .collector import Collector
    bundle = create_task_runtime(manifest, task_id)
    snap = bundle.start_case(case_id)
    objects = objects_for(task_id)
    actions = sorted({c.name for c in snap.template.contracts})
    predicates = sorted(PREDICATES)
    types = sorted(set(objects.values()))
    model = Policy(actions, predicates, types, observation_dim=48, candidate_dim=8, method=method, B=0.5)
    model.eval()
    collector = Collector(bundle, model)
    collector.reset_episode(snap.env_id, snap.episode_id)
    t, result = collector.step(snap)
    out = collector.last_output
    diag = getattr(out, "diagnostics", {}) or {}
    dp_max = 0.0
    dk_max = 0.0
    delta_max = 0.0
    up_max = 0.0
    for cid, diff in (diag.get("differences") or {}).items():
        dp_max = max(dp_max, _absmax(getattr(diff, "dp", None)))
        dk_max = max(dk_max, _absmax(getattr(diff, "dk", None)))
    for cid, residual in (diag.get("delta") or {}).items():
        delta_max = max(delta_max, _absmax(residual))
    for cid, up in (diag.get("up") or {}).items():
        up_max = max(up_max, _absmax(up))
    selected = None if t is None else t.selected_candidate_id
    report = {
        "task_id": task_id,
        "method": method,
        "case_id": case_id,
        "prior_edge_count": len(snap.prior_edges),
        "mask_true": int(sum(snap.mask)),
        "candidate_ids": list(snap.candidate_ids),
        "mask": list(snap.mask),
        "selected_candidate_id": selected,
        "source_prior_edge_count": len(getattr(bundle, "original_prior_edges", ()) or ()),
        "effective_prior_edge_count": 0 if method in ("B0", "B2") else len(snap.prior_edges),
        "transition": None if t is None else {
            "duration": t.duration,
            "reward": t.reward,
            "terminated": t.terminated,
            "truncated": t.truncated,
            "reason": t.reason,
            "selected_candidate_id": t.selected_candidate_id,
        },
        "result": result if isinstance(result, dict) else {
            "success": result.success,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "reason": result.reason,
        },
        "controller_exit": None if collector.last_execution is None else getattr(collector.last_execution, "controller_exit", None) if not isinstance(collector.last_execution, dict) else collector.last_execution.get("controller_exit"),
        "diagnostics": {
            "dp_abs_max": dp_max,
            "dk_abs_max": dk_max,
            "delta_abs_max": delta_max,
            "up_abs_max": up_max,
            "b0_no_differences": method == "B0" and not diag.get("differences"),
            "b1_no_nominal_successor": method == "B1" and not diag.get("differences"),
        },
        "optimizer_steps": 0,
        "loss_computed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{task_id}_{method}_dry_run.json", report)
    bundle.environment.close()
    return report


def cache_stats(root, caches):
    rows = []
    nonempty = 0
    empty = 0
    accepted_total = 0
    raw_total = 0
    evaluable = False
    evaluable_case = None
    for rec in caches:
        cdir = root / rec["cache_dir"]
        accepted = json.loads((cdir / "accepted_relations.json").read_text(encoding="utf-8")) if (cdir / "accepted_relations.json").exists() else []
        parsed = json.loads((cdir / "parsed_relations.json").read_text(encoding="utf-8")) if (cdir / "parsed_relations.json").exists() else {}
        edges = json.loads((cdir / "final_edges.json").read_text(encoding="utf-8")) if (cdir / "final_edges.json").exists() else []
        n_acc = len(accepted or [])
        n_raw = len((parsed or {}).get("relations") or [])
        accepted_total += n_acc
        raw_total += n_raw
        if n_acc:
            nonempty += 1
        else:
            empty += 1
        if n_acc and edges:
            evaluable = True
            if evaluable_case is None:
                evaluable_case = rec["case_id"]
        rows.append({**rec, "accepted_count": n_acc, "raw_count": n_raw, "final_edges": len(edges or [])})
    n = max(len(caches), 1)
    return {
        "count": len(caches),
        "complete": sum(1 for r in caches if r.get("status") in ("SUCCESS", "REUSED")),
        "empty_rate": empty / n,
        "nonempty_accepted": nonempty,
        "accepted_total": accepted_total,
        "raw_total": raw_total,
        "prior_mechanism_evaluable": evaluable,
        "natural_nonempty_case": evaluable_case,
        "rows": rows,
    }


def freeze_resolved_task(root, task_id, timing, split_rel):
    role = TASK_ROLE[task_id]
    if task_id == "T_A":
        instruction = "Place both the red target cube and the yellow second cube into the openable blue container. Opening the container is a shared prerequisite. The green buffer may be used if needed."
        goals = [
            {"predicate": "Inside", "arguments": ["target", "container"], "sign": 1},
            {"predicate": "Inside", "arguments": ["second_object", "container"], "sign": 1},
        ]
        structure = "shared_prerequisite_multi_object"
    else:
        instruction = "Place the red target cube into the openable blue container. The cyan interferer starts between the target and the container and must be relocated with PICK+PLACE_BUFFER; moving it is not itself the goal."
        goals = [{"predicate": "Inside", "arguments": ["target", "container"], "sign": 1}]
        structure = "intermediate_relocation"
    doc = {
        "task_id": task_id,
        "task_version": "cp-disr-v2.1-stage-2a-p0",
        "status": "RUNTIME_BOUND_STAGE_2A_NOT_STARTED",
        "platform_ref": "LIBERO_CP_DISR_CLEAN@8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "typed_entities": {
            "target": {"type": "object"},
            role: {"type": "object"},
            "container": {"type": "container"},
            "buffer": {"type": "buffer"},
        },
        "instruction": instruction,
        "signed_goals": goals,
        "required_skill_schemas": ["OPEN", "PICK", "PLACE", "PLACE_BUFFER"],
        "grounded_action_ids": grounded_ids(task_id),
        "move_decision": "REJECTED_FOR_ALL_STAGE_2A_TASKS",
        "move_alternative": ["PICK", "PLACE_BUFFER"],
        "structural_signature": structure,
        "task_evaluator_ref": "src/cp_disr/platforms/libero/task_evaluator.py",
        "controller_ref": "src/cp_disr/platforms/libero/skill_executor.py",
        "verifier_ref": "src/cp_disr/platforms/libero/verifier.py",
        "contract_registry_ref": "configs/runtime/stage_2a_contract_registry.yaml",
        "split_ref": split_rel,
        "deadline_seconds": timing["task_deadline_seconds"],
        "reference_skill_seconds": timing["reference_skill_seconds"],
        "stage_2a_runtime_ready": False,
    }
    path = root / f"configs/tasks/resolved/{task_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def freeze_stage_2a_manifest(root, timings, splits):
    factory_path = root / "src/cp_disr/platforms/libero/runtime_factory.py"
    d0 = _read(root / "experiments/manifests/runtime_manifest.yaml")
    rt = {
        "manifest_status": "STAGE_2A_RUNTIME_BOUND_TRAINING_NOT_STARTED",
        "execution_host_identity": d0.get("execution_host_identity"),
        "runtime": dict(d0.get("runtime") or {}),
        "runtime_factory": {
            "module": "cp_disr.platforms.libero.runtime_factory",
            "factory": "create_stage_2a_runtime",
            "source_path": str(factory_path),
            "sha256": sha_file(factory_path),
        },
        "vlm_runtime": d0.get("vlm_runtime"),
        "note": "Separate Stage 2A runtime manifest. D0 Stage 1A runtime_manifest.yaml is not modified.",
    }
    rt["runtime"]["environment_version"] = "stage-2a-p0"
    rt["runtime"]["task_assets"] = {
        "T_A": "src/cp_disr/platforms/libero/d0_env.py",
        "T_C": "src/cp_disr/platforms/libero/d0_env.py",
        "license": "assets/cp_disr/LICENSE",
        "owned_by": "cp_disr_project",
    }
    rt["runtime"]["skill_timeouts"] = {tid: timings[tid]["skill_timeouts"] for tid in ("T_A", "T_C")}
    rt["runtime"]["task_deadlines"] = {tid: timings[tid]["task_deadline_seconds"] for tid in ("T_A", "T_C")}
    rt["runtime"]["reference_skill_seconds_by_task"] = {tid: timings[tid]["reference_skill_seconds"] for tid in ("T_A", "T_C")}
    rt["runtime"]["task_splits"] = {"T_A": "configs/splits/T_A_stage_2a.json", "T_C": "configs/splits/T_C_stage_2a.json"}
    rt["runtime"]["stage_2a_contract_path"] = "configs/runtime/stage_2a_contract_registry.yaml"
    rt["runtime"]["task_evaluator_version"] = {
        "T_A": "cp-disr-ta-task-evaluator-v1",
        "T_C": "cp-disr-tc-task-evaluator-v1",
    }
    path = root / "experiments/manifests/stage_2a_runtime_manifest.yaml"
    path.write_text(yaml.safe_dump(rt, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path, rt

def historical_fingerprints(root: Path):
    return {
        "vlm_cache_train": fingerprint_tree(root / "experiments/vlm_cache/train"),
        "vlm_cache_dev": fingerprint_tree(root / "experiments/vlm_cache/dev"),
        "stage_0c": fingerprint_tree(root / "experiments/part_0_validation/stage_0c"),
        "stage_1a": fingerprint_tree(root / "experiments/part_1_smoke/stage_1a", exclude_parts=("errata",)),
        "stage_1a_p0": fingerprint_tree(root / "experiments/part_1_smoke/stage_1a_p0"),
        "d0_runtime_manifest": sha_bytes((root / "experiments/manifests/runtime_manifest.yaml").read_bytes()) if (root / "experiments/manifests/runtime_manifest.yaml").exists() else "MISSING",
        "d0_split": sha_bytes((root / "configs/splits/D0_stage_1a.json").read_bytes()) if (root / "configs/splits/D0_stage_1a.json").exists() else "MISSING",
    }


def summarize_ctrl(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["skill_id"]].append(r)
    summary = {}
    blocked = []
    for sid, rs in by.items():
        n = sum(1 for r in rs if r["verified_success"])
        wrong = sum(1 for r in rs if r["wrong_object"])
        nan = sum(1 for r in rs if r["nan_blocked"])
        summary[sid] = {"verified_success": n, "attempts": len(rs), "wrong_object": wrong, "nan": nan, "pass": n >= 4 and wrong == 0 and nan == 0}
        if not summary[sid]["pass"]:
            blocked.append(sid)
    return summary, blocked


def write_cache_report(path, task_id, stats):
    lines = [
        f"# {task_id} Stage 2A cache report",
        "",
        f"- count: {stats['count']}",
        f"- complete: {stats['complete']}",
        f"- empty_rate: {stats['empty_rate']}",
        f"- nonempty_accepted: {stats['nonempty_accepted']}",
        f"- prior_mechanism_evaluable: {stats['prior_mechanism_evaluable']}",
        f"- natural_nonempty_case: {stats['natural_nonempty_case']}",
        "",
        "Empty prior is legal. No semantic retry and no scene resampling were performed.",
        "",
    ]
    write_text(path, "\n".join(lines))


def run_all(root: Path, gpu=0):
    started = time.time()
    root = Path(root)
    out = root / P0
    out.mkdir(parents=True, exist_ok=True)
    notes = []
    status = "PASS"
    before = historical_fingerprints(root)
    write_json(out / "historical_fingerprint_before.json", before)

    write_text(out / "action_registry_audit.md", "\n".join([
        "# Action registry audit",
        "",
        "- Formal Stage 2A schemas: OPEN, PICK, PLACE, PLACE_BUFFER.",
        "- MOVE is rejected for all Stage 2A tasks and is not a PLACE_BUFFER alias.",
        "- T_C intermediate relocation is PICK(interferer) + PLACE_BUFFER(interferer, buffer).",
        "- Stage 0C T_A/T_C caches are historical and not reused.",
        "",
    ]))

    splits = {}
    for task_id in ("T_A", "T_C"):
        splits[task_id] = build_task_split(task_id)
        write_json(root / f"configs/splits/{task_id}_stage_2a.json", splits[task_id])
        write_json(out / f"split_manifest_{task_id}.json", splits[task_id])

    ctrl_rows = []
    timings = {}
    ver_rows = []
    ev_rows = []
    leak_rows = []
    timing_path = out / "timing_calibration.json"
    reuse_ctrl = timing_path.exists() and (out / "controller_qualification.csv").exists()
    if reuse_ctrl:
        timings = json.loads(timing_path.read_text(encoding="utf-8"))
        _log("reusing controller timing from previous P0 attempt")
    for task_id in ("T_A", "T_C"):
        if reuse_ctrl and task_id in timings:
            prev = json.loads((out / f"controller_qualification_{task_id}.json").read_text(encoding="utf-8"))
            if prev.get("blocked"):
                reuse_ctrl = False
        if reuse_ctrl and task_id in timings:
            _log("skip controller qualification " + task_id)
            rows = []
            durations = {}
        else:
            _log("qualify controllers " + task_id)
            rows, durations = qualify_controllers(task_id, gpu=gpu)
        if rows:
            summary, blocked = summarize_ctrl(rows)
            write_json(out / f"controller_qualification_{task_id}.json", {"summary": summary, "blocked": blocked})
            ctrl_rows.extend(rows)
            if blocked:
                status = "NEEDS_RERUN"
                notes.append(f"{task_id} controller qualification below 4/5: " + ",".join(blocked))
            else:
                timings[task_id] = calibrate_timing(task_id, durations)
        else:
            blocked = []
        _log("qualify verifier " + task_id)
        vrows = qualify_verifier(task_id, gpu=gpu)
        ver_rows.extend(vrows)
        if any(r["leaked_hidden_into_verifier"] for r in vrows):
            status = "NEEDS_RERUN"
            notes.append(task_id + " verifier leaked hidden state")
        _log("evaluator tests " + task_id)
        erows = evaluator_unit_checks(task_id, gpu=gpu)
        ev_rows.extend(erows)
        named = [r for r in erows if isinstance(r.get("pass"), bool)]
        if not named or not all(r.get("pass") for r in named):
            status = "NEEDS_RERUN"
            notes.append(task_id + " evaluator tests failed")
        leak = hidden_truth_leakage_check(task_id, gpu=gpu)
        leak_rows.append(leak)
        if leak["leaked"]:
            status = "NEEDS_RERUN"
            notes.append(task_id + " hidden-state leakage")

    if ctrl_rows:
        write_csv(out / "controller_qualification.csv", ctrl_rows)
    write_csv(out / "verifier_qualification.csv", [{"task_id": r["task_id"], "scenario": r["scenario"], "leaked": r["leaked_hidden_into_verifier"]} for r in ver_rows])
    write_json(out / "evaluator_tests.json", ev_rows)
    timing_rows = []
    for tid, timing in timings.items():
        timing_rows.append({
            "task_id": tid,
            "deadline_seconds": timing["task_deadline_seconds"],
            "d_ref": timing["reference_skill_seconds"],
            "OPEN": timing["skill_timeouts"]["OPEN"],
            "PICK": timing["skill_timeouts"]["PICK"],
            "PLACE": timing["skill_timeouts"]["PLACE"],
            "PLACE_BUFFER": timing["skill_timeouts"]["PLACE_BUFFER"],
            "copied_from_d0": False,
        })
    write_csv(out / "timing_calibration.csv", timing_rows)
    write_json(out / "timing_calibration.json", timings)

    interference = None
    try:
        interference = tc_interference_probe(gpu=gpu)
        write_json(out / "tc_interference_probe.json", interference)
    except Exception as exc:  # noqa: BLE001
        notes.append("T_C interference probe failed: " + type(exc).__name__)
        write_json(out / "tc_interference_probe.json", {"error": type(exc).__name__, "detail": str(exc)[:300]})

    if "T_A" in timings and "T_C" in timings:
        for task_id in ("T_A", "T_C"):
            freeze_resolved_task(root, task_id, timings[task_id], f"configs/splits/{task_id}_stage_2a.json")
        freeze_stage_2a_manifest(root, timings, splits)
    else:
        status = "NEEDS_RERUN" if status == "PASS" else status
        notes.append("timing missing; cannot freeze runtime")

    caches = {"T_A": [], "T_C": []}
    cache_stats_by_task = {}
    scripted = {"T_A": [], "T_C": []}
    dry_reports = []
    if "T_A" in timings and "T_C" in timings:
        for task_id in ("T_A", "T_C"):
            _log("capture RGB " + task_id)
            mapping = capture_split_images(root, task_id, splits[task_id], gpu=gpu)
            _log("generate caches " + task_id)
            try:
                caches[task_id] = generate_task_caches(root, task_id, mapping, gpu=gpu)
            except Exception as exc:  # noqa: BLE001
                status = "BLOCKED"
                notes.append(task_id + " cache generation failed: " + type(exc).__name__ + ": " + str(exc)[:200])
                caches[task_id] = [{"status": "ERROR", "error": type(exc).__name__, "detail": str(exc)[:300]}]
                break
            cmap = {r.get("case_id"): r for r in caches[task_id] if r.get("case_id")}
            for row in splits[task_id]["train"] + splits[task_id]["dev"]:
                rec = cmap.get(row["case_id"])
                if rec:
                    row["cache_dir"] = rec.get("cache_dir")
                    row["cache_key"] = rec.get("cache_key")
                    row["cache_status"] = rec.get("status")
            write_json(root / f"configs/splits/{task_id}_stage_2a.json", splits[task_id])
            write_json(out / f"split_manifest_{task_id}.json", splits[task_id])
            success_rows = [r for r in caches[task_id] if r.get("cache_dir") and r.get("status") in ("SUCCESS", "REUSED")]
            stats = cache_stats(root, success_rows)
            cache_stats_by_task[task_id] = stats
            write_cache_report(out / f"cache_report_{task_id}.md", task_id, stats)
            write_json(out / f"cache_generation_{task_id}.json", caches[task_id])
            success_n = len([r for r in success_rows if (root / r["cache_dir"] / "COMPLETE").exists()])
            if success_n != 84:
                status = "NEEDS_RERUN" if status == "PASS" else status
                notes.append(task_id + " cache set incomplete")

        if status != "BLOCKED" and "T_A" in timings:
            freeze_stage_2a_manifest(root, timings, splits)
            man_path = root / "experiments/manifests/stage_2a_runtime_manifest.yaml"
            manifest = _read(man_path)
            for task_id in ("T_A", "T_C"):
                csv_path = out / f"scripted_dev20_{task_id}.csv"
                reused_rows = []
                if csv_path.exists():
                    with csv_path.open(encoding="utf-8", newline="") as handle:
                        reused_rows = list(csv.DictReader(handle))
                    def _truthy(value):
                        return str(value).strip().lower() in ("true", "1", "yes")
                    if len(reused_rows) == 20 and all(_truthy(r.get("success")) for r in reused_rows) and not any(_truthy(r.get("repeat_reward")) or _truthy(r.get("hidden_in_policy")) or _truthy(r.get("controller_exception")) for r in reused_rows):
                        scripted[task_id] = [{**r, "success": True, "repeat_reward": False, "hidden_in_policy": False, "controller_exception": False} for r in reused_rows]
                        _log("reuse scripted dev20 " + task_id)
                        continue
                _log("scripted dev20 " + task_id)
                scripted[task_id] = scripted_dev20(task_id, splits[task_id], timings[task_id]["skill_timeouts"], timings[task_id]["task_deadline_seconds"], gpu=gpu)
                write_csv(out / f"scripted_dev20_{task_id}.csv", scripted[task_id])
                if (not scripted[task_id]) or (not all(r["success"] for r in scripted[task_id])) or any(r["repeat_reward"] or r["hidden_in_policy"] or r["controller_exception"] for r in scripted[task_id]):
                    status = "NEEDS_RERUN" if status == "PASS" else status
                    notes.append(task_id + " scripted dev20 incomplete")
            _log("eight method dry-runs")
            for task_id in ("T_A", "T_C"):
                default_case = splits[task_id]["dev"][0]["case_id"]
                nonempty = (cache_stats_by_task.get(task_id) or {}).get("natural_nonempty_case")
                for method in ("B0", "B1", "B2", "Full"):
                    case_id = default_case
                    if method == "Full" and nonempty:
                        case_id = nonempty
                    try:
                        rep = model_dry_run(root, task_id, method, case_id, manifest, out / "dry_run")
                    except Exception as exc:  # noqa: BLE001
                        rep = {"task_id": task_id, "method": method, "case_id": case_id, "error": type(exc).__name__, "detail": str(exc)[:300], "transition": None, "optimizer_steps": 0}
                        status = "NEEDS_RERUN" if status == "PASS" else status
                        notes.append(f"{task_id} {method} dry-run failed")
                    dry_reports.append(rep)
                    if method == "B2":
                        if rep.get("effective_prior_edge_count") not in (0, None) or (rep.get("diagnostics") or {}).get("dp_abs_max", 0) > 1e-8 or (rep.get("diagnostics") or {}).get("delta_abs_max", 0) > 1e-8:
                            status = "NEEDS_RERUN" if status == "PASS" else status
                            notes.append(f"{task_id} B2 prior/DP/Delta not strictly zero")
                    if not (rep.get("transition") and rep["transition"].get("selected_candidate_id")):
                        status = "NEEDS_RERUN" if status == "PASS" else status
                        notes.append(f"{task_id} {method} dry-run missing real selected action")
            write_json(out / "eight_method_dry_run.json", dry_reports)

    runner_report = validate_runner(root)
    if not runner_report.get("passed"):
        status = "NEEDS_RERUN" if status == "PASS" else status
        notes.append("24-job runner validation failed")

    _log("stage 0B unit tests")
    import subprocess, sys
    rc = subprocess.call([sys.executable, "-m", "pytest", str(root / "tests"), "-m", "pure or torch_runtime", "-q"], cwd=root)
    if rc != 0:
        status = "NEEDS_RERUN" if status == "PASS" else status
        notes.append("Stage 0B pytest failed rc=" + str(rc))

    after = historical_fingerprints(root)
    write_json(out / "historical_fingerprint_after.json", after)
    if after != before:
        status = "BLOCKED"
        notes.append("Stage 0C or Stage 1A historical fingerprint changed")

    ta_ok = bool(scripted["T_A"]) and all(r["success"] for r in scripted["T_A"]) and len(scripted["T_A"]) == 20
    tc_ok = bool(scripted["T_C"]) and all(r["success"] for r in scripted["T_C"]) and len(scripted["T_C"]) == 20
    dry_ok = {m: bool(dry_reports) and all(bool(r.get("transition")) for r in dry_reports if r.get("method") == m) and any(r.get("method") == m for r in dry_reports) for m in ("B0", "B1", "B2", "Full")}
    leakage = any(r.get("leaked") for r in leak_rows)
    def _ok_cache(rows):
        return [r for r in rows if r.get("cache_dir") and r.get("status") in ("SUCCESS", "REUSED")]
    ta_ok_caches = _ok_cache(caches["T_A"])
    tc_ok_caches = _ok_cache(caches["T_C"])
    ta_cache_n = len(ta_ok_caches)
    tc_cache_n = len(tc_ok_caches)
    runtime_ready = (
        status in ("PASS", "PASS_WITH_NOTES")
        and ta_ok and tc_ok
        and ta_cache_n == 84 and tc_cache_n == 84
        and all(dry_ok.values())
        and not leakage
        and runner_report.get("passed")
        and rc == 0
    )
    if runtime_ready and any(not (cache_stats_by_task.get(t) or {}).get("prior_mechanism_evaluable") for t in ("T_A", "T_C")):
        status = "PASS_WITH_NOTES"
        notes.append("one or both tasks lack a natural nonempty prior+patch case; prior mechanism claim is weak/N/A")
    elif runtime_ready:
        status = "PASS"

    readiness = {
        "t_a_runtime_ready": ta_ok and ta_cache_n == 84,
        "t_c_runtime_ready": tc_ok and tc_cache_n == 84,
        "t_a_train_scene_count": 64,
        "t_a_dev_scene_count": 20,
        "t_c_train_scene_count": 64,
        "t_c_dev_scene_count": 20,
        "t_a_train_cache_count": len([r for r in ta_ok_caches if r.get("split") == "train"]),
        "t_a_dev_cache_count": len([r for r in ta_ok_caches if r.get("split") == "dev"]),
        "t_c_train_cache_count": len([r for r in tc_ok_caches if r.get("split") == "train"]),
        "t_c_dev_cache_count": len([r for r in tc_ok_caches if r.get("split") == "dev"]),
        "t_a_scripted_dev_passed": ta_ok,
        "t_c_scripted_dev_passed": tc_ok,
        "b0_dry_run_passed": bool(dry_ok.get("B0")),
        "b1_dry_run_passed": bool(dry_ok.get("B1")),
        "b2_dry_run_passed": bool(dry_ok.get("B2")),
        "full_dry_run_passed": bool(dry_ok.get("Full")),
        "hidden_truth_leakage": bool(leakage),
        "run_registry_count": 24,
        "stage_2a_runtime_ready": bool(runtime_ready),
    }
    write_json(out / "stage_2a_readiness.json", readiness)
    st_path = root / "experiments/stage_status/stage_2a.json"
    st = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {"stage": "2A", "status": "NOT_STARTED"}
    st["status"] = "NOT_STARTED"
    st["readiness"] = readiness
    st["p0_status"] = status
    st["p0_notes"] = notes
    write_json(st_path, st)

    prior_eval = {t: bool((cache_stats_by_task.get(t) or {}).get("prior_mechanism_evaluable")) for t in ("T_A", "T_C")}
    write_text(out / "task_binding_report.md", "\n".join([
        "# Task binding report",
        "",
        "## T_A",
        "Real multi-object shared-prerequisite task. Signed goals: Inside(target, container) AND Inside(second_object, container). Reward only when both first become true.",
        "",
        "## T_C",
        "Intermediate relocation. Signed goal: Inside(target, container) only. Interferer starts on the target-to-container segment at 0.038 m from target. Relocation uses PICK(interferer)+PLACE_BUFFER, not MOVE. Interferer-at-buffer reward is 0.",
        "",
        json.dumps({"interference": interference, "prior_mechanism_evaluable": prior_eval}, indent=2, default=str),
        "",
    ]))
    write_text(out / "eight_method_dry_run_report.md", "# Eight method dry-run\n\n" + json.dumps(dry_reports, indent=2, default=str)[:50000] + "\n")
    write_text(out / "hidden_truth_leakage_report.md", "# Hidden-state leakage\n\n" + json.dumps(leak_rows, indent=2) + "\n")
    summary = [
        "# Stage 2A-P0 summary",
        "",
        f"- p0_status: {status}",
        f"- stage_2A_status: NOT_STARTED",
        f"- notes: {notes}",
        f"- readiness: {json.dumps(readiness)}",
        f"- prior_mechanism_evaluable: {prior_eval}",
        f"- elapsed_seconds: {time.time() - started}",
        f"- optimizer_steps: 0",
        "",
    ]
    write_text(out / "stage_2a_p0_summary.md", "\n".join(summary))
    _log("P0 finished status=%s notes=%s" % (status, notes))
    return {"status": status, "notes": notes, "readiness": readiness, "elapsed_seconds": time.time() - started}


def cmd_stage_2a_p0(root: Path, gpu=0):
    return run_all(root, gpu=gpu)
