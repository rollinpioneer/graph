#!/usr/bin/env python3
"""No-learning production executor traces for clock recovery. Not test-ID eval."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
WALL_BUDGET_S = 90.0
STEP_BUDGET = 700
SKILL_TIMEOUT = 8.0


def load_row(split_rel, case_id):
    doc = json.loads((ROOT / split_rel).read_text(encoding="utf-8"))
    for split in ("train", "dev"):
        for row in doc.get(split) or []:
            if row.get("case_id") == case_id:
                return row, split
    raise KeyError(case_id)


def spec_from_row(row, task_id):
    from cp_disr.platforms.libero.d0_env import CaseSpec
    return CaseSpec(
        case_id=row["case_id"],
        split=row.get("split") or "train",
        seed=int(row["seed"]),
        target_xy=tuple(row["target_xy"]),
        second_xy=tuple(row["second_xy"]),
        container_xy=tuple(row["container_xy"]),
        buffer_xy=tuple(row["buffer_xy"]),
        lid_closed=bool(row.get("lid_closed", True)),
        task_id=task_id,
        second_role=str(row.get("second_role") or ("interferer" if task_id == "T_C" else "second_object")),
        deadline=float(row.get("deadline") or 90.0),
    )


def wrap_env(env):
    orig = env.step_osc
    state = {"steps": 0, "t0": time.monotonic()}

    def guarded(dpos, dgrip, n=1):
        if time.monotonic() - state["t0"] > WALL_BUDGET_S:
            raise RuntimeError("DIAGNOSTIC_WALL_BUDGET")
        state["steps"] += int(n)
        if state["steps"] > STEP_BUDGET:
            raise RuntimeError("DIAGNOSTIC_CONTROL_STEP_BUDGET")
        return orig(dpos, dgrip, n)

    env.step_osc = guarded
    return state


def run_trace(task_id, case_id, split_rel, sequence, inject_missing_lid=False, label="natural"):
    from cp_disr.platforms.libero.d0_env import make_env
    from cp_disr.platforms.libero.clock import DurationProvider
    from cp_disr.platforms.libero.safety import SafetyManager
    from cp_disr.platforms.libero.skill_executor import SkillExecutor
    from cp_disr.platforms.libero.perception import PerceptionAdapter

    row, split = load_row(split_rel, case_id)
    spec = spec_from_row(row, task_id)
    env = make_env(spec, gpu=0)
    rec = {
        "label": label,
        "task_id": task_id,
        "case_id": case_id,
        "split": split,
        "inject_missing_lid": bool(inject_missing_lid),
        "replay_of_historical_memory": False,
        "sequence": list(sequence),
        "events": [],
        "diagnostic_abort": False,
        "ok_for_recovery_gate": False,
    }
    try:
        env.reset()
        if hasattr(env, "_apply_case_poses"):
            env._apply_case_poses()
        if hasattr(env, "_open_gripper_reset"):
            env._open_gripper_reset()
        clock = DurationProvider(env)
        safety = SafetyManager(env)
        perception = PerceptionAdapter(env)
        raw_refresh = lambda e=env, p=perception: p.infer(e.public_observation())
        if inject_missing_lid:
            def injected_refresh(raw=raw_refresh, e=env):
                raw()
                perc = dict(e.last_perception or {})
                perc.pop("lid", None)
                e.last_perception = perc
                return perc
            env.refresh_perception = injected_refresh
            rec["injection"] = "refresh_wrapper_drops_lid_blob_keep_images_and_cache_unmodified"
        else:
            env.refresh_perception = raw_refresh
        guard = wrap_env(env)
        ex = SkillExecutor(env, safety, clock)
        env.refresh_perception()
        rec["natural_lid_in_perception_before_inject"] = True
        rec["lid_in_perception_at_skill_start"] = "lid" in (env.last_perception or {})
        rec["t0"] = clock.now_seconds()
        for cid in sequence:
            t1 = float(env.sim.data.time)
            n1 = int(guard["steps"])
            out = None
            err = None
            try:
                out = ex.execute(cid, SKILL_TIMEOUT)
            except Exception as exc:
                err = "%s:%s" % (type(exc).__name__, str(exc).splitlines()[0][:160])
                if "DIAGNOSTIC" in str(exc):
                    rec["diagnostic_abort"] = True
                    rec["diagnostic_reason"] = err
            ev = {
                "candidate_id": cid,
                "raw_sim_start": t1,
                "raw_sim_end": float(env.sim.data.time),
                "control_steps": int(guard["steps"]) - n1,
                "executor": out,
                "error": err,
                "elapsed": clock.now_seconds(),
            }
            if out:
                ev["clock_duration"] = float(out["end_seconds"]) - float(out["start_seconds"])
                ev["duration_matches_sim"] = abs(float(out["sim_duration"]) - (float(env.sim.data.time) - t1)) < 1e-9
                ev["zero_physics"] = int(out["steps"]) == 0 and float(out["sim_duration"]) <= 0.0
            rec["events"].append(ev)
            if rec["diagnostic_abort"]:
                break
        rec["t_end"] = clock.now_seconds()
        rec["control_steps_total"] = int(guard["steps"])
        rec["gripper_qpos_final"] = np.asarray(env.public_observation()["gripper_qpos"], dtype=float).tolist()
        exits = [e["executor"].get("controller_exit") for e in rec["events"] if e.get("executor")]
        rec["controller_exits"] = exits
        rec["ok_for_recovery_gate"] = (not rec["diagnostic_abort"]) and (not any(str(x).startswith("INTERRUPT_CONTROLLER_EXCEPTION:RuntimeError") for x in exits))
        if inject_missing_lid:
            rec["ok_for_recovery_gate"] = (
                not rec["diagnostic_abort"]
                and any(x in {"EXECUTION_FAILED", "CRITICAL_FACT_LOST", "REJECTED_UNSAFE_OR_INVALID"} for x in exits)
                and not any(str(x).startswith("INTERRUPT_CONTROLLER_EXCEPTION:RuntimeError") for x in exits)
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
    return rec


def main():
    out_dir = Path("/tmp/clock_recovery_traces")
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    traces.append(run_trace("T_B", "T_B_train_36", "configs/splits/T_B_stage_2a_v11.json", ["a:OPEN:container:v1"], False, "T_B_train_36_natural_open"))
    traces.append(run_trace("T_B", "T_B_train_36", "configs/splits/T_B_stage_2a_v11.json", ["a:OPEN:container:v1"], True, "T_B_train_36_injected_missing_lid"))
    traces.append(run_trace("T_B", "T_B_train_53", "configs/splits/T_B_stage_2a_v11.json", ["a:OPEN:container:v1"], False, "T_B_train_53_natural_open"))
    traces.append(run_trace("T_C", "T_C_dev_00", "configs/splits/T_C_stage_2a_v11.json", ["a:OPEN:container:v1"], False, "T_C_dev_00_natural_open"))
    path = out_dir / "finite_runtime_traces.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in traces:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    summary = {
        "n_traces": len(traces),
        "labels": [t["label"] for t in traces],
        "ok": [t["label"] for t in traces if t.get("ok_for_recovery_gate")],
        "not_ok": [t["label"] for t in traces if not t.get("ok_for_recovery_gate")],
        "path": str(path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for t in traces:
        print("TRACE", t["label"], "exits", t.get("controller_exits"), "ok", t.get("ok_for_recovery_gate"), "diag", t.get("diagnostic_abort"), "lid", t.get("natural_lid_in_perception"))


if __name__ == "__main__":
    main()
