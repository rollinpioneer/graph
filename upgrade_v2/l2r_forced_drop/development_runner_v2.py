from __future__ import annotations

import json
import csv
from pathlib import Path

from .case_registry import CASES, case_by_id
from .intervention import make_force_pulse, run_force_pulse
from .protocol import PulseLevel, CALIBRATION_DURATION_S, pulse_levels
from .physical_reference import evaluate_loss_trace
from .simulator import ControlledForcedDropTabletop
from .trace_recorder_v2 import snapshot
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec

FAMILIES = (
    ("L2RAR2_FDROP_DEV_00_860000", 860000, 86100000),
    ("L2RAR2_FDROP_DEV_01_860001", 860001, 86100100),
    ("L2RAR2_FDROP_DEV_02_860002", 860002, 86100200),
    ("L2RAR2_FDROP_DEV_03_860003", 860003, 86100300),
)


def _capture(sim, trace, family, case_id, seed, phase, *, pre_hold_verified=False,
             commanded_release=False):
    trace.append(snapshot(sim, step=len(trace), phase=phase, case_id=case_id,
                          family_id=family, seed=seed,
                          pre_hold_verified=pre_hold_verified,
                          commanded_release=commanded_release))


def _prehold(sim, trace, family, case_id, seed, duration_steps=10):
    _capture(sim, trace, family, case_id, seed, "initial_open_no_contact")
    sim.perform("approach_object")
    _capture(sim, trace, family, case_id, seed, "after_approach")
    sim.perform("close_gripper")
    _capture(sim, trace, family, case_id, seed, "after_close")
    sim.perform("lift")
    _capture(sim, trace, family, case_id, seed, "after_lift")
    for _ in range(duration_steps):
        sim.physics_step()
        _capture(sim, trace, family, case_id, seed, "pre_hold")
    verified = bool(all(r["weld_active"] and r["numeric_health"]["passed"] for r in trace[-duration_steps:]))
    for row in trace[-duration_steps:]:
        row["pre_hold_verified"] = verified
    sim.pre_hold_verified = verified
    return verified


def _pulse(sim, trace, family, case_id, seed, selected_level, duration_s=CALIBRATION_DURATION_S):
    object_body = int(sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, "object"))
    body = int(sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, "gripper"))
    pulse = make_force_pulse(float(sim.model.body_mass[object_body]),
                             sim.data.xmat[body].reshape(3, 3), selected_level,
                             duration_s=duration_s)
    force_start_time = None
    def event(current_sim, name):
        nonlocal force_start_time
        if name == "physics_step":
            _capture(current_sim, trace, family, case_id, seed, "force_pulse_step",
                     pre_hold_verified=True)
            trace[-1]["pulse_step"] = sum(r.get("pulse_step", 0) > 0 for r in trace) + 1
        elif name in {"post_weld_off_pre_force", "force_pulse_started"}:
            if name == "force_pulse_started":
                force_start_time = float(current_sim.data.time)
            _capture(current_sim, trace, family, case_id, seed, name,
                     pre_hold_verified=True)
    run_force_pulse(sim, pulse, trace=event)
    return force_start_time


def _run_case(root: Path, family: str, family_seed: int, rollout_seed: int,
              case_id: str, selected_level, duration_s=CALIBRATION_DURATION_S):
    case = case_by_id(case_id)
    online_root, reference_dir = root / "online", root / "reference"
    online_root.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    variant = "brief_hold" if case_id.startswith("F6_") else "default"
    scenario = "missed_grasp_then_retry" if case_id.startswith("F1_") else "normal_pick_place"
    sim = ControlledForcedDropTabletop(
        family_spec(family, scenario, family_seed, rollout_seed,
                    probe_variant=variant), rollout_seed)
    trace = []
    force_start_time = None
    if case_id.startswith("F1_"):
        _capture(sim, trace, family, case_id, rollout_seed, "initial_open_no_contact")
        sim.perform("approach_object")
        _capture(sim, trace, family, case_id, rollout_seed, "after_approach")
        sim.perform("close_gripper")
        _capture(sim, trace, family, case_id, rollout_seed, "after_close")
        for _ in range(10):
            sim.physics_step(); _capture(sim, trace, family, case_id, rollout_seed, "pre_hold")
        # The missed-grasp scenario makes the first close physically fail to attach.
        sim.pre_hold_verified = False
        _capture(sim, trace, family, case_id, rollout_seed, "hold_attempt_end")
        trace[-1]["event"] = "hold_request_without_stable_hold"
        trace[-1]["retry_required"] = True
        result = {"pre_hold_verified": False, "physical_loss_confirmed": False,
                  "resolvable": True, "resolution": "retry_grasp"}
    elif case_id.startswith("F2_"):
        _capture(sim, trace, family, case_id, rollout_seed, "initial_open_no_contact")
        sim.perform("approach_object")
        _capture(sim, trace, family, case_id, rollout_seed, "after_approach")
        for _ in range(10):
            sim.physics_step(); _capture(sim, trace, family, case_id, rollout_seed, "touch_only")
        result = {"pre_hold_verified": False, "physical_loss_confirmed": False,
                  "resolvable": True, "resolution": "touch_only"}
    else:
        verified = _prehold(sim, trace, family, case_id, rollout_seed,
                            10 if not case_id.startswith("F6_") else 10)
        if verified:
            if case_id.startswith("F3_"):
                pass
            elif case_id.startswith("F4_"):
                sim.disable_weld_for_intervention(); sim.forward()
                _capture(sim, trace, family, case_id, rollout_seed,
                         "post_weld_off_pre_force", pre_hold_verified=True)
            elif case_id.startswith("F5_") or case_id.startswith("F6_"):
                force_start_time = _pulse(sim, trace, family, case_id, rollout_seed,
                                          selected_level, duration_s)
            elif case_id.startswith("F7_"):
                sim.perform("open_gripper")
                _capture(sim, trace, family, case_id, rollout_seed,
                         "commanded_release", pre_hold_verified=True,
                         commanded_release=True)
            elif case_id.startswith("F8_"):
                sim.perform("transport_to_target")
                _capture(sim, trace, family, case_id, rollout_seed,
                         "transport_before_force", pre_hold_verified=True)
                force_start_time = _pulse(sim, trace, family, case_id, rollout_seed,
                                          selected_level, duration_s)
        for _ in range(90):
            sim.physics_step(); _capture(sim, trace, family, case_id, rollout_seed,
                                         "observation", pre_hold_verified=sim.pre_hold_verified,
                                         commanded_release=case_id.startswith("F7_"))
        result = evaluate_loss_trace(trace, pre_hold_verified=sim.pre_hold_verified,
                                     force_start_time=force_start_time)
        result["force_start_time"] = force_start_time
        result["pre_hold_verified"] = sim.pre_hold_verified
        result["resolvable"] = True

    # Keep candidate-facing data physically separate from the reference.
    online_rows = []
    for index, row in enumerate(trace):
        if "time" not in row:
            online = dict(row)
            online.setdefault("capture_order", index)
            online.setdefault("time", float(trace[index - 1].get("time", 0.0)) if index else 0.0)
        else:
            online = {key: row.get(key) for key in (
                "time", "phase", "action", "object_world_position",
                "gripper_world_position", "gripper_command", "contacts",
                "support_force_ratio_mg", "inside_capture", "commanded_release")}
            online["capture_order"] = index
            online["contact_present"] = bool(row.get("contact_present", bool(row.get("contacts"))))
        online.update({"attempt_id": 1, "attempt_phase": "post_contact_motion",
                       "attempt_active": index < len(trace) - 1,
                       "attempt_end": index == len(trace) - 1,
                       "attempt_end_reason": "release" if case_id.startswith("F7_") else "segment_complete",
                       "requested_effect": case.requested_effect, "context_valid": True})
        online_rows.append(online)
    trace_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in online_rows)
    (online_root / "physics_trace.jsonl").write_text(trace_text, encoding="utf-8")
    reference = {"schema": "l2rar2_r16_physical_reference_v2", "family_id": family,
                 "family_seed": family_seed, "rollout_seed": rollout_seed,
                 "case_id": case_id, "requested_effect": case.requested_effect,
                 "reference_action": case.reference_action, "trace": trace,
                 "trace_complete": len(trace) >= 10,
                 "numeric_health_pass": all(r.get("numeric_health", {}).get("passed", True)
                                            for r in trace if "numeric_health" in r),
                 **result,
                 "commanded_release": case_id.startswith("F7_")}
    (reference_dir / "physical_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (online_root / "status.json").write_text(json.dumps({"family_id": family,
        "case_id": case_id, "trace_complete": reference["trace_complete"],
        "numeric_health_pass": reference["numeric_health_pass"],
        "selected_level": selected_level.level_id if hasattr(selected_level, "level_id") else selected_level}, indent=2) + "\n")
    return reference


def run_development_v2(output_root: Path, *, selected_level: str | None, authorized: bool = True):
    if not authorized:
        raise PermissionError("R16 development grant required")
    if not selected_level:
        raise RuntimeError("CALIBRATION_SELECTION_REQUIRED")
    levels = {p.level_id: (p, CALIBRATION_DURATION_S) for p in pulse_levels()}
    levels["E2_diag40_d0p20"] = (PulseLevel("E2_diag40_d0p20", (40.0, 40.0, -8.0)), 0.20)
    if selected_level not in levels:
        raise ValueError("unknown selected intervention level")
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for family, family_seed, rollout_seed_base in FAMILIES:
        for index, case in enumerate(CASES):
            seed = rollout_seed_base + index
            rows.append(_run_case(output_root / f"{family}_{case.case_id}", family,
                                  family_seed, seed, case.case_id, levels[selected_level][0],
                                  levels[selected_level][1]))
    manifest = [{k: row.get(k) for k in ("family_id", "family_seed", "rollout_seed",
                 "case_id", "trace_complete", "numeric_health_pass",
                 "physical_loss_confirmed", "pre_hold_verified", "commanded_release")}
                for row in rows]
    (output_root / "rollout_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fields = ("family_id", "family_seed", "rollout_seed", "case_id",
              "trace_complete", "numeric_health_pass", "physical_loss_confirmed",
              "pre_hold_verified", "commanded_release")
    for filename in ("rollout_manifest.csv", "per_rollout_status.csv"):
        with (output_root / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(manifest)
    return {"schema": "l2rar2_r16_development_v2_result", "status": "DEVELOPMENT_COMPLETE",
            "rollouts": len(rows), "rows": rows}
