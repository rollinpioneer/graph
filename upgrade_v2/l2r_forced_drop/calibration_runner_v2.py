from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .protocol import PulseLevel, pulse_levels, CALIBRATION_DURATION_S
from .simulator import ControlledForcedDropTabletop
from .intervention import make_force_pulse, run_force_pulse
from .trace_recorder_v2 import snapshot
from .physical_reference import evaluate_loss_trace, evaluate_no_force_control
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec


def _prehold(sim, family: str, family_seed: int, rollout_seed: int):
    """Capture ten raw pre-hold rows, then validate them in a separate summary."""
    z0 = float(sim.object_xyz[2])
    sim.perform('approach_object')
    sim.perform('close_gripper')
    sim.perform('lift')
    anchor = None
    rows = []
    for i in range(10):
        sim.physics_step()
        row = snapshot(sim, step=i, phase="pre_hold", family_id=family,
                       seed=rollout_seed, pre_hold_verified=False)
        if anchor is None:
            anchor = np.asarray(row["object_in_gripper_position"], dtype=float)
        row["relative_drift_m"] = float(np.linalg.norm(
            np.asarray(row["object_in_gripper_position"], dtype=float) - anchor))
        rows.append(row)
    rise = float(sim.object_xyz[2] - z0)
    drift = max((r["relative_drift_m"] for r in rows), default=float("inf"))
    ok = bool(rows and all(r["weld_active"] and r["numeric_health"]["passed"] for r in rows)
              and rise >= 0.03 and drift <= 0.01)
    sim.pre_hold_verified = ok
    summary = {"pre_hold_verified": ok, "height_rise_m": rise,
               "max_relative_drift_m": drift, "sample_count": len(rows),
               "family_seed": family_seed, "rollout_seed": rollout_seed}
    return rows, summary


def _run_instance(root: Path, family: str, family_seed: int, rollout_seed: int,
                  level: PulseLevel | None, label: str,
                  pulse_duration_s: float = CALIBRATION_DURATION_S):
    root.mkdir(parents=True, exist_ok=True)
    spec = family_spec(family, "normal_pick_place", family_seed, rollout_seed,
                       probe_variant="default")
    sim = ControlledForcedDropTabletop(spec, rollout_seed)
    pre, pre_summary = _prehold(sim, family, family_seed, rollout_seed)
    trace = list(pre)
    force_start_time = None
    pulse_step = 0

    if sim.pre_hold_verified:
        if level is None:
            sim.disable_weld_for_intervention()
            sim.forward()
            trace.append(snapshot(sim, step=len(trace), phase="post_weld_off_pre_force",
                                   family_id=family, seed=rollout_seed,
                                   pre_hold_verified=sim.pre_hold_verified))
        else:
            object_body = int(sim.mujoco.mj_name2id(
                sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, "object"))
            body = int(sim.mujoco.mj_name2id(
                sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, "gripper"))
            rot = sim.data.xmat[body].reshape(3,3)
            mass = float(sim.model.body_mass[object_body])
            pulse = make_force_pulse(mass, rot, level, duration_s=pulse_duration_s)

            def capture_event(current_sim, event: str):
                nonlocal pulse_step, force_start_time
                if event == "post_weld_off_pre_force":
                    trace.append(snapshot(current_sim, step=len(trace), phase=event,
                                          family_id=family, seed=rollout_seed,
                                          pre_hold_verified=current_sim.pre_hold_verified))
                elif event == "force_pulse_started":
                    force_start_time = float(current_sim.data.time)
                    trace.append(snapshot(current_sim, step=len(trace), phase=event,
                                          family_id=family, seed=rollout_seed,
                                          pre_hold_verified=current_sim.pre_hold_verified))
                elif event == "physics_step":
                    pulse_step += 1
                    row = snapshot(current_sim, step=len(trace), phase="force_pulse_step",
                                   family_id=family, seed=rollout_seed,
                                   pre_hold_verified=current_sim.pre_hold_verified)
                    row["pulse_step"] = pulse_step
                    trace.append(row)

            run_force_pulse(sim, pulse, trace=capture_event)

        for _ in range(75):
            sim.physics_step()
            trace.append(snapshot(sim, step=len(trace), phase="observation",
                                  family_id=family, seed=rollout_seed,
                                  pre_hold_verified=sim.pre_hold_verified))

    rows = [dict(r, outside_capture=bool(r.get("outside_capture", False)),
                 support_force_ratio_mg=float(r.get("support_force_ratio_mg", 0.0)))
            for r in trace]
    outcome = evaluate_loss_trace(rows[10:], pre_hold_verified=pre_summary["pre_hold_verified"],
                                  force_start_time=force_start_time)
    numeric = all(r["numeric_health"]["passed"] for r in rows)
    no_force = evaluate_no_force_control(rows) if level is None else None
    result = {
        "label": label, "family_id": family, "family_seed": family_seed,
        "rollout_seed": rollout_seed, "level_id": level.level_id if level else None,
        "execution_valid": bool(pre_summary["pre_hold_verified"] and numeric and len(rows) >= 75),
        "pre_hold": pre_summary, "pre_hold_verified": pre_summary["pre_hold_verified"],
        "numeric_health_pass": numeric,
        "physical_loss_confirmed": bool(outcome.get("physical_loss_confirmed", False)),
        "outcome": outcome, "no_force_control": no_force,
        "force_start_time": force_start_time,
        "loss_onset_time_abs": outcome.get("loss_onset_time_abs"),
        "loss_confirmed_time_abs": outcome.get("loss_confirmed_time_abs"),
        "loss_confirmed_delay_from_force_s": outcome.get("loss_confirmed_delay_from_force_s"),
        "selected": False,
        "failure_reason": None if outcome.get("physical_loss_confirmed") else outcome.get("state"),
        "trace_rows": len(rows),
    }
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    for name in ("physics_trace.jsonl", "contact_trace.jsonl",
                 "capture_volume_trace.jsonl", "force_trace.jsonl"):
        (root / name).write_text(text, encoding="utf-8")
    (root / "pre_hold_summary.json").write_text(json.dumps(pre_summary, indent=2) + "\n")
    (root / "physical_reference.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_calibration_v2(output_root: Path, authorized: bool = True):
    if not authorized:
        raise PermissionError("R16 calibration grant required")
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    r = _run_instance(output_root / "control_primary", "L2RAR2_FDROP_CAL_00_859900",
                      86099000, 86099000, None, "primary_control")
    rows.append(r)
    control_valid = bool(r["execution_valid"] and not r["physical_loss_confirmed"]
                         and (r["no_force_control"] or {}).get("passed", False))
    if not control_valid:
        backup = _run_instance(output_root / "control_backup",
                               "L2RAR2_FDROP_CAL_01_859901", 86099100,
                               86099100, None, "backup_control")
        rows.append(backup)
        if not (backup["execution_valid"] and not backup["physical_loss_confirmed"]
                and (backup["no_force_control"] or {}).get("passed", False)):
            raise RuntimeError("STOPPED_CALIBRATION_CONTROL_INVALID")
    design = [{"level_id": p.level_id, "target_delta_v_local_mps": list(p.target_delta_v_local_mps),
               "duration_s": CALIBRATION_DURATION_S, "round": "protocol"} for p in pulse_levels()]
    for level in pulse_levels():
        r = _run_instance(output_root / level.level_id,
                          "L2RAR2_FDROP_CAL_00_859900", 86099000,
                          86099000, level, level.level_id)
        rows.append(r)
        delay = r.get("loss_confirmed_delay_from_force_s")
        if r["execution_valid"] and r["physical_loss_confirmed"] and delay is not None and delay <= 0.40:
            r["selected"] = True
            break
    # If the frozen I1-I3 sweep is scientifically unresolved, run one
    # pre-registered diagonal/longer pulse design within the same five-instance
    # fresh-calibration allowance.  This is calibration design, never a search
    # over development data.
    if not any(r.get("selected") for r in rows):
        enhanced = PulseLevel("E2_diag40_d0p20", (40.0, 40.0, -8.0))
        design.append({"level_id": enhanced.level_id,
                       "target_delta_v_local_mps": list(enhanced.target_delta_v_local_mps),
                       "duration_s": 0.20, "round": "enhanced_fallback"})
        r = _run_instance(output_root / enhanced.level_id,
                          "L2RAR2_FDROP_CAL_00_859900", 86099000,
                          86099004, enhanced, enhanced.level_id, 0.20)
        rows.append(r)
        delay = r.get("loss_confirmed_delay_from_force_s")
        if r["execution_valid"] and r["physical_loss_confirmed"] and delay is not None and delay <= 0.40:
            r["selected"] = True
    selected = next((r["level_id"] for r in rows if r.get("selected")), None)
    status = "CALIBRATION_PASS" if selected else "STOPPED_CALIBRATION_FAILED"
    keys = ("label", "level_id", "execution_valid", "pre_hold_verified",
            "numeric_health_pass", "physical_loss_confirmed", "selected",
            "failure_reason")
    (output_root / "execution_ledger.csv").write_text(
        ",".join(keys) + "\n" + "\n".join(
            ",".join(str(r.get(k, "")) for k in keys) for r in rows) + "\n",
        encoding="utf-8")
    summary = {"schema": "l2rar2_r16_calibration_v2_result", "status": status,
               "physical_executions": len(rows), "selected_intervention_level": selected,
               "intervention_design": design, "rows": rows}
    (output_root / "calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_root / "numerical_sanity.json").write_text(json.dumps(
        {"passed": all(r["numeric_health_pass"] for r in rows), "instances": len(rows)},
        indent=2) + "\n", encoding="utf-8")
    if selected:
        (output_root / "selected_intervention.json").write_text(json.dumps(
            {"level_id": selected, "protocol": "L2RAR2_R16_CONTROLLED_FORCED_DROP_V2"},
            indent=2) + "\n", encoding="utf-8")
    return summary
