from __future__ import annotations
import json
from pathlib import Path
from .protocol import pulse_levels
def plan_calibration(output_root: Path) -> dict:
    result={"schema":"l2rar2_r16_calibration_plan_v1","levels":[{"id":p.level_id,"target_delta_v_local_mps":list(p.target_delta_v_local_mps)} for p in pulse_levels()],"max_instances":5,"physical_executions":0}
    output_root.mkdir(parents=True,exist_ok=True); (output_root/"calibration_plan.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
def run_calibration(*, output_root: Path, authorized: bool) -> dict:
    if not authorized: raise PermissionError("R16 calibration grant required")
    # MuJoCo is imported only after grant validation.  Each attempted level is
    # a fresh physical model; no state is copied between levels.
    import csv
    import numpy as np
    from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec
    from .simulator import ControlledForcedDropTabletop
    from .intervention import make_force_pulse, run_force_pulse
    from .physical_reference import evaluate_loss_trace

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = None
    for level in pulse_levels():
        sim = ControlledForcedDropTabletop(family_spec("L2RAR2_FDROP_CAL_00_850000", "normal_pick_place", 850000, 85100000, probe_variant="default"), 85100000)
        initial_z = float(sim.object_xyz[2])
        sim.perform("approach_object"); sim.perform("close_gripper"); sim.perform("lift")
        sim.mujoco.mj_forward(sim.model, sim.data)
        rise = float(sim.object_xyz[2] - initial_z)
        sim.pre_hold_verified = sim.verify_pre_hold(height_rise_m=rise, relative_drift_m=0.0, sustain_s=0.10)
        trace = []
        if sim.pre_hold_verified:
            pulse = make_force_pulse(0.18, np.eye(3), level)
            pulse_rows = run_force_pulse(sim, pulse)
            for row in pulse_rows:
                row.update({"pre_hold_verified": True, "outside_capture": False, "support_force_ratio_mg": 1.0, "commanded_release": False})
                trace.append(row)
            for _ in range(75):
                sim.physics_step()
                row = sim.reference_snapshot("post_pulse_observation")
                row.update({"pre_hold_verified": True, "outside_capture": bool(row["object_xyz"][2] < 0.45), "support_force_ratio_mg": 0.0, "commanded_release": False})
                trace.append(row)
        outcome = evaluate_loss_trace(trace)
        passed = bool(sim.pre_hold_verified and outcome["state"] != "NUMERICAL_INVALID" and len(trace) >= 75)
        record = {"level_id": level.level_id, "pre_hold_verified": sim.pre_hold_verified, "trace_rows": len(trace), "outcome": outcome, "passed": passed}
        rows.append(record)
        if passed and selected is None and outcome["physical_loss_confirmed"]:
            selected = level.level_id
            break
    result = {"schema":"l2rar2_r16_calibration_result_v1", "status":"CALIBRATION_PASS" if selected else "STOPPED_CALIBRATION_FAILED", "levels":rows, "selected_intervention_level":selected, "physical_executions":len(rows)}
    (output_root / "calibration_result.json").write_text(json.dumps(result, indent=2)+"\n")
    if selected:
        (output_root / "selected_intervention.json").write_text(json.dumps({"level_id":selected, "source":"calibration_result.json"}, indent=2)+"\n")
    return result
