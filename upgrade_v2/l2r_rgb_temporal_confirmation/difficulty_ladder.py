from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from upgrade_v2.l2r_forced_drop.intervention import make_force_pulse, run_force_pulse
from upgrade_v2.l2r_forced_drop.physical_reference import evaluate_loss_trace
from upgrade_v2.l2r_forced_drop.protocol import PulseLevel
from upgrade_v2.l2r_forced_drop.simulator import ControlledForcedDropTabletop
from upgrade_v2.l2r_forced_drop.trace_recorder_v2 import snapshot
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec

from .case_registry import CALIBRATION_FAMILIES
from .io_utils import read_json, sha256_file, write_csv, write_json
from .protocol import BASE_DELTA_V, PULSE_DURATION_S, SCALES


def level_id(scale: float) -> str:
    return "D" + str(SCALES.index(scale)) if scale in SCALES else "DMID_" + str(scale).replace(".", "p")


def pulse_for_scale(scale: float) -> PulseLevel:
    return PulseLevel(level_id(scale), tuple(float(scale) * value for value in BASE_DELTA_V))


def _run_instance(root: Path, family: str, family_seed: int, rollout_seed: int,
                  level: PulseLevel) -> dict:
    """R17 calibration with a separate settling interval before verification."""
    root.mkdir(parents=True, exist_ok=False)
    sim = ControlledForcedDropTabletop(
        family_spec(family, "normal_pick_place", family_seed, rollout_seed,
                    probe_variant="default"), rollout_seed)
    sim.perform("approach_object"); sim.perform("close_gripper"); sim.perform("lift")
    trace = []
    for _ in range(10):
        sim.physics_step()
        trace.append(snapshot(sim, step=len(trace), phase="pre_hold_settle", family_id=family,
                              seed=rollout_seed, pre_hold_verified=False))
    verification = []
    anchor = None
    for _ in range(50):
        sim.physics_step()
        row = snapshot(sim, step=len(trace), phase="pre_hold", family_id=family,
                       seed=rollout_seed, pre_hold_verified=False)
        if anchor is None: anchor = np.asarray(row["object_in_gripper_position"], dtype=float)
        row["relative_drift_m"] = float(np.linalg.norm(
            np.asarray(row["object_in_gripper_position"], dtype=float) - anchor))
        trace.append(row); verification.append(row)
    drift = max((row["relative_drift_m"] for row in verification), default=float("inf"))
    verified = bool(len(verification) == 50 and drift <= 0.01 and
                    all(row["weld_active"] and row["numeric_health"]["passed"] for row in verification))
    sim.pre_hold_verified = verified
    force_start_time = None
    if verified:
        object_body = int(sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, "object"))
        gripper_body = int(sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, "gripper"))
        pulse = make_force_pulse(float(sim.model.body_mass[object_body]),
                                 sim.data.xmat[gripper_body].reshape(3, 3), level,
                                 duration_s=PULSE_DURATION_S)

        def event(current, name):
            nonlocal force_start_time
            if name == "force_pulse_started": force_start_time = float(current.data.time)
            phase = "force_pulse_step" if name == "physics_step" else name
            if name in {"post_weld_off_pre_force", "force_pulse_started", "physics_step", "force_pulse_ended"}:
                trace.append(snapshot(current, step=len(trace), phase=phase, family_id=family,
                                      seed=rollout_seed, pre_hold_verified=verified))
        run_force_pulse(sim, pulse, trace=event)
        for _ in range(100):
            sim.physics_step()
            trace.append(snapshot(sim, step=len(trace), phase="observation", family_id=family,
                                  seed=rollout_seed, pre_hold_verified=verified))
    outcome = evaluate_loss_trace(trace, pre_hold_verified=verified, force_start_time=force_start_time)
    numeric = all(row["numeric_health"]["passed"] for row in trace)
    result = {"family_id": family, "family_seed": family_seed, "rollout_seed": rollout_seed,
              "level_id": level.level_id, "execution_valid": bool(verified and numeric),
              "pre_hold_verified": verified, "numeric_health_pass": numeric,
              "physical_loss_confirmed": bool(outcome.get("physical_loss_confirmed")),
              "force_start_time": force_start_time,
              "loss_onset_time_abs": outcome.get("loss_onset_time_abs"),
              "loss_confirmed_time_abs": outcome.get("loss_confirmed_time_abs"),
              "loss_confirmed_delay_from_force_s": outcome.get("loss_confirmed_delay_from_force_s"),
              "outcome": outcome, "trace_rows": len(trace)}
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in trace)
    for name in ("physics_trace.jsonl", "contact_trace.jsonl", "capture_volume_trace.jsonl", "force_trace.jsonl"):
        (root / name).write_text(text, encoding="utf-8")
    write_json(root / "pre_hold_summary.json", {"pre_hold_verified": verified,
                                                  "settling_duration_s": 0.10,
                                                  "verification_duration_s": 0.50,
                                                  "max_relative_drift_m": drift})
    write_json(root / "physical_reference.json", result)
    return result


def choose_ladder(levels: list[dict]) -> dict:
    ordered = sorted(levels, key=lambda row: float(row["scale"]))
    weak_rows = [row for row in ordered if int(row["loss_count"]) == 0]
    strong_rows = [row for row in ordered if int(row["loss_count"]) == 4]
    if not weak_rows or not strong_rows:
        raise RuntimeError("DIFFICULTY_LADDER_UNRESOLVED")
    weak = max(weak_rows, key=lambda row: float(row["scale"]))
    strong = min(strong_rows, key=lambda row: float(row["scale"]))
    if float(weak["scale"]) >= float(strong["scale"]):
        raise RuntimeError("DIFFICULTY_LADDER_NOT_ORDERED")
    between = [row for row in ordered if float(weak["scale"]) < float(row["scale"]) < float(strong["scale"])]
    if not between:
        return {"weak": weak, "medium": None, "strong": strong,
                "midpoint_required": (float(weak["scale"]) + float(strong["scale"])) / 2.0}
    medium = min(between, key=lambda row: (abs(int(row["loss_count"]) - 2), float(row["scale"])))
    return {"weak": weak, "medium": medium, "strong": strong, "midpoint_required": None}


def _run_level(output_root: Path, scale: float, ordinal: int) -> tuple[dict, list[dict]]:
    pulse = pulse_for_scale(scale)
    rows = []
    for family, family_seed, rollout_seed_base in CALIBRATION_FAMILIES:
        rollout_seed = rollout_seed_base + ordinal
        root = output_root / pulse.level_id / family
        result = _run_instance(root, family, family_seed, rollout_seed, pulse)
        rows.append(result)
    delays = [float(row["loss_confirmed_delay_from_force_s"]) for row in rows
              if row.get("loss_confirmed_delay_from_force_s") is not None]
    summary = {
        "level_id": pulse.level_id,
        "scale": float(scale),
        "target_delta_v_local_mps": list(pulse.target_delta_v_local_mps),
        "duration_s": PULSE_DURATION_S,
        "rollouts": len(rows),
        "loss_count": sum(bool(row.get("physical_loss_confirmed")) for row in rows),
        "numeric_pass_count": sum(bool(row.get("numeric_health_pass")) for row in rows),
        "prehold_count": sum(bool(row.get("pre_hold_verified")) for row in rows),
        "median_loss_delay_s": statistics.median(delays) if delays else None,
    }
    return summary, rows


def run_difficulty_calibration(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    level_rows: list[dict] = []
    rollout_rows: list[dict] = []
    for ordinal, scale in enumerate(SCALES):
        level, rows = _run_level(output_root, scale, ordinal)
        level_rows.append(level)
        rollout_rows.extend(rows)
    selection = choose_ladder(level_rows)
    if selection["midpoint_required"] is not None:
        midpoint = float(selection["midpoint_required"])
        level, rows = _run_level(output_root, midpoint, len(SCALES))
        level_rows.append(level)
        rollout_rows.extend(rows)
        selection = choose_ladder(level_rows)
        if selection["medium"] is None:
            selection["medium"] = level
    selected = {
        "schema": "l2rar2_r17_selected_difficulty_ladder_v1",
        "status": "DIFFICULTY_LADDER_FROZEN",
        "base_delta_v_local_mps": list(BASE_DELTA_V),
        "duration_s": PULSE_DURATION_S,
        "parameter_search_allowed": False,
        "weak": selection["weak"], "medium": selection["medium"], "strong": selection["strong"],
        "midpoint_extensions_used": sum(str(row["level_id"]).startswith("DMID") for row in level_rows),
    }
    write_csv(output_root / "per_level_results.csv", level_rows)
    compact = [{key: row.get(key) for key in (
        "family_id", "family_seed", "rollout_seed", "level_id", "execution_valid",
        "pre_hold_verified", "numeric_health_pass", "physical_loss_confirmed",
        "loss_onset_time_abs", "loss_confirmed_time_abs", "loss_confirmed_delay_from_force_s")}
        for row in rollout_rows]
    write_csv(output_root / "per_rollout_reference.csv", compact)
    write_json(output_root / "selected_difficulty_ladder.json", selected)
    traces = []
    for path in sorted(output_root.rglob("physics_trace.jsonl")):
        traces.append({"path": str(path.relative_to(output_root)), "sha256": sha256_file(path),
                       "bytes": path.stat().st_size})
    write_json(output_root / "calibration_trace_manifest.json",
               {"schema": "l2rar2_r17_calibration_trace_manifest_v1", "traces": traces})
    write_json(output_root / "numeric_health.json", {
        "schema": "l2rar2_r17_calibration_numeric_health_v1",
        "passed": all(bool(row.get("numeric_health_pass")) for row in rollout_rows),
        "rollouts": len(rollout_rows),
    })
    return selected


def validate_difficulty(calibration_root: Path) -> dict:
    selected = read_json(calibration_root / "selected_difficulty_ladder.json")
    errors = []
    if selected.get("status") != "DIFFICULTY_LADDER_FROZEN": errors.append("status")
    scales = [float(selected[name]["scale"]) for name in ("weak", "medium", "strong")]
    if not scales[0] < scales[1] < scales[2]: errors.append("ordering")
    if int(selected["weak"]["loss_count"]) != 0: errors.append("weak_not_0_of_4")
    if int(selected["strong"]["loss_count"]) != 4: errors.append("strong_not_4_of_4")
    if int(selected.get("midpoint_extensions_used", 0)) > 1: errors.append("midpoint_limit")
    return {"schema": "l2rar2_r17_difficulty_validation_v1",
            "status": "PASS" if not errors else "FAIL", "errors": errors, "scales": scales}
