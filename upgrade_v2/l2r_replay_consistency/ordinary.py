from __future__ import annotations

import copy
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .comparison import (
    compare_action_end_geometry,
    compare_actions,
    compare_callback_order,
    compare_callback_to_return,
    compare_controls,
    compare_events,
    state_health,
)
from .protocol import canonical_hash, sha256


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _contact_summary(sim: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contact_id in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        rows.append({
            "contact_id": contact_id,
            "geom1": sim.mujoco.mj_id2name(sim.model, sim.mujoco.mjtObj.mjOBJ_GEOM, geom1) or str(geom1),
            "geom2": sim.mujoco.mj_id2name(sim.model, sim.mujoco.mjtObj.mjOBJ_GEOM, geom2) or str(geom2),
            "distance_m": float(contact.dist),
        })
    return rows


def _snapshot(sim: Any, *, sampling_point: str, sequence: int, action: str | None) -> dict[str, Any]:
    rng_state = copy.deepcopy(sim.rng.bit_generator.state)
    oracle = sim.oracle_snapshot()
    return {
        "sequence": sequence,
        "sampling_point": sampling_point,
        "action": action,
        "action_index": int(sim.action_index),
        "time": float(sim.data.time),
        "qpos": [float(value) for value in sim.data.qpos],
        "qvel": [float(value) for value in sim.data.qvel],
        "qacc_warmstart": [float(value) for value in sim.data.qacc_warmstart],
        "mocap_pos": np.asarray(sim.data.mocap_pos, dtype=float).tolist(),
        "mocap_quat": np.asarray(sim.data.mocap_quat, dtype=float).tolist(),
        "eq_active": [int(value) for value in sim.data.eq_active],
        "model_eq_data": np.asarray(sim.model.eq_data, dtype=float).tolist(),
        "rng_state": rng_state,
        "rng_state_sha256": canonical_hash(rng_state),
        "contact_pair_summary": _contact_summary(sim),
        "ncon": int(sim.data.ncon),
        "object_xyz": [float(value) for value in sim.object_xyz],
        "gripper_xyz": [float(value) for value in sim.data.mocap_pos[0]],
        "weld_state": int(bool(oracle["weld_state"])),
        "attached": bool(sim.attached),
    }


def run_ordinary(protocol: dict[str, Any], cache_root: Path, output_root: Path) -> dict[str, Any]:
    import cv2
    import mujoco

    from upgrade_v2.l2r_hold_evidence.probe_adapter import control_variant, perform
    from upgrade_v2.l2r_task_context.collector import CASES
    from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec
    from upgrade_v2.visual_refine_l2.repaired_simulator import ATTACH_RELPOSE_VERSION, AttachRelposeDynamicTabletop
    from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer
    from upgrade_v2.visual_refine_l2.vision import detect_frame

    actual_environment = {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    if actual_environment != protocol["required_environment"]:
        raise RuntimeError(f"runtime environment mismatch: {actual_environment!r}")

    case = CASES[protocol["case_id"]]
    if canonical_hash(case["program"]) != protocol["program_sha256"]:
        raise RuntimeError("program hash changed after authorization")
    spec = family_spec(
        protocol["root_family_id"],
        case["scenario"],
        int(protocol["family_seed"]),
        int(protocol["rollout_seed"]),
        probe_variant=case.get("probe_variant", "default"),
    )
    physical = {
        "object_radius": spec.object_radius,
        "friction": spec.friction,
        "camera_jitter": spec.camera_jitter,
        "object_x": spec.object_x,
        "object_y": spec.object_y,
        "target_x": spec.target_x,
        "target_y": spec.target_y,
    }
    if canonical_hash(physical) != protocol["physical_spec_sha256"]:
        raise RuntimeError("physical specification changed after authorization")
    if control_variant(int(protocol["family_index"]))["variant_id"] != protocol["control_variant"]:
        raise RuntimeError("control variant changed after authorization")

    started = datetime.now(timezone.utc)
    _write_json(output_root / "model_construction_started.json", {
        "schema": "l2rar2_r14_model_construction_v1",
        "instance_index": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    sim = AttachRelposeDynamicTabletop(spec, int(protocol["rollout_seed"]))
    states: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    action_end_states: list[dict[str, Any]] = []
    return_states: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    sequence = 0
    capture_order = 0
    render_count = 0
    current_action = "before_program"

    def save_state(sampling_point: str, action: str | None) -> dict[str, Any]:
        nonlocal sequence
        row = _snapshot(sim, sampling_point=sampling_point, sequence=sequence, action=action)
        sequence += 1
        states.append(row)
        return row

    save_state("initial_after_forward", None)
    with TabletopRenderer(sim.model) as renderer:
        def capture(current: Any, phase: str, action: str) -> None:
            nonlocal capture_order, render_count
            callback_steps = ["callback_enter", "render_start"]
            image_path = output_root / "rgb" / "front" / f"frame_{capture_order:05d}.jpg"
            renderer.save_jpeg(renderer.render(current.data, "front", spec.camera_jitter), image_path)
            render_count += 1
            callback_steps.extend(("render_end", "state_saved"))
            state = save_state(f"{phase}_callback", action)
            if phase == "action_end":
                action_end_states.append(state)
            callback_steps.append("callback_exit")
            captures.append({
                "capture_order": capture_order,
                "phase": phase,
                "action": action,
                "action_index": int(current.action_index),
                "time": float(current.data.time),
                "image_path": str(image_path.resolve()),
                "image_sha256": sha256(image_path),
                "callback_steps": callback_steps,
            })
            capture_order += 1

        def control_callback(current: Any, control: dict[str, Any]) -> None:
            del current, control
            capture(sim, "control_tick", current_action)

        def action_end_callback(current: Any, action: str, action_index: int, result: dict[str, Any]) -> None:
            del current, action_index, result
            capture(sim, "action_end", action)

        sim._r1_control_callback = control_callback
        sim._r1_action_end_callback = action_end_callback
        for planned_action in case["program"]:
            current_action = planned_action
            save_state("before_action", planned_action)
            result = perform(sim, planned_action, int(protocol["family_index"]))
            low = copy.deepcopy(result.get("low_level_control_sequence", []))
            actions.append({
                "action_index": int(result["action_index"]),
                "action": planned_action,
                "start_time": float(result["start_time"]),
                "end_time": float(result["end_time"]),
            })
            controls.append({"action_index": int(result["action_index"]), "action": planned_action, "controls": low})
            return_states.append(save_state("after_perform_return", planned_action))

    visual_action_end: list[dict[str, Any]] = []
    for capture_row in captures:
        if capture_row["phase"] != "action_end":
            continue
        detected = detect_frame(Path(capture_row["image_path"]))
        visual_action_end.append({
            "capture_order": capture_row["capture_order"],
            "object_centroid": detected.get("object_centroid"),
            "gripper_centroid": detected.get("gripper_centroid"),
            "object_confidence": detected.get("object_confidence"),
            "gripper_confidence": detected.get("gripper_confidence"),
            "width": detected.get("width"),
            "height": detected.get("height"),
        })

    tolerances = protocol["tolerances"]
    gates = {
        "action_sequence": compare_actions(actions, cache_root, float(tolerances["time_atol"])),
        "low_level_controls": compare_controls(
            controls,
            cache_root,
            time_atol=float(tolerances["time_atol"]),
            mocap_atol=float(tolerances["mocap_atol"]),
        ),
        "event_sequence": compare_events(copy.deepcopy(sim.events), cache_root),
        "action_end_geometry": compare_action_end_geometry(action_end_states, cache_root, float(tolerances["geometry_atol"])),
        "callback_capture_order": compare_callback_order(captures, cache_root, float(tolerances["time_atol"])),
        "callback_to_perform_return": compare_callback_to_return(action_end_states, return_states),
        "state_numeric_health": state_health(states),
    }
    metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
    cache_control_count = int(metadata["dense_observation_count"])
    expected_render_count = int(protocol["expected_renderer_callback_count"])
    actual_control_count = sum(row["phase"] == "control_tick" for row in captures)
    actual_action_end_count = sum(row["phase"] == "action_end" for row in captures)
    gates["renderer_callback_count"] = {
        "passed": bool(
            render_count == expected_render_count == len(captures)
            and actual_control_count == cache_control_count == int(protocol["expected_control_callback_count"])
            and actual_action_end_count == int(protocol["expected_action_end_callback_count"])
        ),
        "actual": render_count,
        "expected": expected_render_count,
        "capture_rows": len(captures),
        "actual_control_callbacks": actual_control_count,
        "expected_control_callbacks": int(protocol["expected_control_callback_count"]),
        "actual_action_end_callbacks": actual_action_end_count,
        "expected_action_end_callbacks": int(protocol["expected_action_end_callback_count"]),
    }
    all_passed = all(bool(value["passed"]) for value in gates.values())

    _write_jsonl(output_root / "state_trace.jsonl", states)
    _write_jsonl(output_root / "callback_capture_trace.jsonl", captures)
    _write_jsonl(output_root / "actions.jsonl", actions)
    _write_jsonl(output_root / "low_level_controls.jsonl", controls)
    _write_jsonl(output_root / "events.jsonl", copy.deepcopy(sim.events))
    _write_json(output_root / "visual_action_end_diagnostics.json", {
        "schema": "l2rar2_r14_visual_action_end_diagnostics_v1",
        "main_gate": False,
        "reason": "locked R11 action-end geometry contract compares oracle object_xyz and gripper_xyz",
        "rows": visual_action_end,
    })
    _write_json(output_root / "ordinary_replay_comparison.json", {
        "schema": "l2rar2_r14_ordinary_replay_comparison_v1",
        "status": "PASS" if all_passed else "FAIL",
        "main_gates": gates,
        "all_main_gates_passed": all_passed,
        "stop_action": None if all_passed else "STOP_AFTER_EXECUTION_1",
    })
    result = {
        "schema": "l2rar2_r14_ordinary_replay_result_v1",
        "status": "R14_ORDINARY_REPLAY_PASS" if all_passed else "STOP_AFTER_EXECUTION_1",
        "stage": protocol["stage"],
        "executions_used": 1,
        "authorized_instances": 1,
        "automatic_retry": False,
        "root_family_id": protocol["root_family_id"],
        "case_id": protocol["case_id"],
        "rollout_seed": protocol["rollout_seed"],
        "repair_version": ATTACH_RELPOSE_VERSION,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "opencv": cv2.__version__,
        },
        "main_gate_pass_count": sum(bool(value["passed"]) for value in gates.values()),
        "main_gate_count": len(gates),
        "all_main_gates_passed": all_passed,
        "instrumented_replay_executions": 0,
        "r16_calibration_executions": 0,
        "r16_development_executions": 0,
    }
    _write_json(output_root / "result.json", result)
    if not all_passed:
        (output_root / "STOP_AFTER_EXECUTION_1").write_text("ordinary replay main-gate mismatch; no retry or later stage authorized\n", encoding="utf-8")
    return result
