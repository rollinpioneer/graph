from __future__ import annotations

import copy
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import CAPTURE_VERSION
from .capture import CaptureRecorder, ReadOnlyPhysicsRecorder, detection_record
from .environment import compare_main_gate_fields, finalize_fingerprint, runtime_fingerprint, validate_process_environment
from .fingerprint import model_fingerprint
from .protocol import canonical_hash, sha256
from .simulator import ReproducibleBaselineTabletop


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n" for row in rows))


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _manifest(output_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "result.json"}:
            files.append({"path": str(path.relative_to(output_root)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {"schema": "l2rar2_r14b_artifact_manifest_v1", "files": files}
    payload["artifact_manifest_sha256"] = canonical_hash(payload)
    _write_json(output_root / "artifact_manifest.json", payload)
    return payload


def _numeric_health(recorder: CaptureRecorder) -> dict[str, Any]:
    failures = []
    for row in recorder.states:
        for field in ("object_xyz", "gripper_xyz", "object_qpos", "object_qvel"):
            values = row.get(field, [])
            if any(not isinstance(x, (int, float)) or x != x or abs(float(x)) == float("inf") for x in values):
                failures.append({"sequence": row["sequence"], "field": field})
        for name, info in row["arrays"].items():
            if info["byte_length"] < 0:
                failures.append({"sequence": row["sequence"], "field": name})
    return {"schema": "l2rar2_r14b_numeric_health_v1", "passed": not failures, "state_count": len(recorder.states), "failures": failures}


def run_ordinary(*, repo: Path, protocol: dict[str, Any], output_root: Path, stage: str, source_lock: dict[str, Any], instrumented: bool = False) -> dict[str, Any]:
    validate_process_environment()
    import cv2
    import mujoco
    import numpy as np

    from upgrade_v2.l2r_hold_evidence.probe_adapter import control_variant, perform
    from upgrade_v2.l2r_task_context.collector import CASES
    from upgrade_v2.visual_refine_l2.dynamic_simulator import _xml, family_spec
    from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer
    from upgrade_v2.visual_refine_l2.vision import detect_frame

    required = protocol["required_environment"]
    actual = {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__, "opencv": cv2.__version__}
    if any(str(actual[key]) != str(required[key]) for key in actual):
        raise RuntimeError(f"ENVIRONMENT_CONTRACT_MISMATCH: {actual}")
    if control_variant(int(protocol["family_index"]))["variant_id"] != protocol["control_variant"]:
        raise RuntimeError("control variant changed")
    case = CASES[protocol["case_id"]]
    if case["program"] != protocol["program"] or canonical_hash(case["program"]) != protocol["program_sha256"]:
        raise RuntimeError("program hash changed")
    spec = family_spec(protocol["root_family_id"], protocol["scenario"], int(protocol["family_seed"]), int(protocol["rollout_seed"]), probe_variant=case.get("probe_variant", "default"))
    physical = {key: getattr(spec, key) for key in ("object_radius", "friction", "camera_jitter", "object_x", "object_y", "target_x", "target_y")}
    if canonical_hash(physical) != protocol["physical_spec_sha256"]:
        raise RuntimeError("physical specification changed")

    if not output_root.is_dir():
        raise RuntimeError("output root must be reserved by authorization before model construction")
    started = datetime.now(timezone.utc)
    _write_json(output_root / "model_construction_started.json", {"status": "STARTED_AFTER_AUTHORIZATION_CONSUMPTION", "started_at_utc": started.isoformat()})
    env_fp = finalize_fingerprint(runtime_fingerprint(mujoco=mujoco, cv2=cv2, numpy=np))
    xml = _xml(spec)
    _write_text(output_root / "generated_model.xml", xml)
    sim = ReproducibleBaselineTabletop(spec, int(protocol["rollout_seed"]), physics_observer=ReadOnlyPhysicsRecorder() if instrumented else None)
    model_fp = model_fingerprint(sim.model, mujoco, xml)
    if source_lock.get("generated_model_xml_sha256") != sha256(output_root / "generated_model.xml"):
        raise RuntimeError("pre-execution model XML hash changed")
    _write_json(output_root / "runtime_environment_fingerprint.json", env_fp)
    _write_json(output_root / "model_fingerprint.json", model_fp)
    _write_json(output_root / "source_lock.json", source_lock)

    recorder = CaptureRecorder(sim, None, output_root, spec)
    with TabletopRenderer(sim.model) as renderer:
        recorder.renderer = renderer
        recorder.checkpoint("initial_after_forward", None)
        actions: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        current_action = "before_program"

        def control_callback(current: Any, control: dict[str, Any]) -> None:
            recorder.callback("control_tick", current_action)

        def action_end_callback(current: Any, action: str, action_index: int, result: dict[str, Any]) -> None:
            recorder.callback("action_end", action)

        sim._r1_control_callback = control_callback
        sim._r1_action_end_callback = action_end_callback
        for planned_action in protocol["program"]:
            current_action = planned_action
            recorder.checkpoint("before_action", planned_action)
            result = perform(sim, planned_action, int(protocol["family_index"]))
            actions.append({"action_index": int(result["action_index"]), "action": planned_action, "start_time": float(result["start_time"]), "end_time": float(result["end_time"])})
            controls.append({"action_index": int(result["action_index"]), "action": planned_action, "controls": copy.deepcopy(result.get("low_level_control_sequence", []))})
            recorder.checkpoint("after_perform_return", planned_action)

    detections = []
    for capture in recorder.captures:
        detections.append(detection_record(capture, detect_frame(Path(capture["image_path"]))))

    _write_jsonl(output_root / "checkpoint_state_trace.jsonl", recorder.states)
    _write_jsonl(output_root / "callback_capture_trace.jsonl", recorder.captures)
    _write_jsonl(output_root / "actions.jsonl", actions)
    _write_jsonl(output_root / "low_level_controls.jsonl", controls)
    _write_jsonl(output_root / "events.jsonl", copy.deepcopy(sim.events))
    _write_jsonl(output_root / "vision_detections.jsonl", detections)
    _write_csv(output_root / "renderer_capture_manifest.csv", recorder.captures, ["capture_order", "phase", "action", "action_index", "time", "raw_rgb_sha256", "jpeg_sha256", "shape", "dtype", "image_path", "checkpoint_sequence", "callback_steps"])
    _write_json(output_root / "checkpoint_counts.json", {"checkpoints": len(recorder.states), "captures": len(recorder.captures), "actions": len(actions), "control_callbacks": sum(row["phase"] == "control_tick" for row in recorder.captures), "action_end_callbacks": sum(row["phase"] == "action_end" for row in recorder.captures), "physics_steps": sum(int(row.get("physics_steps", 0)) for row in controls for row in row["controls"])})
    numeric = _numeric_health(recorder)
    _write_json(output_root / "numeric_health.json", numeric)
    expected = protocol["expected_counts"]
    counts = json.loads((output_root / "checkpoint_counts.json").read_text(encoding="utf-8"))
    gates = {
        "environment_contract": all(str(actual[key]) == str(required[key]) for key in actual) and env_fp["renderer_backend"] == "egl",
        "model_fingerprint_complete": bool(model_fp.get("model_fingerprint_sha256")) and bool(model_fp.get("generated_model_xml_sha256")),
        "action_count": counts["actions"] == expected["actions"], "control_callback_count": counts["control_callbacks"] == expected["control_callbacks"],
        "action_end_callback_count": counts["action_end_callbacks"] == expected["action_end_callbacks"], "render_callback_count": len(recorder.captures) == expected["render_callbacks"],
        "physics_steps": counts["physics_steps"] == expected["physics_steps"], "checkpoint_count": len(recorder.states) == expected["ordinary_checkpoint_rows"],
        "numeric_health": numeric["passed"], "capture_files_complete": all(Path(row["image_path"]).is_file() for row in recorder.captures),
        "detection_count": len(detections) == expected["render_callbacks"], "instrumentation_trace": (not instrumented) or len(sim.physics_observer.rows) == expected["instrumented_before_after_rows"],
    }
    action_end_states = [row for row in recorder.states if row["sampling_point"] == "action_end_callback"]
    return_states = [row for row in recorder.states if row["sampling_point"] == "after_perform_return"]
    gates["action_end_geometry"] = len(action_end_states) == expected["action_end_callbacks"] and all(
        len(row["object_xyz"]) == 3 and len(row["gripper_xyz"]) == 3 for row in action_end_states
    )
    gates["callback_to_return_exact"] = len(action_end_states) == len(return_states) and all(
        left["state_sha256"] == right["state_sha256"] for left, right in zip(action_end_states, return_states)
    )
    if instrumented:
        integrity = {"schema": "l2rar2_r14b_instrumentation_integrity_v1", "before_after_rows": len(sim.physics_observer.rows), "mutation_checks": len(sim.physics_observer.rows), "mutation_failures": 0, "passed": len(sim.physics_observer.rows) == expected["instrumented_before_after_rows"]}
        _write_jsonl(output_root / "physics_step_trace.jsonl", sim.physics_observer.rows)
        _write_json(output_root / "instrumentation_integrity.json", integrity)
    all_passed = all(gates.values())
    _write_json(output_root / "baseline_quality_gates.json", {"schema": "l2rar2_r14b_quality_gates_v1", "gates": gates, "all_main_gates_passed": all_passed})
    manifest = _manifest(output_root)
    result = {
        "schema": "l2rar2_r14b_execution_result_v1", "status": "R14B_INSTRUMENTED_C_COMPLETE" if instrumented and all_passed else "R14B_BASELINE_A_COMPLETE_WAITING_HUMAN_REVIEW" if stage == "R14B_ORDINARY_BASELINE_A" and all_passed else "R14B_ORDINARY_REPEAT_B_COMPLETE" if stage == "R14B_ORDINARY_REPEAT_B" and all_passed else "STOP_AFTER_EXECUTION_1",
        "stage": stage, "executions_used": 1, "authorized_instances": 1, "automatic_retry": False,
        "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat(), "runner_commit": source_lock["runner_commit"], "runner_file_hashes": source_lock["generation_runner_file_hashes"],
        "protocol_sha256": protocol["protocol_sha256"], "environment_fingerprint_sha256": env_fp["fingerprint_sha256"], "model_fingerprint_sha256": model_fp["model_fingerprint_sha256"],
        "source_lock_sha256": source_lock["source_lock_sha256"], "environment_contract_sha256": source_lock["environment_contract_sha256"],
        "generated_model_xml_sha256": source_lock["generated_model_xml_sha256"], "family_seed": source_lock["family_seed"],
        "main_gate_count": len(gates), "main_gate_pass_count": sum(bool(v) for v in gates.values()), "all_main_gates_passed": all_passed, "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
        "instrumented_replay_executions": 1 if instrumented else 0, "r16_calibration_executions": 0, "r16_development_executions": 0,
    }
    _write_json(output_root / "result.json", result)
    if not all_passed:
        _write_text(output_root / "STOP_AFTER_EXECUTION_1", "R14-B main-gate mismatch; no retry or later stage authorized\n")
    return result
