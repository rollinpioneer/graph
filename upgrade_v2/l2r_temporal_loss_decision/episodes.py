from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .online_schema import sanitize
from .timebase import Point, deduplicate_points


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trace(path: Path) -> list[dict[str, Any]]:
    return _jsonl(path) if path.is_file() else []


def _detector(path: Path) -> dict[str, Any]:
    from .visual_separation import detect_frame_boxes
    try:
        return detect_frame_boxes(path)
    except Exception as exc:
        return {"detector_error": type(exc).__name__}


def _relative(row: dict[str, Any]) -> tuple[float, float] | None:
    obj, grip = row.get("object_centroid"), row.get("gripper_centroid")
    if not (isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(grip, (list, tuple)) and len(grip) == 2):
        return None
    try:
        result = (float(obj[0]) - float(grip[0]), float(obj[1]) - float(grip[1]))
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result) else None


def _reference(rows: list[dict[str, Any]], outcome: dict[str, Any], case_id: str) -> dict[str, Any]:
    from upgrade_v2.l2r_forced_drop.physical_reference import evaluate_loss_trace
    commanded = any(bool(row.get("commanded_release")) for row in rows) or "release" in case_id.lower()
    if commanded:
        return {"reference_state": "COMMANDED_RELEASE", "loss_confirmed": False, "onset_ns": None, "confirmed_ns": None}
    # R25 marks true physical-loss tasks through recovery_required; C6/C7 are
    # explicitly no-loss control cases even though they exercise intervention logic.
    if "physical_loss_exists" in outcome:
        loss = bool(outcome.get("physical_loss_exists"))
    elif case_id.startswith("R25C6") or case_id.startswith("R25C7"):
        loss = False
    else:
        loss = bool(outcome.get("recovery_required"))
    # Physical reference timing starts only after the trace itself verifies
    # the held object. Pre-hold approach rows must not create an onset at t=0.
    hold_index = next(
        (index for index, row in enumerate(rows)
         if bool(row.get("pre_hold_verified"))),
        None,
    )
    if hold_index is None:
        return {"reference_state": "UNRESOLVED", "loss_confirmed": False,
                "onset_ns": None, "confirmed_ns": None,
                "evaluator_state": "PREHOLD_UNVERIFIED"}
    evaluated = evaluate_loss_trace(rows[hold_index:], pre_hold_verified=True)
    onset = evaluated.get("loss_onset_time_abs")
    confirmed = evaluated.get("loss_confirmed_time_abs")
    if loss and onset is not None:
        return {"reference_state": "LOSS", "loss_confirmed": True, "onset_ns": int(round(float(onset) * 1_000_000_000)), "confirmed_ns": int(round(float(confirmed or onset) * 1_000_000_000)), "evaluator_state": evaluated.get("state")}
    if not loss:
        return {"reference_state": "NO_LOSS", "loss_confirmed": False, "onset_ns": None, "confirmed_ns": None, "evaluator_state": evaluated.get("state")}
    return {"reference_state": "UNRESOLVED", "loss_confirmed": False, "onset_ns": None, "confirmed_ns": None, "evaluator_state": evaluated.get("state")}


def _rollout(path: Path) -> dict[str, Any]:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    outcome = json.loads((path / "outcome.json").read_text(encoding="utf-8"))
    case_id = str(meta.get("case_id", ""))
    raw_rows = _jsonl(path / "candidate_input/live_observations.jsonl")
    manifest = list(csv.DictReader((path / "online_raw/frame_manifest.csv").open(encoding="utf-8")))
    logical_path = path / "online_raw/logical_observation_manifest.csv"
    logical = list(csv.DictReader(logical_path.open(encoding="utf-8"))) if logical_path.is_file() else []
    by_order = {int(row["capture_order"]): row for row in manifest if row.get("capture_order") is not None}
    by_candidate = {int(row["candidate_capture_order"]): by_order.get(int(row["source_capture_order"]), {}) for row in logical if row.get("candidate_capture_order") is not None}
    observations: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = sanitize(raw)
        order = int(row.get("capture_order", -1)); frame = by_candidate.get(order, by_order.get(order, {}))
        jpeg_path = path / str(frame.get("jpeg_path", "")); det = _detector(jpeg_path) if jpeg_path.is_file() else {"detector_error": "missing_frame"}
        jpeg_hash = str(frame.get("jpeg_sha256", "")); ns = int(row.get("physical_time_ns", frame.get("physical_time_ns", 0)))
        def as_box(value: Any) -> list[float] | None:
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                return None
            x, y, w, h = (float(item) for item in value)
            return [x, y, x + w, y + h]
        row.update({"object_centroid": list(det.get("object_centroid")) if det.get("object_centroid") is not None else row.get("object_centroid"), "gripper_centroid": list(det.get("gripper_centroid")) if det.get("gripper_centroid") is not None else row.get("gripper_centroid"), "object_area": float(det.get("object_area", 0.0)), "gripper_area": float(det.get("gripper_area", 0.0)), "detector_error": bool(det.get("detector_error")), "object_bbox": as_box(det.get("object_bbox")), "gripper_bbox": as_box(det.get("gripper_bbox")), "jpeg_sha256": jpeg_hash, "jpeg_path": str(frame.get("jpeg_path", "")), "frame_missing": not jpeg_path.is_file(), "physical_time_ns": ns})
        observations.append(row)
    observations.sort(key=lambda item: int(item.get("capture_order", -1)))
    unique = deduplicate_points([Point(int(row["physical_time_ns"]), int(row["capture_order"]), _relative(row), bool(_relative(row) is not None and not row.get("detector_error") and not row.get("frame_missing")), bool(row.get("gripper_command") == "open" or row.get("requested_effect") == "RELEASE_OBJECT"), int(row.get("attempt_id", 1)), (int(row["physical_time_ns"]), str(row.get("jpeg_sha256", "")))) for row in observations])
    times = [point.time_ns for point in unique]
    closed = [row for row in observations if row.get("gripper_command") == "closed" and row.get("contact_present") is True and _relative(row) is not None and not row.get("detector_error")]
    hold_ready_ns = int(closed[1]["physical_time_ns"]) if len(closed) >= 2 else None
    baseline_rows = [row for row in closed if hold_ready_ns is not None and int(row["physical_time_ns"]) <= hold_ready_ns + 500_000_000]
    baseline_ready_ns = int(baseline_rows[-1]["physical_time_ns"]) if baseline_rows else None
    action_ns = None
    decisions: list[dict[str, Any]] = []
    output_path = path / "candidate_output/live_decisions.jsonl"
    if not output_path.is_file():
        output_path = path / "candidate_output/factor_decisions.jsonl"
    if output_path.is_file():
        decisions = _jsonl(output_path)
        actions = [int(d.get("physical_time_ns")) for d in decisions if d.get("selected_action") == "recover_object"]
        action_ns = min(actions) if actions else None
    trace = _trace(path / "reference/physics_trace.jsonl")
    reference = _reference(trace, outcome, case_id)
    last_ns = max(times) if times else 0
    evidence_end = min(last_ns, action_ns) if action_ns is not None else last_ns
    root_family = str(meta.get("family_id", ""))
    arm = str(meta.get("arm_id", meta.get("method", "")))
    target_subset = (case_id in {"R24C2_loss_signal_missing", "R24C4_multisource_missing_then_loss"} and arm == "O_C3_CLP3_CANONICAL_TIME") or (case_id.startswith(("R25C1", "R25C2")) and arm.startswith("F1"))
    return {"episode_id": path.name, "root_family_id": root_family, "paired_condition_id": f"{root_family}::{case_id}", "family_id": root_family, "arm_id": arm, "case_id": case_id, "rollout_seed": meta.get("rollout_seed", outcome.get("rollout_seed")), "reference_state": reference["reference_state"], "is_target_observability_subset": target_subset, "physical_loss_onset_ns": reference.get("onset_ns"), "physical_loss_confirmed_ns": reference.get("confirmed_ns"), "online_hold_ready_ns": hold_ready_ns, "baseline_ready_ns": baseline_ready_ns, "first_recovery_command_ns": action_ns, "episode_start_ns": int(times[0]) if times else None, "last_raw_physics_ns": int(max((float(row.get("time", 0.0)) for row in trace), default=0.0) * 1_000_000_000), "last_candidate_observation_ns": last_ns, "evidence_end_ns": evidence_end, "outcome": outcome, "candidate_decisions": decisions if output_path.is_file() else [], "observations": observations, "reference_trace_rows": len(trace)}


def build(resources: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in resources.get("sources", []):
        root = Path(source["path"])
        for metadata in sorted(root.glob("*/metadata.json")):
            records.append(_rollout(metadata.parent))
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (output / "label_vs_subset_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["episode_id", "root_family_id", "case_id", "arm_id", "reference_state", "is_target_observability_subset"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in records)
    with (output / "arm_episode_pairing.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["episode_id", "paired_condition_id", "root_family_id", "arm_id"]; writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in records)
    with (output / "action_censoring.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["episode_id", "evidence_end_ns", "first_recovery_command_ns", "last_candidate_observation_ns"]; writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in records)
    return {"schema": "l2rar2_r27_episode_audit_v1", "episodes": len(records), "families": len({row["root_family_id"] for row in records}), "physical_executions": 0, "mujoco_imported": False}
