from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .io import read_csv, read_json, read_jsonl


EXPECTED = {
    "K1_hold_request_ends_without_hold": ("retry_grasp", "hold_request_normal_end_without_hold"),
    "K2_touch_request_completes_without_hold": ("none", "touch_request_normal_completion"),
    "K3_normal_hold_pause_resume": ("none", "held_pause_resume_no_failure"),
    "K4_regular_hold_loss": ("recover_object", "held_object_loss_recovery_required"),
    "K5_brief_hold_loss": ("recover_object", "held_object_loss_recovery_required"),
    "K6_long_gap_after_loss": ("recover_object", "held_object_loss_recovery_required"),
    "K7_commanded_release": ("none", "commanded_release"),
    "K8_acquisition_touch_then_continue": ("none", "active_acquisition_transient_contact"),
}


def _vector(value: Any) -> list[float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    return [float(item) for item in parsed]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _verified_hold_intervals(oracle: list[dict[str, str]]) -> list[dict[str, Any]]:
    ordered = sorted(oracle, key=lambda row: (float(row["time"]), int(row.get("capture_order", 0))))
    intervals: list[dict[str, Any]] = []
    active: list[dict[str, str]] = []

    def finish(end_time: float) -> None:
        if not active:
            return
        objects = [_vector(row["object_xyz"]) for row in active]
        grippers = [_vector(row["gripper_xyz"]) for row in active]
        object_displacement = _distance(objects[0], objects[-1])
        relative = [[obj[i] - grip[i] for i in range(3)] for obj, grip in zip(objects, grippers)]
        relative_drift = max(_distance(relative[0], point) for point in relative)
        intervals.append({
            "start": float(active[0]["time"]), "end": end_time,
            "object_displacement_m": object_displacement,
            "maximum_relative_position_drift_m": relative_drift,
            "status": "held_verified" if object_displacement >= 0.01 and relative_drift <= 0.02 else "reference_unresolved",
        })

    for row in ordered:
        weld = str(row.get("weld_state", "0")).lower() in {"1", "true"}
        if weld:
            active.append(row)
        elif active:
            finish(float(row["time"]))
            active = []
    if active:
        finish(float(active[-1]["time"]))
    return intervals


def build_reference(meta: dict[str, Any], window_seconds: float = 0.5) -> dict[str, Any]:
    root = Path(meta["path"])
    case_id = meta["case_id"]
    expected, reference_type = EXPECTED[case_id]
    actions = read_csv(root / "actions.csv")
    events = read_jsonl(root / "events.jsonl")
    observations = read_jsonl(root / "observations_dense.jsonl")
    requests = read_jsonl(root / "controller_requests.jsonl")
    oracle = read_csv(root / "oracle_timeline.csv")
    hold_intervals = _verified_hold_intervals(oracle)
    if not requests or requests[0]["requested_effect"] != meta["requested_effect"]:
        return {"status": "reference_unresolved", "reason": "controller_request_missing_or_inconsistent", "events": []}
    event_names = {row.get("event") for row in events}
    if case_id == "K1_hold_request_ends_without_hold" and "missed_grasp" not in event_names:
        return {"status": "reference_unresolved", "reason": "planned missed-grasp event did not occur", "events": []}
    if case_id in {"K2_touch_request_completes_without_hold", "K8_acquisition_touch_then_continue"} and not {"transient_contact", "transient_contact_lost"}.issubset(event_names):
        return {"status": "reference_unresolved", "reason": "planned transient-contact sequence did not occur", "events": []}
    if case_id == "K7_commanded_release" and "released" not in event_names:
        return {"status": "reference_unresolved", "reason": "planned commanded release did not occur", "events": []}
    end_action = next(row for row in actions if str(row["planned_request_end"]).lower() in {"true", "1"})
    end_time = float(end_action["end_time"])
    event_time = end_time
    if expected == "recover_object":
        loss = next((float(row["time"]) for row in events if row.get("event") == "contact_lost"), None)
        if loss is None:
            return {"status": "reference_unresolved", "reason": "planned held loss did not occur", "events": []}
        verified = next((interval for interval in hold_intervals if interval["start"] <= loss + 1e-9 <= interval["end"] + 1e-9 and interval["status"] == "held_verified"), None)
        if verified is None:
            return {"status": "reference_unresolved", "reason": "held loss lacks frozen physical proxy support", "events": [], "hold_intervals": hold_intervals}
        event_time = loss
    elif case_id == "K8_acquisition_touch_then_continue":
        lost = next((float(row["time"]) for row in events if row.get("event") == "transient_contact_lost"), None)
        if lost is None:
            return {"status": "reference_unresolved", "reason": "planned transient contact loss did not occur", "events": []}
        event_time = lost
    elif case_id == "K3_normal_hold_pause_resume":
        if not any(interval["status"] == "held_verified" for interval in hold_intervals):
            return {"status": "reference_unresolved", "reason": "pause case did not establish frozen physical hold proxy", "events": [], "hold_intervals": hold_intervals}
        post_end = [float(row["time"]) for row in observations if float(row["time"]) >= end_time - 1e-9]
        event_time = post_end[0] if post_end else end_time
    elif case_id == "K7_commanded_release":
        released = next((float(row["time"]) for row in events if row.get("event") == "released"), None)
        event_time = released if released is not None else end_time

    end = min(event_time + window_seconds, max(float(row["time"]) for row in observations))
    return {
        "status": "reference_labeled",
        "version": "l2rar2_task_physical_reference_v1",
        "hold_intervals": hold_intervals,
        "events": [{
            "event_id": f"{meta['rollout_id']}:primary",
            "rollout_id": meta["rollout_id"],
            "root_family_id": meta["root_family_id"],
            "case_id": case_id,
            "expected_action": expected,
            "reference_type": reference_type,
            "decision_window_start": event_time,
            "decision_window_end": end,
            "physical_label_status": "reference_labeled",
            "request_provenance": "controller_dispatch_pre_action",
            "reference_only_sources": "case registry; events.jsonl; oracle_timeline.csv",
        }],
    }
