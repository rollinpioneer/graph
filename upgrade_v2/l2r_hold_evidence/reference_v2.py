from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .inputs import read_csv, read_jsonl


def _float(value: Any) -> float:
    return float(value)


def _vector(value: Any) -> list[float] | None:
    if value in (None, ""):
        return None
    parsed = json.loads(value) if isinstance(value, str) else value
    return [float(item) for item in parsed]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def build_reference(rollout: dict[str, Any]) -> dict[str, Any]:
    root = Path(rollout["rollout_path"])
    events = read_jsonl(root / "events.jsonl") if (root / "events.jsonl").is_file() else []
    actions = read_csv(root / "actions.csv") if (root / "actions.csv").is_file() else []
    oracle = read_csv(root / "oracle_timeline.csv") if (root / "oracle_timeline.csv").is_file() else []
    ordered = sorted(oracle, key=lambda row: (_float(row["time"]), int(row.get("capture_order", 0))))
    dense_observations = read_jsonl(root / "observations_dense.jsonl") if (root / "observations_dense.jsonl").is_file() else []
    history_available = bool(dense_observations) and not any(
        row.get("observation_masked") or row.get("observation_missing") for row in dense_observations
    )
    intervals = []
    active_rows: list[dict[str, str]] = []

    def finish_interval(end_time: float) -> None:
        if not active_rows:
            return
        object_points = [_vector(row.get("object_xyz")) for row in active_rows]
        gripper_points = [_vector(row.get("gripper_xyz")) for row in active_rows]
        valid = all(point is not None for point in object_points + gripper_points)
        object_displacement = _distance(object_points[0], object_points[-1]) if valid else None
        relative = [[o[i] - g[i] for i in range(3)] for o, g in zip(object_points, gripper_points)] if valid else []
        relative_drift = max((_distance(relative[0], point) for point in relative), default=None) if valid else None
        verified = bool(valid and object_displacement >= 0.01 and relative_drift <= 0.02)
        intervals.append({
            "start": _float(active_rows[0]["time"]),
            "end": end_time,
            "status": "held_verified" if verified else "reference_unresolved",
            "weld_active": True,
            "object_displacement_m": object_displacement,
            "maximum_relative_position_drift_m": relative_drift,
            "world_position_error_limit_m": 0.02,
            "minimum_object_displacement_m": 0.01,
        })

    for row in ordered:
        weld = str(row.get("weld_state", "0")).lower() in {"1", "true"}
        if weld:
            active_rows.append(row)
        elif active_rows:
            finish_interval(_float(row["time"]))
            active_rows = []
    if active_rows:
        finish_interval(_float(active_rows[-1]["time"]))
    def state_before(time: float) -> dict[str, Any] | None:
        rows = [row for row in ordered if _float(row["time"]) <= time + 1e-9]
        return rows[-1] if rows else None

    def state_after(time: float) -> dict[str, Any] | None:
        rows = [row for row in ordered if _float(row["time"]) >= time - 1e-9]
        return rows[0] if rows else None

    def first_observation(stream: str, time: float) -> float | None:
        name = "observations_dense.jsonl" if stream == "dense" else "observations_action_end.jsonl"
        path = root / name
        if not path.is_file():
            return None
        rows = read_jsonl(path)
        times = [float(row["time"]) for row in rows if row.get("time") is not None and not row.get("observation_masked") and not row.get("observation_missing") and float(row["time"]) >= time - 1e-9]
        return min(times) if times else None

    decision_events = []
    for index, event in enumerate(events):
        name = event.get("event")
        time = _float(event["time"])
        if name == "missed_grasp":
            if not history_available:
                action, label = "needs_observation", "history_insufficient_for_missed_grasp_classification"
            else:
                action, label = "retry_grasp", "missed_grasp_retry_required"
        elif name == "contact_lost":
            hold_interval = next((interval for interval in intervals if interval["start"] <= time <= interval["end"] and interval["status"] == "held_verified"), None)
            held_before = hold_interval is not None
            before = state_before(time)
            weld_before = bool(before and str(before.get("weld_state", "0")).lower() in {"1", "true"})
            intervention = not history_available
            if weld_before and not held_before:
                action, label = "needs_observation", "held_object_loss_reference_unresolved"
            elif held_before and intervention:
                action, label = "needs_observation", "history_insufficient_for_loss_classification"
            elif held_before:
                action, label = "recover_object", "held_object_loss_recovery_required"
            else:
                action, label = "none", "touch_without_stable_hold"
        elif name == "released":
            action, label = "none", "release_expected"
        elif name == "transient_contact_lost":
            action, label = "none", "touch_without_stable_hold"
        else:
            continue
        hold_interval = next((interval for interval in intervals if interval["start"] <= time <= interval["end"] and interval["status"] == "held_verified"), None)
        resolution = next((float(row["time"]) for row in events[index + 1:] if row.get("event") in {"recovery_achieved", "contact_reestablished"} and float(row.get("time", 0.0)) >= time), None)
        correction_start = next((float(row["start_time"]) for row in actions if row.get("action") in {"retry", "recover"} and float(row.get("start_time", 0.0)) >= time), None)
        endpoints = [time + 0.5] + [value for value in (resolution, correction_start) if value is not None]
        physical_status = "reference_unresolved" if label == "held_object_loss_reference_unresolved" else "reference_labeled"
        decision_events.append({
            "event_id": f"{rollout['rollout_id']}_event_{index:03d}",
            "parent_family_id": rollout["root_family_id"],
            "reference_event_start": time,
            "reference_event_end": time,
            "reference_type": label,
            "expected_action": action,
            "reference_hold_interval": hold_interval,
            "physical_label_status": physical_status,
            "physical_evidence_paths": [str((root / "events.jsonl").resolve()), str((root / "oracle_timeline.csv").resolve())],
            "first_possible_observation_time_dense": first_observation("dense", time),
            "first_possible_observation_time_action_end": first_observation("action_end", time),
            "decision_window_start": time,
            "decision_window_end": min(endpoints),
            "resolution_time": resolution,
            "observation_quality": "recorded" if history_available else "history_masked",
            "history_available_from_stream": history_available,
            "reference_version": "timestamp_aligned_proxy_hold_v2",
        })
    return {"hold_reference_intervals": intervals, "decision_reference_events": decision_events, "reference_version": "timestamp_aligned_proxy_hold_v2"}
