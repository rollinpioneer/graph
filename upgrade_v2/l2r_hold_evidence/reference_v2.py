from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inputs import read_csv, read_jsonl


def _float(value: Any) -> float:
    return float(value)


def build_reference(rollout: dict[str, Any]) -> dict[str, Any]:
    root = Path(rollout["rollout_path"])
    events = read_jsonl(root / "events.jsonl") if (root / "events.jsonl").is_file() else []
    actions = read_csv(root / "actions.csv") if (root / "actions.csv").is_file() else []
    oracle = read_csv(root / "oracle_timeline.csv") if (root / "oracle_timeline.csv").is_file() else []
    intervals = []
    active_start = None
    for row in oracle:
        t, weld = _float(row["time"]), str(row.get("weld_state", "0")) in {"1", "true"}
        if weld and active_start is None:
            active_start = t
        if not weld and active_start is not None:
            intervals.append({"start": active_start, "end": t, "status": "held"})
            active_start = None
    if active_start is not None and oracle:
        intervals.append({"start": active_start, "end": _float(oracle[-1]["time"]), "status": "held"})
    def state_before(time: float) -> dict[str, Any] | None:
        rows = [row for row in oracle if _float(row["time"]) < time - 1e-9]
        return rows[-1] if rows else None

    def state_after(time: float) -> dict[str, Any] | None:
        rows = [row for row in oracle if _float(row["time"]) >= time - 1e-9]
        return rows[0] if rows else None

    def first_observation(stream: str, time: float) -> float | None:
        name = "observations_dense.jsonl" if stream == "dense" else "observations_action_end.jsonl"
        path = root / name
        if not path.is_file():
            return None
        rows = read_jsonl(path)
        times = [float(row["time"]) for row in rows if row.get("time") is not None and float(row["time"]) >= time - 1e-9]
        return min(times) if times else None

    decision_events = []
    for index, event in enumerate(events):
        name = event.get("event")
        time = _float(event["time"])
        if name == "missed_grasp":
            if rollout.get("observation_intervention", False):
                action, label = "needs_observation", "history_insufficient_for_missed_grasp_classification"
            else:
                action, label = "retry_grasp", "missed_grasp_retry_required"
        elif name == "contact_lost":
            before = state_before(time)
            held_before = bool(before and str(before.get("weld_state", "0")).lower() in {"1", "true"})
            intervention = bool(rollout.get("observation_intervention", False))
            if held_before and intervention:
                action, label = "needs_observation", "history_insufficient_for_loss_classification"
            elif held_before:
                action, label = "recover_object", "held_object_loss_recovery_required"
            else:
                action, label = "none", "touch_without_stable_hold"
        elif name == "released":
            action, label = "none", "release_expected"
        else:
            continue
        decision_events.append({
            "event_id": f"{rollout['rollout_id']}_event_{index:03d}",
            "parent_family_id": rollout["root_family_id"],
            "reference_event_start": time,
            "reference_event_end": time,
            "reference_type": label,
            "expected_action": action,
            "reference_hold_interval": next((interval for interval in intervals if interval["start"] <= time <= interval["end"]), None),
            "physical_label_status": "reference_labeled",
            "physical_evidence_paths": [str((root / "events.jsonl").resolve()), str((root / "oracle_timeline.csv").resolve())],
            "first_possible_observation_time_dense": first_observation("dense", time),
            "first_possible_observation_time_action_end": first_observation("action_end", time),
            "decision_window_start": time,
            "decision_window_end": time + 0.5,
            "resolution_time": next((float(row["time"]) for row in events[index + 1:] if row.get("event") in {"recovery_achieved", "contact_reestablished"} and float(row.get("time", 0.0)) >= time), None),
            "observation_quality": "history_masked" if rollout.get("observation_intervention", False) else "recorded",
            "history_available_from_stream": not rollout.get("observation_intervention", False),
            "reference_version": "timestamp_aligned_proxy_hold_v2",
        })
    return {"hold_reference_intervals": intervals, "decision_reference_events": decision_events, "reference_version": "timestamp_aligned_proxy_hold_v2"}
