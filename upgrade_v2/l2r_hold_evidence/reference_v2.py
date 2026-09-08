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
    decision_events = []
    for index, event in enumerate(events):
        name = event.get("event")
        if name == "missed_grasp":
            action, label = "retry_grasp", "missed_grasp_retry_required"
        elif name == "contact_lost":
            action, label = "recover_object", "held_object_loss_recovery_required" if intervals else "touch_without_stable_hold"
        elif name == "released":
            action, label = "none", "release_expected"
        else:
            continue
        time = _float(event["time"])
        decision_events.append({"event_id": f"{rollout['rollout_id']}_event_{index:03d}", "reference_event_start": time, "reference_event_end": time, "reference_type": label, "expected_action": action, "physical_label_status": "reference_labeled", "physical_evidence_paths": [str((root / "events.jsonl").resolve()), str((root / "oracle_timeline.csv").resolve())], "observation_quality": "recorded", "history_available_from_stream": not rollout.get("observation_intervention", False)})
    return {"hold_reference_intervals": intervals, "decision_reference_events": decision_events, "reference_version": "timestamp_aligned_proxy_hold_v2"}
