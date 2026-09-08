"""Offline reference-event extraction.

This module is intentionally never imported by the online event-memory path.
It may read simulator events, action intervals and oracle timelines solely to
label already-recorded diagnostic windows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inputs import read_csv


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _interval_for_time(actions: list[dict[str, str]], time: float) -> tuple[int | None, float | None, float | None, str | None]:
    for index, row in enumerate(actions):
        start, end = float(row["start_time"]), float(row["end_time"])
        if start - 1e-9 <= time <= end + 1e-9:
            return index, start, end, row["action"]
    return None, None, None, None


def extract_reference_events(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(rollout["rollout_path"])
    actions = read_csv(root / "actions.csv")
    events = read_jsonl(root / "events.jsonl")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    result = []
    event_map = {
        "missed_grasp": ("missed_grasp_retry_required", "retry_grasp"),
        "contact_lost": ("held_object_loss_recovery_required", "recover_object"),
        "released": ("release_expected", "release"),
        "recovery_achieved": ("recovery_achieved_observed", "none"),
    }
    counter = 0
    for event in events:
        name = event.get("event")
        if name not in event_map:
            continue
        counter += 1
        kind, action = event_map[name]
        idx, start, end, action_name = _interval_for_time(actions, float(event["time"]))
        observable = bool(event.get("observable_online", False))
        result.append({
            "event_id": f"{rollout['rollout_id']}_event_{counter:02d}",
            "rollout_id": rollout["rollout_id"], "parent_family_id": rollout["root_family_id"],
            "onset_interval_start": start, "onset_interval_end": end, "resolution_interval": None,
            "frame_index": idx, "reference_event_type": kind, "reference_action_class": action,
            "reference_source": "events.jsonl+actions.csv+metadata.json",
            "reference_observable_at_decision": observable,
            "reference_history_complete": True, "reference_label_status": "reference_labeled",
            "source_event_name": name, "source_action_name": action_name,
            "source_metadata_flags": {"scenario": metadata.get("scenario"), "recovery_achieved": metadata.get("recovery_achieved")},
        })
    # A rollout with no event is still useful as a negative opportunity, but
    # it is not an event label and is kept outside event metrics.
    return result


def build_event_index(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rollout in resolved["rollouts"]:
        rows.extend(extract_reference_events(rollout))
    return rows
