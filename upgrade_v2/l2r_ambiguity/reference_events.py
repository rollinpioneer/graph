"""Offline reference-event extraction from recorded rollout evidence.

This module is intentionally kept out of the online event-memory path. It may
use event logs, action intervals and simulator-only state to label a recorded
window, but it never uses a stratum/scenario name as a label source.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inputs import read_csv


EPS = 1e-9
ACTIONABLE_TYPES = {
    "missed_grasp_retry_required",
    "held_object_loss_recovery_required",
    "touch_without_stable_hold",
    "release_expected",
    "needs_observation",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _event_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "events.jsonl"
    return read_jsonl(path) if path.is_file() else []


def _oracle_rows(root: Path) -> list[dict[str, str]]:
    path = root / "oracle_timeline.csv"
    return read_csv(path) if path.is_file() else []


def _actions(root: Path) -> list[dict[str, str]]:
    path = root / "actions.csv"
    return read_csv(path) if path.is_file() else []


def _metadata(root: Path) -> dict[str, Any]:
    path = root / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _float(value: Any) -> float:
    return float(value)


def _interval_for_time(
    actions: list[dict[str, str]],
    time: float,
    preferred_actions: set[str] | None = None,
) -> tuple[int | None, float | None, float | None, str | None]:
    """Align an event to an action interval, preferring exact start boundaries."""
    candidates = []
    for index, row in enumerate(actions):
        start, end = _float(row["start_time"]), _float(row["end_time"])
        if start - EPS <= time <= end + EPS:
            candidates.append((index, start, end, row["action"]))
    if not candidates:
        return None, None, None, None
    preferred_actions = preferred_actions or set()
    exact_start = [row for row in candidates if abs(row[1] - time) <= EPS]
    for group in (exact_start, candidates):
        preferred = [row for row in group if row[3] in preferred_actions]
        if preferred:
            return preferred[0]
        if group:
            return group[0]
    return None, None, None, None


def _state_at_or_before(oracle: list[dict[str, str]], time: float) -> dict[str, str] | None:
    rows = [row for row in oracle if _float(row["time"]) <= time + EPS]
    return rows[-1] if rows else None


def _state_at_or_after(oracle: list[dict[str, str]], time: float) -> dict[str, str] | None:
    rows = [row for row in oracle if _float(row["time"]) >= time - EPS]
    return rows[0] if rows else None


def _weld(row: dict[str, str] | None) -> str:
    if row is None or "weld_state" not in row:
        return "unknown"
    return "true" if str(row["weld_state"]).lower() in {"1", "true"} else "false"


def _contact(row: dict[str, str] | None) -> str:
    if row is None or "contact_present" not in row:
        return "unknown"
    return "true" if str(row["contact_present"]).lower() in {"1", "true"} else "false"


def _release_before_or_at(actions: list[dict[str, str]], time: float) -> bool:
    return any(
        row["action"] in {"open_gripper", "release", "commanded_release"}
        and _float(row["start_time"]) <= time + EPS
        for row in actions
    )


def _action_for_event(
    actions: list[dict[str, str]], name: str, time: float
) -> tuple[int | None, float | None, float | None, str | None]:
    preferred = {
        "released": {"open_gripper", "release", "commanded_release"},
        "recovery_achieved": {"recover", "retry", "regrasp"},
        "contact_established": {"recover", "retry", "regrasp", "close_gripper"},
    }.get(name, set())
    return _interval_for_time(actions, time, preferred)


def _resolution(
    events: list[dict[str, Any]], actions: list[dict[str, str]], onset_time: float
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for index, event in enumerate(events):
        name = event.get("event")
        time = _float(event["time"]) if event.get("time") is not None else None
        if time is None or time + EPS < onset_time:
            continue
        if name not in {"recovery_achieved", "contact_established", "contact_reestablished"}:
            continue
        action_index, start, end, action_name = _action_for_event(actions, str(name), time)
        interval = None if start is None else {
            "action_index": action_index,
            "start": start,
            "end": end,
            "action": action_name,
        }
        return {"event_index": index, "event_name": name, "time": time, "interval": interval}, interval
    return None, None


def _base_record(
    rollout: dict[str, Any],
    event_id: str,
    event: dict[str, Any],
    event_index: int,
    event_type: str,
    action_class: str,
    label_status: str,
    observable: bool,
    history_complete: bool,
    actions: list[dict[str, str]],
    oracle: list[dict[str, str]],
    all_events: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    name = str(event.get("event"))
    time = _float(event["time"])
    action_index, start, end, action_name = _action_for_event(actions, name, time)
    resolution_event, resolution_interval = _resolution(all_events, actions, time)
    if resolution_event and resolution_event["event_index"] == event_index:
        resolution_event, resolution_interval = None, None
    evidence = {
        **evidence,
        "source_event_index": event_index,
        "source_event_time": time,
        "source_event_observable_online": event.get("observable_online"),
        "matched_action_index": action_index,
        "matched_action": action_name,
    }
    if resolution_event:
        evidence["resolution_event"] = resolution_event
    return {
        "event_id": event_id,
        "rollout_id": rollout["rollout_id"],
        "parent_family_id": rollout["root_family_id"],
        "onset_interval_start": start,
        "onset_interval_end": end,
        "resolution_interval": resolution_interval,
        "frame_index": action_index - 1 if action_index is not None else None,
        "decision_frame_index": action_index - 1 if action_index is not None else None,
        "reference_event_type": event_type,
        "reference_action_class": action_class,
        "reference_source": "events.jsonl+actions.csv+oracle_timeline.csv+termination.json",
        "reference_observable_at_decision": observable,
        "reference_history_complete": history_complete,
        "reference_label_status": label_status,
        "source_event_name": name,
        "source_action_name": action_name,
        "evidence_summary": evidence,
    }


def extract_probe_reference_events(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract primary semantic events from a probe's recorded evidence."""
    root = Path(rollout["rollout_path"])
    actions, events, oracle = _actions(root), _event_rows(root), _oracle_rows(root)
    metadata = _metadata(root)
    intervention = bool(metadata.get("observation_intervention", False) or rollout.get("observation_intervention", False))
    history_complete = not intervention
    records: list[dict[str, Any]] = []
    seen_primary = False
    counter = 0

    for event_index, event in enumerate(events):
        name = event.get("event")
        if name not in {"missed_grasp", "contact_lost", "transient_contact", "transient_contact_lost", "released"}:
            continue
        time = _float(event["time"])
        before = _state_at_or_before(oracle, time)
        after = _state_at_or_after(oracle, time)
        weld_before, weld_after = _weld(before), _weld(after)
        release_before = _release_before_or_at(actions, time)
        stable_before = weld_before == "true"
        stable_known = weld_before != "unknown"

        if name == "missed_grasp":
            event_type = "missed_grasp_retry_required"
            evidence_ok = stable_known and not stable_before
            action_class = "needs_observation" if intervention or not evidence_ok else "retry_grasp"
            label_status = "observation_intervention" if intervention else "reference_labeled" if evidence_ok else "reference_unresolved"
            observable = bool(not intervention and evidence_ok)
        elif name == "contact_lost":
            if release_before:
                event_type, action_class = "release_expected", "none"
                label_status, observable, evidence_ok = "reference_labeled", True, True
            elif stable_before:
                event_type = "held_object_loss_recovery_required"
                action_class = "needs_observation" if intervention else "recover_object"
                label_status, observable, evidence_ok = ("observation_intervention", False, True) if intervention else ("reference_labeled", True, True)
            elif stable_known:
                event_type, action_class = "touch_without_stable_hold", "none"
                label_status, observable, evidence_ok = "reference_labeled", True, True
            else:
                event_type, action_class = "needs_observation", "needs_observation"
                label_status, observable, evidence_ok = "reference_unresolved", False, False
        elif name in {"transient_contact", "transient_contact_lost"}:
            event_type, action_class = "touch_without_stable_hold", "none"
            label_status, observable, evidence_ok = "reference_labeled", True, True
        else:  # explicit released event
            event_type, action_class = "release_expected", "none"
            evidence_ok = release_before and (stable_before or weld_after == "true")
            label_status, observable = ("reference_labeled", True) if evidence_ok else ("reference_unresolved", False)

        # A transient contact is represented by its first actual contact event;
        # the later loss marker remains available in the evidence files.
        if name == "transient_contact_lost" and any(r["source_event_name"] == "transient_contact" for r in records):
            continue
        if seen_primary and event_type in ACTIONABLE_TYPES:
            continue
        counter += 1
        record = _base_record(
            rollout,
            f"{rollout['rollout_id']}_event_{counter:02d}",
            event,
            event_index,
            event_type,
            action_class,
            label_status,
            observable,
            history_complete,
            actions,
            oracle,
            events,
            {
                "oracle_weld_before": weld_before,
                "oracle_weld_at_or_after": weld_after,
                "oracle_contact_at_or_after": _contact(after),
                "explicit_release_before_or_at_event": release_before,
                "evidence_sufficient_for_type": evidence_ok,
                "metadata_observation_intervention": intervention,
            },
        )
        records.append(record)
        seen_primary = True
    return records


def extract_reference_events(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    """Backward-compatible event index for legacy diagnosis."""
    records = extract_probe_reference_events(rollout)
    if records:
        return records
    # Some historical records contain only an event log and action intervals.
    root = Path(rollout["rollout_path"])
    actions, events = _actions(root), _event_rows(root)
    event_map = {
        "missed_grasp": ("missed_grasp_retry_required", "retry_grasp"),
        "contact_lost": ("held_object_loss_recovery_required", "recover_object"),
        "released": ("release_expected", "release"),
        "recovery_achieved": ("recovery_achieved_observed", "none"),
    }
    result = []
    counter = 0
    for index, event in enumerate(events):
        name = event.get("event")
        if name not in event_map:
            continue
        counter += 1
        kind, action = event_map[name]
        action_index, start, end, action_name = _action_for_event(actions, str(name), _float(event["time"]))
        result.append({
            "event_id": f"{rollout['rollout_id']}_event_{counter:02d}",
            "rollout_id": rollout["rollout_id"],
            "parent_family_id": rollout["root_family_id"],
            "onset_interval_start": start,
            "onset_interval_end": end,
            "resolution_interval": None,
            "frame_index": action_index - 1 if action_index is not None else None,
            "decision_frame_index": action_index - 1 if action_index is not None else None,
            "reference_event_type": kind,
            "reference_action_class": action,
            "reference_source": "events.jsonl+actions.csv",
            "reference_observable_at_decision": bool(event.get("observable_online", False)),
            "reference_history_complete": True,
            "reference_label_status": "reference_labeled",
            "source_event_name": name,
            "source_action_name": action_name,
            "evidence_summary": {"source_event_index": index},
        })
    return result


def build_event_index(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rollout in resolved["rollouts"]:
        rows.extend(extract_reference_events(rollout))
    return rows
