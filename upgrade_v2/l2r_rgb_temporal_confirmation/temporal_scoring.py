from __future__ import annotations

from typing import Any

OUTCOMES = (
    "CORRECT_NONE", "CORRECT_IN_WINDOW", "EARLY_ACTION", "LATE_ACTION",
    "MISSED_REQUIRED_ACTION", "WRONG_ACTION", "FALSE_RETRY", "FALSE_RECOVERY",
    "REFERENCE_UNRESOLVED",
)


def earliest_non_none(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [row for row in actions if row.get("action") in {"retry_grasp", "recover_object"}]
    return min(rows, key=lambda row: (float(row.get("time", float("inf"))), int(row.get("capture_order", 10**9)))) if rows else None


def score_temporal(*, truth_action: str, event_onset: float | None, deadline: float | None,
                   actions: list[dict[str, Any]], resolvable: bool = True) -> dict[str, Any]:
    first = earliest_non_none(actions)
    if not resolvable:
        outcome = "REFERENCE_UNRESOLVED"
    elif truth_action == "none":
        outcome = "CORRECT_NONE" if first is None else (
            "FALSE_RETRY" if first["action"] == "retry_grasp" else "FALSE_RECOVERY")
    elif first is None:
        outcome = "MISSED_REQUIRED_ACTION"
    elif first["action"] != truth_action:
        outcome = "WRONG_ACTION"
    elif event_onset is None or deadline is None:
        outcome = "REFERENCE_UNRESOLVED"
    elif float(first["time"]) < float(event_onset) - 1e-9:
        outcome = "EARLY_ACTION"
    elif float(first["time"]) <= float(deadline) + 1e-9:
        outcome = "CORRECT_IN_WINDOW"
    else:
        outcome = "LATE_ACTION"
    return {
        "temporal_outcome": outcome,
        "accurate": outcome in {"CORRECT_NONE", "CORRECT_IN_WINDOW"},
        "primary_action": None if first is None else first["action"],
        "primary_action_time": None if first is None else float(first["time"]),
        "primary_action_capture_order": None if first is None else first.get("capture_order"),
        "event_onset": event_onset, "deadline": deadline,
        "all_actions": actions,
    }
