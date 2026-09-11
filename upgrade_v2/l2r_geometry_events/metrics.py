"""R15-B metric definitions: outcome enum and separated anomaly counters."""

from __future__ import annotations

from collections import Counter
from typing import Any

CORRECT = "CORRECT"
WRONG_ACTION = "WRONG_ACTION"
MISSED_REQUIRED_ACTION = "MISSED_REQUIRED_ACTION"
REFERENCE_UNRESOLVED = "REFERENCE_UNRESOLVED"

ACTION_EXPECTED = {"retry_grasp", "recover_object"}


def classify_outcome(expected: str | None, selected: str | None, reference_labeled: bool) -> str:
    """Classify one decision row.  ``none == none`` is CORRECT."""
    if not reference_labeled or expected is None:
        return REFERENCE_UNRESOLVED
    if expected == selected:
        return CORRECT
    if expected == "none" and selected in ACTION_EXPECTED:
        return WRONG_ACTION
    if expected in ACTION_EXPECTED and selected == "none":
        return MISSED_REQUIRED_ACTION
    if expected in ACTION_EXPECTED and selected in ACTION_EXPECTED:
        return WRONG_ACTION
    return WRONG_ACTION


def action_correct(expected: str | None, selected: str | None) -> bool:
    return expected is not None and expected == selected


def _rate(rows: list[dict[str, Any]], predicate) -> float | None:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-method summary with the three anomaly families kept separate."""
    rollout_count = len(rows)
    correct = sum(1 for row in rows if row["outcome"] == CORRECT)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)

    def case_rows(*names: str) -> list[dict[str, Any]]:
        return [row for name in names for row in by_case.get(name, [])]

    positives = case_rows("K1_hold_request_ends_without_hold")
    negatives = case_rows(
        "K2_touch_request_completes_without_hold",
        "K3_normal_hold_pause_resume",
        "K7_commanded_release",
        "K8_acquisition_touch_then_continue",
    )
    losses = case_rows("K4_regular_hold_loss", "K5_brief_hold_loss", "K6_long_gap_after_loss")
    hold_events: Counter = Counter()
    for row in rows:
        for route in str(row.get("hold_entry_routes") or "").split(","):
            if route:
                hold_events[route] += 1
    transition_total = sum(int(row.get("hold_transition_event_count") or 0) for row in rows)
    return {
        "rollouts": rollout_count,
        "legacy_action_correct_count": correct,
        "legacy_action_accuracy": (correct / rollout_count) if rollout_count else None,
        "outcome_counts": dict(Counter(row["outcome"] for row in rows)),
        "K1_recall": _rate(positives, lambda row: row["outcome"] == CORRECT),
        "negative_window_false_emergency_rate": _rate(negatives, lambda row: row["outcome"] == WRONG_ACTION),
        "negative_window_false_retry": sum(1 for row in negatives if row.get("selected_action") == "retry_grasp"),
        "negative_window_false_recovery": sum(1 for row in negatives if row.get("selected_action") == "recover_object"),
        "K4_K5_K6_legacy_agreement": _rate(losses, lambda row: row["outcome"] == CORRECT),
        "premature_emergency": sum(1 for row in rows if row.get("premature_emergency")),
        "premature_retry": sum(1 for row in rows if row.get("premature_retry")),
        "premature_recovery": sum(1 for row in rows if row.get("premature_recovery")),
        "post_window_emergency": sum(1 for row in rows if row.get("post_window_emergency")),
        "hold_transition_events": transition_total,
        "hold_entry_route_counts": {key: value for key, value in hold_events.items() if key},
        "unknown_row_count": sum(int(row.get("unknown_rows") or 0) for row in rows),
        "independent_physical_loss_status": "UNRESOLVED",
        "physical_loss_recall": "NOT_ESTIMABLE",
        "input_tier": rows[0].get("input_tier") if rows else None,
    }
