"""Pure R12 event-boundary and contact-stream audit helpers."""
from __future__ import annotations

from typing import Any


EPSILON_SECONDS = 1e-9


def event_relation(sample_time: float, event_time: float, *, sample_order: int | None = None, event_order: int | None = None, order_provenance_verified: bool = False) -> str:
    delta = sample_time - event_time
    if delta < -EPSILON_SECONDS:
        return "STRICTLY_BEFORE_EVENT"
    if delta > EPSILON_SECONDS:
        return "STRICTLY_AFTER_EVENT"
    if order_provenance_verified and sample_order is not None and event_order is not None:
        return "SAME_TIMESTAMP_AFTER_EVENT" if sample_order > event_order else "SAME_TIMESTAMP_BEFORE_EVENT"
    return "SAME_TIMESTAMP_ORDER_UNKNOWN"


def first_saved_contact_loss(rows: list[dict[str, Any]], window_start: float, window_end: float) -> dict[str, Any]:
    times = [float(row["time"]) for row in rows]
    if times != sorted(times):
        raise ValueError("contact stream is not sorted")
    for left, right in zip(rows, rows[1:]):
        if float(left["time"]) == float(right["time"]):
            raise ValueError("same-time contact samples need an order domain")
    prior = None
    for row in rows:
        time = float(row["time"])
        contact = row.get("contact_present")
        if time < window_start:
            prior = contact
            continue
        if time > window_end:
            break
        if prior is True and contact is False:
            return {"status": "CONTACT_TRANSITION_SAVED", "time": time, "physical_loss_verified": False}
        prior = contact
    return {"status": "NOT_OBSERVED_IN_AVAILABLE_WINDOW", "time": None, "physical_loss_verified": False}


def assess_loss_evidence_metadata(*, contact_pair_count: int | None, object_speed: float | None, force_history_available: bool, relative_motion_history_available: bool) -> dict[str, Any]:
    return {"contact_pair_count": contact_pair_count, "object_speed": object_speed, "force_history_available": force_history_available, "relative_motion_history_available": relative_motion_history_available, "task_loss_status": "NOT_ASSESSED_BY_R12", "eligible_for_legacy_relabel": False}
