"""Reference-layer weld transition and drift audit.

``weld_state`` is read here only.  It never reaches a candidate or the state
machine: this module describes the cache, it does not predict anything.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .adapters import distance3
from .features import relative_vector

ORDER_STRICTLY_LATER = "STRICTLY_LATER_TIME"
ORDER_SAME_TIME_ORDERED = "SAME_TIME_ORDERED_BY_CAPTURE_ORDER"
ORDER_SAME_TIME_UNKNOWN = "SAME_TIME_ORDER_UNKNOWN"
ORDER_NO_TRANSITION = "NO_TRANSITION_OBSERVED"

REL_INSIDE = "EVENT_INSIDE_TRANSITION_INTERVAL"
REL_BEFORE = "EVENT_BEFORE_OBSERVED_TRANSITION"
REL_AFTER = "EVENT_AFTER_OBSERVED_TRANSITION"
REL_NOT_COMPARABLE = "NOT_COMPARABLE"


def read_oracle_rows(rollout_dir: Path) -> list[dict[str, str]]:
    path = Path(rollout_dir) / "oracle_timeline.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(
        rows,
        key=lambda row: (float(row.get("time", 0.0) or 0.0), int(float(row.get("capture_order", 0) or 0))),
    )


def _is_on(row: dict[str, str]) -> bool:
    return str(row.get("weld_state", "0")).strip().lower() in {"1", "true"}


def weld_transition(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Detect a real ``1 -> 0`` weld transition between adjacent rows."""
    result: dict[str, Any] = {
        "transition_found": False,
        "last_on_time": None,
        "last_on_capture_order": None,
        "first_off_time": None,
        "first_off_capture_order": None,
        "transition_interval_start": None,
        "transition_interval_end": None,
        "transition_order_status": ORDER_NO_TRANSITION,
    }
    for previous, current in zip(rows, rows[1:]):
        if _is_on(previous) and not _is_on(current):
            previous_time = float(previous["time"])
            current_time = float(current["time"])
            previous_order = int(float(previous.get("capture_order", 0) or 0))
            current_order = int(float(current.get("capture_order", 0) or 0))
            if current_time > previous_time:
                status = ORDER_STRICTLY_LATER
            elif current_order > previous_order:
                status = ORDER_SAME_TIME_ORDERED
            else:
                status = ORDER_SAME_TIME_UNKNOWN
            result.update(
                {
                    "transition_found": True,
                    "last_on_time": previous_time,
                    "last_on_capture_order": previous_order,
                    "first_off_time": current_time,
                    "first_off_capture_order": current_order,
                    "transition_interval_start": previous_time,
                    "transition_interval_end": current_time,
                    "transition_order_status": status,
                }
            )
            return result
    return result


def contact_lost_time(rollout_dir: Path) -> float | None:
    path = Path(rollout_dir) / "events.jsonl"
    if not path.is_file():
        return None
    import json

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("event") == "contact_lost":
            return float(row["time"])
    return None


def event_transition_relation(event_time: float | None, transition: dict[str, Any]) -> str:
    if event_time is None or not transition.get("transition_found"):
        return REL_NOT_COMPARABLE
    start = transition["transition_interval_start"]
    end = transition["transition_interval_end"]
    if start - 1e-9 <= event_time <= end + 1e-9:
        return REL_INSIDE
    return REL_BEFORE if event_time < start else REL_AFTER


def drift_audit(
    samples: list[dict[str, Any]],
    anchor_relative: list[float] | None,
    anchor_time: float | None,
    anchor_order: int | None,
    prefix: str,
) -> dict[str, Any]:
    """Max relative drift from one anchor, plus the window and sample count."""
    result = {
        f"{prefix}_anchor_time": anchor_time,
        f"{prefix}_anchor_capture_order": anchor_order,
        f"{prefix}_max_drift_m": None,
        f"{prefix}_window_start": anchor_time,
        f"{prefix}_window_end": None,
        f"{prefix}_sample_count": 0,
    }
    if anchor_relative is None or anchor_time is None:
        return result
    drifts = []
    last_time = None
    for sample in samples:
        time_value = sample.get("time")
        if time_value is None or float(time_value) < float(anchor_time) - 1e-9:
            continue
        current = relative_vector(sample)
        if current is None:
            continue
        drifts.append(distance3(anchor_relative, current))
        last_time = float(time_value)
    result[f"{prefix}_max_drift_m"] = round(max(drifts), 6) if drifts else None
    result[f"{prefix}_window_end"] = last_time
    result[f"{prefix}_sample_count"] = len(drifts)
    return result


def relative_at_order(samples: list[dict[str, Any]], capture_order: int | None, time_value: float | None) -> list[float] | None:
    for sample in samples:
        if capture_order is not None and sample.get("capture_order") == capture_order:
            return relative_vector(sample)
    for sample in samples:
        if time_value is not None and sample.get("time") is not None and abs(float(sample["time"]) - float(time_value)) <= 1e-9:
            return relative_vector(sample)
    return None
