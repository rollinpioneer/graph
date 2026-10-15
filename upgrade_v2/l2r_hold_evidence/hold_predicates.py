from __future__ import annotations

from collections import deque
from typing import Any

TRUE, FALSE, UNKNOWN = "true", "false", "unknown"


def tri(value: Any) -> str:
    if value in (TRUE, FALSE, UNKNOWN):
        return value
    if value is True:
        return TRUE
    if value is False:
        return FALSE
    return UNKNOWN


def evidence_from_interval(
    row: dict[str, Any],
    *,
    closed: str = UNKNOWN,
    contact: str = UNKNOWN,
    previous_closed: str = UNKNOWN,
    previous_contact: str = UNKNOWN,
    rho_max: float = 0.35,
    direction_min: float = 0.80,
    min_motion: float = 0.004,
) -> str:
    if closed != TRUE or contact != TRUE or previous_closed != TRUE or previous_contact != TRUE:
        return UNKNOWN
    valid = row.get("effective_motion_interval") and row.get("identity_ok") and row.get("direction_cosine") is not None and row.get("relative_vector_error") is not None
    if not valid:
        return UNKNOWN
    if min(float(row["object_displacement_norm"]), float(row["gripper_displacement_norm"])) < min_motion:
        return UNKNOWN
    return TRUE if float(row["direction_cosine"]) >= direction_min and float(row["relative_vector_error"]) <= rho_max else FALSE


def evaluate_candidate(observations: list[dict[str, Any]], geometry: list[dict[str, Any]], candidate_id: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or {}
    hold = False
    supported_time = 0.0
    supported_displacement = 0.0
    qualifying_intervals: deque[tuple[float, float, float, tuple[float, float], tuple[float, float]]] = deque()
    run_length = 0
    rows = []
    previous_contact = UNKNOWN
    previous_closed = UNKNOWN
    for obs, geo in zip(observations, geometry):
        p = obs.get("predicates", obs)
        closed, contact, opened = tri(p.get("gripper_command_closed")), tri(p.get("contact_present")), tri(p.get("gripper_command_open"))
        if candidate_id == "B_count1":
            evidence = tri(p.get("stable_hold_observed"))
        elif candidate_id == "B_count2":
            evidence = tri(p.get("stable_hold_observed"))
        else:
            evidence = evidence_from_interval(
                geo,
                closed=closed,
                contact=contact,
                previous_closed=previous_closed,
                previous_contact=previous_contact,
                rho_max=float(config.get("relative_rho_max", 0.35)),
                min_motion=float(config.get("min_motion", 0.004)),
            )
        if candidate_id == "B_count2":
            if evidence == TRUE:
                run_length += 1
                evidence = TRUE if run_length >= 2 else UNKNOWN
            else:
                run_length = 0
        if candidate_id.startswith("C4"):
            dt = geo.get("dt_seconds")
            continuity_valid = bool(
                closed == TRUE
                and contact == TRUE
                and previous_closed == TRUE
                and previous_contact == TRUE
                and geo.get("effective_motion_interval")
                and dt is not None
                and 0.0 < float(dt) <= 0.25
            )
            interval_qualifies = bool(continuity_valid and evidence == TRUE)
            current_time = float(obs.get("time")) if obs.get("time") is not None else None
            if interval_qualifies and current_time is not None:
                displacement = min(
                    float(geo.get("object_displacement_norm") or 0.0),
                    float(geo.get("gripper_displacement_norm") or 0.0),
                )
                relative_previous = geo.get("relative_position_previous")
                relative_current = geo.get("relative_position_current")
                qualifying_intervals.append((current_time, float(dt), displacement, relative_previous, relative_current))
                window = float(config.get("supported_window_seconds", 0.6))
                while qualifying_intervals and current_time - qualifying_intervals[0][0] > window + 1e-9:
                    qualifying_intervals.popleft()
                supported_time = sum(item[1] for item in qualifying_intervals)
                supported_displacement = sum(item[2] for item in qualifying_intervals)
                anchor = qualifying_intervals[0][3]
                window_drift = max(
                    ((item[4][0] - anchor[0]) ** 2 + (item[4][1] - anchor[1]) ** 2) ** 0.5
                    for item in qualifying_intervals
                )
                evidence = TRUE if (
                    supported_time >= float(config.get("supported_time_min", 0.1))
                    and supported_displacement >= float(config.get("supported_displacement_min", 0.004))
                    and window_drift <= float(config.get("relative_position_window_drift_max", 0.01))
                ) else FALSE
            elif continuity_valid and current_time is not None:
                window = float(config.get("supported_window_seconds", 0.6))
                while qualifying_intervals and current_time - qualifying_intervals[0][0] > window + 1e-9:
                    qualifying_intervals.popleft()
                supported_time = sum(item[1] for item in qualifying_intervals)
                supported_displacement = sum(item[2] for item in qualifying_intervals)
                evidence = FALSE
            else:
                # A missing, masked, stale, non-contact, or non-qualifying
                # interval breaks continuity; never bridge it by interpolation.
                supported_time = 0.0
                supported_displacement = 0.0
                qualifying_intervals.clear()
                evidence = UNKNOWN if dt is None or closed == UNKNOWN or contact == UNKNOWN else FALSE
        if closed == TRUE and contact == TRUE and evidence == TRUE:
            hold = True
        if opened == TRUE or (contact == FALSE and previous_contact == TRUE and hold):
            supported_time = 0.0
            supported_displacement = 0.0
            qualifying_intervals.clear()
            hold = False
        rows.append({"frame_index": obs.get("frame_index"), "time": obs.get("time"), "hold_evidence": evidence, "hold_memory": TRUE if hold else FALSE, "supported_time": supported_time, "supported_displacement": supported_displacement, "contact": contact, "closed": closed, "open": opened, "geometry": geo})
        previous_contact = contact
        previous_closed = closed
    return rows
