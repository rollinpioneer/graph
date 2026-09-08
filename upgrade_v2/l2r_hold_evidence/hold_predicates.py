from __future__ import annotations

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


def evidence_from_interval(row: dict[str, Any], *, rho_max: float = 0.35, direction_min: float = 0.80, min_motion: float = 0.004) -> str:
    valid = row.get("effective_motion_interval") and row.get("identity_ok") and row.get("direction_cosine") is not None and row.get("relative_vector_error") is not None
    if not valid:
        return UNKNOWN if row.get("identity_ok") is None else FALSE
    if min(float(row["object_displacement_norm"]), float(row["gripper_displacement_norm"])) < min_motion:
        return FALSE
    return TRUE if float(row["direction_cosine"]) >= direction_min and float(row["relative_vector_error"]) <= rho_max else FALSE


def evaluate_candidate(observations: list[dict[str, Any]], geometry: list[dict[str, Any]], candidate_id: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or {}
    hold = False
    supported_time = 0.0
    supported_displacement = 0.0
    rows = []
    previous_contact = UNKNOWN
    for obs, geo in zip(observations, geometry):
        p = obs.get("predicates", obs)
        closed, contact, opened = tri(p.get("gripper_command_closed")), tri(p.get("contact_present")), tri(p.get("gripper_command_open"))
        if candidate_id == "B_count1":
            evidence = tri(p.get("stable_hold_observed"))
        elif candidate_id == "B_count2":
            evidence = tri(p.get("stable_hold_observed"))
        else:
            evidence = evidence_from_interval(geo, rho_max=float(config.get("relative_rho_max", 0.35)), min_motion=float(config.get("min_motion", 0.004)))
        if candidate_id == "B_count2":
            if evidence == TRUE:
                config["_run"] = int(config.get("_run", 0)) + 1
                evidence = TRUE if config["_run"] >= 2 else FALSE
            else:
                config["_run"] = 0
        if candidate_id.startswith("C4") and evidence == TRUE:
            dt = float(geo.get("dt_seconds") or 0.0)
            supported_time += dt
            supported_displacement += min(float(geo.get("object_displacement_norm") or 0.0), float(geo.get("gripper_displacement_norm") or 0.0))
            evidence = TRUE if supported_time >= float(config.get("supported_time_min", 0.1)) and supported_displacement >= float(config.get("supported_displacement_min", 0.004)) else FALSE
        if closed == TRUE and contact == TRUE and evidence == TRUE:
            hold = True
        if opened == TRUE or (contact == FALSE and previous_contact == TRUE and hold):
            supported_time = 0.0
            supported_displacement = 0.0
        rows.append({"frame_index": obs.get("frame_index"), "time": obs.get("time"), "hold_evidence": evidence, "hold_memory": TRUE if hold else FALSE, "supported_time": supported_time, "supported_displacement": supported_displacement, "contact": contact, "closed": closed, "open": opened, "geometry": geo})
        previous_contact = contact
    return rows
