from __future__ import annotations

import math
from typing import Any


def _point(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        return float(value[0]), float(value[1])
    except (IndexError, TypeError, ValueError):
        return None


def _norm(vector: tuple[float, float] | None) -> float | None:
    return None if vector is None else math.hypot(*vector)


def adjacent_features(previous: dict[str, Any] | None, current: dict[str, Any], diagonal: float | None = None, noise_floor: float = 0.001) -> dict[str, Any]:
    diagonal = float(diagonal or math.hypot(float(current.get("width", 640)), float(current.get("height", 480))))
    po0, pg0 = _point(previous.get("object_centroid")) if previous else None, _point(previous.get("gripper_centroid")) if previous else None
    po1, pg1 = _point(current.get("object_centroid")), _point(current.get("gripper_centroid"))
    object_vec = None if po0 is None or po1 is None else ((po1[0] - po0[0]) / diagonal, (po1[1] - po0[1]) / diagonal)
    gripper_vec = None if pg0 is None or pg1 is None else ((pg1[0] - pg0[0]) / diagonal, (pg1[1] - pg0[1]) / diagonal)
    object_norm, gripper_norm = _norm(object_vec), _norm(gripper_vec)
    dt = None
    if previous and current.get("time") is not None and previous.get("time") is not None:
        dt = float(current["time"]) - float(previous["time"])
    cosine = None
    rho = None
    if object_norm is not None and gripper_norm is not None and object_norm > noise_floor and gripper_norm > noise_floor:
        cosine = (object_vec[0] * gripper_vec[0] + object_vec[1] * gripper_vec[1]) / (object_norm * gripper_norm)
        rho = math.hypot(object_vec[0] - gripper_vec[0], object_vec[1] - gripper_vec[1]) / (object_norm + gripper_norm + 1e-12)
    relative_drift = None
    if po0 is not None and pg0 is not None and po1 is not None and pg1 is not None:
        before = ((po0[0] - pg0[0]) / diagonal, (po0[1] - pg0[1]) / diagonal)
        after = ((po1[0] - pg1[0]) / diagonal, (po1[1] - pg1[1]) / diagonal)
        relative_drift = math.hypot(after[0] - before[0], after[1] - before[1])
    valid_identity = all(v is not None for v in (po0, pg0, po1, pg1))
    effective = bool(valid_identity and dt is not None and 0 < dt <= 0.25 and object_norm is not None and gripper_norm is not None)
    return {
        "object_displacement_vector": object_vec,
        "gripper_displacement_vector": gripper_vec,
        "object_displacement_norm": object_norm,
        "gripper_displacement_norm": gripper_norm,
        "dt_seconds": dt,
        "direction_cosine": cosine,
        "relative_vector_error": rho,
        "relative_position_drift": relative_drift,
        "identity_ok": valid_identity,
        "effective_motion_interval": effective,
        "camera_geometry_valid": True,
    }


def build_features(observations: list[dict[str, Any]], noise_floor: float = 0.001) -> list[dict[str, Any]]:
    output = []
    previous = None
    for index, row in enumerate(observations):
        geometry = adjacent_features(previous, row, noise_floor=noise_floor)
        output.append({"frame_index": row.get("frame_index", index), "time": row.get("time"), **geometry})
        previous = row
    return output
