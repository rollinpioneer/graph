"""Pure geometry features: relative height, 3D co-motion, relative vector.

All functions are side-effect free and return ``None`` when the input is not
usable.  No feature fabricates orientation: the cache stores no object or
gripper quaternion, so only world-frame relative translation is computed.
"""

from __future__ import annotations

import math
from typing import Any

from .adapters import DATA_VALID, distance3, displacement, norm3


def height_baseline(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Anchor z0 at the first valid, gripper-open, clearly-not-in-contact pose."""
    for sample in samples:
        if sample.get("data_quality") != DATA_VALID:
            continue
        object_xyz = sample.get("object_xyz")
        if object_xyz is None:
            continue
        if sample.get("gripper_command") != "open":
            continue
        if sample.get("contact_present") is not False:
            continue
        time_value = sample.get("time")
        return {
            "status": "AVAILABLE",
            "z0": float(object_xyz[2]),
            "anchor_time": float(time_value) if time_value is not None else None,
            "anchor_capture_order": sample.get("capture_order"),
            "anchor_source": "first_valid_open_no_contact",
        }
    return {
        "status": "HEIGHT_BASELINE_UNAVAILABLE",
        "z0": None,
        "anchor_time": None,
        "anchor_capture_order": None,
        "anchor_source": None,
    }


def height_above_baseline(sample: dict[str, Any], z0: float | None) -> float | None:
    if z0 is None or sample.get("data_quality") != DATA_VALID:
        return None
    object_xyz = sample.get("object_xyz")
    if object_xyz is None:
        return None
    return float(object_xyz[2]) - float(z0)


def motion_features(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    max_gap_s: float = 0.25,
) -> dict[str, Any]:
    """Adjacent-sample 3D co-motion features, in metres."""
    blank: dict[str, Any] = {
        "dt_seconds": None,
        "object_step_vector": None,
        "gripper_step_vector": None,
        "object_step_norm": None,
        "gripper_step_norm": None,
        "direction_cosine": None,
        "relative_vector_error": None,
        "motion_interval_valid": False,
        "invalid_reason": None,
    }
    if previous is None:
        blank["invalid_reason"] = "no_previous_sample"
        return blank
    if previous.get("data_quality") != DATA_VALID or current.get("data_quality") != DATA_VALID:
        blank["invalid_reason"] = "sample_quality_not_valid"
        return blank
    previous_time = previous.get("time")
    current_time = current.get("time")
    if previous_time is None or current_time is None:
        blank["invalid_reason"] = "time_unavailable"
        return blank
    dt = float(current_time) - float(previous_time)
    blank["dt_seconds"] = dt
    if dt <= 0:
        blank["invalid_reason"] = "non_positive_dt"
        return blank
    if dt > float(max_gap_s) + 1e-12:
        blank["invalid_reason"] = "gap_exceeds_max_gap_s"
        return blank
    object_step = displacement(previous.get("object_xyz"), current.get("object_xyz"))
    gripper_step = displacement(previous.get("gripper_xyz"), current.get("gripper_xyz"))
    object_norm = norm3(object_step)
    gripper_norm = norm3(gripper_step)
    if object_norm is None or gripper_norm is None:
        blank["invalid_reason"] = "position_unavailable"
        return blank
    cosine = None
    rho = None
    if object_norm > 0.0 and gripper_norm > 0.0:
        dot = sum(object_step[i] * gripper_step[i] for i in range(3))
        cosine = dot / (object_norm * gripper_norm)
        difference = math.sqrt(sum((object_step[i] - gripper_step[i]) ** 2 for i in range(3)))
        rho = difference / (object_norm + gripper_norm + 1e-12)
    return {
        "dt_seconds": dt,
        "object_step_vector": object_step,
        "gripper_step_vector": gripper_step,
        "object_step_norm": object_norm,
        "gripper_step_norm": gripper_norm,
        "direction_cosine": cosine,
        "relative_vector_error": rho,
        "motion_interval_valid": True,
        "invalid_reason": None,
    }


def relative_vector(sample: dict[str, Any]) -> list[float] | None:
    if sample.get("data_quality") != DATA_VALID:
        return None
    object_xyz = sample.get("object_xyz")
    gripper_xyz = sample.get("gripper_xyz")
    if object_xyz is None or gripper_xyz is None:
        return None
    return [float(object_xyz[i]) - float(gripper_xyz[i]) for i in range(3)]


def relative_drift(anchor: list[float] | None, current: list[float] | None) -> float | None:
    return distance3(anchor, current)


def co_motion_qualifies(
    previous: dict[str, Any],
    current: dict[str, Any],
    motion: dict[str, Any],
    protocol: dict[str, float],
) -> bool:
    """A valid, contact-consistent, direction-consistent co-motion interval."""
    if not motion.get("motion_interval_valid"):
        return False
    if previous.get("gripper_command") != "closed" or current.get("gripper_command") != "closed":
        return False
    if previous.get("contact_present") is not True or current.get("contact_present") is not True:
        return False
    object_norm = motion.get("object_step_norm")
    gripper_norm = motion.get("gripper_step_norm")
    cosine = motion.get("direction_cosine")
    rho = motion.get("relative_vector_error")
    if object_norm is None or gripper_norm is None or cosine is None or rho is None:
        return False
    if min(float(object_norm), float(gripper_norm)) < float(protocol["min_motion_3d_m"]):
        return False
    if float(cosine) < float(protocol["direction_cosine_min"]):
        return False
    if float(rho) > float(protocol["relative_rho_max"]):
        return False
    return True
