"""Small builders for pure sequence tests (no simulator, no data files)."""

from __future__ import annotations

from typing import Any

from upgrade_v2.l2r_geometry_events.adapters import DATA_MASKED, DATA_VALID

PROTOCOL: dict[str, Any] = {
    "schema": "l2rar2_geometry_events_protocol_v1",
    "parameters": {
        "height_on_m": 0.02,
        "height_off_m": 0.01,
        "min_motion_3d_m": 0.004,
        "direction_cosine_min": 0.8,
        "relative_rho_max": 0.35,
        "relative_hold_drift_m": 0.01,
        "relative_loss_drift_m": 0.02,
        "evidence_duration_s": 0.05,
        "max_gap_s": 0.25,
    },
}


def sample(
    time: float,
    order: int,
    object_xyz: list[float] | None,
    gripper_xyz: list[float] | None,
    contact: bool | None,
    command: str,
    *,
    quality: str = DATA_VALID,
    attempt_id: int | None = 1,
    attempt_active: bool = True,
    attempt_end: bool = False,
    attempt_end_reason: str | None = None,
    requested_effect: str | None = "HOLD_OBJECT",
) -> dict[str, Any]:
    return {
        "time": time,
        "capture_order": order,
        "object_xyz": object_xyz,
        "gripper_xyz": gripper_xyz,
        "contact_present": contact,
        "gripper_command": command,
        "attempt_id": attempt_id,
        "attempt_phase": "acquiring",
        "attempt_active": attempt_active,
        "attempt_end": attempt_end,
        "attempt_end_reason": attempt_end_reason,
        "attempt_end_sequence": 1,
        "requested_effect": requested_effect,
        "data_quality": quality,
        "data_quality_reason": "capture_order_time_match" if quality == DATA_VALID else "synthetic_masked",
    }


def masked(time: float, order: int) -> dict[str, Any]:
    return sample(time, order, None, None, None, "open", quality=DATA_MASKED)


def hold_sequence(
    steps: int = 8,
    step_z: float = 0.006,
    start_time: float = 0.0,
    dt: float = 0.05,
    contact: bool = True,
) -> list[dict[str, Any]]:
    """Gripper and object rise together; the object keeps a constant offset."""
    samples = []
    for index in range(steps):
        time_value = start_time + index * dt
        z = 0.0 + index * step_z
        samples.append(
            sample(
                time_value,
                index,
                [0.0, 0.0, 0.5 + z],
                [0.0, 0.0, 0.55 + z],
                contact,
                "closed" if index > 0 else "open",
            )
        )
    return samples
