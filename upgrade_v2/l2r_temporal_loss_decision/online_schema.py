from __future__ import annotations

from typing import Any

ALLOWED_FIELDS = frozenset({"physical_time_ns", "capture_order", "attempt_id", "context_valid", "requested_effect", "gripper_command", "attempt_end", "attempt_end_reason", "contact_present", "object_centroid", "gripper_centroid", "object_area", "gripper_area", "object_confidence", "gripper_confidence", "frame_missing", "jpeg_sha256", "attempt_active", "attempt_phase", "time"})
FORBIDDEN_FIELDS = frozenset({"case_id", "family_id", "arm_id", "method", "weld_active", "object_world_position", "support_force_n", "physical_loss_onset_ns", "final_task_success"})


def sanitize(row: dict[str, Any]) -> dict[str, Any]:
    if set(row) & FORBIDDEN_FIELDS:
        raise ValueError("metadata or physical oracle field in online input")
    return {key: value for key, value in row.items() if key in ALLOWED_FIELDS}
