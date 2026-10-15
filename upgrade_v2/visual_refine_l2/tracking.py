"""Small online-only centroid tracker for primitive RGB detections."""

from __future__ import annotations

import math
from typing import Any


def distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def temporal_features(detections: list[dict[str, Any]], stability_window: int) -> list[dict[str, Any]]:
    rows = []
    for index, current in enumerate(detections):
        previous = detections[index - 1] if index else None
        object_delta = distance(current.get("object_centroid"), previous.get("object_centroid") if previous else None)
        gripper_delta = distance(current.get("gripper_centroid"), previous.get("gripper_centroid") if previous else None)
        co_motion = None
        if object_delta is not None and gripper_delta is not None:
            scale = max(object_delta, gripper_delta, 1.0)
            co_motion = max(0.0, 1.0 - abs(object_delta - gripper_delta) / scale)
        window = detections[max(0, index - stability_window + 1):index + 1]
        points = [item.get("object_centroid") for item in window if item.get("object_centroid") is not None]
        max_drift = None
        if len(points) >= 2:
            max_drift = max(distance(points[0], point) or 0.0 for point in points[1:])
        rows.append({"object_delta_px": object_delta, "gripper_delta_px": gripper_delta, "co_motion": co_motion, "window_object_drift_px": max_drift, "history_frames": len(window)})
    return rows
