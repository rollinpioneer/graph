from __future__ import annotations

import math
from typing import Any

from pathlib import Path

import cv2
import numpy as np


def box_gap(object_box: list[float] | None, gripper_box: list[float] | None, scale: float) -> float | None:
    if not object_box or not gripper_box or scale <= 0:
        return None
    dx = max(object_box[0] - gripper_box[2], gripper_box[0] - object_box[2], 0.0)
    dy = max(object_box[1] - gripper_box[3], gripper_box[1] - object_box[3], 0.0)
    return math.hypot(dx, dy) / scale


def _union(*masks: np.ndarray) -> np.ndarray:
    result = masks[0].copy()
    for mask in masks[1:]:
        result = cv2.bitwise_or(result, mask)
    return result


def detect_frame_boxes(path: Path) -> dict[str, Any]:
    """Derive the frozen color-segmentation boxes used by the R27 proxy.

    This is a new feature export layered on top of the existing detector; it
    does not modify the historical detector output or feed physical labels to
    the online decision function.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = _union(
        cv2.inRange(hsv, (0, 90, 45), (12, 255, 255)),
        cv2.inRange(hsv, (168, 80, 40), (180, 255, 255)),
    )
    cyan = cv2.inRange(hsv, (78, 70, 45), (105, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    masks = {
        "object": cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel),
        "gripper": cv2.morphologyEx(cyan, cv2.MORPH_OPEN, kernel),
    }
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        pixels = cv2.findNonZero(mask)
        if pixels is None:
            result[f"{name}_bbox"] = None
            result[f"{name}_centroid"] = None
            result[f"{name}_area"] = 0.0
            continue
        x, y, w, h = cv2.boundingRect(pixels)
        moments = cv2.moments(mask)
        result[f"{name}_bbox"] = [float(x), float(y), float(x + w), float(y + h)]
        result[f"{name}_centroid"] = [
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        ] if moments["m00"] else None
        result[f"{name}_area"] = float(cv2.countNonZero(mask))
    result["object_confidence"] = min(1.0, result["object_area"] / 60.0)
    result["gripper_confidence"] = min(1.0, result["gripper_area"] / 1150.0)
    return result


def visible_event(rows: list[dict[str, Any]], *, now_ns: int, theta: float, sustain_ns: int = 100_000_000, minimum_times: int = 3, scale: float = 1.0) -> dict[str, Any]:
    valid: list[tuple[int, float]] = []
    for row in rows:
        ns = int(row.get("physical_time_ns", -1))
        if ns > now_ns or row.get("frame_missing") or row.get("detector_error"):
            continue
        gap = box_gap(row.get("object_bbox"), row.get("gripper_bbox"), scale)
        if gap is not None and row.get("context_valid") is True and row.get("gripper_command") == "closed":
            valid.append((ns, gap))
    if not valid:
        return {"event": False, "reason": "UNKNOWN", "valid_span_ns": 0}
    selected = [(ns, gap) for ns, gap in valid if gap > theta]
    times = sorted({ns for ns, _ in selected})
    span = times[-1] - times[0] if times else 0
    if times:
        # A >100 ms missing interval cannot silently count as persistence.
        max_interval = max((b - a for a, b in zip(times, times[1:])), default=0)
    else:
        max_interval = 0
    event = (
        len(times) >= minimum_times
        and span >= sustain_ns
        and max_interval <= 100_000_000
    )
    return {
        "event": event,
        "reason": "VISIBLE_SEPARATION_EVIDENCE" if event else "INSUFFICIENT_SEPARATION_SUSTAIN",
        "first_ns": times[0] if times else None,
        "valid_span_ns": span,
        "max_gap_ns": max_interval,
        "max_gap": max((gap for _, gap in selected), default=0.0),
    }
