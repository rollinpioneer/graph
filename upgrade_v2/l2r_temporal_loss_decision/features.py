from __future__ import annotations

import math
from typing import Any

from .baseline import fit
from .timebase import Point, window_features


def relative(row: dict[str, Any]) -> tuple[float, float] | None:
    obj, grip = row.get("object_centroid"), row.get("gripper_centroid")
    if not (isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(grip, (list, tuple)) and len(grip) == 2):
        return None
    try:
        value = (float(obj[0]) - float(grip[0]), float(obj[1]) - float(grip[1]))
    except (TypeError, ValueError):
        return None
    return value if all(math.isfinite(x) for x in value) else None


def points(rows: list[dict[str, Any]]) -> list[Point]:
    return [Point(int(row["physical_time_ns"]), int(row["capture_order"]), relative(row), bool(relative(row) is not None and not row.get("frame_missing") and not row.get("detector_error")), bool(row.get("gripper_command") == "open" or row.get("requested_effect") == "RELEASE_OBJECT"), int(row.get("attempt_id", 1)), (int(row["physical_time_ns"]), str(row.get("jpeg_sha256", "")))) for row in rows]


def endpoint_features(rows: list[dict[str, Any]], now_ns: int, window_ns: int, anchor: tuple[float, float], scale: float) -> dict[str, Any]:
    return window_features(points(rows), now_ns=now_ns, window_ns=window_ns, anchor_xy=anchor, scale_px=scale)


def endpoint_features_points(point_rows: list[Point], now_ns: int, window_ns: int, anchor: tuple[float, float], scale: float) -> dict[str, Any]:
    return window_features(point_rows, now_ns=now_ns, window_ns=window_ns, anchor_xy=anchor, scale_px=scale)
