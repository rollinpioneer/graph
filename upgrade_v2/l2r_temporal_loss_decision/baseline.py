from __future__ import annotations

import math
import statistics
from typing import Any


def fit(rows: list[dict[str, Any]], *, minimum_span_ns: int = 500_000_000, minimum_distinct_times: int = 11, maximum_gap_ns: int = 100_000_000) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("gripper_command") == "closed" and row.get("contact_present") is True and not row.get("attempt_end") and not row.get("frame_missing") and not row.get("detector_error") and row.get("object_centroid") is not None and row.get("gripper_centroid") is not None and float(row.get("object_area", 0.0)) > 0]
    times = sorted({int(row["physical_time_ns"]) for row in eligible})
    span = times[-1] - times[0] if times else 0
    valid = bool(len(times) >= minimum_distinct_times and span >= minimum_span_ns and all(b - a <= maximum_gap_ns for a, b in zip(times, times[1:])))
    rel = [(float(row["object_centroid"][0]) - float(row["gripper_centroid"][0]), float(row["object_centroid"][1]) - float(row["gripper_centroid"][1])) for row in eligible]
    anchor = (statistics.median([x[0] for x in rel]), statistics.median([x[1] for x in rel])) if rel else None
    areas = [float(row["object_area"]) for row in eligible if float(row.get("object_area", 0.0)) > 0]
    scale = math.sqrt(statistics.median(areas) / math.pi) if areas else None
    return {"ready": valid and anchor is not None and scale is not None and scale > 0, "eligible_count": len(eligible), "distinct_times": len(times), "span_ns": span, "max_gap_ns": max((b-a for a,b in zip(times,times[1:])), default=0), "anchor_xy": anchor, "scale_px": scale, "baseline_noise": (statistics.median([math.dist(item, anchor) for item in rel]) if anchor and rel else None)}
