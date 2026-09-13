from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

MAX_VALID_GAP_NS = 100_000_000


def integer_time(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("time must be a nonnegative integer nanosecond timestamp")
    return value


@dataclass(frozen=True)
class Point:
    time_ns: int
    capture_order: int
    relative_xy: tuple[float, float] | None
    valid: bool = True
    release: bool = False
    attempt_id: int = 1
    evidence_id: tuple[int, str] | None = None

    def __post_init__(self) -> None:
        integer_time(self.time_ns)
        if isinstance(self.capture_order, bool) or not isinstance(self.capture_order, int):
            raise ValueError("capture_order must be integer")
        if self.relative_xy is not None and (len(self.relative_xy) != 2 or not all(math.isfinite(x) for x in self.relative_xy)):
            raise ValueError("relative_xy must contain finite values")


def validate_order(points: Sequence[Point]) -> None:
    for before, after in zip(points, points[1:]):
        if after.capture_order <= before.capture_order or after.time_ns < before.time_ns:
            raise ValueError("nonmonotonic stream")


def deduplicate_points(points: Sequence[Point]) -> list[Point]:
    validate_order(points)
    seen: set[tuple[int, str]] = set(); result: list[Point] = []
    for point in points:
        if point.evidence_id is not None and point.evidence_id in seen:
            continue
        if point.evidence_id is not None:
            seen.add(point.evidence_id)
        result.append(point)
    return result


def window_features(points: Sequence[Point], *, now_ns: int, window_ns: int, anchor_xy: tuple[float, float], scale_px: float) -> dict[str, Any]:
    integer_time(now_ns)
    if window_ns <= 0 or not math.isfinite(scale_px) or scale_px <= 0:
        raise ValueError("invalid window or scale")
    validate_order([point for point in points if point.time_ns <= now_ns])
    # Only the latest contiguous valid prefix can contribute to the window.
    # Walking backwards avoids rescanning the entire episode for every frame.
    suffix_rev: list[Point] = []
    current_attempt: int | None = None
    for point in reversed(points):
        if point.time_ns > now_ns:
            continue
        if current_attempt is None:
            current_attempt = point.attempt_id
        if point.attempt_id != current_attempt or point.release or not point.valid or point.relative_xy is None:
            break
        if suffix_rev and suffix_rev[-1].time_ns - point.time_ns > MAX_VALID_GAP_NS:
            break
        suffix_rev.append(point)
    suffix = list(reversed(suffix_rev))
    window = [point for point in suffix if now_ns - window_ns <= point.time_ns <= now_ns]
    if not window or window[-1].time_ns != now_ns:
        return {"valid": False, "reason": "NO_CURRENT_VALID_EVIDENCE"}
    span = window[-1].time_ns - window[0].time_ns
    minimum = max(3, math.ceil(0.8 * window_ns / 50_000_000) + 1)
    if span * 5 < window_ns * 4 or len(window) < minimum:
        return {"valid": False, "reason": "INSUFFICIENT_WINDOW_SUPPORT", "span_ns": span, "points": len(window)}
    distances = [math.dist(point.relative_xy, anchor_xy) / scale_px for point in window]
    path = sum(math.dist(a.relative_xy, b.relative_xy) / scale_px for a, b in zip(window, window[1:]))
    gain = distances[-1] - distances[0]
    efficiency = max(gain, 0.0) / path if path > 0 else 0.0
    return {"valid": True, "reason": "VALID", "time_ns": now_ns, "window_ns": window_ns, "span_ns": span, "points": len(window), "amplitude": distances[-1], "radial_gain": gain, "path_length": path, "outward_efficiency": min(1.0, efficiency), "radial_speed_per_s": gain * 1_000_000_000 / span}
