from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import endpoint_features
from .visual_separation import visible_event


GATE_BLOCKERS = (
    "BASELINE_NOT_READY",
    "VISUAL_MISSING_OR_UNAVAILABLE",
    "WINDOW_INVALID",
    "AMPLITUDE_BELOW_THRESHOLD",
    "RADIAL_GROWTH_BELOW_THRESHOLD",
    "MOTION_EFFICIENCY_BELOW_THRESHOLD",
)


def motion_gate_diagnostics(
    rows: list[dict[str, Any]],
    row_index: int,
    *,
    window_ns: int = 500_000_000,
    theta_motion: float = 0.35,
    anchor_xy: tuple[float, float] | None = None,
    scale_px: float | None = None,
) -> dict[str, Any]:
    """Return raw B2 gate values and every blocker for one frame.

    This helper is diagnostic-only. It applies the frozen B2 thresholds and
    never emits a decision or changes the online state. Multiple blockers are
    retained when applicable (for example, a missing current visual also makes
    the temporal window invalid).
    """
    row = rows[row_index]
    ns = int(row["physical_time_ns"])
    order = int(row["capture_order"])
    blockers: list[str] = []
    baseline_ready = anchor_xy is not None and scale_px is not None and scale_px > 0
    if not baseline_ready:
        blockers.append("BASELINE_NOT_READY")

    visual_available = not bool(row.get("frame_missing") or row.get("detector_error")) and row.get("context_valid") is True
    if not visual_available:
        blockers.append("VISUAL_MISSING_OR_UNAVAILABLE")

    features: dict[str, Any] = {}
    if baseline_ready:
        try:
            features = endpoint_features(rows[: row_index + 1], ns, window_ns, anchor_xy, float(scale_px))
        except (TypeError, ValueError):
            features = {"valid": False, "reason": "INVALID_FEATURE_INPUT"}
        if not features.get("valid"):
            blockers.append("WINDOW_INVALID")
        else:
            amplitude = float(features.get("amplitude", 0.0))
            radial_gain = float(features.get("radial_gain", 0.0))
            efficiency = float(features.get("outward_efficiency", 0.0))
            if amplitude < theta_motion:
                blockers.append("AMPLITUDE_BELOW_THRESHOLD")
            if radial_gain < theta_motion / 2:
                blockers.append("RADIAL_GROWTH_BELOW_THRESHOLD")
            if efficiency < 0.65:
                blockers.append("MOTION_EFFICIENCY_BELOW_THRESHOLD")

    return {
        "observation_ns": ns,
        "capture_order": order,
        "baseline_ready": baseline_ready,
        "visual_available": visual_available,
        "window_ns": window_ns,
        "theta_motion": theta_motion,
        "radial_gain_threshold": theta_motion / 2,
        "outward_efficiency_threshold": 0.65,
        "window_valid": bool(features.get("valid")),
        "window_reason": features.get("reason"),
        "window_span_ns": features.get("span_ns"),
        "window_points": features.get("points"),
        "amplitude": features.get("amplitude"),
        "radial_gain": features.get("radial_gain"),
        "path_length": features.get("path_length"),
        "outward_efficiency": features.get("outward_efficiency"),
        "radial_speed_per_s": features.get("radial_speed_per_s"),
        "blockers": blockers,
        "gate_pass": not blockers,
    }


@dataclass
class TemporalDecision:
    method: str
    window_ns: int = 500_000_000
    theta_motion: float = 0.35
    theta_sep: float = 0.10
    historical_hold: bool = True
    anchor_xy: tuple[float, float] | None = None
    scale_px: float | None = None
    emitted: bool = False

    def reset(self) -> None:
        self.emitted = False

    def step(self, rows: list[dict[str, Any]], row_index: int, *, evidence_end_ns: int | None = None) -> dict[str, Any]:
        row = rows[row_index]; ns = int(row["physical_time_ns"]); order = int(row["capture_order"])
        if row.get("gripper_command") == "open" or row.get("requested_effect") == "RELEASE_OBJECT" or row.get("attempt_end_reason") == "release":
            self.emitted = False
            return {"state": "RELEASE_EXPECTED", "reason": "RELEASE_INTENT", "source": "release", "observation_ns": ns, "capture_order": order}
        if not self.historical_hold:
            return {"state": "HOLD_UNVERIFIED", "reason": "HISTORICAL_HOLD_NOT_READY", "source": None, "observation_ns": ns, "capture_order": order}
        if self.anchor_xy is None or self.scale_px is None:
            return {"state": "BASELINE_PENDING", "reason": "BASELINE_PENDING", "source": None, "observation_ns": ns, "capture_order": order}
        if self.emitted:
            return {"state": "LOSS_EVIDENCE", "reason": "LOSS_ALREADY_EMITTED", "source": "latched", "observation_ns": ns, "capture_order": order}
        prefix = rows[: row_index + 1]
        sep = visible_event(prefix, now_ns=ns, theta=self.theta_sep, scale=self.scale_px)
        motion = endpoint_features(prefix, ns, self.window_ns, self.anchor_xy, self.scale_px)
        motion_event = bool(motion.get("valid") and motion.get("amplitude", 0) >= self.theta_motion and motion.get("radial_gain", 0) >= self.theta_motion / 2 and motion.get("outward_efficiency", 0) >= 0.65)
        if self.method in {"B1", "B3"} and sep.get("event"):
            self.emitted = True
            return {"state": "LOSS_EVIDENCE", "reason": "VISIBLE_SEPARATION_EVIDENCE", "source": "B1", "observation_ns": ns, "capture_order": order, "window_start_ns": sep.get("first_ns"), "window_end_ns": ns, "valid_span_ns": sep.get("valid_span_ns")}
        if self.method in {"B2", "B3"} and motion_event:
            self.emitted = True
            return {"state": "LOSS_EVIDENCE", "reason": "TEMPORAL_RELATIVE_MOTION_EVIDENCE", "source": "B2", "observation_ns": ns, "capture_order": order, "window_start_ns": ns - self.window_ns, "window_end_ns": ns, "valid_span_ns": motion.get("span_ns")}
        if row.get("frame_missing") or row.get("detector_error") or row.get("context_valid") is not True:
            return {"state": "OBSERVATION_UNAVAILABLE", "reason": "OBSERVATION_UNAVAILABLE", "source": None, "observation_ns": ns, "capture_order": order}
        return {"state": "UNCERTAIN_SLIP", "reason": "NO_LOSS_EVIDENCE_YET", "source": None, "observation_ns": ns, "capture_order": order}
