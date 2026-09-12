from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


CONTACT_REQUIRED_OBSERVATIONS = 2
CONTACT_MAX_GAP_NS = 100_000_000
RGB_REQUIRED_OBSERVATIONS = 2
RGB_MAX_GAP_NS = 150_000_000
RGB_RELATIVE_DISPLACEMENT_PX = 6.0
RGB_OBJECT_MISSING_CONFIDENCE = 0.20
RGB_GRIPPER_VISIBLE_CONFIDENCE = 0.50
HOLD_REQUIRED_OBSERVATIONS = 2


@dataclass(frozen=True)
class FusionProposal:
    selected_action: str = "none"
    reason_code: str = "NO_FALLBACK_LOSS_EVIDENCE"
    proposal_source: str | None = None
    signal_observed: bool = False
    physical_time_ns: int | None = None
    capture_order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LossObservabilityFusionV1:
    """Online-only fallback proposal source after a verified historical hold.

    The module does not modify or wrap CLP3.  It consumes only canonical-time
    online observations and emits a proposal from either sustained contact
    absence or sustained real-RGB object/gripper detach evidence.
    """

    def __init__(self) -> None:
        self.previous_order: int | None = None
        self.previous_ns: int | None = None
        self.hold_count = 0
        self.historical_hold = False
        self.baseline_relative: deque[tuple[float, float]] = deque(maxlen=5)
        self.contact_count = 0
        self.contact_last_ns: int | None = None
        self.rgb_count = 0
        self.rgb_last_ns: int | None = None
        self.emitted = False

    @staticmethod
    def _relative(row: dict[str, Any]) -> tuple[float, float] | None:
        obj, gripper = row.get("object_centroid"), row.get("gripper_centroid")
        if not (isinstance(obj, (list, tuple)) and len(obj) == 2
                and isinstance(gripper, (list, tuple)) and len(gripper) == 2):
            return None
        try:
            values = float(obj[0]) - float(gripper[0]), float(obj[1]) - float(gripper[1])
        except (TypeError, ValueError):
            return None
        return values if all(math.isfinite(value) for value in values) else None

    def _baseline(self) -> tuple[float, float] | None:
        if not self.baseline_relative:
            return None
        return (statistics.median(value[0] for value in self.baseline_relative),
                statistics.median(value[1] for value in self.baseline_relative))

    def _clear_evidence(self) -> None:
        self.contact_count = 0
        self.contact_last_ns = None
        self.rgb_count = 0
        self.rgb_last_ns = None

    def rearm_after_verified_hold(self) -> None:
        """Open one new loss episode without changing the frozen candidate."""
        self.historical_hold = True
        self.hold_count = HOLD_REQUIRED_OBSERVATIONS
        self.emitted = False
        self._clear_evidence()

    def step(self, row: dict[str, Any]) -> dict[str, Any]:
        ns, order = row.get("physical_time_ns"), row.get("capture_order")
        valid_clock = (isinstance(ns, int) and not isinstance(ns, bool)
                       and isinstance(order, int)
                       and (self.previous_order is None or order > self.previous_order)
                       and (self.previous_ns is None or ns >= self.previous_ns))
        release = (row.get("gripper_command") == "open"
                   or row.get("requested_effect") == "RELEASE_OBJECT"
                   or row.get("attempt_end_reason") == "release")
        if not valid_clock or row.get("context_valid") is not True:
            return FusionProposal(reason_code="INVALID_OR_NONMONOTONIC_OBSERVATION",
                                  physical_time_ns=ns if isinstance(ns, int) else None,
                                  capture_order=order if isinstance(order, int) else None).to_dict()
        self.previous_order, self.previous_ns = order, ns
        if release:
            self._clear_evidence()
            self.emitted = False
            return FusionProposal(reason_code="RELEASE_SUPPRESSES_FALLBACK",
                                  physical_time_ns=ns, capture_order=order).to_dict()

        contact = row.get("contact_present")
        relative = self._relative(row)
        holding = (row.get("gripper_command") == "closed"
                   and row.get("requested_effect") == "HOLD_OBJECT"
                   and contact is True)
        if holding:
            self.hold_count += 1
            if relative is not None and not self.historical_hold:
                self.baseline_relative.append(relative)
            if self.hold_count >= HOLD_REQUIRED_OBSERVATIONS:
                self.historical_hold = True
        elif not self.historical_hold:
            self.hold_count = 0

        if not self.historical_hold or self.emitted:
            return FusionProposal(reason_code="HISTORICAL_HOLD_NOT_ARMED" if not self.historical_hold else "FALLBACK_ALREADY_EMITTED",
                                  physical_time_ns=ns, capture_order=order).to_dict()

        if contact is False:
            if self.contact_last_ns is not None and 0 < ns - self.contact_last_ns <= CONTACT_MAX_GAP_NS:
                self.contact_count += 1
            elif self.contact_last_ns != ns:
                self.contact_count = 1
            self.contact_last_ns = ns
        elif contact is True:
            self.contact_count = 0
            self.contact_last_ns = None

        baseline = self._baseline()
        displacement = math.dist(relative, baseline) if relative is not None and baseline is not None else None
        rgb_detached = bool(
            float(row.get("gripper_confidence") or 0.0) >= RGB_GRIPPER_VISIBLE_CONFIDENCE
            and ((row.get("object_centroid") is None
                  and float(row.get("object_confidence") or 0.0) < RGB_OBJECT_MISSING_CONFIDENCE)
                 or (displacement is not None and displacement >= RGB_RELATIVE_DISPLACEMENT_PX))
        )
        if rgb_detached:
            if self.rgb_last_ns is not None and 0 < ns - self.rgb_last_ns <= RGB_MAX_GAP_NS:
                self.rgb_count += 1
            elif self.rgb_last_ns != ns:
                self.rgb_count = 1
            self.rgb_last_ns = ns
        else:
            self.rgb_count = 0
            self.rgb_last_ns = None

        source = None
        reason = "NO_FALLBACK_LOSS_EVIDENCE"
        if self.contact_count >= CONTACT_REQUIRED_OBSERVATIONS:
            source, reason = "SUSTAINED_CONTACT_ABSENCE", "FALLBACK_CONTACT_ABSENCE_CONFIRMED"
        elif self.rgb_count >= RGB_REQUIRED_OBSERVATIONS:
            source, reason = "REAL_RGB_DETACH", "FALLBACK_RGB_DETACH_CONFIRMED"
        if source is not None:
            self.emitted = True
            return FusionProposal("recover_object", reason, source, True, ns, order).to_dict()
        return FusionProposal(physical_time_ns=ns, capture_order=order).to_dict()


def run_fusion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fusion = LossObservabilityFusionV1()
    return [fusion.step(row) for row in rows]
